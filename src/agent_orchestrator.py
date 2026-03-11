import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.tool_gateway import ToolGateway
from src.session_manager import SessionManager
from src.model_backend import (
    ModelBackend,
    ModelTurn,
    ToolCall,
    ToolResult,
    TOOL_SCHEMAS,
    tool_schemas_for_scope,
    create_backend,
)


@dataclass
class ToolCallSpec:
    tool_name: str
    args: dict
    rationale: str = ""


@dataclass
class ExecutedToolCall:
    tool_name: str
    args: dict
    result: Any
    error: Optional[str] = None


@dataclass
class Citation:
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: str = ""
    source_type: str = "file"
    ref_id: Optional[str] = None


@dataclass
class FinalResponse:
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    patch_diff: Optional[str] = None
    next_actions: list[str] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    tool_call_plan: list[ToolCallSpec]
    executed_tool_calls: list[ExecutedToolCall]
    consolidated_evidence: list[Citation]
    final_response: FinalResponse
    raw_turns: list[dict] = field(default_factory=list)


MODES = ("explain", "locate", "suggest", "patch")
SCOPES = ("files-only", "include-pr")

MODE_INSTRUCTIONS = {
    "explain": (
        "Provide a thorough explanation with code citations. "
        "Reference specific file paths and line numbers."
    ),
    "locate": (
        "Identify exactly which files and line ranges implement the requested functionality. "
        "Be concise — list locations first, brief explanation second."
    ),
    "suggest": (
        "Suggest concrete next development steps. "
        "For each suggestion include an impact label (high/medium/low) and an effort label (high/medium/low). "
        "End your response with a 'Next Actions' list."
    ),
    "patch": (
        "Propose a code change that addresses the request. "
        "Output the change as a unified diff (patch format) after your explanation."
    ),
}


# Default model map: each mode routes to a specialist model.
# Override with model_map param or use a single model for all modes.
DEFAULT_MODEL_MAP: dict[str, str] = {
    "explain": "gemini-2.5-flash",
    "locate":  "gemini-2.5-flash",
    "suggest": "gemini-2.5-flash",
    "patch":   "gemini-2.5-flash",
}

# Intent keywords used by _classify_intent for auto-routing
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "patch":   ["fix", "patch", "change", "modify", "update", "refactor", "diff", "edit"],
    "locate":  ["where", "find", "locate", "which file", "what file", "defined", "implemented"],
    "suggest": ["suggest", "improve", "recommend", "next step", "should", "todo", "how to"],
    "explain": ["explain", "what", "how", "why", "describe", "walk through", "overview"],
}


class AgentOrchestrator:

    def __init__(
                    self, 
                    gateway: ToolGateway, 
                    session: Optional[SessionManager] = None,
                    api_key: Optional[str] = None, 
                    model: str = "gemini-2.5-flash",
                    model_map: Optional[dict[str, str]] = None,
                    backend: Optional[ModelBackend] = None,
                    backend_type: str = "gemini",
                    **backend_kwargs,
                ):
        self.gateway = gateway
        self.session = session
        self.model = model
        self.model_map = model_map if model_map is not None else {m: model for m in MODES}

        # Use provided backend, or create one from backend_type
        if backend is not None:
            self.backend = backend
        else:
            self.backend = create_backend(
                backend_type,
                model=model,
                api_key=api_key,
                **backend_kwargs,
            )

    @staticmethod
    def _classify_intent(query: str) -> str:
        """Auto-detect mode from query text using keyword heuristics."""
        q = query.lower()
        scores = {mode: 0 for mode in MODES}
        for mode, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    scores[mode] += 1
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "explain"

    def run(self, query: str, mode: str = "explain", scope: str = "include-pr",
            max_turns: int = 10, verbose: bool = False) -> OrchestratorResult:
        if mode == "auto":
            mode = self._classify_intent(query)
            if verbose:
                print(f"[Router] Auto-classified mode: {mode}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")

        if verbose:
            print(f"[Router] Using backend {self.backend.model_name} for mode {mode}")

        system_prompt = self._build_system_prompt(mode, scope)
        tools = tool_schemas_for_scope(scope)

        # Build conversation using backend message formatting
        messages = [
            self.backend.format_system_message(system_prompt),
            self.backend.format_user_message(query),
        ]

        executed_calls: list[ExecutedToolCall] = []
        raw_plan: list[ToolCallSpec] = []
        final_text = ""

        turn = 0
        while turn < max_turns:
            turn += 1
            if verbose:
                print(f"\n[Orchestrator] Turn {turn}")

            model_turn = self.backend.generate(messages, tools, temperature=0.2)

            if model_turn.has_tool_calls:
                if turn == 1:
                    raw_plan = [
                        ToolCallSpec(tool_name=tc.name, args=tc.arguments)
                        for tc in model_turn.tool_calls
                    ]

                if model_turn.text:
                    final_text = model_turn.text

                # Append assistant response to conversation
                messages.append(self.backend.format_assistant_message(model_turn))

                # Execute each tool call
                tool_results: list[ToolResult] = []
                for tc in model_turn.tool_calls:
                    if verbose:
                        print(f"  [Tool] {tc.name}({tc.arguments})")

                    result = self._execute_tool(tc.name, tc.arguments, scope)
                    error = result.get("error") if isinstance(result, dict) else None
                    executed_calls.append(
                        ExecutedToolCall(tool_name=tc.name, args=tc.arguments, result=result, error=error)
                    )

                    if verbose and error:
                        print(f"    [Error] {error}")

                    tool_results.append(ToolResult(name=tc.name, result=result))

                # Append tool results
                formatted = self.backend.format_tool_results(tool_results)
                if isinstance(formatted, list):
                    messages.extend(formatted)
                else:
                    messages.append(formatted)

            elif model_turn.has_text:
                final_text = model_turn.text
                if verbose:
                    print("[Orchestrator] Final answer received.")
                break
            else:
                # No text and no tool calls — stop
                break

        if not final_text.strip():
            # Ask the model for a final answer based on gathered evidence
            messages.append(
                self.backend.format_user_message(
                    "Please provide your final answer based on the evidence gathered."
                )
            )
            fallback = self.backend.generate(messages, tools=[], temperature=0.2)
            if fallback.text:
                final_text = fallback.text

        evidence = self._consolidate_evidence(executed_calls)
        final_response = self._compose_response(final_text, evidence, mode)

        if self.session is not None:
            self.session.add_query(query, final_response.answer_text[:300])
            self.session.add_evidence(
                [
                    {"file_path": c.file_path, "start_line": c.start_line, "end_line": c.end_line}
                    for c in evidence
                    if c.source_type == "file"
                ]
            )

        return OrchestratorResult(
            tool_call_plan=raw_plan,
            executed_tool_calls=executed_calls,
            consolidated_evidence=evidence,
            final_response=final_response,
            raw_turns=self.backend.serialize_messages(messages),
        )

    def _consolidate_evidence(self, executed: list[ExecutedToolCall]) -> list[Citation]:
        seen: set[tuple] = set()
        citations: list[Citation] = []

        for call in executed:
            result = call.result
            if not isinstance(result, dict):
                continue

            if call.tool_name == "search_repo" and "results" in result:
                for item in result["results"]:
                    key = (item.get("file_path"), item.get("start_line"), item.get("end_line"))
                    if key not in seen:
                        seen.add(key)
                        citations.append(Citation(
                            file_path=item.get("file_path", ""),
                            start_line=item.get("start_line"),
                            end_line=item.get("end_line"),
                            snippet=item.get("snippet", ""),
                            source_type="file",
                        ))

            elif call.tool_name == "open_file" and "file_path" in result:
                key = (result["file_path"], result.get("start_line"), result.get("end_line"))
                if key not in seen:
                    seen.add(key)
                    text = result.get("text", "")
                    citations.append(Citation(
                        file_path=result["file_path"],
                        start_line=result.get("start_line"),
                        end_line=result.get("end_line"),
                        snippet=text[:200] + ("..." if len(text) > 200 else ""),
                        source_type="file",
                    ))

            elif call.tool_name == "get_issues" and "issues" in result:
                for issue in result["issues"]:
                    key = ("issue", issue.get("number"))
                    if key not in seen:
                        seen.add(key)
                        citations.append(Citation(
                            file_path=issue.get("url", ""),
                            snippet=issue.get("title", ""),
                            source_type="issue",
                            ref_id=str(issue.get("number", "")),
                        ))

            elif call.tool_name == "get_pull_requests" and "pull_requests" in result:
                for pr in result["pull_requests"]:
                    key = ("pr", pr.get("number"))
                    if key not in seen:
                        seen.add(key)
                        citations.append(Citation(
                            file_path=pr.get("url", ""),
                            snippet=pr.get("title", ""),
                            source_type="pr",
                            ref_id=str(pr.get("number", "")),
                        ))

        return citations

    def _compose_response(self, raw_text: str, evidence: list[Citation], mode: str) -> FinalResponse:
        patch_diff = None
        next_actions: list[str] = []
        answer_text = raw_text

        if mode == "patch":
            patch_diff, answer_text = self._extract_patch(raw_text)

        next_actions, answer_text = self._extract_next_actions(answer_text)

        return FinalResponse(
            answer_text=answer_text.strip(),
            citations=evidence,
            patch_diff=patch_diff,
            next_actions=next_actions,
        )

    @staticmethod
    def _extract_patch(text: str):
        match = re.search(r"```diff\n(.*?)```", text, re.DOTALL)
        if match:
            patch = match.group(1)
            cleaned = text[: match.start()] + text[match.end():]
            return patch.strip(), cleaned.strip()
        lines = text.splitlines()
        diff_lines = []
        rest_lines = []
        in_diff = False
        for line in lines:
            if line.startswith("--- ") or line.startswith("+++ "):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
            else:
                rest_lines.append(line)
        if diff_lines:
            return "\n".join(diff_lines), "\n".join(rest_lines)
        return None, text

    @staticmethod
    def _extract_next_actions(text: str):
        pattern = re.compile(
            r"(?:Next Actions?|Next Steps?|Suggested Steps?|Recommendations?)\s*:?\s*\n"
            r"((?:[ \t]*[-*\d•].*\n?)+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return [], text

        block = match.group(1)
        actions = [
            re.sub(r"^[ \t]*[-*\d•\.\)]+[ \t]*", "", line).strip()
            for line in block.splitlines()
            if line.strip()
        ]
        cleaned = text[: match.start()] + text[match.end():]
        return [a for a in actions if a], cleaned.strip()

    @staticmethod
    def _serialize_contents(contents: list) -> list[dict]:
        """Legacy serializer — backends now handle serialization via serialize_messages()."""
        turns = []
        for content in contents:
            if isinstance(content, dict):
                turns.append(content)
                continue
            role = getattr(content, "role", "unknown")
            parts_out = []
            for part in getattr(content, "parts", []):
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    parts_out.append({"type": "function_call", "name": fc.name, "args": dict(fc.args)})
                elif getattr(part, "function_response", None):
                    fr = part.function_response
                    parts_out.append({"type": "function_response", "name": fr.name, "response": fr.response})
                elif getattr(part, "text", None):
                    parts_out.append({"type": "text", "text": part.text})
            turns.append({"role": role, "parts": parts_out})
        return turns

    def _build_system_prompt(self, mode: str, scope: str) -> str:
        stats = self.gateway.stats()
        session_ctx = ""
        if self.session is not None:
            ctx = self.session.get_llm_context()
            recent = ctx.get("recent_queries", [])
            if recent:
                summaries = "\n".join(
                    f"  - {q['query']}: {q['summary']}" for q in recent[-3:]
                )
                session_ctx = f"\nRecent conversation history:\n{summaries}\n"

        scope_note = (
            "You may use all available tools including issue and PR lookups."
            if scope == "include-pr"
            else "Only use file-based tools (search_repo, open_file, get_repo_stats). Do NOT call get_issues or get_pull_requests."
        )

        return f"""You are an expert repository assistant. You have tools to search code, read files, and query GitHub issues/PRs.

Repository Context:
  Path: {stats.get('repo_path', 'N/A')}
  Files: {stats.get('total_files', 0)}
  Chunks: {stats.get('total_chunks', 0)}
  Issues: {stats.get('total_issues', 0)}
  Pull Requests: {stats.get('total_prs', 0)}
{session_ctx}
Mode: {mode.upper()}
{MODE_INSTRUCTIONS[mode]}

Scope: {scope_note}

General guidelines:
- Always cite specific files and line numbers when referencing code.
- Use multiple tool calls in a turn when gathering broad evidence.
- If you cannot find something, say so clearly rather than guessing.
- Structure your final answer clearly with headings if appropriate."""

    def _define_tools(self):
        """Legacy — kept for backward compat; tools now come from model_backend.TOOL_SCHEMAS."""
        return TOOL_SCHEMAS

    def _tools_for_scope(self, scope: str):
        """Legacy — kept for backward compat; use tool_schemas_for_scope() instead."""
        return tool_schemas_for_scope(scope)

    def _execute_tool(self, tool_name: str, args: dict, scope: str) -> dict:
        try:
            if tool_name == "search_repo":
                results = self.gateway.search_repo(args.get("query"), top_k=args.get("top_k", 5))
                return {"results": results, "count": len(results)}

            elif tool_name == "open_file":
                return self.gateway.open_file(
                    args.get("path"),
                    args.get("start_line"),
                    args.get("end_line"),
                )

            elif tool_name == "get_issues":
                if scope == "files-only":
                    return {"error": "get_issues not available in files-only scope"}
                results = self.gateway.get_issues(
                    query=args.get("query"),
                    state=args.get("state", "open"),
                    limit=args.get("limit", 10),
                )
                return {"issues": results, "count": len(results)}

            elif tool_name == "get_pull_requests":
                if scope == "files-only":
                    return {"error": "get_pull_requests not available in files-only scope"}
                results = self.gateway.get_pull_requests(
                    query=args.get("query"),
                    state=args.get("state", "open"),
                    limit=args.get("limit", 10),
                )
                return {"pull_requests": results, "count": len(results)}

            elif tool_name == "get_repo_stats":
                return self.gateway.stats()

            elif tool_name == "list_files":
                files = self.gateway.list_files(
                    path_prefix=args.get("path_prefix"),
                    extensions=args.get("extensions"),
                )
                return {"files": files, "count": len(files)}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": str(e)}
