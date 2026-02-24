import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google import genai
from google.genai import types

from src.tool_gateway import ToolGateway
from src.session_manager import SessionManager


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


class AgentOrchestrator:

    def __init__(self, gateway: ToolGateway, session: Optional[SessionManager] = None,
                 api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.gateway = gateway
        self.session = session
        self.model = model

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        self.client = genai.Client(api_key=api_key)
        self._tools = self._define_tools()

    def run(self, query: str, mode: str = "explain", scope: str = "include-pr",
            max_turns: int = 10, verbose: bool = False) -> OrchestratorResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")

        system_prompt = self._build_system_prompt(mode, scope)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=system_prompt + "\n\nRequest: " + query)],
            )
        ]

        executed_calls: list[ExecutedToolCall] = []
        raw_plan: list[ToolCallSpec] = []
        final_text = ""

        turn = 0
        while turn < max_turns:
            turn += 1
            if verbose:
                print(f"\n[Orchestrator] Turn {turn}")

            tools = self._tools_for_scope(scope)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(tools=tools, temperature=0.2),
            )
            time.sleep(1)

            if not response.candidates or not response.candidates[0].content.parts:
                break

            function_calls = []
            text_parts = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_parts.append(part.text)

            if function_calls:
                if turn == 1:
                    raw_plan = [
                        ToolCallSpec(tool_name=fc.name, args=dict(fc.args))
                        for fc in function_calls
                    ]

                if text_parts:
                    final_text = "\n".join(text_parts)

                contents.append(response.candidates[0].content)
                response_parts = []

                for fc in function_calls:
                    args = dict(fc.args)
                    if verbose:
                        print(f"  [Tool] {fc.name}({args})")

                    result = self._execute_tool(fc.name, args, scope)
                    error = result.get("error") if isinstance(result, dict) else None
                    executed_calls.append(
                        ExecutedToolCall(tool_name=fc.name, args=args, result=result, error=error)
                    )

                    if verbose and error:
                        print(f"    [Error] {error}")

                    response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name, response=result
                            )
                        )
                    )

                contents.append(types.Content(role="user", parts=response_parts))

            elif text_parts:
                final_text = "\n".join(text_parts)
                if verbose:
                    print("[Orchestrator] Final answer received.")
                break

        if not final_text.strip():
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text="Please provide your final answer based on the evidence gathered.")],
            ))
            fallback = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            time.sleep(1)
            for part in fallback.candidates[0].content.parts:
                if part.text:
                    final_text += part.text

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
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_repo",
                        description="Search the repository code for files and functions matching a query.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "What to search for"},
                                "top_k": {"type": "integer", "description": "Max results", "default": 5},
                            },
                            "required": ["query"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="open_file",
                        description="Read the contents of a specific file or line range.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path relative to repo root"},
                                "start_line": {"type": "integer", "description": "Start line (1-indexed)"},
                                "end_line": {"type": "integer", "description": "End line (inclusive)"},
                            },
                            "required": ["path"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="get_issues",
                        description="Get GitHub issues for the repository.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search text"},
                                "state": {"type": "string", "description": "open | closed | all", "default": "open"},
                                "limit": {"type": "integer", "description": "Max results", "default": 10},
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="get_pull_requests",
                        description="Get GitHub pull requests for the repository.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search text"},
                                "state": {"type": "string", "description": "open | closed | all", "default": "open"},
                                "limit": {"type": "integer", "description": "Max results", "default": 10},
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="get_repo_stats",
                        description="Get repository statistics.",
                        parameters={"type": "object", "properties": {}},
                    ),
                    types.FunctionDeclaration(
                        name="list_files",
                        description="List all files in the repository. Use this first to understand the project structure before searching or opening files.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path_prefix": {"type": "string", "description": "Only list files under this directory (e.g. 'src/')"},
                                "extensions": {"type": "array", "items": {"type": "string"}, "description": "Filter by file extensions (e.g. ['.ts', '.py'])"},
                            },
                        },
                    ),
                ]
            )
        ]

    def _tools_for_scope(self, scope: str):
        if scope == "files-only":
            return [
                types.Tool(
                    function_declarations=[
                        decl
                        for decl in self._tools[0].function_declarations
                        if decl.name not in ("get_issues", "get_pull_requests")  # list_files always available
                    ]
                )
            ]
        return self._tools

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
