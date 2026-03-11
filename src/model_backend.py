"""
Model backend abstraction for repo-assist.

Provides a pluggable interface so the AgentOrchestrator can use any LLM
(Gemini, Qwen via Tinker, OpenAI, etc.) without changes to the tool-execution
or orchestration logic.

Each backend implements:
  - generate():  one model turn (may produce tool calls or a text response)
  - build_tool_definitions():  format tool schemas for the model
  - build_system_prompt():  wrap system text in the model's expected format

The orchestrator owns the ReAct loop and tool execution; the backend just
handles model I/O and response parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Shared types ────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    """One model response — either text, tool calls, or both."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass
class ToolResult:
    """Result of executing a tool, fed back to the model."""
    name: str
    result: dict[str, Any]


# ── Tool schema (model-agnostic) ───────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_repo",
        "description": "Search the repository code for files and functions matching a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "open_file",
        "description": "Read the contents of a specific file or line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "start_line": {"type": "integer", "description": "Start line (1-indexed)"},
                "end_line": {"type": "integer", "description": "End line (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List all files in the repository. Use this first to understand "
            "the project structure before searching or opening files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Only list files under this directory (e.g. 'src/')",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by file extensions (e.g. ['.ts', '.py'])",
                },
            },
        },
    },
    {
        "name": "get_issues",
        "description": "Get GitHub issues for the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "state": {"type": "string", "description": "open | closed | all"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
        },
    },
    {
        "name": "get_pull_requests",
        "description": "Get GitHub pull requests for the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text"},
                "state": {"type": "string", "description": "open | closed | all"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
        },
    },
    {
        "name": "get_repo_stats",
        "description": "Get repository statistics (file count, chunks, issues, PRs).",
        "parameters": {"type": "object", "properties": {}},
    },
]


def tool_schemas_for_scope(scope: str) -> list[dict[str, Any]]:
    """Filter tool schemas by scope (files-only excludes GitHub tools)."""
    if scope == "files-only":
        excluded = {"get_issues", "get_pull_requests"}
        return [t for t in TOOL_SCHEMAS if t["name"] not in excluded]
    return TOOL_SCHEMAS


# ── Abstract base class ────────────────────────────────────────────

class ModelBackend(ABC):
    """Abstract interface for an LLM that can make tool calls."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ModelTurn:
        """Run one model turn.

        Args:
            messages: Conversation history in the backend's native format.
            tools: Tool schemas (from TOOL_SCHEMAS).
            temperature: Sampling temperature.

        Returns:
            A ModelTurn with text and/or tool_calls.
        """
        ...

    @abstractmethod
    def format_system_message(self, text: str) -> dict[str, Any]:
        """Wrap system prompt text in the backend's message format."""
        ...

    @abstractmethod
    def format_user_message(self, text: str) -> dict[str, Any]:
        """Wrap user text in the backend's message format."""
        ...

    @abstractmethod
    def format_assistant_message(self, turn: ModelTurn) -> dict[str, Any]:
        """Wrap a model response in the backend's message format."""
        ...

    @abstractmethod
    def format_tool_results(self, results: list[ToolResult]) -> dict[str, Any] | list[dict[str, Any]]:
        """Wrap tool execution results in the backend's message format."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    def serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict]:
        """Serialize internal message format into plain dicts for export.

        Default: return as-is.  Backends can override for Gemini Content
        objects, etc.
        """
        return messages


# ── Gemini backend ──────────────────────────────────────────────────

class GeminiBackend(ModelBackend):
    """Google Gemini via google-genai SDK (native function-calling)."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai
        from google.genai import types as gtypes

        self._model = model
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required for GeminiBackend")
        self.client = genai.Client(api_key=api_key)
        self._gtypes = gtypes
        self._genai = genai

    @property
    def model_name(self) -> str:
        return self._model

    # -- Gemini-native tool format ------------------------------------------------

    def _to_gemini_tools(self, tools: list[dict]) -> list:
        gtypes = self._gtypes
        declarations = []
        for t in tools:
            declarations.append(
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"],
                )
            )
        return [gtypes.Tool(function_declarations=declarations)]

    # -- Message formatting -------------------------------------------------------

    def format_system_message(self, text: str) -> dict[str, Any]:
        gtypes = self._gtypes
        return {
            "_gemini_content": gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=text)],
            )
        }

    def format_user_message(self, text: str) -> dict[str, Any]:
        gtypes = self._gtypes
        return {
            "_gemini_content": gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=text)],
            )
        }

    def format_assistant_message(self, turn: ModelTurn) -> dict[str, Any]:
        # We don't manually build assistant messages — Gemini returns them.
        # This is only needed if we want to inject; usually we store the raw
        # Content object returned by the API.
        gtypes = self._gtypes
        parts = []
        if turn.text:
            parts.append(gtypes.Part(text=turn.text))
        for tc in turn.tool_calls:
            parts.append(gtypes.Part(
                function_call=gtypes.FunctionCall(name=tc.name, args=tc.arguments)
            ))
        return {"_gemini_content": gtypes.Content(role="model", parts=parts)}

    def format_tool_results(self, results: list[ToolResult]) -> dict[str, Any]:
        gtypes = self._gtypes
        parts = [
            gtypes.Part(
                function_response=gtypes.FunctionResponse(name=r.name, response=r.result)
            )
            for r in results
        ]
        return {"_gemini_content": gtypes.Content(role="user", parts=parts)}

    def _extract_contents(self, messages: list[dict]) -> list:
        return [m["_gemini_content"] for m in messages]

    # -- Generation ---------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ModelTurn:
        gtypes = self._gtypes
        contents = self._extract_contents(messages)
        gemini_tools = self._to_gemini_tools(tools)

        response = self.client.models.generate_content(
            model=self._model,
            contents=contents,
            config=gtypes.GenerateContentConfig(tools=gemini_tools, temperature=temperature),
        )
        time.sleep(1)  # rate limit

        if not response.candidates or not response.candidates[0].content.parts:
            return ModelTurn()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_calls.append(
                    ToolCall(name=part.function_call.name, arguments=dict(part.function_call.args))
                )
            elif part.text:
                text_parts.append(part.text)

        # Also store the raw Content for conversation threading
        raw_content = response.candidates[0].content

        turn = ModelTurn(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
        )
        # Attach raw content so the orchestrator can append it directly
        turn._raw_gemini_content = raw_content  # type: ignore[attr-defined]
        return turn

    def serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict]:
        """Serialize Gemini Content objects to plain dicts."""
        turns = []
        for msg in messages:
            content = msg.get("_gemini_content")
            if content is None:
                turns.append(msg)
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


# ── Qwen/Tinker backend ────────────────────────────────────────────

# Qwen3 tool-calling protocol:
#   System message includes tool definitions in a specific XML-ish format.
#   The model emits tool calls with:  <tool_call>\n{"name":..., "arguments":...}\n</tool_call>
#   Tool results are fed back as:     <tool_response>\n{...}\n</tool_response>

QWEN_TOOL_PREAMBLE = (
    "You are a helpful assistant with access to the following tools. "
    "You can call one or more tools to assist with the user's query. "
    "You are provided with tool descriptions and their parameters.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tool_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{\"name\": \"<function-name>\", \"arguments\": {{<args-json-object>}}}}\n'
    "</tool_call>"
)


def _build_qwen_tool_block(tools: list[dict]) -> str:
    """Build the Qwen3 tool definition block for the system prompt."""
    tool_defs = []
    for t in tools:
        td = {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        tool_defs.append(json.dumps(td))
    tool_json = "\n".join(tool_defs)
    return QWEN_TOOL_PREAMBLE.format(tool_json=tool_json)


def _parse_qwen_tool_calls(text: str) -> tuple[str | None, list[ToolCall]]:
    """Parse Qwen3 model output for <tool_call> blocks.

    Returns (remaining_text, list_of_tool_calls).
    """
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    tool_calls: list[ToolCall] = []
    remaining = text

    for match in pattern.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append(ToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, KeyError):
            continue

    # Remove tool_call blocks from remaining text
    remaining = pattern.sub("", remaining).strip()
    remaining = remaining if remaining else None

    return remaining, tool_calls


class QwenTinkerBackend(ModelBackend):
    """Qwen3 via Tinker SamplingClient (supports tool-calling ReAct loop).

    This backend works with a pre-loaded Tinker SamplingClient (e.g., from a
    fine-tuned LoRA checkpoint).  It constructs prompts using Qwen3's chat
    template and parses <tool_call> blocks from the output.
    """

    def __init__(
        self,
        sampling_client,
        tokenizer,
        model_label: str = "Qwen/Qwen3-8B+LoRA",
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ):
        self._sampling_client = sampling_client
        self._tokenizer = tokenizer
        self._model_label = model_label
        self._max_tokens = max_tokens
        self._disable_thinking = disable_thinking

    @property
    def model_name(self) -> str:
        return self._model_label

    # -- Message formatting (plain dict with role/content) -----------------------

    def format_system_message(self, text: str) -> dict[str, Any]:
        return {"role": "system", "content": text}

    def format_user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def format_assistant_message(self, turn: ModelTurn) -> dict[str, Any]:
        parts: list[str] = []
        if turn.text:
            parts.append(turn.text)
        for tc in turn.tool_calls:
            tc_json = json.dumps({"name": tc.name, "arguments": tc.arguments})
            parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
        return {"role": "assistant", "content": "\n".join(parts)}

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        """Each tool result becomes a user message with <tool_response> tags."""
        msgs = []
        for r in results:
            resp_str = json.dumps(r.result, default=str)
            # Truncate very large tool results
            if len(resp_str) > 4000:
                resp_str = resp_str[:4000] + "...(truncated)"
            msgs.append({
                "role": "user",
                "content": f"<tool_response>\n{resp_str}\n</tool_response>",
            })
        return msgs

    def _build_prompt_text(self, messages: list[dict], tools: list[dict]) -> str:
        """Build a Qwen3 chat-template prompt string from messages."""
        parts: list[str] = []
        tool_block = _build_qwen_tool_block(tools)

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # Prepend tool definitions to the system prompt
                combined = tool_block + "\n\n" + content
                parts.append(f"<|im_start|>system\n{combined}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                # Wrap in <think> to match the renderer's training format
                parts.append(f"<|im_start|>assistant\n<think>\n{content}<|im_end|>")

        # Open the assistant turn
        # The qwen3 renderer wraps assistant content inside <think> tags,
        # so we just open <think> and let the model generate from there.
        # Tool calls appear inside <think> in the training data.
        parts.append("<|im_start|>assistant\n<think>\n")

        return "\n".join(parts)

    # -- Generation ---------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ModelTurn:
        from tinker import types as ttypes

        prompt_text = self._build_prompt_text(messages, tools)
        prompt_tokens = self._tokenizer.encode(prompt_text)
        prompt = ttypes.ModelInput.from_ints(tokens=prompt_tokens)

        sampling_params = ttypes.SamplingParams(
            max_tokens=self._max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )

        result = self._sampling_client.sample(prompt, 1, sampling_params).result()
        raw_text = self._tokenizer.decode(result.sequences[0].tokens).strip()

        # Strip stop tokens
        for stop_tok in ["<|im_end|>", "<|endoftext|>"]:
            if raw_text.endswith(stop_tok):
                raw_text = raw_text[: -len(stop_tok)].strip()

        # Parse tool calls FIRST (before stripping <think>), because
        # the qwen3 renderer puts <tool_call> blocks inside <think>
        remaining_text, tool_calls = _parse_qwen_tool_calls(raw_text)

        # Now strip thinking blocks from the remaining text
        if remaining_text:
            remaining_text = re.sub(r"<think>.*?</think>", "", remaining_text, flags=re.DOTALL).strip()
            remaining_text = re.sub(r"<think>.*", "", remaining_text, flags=re.DOTALL).strip()
            # Also strip </think> that might appear before the real answer
            remaining_text = remaining_text.replace("</think>", "").strip()
            remaining_text = remaining_text if remaining_text else None

        return ModelTurn(text=remaining_text, tool_calls=tool_calls)


# ── Factory ─────────────────────────────────────────────────────────

def create_backend(
    backend_type: str,
    *,
    # Gemini params
    model: str | None = None,
    api_key: str | None = None,
    # Qwen/Tinker params
    checkpoint: str | None = None,
    base_model: str | None = None,
    max_tokens: int = 2048,
    disable_thinking: bool = True,
    # Pre-built Tinker objects (for use in training scripts)
    sampling_client: Any | None = None,
    tokenizer: Any | None = None,
) -> ModelBackend:
    """Create a model backend by type.

    Args:
        backend_type: "gemini" or "qwen"
        model: Gemini model name (for gemini backend)
        api_key: API key (for gemini backend)
        checkpoint: tinker:// checkpoint path or file containing one (for qwen backend)
        base_model: Base model name like "Qwen/Qwen3-8B" (for qwen backend)
        max_tokens: Max generation tokens (for qwen backend)
        disable_thinking: Skip Qwen3 thinking mode (for qwen backend)
        sampling_client: Pre-built Tinker SamplingClient (for qwen backend)
        tokenizer: Pre-built tokenizer (for qwen backend)
    """
    if backend_type == "gemini":
        return GeminiBackend(
            model=model or "gemini-2.5-flash",
            api_key=api_key,
        )

    elif backend_type == "qwen":
        if sampling_client is not None and tokenizer is not None:
            return QwenTinkerBackend(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                model_label=base_model or "Qwen/Qwen3-8B+LoRA",
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )

        # Build from checkpoint
        if checkpoint is None:
            raise ValueError("QwenTinkerBackend requires --checkpoint or pre-built sampling_client")

        import tinker

        # Resolve checkpoint path
        ckpt = checkpoint.strip()
        if not ckpt.startswith("tinker://"):
            from pathlib import Path
            p = Path(ckpt)
            if p.is_file():
                ckpt = p.read_text(encoding="utf-8").strip()
            else:
                raise ValueError(f"checkpoint must be tinker:// URI or file (got {ckpt!r})")

        service = tinker.ServiceClient()
        tc = service.create_training_client_from_state(path=ckpt)
        tok = tc.get_tokenizer()
        sc = tc.save_weights_and_get_sampling_client()

        return QwenTinkerBackend(
            sampling_client=sc,
            tokenizer=tok,
            model_label=f"{base_model or 'Qwen/Qwen3-8B'}+LoRA",
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Use 'gemini' or 'qwen'.")
