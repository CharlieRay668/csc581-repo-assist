"""
Convert agent output JSONL (from run_task_batch) into Tinker-compatible
conversations.jsonl for supervised fine-tuning.

Usage:
    python -m src.convert_trajectories \
        --input  eval/prfconnect/real_results/agent_outputs.prfc-connect.heldout.jsonl \
        --output training_data/conversations.jsonl \
        --mode   explain          # optional: filter by mode
"""

import argparse
import json
from pathlib import Path


def build_system_message(row: dict) -> str:
    """Reconstruct a minimal system prompt from task metadata."""
    mode = row.get("mode", "explain")
    mode_instructions = {
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
            "For each suggestion include an impact label (high/medium/low) and an effort label (high/medium/low)."
        ),
        "patch": (
            "Propose a code change that addresses the request. "
            "Output the change as a unified diff (patch format) after your explanation."
        ),
    }
    return (
        f"You are an expert repository assistant. "
        f"Mode: {mode.upper()}. {mode_instructions.get(mode, '')}\n"
        f"Always cite specific files and line numbers."
    )


def convert_raw_turns(raw_turns: list[dict]) -> list[dict]:
    """Convert raw_turns (serialized Gemini contents) to chat messages."""
    messages = []
    for turn in raw_turns:
        role = turn.get("role", "user")
        parts = turn.get("parts", [])

        for part in parts:
            ptype = part.get("type")

            if ptype == "text":
                chat_role = "assistant" if role == "model" else "user"
                messages.append({"role": chat_role, "content": part["text"]})

            elif ptype == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "name": part["name"],
                        "arguments": part.get("args", {}),
                    }],
                })

            elif ptype == "function_response":
                # Truncate large tool results to keep dataset manageable
                resp = part.get("response", {})
                resp_str = json.dumps(resp, default=str)
                if len(resp_str) > 2000:
                    resp_str = resp_str[:2000] + "...(truncated)"
                messages.append({
                    "role": "tool",
                    "name": part["name"],
                    "content": resp_str,
                })

    return messages


def convert_from_tool_calls(row: dict) -> list[dict]:
    """Fallback: build messages from tool_calls + answer when raw_turns is absent."""
    messages = []
    for tc in row.get("tool_calls", []):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "name": tc["tool_name"],
                "arguments": tc.get("args", {}),
            }],
        })
        result_str = json.dumps(tc.get("result", {}), default=str)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "...(truncated)"
        messages.append({
            "role": "tool",
            "name": tc["tool_name"],
            "content": result_str,
        })
    return messages


def convert_row(row: dict) -> dict | None:
    """Convert a single agent output row to a conversations.jsonl entry."""
    if row.get("status") != "ok":
        return None

    question = row.get("question", "")
    answer = row.get("answer_text", "")
    if not question or not answer:
        return None

    system_msg = build_system_message(row)
    messages = [{"role": "system", "content": system_msg}]
    messages.append({"role": "user", "content": question})

    # Prefer raw_turns if available, else fall back to tool_calls
    raw_turns = row.get("raw_turns", [])
    if raw_turns:
        messages.extend(convert_raw_turns(raw_turns))
    else:
        messages.extend(convert_from_tool_calls(row))

    # Always append the final answer so the conversation ends with role=assistant
    messages.append({"role": "assistant", "content": answer})

    return {
        "task_id": row.get("task_id"),
        "mode": row.get("mode"),
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert agent outputs to Tinker conversations.jsonl for SFT.",
    )
    parser.add_argument("--input", required=True, help="Agent output JSONL from run_task_batch")
    parser.add_argument("--output", required=True, help="Output conversations.jsonl path")
    parser.add_argument("--mode", default=None, help="Filter to a specific mode (explain/locate/suggest/patch)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    if args.mode:
        rows = [r for r in rows if r.get("mode") == args.mode]

    converted = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            entry = convert_row(row)
            if entry is None:
                skipped += 1
                continue
            out.write(json.dumps(entry) + "\n")
            converted += 1

    print(f"Converted {converted} trajectories, skipped {skipped}.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
