"""
Evaluate a fine-tuned LoRA checkpoint via Tinker SamplingClient.

Loads a *persisted* checkpoint by its tinker:// path (no retraining needed),
creates a SamplingClient, and runs inference on each eval task using a
tool-calling ReAct loop (if --repo-path is given) or single-shot generation.

The model can call search_repo, open_file, list_files, etc. via ToolGateway
to ground its answers — the same tools the Gemini agent uses.

Usage:
    # With tool calling (recommended):
    python -m src.training.eval_sft \
        --tasks       eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers     eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --spec        eval/eval_spec.json \
        --checkpoint  training_data/last_checkpoint.txt \
        --repo-path   hack4impact-repos/prfc-connect \
        --output-dir  eval/prfconnect/sft_results \
        --baseline    eval/prfconnect/real_results/output.prfc-connect.auto.json

    # Without tool calling (single-shot, like old behavior):
    python -m src.training.eval_sft \
        --tasks       eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers     eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --spec        eval/eval_spec.json \
        --checkpoint  training_data/last_checkpoint.txt \
        --output-dir  eval/prfconnect/sft_results
"""

import argparse
import json
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import tinker
from tinker import types


# ── helpers ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_checkpoint(raw: str) -> str:
    """Accept a tinker:// URI *or* a path to a file that contains one."""
    if raw.startswith("tinker://"):
        return raw.strip()
    p = Path(raw)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    raise ValueError(
        f"--checkpoint must be a tinker:// URI or a file containing one (got {raw!r})"
    )


def extract_file_paths(text: str) -> list[str]:
    pattern = r"\b(?:src|docs|prisma|test|e2e|public|types)/[A-Za-z0-9_\-./\[\]]+\.[A-Za-z0-9]+"
    paths = re.findall(pattern, text or "")
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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
        "For each suggestion include an impact label (high/medium/low) and an effort label (high/medium/low)."
    ),
    "patch": (
        "Propose a code change that addresses the request. "
        "Output the change as a unified diff (patch format) after your explanation."
    ),
}


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SFT model")
    parser.add_argument("--tasks", required=True, help="Tasks JSONL path")
    parser.add_argument("--answers", required=True, help="Answer key JSONL path")
    parser.add_argument("--spec", required=True, help="eval_spec.json path")
    parser.add_argument(
        "--checkpoint", required=False, default=None,
        help="tinker:// checkpoint path, or a file that contains one "
             "(e.g. training_data/last_checkpoint.txt)",
    )
    parser.add_argument("--base-only", action="store_true",
                        help="Evaluate the base model without any LoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name (for labelling)")
    parser.add_argument("--output-dir", required=True, help="Directory for eval outputs")
    parser.add_argument("--baseline", default=None, help="Optional baseline output.json for comparison")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N tasks")
    parser.add_argument(
        "--repo-path", default=None,
        help="Path to the target repository for tool-calling eval. "
             "If provided, the model can use search_repo, open_file, etc. "
             "If omitted, falls back to single-shot generation.",
    )
    parser.add_argument("--max-tool-turns", type=int, default=8, help="Max ReAct loop turns per task")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    # ── Connect to Tinker & create backend ──────────────────────────
    print("Connecting to Tinker...")
    from src.model_backend import QwenTinkerBackend, tool_schemas_for_scope, ToolResult

    service = tinker.ServiceClient()

    if args.base_only:
        # Base model eval: create a fresh LoRA adapter (B=0, so output = base model)
        print(f"Creating base model client for {args.model} (no fine-tuning)...")
        tc = service.create_lora_training_client(
            base_model=args.model,
            rank=32,
        )
        model_label = f"{args.model} (base)"
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --base-only is set")
        ckpt_path = resolve_checkpoint(args.checkpoint)
        print(f"Checkpoint: {ckpt_path}")
        print("Loading checkpoint into training client...")
        tc = service.create_training_client_from_state(path=ckpt_path)
        model_label = f"{args.model}+LoRA"

    tokenizer = tc.get_tokenizer()
    print(f"Tokenizer loaded ({args.model})")

    print("Creating sampling client...")
    sampling_client = tc.save_weights_and_get_sampling_client()
    print("Sampling client ready.")

    backend = QwenTinkerBackend(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        model_label=model_label,
        max_tokens=args.max_tokens,
        disable_thinking=True,
    )

    # ── Set up tool gateway ─────────────────────────────────────────
    gateway = None
    if args.repo_path:
        from src.tool_gateway import ToolGateway
        try:
            gateway = ToolGateway(repo_path=args.repo_path)
            stats = gateway.stats()
            print(f"Tool gateway loaded: {stats.get('total_files', 0)} files, "
                  f"{stats.get('total_chunks', 0)} chunks indexed")
        except Exception as e:
            print(f"Warning: Could not load tool gateway ({e}). Falling back to single-shot.")
            gateway = None
    else:
        print("No --repo-path given; using single-shot generation (no tool calling).")

    tools = tool_schemas_for_scope("include-pr") if gateway else []

    def execute_tool(tool_name: str, args_dict: dict) -> dict:
        """Execute a tool call via ToolGateway."""
        if gateway is None:
            return {"error": "No tool gateway available"}
        try:
            if tool_name == "search_repo":
                results = gateway.search_repo(args_dict.get("query"), top_k=args_dict.get("top_k", 5))
                return {"results": results, "count": len(results)}
            elif tool_name == "open_file":
                return gateway.open_file(args_dict.get("path"), args_dict.get("start_line"), args_dict.get("end_line"))
            elif tool_name == "list_files":
                files = gateway.list_files(path_prefix=args_dict.get("path_prefix"), extensions=args_dict.get("extensions"))
                return {"files": files, "count": len(files)}
            elif tool_name == "get_repo_stats":
                return gateway.stats()
            elif tool_name == "get_issues":
                results = gateway.get_issues(query=args_dict.get("query"), state=args_dict.get("state", "open"), limit=args_dict.get("limit", 10))
                return {"issues": results, "count": len(results)}
            elif tool_name == "get_pull_requests":
                results = gateway.get_pull_requests(query=args_dict.get("query"), state=args_dict.get("state", "open"), limit=args_dict.get("limit", 10))
                return {"pull_requests": results, "count": len(results)}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    # ── Inference loop (ReAct with tools) ───────────────────────────
    agent_outputs: list[dict] = []
    for idx, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        question = task.get("question", "")
        mode = task.get("mode", "explain")

        system_msg = (
            f"You are an expert repository assistant. "
            f"Mode: {mode.upper()}. {MODE_INSTRUCTIONS.get(mode, '')}\n"
            f"Always cite specific files and line numbers."
        )

        print(f"  [{idx}/{len(tasks)}] {task_id} ({mode})...", end=" ", flush=True)
        start_ms = time.time() * 1000

        try:
            messages = [
                backend.format_system_message(system_msg),
                backend.format_user_message(question),
            ]

            answer_text = ""
            total_tool_calls = 0

            # ReAct loop
            for t in range(args.max_tool_turns):
                model_turn = backend.generate(messages, tools, temperature=args.temperature)

                if model_turn.has_tool_calls and gateway is not None:
                    messages.append(backend.format_assistant_message(model_turn))
                    tool_results = []
                    for tc_item in model_turn.tool_calls:
                        result = execute_tool(tc_item.name, tc_item.arguments)
                        tool_results.append(ToolResult(name=tc_item.name, result=result))
                        total_tool_calls += 1
                    formatted = backend.format_tool_results(tool_results)
                    if isinstance(formatted, list):
                        messages.extend(formatted)
                    else:
                        messages.append(formatted)
                else:
                    answer_text = model_turn.text or ""
                    break

            if not answer_text.strip():
                messages.append(backend.format_user_message(
                    "Please provide your final answer based on the evidence gathered."
                ))
                final_turn = backend.generate(messages, tools=[], temperature=args.temperature)
                answer_text = final_turn.text or ""

            status = "ok"
            error = None
        except Exception as e:
            answer_text = ""
            total_tool_calls = 0
            status = "error"
            error = str(e)

        elapsed_ms = int(time.time() * 1000 - start_ms)

        paths = extract_file_paths(answer_text) if status == "ok" else []
        citations = [{"source_type": "file", "file_path": fp, "snippet": ""} for fp in paths]

        row = {
            "task_id": task_id,
            "repo": task.get("repo", ""),
            "model": f"{args.model}+LoRA",
            "mode": mode,
            "scope": "include-pr",
            "status": status,
            "error": error,
            "question": question,
            "answer_text": answer_text,
            "citations": citations,
            "patch_diff": None,
            "next_actions": [],
            "tool_call_count": total_tool_calls,
            "latency_ms": elapsed_ms,
            "started_at": "",
            "finished_at": "",
            "session_id": "sft-eval-tools" if gateway else "sft-eval",
        }
        agent_outputs.append(row)

        wc = len(answer_text.split())
        status_str = "ok" if status == "ok" else f"ERR: {error}"
        tc_str = f", {total_tool_calls} tool calls" if total_tool_calls else ""
        print(f"{status_str} ({wc} words{tc_str}, {elapsed_ms}ms)")

    # ── Write agent outputs ─────────────────────────────────────────
    agent_out_path = out_dir / "agent_outputs.sft.jsonl"
    with agent_out_path.open("w") as f:
        for row in agent_outputs:
            f.write(json.dumps(row) + "\n")
    print(f"\nAgent outputs: {agent_out_path}")

    # ── Generate eval artifacts ─────────────────────────────────────
    print("Generating eval artifacts...")
    from src.generate_eval_artifacts import main as gen_main
    import sys
    old_argv = sys.argv
    sys.argv = [
        "generate_eval_artifacts",
        "--tasks", args.tasks,
        "--answers", args.answers,
        "--agent-outputs", str(agent_out_path),
        "--ratings-out", str(out_dir / "ratings.sft.jsonl"),
        "--retrieval-out", str(out_dir / "retrieval.sft.jsonl"),
        "--run-results-out", str(out_dir / "run_results.sft.jsonl"),
        "--rater-id", "auto_rater_sft_v1",
    ]
    gen_main()
    sys.argv = old_argv

    # ── Compute scorecard ───────────────────────────────────────────
    print("Computing scorecard...")
    from src.eval_runner import (
        build_scorecard, load_json, load_jsonl as eval_load_jsonl, compare_to_baseline,
    )

    spec = load_json(args.spec)
    ratings = eval_load_jsonl(str(out_dir / "ratings.sft.jsonl"))
    results = eval_load_jsonl(str(out_dir / "run_results.sft.jsonl"))
    retrieval = eval_load_jsonl(str(out_dir / "retrieval.sft.jsonl"))

    report = build_scorecard(spec, ratings, results, retrieval)

    if args.baseline:
        baseline = load_json(args.baseline)
        report["delta_vs_baseline"] = compare_to_baseline(report, baseline)

    report_path = out_dir / "output.sft.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nScorecard: {report_path}")

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SFT MODEL EVALUATION RESULTS")
    print("=" * 60)
    q = report.get("quality", {})
    print(f"  Pass rate:        {q.get('pass_rate', 'N/A')}")
    print(f"  Rubric mean:      {q.get('rubric_mean_total', 'N/A')}")
    print(f"  Critical errors:  {q.get('critical_error_rate', 'N/A')}")

    bc = q.get("by_criterion", {})
    for k in ["correctness", "grounding", "relevance", "clarity"]:
        v = bc.get(k, {})
        print(f"    {k}: {v.get('mean', 'N/A')}")

    ts = report.get("task_success", {})
    print(f"  Success rate:     {ts.get('success_rate', 'N/A')}")
    print(f"  Exact match:      {ts.get('exact_match_rate', 'N/A')}")

    r = report.get("retrieval", {})
    print(f"  P@3:              {r.get('p@3', 'N/A')}")
    print(f"  R@3:              {r.get('r@3', 'N/A')}")
    print(f"  nDCG@3:           {r.get('ndcg@3', 'N/A')}")

    if "delta_vs_baseline" in report:
        print("\n  Δ vs Baseline (Gemini):")
        deltas = report["delta_vs_baseline"]
        for k in [
            "quality.pass_rate", "quality.rubric_mean_total",
            "task_success.success_rate",
            "retrieval.p@3", "retrieval.r@3", "retrieval.ndcg@3",
        ]:
            d = deltas.get(k)
            if d is not None:
                sign = "+" if d >= 0 else ""
                print(f"    {k}: {sign}{d:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
