"""
Evaluate a fine-tuned LoRA checkpoint via Tinker SamplingClient.

Loads a *persisted* checkpoint by its tinker:// path (no retraining needed),
creates a SamplingClient, runs inference on each eval task, then generates
eval artifacts and computes the scorecard using the same metrics as the
baseline (Gemini) evaluation.

Usage:
    python -m src.training.eval_sft \
        --tasks       eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers     eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --spec        eval/eval_spec.json \
        --checkpoint  tinker://RUN_ID:train:0/weights/sft-epoch-7 \
        --output-dir  eval/prfconnect/sft_results \
        --baseline    eval/prfconnect/real_results/output.prfc-connect.auto.json

    # Or read checkpoint path from the file saved by sft_train.py:
    python -m src.training.eval_sft \
        --tasks       eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers     eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --spec        eval/eval_spec.json \
        --checkpoint  training_data/last_checkpoint.txt \
        --output-dir  eval/prfconnect/sft_results \
        --baseline    eval/prfconnect/real_results/output.prfc-connect.auto.json
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
        "--checkpoint", required=True,
        help="tinker:// checkpoint path, or a file that contains one "
             "(e.g. training_data/last_checkpoint.txt)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name (for labelling)")
    parser.add_argument("--output-dir", required=True, help="Directory for eval outputs")
    parser.add_argument("--baseline", default=None, help="Optional baseline output.json for comparison")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N tasks")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    # ── Connect to Tinker & create sampling client ──────────────────
    print("Connecting to Tinker...")
    service = tinker.ServiceClient()

    # Load persisted checkpoint into a training client, then get a sampling client.
    # (create_sampling_client needs sampler_weights, but save_state creates weights;
    #  this route converts on the fly.)
    print("Loading checkpoint into training client...")
    tc = service.create_training_client_from_state(path=ckpt_path)
    tokenizer = tc.get_tokenizer()
    print(f"Tokenizer loaded ({args.model})")

    print("Creating sampling client from loaded weights...")
    sampling_client = tc.save_weights_and_get_sampling_client()
    print("Sampling client ready.")

    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=["<|im_end|>"],
    )

    # ── Inference loop ──────────────────────────────────────────────
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

        # Pre-fill assistant turn with closed <think> block to skip thinking
        prompt_text = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n\n"
        )
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt = types.ModelInput.from_ints(tokens=prompt_tokens)

        print(f"  [{idx}/{len(tasks)}] {task_id} ({mode})...", end=" ", flush=True)
        start_ms = time.time() * 1000

        try:
            result = sampling_client.sample(prompt, 1, sampling_params).result()
            seq = result.sequences[0]
            answer_text = tokenizer.decode(seq.tokens).strip()
            # Strip Qwen3 thinking blocks (closed or unclosed)
            answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()
            answer_text = re.sub(r"<think>.*", "", answer_text, flags=re.DOTALL).strip()
            for stop_tok in ["<|im_end|>", "<|endoftext|>"]:
                if answer_text.endswith(stop_tok):
                    answer_text = answer_text[: -len(stop_tok)].strip()
            status = "ok"
            error = None
        except Exception as e:
            answer_text = ""
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
            "tool_call_count": 0,
            "latency_ms": elapsed_ms,
            "started_at": "",
            "finished_at": "",
            "session_id": "sft-eval",
        }
        agent_outputs.append(row)

        wc = len(answer_text.split())
        status_str = "ok" if status == "ok" else f"ERR: {error}"
        print(f"{status_str} ({wc} words, {elapsed_ms}ms)")

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
