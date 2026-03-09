"""
Supervised fine-tuning via Tinker API using cross-entropy loss.

Reads conversation trajectories (from convert_trajectories.py) and fine-tunes
a LoRA adapter on a Tinker-supported base model.

Usage:
    python -m src.training.sft_train \
        --data   training_data/conversations.jsonl \
        --model  Qwen/Qwen3-8B \
        --epochs 3 \
        --lr     1e-4
"""

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import torch
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info


def load_conversations(path: str) -> list[dict]:
    """Load conversations.jsonl → list of {task_id, mode, messages}."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def messages_to_renderer_format(messages: list[dict]) -> list[dict]:
    """Convert our message format to the renderer's expected format.

    The renderer expects: [{role: "system"|"user"|"assistant", content: str}, ...]
    We collapse tool calls/responses into assistant/user turns.
    """
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            out.append({"role": "system", "content": content})
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                tc_str = json.dumps({"name": tc["name"], "arguments": tc["arguments"]})
                out.append({"role": "assistant", "content": tc_str})
            else:
                out.append({"role": "assistant", "content": content or ""})
        elif role == "tool":
            # Tool response → user turn with tool context
            tool_content = msg.get("content", "")
            tool_name = msg.get("name", "tool")
            out.append({"role": "user", "content": f"[{tool_name}]: {tool_content}"})

    return out


def messages_to_datum(
    messages: list[dict], renderer: renderers.Renderer, max_length: int | None = None
) -> types.Datum | None:
    """Convert a message list to a Tinker Datum for SFT."""
    convo = messages_to_renderer_format(messages)

    if len(convo) < 2:
        return None

    # Ensure the last message is from assistant (what we're training on)
    if convo[-1]["role"] != "assistant":
        return None

    try:
        tokens, weights = renderer.build_supervised_example(convo)
    except Exception as e:
        return None

    if len(tokens) < 2:
        return None

    # Truncate if needed
    if max_length is not None:
        tokens = tokens[:max_length]
        weights = weights[:max_length]

    # Standard SFT: predict next token (shift by 1)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens.tolist()),
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": tinker.TensorData(
                data=[int(x) for x in target_tokens.tolist()],
                dtype="int64",
                shape=list(target_tokens.shape),
            ),
        },
    )


def run_post_training_eval(
    training_client,
    tokenizer,
    model_name: str,
    checkpoint_name: str,
    eval_tasks_path: str,
    eval_answers_path: str,
    eval_spec_path: str,
    eval_output_dir: str,
    eval_baseline_path: str | None,
) -> None:
    """Run evaluation immediately after training using the live training session."""
    import re
    import time as _time

    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    out_dir = Path(eval_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = []
    for line in Path(eval_tasks_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} eval tasks")

    # Get a sampling client from the fine-tuned weights
    print("Creating sampling client from trained weights...")
    sampling_client = training_client.save_weights_and_get_sampling_client()
    print("Sampling client ready.")

    sampling_params = types.SamplingParams(
        max_tokens=2048,
        temperature=0.3,
        stop=["<|im_end|>"],
    )

    mode_instructions = {
        "explain": "Provide a thorough explanation with code citations. Reference specific file paths and line numbers.",
        "locate": "Identify exactly which files and line ranges implement the requested functionality. Be concise — list locations first, brief explanation second.",
        "suggest": "Suggest concrete next development steps. For each suggestion include an impact label (high/medium/low) and an effort label (high/medium/low).",
        "patch": "Propose a code change that addresses the request. Output the change as a unified diff (patch format) after your explanation.",
    }

    agent_outputs = []
    for idx, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        question = task.get("question", "")
        mode = task.get("mode", "explain")

        system_msg = (
            f"You are an expert repository assistant. "
            f"Mode: {mode.upper()}. {mode_instructions.get(mode, '')}\n"
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
        start_ms = _time.time() * 1000

        try:
            result = sampling_client.sample(prompt, 1, sampling_params).result()
            seq = result.sequences[0]
            answer_tokens = seq.tokens
            answer_text = tokenizer.decode(answer_tokens).strip()
            # Strip Qwen3 thinking blocks if present (closed or unclosed)
            answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL).strip()
            answer_text = re.sub(r'<think>.*', '', answer_text, flags=re.DOTALL).strip()
            for stop_tok in ["<|im_end|>", "<|endoftext|>"]:
                if answer_text.endswith(stop_tok):
                    answer_text = answer_text[:-len(stop_tok)].strip()
            status = "ok"
            error = None
        except Exception as e:
            answer_text = ""
            status = "error"
            error = str(e)

        elapsed_ms = int(_time.time() * 1000 - start_ms)

        # Extract file path citations from answer text
        paths = re.findall(
            r"\b(?:src|docs|prisma|test|e2e|public|types)/[A-Za-z0-9_\-./\[\]]+\.[A-Za-z0-9]+",
            answer_text or "",
        )
        seen = set()
        citations = []
        for fp in paths:
            if fp not in seen:
                seen.add(fp)
                citations.append({"source_type": "file", "file_path": fp, "snippet": ""})

        row = {
            "task_id": task_id,
            "repo": task.get("repo", ""),
            "model": f"{model_name}+LoRA({checkpoint_name})",
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

    # Write agent outputs
    agent_out_path = out_dir / "agent_outputs.sft.jsonl"
    with agent_out_path.open("w") as f:
        for row in agent_outputs:
            f.write(json.dumps(row) + "\n")
    print(f"\nAgent outputs: {agent_out_path}")

    # Generate eval artifacts
    print("Generating eval artifacts...")
    from src.generate_eval_artifacts import main as gen_main
    import sys
    old_argv = sys.argv
    sys.argv = [
        "generate_eval_artifacts",
        "--tasks", eval_tasks_path,
        "--answers", eval_answers_path,
        "--agent-outputs", str(agent_out_path),
        "--ratings-out", str(out_dir / "ratings.sft.jsonl"),
        "--retrieval-out", str(out_dir / "retrieval.sft.jsonl"),
        "--run-results-out", str(out_dir / "run_results.sft.jsonl"),
        "--rater-id", "auto_rater_sft_v1",
    ]
    gen_main()
    sys.argv = old_argv

    # Compute scorecard
    print("Computing scorecard...")
    from src.eval_runner import build_scorecard, load_json, load_jsonl as eval_load_jsonl, compare_to_baseline

    spec = load_json(eval_spec_path)
    ratings = eval_load_jsonl(str(out_dir / "ratings.sft.jsonl"))
    results = eval_load_jsonl(str(out_dir / "run_results.sft.jsonl"))
    retrieval = eval_load_jsonl(str(out_dir / "retrieval.sft.jsonl"))

    report = build_scorecard(spec, ratings, results, retrieval)

    if eval_baseline_path:
        baseline = load_json(eval_baseline_path)
        report["delta_vs_baseline"] = compare_to_baseline(report, baseline)

    report_path = out_dir / "output.sft.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Scorecard: {report_path}")

    # Print summary
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
        print("\n  Δ vs Baseline (Gemini agent):")
        deltas = report["delta_vs_baseline"]
        for k in ["quality.pass_rate", "quality.rubric_mean_total", "task_success.success_rate",
                   "retrieval.p@3", "retrieval.r@3", "retrieval.ndcg@3"]:
            d = deltas.get(k)
            if d is not None:
                sign = "+" if d >= 0 else ""
                print(f"    {k}: {sign}{d:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SFT training via Tinker")
    parser.add_argument("--data", required=True, help="conversations.jsonl path")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model on Tinker")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--log-path", default="/tmp/tinker-sft", help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print stats only")
    parser.add_argument("--eval-tasks", default=None, help="Tasks JSONL for post-training eval")
    parser.add_argument("--eval-answers", default=None, help="Answer key JSONL for post-training eval")
    parser.add_argument("--eval-spec", default=None, help="eval_spec.json for post-training eval")
    parser.add_argument("--eval-output-dir", default=None, help="Output dir for eval results")
    parser.add_argument("--eval-baseline", default=None, help="Baseline output.json for delta comparison")
    args = parser.parse_args()

    # --- Load data ---
    entries = load_conversations(args.data)
    print(f"Loaded {len(entries)} conversations from {args.data}")

    if args.dry_run:
        lens = [len(e["messages"]) for e in entries]
        print(f"  Turn counts: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.1f}")
        modes = {}
        for e in entries:
            modes[e.get("mode", "?")] = modes.get(e.get("mode", "?"), 0) + 1
        print(f"  Mode distribution: {modes}")
        return

    # --- Connect to Tinker ---
    service = tinker.ServiceClient()  # reads TINKER_API_KEY from env
    training_client = service.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )

    # Get tokenizer and renderer
    tokenizer = training_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # --- Convert to Datums ---
    datums = []
    skipped = 0
    for entry in entries:
        d = messages_to_datum(entry["messages"], renderer)
        if d is not None:
            datums.append(d)
        else:
            skipped += 1

    print(f"Prepared {len(datums)} training examples ({skipped} skipped)")
    if not datums:
        print("No valid training data — exiting.")
        return

    # --- Training loop ---
    log_path = Path(args.log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(datums), args.batch_size):
            batch = datums[i : i + args.batch_size]

            # Submit forward_backward and optim_step (pipelining)
            fwdbwd_future = training_client.forward_backward(
                data=batch,
                loss_fn="cross_entropy",
            )
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=args.lr),
            )

            # Wait for results
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            # Compute weighted mean NLL
            total_weighted_lp = 0.0
            total_w = 0.0
            for out, d in zip(fwdbwd_result.loss_fn_outputs, batch):
                lp = out["logprobs"].to_torch()
                w = d.loss_fn_inputs["weights"].to_torch()
                total_weighted_lp += lp.dot(w).item()
                total_w += w.sum().item()
            loss = -total_weighted_lp / max(total_w, 1.0)

            epoch_loss += loss
            num_batches += 1
            total_steps += 1

            if total_steps % 10 == 0:
                print(f"  [Step {total_steps}] loss={loss:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} — avg_loss={avg_loss:.4f} ({num_batches} batches)")

        # Save checkpoint (final epoch gets no TTL so it persists)
        ckpt_name = f"sft-epoch-{epoch}"
        is_final = (epoch == args.epochs)
        save_kwargs = {"name": ckpt_name}
        if is_final:
            save_kwargs["ttl_seconds"] = None   # never expires
        save_resp = training_client.save_state(**save_kwargs).result()
        ckpt_path = getattr(save_resp, "path", ckpt_name)
        print(f"  Saved checkpoint: {ckpt_name} → {ckpt_path}")
        if is_final:
            print(f"  (persistent — no TTL)")

    print("SFT training complete.")
    print(f"Final checkpoint path: {ckpt_path}")

    # Write checkpoint path to a file so standalone eval can reuse it
    ckpt_file = Path(args.data).parent / "last_checkpoint.txt"
    ckpt_file.write_text(ckpt_path + "\n")
    print(f"Checkpoint path saved to {ckpt_file}")

    # --- Optional post-training evaluation ---
    if args.eval_tasks and args.eval_answers and args.eval_spec:
        run_post_training_eval(
            training_client=training_client,
            tokenizer=tokenizer,
            model_name=args.model,
            checkpoint_name=ckpt_name,
            eval_tasks_path=args.eval_tasks,
            eval_answers_path=args.eval_answers,
            eval_spec_path=args.eval_spec,
            eval_output_dir=args.eval_output_dir or "eval/prfconnect/sft_results",
            eval_baseline_path=args.eval_baseline,
        )


if __name__ == "__main__":
    main()
