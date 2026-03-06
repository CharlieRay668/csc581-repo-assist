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

import numpy as np
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
    messages: list[dict], renderer: renderers.Renderer
) -> types.Datum | None:
    """Convert a message list to a Tinker Datum for SFT."""
    convo = messages_to_renderer_format(messages)

    if len(convo) < 2:
        return None

    # Ensure the last message is from assistant (what we're training on)
    if convo[-1]["role"] != "assistant":
        return None

    try:
        model_input, weights = renderer.build_supervised_example(convo)
    except Exception:
        return None

    tokens = model_input.to_ints()
    if len(tokens) < 2:
        return None

    # Standard SFT: predict next token
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]  # shift weights to align with targets

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
        ),
    )


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

            # Compute weighted average NLL
            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [d.loss_fn_inputs["weights"].tolist() for d in batch]
            )
            loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)

            epoch_loss += loss
            num_batches += 1
            total_steps += 1

            if total_steps % 10 == 0:
                print(f"  [Step {total_steps}] loss={loss:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} — avg_loss={avg_loss:.4f} ({num_batches} batches)")

        # Save checkpoint
        ckpt_name = f"sft-epoch-{epoch}"
        training_client.save_state(name=ckpt_name).result()
        print(f"  Saved checkpoint: {ckpt_name}")

    print("SFT training complete.")


if __name__ == "__main__":
    main()
