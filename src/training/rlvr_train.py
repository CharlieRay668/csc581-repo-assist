"""
Reinforcement Learning with Verifiable Rewards (RLVR) via Tinker + tinker-cookbook.

Uses a ProblemEnv to evaluate file-retrieval answers against ground truth.
The agent receives a repo question, produces a list of relevant files, and
gets a reward based on precision/recall vs. expected files.

Usage:
    python -m src.training.rlvr_train \
        --data   eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --model  Qwen/Qwen3-8B
"""

import argparse
import json
import re
from pathlib import Path

from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl import train


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def extract_file_list(text: str) -> list[str]:
    """Pull file paths from model output text."""
    # Look for paths that look like src/... or components/... etc.
    pattern = r"(?:^|\s)([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)"
    return list(dict.fromkeys(re.findall(pattern, text)))


def file_f1(predicted: list[str], expected: list[str]) -> float:
    """F1 between predicted and expected file lists."""
    pred_set = set(predicted)
    exp_set = set(expected)
    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Problem environment for file-retrieval tasks
# ---------------------------------------------------------------------------

class FileRetrievalEnv(ProblemEnv):
    """A Tinker RL environment for repo file-retrieval tasks.

    get_question():          returns the user's task description
    check_answer(answer):    F1 >= threshold → True
    check_format(answer):    contains at least one plausible file path → True
    get_reference_answer():  comma-separated expected file list
    """

    def __init__(self, question: str, expected_files: list[str], f1_threshold: float = 0.5):
        self.question = question
        self.expected_files = expected_files
        self.f1_threshold = f1_threshold

    def get_question(self) -> str:
        return self.question

    def check_answer(self, answer: str) -> bool:
        predicted = extract_file_list(answer)
        return file_f1(predicted, self.expected_files) >= self.f1_threshold

    def check_format(self, answer: str) -> bool:
        # Must contain at least one file-path-like string
        return bool(re.search(r"[a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+", answer))

    def get_reference_answer(self) -> str:
        return ", ".join(self.expected_files)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_task_answer_pairs(
    task_path: str, answer_path: str
) -> list[tuple[str, list[str]]]:
    """Load task questions and their expected file lists.

    Returns: [(question, [expected_file, ...]), ...]
    """
    tasks = {}
    for line in Path(task_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = obj.get("task_id", obj.get("id"))
        tasks[tid] = obj.get("question", obj.get("input", ""))

    pairs = []
    for line in Path(answer_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = obj.get("task_id", obj.get("id"))
        q = tasks.get(tid)
        if q is None:
            continue
        expected = obj.get("expected_files", obj.get("files", []))
        if isinstance(expected, str):
            expected = [f.strip() for f in expected.split(",") if f.strip()]
        pairs.append((q, expected))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RLVR training via Tinker")
    parser.add_argument("--data", required=True, help="tasks JSONL path")
    parser.add_argument("--answers", required=True, help="answers JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model on Tinker")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--num-samples", type=int, default=8, help="Samples per prompt (GRPO)")
    parser.add_argument("--log-path", default="/tmp/tinker-rlvr", help="Log directory")
    parser.add_argument("--loss-fn", default="cispo", help="RL loss: cispo|ppo|importance_sampling")
    parser.add_argument("--f1-threshold", type=float, default=0.5, help="F1 threshold for reward")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print stats only")
    args = parser.parse_args()

    # --- Load data ---
    pairs = load_task_answer_pairs(args.data, args.answers)
    print(f"Loaded {len(pairs)} task-answer pairs")

    if args.dry_run:
        file_counts = [len(p[1]) for p in pairs]
        print(f"  Expected file counts: min={min(file_counts)}, max={max(file_counts)}, avg={sum(file_counts)/len(file_counts):.1f}")
        return

    if not pairs:
        print("No valid pairs — exiting.")
        return

    # --- Build RL dataset ---
    def make_dataset_builder() -> RLDatasetBuilder:
        builder = RLDatasetBuilder()
        for question, expected_files in pairs:
            env_thunk = lambda q=question, ef=expected_files: FileRetrievalEnv(
                question=q, expected_files=ef, f1_threshold=args.f1_threshold
            )
            builder.add(ProblemGroupBuilder(env_thunk=env_thunk, num_envs=args.num_samples))
        return builder

    # --- Configure training ---
    config = train.Config(
        model_name=args.model,
        dataset_builder=make_dataset_builder,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        loss_fn=args.loss_fn,
        log_path=args.log_path,
    )

    print(f"Starting RLVR training:")
    print(f"  model      = {args.model}")
    print(f"  loss_fn    = {args.loss_fn}")
    print(f"  lr         = {args.lr}")
    print(f"  num_samples= {args.num_samples}")
    print(f"  max_tokens = {args.max_tokens}")
    print(f"  tasks      = {len(pairs)}")

    # --- Run training loop ---
    train.main(config)

    print("RLVR training complete.")


if __name__ == "__main__":
    main()
