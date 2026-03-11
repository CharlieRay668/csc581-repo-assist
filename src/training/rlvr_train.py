"""
Reinforcement Learning with Verifiable Rewards (RLVR) for Repo-Assist.

Uses Tinker's tinker_cookbook RL framework to train a model to answer
repository questions, with rewards based on file-retrieval accuracy.

The model is trained with Qwen3's native tool-calling format (<tool_call>
tags) in the system prompt so it learns to invoke tools like search_repo
and open_file.  At inference time, the eval_sft.py / AgentOrchestrator
ReAct loop executes these tool calls via ToolGateway.

Reward signal:
  - check_answer():  F1 between predicted and expected file paths >= threshold
                     If --repo-path is given, tool calls in the model output
                     are executed and the final text is scored.
  - check_format():  answer contains at least one plausible file path
                     OR a well-formed <tool_call> block.

Usage:
    # Dry run — load data, print stats, verify everything parses
    python -m src.training.rlvr_train \
        --data    eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --dry-run

    # Real training from base model
    python -m src.training.rlvr_train \
        --data    eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --model   Qwen/Qwen3-8B

    # Continue from SFT checkpoint, with tool-augmented rewards
    python -m src.training.rlvr_train \
        --data    eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
        --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
        --model   Qwen/Qwen3-8B \
        --load-checkpoint training_data/last_checkpoint.txt \
        --repo-path hack4impact-repos/prfc-connect
"""

import argparse
import asyncio
import json
import math
import re
from functools import partial
from pathlib import Path
from typing import Sequence

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.rl import train
from tinker_cookbook.tokenizer_utils import get_tokenizer


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def extract_file_list(text: str) -> list[str]:
    """Pull file paths out of model output text.

    Matches paths like  src/app/api/referrals/route.ts  or  prisma/schema.prisma.
    De-duplicates while preserving first-occurrence order.
    """
    pattern = r"\b(?:src|docs|prisma|test|e2e|public|types)/[A-Za-z0-9_\-./\[\]]+\.[A-Za-z0-9]+"
    paths = re.findall(pattern, text or "")
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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
# ProblemEnv for file-retrieval tasks
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert repository assistant for the PRFC Connect codebase "
    "(a Next.js referral-management application). "
    "Answer the developer's question by citing specific source files. "
    "Always include file paths like src/app/api/referrals/route.ts in your answer.\n"
    "You have access to tools — use <tool_call> tags to invoke them."
)

# Optional: shared tool gateway instance for reward-time tool execution
_TOOL_GATEWAY = None


def set_tool_gateway(gateway):
    """Set a shared ToolGateway for use in reward computation.

    When set, the check_answer() method will execute any <tool_call> blocks
    in the model output and include the resulting file paths in scoring.
    """
    global _TOOL_GATEWAY
    _TOOL_GATEWAY = gateway


def _execute_tool_calls_for_reward(text: str) -> str:
    """Execute <tool_call> blocks in text via the shared gateway, return augmented text.

    This lets the reward function score tool-using outputs by actually running
    the tools and appending the results.  If no gateway is set, returns text unchanged.
    """
    from src.model_backend import _parse_qwen_tool_calls

    if _TOOL_GATEWAY is None:
        return text

    remaining, tool_calls = _parse_qwen_tool_calls(text)
    if not tool_calls:
        return text

    augmented = text
    for tc in tool_calls:
        try:
            if tc.name == "search_repo":
                results = _TOOL_GATEWAY.search_repo(tc.arguments.get("query"), top_k=tc.arguments.get("top_k", 5))
                result_text = json.dumps({"results": results, "count": len(results)}, default=str)
            elif tc.name == "open_file":
                result = _TOOL_GATEWAY.open_file(tc.arguments.get("path"), tc.arguments.get("start_line"), tc.arguments.get("end_line"))
                result_text = json.dumps(result, default=str)
            elif tc.name == "list_files":
                files = _TOOL_GATEWAY.list_files(path_prefix=tc.arguments.get("path_prefix"), extensions=tc.arguments.get("extensions"))
                result_text = json.dumps({"files": files}, default=str)
            else:
                continue
            # Append tool result paths to augmented text for scoring
            augmented += f"\n{result_text}"
        except Exception:
            continue

    return augmented


class FileRetrievalEnv(ProblemEnv):
    """Tinker RL environment for repo file-retrieval tasks.

    Reward = check_answer (binary: F1 >= threshold) + format_coef * (check_format - 1)
    This matches the ProblemEnv.step() reward formula.

    When a ToolGateway is set via set_tool_gateway(), check_answer() will
    execute any <tool_call> blocks in the model output and include the
    resulting file paths in the F1 computation.
    """

    def __init__(
        self,
        question: str,
        expected_files: list[str],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        f1_threshold: float = 0.5,
    ):
        super().__init__(renderer, convo_prefix, format_coef=0.1)
        self.question = question
        self.expected_files = expected_files
        self.f1_threshold = f1_threshold

    def get_question(self) -> str:
        return self.question

    def check_answer(self, sample_str: str) -> bool:
        """True if file-path F1 >= threshold.

        If a ToolGateway is available, executes tool calls in the output
        and includes discovered file paths in scoring.
        """
        # Augment with tool execution results if gateway is available
        augmented = _execute_tool_calls_for_reward(sample_str)
        predicted = extract_file_list(augmented)
        return file_f1(predicted, self.expected_files) >= self.f1_threshold

    def check_format(self, sample_str: str) -> bool:
        """True if the answer contains a plausible file path OR a well-formed <tool_call>."""
        has_file_path = bool(re.search(
            r"\b(?:src|docs|prisma|test|e2e|public|types)/[A-Za-z0-9_\-./]+\.[A-Za-z0-9]+",
            sample_str,
        ))
        has_tool_call = bool(re.search(
            r"<tool_call>\s*\{.*?\"name\".*?\}\s*</tool_call>",
            sample_str,
            re.DOTALL,
        ))
        return has_file_path or has_tool_call

    def get_reference_answer(self) -> str:
        return ", ".join(self.expected_files)


# ---------------------------------------------------------------------------
# RLDataset / RLDatasetBuilder  (mirrors MathDataset / MathDatasetBuilder)
# ---------------------------------------------------------------------------

def _load_task_answer_pairs(
    task_path: str, answer_path: str
) -> list[dict]:
    """Load tasks joined with their expected files.

    Returns list of dicts with keys: task_id, question, expected_files, mode, difficulty.
    """
    tasks: dict[str, dict] = {}
    for line in Path(task_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = obj.get("task_id", obj.get("id"))
        tasks[tid] = obj

    pairs: list[dict] = []
    for line in Path(answer_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ans = json.loads(line)
        tid = ans.get("task_id", ans.get("id"))
        task_obj = tasks.get(tid)
        if task_obj is None:
            continue
        expected = ans.get("expected_files", ans.get("files", []))
        if isinstance(expected, str):
            expected = [f.strip() for f in expected.split(",") if f.strip()]
        # Skip tasks with no expected files (e.g. some suggest tasks)
        # — RLVR needs a verifiable ground truth
        if not expected:
            continue
        pairs.append({
            "task_id": tid,
            "question": task_obj.get("question", task_obj.get("input", "")),
            "expected_files": expected,
            "mode": task_obj.get("mode", "locate"),
            "difficulty": task_obj.get("difficulty", "easy"),
        })

    return pairs


class RepoAssistDataset(RLDataset):
    """An RL dataset over repo-assist file-retrieval tasks."""

    def __init__(
        self,
        pairs: list[dict],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        f1_threshold: float = 0.5,
    ):
        self.pairs = pairs
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.f1_threshold = f1_threshold

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.pairs))
        assert start < end, f"Batch index {index} out of range"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    FileRetrievalEnv,
                    question=p["question"],
                    expected_files=p["expected_files"],
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                    f1_threshold=self.f1_threshold,
                ),
                num_envs=self.group_size,
                dataset_name="repo_assist",
            )
            for p in self.pairs[start:end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.batch_size)


@chz.chz
class RepoAssistDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally test) RL datasets from tasks + answers."""

    task_path: str
    answer_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str = "qwen3_disable_thinking"
    f1_threshold: float = 0.5
    train_frac: float = 0.8
    seed: int = 42

    async def __call__(self) -> tuple[RepoAssistDataset, RepoAssistDataset | None]:
        import random
        from src.model_backend import _build_qwen_tool_block, TOOL_SCHEMAS

        all_pairs = _load_task_answer_pairs(self.task_path, self.answer_path)
        rng = random.Random(self.seed)
        rng.shuffle(all_pairs)

        split = int(len(all_pairs) * self.train_frac)
        train_pairs = all_pairs[:split]
        test_pairs = all_pairs[split:] if split < len(all_pairs) else None

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Include tool definitions in the system prompt so the model learns
        # the <tool_call> format during RL training
        tool_block = _build_qwen_tool_block(TOOL_SCHEMAS)
        system_content = tool_block + "\n\n" + SYSTEM_PROMPT

        convo_prefix: list[renderers.Message] = [
            {"role": "system", "content": system_content},
        ]

        train_ds = RepoAssistDataset(
            pairs=train_pairs,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            f1_threshold=self.f1_threshold,
        )
        test_ds = (
            RepoAssistDataset(
                pairs=test_pairs,
                batch_size=self.batch_size,
                group_size=1,  # single sample for test
                renderer=renderer,
                convo_prefix=convo_prefix,
                f1_threshold=self.f1_threshold,
            )
            if test_pairs
            else None
        )
        return train_ds, test_ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_checkpoint(raw: str | None) -> str | None:
    """Accept a tinker:// URI, a file containing one, or None."""
    if raw is None:
        return None
    if raw.startswith("tinker://"):
        return raw.strip()
    p = Path(raw)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    raise ValueError(f"--load-checkpoint must be a tinker:// URI or file (got {raw!r})")


def main():
    parser = argparse.ArgumentParser(
        description="RLVR training for repo-assist via Tinker"
    )
    parser.add_argument("--data", required=True, help="Tasks JSONL path")
    parser.add_argument("--answers", required=True, help="Answers JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Samples per prompt (GRPO group size)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Tasks per training batch")
    parser.add_argument(
        "--loss-fn", default="importance_sampling",
        choices=["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"],
        help="RL loss function",
    )
    parser.add_argument("--f1-threshold", type=float, default=0.5,
                        help="F1 threshold for binary reward")
    parser.add_argument("--renderer", default="qwen3_disable_thinking",
                        help="Chat renderer (qwen3, qwen3_disable_thinking, etc.)")
    parser.add_argument("--log-path", default="training_data/rlvr_logs",
                        help="Log/checkpoint directory")
    parser.add_argument("--load-checkpoint", default=None,
                        help="tinker:// checkpoint or file to continue from (e.g. SFT checkpoint)")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and print stats only")
    parser.add_argument("--repo-path", default=None,
                        help="Path to target repository for tool-augmented rewards. "
                             "If given, tool calls in model output are executed during "
                             "reward computation to discover file paths.")
    args = parser.parse_args()

    # ── Load & validate data ────────────────────────────────────────
    pairs = _load_task_answer_pairs(args.data, args.answers)
    print(f"Loaded {len(pairs)} task-answer pairs with verifiable file lists")

    if not pairs:
        print("No valid pairs with expected_files — nothing to train on.")
        return

    # Stats
    file_counts = [len(p["expected_files"]) for p in pairs]
    modes = {}
    diffs = {}
    for p in pairs:
        modes[p["mode"]] = modes.get(p["mode"], 0) + 1
        diffs[p["difficulty"]] = diffs.get(p["difficulty"], 0) + 1

    print(f"  Expected files per task: min={min(file_counts)}, "
          f"max={max(file_counts)}, avg={sum(file_counts)/len(file_counts):.1f}")
    print(f"  By mode: {modes}")
    print(f"  By difficulty: {diffs}")
    print(f"  Train/test split: {int(len(pairs)*0.8)} / {len(pairs) - int(len(pairs)*0.8)}")

    if args.dry_run:
        print("\n[dry-run] Would train with:")
        print(f"  model       = {args.model}")
        print(f"  loss_fn     = {args.loss_fn}")
        print(f"  lr          = {args.lr}")
        print(f"  group_size  = {args.group_size}")
        print(f"  batch_size  = {args.batch_size}")
        print(f"  max_tokens  = {args.max_tokens}")
        print(f"  renderer    = {args.renderer}")
        print(f"  f1_threshold= {args.f1_threshold}")
        ckpt = resolve_checkpoint(args.load_checkpoint)
        print(f"  checkpoint  = {ckpt or '(none — training from base)'}")
        return

    # ── Build config ────────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(args.load_checkpoint)

    # Set up tool gateway for reward computation if repo path given
    if args.repo_path:
        from src.tool_gateway import ToolGateway
        try:
            gw = ToolGateway(repo_path=args.repo_path)
            set_tool_gateway(gw)
            stats = gw.stats()
            print(f"Tool gateway loaded for rewards: {stats.get('total_files', 0)} files indexed")
        except Exception as e:
            print(f"Warning: Could not load tool gateway ({e}). Rewards will not execute tool calls.")

    dataset_builder = RepoAssistDatasetBuilder(
        task_path=args.data,
        answer_path=args.answers,
        batch_size=args.batch_size,
        group_size=args.group_size,
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        f1_threshold=args.f1_threshold,
    )

    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    config = train.Config(
        model_name=args.model,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        loss_fn=args.loss_fn,
        log_path=args.log_path,
        load_checkpoint_path=ckpt_path,
        eval_every=args.eval_every,
        save_every=args.save_every,
        temperature=1.0,  # RL needs exploration
    )

    print(f"\nStarting RLVR training:")
    print(f"  model       = {args.model}")
    print(f"  loss_fn     = {args.loss_fn}")
    print(f"  lr          = {args.lr}")
    print(f"  group_size  = {args.group_size} samples/prompt")
    print(f"  batch_size  = {args.batch_size} tasks/batch")
    print(f"  max_tokens  = {args.max_tokens}")
    print(f"  renderer    = {args.renderer}")
    print(f"  f1_threshold= {args.f1_threshold}")
    print(f"  checkpoint  = {ckpt_path or '(none — base model)'}")
    print()

    # ── Run ─────────────────────────────────────────────────────────
    asyncio.run(train.main(config))

    # ── Capture final checkpoint path ───────────────────────────────
    # train.main saves checkpoints to {log_path}/checkpoints.jsonl
    ckpt_file = Path(args.log_path) / "checkpoints.jsonl"
    final_ckpt = None
    if ckpt_file.exists():
        lines = ckpt_file.read_text().strip().splitlines()
        for line in reversed(lines):
            entry = json.loads(line)
            if "state_path" in entry:
                final_ckpt = entry["state_path"]
                break

    if final_ckpt:
        # Save for easy access by eval_sft.py
        ckpt_out = Path(args.log_path) / "last_checkpoint.txt"
        ckpt_out.write_text(final_ckpt + "\n")
        print(f"\nFinal checkpoint: {final_ckpt}")
        print(f"Saved to: {ckpt_out}")
        print(f"\nTo evaluate, run:")
        print(f"  python -m src.training.eval_sft \\")
        print(f"    --checkpoint {ckpt_out} \\")
        print(f"    --tasks {args.data} \\")
        print(f"    --answers {args.answers} \\")
        print(f"    --spec eval/eval_spec.json \\")
        print(f"    --output-dir eval/prfconnect/rlvr_results \\")
        print(f"    --baseline eval/prfconnect/real_results/output.prfc-connect.auto.json")
    else:
        print("\nWarning: could not find final checkpoint in checkpoints.jsonl")

    print("\nRLVR training complete.")


if __name__ == "__main__":
    main()
