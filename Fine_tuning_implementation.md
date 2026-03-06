# Repo Assist — Implementation Reference

**Course:** CSC 581  
**Date:** March 5, 2026  
**Purpose:** Source-of-truth for writing class deliverables. Documents the full system architecture, what was built, and — in particular — the fine-tuning pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Core Components](#3-core-components)
4. [Evaluation Framework](#4-evaluation-framework)
5. [Multi-Agent Routing](#5-multi-agent-routing)
6. [Fine-Tuning Pipeline (SFT)](#6-fine-tuning-pipeline-sft)
7. [Fine-Tuning Pipeline (RLVR)](#7-fine-tuning-pipeline-rlvr)
8. [Data Flow — End to End](#8-data-flow--end-to-end)
9. [Tinker Platform Details](#9-tinker-platform-details)
10. [Commands Reference](#10-commands-reference)
11. [File Index](#11-file-index)

---

## 1. System Overview

Repo Assist is a **ReAct-based agent** that answers developer questions about a codebase. Given a repository and a natural-language question, it autonomously searches code, reads files, queries GitHub issues/PRs, and produces a grounded answer with file citations.

The system extends beyond a single-model agent into a **multi-agent specialist architecture** with fine-tuning infrastructure. Each question mode (explain, locate, suggest, patch) can be routed to a dedicated specialist model that has been fine-tuned for that task via the Tinker platform.

### Key Capabilities

| Capability | Description |
|---|---|
| **Code search & retrieval** | Chunked ingestion with semantic search over repo files |
| **Tool-use reasoning** | ReAct loop with function calling (search, read file, list files, GitHub API) |
| **Mode-specific answers** | 4 modes with tailored system prompts and output formats |
| **Evaluation framework** | 100-task held-out set with automated correctness/grounding/retrieval metrics |
| **Trajectory logging** | Full agent reasoning chains serialized for training data |
| **SFT pipeline** | Supervised fine-tuning via Tinker cross-entropy loss |
| **RLVR pipeline** | Reinforcement learning with verifiable file-retrieval rewards |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLI / Batch Runner                    │
│                    (cli.py, run_task_batch.py)                │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   Session Manager        │    │   Agent Orchestrator         │
│   (session_manager.py)   │    │   (agent_orchestrator.py)    │
│                          │    │                              │
│  • Session persistence   │    │  • ReAct loop (max N turns)  │
│  • Query history         │    │  • Mode-based routing        │
│  • Evidence tracking     │    │  • Intent classification     │
└──────────────────────────┘    │  • Trajectory serialization  │
                                └──────────────┬───────────────┘
                                               │
                                ┌──────────────▼───────────────┐
                                │      Tool Gateway            │
                                │      (tool_gateway.py)       │
                                │                              │
                                │  search_repo()               │
                                │  open_file()                 │
                                │  list_files()                │
                                │  get_issues()                │
                                │  get_pull_requests()         │
                                │  get_repo_stats()            │
                                └──────────────┬───────────────┘
                                               │
                                ┌──────────────▼───────────────┐
                                │    Repo Ingestion            │
                                │    (repo_ingestion.py)       │
                                │                              │
                                │  • File walking & chunking   │
                                │  • GitHub API integration    │
                                │  • Substring search          │
                                └──────────────────────────────┘
```

### Training Pipeline (extends the above)

```
Agent Outputs (JSONL)
        │
        ▼
┌─────────────────────┐     ┌─────────────────────────┐
│ convert_trajectories │────▶│  conversations.jsonl     │
│ (convert_trajectories.py)  │  (SFT training data)    │
└─────────────────────┘     └───────────┬─────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  sft_train.py          │
                            │  (Tinker cross-entropy)│
                            └───────────────────────┘

Task/Answer Pairs (JSONL)
        │
        ▼
┌─────────────────────────────────┐
│  rlvr_train.py                   │
│  (Tinker RLVR — self-play with  │
│   verifiable file-retrieval      │
│   rewards, no pre-generated      │
│   trajectories needed)           │
└─────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Repo Ingestion (`src/repo_ingestion.py`)

Walks a repository directory and produces a searchable in-memory index.

- **Chunking:** Files split into fixed-size segments of up to 40 lines with 5-line overlap between consecutive chunks. Files under 10 lines are kept as a single chunk.
- **Filtering:** Ignores `node_modules`, `.git`, binary files, lock files, etc.
- **Fingerprinting:** Each file is hashed using truncated SHA-256 (16 hex chars) for change detection metadata.
- **Search:** Case-insensitive substring matching over chunks (not TF-IDF or embedding-based). Results are returned in file-walk order, without relevance ranking.
- **GitHub integration:** Fetches issues and PRs via GitHub REST API

### 3.2 Tool Gateway (`src/tool_gateway.py`)

Wraps `RepoIngestion` into a tool interface consumed by the orchestrator. Provides 6 tools:

| Tool | Description |
|---|---|
| `search_repo(query, top_k)` | Semantic search over code chunks |
| `open_file(path, start_line, end_line)` | Read file contents / line ranges |
| `list_files(path_prefix, extensions)` | List repository file tree |
| `get_issues(query, state, limit)` | Search GitHub issues |
| `get_pull_requests(query, state, limit)` | Search GitHub PRs |
| `get_repo_stats()` | File/chunk/issue/PR counts |

### 3.3 Agent Orchestrator (`src/agent_orchestrator.py`)

The core ReAct agent. 573 lines. Key design decisions:

- **LLM:** Gemini 2.5 Flash via `google-genai` SDK (temperature 0.2)
- **Function calling:** Native Gemini `FunctionDeclaration` tool use (not prompt-based)
- **Turn limit:** Configurable `max_turns` (default 10) to prevent runaway loops
- **Scope control:** `files-only` vs `include-pr` restricts available tools
- **Fallback:** If the model stops without a text answer, prompts it once more for a final answer

**Output dataclass — `OrchestratorResult`:**

| Field | Type | Description |
|---|---|---|
| `tool_call_plan` | `list[ToolCallSpec]` | First-turn planned tool calls |
| `executed_tool_calls` | `list[ExecutedToolCall]` | All tool calls with results |
| `consolidated_evidence` | `list[Citation]` | Deduplicated file/issue/PR citations |
| `final_response` | `FinalResponse` | Answer text, citations, patch diff, next actions |
| `raw_turns` | `list[dict]` | **Full serialized reasoning chain for training** |

### 3.4 Session Manager (`src/session_manager.py`)

Persists conversation state across queries in `~/.repo_assist/sessions/`. Tracks recent queries and selected evidence to provide conversational context.

### 3.5 Batch Runner (`src/run_task_batch.py`)

Runs the agent across a JSONL task file and writes structured output. Key features:
- Accepts `--model-map` for per-mode model routing
- Outputs `raw_turns` and `tool_calls` arrays per task for trajectory logging
- Supports `--offset`, `--limit`, `--append` for resumable batch runs

---

## 4. Evaluation Framework

### 4.1 Task Dataset

**100 held-out tasks** in `eval/prfconnect/tasks.prfc-connect.heldout.jsonl`, stratified:

| Stratum | Count | Modes |
|---|---|---|
| Easy | 40 | Mostly `locate` |
| Medium | 40 | Mix of `locate` and `explain` |
| Hard | 20 | `explain` and `suggest` |

Each task has a ground-truth answer with `expected_files`, `reference_answer`, and `required_keywords`.

### 4.2 Evaluation Metrics

Defined in `eval/eval_spec.json`, scored by `src/eval_runner.py` and `src/generate_eval_artifacts.py`:

| Metric | Scale | What it measures |
|---|---|---|
| Correctness | 0–2 | Factual accuracy of the answer |
| Grounding | 0–2 | Claims backed by cited evidence |
| Relevance | 0–2 | Does the answer address the question's intent |
| Retrieval Precision | 0.0–1.0 | Fraction of retrieved files that are relevant |
| Retrieval Recall | 0.0–1.0 | Fraction of relevant files that were retrieved |
| Latency | milliseconds | End-to-end response time |

### 4.3 Existing Results

Baseline results with Gemini 2.5 Flash (all modes) are stored in:
- `eval/prfconnect/real_results/agent_outputs.prfc-connect.heldout.jsonl` — 100 agent outputs
- `eval/prfconnect/real_results/ratings.prfc-connect.heldout.auto.jsonl` — automated quality scores
- `eval/prfconnect/real_results/retrieval.prfc-connect.heldout.auto.jsonl` — retrieval metrics

---

## 5. Multi-Agent Routing

### 5.1 Intent Classification

Added `_classify_intent()` to `AgentOrchestrator` — a keyword-heuristic router that auto-detects which mode a query belongs to:

```python
_INTENT_KEYWORDS = {
    "patch":   ["fix", "patch", "change", "modify", "update", "refactor", "diff", "edit"],
    "locate":  ["where", "find", "locate", "which file", "what file", "defined", "implemented"],
    "suggest": ["suggest", "improve", "recommend", "next step", "should", "todo", "how to"],
    "explain": ["explain", "what", "how", "why", "describe", "walk through", "overview"],
}
```

When `mode="auto"`, the router scores the query against these keyword lists and selects the highest-scoring mode (defaulting to `explain`).

### 5.2 Model Map

The orchestrator now accepts a `model_map: dict[str, str]` that routes each mode to a different model:

```python
DEFAULT_MODEL_MAP = {
    "explain": "gemini-2.5-flash",
    "locate":  "gemini-2.5-flash",
    "suggest": "gemini-2.5-flash",
    "patch":   "gemini-2.5-flash",
}
```

After fine-tuning, these values would swap to Tinker-hosted specialist models (e.g., `Qwen/Qwen3-30B-A3B` for explain, `moonshotai/Kimi-K2-Thinking` for patch).

### 5.3 Routing Flow

```
User query
    │
    ▼
_classify_intent(query)  ←── keyword heuristics
    │
    ▼
mode = "locate" (e.g.)
    │
    ▼
active_model = model_map["locate"]  ←── specialist lookup
    │
    ▼
ReAct loop with active_model
```

---

## 6. Fine-Tuning Pipeline (SFT)

> **This is the primary fine-tuning approach.** SFT teaches a model to reproduce expert agent behavior by training on recorded trajectories.

### 6.1 Trajectory Logging

**What changed:** `OrchestratorResult` now includes a `raw_turns` field — the full serialized Gemini conversation (user messages, model reasoning, function calls, function responses) as plain dicts.

**Serialization** (`_serialize_contents()`): Converts Gemini SDK `Content` objects into JSON-safe dicts with typed parts:
- `{"type": "text", "text": "..."}` — model reasoning or user messages
- `{"type": "function_call", "name": "search_repo", "args": {...}}` — tool invocations
- `{"type": "function_response", "name": "search_repo", "response": {...}}` — tool results

The batch runner (`run_task_batch.py`) writes these into the output JSONL per task.

### 6.2 Trajectory Conversion (`src/convert_trajectories.py`)

Converts raw agent output JSONL into `conversations.jsonl` — the format expected by the SFT trainer.

**Input:** Agent output JSONL from `run_task_batch.py`  
**Output:** `training_data/conversations.jsonl`

Each output row has:
```json
{
  "task_id": "PRFC-E001",
  "mode": "locate",
  "messages": [
    {"role": "system", "content": "You are an expert repository assistant..."},
    {"role": "user", "content": "Where is the POST /api/referrals handler?"},
    {"role": "assistant", "content": null, "tool_calls": [{"name": "search_repo", "arguments": {...}}]},
    {"role": "tool", "name": "search_repo", "content": "{...results...}"},
    {"role": "assistant", "content": "The handler is in src/app/api/referrals/route.ts..."}
  ]
}
```

**Key design decisions:**
- Prefers `raw_turns` (full reasoning chain) when available
- Falls back to `tool_calls` + final answer when `raw_turns` is absent
- Truncates tool responses to 2KB to keep training data manageable
- Supports `--mode` filtering to train mode-specific specialists
- Reconstructs system prompts from task metadata

**Usage:**
```bash
python -m src.convert_trajectories \
    --input  eval/prfconnect/real_results/agent_outputs.prfc-connect.heldout.jsonl \
    --output training_data/conversations.jsonl \
    --mode   locate    # optional: train only a locate specialist
```

### 6.3 SFT Training Script (`src/training/sft_train.py`)

Performs supervised fine-tuning using the **Tinker API** with cross-entropy loss.

**How it works, step by step:**

1. **Load conversations** from `conversations.jsonl`
2. **Connect to Tinker** — creates a LoRA training client on the specified base model
3. **Tokenize** — uses `tinker_cookbook.renderers` to convert chat messages into token sequences with loss weights:
   - `renderer.build_supervised_example(messages)` → `(ModelInput, weights)`
   - Weights are 0 for input tokens (user/system) and 1 for target tokens (assistant) — the model only learns to predict assistant turns
4. **Construct Datums** — each training example becomes a `Datum`:
   ```python
   Datum(
       model_input=ModelInput.from_ints(tokens=input_tokens),  # tokens[:-1]
       loss_fn_inputs=dict(
           weights=weights[1:],          # shifted to align with targets
           target_tokens=tokens[1:],     # next-token prediction targets
       ),
   )
   ```
5. **Training loop** — for each epoch, iterate over mini-batches:
   ```python
   fwdbwd_future = training_client.forward_backward(data=batch, loss_fn="cross_entropy")
   optim_future  = training_client.optim_step(AdamParams(learning_rate=lr))
   ```
   - `forward_backward()` and `optim_step()` are **pipelined as async futures** — the optimizer step is submitted immediately while loss computation is still in flight
   - Loss is computed from `fwdbwd_result.loss_fn_outputs` (weighted negative log-likelihood)
6. **Checkpointing** — `training_client.save_state(name="sft-epoch-N")` after each epoch

**CLI:**
```bash
# Dry run (data validation only)
python -m src.training.sft_train --data training_data/conversations.jsonl --dry-run

# Real training
python -m src.training.sft_train \
    --data       training_data/conversations.jsonl \
    --model      Qwen/Qwen3-8B \
    --epochs     3 \
    --lr         1e-4 \
    --batch-size 4 \
    --lora-rank  32
```

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Any Tinker-supported model |
| `--epochs` | 3 | Full passes over the dataset |
| `--lr` | 1e-4 | Learning rate for AdamW |
| `--batch-size` | 4 | Micro-batch size per forward/backward |
| `--lora-rank` | 32 | LoRA adapter rank (higher = more capacity) |

---

## 7. Fine-Tuning Pipeline (RLVR)

> **RLVR (Reinforcement Learning with Verifiable Rewards)** improves the model through self-play. Unlike SFT, it does NOT require pre-generated trajectories — the model generates its own answers and gets a binary reward based on correctness.

### 7.1 Core Concept

For each task:
1. The model receives a question (e.g., "Where is the POST /api/referrals handler?")
2. It generates N candidate answers (GRPO — Group Relative Policy Optimization)
3. Each answer is checked against ground truth using a **verifiable reward function**
4. Correct answers get reward 1, incorrect get reward 0
5. The policy is updated to increase the probability of correct answers

### 7.2 Reward Design — `FileRetrievalEnv`

The RLVR script defines a `FileRetrievalEnv` that extends Tinker's `ProblemEnv` abstract class:

```python
class FileRetrievalEnv(ProblemEnv):
    def get_question(self) -> str:
        return self.question

    def check_answer(self, answer: str) -> bool:
        predicted = extract_file_list(answer)         # regex-extract file paths
        return file_f1(predicted, self.expected_files) >= self.f1_threshold

    def check_format(self, answer: str) -> bool:
        return bool(re.search(r"[a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+", answer))

    def get_reference_answer(self) -> str:
        return ", ".join(self.expected_files)
```

**Reward signal:** F1 score between predicted file paths and expected file paths. If F1 ≥ 0.5 (configurable), the answer is "correct." This is a **verifiable** reward — no LLM judge needed.

**File extraction:** `extract_file_list()` uses regex to pull file-path-like strings (e.g., `src/app/api/referrals/route.ts`) from model output.

**F1 calculation:**
$$F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

where precision = (correct predictions / total predictions) and recall = (correct predictions / total expected files).

### 7.3 Training Configuration

The RLVR script uses `tinker_cookbook.rl.train.Config` to configure the training loop:

```python
config = train.Config(
    model_name=args.model,
    dataset_builder=make_dataset_builder,   # creates ProblemGroupBuilder per task
    learning_rate=args.lr,
    lora_rank=args.lora_rank,
    max_tokens=args.max_tokens,             # max generation length
    loss_fn=args.loss_fn,                   # "cispo" (default), "ppo", or "importance_sampling"
    log_path=args.log_path,
)
train.main(config)  # tinker-cookbook handles the full loop
```

The training loop (handled by `tinker_cookbook.rl.train.main()`) does:
1. Build `RLDataset` from `ProblemGroupBuilder`s
2. For each batch: sample N completions per prompt via `SamplingClient`
3. Evaluate each completion with `check_answer()` and `check_format()`
4. Compute advantages using GRPO (relative to the group)
5. Update policy via `forward_backward()` with the chosen RL loss
6. Repeat

**CLI:**
```bash
# Dry run
python -m src.training.rlvr_train \
    --data    eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --dry-run

# Real training
python -m src.training.rlvr_train \
    --data         eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers      eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --model        Qwen/Qwen3-8B \
    --lr           1e-5 \
    --num-samples  8 \
    --loss-fn      cispo \
    --f1-threshold 0.5
```

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Base model for RL fine-tuning |
| `--lr` | 1e-5 | Lower than SFT (RL is less stable) |
| `--num-samples` | 8 | GRPO group size — completions per prompt |
| `--loss-fn` | `cispo` | Clipped Importance Sampling Policy Optimization |
| `--max-tokens` | 2048 | Max generation length per sample |
| `--f1-threshold` | 0.5 | Minimum F1 for a "correct" answer |
| `--lora-rank` | 32 | LoRA adapter rank |

### 7.4 SFT vs. RLVR — When to Use Which

| Dimension | SFT | RLVR |
|---|---|---|
| **Requires** | Pre-generated agent trajectories | Task/answer pairs only |
| **Learns from** | Imitating expert behavior | Trial-and-error self-play |
| **Strengths** | Fast convergence, stable, predictable | Discovers novel strategies, optimizes for outcomes |
| **Weaknesses** | Bounded by teacher quality | Higher variance, needs more compute |
| **Best for** | Explain, suggest (open-ended) | Locate, patch (verifiable outputs) |
| **Data needed** | 100–500 trajectories | 50–100 task/answer pairs (model generates its own data) |
| **Recommended order** | **First** — establishes baseline behavior | **Second** — refines on top of SFT |

---

## 8. Data Flow — End to End

### Phase 1: Generate Training Data

```bash
# 1. Run agent on held-out tasks (generates trajectories with raw_turns)
python -m src.run_task_batch \
    --repo-path hack4impact-repos/prfc-connect \
    --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --output training_data/agent_outputs_with_traces.jsonl \
    --model gemini-2.5-flash

# 2. Convert to SFT format
python -m src.convert_trajectories \
    --input  training_data/agent_outputs_with_traces.jsonl \
    --output training_data/conversations.jsonl
```

### Phase 2: SFT Training

```bash
# 3. Fine-tune a specialist (e.g., locate mode only)
python -m src.training.sft_train \
    --data  training_data/conversations.jsonl \
    --model Qwen/Qwen3-8B \
    --epochs 3 --lr 1e-4
# Produces checkpoint: "sft-epoch-3"
```

### Phase 3: RLVR Refinement

```bash
# 4. RL fine-tune on top of SFT checkpoint
python -m src.training.rlvr_train \
    --data    eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --model   Qwen/Qwen3-8B \
    --loss-fn cispo
```

### Phase 4: Deploy Specialist

```bash
# 5. Update model_map to use fine-tuned model
python -m src.run_task_batch \
    --repo-path hack4impact-repos/prfc-connect \
    --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --output eval_results/specialist_outputs.jsonl \
    --model-map '{"locate": "tinker://sft-epoch-3", "explain": "gemini-2.5-flash"}'
```

### Current Training Data Inventory

| Source | Rows | Has `raw_turns`? | Usable for SFT? |
|---|---|---|---|
| `agent_outputs.prfc-connect.heldout.jsonl` | 100 | ❌ (pre-dates trajectory logging) | Shallow only (system → user → assistant, 3 msgs each) |
| Re-run with updated orchestrator | ~100 | ✅ | Full trajectories (~13–15 msgs each with tool chains) |
| Temperature-varied re-runs (3–5×) | ~300–500 | ✅ | Best SFT quality (diverse reasoning paths) |

**For RLVR:** No pre-generated data needed. The 100 task/answer pairs in `eval/prfconnect/` are sufficient.

---

## 9. Tinker Platform Details

### 9.1 What is Tinker?

Tinker is a cloud platform from [Thinking Machines Lab](https://tinker-docs.thinkingmachines.ai/) for fine-tuning LLMs. Key points:

- **Your code runs locally** on a CPU-only machine — Tinker handles GPU compute remotely
- **LoRA-only** fine-tuning (not full fine-tuning) — Thinking Machines argues LoRA matches full FT quality
- **Async API** — `forward_backward()` and `optim_step()` return futures for pipelining
- **Environment variable:** `TINKER_API_KEY` must be set

### 9.2 SDK Packages

| Package | Version | Purpose |
|---|---|---|
| `tinker` | 0.14.0 | Core SDK — `ServiceClient`, `TrainingClient`, `SamplingClient`, `Datum`, `ModelInput`, `AdamParams` |
| `tinker-cookbook` | 0.1.0 | Higher-level helpers — renderers, RL environments, training loop |

### 9.3 Key API Objects

**`ServiceClient`** — entry point:
```python
service = tinker.ServiceClient()  # reads TINKER_API_KEY from env
training_client = service.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)
```

**`Datum`** — single training example:
```python
Datum(
    model_input=ModelInput.from_ints(tokens=[...]),
    loss_fn_inputs=dict(weights=[...], target_tokens=[...]),
)
```

**`forward_backward()`** — compute loss and gradients:
```python
future = training_client.forward_backward(
    data=[datum1, datum2, ...],
    loss_fn="cross_entropy",  # or "cispo", "ppo", "importance_sampling", "dro"
)
result = future.result()  # ForwardBackwardOutput
```

**`optim_step()`** — apply gradients:
```python
training_client.optim_step(AdamParams(learning_rate=1e-4)).result()
```

**`save_state()`** — checkpoint:
```python
training_client.save_state(name="my-checkpoint", ttl_seconds=86400).result()
```

### 9.4 Recommended Model Lineup

| Model | Best Mode | Rationale |
|---|---|---|
| `Qwen/Qwen3-8B` | Router / locate | Small, fast, good for classification and concise retrieval |
| `Qwen/Qwen3-30B-A3B` | Explain / suggest | Hybrid MoE — thorough reasoning, cost-effective |
| `moonshotai/Kimi-K2-Thinking` | Patch / deep explain | Long chain-of-thought, code-specialized |

---

## 10. Commands Reference

### Running the Agent

```bash
# Interactive CLI
python -m src.cli --repo-path hack4impact-repos/prfc-connect

# Batch evaluation
python -m src.run_task_batch \
    --repo-path hack4impact-repos/prfc-connect \
    --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --output results.jsonl \
    --model gemini-2.5-flash
```

### Evaluation

```bash
# Generate eval artifacts (ratings, retrieval metrics)
python -m src.generate_eval_artifacts \
    --run-results results.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --output-dir eval_output/

# Run evaluation report
python -m src.eval_runner --config eval/eval_spec.json --results-dir eval_output/
```

### Training Data Pipeline

```bash
# Convert agent outputs → SFT training data
python -m src.convert_trajectories \
    --input  results.jsonl \
    --output training_data/conversations.jsonl \
    --mode   locate  # optional filter

# SFT dry run
python -m src.training.sft_train --data training_data/conversations.jsonl --dry-run

# RLVR dry run
python -m src.training.rlvr_train \
    --data eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --dry-run
```

### Environment Setup

```bash
# Python 3.12 required (Tinker needs >= 3.11)
brew install python@3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or requirements_backup.txt
pip install tinker tinker-cookbook

# Required env vars
export GEMINI_API_KEY="..."
export GITHUB_TOKEN="..."
export TINKER_API_KEY="..."   # for training only
```

---

## 11. File Index

| File | Lines | Purpose |
|---|---|---|
| `src/agent_orchestrator.py` | 573 | ReAct agent loop, mode routing, trajectory serialization |
| `src/tool_gateway.py` | 279 | Tool interface wrapping repo ingestion |
| `src/repo_ingestion.py` | 467 | File walking, chunking, substring search, GitHub API |
| `src/session_manager.py` | 127 | Session persistence and query history |
| `src/cli.py` | — | Interactive CLI entry point |
| `src/run_task_batch.py` | 258 | Batch runner for evaluation / trajectory generation |
| `src/eval_runner.py` | 429 | Evaluation metrics computation |
| `src/generate_eval_artifacts.py` | 396 | Automated rating and retrieval scoring |
| `src/convert_trajectories.py` | 176 | Agent outputs → SFT conversations converter |
| `src/training/sft_train.py` | 209 | SFT via Tinker cross-entropy loss |
| `src/training/rlvr_train.py` | 190 | RLVR via Tinker + tinker-cookbook |
| `docs/multi-agent-sft-feasibility.md` | 402 | Feasibility analysis for multi-agent + fine-tuning |
| `eval/eval_spec.json` | 79 | Evaluation rubric and dataset config |
| `eval/prfconnect/` | — | 100-task held-out benchmark + baseline results |
