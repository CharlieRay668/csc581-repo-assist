# Fine-Tuning a Code-Understanding Model

## Overview

We built an AI assistant that answers developer questions about a codebase — things like "Where is the login handler implemented?" or "How does the referral email system work?" Our baseline system uses **Gemini 2.5 Flash** (a large commercial model from Google) to power a ReAct agent that can search code, read files, and compose answers.

The goal of this work was to **fine-tune a smaller open-source model (Qwen3-8B)** to perform the same task — answering repository questions — without needing the multi-step tool-calling loop that the Gemini agent relies on. Instead of teaching the model to *use tools*, we teach it to *directly produce good answers* by learning from the Gemini agent's best outputs.

---

## The Target Repository

All training and evaluation is done against a single real-world codebase: **PRFC Connect**, a Next.js web application built by Hack4Impact. It's a referral management system with:

- ~163 source files (TypeScript, React, Prisma ORM)
- API routes, server actions, authentication, email services
- Unit tests, end-to-end tests, database migrations
- Documentation and architecture decision records

This gives us a realistic, mid-sized codebase where questions range from simple file lookups to nuanced architectural explanations.

---

## How We Built the Training Data

### Step 1: Generate Questions (100 Tasks)

We created 100 hand-written questions about the PRFC Connect codebase, split across three difficulty levels and three answer modes:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy       | 40    | Single-file lookups ("Where is X defined?") |
| Medium     | 40    | Multi-file explanations ("How does the auth flow work?") |
| Hard       | 20    | Architectural suggestions ("What should we build next?") |

| Mode     | Count | What the model should produce |
|----------|-------|-------------------------------|
| Locate   | 40    | File paths and line numbers |
| Explain  | 40    | Detailed explanations with code citations |
| Suggest  | 20    | Development recommendations with effort/impact ratings |

**Example tasks:**
- *"Where is the `POST /api/referrals` handler implemented?"* (easy / locate)
- *"How does the email service send referral emails?"* (medium / explain)
- *"What new features would improve the admin dashboard?"* (hard / suggest)

### Step 2: Run the Gemini Agent on Every Question

We ran our full Gemini-powered ReAct agent on all 100 questions. For each question, the agent:

1. Reads the question
2. Decides which tool to call (search code, open a file, list files, etc.)
3. Reads the tool result
4. Repeats steps 2–3 as needed (typically 2–6 tool calls)
5. Produces a final answer

This generates a **trajectory** — the full conversation including the system prompt, the user question, every tool call and its result, and the final answer. Here's a simplified example:

```
System:  You are an expert repository assistant. Mode: LOCATE...
User:    Where is the POST /api/referrals handler implemented?
Agent:   [calls search_repo("POST /api/referrals")]
Tool:    {results: [...]}
Agent:   [calls open_file("src/app/api/referral/route.ts")]
Tool:    {file content...}
Agent:   The POST handler is in src/app/api/referral/route.ts (lines 11-77)...
```

### Step 3: Convert Trajectories to Training Conversations

Each trajectory is converted into a multi-turn conversation in the standard chat format that language models expect:

```json
{
  "task_id": "PRFC-E001",
  "mode": "locate",
  "messages": [
    {"role": "system", "content": "You are an expert repository assistant..."},
    {"role": "user", "content": "Where is the POST /api/referrals handler..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "{results: [...]}"},
    {"role": "assistant", "content": "The handler is in src/app/api/referral/route.ts..."}
  ]
}
```

**Training data statistics:**
- **100 conversations** total
- **6 to 39 messages** per conversation (average: 13.8)
- Every conversation ends with an assistant message (the final answer)
- Messages include system prompts, user questions, tool calls, tool results, and final answers

The key insight: the model learns to produce the *same quality of answer* that the multi-step Gemini agent produces, but in a single forward pass — no tool calls needed at inference time.

---

## The Fine-Tuning Process

### Base Model

**Qwen3-8B** — an 8-billion-parameter open-source language model. It has strong code understanding capabilities and supports structured chat formats.

### Training Method: LoRA (Low-Rank Adaptation)

Rather than updating all 8 billion parameters (which would be extremely expensive), we use **LoRA** — a technique that adds small trainable "adapter" layers on top of the frozen base model. Think of it like adding Post-it notes to a textbook rather than rewriting the whole book.

| Setting | Value |
|---------|-------|
| LoRA rank | 32 |
| Learning rate | 0.0001 |
| Batch size | 4 |
| Optimizer | Adam |
| Epochs | 7 |
| Training examples | 100 |
| Steps per epoch | 25 |
| Total training steps | 175 |

### Training Infrastructure

Training runs on **Tinker**, a cloud GPU platform that provides the base model and handles GPU provisioning. We send our training data to Tinker's API, which runs the actual weight updates on their hardware and stores the resulting checkpoints.

### Training Progress (Loss Curve)

The **loss** measures how far the model's predictions are from the correct training answers. Lower = better.

| Epoch | Average Loss | Change |
|-------|-------------|--------|
| 1     | 0.845       | —      |
| 2     | 0.598       | −29%   |
| 3     | 0.480       | −20%   |
| 4     | 0.337       | −30%   |
| 5     | 0.190       | −44%   |
| 6     | 0.192       | +1%    |
| 7     | 0.186       | −3%    |

The model improves rapidly through epochs 1–5, then plateaus at epochs 5–7 — a sign that it has learned about as much as it can from this dataset. The final loss of 0.186 means the model's predictions closely match the training data.

---

## Evaluation

### What We Measure

We evaluate on the **same 100 tasks** used for training (held-out answer keys were used for scoring — the model never sees the answer keys during training). Each response is scored on a rubric with four criteria:

| Criterion     | Scale | What it measures |
|---------------|-------|------------------|
| Correctness   | 0–2   | Is the answer factually right? |
| Grounding     | 0–2   | Are claims supported by cited code? |
| Relevance     | 0–2   | Does it address the actual question? |
| Clarity       | 0–2   | Is the answer clear and actionable? |

**Total score: 0–8 per task.** A task "passes" if it scores ≥6 with no critical errors (hallucinations, unsafe guidance, or wrong conclusions).

We also measure:
- **File citation accuracy** — does the model point to the right files?
- **Retrieval metrics** (Precision, Recall, nDCG at k=3,5,10) — how well does it find the relevant source files?
- **Latency** — how fast does it respond?

### Understanding the Metrics

Here's what each metric means, how it's computed, and why it matters:

#### Quality Metrics (from the rubric)

**Rubric mean (out of 8)** — The average total score across all tasks. Each task is scored on four criteria (correctness, grounding, relevance, clarity) each worth 0–2, so the maximum is 8. A higher rubric mean means the model consistently produces better answers overall.

**Pass rate** — The percentage of tasks that scored ≥6 out of 8 *and* had no critical errors. This is our primary quality gate. A task can score well on most criteria but still "fail" if it hallucinates a file that doesn't exist (critical error).

**Critical error rate** — The percentage of tasks where the model made a serious mistake: hallucinating code that doesn't exist, giving unsafe guidance, or reaching a fundamentally wrong conclusion. Even a small critical error rate is concerning because it means the model sometimes makes things up.

**Success rate** — The percentage of tasks where the model found the right files and included the right keywords, calibrated by difficulty:
- *Easy tasks*: must find 100% of expected files
- *Medium tasks*: must find ≥50% of expected files and ≥34% of required keywords
- *Hard tasks*: must find ≥34% of expected files and include at least one citation

This is stricter than the rubric pass rate because it checks whether the model actually found the right *specific* information, not just whether the answer reads well.

#### Retrieval Metrics (file citation quality)

These measure how well the model identifies the correct source files. Each task has a set of "expected files" — the files that a correct answer should reference. The model's cited files are treated as a ranked list and compared against those expected files.

**Precision@3** — Of the model's top 3 cited files, what fraction are actually relevant? A score of 0.348 means roughly 1 out of every 3 cited files is correct. Higher is better.

**Recall@3** — Of all the files the model *should* have cited, what fraction did it actually cite in its top 3? A score of 0.528 means the model finds about half the relevant files. Higher is better.

**nDCG@3** — Normalized Discounted Cumulative Gain. This is a standard information retrieval metric that rewards putting the most important files first. If the model cites the right file at position 1, that counts more than citing it at position 3. Scores range from 0 to 1; higher is better. A score of 0.494 means the model's ranking is about halfway to perfect.

> **Why @3?** We focus on the top 3 citations because developers typically look at the first few results. We also compute @5 and @10 but @3 is the primary benchmark.

#### System Metrics

**Mean latency** — Average time to produce a response, in seconds. For the Gemini agent this includes multiple round-trip API calls (search, read file, search again, etc.). For the fine-tuned model this is a single inference pass.

**Cost per task** — Dollar cost of API usage per task. The Gemini agent costs ~$0.015/task in API fees. The fine-tuned model runs on Tinker's GPU infrastructure, which has different cost dynamics (pay for GPU time rather than per-token).

### How We Compute These for the Fine-Tuned Model

**Yes — all metrics are fully automated.** There is no manual grading or human review involved. The entire pipeline runs with a single command and produces the final scorecard JSON.

Here's the two-stage process:

#### Stage 1: Score each task (`generate_eval_artifacts.py`)

This script takes three inputs:
- The **task list** (100 questions with difficulty, mode, etc.)
- The **answer keys** (expected files, required keywords, reference answers)
- The **model outputs** (the generated answers with extracted file citations)

For each task, it automatically computes scores by comparing the model's output against the answer key:

| What it checks | How it checks | What it produces |
|----------------|---------------|------------------|
| Did the model mention the right files? | String-match expected file paths against the answer text | Correctness score (0–2) |
| Did the model include required keywords? | Case-insensitive substring search | Feeds into correctness blend |
| Did the model cite files formally? | Counts file-path citations extracted from the answer | Grounding score (0–2) |
| Did citations point to real files? | Checks if cited files exist on disk | Grounding detail |
| Is the answer the right length? | Word count heuristic (too short = 1, too long = 1, Goldilocks = 2) | Clarity score (0–2) |
| Did the model completely miss? | Known-answer task with 0 expected files found → critical error | Critical error flag |
| Did the model find enough? | Thresholds vary by difficulty (easy=100%, medium=50%, hard=34%) | Success flag |

It produces three JSONL files:
- **ratings.jsonl** — Per-task rubric scores (correctness, grounding, relevance, clarity, critical error)
- **retrieval.jsonl** — Per-task ranked file lists vs. expected files (for Precision/Recall/nDCG)
- **run_results.jsonl** — Per-task success, latency, cost, and grounding details

#### Stage 2: Aggregate into scorecard (`eval_runner.py`)

This script reads the three JSONL files from Stage 1 and computes all the summary metrics:

- **Rubric mean** — averages the per-task total scores
- **Pass rate** — counts tasks with total ≥ 6 and no critical error
- **Precision@k, Recall@k, nDCG@k** — standard IR formulas over the ranked file lists
- **Success rate** — counts tasks that met the difficulty-specific thresholds
- **Bootstrap confidence intervals** — resamples 1,000 times to estimate statistical uncertainty
- **Comparison to baseline** — computes deltas (Δ) against the Gemini results

The final output is a single JSON file (like `output.prfc-connect.auto.json`) with every metric in one place.

#### Running it

The eval script (`eval_sft.py`) chains both stages automatically after generating model answers. It's a single command:

```
python -m src.training.eval_sft \
    --checkpoint "tinker://...checkpoint-path..." \
    --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --eval-spec eval/eval_spec.json \
    --output-dir eval/prfconnect/sft_results \
    --baseline eval/prfconnect/real_results/output.prfc-connect.auto.json
```

This produces model answers → scores them → computes the scorecard → prints a comparison against the Gemini baseline, all without human intervention.

### Baseline Results (Gemini 2.5 Flash Agent)

This is what the full Gemini-powered agent scores on the same 100 tasks:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Rubric mean (out of 8) | 5.84 | Answers are generally good but not perfect |
| Pass rate | 57% | Just over half the answers meet quality bar |
| Critical error rate | 3% | Rarely hallucinates, but it does happen |
| Success rate | 45% | Finds the right files less than half the time |
| Precision@3 | 0.348 | ~1 in 3 cited files is correct |
| Recall@3 | 0.528 | Finds ~half the relevant files |
| nDCG@3 | 0.494 | File ranking quality is middling |
| Mean latency | 21.3 seconds | Includes multiple tool-call round-trips |
| Cost per task | $0.015 | Google API fees per question |

### Fine-Tuned Model Results (100/100 tasks)

All 100 tasks completed successfully with no errors:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Rubric mean (out of 8) | 3.85 | Below the quality bar — answers read okay but miss key details |
| Pass rate | 8% | Very few answers meet the ≥6 + no-critical-error threshold |
| Critical error rate | 33% | One-third of answers contain hallucinated files or wrong conclusions |
| Success rate | 7% | Rarely finds the right specific files |
| Precision@3 | 0.065 | Almost none of the cited files are correct |
| Recall@3 | 0.065 | Finds almost none of the relevant files |
| nDCG@3 | 0.068 | File ranking quality is near zero |
| Mean latency | 27.0 seconds | Single inference pass (no tool calls) |
| Cost per task | $0.014 | Tinker GPU time |

**Per-task-type output characteristics:**
- **Locate tasks** (40): 3–125 words, 6–15 seconds
- **Explain tasks** (40): 425–1,073 words, 27–53 seconds
- **Suggest tasks** (20): 510–1,183 words, 25–55 seconds

### RLVR Model Results (100/100 tasks)

We then trained with RLVR (Reinforcement Learning with Verifiable Rewards), initializing from the SFT checkpoint and running 20 training steps with 8 samples per prompt. The reward signal is based on file-path F1 against ground-truth answer keys.

**RLVR training metrics (20 steps):**

| Step | Correct Rate | Format Rate | Reward |
|------|-------------|-------------|--------|
| 0 | 25.0% | 78.1% | 0.228 |
| 2 | 31.2% | 96.9% | 0.309 |
| 9 | 21.9% | 87.5% | 0.206 |
| 14 | 31.2% | 93.8% | 0.306 |
| 18 | 43.8% | 81.2% | 0.419 |
| 19 | 0.0% | 90.6% | −0.009 |

The reward signal is noisy (high variance across batches of 4 tasks × 8 samples), which is typical for RL with small batch sizes. Step 18 reached the highest correct rate (43.8%) but the final step happened to land on hard tasks.

**Evaluation results:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Rubric mean (out of 8) | 3.65 | Slightly worse than SFT |
| Pass rate | 3% | Almost no answers pass the quality gate |
| Critical error rate | 37% | More hallucinations than SFT |
| Success rate | 3% | Almost never finds the right files |
| Precision@3 | 0.025 | Worse than SFT's already-low 0.065 |
| Recall@3 | 0.021 | Near zero |
| nDCG@3 | 0.020 | Near zero |
| Mean latency | 33.6 seconds | Slower than SFT |
| Cost per task | $0.013 | Similar |

### Three-Way Comparison

| Metric | Gemini Agent | SFT (Qwen3-8B) | RLVR (Qwen3-8B) |
|--------|-------------|-----------------|------------------|
| Rubric mean | **5.84** | 3.85 | 3.65 |
| Pass rate | **57%** | 8% | 3% |
| Critical error rate | **3%** | 33% | 37% |
| Success rate | **45%** | 7% | 3% |
| Precision@3 | **0.348** | 0.065 | 0.025 |
| Recall@3 | **0.528** | 0.065 | 0.021 |
| nDCG@3 | **0.494** | 0.068 | 0.020 |
| Mean latency | **21.3s** | 27.0s | 33.6s |
| Cost/task | $0.015 | $0.014 | $0.013 |

### Analysis: Why Both SFT and RLVR Underperformed

Both fine-tuning approaches scored dramatically worse than the Gemini baseline. RLVR actually performed *slightly worse* than SFT despite being designed to fix SFT's problems. Here's why:

**The fundamental problem: no tool access at inference.** The Gemini agent dynamically searches and reads files during each question. Both fine-tuned models must answer from memory — but 100 training examples isn't enough to memorize the contents and locations of 163 source files.

**Why SFT failed:**
1. **Surface-level imitation** — The model learned to write answers that *look like* the Gemini agent's outputs (reasonable structure, appropriate length) but without factual grounding.
2. **Hallucinated file paths** — 33% critical error rate shows it invents plausible paths that don't exist.

**Why RLVR didn't fix it:**
1. **Reward signal too sparse** — With only 100 tasks and F1-based binary rewards, the model gets very few positive signals to learn from. Most of the 8 samples per prompt are wrong, so GRPO has little contrast to work with.
2. **Small dataset, single pass** — 20 training steps (one pass through 80 tasks at batch size 4) isn't enough for RL to meaningfully shift the distribution. Math-RL benchmarks typically train for thousands of steps on tens of thousands of problems.
3. **The knowledge gap can't be bridged by RL alone** — RLVR can teach the model to *prefer* citing correct files over incorrect ones, but only if the model has some probability of generating correct files in the first place. With P@3 at 0.065 pre-RLVR, the model almost never generates correct paths, so there's no signal to reinforce.
4. **Possible reward hacking** — The model may have learned to generate fewer citations (reducing false positives) rather than learning correct file paths, which would explain the drop from P@3 0.065 → 0.025.

---

## Key Differences: All Three Approaches

| Aspect | Gemini Agent | SFT (Qwen3-8B) | RLVR (SFT→RL) |
|--------|-------------|-----------------|----------------|
| Model size | ~100B+ params (est.) | 8B params | 8B params |
| Training method | N/A (pre-trained) | Supervised (token matching) | RL (reward-based) |
| Training signal | — | "Copy these tokens" | "Did you cite the right files?" |
| API cost | ~$0.015/task | ~$0.014/task | ~$0.013/task |
| Approach | Multi-step tool use | Single-shot generation | Single-shot generation |
| Latency | ~21s | ~27s | ~34s |
| Tool calls | 2–6 per task | 0 | 0 |
| Training data | — | 100 conversations | 80 tasks × 8 samples |
| Training steps | — | 175 (7 epochs) | 20 (1 pass) |
| Pass rate | **57%** | 8% | 3% |
| File accuracy (P@3) | **0.348** | 0.065 | 0.025 |

The core takeaway: **this task requires runtime access to code.** Neither SFT nor RLVR can substitute for the ability to dynamically search and read files during inference. The Gemini agent's advantage isn't its model size — it's its tools.

---

## Limitations & Lessons Learned

1. **Tool access is non-negotiable for retrieval tasks** — Both SFT and RLVR failed because the fine-tuned model has no way to search or read files at inference time. For tasks that require finding specific information in a codebase, a smaller model with tools will always beat a larger model without them.

2. **SFT teaches style, not knowledge** — The SFT model learned to produce well-structured, appropriately-lengthed answers that *read like* expert responses. But it couldn't back them up with correct file paths because it has no runtime access to the code.

3. **RLVR needs sufficient base capability** — RL can only reinforce behaviors the model already exhibits with some probability. With P@3 at 0.065 pre-RLVR, correct file citations were too rare for the reward signal to have meaningful effect.

4. **Small datasets limit both approaches** — 100 examples for SFT and 80 tasks for RLVR are far below typical benchmarks (1,000+ for SFT, 10,000+ for RL). More data, more training steps, and a larger model might partially close the gap.

### What Would Actually Work

Based on these results, the most promising paths forward would be:

- **Fine-tune the model to use tools** (tool-calling SFT) rather than to produce answers directly. Teach the 8B model to emit `search_repo(...)` and `open_file(...)` calls, then use the existing tool infrastructure.
- **RAG (Retrieval-Augmented Generation)** — Embed the codebase into a vector store and inject relevant file snippets into the prompt at inference time, giving the model the context it needs.
- **Significantly more training data** — Scale from 100 to 1,000+ diverse examples, covering more files and more question types, so the model has a realistic chance of memorizing enough of the codebase.

---

## Files and Artifacts

| File | Description |
|------|-------------|
| `training_data/conversations.jsonl` | 100 training conversations |
| `src/training/sft_train.py` | SFT training script (LoRA fine-tuning via Tinker) |
| `src/training/eval_sft.py` | Standalone evaluation script |
| `src/training/rlvr_train.py` | RLVR training script (reinforcement learning via Tinker) |
| `src/convert_trajectories.py` | Converts agent traces → training format |
| `eval/eval_spec.json` | Evaluation rubric and metrics specification |
| `eval/prfconnect/tasks.prfc-connect.heldout.jsonl` | 100 evaluation tasks |
| `eval/prfconnect/answers.prfc-connect.heldout.jsonl` | Answer keys for scoring |
| `eval/prfconnect/real_results/output.prfc-connect.auto.json` | Baseline (Gemini) evaluation results |
| `eval/prfconnect/sft_results/output.sft.json` | SFT model evaluation results (100/100 tasks) |
| `eval/prfconnect/rlvr_results/output.sft.json` | RLVR model evaluation results (100/100 tasks) |
| `training_data/rlvr_logs/metrics.jsonl` | RLVR training metrics (reward curve, 20 steps) |
