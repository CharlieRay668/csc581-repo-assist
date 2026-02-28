# Evaluation Files

This directory contains the evaluation framework and PRFC Connect held-out datasets/results.

## Structure

- `eval_spec.json`: rubric, metric config, reporting rules.
- `prfconnect/tasks.prfc-connect.heldout.jsonl`: 100-task held-out set.
- `prfconnect/answers.prfc-connect.heldout.jsonl`: answer key / expected files and validation hints.
- `prfconnect/seed_results/`: synthetic baseline artifacts.
- `prfconnect/real_results/`: artifacts generated from real agent outputs.

## 1) Run Seed Evaluation

```bash
python3 -m src.eval_runner \
  --spec eval/eval_spec.json \
  --ratings eval/prfconnect/seed_results/ratings.prfc-connect.heldout.seed.jsonl \
  --results eval/prfconnect/seed_results/run_results.prfc-connect.heldout.seed.jsonl \
  --retrieval eval/prfconnect/seed_results/retrieval.prfc-connect.heldout.seed.jsonl \
  --output eval/prfconnect/seed_results/output.prfc-connect.seed.json
```

## 2) Generate Real Agent Outputs (Batch)

```bash
python3 -m src.run_task_batch \
  --repo-path /Users/jasonjelincic/Coding/CSC581/csc581-repo-assist/h4i_repos/prfc-connect \
  --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
  --output eval/prfconnect/real_results/agent_outputs.prfc-connect.heldout.jsonl \
  --default-scope files-only \
  --default-mode explain
```

## 3) Convert Agent Outputs to Eval Artifacts

```bash
python3 -m src.generate_eval_artifacts \
  --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
  --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
  --agent-outputs eval/prfconnect/real_results/agent_outputs.prfc-connect.heldout.jsonl \
  --ratings-out eval/prfconnect/real_results/ratings.prfc-connect.heldout.auto.jsonl \
  --retrieval-out eval/prfconnect/real_results/retrieval.prfc-connect.heldout.auto.jsonl \
  --run-results-out eval/prfconnect/real_results/run_results.prfc-connect.heldout.auto.jsonl \
  --rater-id auto_rater_prfc_v1
```

## 4) Run Evaluation on Real Results

```bash
python3 -m src.eval_runner \
  --spec eval/eval_spec.json \
  --ratings eval/prfconnect/real_results/ratings.prfc-connect.heldout.auto.jsonl \
  --results eval/prfconnect/real_results/run_results.prfc-connect.heldout.auto.jsonl \
  --retrieval eval/prfconnect/real_results/retrieval.prfc-connect.heldout.auto.jsonl \
  --output eval/prfconnect/real_results/output.prfc-connect.auto.json
```
