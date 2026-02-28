# Evaluation Files

This directory contains a minimal evaluation framework for pipeline validation.

## Files

- `eval_spec.json`: rubric, sampling, metric configuration, and reporting rules.
- `tasks.example.jsonl`: task metadata template.
- `ratings.example.jsonl`: human/LLM-judge rubric labels template.
- `run_results.example.jsonl`: task outcomes + performance + grounding template.
- `retrieval.example.jsonl`: retrieval ranking template.

## Run

```bash
python -m src.eval_runner \
  --spec eval/eval_spec.json \
  --ratings eval/ratings.example.jsonl \
  --results eval/run_results.example.jsonl \
  --retrieval eval/retrieval.example.jsonl \
  --output eval/report.json
```

