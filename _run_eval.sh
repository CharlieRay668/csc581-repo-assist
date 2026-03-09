#!/bin/bash
cd "/Users/charlieray/Desktop/CSC 581/csc581-repo-assist"
source venv/bin/activate
set -a
source .env
set +a
export PYTHONUNBUFFERED=1
python -m src.training.eval_sft \
    --tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --spec eval/eval_spec.json \
    --checkpoint training_data/last_checkpoint.txt \
    --output-dir eval/prfconnect/sft_results \
    --baseline eval/prfconnect/real_results/output.prfc-connect.auto.json
