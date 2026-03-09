#!/bin/bash
cd /Users/charlieray/Desktop/CSC\ 581/csc581-repo-assist
source venv/bin/activate
set -a
source .env
set +a
export PYTHONUNBUFFERED=1
python -m src.training.sft_train \
    --data training_data/conversations.jsonl \
    --model Qwen/Qwen3-8B \
    --epochs 7 \
    --lr 1e-4 \
    --batch-size 4 \
    --lora-rank 32 \
    --eval-tasks eval/prfconnect/tasks.prfc-connect.heldout.jsonl \
    --eval-answers eval/prfconnect/answers.prfc-connect.heldout.jsonl \
    --eval-spec eval/eval_spec.json \
    --eval-output-dir eval/prfconnect/sft_results \
    --eval-baseline eval/prfconnect/real_results/output.prfc-connect.auto.json
