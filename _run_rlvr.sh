#!/usr/bin/env bash
# Run RLVR training, then evaluate the resulting checkpoint.
# Usage:  bash _run_rlvr.sh
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
set -a && source .env && set +a

TASKS="eval/prfconnect/tasks.prfc-connect.heldout.jsonl"
ANSWERS="eval/prfconnect/answers.prfc-connect.heldout.jsonl"
SPEC="eval/eval_spec.json"
BASELINE="eval/prfconnect/real_results/output.prfc-connect.auto.json"
LOG_DIR="training_data/rlvr_logs"
EVAL_DIR="eval/prfconnect/rlvr_results"

echo "=== Stage 1: RLVR Training ==="
python -m src.training.rlvr_train \
    --data "$TASKS" \
    --answers "$ANSWERS" \
    --model Qwen/Qwen3-8B \
    --lr 1e-5 \
    --lora-rank 32 \
    --group-size 8 \
    --batch-size 4 \
    --max-tokens 2048 \
    --loss-fn importance_sampling \
    --renderer qwen3_disable_thinking \
    --log-path "$LOG_DIR" \
    --load-checkpoint training_data/last_checkpoint.txt \
    --repo-path hack4impact-repos/prfc-connect \
    --eval-every 5 \
    --save-every 5

echo ""
echo "=== Stage 2: Evaluate RLVR checkpoint ==="
RLVR_CKPT="$LOG_DIR/last_checkpoint.txt"
if [ ! -f "$RLVR_CKPT" ]; then
    echo "ERROR: No checkpoint found at $RLVR_CKPT"
    exit 1
fi

mkdir -p "$EVAL_DIR"
python -m src.training.eval_sft \
    --checkpoint "$RLVR_CKPT" \
    --tasks "$TASKS" \
    --answers "$ANSWERS" \
    --spec "$SPEC" \
    --output-dir "$EVAL_DIR" \
    --baseline "$BASELINE" \
    --repo-path hack4impact-repos/prfc-connect

echo ""
echo "=== Done ==="
echo "RLVR scorecard: $EVAL_DIR/output.sft.json"
echo "Compare with SFT: eval/prfconnect/sft_results/output.sft.json"
echo "Compare with baseline: $BASELINE"
