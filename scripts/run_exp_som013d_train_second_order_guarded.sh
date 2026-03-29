#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
OUT_ROOT="${2:-$RUNS_ROOT/exp_som013d_train_second_order_from_${RUN_NAME}}"
FEATURE_DIR="${FEATURE_DIR:-$RUN_DIR/bundle/second_order_som/features}"
EPOCHS="${EPOCHS:-24}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-0}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$OUT_ROOT"

python -m scripts.preflight_postfreeze_experiments \
  --run-dir "$RUN_DIR" \
  --experiment som013d \
  --output-json "$OUT_ROOT/preflight.json"

python -m scripts.train_second_order_som \
  --feature-dir "$FEATURE_DIR" \
  --output-dir "$OUT_ROOT" \
  --epochs "$EPOCHS" \
  --stage1-epochs "$STAGE1_EPOCHS" \
  --stage2-epochs "$STAGE2_EPOCHS" \
  --stage3-epochs "$STAGE3_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --device "$DEVICE"
