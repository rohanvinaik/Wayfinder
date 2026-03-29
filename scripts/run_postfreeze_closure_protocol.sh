#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
FEATURE_DIR="${FEATURE_DIR:-$RUN_DIR/bundle/second_order_som/features}"
TRAIN_OUT="${TRAIN_OUT:-$RUNS_ROOT/exp_som013d_train_second_order_from_${RUN_NAME}}"
BRIDGE_OUT="${BRIDGE_OUT:-$RUNS_ROOT/exp_dd015_integrated_bridge_from_${RUN_NAME}}"
LEARNED_BRIDGE_OUT="${LEARNED_BRIDGE_OUT:-$RUNS_ROOT/exp_dd015_integrated_bridge_learned_from_${RUN_NAME}}"
RUN_HEADROOM="${RUN_HEADROOM:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_BRIDGE_DETERMINISTIC="${RUN_BRIDGE_DETERMINISTIC:-1}"
RUN_BRIDGE_LEARNED="${RUN_BRIDGE_LEARNED:-1}"

if [[ "$RUN_HEADROOM" == "1" ]]; then
  "$ROOT/scripts/run_exp_som013a_guarded.sh" "$RUN_DIR"
fi

python -m scripts.build_second_order_feature_dataset \
  --packets "$RUN_DIR/bundle/second_order_som/second_order_packets.jsonl" \
  --output-dir "$FEATURE_DIR"

if [[ "$RUN_TRAIN" == "1" ]]; then
  "$ROOT/scripts/run_exp_som013d_train_second_order_guarded.sh" "$RUN_DIR" "$TRAIN_OUT"
fi

if [[ "$RUN_BRIDGE_DETERMINISTIC" == "1" ]]; then
  "$ROOT/scripts/run_exp_dd015_integrated_bridge_guarded.sh" "$RUN_DIR" "$BRIDGE_OUT"
fi

if [[ "$RUN_BRIDGE_LEARNED" == "1" ]]; then
  export SECOND_ORDER_MODEL="$TRAIN_OUT/model.pt"
  export SECOND_ORDER_METADATA="$TRAIN_OUT/metadata_snapshot.json"
  "$ROOT/scripts/run_exp_dd015_integrated_bridge_guarded.sh" "$RUN_DIR" "$LEARNED_BRIDGE_OUT"
fi
