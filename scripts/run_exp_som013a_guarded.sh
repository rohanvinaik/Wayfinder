#!/bin/zsh
set -euo pipefail

RUN_DIR="${1:-/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2}"
BUNDLE_DIR="$RUN_DIR/bundle"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
OUT_ROOT="${2:-$RUNS_ROOT/exp_som013a_headroom_from_${RUN_NAME}}"

LOCAL_LIMIT="${LOCAL_LIMIT:-64}"
LOCAL_OFFSET="${LOCAL_OFFSET:-0}"
PLANNER_LIMIT="${PLANNER_LIMIT:-0}"
ORACLE_LIMIT="${ORACLE_LIMIT:-0}"

BUDGETS="${BUDGETS:-128,256}"
DEPTHS="${DEPTHS:-2,4,8}"
DEPTH_TIMEOUT="${DEPTH_TIMEOUT:-45}"
ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-60}"

mkdir -p "$OUT_ROOT"

python -m scripts.preflight_postfreeze_experiments \
  --run-dir "$RUN_DIR" \
  --experiment som013a \
  --output-json "$OUT_ROOT/preflight.json"

planner_args=()
oracle_args=()
if [[ "$PLANNER_LIMIT" != "0" ]]; then
  planner_args+=(--limit "$PLANNER_LIMIT")
fi
if [[ "$ORACLE_LIMIT" != "0" ]]; then
  oracle_args+=(--limit "$ORACLE_LIMIT")
fi

echo "[exp-som-013a] run dir: $RUN_DIR"
echo "[exp-som-013a] output root: $OUT_ROOT"
echo "[exp-som-013a] local limit/offset: $LOCAL_LIMIT / $LOCAL_OFFSET"
echo "[exp-som-013a] planner limit: $PLANNER_LIMIT"
echo "[exp-som-013a] oracle limit: $ORACLE_LIMIT"
echo "[exp-som-013a] budgets/depths: $BUDGETS / $DEPTHS"

python -m scripts.run_exp_som012_depth_ladder \
  --inputs "$BUNDLE_DIR/hard_proof_planner.jsonl" \
  --output-dir "$OUT_ROOT/depth_ladder_planner" \
  --budgets "$BUDGETS" \
  --depths "$DEPTHS" \
  --per-theorem-timeout "$DEPTH_TIMEOUT" \
  "${planner_args[@]}" \
  --cosine-rw \
  --cosine-rw-seq

python -m scripts.run_exp_som012_depth_ladder \
  --inputs "$BUNDLE_DIR/hard_proof_local.jsonl" \
  --output-dir "$OUT_ROOT/depth_ladder_local" \
  --budgets "$BUDGETS" \
  --depths "$DEPTHS" \
  --per-theorem-timeout "$DEPTH_TIMEOUT" \
  --limit "$LOCAL_LIMIT" \
  --offset "$LOCAL_OFFSET" \
  --cosine-rw \
  --cosine-rw-seq

python -m scripts.run_exp_som012_oracle_gap \
  --inputs "$BUNDLE_DIR/hard_proof_all.jsonl" \
  --resolution-packets "$BUNDLE_DIR/hard_resolution_layer/resolution_packets.jsonl" \
  --output-dir "$OUT_ROOT/oracle_gap" \
  --per-theorem-timeout "$ORACLE_TIMEOUT" \
  "${oracle_args[@]}"
