#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
OUT_ROOT="${2:-$RUNS_ROOT/exp_dd015_integrated_bridge_from_${RUN_NAME}}"

LIMIT="${LIMIT:-40}"
PER_THEOREM_TIMEOUT="${PER_THEOREM_TIMEOUT:-420}"
RESTART_EVERY="${RESTART_EVERY:-8}"
DUCKY_FIRST_MAX_PROGRAMS="${DUCKY_FIRST_MAX_PROGRAMS:-24}"
DUCKY_FIRST_MAX_ROUNDS="${DUCKY_FIRST_MAX_ROUNDS:-3}"
DUCKY_SECOND_MAX_PROGRAMS="${DUCKY_SECOND_MAX_PROGRAMS:-20}"
DUCKY_SECOND_MAX_ROUNDS="${DUCKY_SECOND_MAX_ROUNDS:-2}"
SECOND_ORDER_MODEL="${SECOND_ORDER_MODEL:-}"
SECOND_ORDER_METADATA="${SECOND_ORDER_METADATA:-}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-cpu}"
THEOREM_ID="${THEOREM_ID:-}"
SELECTION_SOURCE="${SELECTION_SOURCE:-validated_progress}"
VALIDATED_SEED_PATH="${VALIDATED_SEED_PATH:-}"
ALLOW_UNVALIDATED_BACKFILL="${ALLOW_UNVALIDATED_BACKFILL:-true}"

mkdir -p "$OUT_ROOT"

python -m scripts.preflight_postfreeze_experiments \
  --run-dir "$RUN_DIR" \
  --experiment dd015 \
  --output-json "$OUT_ROOT/preflight.json"

cmd=(
  python -m scripts.run_exp_dd015_integrated_bridge
  --run-dir "$RUN_DIR"
  --output-dir "$OUT_ROOT"
  --limit "$LIMIT"
  --per-theorem-timeout "$PER_THEOREM_TIMEOUT"
  --restart-every "$RESTART_EVERY"
  --ducky-first-max-programs "$DUCKY_FIRST_MAX_PROGRAMS"
  --ducky-first-max-rounds "$DUCKY_FIRST_MAX_ROUNDS"
  --ducky-second-max-programs "$DUCKY_SECOND_MAX_PROGRAMS"
  --ducky-second-max-rounds "$DUCKY_SECOND_MAX_ROUNDS"
  --controller-device "$CONTROLLER_DEVICE"
  --selection-source "$SELECTION_SOURCE"
)

if [[ -n "$SECOND_ORDER_MODEL" ]]; then
  cmd+=(--second-order-model "$SECOND_ORDER_MODEL")
fi
if [[ -n "$SECOND_ORDER_METADATA" ]]; then
  cmd+=(--second-order-metadata "$SECOND_ORDER_METADATA")
fi
if [[ -n "$THEOREM_ID" ]]; then
  cmd+=(--theorem-id "$THEOREM_ID")
fi
if [[ -n "$VALIDATED_SEED_PATH" ]]; then
  cmd+=(--validated-seed-path "$VALIDATED_SEED_PATH")
fi
if [[ "$ALLOW_UNVALIDATED_BACKFILL" == "false" ]]; then
  cmd+=(--no-allow-unvalidated-backfill)
fi

"${cmd[@]}"
