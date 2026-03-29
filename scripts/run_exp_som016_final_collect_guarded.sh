#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
OUT_ROOT="${1:-$ROOT/runs/exp_som016_final_random2000_r1}"

CHECKPOINT="${CHECKPOINT:-$ROOT/models/NAV-004_step5000.pt}"
MATHLIB_CORPUS="${MATHLIB_CORPUS:-$ROOT/data/leandojo_mathlib.jsonl}"
LEAN_PROJECT="${LEAN_PROJECT:-$ROOT/data/lean_project}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/wayfinder.yaml}"
DEVICE="${DEVICE:-mps}"
BACKEND="${BACKEND:-pantograph}"

SAMPLE_SIZE="${SAMPLE_SIZE:-2000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"
BUDGET="${BUDGET:-600}"
PER_THEOREM_TIMEOUT="${PER_THEOREM_TIMEOUT:-300}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
PROBE_K="${PROBE_K:-3}"
PROBE_LEAN="${PROBE_LEAN:-1}"
RESUME="${RESUME:-0}"
TEMPORAL_MODE="${TEMPORAL_MODE:-arbiter_full}"
SELECTOR_PATH="${SELECTOR_PATH:-$ROOT/models/apply_exec_selector_v1.pt}"
EXEC_APPLY_SELECTOR_PATH="${EXEC_APPLY_SELECTOR_PATH:-$ROOT/models/apply_exec_selector_v2.pt}"
APPLY_TRIGGER_PATH="${APPLY_TRIGGER_PATH:-$ROOT/models/apply_trigger_v3.pt}"
FAMILY_CLASSIFIER_TORCH_PATH="${FAMILY_CLASSIFIER_TORCH_PATH:-$ROOT/models/som_torch_v1/best.pt}"
DISABLE_DR_DUCKY="${DISABLE_DR_DUCKY:-0}"
DR_DUCKY_MAX_PROGRAMS="${DR_DUCKY_MAX_PROGRAMS:-24}"
DR_DUCKY_MAX_ROUNDS="${DR_DUCKY_MAX_ROUNDS:-3}"
DR_DUCKY_GOAL_LIMIT="${DR_DUCKY_GOAL_LIMIT:-3}"

mkdir -p "$OUT_ROOT"

args=(
  -m scripts.run_exp_som016_final_collect
  --config "$CONFIG_PATH"
  --checkpoint "$CHECKPOINT"
  --mathlib-corpus "$MATHLIB_CORPUS"
  --output-dir "$OUT_ROOT"
  --device "$DEVICE"
  --backend "$BACKEND"
  --lean-project "$LEAN_PROJECT"
  --sample-size "$SAMPLE_SIZE"
  --sample-seed "$SAMPLE_SEED"
  --temporal "$TEMPORAL_MODE"
  --budget "$BUDGET"
  --per-theorem-timeout "$PER_THEOREM_TIMEOUT"
  --flush-every "$FLUSH_EVERY"
  --selector "$SELECTOR_PATH"
  --probe-k "$PROBE_K"
  --family-classifier-torch-path "$FAMILY_CLASSIFIER_TORCH_PATH"
  --exec-apply-selector-path "$EXEC_APPLY_SELECTOR_PATH"
  --apply-trigger-path "$APPLY_TRIGGER_PATH"
  --dr-ducky-max-programs "$DR_DUCKY_MAX_PROGRAMS"
  --dr-ducky-max-rounds "$DR_DUCKY_MAX_ROUNDS"
  --dr-ducky-goal-limit "$DR_DUCKY_GOAL_LIMIT"
)

if [[ -n "$SAMPLE_MANIFEST" ]]; then
  args+=(--sample-manifest "$SAMPLE_MANIFEST")
fi

if [[ "$PROBE_LEAN" == "1" ]]; then
  args+=(--probe-lean)
fi

if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi

if [[ "$DISABLE_DR_DUCKY" == "1" ]]; then
  args+=(--disable-dr-ducky)
fi

echo "[exp-som-016] output: $OUT_ROOT"
echo "[exp-som-016] sample size/seed: $SAMPLE_SIZE / $SAMPLE_SEED"
echo "[exp-som-016] sample manifest: ${SAMPLE_MANIFEST:-<fresh sample>}"
echo "[exp-som-016] budget: $BUDGET"
echo "[exp-som-016] timeout: $PER_THEOREM_TIMEOUT"
echo "[exp-som-016] temporal mode: $TEMPORAL_MODE"
echo "[exp-som-016] family classifier: $FAMILY_CLASSIFIER_TORCH_PATH"
echo "[exp-som-016] probe lean: $PROBE_LEAN"
echo "[exp-som-016] resume: $RESUME"

exec caffeinate -dimsu python "${args[@]}" 2>&1 | tee "$OUT_ROOT/launcher.log"
