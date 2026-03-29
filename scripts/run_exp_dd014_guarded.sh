#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
MODE="${2:?mode required: eqsat|proof_dsl|relational|integrated}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"

LIMIT="${LIMIT:-120}"
ROW_TIMEOUT="${ROW_TIMEOUT:-180}"
RESTART_EVERY="${RESTART_EVERY:-12}"
MAX_PROGRAMS="${MAX_PROGRAMS:-24}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

backend_args=()
goal_args=()
experiment=""
run_stem=""

case "$MODE" in
  eqsat)
    experiment="dd014a"
    run_stem="exp_dd014a_eqsat_from_${RUN_NAME}"
    backend_args+=(--backend-family egglog_eqsat)
    goal_args+=(--goal-bucket equality)
    ;;
  proof_dsl)
    experiment="dd014b"
    run_stem="exp_dd014b_proof_dsl_from_${RUN_NAME}"
    backend_args+=(--backend-family rosette_proof_dsl)
    ;;
  relational)
    experiment="dd014c"
    run_stem="exp_dd014c_relational_from_${RUN_NAME}"
    backend_args+=(--backend-family kodkod_relational)
    goal_args+=(--goal-bucket membership --goal-bucket exists --goal-bucket subset)
    ;;
  integrated)
    experiment="dd014d"
    run_stem="exp_dd014d_integrated_from_${RUN_NAME}"
    ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

OUT_ROOT="${3:-$RUNS_ROOT/$run_stem}"
mkdir -p "$OUT_ROOT"

python -m scripts.preflight_postfreeze_experiments \
  --run-dir "$RUN_DIR" \
  --experiment "$experiment" \
  --output-json "$OUT_ROOT/preflight.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit "$LIMIT" \
  --row-timeout-seconds "$ROW_TIMEOUT" \
  --restart-every "$RESTART_EVERY" \
  --max-programs "$MAX_PROGRAMS" \
  --max-rounds "$MAX_ROUNDS" \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  "${backend_args[@]}" \
  "${goal_args[@]}" \
  --output-json "$OUT_ROOT/summary.json" \
  --output-jsonl "$OUT_ROOT/rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_ROOT/engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_ROOT/projector_outcomes.jsonl" \
  --closure-report-json "$OUT_ROOT/closure_report.json"
