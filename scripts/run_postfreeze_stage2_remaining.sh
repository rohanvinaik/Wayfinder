#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
ORCH_ROOT="${2:-$RUNS_ROOT/postfreeze_stage2_from_${RUN_NAME}}"

HEADROOM_OUT="${HEADROOM_OUT:-$RUNS_ROOT/exp_som013a_headroom_full_from_${RUN_NAME}}"
DD014A_OUT="${DD014A_OUT:-$RUNS_ROOT/exp_dd014a_eqsat_full_from_${RUN_NAME}}"
DD014B_OUT="${DD014B_OUT:-$RUNS_ROOT/exp_dd014b_proof_dsl_full_from_${RUN_NAME}}"
DD014C_OUT="${DD014C_OUT:-$RUNS_ROOT/exp_dd014c_relational_full_from_${RUN_NAME}}"
DD014D_OUT="${DD014D_OUT:-$RUNS_ROOT/exp_dd014d_integrated_full_from_${RUN_NAME}}"
DD015_DET_OUT="${DD015_DET_OUT:-$RUNS_ROOT/exp_dd015_integrated_bridge_seeded_from_${RUN_NAME}}"

mkdir -p "$ORCH_ROOT"

{
  echo "[postfreeze-stage2] run dir: $RUN_DIR"
  echo "[postfreeze-stage2] orchestration root: $ORCH_ROOT"
  echo "[postfreeze-stage2] start: $(date)"

  echo
  echo "[1/6] exp_som013a headroom full"
  LOCAL_LIMIT="${LOCAL_LIMIT:-128}" \
  PLANNER_LIMIT="${PLANNER_LIMIT:-7}" \
  ORACLE_LIMIT="${ORACLE_LIMIT:-128}" \
  BUDGETS="${BUDGETS:-128,256}" \
  DEPTHS="${DEPTHS:-2,4,8}" \
  DEPTH_TIMEOUT="${DEPTH_TIMEOUT:-45}" \
  ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-60}" \
    "$ROOT/scripts/run_exp_som013a_guarded.sh" "$RUN_DIR" "$HEADROOM_OUT"

  echo
  echo "[2/6] exp_dd014a eqsat backend"
  LIMIT="${DD014_LIMIT:-120}" \
  ROW_TIMEOUT="${DD014_ROW_TIMEOUT:-180}" \
  RESTART_EVERY="${DD014_RESTART_EVERY:-12}" \
  MAX_PROGRAMS="${DD014_MAX_PROGRAMS:-24}" \
  MAX_ROUNDS="${DD014_MAX_ROUNDS:-3}" \
    "$ROOT/scripts/run_exp_dd014a_eqsat_guarded.sh" "$RUN_DIR" "$DD014A_OUT"

  echo
  echo "[3/6] exp_dd014b proof_dsl backend"
  LIMIT="${DD014_LIMIT:-120}" \
  ROW_TIMEOUT="${DD014_ROW_TIMEOUT:-180}" \
  RESTART_EVERY="${DD014_RESTART_EVERY:-12}" \
  MAX_PROGRAMS="${DD014_MAX_PROGRAMS:-24}" \
  MAX_ROUNDS="${DD014_MAX_ROUNDS:-3}" \
    "$ROOT/scripts/run_exp_dd014b_proof_dsl_guarded.sh" "$RUN_DIR" "$DD014B_OUT"

  echo
  echo "[4/6] exp_dd014c relational backend"
  LIMIT="${DD014_LIMIT:-120}" \
  ROW_TIMEOUT="${DD014_ROW_TIMEOUT:-180}" \
  RESTART_EVERY="${DD014_RESTART_EVERY:-12}" \
  MAX_PROGRAMS="${DD014_MAX_PROGRAMS:-24}" \
  MAX_ROUNDS="${DD014_MAX_ROUNDS:-3}" \
    "$ROOT/scripts/run_exp_dd014c_relational_guarded.sh" "$RUN_DIR" "$DD014C_OUT"

  echo
  echo "[5/6] exp_dd014d integrated backend"
  LIMIT="${DD014_LIMIT:-120}" \
  ROW_TIMEOUT="${DD014_ROW_TIMEOUT:-180}" \
  RESTART_EVERY="${DD014_RESTART_EVERY:-12}" \
  MAX_PROGRAMS="${DD014_MAX_PROGRAMS:-24}" \
  MAX_ROUNDS="${DD014_MAX_ROUNDS:-3}" \
    "$ROOT/scripts/run_exp_dd014d_integrated_guarded.sh" "$RUN_DIR" "$DD014D_OUT"

  echo
  echo "[6/6] exp_dd015 deterministic seeded bridge"
  LIMIT="${DD015_LIMIT:-40}" \
  PER_THEOREM_TIMEOUT="${DD015_TIMEOUT:-420}" \
  RESTART_EVERY="${DD015_RESTART_EVERY:-8}" \
  SELECTION_SOURCE="${DD015_SELECTION_SOURCE:-validated_progress}" \
  ALLOW_UNVALIDATED_BACKFILL="${DD015_ALLOW_UNVALIDATED_BACKFILL:-true}" \
    "$ROOT/scripts/run_exp_dd015_integrated_bridge_guarded.sh" "$RUN_DIR" "$DD015_DET_OUT"

  echo
  echo "[postfreeze-stage2] done: $(date)"
} | tee "$ORCH_ROOT/run.log"
