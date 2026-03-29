#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="$ROOT/runs/exp_som012_hard_eval_r2"
OUT_DIR="$RUN_DIR/bundle/dr_ducky"

export PYTHONUNBUFFERED=1

rm -f \
  "$OUT_DIR/executor_validation_stratified120_summary.json" \
  "$OUT_DIR/executor_validation_stratified120_rows.jsonl" \
  "$OUT_DIR/executor_validation_stratified120_engine_outcomes.jsonl" \
  "$OUT_DIR/executor_validation_stratified120_projector_outcomes.jsonl" \
  "$OUT_DIR/executor_validation_stratified120_closure_report.json" \
  "$OUT_DIR/ablation_single_goal_near_miss_summary.json" \
  "$OUT_DIR/ablation_single_goal_near_miss_rows.jsonl" \
  "$OUT_DIR/ablation_single_goal_near_miss_engine_outcomes.jsonl" \
  "$OUT_DIR/ablation_single_goal_near_miss_projector_outcomes.jsonl" \
  "$OUT_DIR/ablation_single_goal_near_miss_closure_report.json" \
  "$OUT_DIR/ablation_single_goal_stall_summary.json" \
  "$OUT_DIR/ablation_single_goal_stall_rows.jsonl" \
  "$OUT_DIR/ablation_single_goal_stall_engine_outcomes.jsonl" \
  "$OUT_DIR/ablation_single_goal_stall_projector_outcomes.jsonl" \
  "$OUT_DIR/ablation_single_goal_stall_closure_report.json" \
  "$OUT_DIR/ablation_multi_goal_small_progress_summary.json" \
  "$OUT_DIR/ablation_multi_goal_small_progress_rows.jsonl" \
  "$OUT_DIR/ablation_multi_goal_small_progress_engine_outcomes.jsonl" \
  "$OUT_DIR/ablation_multi_goal_small_progress_projector_outcomes.jsonl" \
  "$OUT_DIR/ablation_multi_goal_small_progress_closure_report.json" \
  "$OUT_DIR/ablation_multi_goal_large_progress_summary.json" \
  "$OUT_DIR/ablation_multi_goal_large_progress_rows.jsonl" \
  "$OUT_DIR/ablation_multi_goal_large_progress_engine_outcomes.jsonl" \
  "$OUT_DIR/ablation_multi_goal_large_progress_projector_outcomes.jsonl" \
  "$OUT_DIR/ablation_multi_goal_large_progress_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 120 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --output-json "$OUT_DIR/executor_validation_stratified120_summary.json" \
  --output-jsonl "$OUT_DIR/executor_validation_stratified120_rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_DIR/executor_validation_stratified120_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_DIR/executor_validation_stratified120_projector_outcomes.jsonl" \
  --closure-report-json "$OUT_DIR/executor_validation_stratified120_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 80 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --residual-bucket single_goal_near_miss \
  --output-json "$OUT_DIR/ablation_single_goal_near_miss_summary.json" \
  --output-jsonl "$OUT_DIR/ablation_single_goal_near_miss_rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_DIR/ablation_single_goal_near_miss_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_DIR/ablation_single_goal_near_miss_projector_outcomes.jsonl" \
  --closure-report-json "$OUT_DIR/ablation_single_goal_near_miss_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 80 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --residual-bucket single_goal_stall \
  --output-json "$OUT_DIR/ablation_single_goal_stall_summary.json" \
  --output-jsonl "$OUT_DIR/ablation_single_goal_stall_rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_DIR/ablation_single_goal_stall_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_DIR/ablation_single_goal_stall_projector_outcomes.jsonl" \
  --closure-report-json "$OUT_DIR/ablation_single_goal_stall_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 80 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --residual-bucket multi_goal_small_progress \
  --output-json "$OUT_DIR/ablation_multi_goal_small_progress_summary.json" \
  --output-jsonl "$OUT_DIR/ablation_multi_goal_small_progress_rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_DIR/ablation_multi_goal_small_progress_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_DIR/ablation_multi_goal_small_progress_projector_outcomes.jsonl" \
  --closure-report-json "$OUT_DIR/ablation_multi_goal_small_progress_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 80 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --residual-bucket multi_goal_large_progress \
  --output-json "$OUT_DIR/ablation_multi_goal_large_progress_summary.json" \
  --output-jsonl "$OUT_DIR/ablation_multi_goal_large_progress_rows.jsonl" \
  --engine-outcomes-jsonl "$OUT_DIR/ablation_multi_goal_large_progress_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$OUT_DIR/ablation_multi_goal_large_progress_projector_outcomes.jsonl" \
  --closure-report-json "$OUT_DIR/ablation_multi_goal_large_progress_closure_report.json"
