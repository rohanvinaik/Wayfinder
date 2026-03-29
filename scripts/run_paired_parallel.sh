#!/bin/zsh
# Parallel paired experiment: run two halves of the manifest simultaneously.
# Usage: zsh scripts/run_paired_parallel.sh [A|C]
#
# Splits the 2000-theorem manifest into two 1000-theorem halves and runs
# them as independent processes. Merge results with:
#   cat runs/paired_cond${COND}_first/details.jsonl runs/paired_cond${COND}_second/details.jsonl > merged.jsonl

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
CHECKPOINT="$ROOT/models/NAV-004_step5000.pt"
CORPUS="$ROOT/data/leandojo_mathlib.jsonl"
LEAN_PROJECT="$ROOT/data/lean_project"
FAMILY_CLF="$ROOT/models/som_torch_repair_v1/best.pt"
EXEC_SELECTOR="$ROOT/models/apply_exec_selector_v2.pt"
PROBE_SELECTOR="$ROOT/models/apply_exec_selector_v1.pt"
APPLY_TRIGGER="$ROOT/models/apply_trigger_v3.pt"
MANIFEST_FIRST="$ROOT/runs/paired_condA_baseline/manifest_first_half.jsonl"
MANIFEST_SECOND="$ROOT/runs/paired_condA_baseline/manifest_second_half.jsonl"

CONDITION="${1:-A}"

_common_args=(
  --config "$ROOT/configs/wayfinder.yaml"
  --checkpoint "$CHECKPOINT"
  --mathlib-corpus "$CORPUS"
  --device mps
  --backend pantograph
  --lean-project "$LEAN_PROJECT"
  --sample-size 1000
  --sample-seed 42
  --temporal arbiter_full
  --budget 600
  --per-theorem-timeout 300
  --flush-every 1
  --selector "$PROBE_SELECTOR"
  --probe-k 3
  --family-classifier-torch-path "$FAMILY_CLF"
  --exec-apply-selector-path "$EXEC_SELECTOR"
  --apply-trigger-path "$APPLY_TRIGGER"
  --probe-lean
)

case "$CONDITION" in
  A)
    echo "=== Condition A: Baseline (no Ducky) — PARALLEL ==="
    OUT_FIRST="$ROOT/runs/paired_condA_first"
    OUT_SECOND="$ROOT/runs/paired_condA_second"
    EXTRA_ARGS=(--disable-dr-ducky)
    ;;
  C)
    echo "=== Condition C: Ducky + Wesker v3 — PARALLEL ==="
    OUT_FIRST="$ROOT/runs/paired_condC_first"
    OUT_SECOND="$ROOT/runs/paired_condC_second"
    EXTRA_ARGS=(
      --dr-ducky-max-programs 24
      --dr-ducky-max-rounds 3
      --dr-ducky-goal-limit 3
      --second-order-model "$ROOT/runs/exp_som013d_wesker_v3/model.pt"
      --second-order-metadata "$ROOT/data/wesker_v3/features/metadata.json"
    )
    ;;
  *)
    echo "Usage: $0 [A|C]"
    exit 1
    ;;
esac

mkdir -p "$OUT_FIRST" "$OUT_SECOND"

echo "Launching first half (theorems 1-1000) → $OUT_FIRST"
caffeinate -dimsu python -m scripts.run_exp_som016_final_collect \
  "${_common_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "$OUT_FIRST" \
  --sample-manifest "$MANIFEST_FIRST" \
  > "$OUT_FIRST/nohup.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

echo "Launching second half (theorems 1001-2000) → $OUT_SECOND"
caffeinate -dimsu python -m scripts.run_exp_som016_final_collect \
  "${_common_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "$OUT_SECOND" \
  --sample-manifest "$MANIFEST_SECOND" \
  > "$OUT_SECOND/nohup.log" 2>&1 &
PID2=$!
echo "  PID: $PID2"

echo ""
echo "Both halves running. Monitor with:"
echo "  zsh scripts/dashboard.sh"
echo ""
echo "To merge results when done:"
echo "  cat $OUT_FIRST/details.jsonl $OUT_SECOND/details.jsonl > runs/paired_cond${CONDITION}_merged/details.jsonl"
echo ""
echo "PIDs: $PID1, $PID2"
