#!/bin/zsh
# Paired theorem-search experiment: Condition A (baseline) then Condition C (Ducky + Wesker v3)
# Reuses the r6 sample manifest for identical theorem set.
# Run from project root: zsh scripts/run_paired_experiment.sh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
MANIFEST="$ROOT/runs/exp_som016_final_random2000_r1/sample_manifest.jsonl"
CHECKPOINT="$ROOT/models/NAV-004_step5000.pt"
CORPUS="$ROOT/data/leandojo_mathlib.jsonl"
LEAN_PROJECT="$ROOT/data/lean_project"
FAMILY_CLF="$ROOT/models/som_torch_repair_v1/best.pt"
EXEC_SELECTOR="$ROOT/models/apply_exec_selector_v2.pt"
PROBE_SELECTOR="$ROOT/models/apply_exec_selector_v1.pt"
APPLY_TRIGGER="$ROOT/models/apply_trigger_v3.pt"

CONDITION="${1:-A}"

case "$CONDITION" in
  A)
    echo "=== Condition A: Baseline (no Ducky) ==="
    OUT="$ROOT/runs/paired_condA_baseline"
    mkdir -p "$OUT"
    caffeinate -dimsu python -m scripts.run_exp_som016_final_collect \
      --config "$ROOT/configs/wayfinder.yaml" \
      --checkpoint "$CHECKPOINT" \
      --mathlib-corpus "$CORPUS" \
      --output-dir "$OUT" \
      --device mps \
      --backend pantograph \
      --lean-project "$LEAN_PROJECT" \
      --sample-size 2000 \
      --sample-seed 42 \
      --temporal arbiter_full \
      --budget 600 \
      --per-theorem-timeout 300 \
      --flush-every 1 \
      --selector "$PROBE_SELECTOR" \
      --probe-k 3 \
      --family-classifier-torch-path "$FAMILY_CLF" \
      --exec-apply-selector-path "$EXEC_SELECTOR" \
      --apply-trigger-path "$APPLY_TRIGGER" \
      --sample-manifest "$MANIFEST" \
      --probe-lean \
      --disable-dr-ducky
    ;;
  C)
    echo "=== Condition C: Ducky + Wesker v3 SoM ==="
    OUT="$ROOT/runs/paired_condC_ducky_wesker"
    mkdir -p "$OUT"
    caffeinate -dimsu python -m scripts.run_exp_som016_final_collect \
      --config "$ROOT/configs/wayfinder.yaml" \
      --checkpoint "$CHECKPOINT" \
      --mathlib-corpus "$CORPUS" \
      --output-dir "$OUT" \
      --device mps \
      --backend pantograph \
      --lean-project "$LEAN_PROJECT" \
      --sample-size 2000 \
      --sample-seed 42 \
      --temporal arbiter_full \
      --budget 600 \
      --per-theorem-timeout 300 \
      --flush-every 1 \
      --selector "$PROBE_SELECTOR" \
      --probe-k 3 \
      --family-classifier-torch-path "$FAMILY_CLF" \
      --exec-apply-selector-path "$EXEC_SELECTOR" \
      --apply-trigger-path "$APPLY_TRIGGER" \
      --dr-ducky-max-programs 24 \
      --dr-ducky-max-rounds 3 \
      --dr-ducky-goal-limit 3 \
      --sample-manifest "$MANIFEST" \
      --probe-lean \
      --second-order-model "$ROOT/runs/exp_som013d_wesker_v3/model.pt" \
      --second-order-metadata "$ROOT/data/wesker_v3/features/metadata.json"
    ;;
  *)
    echo "Usage: $0 [A|C]"
    echo "  A = Baseline (no Ducky)"
    echo "  C = Ducky + Wesker v3 SoM"
    exit 1
    ;;
esac
