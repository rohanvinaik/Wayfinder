#!/bin/bash
set -euo pipefail
cd /Users/rohanvinaik/Projects/Wayfinder

COMMON="--config configs/wayfinder.yaml \
    --checkpoint models/NAV-002_step5000.pt \
    --backend pantograph \
    --lean-project data/lean_project \
    --lean-imports Mathlib \
    --device mps \
    --theorems data/mathlib_benchmark_50.jsonl"

EXTRA_FLAGS="${EXTRA_FLAGS:-}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-5}"
PYTHON_RUNNER="${PYTHON_RUNNER:-PYTHONUNBUFFERED=1 python -u -m scripts.run_benchmark}"

echo "=== Starting 3-mode Mathlib benchmark $(date) ==="

echo "--- Mode 1: learned_only ---"
eval "$PYTHON_RUNNER" $COMMON \
    --search-mode learned_only \
    --output runs/mathlib-learned-only/benchmark_results.json \
    $EXTRA_FLAGS \
    2>&1 | tee runs/mathlib-learned-only.log | tail -"$LOG_TAIL_LINES"
echo "--- Mode 1 done $(date) ---"

echo "--- Mode 2: learned_structural ---"  
eval "$PYTHON_RUNNER" $COMMON \
    --search-mode learned_structural \
    --output runs/mathlib-learned-structural/benchmark_results.json \
    $EXTRA_FLAGS \
    2>&1 | tee runs/mathlib-learned-structural.log | tail -"$LOG_TAIL_LINES"
echo "--- Mode 2 done $(date) ---"

echo "--- Mode 3: full ---"
eval "$PYTHON_RUNNER" $COMMON \
    --search-mode full \
    --output runs/mathlib-full/benchmark_results.json \
    $EXTRA_FLAGS \
    2>&1 | tee runs/mathlib-full.log | tail -"$LOG_TAIL_LINES"
echo "--- Mode 3 done $(date) ---"

echo "=== All modes complete $(date) ==="

# Run analysis on each
for mode in learned-only learned-structural full; do
    echo ""
    echo "=== Analysis: $mode ==="
    python -m scripts.analyze_search_traces \
        runs/mathlib-$mode/benchmark_results.jsonl 2>&1 | head -25
done
