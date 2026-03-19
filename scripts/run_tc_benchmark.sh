#!/bin/bash
set -e
cd /Users/rohanvinaik/Projects/Wayfinder

COMMON="--config configs/wayfinder.yaml \
    --checkpoint models/NAV-002_step5000.pt \
    --backend pantograph \
    --lean-project data/lean_project \
    --lean-imports Mathlib \
    --device mps \
    --theorems data/mathlib_benchmark_50.jsonl"

echo "=== TC Benchmark $(date) ==="

for mode in off shadow active; do
    echo "--- Mode: $mode ---"
    python -m scripts.run_benchmark $COMMON \
        --temporal $mode \
        --output runs/tc-mathlib-$mode/benchmark_results.json \
        2>&1 | tail -10
    echo "--- $mode done $(date) ---"
done

echo "=== All modes complete ==="

# Summary
for mode in off shadow active; do
    python3 -c "
import json
with open('runs/tc-mathlib-$mode/benchmark_results.json') as f:
    r = json.load(f)
bm = r['benchmark']
print(f'$mode: {bm[\"raw_success\"]}/{bm[\"total_theorems\"]} ({100*bm[\"raw_success_rate\"]:.0f}%) time={r[\"efficiency\"][\"total_time_s\"]:.0f}s')
lanes = bm.get('by_close_lane', {})
for l,c in sorted(lanes.items()):
    if c: print(f'  {l}: {c}')
"
done
