#!/usr/bin/env bash
# Parallel EXP-050 trigger collection across N shards.
# Each shard spawns its own Pantograph server and writes to a separate output file.
# After all shards complete, merge into a single JSONL.
#
# Usage:
#   bash scripts/collect_trigger_parallel.sh [NUM_SHARDS] [TOTAL_THEOREMS]
#
# Defaults: 8 shards, 29186 theorems (full apply-shaped set)

set -euo pipefail

NUM_SHARDS="${1:-8}"
TOTAL="${2:-29186}"
SHARD_SIZE=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))

THEOREMS="data/apply_theorems_full.jsonl"
CONFIG="configs/wayfinder.yaml"
CHECKPOINT="models/NAV-004_step5000.pt"
SELECTOR="models/apply_exec_selector_v1.pt"
LEAN_PROJECT="data/lean_project/"
DB="data/proof_network_v3.db"
OUTPUT_DIR="data/trigger_shards"
FINAL_OUTPUT="data/apply_trigger_train_full.jsonl"
LOG_DIR="runs/trigger_shards"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=== EXP-050 Parallel Collection ==="
echo "  Shards:          $NUM_SHARDS"
echo "  Total theorems:  $TOTAL"
echo "  Shard size:      $SHARD_SIZE"
echo "  Output:          $FINAL_OUTPUT"
echo ""

PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    OFFSET=$(( i * SHARD_SIZE ))
    LIMIT=$SHARD_SIZE
    OUT="$OUTPUT_DIR/shard_${i}.jsonl"
    LOG="$LOG_DIR/shard_${i}.log"

    echo "  Launching shard $i: offset=$OFFSET limit=$LIMIT → $OUT"

    python3 -m scripts.collect_trigger_states \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --selector "$SELECTOR" \
        --theorems "$THEOREMS" \
        --lean-project "$LEAN_PROJECT" \
        --backend pantograph \
        --lean-imports Mathlib \
        --db "$DB" \
        --output "$OUT" \
        --offset "$OFFSET" \
        --limit "$LIMIT" \
        --probe-lean \
        --probe-k 5 \
        --budget 300 \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_SHARDS shards launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f runs/trigger_shards/shard_*.log"
echo ""

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "  Shard $i (PID $PID): done"
    else
        echo "  Shard $i (PID $PID): FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Merging shards → $FINAL_OUTPUT ==="
cat "$OUTPUT_DIR"/shard_*.jsonl > "$FINAL_OUTPUT"
TOTAL_ROWS=$(wc -l < "$FINAL_OUTPUT")
POSITIVES=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$FINAL_OUTPUT')]
pos = sum(1 for r in rows if r.get('can_apply') == 1)
by_stage = {}
for r in rows:
    s = r.get('search_stage','?')
    by_stage.setdefault(s,[0,0])
    by_stage[s][1] += 1
    if r.get('trigger_label') == 1:
        by_stage[s][0] += 1
print(f'Total rows: {len(rows)}, Positives: {pos} ({100*pos/max(len(rows),1):.1f}%)')
for s,(p,n) in sorted(by_stage.items()):
    print(f'  {s}: {p}/{n} ({100*p/max(n,1):.1f}%)')
" 2>/dev/null)

echo "$POSITIVES"
echo ""
echo "=== Collection complete ==="
echo "  Shards failed: $FAILED"
echo "  Total rows: $TOTAL_ROWS"
echo "  Output: $FINAL_OUTPUT"
echo ""
echo "Next: python3 -m scripts.train_trigger_classifier --data $FINAL_OUTPUT --output models/apply_trigger_v1.pt"

exit $FAILED
