#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/wayfinder.yaml}"
DEVICE="${DEVICE:-mps}"
TEMPLATE_RUN_ID="${TEMPLATE_RUN_ID:-TC-AUX-001}"
NAV_RUN_ID="${NAV_RUN_ID:-NAV-AUX-001}"
SKIP_DATA="${SKIP_DATA:-0}"
RUN_TEMPLATE="${RUN_TEMPLATE:-1}"
RUN_NAV="${RUN_NAV:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"
BENCHMARK_THEOREMS="${BENCHMARK_THEOREMS:-data/mathlib_benchmark_50.jsonl}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-runs/${NAV_RUN_ID}_benchmark.json}"
BENCHMARK_EXTRA_FLAGS="${BENCHMARK_EXTRA_FLAGS:---search-mode full --cosine-rw --cosine-rw-seq}"

echo "=== Wayfinder main experiment ==="
echo "config: $CONFIG"
echo "device: $DEVICE"
echo "template run: $TEMPLATE_RUN_ID"
echo "navigator run: $NAV_RUN_ID"

if [[ "$SKIP_DATA" != "1" ]]; then
  echo "[1/4] Enhanced controller data pipeline"
  ./scripts/run_enhanced_controller_pipeline.sh
else
  echo "[1/4] Skipping data pipeline"
fi

if [[ "$RUN_TEMPLATE" == "1" ]]; then
  echo "[2/4] Train template classifier with move supervision"
  python -m scripts.train_template_classifier \
    --config "$CONFIG" \
    --run-id "$TEMPLATE_RUN_ID" \
    --device "$DEVICE"
else
  echo "[2/4] Skipping template classifier training"
fi

if [[ "$RUN_NAV" == "1" ]]; then
  echo "[3/4] Train navigator with descriptive move supervision"
  python scripts/train_navigator.py \
    --config "$CONFIG" \
    --run-id "$NAV_RUN_ID" \
    --device "$DEVICE"
else
  echo "[3/4] Skipping navigator training"
fi

NAV_CKPT="$(ls -1t models/${NAV_RUN_ID}_step*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "$NAV_CKPT" ]]; then
  echo "[4/4] No navigator checkpoint found; benchmark skipped"
  exit 0
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  echo "[4/4] Benchmark unified rewrite executor"
  PYTHONUNBUFFERED=1 python -u -m scripts.run_benchmark \
    --config "$CONFIG" \
    --checkpoint "$NAV_CKPT" \
    --device "$DEVICE" \
    --theorems "$BENCHMARK_THEOREMS" \
    --output "$BENCHMARK_OUTPUT" \
    $BENCHMARK_EXTRA_FLAGS
else
  echo "[4/4] Benchmark skipped"
  echo "Latest navigator checkpoint: $NAV_CKPT"
  echo "Run benchmark with:"
  echo "  python -m scripts.run_benchmark --config $CONFIG --checkpoint $NAV_CKPT --device $DEVICE --theorems $BENCHMARK_THEOREMS --output $BENCHMARK_OUTPUT $BENCHMARK_EXTRA_FLAGS"
fi
