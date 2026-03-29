#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/9] Canonical local data"
if [[ "${FORCE_CANONICAL:-0}" == "1" || ! -f data/canonical/canonical_residual_train.jsonl ]]; then
  python -m scripts.build_canonical_local_data
else
  echo "  using existing canonical data"
fi

echo "[2/9] Subtask training data"
python -m scripts.build_subtask_training_data \
  --input data/canonical/canonical_residual_train.jsonl \
  --output data/canonical/subtask_train.jsonl

echo "[3/9] Subtask validation"
python -m scripts.validate_subtask_ir \
  --input data/canonical/canonical_residual_eval.jsonl

echo "[4/9] Navigational training data with canonical move metadata"
python scripts/build_nav_training_data.py \
  --input data/proof_network_entities.jsonl \
  --output data/nav_training.jsonl \
  --canonical data/canonical/canonical_residual_train.jsonl \
  --canonical data/canonical/canonical_residual_eval.jsonl

echo "[5/9] Template extraction with theorem-level move profiles"
python -m scripts.extract_templates \
  --data data/nav_training.jsonl \
  --output-dir data

echo "[6/9] Compact move inventory"
python -m scripts.build_move_inventory \
  --input data/canonical/subtask_train.jsonl \
  --min-support 100 \
  --output data/move_inventory.json

echo "[7/9] Template narrative dataset"
python -m scripts.build_template_narrative_dataset \
  --input data/nav_train_templates.jsonl \
  --output data/template_narrative_train.jsonl

echo "[8/9] Temporal training dataset"
TRACE_INPUTS=(runs/*/benchmark_results.jsonl runs/*/*.jsonl)
EXISTING_TRACES=()
for f in "${TRACE_INPUTS[@]}"; do
  [[ -f "$f" ]] && EXISTING_TRACES+=("$f")
done
if [[ ${#EXISTING_TRACES[@]} -gt 0 ]]; then
  python -m scripts.build_temporal_dataset \
    --inputs "${EXISTING_TRACES[@]}" \
    --output data/temporal_train.jsonl
else
  echo "  no trace files found — skipping temporal dataset"
fi

echo "[9/9] Strategy memory"
if [[ ${#EXISTING_TRACES[@]} -gt 0 ]]; then
  python -m scripts.mine_strategy_memory \
    --inputs "${EXISTING_TRACES[@]}" \
    --templates data/nav_train_templates.jsonl \
    --output data/strategy_memory.json \
    --min-support 3
else
  echo "  no trace files found — skipping strategy memory"
fi

echo "Enhanced controller data pipeline complete."
