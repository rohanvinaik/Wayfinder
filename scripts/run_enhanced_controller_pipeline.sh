#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/6] Canonical local data"
if [[ "${FORCE_CANONICAL:-0}" == "1" || ! -f data/canonical/canonical_residual_train.jsonl ]]; then
  python -m scripts.build_canonical_local_data
else
  echo "  using existing canonical data"
fi

echo "[2/6] Subtask training data"
python -m scripts.build_subtask_training_data \
  --input data/canonical/canonical_residual_train.jsonl \
  --output data/canonical/subtask_train.jsonl

echo "[3/6] Subtask validation"
python -m scripts.validate_subtask_ir \
  --input data/canonical/canonical_residual_eval.jsonl

echo "[4/6] Navigational training data with canonical move metadata"
python scripts/build_nav_training_data.py \
  --input data/proof_network_entities.jsonl \
  --output data/nav_training.jsonl \
  --canonical data/canonical/canonical_residual_train.jsonl \
  --canonical data/canonical/canonical_residual_eval.jsonl

echo "[5/6] Template extraction with theorem-level move profiles"
python -m scripts.extract_templates \
  --data data/nav_training.jsonl \
  --output-dir data

echo "[6/6] Compact move inventory"
python -m scripts.build_move_inventory \
  --input data/canonical/subtask_train.jsonl \
  --min-support 100 \
  --output data/move_inventory.json

echo "Enhanced controller data pipeline complete."
