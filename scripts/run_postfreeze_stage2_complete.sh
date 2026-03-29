#!/bin/zsh

set -euo pipefail

ROOT="/Users/rohanvinaik/Projects/Wayfinder"
RUN_DIR="${1:-$ROOT/runs/exp_som012_hard_eval_r2}"
RUN_NAME="${RUN_DIR:t}"
RUNS_ROOT="${RUN_DIR:h}"
SMOKE_ROOT="${2:-$RUNS_ROOT}"
AUDIT_JSON="${3:-$RUNS_ROOT/postfreeze_stage2_complete_from_${RUN_NAME}/smoke_audit.json}"
ORCH_ROOT="${4:-$RUNS_ROOT/postfreeze_stage2_complete_from_${RUN_NAME}}"

mkdir -p "$ORCH_ROOT"

python -m scripts.audit_postfreeze_stage2_smoke \
  --run-dir "$RUN_DIR" \
  --smoke-root "$SMOKE_ROOT" \
  --output-json "$AUDIT_JSON" > "$ORCH_ROOT/audit_stdout.log"

eval "$(python - <<'PY' "$AUDIT_JSON"
import json, shlex, sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text())
env = obj["recommended_full_env"]
runs = obj["recommended_complete_runs"]
for key, value in env.items():
    print(f"export {key}={shlex.quote(value)}")
for key, value in runs.items():
    print(f"export {key.upper()}_OUT={shlex.quote(value)}")
PY
)"

exec "$ROOT/scripts/run_postfreeze_stage2_remaining.sh" "$RUN_DIR" "$ORCH_ROOT"
