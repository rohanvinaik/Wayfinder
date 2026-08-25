#!/bin/bash
# Loop pass 2 — the re-defined universe, paired (THE_REFOUNDING §10.7).
#
# Corpus: the 771 pass-1 model-failures (progressed + blocked_unproven;
# infra excluded — a separate repair track). Both arms run current HEAD,
# which pass 1 did NOT carry: the slate-diversity fix (EXP-RF-008's
# skeletons finally execute) and the mined Ducky-regime read. The pair
# isolates the re-mined negative channel:
#
#   1. control — deterministic, traced, gate OFF
#   2. treated — + censor_store_v2 (92 fences, current lane population)
#
# Metrics: new closes on the residual (every one is a net cumulative
# solve), solve delta treated-vs-control (must be >= 0), attempts saved
# (the re-mined store's budget refund), abstention ledger. Traces stay
# on — pass 2's residual is pass 3's mining corpus.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CKPT=models/NAV-004_step5000.pt
MANIFEST=runs/loop_pass2_residual_manifest.jsonl

run_arm () {
  local name="$1" config="$2"
  echo "=== ARM $name — $(date '+%H:%M:%S') ==="
  caffeinate -i $PY -m scripts.run_benchmark \
    --config "$config" --checkpoint "$CKPT" --mode v1 \
    --theorems "$MANIFEST" --limit 771 \
    --output "runs/loop_pass2_${name}.json" \
    > "runs/loop_pass2_${name}.log" 2>&1
  echo "    exit $? — report runs/loop_pass2_${name}.json"
}

run_arm control configs/wayfinder_deterministic_traced.yaml
run_arm treated configs/wayfinder_pass2_treated.yaml

$PY - <<'EOF'
import json
from pathlib import Path

def load(name):
    p = Path(f"runs/loop_pass2_{name}.json")
    return json.loads(p.read_text()) if p.exists() else None

print("\n=== LOOP PASS 2 PAIRED SUMMARY (residual n=771) ===")
ctrl, treat = load("control"), load("treated")
if ctrl and treat:
    cb, tb = ctrl["benchmark"], treat["benchmark"]
    ce, te = ctrl["efficiency"], treat["efficiency"]
    drops = sum(d.get("censor_drop_count", 0) for d in treat.get("details", []))
    print(f"  new closes   control {cb['raw_success']}  vs  treated {tb['raw_success']}  (delta {tb['raw_success']-cb['raw_success']:+d})")
    print(f"  honest       control {cb['honest_success']}  vs  treated {tb['honest_success']}")
    print(f"  attempts/thm control {ce['avg_attempts_per_theorem']}  vs  treated {te['avg_attempts_per_theorem']}")
    print(f"  time/thm     control {ce['avg_time_per_theorem_s']}s  vs  treated {te['avg_time_per_theorem_s']}s")
    print(f"  lane removals under warrant (treated): {drops}")
    print(f"  cumulative vs pass 1: 1028 + {tb['raw_success']} best-arm = {1028 + max(cb['raw_success'], tb['raw_success'])}/2000")
else:
    print("  incomplete (missing report)")
EOF
echo "=== DONE — $(date '+%H:%M:%S') ==="
