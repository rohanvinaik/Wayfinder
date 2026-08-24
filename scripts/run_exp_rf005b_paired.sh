#!/bin/bash
# EXP-RF-005b — the live paired censor-gate experiment (THE_REFOUNDING §9).
#
# Four arms, sequential (Pantograph is a per-process singleton), each writing
# its own report so partial completion still yields data:
#
#   1. MiniF2F  control  — deterministic path, gate off        (limit 100)
#   2. MiniF2F  treated  — + full-corpus store (CROSS-CORPUS:  (limit 100)
#                          the store was mined on Mathlib traces, so MiniF2F
#                          is fully out-of-sample by construction)
#   3. Mathlib½B control — deterministic path on the HALF-B    (limit 150)
#                          manifest, gate off
#   4. Mathlib½B treated — + the HALF-A store (the 005a split  (limit 150)
#                          discipline, live: fences never meet
#                          their mining theorems)
#
# Metrics per pair: raw/honest solve delta (must be >= 0), attempts saved,
# censor_drop ledger totals. Run under caffeinate so system sleep cannot
# kill the arms. Logs + reports land in runs/exp_rf005b_*.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CKPT=models/NAV-004_step5000.pt   # required by the CLI, never loaded in deterministic mode

run_arm () {
  local name="$1" config="$2" extra="$3"
  echo "=== ARM $name — $(date '+%H:%M:%S') ==="
  # shellcheck disable=SC2086
  caffeinate -i $PY -m scripts.run_benchmark \
    --config "$config" --checkpoint "$CKPT" --mode v1 \
    --output "runs/exp_rf005b_${name}.json" $extra \
    > "runs/exp_rf005b_${name}.log" 2>&1
  echo "    exit $? — report runs/exp_rf005b_${name}.json"
}

run_arm minif2f_control configs/wayfinder_deterministic.yaml               "--limit 100"
run_arm minif2f_treated configs/wayfinder_deterministic_censored.yaml      "--limit 100"
run_arm mathlibB_control configs/wayfinder_deterministic.yaml              "--limit 150 --theorems runs/exp_rf005b_halfB_manifest.jsonl"
run_arm mathlibB_treated configs/wayfinder_deterministic_censored_halfA.yaml "--limit 150 --theorems runs/exp_rf005b_halfB_manifest.jsonl"

# The paired summary — consumed signals only, straight from the reports.
$PY - <<'EOF'
import json
from pathlib import Path

def load(name):
    p = Path(f"runs/exp_rf005b_{name}.json")
    return json.loads(p.read_text()) if p.exists() else None

print("\n=== EXP-RF-005b PAIRED SUMMARY ===")
for corpus, ctrl_name, treat_name in (
    ("MiniF2F (cross-corpus store)", "minif2f_control", "minif2f_treated"),
    ("Mathlib half-B (half-A store)", "mathlibB_control", "mathlibB_treated"),
):
    ctrl, treat = load(ctrl_name), load(treat_name)
    if not (ctrl and treat):
        print(f"{corpus}: incomplete (missing report)")
        continue
    cb, tb = ctrl["benchmark"], treat["benchmark"]
    ce, te = ctrl["efficiency"], treat["efficiency"]
    drops = sum(d.get("censor_drop_count", 0) for d in treat.get("details", []))
    print(f"{corpus}:")
    print(f"  raw    control {cb['raw_success']}/{cb['total_theorems']}  vs  treated {tb['raw_success']}/{tb['total_theorems']}  (delta {tb['raw_success']-cb['raw_success']:+d})")
    print(f"  honest control {cb['honest_success']}  vs  treated {tb['honest_success']}  (delta {tb['honest_success']-cb['honest_success']:+d})")
    print(f"  attempts/theorem  control {ce['avg_attempts_per_theorem']}  vs  treated {te['avg_attempts_per_theorem']}")
    print(f"  time/theorem      control {ce['avg_time_per_theorem_s']}s  vs  treated {te['avg_time_per_theorem_s']}s")
    print(f"  abstention ledger (treated): {drops} lane removals under warrant")
EOF
echo "=== DONE — $(date '+%H:%M:%S') ==="
