#!/bin/zsh
set -euo pipefail
SCRIPT_DIR="${0:a:h}"
exec "$SCRIPT_DIR/run_exp_dd014_guarded.sh" "${1:-/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2}" integrated "${2:-}"
