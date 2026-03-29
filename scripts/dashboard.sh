#!/bin/zsh
# Wayfinder experiment dashboard
# Usage: zsh scripts/dashboard.sh
#   or:  watch -n60 zsh scripts/dashboard.sh

ROOT="/Users/rohanvinaik/Projects/Wayfinder/runs"

_cond() {
  local dir="$1" label="$2" target="$3"
  if [[ ! -f "$dir/summary.json" ]]; then
    printf "  %-12s  waiting...\n" "$label"
    return
  fi
  python3 -c "
import json
d = json.loads(open('$dir/summary.json').read())
b = d.get('benchmark', {})
total = b.get('total_theorems', 0)
target = int('$target') if '$target' else 2000
if total == 0: exit()
proved = b.get('raw_success', 0)
started = b.get('started_theorems', 0)
skip = b.get('skipped_start', 0)
br = d.get('bridge', {})
br_c = br.get('closed', 0)
br_i = br.get('invoked', 0)
eff = d.get('efficiency', {})
elapsed_h = eff.get('elapsed_s', 0) / 3600
avg_t = eff.get('avg_time_per_theorem_s', 0)
remaining = target - total
eta_h = (remaining * avg_t / 3600) if avg_t > 0 else 0

bar_width = 20
filled = int(total / target * bar_width)
bar = '█' * filled + '░' * (bar_width - filled)

started_rate = proved / started * 100 if started else 0
bridge_str = f'{br_c}/{br_i} ({br_c/br_i*100:.0f}%)' if br_i else '-'

print(f'  {\"$label\":12s} [{bar}] {total:>4d}/{target}')
print(f'               proved={proved} ({proved/total*100:.1f}%)  proved|started={started_rate:.1f}%  skip={skip}')
print(f'               bridge {bridge_str}  elapsed={elapsed_h:.1f}h  ETA ~{eta_h:.0f}h')
" 2>/dev/null
}

_bridge() {
  python3 -c "
import json
from collections import Counter
closed = 0; total = 0; prog = 0; close_by = Counter()
for path in ['$ROOT/paired_condC_first/hardtail_bridge_rows.jsonl',
             '$ROOT/paired_condC_second/hardtail_bridge_rows.jsonl']:
    try:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                total += 1
                if row.get('closed'):
                    closed += 1
                    close_by[row.get('closed_by','?')[:25]] += 1
                elif any(isinstance(s, dict) and s.get('progressed') for s in (row.get('stage_trace') or [])):
                    prog += 1
    except: pass
if total == 0:
    print('  Bridge: no data yet')
    exit()
open_c = total - closed
print(f'  Bridge')
print(f'    closed:   {closed}/{total} ({closed/total*100:.1f}%)')
print(f'    progress: {prog}/{open_c} open show progress ({prog/max(open_c,1)*100:.0f}%)')
if close_by:
    top = close_by.most_common(3)
    parts = [f'{stage} ({c})' for stage, c in top]
    print(f'    by:       {\"  \".join(parts)}')
" 2>/dev/null
}

_latest() {
  for dir in "$ROOT/paired_condC_first" "$ROOT/paired_condC_second"; do
    if [[ -f "$dir/run.log" ]]; then
      local label=$(basename "$dir" | sed 's/paired_condC_//')
      local last=$(tail -1 "$dir/run.log" 2>/dev/null | grep -o '\[.*\].*' | head -1)
      if [[ -n "$last" ]]; then
        printf "  %-6s %s\n" "$label" "$last"
      fi
    fi
  done
}

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║  Wayfinder — Condition C (fresh run, all fixes)      ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""
_cond "$ROOT/paired_condC_first"  "C (1st)" 1000
_cond "$ROOT/paired_condC_second" "C (2nd)" 1000
echo ""
_bridge
echo ""
echo "  ─────────────────────────────────────────────────────────"
echo "  Reference (Condition A — no Ducky, complete)"
echo "  A total: 2000 theorems, 1051 proved (52.6%), 230 skipped"
echo "  A proved|started: 1051/1770 = 59.4%"
echo "  ─────────────────────────────────────────────────────────"
echo ""
_latest
echo "  $(date '+%H:%M:%S')"
echo ""
