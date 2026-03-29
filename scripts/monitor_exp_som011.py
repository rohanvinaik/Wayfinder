from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

OUT = Path("runs/exp_som011_paired_2000/MONITOR.txt")
BASELINE = Path("runs/exp058_decisive_2000/summary.json")
PIDS = "40194,40195,40534,40535"


def _process_block() -> str:
    proc = subprocess.run(
        ["ps", "-p", PIDS, "-o", "pid=,etime=,pcpu=,pmem=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.rstrip() or "(none)"


def _baseline_block() -> str:
    if not BASELINE.exists():
        return "(baseline summary missing)"
    data = json.loads(BASELINE.read_text())
    entry = data.get("cosine_rw_only", {})
    return (
        f"EXP-058 baseline: {entry.get('proved')}/{entry.get('total')} "
        f"success_rate={entry.get('success_rate')}"
    )


def _write_once() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(str(path) for path in OUT.parent.glob("*") if path.is_file())
    lines = [
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Processes",
        _process_block(),
        "",
        "Output files",
        *(files or ["(none yet)"]),
        "",
        "Baseline",
        _baseline_block(),
    ]
    OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        _write_once()
        return

    while True:
        _write_once()
        time.sleep(60)


if __name__ == "__main__":
    main()
