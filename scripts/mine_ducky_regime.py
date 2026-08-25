"""Mine the Dr.-Ducky-regime read from the population of her observed closes.

THE_REFOUNDING §9 (round 5) + §10.7: four solves lost to bulk-rewrite
preemption were goals `goal_tail_regime` reads as '' — a discrimination-
guarantee gap (Ducky-closable goals sharing the not-tail read). The fix
is a read MINED from the population of historical dr_ducky closes, never
markers tuned to the four names.

Consumes traced detail JSONLs (collect_trace: true runs). For every step
where the dr_ducky lane made progress, reads the goal's full signature;
compares against the all-steps base rate; reports the signature cells
where her closes concentrate, with coverage/purity per cell. The output
is DATA for authoring the read (mirroring the mined-norms discipline) —
this script proposes; the read is written and pinned by hand.

Usage:
    python -m scripts.mine_ducky_regime runs/loop_pass1b_traced.jsonl \
        [more.jsonl ...] [--min-closes 3] [--min-ratio 3.0]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.structural_reads import read_goal


def regime_cells(
    ducky_counts: dict[tuple, int],
    population_counts: dict[tuple, int],
    total_ducky: int,
    total_population: int,
    min_closes: int = 3,
    min_ratio: float = 3.0,
) -> list[dict]:
    """The pure decision: cells where Ducky closes concentrate.

    A cell qualifies when it holds at least ``min_closes`` observed closes
    AND its close-rate lift over the population base rate is at least
    ``min_ratio``. Returns cells sorted by close count, each with its
    evidence (closes, population share, lift) — never a bare verdict.
    """
    if total_ducky <= 0 or total_population <= 0:
        return []
    base_rate = total_ducky / total_population
    out: list[dict] = []
    for cell, closes in ducky_counts.items():
        pop = population_counts.get(cell, 0)
        if closes < min_closes or pop <= 0:
            continue
        lift = (closes / pop) / base_rate
        if lift < min_ratio:
            continue
        bucket, sig = cell
        out.append(
            {
                "bucket": bucket,
                "signature": list(sig),
                "ducky_closes": closes,
                "population_steps": pop,
                "cell_close_rate": round(closes / pop, 4),
                "base_rate": round(base_rate, 4),
                "lift": round(lift, 2),
            }
        )
    out.sort(key=lambda c: (-c["ducky_closes"], -c["lift"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpora", nargs="+", type=Path)
    parser.add_argument("--min-closes", type=int, default=3)
    parser.add_argument("--min-ratio", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("runs/ducky_regime_cells.json"))
    args = parser.parse_args()

    ducky: Counter = Counter()
    population: Counter = Counter()
    ducky_goals: list[str] = []
    sig_cache: dict[str, tuple] = {}

    def cell_of(goal_text: str) -> tuple:
        if goal_text not in sig_cache:
            reads = read_goal(goal_text)
            sig_cache[goal_text] = (reads.goal_bucket, reads.signature())
        return sig_cache[goal_text]

    steps = rows = 0
    for corpus in args.corpora:
        with corpus.open() as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                for st in row.get("step_trace") or []:
                    goal = st.get("goal_before") or st.get("attempted_goal") or ""
                    if not goal:
                        continue
                    steps += 1
                    cell = cell_of(goal)
                    population[cell] += 1
                    if st.get("lane") == "dr_ducky" and st.get("progress"):
                        ducky[cell] += 1
                        ducky_goals.append(goal[:200])

    cells = regime_cells(
        dict(ducky),
        dict(population),
        sum(ducky.values()),
        sum(population.values()),
        min_closes=args.min_closes,
        min_ratio=args.min_ratio,
    )
    covered = sum(c["ducky_closes"] for c in cells)
    report = {
        "corpora": [str(p) for p in args.corpora],
        "rows": rows,
        "steps": steps,
        "ducky_progress_steps": sum(ducky.values()),
        "distinct_ducky_cells": len(ducky),
        "qualifying_cells": cells,
        "coverage_of_ducky_closes": round(covered / max(sum(ducky.values()), 1), 3),
        "sample_ducky_goals": ducky_goals[:20],
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"{rows} rows, {steps} steps → {sum(ducky.values())} dr_ducky progress steps "
        f"in {len(ducky)} cells; {len(cells)} qualify "
        f"(coverage {report['coverage_of_ducky_closes']:.0%}) → {args.out}"
    )


if __name__ == "__main__":
    main()
