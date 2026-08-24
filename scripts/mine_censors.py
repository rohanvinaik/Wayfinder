#!/usr/bin/env python3
"""Mine admissible censors from gate-off search traces (THE_REFOUNDING §3, Wave 2).

The spine: every FAILED step in the trace corpus records the goal surface and
the lanes that attempted it — each is an observed near-miss for that lane's
family on that goal's censor-slice signature. A (slice, family) pair becomes
a candidate when its near-miss mass meets the floors and the family NEVER
succeeded on that slice anywhere in the corpus (one success kills the
candidate: the fence would be false, not weak). Candidates are adopted in
greedy max-κ order to the κ-knee, plurality-guarded per slice.

The corpus MUST be gate-off runs (no censor was consulted when the traces
were made) — that is what makes the spine structurally incapable of
self-confirmation. Re-mining after censors deploy must use fresh gate-off
runs, never gated ones.

Floors are explicit v1 mining parameters (CLI-overridable), reported in the
store artifact — they are operating points, not theory constants.

Usage:
  python -m scripts.mine_censors [--corpus runs/corpus/mathlib_test_2000.jsonl]
      [--min-attempts 25] [--min-theorems 8] [--min-kappa 10]
      [--out runs/censor_store_v1.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from src.censor_store import (
    ADMISSIBLE,
    FAMILY_OF_LANE,
    Censor,
    censor_key,
    plurality_verdict,
    rank_by_kappa,
    slice_signature,
    spine_verdict,
)
from src.structural_reads import read_goal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("runs/corpus/mathlib_test_2000.jsonl"))
    parser.add_argument("--min-attempts", type=int, default=25)
    parser.add_argument("--min-theorems", type=int, default=8)
    parser.add_argument("--min-kappa", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/censor_store_v1.json"))
    args = parser.parse_args()

    # (key, family) → spine accumulation over the gate-off corpus.
    failure_events: dict[tuple[str, str], set] = defaultdict(set)
    attempt_theorems: dict[tuple[str, str], set] = defaultdict(set)
    successes: dict[tuple[str, str], int] = defaultdict(int)
    families_attempted: dict[str, set] = defaultdict(set)
    key_cache: dict[str, str] = {}

    def key_of(goal_text: str) -> str:
        if goal_text not in key_cache:
            reads = read_goal(goal_text)
            key_cache[goal_text] = censor_key(reads.goal_bucket, slice_signature(reads))
        return key_cache[goal_text]

    rows = steps = 0
    with args.corpus.open() as handle:
        for line in handle:
            row = json.loads(line)
            theorem = row.get("theorem_id", "")
            for st in row.get("step_trace") or []:
                goal = st.get("goal_before") or st.get("attempted_goal") or ""
                if not goal:
                    continue
                steps += 1
                key = key_of(goal)
                if st.get("progress"):
                    family = FAMILY_OF_LANE.get(st.get("lane", ""), st.get("lane", ""))
                    if family:
                        successes[(key, family)] += 1
                        families_attempted[key].add(family)
                    continue
                tried = st.get("lanes_tried") or []
                for lane in tried:
                    family = FAMILY_OF_LANE.get(lane)
                    if family is None:
                        continue
                    pair = (key, family)
                    failure_events[pair].add((theorem, st.get("step"), family))
                    attempt_theorems[pair].add(theorem)
                    families_attempted[key].add(family)
            rows += 1

    # Clause (i): spine floors + the zero-success guard.
    spine_ok: dict[tuple[str, str], frozenset] = {}
    rejected = {"insufficient_spine": 0, "success_observed": 0}
    for pair, events in failure_events.items():
        verdict = spine_verdict(
            len(events),
            len(attempt_theorems[pair]),
            successes.get(pair, 0),
            min_attempts=args.min_attempts,
            min_theorems=args.min_theorems,
        )
        if verdict == ADMISSIBLE:
            spine_ok[pair] = frozenset(events)
        else:
            rejected[verdict] += 1

    # Greedy max-κ adoption to the knee, with clause (ii) checked at adoption.
    fenced_per_key: dict[str, set] = defaultdict(set)
    adopted: list[Censor] = []
    plurality_blocked = 0
    for (key, family), kappa in rank_by_kappa(spine_ok, min_kappa=args.min_kappa):
        after = len(fenced_per_key[key] | {family})
        if plurality_verdict(len(families_attempted[key]), after) != ADMISSIBLE:
            plurality_blocked += 1
            continue
        fenced_per_key[key].add(family)
        adopted.append(
            Censor(
                key=key,
                family=family,
                attempts=len(failure_events[(key, family)]),
                theorems=len(attempt_theorems[(key, family)]),
                successes=0,
                kappa=kappa,
                source=str(args.corpus),
            )
        )

    store = {
        "version": 1,
        "mined_from": str(args.corpus),
        "mining_parameters": {
            "min_attempts": args.min_attempts,
            "min_theorems": args.min_theorems,
            "min_kappa": args.min_kappa,
        },
        "corpus": {"theorem_rows": rows, "steps_read": steps},
        "candidates": len(failure_events),
        "rejected": rejected,
        "plurality_blocked": plurality_blocked,
        "adopted": [asdict(c) for c in adopted],
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(store, indent=2))

    total_kappa = sum(c.kappa for c in adopted)
    print(
        f"censor mining · {rows} theorems · {steps} steps · "
        f"{len(failure_events)} candidates → {len(adopted)} adopted "
        f"(spine-rejected {rejected['insufficient_spine']}, "
        f"false-fence-rejected {rejected['success_observed']}, "
        f"plurality-blocked {plurality_blocked})"
    )
    print(f"  fenced failure-event mass (Σκ): {total_kappa}")
    for c in adopted[:10]:
        print(f"  κ={c.kappa:4d} · {c.family:10s} on {c.key}  ({c.theorems} theorems)")
    print(f"store → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
