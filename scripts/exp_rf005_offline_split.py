#!/usr/bin/env python3
"""EXP-RF-005a — the censor gate, evaluated OUT-OF-SAMPLE by corpus split.

The offline half of THE_REFOUNDING §9's headline experiment. Evaluating
fences on the corpus they were mined from would be self-confirmation (the
spine IS those failures), so this run splits the gate-off trace corpus by
deterministic theorem-hash parity, mines a store on half A with the standard
pipeline, and evaluates on unseen half B:

  saved_attempts    failed (goal, lane) events in B whose family the A-store
                    fences — Lean calls the gate would not have spent.
  fenced_successes  progressing steps in B whose succeeding family the
                    A-store fences — THE COST CELL. The theory's claim is
                    that admissible fences generalize: this must be ~0. A
                    non-trivial count falsifies the mined store's slice
                    keying, and the honest response is a keying fix or floor
                    raise recorded here — never deleting the cell.

Both cells are reported per-family with the store's floors, deterministic
end to end (hash split, no RNG). Report → runs/exp_rf005a_report.json.

Usage: python -m scripts.exp_rf005_offline_split
    [--corpus runs/corpus/mathlib_test_2000.jsonl]
    [--min-attempts 25] [--min-theorems 8] [--min-kappa 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.censor_store import (
    ADMISSIBLE,
    FAMILY_OF_LANE,
    censor_key,
    fenced_families,
    plurality_verdict,
    rank_by_kappa,
    slice_signature,
    spine_verdict,
)
from src.structural_reads import read_goal

REPORT_PATH = Path("runs/exp_rf005a_report.json")


def in_half_a(theorem_id: str) -> bool:
    """Deterministic split: sha256 parity of the theorem id."""
    return hashlib.sha256(theorem_id.encode()).digest()[0] % 2 == 0


def mine_half_a(rows: list[dict], args) -> dict[str, set[str]]:
    """The standard mining pipeline (mirrors scripts/mine_censors.py), on half A only."""
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

    for row in rows:
        theorem = row.get("theorem_id", "")
        for st in row.get("step_trace") or []:
            goal = st.get("goal_before") or st.get("attempted_goal") or ""
            if not goal:
                continue
            key = key_of(goal)
            if st.get("progress"):
                family = FAMILY_OF_LANE.get(st.get("lane", ""), st.get("lane", ""))
                if family:
                    successes[(key, family)] += 1
                    families_attempted[key].add(family)
                continue
            for lane in st.get("lanes_tried") or []:
                family = FAMILY_OF_LANE.get(lane)
                if family is None:
                    continue
                pair = (key, family)
                failure_events[pair].add((theorem, st.get("step"), family))
                attempt_theorems[pair].add(theorem)
                families_attempted[key].add(family)

    spine_ok = {
        pair: frozenset(events)
        for pair, events in failure_events.items()
        if spine_verdict(
            len(events), len(attempt_theorems[pair]), successes.get(pair, 0),
            args.min_attempts, args.min_theorems,
        )
        == ADMISSIBLE
    }
    fences: dict[str, set[str]] = defaultdict(set)
    for (key, family), _kappa in rank_by_kappa(spine_ok, args.min_kappa):
        after = len(fences[key] | {family})
        if plurality_verdict(len(families_attempted[key]), after) == ADMISSIBLE:
            fences[key].add(family)
    return dict(fences)


def evaluate_half_b(rows: list[dict], fences: dict[str, set[str]]) -> dict:
    """Out-of-sample: what the A-mined gate would have saved and cost on B."""
    saved = Counter()
    fenced_success = Counter()
    total_failed_attempts = 0
    total_successes = 0
    key_cache: dict[str, str] = {}

    for row in rows:
        for st in row.get("step_trace") or []:
            goal = st.get("goal_before") or st.get("attempted_goal") or ""
            if not goal:
                continue
            if goal not in key_cache:
                reads = read_goal(goal)
                key_cache[goal] = censor_key(reads.goal_bucket, slice_signature(reads))
            key = key_cache[goal]
            bucket, _, signs = key.partition("|")
            fenced = fenced_families(bucket, tuple(int(s) for s in signs.split(",")), fences)
            if st.get("progress"):
                total_successes += 1
                family = FAMILY_OF_LANE.get(st.get("lane", ""), st.get("lane", ""))
                if family in fenced:
                    fenced_success[family] += 1
                continue
            for lane in st.get("lanes_tried") or []:
                family = FAMILY_OF_LANE.get(lane)
                if family is None:
                    continue
                total_failed_attempts += 1
                if family in fenced:
                    saved[family] += 1

    return {
        "total_failed_lane_attempts": total_failed_attempts,
        "total_successes": total_successes,
        "saved_attempts": sum(saved.values()),
        "saved_by_family": dict(saved),
        "saved_rate": round(sum(saved.values()) / max(total_failed_attempts, 1), 4),
        "fenced_successes": sum(fenced_success.values()),
        "fenced_successes_by_family": dict(fenced_success),
        "fenced_success_rate": round(sum(fenced_success.values()) / max(total_successes, 1), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("runs/corpus/mathlib_test_2000.jsonl"))
    parser.add_argument("--min-attempts", type=int, default=25)
    parser.add_argument("--min-theorems", type=int, default=8)
    parser.add_argument("--min-kappa", type=int, default=10)
    args = parser.parse_args()

    half_a: list[dict] = []
    half_b: list[dict] = []
    with args.corpus.open() as handle:
        for line in handle:
            row = json.loads(line)
            (half_a if in_half_a(row.get("theorem_id", "")) else half_b).append(row)

    fences = mine_half_a(half_a, args)
    evaluation = evaluate_half_b(half_b, fences)

    report = {
        "experiment": "EXP-RF-005a",
        "split": {"half_a_theorems": len(half_a), "half_b_theorems": len(half_b)},
        "mining_parameters": {
            "min_attempts": args.min_attempts,
            "min_theorems": args.min_theorems,
            "min_kappa": args.min_kappa,
        },
        "store": {
            "keys": len(fences),
            "fences": sum(len(v) for v in fences.values()),
        },
        "out_of_sample": evaluation,
        "reading": (
            "saved_attempts is the budget the gate pays out-of-sample; "
            "fenced_successes is the cost cell and the theory predicts ~0 — "
            "a nonzero count is a keying/floor finding, not a deletion target."
        ),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    ev = evaluation
    print(
        f"EXP-RF-005a · A={len(half_a)} B={len(half_b)} theorems · "
        f"{report['store']['fences']} fences on {report['store']['keys']} keys"
    )
    print(
        f"  saved (out-of-sample): {ev['saved_attempts']} of "
        f"{ev['total_failed_lane_attempts']} failed lane-attempts "
        f"({ev['saved_rate']:.1%}) · by family {ev['saved_by_family']}"
    )
    print(
        f"  COST CELL — fenced successes: {ev['fenced_successes']} of "
        f"{ev['total_successes']} successes ({ev['fenced_success_rate']:.2%})"
        f" · by family {ev['fenced_successes_by_family']}"
    )
    print(f"report → {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
