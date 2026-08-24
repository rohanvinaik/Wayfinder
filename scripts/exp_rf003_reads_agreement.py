#!/usr/bin/env python3
"""EXP-RF-003 — computed structural reads vs the frozen navigation labels.

Wave-1 gate of THE_REFOUNDING (§9): measure, on the frozen eval set
(data/nav_eval.jsonl, never modified), how the deterministic bank reads relate
to the supervision labels the navigator was trained toward, and run the
Peitho discrimination audit (§4.1b law 3) over the read signatures.

Honest framing — this is NOT a single accuracy number. The labels are
step-supervision: several banks were labeled from the TACTIC the proof
applied (proof-side knowledge an open goal does not have), so for those
banks the read's job is *non-contradiction* (abstain or agree), not
prediction. Per-bank rows report:

  agree          read sign == label sign
  abstain        read 0 vs label ±1 — the honest no-opinion, not an error
  contradict     read ±1 vs label ∓1 — the only genuinely bad cell
  label_zero     label 0 (no supervision opinion) — uninformative row

The discrimination audit groups goals by read signature and reports
signatures whose goals close under many distinct tactic families — a
collapsed signature is a MISSING DIMENSION (add an orthogonal read), never
a threshold to tune.

Deterministic: stride sampling, no RNG. Report → runs/exp_rf003_report.json.

Usage: python -m scripts.exp_rf003_reads_agreement [--limit 4000] [--stride 2]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.structural_reads import BANKS, read_goal

EVAL_PATH = Path("data/nav_eval.jsonl")
REPORT_PATH = Path("runs/exp_rf003_report.json")


def tactic_family(tactic: str) -> str:
    """First token of the ground-truth tactic — the coarse family key."""
    head = (tactic or "").strip().split()
    return head[0] if head else "<empty>"


def compare(read_sign: int, label_sign: int) -> str:
    if label_sign == 0:
        return "label_zero"
    if read_sign == label_sign:
        return "agree"
    if read_sign == 0:
        return "abstain"
    return "contradict"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    args = parser.parse_args()

    per_bank: dict[str, Counter] = {bank: Counter() for bank in BANKS}
    signature_tactics: dict[tuple, Counter] = defaultdict(Counter)
    signature_counts: Counter = Counter()
    rows = 0

    with args.eval_path.open() as handle:
        for idx, line in enumerate(handle):
            if idx % args.stride:
                continue
            if rows >= args.limit:
                break
            example = json.loads(line)
            goal_state = example.get("goal_state", "")
            labels = example.get("nav_directions", {}) or {}
            if not goal_state or not labels:
                continue
            reads = read_goal(goal_state)
            directions = reads.directions()
            for bank in BANKS:
                if bank in labels:
                    per_bank[bank][compare(directions[bank], int(labels[bank]))] += 1
            sig = reads.signature()
            signature_counts[sig] += 1
            signature_tactics[sig][tactic_family(example.get("ground_truth_tactic", ""))] += 1
            rows += 1

    bank_rows = {}
    for bank in BANKS:
        counts = per_bank[bank]
        informative = counts["agree"] + counts["abstain"] + counts["contradict"]
        bank_rows[bank] = {
            **dict(counts),
            "informative": informative,
            "agree_rate": round(counts["agree"] / informative, 4) if informative else None,
            "contradict_rate": round(counts["contradict"] / informative, 4) if informative else None,
        }

    # Discrimination audit: signatures carrying many goals across many families.
    collapsed = []
    for sig, families in signature_tactics.items():
        n = signature_counts[sig]
        if n < 20:
            continue
        top_family, top_n = families.most_common(1)[0]
        purity = top_n / n
        if len(families) >= 5 and purity < 0.5:
            collapsed.append(
                {
                    "signature": list(sig),
                    "goals": n,
                    "families": len(families),
                    "top_family": top_family,
                    "top_family_share": round(purity, 3),
                    "family_sample": dict(families.most_common(6)),
                }
            )
    collapsed.sort(key=lambda row: -row["goals"])

    report = {
        "experiment": "EXP-RF-003",
        "rows": rows,
        "distinct_signatures": len(signature_counts),
        "per_bank": bank_rows,
        "collapsed_signatures": collapsed[:12],
        "collapsed_count": len(collapsed),
        "note": (
            "Labels are step-supervision (tactic-derived for several banks); "
            "abstain is honest no-opinion, contradict is the failure cell. "
            "A collapsed signature is a missing dimension, never a threshold."
        ),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"EXP-RF-003 · {rows} goals · {len(signature_counts)} distinct signatures")
    for bank, row in bank_rows.items():
        rate = row["agree_rate"]
        contra = row["contradict_rate"]
        print(
            f"  {bank:14s} agree={rate if rate is not None else '—':<7} "
            f"contradict={contra if contra is not None else '—':<7} "
            f"(informative n={row['informative']})"
        )
    print(f"  collapsed signatures (≥20 goals, ≥5 families, purity<0.5): {len(collapsed)}")
    for row in collapsed[:4]:
        print(
            f"    {tuple(row['signature'])} — {row['goals']} goals, "
            f"{row['families']} families, top {row['top_family']} {row['top_family_share']}"
        )
    print(f"report → {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
