"""EXP-050/051 dataset profiler.

Summarizes the trigger dataset for sanity checking before training.

Reports:
  - Total rows by row_type (trigger_state vs goal_start_failure)
  - Positives by stage
  - Accepted candidates per pool
  - Theorem-group counts
  - Startability mix
  - Namespace distribution of positives

Usage:
    python -m scripts.evaluate_trigger_dataset \\
        --data data/apply_trigger_train_full.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("Empty dataset.")
        return

    # Split by row_type
    trigger_rows = [r for r in rows if r.get("row_type") == "trigger_state"]
    start_fail_rows = [r for r in rows if r.get("row_type") == "goal_start_failure"]
    other_rows = [
        r for r in rows if r.get("row_type") not in ("trigger_state", "goal_start_failure")
    ]

    print("=" * 65)
    print("Trigger Dataset Profile")
    print("=" * 65)
    print(f"  Total rows:            {len(rows)}")
    print(f"    trigger_state:       {len(trigger_rows)}")
    print(f"    goal_start_failure:  {len(start_fail_rows)}")
    if other_rows:
        print(f"    other/unknown:       {len(other_rows)}")

    # --- Trigger rows ---
    if trigger_rows:
        print(f"\n--- Trigger States ({len(trigger_rows)} rows) ---")

        # Label source breakdown
        source_counts = Counter(r.get("label_source", "?") for r in trigger_rows)
        print("\n  Label sources:")
        for src, cnt in source_counts.most_common():
            print(f"    {src:20s}: {cnt}")

        # can_apply breakdown
        lean_rows = [r for r in trigger_rows if r.get("label_source") == "lean_probe"]
        can_apply_pos = sum(1 for r in lean_rows if r.get("can_apply") == 1)
        can_apply_neg = sum(1 for r in lean_rows if r.get("can_apply") == 0)
        can_apply_none = sum(1 for r in lean_rows if r.get("can_apply") is None)
        print(f"\n  can_apply (lean_probe only, n={len(lean_rows)}):")
        print(
            f"    positive:  {can_apply_pos} ({100 * can_apply_pos / max(len(lean_rows), 1):.1f}%)"
        )
        print(f"    negative:  {can_apply_neg}")
        if can_apply_none:
            print(f"    null:      {can_apply_none}")

        # selector_top1_accepted diagnostic
        sel_rows = [r for r in lean_rows if r.get("selector_top1_accepted") is not None]
        sel_pos = sum(1 for r in sel_rows if r.get("selector_top1_accepted") == 1)
        if sel_rows:
            print(f"\n  selector_top1_accepted (diagnostic, n={len(sel_rows)}):")
            print(f"    positive:  {sel_pos} ({100 * sel_pos / max(len(sel_rows), 1):.1f}%)")

        # Per-stage breakdown
        stage_counts: dict[str, dict[str, int]] = {}
        for r in lean_rows:
            s = r.get("search_stage", "?")
            stage_counts.setdefault(s, {"total": 0, "pos": 0})
            stage_counts[s]["total"] += 1
            if r.get("can_apply") == 1:
                stage_counts[s]["pos"] += 1

        print("\n  Per-stage (lean_probe):")
        for stage, sc in sorted(stage_counts.items()):
            pct = 100 * sc["pos"] / max(sc["total"], 1)
            print(f"    {stage:20s}: {sc['pos']:4d}/{sc['total']:<4d} ({pct:.1f}%)")

        # Pool stats
        probed_counts = [r.get("num_candidates_probed", 0) for r in lean_rows]
        accepted_counts = [r.get("num_accepted_in_pool", 0) for r in lean_rows]
        considered_counts = [r.get("num_candidates_considered", 0) for r in lean_rows]
        if probed_counts:
            print("\n  Probe pool stats (lean_probe):")
            print(
                f"    candidates considered: mean={sum(considered_counts) / len(considered_counts):.1f}"
            )
            print(f"    candidates probed:     mean={sum(probed_counts) / len(probed_counts):.1f}")
            print(
                f"    accepted per pool:     mean={sum(accepted_counts) / len(accepted_counts):.2f}"
            )
            multi_accepted = sum(1 for a in accepted_counts if a > 1)
            if multi_accepted:
                print(f"    pools with >1 accepted: {multi_accepted}")

        # Theorem-group counts
        theorem_ids = set(r.get("theorem_id", "") for r in trigger_rows)
        pos_theorem_ids = set(r.get("theorem_id", "") for r in lean_rows if r.get("can_apply") == 1)
        rows_per_theorem = Counter(r.get("theorem_id", "") for r in trigger_rows)
        print("\n  Theorem coverage:")
        print(f"    unique theorems:      {len(theorem_ids)}")
        print(f"    theorems with pos:    {len(pos_theorem_ids)}")
        print(
            f"    rows/theorem: mean={sum(rows_per_theorem.values()) / max(len(rows_per_theorem), 1):.1f}, "
            f"max={max(rows_per_theorem.values()) if rows_per_theorem else 0}"
        )

        # Namespace distribution of positives
        ns_pos: Counter[str] = Counter()
        ns_total: Counter[str] = Counter()
        for r in lean_rows:
            ns = r.get("namespace_prefix", "?")
            ns_total[ns] += 1
            if r.get("can_apply") == 1:
                ns_pos[ns] += 1
        if ns_pos:
            print("\n  Top namespaces by positives:")
            for ns, cnt in ns_pos.most_common(10):
                print(f"    {ns:30s}: {cnt}/{ns_total[ns]} pos")

        # Feedback categories
        feedback_counts = Counter(
            r.get("best_feedback_category", "")
            for r in lean_rows
            if r.get("best_feedback_category")
        )
        if feedback_counts:
            print("\n  Feedback categories (lean_probe):")
            for cat, cnt in feedback_counts.most_common(10):
                print(f"    {cat:30s}: {cnt}")

    # --- Goal start failures ---
    if start_fail_rows:
        print(f"\n--- Goal Start Failures ({len(start_fail_rows)} rows) ---")
        status_counts = Counter(r.get("goal_start_status", "?") for r in start_fail_rows)
        for status, cnt in status_counts.most_common():
            print(f"    {status:30s}: {cnt}")

        fail_cats = Counter(r.get("failure_category", "?") for r in start_fail_rows)
        print("\n  Failure categories:")
        for cat, cnt in fail_cats.most_common(10):
            print(f"    {cat:30s}: {cnt}")

        repaired = sum(1 for r in start_fail_rows if r.get("repair_success"))
        print(
            f"\n  Repair attempted: {sum(1 for r in start_fail_rows if r.get('repair_attempted'))}"
        )
        print(f"  Repair succeeded: {repaired}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
