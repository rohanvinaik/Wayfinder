"""
Anchor gap analysis — iterative recall validation for the proof network.

Standalone worker (ModelAtlas pattern): reads proof_network.db, writes
gap analysis results to JSONL. Zero Wayfinder src/ imports.

For N random proof steps:
  1. Build "perfect" query from ground-truth (correct bank positions + anchors)
  2. navigate(proof_network, perfect_query, limit=16)
  3. Is ground-truth premise in top-16?
  4. For each miss: what anchors *would have* connected goal to premise?
  5. Cluster gap anchors, report for network enrichment

Iterate until: Top-16 recall >= 70%.

Usage:
    python scripts/anchor_gap_analysis.py --db data/proof_network.db --samples 500
    python scripts/anchor_gap_analysis.py --db data/proof_network.db --resume
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.nav_contracts import StructuredQuery
from src.proof_network import navigate


@dataclass
class GapRecord:
    """A single gap analysis result for one proof step."""

    theorem_id: str
    goal_state: str
    ground_truth_premises: list[str]
    retrieved_premises: list[str]
    recall_at_16: float
    missed_premises: list[str]
    gap_anchors: list[str]
    gap_by_category: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "theorem_id": self.theorem_id,
            "goal_state": self.goal_state[:200],
            "ground_truth_premises": self.ground_truth_premises,
            "retrieved_premises": self.retrieved_premises,
            "recall_at_16": self.recall_at_16,
            "missed_premises": self.missed_premises,
            "gap_anchors": self.gap_anchors,
        }
        if self.gap_by_category:
            result["gap_by_category"] = self.gap_by_category
        return result


def load_proof_steps(db_path: str, sample_size: int) -> list[dict]:
    """Load random proof steps with ground-truth premises from the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT e.id, e.name, e.entity_type, e.namespace
        FROM entities e
        WHERE e.entity_type = 'lemma'
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (sample_size,),
    ).fetchall()

    steps = []
    for row in rows:
        theorem_id = row["id"]
        premises = conn.execute(
            """
            SELECT e.name
            FROM accessible_premises ap
            JOIN entities e ON e.id = ap.premise_id
            WHERE ap.theorem_id = ?
            """,
            (theorem_id,),
        ).fetchall()

        positions = conn.execute(
            """
            SELECT bank, sign, depth
            FROM entity_positions
            WHERE entity_id = ?
            """,
            (theorem_id,),
        ).fetchall()

        anchors = conn.execute(
            """
            SELECT a.label
            FROM entity_anchors ea
            JOIN anchors a ON a.id = ea.anchor_id
            WHERE ea.entity_id = ?
            """,
            (theorem_id,),
        ).fetchall()

        steps.append(
            {
                "theorem_id": theorem_id,
                "name": row["name"],
                "namespace": row["namespace"],
                "premises": [p["name"] for p in premises],
                "positions": {p["bank"]: (p["sign"], p["depth"]) for p in positions},
                "anchors": [a["label"] for a in anchors],
            }
        )

    conn.close()
    return steps


def _resolve_anchor_ids(conn: sqlite3.Connection, labels: list[str]) -> list[int]:
    """Map anchor labels to IDs."""
    if not labels:
        return []
    ph = ",".join("?" * len(labels))
    rows = conn.execute(
        f"SELECT id FROM anchors WHERE label IN ({ph})",  # nosec B608
        labels,
    ).fetchall()
    return [r[0] for r in rows]


def build_perfect_query(conn: sqlite3.Connection, step: dict) -> StructuredQuery:
    """Build a perfect StructuredQuery from ground-truth bank positions and anchors.

    Searches the full entity space (no accessible_theorem_id filter) to
    measure whether bank+anchor navigation alone retrieves correct premises.
    Using accessible_theorem_id would leak ground truth since accessible_premises
    is populated from used premises (the labels themselves).
    """
    anchor_ids = _resolve_anchor_ids(conn, step["anchors"])
    return StructuredQuery(
        bank_directions={bank: sign for bank, (sign, _depth) in step["positions"].items()},
        bank_confidences={bank: 1.0 for bank in step["positions"]},
        prefer_anchors=anchor_ids,
        prefer_weights=[1.0] * len(anchor_ids),
    )


def navigate_with_query(
    conn: sqlite3.Connection, query: StructuredQuery, limit: int = 16
) -> list[str]:
    """Execute a navigational query using the real proof network navigate()."""
    results = navigate(conn, query, limit=limit, entity_type="lemma")
    return [r.name for r in results]


def find_gap_anchors_from_conn(
    conn: sqlite3.Connection, missed_premise: str, step_anchors: list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Find anchors of a missed premise that differ from the query's anchors.

    Returns (flat_gap_list, {category: [gap_anchor_labels]}).
    Excludes trivial anchors (general, broad .lake paths) from gap surfacing.
    """
    row = conn.execute("SELECT id FROM entities WHERE name = ?", (missed_premise,)).fetchone()
    if not row:
        return [], {}

    premise_anchors = conn.execute(
        """
        SELECT a.label, a.category
        FROM entity_anchors ea
        JOIN anchors a ON a.id = ea.anchor_id
        WHERE ea.entity_id = ?
        """,
        (row[0],),
    ).fetchall()

    step_set = set(step_anchors)
    gap_flat: list[str] = []
    gap_by_cat: dict[str, list[str]] = {}

    for label, category in premise_anchors:
        if label in step_set:
            continue
        # Skip trivial anchors when surfacing gaps
        if label == "general":
            continue
        if label.startswith("dir:.lake/packages"):
            continue
        gap_flat.append(label)
        gap_by_cat.setdefault(category, []).append(label)

    return sorted(gap_flat), gap_by_cat


def analyze_step(conn: sqlite3.Connection, step: dict) -> GapRecord:
    """Run gap analysis on a single proof step."""
    query = build_perfect_query(conn, step)
    retrieved = navigate_with_query(conn, query, limit=16)

    gt_set = set(step["premises"])
    retrieved_set = set(retrieved)
    hits = gt_set & retrieved_set
    recall = len(hits) / max(len(gt_set), 1)

    missed = sorted(gt_set - retrieved_set)
    gap_anchors: list[str] = []
    merged_cats: dict[str, list[str]] = {}
    for premise in missed:
        flat, by_cat = find_gap_anchors_from_conn(conn, premise, step["anchors"])
        gap_anchors.extend(flat)
        for cat, labels in by_cat.items():
            merged_cats.setdefault(cat, []).extend(labels)

    return GapRecord(
        theorem_id=step["name"],
        goal_state=step.get("namespace", ""),
        ground_truth_premises=step["premises"],
        retrieved_premises=retrieved,
        recall_at_16=recall,
        missed_premises=missed,
        gap_anchors=gap_anchors,
        gap_by_category=merged_cats if merged_cats else None,
    )


def _summarize_records(records: list[GapRecord], output_path: str) -> dict:
    """Compute and print gap analysis summary statistics."""
    if not records:
        print("No records with premises found.")
        return {"status": "empty", "records": 0}

    avg_recall = sum(r.recall_at_16 for r in records) / len(records)
    perfect = sum(1 for r in records if math.isclose(r.recall_at_16, 1.0))
    zero = sum(1 for r in records if math.isclose(r.recall_at_16, 0.0))

    all_gaps: list[str] = []
    cat_gaps: dict[str, list[str]] = {}
    for r in records:
        all_gaps.extend(r.gap_anchors)
        if r.gap_by_category:
            for cat, labels in r.gap_by_category.items():
                cat_gaps.setdefault(cat, []).extend(labels)

    top_gaps = Counter(all_gaps).most_common(20)

    print("\n=== Gap Analysis Summary ===")
    print(f"  Samples analyzed: {len(records)}")
    print(f"  Average recall@16: {avg_recall:.3f}")
    print(f"  Perfect recall: {perfect}/{len(records)} ({100 * perfect / len(records):.1f}%)")
    print(f"  Zero recall: {zero}/{len(records)} ({100 * zero / len(records):.1f}%)")
    print(f"  Gate: {'PASS' if avg_recall >= 0.70 else 'FAIL'} (target: >= 0.70)")

    # Per-category gap distribution
    if cat_gaps:
        print("\n  Gap distribution by lens category:")
        total_gap_count = sum(len(v) for v in cat_gaps.values())
        for cat in sorted(cat_gaps, key=lambda c: len(cat_gaps[c]), reverse=True):
            count = len(cat_gaps[cat])
            pct = 100 * count / max(total_gap_count, 1)
            top_in_cat = Counter(cat_gaps[cat]).most_common(3)
            top_str = ", ".join(f"{a}({c})" for a, c in top_in_cat)
            print(f"    {cat}: {count} ({pct:.0f}%) — top: {top_str}")

    print("\n  Top 20 gap anchors (would improve recall if added):")
    for anchor, count in top_gaps:
        print(f"    {anchor}: {count} misses")

    cat_summary = {
        cat: {"count": len(labels), "top": Counter(labels).most_common(5)}
        for cat, labels in cat_gaps.items()
    } if cat_gaps else {}

    return {
        "status": "complete",
        "records": len(records),
        "avg_recall_at_16": round(avg_recall, 4),
        "perfect_recall_count": perfect,
        "zero_recall_count": zero,
        "gate_passed": avg_recall >= 0.70,
        "top_gap_anchors": [{"anchor": a, "count": c} for a, c in top_gaps],
        "gap_by_category": {
            cat: {"count": info["count"], "top": [{"anchor": a, "count": c} for a, c in info["top"]]}
            for cat, info in cat_summary.items()
        },
        "output_path": output_path,
    }


def run_analysis(db_path: str, sample_size: int, output_path: str) -> dict:
    """Run full gap analysis and write results."""
    print(f"Loading {sample_size} proof steps from {db_path}...")
    steps = load_proof_steps(db_path, sample_size)
    print(f"  Loaded {len(steps)} steps")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    records: list[GapRecord] = []
    for i, step in enumerate(steps):
        if not step["premises"]:
            continue
        record = analyze_step(conn, step)
        records.append(record)
        if (i + 1) % 50 == 0:
            avg_recall = sum(r.recall_at_16 for r in records) / len(records)
            print(f"  Analyzed {i + 1}/{len(steps)}, avg recall@16: {avg_recall:.3f}")

    conn.close()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")

    return _summarize_records(records, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor gap analysis")
    parser.add_argument("--db", type=str, required=True, help="Path to proof_network.db")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/gap_analysis.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.resume and Path(args.output).exists():
        existing = sum(1 for _ in open(args.output))
        remaining = max(0, args.samples - existing)
        print(f"Resuming: {existing} existing records, {remaining} remaining")
        if remaining == 0:
            print("Already complete.")
            return
        args.samples = remaining

    result = run_analysis(args.db, args.samples, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
