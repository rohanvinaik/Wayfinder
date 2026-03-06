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
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


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

    def to_dict(self) -> dict:
        return {
            "theorem_id": self.theorem_id,
            "goal_state": self.goal_state[:200],
            "ground_truth_premises": self.ground_truth_premises,
            "retrieved_premises": self.retrieved_premises,
            "recall_at_16": self.recall_at_16,
            "missed_premises": self.missed_premises,
            "gap_anchors": self.gap_anchors,
        }


def load_proof_steps(db_path: str, sample_size: int) -> list[dict]:
    """Load random proof steps with ground-truth premises from the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT e.id, e.name, e.type, e.namespace
        FROM entities e
        WHERE e.type = 'theorem'
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


def build_perfect_query(step: dict) -> dict:
    """Build a perfect query from ground-truth bank positions and anchors."""
    return {
        "bank_directions": {bank: sign for bank, (sign, _depth) in step["positions"].items()},
        "bank_confidences": {bank: 1.0 for bank in step["positions"]},
        "anchors": step["anchors"],
    }


def navigate_with_query(db_path: str, query: dict, limit: int = 16) -> list[str]:
    """Execute a navigational query against the proof network.

    Simplified standalone version — scores entities by bank alignment
    and shared anchors without importing src/.
    """
    conn = sqlite3.connect(db_path)

    directions = query["bank_directions"]
    anchor_labels = query.get("anchors", [])

    # Score by bank alignment
    bank_clauses = []
    bank_params: list = []
    for bank, sign in directions.items():
        bank_clauses.append(
            "SELECT entity_id, "
            "CASE WHEN sign = ? THEN 1.0 WHEN sign = 0 THEN 0.5 ELSE 0.1 END AS score "
            "FROM entity_positions WHERE bank = ?"
        )
        bank_params.extend([sign, bank])

    if not bank_clauses:
        conn.close()
        return []

    union_sql = " UNION ALL ".join(bank_clauses)
    scored_sql = f"""
        SELECT entity_id, SUM(score) as total_score
        FROM ({union_sql})
        GROUP BY entity_id
        ORDER BY total_score DESC
        LIMIT ?
    """
    bank_params.append(limit * 4)

    candidates = conn.execute(scored_sql, bank_params).fetchall()
    candidate_ids = [c[0] for c in candidates]

    if not candidate_ids:
        conn.close()
        return []

    # Boost by shared anchors
    if anchor_labels:
        placeholders = ",".join("?" * len(anchor_labels))
        anchor_sql = f"""
            SELECT ea.entity_id, COUNT(*) as shared
            FROM entity_anchors ea
            JOIN anchors a ON a.id = ea.anchor_id
            WHERE a.label IN ({placeholders})
            AND ea.entity_id IN ({",".join("?" * len(candidate_ids))})
            GROUP BY ea.entity_id
        """
        anchor_rows = conn.execute(anchor_sql, [*anchor_labels, *candidate_ids]).fetchall()
        anchor_boost = {r[0]: r[1] for r in anchor_rows}
    else:
        anchor_boost = {}

    scored = []
    for eid, bank_score in candidates:
        boost = anchor_boost.get(eid, 0) * 0.5
        scored.append((eid, bank_score + boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [s[0] for s in scored[:limit]]

    placeholders = ",".join("?" * len(top_ids))
    names = conn.execute(
        f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
        top_ids,
    ).fetchall()
    name_map = {r[0]: r[1] for r in names}
    conn.close()

    return [name_map.get(eid, "") for eid in top_ids]


def find_gap_anchors(db_path: str, missed_premise: str, step_anchors: list[str]) -> list[str]:
    """Find anchors of a missed premise that differ from the query's anchors."""
    conn = sqlite3.connect(db_path)

    row = conn.execute("SELECT id FROM entities WHERE name = ?", (missed_premise,)).fetchone()
    if not row:
        conn.close()
        return []

    premise_anchors = conn.execute(
        """
        SELECT a.label
        FROM entity_anchors ea
        JOIN anchors a ON a.id = ea.anchor_id
        WHERE ea.entity_id = ?
        """,
        (row[0],),
    ).fetchall()
    conn.close()

    premise_labels = {a[0] for a in premise_anchors}
    step_set = set(step_anchors)
    return sorted(premise_labels - step_set)


def analyze_step(db_path: str, step: dict) -> GapRecord:
    """Run gap analysis on a single proof step."""
    query = build_perfect_query(step)
    retrieved = navigate_with_query(db_path, query, limit=16)

    gt_set = set(step["premises"])
    retrieved_set = set(retrieved)
    hits = gt_set & retrieved_set
    recall = len(hits) / max(len(gt_set), 1)

    missed = sorted(gt_set - retrieved_set)
    gap_anchors: list[str] = []
    for premise in missed:
        gap_anchors.extend(find_gap_anchors(db_path, premise, step["anchors"]))

    return GapRecord(
        theorem_id=step["name"],
        goal_state=step.get("namespace", ""),
        ground_truth_premises=step["premises"],
        retrieved_premises=retrieved,
        recall_at_16=recall,
        missed_premises=missed,
        gap_anchors=gap_anchors,
    )


def run_analysis(db_path: str, sample_size: int, output_path: str) -> dict:
    """Run full gap analysis and write results."""
    print(f"Loading {sample_size} proof steps from {db_path}...")
    steps = load_proof_steps(db_path, sample_size)
    print(f"  Loaded {len(steps)} steps")

    records: list[GapRecord] = []
    for i, step in enumerate(steps):
        if not step["premises"]:
            continue
        record = analyze_step(db_path, step)
        records.append(record)
        if (i + 1) % 50 == 0:
            avg_recall = sum(r.recall_at_16 for r in records) / len(records)
            print(f"  Analyzed {i + 1}/{len(steps)}, avg recall@16: {avg_recall:.3f}")

    # Write results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")

    # Summary
    if not records:
        print("No records with premises found.")
        return {"status": "empty", "records": 0}

    avg_recall = sum(r.recall_at_16 for r in records) / len(records)
    perfect = sum(1 for r in records if r.recall_at_16 == 1.0)
    zero = sum(1 for r in records if r.recall_at_16 == 0.0)

    all_gaps = []
    for r in records:
        all_gaps.extend(r.gap_anchors)
    gap_counter = Counter(all_gaps)
    top_gaps = gap_counter.most_common(20)

    print("\n=== Gap Analysis Summary ===")
    print(f"  Samples analyzed: {len(records)}")
    print(f"  Average recall@16: {avg_recall:.3f}")
    print(f"  Perfect recall: {perfect}/{len(records)} ({100 * perfect / len(records):.1f}%)")
    print(f"  Zero recall: {zero}/{len(records)} ({100 * zero / len(records):.1f}%)")
    print(f"  Gate: {'PASS' if avg_recall >= 0.70 else 'FAIL'} (target: >= 0.70)")
    print("\n  Top 20 gap anchors (would improve recall if added):")
    for anchor, count in top_gaps:
        print(f"    {anchor}: {count} misses")

    return {
        "status": "complete",
        "records": len(records),
        "avg_recall_at_16": round(avg_recall, 4),
        "perfect_recall_count": perfect,
        "zero_recall_count": zero,
        "gate_passed": avg_recall >= 0.70,
        "top_gap_anchors": [{"anchor": a, "count": c} for a, c in top_gaps],
        "output_path": str(out_path),
    }


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
