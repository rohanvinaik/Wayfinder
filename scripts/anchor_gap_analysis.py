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


def _score_candidates_by_bank(
    conn: sqlite3.Connection, directions: dict, limit: int
) -> list[tuple]:
    """Score entities by bank alignment and return ranked candidates."""
    clauses = []
    params: list = []
    for bank, sign in directions.items():
        clauses.append(
            "SELECT entity_id, "
            "CASE WHEN sign = ? THEN 1.0 WHEN sign = 0 THEN 0.5 ELSE 0.1 END AS score "
            "FROM entity_positions WHERE bank = ?"
        )
        params.extend([sign, bank])

    if not clauses:
        return []

    union_sql = " UNION ALL ".join(clauses)
    sql = f"""
        SELECT entity_id, SUM(score) as total_score
        FROM ({union_sql})
        GROUP BY entity_id
        ORDER BY total_score DESC
        LIMIT ?
    """  # nosec B608 — parameterized via ? placeholders
    params.append(limit * 4)
    return conn.execute(sql, params).fetchall()


def _compute_anchor_boost(
    conn: sqlite3.Connection, anchor_labels: list[str], candidate_ids: list[int]
) -> dict[int, int]:
    """Compute anchor overlap boost for candidate entities."""
    if not anchor_labels:
        return {}
    ph_anchors = ",".join("?" * len(anchor_labels))
    ph_ids = ",".join("?" * len(candidate_ids))
    sql = f"""
        SELECT ea.entity_id, COUNT(*) as shared
        FROM entity_anchors ea
        JOIN anchors a ON a.id = ea.anchor_id
        WHERE a.label IN ({ph_anchors})
        AND ea.entity_id IN ({ph_ids})
        GROUP BY ea.entity_id
    """  # nosec B608 — parameterized via ? placeholders
    rows = conn.execute(sql, [*anchor_labels, *candidate_ids]).fetchall()
    return {r[0]: r[1] for r in rows}


def _resolve_entity_names(conn: sqlite3.Connection, entity_ids: list[int]) -> dict[int, str]:
    """Map entity IDs to names."""
    ph = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT id, name FROM entities WHERE id IN ({ph})",  # nosec B608
        entity_ids,
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def navigate_with_query(db_path: str, query: dict, limit: int = 16) -> list[str]:
    """Execute a navigational query against the proof network.

    Simplified standalone version — scores entities by bank alignment
    and shared anchors without importing src/.
    """
    conn = sqlite3.connect(db_path)

    directions = query["bank_directions"]
    candidates = _score_candidates_by_bank(conn, directions, limit)

    if not candidates:
        conn.close()
        return []

    candidate_ids = [c[0] for c in candidates]
    anchor_boost = _compute_anchor_boost(conn, query.get("anchors", []), candidate_ids)

    scored = [(eid, bank_score + anchor_boost.get(eid, 0) * 0.5) for eid, bank_score in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [s[0] for s in scored[:limit]]

    name_map = _resolve_entity_names(conn, top_ids)
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


def _summarize_records(records: list[GapRecord], output_path: str) -> dict:
    """Compute and print gap analysis summary statistics."""
    if not records:
        print("No records with premises found.")
        return {"status": "empty", "records": 0}

    avg_recall = sum(r.recall_at_16 for r in records) / len(records)
    perfect = sum(1 for r in records if math.isclose(r.recall_at_16, 1.0))
    zero = sum(1 for r in records if math.isclose(r.recall_at_16, 0.0))

    all_gaps: list[str] = []
    for r in records:
        all_gaps.extend(r.gap_anchors)
    top_gaps = Counter(all_gaps).most_common(20)

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
        "output_path": output_path,
    }


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
