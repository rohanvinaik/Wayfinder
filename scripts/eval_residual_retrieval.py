"""Evaluate retrieval quality on post-structural residual goals.

For each residual example: build a query from the goal state,
run navigate() on the proof network, check if ground-truth premises
appear in top-k results. Reports recall@k overall and by tactic family.

This is the aligned Task A metric: does retrieval help at the residual level?

Usage:
    python -m scripts.eval_residual_retrieval --samples 5000
    python -m scripts.eval_residual_retrieval --samples 5000 --db data/proof_network_v3.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from src.nav_contracts import StructuredQuery
from src.proof_network import navigate


def _resolve_premise_ids(conn: sqlite3.Connection, premise_names: list[str]) -> set[int]:
    """Map premise names to entity IDs in the proof network."""
    if not premise_names:
        return set()
    ph = ",".join("?" * len(premise_names))
    rows = conn.execute(
        f"SELECT id FROM entities WHERE name IN ({ph})",  # noqa: S608
        premise_names,
    ).fetchall()
    return {r[0] for r in rows}


def _build_query_from_theorem(
    conn: sqlite3.Connection, theorem_id: str, bank_positions: dict
) -> StructuredQuery:
    """Build a StructuredQuery from the theorem's DB entry.

    Uses the theorem's bank positions and anchor set from the proof network,
    NOT the navigator's (often all-zero) direction predictions.
    """
    # Look up theorem entity ID
    row = conn.execute("SELECT id FROM entities WHERE name = ?", (theorem_id,)).fetchone()
    if not row:
        return StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[],
            prefer_weights=[],
        )
    entity_id = row[0]

    # Get bank positions from the training example
    bank_dirs = {}
    bank_confs = {}
    for bank, pos in bank_positions.items():
        if isinstance(pos, list) and len(pos) == 2:
            sign, depth = pos
            if sign != 0:
                bank_dirs[bank] = sign
                bank_confs[bank] = min(depth / 3.0, 1.0) if depth else 0.5

    # Get the theorem's anchors from the DB
    anchor_rows = conn.execute(
        """SELECT ea.anchor_id, ai.idf_value
           FROM entity_anchors ea
           LEFT JOIN anchor_idf ai ON ai.anchor_id = ea.anchor_id
           WHERE ea.entity_id = ?""",
        (entity_id,),
    ).fetchall()

    prefer_anchors = [r[0] for r in anchor_rows]
    prefer_weights = [r[1] if r[1] else 1.0 for r in anchor_rows]

    return StructuredQuery(
        bank_directions=bank_dirs,
        bank_confidences=bank_confs,
        prefer_anchors=prefer_anchors,
        prefer_weights=prefer_weights,
    )


def evaluate(
    residual_path: Path,
    db_path: Path,
    samples: int,
    seed: int = 42,
) -> dict:
    """Evaluate retrieval on residual goals."""
    rng = np.random.default_rng(seed)

    # Load residual examples
    print(f"Loading residual data from {residual_path}...")
    examples = []
    with open(residual_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("ground_truth_premises"):
                examples.append(ex)

    if len(examples) > samples:
        indices = rng.choice(len(examples), samples, replace=False)
        examples = [examples[int(i)] for i in indices]
    print(f"  {len(examples)} examples with premises (sampled from {samples})")

    conn = sqlite3.connect(db_path)

    # Build name→id map for premise lookup
    print("Building entity name→id map...")
    name_to_id = {}
    for row in conn.execute("SELECT id, name FROM entities").fetchall():
        name_to_id[row[1]] = row[0]

    # Evaluate
    ks = [1, 4, 8, 16, 32]
    hits_by_k: dict[int, int] = {k: 0 for k in ks}
    total_with_coverage = 0

    # Per-family tracking
    family_hits: dict[str, dict[int, int]] = {}
    family_total: dict[str, int] = Counter()

    for i, ex in enumerate(examples):
        gt_premises = ex["ground_truth_premises"]
        gt_ids = {name_to_id[p] for p in gt_premises if p in name_to_id}

        if not gt_ids:
            continue  # premises not in DB

        total_with_coverage += 1
        family = ex["tactic_base"]
        family_total[family] += 1

        # Build query from theorem's DB entry (anchors + bank positions)
        query = _build_query_from_theorem(conn, ex["theorem_id"], ex.get("bank_positions", {}))
        results = navigate(conn, query, limit=32, entity_type="lemma")
        retrieved_ids = {r.entity_id for r in results}

        # Check recall at each k
        for k in ks:
            top_k_ids = {r.entity_id for r in results[:k]}
            if gt_ids & top_k_ids:
                hits_by_k[k] += 1
                family_hits.setdefault(family, {}).setdefault(k, 0)
                family_hits[family][k] += 1

        if (i + 1) % 1000 == 0:
            r16 = hits_by_k[16] / max(total_with_coverage, 1)
            print(
                f"  {i + 1}/{len(examples)}: recall@16={r16:.3f} (coverage={total_with_coverage})"
            )

    conn.close()

    # Compute metrics
    print("\n=== Residual Retrieval Eval ===")
    print(f"  Samples: {len(examples)}, with DB coverage: {total_with_coverage}")

    recall = {}
    for k in ks:
        r = hits_by_k[k] / max(total_with_coverage, 1)
        recall[f"recall@{k}"] = round(r, 4)
        print(f"  recall@{k}: {r:.3f}")

    # Per-family
    print("\n  By tactic family (recall@16):")
    family_recall = {}
    for fam in ["rw", "simp", "exact", "apply", "refine"]:
        ft = family_total.get(fam, 0)
        fh = family_hits.get(fam, {}).get(16, 0)
        r = fh / max(ft, 1)
        family_recall[fam] = round(r, 4)
        print(f"    {fam}: {r:.3f} ({fh}/{ft})")

    return {
        **recall,
        "total": total_with_coverage,
        "family_recall_at_16": family_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval residual retrieval")
    parser.add_argument("--data", type=Path, default=Path("data/residual_train.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/proof_network.db"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate(args.data, args.db, args.samples, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
