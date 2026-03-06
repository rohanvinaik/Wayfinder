"""
Resolution — convert navigator output to concrete tactic + premise candidates.

Takes a NavOutput (directions, anchor scores, progress, critic) and resolves
it against the proof network to produce ranked tactic-premise candidates.

This is the symbolic bridge between the neural navigator and the Lean kernel.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.nav_contracts import NavOutput, ScoredEntity, StructuredQuery
from src.proof_network import navigate, spread


@dataclass
class SearchContext:
    """Context for proof search resolution."""

    accessible_theorem_id: int | None = None
    seed_entity_ids: list[int] = field(default_factory=list)
    used_tactics: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """A resolved tactic-premise candidate for the Lean kernel."""

    tactic_name: str
    premises: list[str]
    score: float
    tactic_entity: ScoredEntity | None = None
    premise_entities: list[ScoredEntity] = field(default_factory=list)


def build_query(
    nav_output: NavOutput,
    anchor_id_map: dict[str, int] | None = None,
    anchor_threshold: float = 0.3,
    top_k_anchors: int = 10,
) -> StructuredQuery:
    """Build a StructuredQuery from neural navigator output.

    Args:
        nav_output: Output from ProofNavigator.predict().
        anchor_id_map: Mapping from anchor label strings to DB integer IDs.
        anchor_threshold: Minimum sigmoid score to include an anchor.
        top_k_anchors: Maximum preferred anchors to include.
    """
    bank_directions = dict(nav_output.directions)
    bank_confidences = dict(nav_output.direction_confidences)

    # Convert anchor scores to prefer/avoid lists
    prefer_anchors: list[int] = []
    prefer_weights: list[float] = []
    avoid_anchors: list[int] = []

    if anchor_id_map:
        scored = sorted(nav_output.anchor_scores.items(), key=lambda x: x[1], reverse=True)
        for label, score in scored[:top_k_anchors]:
            if label in anchor_id_map and score >= anchor_threshold:
                prefer_anchors.append(anchor_id_map[label])
                prefer_weights.append(score)

    return StructuredQuery(
        bank_directions=bank_directions,
        bank_confidences=bank_confidences,
        prefer_anchors=prefer_anchors,
        prefer_weights=prefer_weights,
        avoid_anchors=avoid_anchors,
    )


def resolve(
    nav_output: NavOutput,
    conn: sqlite3.Connection,
    context: SearchContext,
    anchor_id_map: dict[str, int] | None = None,
    tactic_limit: int = 8,
    premise_limit: int = 16,
    spread_depth: int = 2,
) -> list[Candidate]:
    """Resolve navigator output to ranked tactic-premise candidates.

    Steps:
        1. Build structured query from nav output
        2. Navigate for tactic entities
        3. Navigate for premise entities
        4. Spread activation from seed entities for re-ranking
        5. Combine and rank candidates
    """
    query = build_query(nav_output, anchor_id_map)

    if context.accessible_theorem_id is not None:
        query.accessible_theorem_id = context.accessible_theorem_id
    if context.seed_entity_ids:
        query.seed_entity_ids = list(context.seed_entity_ids)

    # Retrieve tactic entities
    tactics = navigate(conn, query, limit=tactic_limit, entity_type="tactic")

    # Retrieve premise entities
    premises = navigate(conn, query, limit=premise_limit, entity_type="lemma")

    # Spread activation from seeds for re-ranking
    spread_scores: dict[int, float] = {}
    if context.seed_entity_ids:
        spread_scores = spread(conn, context.seed_entity_ids, max_depth=spread_depth)

    return _combine_candidates(tactics, premises, spread_scores)


def _combine_candidates(
    tactics: list[ScoredEntity],
    premises: list[ScoredEntity],
    spread_scores: dict[int, float],
) -> list[Candidate]:
    """Combine tactic and premise results into ranked candidates."""
    if not tactics:
        return []

    # Boost premise scores with spread activation
    boosted_premises = []
    for p in premises:
        boost = spread_scores.get(p.entity_id, 1.0)
        boosted_score = p.final_score * (1.0 + 0.3 * boost)
        boosted_premises.append((p, boosted_score))
    boosted_premises.sort(key=lambda x: x[1], reverse=True)

    # Each tactic gets paired with the top premises
    candidates: list[Candidate] = []
    premise_names = [p.name for p, _ in boosted_premises]

    for tactic in tactics:
        tactic_boost = spread_scores.get(tactic.entity_id, 1.0)
        score = tactic.final_score * (1.0 + 0.3 * tactic_boost)
        candidates.append(
            Candidate(
                tactic_name=tactic.name,
                premises=premise_names,
                score=score,
                tactic_entity=tactic,
                premise_entities=[p for p, _ in boosted_premises],
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
