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
from src.premise_retrieval import landmark_expand_retrieve
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


# The anchor-label vocabulary lives in src.structural_reads (the vocabulary
# module, importable without numpy); re-exported here for query consumers.
from src.structural_reads import symbolic_query_labels  # noqa: E402


def build_symbolic_query(
    conn: sqlite3.Connection,
    goal_text: str,
    top_k_anchors: int = 24,
) -> StructuredQuery:
    """Build a StructuredQuery from goal TEXT alone — no navigator, no model.

    The deterministic twin of :func:`build_query`. Anchors come from exact
    set intersection between the goal's typed candidate labels
    (:func:`symbolic_query_labels`) and the proof network's anchor
    dictionary; banks stay neutral (empty = the mode). IDF weighting happens
    downstream in anchor scoring, so all matched anchors carry weight 1.0.
    """
    labels = symbolic_query_labels(goal_text)
    prefer_anchors: list[int] = []
    if labels:
        placeholders = ",".join("?" for _ in labels)
        rows = conn.execute(
            f"SELECT id FROM anchors WHERE label IN ({placeholders}) ORDER BY id",  # nosec B608 — "?" placeholders only; values bound
            labels,
        ).fetchall()
        prefer_anchors = [row[0] for row in rows][:top_k_anchors]

    return StructuredQuery(
        bank_directions={},
        bank_confidences={},
        prefer_anchors=prefer_anchors,
        prefer_weights=[1.0] * len(prefer_anchors),
        avoid_anchors=[],
    )


def library_premise_pool(
    conn: sqlite3.Connection,
    goal_text: str,
    limit: int = 200,
) -> list[str]:
    """The premise pool for a goal with NO site: the library universe, by evidence.

    The accessible-premises pool is a site-indexed anti-circularity constraint
    (a theorem replayed at its Mathlib site must not use later lemmas). An
    EXTERNAL goal — MiniF2F, a fresh conjecture — has no site and no
    circularity hazard: every proved lemma in the network is legitimately
    available (its environment is the whole library; cf. LeanExpr.environment).

    "No site" is therefore NOT "no premises" — conflating them is the
    determined-empty/cannot-determine error. But an unconstrained pool is only
    honest when EVIDENCE selects from it: this pool is the lemma entities
    sharing at least one anchor with the goal, scored by summed anchor IDF,
    deterministic tie-break by name. A goal whose tokens match no anchor gets
    an EMPTY pool — no evidence, no candidates, the lane abstains — never an
    unranked dump of the library.
    """
    query = build_symbolic_query(conn, goal_text)
    if not query.prefer_anchors:
        return []
    placeholders = ",".join("?" for _ in query.prefer_anchors)
    rows = conn.execute(
        f"""
        SELECT e.name, SUM(COALESCE(ai.idf_value, 1.0)) AS score
        FROM entity_anchors ea
        JOIN entities e ON e.id = ea.entity_id
        LEFT JOIN anchor_idf ai ON ai.anchor_id = ea.anchor_id
        WHERE ea.anchor_id IN ({placeholders}) AND e.entity_type = 'lemma'
        GROUP BY ea.entity_id
        ORDER BY score DESC, e.name ASC
        LIMIT ?
        """,  # nosec B608 — "?" placeholders only; values bound
        [*query.prefer_anchors, limit],
    ).fetchall()
    return [name for name, _score in rows]


def symbolic_rank_premises(
    conn: sqlite3.Connection,
    goal_text: str,
    premise_names: list[str],
    max_premises: int = 30,
) -> list[str]:
    """Rank a premise pool by IDF-weighted anchor intersection with the goal.

    Fully deterministic: goal tokens -> anchor dictionary (set intersection)
    -> per-premise IDF-weighted overlap score -> stable sort with the
    original pool order as tie-break. No embeddings, no model, no learned
    state. This is the retrieval mechanism the network was built for,
    driven from text instead of from navigator coordinates.
    """
    from src.proof_network import _get_idf_cache
    from src.proof_scoring import _compute_anchor_score

    if not premise_names:
        return []

    query = build_symbolic_query(conn, goal_text)
    if not query.prefer_anchors:
        return list(premise_names[:max_premises])

    placeholders = ",".join("?" for _ in premise_names)
    id_rows = conn.execute(
        f"SELECT id, name FROM entities WHERE name IN ({placeholders})",  # nosec B608 — "?" placeholders only; values bound
        list(premise_names),
    ).fetchall()
    name_to_eid = {name: eid for eid, name in id_rows}

    anchor_sets: dict[int, set[int]] = {}
    eids = list(name_to_eid.values())
    if eids:
        eid_placeholders = ",".join("?" for _ in eids)
        for eid, aid in conn.execute(
            f"SELECT entity_id, anchor_id FROM entity_anchors "
            f"WHERE entity_id IN ({eid_placeholders})",  # nosec B608 — "?" placeholders only; values bound
            eids,
        ).fetchall():
            anchor_sets.setdefault(eid, set()).add(aid)

    idf_cache = _get_idf_cache(conn)
    scored: list[tuple[float, int, str]] = []
    for pool_idx, name in enumerate(premise_names):
        eid = name_to_eid.get(name)
        ea_set = anchor_sets.get(eid, set()) if eid is not None else set()
        score = _compute_anchor_score(ea_set, query, idf_cache)
        scored.append((-score, pool_idx, name))
    scored.sort()
    return [name for _neg, _idx, name in scored[:max_premises]]


def resolve(
    nav_output: NavOutput,
    conn: sqlite3.Connection,
    context: SearchContext,
    anchor_id_map: dict[str, int] | None = None,
    tactic_limit: int = 8,
    premise_limit: int = 16,
    spread_depth: int = 2,
    retrieval_config: dict | None = None,
) -> list[Candidate]:
    """Resolve navigator output to ranked tactic-premise candidates.

    Steps:
        1. Build structured query from nav output
        2. Navigate for tactic entities
        3. Retrieve premise entities (flat navigate or landmark_expand)
        4. Spread activation from seed entities for re-ranking
        5. Combine and rank candidates

    Args:
        retrieval_config: If provided, controls premise retrieval strategy.
            strategy: "flat" (default) or "landmark_expand"
            Plus any landmark_expand_retrieve config keys.
    """
    query = build_query(nav_output, anchor_id_map)

    if context.accessible_theorem_id is not None:
        query.accessible_theorem_id = context.accessible_theorem_id
    if context.seed_entity_ids:
        query.seed_entity_ids = list(context.seed_entity_ids)

    # Retrieve tactic entities (always flat navigate)
    tactics = navigate(conn, query, limit=tactic_limit, entity_type="tactic")

    # Retrieve premise entities
    rcfg = retrieval_config or {}
    strategy = rcfg.get("strategy", "flat")
    if strategy == "landmark_expand":
        premises, _ = landmark_expand_retrieve(query, conn, limit=premise_limit, config=rcfg)
    else:
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
