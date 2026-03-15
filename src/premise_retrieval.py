"""Three-stage premise retrieval: landmark → expand → rerank.

v3 retrieval architecture. Does NOT replace flat navigate() — lives
alongside it as an alternative strategy for v3_runtime.

Stage 1: retrieve_landmarks — find high-confidence traced entities
         using primary lens scoring (proof, constant, semantic, structural).
Stage 2: expand_from_landmarks — walk typed graph links to reach
         premise-only entities within N hops.
Stage 3: rerank_candidates — score expanded pool by multi-lens coherence
         with graph path support.

Config keys (under retrieval:):
    strategy: landmark_expand
    landmark_limit: 64
    expansion_hops: 2
    expansion_topk_per_seed: 24
    expansion_relations: [depends_on, co_used_with, shared_constant, ...]
    rerank_limit: 128
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_network import _get_data_cache, _get_idf_cache
from src.proof_scoring import (
    LENS_CATEGORIES,
    _vectorized_bank_scores,
    compute_observability_score,
)

# Primary lenses decide semantic relevance; support lenses stabilize.
PRIMARY_LENSES = frozenset({"proof", "constant", "semantic", "structural"})
SUPPORT_LENSES = frozenset({"locality", "lexical"})

# Default config
DEFAULT_CONFIG: dict = {
    "landmark_limit": 64,
    "expansion_hops": 1,
    "expansion_topk_per_seed": 16,
    "expansion_relations": [
        "depends_on",
        "co_used_with",
        "shared_constant",
    ],
    "rerank_limit": 128,
    "landmark_traced_bias": False,
}


@dataclass
class ExpandedCandidate:
    """A candidate reached via graph expansion from a landmark."""

    entity_id: int
    name: str
    best_relation: str
    hops: int
    path_score: float
    source_landmark_id: int


@dataclass
class RetrievalTrace:
    """Diagnostic trace for a retrieval call."""

    landmark_count: int = 0
    expanded_pool_size: int = 0
    final_count: int = 0
    landmarks_by_provenance: dict[str, int] = field(default_factory=dict)
    expanded_by_relation: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: Landmark retrieval
# ---------------------------------------------------------------------------


def _compute_primary_seed_score(
    entity_anchors: set[int],
    query_anchors: list[int],
    query_weights: list[float],
    idf_cache: dict[int, float],
    anchor_categories: dict[int, str],
    anchor_confidences: dict[int, float] | None = None,
) -> float:
    """Primary lens scoring for landmark seeding.

    Rules:
        proof alone may seed.
        rare constant alone may seed.
        semantic + structural may seed.
        constant + semantic may seed.
        proof + semantic may seed.
        constant + structural may seed.
        locality alone CANNOT seed.
        lexical alone CANNOT seed.
    """
    # Partition query anchors by category, compute per-lens overlap
    lens_scores: dict[str, float] = {}
    query_by_cat: dict[str, list[tuple[int, float]]] = {}
    for aid, weight in zip(query_anchors, query_weights):
        cat = anchor_categories.get(aid, "general")
        if cat in LENS_CATEGORIES:
            query_by_cat.setdefault(cat, []).append((aid, weight))

    for cat, items in query_by_cat.items():
        total_idf = 0.0
        matched_idf = 0.0
        for aid, weight in items:
            idf = idf_cache.get(aid, 1.0) * weight
            total_idf += idf
            if aid in entity_anchors:
                conf = anchor_confidences.get(aid, 1.0) if anchor_confidences else 1.0
                matched_idf += idf * conf
        if total_idf > 0:
            lens_scores[cat] = matched_idf / total_idf

    proof = lens_scores.get("proof", 0.0)
    constant = lens_scores.get("constant", 0.0)
    semantic = lens_scores.get("semantic", 0.0)
    structural = lens_scores.get("structural", 0.0)
    locality = lens_scores.get("locality", 0.0)
    lexical = lens_scores.get("lexical", 0.0)

    seed_primary = max(
        proof,
        0.95 * constant,
        0.90 * math.sqrt(max(semantic * structural, 0)),
        0.90 * math.sqrt(max(constant * semantic, 0)),
        0.85 * math.sqrt(max(proof * semantic, 0)),
        0.80 * math.sqrt(max(constant * structural, 0)),
    )

    support_bonus = 1.0 + 0.10 * locality + 0.08 * lexical

    return seed_primary * support_bonus


def retrieve_landmarks(
    query: StructuredQuery,
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Stage 1: retrieve high-confidence landmark entities.

    Scores traced entities using primary lens scoring, plus any entities
    reachable via direct depends_on links from accessible premises.
    Locality/lexical alone cannot seed a landmark.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    limit = cfg["landmark_limit"]
    traced_bias = cfg["landmark_traced_bias"]

    data = _get_data_cache(conn)
    idf_cache = _get_idf_cache(conn)

    # Get candidate IDs — prefer traced entities for landmark seeding
    if traced_bias:
        candidate_ids = [
            eid for eid, prov in data.provenances.items()
            if prov == "traced"
        ]
    else:
        candidate_ids = list(data.names.keys())

    if not candidate_ids:
        return [], RetrievalTrace()

    # Bank scores (vectorized)
    bank_scores = _vectorized_bank_scores(candidate_ids, data.positions, query)

    results: list[ScoredEntity] = []
    for i, eid in enumerate(candidate_ids):
        b_score = float(bank_scores[i])
        if b_score <= 0:
            continue

        ea_set = data.anchor_sets.get(eid, set())
        entity_confs = data.anchor_confidences.get(eid)

        primary = _compute_primary_seed_score(
            ea_set,
            query.prefer_anchors,
            query.prefer_weights,
            idf_cache,
            data.anchor_categories,
            entity_confs,
        )

        obs = compute_observability_score(
            data.positions.get(eid, {}),
            ea_set,
            data.provenances.get(eid, "traced"),
            data.anchor_categories,
        )

        # Primary is a bonus, not a gate — bank alignment is the
        # primary landmark signal. Observability separates traced from
        # premise-only. Primary lens boosts when available.
        final = b_score * obs * (1.0 + primary)
        results.append(ScoredEntity(
            entity_id=eid,
            name=data.names.get(eid, ""),
            final_score=final,
            bank_score=b_score,
            anchor_score=primary,
            seed_score=obs,
        ))

    results.sort(key=lambda e: e.final_score, reverse=True)
    landmarks = results[:limit]

    # Trace
    trace = RetrievalTrace(landmark_count=len(landmarks))
    prov_counts: dict[str, int] = defaultdict(int)
    for lm in landmarks:
        prov_counts[data.provenances.get(lm.entity_id, "?")] += 1
    trace.landmarks_by_provenance = dict(prov_counts)

    return landmarks, trace


# ---------------------------------------------------------------------------
# Stage 2: Graph expansion
# ---------------------------------------------------------------------------


def expand_from_landmarks(
    landmarks: list[ScoredEntity],
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> tuple[list[ExpandedCandidate], RetrievalTrace]:
    """Stage 2: walk typed graph links from landmarks to reach premise-only entities.

    BFS up to expansion_hops, following only whitelisted relations.
    Returns candidates with path provenance (relation, hops, score).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    max_hops = cfg["expansion_hops"]
    topk_per_seed = cfg["expansion_topk_per_seed"]
    allowed_relations = set(cfg["expansion_relations"])

    landmark_ids = {lm.entity_id for lm in landmarks}

    data = _get_data_cache(conn)

    # Pre-load typed links from DB (only whitelisted relations)
    if allowed_relations:
        placeholders = ",".join("?" * len(allowed_relations))
        link_rows = conn.execute(
            f"SELECT source_id, target_id, relation, weight "  # nosec B608
            f"FROM entity_links WHERE relation IN ({placeholders})",
            list(allowed_relations),
        ).fetchall()
    else:
        link_rows = []

    # Build adjacency: source → [(target, relation, weight)]
    adj: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for src, tgt, rel, wt in link_rows:
        adj[src].append((tgt, rel, wt))

    # BFS from landmarks
    visited: dict[int, ExpandedCandidate] = {}
    frontier = [(lm_id, lm_id, 0, 1.0) for lm_id in landmark_ids]

    for _ in range(max_hops):
        next_frontier: list[tuple[int, int, int, float]] = []
        for eid, source_lm, hops, path_score in frontier:
            neighbors = adj.get(eid, [])
            # Sort by weight descending, take topk
            neighbors.sort(key=lambda x: x[2], reverse=True)
            for tgt, rel, wt in neighbors[:topk_per_seed]:
                if tgt in landmark_ids:
                    continue  # don't re-expand landmarks
                new_score = path_score * wt
                if tgt in visited and visited[tgt].path_score >= new_score:
                    continue
                cand = ExpandedCandidate(
                    entity_id=tgt,
                    name=data.names.get(tgt, ""),
                    best_relation=rel,
                    hops=hops + 1,
                    path_score=new_score,
                    source_landmark_id=source_lm,
                )
                visited[tgt] = cand
                next_frontier.append((tgt, source_lm, hops + 1, new_score))
        frontier = next_frontier

    expanded = sorted(visited.values(), key=lambda c: c.path_score, reverse=True)

    # Trace
    trace = RetrievalTrace(expanded_pool_size=len(expanded))
    rel_counts: dict[str, int] = defaultdict(int)
    for c in expanded:
        rel_counts[c.best_relation] += 1
    trace.expanded_by_relation = dict(rel_counts)

    return expanded, trace


# ---------------------------------------------------------------------------
# Stage 3: Reranking
# ---------------------------------------------------------------------------


def _compute_primary_coherence(
    entity_anchors: set[int],
    query_anchors: list[int],
    query_weights: list[float],
    idf_cache: dict[int, float],
    anchor_categories: dict[int, str],
    anchor_confidences: dict[int, float] | None = None,
) -> float:
    """Geometric mean over active primary lenses only."""
    query_by_cat: dict[str, list[tuple[int, float]]] = {}
    for aid, weight in zip(query_anchors, query_weights):
        cat = anchor_categories.get(aid, "general")
        if cat in PRIMARY_LENSES:
            query_by_cat.setdefault(cat, []).append((aid, weight))

    scores: dict[str, float] = {}
    for cat, items in query_by_cat.items():
        total_idf = 0.0
        matched_idf = 0.0
        for aid, weight in items:
            idf = idf_cache.get(aid, 1.0) * weight
            total_idf += idf
            if aid in entity_anchors:
                conf = anchor_confidences.get(aid, 1.0) if anchor_confidences else 1.0
                matched_idf += idf * conf
        if total_idf > 0:
            scores[cat] = matched_idf / total_idf

    populated = {c: s for c, s in scores.items() if s > 0}
    if not populated:
        return 0.0
    return math.prod(populated.values()) ** (1.0 / len(populated))


def _compute_support_bonus(
    entity_anchors: set[int],
    query_anchors: list[int],
    query_weights: list[float],
    idf_cache: dict[int, float],
    anchor_categories: dict[int, str],
) -> float:
    """Support lens bonus: 1 + 0.10 * locality + 0.08 * lexical."""
    locality = 0.0
    lexical = 0.0
    for cat_name, _ in [("locality", "loc"), ("lexical", "lex")]:
        total = 0.0
        matched = 0.0
        for aid, weight in zip(query_anchors, query_weights):
            if anchor_categories.get(aid, "") == cat_name:
                idf = idf_cache.get(aid, 1.0) * weight
                total += idf
                if aid in entity_anchors:
                    matched += idf
        if total > 0:
            if cat_name == "locality":
                locality = matched / total
            else:
                lexical = matched / total

    return 1.0 + 0.10 * locality + 0.08 * lexical


def rerank_candidates(
    query: StructuredQuery,
    landmarks: list[ScoredEntity],
    expanded: list[ExpandedCandidate],
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Stage 3: rerank the merged landmark + expanded pool.

    rerank_score = bank_score * observability * primary_coherence
                   * support_bonus * expansion_support
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    limit = cfg["rerank_limit"]

    data = _get_data_cache(conn)
    idf_cache = _get_idf_cache(conn)

    # Merge candidates: landmarks + expanded (deduplicated)
    all_ids: dict[int, float] = {}  # eid → expansion_support
    for lm in landmarks:
        all_ids[lm.entity_id] = 1.0  # landmarks get full expansion support
    for ec in expanded:
        if ec.entity_id not in all_ids or ec.path_score > all_ids[ec.entity_id]:
            all_ids[ec.entity_id] = ec.path_score

    candidate_ids = list(all_ids.keys())
    if not candidate_ids:
        return [], RetrievalTrace()

    # Bank scores (vectorized)
    bank_scores = _vectorized_bank_scores(candidate_ids, data.positions, query)

    results: list[ScoredEntity] = []
    for i, eid in enumerate(candidate_ids):
        b_score = float(bank_scores[i])
        if b_score <= 0:
            continue

        ea_set = data.anchor_sets.get(eid, set())
        entity_confs = data.anchor_confidences.get(eid)

        obs = compute_observability_score(
            data.positions.get(eid, {}),
            ea_set,
            data.provenances.get(eid, "traced"),
            data.anchor_categories,
        )

        primary_coherence = _compute_primary_coherence(
            ea_set,
            query.prefer_anchors,
            query.prefer_weights,
            idf_cache,
            data.anchor_categories,
            entity_confs,
        )

        support = _compute_support_bonus(
            ea_set,
            query.prefer_anchors,
            query.prefer_weights,
            idf_cache,
            data.anchor_categories,
        )

        expansion = all_ids[eid]

        # For premise retrieval, graph path support is the primary signal.
        # Bank alignment and lens coherence refine within the graph-
        # reachable set, but a premise reached via depends_on from a
        # high-scoring landmark IS relevant regardless of anchor overlap.
        semantic_bonus = (1.0 + primary_coherence) * support
        final = expansion * (0.3 + 0.7 * b_score * obs * semantic_bonus)
        if final > 0:
            results.append(ScoredEntity(
                entity_id=eid,
                name=data.names.get(eid, ""),
                final_score=final,
                bank_score=b_score,
                anchor_score=primary_coherence,
                seed_score=expansion,
            ))

    results.sort(key=lambda e: e.final_score, reverse=True)
    final_results = results[:limit]

    trace = RetrievalTrace(final_count=len(final_results))
    return final_results, trace


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def landmark_expand_retrieve(
    query: StructuredQuery,
    conn: sqlite3.Connection,
    limit: int = 16,
    config: dict | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Three-stage retrieval: landmark → expand → rerank.

    Drop-in replacement for navigate() in v3 runtime.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg["rerank_limit"] = max(limit, cfg.get("rerank_limit", 128))

    landmarks, trace1 = retrieve_landmarks(query, conn, cfg)
    expanded, trace2 = expand_from_landmarks(landmarks, conn, cfg)
    results, _ = rerank_candidates(query, landmarks, expanded, conn, cfg)

    # Merge traces
    trace = RetrievalTrace(
        landmark_count=trace1.landmark_count,
        expanded_pool_size=trace2.expanded_pool_size,
        final_count=len(results[:limit]),
        landmarks_by_provenance=trace1.landmarks_by_provenance,
        expanded_by_relation=trace2.expanded_by_relation,
    )

    return results[:limit], trace
