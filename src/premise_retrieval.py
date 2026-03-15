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
class SupportVote:
    """One supporting path from a landmark to a candidate."""

    landmark_id: int
    landmark_score: float
    relation: str
    relation_weight: float
    path_score: float
    source_fanout: int  # outgoing edges from this landmark


@dataclass
class ExpandedCandidate:
    """A candidate reached via graph expansion from landmarks."""

    entity_id: int
    name: str
    supports: list[SupportVote] = field(default_factory=list)

    @property
    def best_relation(self) -> str:
        if not self.supports:
            return ""
        return max(self.supports, key=lambda s: s.path_score).relation

    @property
    def hops(self) -> int:
        return 1  # currently 1-hop expansion

    @property
    def distinct_landmarks(self) -> int:
        return len({s.landmark_id for s in self.supports})


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

    # Compute source fanout per landmark (number of outgoing edges)
    source_fanout: dict[int, int] = {}
    for lm_id in landmark_ids:
        source_fanout[lm_id] = len(adj.get(lm_id, []))

    # Collect ALL supporting paths per candidate (not just best)
    candidate_supports: dict[int, list[SupportVote]] = defaultdict(list)

    for lm in landmarks:
        neighbors = adj.get(lm.entity_id, [])
        neighbors.sort(key=lambda x: x[2], reverse=True)
        fanout = source_fanout.get(lm.entity_id, 1)
        for tgt, rel, wt in neighbors[:topk_per_seed]:
            if tgt in landmark_ids:
                continue
            vote = SupportVote(
                landmark_id=lm.entity_id,
                landmark_score=lm.final_score,
                relation=rel,
                relation_weight=wt,
                path_score=lm.final_score * wt,
                source_fanout=max(fanout, 1),
            )
            candidate_supports[tgt].append(vote)

    # Multi-hop: expand from hop-1 candidates
    if max_hops >= 2:
        hop1_ids = set(candidate_supports.keys())
        for h1_id in hop1_ids:
            best_vote = max(candidate_supports[h1_id], key=lambda v: v.path_score)
            neighbors = adj.get(h1_id, [])
            neighbors.sort(key=lambda x: x[2], reverse=True)
            h1_fanout = len(neighbors)
            for tgt, rel, wt in neighbors[:topk_per_seed]:
                if tgt in landmark_ids or tgt in hop1_ids:
                    continue
                vote = SupportVote(
                    landmark_id=best_vote.landmark_id,
                    landmark_score=best_vote.landmark_score,
                    relation=rel,
                    relation_weight=wt,
                    path_score=best_vote.path_score * wt,
                    source_fanout=max(h1_fanout, 1),
                )
                candidate_supports[tgt].append(vote)

    # Build ExpandedCandidate objects
    visited: dict[int, ExpandedCandidate] = {}
    for eid, supports in candidate_supports.items():
        visited[eid] = ExpandedCandidate(
            entity_id=eid,
            name=data.names.get(eid, ""),
            supports=supports,
        )

    expanded = sorted(
        visited.values(),
        key=lambda c: max(s.path_score for s in c.supports),
        reverse=True,
    )

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


def _compute_consensus(supports: list[SupportVote]) -> float:
    """Weighted consensus: 1 - prod(1 - vote_i), with fanout normalization.

    Each vote is: landmark_score * relation_weight / sqrt(source_fanout).
    Multiple independent high-quality landmarks compound via noisy-OR.
    """
    if not supports:
        return 0.0

    product = 1.0
    for vote in supports:
        raw = (
            vote.landmark_score
            * vote.relation_weight
            / math.sqrt(vote.source_fanout)
        )
        product *= 1.0 - min(raw, 0.95)

    return 1.0 - product


def _compute_diversity_bonus(supports: list[SupportVote]) -> float:
    """Reward support from multiple distinct landmarks."""
    distinct = len({s.landmark_id for s in supports})
    if distinct <= 1:
        return 1.0
    # Diminishing returns: sqrt scaling
    return math.sqrt(distinct)


def _compute_hub_penalty(
    entity_id: int,
    hub_degrees: dict[int, int],
    median_degree: float,
) -> float:
    """Downweight globally over-connected candidates (hubs).

    Entities with far more incoming links than the median are less
    likely to be specifically relevant — they're infrastructure lemmas.
    """
    degree = hub_degrees.get(entity_id, 0)
    if degree <= median_degree:
        return 1.0
    # Gentle penalty: 1 / sqrt(degree / median)
    return 1.0 / math.sqrt(degree / max(median_degree, 1.0))


def rerank_candidates(
    query: StructuredQuery,
    landmarks: list[ScoredEntity],
    expanded: list[ExpandedCandidate],
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Stage 3: rerank by weighted consensus + semantic refinement.

    graph_support = consensus * diversity_bonus * hub_penalty
    semantic_refinement = bank_score * obs * (1 + primary_coherence) * support_bonus
    final = graph_support * (0.2 + 0.8 * semantic_refinement)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    limit = cfg["rerank_limit"]

    data = _get_data_cache(conn)
    idf_cache = _get_idf_cache(conn)

    # Hub penalty: precompute in-degree per candidate (cheap aggregation)
    allowed_relations = set(cfg["expansion_relations"])
    candidate_ids_set = {ec.entity_id for ec in expanded}
    candidate_ids_set.update(lm.entity_id for lm in landmarks)

    hub_degrees: dict[int, int] = {}
    if candidate_ids_set and allowed_relations:
        placeholders_r = ",".join("?" * len(allowed_relations))
        placeholders_e = ",".join("?" * len(candidate_ids_set))
        rows = conn.execute(
            f"SELECT target_id, COUNT(*) FROM entity_links "  # nosec B608
            f"WHERE relation IN ({placeholders_r}) "
            f"AND target_id IN ({placeholders_e}) "
            f"GROUP BY target_id",
            list(allowed_relations) + list(candidate_ids_set),
        ).fetchall()
        hub_degrees = dict(rows)

    degrees = sorted(hub_degrees.values()) if hub_degrees else [1]
    median_degree = degrees[len(degrees) // 2] if degrees else 1.0

    # Only rerank expanded candidates — landmarks are seed theorems, not premises
    landmark_ids = {lm.entity_id for lm in landmarks}
    all_candidates: dict[int, list[SupportVote]] = {}
    for ec in expanded:
        if ec.entity_id not in landmark_ids:
            all_candidates[ec.entity_id] = list(ec.supports)

    candidate_ids = list(all_candidates.keys())
    if not candidate_ids:
        return [], RetrievalTrace()

    # Reranking strategy (configurable for ablations)
    rerank_mode = cfg.get("rerank_mode", "hierarchical")

    # Vectorized bank scores (needed for some modes)
    bank_scores_arr = _vectorized_bank_scores(candidate_ids, data.positions, query)

    results: list[ScoredEntity] = []
    for i, eid in enumerate(candidate_ids):
        supports = all_candidates[eid]

        # Graph support: consensus * diversity * hub penalty
        consensus = _compute_consensus(supports)
        diversity = _compute_diversity_bonus(supports)
        hub_pen = _compute_hub_penalty(eid, hub_degrees, median_degree)
        graph_support = consensus * diversity * hub_pen

        if graph_support <= 0:
            continue

        # Semantic components (computed for all modes, used selectively)
        b_score = float(bank_scores_arr[i])
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

        support_bonus = _compute_support_bonus(
            ea_set,
            query.prefer_anchors,
            query.prefer_weights,
            idf_cache,
            data.anchor_categories,
        )

        # --- Rerank modes (for ablation) ---
        if rerank_mode == "consensus_only":
            final = graph_support

        elif rerank_mode == "consensus_obs":
            final = graph_support * (0.5 + 0.5 * obs)

        elif rerank_mode == "consensus_no_bank":
            # Local semantic without bank_score
            local_sem = obs * (1.0 + primary_coherence) * support_bonus
            final = graph_support * (0.5 + 0.5 * local_sem)

        elif rerank_mode == "consensus_full_semantic":
            semantic = b_score * obs * (1.0 + primary_coherence) * support_bonus
            final = graph_support * (0.2 + 0.8 * semantic)

        else:  # "hierarchical" (default)
            # Graph consensus is monotone-primary.
            # Semantics refines only within consensus bands.
            # bank_score excluded from stage 3 — it's a stage-1 signal.
            local_tiebreak = obs * (1.0 + primary_coherence) * support_bonus
            # Quantize graph_support into bands (0.01 resolution)
            # so that semantics cannot invert consensus ordering
            # across band boundaries.
            band = round(graph_support, 2)
            final = band + 0.009 * min(local_tiebreak, 1.0)

        if final > 0:
            results.append(ScoredEntity(
                entity_id=eid,
                name=data.names.get(eid, ""),
                final_score=final,
                bank_score=b_score,
                anchor_score=consensus,
                seed_score=graph_support,
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
