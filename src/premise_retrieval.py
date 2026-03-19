"""Three-stage premise retrieval: landmark -> expand -> rerank.

v3 retrieval architecture. Does NOT replace flat navigate() -- lives
alongside it as an alternative strategy for v3_runtime.

Stage 1: retrieve_landmarks -- find high-confidence traced entities
         using primary lens scoring (proof, constant, semantic, structural).
Stage 2: expand_from_landmarks -- walk typed graph links to reach
         premise-only entities within N hops.
Stage 3: rerank_candidates -- score expanded pool by multi-lens coherence
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

from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_network import _get_data_cache, _get_idf_cache
from src.proof_scoring import (
    _vectorized_bank_scores,
    compute_observability_score,
)

# Re-exports: backward compatibility for all symbols that moved
# to retrieval_scoring and retrieval_stages.
from src.retrieval_scoring import (  # noqa: F401
    PRIMARY_LENSES,
    SUPPORT_LENSES,
    _compute_bridge_potential,
    _compute_hub_penalty,
    _compute_primary_coherence,
    _compute_primary_seed_score,
    _compute_support_bonus,
)
from src.retrieval_stages import (  # noqa: F401
    DEFAULT_CONFIG,
    ExpandedCandidate,
    RetrievalTrace,
    SupportVote,
    _get_neighborhood_signatures,
    expand_from_landmarks,
    resolve_landmarks,
    retrieve_landmarks,
)

# ---------------------------------------------------------------------------
# Stage 3: Reranking
# ---------------------------------------------------------------------------


def _compute_consensus(supports: list[SupportVote]) -> float:
    """Weighted consensus: 1 - prod(1 - vote_i), with fanout normalization.

    Each vote is: landmark_score * relation_weight / sqrt(source_fanout).
    Multiple independent high-quality landmarks compound via noisy-OR.
    """
    if not supports:
        return 0.0

    product = 1.0
    for vote in supports:
        raw = vote.landmark_score * vote.relation_weight / math.sqrt(vote.source_fanout)
        product *= 1.0 - min(raw, 0.95)

    return 1.0 - product


def _compute_diversity_bonus(supports: list[SupportVote]) -> float:
    """Reward support from multiple distinct landmarks."""
    distinct = len({s.landmark_id for s in supports})
    if distinct <= 1:
        return 1.0
    # Diminishing returns: sqrt scaling
    return math.sqrt(distinct)


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

    # Only rerank expanded candidates -- landmarks are seed theorems, not premises
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
            # bank_score excluded from stage 3 -- it's a stage-1 signal.
            local_tiebreak = obs * (1.0 + primary_coherence) * support_bonus
            # Quantize graph_support into bands (0.01 resolution)
            # so that semantics cannot invert consensus ordering
            # across band boundaries.
            band = round(graph_support, 2)
            final = band + 0.009 * min(local_tiebreak, 1.0)

        if final > 0:
            results.append(
                ScoredEntity(
                    entity_id=eid,
                    name=data.names.get(eid, ""),
                    final_score=final,
                    bank_score=b_score,
                    anchor_score=consensus,
                    seed_score=graph_support,
                )
            )

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
    """Three-stage retrieval: landmark -> expand -> rerank.

    Drop-in replacement for navigate() in v3 runtime.
    When lens_guidance=True in config, adds lens-specialist
    committee resolution after reranking.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg["rerank_limit"] = max(limit, cfg.get("rerank_limit", 128))

    strategy = cfg.get("landmark_strategy", "bridge_potential")
    if strategy == "multi_pass":
        landmarks, trace1 = resolve_landmarks(query, conn, cfg)
    else:
        landmarks, trace1 = retrieve_landmarks(query, conn, cfg)
    expanded, trace2 = expand_from_landmarks(landmarks, conn, cfg)
    results, _ = rerank_candidates(query, landmarks, expanded, conn, cfg)

    # Lens guidance: run specialist committee on reranked candidates
    if cfg.get("lens_guidance", False) and results:
        from src.coherence_engine import (
            build_resolution_decision,
            run_lens_committee,
        )
        from src.lens_guidance import build_guidance_packet
        from src.proof_network import _get_data_cache, _get_idf_cache
        from src.retrieval_stages import _get_neighborhood_signatures

        data = _get_data_cache(conn)
        idf_cache = _get_idf_cache(conn)
        nbr_sigs = _get_neighborhood_signatures(conn, data)

        from src.landmark_freeze import (
            FrozenLandmarkState,
            LandmarkResidualReport,
        )

        frozen = trace1.frozen_state or FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = trace1.residual or LandmarkResidualReport()

        packet = build_guidance_packet(
            query, landmarks, results, frozen, residual,
            expanded, data, nbr_sigs, idf_cache,
        )

        committee = run_lens_committee(packet)
        decision = build_resolution_decision(packet, committee)
        results = decision.premises

    # Merge traces
    trace = RetrievalTrace(
        landmark_count=trace1.landmark_count,
        expanded_pool_size=trace2.expanded_pool_size,
        final_count=len(results[:limit]),
        landmarks_by_provenance=trace1.landmarks_by_provenance,
        expanded_by_relation=trace2.expanded_by_relation,
    )

    return results[:limit], trace
