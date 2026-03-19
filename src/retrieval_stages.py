"""Retrieval pipeline stages 1 and 2: landmark selection + graph expansion.

Stage 1: retrieve_landmarks / resolve_landmarks — find high-confidence
         traced bridge theorems via scoring or multi-pass selector fusion.
Stage 2: expand_from_landmarks — walk typed graph links to reach
         premise-only entities within N hops.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from src.landmark_selectors import (
    ALL_SELECTORS,
    SelectorVote,
    build_hypotheses,
)
from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_network import _DataCache, _get_data_cache, _get_idf_cache
from src.proof_scoring import (
    _vectorized_bank_scores,
    compute_observability_score,
)
from src.retrieval_scoring import _compute_bridge_potential

# Default config (shared across all stages)
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

    # Selector/convergence diagnostics (populated by resolve_landmarks)
    selector_vote_counts: dict[str, int] = field(default_factory=dict)
    convergence_distribution: dict[str, int] = field(default_factory=dict)
    hub_suppressed_count: int = 0

    # Freeze/residual diagnostics (populated by resolve_landmarks with freezing)
    committed_count: int = 0
    ambiguous_pool_size: int = 0
    resolved_from_ambiguous: int = 0
    residual_entropy: float = 0.0
    query_coverage: float = 0.0
    phase: str = ""
    resolve_passes: int = 1

    # Internal state for lens guidance (set by resolve_landmarks)
    frozen_state: object = None  # FrozenLandmarkState when available
    residual: object = None  # LandmarkResidualReport when available


# ---------------------------------------------------------------------------
# Neighborhood cache
# ---------------------------------------------------------------------------

# Module-level cache for neighborhood signatures (cleared with proof_network caches)
_neighborhood_cache: dict[int, dict[int, set[int]]] = {}


def _get_neighborhood_signatures(
    conn: sqlite3.Connection,
    data: _DataCache,
) -> dict[int, set[int]]:
    """Precompute dependency-neighborhood anchor signatures for traced theorems.

    For each traced theorem, collect the union of anchor IDs from its
    depends_on neighbors. This is the "what can this landmark provide?"
    signature -- matching query anchors against this tells us bridge potential.

    Cached per connection.
    """
    conn_id = id(conn)
    if conn_id in _neighborhood_cache:
        return _neighborhood_cache[conn_id]

    # Get all depends_on edges
    dep_rows = conn.execute(
        "SELECT source_id, target_id FROM entity_links WHERE relation = 'depends_on'"
    ).fetchall()

    # Build source -> set of neighbor anchor IDs
    nbr_anchors: dict[int, set[int]] = defaultdict(set)
    for src, tgt in dep_rows:
        tgt_anchors = data.anchor_sets.get(tgt, set())
        nbr_anchors[src].update(tgt_anchors)

    result = dict(nbr_anchors)
    _neighborhood_cache[conn_id] = result
    return result


# ---------------------------------------------------------------------------
# Stage 1: Landmark retrieval
# ---------------------------------------------------------------------------


def retrieve_landmarks(
    query: StructuredQuery,
    conn: sqlite3.Connection,
    config: dict | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Stage 1: retrieve bridge theorems whose neighborhoods contain useful premises.

    Landmarks are NOT premises. They are traced theorems whose depends_on
    frontier is likely to contain the query's needed premises. Scored by
    bridge potential (query-to-neighborhood match), not self-match.

    Hard filters:
        - entity_type = 'lemma', provenance = 'traced'
        - nonzero depends_on outdegree (structurally useful bridge)

    Scoring:
        landmark_score = bank_score * bridge_potential * observability

    Diversity: greedy selection with neighborhood overlap penalty to avoid
    redundant landmarks from the same dependency basin.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    limit = cfg["landmark_limit"]

    data = _get_data_cache(conn)
    idf_cache = _get_idf_cache(conn)

    # Precompute neighborhood signatures
    nbr_sigs = _get_neighborhood_signatures(conn, data)

    # Hard-filter: traced lemmas with nonzero depends_on outdegree
    candidate_ids = [
        eid for eid, prov in data.provenances.items() if prov == "traced" and eid in nbr_sigs
    ]

    if not candidate_ids:
        return [], RetrievalTrace()

    # Bank scores (vectorized)
    bank_scores = _vectorized_bank_scores(candidate_ids, data.positions, query)

    # Score all candidates by bridge potential
    scored: list[tuple[float, int, float, float]] = []  # (score, eid, bridge, bank)
    for i, eid in enumerate(candidate_ids):
        b_score = float(bank_scores[i])
        if b_score <= 0:
            continue

        nbr_sig = nbr_sigs.get(eid, set())
        bridge = _compute_bridge_potential(
            nbr_sig, query.prefer_anchors, query.prefer_weights, idf_cache
        )
        if bridge <= 0:
            continue

        obs = compute_observability_score(
            data.positions.get(eid, {}),
            data.anchor_sets.get(eid, set()),
            "traced",
            data.anchor_categories,
        )

        final = b_score * bridge * obs
        scored.append((final, eid, bridge, b_score))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedy diversity selection: penalize landmarks with highly
    # overlapping depends_on neighborhoods (same dependency basin)
    selected: list[ScoredEntity] = []
    selected_nbrs: set[int] = set()

    for final, eid, bridge, b_score in scored:
        if len(selected) >= limit:
            break

        nbr_sig = nbr_sigs.get(eid, set())
        if selected_nbrs:
            overlap = len(nbr_sig & selected_nbrs) / max(len(nbr_sig), 1)
            if overlap > 0.8:
                continue  # too similar to already-selected landmarks

        selected.append(
            ScoredEntity(
                entity_id=eid,
                name=data.names.get(eid, ""),
                final_score=final,
                bank_score=b_score,
                anchor_score=bridge,
                seed_score=1.0,
            )
        )
        selected_nbrs.update(nbr_sig)

    # Trace
    trace = RetrievalTrace(landmark_count=len(selected))
    prov_counts: dict[str, int] = defaultdict(int)
    for lm in selected:
        prov_counts[data.provenances.get(lm.entity_id, "?")] += 1
    trace.landmarks_by_provenance = dict(prov_counts)

    return selected, trace


# ---------------------------------------------------------------------------
# Stage 1b: Multi-pass landmark resolution
# ---------------------------------------------------------------------------


def resolve_landmarks(
    query: StructuredQuery,
    conn: sqlite3.Connection,
    config: dict | None = None,
    template_hint: str | None = None,
) -> tuple[list[ScoredEntity], RetrievalTrace]:
    """Multi-pass landmark resolution with iterative freeze-resolve.

    Pipeline:
        1. Selector fusion: run 4 selectors, build LandmarkHypotheses
        2. Iterative freeze-resolve (up to 3 passes):
           a. Freeze committed state (conservative: converged + margin-tested isolated)
           b. Compute category-weighted residual
           c. Resolve ambiguous by category-weighted coverage + conflict bifurcation
           d. Re-freeze newly validated, recompute residual
           e. Stop on entropy plateau or coverage stall
        3. Greedy diversity selection from committed + resolved pool
    """
    from src.landmark_freeze import (
        compute_residual,
        extend_frozen_state,
        freeze_committed_state,
        resolve_ambiguous_landmarks,
    )

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    limit = cfg["landmark_limit"]

    data = _get_data_cache(conn)
    idf_cache = _get_idf_cache(conn)
    nbr_sigs = _get_neighborhood_signatures(conn, data)

    # Hard-filter: traced lemmas with nonzero depends_on outdegree
    candidate_ids = [
        eid for eid, prov in data.provenances.items() if prov == "traced" and eid in nbr_sigs
    ]

    if not candidate_ids:
        return [], RetrievalTrace()

    # --- Pass A: Selector fusion ---
    all_votes: dict[int, list[SelectorVote]] = defaultdict(list)
    for selector in ALL_SELECTORS:
        votes = selector(candidate_ids, query, data, idf_cache, conn, nbr_sigs)
        for eid, vote in votes.items():
            all_votes[eid].append(vote)

    hypotheses = build_hypotheses(candidate_ids, dict(all_votes), nbr_sigs, oppose_lambda=0.5)

    if not hypotheses:
        return [], RetrievalTrace()

    # --- Iterative freeze-resolve ---
    frozen, ambiguous = freeze_committed_state(hypotheses, data, nbr_sigs)

    max_passes = 3
    prev_entropy = float("inf")
    prev_coverage = 0.0
    residual = None  # set on first pass, always runs since max_passes >= 1
    all_resolved_eids: set[int] = set()
    all_resolved: list[tuple[float, int]] = []
    resolve_passes = 0

    for pass_idx in range(max_passes):
        residual = compute_residual(
            frozen, query.prefer_anchors, query.prefer_weights,
            data, idf_cache, ambiguous, nbr_sigs,
        )

        # Convergence check (after first pass)
        if pass_idx > 0:
            entropy_decrease = prev_entropy - residual.residual_entropy
            coverage_gain = residual.query_coverage - prev_coverage
            if entropy_decrease < 0.05 and coverage_gain < 0.01:
                break

        prev_entropy = residual.residual_entropy
        prev_coverage = residual.query_coverage

        resolved = resolve_ambiguous_landmarks(
            frozen, residual, ambiguous, data, nbr_sigs, idf_cache,
        )
        resolve_passes += 1

        if not resolved:
            break

        # Take top third as newly validated
        n_promote = max(len(resolved) // 3, 1)
        newly_validated = resolved[:n_promote]
        remaining = resolved[n_promote:]

        for hyp in newly_validated:
            score = (hyp.support_rrf - 0.5 * hyp.oppose_rrf) * 0.95
            all_resolved.append((score, hyp.entity_id))
            all_resolved_eids.add(hyp.entity_id)

        # Re-freeze with newly validated
        frozen = extend_frozen_state(frozen, newly_validated, data, nbr_sigs)

        # Update ambiguous pool: remove promoted, keep remaining + unpromoted
        promoted_ids = {h.entity_id for h in newly_validated}
        ambiguous = [
            h for h in ambiguous if h.entity_id not in promoted_ids
        ]

        # On last pass, add all remaining resolved
        if pass_idx == max_passes - 1:
            for hyp in remaining:
                if hyp.entity_id not in all_resolved_eids:
                    score = (hyp.support_rrf - 0.5 * hyp.oppose_rrf) * 0.95
                    all_resolved.append((score, hyp.entity_id))

    # --- Merge: committed first, then resolved ---
    committed_set = frozen.committed_ids
    merged: list[tuple[float, int]] = []
    for hyp in hypotheses:
        if hyp.entity_id in committed_set:
            score = hyp.support_rrf - 0.5 * hyp.oppose_rrf
            merged.append((score, hyp.entity_id))

    merged.extend(all_resolved)
    merged.sort(key=lambda x: x[0], reverse=True)

    # --- Greedy diversity selection ---
    selected: list[ScoredEntity] = []
    selected_nbrs: set[int] = set()

    for fused_score, eid in merged:
        if len(selected) >= limit:
            break

        nbr_sig = nbr_sigs.get(eid, set())
        if selected_nbrs and nbr_sig:
            overlap = len(nbr_sig & selected_nbrs) / max(len(nbr_sig), 1)
            if overlap > 0.8:
                continue

        selected.append(
            ScoredEntity(
                entity_id=eid,
                name=data.names.get(eid, ""),
                final_score=max(fused_score, 0.0),
                bank_score=fused_score,
                anchor_score=0.0,
                seed_score=1.0,
            )
        )
        selected_nbrs.update(nbr_sig)

    # --- Trace ---
    trace = RetrievalTrace(landmark_count=len(selected))

    prov_counts: dict[str, int] = defaultdict(int)
    for lm in selected:
        prov_counts[data.provenances.get(lm.entity_id, "?")] += 1
    trace.landmarks_by_provenance = dict(prov_counts)

    vote_counts: dict[str, int] = defaultdict(int)
    for votes_list in all_votes.values():
        for v in votes_list:
            key = f"{v.selector_name}:{'+1' if v.vote == 1 else ('-1' if v.vote == -1 else '0')}"
            vote_counts[key] += 1
    trace.selector_vote_counts = dict(vote_counts)

    conv_dist: dict[str, int] = defaultdict(int)
    for hyp in hypotheses:
        conv_dist[hyp.convergence_class] += 1
    trace.convergence_distribution = dict(conv_dist)

    trace.hub_suppressed_count = sum(
        1 for hyp in hypotheses
        if any(v.selector_name == "hub_suppressor" and v.vote == -1 for v in hyp.votes)
    )

    trace.committed_count = len(committed_set)
    trace.ambiguous_pool_size = len(ambiguous)
    trace.resolved_from_ambiguous = len(all_resolved)
    if residual is not None:
        trace.residual_entropy = residual.residual_entropy
        trace.query_coverage = residual.query_coverage
        trace.phase = residual.phase_signal
        trace.residual = residual
    trace.resolve_passes = resolve_passes
    trace.frozen_state = frozen

    return selected, trace


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

    # Build adjacency: source -> [(target, relation, weight)]
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
