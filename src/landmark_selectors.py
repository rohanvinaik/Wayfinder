"""Multi-pass landmark evidence mesh: ternary selectors, signed RRF fusion, convergence.

4 selectors score landmarks as bridge theorems using ternary semantics:
    +1 = explicit support
     0 = abstain/orthogonal
    -1 = explicit anti-evidence (hubs, inaccessibility)

Signed fusion: support_rrf - lambda * oppose_rrf, with convergence classification.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from src.nav_contracts import StructuredQuery
from src.proof_network import _DataCache, get_accessible_premises

# Standard RRF constant
RRF_K = 60


@dataclass
class SelectorVote:
    """One selector's ternary vote on a landmark candidate."""

    vote: int  # +1, 0, -1
    confidence: float  # [0, 1]
    raw_score: float  # selector's raw numeric output
    selector_name: str
    family: str  # "bridge", "match", "suppression", "context"
    evidence: str  # human-readable
    rank: int  # selector-internal rank (0 = not ranked)


@dataclass
class LandmarkHypothesis:
    """A landmark candidate with fused multi-selector evidence."""

    entity_id: int
    votes: list[SelectorVote] = field(default_factory=list)
    support_rrf: float = 0.0
    oppose_rrf: float = 0.0
    convergence_class: str = "isolated"  # "converged" | "isolated" | "conflicted"
    novelty_proxy: float = 1.0  # 1 - overlap with already-selected neighborhoods
    neighborhood_size: int = 0  # depends_on outdegree


# ---------------------------------------------------------------------------
# Cached neighbor entity IDs (for accessibility selector)
# ---------------------------------------------------------------------------

_nbr_entity_cache: dict[int, dict[int, set[int]]] = {}


def _get_neighbor_entity_ids(conn: sqlite3.Connection) -> dict[int, set[int]]:
    """Pre-load depends_on neighbor entity IDs (cached per connection)."""
    conn_id = id(conn)
    if conn_id in _nbr_entity_cache:
        return _nbr_entity_cache[conn_id]

    dep_rows = conn.execute(
        "SELECT source_id, target_id FROM entity_links WHERE relation = 'depends_on'"
    ).fetchall()

    nbr: dict[int, set[int]] = defaultdict(set)
    for src, tgt in dep_rows:
        nbr[src].add(tgt)

    result = dict(nbr)
    _nbr_entity_cache[conn_id] = result
    return result


def clear_selector_caches() -> None:
    """Clear module-level caches. Call between test cases or DB swaps."""
    _nbr_entity_cache.clear()


# ---------------------------------------------------------------------------
# Selector 1: bridge_potential (family: "bridge")
# ---------------------------------------------------------------------------


def select_by_bridge_potential(
    candidate_ids: list[int],
    query: StructuredQuery,
    data: _DataCache,
    idf_cache: dict[int, float],
    conn: sqlite3.Connection,
    nbr_sigs: dict[int, set[int]],
    **kwargs,
) -> dict[int, SelectorVote]:
    """Neighborhood-based bridge scoring.

    Score: IDF-weighted overlap between query anchors and landmark's
    depends_on neighborhood anchors.
    Vote: score > 0 -> +1, score == 0 -> 0. Never -1.
    """
    if not query.prefer_anchors:
        return {}

    scores: list[tuple[int, float]] = []
    for eid in candidate_ids:
        nbr_sig = nbr_sigs.get(eid, set())
        if not nbr_sig:
            continue

        total_idf = 0.0
        matched_idf = 0.0
        for aid, weight in zip(query.prefer_anchors, query.prefer_weights):
            idf = idf_cache.get(aid, 1.0) * weight
            total_idf += idf
            if aid in nbr_sig:
                matched_idf += idf

        score = matched_idf / total_idf if total_idf > 0 else 0.0
        if score > 0:
            scores.append((eid, score))

    if not scores:
        return {}

    scores.sort(key=lambda x: x[1], reverse=True)
    max_score = scores[0][1]

    result: dict[int, SelectorVote] = {}
    for rank_idx, (eid, score) in enumerate(scores):
        result[eid] = SelectorVote(
            vote=1,
            confidence=min(score / max(max_score, 1e-9), 1.0),
            raw_score=score,
            selector_name="bridge_potential",
            family="bridge",
            evidence=f"bridge={score:.3f}",
            rank=rank_idx + 1,
        )

    return result


# ---------------------------------------------------------------------------
# Selector 2: self_match (family: "match")
# ---------------------------------------------------------------------------


def select_by_self_match(
    candidate_ids: list[int],
    query: StructuredQuery,
    data: _DataCache,
    idf_cache: dict[int, float],
    conn: sqlite3.Connection,
    nbr_sigs: dict[int, set[int]],
    **kwargs,
) -> dict[int, SelectorVote]:
    """Cross-check: primary lens overlap between query and landmark's own anchors.

    Vote: top quartile -> +1, rest -> 0. Never -1.
    Confidence capped at 0.6 (cross-check, not primary signal).
    """
    from src.retrieval_scoring import _compute_primary_seed_score

    if not query.prefer_anchors:
        return {}

    scores: list[tuple[int, float]] = []
    for eid in candidate_ids:
        ea = data.anchor_sets.get(eid, set())
        if not ea:
            continue
        confs = data.anchor_confidences.get(eid)
        score = _compute_primary_seed_score(
            ea,
            query.prefer_anchors,
            query.prefer_weights,
            idf_cache,
            data.anchor_categories,
            confs,
        )
        if score > 0:
            scores.append((eid, score))

    if not scores:
        return {}

    scores.sort(key=lambda x: x[1], reverse=True)
    q1_cutoff = max(len(scores) // 4, 1)

    result: dict[int, SelectorVote] = {}
    for rank_idx, (eid, score) in enumerate(scores):
        vote = 1 if rank_idx < q1_cutoff else 0
        result[eid] = SelectorVote(
            vote=vote,
            confidence=min(score, 0.6),
            raw_score=score,
            selector_name="self_match",
            family="match",
            evidence=f"self_match={score:.3f}",
            rank=rank_idx + 1,
        )

    return result


# ---------------------------------------------------------------------------
# Selector 3: accessibility (family: "context")
# ---------------------------------------------------------------------------


def select_by_accessibility(
    candidate_ids: list[int],
    query: StructuredQuery,
    data: _DataCache,
    idf_cache: dict[int, float],
    conn: sqlite3.Connection,
    nbr_sigs: dict[int, set[int]],
    **kwargs,
) -> dict[int, SelectorVote]:
    """Neighborhood-based: check overlap with accessible premise set.

    Vote: neighborhood overlaps accessible set -> +1,
          no accessible_theorem_id -> 0 (abstain),
          accessible_theorem_id set but zero overlap -> 0 (abstain).
    """
    if query.accessible_theorem_id is None:
        return {}

    accessible = get_accessible_premises(conn, query.accessible_theorem_id)
    if not accessible:
        return {}

    nbr_entity_ids = _get_neighbor_entity_ids(conn)

    scores: list[tuple[int, float, int]] = []
    for eid in candidate_ids:
        neighbors = nbr_entity_ids.get(eid, set())
        if not neighbors:
            continue
        overlap = neighbors & accessible
        if overlap:
            score = len(overlap) / len(neighbors)
            scores.append((eid, score, len(overlap)))

    if not scores:
        return {}

    scores.sort(key=lambda x: x[1], reverse=True)

    result: dict[int, SelectorVote] = {}
    for rank_idx, (eid, score, overlap_count) in enumerate(scores):
        result[eid] = SelectorVote(
            vote=1,
            confidence=min(score, 1.0),
            raw_score=score,
            selector_name="accessibility",
            family="context",
            evidence=f"accessible_overlap={overlap_count}",
            rank=rank_idx + 1,
        )

    return result


# ---------------------------------------------------------------------------
# Selector 4: hub_suppressor (family: "suppression")
# ---------------------------------------------------------------------------


def select_by_hub_suppressor(
    candidate_ids: list[int],
    query: StructuredQuery,
    data: _DataCache,
    idf_cache: dict[int, float],
    conn: sqlite3.Connection,
    nbr_sigs: dict[int, set[int]],
    **kwargs,
) -> dict[int, SelectorVote]:
    """True negative selector: suppresses hubs. Never votes +1.

    Vote: in-degree > 10x median -> -1 (conf 0.9),
          in-degree > 5x median -> -1 (conf 0.5),
          else -> 0 (abstain).
    """
    hub_degrees = data.hub_in_degrees
    if not hub_degrees:
        return {}

    degrees = sorted(hub_degrees.values())
    if not degrees:
        return {}
    median_deg = degrees[len(degrees) // 2]
    if median_deg == 0:
        median_deg = 1

    result: dict[int, SelectorVote] = {}
    for eid in candidate_ids:
        deg = hub_degrees.get(eid, 0)
        if deg > 10 * median_deg:
            result[eid] = SelectorVote(
                vote=-1,
                confidence=0.9,
                raw_score=float(deg),
                selector_name="hub_suppressor",
                family="suppression",
                evidence=f"in_degree={deg}>10x_median={10 * median_deg}",
                rank=0,
            )
        elif deg > 5 * median_deg:
            result[eid] = SelectorVote(
                vote=-1,
                confidence=0.5,
                raw_score=float(deg),
                selector_name="hub_suppressor",
                family="suppression",
                evidence=f"in_degree={deg}>5x_median={5 * median_deg}",
                rank=0,
            )

    return result


# All selectors in execution order
ALL_SELECTORS = [
    select_by_bridge_potential,
    select_by_self_match,
    select_by_accessibility,
    select_by_hub_suppressor,
]


# ---------------------------------------------------------------------------
# Signed RRF fusion + convergence classification
# ---------------------------------------------------------------------------


def compute_rrf(votes: list[SelectorVote], vote_sign: int) -> float:
    """Compute confidence-weighted reciprocal rank fusion for votes with a given sign.

    Each vote contributes confidence / (k + rank). This ensures high-confidence
    votes dominate over low-confidence ones at the same rank, and distinguishes
    hub_suppressor's 0.5 vs 0.9 confidence levels.
    """
    total = 0.0
    for v in votes:
        if v.vote == vote_sign:
            rank = v.rank if v.rank > 0 else 1
            total += v.confidence / (RRF_K + rank)
    return total


def classify_convergence(
    votes: list[SelectorVote],
    support_rrf: float,
    oppose_rrf: float,
) -> str:
    """Classify convergence: converged | isolated | conflicted."""
    support_families = {v.family for v in votes if v.vote == 1}
    has_opposition = oppose_rrf > 0

    # Converged: >= 2 different families agree, low opposition
    if len(support_families) >= 2:
        if not has_opposition or oppose_rrf < 0.3 * support_rrf:
            return "converged"

    # Conflicted: significant opposition relative to support
    if has_opposition and support_rrf > 0 and oppose_rrf > 0.5 * support_rrf:
        return "conflicted"

    # Isolated: default for single-family support or weak support
    return "isolated"


def build_hypotheses(
    candidate_ids: list[int],
    all_votes: dict[int, list[SelectorVote]],
    nbr_sigs: dict[int, set[int]],
    selected_nbrs: set[int] | None = None,
    oppose_lambda: float = 0.5,
) -> list[LandmarkHypothesis]:
    """Build LandmarkHypothesis for each candidate from collected votes.

    Returns hypotheses sorted by signed fusion score (support_rrf - lambda * oppose_rrf).
    Only includes candidates with at least one +1 vote.
    """
    hypotheses: list[LandmarkHypothesis] = []

    for eid in candidate_ids:
        votes = all_votes.get(eid, [])
        if not votes:
            continue

        has_support = any(v.vote == 1 for v in votes)
        if not has_support:
            continue

        support_rrf = compute_rrf(votes, +1)
        oppose_rrf = compute_rrf(votes, -1)
        convergence = classify_convergence(votes, support_rrf, oppose_rrf)

        nbr_sig = nbr_sigs.get(eid, set())
        if selected_nbrs and nbr_sig:
            novelty = 1.0 - len(nbr_sig & selected_nbrs) / max(len(nbr_sig), 1)
        else:
            novelty = 1.0

        hypotheses.append(
            LandmarkHypothesis(
                entity_id=eid,
                votes=votes,
                support_rrf=support_rrf,
                oppose_rrf=oppose_rrf,
                convergence_class=convergence,
                novelty_proxy=novelty,
                neighborhood_size=len(nbr_sig),
            )
        )

    # Sort by signed fusion score
    hypotheses.sort(
        key=lambda h: h.support_rrf - oppose_lambda * h.oppose_rrf,
        reverse=True,
    )

    return hypotheses
