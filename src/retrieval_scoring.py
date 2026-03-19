"""Pure scoring functions for premise retrieval stages 1 and 3.

Lens-based scoring: primary seed scoring, bridge potential,
coherence, support bonus, and hub penalty. All functions are
stateless and do not access the database.
"""

from __future__ import annotations

import math

from src.proof_scoring import LENS_CATEGORIES

# Primary lenses decide semantic relevance; support lenses stabilize.
PRIMARY_LENSES = frozenset({"proof", "constant", "semantic", "structural"})
SUPPORT_LENSES = frozenset({"locality", "lexical"})


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


def _compute_bridge_potential(
    nbr_anchor_sig: set[int],
    query_anchors: list[int],
    query_weights: list[float],
    idf_cache: dict[int, float],
) -> float:
    """Score how well a landmark's neighborhood matches the query.

    IDF-weighted overlap between query anchors and the union of
    anchors from the landmark's depends_on neighbors.
    """
    if not nbr_anchor_sig or not query_anchors:
        return 0.0

    total_idf = 0.0
    matched_idf = 0.0
    for aid, weight in zip(query_anchors, query_weights):
        idf = idf_cache.get(aid, 1.0) * weight
        total_idf += idf
        if aid in nbr_anchor_sig:
            matched_idf += idf

    return matched_idf / total_idf if total_idf > 0 else 0.0


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


def _compute_hub_penalty(
    entity_id: int,
    hub_degrees: dict[int, int],
    median_degree: float,
) -> float:
    """Downweight globally over-connected candidates (hubs).

    Entities with far more incoming links than the median are less
    likely to be specifically relevant -- they're infrastructure lemmas.
    """
    degree = hub_degrees.get(entity_id, 0)
    if degree <= median_degree:
        return 1.0
    # Gentle penalty: 1 / sqrt(degree / median)
    return 1.0 / math.sqrt(degree / max(median_degree, 1.0))
