"""Scoring functions for proof network navigation.

Pure computation — no DB access. Computes bank alignment, anchor
relevance, and seed similarity scores for entity ranking.

Extracted from proof_network.py per LintGate file-too-long split proposal.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from src.nav_contracts import BANK_NAMES, StructuredQuery

# Missing bank penalty: not zero, but penalized.
_MISSING_BANK_SCORE = 0.3


@functools.lru_cache(maxsize=256)
def bank_score(entity_signed_pos: int, query_direction: int) -> float:
    """Score a single bank alignment between entity position and query direction."""
    if query_direction == 0:
        return 1.0 / (1.0 + abs(entity_signed_pos))
    alignment = entity_signed_pos * query_direction
    if alignment > 0:
        return 1.0
    if alignment == 0:
        return 0.5
    return 1.0 / (1.0 + abs(alignment))


def compose_bank_scores(
    scores: dict[str, float],
    confidences: dict[str, float],
    mechanism: str = "confidence_weighted",
    floor_epsilon: float = 0.1,
) -> float:
    """Compose per-bank scores into a single alignment score.

    Mechanisms:
        multiplicative      — pure product (precise but fragile)
        confidence_weighted — prod score_i^confidence_i (default, graceful)
        soft_floor          — prod max(score_i, epsilon)
        geometric_mean      — prod^(1/n)
        log_additive        — exp(sum log(score_i))
    """
    if not scores:
        return 1.0

    vals = list(scores.values())

    if mechanism == "multiplicative":
        return math.prod(vals)

    if mechanism == "confidence_weighted":
        return math.prod(s ** confidences.get(bank, 1.0) for bank, s in scores.items())

    if mechanism == "soft_floor":
        return math.prod(max(s, floor_epsilon) for s in vals)

    if mechanism == "geometric_mean":
        return math.prod(vals) ** (1.0 / len(vals))

    if mechanism == "log_additive":
        return math.exp(sum(math.log(max(s, 1e-6)) for s in vals))

    msg = f"Unknown scoring mechanism: {mechanism}"
    raise ValueError(msg)


def _vectorized_bank_scores(
    candidate_ids: list[int],
    positions: dict[int, dict[str, int]],
    query: StructuredQuery,
) -> np.ndarray:
    """Compute bank scores for all candidates using NumPy vectorization.

    Returns 1D array of composite bank scores (one per candidate).
    Uses the confidence_weighted mechanism: prod(score_i^confidence_i).
    """
    n = len(candidate_ids)
    if n == 0:
        return np.array([], dtype=np.float64)

    n_banks = len(BANK_NAMES)
    pos_matrix = np.full((n, n_banks), np.nan, dtype=np.float64)
    for i, eid in enumerate(candidate_ids):
        epos = positions.get(eid, {})
        for j, bank_name in enumerate(BANK_NAMES):
            if bank_name in epos:
                pos_matrix[i, j] = epos[bank_name]

    directions = np.array([query.bank_directions.get(b, 0) for b in BANK_NAMES], dtype=np.float64)
    confidences = np.array(
        [query.bank_confidences.get(b, 1.0) for b in BANK_NAMES], dtype=np.float64
    )

    alignment = pos_matrix * directions

    scores = np.where(
        np.isnan(pos_matrix),
        _MISSING_BANK_SCORE,
        np.where(
            directions == 0,
            1.0 / (1.0 + np.abs(np.nan_to_num(pos_matrix, nan=0.0))),
            np.where(
                alignment > 0,
                1.0,
                np.where(
                    alignment == 0,
                    0.5,
                    1.0 / (1.0 + np.abs(alignment)),
                ),
            ),
        ),
    )

    skip_mask = (directions == 0) & np.isnan(pos_matrix)
    scores[skip_mask] = 1.0

    log_scores = np.log(np.maximum(scores, 1e-10)) * confidences
    composite = np.exp(np.sum(log_scores, axis=1))

    return composite


def _compute_bank_score(
    positions: dict[str, int],
    query: StructuredQuery,
    mechanism: str,
) -> float:
    """Compute composite bank alignment score for one entity."""
    scores: dict[str, float] = {}
    for bank_name in BANK_NAMES:
        direction = query.bank_directions.get(bank_name, 0)
        if direction == 0 and bank_name not in positions:
            continue
        entity_pos = positions.get(bank_name)
        if entity_pos is None:
            scores[bank_name] = _MISSING_BANK_SCORE
        else:
            scores[bank_name] = bank_score(entity_pos, direction)

    return compose_bank_scores(scores, query.bank_confidences, mechanism)


def _compute_anchor_score(
    entity_anchors: set[int],
    query: StructuredQuery,
    idf_cache: dict[int, float],
) -> float:
    """Compute anchor relevance score (IDF-weighted)."""
    if not query.prefer_anchors:
        return 1.0

    total_idf = 0.0
    matched_idf = 0.0
    for aid, weight in zip(query.prefer_anchors, query.prefer_weights):
        idf = idf_cache.get(aid, 1.0) * weight
        total_idf += idf
        if aid in entity_anchors:
            matched_idf += idf

    avoid_penalty = 1.0
    for aid in query.avoid_anchors:
        if aid in entity_anchors:
            avoid_penalty *= 0.5

    if total_idf == 0:
        return avoid_penalty
    return (matched_idf / total_idf) * avoid_penalty


def _compute_seed_score(
    entity_anchors: set[int],
    seed_anchors: set[int],
    idf_cache: dict[int, float],
) -> float:
    """Compute IDF-weighted Jaccard similarity with seed entities."""
    if not seed_anchors:
        return 1.0

    shared = entity_anchors & seed_anchors
    union = entity_anchors | seed_anchors
    if not union:
        return 1.0

    shared_idf = sum(idf_cache.get(a, 1.0) for a in shared)
    union_idf = sum(idf_cache.get(a, 1.0) for a in union)
    if union_idf == 0:
        return 1.0
    return shared_idf / union_idf
