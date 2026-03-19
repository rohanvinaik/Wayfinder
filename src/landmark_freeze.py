"""Contextual freezing + residual isolation for landmark selection.

After selector fusion produces LandmarkHypotheses, this module:
1. Freezes committed state (converged only; isolated requires margin+novelty)
2. Computes residual report (category-weighted uncovered anchors)
3. Resolves ambiguous landmarks by category-weighted residual coverage
4. Supports iterative freeze-resolve via _extend_frozen_state()

Transfer from ARC-3 LevelCrystal: freeze structural facts, reason only about novelty.
Transfer from ARC iterative_decomposer: residual-guided selector family transitions.
Transfer from ARC holographic_resolver: hard_lock/soft/open uncertainty classes.
Transfer from ARC constraint_ledger: monotone vs revisable constraint separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.landmark_selectors import LandmarkHypothesis
from src.proof_network import _DataCache

# Category discriminative gain (Minority Channel Advantage):
# rare channels carry more bits per observation.
CATEGORY_GAIN: dict[str, float] = {
    "semantic": 1.0,
    "structural": 0.9,
    "constant": 0.8,
    "proof": 0.7,
    "locality": 0.3,
    "lexical": 0.3,
    "general": 0.5,
}


@dataclass(frozen=True)
class FrozenLandmarkState:
    """Immutable committed landmark knowledge after selector fusion.

    Analogous to ARC-3 LevelCrystal: structural facts that are now
    effectively certain, applied as priors to collapse the remaining
    hypothesis space.
    """

    committed_ids: frozenset[int]
    committed_scores: tuple[tuple[int, float], ...]
    dominant_anchors: frozenset[int]  # neighborhood anchor union
    dominant_constants: frozenset[int]  # constant-category anchors
    dominant_namespace_clusters: frozenset[str]
    active_lens_family: str  # dominant support family
    neighborhood_bank_centroid: tuple[tuple[str, float], ...]


@dataclass
class LandmarkResidualReport:
    """What the frozen state doesn't cover.

    Guides the ambiguity resolver: which anchor mass is still uncovered,
    which constants remain unsupported, which candidates are locality-only.
    """

    uncovered_anchor_mass: dict[str, float] = field(default_factory=dict)
    unsupported_constants: list[int] = field(default_factory=list)
    locality_only_candidates: list[int] = field(default_factory=list)
    conflict_clusters: list[tuple[int, int]] = field(default_factory=list)
    residual_entropy: float = 0.0
    query_coverage: float = 0.0
    residual_anchor_priorities: list[int] = field(default_factory=list)
    phase_signal: str = "bridge_dominant"


@dataclass
class LandmarkConstraintLedger:
    """Monotone vs revisable constraints for landmark selection."""

    hub_vetoes: frozenset[int] = field(default_factory=frozenset)
    traced_only: bool = True
    active_lens_family: str = "bridge"
    residual_anchor_priorities: list[int] = field(default_factory=list)
    locality_window: frozenset[str] = field(default_factory=frozenset)
    phase: str = "bridge_dominant"


# ---------------------------------------------------------------------------
# Step 1: Freeze committed state (conservative)
# ---------------------------------------------------------------------------


def freeze_committed_state(
    hypotheses: list[LandmarkHypothesis],
    data: _DataCache,
    nbr_sigs: dict[int, set[int]],
) -> tuple[FrozenLandmarkState, list[LandmarkHypothesis]]:
    """Freeze converged landmarks; isolated require margin + novelty.

    Classification:
        hard_lock: converged hypotheses (>= 2 families agree, low opposition)
        soft (promotable): isolated with conf >= 0.8 AND margin AND novelty > 0.3
        open: everything else -> ambiguous pool
    """
    committed: list[LandmarkHypothesis] = []
    ambiguous: list[LandmarkHypothesis] = []

    # First pass: only converged are committed
    for hyp in hypotheses:
        if hyp.convergence_class == "converged":
            committed.append(hyp)
        else:
            ambiguous.append(hyp)

    # Second pass: promote qualifying isolated from ambiguous
    sorted_ambiguous = sorted(
        ambiguous, key=lambda h: h.support_rrf, reverse=True
    )
    promoted: list[LandmarkHypothesis] = []
    remaining: list[LandmarkHypothesis] = []

    for i, hyp in enumerate(sorted_ambiguous):
        if hyp.convergence_class == "isolated":
            max_conf = max(
                (v.confidence for v in hyp.votes if v.vote == 1), default=0.0
            )
            next_best = (
                sorted_ambiguous[i + 1].support_rrf
                if i + 1 < len(sorted_ambiguous)
                else hyp.support_rrf  # no artificial margin for last entry
            )
            margin = hyp.support_rrf - next_best

            if max_conf >= 0.8 and margin > 0.005 and hyp.novelty_proxy > 0.3:
                promoted.append(hyp)
            else:
                remaining.append(hyp)
        else:
            remaining.append(hyp)

    committed.extend(promoted)
    ambiguous = remaining

    frozen = _build_frozen_state(committed, data, nbr_sigs)
    return frozen, ambiguous


def _build_frozen_state(
    committed: list[LandmarkHypothesis],
    data: _DataCache,
    nbr_sigs: dict[int, set[int]],
) -> FrozenLandmarkState:
    """Build a FrozenLandmarkState from a list of committed hypotheses."""
    committed_ids = frozenset(h.entity_id for h in committed)
    committed_scores = tuple(
        (h.entity_id, h.support_rrf - 0.5 * h.oppose_rrf) for h in committed
    )

    dominant_anchor_set: set[int] = set()
    for h in committed:
        dominant_anchor_set |= nbr_sigs.get(h.entity_id, set())

    dominant_constants = frozenset(
        aid
        for aid in dominant_anchor_set
        if data.anchor_categories.get(aid) == "constant"
    )

    namespaces: set[str] = set()
    for h in committed:
        name = data.names.get(h.entity_id, "")
        parts = name.rsplit(".", 1)
        if len(parts) > 1:
            namespaces.add(parts[0])

    family_counts: dict[str, int] = {}
    for h in committed:
        for v in h.votes:
            if v.vote == 1:
                family_counts[v.family] = family_counts.get(v.family, 0) + 1
    active_lens = (
        max(family_counts, key=lambda k: family_counts[k])
        if family_counts
        else "bridge"
    )

    bank_sums: dict[str, float] = {}
    bank_n: dict[str, int] = {}
    for h in committed:
        pos = data.positions.get(h.entity_id, {})
        for bank, val in pos.items():
            bank_sums[bank] = bank_sums.get(bank, 0.0) + val
            bank_n[bank] = bank_n.get(bank, 0) + 1
    centroid = tuple(
        (bank, bank_sums[bank] / bank_n[bank]) for bank in sorted(bank_sums)
    )

    return FrozenLandmarkState(
        committed_ids=committed_ids,
        committed_scores=committed_scores,
        dominant_anchors=frozenset(dominant_anchor_set),
        dominant_constants=dominant_constants,
        dominant_namespace_clusters=frozenset(namespaces),
        active_lens_family=active_lens,
        neighborhood_bank_centroid=centroid,
    )


def extend_frozen_state(
    frozen: FrozenLandmarkState,
    new_hyps: list[LandmarkHypothesis],
    data: _DataCache,
    nbr_sigs: dict[int, set[int]],
) -> FrozenLandmarkState:
    """Extend a frozen state with newly validated hypotheses (iterative freeze)."""
    new_ids = frozenset(h.entity_id for h in new_hyps)
    new_scores = tuple(
        (h.entity_id, h.support_rrf - 0.5 * h.oppose_rrf) for h in new_hyps
    )

    new_anchors = set(frozen.dominant_anchors)
    for h in new_hyps:
        new_anchors |= nbr_sigs.get(h.entity_id, set())

    new_constants = frozenset(
        aid
        for aid in new_anchors
        if data.anchor_categories.get(aid) == "constant"
    )

    new_ns = set(frozen.dominant_namespace_clusters)
    for h in new_hyps:
        name = data.names.get(h.entity_id, "")
        parts = name.rsplit(".", 1)
        if len(parts) > 1:
            new_ns.add(parts[0])

    return FrozenLandmarkState(
        committed_ids=frozen.committed_ids | new_ids,
        committed_scores=frozen.committed_scores + new_scores,
        dominant_anchors=frozenset(new_anchors),
        dominant_constants=frozen.dominant_constants | new_constants,
        dominant_namespace_clusters=frozenset(new_ns),
        active_lens_family=frozen.active_lens_family,
        neighborhood_bank_centroid=frozen.neighborhood_bank_centroid,
    )


# ---------------------------------------------------------------------------
# Step 2: Compute residual
# ---------------------------------------------------------------------------


def compute_residual(
    frozen: FrozenLandmarkState,
    query_anchors: list[int],
    query_weights: list[float],
    data: _DataCache,
    idf_cache: dict[int, float],
    ambiguous: list[LandmarkHypothesis],
    nbr_sigs: dict[int, set[int]],
) -> LandmarkResidualReport:
    """Compute what the frozen state doesn't cover."""
    uncovered_by_cat: dict[str, float] = {}
    total_by_cat: dict[str, float] = {}
    for aid, weight in zip(query_anchors, query_weights):
        cat = data.anchor_categories.get(aid, "general")
        idf = idf_cache.get(aid, 1.0) * weight
        total_by_cat[cat] = total_by_cat.get(cat, 0.0) + idf
        if aid not in frozen.dominant_anchors:
            uncovered_by_cat[cat] = uncovered_by_cat.get(cat, 0.0) + idf

    uncovered_mass = {
        cat: uncovered_by_cat.get(cat, 0.0) / total_by_cat[cat]
        for cat in total_by_cat
        if total_by_cat[cat] > 0
    }

    unsupported = [
        aid
        for aid in query_anchors
        if data.anchor_categories.get(aid) == "constant"
        and aid not in frozen.dominant_anchors
    ]

    locality_only: list[int] = []
    for hyp in ambiguous:
        support_families = {v.family for v in hyp.votes if v.vote == 1}
        if support_families and support_families <= {"context"}:
            locality_only.append(hyp.entity_id)

    conflicts: list[tuple[int, int]] = []
    conflicted_ids = [
        h.entity_id for h in ambiguous if h.convergence_class == "conflicted"
    ]
    for i in range(len(conflicted_ids)):
        for j in range(i + 1, min(len(conflicted_ids), i + 20)):
            sig_i = nbr_sigs.get(conflicted_ids[i], set())
            sig_j = nbr_sigs.get(conflicted_ids[j], set())
            if sig_i and sig_j:
                overlap = len(sig_i & sig_j) / max(len(sig_i | sig_j), 1)
                if overlap > 0.5:
                    conflicts.append((conflicted_ids[i], conflicted_ids[j]))

    total_uncovered = sum(uncovered_by_cat.values())
    if total_uncovered > 0:
        probs = [v / total_uncovered for v in uncovered_by_cat.values() if v > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
    else:
        entropy = 0.0

    covered = sum(1 for aid in query_anchors if aid in frozen.dominant_anchors)
    coverage = covered / max(len(query_anchors), 1)

    uncovered_aids = [
        aid for aid in query_anchors if aid not in frozen.dominant_anchors
    ]
    uncovered_aids.sort(key=lambda a: idf_cache.get(a, 1.0), reverse=True)

    if uncovered_mass:
        primary_uncovered = {
            k: v
            for k, v in uncovered_mass.items()
            if k in ("constant", "semantic", "structural", "proof")
        }
        if primary_uncovered:
            top_cat = max(primary_uncovered, key=lambda k: primary_uncovered[k])
            if top_cat in ("constant", "semantic"):
                phase = "constant_recovery"
            elif top_cat == "structural":
                phase = "structure_matching"
            else:
                phase = "bridge_dominant"
        else:
            phase = "bridge_dominant"
    else:
        phase = "bridge_dominant"

    return LandmarkResidualReport(
        uncovered_anchor_mass=uncovered_mass,
        unsupported_constants=unsupported,
        locality_only_candidates=locality_only,
        conflict_clusters=conflicts,
        residual_entropy=entropy,
        query_coverage=coverage,
        residual_anchor_priorities=uncovered_aids,
        phase_signal=phase,
    )


# ---------------------------------------------------------------------------
# Step 3: Resolve ambiguous landmarks (category-weighted + bifurcation)
# ---------------------------------------------------------------------------


def resolve_ambiguous_landmarks(
    frozen: FrozenLandmarkState,
    residual: LandmarkResidualReport,
    ambiguous: list[LandmarkHypothesis],
    data: _DataCache,
    nbr_sigs: dict[int, set[int]],
    idf_cache: dict[int, float],
) -> list[LandmarkHypothesis]:
    """Score ambiguous landmarks by category-weighted residual coverage.

    Category-weighted (Minority Channel Advantage): semantic/structural
    matches are worth more than locality matches.

    Conflict bifurcation (HV Instability as Signal): conflicted landmarks
    that cover novel high-value territory score without penalty; those that
    don't get a steeper penalty than before.
    """
    if not ambiguous:
        return []

    uncovered_anchors = set(residual.residual_anchor_priorities)
    query_constants_set = set(residual.unsupported_constants)
    conflict_eids = {
        eid for pair in residual.conflict_clusters for eid in pair
    }
    locality_only_set = set(residual.locality_only_candidates)

    # Pre-compute denominator for category-weighted residual coverage
    denom = sum(
        idf_cache.get(a, 1.0)
        * CATEGORY_GAIN.get(data.anchor_categories.get(a, "general"), 0.5)
        for a in uncovered_anchors
    )

    scored: list[tuple[float, LandmarkHypothesis]] = []
    for hyp in ambiguous:
        nbr = nbr_sigs.get(hyp.entity_id, set())

        # 1. Category-weighted residual coverage
        if uncovered_anchors and nbr:
            covered_in_nbr = nbr & uncovered_anchors
            numer = sum(
                idf_cache.get(a, 1.0)
                * CATEGORY_GAIN.get(
                    data.anchor_categories.get(a, "general"), 0.5
                )
                for a in covered_in_nbr
            )
            residual_cov = numer / max(denom, 1e-9)
        else:
            residual_cov = 0.0

        # 2. Constant match
        nbr_constants = {
            aid
            for aid in nbr
            if data.anchor_categories.get(aid) == "constant"
        }
        if query_constants_set and nbr_constants:
            const_match = len(nbr_constants & query_constants_set) / max(
                len(query_constants_set), 1
            )
        else:
            const_match = 0.0

        # 3. Compatibility with frozen dominant
        if frozen.dominant_anchors and nbr:
            compat = len(nbr & frozen.dominant_anchors) / max(len(nbr), 1)
        else:
            compat = 0.0

        # Conflict bifurcation: productive conflicts score well,
        # unproductive conflicts get steeper penalty
        if hyp.entity_id in conflict_eids:
            novel_in_nbr = nbr & uncovered_anchors
            if novel_in_nbr:
                novel_value = sum(
                    idf_cache.get(a, 1.0)
                    * CATEGORY_GAIN.get(
                        data.anchor_categories.get(a, "general"), 0.5
                    )
                    for a in novel_in_nbr
                )
                novel_frac = novel_value / max(denom, 1e-9)
                # Productive: covers >10% of residual value
                conflict_pen = 1.1 if novel_frac > 0.1 else 0.5
            else:
                conflict_pen = 0.5
        else:
            conflict_pen = 1.0

        # Locality-only suppression
        locality_pen = 0.5 if hyp.entity_id in locality_only_set else 1.0

        score = (
            (0.6 * residual_cov + 0.25 * const_match + 0.15 * compat)
            * conflict_pen
            * locality_pen
        )

        if score > 0:
            scored.append((score, hyp))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [hyp for _, hyp in scored]


# ---------------------------------------------------------------------------
# Step 4: Build constraint ledger
# ---------------------------------------------------------------------------


def build_constraint_ledger(
    frozen: FrozenLandmarkState,
    residual: LandmarkResidualReport,
    hub_vetoes: frozenset[int] = frozenset(),
) -> LandmarkConstraintLedger:
    """Build constraint ledger from frozen state and residual."""
    return LandmarkConstraintLedger(
        hub_vetoes=hub_vetoes,
        traced_only=True,
        active_lens_family=frozen.active_lens_family,
        residual_anchor_priorities=list(residual.residual_anchor_priorities),
        locality_window=frozen.dominant_namespace_clusters,
        phase=residual.phase_signal,
    )
