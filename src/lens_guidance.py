"""Lens guidance contracts and packet builder.

Bridges deterministic retrieval collapse and lens-specialist resolution.
The GuidancePacket captures the full state of the deterministic pipeline
so that small learned models can resolve the ambiguous remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.landmark_freeze import FrozenLandmarkState, LandmarkResidualReport
from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_network import _DataCache
from src.retrieval_stages import ExpandedCandidate


@dataclass
class LensVote:
    """One lens specialist's typed output on a candidate."""

    candidate_id: int
    vote: int  # +1 support, 0 abstain, -1 oppose
    confidence: float  # [0, 1]
    lens_name: str
    evidence: str


@dataclass
class LensCommitteeState:
    """Coherence state from fusing multiple lens votes."""

    candidate_id: int
    votes: list[LensVote] = field(default_factory=list)
    coherence_state: str = "degraded"
    fused_score: float = 0.0
    action: str = "expand_more"


@dataclass
class GuidancePacket:
    """Deterministic collapse output fed to lens specialists.

    Each candidate carries its ACTUAL anchor set and computed overlaps,
    not neighborhood proxies. The packet is the complete evidence surface
    for lens specialist decisions.
    """

    query: StructuredQuery
    landmark_committee: list[ScoredEntity]
    candidate_premises: list[ScoredEntity]
    residual: LandmarkResidualReport
    frozen_state: FrozenLandmarkState
    conflict_clusters: list[tuple[int, int]] = field(default_factory=list)
    locality_only_warnings: list[int] = field(default_factory=list)
    negative_evidence: dict[int, list[str]] = field(default_factory=dict)
    # Per-candidate evidence (keyed by entity_id)
    candidate_anchor_sets: dict[int, set[int]] = field(default_factory=dict)
    candidate_summaries: dict[int, dict] = field(default_factory=dict)
    idf_cache: dict[int, float] = field(default_factory=dict)
    anchor_categories: dict[int, str] = field(default_factory=dict)


@dataclass
class ResolutionDecision:
    """Final resolution: ordered premises to try."""

    premises: list[ScoredEntity] = field(default_factory=list)
    committee_states: list[LensCommitteeState] = field(default_factory=list)


def build_guidance_packet(
    query: StructuredQuery,
    landmarks: list[ScoredEntity],
    candidates: list[ScoredEntity],
    frozen_state: FrozenLandmarkState,
    residual: LandmarkResidualReport,
    expanded: list[ExpandedCandidate],
    data: _DataCache,
    nbr_sigs: dict[int, set[int]],
    idf_cache: dict[int, float],
) -> GuidancePacket:
    """Build a GuidancePacket from existing pipeline output.

    Uses data.anchor_sets for each candidate's actual anchors (not
    nbr_sigs, which are landmark neighborhood signatures).
    """
    uncovered = set(residual.residual_anchor_priorities)
    unsupported_const = set(residual.unsupported_constants)

    # Per-candidate: actual anchor sets and computed summaries
    anchor_sets: dict[int, set[int]] = {}
    summaries: dict[int, dict] = {}
    for c in candidates:
        ea = data.anchor_sets.get(c.entity_id, set())
        anchor_sets[c.entity_id] = ea

        const_anchors = {
            a for a in ea if data.anchor_categories.get(a) == "constant"
        }
        summaries[c.entity_id] = {
            "anchor_count": len(ea),
            "constant_count": len(const_anchors),
            "residual_overlap": len(ea & uncovered),
            "unsupported_const_overlap": len(const_anchors & unsupported_const),
            "frozen_overlap": len(ea & frozen_state.dominant_anchors),
        }

    # Negative evidence: hub in-degrees + landmark opposition
    neg_evidence: dict[int, list[str]] = {}
    hub_degrees = data.hub_in_degrees
    if hub_degrees:
        degrees = sorted(hub_degrees.values())
        median_deg = degrees[len(degrees) // 2] if degrees else 1
        if median_deg == 0:
            median_deg = 1
        for c in candidates:
            deg = hub_degrees.get(c.entity_id, 0)
            if deg > 5 * median_deg:
                neg_evidence.setdefault(c.entity_id, []).append(
                    f"hub_in_degree={deg}>5x_median={5 * median_deg}"
                )

    # Support path provenance from expanded candidates
    support_map: dict[int, list[int]] = {}
    for ec in expanded:
        support_map[ec.entity_id] = [sv.landmark_id for sv in ec.supports]

    return GuidancePacket(
        query=query,
        landmark_committee=landmarks,
        candidate_premises=candidates,
        residual=residual,
        frozen_state=frozen_state,
        conflict_clusters=list(residual.conflict_clusters),
        locality_only_warnings=list(residual.locality_only_candidates),
        negative_evidence=neg_evidence,
        candidate_anchor_sets=anchor_sets,
        candidate_summaries=summaries,
        idf_cache=idf_cache,
        anchor_categories=dict(data.anchor_categories),
    )
