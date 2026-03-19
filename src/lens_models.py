"""Lens specialist models: typed voters on candidate premises.

Each specialist examines candidates through one interpretive lens
and returns a typed vote (+1/0/-1) with confidence. Initially
rule-based; designed for drop-in replacement with small trained models.

Specialists use actual candidate anchor sets from the GuidancePacket,
not neighborhood proxies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.lens_guidance import GuidancePacket, LensVote
from src.nav_contracts import ScoredEntity
from src.retrieval_scoring import (
    _compute_primary_coherence,
)


class LensSpecialist(ABC):
    """Abstract interface for a lens specialist."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote: ...

    def vote_batch(self, packet: GuidancePacket) -> list[LensVote]:
        return [self.vote(packet, c) for c in packet.candidate_premises]


class BridgeCoherenceLens(LensSpecialist):
    """Primary lens coherence using the candidate's actual anchor set."""

    @property
    def name(self) -> str:
        return "bridge_coherence"

    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote:
        ea = packet.candidate_anchor_sets.get(candidate.entity_id, set())
        if not ea:
            return LensVote(
                candidate.entity_id, 0, 0.0, self.name, "no_anchors"
            )

        coherence = _compute_primary_coherence(
            ea,
            packet.query.prefer_anchors,
            packet.query.prefer_weights,
            packet.idf_cache,
            packet.anchor_categories,
        )

        if coherence > 0.1:
            v, conf = 1, min(coherence, 1.0)
        elif coherence < 0.01:
            v, conf = -1, 0.5
        else:
            v, conf = 0, coherence

        return LensVote(
            candidate.entity_id, v, conf, self.name,
            f"coherence={coherence:.3f}",
        )


class ResidualCoverageLens(LensSpecialist):
    """Checks actual overlap between candidate anchors and uncovered residual."""

    @property
    def name(self) -> str:
        return "residual_coverage"

    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote:
        summary = packet.candidate_summaries.get(candidate.entity_id, {})
        residual_overlap = summary.get("residual_overlap", 0)
        uncovered_count = len(packet.residual.residual_anchor_priorities)

        if uncovered_count == 0:
            return LensVote(
                candidate.entity_id, 0, 0.0, self.name, "no_uncovered"
            )

        coverage = residual_overlap / max(uncovered_count, 1)

        if coverage > 0.05:
            v = 1
        elif residual_overlap == 0 and candidate.entity_id in {
            eid for pair in packet.conflict_clusters for eid in pair
        }:
            v = -1
        else:
            v = 0

        return LensVote(
            candidate.entity_id, v, min(coverage, 1.0), self.name,
            f"overlap={residual_overlap}/{uncovered_count}",
        )


class ConstantMatchLens(LensSpecialist):
    """Checks actual overlap between candidate constants and unsupported constants."""

    @property
    def name(self) -> str:
        return "constant_match"

    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote:
        unsupported = set(packet.residual.unsupported_constants)
        if not unsupported:
            return LensVote(
                candidate.entity_id, 0, 0.0, self.name,
                "no_unsupported_constants",
            )

        summary = packet.candidate_summaries.get(candidate.entity_id, {})
        const_overlap = summary.get("unsupported_const_overlap", 0)

        if const_overlap > 0:
            v, conf = 1, 0.8
        else:
            v, conf = 0, 0.0

        return LensVote(
            candidate.entity_id, v, conf, self.name,
            f"const_overlap={const_overlap}",
        )


class LocalityGuardLens(LensSpecialist):
    """Negative selector: opposes locality-only candidates. Never +1."""

    @property
    def name(self) -> str:
        return "locality_guard"

    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote:
        if candidate.entity_id in packet.locality_only_warnings:
            return LensVote(
                candidate.entity_id, -1, 0.6, self.name, "locality_only"
            )
        return LensVote(
            candidate.entity_id, 0, 0.0, self.name, "not_locality_only"
        )


class HubPenaltyLens(LensSpecialist):
    """Negative selector: opposes hub entities using actual hub evidence."""

    @property
    def name(self) -> str:
        return "hub_penalty"

    def vote(self, packet: GuidancePacket, candidate: ScoredEntity) -> LensVote:
        neg = packet.negative_evidence.get(candidate.entity_id, [])
        if neg:
            return LensVote(
                candidate.entity_id, -1, 0.7, self.name,
                f"hub:{neg[0]}",
            )
        return LensVote(
            candidate.entity_id, 0, 0.0, self.name, "normal"
        )


# Registry of all rule-based specialists
ALL_SPECIALISTS: list[LensSpecialist] = [
    BridgeCoherenceLens(),
    ResidualCoverageLens(),
    ConstantMatchLens(),
    LocalityGuardLens(),
    HubPenaltyLens(),
]
