"""LintGate-style coherence engine for lens committee fusion.

Fuses typed LensVote instances into coherence states and resolution actions.
Uses discrete state classification (not Kuramoto), matching the interpretable
typed-vote pattern from the landmark selector layer.
"""

from __future__ import annotations

from src.lens_guidance import (
    GuidancePacket,
    LensCommitteeState,
    LensVote,
    ResolutionDecision,
)
from src.lens_models import ALL_SPECIALISTS, LensSpecialist
from src.nav_contracts import ScoredEntity

# Action priority for sorting (lower = higher priority)
ACTION_PRIORITY = {
    "act": 0,
    "trust_lens": 1,
    "bifurcate": 2,
    "expand_more": 3,
}


def classify_committee_state(votes: list[LensVote]) -> str:
    """Classify coherence state from lens committee votes.

    States:
        stable:     >= 2 lenses agree (+1), opposition < 0.3x support
        coupled:    >= 2 lenses agree (+1), mild opposition (< 0.5x)
        isolated:   exactly 1 lens supports, no opposition
        conflicted: opposition >= 0.5x support, with some support
        degraded:   no support at all
    """
    support = [v for v in votes if v.vote == 1]
    oppose = [v for v in votes if v.vote == -1]

    support_mass = sum(v.confidence for v in support)
    oppose_mass = sum(v.confidence for v in oppose)

    support_count = len(support)

    if support_count == 0:
        return "degraded"

    if support_count >= 2:
        if oppose_mass < 0.3 * support_mass:
            return "stable"
        if oppose_mass < 0.5 * support_mass:
            return "coupled"

    if oppose_mass >= 0.5 * support_mass and support_count >= 1:
        return "conflicted"

    if support_count == 1 and len(oppose) == 0:
        return "isolated"

    return "isolated"


def compute_fused_score(votes: list[LensVote]) -> float:
    """Confidence-weighted fusion: support_mass - 0.5 * oppose_mass."""
    support_mass = sum(v.confidence for v in votes if v.vote == 1)
    oppose_mass = sum(v.confidence for v in votes if v.vote == -1)
    return support_mass - 0.5 * oppose_mass


def _state_to_action(state: str) -> str:
    """Map coherence state to resolution action."""
    return {
        "stable": "act",
        "coupled": "act",
        "isolated": "trust_lens",
        "conflicted": "bifurcate",
        "degraded": "expand_more",
    }.get(state, "expand_more")


def run_lens_committee(
    packet: GuidancePacket,
    specialists: list[LensSpecialist] | None = None,
) -> list[LensCommitteeState]:
    """Run all lens specialists on all candidates, return committee states."""
    if specialists is None:
        specialists = ALL_SPECIALISTS

    # Collect all votes per candidate
    votes_by_candidate: dict[int, list[LensVote]] = {}
    for specialist in specialists:
        for vote in specialist.vote_batch(packet):
            votes_by_candidate.setdefault(vote.candidate_id, []).append(vote)

    # Classify each candidate
    states: list[LensCommitteeState] = []
    for candidate in packet.candidate_premises:
        cid = candidate.entity_id
        votes = votes_by_candidate.get(cid, [])
        coherence = classify_committee_state(votes)
        fused = compute_fused_score(votes)
        action = _state_to_action(coherence)

        states.append(
            LensCommitteeState(
                candidate_id=cid,
                votes=votes,
                coherence_state=coherence,
                fused_score=fused,
                action=action,
            )
        )

    return states


def build_resolution_decision(
    packet: GuidancePacket,
    committee_states: list[LensCommitteeState],
) -> ResolutionDecision:
    """Build final ordered premise list modulated by committee evidence.

    The committee MODULATES the reranked score, not replaces it:
        combined = rerank_score * (1 + modulation * fused_score)
    where modulation scales with committee informativeness.

    Action tiers provide secondary ordering only when scores are tied.
    Degraded candidates (no lens support) are demoted but not dropped.
    """
    state_map = {s.candidate_id: s for s in committee_states}

    # Measure committee informativeness: fraction of non-abstain votes
    total_votes = sum(len(s.votes) for s in committee_states)
    active_votes = sum(
        1 for s in committee_states for v in s.votes if v.vote != 0
    )
    informativeness = active_votes / max(total_votes, 1)

    # Modulation strength: scale with informativeness (0 when all abstain)
    modulation = 0.3 * informativeness

    scored: list[tuple[float, int, ScoredEntity]] = []
    for candidate in packet.candidate_premises:
        state = state_map.get(candidate.entity_id)
        fused = state.fused_score if state else 0.0
        action_pri = ACTION_PRIORITY.get(state.action, 99) if state else 99

        # Modulate reranked score, don't replace it
        combined = candidate.final_score * (1.0 + modulation * fused)

        # Demote degraded candidates
        if state and state.action == "expand_more":
            combined *= 0.5

        scored.append((combined, action_pri, candidate))

    scored.sort(key=lambda x: (-x[0], x[1]))

    sorted_states = sorted(
        committee_states,
        key=lambda s: (ACTION_PRIORITY.get(s.action, 99), -s.fused_score),
    )

    return ResolutionDecision(
        premises=[c for _, _, c in scored],
        committee_states=sorted_states,
    )
