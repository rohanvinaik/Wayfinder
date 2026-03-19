"""Tests for lens guidance, lens models, and coherence engine."""

import unittest

from src.coherence_engine import (
    build_resolution_decision,
    classify_committee_state,
    compute_fused_score,
    run_lens_committee,
)
from src.landmark_freeze import FrozenLandmarkState, LandmarkResidualReport
from src.lens_guidance import GuidancePacket, LensCommitteeState, LensVote
from src.lens_models import (
    ALL_SPECIALISTS,
    ConstantMatchLens,
    LensSpecialist,
    LocalityGuardLens,
)
from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_network import clear_caches


def _scored(eid, name="", score=1.0):
    return ScoredEntity(eid, name, score, score, 0.0, 1.0)


def _empty_frozen():
    return FrozenLandmarkState(
        committed_ids=frozenset(),
        committed_scores=(),
        dominant_anchors=frozenset(),
        dominant_constants=frozenset(),
        dominant_namespace_clusters=frozenset(),
        active_lens_family="bridge",
        neighborhood_bank_centroid=(),
    )


def _make_packet(**kwargs):
    defaults = {
        "query": StructuredQuery(
            bank_directions={"structure": 1},
            bank_confidences={"structure": 0.8},
            prefer_anchors=[1, 2],
            prefer_weights=[1.0, 1.0],
        ),
        "landmark_committee": [_scored(100)],
        "candidate_premises": [_scored(1), _scored(2)],
        "residual": LandmarkResidualReport(),
        "frozen_state": _empty_frozen(),
        "idf_cache": {1: 1.0, 2: 1.0},
        "anchor_categories": {},
        "candidate_anchor_sets": {1: {1, 2, 10}, 2: {3}},
        "candidate_summaries": {
            1: {
                "anchor_count": 3, "constant_count": 2, "residual_overlap": 1,
                "unsupported_const_overlap": 1, "frozen_overlap": 0,
            },
            2: {
                "anchor_count": 1, "constant_count": 0, "residual_overlap": 0,
                "unsupported_const_overlap": 0, "frozen_overlap": 0,
            },
        },
    }
    defaults.update(kwargs)
    return GuidancePacket(**defaults)


def _lens_vote(cid, vote=1, conf=0.8, name="test"):
    return LensVote(cid, vote, conf, name, "")


# ---------------------------------------------------------------------------
# Coherence engine tests
# ---------------------------------------------------------------------------


class TestClassifyCommitteeState(unittest.TestCase):

    def test_stable_two_support_no_opposition(self):
        votes = [_lens_vote(1, 1, 0.8, "a"), _lens_vote(1, 1, 0.7, "b")]
        self.assertEqual(classify_committee_state(votes), "stable")

    def test_isolated_single_support(self):
        votes = [_lens_vote(1, 1, 0.8, "a"), _lens_vote(1, 0, 0.0, "b")]
        self.assertEqual(classify_committee_state(votes), "isolated")

    def test_coupled_support_with_mild_opposition(self):
        # oppose_mass=0.6 vs support_mass=1.5 -> 0.4x -> between 0.3 and 0.5
        votes = [
            _lens_vote(1, 1, 0.8, "a"),
            _lens_vote(1, 1, 0.7, "b"),
            _lens_vote(1, -1, 0.6, "c"),
        ]
        self.assertEqual(classify_committee_state(votes), "coupled")

    def test_conflicted_significant_opposition(self):
        votes = [_lens_vote(1, 1, 0.5, "a"), _lens_vote(1, -1, 0.5, "b")]
        self.assertEqual(classify_committee_state(votes), "conflicted")

    def test_degraded_no_support(self):
        votes = [_lens_vote(1, 0, 0.0, "a"), _lens_vote(1, -1, 0.5, "b")]
        self.assertEqual(classify_committee_state(votes), "degraded")

    def test_degraded_empty(self):
        self.assertEqual(classify_committee_state([]), "degraded")


class TestFusedScore(unittest.TestCase):

    def test_support_minus_half_oppose(self):
        votes = [_lens_vote(1, 1, 0.8), _lens_vote(1, -1, 0.4)]
        self.assertAlmostEqual(compute_fused_score(votes), 0.8 - 0.2)

    def test_abstain_not_counted(self):
        votes = [_lens_vote(1, 1, 0.8), _lens_vote(1, 0, 0.5)]
        self.assertAlmostEqual(compute_fused_score(votes), 0.8)


class TestResolutionDecision(unittest.TestCase):

    def test_act_before_trust_lens(self):
        packet = _make_packet()
        states = [
            LensCommitteeState(1, [], "isolated", 0.5, "trust_lens"),
            LensCommitteeState(2, [], "stable", 0.8, "act"),
        ]
        decision = build_resolution_decision(packet, states)
        self.assertEqual(decision.premises[0].entity_id, 2)

    def test_degraded_demoted(self):
        packet = _make_packet()
        states = [
            LensCommitteeState(1, [], "degraded", 0.0, "expand_more"),
            LensCommitteeState(2, [], "stable", 0.8, "act"),
        ]
        decision = build_resolution_decision(packet, states)
        ids = [p.entity_id for p in decision.premises]
        # Degraded is demoted (lower score), not dropped
        self.assertIn(2, ids)
        self.assertEqual(ids[0], 2)  # stable candidate ranked first


# ---------------------------------------------------------------------------
# Specialist tests
# ---------------------------------------------------------------------------


class TestSpecialistInterface(unittest.TestCase):

    def test_all_specialists_are_lens_specialist(self):
        for s in ALL_SPECIALISTS:
            self.assertIsInstance(s, LensSpecialist)

    def test_all_specialists_have_name(self):
        names = {s.name for s in ALL_SPECIALISTS}
        self.assertEqual(len(names), len(ALL_SPECIALISTS))

    def test_all_vote_returns_lens_vote(self):
        packet = _make_packet()
        candidate = _scored(1)
        for s in ALL_SPECIALISTS:
            vote = s.vote(packet, candidate)
            self.assertIsInstance(vote, LensVote)
            self.assertIn(vote.vote, (-1, 0, 1))


class TestLocalityGuardLens(unittest.TestCase):

    def test_opposes_locality_only(self):
        packet = _make_packet(locality_only_warnings=[1])
        vote = LocalityGuardLens().vote(packet, _scored(1))
        self.assertEqual(vote.vote, -1)

    def test_never_votes_positive(self):
        packet = _make_packet()
        vote = LocalityGuardLens().vote(packet, _scored(1))
        self.assertLessEqual(vote.vote, 0)


class TestConstantMatchLens(unittest.TestCase):

    def test_supports_when_constants_present(self):
        packet = _make_packet(
            residual=LandmarkResidualReport(unsupported_constants=[10]),
            candidate_summaries={
                1: {
                    "anchor_count": 5, "constant_count": 3, "residual_overlap": 0,
                    "unsupported_const_overlap": 2, "frozen_overlap": 0,
                },
                2: {
                    "anchor_count": 1, "constant_count": 0, "residual_overlap": 0,
                    "unsupported_const_overlap": 0, "frozen_overlap": 0,
                },
            },
        )
        vote = ConstantMatchLens().vote(packet, _scored(1))
        self.assertEqual(vote.vote, 1)

    def test_abstains_no_unsupported(self):
        packet = _make_packet()
        vote = ConstantMatchLens().vote(packet, _scored(1))
        self.assertEqual(vote.vote, 0)


# ---------------------------------------------------------------------------
# Committee integration
# ---------------------------------------------------------------------------


class TestRunLensCommittee(unittest.TestCase):

    def test_returns_states_for_all_candidates(self):
        packet = _make_packet()
        states = run_lens_committee(packet)
        self.assertEqual(len(states), len(packet.candidate_premises))

    def test_each_state_has_coherence(self):
        packet = _make_packet()
        states = run_lens_committee(packet)
        valid = {"stable", "isolated", "coupled", "conflicted", "degraded"}
        for s in states:
            self.assertIn(s.coherence_state, valid)


class TestEndToEnd(unittest.TestCase):
    """Integration: lens guidance through landmark_expand_retrieve."""

    def setUp(self):
        clear_caches()
        from src.landmark_selectors import clear_selector_caches

        clear_selector_caches()
        # Reuse the test DB builder from landmark selector tests
        from tests.test_landmark_selectors import _build_test_db

        self.conn = _build_test_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def test_lens_guidance_flag(self):
        from src.premise_retrieval import landmark_expand_retrieve

        query = StructuredQuery(
            bank_directions={"structure": 1},
            bank_confidences={"structure": 0.8},
            prefer_anchors=[1, 2],
            prefer_weights=[1.0, 1.0],
        )
        results, trace = landmark_expand_retrieve(
            query,
            self.conn,
            limit=4,
            config={"landmark_strategy": "multi_pass", "lens_guidance": True},
        )
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
