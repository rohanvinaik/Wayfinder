"""Tests for landmark_freeze — conservative freeze, category-weighted residual, bifurcation."""

import unittest

from src.landmark_freeze import (
    CATEGORY_GAIN,
    FrozenLandmarkState,
    LandmarkResidualReport,
    compute_residual,
    extend_frozen_state,
    freeze_committed_state,
    resolve_ambiguous_landmarks,
)
from src.landmark_selectors import LandmarkHypothesis, SelectorVote
from src.proof_network import _DataCache


def _make_data(**kwargs):
    return _DataCache(
        positions=kwargs.get("positions", {}),
        anchor_sets=kwargs.get("anchor_sets", {}),
        names=kwargs.get("names", {}),
        provenances=kwargs.get("provenances", {}),
        anchor_categories=kwargs.get("anchor_categories", {}),
    )


def _vote(vote=1, confidence=0.9, family="bridge", name="test", rank=1):
    return SelectorVote(
        vote=vote,
        confidence=confidence,
        raw_score=1.0,
        selector_name=name,
        family=family,
        evidence="",
        rank=rank,
    )


def _hyp(eid, convergence="isolated", votes=None, support_rrf=0.016, novelty=1.0):
    return LandmarkHypothesis(
        entity_id=eid,
        votes=votes or [_vote()],
        support_rrf=support_rrf,
        oppose_rrf=0.0,
        convergence_class=convergence,
        novelty_proxy=novelty,
        neighborhood_size=5,
    )


class TestConservativeFreeze(unittest.TestCase):
    """Test that freeze_committed_state is conservative."""

    def setUp(self):
        self.data = _make_data(names={1: "a", 2: "b", 3: "c", 4: "d"})
        self.nbr_sigs: dict[int, set[int]] = {1: {10}, 2: {20}, 3: {30}, 4: {40}}

    def test_converged_always_committed(self):
        hyps = [_hyp(1, convergence="converged")]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        self.assertIn(1, frozen.committed_ids)
        self.assertEqual(len(ambiguous), 0)

    def test_conflicted_never_committed(self):
        hyps = [_hyp(1, convergence="conflicted")]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        self.assertNotIn(1, frozen.committed_ids)
        self.assertEqual(len(ambiguous), 1)

    def test_isolated_low_conf_stays_ambiguous(self):
        v = _vote(confidence=0.6)
        hyps = [_hyp(1, convergence="isolated", votes=[v])]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        self.assertNotIn(1, frozen.committed_ids)
        self.assertEqual(len(ambiguous), 1)

    def test_isolated_no_margin_stays_ambiguous(self):
        """Two isolated with same support_rrf -> no margin -> both stay ambiguous."""
        hyps = [
            _hyp(1, convergence="isolated", support_rrf=0.016, novelty=0.5),
            _hyp(2, convergence="isolated", support_rrf=0.016, novelty=0.5),
        ]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        self.assertEqual(len(frozen.committed_ids), 0)
        self.assertEqual(len(ambiguous), 2)

    def test_isolated_low_novelty_stays_ambiguous(self):
        hyps = [
            _hyp(1, convergence="isolated", support_rrf=0.02, novelty=0.1),
            _hyp(2, convergence="isolated", support_rrf=0.005, novelty=0.5),
        ]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        # Entity 1 has margin but low novelty -> stays ambiguous
        self.assertNotIn(1, frozen.committed_ids)

    def test_isolated_passes_all_three_gets_committed(self):
        hyps = [
            _hyp(1, convergence="isolated", support_rrf=0.02, novelty=0.5),
            _hyp(2, convergence="isolated", support_rrf=0.005, novelty=0.5),
        ]
        frozen, ambiguous = freeze_committed_state(hyps, self.data, self.nbr_sigs)
        # Entity 1: conf=0.9, margin=0.015>0.005, novelty=0.5>0.3
        self.assertIn(1, frozen.committed_ids)


class TestResidualCategoryAccounting(unittest.TestCase):
    """Test category-weighted residual coverage in the resolver."""

    def test_semantic_weighted_higher_than_locality(self):
        self.assertGreater(CATEGORY_GAIN["semantic"], CATEGORY_GAIN["locality"])
        self.assertGreater(CATEGORY_GAIN["structural"], CATEGORY_GAIN["locality"])

    def test_category_gain_ordering(self):
        gains = CATEGORY_GAIN
        self.assertGreaterEqual(gains["semantic"], gains["structural"])
        self.assertGreaterEqual(gains["structural"], gains["constant"])
        self.assertGreaterEqual(gains["constant"], gains["proof"])
        self.assertGreaterEqual(gains["proof"], gains["locality"])

    def test_weighted_residual_prefers_semantic_coverage(self):
        """Entity covering a semantic anchor beats one covering a locality anchor."""
        data = _make_data(
            anchor_categories={10: "semantic", 20: "locality"},
        )
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = LandmarkResidualReport(
            residual_anchor_priorities=[10, 20],
            unsupported_constants=[],
        )
        # Entity A covers semantic anchor 10, entity B covers locality anchor 20
        hyp_a = _hyp(1, convergence="isolated")
        hyp_b = _hyp(2, convergence="isolated")
        nbr_sigs = {1: {10}, 2: {20}}
        idf = {10: 1.0, 20: 1.0}

        resolved = resolve_ambiguous_landmarks(
            frozen, residual, [hyp_a, hyp_b], data, nbr_sigs, idf
        )
        self.assertEqual(resolved[0].entity_id, 1)  # semantic wins


class TestConflictBifurcation(unittest.TestCase):
    """Test that productive conflicts score well, unproductive get penalty."""

    def test_productive_conflict_no_penalty(self):
        """Conflicted entity covering novel semantic territory scores >= 1.0x."""
        data = _make_data(anchor_categories={10: "semantic", 20: "semantic"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = LandmarkResidualReport(
            residual_anchor_priorities=[10, 20],
            conflict_clusters=[(1, 99)],
        )
        hyp = _hyp(1, convergence="conflicted")
        nbr_sigs = {1: {10, 20}}  # covers all uncovered
        idf = {10: 1.0, 20: 1.0}

        resolved = resolve_ambiguous_landmarks(
            frozen, residual, [hyp], data, nbr_sigs, idf
        )
        self.assertEqual(len(resolved), 1)

    def test_unproductive_conflict_steep_penalty(self):
        """Conflicted entity covering nothing gets 0.5x (steeper than old 0.7x)."""
        data = _make_data(anchor_categories={10: "semantic"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = LandmarkResidualReport(
            residual_anchor_priorities=[10],
            conflict_clusters=[(1, 99)],
        )
        # Hyp covers anchor 99, not the uncovered anchor 10
        hyp_conflict = _hyp(1, convergence="conflicted")
        hyp_clean = _hyp(2, convergence="isolated")
        nbr_sigs = {1: {99}, 2: {10}}  # conflict misses, clean hits
        idf = {10: 1.0, 99: 1.0}

        resolved = resolve_ambiguous_landmarks(
            frozen, residual, [hyp_conflict, hyp_clean], data, nbr_sigs, idf
        )
        # Clean entity should rank first
        self.assertEqual(resolved[0].entity_id, 2)


class TestLocalityOnlyDetection(unittest.TestCase):
    """Test that locality-only candidates are detected and penalized."""

    def test_context_only_flagged(self):
        hyp = _hyp(1, votes=[_vote(family="context")])
        data = _make_data(anchor_categories={10: "locality"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = compute_residual(
            frozen, [10], [1.0], data, {10: 1.0}, [hyp], {1: {10}},
        )
        self.assertIn(1, residual.locality_only_candidates)

    def test_bridge_plus_context_not_flagged(self):
        hyp = _hyp(1, votes=[_vote(family="bridge"), _vote(family="context")])
        data = _make_data(anchor_categories={10: "locality"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = compute_residual(
            frozen, [10], [1.0], data, {10: 1.0}, [hyp], {1: {10}},
        )
        self.assertNotIn(1, residual.locality_only_candidates)


class TestExtendFrozenState(unittest.TestCase):
    """Test iterative freeze extension."""

    def test_extends_committed_ids(self):
        data = _make_data(
            names={1: "a.b", 2: "c.d"},
            anchor_categories={10: "constant"},
        )
        frozen = FrozenLandmarkState(
            committed_ids=frozenset({1}),
            committed_scores=((1, 0.5),),
            dominant_anchors=frozenset({10}),
            dominant_constants=frozenset({10}),
            dominant_namespace_clusters=frozenset({"a"}),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        new_hyp = _hyp(2)
        extended = extend_frozen_state(frozen, [new_hyp], data, {2: {20}})
        self.assertIn(1, extended.committed_ids)
        self.assertIn(2, extended.committed_ids)
        self.assertIn(20, extended.dominant_anchors)


class TestPhaseTransition(unittest.TestCase):
    """Test phase signal detection."""

    def test_constant_recovery_phase(self):
        data = _make_data(anchor_categories={10: "constant"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = compute_residual(
            frozen, [10], [1.0], data, {10: 1.0}, [], {},
        )
        self.assertEqual(residual.phase_signal, "constant_recovery")

    def test_structure_matching_phase(self):
        data = _make_data(anchor_categories={10: "structural"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = compute_residual(
            frozen, [10], [1.0], data, {10: 1.0}, [], {},
        )
        self.assertEqual(residual.phase_signal, "structure_matching")

    def test_bridge_dominant_default(self):
        data = _make_data(anchor_categories={10: "locality"})
        frozen = FrozenLandmarkState(
            committed_ids=frozenset(),
            committed_scores=(),
            dominant_anchors=frozenset(),
            dominant_constants=frozenset(),
            dominant_namespace_clusters=frozenset(),
            active_lens_family="bridge",
            neighborhood_bank_centroid=(),
        )
        residual = compute_residual(
            frozen, [10], [1.0], data, {10: 1.0}, [], {},
        )
        # locality-only residual -> bridge_dominant (no primary lens uncovered)
        self.assertEqual(residual.phase_signal, "bridge_dominant")


if __name__ == "__main__":
    unittest.main()
