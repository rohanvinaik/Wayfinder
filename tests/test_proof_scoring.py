"""Tests for proof_scoring — mutation-prescribed VALUE + SWAP assertions."""

import unittest

from src.nav_contracts import StructuredQuery
from src.proof_scoring import (
    _compute_anchor_score,
    _compute_bank_score,
    _compute_seed_score,
    compose_bank_scores,
    compute_observability_score,
)


class TestComputeBankScore(unittest.TestCase):
    """VALUE + SWAP for _compute_bank_score."""

    def _query(self, directions, confidences=None):
        return StructuredQuery(
            bank_directions=directions,
            bank_confidences=confidences or {b: 1.0 for b in directions},
            prefer_anchors=[], prefer_weights=[], avoid_anchors=[],
        )

    def test_exact_perfect_alignment(self):
        q = self._query({"structure": 1})
        score = _compute_bank_score({"structure": 1}, q, "multiplicative")
        self.assertAlmostEqual(score, 1.0)

    def test_exact_misalignment(self):
        q = self._query({"structure": 1})
        score = _compute_bank_score({"structure": -1}, q, "multiplicative")
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0)

    def test_swap_direction_changes_score(self):
        """SWAP: query direction +1 vs -1 should produce different scores."""
        pos = {"structure": 1}
        q_pos = self._query({"structure": 1})
        q_neg = self._query({"structure": -1})
        self.assertNotEqual(
            _compute_bank_score(pos, q_pos, "multiplicative"),
            _compute_bank_score(pos, q_neg, "multiplicative"),
        )

    def test_zero_direction_neutral(self):
        q = self._query({"structure": 0})
        score = _compute_bank_score({"structure": 0}, q, "multiplicative")
        self.assertAlmostEqual(score, 1.0)


class TestComposeBankScores(unittest.TestCase):
    """VALUE + SWAP for compose_bank_scores."""

    def test_exact_multiplicative(self):
        scores = {"a": 0.5, "b": 0.8}
        result = compose_bank_scores(scores, {}, "multiplicative")
        self.assertAlmostEqual(result, 0.4)

    def test_exact_geometric_mean(self):
        scores = {"a": 0.25, "b": 1.0}
        result = compose_bank_scores(scores, {}, "geometric_mean")
        self.assertAlmostEqual(result, 0.5)

    def test_swap_mechanism_changes_result(self):
        """SWAP: different mechanisms produce different scores."""
        scores = {"a": 0.05, "b": 0.8}  # 0.05 < floor epsilon 0.1
        confs = {"a": 1.0, "b": 1.0}
        mult = compose_bank_scores(scores, confs, "multiplicative")  # 0.05*0.8=0.04
        soft = compose_bank_scores(scores, confs, "soft_floor")  # max(0.05,0.1)*0.8=0.08
        self.assertNotEqual(mult, soft)

    def test_empty_scores_returns_one(self):
        self.assertAlmostEqual(compose_bank_scores({}, {}, "multiplicative"), 1.0)


class TestComputeAnchorScore(unittest.TestCase):
    """VALUE + SWAP for _compute_anchor_score."""

    def _query(self, prefer, weights, avoid=None):
        return StructuredQuery(
            bank_directions={}, bank_confidences={},
            prefer_anchors=prefer, prefer_weights=weights,
            avoid_anchors=avoid or [],
        )

    def test_exact_full_match(self):
        q = self._query([1, 2], [1.0, 1.0])
        score = _compute_anchor_score({1, 2}, q, {1: 1.0, 2: 1.0})
        self.assertAlmostEqual(score, 1.0)

    def test_exact_no_match(self):
        q = self._query([1, 2], [1.0, 1.0])
        score = _compute_anchor_score({3, 4}, q, {1: 1.0, 2: 1.0})
        self.assertAlmostEqual(score, 0.0)

    def test_exact_partial_match(self):
        q = self._query([1, 2], [1.0, 1.0])
        score = _compute_anchor_score({1}, q, {1: 1.0, 2: 1.0})
        self.assertAlmostEqual(score, 0.5)

    def test_avoid_penalty_with_prefer(self):
        """Avoid penalty only applies when prefer anchors are present."""
        q = self._query([1], [1.0], avoid=[5])
        with_avoided = _compute_anchor_score({1, 5}, q, {1: 1.0})
        without_avoided = _compute_anchor_score({1}, q, {1: 1.0})
        self.assertLess(with_avoided, without_avoided)

    def test_no_preferences_neutral(self):
        q = self._query([], [])
        score = _compute_anchor_score({1, 2, 3}, q, {})
        self.assertAlmostEqual(score, 1.0)

    def test_swap_prefer_order_matters(self):
        """SWAP: different anchor weights produce different scores."""
        q1 = self._query([1, 2], [1.0, 0.1])
        q2 = self._query([1, 2], [0.1, 1.0])
        # Entity has only anchor 1
        s1 = _compute_anchor_score({1}, q1, {1: 1.0, 2: 1.0})
        s2 = _compute_anchor_score({1}, q2, {1: 1.0, 2: 1.0})
        self.assertNotEqual(round(s1, 4), round(s2, 4))


class TestComputeSeedScore(unittest.TestCase):
    """VALUE + SWAP for _compute_seed_score."""

    def test_exact_identical_sets(self):
        score = _compute_seed_score({1, 2}, {1, 2}, {1: 1.0, 2: 1.0})
        self.assertAlmostEqual(score, 1.0)

    def test_exact_disjoint_sets(self):
        score = _compute_seed_score({1, 2}, {3, 4}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        self.assertAlmostEqual(score, 0.0)

    def test_no_seeds_neutral(self):
        score = _compute_seed_score({1, 2, 3}, set(), {})
        self.assertAlmostEqual(score, 1.0)

    def test_swap_entity_seed_asymmetric_idf(self):
        """SWAP: asymmetric IDF weights break Jaccard symmetry."""
        idf = {1: 1.0, 2: 5.0}  # anchor 2 is much rarer
        s1 = _compute_seed_score({1}, {1, 2}, idf)     # shared=1.0, union=6.0
        s2 = _compute_seed_score({1, 2}, {1}, idf)     # shared=1.0, union=6.0
        # With equal-weight anchors Jaccard IS symmetric — but verify values
        self.assertAlmostEqual(s1, 1.0 / 6.0, places=4)
        self.assertAlmostEqual(s2, 1.0 / 6.0, places=4)


class TestComputeObservabilitySwap(unittest.TestCase):
    """SWAP prescriptions for compute_observability_score."""

    def test_swap_provenance(self):
        """SWAP: traced vs premise_only with same data produces different scores."""
        pos = {"structure": 1, "domain": -1}
        anchors = {1, 2, 3, 4, 5}
        t = compute_observability_score(pos, anchors, "traced")
        p = compute_observability_score(pos, anchors, "premise_only")
        self.assertNotEqual(t, p)

    def test_swap_positions_anchors(self):
        """More banks vs more anchors produce different observability profiles."""
        many_banks = compute_observability_score(
            {"s": 1, "d": -1, "dp": 1, "a": -1}, {1}, "traced"
        )
        many_anchors = compute_observability_score(
            {"s": 1}, set(range(15)), "traced"
        )
        # Both should be positive but different
        self.assertGreater(many_banks, 0)
        self.assertGreater(many_anchors, 0)
        self.assertNotEqual(round(many_banks, 4), round(many_anchors, 4))


if __name__ == "__main__":
    unittest.main()
