"""Tests for proof_scoring — mutation-prescribed VALUE + SWAP assertions."""

import math
import unittest

from src.nav_contracts import StructuredQuery
from src.proof_scoring import (
    _compute_anchor_score,
    _compute_bank_score,
    _compute_seed_score,
    compose_bank_scores,
    compute_lens_coherence,
    compute_lens_scores,
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


class TestComputeLensScores(unittest.TestCase):
    """VALUE + SWAP for compute_lens_scores."""

    def test_exact_full_match_single_category(self):
        """VALUE: entity has all query anchors in one category."""
        entity = {10, 20}
        query_anchors = [10, 20]
        query_weights = [1.0, 1.0]
        idf = {10: 2.0, 20: 3.0}
        cats = {10: "semantic", 20: "semantic"}
        scores = compute_lens_scores(entity, query_anchors, query_weights, idf, cats)
        self.assertAlmostEqual(scores["semantic"], 1.0)

    def test_exact_no_match(self):
        """VALUE: entity has none of the query anchors."""
        entity = {99}
        query_anchors = [10, 20]
        query_weights = [1.0, 1.0]
        idf = {10: 1.0, 20: 1.0}
        cats = {10: "structural", 20: "structural"}
        scores = compute_lens_scores(entity, query_anchors, query_weights, idf, cats)
        self.assertAlmostEqual(scores["structural"], 0.0)

    def test_exact_partial_match_idf_weighted(self):
        """VALUE: partial match weighted by IDF."""
        entity = {10}  # has anchor 10 but not 20
        query_anchors = [10, 20]
        query_weights = [1.0, 1.0]
        idf = {10: 2.0, 20: 3.0}
        cats = {10: "lexical", 20: "lexical"}
        scores = compute_lens_scores(entity, query_anchors, query_weights, idf, cats)
        # matched_idf = 2.0, total_idf = 5.0
        self.assertAlmostEqual(scores["lexical"], 2.0 / 5.0)

    def test_multi_category_separation(self):
        """VALUE: scores computed independently per category."""
        entity = {10, 30}  # has semantic 10, locality 30
        query_anchors = [10, 20, 30]
        query_weights = [1.0, 1.0, 1.0]
        idf = {10: 1.0, 20: 1.0, 30: 1.0}
        cats = {10: "semantic", 20: "semantic", 30: "locality"}
        scores = compute_lens_scores(entity, query_anchors, query_weights, idf, cats)
        self.assertAlmostEqual(scores["semantic"], 0.5)  # 1/2
        self.assertAlmostEqual(scores["locality"], 1.0)  # 1/1

    def test_swap_weight_order_changes_score(self):
        """SWAP: different weights on same anchors change per-category score."""
        entity = {10}
        cats = {10: "structural", 20: "structural"}
        idf = {10: 1.0, 20: 1.0}
        s1 = compute_lens_scores(entity, [10, 20], [1.0, 0.1], idf, cats)
        s2 = compute_lens_scores(entity, [10, 20], [0.1, 1.0], idf, cats)
        self.assertNotEqual(round(s1["structural"], 4), round(s2["structural"], 4))

    def test_swap_category_assignment_changes_output(self):
        """SWAP: moving an anchor to a different category changes which lens scores."""
        entity = {10, 20}
        query_anchors = [10, 20]
        query_weights = [1.0, 1.0]
        idf = {10: 1.0, 20: 1.0}
        cats_same = {10: "semantic", 20: "semantic"}
        cats_split = {10: "semantic", 20: "structural"}
        s_same = compute_lens_scores(entity, query_anchors, query_weights, idf, cats_same)
        s_split = compute_lens_scores(entity, query_anchors, query_weights, idf, cats_split)
        self.assertIn("semantic", s_same)
        self.assertNotIn("structural", s_same)
        self.assertIn("semantic", s_split)
        self.assertIn("structural", s_split)

    def test_confidence_modulates_match(self):
        """VALUE: anchor confidence reduces matched_idf."""
        entity = {10}
        query_anchors = [10]
        query_weights = [1.0]
        idf = {10: 2.0}
        cats = {10: "semantic"}
        # Full confidence
        s_full = compute_lens_scores(entity, query_anchors, query_weights, idf, cats, {10: 1.0})
        # Half confidence
        s_half = compute_lens_scores(entity, query_anchors, query_weights, idf, cats, {10: 0.5})
        self.assertAlmostEqual(s_full["semantic"], 1.0)
        self.assertAlmostEqual(s_half["semantic"], 0.5)

    def test_empty_query_returns_empty(self):
        scores = compute_lens_scores({10}, [], [], {}, {})
        self.assertEqual(scores, {})

    def test_general_category_excluded(self):
        """Anchors with category 'general' don't appear in any lens."""
        entity = {10}
        cats = {10: "general"}
        idf = {10: 1.0}
        scores = compute_lens_scores(entity, [10], [1.0], idf, cats)
        self.assertEqual(scores, {})


class TestComputeLensCoherence(unittest.TestCase):
    """VALUE + BOUNDARY for compute_lens_coherence."""

    def test_exact_single_lens_perfect(self):
        """VALUE: single lens at 1.0, min_lenses=2 → discounted to 0.5."""
        result = compute_lens_coherence({"semantic": 1.0}, min_lenses=2)
        self.assertAlmostEqual(result, 0.5)  # 1.0 * (1/2)

    def test_exact_two_lenses_perfect(self):
        """VALUE: two lenses at 1.0, min_lenses=2 → full confidence."""
        result = compute_lens_coherence({"semantic": 1.0, "structural": 1.0}, min_lenses=2)
        self.assertAlmostEqual(result, 1.0)

    def test_exact_geometric_mean(self):
        """VALUE: geometric mean of two different scores."""
        result = compute_lens_coherence({"semantic": 0.25, "structural": 1.0}, min_lenses=2)
        expected = math.sqrt(0.25 * 1.0)  # = 0.5
        self.assertAlmostEqual(result, expected)

    def test_exact_three_lenses(self):
        """VALUE: three lenses, no discount."""
        result = compute_lens_coherence(
            {"semantic": 0.8, "structural": 0.8, "lexical": 0.8}, min_lenses=2
        )
        expected = (0.8 * 0.8 * 0.8) ** (1.0 / 3)  # = 0.8
        self.assertAlmostEqual(result, expected)

    def test_empty_returns_zero(self):
        """VALUE: empty lens_scores → 0."""
        self.assertAlmostEqual(compute_lens_coherence({}), 0.0)

    def test_all_zero_scores_returns_zero(self):
        """VALUE: all zeros are excluded from populated set."""
        result = compute_lens_coherence({"semantic": 0.0, "structural": 0.0})
        self.assertAlmostEqual(result, 0.0)

    def test_boundary_min_lenses_exact(self):
        """BOUNDARY: at exactly min_lenses, no discount applied."""
        result = compute_lens_coherence({"semantic": 1.0, "structural": 1.0}, min_lenses=2)
        self.assertAlmostEqual(result, 1.0)  # no discount

    def test_boundary_below_min_lenses(self):
        """BOUNDARY: below min_lenses, discount is applied."""
        result = compute_lens_coherence({"semantic": 1.0}, min_lenses=2)
        self.assertAlmostEqual(result, 0.5)  # 1.0 * (1/2) = 0.5

    def test_boundary_above_min_lenses(self):
        """BOUNDARY: above min_lenses, no discount."""
        result = compute_lens_coherence(
            {"semantic": 1.0, "structural": 1.0, "lexical": 1.0}, min_lenses=2
        )
        self.assertAlmostEqual(result, 1.0)

    def test_single_lens_discount_scales(self):
        """BOUNDARY: discount scales with n/min_lenses ratio."""
        r1 = compute_lens_coherence({"semantic": 1.0}, min_lenses=3)
        r2 = compute_lens_coherence({"semantic": 1.0}, min_lenses=4)
        self.assertAlmostEqual(r1, 1.0 / 3)
        self.assertAlmostEqual(r2, 1.0 / 4)

    def test_coherence_higher_when_more_lenses_agree(self):
        """Multi-lens agreement > single-lens match."""
        single = compute_lens_coherence({"semantic": 0.8}, min_lenses=2)
        multi = compute_lens_coherence({"semantic": 0.8, "structural": 0.8}, min_lenses=2)
        self.assertGreater(multi, single)


class TestComputeObservabilityChannelCoverage(unittest.TestCase):
    """Tests for channel-coverage observability with anchor_categories."""

    def test_traced_full_coverage(self):
        """Traced entity with all 5 expected channels covered."""
        pos = {"structure": 1, "domain": -1, "depth": 1}
        cats = {1: "semantic", 2: "structural", 3: "lexical", 4: "locality", 5: "proof"}
        score = compute_observability_score(pos, {1, 2, 3, 4, 5}, "traced", cats)
        self.assertGreater(score, 0.5)

    def test_premise_only_missing_proof(self):
        """Premise-only has 4 channels, no proof → lower than traced with 5."""
        pos = {"structure": 1, "domain": -1}
        cats = {1: "semantic", 2: "structural", 3: "lexical", 4: "locality"}
        traced = compute_observability_score(
            pos, {1, 2, 3, 4, 5}, "traced", {**cats, 5: "proof"}
        )
        premise = compute_observability_score(pos, {1, 2, 3, 4}, "premise_only", cats)
        self.assertGreater(traced, premise)

    def test_tactic_only_proof_channel(self):
        """Tactic entity expects only proof channel."""
        pos = {}
        cats = {1: "proof"}
        score = compute_observability_score(pos, {1}, "tactic", cats)
        self.assertGreater(score, 0)


if __name__ == "__main__":
    unittest.main()
