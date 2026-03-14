"""Characterization tests for proof_network scoring functions.

Tests compose_bank_scores, bank_score, _score_candidates, and
the internal scoring helpers to close the mutation testing gap.
"""

import unittest

from src.nav_contracts import BANK_NAMES, ScoredEntity, StructuredQuery
from src.proof_network import (
    _compute_anchor_score,
    _compute_bank_score,
    _compute_seed_score,
    _score_candidates,
    bank_score,
    compose_bank_scores,
)


class TestBankScore(unittest.TestCase):
    """Test bank_score: single bank alignment scoring."""

    def test_zero_query_zero_entity(self):
        # query doesn't care, entity at origin → 1/(1+0) = 1.0
        self.assertAlmostEqual(bank_score(0, 0), 1.0)

    def test_zero_query_positive_entity(self):
        # query doesn't care, entity at +2 → 1/(1+2) = 1/3
        self.assertAlmostEqual(bank_score(2, 0), 1.0 / 3.0)

    def test_zero_query_negative_entity(self):
        # query doesn't care, entity at -3 → 1/(1+3) = 0.25
        self.assertAlmostEqual(bank_score(-3, 0), 0.25)

    def test_aligned_positive(self):
        # entity +2, query +1 → alignment=2 > 0 → 1.0
        self.assertAlmostEqual(bank_score(2, 1), 1.0)

    def test_aligned_negative(self):
        # entity -1, query -1 → alignment=1 > 0 → 1.0
        self.assertAlmostEqual(bank_score(-1, -1), 1.0)

    def test_neutral_entity_positive_query(self):
        # entity 0, query +1 → alignment=0 → 0.5
        self.assertAlmostEqual(bank_score(0, 1), 0.5)

    def test_neutral_entity_negative_query(self):
        # entity 0, query -1 → alignment=0 → 0.5
        self.assertAlmostEqual(bank_score(0, -1), 0.5)

    def test_misaligned(self):
        # entity +1, query -1 → alignment=-1 < 0 → 1/(1+1) = 0.5
        self.assertAlmostEqual(bank_score(1, -1), 0.5)

    def test_strongly_misaligned(self):
        # entity +3, query -1 → alignment=-3 < 0 → 1/(1+3) = 0.25
        self.assertAlmostEqual(bank_score(3, -1), 0.25)

    def test_opposite_deep(self):
        # entity -2, query +1 → alignment=-2 < 0 → 1/(1+2) = 1/3
        self.assertAlmostEqual(bank_score(-2, 1), 1.0 / 3.0)


class TestComposeBankScores(unittest.TestCase):
    """Test compose_bank_scores: all 5 scoring mechanisms."""

    def test_empty_scores(self):
        """Empty scores dict → 1.0 for all mechanisms."""
        for mech in [
            "multiplicative",
            "confidence_weighted",
            "soft_floor",
            "geometric_mean",
            "log_additive",
        ]:
            self.assertAlmostEqual(compose_bank_scores({}, {}, mech), 1.0, msg=mech)

    # --- multiplicative ---

    def test_multiplicative_single(self):
        scores = {"structure": 0.7}
        result = compose_bank_scores(scores, {}, "multiplicative")
        self.assertAlmostEqual(result, 0.7)

    def test_multiplicative_two_banks(self):
        scores = {"structure": 0.5, "domain": 0.4}
        result = compose_bank_scores(scores, {}, "multiplicative")
        self.assertAlmostEqual(result, 0.2)

    def test_multiplicative_with_one(self):
        scores = {"structure": 1.0, "domain": 0.8}
        result = compose_bank_scores(scores, {}, "multiplicative")
        self.assertAlmostEqual(result, 0.8)

    def test_multiplicative_with_zero(self):
        scores = {"structure": 0.0, "domain": 0.9}
        result = compose_bank_scores(scores, {}, "multiplicative")
        self.assertAlmostEqual(result, 0.0)

    # --- confidence_weighted ---

    def test_confidence_weighted_full_confidence(self):
        scores = {"structure": 0.5, "domain": 0.4}
        confs = {"structure": 1.0, "domain": 1.0}
        result = compose_bank_scores(scores, confs, "confidence_weighted")
        # same as multiplicative when all confidences = 1
        self.assertAlmostEqual(result, 0.5 * 0.4)

    def test_confidence_weighted_zero_confidence(self):
        scores = {"structure": 0.5, "domain": 0.4}
        confs = {"structure": 0.0, "domain": 0.0}
        result = compose_bank_scores(scores, confs, "confidence_weighted")
        # s^0 = 1.0 for all → product = 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_confidence_weighted_partial_confidence(self):
        scores = {"structure": 0.25}
        confs = {"structure": 0.5}
        result = compose_bank_scores(scores, confs, "confidence_weighted")
        # 0.25^0.5 = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_confidence_weighted_missing_confidence_defaults_to_one(self):
        scores = {"structure": 0.5}
        confs = {}  # no confidence entry → defaults to 1.0
        result = compose_bank_scores(scores, confs, "confidence_weighted")
        # 0.5^1.0 = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_confidence_weighted_multi_bank(self):
        scores = {"structure": 0.5, "domain": 1.0, "depth": 0.25}
        confs = {"structure": 1.0, "domain": 0.5, "depth": 0.5}
        result = compose_bank_scores(scores, confs, "confidence_weighted")
        # 0.5^1.0 * 1.0^0.5 * 0.25^0.5 = 0.5 * 1.0 * 0.5 = 0.25
        self.assertAlmostEqual(result, 0.25)

    # --- soft_floor ---

    def test_soft_floor_above_epsilon(self):
        scores = {"structure": 0.5, "domain": 0.3}
        result = compose_bank_scores(scores, {}, "soft_floor", floor_epsilon=0.1)
        # Both above 0.1 → product unchanged
        self.assertAlmostEqual(result, 0.15)

    def test_soft_floor_clamps_zero(self):
        scores = {"structure": 0.0, "domain": 0.5}
        result = compose_bank_scores(scores, {}, "soft_floor", floor_epsilon=0.1)
        # max(0.0, 0.1) * max(0.5, 0.1) = 0.1 * 0.5 = 0.05
        self.assertAlmostEqual(result, 0.05)

    def test_soft_floor_clamps_below_epsilon(self):
        scores = {"structure": 0.05, "domain": 0.5}
        result = compose_bank_scores(scores, {}, "soft_floor", floor_epsilon=0.1)
        # max(0.05, 0.1) * max(0.5, 0.1) = 0.1 * 0.5 = 0.05
        self.assertAlmostEqual(result, 0.05)

    # --- geometric_mean ---

    def test_geometric_mean_single(self):
        scores = {"structure": 0.64}
        result = compose_bank_scores(scores, {}, "geometric_mean")
        # 0.64^(1/1) = 0.64
        self.assertAlmostEqual(result, 0.64)

    def test_geometric_mean_two(self):
        scores = {"structure": 0.25, "domain": 1.0}
        result = compose_bank_scores(scores, {}, "geometric_mean")
        # (0.25 * 1.0)^(1/2) = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_geometric_mean_three(self):
        scores = {"structure": 0.5, "domain": 0.5, "depth": 0.5}
        result = compose_bank_scores(scores, {}, "geometric_mean")
        # (0.125)^(1/3) = 0.5
        self.assertAlmostEqual(result, 0.5)

    # --- log_additive ---

    def test_log_additive_single(self):
        scores = {"structure": 0.5}
        result = compose_bank_scores(scores, {}, "log_additive")
        # exp(log(0.5)) = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_log_additive_two(self):
        scores = {"structure": 0.5, "domain": 0.4}
        result = compose_bank_scores(scores, {}, "log_additive")
        # exp(log(0.5) + log(0.4)) = 0.5 * 0.4 = 0.2
        self.assertAlmostEqual(result, 0.2, places=5)

    def test_log_additive_zero_clamped(self):
        scores = {"structure": 0.0, "domain": 0.5}
        result = compose_bank_scores(scores, {}, "log_additive")
        # 0.0 clamped to 1e-6 → exp(log(1e-6) + log(0.5)) ≈ 5e-7
        self.assertAlmostEqual(result, 5e-7, places=12)

    # --- error ---

    def test_unknown_mechanism_raises(self):
        with self.assertRaises(ValueError):
            compose_bank_scores({"a": 0.5}, {}, "bogus_method")


class TestComputeBankScore(unittest.TestCase):
    """Test _compute_bank_score: entity positions × query → compose."""

    def test_no_positions_no_directions(self):
        query = StructuredQuery(
            bank_directions={b: 0 for b in BANK_NAMES},
            bank_confidences={b: 1.0 for b in BANK_NAMES},
        )
        # No positions, zero directions → all skipped → empty scores → 1.0
        result = _compute_bank_score({}, query, "multiplicative")
        self.assertAlmostEqual(result, 1.0)

    def test_aligned_single_bank(self):
        query = StructuredQuery(
            bank_directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            bank_confidences={b: 1.0 for b in BANK_NAMES},
        )
        positions = {"structure": 2}  # aligned with +1 query
        result = _compute_bank_score(positions, query, "multiplicative")
        # bank_score(2, 1) = 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_missing_position_gets_penalty(self):
        query = StructuredQuery(
            bank_directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            bank_confidences={b: 1.0 for b in BANK_NAMES},
        )
        # No position for structure, but query wants +1 → _MISSING_BANK_SCORE = 0.3
        result = _compute_bank_score({}, query, "multiplicative")
        self.assertAlmostEqual(result, 0.3)


class TestComputeAnchorScore(unittest.TestCase):
    """Test _compute_anchor_score: IDF-weighted anchor matching."""

    def test_no_prefer_anchors(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[],
            prefer_weights=[],
        )
        result = _compute_anchor_score({1, 2, 3}, query, {})
        self.assertAlmostEqual(result, 1.0)

    def test_full_match(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10, 20],
            prefer_weights=[1.0, 1.0],
        )
        idf = {10: 2.0, 20: 3.0}
        entity_anchors = {10, 20, 30}
        result = _compute_anchor_score(entity_anchors, query, idf)
        # matched_idf = 2.0 + 3.0 = 5.0, total_idf = 5.0 → 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_partial_match(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10, 20],
            prefer_weights=[1.0, 1.0],
        )
        idf = {10: 2.0, 20: 3.0}
        entity_anchors = {10}  # only matches anchor 10
        result = _compute_anchor_score(entity_anchors, query, idf)
        # matched_idf = 2.0, total_idf = 5.0 → 0.4
        self.assertAlmostEqual(result, 0.4)

    def test_no_match(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10, 20],
            prefer_weights=[1.0, 1.0],
        )
        idf = {10: 2.0, 20: 3.0}
        entity_anchors = {99}
        result = _compute_anchor_score(entity_anchors, query, idf)
        # matched_idf = 0, total_idf = 5.0 → 0.0
        self.assertAlmostEqual(result, 0.0)

    def test_avoid_anchor_penalty(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10],
            prefer_weights=[1.0],
            avoid_anchors=[20],
        )
        idf = {10: 1.0}
        entity_anchors = {10, 20}  # matches prefer AND avoid
        result = _compute_anchor_score(entity_anchors, query, idf)
        # matched_idf/total_idf = 1.0 * avoid_penalty(0.5) = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_multiple_avoid_anchors(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10],
            prefer_weights=[1.0],
            avoid_anchors=[20, 30],
        )
        idf = {10: 1.0}
        entity_anchors = {10, 20, 30}  # hits both avoid anchors
        result = _compute_anchor_score(entity_anchors, query, idf)
        # 1.0 * 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(result, 0.25)

    def test_weights_scale_idf(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10, 20],
            prefer_weights=[2.0, 0.5],
        )
        idf = {10: 1.0, 20: 1.0}
        entity_anchors = {10}
        result = _compute_anchor_score(entity_anchors, query, idf)
        # matched_idf = 1.0*2.0 = 2.0, total_idf = 1.0*2.0 + 1.0*0.5 = 2.5 → 0.8
        self.assertAlmostEqual(result, 0.8)

    def test_missing_idf_defaults_to_one(self):
        query = StructuredQuery(
            bank_directions={},
            bank_confidences={},
            prefer_anchors=[10],
            prefer_weights=[1.0],
        )
        entity_anchors = {10}
        result = _compute_anchor_score(entity_anchors, query, {})
        # idf.get(10, 1.0) = 1.0, matched = 1.0, total = 1.0 → 1.0
        self.assertAlmostEqual(result, 1.0)


class TestComputeSeedScore(unittest.TestCase):
    """Test _compute_seed_score: IDF-weighted Jaccard with seeds."""

    def test_no_seeds(self):
        result = _compute_seed_score({1, 2}, set(), {})
        self.assertAlmostEqual(result, 1.0)

    def test_perfect_overlap(self):
        idf = {1: 2.0, 2: 3.0}
        result = _compute_seed_score({1, 2}, {1, 2}, idf)
        # shared = {1,2}, union = {1,2} → 5.0/5.0 = 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_no_overlap(self):
        idf = {1: 2.0, 2: 3.0, 3: 1.0, 4: 1.0}
        result = _compute_seed_score({1, 2}, {3, 4}, idf)
        # shared = empty → 0, union = {1,2,3,4} → 0/7.0 = 0.0
        self.assertAlmostEqual(result, 0.0)

    def test_partial_overlap(self):
        idf = {1: 2.0, 2: 3.0, 3: 1.0}
        result = _compute_seed_score({1, 2}, {2, 3}, idf)
        # shared = {2} → idf=3.0, union = {1,2,3} → 2+3+1=6.0 → 3.0/6.0 = 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_empty_entity_anchors(self):
        idf = {1: 2.0}
        result = _compute_seed_score(set(), {1}, idf)
        # shared = empty, union = {1} → 0/2.0 = 0.0
        self.assertAlmostEqual(result, 0.0)


class TestScoreCandidates(unittest.TestCase):
    """Test _score_candidates: full scoring pipeline (no DB)."""

    def _make_query(self, **kwargs):
        defaults = {
            "bank_directions": {b: 0 for b in BANK_NAMES},
            "bank_confidences": {b: 1.0 for b in BANK_NAMES},
        }
        defaults.update(kwargs)
        return StructuredQuery(**defaults)

    def test_empty_candidates(self):
        query = self._make_query()
        result = _score_candidates([], {}, {}, {}, {}, query, set(), "multiplicative")
        self.assertEqual(result, [])

    def test_single_candidate_basic(self):
        query = self._make_query()
        result = _score_candidates(
            candidate_ids=[1],
            positions={},
            entity_anchor_sets={},
            idf_cache={},
            names={1: "Nat.add"},
            query=query,
            seed_anchors=set(),
            mechanism="multiplicative",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Nat.add")
        self.assertEqual(result[0].entity_id, 1)
        # All neutral → 1.0 * 1.0 * 1.0
        self.assertAlmostEqual(result[0].final_score, 1.0)

    def test_scoring_ranks_by_alignment(self):
        query = self._make_query(
            bank_directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
        )
        result = _score_candidates(
            candidate_ids=[1, 2],
            positions={
                1: {"structure": 2},  # aligned → bank_score = 1.0
                2: {"structure": -2},  # misaligned → bank_score = 1/3
            },
            entity_anchor_sets={},
            idf_cache={},
            names={1: "good", 2: "bad"},
            query=query,
            seed_anchors=set(),
            mechanism="multiplicative",
        )
        self.assertEqual(len(result), 2)
        # Entity 1 should rank first (higher score)
        self.assertEqual(result[0].name, "good")
        self.assertGreater(result[0].final_score, result[1].final_score)

    def test_zero_score_excluded(self):
        """Candidates with final_score=0 are not included."""
        query = self._make_query(
            prefer_anchors=[99],
            prefer_weights=[1.0],
        )
        result = _score_candidates(
            candidate_ids=[1],
            positions={},
            entity_anchor_sets={1: set()},  # no matching anchors
            idf_cache={99: 1.0},
            names={1: "no_match"},
            query=query,
            seed_anchors=set(),
            mechanism="multiplicative",
        )
        # anchor_score = 0.0 → final = 0.0 → excluded
        self.assertEqual(result, [])

    def test_final_score_is_product_of_three(self):
        """final_score = bank_score * anchor_score * seed_score."""
        query = self._make_query(
            bank_directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            prefer_anchors=[10],
            prefer_weights=[1.0],
        )
        idf = {10: 1.0, 1: 1.0, 2: 1.0}
        result = _score_candidates(
            candidate_ids=[1],
            positions={1: {"structure": 2}},
            entity_anchor_sets={1: {10, 1, 2}},  # include 2 so seed overlap > 0
            idf_cache=idf,
            names={1: "test"},
            query=query,
            seed_anchors={2},
            mechanism="multiplicative",
        )
        self.assertEqual(len(result), 1)
        r = result[0]
        self.assertAlmostEqual(r.final_score, r.bank_score * r.anchor_score * r.seed_score)

    def test_sorted_descending(self):
        query = self._make_query()
        result = _score_candidates(
            candidate_ids=[1, 2, 3],
            positions={},
            entity_anchor_sets={},
            idf_cache={},
            names={1: "a", 2: "b", 3: "c"},
            query=query,
            seed_anchors=set(),
            mechanism="multiplicative",
        )
        scores = [r.final_score for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_returns_scored_entity_type(self):
        query = self._make_query()
        result = _score_candidates(
            candidate_ids=[1],
            positions={},
            entity_anchor_sets={},
            idf_cache={},
            names={1: "test"},
            query=query,
            seed_anchors=set(),
            mechanism="multiplicative",
        )
        self.assertIsInstance(result[0], ScoredEntity)
        self.assertIsInstance(result[0].bank_score, float)
        self.assertIsInstance(result[0].anchor_score, float)
        self.assertIsInstance(result[0].seed_score, float)


if __name__ == "__main__":
    unittest.main()
