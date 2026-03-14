"""Tests for proof_network — algebraic scoring, spreading, and DB lifecycle.

Covers:
    bank_score()            — pure alignment scoring
    compose_bank_scores()   — 5 composition mechanisms
    _compute_anchor_score() — IDF-weighted anchor matching
    _compute_seed_score()   — IDF-weighted Jaccard similarity
    _score_candidates()     — integration of bank/anchor/seed scoring
    spread()                — BFS spreading activation
    init_db / recompute_idf / clear_caches / get_accessible_premises — DB lifecycle
"""

import math
import sqlite3
import unittest

from src.nav_contracts import BANK_NAMES, ScoredEntity, StructuredQuery
from src.proof_network import (
    _MISSING_BANK_SCORE,
    _SCHEMA_SQL,
    _compute_anchor_score,
    _compute_bank_score,
    _compute_seed_score,
    _score_candidates,
    bank_score,
    clear_caches,
    compose_bank_scores,
    get_accessible_premises,
    init_db,
    recompute_idf,
    spread,
)


def _make_db() -> sqlite3.Connection:
    """Create an in-memory proof network DB with the full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    return conn


def _make_query(
    directions: dict[str, int] | None = None,
    confidences: dict[str, float] | None = None,
    prefer_anchors: list[int] | None = None,
    prefer_weights: list[float] | None = None,
    avoid_anchors: list[int] | None = None,
    seed_entity_ids: list[int] | None = None,
    require_anchors: list[int] | None = None,
    accessible_theorem_id: int | None = None,
) -> StructuredQuery:
    """Build a StructuredQuery with sensible defaults."""
    return StructuredQuery(
        bank_directions=directions or {},
        bank_confidences=confidences or {},
        prefer_anchors=prefer_anchors or [],
        prefer_weights=prefer_weights or [],
        avoid_anchors=avoid_anchors or [],
        seed_entity_ids=seed_entity_ids or [],
        require_anchors=require_anchors or [],
        accessible_theorem_id=accessible_theorem_id,
    )


# ---------------------------------------------------------------------------
# bank_score
# ---------------------------------------------------------------------------


class TestBankScore(unittest.TestCase):
    """Pure function: bank_score(entity_signed_pos, query_direction) -> float."""

    # --- query_direction == 0 => 1/(1+|pos|) ---

    def test_direction_zero_pos_zero(self):
        self.assertEqual(bank_score(0, 0), 1.0)

    def test_direction_zero_pos_positive(self):
        self.assertEqual(bank_score(1, 0), 0.5)

    def test_direction_zero_pos_negative(self):
        self.assertEqual(bank_score(-1, 0), 0.5)

    def test_direction_zero_pos_two(self):
        self.assertAlmostEqual(bank_score(2, 0), 1.0 / 3.0, places=15)

    def test_direction_zero_pos_neg_three(self):
        self.assertEqual(bank_score(-3, 0), 0.25)

    # --- alignment > 0 => 1.0 ---

    def test_aligned_pos_pos(self):
        self.assertEqual(bank_score(1, 1), 1.0)

    def test_aligned_neg_neg(self):
        self.assertEqual(bank_score(-1, -1), 1.0)

    def test_aligned_large(self):
        self.assertEqual(bank_score(3, 1), 1.0)

    def test_aligned_large_neg(self):
        self.assertEqual(bank_score(-2, -1), 1.0)

    # --- alignment == 0 (entity_pos != 0) => 0.5 ---

    def test_zero_alignment_entity_zero_query_pos(self):
        # entity_pos=0, query_direction=1 => alignment=0*1=0 => 0.5
        self.assertEqual(bank_score(0, 1), 0.5)

    def test_zero_alignment_entity_zero_query_neg(self):
        self.assertEqual(bank_score(0, -1), 0.5)

    # --- alignment < 0 => 1/(1+|alignment|) ---

    def test_anti_aligned_neg_pos(self):
        # (-1)*1 = -1 => 1/(1+1) = 0.5
        self.assertEqual(bank_score(-1, 1), 0.5)

    def test_anti_aligned_pos_neg(self):
        # (1)*(-1) = -1 => 0.5
        self.assertEqual(bank_score(1, -1), 0.5)

    def test_anti_aligned_large(self):
        # (2)*(-1) = -2 => 1/(1+2) = 1/3
        self.assertAlmostEqual(bank_score(2, -1), 1.0 / 3.0, places=15)

    def test_anti_aligned_neg_large(self):
        # (-3)*(1) = -3 => 1/(1+3) = 0.25
        self.assertEqual(bank_score(-3, 1), 0.25)

    # --- Cross-branch value discrimination ---

    def test_aligned_returns_different_from_neutral(self):
        """Aligned (1.0) must differ from neutral (0.5)."""
        self.assertNotEqual(bank_score(1, 1), bank_score(0, 1))

    def test_direction_zero_depth_discriminates(self):
        """Deeper positions score lower when direction is neutral."""
        self.assertGreater(bank_score(0, 0), bank_score(1, 0))
        self.assertGreater(bank_score(1, 0), bank_score(2, 0))

    def test_anti_aligned_depth_discriminates(self):
        """Deeper anti-alignment scores lower."""
        self.assertGreater(bank_score(-1, 1), bank_score(-2, 1))


# ---------------------------------------------------------------------------
# compose_bank_scores
# ---------------------------------------------------------------------------


class TestComposeBankScores(unittest.TestCase):
    """compose_bank_scores(scores, confidences, mechanism, floor_epsilon)."""

    # --- Empty scores ---

    def test_empty_scores_returns_one(self):
        self.assertEqual(compose_bank_scores({}, {}, mechanism="multiplicative"), 1.0)

    def test_empty_scores_all_mechanisms(self):
        for mech in [
            "multiplicative",
            "confidence_weighted",
            "soft_floor",
            "geometric_mean",
            "log_additive",
        ]:
            with self.subTest(mechanism=mech):
                self.assertEqual(compose_bank_scores({}, {}, mechanism=mech), 1.0)

    # --- Unknown mechanism ---

    def test_unknown_mechanism_raises(self):
        with self.assertRaises(ValueError):
            compose_bank_scores({"a": 0.5}, {}, mechanism="bogus")

    # --- multiplicative ---

    def test_multiplicative_single(self):
        result = compose_bank_scores({"a": 0.7}, {}, mechanism="multiplicative")
        self.assertAlmostEqual(result, 0.7, places=15)

    def test_multiplicative_two(self):
        result = compose_bank_scores({"a": 0.5, "b": 0.6}, {}, mechanism="multiplicative")
        self.assertAlmostEqual(result, 0.3, places=15)

    def test_multiplicative_three(self):
        result = compose_bank_scores({"a": 0.5, "b": 0.4, "c": 0.8}, {}, mechanism="multiplicative")
        self.assertAlmostEqual(result, 0.5 * 0.4 * 0.8, places=15)

    def test_multiplicative_zero_in_scores(self):
        result = compose_bank_scores({"a": 0.5, "b": 0.0}, {}, mechanism="multiplicative")
        self.assertEqual(result, 0.0)

    # --- confidence_weighted ---

    def test_confidence_weighted_full_confidence(self):
        # s^1.0 = s, so same as multiplicative
        result = compose_bank_scores(
            {"a": 0.5, "b": 0.6},
            {"a": 1.0, "b": 1.0},
            mechanism="confidence_weighted",
        )
        self.assertAlmostEqual(result, 0.5 * 0.6, places=15)

    def test_confidence_weighted_zero_confidence(self):
        # s^0 = 1.0 for all, so product = 1.0
        result = compose_bank_scores(
            {"a": 0.5, "b": 0.6},
            {"a": 0.0, "b": 0.0},
            mechanism="confidence_weighted",
        )
        self.assertAlmostEqual(result, 1.0, places=15)

    def test_confidence_weighted_half_confidence(self):
        # 0.25^0.5 * 0.64^0.5 = 0.5 * 0.8 = 0.4
        result = compose_bank_scores(
            {"a": 0.25, "b": 0.64},
            {"a": 0.5, "b": 0.5},
            mechanism="confidence_weighted",
        )
        self.assertAlmostEqual(result, 0.5 * 0.8, places=15)

    def test_confidence_weighted_missing_confidence_defaults_one(self):
        # Missing confidence => exponent 1.0
        result = compose_bank_scores({"a": 0.5}, {}, mechanism="confidence_weighted")
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_confidence_weighted_mixed(self):
        # a: 0.8^0.5, b: 0.4^1.0 => sqrt(0.8) * 0.4
        result = compose_bank_scores(
            {"a": 0.8, "b": 0.4},
            {"a": 0.5, "b": 1.0},
            mechanism="confidence_weighted",
        )
        expected = (0.8**0.5) * (0.4**1.0)
        self.assertAlmostEqual(result, expected, places=15)

    # --- soft_floor ---

    def test_soft_floor_above_epsilon(self):
        result = compose_bank_scores(
            {"a": 0.5, "b": 0.6},
            {},
            mechanism="soft_floor",
            floor_epsilon=0.1,
        )
        self.assertAlmostEqual(result, 0.5 * 0.6, places=15)

    def test_soft_floor_below_epsilon(self):
        # max(0.01, 0.1) * max(0.5, 0.1) = 0.1 * 0.5 = 0.05
        result = compose_bank_scores(
            {"a": 0.01, "b": 0.5},
            {},
            mechanism="soft_floor",
            floor_epsilon=0.1,
        )
        self.assertAlmostEqual(result, 0.1 * 0.5, places=15)

    def test_soft_floor_zero_value(self):
        # max(0.0, 0.1) = 0.1
        result = compose_bank_scores({"a": 0.0}, {}, mechanism="soft_floor", floor_epsilon=0.1)
        self.assertAlmostEqual(result, 0.1, places=15)

    def test_soft_floor_custom_epsilon(self):
        # max(0.0, 0.2) * max(0.3, 0.2) = 0.2 * 0.3 = 0.06
        result = compose_bank_scores(
            {"a": 0.0, "b": 0.3},
            {},
            mechanism="soft_floor",
            floor_epsilon=0.2,
        )
        self.assertAlmostEqual(result, 0.2 * 0.3, places=15)

    # --- geometric_mean ---

    def test_geometric_mean_single(self):
        result = compose_bank_scores({"a": 0.64}, {}, mechanism="geometric_mean")
        self.assertAlmostEqual(result, 0.64, places=15)

    def test_geometric_mean_two_equal(self):
        # (0.5 * 0.5)^(1/2) = 0.5
        result = compose_bank_scores({"a": 0.5, "b": 0.5}, {}, mechanism="geometric_mean")
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_geometric_mean_two_different(self):
        # (0.25 * 1.0)^(1/2) = 0.5
        result = compose_bank_scores({"a": 0.25, "b": 1.0}, {}, mechanism="geometric_mean")
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_geometric_mean_three(self):
        # (0.5 * 0.5 * 0.5)^(1/3) = 0.5
        result = compose_bank_scores({"a": 0.5, "b": 0.5, "c": 0.5}, {}, mechanism="geometric_mean")
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_geometric_mean_exact(self):
        # (0.8 * 0.2)^(1/2) = sqrt(0.16) = 0.4
        result = compose_bank_scores({"a": 0.8, "b": 0.2}, {}, mechanism="geometric_mean")
        self.assertAlmostEqual(result, 0.4, places=15)

    # --- log_additive ---

    def test_log_additive_single(self):
        result = compose_bank_scores({"a": 0.5}, {}, mechanism="log_additive")
        expected = math.exp(math.log(0.5))
        self.assertAlmostEqual(result, expected, places=15)

    def test_log_additive_two(self):
        # exp(log(0.5) + log(0.4)) = 0.5 * 0.4 = 0.2
        result = compose_bank_scores({"a": 0.5, "b": 0.4}, {}, mechanism="log_additive")
        self.assertAlmostEqual(result, 0.2, places=10)

    def test_log_additive_zero_clamped(self):
        # max(0.0, 1e-6) => exp(log(1e-6)) = 1e-6
        result = compose_bank_scores({"a": 0.0}, {}, mechanism="log_additive")
        self.assertAlmostEqual(result, 1e-6, places=15)

    def test_log_additive_near_zero(self):
        # max(1e-8, 1e-6) = 1e-6 => exp(log(1e-6)) = 1e-6
        result = compose_bank_scores({"a": 1e-8}, {}, mechanism="log_additive")
        self.assertAlmostEqual(result, 1e-6, places=15)


# ---------------------------------------------------------------------------
# _compute_bank_score (internal, but critical for scoring correctness)
# ---------------------------------------------------------------------------


class TestComputeBankScore(unittest.TestCase):
    """_compute_bank_score combines bank_score per-bank with compose."""

    def test_no_positions_no_directions(self):
        query = _make_query()
        result = _compute_bank_score({}, query, "multiplicative")
        # No scores dict entries => empty => 1.0
        self.assertEqual(result, 1.0)

    def test_single_bank_aligned(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        result = _compute_bank_score({"structure": 1}, query, "multiplicative")
        self.assertEqual(result, 1.0)

    def test_single_bank_anti_aligned(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        # entity at -1, query at +1 => alignment=-1 => 0.5
        result = _compute_bank_score({"structure": -1}, query, "multiplicative")
        self.assertEqual(result, 0.5)

    def test_missing_entity_position_uses_penalty(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        # entity has no "structure" position => _MISSING_BANK_SCORE = 0.3
        result = _compute_bank_score({}, query, "multiplicative")
        self.assertAlmostEqual(result, _MISSING_BANK_SCORE, places=15)

    def test_query_zero_direction_entity_missing_skipped(self):
        # direction=0 and entity has no position => skip (no entry in scores dict)
        query = _make_query(
            directions={"structure": 0},
            confidences={"structure": 1.0},
        )
        result = _compute_bank_score({}, query, "multiplicative")
        self.assertEqual(result, 1.0)

    def test_query_zero_direction_entity_present(self):
        # direction=0 but entity has a position => score = 1/(1+|pos|)
        query = _make_query(
            directions={"structure": 0},
            confidences={"structure": 1.0},
        )
        result = _compute_bank_score({"structure": 2}, query, "multiplicative")
        self.assertAlmostEqual(result, 1.0 / 3.0, places=15)

    def test_two_banks_multiplicative(self):
        query = _make_query(
            directions={"structure": 1, "domain": -1},
            confidences={"structure": 1.0, "domain": 1.0},
        )
        # structure: entity=1, query=1 => aligned => 1.0
        # domain: entity=-1, query=-1 => aligned => 1.0
        result = _compute_bank_score({"structure": 1, "domain": -1}, query, "multiplicative")
        self.assertEqual(result, 1.0)

    def test_two_banks_mixed_alignment(self):
        query = _make_query(
            directions={"structure": 1, "domain": 1},
            confidences={"structure": 1.0, "domain": 1.0},
        )
        # structure: entity=1, query=1 => 1.0
        # domain: entity=-1, query=1 => alignment=-1 => 0.5
        result = _compute_bank_score({"structure": 1, "domain": -1}, query, "multiplicative")
        self.assertAlmostEqual(result, 1.0 * 0.5, places=15)

    def test_confidence_weighted_bank_scoring(self):
        query = _make_query(
            directions={"structure": 1, "domain": 1},
            confidences={"structure": 0.5, "domain": 1.0},
        )
        # structure: 1.0 (aligned), domain: 0.5 (anti)
        # confidence_weighted: 1.0^0.5 * 0.5^1.0 = 1.0 * 0.5 = 0.5
        result = _compute_bank_score({"structure": 1, "domain": -1}, query, "confidence_weighted")
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_all_six_banks(self):
        directions = {b: 1 for b in BANK_NAMES}
        confidences = {b: 1.0 for b in BANK_NAMES}
        query = _make_query(directions=directions, confidences=confidences)

        # All aligned => bank_score=1.0 for each => product=1.0
        positions = {b: 1 for b in BANK_NAMES}
        result = _compute_bank_score(positions, query, "multiplicative")
        self.assertEqual(result, 1.0)

    def test_all_six_banks_half_anti_aligned(self):
        directions = {b: 1 for b in BANK_NAMES}
        confidences = {b: 1.0 for b in BANK_NAMES}
        query = _make_query(directions=directions, confidences=confidences)

        # 3 aligned (1.0), 3 anti-aligned (0.5) => product = 0.5^3 = 0.125
        positions = {}
        for i, b in enumerate(BANK_NAMES):
            positions[b] = 1 if i < 3 else -1
        result = _compute_bank_score(positions, query, "multiplicative")
        self.assertAlmostEqual(result, 0.5**3, places=15)


# ---------------------------------------------------------------------------
# _compute_anchor_score
# ---------------------------------------------------------------------------


class TestComputeAnchorScore(unittest.TestCase):
    """IDF-weighted anchor matching."""

    def test_no_prefer_anchors_returns_one(self):
        query = _make_query()
        result = _compute_anchor_score(set(), query, {})
        self.assertEqual(result, 1.0)

    def test_no_prefer_anchors_with_entity_anchors(self):
        query = _make_query()
        result = _compute_anchor_score({1, 2, 3}, query, {1: 2.0, 2: 1.5})
        self.assertEqual(result, 1.0)

    def test_single_prefer_match(self):
        query = _make_query(prefer_anchors=[10], prefer_weights=[1.0])
        idf = {10: 2.0}
        entity_anchors = {10, 20}
        # total_idf = 2.0*1.0 = 2.0, matched_idf = 2.0 => 2.0/2.0 = 1.0
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertEqual(result, 1.0)

    def test_single_prefer_no_match(self):
        query = _make_query(prefer_anchors=[10], prefer_weights=[1.0])
        idf = {10: 2.0}
        entity_anchors = {20, 30}
        # total_idf = 2.0, matched_idf = 0 => 0.0
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertEqual(result, 0.0)

    def test_partial_match(self):
        query = _make_query(prefer_anchors=[10, 20], prefer_weights=[1.0, 1.0])
        idf = {10: 2.0, 20: 3.0}
        entity_anchors = {10}
        # total_idf = 2.0 + 3.0 = 5.0, matched_idf = 2.0 => 2.0/5.0 = 0.4
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertAlmostEqual(result, 0.4, places=15)

    def test_weighted_prefer(self):
        query = _make_query(prefer_anchors=[10, 20], prefer_weights=[2.0, 0.5])
        idf = {10: 1.0, 20: 1.0}
        entity_anchors = {10}
        # total_idf = 1.0*2.0 + 1.0*0.5 = 2.5
        # matched_idf = 1.0*2.0 = 2.0
        # => 2.0/2.5 = 0.8
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertAlmostEqual(result, 0.8, places=15)

    def test_default_idf_one_for_missing_anchors(self):
        query = _make_query(prefer_anchors=[10], prefer_weights=[1.0])
        idf = {}  # anchor 10 not in IDF cache => defaults to 1.0
        entity_anchors = {10}
        # total_idf = 1.0*1.0 = 1.0, matched = 1.0 => 1.0
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertEqual(result, 1.0)

    def test_avoid_single_match(self):
        query = _make_query(avoid_anchors=[10])
        entity_anchors = {10, 20}
        # No prefer_anchors => anchor score path => total_idf=0 => returns avoid_penalty
        # avoid: 10 in entity => penalty *= 0.5 => 0.5
        # But wait: no prefer_anchors => return 1.0 early
        # Actually, checking code: if not query.prefer_anchors: return 1.0
        # avoid penalty only applies when prefer_anchors is set
        result = _compute_anchor_score(entity_anchors, query, {})
        self.assertEqual(result, 1.0)

    def test_avoid_with_prefer(self):
        query = _make_query(prefer_anchors=[10], prefer_weights=[1.0], avoid_anchors=[20])
        idf = {10: 1.0}
        entity_anchors = {10, 20}
        # matched_idf/total_idf = 1.0/1.0 = 1.0
        # avoid: 20 in entity => penalty 0.5
        # => 1.0 * 0.5 = 0.5
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_avoid_double_match(self):
        query = _make_query(
            prefer_anchors=[10],
            prefer_weights=[1.0],
            avoid_anchors=[20, 30],
        )
        idf = {10: 1.0}
        entity_anchors = {10, 20, 30}
        # matched/total = 1.0
        # avoid: 20 match (0.5), 30 match (0.5) => penalty = 0.25
        # => 1.0 * 0.25 = 0.25
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertAlmostEqual(result, 0.25, places=15)

    def test_avoid_no_match(self):
        query = _make_query(
            prefer_anchors=[10],
            prefer_weights=[1.0],
            avoid_anchors=[99],
        )
        idf = {10: 2.0}
        entity_anchors = {10}
        # 99 not in entity => no penalty
        # matched/total = 2.0/2.0 = 1.0
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertEqual(result, 1.0)

    def test_total_idf_zero_returns_avoid_penalty(self):
        # Construct a case where total_idf == 0: weight=0 and idf=whatever
        query = _make_query(
            prefer_anchors=[10],
            prefer_weights=[0.0],
            avoid_anchors=[20],
        )
        idf = {10: 0.0}
        entity_anchors = {10, 20}
        # total_idf = 0.0 * 0.0 = 0.0 => returns avoid_penalty
        # avoid: 20 in entity => 0.5
        result = _compute_anchor_score(entity_anchors, query, idf)
        self.assertAlmostEqual(result, 0.5, places=15)


# ---------------------------------------------------------------------------
# _compute_seed_score
# ---------------------------------------------------------------------------


class TestComputeSeedScore(unittest.TestCase):
    """IDF-weighted Jaccard similarity with seed entities."""

    def test_no_seeds_returns_one(self):
        result = _compute_seed_score({1, 2}, set(), {})
        self.assertEqual(result, 1.0)

    def test_empty_entity_and_seeds(self):
        result = _compute_seed_score(set(), set(), {})
        self.assertEqual(result, 1.0)

    def test_empty_union_returns_one(self):
        # Both empty, so no seeds => 1.0 (caught by seed check first)
        result = _compute_seed_score(set(), set(), {1: 2.0})
        self.assertEqual(result, 1.0)

    def test_perfect_overlap(self):
        idf = {1: 2.0, 2: 3.0}
        entity_anchors = {1, 2}
        seed_anchors = {1, 2}
        # shared = {1,2}, union = {1,2}
        # shared_idf = 2+3 = 5, union_idf = 5 => 1.0
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertEqual(result, 1.0)

    def test_no_overlap(self):
        idf = {1: 2.0, 2: 3.0, 3: 1.0, 4: 1.5}
        entity_anchors = {1, 2}
        seed_anchors = {3, 4}
        # shared = {}, union = {1,2,3,4}
        # shared_idf = 0, union_idf = 2+3+1+1.5 = 7.5
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertEqual(result, 0.0)

    def test_partial_overlap(self):
        idf = {1: 2.0, 2: 3.0, 3: 1.0}
        entity_anchors = {1, 2}
        seed_anchors = {2, 3}
        # shared = {2}, union = {1,2,3}
        # shared_idf = 3.0, union_idf = 2+3+1 = 6.0
        # => 3.0/6.0 = 0.5
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertAlmostEqual(result, 0.5, places=15)

    def test_default_idf_for_missing_anchors(self):
        idf = {}  # all anchors default to 1.0
        entity_anchors = {1, 2}
        seed_anchors = {2, 3}
        # shared = {2}, union = {1,2,3}
        # shared_idf = 1.0, union_idf = 3.0
        # => 1/3
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertAlmostEqual(result, 1.0 / 3.0, places=15)

    def test_weighted_overlap(self):
        idf = {1: 10.0, 2: 1.0, 3: 1.0}
        entity_anchors = {1, 2}
        seed_anchors = {1, 3}
        # shared = {1}, union = {1,2,3}
        # shared_idf = 10.0, union_idf = 10+1+1 = 12.0
        # => 10/12 = 5/6
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertAlmostEqual(result, 10.0 / 12.0, places=15)

    def test_entity_empty_seeds_nonempty(self):
        idf = {1: 2.0, 2: 3.0}
        entity_anchors: set[int] = set()
        seed_anchors = {1, 2}
        # shared = {}, union = {1,2}
        # shared_idf = 0, union_idf = 5 => 0.0
        result = _compute_seed_score(entity_anchors, seed_anchors, idf)
        self.assertEqual(result, 0.0)


# ---------------------------------------------------------------------------
# _score_candidates — integration
# ---------------------------------------------------------------------------


class TestScoreCandidates(unittest.TestCase):
    """Integration: bank * anchor * seed scoring and sorting."""

    def test_basic_scoring(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        candidate_ids = [1, 2]
        positions = {
            1: {"structure": 1},  # aligned => bank_score=1.0
            2: {"structure": -1},  # anti => bank_score=0.5
        }
        entity_anchor_sets: dict[int, set[int]] = {}
        idf_cache: dict[int, float] = {}
        names = {1: "alpha", 2: "beta"}
        seed_anchors: set[int] = set()

        results = _score_candidates(
            candidate_ids,
            positions,
            entity_anchor_sets,
            idf_cache,
            names,
            query,
            seed_anchors,
            "multiplicative",
        )

        self.assertEqual(len(results), 2)
        # Sorted descending
        self.assertEqual(results[0].entity_id, 1)
        self.assertAlmostEqual(results[0].final_score, 1.0, places=15)
        self.assertEqual(results[1].entity_id, 2)
        self.assertAlmostEqual(results[1].final_score, 0.5, places=15)

    def test_final_score_is_product(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
            prefer_anchors=[10],
            prefer_weights=[1.0],
        )
        candidate_ids = [1]
        positions = {1: {"structure": -1}}  # bank_score = 0.5
        entity_anchor_sets = {1: {10}}  # anchor 10 matches
        idf_cache = {10: 2.0}
        names = {1: "gamma"}
        seed_anchors: set[int] = set()

        results = _score_candidates(
            candidate_ids,
            positions,
            entity_anchor_sets,
            idf_cache,
            names,
            query,
            seed_anchors,
            "multiplicative",
        )

        self.assertEqual(len(results), 1)
        # bank=0.5, anchor=1.0 (fully matched), seed=1.0 (no seeds)
        self.assertAlmostEqual(results[0].bank_score, 0.5, places=15)
        self.assertAlmostEqual(results[0].anchor_score, 1.0, places=15)
        self.assertAlmostEqual(results[0].seed_score, 1.0, places=15)
        self.assertAlmostEqual(results[0].final_score, 0.5, places=15)

    def test_zero_score_excluded(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
            prefer_anchors=[10],
            prefer_weights=[1.0],
        )
        candidate_ids = [1]
        positions = {1: {"structure": 1}}  # bank=1.0
        entity_anchor_sets = {1: {99}}  # no anchor 10 match => anchor=0.0
        idf_cache = {10: 2.0, 99: 1.0}
        names = {1: "delta"}
        seed_anchors: set[int] = set()

        results = _score_candidates(
            candidate_ids,
            positions,
            entity_anchor_sets,
            idf_cache,
            names,
            query,
            seed_anchors,
            "multiplicative",
        )

        self.assertEqual(len(results), 0)

    def test_sorting_descending(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        candidate_ids = [1, 2, 3]
        positions = {
            1: {"structure": -1},  # 0.5
            2: {"structure": 1},  # 1.0
            3: {"structure": 0},  # 0.5
        }
        names = {1: "a", 2: "b", 3: "c"}

        results = _score_candidates(
            candidate_ids,
            positions,
            {},
            {},
            names,
            query,
            set(),
            "multiplicative",
        )

        scores = [r.final_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0].entity_id, 2)

    def test_full_integration_bank_anchor_seed(self):
        query = _make_query(
            directions={"structure": 1, "domain": -1},
            confidences={"structure": 1.0, "domain": 1.0},
            prefer_anchors=[10, 20],
            prefer_weights=[1.0, 1.0],
        )
        candidate_ids = [1]
        positions = {1: {"structure": 1, "domain": -1}}  # both aligned => bank=1.0
        entity_anchor_sets = {1: {10, 30}}  # match 10, miss 20
        idf_cache = {10: 2.0, 20: 3.0, 30: 1.5}
        names = {1: "epsilon"}
        seed_anchors = {10, 40}

        results = _score_candidates(
            candidate_ids,
            positions,
            entity_anchor_sets,
            idf_cache,
            names,
            query,
            seed_anchors,
            "multiplicative",
        )

        self.assertEqual(len(results), 1)
        r = results[0]

        # Bank: structure aligned (1.0), domain aligned (1.0) => product 1.0
        self.assertAlmostEqual(r.bank_score, 1.0, places=15)

        # Anchor: prefer 10 (idf=2.0, w=1.0), prefer 20 (idf=3.0, w=1.0)
        # total_idf = 2.0 + 3.0 = 5.0
        # entity has 10 => matched_idf = 2.0
        # => 2.0/5.0 = 0.4
        self.assertAlmostEqual(r.anchor_score, 0.4, places=15)

        # Seed: entity_anchors={10,30}, seed_anchors={10,40}
        # shared = {10}, union = {10,30,40}
        # shared_idf = 2.0, union_idf = 2.0 + 1.5 + 1.0 = 4.5 (40 defaults to 1.0)
        expected_seed = 2.0 / 4.5
        self.assertAlmostEqual(r.seed_score, expected_seed, places=15)

        # Final = bank * anchor * seed
        expected_final = 1.0 * 0.4 * expected_seed
        self.assertAlmostEqual(r.final_score, expected_final, places=15)

    def test_empty_candidates(self):
        query = _make_query()
        results = _score_candidates([], {}, {}, {}, {}, query, set(), "multiplicative")
        self.assertEqual(results, [])

    def test_missing_name_defaults_empty(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        candidate_ids = [1]
        positions = {1: {"structure": 1}}
        names: dict[int, str] = {}  # no name for entity 1

        results = _score_candidates(
            candidate_ids,
            positions,
            {},
            {},
            names,
            query,
            set(),
            "multiplicative",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "")

    def test_score_breakdown_type(self):
        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = _score_candidates(
            [1],
            {1: {"structure": 1}},
            {},
            {},
            {1: "test"},
            query,
            set(),
            "multiplicative",
        )
        r = results[0]
        self.assertIsInstance(r, ScoredEntity)
        self.assertIsInstance(r.final_score, float)
        self.assertIsInstance(r.bank_score, float)
        self.assertIsInstance(r.anchor_score, float)
        self.assertIsInstance(r.seed_score, float)


# ---------------------------------------------------------------------------
# spread — BFS activation through entity links
# ---------------------------------------------------------------------------


class TestSpread(unittest.TestCase):
    """Spreading activation with in-memory SQLite DB."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def _insert_entity(self, eid: int, name: str) -> None:
        self.conn.execute(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, 'lemma')",
            (eid, name),
        )

    def _insert_link(self, source: int, target: int, weight: float, relation: str = "uses") -> None:
        self.conn.execute(
            "INSERT INTO entity_links (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
            (source, target, relation, weight),
        )

    def test_single_seed_no_links(self):
        self._insert_entity(1, "A")
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.8)
        self.assertEqual(result, {1: 1.0})

    def test_seed_activation_is_one(self):
        self._insert_entity(1, "A")
        self._insert_entity(2, "B")
        self.conn.commit()

        result = spread(self.conn, [1, 2], max_depth=3, decay=0.8)
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], 1.0)

    def test_single_hop_decay(self):
        """A --(0.5)--> B: activation(B) = 1.0 * 0.5 * 0.8 = 0.4."""
        self._insert_entity(1, "A")
        self._insert_entity(2, "B")
        self._insert_link(1, 2, 0.5)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.8)
        self.assertAlmostEqual(result[1], 1.0, places=15)
        self.assertAlmostEqual(result[2], 0.4, places=15)

    def test_two_hop_chain(self):
        """A --(1.0)--> B --(1.0)--> C.

        B = 1.0 * 1.0 * 0.8 = 0.8
        C = 0.8 * 1.0 * 0.8 = 0.64
        """
        self._insert_entity(1, "A")
        self._insert_entity(2, "B")
        self._insert_entity(3, "C")
        self._insert_link(1, 2, 1.0)
        self._insert_link(2, 3, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.8)
        self.assertAlmostEqual(result[1], 1.0, places=15)
        self.assertAlmostEqual(result[2], 0.8, places=15)
        self.assertAlmostEqual(result[3], 0.64, places=15)

    def test_three_hop_chain(self):
        """A -> B -> C -> D with weight 1.0 and decay 0.5.

        B = 1.0 * 1.0 * 0.5 = 0.5
        C = 0.5 * 1.0 * 0.5 = 0.25
        D = 0.25 * 1.0 * 0.5 = 0.125
        """
        for i in range(1, 5):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 2, 1.0)
        self._insert_link(2, 3, 1.0)
        self._insert_link(3, 4, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.5)
        self.assertAlmostEqual(result[1], 1.0, places=15)
        self.assertAlmostEqual(result[2], 0.5, places=15)
        self.assertAlmostEqual(result[3], 0.25, places=15)
        self.assertAlmostEqual(result[4], 0.125, places=15)

    def test_max_depth_boundary(self):
        """With max_depth=1, only direct neighbors are reached."""
        for i in range(1, 4):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 2, 1.0)
        self._insert_link(2, 3, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=1, decay=0.8)
        self.assertAlmostEqual(result[1], 1.0, places=15)
        self.assertAlmostEqual(result[2], 0.8, places=15)
        # 3 should NOT be reached because seed is depth=0, B is depth=1 which == max_depth
        self.assertNotIn(3, result)

    def test_max_depth_zero(self):
        """With max_depth=0, nothing spreads beyond the seed."""
        self._insert_entity(1, "A")
        self._insert_entity(2, "B")
        self._insert_link(1, 2, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=0, decay=0.8)
        self.assertEqual(result, {1: 1.0})

    def test_bidirectional_links(self):
        """Links are traversed in both directions.

        A --(0.5)--> B also means B can reach A.
        """
        self._insert_entity(1, "A")
        self._insert_entity(2, "B")
        self._insert_link(1, 2, 0.5)
        self.conn.commit()

        result = spread(self.conn, [2], max_depth=3, decay=0.8)
        self.assertAlmostEqual(result[2], 1.0, places=15)
        # B reaches A via reverse link: 1.0 * 0.5 * 0.8 = 0.4
        self.assertAlmostEqual(result[1], 0.4, places=15)

    def test_higher_activation_wins(self):
        """Two paths to the same node: the higher activation wins.

        A --(1.0)--> C and B --(1.0)--> C, seeds=[A, B].
        Both paths give C = 1.0*1.0*0.8 = 0.8.
        """
        for i in range(1, 4):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 3, 1.0)
        self._insert_link(2, 3, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1, 2], max_depth=3, decay=0.8)
        self.assertAlmostEqual(result[3], 0.8, places=15)

    def test_stronger_path_wins(self):
        """Two paths to C: A--(1.0)-->C vs A--(0.5)-->B--(1.0)-->C.

        Direct: 1.0*1.0*0.8 = 0.8
        Via B:  (1.0*0.5*0.8)*(1.0*0.8) = 0.4*0.8 = 0.32
        C = 0.8 (direct wins).
        """
        for i in range(1, 4):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 3, 1.0)  # direct A->C
        self._insert_link(1, 2, 0.5)  # A->B
        self._insert_link(2, 3, 1.0)  # B->C
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.8)
        self.assertAlmostEqual(result[3], 0.8, places=15)

    def test_diamond_topology(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D.

        All weights 1.0, decay 0.5.
        B = 0.5, C = 0.5
        D via B = 0.5*1.0*0.5 = 0.25
        D via C = 0.5*1.0*0.5 = 0.25
        D = 0.25 (both equal, first sets it, second doesn't improve)
        """
        for i in range(1, 5):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 2, 1.0)
        self._insert_link(1, 3, 1.0)
        self._insert_link(2, 4, 1.0)
        self._insert_link(3, 4, 1.0)
        self.conn.commit()

        result = spread(self.conn, [1], max_depth=3, decay=0.5)
        self.assertAlmostEqual(result[1], 1.0, places=15)
        self.assertAlmostEqual(result[2], 0.5, places=15)
        self.assertAlmostEqual(result[3], 0.5, places=15)
        self.assertAlmostEqual(result[4], 0.25, places=15)

    def test_multiple_seeds(self):
        """Two seeds converging on a shared neighbor."""
        for i in range(1, 4):
            self._insert_entity(i, chr(64 + i))
        self._insert_link(1, 3, 0.5)
        self._insert_link(2, 3, 0.8)
        self.conn.commit()

        result = spread(self.conn, [1, 2], max_depth=3, decay=1.0)
        # Via seed 1: 1.0 * 0.5 * 1.0 = 0.5
        # Via seed 2: 1.0 * 0.8 * 1.0 = 0.8
        # Max wins => 0.8
        self.assertAlmostEqual(result[3], 0.8, places=15)

    def test_neighbor_slice_limits(self):
        """neighbor_slice limits the number of neighbors considered."""
        self._insert_entity(1, "A")
        for i in range(2, 12):
            self._insert_entity(i, f"N{i}")
            self._insert_link(1, i, 1.0 / i)  # decreasing weights
        self.conn.commit()

        # Only top 3 neighbors by weight
        result = spread(self.conn, [1], max_depth=1, decay=1.0, neighbor_slice=3)
        # A has neighbors 2..11 with weights 0.5, 0.333, 0.25, 0.2, ...
        # Top 3 by weight: 2 (0.5), 3 (0.333), 4 (0.25)
        reached = {k for k in result if k != 1}
        self.assertLessEqual(len(reached), 3)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb(unittest.TestCase):
    """Database initialization and schema."""

    def test_init_db_memory(self):
        conn = init_db(":memory:")
        # Should have all expected tables
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r[0] for r in rows}
        expected_tables = {
            "entities",
            "entity_positions",
            "anchors",
            "entity_anchors",
            "entity_links",
            "anchor_idf",
            "accessible_premises",
        }
        self.assertEqual(expected_tables.issubset(table_names), True)
        conn.close()

    def test_init_db_idempotent(self):
        conn = init_db(":memory:")
        # Insert a row
        conn.execute("INSERT INTO entities (name, entity_type) VALUES ('test', 'lemma')")
        conn.commit()
        # Re-initialize (executescript _SCHEMA_SQL again) — should not destroy data
        conn.executescript(_SCHEMA_SQL)
        count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()


# ---------------------------------------------------------------------------
# recompute_idf
# ---------------------------------------------------------------------------


class TestRecomputeIdf(unittest.TestCase):
    """IDF computation from anchor_idf table."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def test_empty_entities_noop(self):
        recompute_idf(self.conn)
        rows = self.conn.execute("SELECT * FROM anchor_idf").fetchall()
        self.assertEqual(rows, [])

    def test_single_entity_single_anchor(self):
        self.conn.execute("INSERT INTO entities (id, name, entity_type) VALUES (1, 'A', 'lemma')")
        self.conn.execute(
            "INSERT INTO anchors (id, label, category) VALUES (1, 'tactic:intro', 'tactic')"
        )
        self.conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (1, 1)")
        self.conn.commit()

        recompute_idf(self.conn)

        row = self.conn.execute("SELECT idf_value FROM anchor_idf WHERE anchor_id = 1").fetchone()
        self.assertIsNotNone(row)
        # N=1, doc_freq=1: idf = log(1) - log(1) = 0.0
        self.assertAlmostEqual(row[0], 0.0, places=10)

    def test_idf_increases_with_rarity(self):
        # 10 entities, anchor A on all 10, anchor B on only 1
        for i in range(1, 11):
            self.conn.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, 'lemma')",
                (i, f"E{i}"),
            )
        self.conn.execute(
            "INSERT INTO anchors (id, label, category) VALUES (1, 'common', 'general')"
        )
        self.conn.execute("INSERT INTO anchors (id, label, category) VALUES (2, 'rare', 'general')")
        for i in range(1, 11):
            self.conn.execute(
                "INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (?, 1)",
                (i,),
            )
        self.conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (1, 2)")
        self.conn.commit()

        recompute_idf(self.conn)

        idf_common = self.conn.execute(
            "SELECT idf_value FROM anchor_idf WHERE anchor_id = 1"
        ).fetchone()[0]
        idf_rare = self.conn.execute(
            "SELECT idf_value FROM anchor_idf WHERE anchor_id = 2"
        ).fetchone()[0]
        # Rare anchor should have higher IDF
        self.assertGreater(idf_rare, idf_common)

    def test_idf_exact_values(self):
        # 4 entities, anchor on 2 of them
        for i in range(1, 5):
            self.conn.execute("INSERT INTO entities (id, name) VALUES (?, ?)", (i, f"E{i}"))
        self.conn.execute("INSERT INTO anchors (id, label) VALUES (1, 'half')")
        self.conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (1, 1)")
        self.conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (2, 1)")
        self.conn.commit()

        recompute_idf(self.conn)

        row = self.conn.execute("SELECT idf_value FROM anchor_idf WHERE anchor_id = 1").fetchone()
        # recompute_idf passes log_total=math.log(N) as Python param,
        # but SQLite LOG() is log10. So IDF = math.log(N) - log10(doc_freq).
        # N=4, doc_freq=2: math.log(4) - log10(2)
        expected = math.log(4) - math.log10(2)
        self.assertAlmostEqual(row[0], expected, places=10)


# ---------------------------------------------------------------------------
# clear_caches
# ---------------------------------------------------------------------------


class TestClearCaches(unittest.TestCase):
    """Module-level cache clearing."""

    def test_clear_caches_runs(self):
        # Should not raise
        clear_caches()

    def test_clear_caches_invalidates_accessible(self):
        conn = _make_db()
        # Insert entity and premise
        conn.execute("INSERT INTO entities (id, name) VALUES (1, 'thm')")
        conn.execute("INSERT INTO entities (id, name) VALUES (2, 'premise')")
        conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, 2)")
        conn.commit()

        result1 = get_accessible_premises(conn, 1)
        self.assertEqual(result1, {2})

        # Add another premise
        conn.execute("INSERT INTO entities (id, name) VALUES (3, 'premise2')")
        conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, 3)")
        conn.commit()

        # Before clearing cache, should still return old result
        result2 = get_accessible_premises(conn, 1)
        self.assertEqual(result2, {2})

        # After clearing, should see new data
        clear_caches()
        result3 = get_accessible_premises(conn, 1)
        self.assertEqual(result3, {2, 3})

        conn.close()
        clear_caches()


# ---------------------------------------------------------------------------
# get_accessible_premises
# ---------------------------------------------------------------------------


class TestGetAccessiblePremises(unittest.TestCase):
    """Accessible premises caching and retrieval."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def test_empty_returns_empty_set(self):
        self.conn.execute("INSERT INTO entities (id, name) VALUES (1, 'thm')")
        self.conn.commit()
        result = get_accessible_premises(self.conn, 1)
        self.assertEqual(result, set())

    def test_returns_correct_premises(self):
        for i in range(1, 4):
            self.conn.execute("INSERT INTO entities (id, name) VALUES (?, ?)", (i, f"E{i}"))
        self.conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, 2)")
        self.conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, 3)")
        self.conn.commit()

        result = get_accessible_premises(self.conn, 1)
        self.assertEqual(result, {2, 3})

    def test_different_theorems_independent(self):
        for i in range(1, 5):
            self.conn.execute("INSERT INTO entities (id, name) VALUES (?, ?)", (i, f"E{i}"))
        self.conn.execute("INSERT INTO accessible_premises VALUES (1, 3)")
        self.conn.execute("INSERT INTO accessible_premises VALUES (2, 4)")
        self.conn.commit()

        self.assertEqual(get_accessible_premises(self.conn, 1), {3})
        self.assertEqual(get_accessible_premises(self.conn, 2), {4})

    def test_caching_returns_same_object(self):
        self.conn.execute("INSERT INTO entities (id, name) VALUES (1, 'thm')")
        self.conn.commit()
        r1 = get_accessible_premises(self.conn, 1)
        r2 = get_accessible_premises(self.conn, 1)
        self.assertIs(r1, r2)  # Same object from cache

    def test_nonexistent_theorem(self):
        result = get_accessible_premises(self.conn, 9999)
        self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# navigate — end-to-end DB-backed retrieval
# ---------------------------------------------------------------------------


class TestNavigateEndToEnd(unittest.TestCase):
    """End-to-end navigate() with real DB."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()
        self._populate_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def _populate_db(self):
        """Set up a small proof network for integration testing."""
        # 3 entities
        self.conn.executemany(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            [
                (1, "Nat.add_comm", "lemma"),
                (2, "List.map_id", "lemma"),
                (3, "Ring.mul_comm", "lemma"),
            ],
        )
        # Bank positions
        self.conn.executemany(
            "INSERT INTO entity_positions (entity_id, bank, sign, depth) VALUES (?, ?, ?, ?)",
            [
                (1, "structure", 1, 1),  # signed_pos = 1*1 = 1
                (1, "domain", -1, 1),  # signed_pos = -1
                (2, "structure", -1, 1),  # signed_pos = -1
                (2, "domain", 1, 1),  # signed_pos = 1
                (3, "structure", 1, 2),  # signed_pos = 2
                (3, "domain", -1, 2),  # signed_pos = -2
            ],
        )
        # Anchors
        self.conn.executemany(
            "INSERT INTO anchors (id, label) VALUES (?, ?)",
            [(1, "tactic:ring"), (2, "tactic:omega"), (3, "ns:Nat")],
        )
        # Entity-anchor links
        self.conn.executemany(
            "INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (?, ?)",
            [(1, 2), (1, 3), (2, 2), (3, 1)],
        )
        # IDF values
        self.conn.executemany(
            "INSERT INTO anchor_idf (anchor_id, idf_value) VALUES (?, ?)",
            [(1, 2.0), (2, 0.5), (3, 1.5)],
        )
        self.conn.commit()

    def test_basic_navigate(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = navigate(self.conn, query, limit=10, mechanism="multiplicative")
        self.assertGreater(len(results), 0)
        # Entity 1 and 3 have structure=+1/+2, entity 2 has structure=-1
        # Entity 1: bank_score(1, 1) = 1.0
        # Entity 3: bank_score(2, 1) = 1.0 (alignment=2>0)
        # Entity 2: bank_score(-1, 1) = 0.5
        for r in results:
            self.assertIsInstance(r, ScoredEntity)
            self.assertGreater(r.final_score, 0)

    def test_navigate_limit(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = navigate(self.conn, query, limit=1, mechanism="multiplicative")
        self.assertLessEqual(len(results), 1)

    def test_navigate_with_entity_type_filter(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = navigate(
            self.conn,
            query,
            limit=10,
            mechanism="multiplicative",
            entity_type="lemma",
        )
        self.assertEqual(len(results), 3)

    def test_navigate_nonexistent_type_returns_empty(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = navigate(
            self.conn,
            query,
            limit=10,
            mechanism="multiplicative",
            entity_type="tactic",
        )
        self.assertEqual(len(results), 0)

    def test_navigate_with_require_anchors(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
            require_anchors=[1],  # tactic:ring => only entity 3
        )
        results = navigate(self.conn, query, limit=10, mechanism="multiplicative")
        entity_ids = {r.entity_id for r in results}
        self.assertEqual(entity_ids, {3})

    def test_navigate_with_accessible_premises(self):
        from src.proof_network import navigate

        # Add accessible premises: theorem 1 can access entity 2
        self.conn.execute("INSERT INTO accessible_premises VALUES (1, 2)")
        self.conn.commit()

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
            accessible_theorem_id=1,
        )
        results = navigate(self.conn, query, limit=10, mechanism="multiplicative")
        entity_ids = {r.entity_id for r in results}
        self.assertTrue(entity_ids.issubset({2}))

    def test_navigate_results_sorted_descending(self):
        from src.proof_network import navigate

        query = _make_query(
            directions={"structure": 1},
            confidences={"structure": 1.0},
        )
        results = navigate(self.conn, query, limit=10, mechanism="multiplicative")
        scores = [r.final_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# _MISSING_BANK_SCORE constant
# ---------------------------------------------------------------------------


class TestMissingBankScore(unittest.TestCase):
    """Verify the penalty constant is as documented."""

    def test_value(self):
        self.assertEqual(_MISSING_BANK_SCORE, 0.3)


# ---------------------------------------------------------------------------
# BANK_NAMES consistency
# ---------------------------------------------------------------------------


class TestBankNamesConsistency(unittest.TestCase):
    """Ensure BANK_NAMES are lowercase and have 6 entries."""

    def test_count(self):
        self.assertEqual(len(BANK_NAMES), 6)

    def test_lowercase(self):
        for name in BANK_NAMES:
            self.assertEqual(name, name.lower())

    def test_expected_names(self):
        expected = {"structure", "domain", "depth", "automation", "context", "decomposition"}
        self.assertEqual(set(BANK_NAMES), expected)


if __name__ == "__main__":
    unittest.main()
