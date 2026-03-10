"""Tests for resolution — build_query, _combine_candidates, resolve (mocked)."""

import unittest
from unittest.mock import patch

from src.nav_contracts import NavOutput, ScoredEntity, StructuredQuery
from src.resolution import (
    Candidate,
    SearchContext,
    _combine_candidates,
    build_query,
    resolve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nav_output(
    anchor_scores=None,
    directions=None,
    direction_confidences=None,
):
    return NavOutput(
        directions=directions or {"structure": 1, "domain": 0},
        direction_confidences=direction_confidences or {"structure": 0.9, "domain": 0.5},
        anchor_scores=anchor_scores or {},
        progress=0.5,
        critic_score=0.8,
    )


def _make_scored_entity(entity_id=1, name="simp", final_score=0.9):
    return ScoredEntity(
        entity_id=entity_id,
        name=name,
        final_score=final_score,
        bank_score=0.5,
        anchor_score=0.3,
        seed_score=0.1,
    )


# ---------------------------------------------------------------------------
# SearchContext / Candidate dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses(unittest.TestCase):
    def test_search_context_defaults(self):
        ctx = SearchContext()
        self.assertIsNone(ctx.accessible_theorem_id)
        self.assertEqual(ctx.seed_entity_ids, [])
        self.assertEqual(ctx.used_tactics, [])

    def test_search_context_field_isolation(self):
        c1 = SearchContext()
        c2 = SearchContext()
        c1.seed_entity_ids.append(42)
        self.assertEqual(c2.seed_entity_ids, [])

    def test_candidate_defaults(self):
        c = Candidate(tactic_name="rfl", premises=[], score=0.5)
        self.assertIsNone(c.tactic_entity)
        self.assertEqual(c.premise_entities, [])

    def test_candidate_field_isolation(self):
        c1 = Candidate(tactic_name="a", premises=[], score=0.5)
        c2 = Candidate(tactic_name="b", premises=[], score=0.5)
        c1.premise_entities.append(_make_scored_entity())
        self.assertEqual(c2.premise_entities, [])


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------


class TestBuildQuery(unittest.TestCase):
    def test_basic_directions_and_confidences(self):
        nav = _make_nav_output()
        q = build_query(nav)
        self.assertEqual(q.bank_directions, {"structure": 1, "domain": 0})
        self.assertEqual(q.bank_confidences, {"structure": 0.9, "domain": 0.5})

    def test_no_anchors_when_no_map(self):
        nav = _make_nav_output(anchor_scores={"anc_a": 0.9})
        q = build_query(nav, anchor_id_map=None)
        self.assertEqual(q.prefer_anchors, [])
        self.assertEqual(q.prefer_weights, [])

    def test_anchors_filtered_by_threshold(self):
        nav = _make_nav_output(anchor_scores={"a": 0.8, "b": 0.1})
        q = build_query(nav, anchor_id_map={"a": 10, "b": 20}, anchor_threshold=0.3)
        self.assertEqual(q.prefer_anchors, [10])
        self.assertAlmostEqual(q.prefer_weights[0], 0.8)

    def test_anchors_respect_top_k(self):
        scores = {f"a{i}": 0.9 - i * 0.01 for i in range(10)}
        id_map = {f"a{i}": i + 100 for i in range(10)}
        nav = _make_nav_output(anchor_scores=scores)
        q = build_query(nav, anchor_id_map=id_map, top_k_anchors=3)
        self.assertEqual(len(q.prefer_anchors), 3)

    def test_anchors_missing_from_map_are_skipped(self):
        nav = _make_nav_output(anchor_scores={"known": 0.9, "unknown": 0.8})
        q = build_query(nav, anchor_id_map={"known": 1})
        self.assertEqual(q.prefer_anchors, [1])
        self.assertEqual(len(q.prefer_weights), 1)

    def test_avoid_anchors_always_empty(self):
        nav = _make_nav_output()
        q = build_query(nav, anchor_id_map={"a": 1})
        self.assertEqual(q.avoid_anchors, [])

    def test_returns_structured_query(self):
        nav = _make_nav_output()
        q = build_query(nav)
        self.assertIsInstance(q, StructuredQuery)


# ---------------------------------------------------------------------------
# _combine_candidates
# ---------------------------------------------------------------------------


class TestCombineCandidates(unittest.TestCase):
    def test_empty_tactics_returns_empty(self):
        self.assertEqual(_combine_candidates([], [], {}), [])

    def test_single_tactic_no_premises(self):
        tactic = _make_scored_entity(entity_id=1, name="rfl", final_score=0.9)
        result = _combine_candidates([tactic], [], {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].tactic_name, "rfl")
        self.assertEqual(result[0].premises, [])

    def test_premises_names_attached_to_each_candidate(self):
        t1 = _make_scored_entity(entity_id=1, name="apply", final_score=0.9)
        t2 = _make_scored_entity(entity_id=2, name="exact", final_score=0.8)
        p1 = _make_scored_entity(entity_id=10, name="Nat.add_comm", final_score=0.7)
        p2 = _make_scored_entity(entity_id=11, name="Nat.mul_comm", final_score=0.6)
        result = _combine_candidates([t1, t2], [p1, p2], {})
        for c in result:
            self.assertEqual(c.premises, ["Nat.add_comm", "Nat.mul_comm"])

    def test_spread_boost_increases_score(self):
        tactic = _make_scored_entity(entity_id=1, name="rfl", final_score=1.0)
        unboosted = _combine_candidates([tactic], [], {})
        boosted = _combine_candidates([tactic], [], {1: 5.0})
        self.assertGreater(boosted[0].score, unboosted[0].score)

    def test_premise_spread_boost_reorders(self):
        p_low = _make_scored_entity(entity_id=10, name="low", final_score=0.5)
        p_high = _make_scored_entity(entity_id=11, name="high", final_score=0.6)
        tactic = _make_scored_entity(entity_id=1, name="t", final_score=0.9)
        # Boost the low-scoring premise
        result = _combine_candidates([tactic], [p_low, p_high], {10: 10.0})
        # "low" should now be first in premises due to boost
        self.assertEqual(result[0].premises[0], "low")

    def test_candidates_sorted_by_score_descending(self):
        t1 = _make_scored_entity(entity_id=1, name="low", final_score=0.3)
        t2 = _make_scored_entity(entity_id=2, name="high", final_score=0.9)
        result = _combine_candidates([t1, t2], [], {})
        self.assertEqual(result[0].tactic_name, "high")
        self.assertEqual(result[1].tactic_name, "low")

    def test_tactic_entity_preserved(self):
        tactic = _make_scored_entity(entity_id=42, name="simp", final_score=0.8)
        result = _combine_candidates([tactic], [], {})
        self.assertEqual(result[0].tactic_entity.entity_id, 42)

    def test_premise_entities_preserved(self):
        tactic = _make_scored_entity(entity_id=1, name="t", final_score=0.9)
        p = _make_scored_entity(entity_id=10, name="lem", final_score=0.7)
        result = _combine_candidates([tactic], [p], {})
        self.assertEqual(len(result[0].premise_entities), 1)
        self.assertEqual(result[0].premise_entities[0].entity_id, 10)


# ---------------------------------------------------------------------------
# resolve (with mocked navigate/spread)
# ---------------------------------------------------------------------------


class TestResolve(unittest.TestCase):
    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_basic_resolve(self, mock_navigate, mock_spread):
        mock_navigate.return_value = [
            _make_scored_entity(entity_id=1, name="rfl", final_score=0.9),
        ]
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext()

        result = resolve(nav, "fake_conn", ctx)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].tactic_name, "rfl")

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_accessible_theorem_id_passed_to_query(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext(accessible_theorem_id=42)

        resolve(nav, "fake_conn", ctx)

        # Check navigate was called and the query had accessible_theorem_id
        call_args = mock_navigate.call_args_list[0]
        query_arg = call_args[0][1]  # second positional arg
        self.assertEqual(query_arg.accessible_theorem_id, 42)

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_seed_entities_passed_to_query(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext(seed_entity_ids=[10, 20])

        resolve(nav, "fake_conn", ctx)

        call_args = mock_navigate.call_args_list[0]
        query_arg = call_args[0][1]
        self.assertEqual(query_arg.seed_entity_ids, [10, 20])

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_spread_called_when_seeds_exist(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext(seed_entity_ids=[5])

        resolve(nav, "fake_conn", ctx)

        mock_spread.assert_called_once()

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_spread_not_called_when_no_seeds(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext(seed_entity_ids=[])

        resolve(nav, "fake_conn", ctx)

        mock_spread.assert_not_called()

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_navigate_called_for_tactics_and_premises(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext()

        resolve(nav, "fake_conn", ctx)

        self.assertEqual(mock_navigate.call_count, 2)
        calls = mock_navigate.call_args_list
        self.assertEqual(calls[0][1]["entity_type"], "tactic")
        self.assertEqual(calls[1][1]["entity_type"], "lemma")

    @patch("src.resolution.spread")
    @patch("src.resolution.navigate")
    def test_tactic_and_premise_limits_passed(self, mock_navigate, mock_spread):
        mock_navigate.return_value = []
        mock_spread.return_value = {}
        nav = _make_nav_output()
        ctx = SearchContext()

        resolve(nav, "fake_conn", ctx, tactic_limit=4, premise_limit=8)

        calls = mock_navigate.call_args_list
        self.assertEqual(calls[0][1]["limit"], 4)
        self.assertEqual(calls[1][1]["limit"], 8)


if __name__ == "__main__":
    unittest.main()
