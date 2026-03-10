"""Tests for proof_search — SearchResult, _try_hammer, _try_candidates."""

import unittest
from unittest.mock import MagicMock, patch

from src.nav_contracts import NavOutput, TacticResult
from src.proof_search import (
    SearchResult,
    _SearchState,
    _try_candidates,
    _try_hammer,
)
from src.resolution import Candidate, SearchContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nav_output(automation=0, critic=0.5):
    """Build a NavOutput with a specified automation direction."""
    return NavOutput(
        directions={"automation": automation} if automation is not None else {},
        direction_confidences={"automation": 0.9},
        anchor_scores={},
        progress=0.5,
        critic_score=critic,
    )


def _make_candidate(tactic="apply", premises=None, score=0.9):
    """Build a Candidate with optional premises."""
    return Candidate(
        tactic_name=tactic,
        premises=premises or [],
        score=score,
    )


def _make_tactic_result(success, tactic="apply", premises=None, new_goals=None):
    """Build a TacticResult."""
    return TacticResult(
        success=success,
        tactic=tactic,
        premises=premises or [],
        new_goals=new_goals or [],
    )


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult(unittest.TestCase):

    def test_defaults(self):
        r = SearchResult(success=True, theorem_id="thm1")
        self.assertEqual(r.success, True)
        self.assertEqual(r.theorem_id, "thm1")
        self.assertEqual(r.tactics_used, [])
        self.assertEqual(r.attempts, 0)
        self.assertEqual(r.goals_closed, 0)
        self.assertEqual(r.goals_remaining, 0)

    def test_factory_isolation(self):
        r1 = SearchResult(success=False, theorem_id="a")
        r2 = SearchResult(success=False, theorem_id="b")
        r1.tactics_used.append("rfl")
        self.assertEqual(r2.tactics_used, [])


# ---------------------------------------------------------------------------
# _try_hammer
# ---------------------------------------------------------------------------


class TestTryHammer(unittest.TestCase):

    @patch("src.proof_search.resolve")
    def test_success_mutates_state(self, mock_resolve):
        """Hammer succeeds: goal popped, closed, tactic recorded, seeds cleared."""
        candidate = _make_candidate(tactic="omega", premises=["lem1", "lem2"])
        mock_resolve.return_value = [candidate]

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(
            success=True, tactic="omega", premises=["lem1"], new_goals=["subgoal1"],
        )

        state = _SearchState(open_goals=["goal_A", "goal_B"])
        context = SearchContext(seed_entity_ids=[10, 20])
        nav = _make_nav_output(automation=-1)

        result = _try_hammer("goal_A", 0, nav, state, MagicMock(), lean, context, None)

        self.assertEqual(result, True)
        # goal_A removed from open_goals, subgoal1 added
        self.assertEqual(state.open_goals, ["goal_B", "subgoal1"])
        self.assertEqual(state.closed_goals, ["goal_A"])
        self.assertEqual(state.tactics_used, ["omega"])
        self.assertEqual(state.attempts, 1)
        # seed_entity_ids cleared on success
        self.assertEqual(context.seed_entity_ids, [])

    @patch("src.proof_search.resolve")
    def test_failure_increments_attempts(self, mock_resolve):
        """Hammer fails: attempts incremented, returns False, state otherwise unchanged."""
        mock_resolve.return_value = [_make_candidate(premises=["p"])]

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(success=False, tactic="omega")

        state = _SearchState(open_goals=["goal_A"])
        context = SearchContext()
        nav = _make_nav_output(automation=-1)

        result = _try_hammer("goal_A", 0, nav, state, MagicMock(), lean, context, None)

        self.assertEqual(result, False)
        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.open_goals, ["goal_A"])
        self.assertEqual(state.closed_goals, [])

    @patch("src.proof_search.resolve")
    def test_no_candidates_hammer_with_empty_premises(self, mock_resolve):
        """When resolve returns empty list, hammer is called with empty premises."""
        mock_resolve.return_value = []

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(success=False, tactic="hammer")

        state = _SearchState(open_goals=["g"])
        context = SearchContext()
        nav = _make_nav_output(automation=-1)

        _try_hammer("g", 0, nav, state, MagicMock(), lean, context, None)

        lean.try_hammer.assert_called_once_with("g", [])

    @patch("src.proof_search.resolve")
    def test_premises_truncated_to_16(self, mock_resolve):
        """Hammer premises are capped at 16 from the first candidate."""
        long_premises = [f"p{i}" for i in range(20)]
        mock_resolve.return_value = [_make_candidate(premises=long_premises)]

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(success=False, tactic="h")

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        _try_hammer("g", 0, nav, state, MagicMock(), lean, SearchContext(), None)

        called_premises = lean.try_hammer.call_args[0][1]
        self.assertEqual(len(called_premises), 16)
        self.assertEqual(called_premises, long_premises[:16])

    @patch("src.proof_search.resolve")
    def test_resolve_called_with_premise_limit_16(self, mock_resolve):
        """resolve() is called with premise_limit=16 for hammer."""
        mock_resolve.return_value = []

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(success=False, tactic="h")

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        _try_hammer("g", 0, nav, state, MagicMock(), lean, SearchContext(), {"a": 1})

        _, kwargs = mock_resolve.call_args
        self.assertEqual(kwargs["premise_limit"], 16)


# ---------------------------------------------------------------------------
# _try_candidates
# ---------------------------------------------------------------------------


class TestTryCandidates(unittest.TestCase):

    @patch("src.proof_search.resolve")
    def test_first_candidate_succeeds(self, mock_resolve):
        """First candidate succeeds: state updated, returns True."""
        mock_resolve.return_value = [
            _make_candidate(tactic="exact", premises=["h"]),
            _make_candidate(tactic="apply", premises=["h2"]),
        ]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True, tactic="exact h", new_goals=[],
        )

        state = _SearchState(open_goals=["goal_X", "goal_Y"])
        nav = _make_nav_output()

        result = _try_candidates(
            "goal_X", 0, nav, state, MagicMock(), lean, SearchContext(), None, 8,
        )

        self.assertEqual(result, True)
        self.assertEqual(state.open_goals, ["goal_Y"])
        self.assertEqual(state.closed_goals, ["goal_X"])
        self.assertEqual(state.tactics_used, ["exact h"])
        self.assertEqual(state.attempts, 1)
        # Only tried one candidate
        self.assertEqual(lean.try_tactic.call_count, 1)

    @patch("src.proof_search.resolve")
    def test_all_candidates_fail(self, mock_resolve):
        """All candidates fail: returns False, state goals unchanged."""
        mock_resolve.return_value = [
            _make_candidate(tactic="t1"),
            _make_candidate(tactic="t2"),
        ]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(success=False, tactic="t")

        state = _SearchState(open_goals=["g1"])
        nav = _make_nav_output()

        result = _try_candidates(
            "g1", 0, nav, state, MagicMock(), lean, SearchContext(), None, 8,
        )

        self.assertEqual(result, False)
        self.assertEqual(state.open_goals, ["g1"])
        self.assertEqual(state.closed_goals, [])
        self.assertEqual(state.attempts, 2)  # tried both candidates

    @patch("src.proof_search.resolve")
    def test_respects_max_candidates(self, mock_resolve):
        """Only max_candidates candidates are tried even if more are available."""
        mock_resolve.return_value = [
            _make_candidate(tactic=f"t{i}") for i in range(10)
        ]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(success=False, tactic="t")

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        _try_candidates("g", 0, nav, state, MagicMock(), lean, SearchContext(), None, 3)

        self.assertEqual(lean.try_tactic.call_count, 3)
        self.assertEqual(state.attempts, 3)

    @patch("src.proof_search.resolve")
    def test_second_candidate_succeeds(self, mock_resolve):
        """First fails, second succeeds: attempts=2, state correct."""
        mock_resolve.return_value = [
            _make_candidate(tactic="t1", premises=[]),
            _make_candidate(tactic="t2", premises=["h"]),
        ]

        lean = MagicMock()
        lean.try_tactic.side_effect = [
            _make_tactic_result(success=False, tactic="t1"),
            _make_tactic_result(success=True, tactic="t2 h", new_goals=["sub"]),
        ]

        state = _SearchState(open_goals=["g1"])
        nav = _make_nav_output()

        result = _try_candidates(
            "g1", 0, nav, state, MagicMock(), lean, SearchContext(), None, 8,
        )

        self.assertEqual(result, True)
        self.assertEqual(state.attempts, 2)
        self.assertEqual(state.closed_goals, ["g1"])
        self.assertEqual(state.open_goals, ["sub"])
        self.assertEqual(state.tactics_used, ["t2 h"])

    @patch("src.proof_search.resolve")
    def test_new_goals_appended(self, mock_resolve):
        """Success with new_goals: they are appended to open_goals."""
        mock_resolve.return_value = [_make_candidate(tactic="cases")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True, tactic="cases", new_goals=["case1", "case2"],
        )

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        _try_candidates("g", 0, nav, state, MagicMock(), lean, SearchContext(), None, 8)

        self.assertEqual(state.open_goals, ["case1", "case2"])
        self.assertEqual(state.closed_goals, ["g"])

    @patch("src.proof_search.resolve")
    def test_empty_candidates_returns_false(self, mock_resolve):
        """No candidates from resolve: returns False immediately."""
        mock_resolve.return_value = []

        lean = MagicMock()
        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        result = _try_candidates(
            "g", 0, nav, state, MagicMock(), lean, SearchContext(), None, 8,
        )

        self.assertEqual(result, False)
        self.assertEqual(state.attempts, 0)
        lean.try_tactic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
