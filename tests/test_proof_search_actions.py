"""Tests for proof_search — SearchResult, _try_hammer, _try_candidates,
_should_hammer, _build_tactic_text, _close_goal, _cached_infer, _select_goal,
_search_step."""

import unittest
from unittest.mock import MagicMock, patch

from src.nav_contracts import NavOutput, TacticResult
from src.proof_search import (
    Pipeline,
    SearchConfig,
    SearchResult,
    _build_tactic_text,
    _cached_infer,
    _close_goal,
    _search_step,
    _SearchEnv,
    _SearchState,
    _select_goal,
    _should_hammer,
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
            success=True,
            tactic="omega",
            premises=["lem1"],
            new_goals=["subgoal1"],
        )

        state = _SearchState(open_goals=["goal_A", "goal_B"])
        context = SearchContext(seed_entity_ids=[10, 20])
        nav = _make_nav_output(automation=-1)

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_hammer("goal_A", 0, nav, state, env, context)

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_hammer("goal_A", 0, nav, state, env, context)

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        _try_hammer("g", 0, nav, state, env, context)

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        _try_hammer("g", 0, nav, state, env, SearchContext())

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map={"a": 1}, max_candidates=8)
        _try_hammer("g", 0, nav, state, env, SearchContext())

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
            success=True,
            tactic="exact h",
            new_goals=[],
        )

        state = _SearchState(open_goals=["goal_X", "goal_Y"])
        nav = _make_nav_output()

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_candidates("goal_X", 0, nav, state, env, SearchContext())

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_candidates("g1", 0, nav, state, env, SearchContext())

        self.assertEqual(result, False)
        self.assertEqual(state.open_goals, ["g1"])
        self.assertEqual(state.closed_goals, [])
        self.assertEqual(state.attempts, 2)  # tried both candidates

    @patch("src.proof_search.resolve")
    def test_respects_max_candidates(self, mock_resolve):
        """Only max_candidates candidates are tried even if more are available."""
        mock_resolve.return_value = [_make_candidate(tactic=f"t{i}") for i in range(10)]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(success=False, tactic="t")

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=3)
        _try_candidates("g", 0, nav, state, env, SearchContext())

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

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_candidates("g1", 0, nav, state, env, SearchContext())

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
            success=True,
            tactic="cases",
            new_goals=["case1", "case2"],
        )

        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        _try_candidates("g", 0, nav, state, env, SearchContext())

        self.assertEqual(state.open_goals, ["case1", "case2"])
        self.assertEqual(state.closed_goals, ["g"])

    @patch("src.proof_search.resolve")
    def test_empty_candidates_returns_false(self, mock_resolve):
        """No candidates from resolve: returns False immediately."""
        mock_resolve.return_value = []

        lean = MagicMock()
        state = _SearchState(open_goals=["g"])
        nav = _make_nav_output()

        env = _SearchEnv(conn=MagicMock(), lean=lean, anchor_id_map=None, max_candidates=8)
        result = _try_candidates("g", 0, nav, state, env, SearchContext())

        self.assertEqual(result, False)
        self.assertEqual(state.attempts, 0)
        lean.try_tactic.assert_not_called()


# ---------------------------------------------------------------------------
# _should_hammer
# ---------------------------------------------------------------------------


class TestShouldHammer(unittest.TestCase):
    def test_automation_neg1_returns_true(self):
        """automation=-1 triggers hammer delegation."""
        nav = _make_nav_output(automation=-1)
        self.assertTrue(_should_hammer(nav))

    def test_automation_zero_returns_false(self):
        """automation=0 does not trigger hammer."""
        nav = _make_nav_output(automation=0)
        self.assertFalse(_should_hammer(nav))

    def test_automation_pos1_returns_false(self):
        """automation=+1 does not trigger hammer."""
        nav = _make_nav_output(automation=1)
        self.assertFalse(_should_hammer(nav))

    def test_missing_automation_key_returns_false(self):
        """Missing automation key defaults to 0, no hammer."""
        nav = _make_nav_output(automation=None)
        self.assertFalse(_should_hammer(nav))


# ---------------------------------------------------------------------------
# _build_tactic_text
# ---------------------------------------------------------------------------


class TestBuildTacticText(unittest.TestCase):
    def test_no_premises(self):
        """No premises returns just the tactic name."""
        c = _make_candidate(tactic="rfl", premises=[])
        self.assertEqual(_build_tactic_text(c), "rfl")

    def test_with_premises(self):
        """Premises are joined with spaces after the tactic name."""
        c = _make_candidate(tactic="apply", premises=["h1", "h2"])
        self.assertEqual(_build_tactic_text(c), "apply h1 h2")

    def test_more_than_4_premises_truncated(self):
        """Only the first 4 premises are included."""
        c = _make_candidate(tactic="exact", premises=["a", "b", "c", "d", "e", "f"])
        result = _build_tactic_text(c)
        self.assertEqual(result, "exact a b c d")

    def test_exactly_4_premises(self):
        """Exactly 4 premises are all included."""
        c = _make_candidate(tactic="simp", premises=["p1", "p2", "p3", "p4"])
        self.assertEqual(_build_tactic_text(c), "simp p1 p2 p3 p4")


# ---------------------------------------------------------------------------
# _close_goal
# ---------------------------------------------------------------------------


class TestCloseGoal(unittest.TestCase):
    def test_removes_goal_at_correct_index(self):
        """Goal is popped from open_goals at the given index."""
        state = _SearchState(open_goals=["g0", "g1", "g2"])
        _close_goal("g1", 1, "tac", [], state)
        self.assertNotIn("g1", state.open_goals)
        self.assertEqual(state.open_goals, ["g0", "g2"])

    def test_adds_goal_to_closed(self):
        """Closed goal is appended to closed_goals."""
        state = _SearchState(open_goals=["g0"])
        _close_goal("g0", 0, "rfl", [], state)
        self.assertEqual(state.closed_goals, ["g0"])

    def test_appends_tactic(self):
        """Tactic is appended to tactics_used."""
        state = _SearchState(open_goals=["g0"])
        _close_goal("g0", 0, "omega", [], state)
        self.assertEqual(state.tactics_used, ["omega"])

    def test_extends_open_goals_with_new_goals(self):
        """New sub-goals are appended to open_goals."""
        state = _SearchState(open_goals=["g0", "g1"])
        _close_goal("g0", 0, "cases", ["sub1", "sub2"], state)
        self.assertEqual(state.open_goals, ["g1", "sub1", "sub2"])

    def test_clears_infer_cache(self):
        """Inference cache is cleared after closing a goal."""
        state = _SearchState(open_goals=["g0"])
        state._infer_cache["g0"] = _make_nav_output()
        state._infer_cache["other"] = _make_nav_output()
        _close_goal("g0", 0, "tac", [], state)
        self.assertEqual(state._infer_cache, {})


# ---------------------------------------------------------------------------
# _cached_infer
# ---------------------------------------------------------------------------


class TestCachedInfer(unittest.TestCase):
    def _make_pipeline(self, nav_output):
        """Build a mock Pipeline whose _infer returns nav_output."""
        pipeline = MagicMock(spec=Pipeline)
        pipeline.encoder.encode.return_value = "emb"
        pipeline.analyzer.return_value = ("feat", None, None)
        pipeline.bridge.return_value = "bridge_out"
        pipeline.navigator.predict.return_value = nav_output
        return pipeline

    @patch("src.proof_search._infer")
    def test_first_call_invokes_infer(self, mock_infer):
        """First call runs _infer and stores result in cache."""
        nav = _make_nav_output(critic=0.7)
        mock_infer.return_value = nav
        pipeline = MagicMock(spec=Pipeline)
        state = _SearchState(open_goals=["g"])

        result = _cached_infer("g", pipeline, state)

        self.assertEqual(result, nav)
        mock_infer.assert_called_once_with("g", pipeline)
        self.assertIn("g", state._infer_cache)

    @patch("src.proof_search._infer")
    def test_second_call_returns_cached(self, mock_infer):
        """Second call with same goal returns cached, no extra _infer call."""
        nav = _make_nav_output(critic=0.8)
        mock_infer.return_value = nav
        pipeline = MagicMock(spec=Pipeline)
        state = _SearchState(open_goals=["g"])

        _cached_infer("g", pipeline, state)
        result2 = _cached_infer("g", pipeline, state)

        self.assertEqual(result2, nav)
        self.assertEqual(mock_infer.call_count, 1)

    @patch("src.proof_search._infer")
    def test_after_cache_clear_calls_infer_again(self, mock_infer):
        """After clearing the cache, _infer is called again."""
        nav1 = _make_nav_output(critic=0.5)
        nav2 = _make_nav_output(critic=0.9)
        mock_infer.side_effect = [nav1, nav2]
        pipeline = MagicMock(spec=Pipeline)
        state = _SearchState(open_goals=["g"])

        result1 = _cached_infer("g", pipeline, state)
        state._infer_cache.clear()
        result2 = _cached_infer("g", pipeline, state)

        self.assertEqual(result1.critic_score, 0.5)
        self.assertEqual(result2.critic_score, 0.9)
        self.assertEqual(mock_infer.call_count, 2)


# ---------------------------------------------------------------------------
# _select_goal
# ---------------------------------------------------------------------------


class TestSelectGoal(unittest.TestCase):
    @patch("src.proof_search._infer")
    def test_single_goal_returns_index_zero(self, mock_infer):
        """Single goal returns (goal, 0) without calling inference."""
        state = _SearchState(open_goals=["only_goal"])
        goal, idx = _select_goal(["only_goal"], MagicMock(), "cpu", state)

        self.assertEqual(goal, "only_goal")
        self.assertEqual(idx, 0)
        mock_infer.assert_not_called()

    @patch("src.proof_search._infer")
    def test_multiple_goals_selects_highest_critic(self, mock_infer):
        """Selects the goal with the highest critic score."""
        nav_low = _make_nav_output(critic=0.2)
        nav_high = _make_nav_output(critic=0.9)
        nav_mid = _make_nav_output(critic=0.5)
        mock_infer.side_effect = [nav_low, nav_high, nav_mid]

        state = _SearchState(open_goals=["g0", "g1", "g2"])
        goal, idx = _select_goal(["g0", "g1", "g2"], MagicMock(), "cpu", state)

        self.assertEqual(goal, "g1")
        self.assertEqual(idx, 1)

    @patch("src.proof_search._infer")
    def test_populates_infer_cache(self, mock_infer):
        """With state, _select_goal populates _infer_cache for all goals."""
        nav_a = _make_nav_output(critic=0.3)
        nav_b = _make_nav_output(critic=0.7)
        mock_infer.side_effect = [nav_a, nav_b]

        state = _SearchState(open_goals=["a", "b"])
        _select_goal(["a", "b"], MagicMock(), "cpu", state)

        self.assertIn("a", state._infer_cache)
        self.assertIn("b", state._infer_cache)


# ---------------------------------------------------------------------------
# _search_step
# ---------------------------------------------------------------------------


class TestSearchStep(unittest.TestCase):
    @patch("src.proof_search._try_candidates")
    @patch("src.proof_search._try_hammer")
    @patch("src.proof_search._cached_infer")
    @patch("src.proof_search._select_goal")
    def test_hammer_delegation_path(self, mock_select, mock_infer, mock_hammer, mock_candidates):
        """When hammer_delegation=True and automation=-1, tries hammer first."""
        nav = _make_nav_output(automation=-1)
        mock_select.return_value = ("g0", 0)
        mock_infer.return_value = nav
        mock_hammer.return_value = True

        state = _SearchState(open_goals=["g0"])
        env = _SearchEnv(conn=MagicMock(), lean=MagicMock(), anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(hammer_delegation=True)
        context = SearchContext()

        _search_step(MagicMock(), state, env, context, cfg)

        mock_hammer.assert_called_once()
        mock_candidates.assert_not_called()

    @patch("src.proof_search._try_candidates")
    @patch("src.proof_search._try_hammer")
    @patch("src.proof_search._cached_infer")
    @patch("src.proof_search._select_goal")
    def test_candidate_success_path(self, mock_select, mock_infer, mock_hammer, mock_candidates):
        """When automation=0, skips hammer and tries candidates."""
        nav = _make_nav_output(automation=0)
        mock_select.return_value = ("g0", 0)
        mock_infer.return_value = nav
        mock_candidates.return_value = True

        state = _SearchState(open_goals=["g0"])
        env = _SearchEnv(conn=MagicMock(), lean=MagicMock(), anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(hammer_delegation=True)
        context = SearchContext()

        _search_step(MagicMock(), state, env, context, cfg)

        mock_hammer.assert_not_called()
        mock_candidates.assert_called_once()

    @patch("src.proof_search._try_candidates")
    @patch("src.proof_search._try_hammer")
    @patch("src.proof_search._cached_infer")
    @patch("src.proof_search._select_goal")
    def test_all_fail_rotates_goal_and_increments(
        self, mock_select, mock_infer, mock_hammer, mock_candidates
    ):
        """All paths fail: goal rotated to back, attempts incremented."""
        nav = _make_nav_output(automation=-1)
        mock_select.return_value = ("g0", 0)
        mock_infer.return_value = nav
        mock_hammer.return_value = False
        mock_candidates.return_value = False

        state = _SearchState(open_goals=["g0", "g1", "g2"])
        env = _SearchEnv(conn=MagicMock(), lean=MagicMock(), anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(hammer_delegation=True)
        context = SearchContext()

        _search_step(MagicMock(), state, env, context, cfg)

        # g0 rotated to end
        self.assertEqual(state.open_goals, ["g1", "g2", "g0"])
        self.assertEqual(state.attempts, 1)

    @patch("src.proof_search._try_candidates")
    @patch("src.proof_search._try_hammer")
    @patch("src.proof_search._cached_infer")
    @patch("src.proof_search._select_goal")
    def test_single_goal_all_fail_no_rotation(
        self, mock_select, mock_infer, mock_hammer, mock_candidates
    ):
        """Single goal, all fail: no rotation (len==1 guard), attempts incremented."""
        nav = _make_nav_output(automation=0)
        mock_select.return_value = ("g0", 0)
        mock_infer.return_value = nav
        mock_candidates.return_value = False

        state = _SearchState(open_goals=["g0"])
        env = _SearchEnv(conn=MagicMock(), lean=MagicMock(), anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(hammer_delegation=False)
        context = SearchContext()

        _search_step(MagicMock(), state, env, context, cfg)

        self.assertEqual(state.open_goals, ["g0"])
        self.assertEqual(state.attempts, 1)

    @patch("src.proof_search._try_candidates")
    @patch("src.proof_search._try_hammer")
    @patch("src.proof_search._cached_infer")
    @patch("src.proof_search._select_goal")
    def test_hammer_disabled_skips_hammer(
        self, mock_select, mock_infer, mock_hammer, mock_candidates
    ):
        """hammer_delegation=False skips hammer even with automation=-1."""
        nav = _make_nav_output(automation=-1)
        mock_select.return_value = ("g0", 0)
        mock_infer.return_value = nav
        mock_candidates.return_value = True

        state = _SearchState(open_goals=["g0"])
        env = _SearchEnv(conn=MagicMock(), lean=MagicMock(), anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(hammer_delegation=False)
        context = SearchContext()

        _search_step(MagicMock(), state, env, context, cfg)

        mock_hammer.assert_not_called()
        mock_candidates.assert_called_once()


if __name__ == "__main__":
    unittest.main()
