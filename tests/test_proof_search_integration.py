"""Tests for proof_search — search() integration with mocks."""

import unittest
from unittest.mock import MagicMock, patch

from src.nav_contracts import NavOutput, TacticResult
from src.proof_search import SearchConfig, search
from src.resolution import Candidate

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


def _make_pipeline():
    """Build a fully-mocked Pipeline.

    We avoid spec=Pipeline because MagicMock(spec=dataclass) blocks attribute
    access on the mock instance. A plain MagicMock with pre-wired return values
    lets _infer chain (encoder -> analyzer -> bridge -> navigator) run cleanly.
    """
    pipeline = MagicMock()
    pipeline.encoder.encode.return_value = "embeddings"
    pipeline.analyzer.return_value = ("features", None, None)
    pipeline.bridge.return_value = "bridge_out"
    pipeline.navigator.predict.return_value = _make_nav_output()
    return pipeline


def _make_tactic_result(success, tactic="apply", premises=None, new_goals=None):
    """Build a TacticResult."""
    return TacticResult(
        success=success,
        tactic=tactic,
        premises=premises or [],
        new_goals=new_goals or [],
    )


# ---------------------------------------------------------------------------
# search (integration with mocks)
# ---------------------------------------------------------------------------


class TestSearch(unittest.TestCase):
    def _setup_pipeline_for_search(self, nav_output=None):
        """Build pipeline mock that returns a fixed NavOutput through _infer."""
        nav = nav_output or _make_nav_output()
        pipeline = _make_pipeline()
        pipeline.navigator.predict.return_value = nav
        return pipeline

    @patch("src.proof_search.resolve")
    def test_budget_exhaustion(self, mock_resolve):
        """Budget=1 with failing candidates: success=False, goals_remaining>0."""
        mock_resolve.return_value = [_make_candidate(tactic="t1")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(success=False, tactic="t1")

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(budget=1, hammer_delegation=False)

        result = search(
            "thm1",
            "initial_goal",
            pipeline,
            MagicMock(),
            lean,
            config=cfg,
        )

        self.assertEqual(result.success, False)
        self.assertEqual(result.theorem_id, "thm1")
        self.assertEqual(result.goals_remaining, 1)

    @patch("src.proof_search._try_structural_fallback", return_value=False)
    @patch("src.proof_search.resolve")
    def test_all_goals_closed(self, mock_resolve, _mock_structural):
        """Single goal, first candidate succeeds with no new goals: success."""
        mock_resolve.return_value = [_make_candidate(tactic="rfl")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="rfl",
            new_goals=[],
        )

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(hammer_delegation=False)

        result = search("thm_easy", "⊢ True", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, True)
        self.assertEqual(result.theorem_id, "thm_easy")
        self.assertEqual(result.goals_closed, 1)
        self.assertEqual(result.goals_remaining, 0)
        self.assertEqual(result.tactics_used, ["rfl"])

    @patch("src.proof_search.resolve")
    def test_hammer_delegation_path(self, mock_resolve):
        """Hammer delegation: automation=-1 triggers try_hammer first."""
        hammer_nav = _make_nav_output(automation=-1)
        pipeline = self._setup_pipeline_for_search(nav_output=hammer_nav)

        mock_resolve.return_value = [_make_candidate(tactic="omega", premises=["h"])]

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(
            success=True,
            tactic="omega",
            premises=["h"],
            new_goals=[],
        )

        cfg = SearchConfig(hammer_delegation=True)

        result = search("thm_auto", "⊢ 1 + 1 = 2", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, True)
        self.assertEqual(result.goals_closed, 1)
        self.assertEqual(result.goals_remaining, 0)
        lean.try_hammer.assert_called()

    @patch("src.proof_search.resolve")
    def test_hammer_disabled_skips_hammer(self, mock_resolve):
        """When hammer_delegation=False, hammer is never called."""
        hammer_nav = _make_nav_output(automation=-1)
        pipeline = self._setup_pipeline_for_search(nav_output=hammer_nav)

        mock_resolve.return_value = [_make_candidate(tactic="exact")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="exact",
            new_goals=[],
        )

        cfg = SearchConfig(hammer_delegation=False)

        result = search("thm", "g", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, True)
        lean.try_hammer.assert_not_called()

    @patch("src.proof_search.resolve")
    def test_default_config_used_when_none(self, mock_resolve):
        """config=None uses SearchConfig defaults."""
        mock_resolve.return_value = [_make_candidate(tactic="rfl")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="rfl",
            new_goals=[],
        )

        pipeline = self._setup_pipeline_for_search()

        result = search("thm", "g", pipeline, MagicMock(), lean, config=None)

        self.assertEqual(result.success, True)

    @patch("src.proof_search._try_structural_fallback", return_value=False)
    @patch("src.proof_search.resolve")
    def test_accessible_premises_passed_to_context(self, mock_resolve, _mock_structural):
        """accessible_theorem_id flows through when accessible_premises=True."""
        mock_resolve.return_value = [_make_candidate(tactic="rfl")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="rfl",
            new_goals=[],
        )

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(accessible_premises=True, hammer_delegation=False)

        search("thm", "g", pipeline, MagicMock(), lean, config=cfg, accessible_theorem_id=42)

        # resolve was called; verify context had the accessible_theorem_id
        call_args = mock_resolve.call_args
        context_arg = call_args[0][2]  # third positional arg is context
        self.assertEqual(context_arg.accessible_theorem_id, 42)

    @patch("src.proof_search._try_structural_fallback", return_value=False)
    @patch("src.proof_search.resolve")
    def test_accessible_premises_disabled_clears_theorem_id(
        self, mock_resolve, _mock_structural
    ):
        """accessible_premises=False means context.accessible_theorem_id is None."""
        mock_resolve.return_value = [_make_candidate(tactic="rfl")]

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="rfl",
            new_goals=[],
        )

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(accessible_premises=False, hammer_delegation=False)

        search("thm", "g", pipeline, MagicMock(), lean, config=cfg, accessible_theorem_id=42)

        call_args = mock_resolve.call_args
        context_arg = call_args[0][2]
        self.assertIsNone(context_arg.accessible_theorem_id)

    @patch("src.proof_search.resolve")
    def test_multi_goal_rotation_on_failure(self, mock_resolve):
        """When candidates fail and multiple goals exist, failed goal rotates."""
        call_count = [0]

        def resolve_side_effect(*args, **kwargs):
            call_count[0] += 1
            return [_make_candidate(tactic=f"t{call_count[0]}")]

        mock_resolve.side_effect = resolve_side_effect

        lean = MagicMock()
        lean.try_tactic.return_value = _make_tactic_result(success=False, tactic="t")

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(budget=2, hammer_delegation=False, max_candidates_per_step=1)

        result = search("thm", "g1", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, False)
        self.assertEqual(result.goals_remaining, 1)

    @patch("src.proof_search.resolve")
    def test_hammer_fails_falls_through_to_candidates(self, mock_resolve):
        """Hammer fails, then candidates are tried in the same iteration."""
        hammer_nav = _make_nav_output(automation=-1)
        pipeline = self._setup_pipeline_for_search(nav_output=hammer_nav)

        mock_resolve.return_value = [_make_candidate(tactic="exact", premises=["h"])]

        lean = MagicMock()
        lean.try_hammer.return_value = _make_tactic_result(success=False, tactic="hammer")
        lean.try_tactic.return_value = _make_tactic_result(
            success=True,
            tactic="exact h",
            new_goals=[],
        )

        cfg = SearchConfig(hammer_delegation=True)

        result = search("thm", "g", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, True)
        lean.try_hammer.assert_called()
        lean.try_tactic.assert_called()

    @patch("src.proof_search._try_structural_fallback", return_value=False)
    @patch("src.proof_search.resolve")
    def test_subgoals_spawn_and_close(self, mock_resolve, _mock_struct):
        """First tactic spawns a subgoal; second iteration closes it."""
        mock_resolve.return_value = [_make_candidate(tactic="intro")]

        tactic_results = [
            _make_tactic_result(success=True, tactic="intro", new_goals=["sub1"]),
            _make_tactic_result(success=True, tactic="intro", new_goals=[]),
        ]

        lean = MagicMock()
        lean.try_tactic.side_effect = tactic_results

        pipeline = self._setup_pipeline_for_search()
        cfg = SearchConfig(hammer_delegation=False)

        result = search("thm_sub", "g0", pipeline, MagicMock(), lean, config=cfg)

        self.assertEqual(result.success, True)
        self.assertEqual(result.goals_closed, 2)
        self.assertEqual(result.goals_remaining, 0)
        self.assertEqual(result.tactics_used, ["intro", "intro"])


if __name__ == "__main__":
    unittest.main()
