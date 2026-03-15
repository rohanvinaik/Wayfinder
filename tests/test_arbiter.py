"""Tests for the SoM arbiter — orchestrator module."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from src.arbiter import (
    SoMSlots,
    _build_tactic_text,
    _close_goal,
    _recognize_with_retry,
    _should_hammer,
    _SoMSearchState,
)
from src.nav_contracts import NavOutput, TacticResult
from src.proof_search import SearchConfig, SearchResult
from src.resolution import Candidate
from src.som_contracts import RecognitionOutput, SubgoalSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nav_output(
    directions=None,
    confidences=None,
    anchor_scores=None,
    progress=0.5,
    critic=0.8,
):
    return NavOutput(
        directions=directions or {"domain": 1, "context": 0},
        direction_confidences=confidences or {"domain": 0.9, "context": 0.7},
        anchor_scores=anchor_scores or {"simp": 0.8},
        progress=progress,
        critic_score=critic,
    )


def _make_recognition(
    template_id=0,
    template_name="DECIDE",
    confidence=0.9,
    top_k=None,
):
    feats = torch.randn(64)
    if top_k is None:
        top_k = [(template_id, template_name, confidence)]
    return RecognitionOutput(
        template_id=template_id,
        template_name=template_name,
        template_confidence=confidence,
        template_features=feats,
        top_k_templates=top_k,
    )


def _make_candidate(tactic="apply", premises=None, score=0.9):
    return Candidate(
        tactic_name=tactic,
        premises=premises or [],
        score=score,
    )


def _make_tactic_result(success=True, tactic="simp", new_goals=None):
    return TacticResult(
        success=success,
        tactic=tactic,
        premises=[],
        new_goals=new_goals or [],
    )


# ---------------------------------------------------------------------------
# Tests: _should_hammer
# ---------------------------------------------------------------------------


class TestShouldHammer(unittest.TestCase):
    """Tests for the _should_hammer pure function."""

    def test_automation_minus_one_triggers_hammer(self):
        nav = _make_nav_output(directions={"automation": -1})
        spec = SubgoalSpec(subgoal_type="close")
        self.assertTrue(_should_hammer(nav, spec))

    def test_hint_automation_minus_one_triggers_hammer(self):
        nav = _make_nav_output(directions={"automation": 0})
        spec = SubgoalSpec(subgoal_type="close", bank_hints={"automation": -1})
        self.assertTrue(_should_hammer(nav, spec))

    def test_no_automation_signal_no_hammer(self):
        nav = _make_nav_output(directions={"automation": 0})
        spec = SubgoalSpec(subgoal_type="close", bank_hints={"automation": 0})
        self.assertFalse(_should_hammer(nav, spec))

    def test_positive_automation_no_hammer(self):
        nav = _make_nav_output(directions={"automation": 1})
        spec = SubgoalSpec(subgoal_type="close", bank_hints={"automation": 1})
        self.assertFalse(_should_hammer(nav, spec))

    def test_missing_automation_key_defaults_zero(self):
        nav = _make_nav_output(directions={"structure": 1})
        spec = SubgoalSpec(subgoal_type="close")
        self.assertFalse(_should_hammer(nav, spec))

    def test_either_signal_suffices(self):
        """If nav says -1 but hint says +1, still hammer (OR logic)."""
        nav = _make_nav_output(directions={"automation": -1})
        spec = SubgoalSpec(subgoal_type="close", bank_hints={"automation": 1})
        self.assertTrue(_should_hammer(nav, spec))


# ---------------------------------------------------------------------------
# Tests: _build_tactic_text
# ---------------------------------------------------------------------------


class TestBuildTacticText(unittest.TestCase):
    """Tests for the _build_tactic_text pure function."""

    def test_no_premises(self):
        c = _make_candidate(tactic="simp", premises=[])
        self.assertEqual(_build_tactic_text(c), "simp")

    def test_with_premises(self):
        c = _make_candidate(tactic="apply", premises=["lemma1", "lemma2"])
        self.assertEqual(_build_tactic_text(c), "apply lemma1 lemma2")

    def test_truncates_to_four_premises(self):
        c = _make_candidate(
            tactic="exact",
            premises=["a", "b", "c", "d", "e", "f"],
        )
        result = _build_tactic_text(c)
        self.assertEqual(result, "exact a b c d")
        # Only 4 premises included
        self.assertEqual(result.count(" "), 4)

    def test_single_premise(self):
        c = _make_candidate(tactic="rw", premises=["h"])
        self.assertEqual(_build_tactic_text(c), "rw h")


# ---------------------------------------------------------------------------
# Tests: _close_goal
# ---------------------------------------------------------------------------


class TestCloseGoal(unittest.TestCase):
    """Tests for _close_goal state mutation."""

    def test_closes_goal_and_records_tactic(self):
        state = _SoMSearchState(open_goals=["goal_A", "goal_B"])
        result = _make_tactic_result(success=True, tactic="simp")
        _close_goal("goal_A", 0, result, state)
        self.assertNotIn("goal_A", state.open_goals)
        self.assertIn("goal_A", state.closed_goals)
        self.assertIn("simp", state.tactics_used)

    def test_adds_new_goals_from_result(self):
        state = _SoMSearchState(open_goals=["goal_A"])
        result = _make_tactic_result(success=True, tactic="have", new_goals=["sub1", "sub2"])
        _close_goal("goal_A", 0, result, state)
        self.assertEqual(len(state.open_goals), 2)
        self.assertIn("sub1", state.open_goals)
        self.assertIn("sub2", state.open_goals)

    def test_removes_correct_goal_by_index(self):
        state = _SoMSearchState(open_goals=["X", "Y", "Z"])
        result = _make_tactic_result(success=True, tactic="exact")
        _close_goal("Y", 1, result, state)
        self.assertEqual(state.open_goals, ["X", "Z"])
        self.assertEqual(state.closed_goals, ["Y"])

    def test_empty_new_goals(self):
        state = _SoMSearchState(open_goals=["goal_A"])
        result = _make_tactic_result(success=True, tactic="omega", new_goals=[])
        _close_goal("goal_A", 0, result, state)
        self.assertEqual(len(state.open_goals), 0)
        self.assertEqual(len(state.closed_goals), 1)

    def test_clears_feature_cache(self):
        """_close_goal must invalidate the feature cache."""
        state = _SoMSearchState(open_goals=["A", "B"])
        state._feature_cache["A"] = torch.randn(256)
        state._feature_cache["B"] = torch.randn(256)
        self.assertEqual(len(state._feature_cache), 2)
        result = _make_tactic_result(success=True, tactic="simp")
        _close_goal("A", 0, result, state)
        self.assertEqual(len(state._feature_cache), 0)


# ---------------------------------------------------------------------------
# Tests: _SoMSearchState
# ---------------------------------------------------------------------------


class TestSoMSearchState(unittest.TestCase):
    """Tests for the mutable search state dataclass."""

    def test_defaults(self):
        state = _SoMSearchState(open_goals=["g"])
        self.assertEqual(state.open_goals, ["g"])
        self.assertEqual(state.closed_goals, [])
        self.assertEqual(state.tactics_used, [])
        self.assertEqual(state.attempts, 0)
        self.assertEqual(state.templates_tried, {})

    def test_field_isolation(self):
        s1 = _SoMSearchState(open_goals=["a"])
        s2 = _SoMSearchState(open_goals=["b"])
        s1.closed_goals.append("x")
        self.assertEqual(len(s2.closed_goals), 0)

    def test_templates_tried_isolation(self):
        s1 = _SoMSearchState(open_goals=["a"])
        s2 = _SoMSearchState(open_goals=["b"])
        s1.templates_tried["g1"] = [0, 1]
        self.assertNotIn("g1", s2.templates_tried)


# ---------------------------------------------------------------------------
# Tests: _recognize_with_retry
# ---------------------------------------------------------------------------


class TestRecognizeWithRetry(unittest.TestCase):
    """Tests for template retry logic."""

    def _make_classifier_mock(self, top_k):
        """Create a mock classifier that returns specified top_k."""
        classifier = MagicMock()
        recognition = _make_recognition(
            template_id=top_k[0][0],
            template_name=top_k[0][1],
            confidence=top_k[0][2],
            top_k=top_k,
        )
        classifier.predict.return_value = recognition
        return classifier

    def test_returns_first_untried(self):
        top_k = [(0, "DECIDE", 0.5), (1, "REWRITE_CHAIN", 0.3), (2, "APPLY_CHAIN", 0.2)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[])
        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, 0)
        self.assertEqual(result.template_name, "DECIDE")

    def test_skips_tried_template(self):
        top_k = [(0, "DECIDE", 0.5), (1, "REWRITE_CHAIN", 0.3), (2, "APPLY_CHAIN", 0.2)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[0])
        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, 1)
        self.assertEqual(result.template_name, "REWRITE_CHAIN")

    def test_skips_multiple_tried(self):
        top_k = [(0, "DECIDE", 0.5), (1, "REWRITE_CHAIN", 0.3), (2, "APPLY_CHAIN", 0.2)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[0, 1])
        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, 2)

    def test_returns_none_when_all_tried(self):
        top_k = [(0, "DECIDE", 0.5), (1, "REWRITE_CHAIN", 0.3)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[0, 1])
        self.assertIsNone(result)

    def test_preserves_template_features(self):
        top_k = [(0, "DECIDE", 0.9)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[])
        self.assertIsNotNone(result)
        self.assertEqual(result.template_features.shape, (64,))

    def test_preserves_top_k_in_output(self):
        top_k = [(0, "DECIDE", 0.5), (1, "REWRITE_CHAIN", 0.3)]
        classifier = self._make_classifier_mock(top_k)
        features = torch.randn(1, 256)

        result = _recognize_with_retry(features, classifier, tried_template_ids=[])
        self.assertIsNotNone(result)
        self.assertEqual(len(result.top_k_templates), 2)


# ---------------------------------------------------------------------------
# Tests: SoMSlots
# ---------------------------------------------------------------------------


class TestSoMSlots(unittest.TestCase):
    """Tests for the SoMSlots dataclass."""

    def test_construction_with_mocks(self):
        slots = SoMSlots(
            encoder=MagicMock(),
            analyzer=MagicMock(),
            classifier=MagicMock(),
            sketch_predictor=MagicMock(),
            execution=MagicMock(),
            lean=MagicMock(),
        )
        self.assertIsNotNone(slots.encoder)
        self.assertIsNotNone(slots.lean)

    def test_all_fields_present(self):
        slots = SoMSlots(
            encoder=MagicMock(),
            analyzer=MagicMock(),
            classifier=MagicMock(),
            sketch_predictor=MagicMock(),
            execution=MagicMock(),
            lean=MagicMock(),
        )
        field_names = {"encoder", "analyzer", "classifier", "sketch_predictor", "execution", "lean"}
        for name in field_names:
            self.assertTrue(hasattr(slots, name), f"Missing field: {name}")


# ---------------------------------------------------------------------------
# Tests: som_search (integration with mocks)
# ---------------------------------------------------------------------------


class TestSomSearchIntegration(unittest.TestCase):
    """Integration tests for som_search using mocked slots."""

    def _make_mock_slots(self, lean_results=None):
        """Create SoMSlots with mocked components.

        Args:
            lean_results: List of TacticResult to return from try_tactic.
        """
        if lean_results is None:
            lean_results = [_make_tactic_result(success=True, tactic="simp")]

        encoder = MagicMock()
        encoder.encode.return_value = torch.randn(1, 384)

        analyzer = MagicMock()
        analyzer.return_value = (torch.randn(1, 256), None, None)

        classifier = MagicMock()
        recognition = _make_recognition(
            template_id=0,
            template_name="DECIDE",
            confidence=0.9,
            top_k=[(0, "DECIDE", 0.9)],
        )
        classifier.predict.return_value = recognition

        sketch_predictor = MagicMock()
        from src.som_contracts import PlanningOutput

        sketch_predictor.predict.return_value = PlanningOutput(
            sketch=[SubgoalSpec(subgoal_type="automation_close", bank_hints={"automation": 0})],
            total_estimated_depth=1,
            template_id=0,
        )

        execution = MagicMock()
        execution.predict.return_value = _make_nav_output()

        lean = MagicMock()
        lean.try_tactic.side_effect = lean_results
        lean.try_hammer.side_effect = lean_results

        return SoMSlots(
            encoder=encoder,
            analyzer=analyzer,
            classifier=classifier,
            sketch_predictor=sketch_predictor,
            execution=execution,
            lean=lean,
        )

    @patch("src.arbiter.resolve")
    def test_successful_proof(self, mock_resolve):
        """Single goal closed by first tactic attempt."""
        mock_resolve.return_value = [_make_candidate(tactic="simp")]
        slots = self._make_mock_slots(
            lean_results=[_make_tactic_result(success=True, tactic="simp")]
        )
        import sqlite3

        conn = sqlite3.connect(":memory:")

        from src.arbiter import SoMSearchParams, som_search

        result = som_search(
            theorem_id="thm1",
            initial_goal="goal_state",
            slots=slots,
            conn=conn,
            params=SoMSearchParams(config=SearchConfig(budget=10, hammer_delegation=False)),
        )
        conn.close()

        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.success, True)
        self.assertEqual(result.theorem_id, "thm1")
        self.assertGreater(result.goals_closed, 0)
        self.assertEqual(result.goals_remaining, 0)

    @patch("src.arbiter.resolve")
    def test_failed_proof_budget_exhausted(self, mock_resolve):
        """All tactics fail, budget is exhausted."""
        mock_resolve.return_value = [_make_candidate(tactic="simp")]
        slots = self._make_mock_slots(
            lean_results=[_make_tactic_result(success=False, tactic="simp")] * 20
        )
        import sqlite3

        conn = sqlite3.connect(":memory:")

        from src.arbiter import SoMSearchParams, som_search

        result = som_search(
            theorem_id="thm2",
            initial_goal="hard_goal",
            slots=slots,
            conn=conn,
            params=SoMSearchParams(
                config=SearchConfig(budget=5, hammer_delegation=False),
                max_template_retries=2,
            ),
        )
        conn.close()

        self.assertFalse(result.success)
        self.assertEqual(result.goals_remaining, 1)
        self.assertLessEqual(result.attempts, 5)

    @patch("src.arbiter.resolve")
    def test_template_retry_limit(self, mock_resolve):
        """Template retry is bounded by max_template_retries."""
        mock_resolve.return_value = [_make_candidate(tactic="simp")]
        slots = self._make_mock_slots(
            lean_results=[_make_tactic_result(success=False, tactic="simp")] * 50
        )
        import sqlite3

        conn = sqlite3.connect(":memory:")

        from src.arbiter import SoMSearchParams, som_search

        result = som_search(
            theorem_id="thm3",
            initial_goal="hard_goal",
            slots=slots,
            conn=conn,
            params=SoMSearchParams(
                config=SearchConfig(budget=20, hammer_delegation=False),
                max_template_retries=1,
            ),
        )
        conn.close()

        self.assertFalse(result.success)

    @patch("src.arbiter.resolve")
    def test_proof_with_new_subgoals(self, mock_resolve):
        """Closing a goal that spawns subgoals — both must be closed."""
        mock_resolve.return_value = [_make_candidate(tactic="have")]
        lean_results = [
            # First goal spawns 2 subgoals
            _make_tactic_result(success=True, tactic="have", new_goals=["sub1", "sub2"]),
            # Close sub1
            _make_tactic_result(success=True, tactic="simp"),
            # Close sub2
            _make_tactic_result(success=True, tactic="omega"),
        ]
        slots = self._make_mock_slots(lean_results=lean_results)
        import sqlite3

        conn = sqlite3.connect(":memory:")

        from src.arbiter import SoMSearchParams, som_search

        result = som_search(
            theorem_id="thm4",
            initial_goal="composed_goal",
            slots=slots,
            conn=conn,
            params=SoMSearchParams(config=SearchConfig(budget=20, hammer_delegation=False)),
        )
        conn.close()

        self.assertEqual(result.success, True)
        self.assertEqual(result.goals_closed, 3)
        self.assertEqual(result.goals_remaining, 0)

    @patch("src.arbiter.resolve")
    def test_returns_search_result_type(self, mock_resolve):
        mock_resolve.return_value = []
        slots = self._make_mock_slots()
        import sqlite3

        conn = sqlite3.connect(":memory:")

        from src.arbiter import SoMSearchParams, som_search

        result = som_search(
            theorem_id="thm5",
            initial_goal="goal",
            slots=slots,
            conn=conn,
            params=SoMSearchParams(
                config=SearchConfig(budget=3, hammer_delegation=False),
                max_template_retries=1,
            ),
        )
        conn.close()

        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.theorem_id, "thm5")


if __name__ == "__main__":
    unittest.main()
