"""Tests for proof_search — pure functions and dataclasses."""

import unittest

from src.nav_contracts import NavOutput
from src.proof_search import (
    SearchConfig,
    _build_tactic_text,
    _SearchState,
    _should_hammer,
)
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


# ---------------------------------------------------------------------------
# _should_hammer
# ---------------------------------------------------------------------------


class TestShouldHammer(unittest.TestCase):
    def test_automation_neg1_returns_true(self):
        nav = _make_nav_output(automation=-1)
        self.assertEqual(_should_hammer(nav), True)

    def test_automation_zero_returns_false(self):
        nav = _make_nav_output(automation=0)
        self.assertEqual(_should_hammer(nav), False)

    def test_automation_pos1_returns_false(self):
        nav = _make_nav_output(automation=1)
        self.assertEqual(_should_hammer(nav), False)

    def test_automation_key_missing_returns_false(self):
        nav = _make_nav_output(automation=None)  # directions dict is empty
        self.assertEqual(_should_hammer(nav), False)


# ---------------------------------------------------------------------------
# _build_tactic_text
# ---------------------------------------------------------------------------


class TestBuildTacticText(unittest.TestCase):
    def test_no_premises_returns_tactic_name(self):
        c = _make_candidate(tactic="rfl", premises=[])
        self.assertEqual(_build_tactic_text(c), "rfl")

    def test_one_premise(self):
        c = _make_candidate(tactic="exact", premises=["Nat.add_comm"])
        self.assertEqual(_build_tactic_text(c), "exact Nat.add_comm")

    def test_four_premises_all_included(self):
        prems = ["a", "b", "c", "d"]
        c = _make_candidate(tactic="apply", premises=prems)
        self.assertEqual(_build_tactic_text(c), "apply a b c d")

    def test_six_premises_only_first_four(self):
        prems = ["a", "b", "c", "d", "e", "f"]
        c = _make_candidate(tactic="simp", premises=prems)
        self.assertEqual(_build_tactic_text(c), "simp a b c d")


# ---------------------------------------------------------------------------
# _SearchState dataclass
# ---------------------------------------------------------------------------


class TestSearchState(unittest.TestCase):
    def test_open_goals_required(self):
        state = _SearchState(open_goals=["goal1"])
        self.assertEqual(state.open_goals, ["goal1"])

    def test_defaults_empty(self):
        state = _SearchState(open_goals=["g"])
        self.assertEqual(state.closed_goals, [])
        self.assertEqual(state.tactics_used, [])
        self.assertEqual(state.attempts, 0)

    def test_default_factory_isolation(self):
        """Two instances must have independent lists."""
        s1 = _SearchState(open_goals=["a"])
        s2 = _SearchState(open_goals=["b"])
        s1.closed_goals.append("x")
        s1.tactics_used.append("t")
        self.assertEqual(s2.closed_goals, [])
        self.assertEqual(s2.tactics_used, [])


# ---------------------------------------------------------------------------
# SearchConfig dataclass
# ---------------------------------------------------------------------------


class TestSearchConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.budget, 600)
        self.assertEqual(cfg.hammer_delegation, True)
        self.assertEqual(cfg.accessible_premises, True)
        self.assertEqual(cfg.max_candidates_per_step, 8)
        self.assertEqual(cfg.device, "cpu")


if __name__ == "__main__":
    unittest.main()
