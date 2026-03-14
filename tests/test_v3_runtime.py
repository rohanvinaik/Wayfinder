"""Tests for pure functions in src/v3_runtime.py."""

import unittest

from src.nav_contracts import NavOutput
from src.resolution import Candidate
from src.v3_runtime import _build_tactic_text, _should_hammer


def _make_nav_output(directions: dict[str, int]) -> NavOutput:
    """Helper to build a NavOutput with only directions populated."""
    return NavOutput(
        directions=directions,
        direction_confidences={},
        anchor_scores={},
        progress=0.0,
        critic_score=0.0,
    )


def _make_candidate(tactic_name: str, premises: list[str]) -> Candidate:
    """Helper to build a Candidate with tactic_name and premises."""
    return Candidate(tactic_name=tactic_name, premises=premises, score=0.0)


class TestShouldHammer(unittest.TestCase):
    """Tests for _should_hammer(nav_output)."""

    def test_automation_minus_one_returns_true(self) -> None:
        nav = _make_nav_output({"automation": -1})
        self.assertTrue(_should_hammer(nav))

    def test_automation_zero_returns_false(self) -> None:
        nav = _make_nav_output({"automation": 0})
        self.assertFalse(_should_hammer(nav))

    def test_automation_plus_one_returns_false(self) -> None:
        nav = _make_nav_output({"automation": 1})
        self.assertFalse(_should_hammer(nav))

    def test_missing_automation_key_returns_false(self) -> None:
        nav = _make_nav_output({"structure": 1, "domain": -1})
        self.assertFalse(_should_hammer(nav))

    def test_other_banks_do_not_affect_result(self) -> None:
        nav = _make_nav_output(
            {
                "structure": 1,
                "domain": -1,
                "depth": 0,
                "automation": -1,
                "context": 1,
                "decomposition": -1,
            }
        )
        self.assertTrue(_should_hammer(nav))

    def test_empty_directions_returns_false(self) -> None:
        nav = _make_nav_output({})
        self.assertFalse(_should_hammer(nav))


class TestBuildTacticText(unittest.TestCase):
    """Tests for _build_tactic_text(candidate)."""

    def test_no_premises_returns_tactic_name_only(self) -> None:
        candidate = _make_candidate("rw", [])
        self.assertEqual(_build_tactic_text(candidate), "rw")

    def test_one_premise(self) -> None:
        candidate = _make_candidate("apply", ["Nat.add_comm"])
        self.assertEqual(_build_tactic_text(candidate), "apply Nat.add_comm")

    def test_four_premises_all_included(self) -> None:
        premises = ["lemma_a", "lemma_b", "lemma_c", "lemma_d"]
        candidate = _make_candidate("simp", premises)
        self.assertEqual(
            _build_tactic_text(candidate),
            "simp lemma_a lemma_b lemma_c lemma_d",
        )

    def test_more_than_four_premises_truncated(self) -> None:
        premises = ["p1", "p2", "p3", "p4", "p5", "p6"]
        candidate = _make_candidate("simp", premises)
        result = _build_tactic_text(candidate)
        self.assertEqual(result, "simp p1 p2 p3 p4")

    def test_empty_tactic_name_no_premises(self) -> None:
        candidate = _make_candidate("", [])
        self.assertEqual(_build_tactic_text(candidate), "")

    def test_empty_tactic_name_with_premises(self) -> None:
        candidate = _make_candidate("", ["lemma_x"])
        self.assertEqual(_build_tactic_text(candidate), " lemma_x")


if __name__ == "__main__":
    unittest.main()
