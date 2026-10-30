"""Wayfinder's search on a residual's holes where they sit: each hole loaded as the only `sorry`
(the others closed by a placeholder axiom, so Lean returns that hole's goal alone), and what the
search builds spliced back — complete, or its validated steps with the goals still open marked."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from src import hole_fill as hf
from src import hole_search as hs

DECL = """theorem t (a : ℕ) (h : a = 1 ∨ a = 2) : a * a ≤ 4 ∧ 0 ≤ a := by
  constructor
  -- WAYFINDER RESIDUAL 1: close this goal
  --   ⊢ a * a ≤ 4
  · sorry
  -- WAYFINDER RESIDUAL 2: close this goal
  --   ⊢ 0 ≤ a
  · sorry"""

FILE = "import Mathlib\nimport Pabkit\n\nopen Real\n\n" + DECL + "\n"


def _rec(goal, gid, before_ids, tactic, new_ids=(), new_goals=None):
    new_goals = list(new_goals or [goal] * len(new_ids))
    return {"goal": goal, "goal_id": gid, "goal_idx": list(before_ids).index(gid), "open_before": [goal] * len(before_ids),
            "open_before_ids": list(before_ids), "tactic": tactic, "lean_tactic": tactic, "new_goals": new_goals,
            "new_goal_ids": list(new_ids), "provenance": "test", "prepend": False}  # fmt: skip


def _tactic_lines(block: str) -> list[str]:
    return [ln.strip().removeprefix("· ") for ln in block.split("\n") if ln.strip() and not ln.strip().startswith("--")]


class HoleSourceTest(unittest.TestCase):
    def test_every_hole_but_one_is_closed_by_the_placeholder(self):
        text = hs.only_hole(DECL, 1)
        self.assertEqual(text.count("sorry"), 1)
        self.assertIn(f"· exact {hs.PLACEHOLDER}", text)
        self.assertEqual(text.split("\n")[4], "  · sorry")  # hole 1 stays where it was

    def test_the_source_is_the_file_without_imports_with_the_axiom_above(self):
        src = hs.hole_source(FILE, DECL, 2)
        self.assertNotIn("import ", src)
        self.assertIn("open Real", src)
        self.assertLess(src.index(hs.PLACEHOLDER_AXIOM), src.index("theorem t"))
        self.assertEqual(src.count("sorry"), 1)


class HoleBlockTest(unittest.TestCase):
    def test_partial_progress_is_kept_with_the_open_goal_marked(self):
        log = [_rec("a * a ≤ 4", 1, [1], "rcases h with h | h", (2, 3)),
               _rec("a * a ≤ 4", 2, [2, 3], "subst h; norm_num")]  # fmt: skip
        block, n = hs.hole_block(["a * a ≤ 4"], [1], log)
        self.assertEqual(n, 1)
        self.assertEqual(_tactic_lines(block), ["rcases h with h | h", "subst h; norm_num", "sorry"])
        self.assertEqual(len(hf.hole_sites(block)), 1)

    def test_a_closed_hole_is_a_complete_block(self):
        log = [_rec("a * a ≤ 4", 1, [1], "rcases h with h | h", (2, 3)),
               _rec("a * a ≤ 4", 2, [2, 3], "subst h; norm_num"), _rec("a * a ≤ 4", 3, [3], "subst h; norm_num")]  # fmt: skip
        block, n = hs.hole_block(["a * a ≤ 4"], [1], log)
        self.assertEqual(n, 0)
        self.assertNotIn("sorry", block)

    def test_an_unproven_commitment_is_withdrawn(self):
        # `use 3` chose a witness and left `3 > 2` open: nothing it chose is proven, so it does not stand
        log = [_rec("∃ n, n > 2", 1, [1], "use 3", (2,), ["3 > 2"])]
        block, n = hs.hole_block(["∃ n, n > 2"], [1], log)
        self.assertEqual(n, 1)
        self.assertEqual(_tactic_lines(block), ["sorry"])
        self.assertIn("use 3", block)  # noted above the hole as tried and withdrawn


class GoalCommentsTest(unittest.TestCase):
    """Lean's goal under each hole, as Goedel reads it: a probe line per hole, then the comments."""

    def test_the_probe_reports_each_hole_on_its_own_line(self):
        probe, at = hs.probe_declaration(DECL)
        lines = probe.split("\n")
        self.assertEqual(sorted(at.values()), [1, 2])
        for i in at:
            self.assertEqual(lines[i].strip(), "· trace_state")
            self.assertEqual(lines[i + 1].strip(), "sorry")

    def test_the_goals_replace_each_holes_comments(self):
        goals = {1: "a : ℕ\nh : a = 1 ∨ a = 2\n⊢ a * a ≤ 4", 2: "a : ℕ\n⊢ 0 ≤ a"}
        out = hs.with_goals(DECL, goals)
        self.assertIn("  --   h : a = 1 ∨ a = 2", out)
        self.assertNotIn("--   ⊢ 0 ≤ a\n  --   ⊢ 0 ≤ a", out)
        self.assertEqual(sorted(hf.hole_sites(out)), [1, 2])
        self.assertEqual(out.count("sorry"), 2)


class SpliceTest(unittest.TestCase):
    def test_a_block_with_its_own_hole_is_spliced_and_the_holes_renumbered(self):
        block = "rcases h with h | h\n· omega\n-- WAYFINDER RESIDUAL 1: close this goal\n--   ⊢ a * a ≤ 4\n· sorry"
        out = hs.splice(DECL, 1, block)
        self.assertEqual(sorted(hf.hole_sites(out)), [1, 2])
        self.assertIn("rcases h with h | h", out)
        self.assertEqual(hs.open_holes(out), 2)
        self.assertEqual(hs.unmarked_sorries(out), 0)


_PROJECT = Path(__file__).resolve().parents[1] / "data" / "lean_project_v428"
_LIVE = importlib.util.find_spec("pantograph") is not None and (_PROJECT / ".lake").exists()

TWO_HOLES = """theorem wf_two_holes (a b : ℕ) (h : a = b) : a + 0 = b ∧ b = a := by
  constructor
  -- WAYFINDER RESIDUAL 1: close this goal
  · sorry
  -- WAYFINDER RESIDUAL 2: close this goal
  · have hb : b = a := h.symm
    -- WAYFINDER RESIDUAL 3: close this goal
    sorry"""


@unittest.skipUnless(_LIVE, "PyPantograph or the local Mathlib project is missing")
class LiveHoleGoalTest(unittest.TestCase):
    """Each hole loads as exactly one goal — the one Lean has there, with that hole's hypotheses."""

    @classmethod
    def setUpClass(cls):
        from src.lean_interface import LeanConfig, LeanKernel

        cls.k = LeanKernel(LeanConfig(backend="pantograph", timeout=300, project_root=str(_PROJECT),
                                      imports=["Mathlib"]))  # fmt: skip

    @classmethod
    def tearDownClass(cls):
        cls.k.close()

    def test_each_hole_is_its_own_goal(self):
        file_text = "import Mathlib\n\n" + TWO_HOLES + "\n"
        g1 = self.k.goal_from_source(hs.hole_source(file_text, TWO_HOLES, 1))
        g3 = self.k.goal_from_source(hs.hole_source(file_text, TWO_HOLES, 3))
        self.assertEqual((g1, g3), ("a + 0 = b", "b = a"))
        self.assertNotIn("hb", [n for n, _ in self.k.goal_variables(g1)])
        self.assertIn("hb", [n for n, _ in self.k.goal_variables(g3)])
        self.assertEqual(self.k.try_tactic(g3, "exact hb").new_goals, [])


if __name__ == "__main__":
    unittest.main()
