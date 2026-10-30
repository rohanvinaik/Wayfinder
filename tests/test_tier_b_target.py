"""Tier B starts the declaration asked for, not the first `sorry` stub above it.

A paper's file is a column of `:= by sorry` stubs; the support slice and the source
replay carry the earlier ones into the wrapper, and `frontend.distil` returns one
target per sorry-bearing declaration in source order.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from src.lean_interface import LeanKernel, _count_sorry_decls, _target_state

WRAPPER = """open Finset

variable {n : ℕ}

def obstructionClass (γ : Fin n → Fin n → ℕ) (s : Finset (Fin n × Fin n)) : ℕ :=
  s.sum (fun e => γ e.1 e.2)

theorem obstruction_zero_iff_all_zero (γ : Fin n → Fin n → ℕ) (s : Finset (Fin n × Fin n)) :
    obstructionClass γ s = 0 ↔ ∀ e ∈ s, γ e.1 e.2 = 0 := by
  sorry

@[simp] theorem sheaf_canonical_iff_all_zero (γ : Fin n → Fin n → ℕ) : True := by
  -- a comment that says sorry is not a sorry
  sorry
theorem _wayfinder_decl_3 (γ : Fin n → Fin n → ℕ) : γ = γ := by
  sorry
"""


def target(goal: str) -> SimpleNamespace:
    return SimpleNamespace(goal_state=SimpleNamespace(goals=[goal]))


class TierBTargetTest(unittest.TestCase):
    def test_counts_sorry_bearing_declarations(self):
        self.assertEqual(_count_sorry_decls(WRAPPER), 3)
        self.assertEqual(_count_sorry_decls("theorem t : True := trivial\n"), 0)

    def test_the_last_target_is_the_declaration_under_replay(self):
        targets = [target("first stub"), target("second stub"), target("the one asked for")]
        self.assertEqual(_target_state(targets, WRAPPER).goals, ["the one asked for"])

    def test_a_missing_target_is_not_misattributed(self):
        # the declaration under replay did not elaborate: only the two stubs came back
        self.assertIsNone(_target_state([target("first stub"), target("second stub")], WRAPPER))

    def test_a_lone_target_is_unchanged(self):
        self.assertEqual(_target_state([target("g")], "theorem t : True := by\n  sorry\n").goals, ["g"])
        self.assertIsNone(_target_state([], WRAPPER))


@unittest.skipUnless(importlib.util.find_spec("pantograph"), "PyPantograph not installed")
class SiteForTest(unittest.TestCase):
    """A tactic acts on its own goal, and the result holds only the goals it produced."""

    def kernel(self) -> LeanKernel:
        k = LeanKernel.__new__(LeanKernel)  # no server: the site is pure bookkeeping
        k._current_env_key = "env"
        k._goal_index = {("env", "q"): 1}
        return k

    def test_the_second_goal_of_a_state_is_addressed_where_it_sits(self):
        site = self.kernel()._site_for("q")
        self.assertEqual((site.goal_id, site.auto_resume), (1, False))

    def test_unknown_goals_default_to_the_first_and_explicit_ids_win(self):
        k = self.kernel()
        self.assertEqual(k._site_for("p").goal_id, 0)
        self.assertEqual(k._site_for("q", goal_id=3).goal_id, 3)


if __name__ == "__main__":
    unittest.main()
