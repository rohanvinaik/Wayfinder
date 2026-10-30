"""Goal identity: Wayfinder names each goal by the live Lean goal it is, never by its target text.

A move that changes only the hypotheses — `have`, `obtain`, `rcases` — is progress, and the goals it
leaves keep their new facts. Identity never comes from the text; loops are still seen, because two
goals with the same hypotheses and target are the same content whatever their handles."""

from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import unittest
from pathlib import Path

from src import dr_ducky_executor as dx
from src import proof_search as ps
from src import residual_handoff as rh
from src.lean_interface import GoalRef, LeanConfig, LeanKernel
from src.nav_contracts import TacticResult

WF = Path(__file__).resolve().parents[1]
MATHLIB_PROJECT = WF / "data" / "lean_project_v428"  # the search and the gate: Lean 4.28
_LIVE = importlib.util.find_spec("pantograph") is not None and (MATHLIB_PROJECT / ".lake").exists()


class GoalRefTest(unittest.TestCase):
    """The handle's text is the target every lane reads; its identity is the live goal."""

    def test_one_target_two_goals(self):
        a, b = GoalRef("p", 1, "h : q\n⊢ p"), GoalRef("p", 2, "⊢ p")
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)
        self.assertEqual(a, "p")  # a plain-text comparison stays textual

    def test_identity_survives_str_copy_and_pickle(self):
        a = GoalRef("p", 7, "⊢ p")
        for b in (str(a), copy.copy(a), copy.deepcopy(a), pickle.loads(pickle.dumps(a))):
            self.assertEqual((b, b.gid, b.rendering), (a, 7, "⊢ p"))

    def test_text_views_and_serialization_are_the_target(self):
        a = GoalRef("x + 1 > 0", 3, "x : ℕ\n⊢ x + 1 > 0")
        self.assertEqual(json.dumps(a), '"x + 1 > 0"')
        self.assertEqual(f"{a}", "x + 1 > 0")
        self.assertEqual(a.strip(), "x + 1 > 0")


class GoalContentTest(unittest.TestCase):
    """Loop and duplicate reads compare whole goals — hypotheses and target — not handles."""

    @staticmethod
    def tags(before, after, tactic):
        return ps._progress_pathology_tags(goals_before=before, goals_after=after, tactic=tactic,
                                           recent_signatures=[ps._goal_signature(before)], loop_window=3)  # fmt: skip

    def test_a_hypothesis_only_move_is_not_a_loop(self):
        before = [GoalRef("a % 2 = 0", 1, "a : ℕ\nh : ∃ x, a = 2 * x\n⊢ a % 2 = 0")]
        after = [GoalRef("a % 2 = 0", 2, "a x : ℕ\nhx : a = 2 * x\n⊢ a % 2 = 0")]
        self.assertNotIn("state_loop", self.tags(before, after, "obtain ⟨x, hx⟩ := h"))

    def test_fresh_handles_do_not_hide_a_genuine_loop(self):
        g1, g2 = GoalRef("p", 1, "h : q\n⊢ p"), GoalRef("p", 2, "h : q\n⊢ p")
        self.assertIn("state_loop", self.tags([g1], [g2], "simp"))

    def test_case_branches_with_one_target_are_not_duplicates(self):
        root = GoalRef("a * a ≤ 4", 1, "a : ℕ\nh : a = 1 ∨ a = 2\n⊢ a * a ≤ 4")
        b1 = GoalRef("a * a ≤ 4", 2, "a : ℕ\nh : a = 1\n⊢ a * a ≤ 4")
        b2 = GoalRef("a * a ≤ 4", 3, "a : ℕ\nh : a = 2\n⊢ a * a ≤ 4")
        self.assertNotIn("duplicate_goal_progress", self.tags([root], [b1, b2], "rcases h with h | h"))

    def test_one_shot_guards_are_per_goal_content(self):
        original = GoalRef("a * b + a > 0", 1, "a b : ℝ\n⊢ a * b + a > 0")
        continuation = GoalRef("a * b + a > 0", 2, "a b : ℝ\nhab : 0 < a * b\n⊢ a * b + a > 0")
        same = GoalRef("a * b + a > 0", 3, "a b : ℝ\n⊢ a * b + a > 0")
        self.assertNotEqual(ps._tried_key("dr_ducky", original), ps._tried_key("dr_ducky", continuation))
        self.assertEqual(ps._tried_key("dr_ducky", original), ps._tried_key("dr_ducky", same))
        self.assertEqual(ps._tried_key("dr_ducky", "plain"), ps._tried_key("dr_ducky", "plain"))


class TreeIdentityTest(unittest.TestCase):
    """The committed tree pairs each closure with its own goal, even when targets coincide."""

    @staticmethod
    def rec(goal_id, before_ids, tactic, new_ids=(), text="a * a ≤ 4"):
        return {"goal": text, "goal_id": goal_id, "goal_idx": before_ids.index(goal_id),
                "open_before": [text] * len(before_ids), "open_before_ids": list(before_ids),
                "tactic": tactic, "lean_tactic": tactic, "new_goals": [text] * len(new_ids),
                "new_goal_ids": list(new_ids), "provenance": "test", "prepend": False}  # fmt: skip

    def test_branches_with_one_target_keep_their_own_subtrees(self):
        log = [self.rec(1, [1], "rcases h with h | h", (2, 3)),
               self.rec(3, [3, 2], "subst h; norm_num"),  # the search rotated: the second branch first
               self.rec(2, [2], "omega")]  # fmt: skip
        tree = rh.build_tree(["a * a ≤ 4"], log, initial_ids=[1])
        by_gid = {n.gid: n for n in tree.nodes}
        self.assertEqual((by_gid[2].tactic, by_gid[3].tactic), ("omega", "subst h; norm_num"))
        self.assertEqual(tree.nodes[0].children, [1, 2])  # node order is the rcases goals' order
        self.assertEqual(tree.open_ids, [])

    def test_a_log_without_ids_still_builds_by_text(self):
        log = [{"goal": "p ∧ q", "goal_idx": 0, "open_before": ["p ∧ q"], "tactic": "constructor",
                "lean_tactic": "constructor", "new_goals": ["p", "q"], "provenance": "t", "prepend": False},
               {"goal": "q", "goal_idx": 1, "open_before": ["p", "q"], "tactic": "trivial",
                "lean_tactic": "trivial", "new_goals": [], "provenance": "t", "prepend": False}]  # fmt: skip
        tree = rh.build_tree(["p ∧ q"], log)
        self.assertEqual([tree.nodes[i].goal for i in tree.open_ids], ["p"])

    def test_a_closure_of_another_goal_is_refused(self):
        log = [self.rec(1, [1], "rcases h with h | h", (2, 3)), {**self.rec(3, [2, 3], "omega"), "goal_idx": 0}]
        with self.assertRaises(rh.ReplayError):
            rh.build_tree(["a * a ≤ 4"], log, initial_ids=[1])


class _StubLean:
    """A kernel whose moves are a table: (goal text, tactic) -> the new goals' texts, or absent."""

    def __init__(self, moves: dict[tuple[str, str], list[str]]):
        self.moves, self.n = moves, 100

    def try_tactic(self, goal, tactic, goal_id=0):
        out = self.moves.get((str.__str__(goal), tactic))
        if out is None:
            return TacticResult(success=False, tactic=tactic, premises=[], new_goals=[], error_message="no")
        new = []
        for text in out:
            self.n += 1
            new.append(GoalRef(text, self.n, f"h{self.n} : True\n⊢ {text}"))
        return TacticResult(success=True, tactic=tactic, premises=[], new_goals=new)


def _program(*tactics: str) -> dx.DuckyProgram:
    return dx.DuckyProgram(program_id="t", bank="b", specialist="s", tactics=list(tactics), rationale="", score=1.0)


class DuckyFrontierTest(unittest.TestCase):
    """Dr. Ducky runs a program as Lean runs a tactic block: each step on the first open goal, every
    goal kept; a program closes only when no goal is left."""

    def test_a_closed_child_is_not_a_closed_program(self):
        lean = _StubLean({("p ∧ q", "constructor"): ["p", "q"], ("p", "ring"): []})
        run = dx.execute_ducky_program(lean, GoalRef("p ∧ q", 1, "⊢ p ∧ q"), _program("constructor"))
        self.assertFalse(run.closed)
        self.assertEqual([str.__str__(g) for g in run.goals_after], ["q"])

    def test_every_goal_closed_in_order_closes_the_program(self):
        lean = _StubLean({("p ∧ q", "constructor"): ["p", "q"], ("p", "ring"): [], ("q", "ring"): []})
        run = dx.execute_ducky_program(lean, GoalRef("p ∧ q", 1, "⊢ p ∧ q"), _program("constructor"))
        self.assertTrue(run.closed)
        self.assertEqual(run.tactics_applied, ["constructor", "ring", "ring"])

    def test_a_hypothesis_only_step_carries_on(self):
        lean = _StubLean({("a % 2 = 0", "obtain ⟨x, hx⟩ := h"): ["a % 2 = 0"]})
        lean.moves[("a % 2 = 0", "omega")] = []
        prog = _program("obtain ⟨x, hx⟩ := h", "omega")
        run = dx.execute_ducky_program(lean, GoalRef("a % 2 = 0", 1, "h : ∃ x, a = 2 * x\n⊢ a % 2 = 0"), prog)
        self.assertTrue(run.closed)
        self.assertEqual(run.tactics_applied, ["obtain ⟨x, hx⟩ := h", "omega"])

    def test_the_replay_is_what_was_applied(self):
        lean = _StubLean({("p ∧ q", "constructor"): ["p", "q"]})
        run = dx.execute_ducky_program(lean, GoalRef("p ∧ q", 1, "⊢ p ∧ q"), _program("constructor", "bogus"))
        self.assertEqual(run.script, "constructor")
        chain = dx._combined_program([run], frontier=run.goals_after)
        self.assertEqual((chain["script"], len(chain["goals_after"])), ("constructor", 2))


class _Goal:
    variables: list = []
    target = "p"


class _State:
    goals = [_Goal()]


class _Server:
    def __init__(self, error: str):
        self.error = error

    def goal_tactic(self, state, tactic, site):
        raise RuntimeError(self.error)


class HeartbeatTimeoutTest(unittest.TestCase):
    """A tactic that runs out of heartbeats has failed: the search goes on, and the server restarts
    before the next goal is created — never under a search's live goals. A dead server still ends it."""

    def kernel(self, error: str) -> tuple[LeanKernel, list]:
        k = LeanKernel(LeanConfig(backend="pantograph"))
        restarts: list = []
        k._server = _Server(error)
        k._ensure_server = lambda: k._server  # type: ignore[method-assign]
        k._restart_server = lambda: restarts.append(1)  # type: ignore[method-assign]
        return k, restarts

    def test_a_timeout_fails_the_tactic_and_defers_the_restart(self):
        k, restarts = self.kernel("{'desc': '(deterministic) timeout at `whnf`, maximum number of heartbeats (200000)'}")
        r = k.try_tactic(k._register(_State(), 0), "apply?")
        self.assertEqual((r.success, restarts, k._restart_pending), (False, [], True))
        k._restart_if_pending()
        self.assertEqual((restarts, k._restart_pending), ([1], False))

    def test_a_broken_pipe_still_ends_the_search(self):
        from src.lean_interface import ServerCrashError

        k, restarts = self.kernel("broken pipe")
        with self.assertRaises(ServerCrashError):
            k.try_tactic(k._register(_State(), 0), "simp")
        self.assertEqual(restarts, [1])


@unittest.skipUnless(_LIVE, "PyPantograph or the local Mathlib project is missing")
class LiveGoalIdentityTest(unittest.TestCase):
    """Against Lean itself (Mathlib, the search's Lean): the three moves measured to fail before."""

    @classmethod
    def setUpClass(cls):
        cls.k = LeanKernel(LeanConfig(backend="pantograph", timeout=300, project_root=str(MATHLIB_PROJECT),
                                      imports=["Mathlib"]))  # fmt: skip

    @classmethod
    def tearDownClass(cls):
        cls.k.close()

    def opened(self, statement: str, intro: str):
        r = self.k.try_tactic(self.k.goal_start(statement), intro)
        self.assertTrue(r.success)
        return r.new_goals[0]

    def test_have_dependency(self):
        g = self.opened("∀ (a b : ℝ), 0 < a → 0 < b → a * b + a > 0", "intro a b ha hb")
        self.assertFalse(self.k.try_tactic(g, "linarith").success)
        r = self.k.try_tactic(g, "have hab : 0 < a * b")
        claim, cont = r.new_goals
        self.assertEqual((claim, cont), ("0 < a * b", "a * b + a > 0"))
        self.assertNotEqual(cont, g)
        done = self.k.try_tactic(cont, "linarith")
        self.assertEqual((done.success, done.new_goals), (True, []))
        self.assertTrue(self.k.try_tactic(claim, "exact mul_pos ha hb").success)  # still its own obligation
        self.assertFalse(self.k.try_tactic(g, "linarith").success)  # MUST-NOT: the original keeps its context
        self.assertIn(("hab", "0 < a * b"), self.k.goal_variables(cont))
        self.assertNotIn("hab", [n for n, _ in self.k.goal_variables(g)])

    def test_context_progress(self):
        g = self.opened("∀ (a : ℕ), (∃ x, a = x * x) → ∃ y, a = y * y", "intro a h")
        r = self.k.try_tactic(g, "obtain ⟨x, hx⟩ := h")
        self.assertTrue(r.success)
        self.assertIn("hx", [n for n, _ in self.k.goal_variables(r.new_goals[0])])
        done = self.k.try_tactic(r.new_goals[0], "exact ⟨x, hx⟩")
        self.assertEqual((done.success, done.new_goals), (True, []))

    def test_a_goal_built_from_its_printed_statement_starts_a_search(self):
        # tier A2: nothing in the environment by this name, only the printed goal (miniF2F's rows)
        r = self.k.goal_via_file_context(theorem_full_name="wf_identity_missing_decl", file_path="", prefix_tactics=[],
                                         project_root=str(MATHLIB_PROJECT), fallback_goal_pp="⊢ ∀ (n : ℕ), n + 0 = n")  # fmt: skip
        self.assertEqual((r.success, r.tier_used), (True, "A2"))
        done = self.k.try_tactic(r.goal_state, "intro n; simp")
        self.assertEqual((done.success, done.new_goals), (True, []))

    def test_branch_isolation(self):
        g = self.opened("∀ (a : ℕ), (a = 1 ∨ a = 2) → a * a ≤ 4", "intro a h")
        r = self.k.try_tactic(g, "rcases h with h | h")
        self.assertEqual(len(r.new_goals), 2)
        self.assertNotEqual(*r.new_goals)
        for branch in r.new_goals:
            done = self.k.try_tactic(branch, "subst h; norm_num")
            self.assertEqual((done.success, done.new_goals), (True, []))


if __name__ == "__main__":
    unittest.main()
