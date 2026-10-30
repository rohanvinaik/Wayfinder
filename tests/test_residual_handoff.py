"""The residual hand-off: the search's stopping state rendered as a Lean proof with holes.

Pure-function pins (no Lean). The compile loop lives in wayfinder_mcp; here the
tree, the rendering, and the reading-back of Lean messages are fixed.
"""

from __future__ import annotations

import unittest

import src.residual_handoff as rh


def rec(goal, idx, tactic, new_goals, open_before, lean=None, prepend=False, provenance="learned"):
    return {
        "goal": goal,
        "goal_idx": idx,
        "open_before": open_before,
        "tactic": tactic,
        "lean_tactic": tactic if lean is None else lean,
        "new_goals": new_goals,
        "provenance": provenance,
        "prepend": prepend,
    }


# The aime_1983_p2 run: a single chain, one goal left open.
CHAIN = [
    rec("G0", 0, "norm_num", ["G1"], ["G0"]),
    rec("G1", 0, "aesop", ["G2"], ["G1"]),
    rec("G2", 0, "ring", ["G3"], ["G2"]),
]

# constructor opens two goals; the first is closed, the second left open.
BRANCH = [
    rec("p ∧ q", 0, "constructor", ["p", "q"], ["p ∧ q"]),
    rec("p", 0, "exact hp", [], ["p", "q"]),
]


class TreeTest(unittest.TestCase):
    def test_chain(self):
        tree = rh.build_tree(["G0"], CHAIN)
        self.assertEqual([tree.nodes[i].goal for i in tree.open_ids], ["G3"])
        body = rh.render(tree).body
        self.assertEqual([ln.strip() for ln in body if not ln.strip().startswith("--")], ["norm_num", "aesop", "ring", "sorry"])

    def test_branch_uses_bullets(self):
        tree = rh.build_tree(["p ∧ q"], BRANCH)
        body = rh.render(tree, hole_goals={2: "hq : q\n⊢ q"}).body
        self.assertEqual(body[0], "  constructor")
        self.assertEqual(body[1], "  · exact hp")
        self.assertIn(f"-- {rh.HOLE_MARK} 1: close this goal", body[2])
        self.assertTrue(any(ln.strip() == "--   ⊢ q" for ln in body))
        self.assertEqual(body[-1], "  · sorry")

    def test_a_hole_comment_sits_with_its_own_bullet(self):
        log = [
            rec("A ∧ B", 0, "constructor", ["A", "B"], ["A ∧ B"]),
            rec("A", 0, "simp", [], ["A", "B"]),
        ]
        body = rh.render(rh.build_tree(["A ∧ B"], log)).body
        self.assertEqual(body[1], "  · simp")
        self.assertEqual(body[2], f"  -- {rh.HOLE_MARK} 1: close this goal")
        self.assertEqual(body[-1], "  · sorry")

    def test_rotated_open_list_is_resynced(self):
        log = [
            rec("p ∧ q", 0, "constructor", ["p", "q"], ["p ∧ q"]),
            rec("p", 1, "exact hp", [], ["q", "p"]),  # the search had rotated p to the end
        ]
        tree = rh.build_tree(["p ∧ q"], log)
        self.assertEqual([tree.nodes[i].goal for i in tree.open_ids], ["q"])

    def test_prepended_goals_come_first(self):
        log = [
            rec("A ∧ B", 0, "constructor", ["A", "B"], ["A ∧ B"]),
            rec("A", 0, "apply f", ["A1", "A2"], ["A", "B"], prepend=True),
        ]
        tree = rh.build_tree(["A ∧ B"], log)
        self.assertEqual([tree.nodes[i].goal for i in tree.open_ids], ["A1", "A2", "B"])

    def test_divergence_is_an_error(self):
        with self.assertRaises(rh.ReplayError):
            rh.build_tree(["G0"], [rec("G0", 0, "simp", ["G1"], ["G0"]), rec("Gx", 0, "ring", [], ["Gx"])])

    def test_unreplayable_closure_is_a_hole(self):
        log = [rec("G0", 0, "rw_seq(2)", ["G1"], ["G0"], lean=None)]
        log[0]["lean_tactic"] = None
        tree = rh.build_tree(["G0"], log)
        self.assertEqual(rh.render(tree).holes, [0])

    def test_forced_hole_drops_its_subtree_only(self):
        tree = rh.build_tree(["p ∧ q"], BRANCH)
        out = rh.render(tree, forced_holes={1})
        self.assertEqual(out.holes, [1, 2])
        self.assertEqual(out.kept, [0])


class CommitmentTest(unittest.TestCase):
    """A residual keeps validated progress only: a step that chose (a disjunct, a witness) or gave
    something up stands only if everything under it closed. imo_1997_p5's search kept `left`, so
    its hole was `x = 1 ∧ y = 1`, false for (16, 2): no prover could ever close it."""

    # imo_1997_p5's shape: intros, then `left` commits to the first solution and its hole stays open
    LEFT_OPEN = [
        rec("⊢ P → A ∨ B", 0, "intros", ["h : P\n⊢ A ∨ B"], ["⊢ P → A ∨ B"]),
        rec("h : P\n⊢ A ∨ B", 0, "left", ["h : P\n⊢ A"], ["h : P\n⊢ A ∨ B"]),
        rec("h : P\n⊢ A", 0, "norm_num", ["h : P\n⊢ A'"], ["h : P\n⊢ A"]),
    ]

    def test_a_choice_with_a_hole_under_it_is_unvalidated(self):
        tree = rh.build_tree(["⊢ P → A ∨ B"], self.LEFT_OPEN)
        self.assertEqual(rh.unvalidated_commitments(tree, tree.open_ids, set()), {1})
        self.assertEqual(rh.attempted(tree, 1), ["left", "norm_num"])

    def test_a_choice_whose_branch_closed_stands(self):
        log = self.LEFT_OPEN[:2] + [rec("h : P\n⊢ A", 0, "exact h.1", [], ["h : P\n⊢ A"])]
        tree = rh.build_tree(["⊢ P → A ∨ B"], log)
        self.assertEqual(rh.unvalidated_commitments(tree, tree.open_ids, set()), set())

    def test_splitting_steps_and_rewrites_never_commit(self):
        tree = rh.build_tree(["p ∧ q"], BRANCH)  # constructor on ∧, one side left open
        self.assertEqual(rh.unvalidated_commitments(tree, tree.open_ids, set()), set())
        for tactic in ("intros", "simp_all", "norm_num at h ⊢", "field_simp", "rcases h with ⟨x, hx⟩", "aesop"):
            self.assertFalse(rh.commits(rh.Node(goal="⊢ G", tactic=tactic)), tactic)

    def test_witnesses_and_compound_steps_commit(self):
        for tactic in ("use 1; aesop; congr; ring", "exists 0", "right", "congr 1", "apply foo", "exfalso"):
            self.assertTrue(rh.commits(rh.Node(goal="⊢ G", tactic=tactic)), tactic)
        self.assertTrue(rh.commits(rh.Node(goal="⊢ ∃ b > 0, f b = 0", tactic="constructor")))
        self.assertFalse(rh.commits(rh.Node(goal="⊢ A ↔ B", tactic="constructor")))

    def test_an_outer_withdrawal_covers_the_ones_under_it(self):
        log = [
            rec("⊢ ∃ b, A b ∨ B b", 0, "use 1", ["⊢ A 1 ∨ B 1"], ["⊢ ∃ b, A b ∨ B b"]),
            rec("⊢ A 1 ∨ B 1", 0, "left", ["⊢ A 1"], ["⊢ A 1 ∨ B 1"]),
        ]
        tree = rh.build_tree(["⊢ ∃ b, A b ∨ B b"], log)
        self.assertEqual(rh.unvalidated_commitments(tree, tree.open_ids, set()), {0})

    def test_the_hole_carries_what_was_tried_and_none_of_it_stands(self):
        tree = rh.build_tree(["⊢ P → A ∨ B"], self.LEFT_OPEN)
        body = rh.render(tree, forced_holes={1}, tried={1: ["left", "norm_num"]}).body
        steps = [ln.strip() for ln in body if not ln.strip().startswith("--")]
        self.assertEqual(steps, ["intros", "sorry"])  # `left` and what followed are not in the residual
        self.assertIn(f"-- {rh.TRIED_MARK}: left ; norm_num", "\n".join(ln.strip() for ln in body))


class DeclarationTest(unittest.TestCase):
    def test_binder_names(self):
        head = (
            "theorem aime (x p : ℝ) (f : ℝ → ℝ) (h₀ : 0 < p ∧ p < 15) {α : Type*} [inst : Fintype α] "
            "[DecidableEq α] ⦃n : ℕ⦄ : 15 ≤ f x := by"
        )
        self.assertEqual(rh.binder_names(head), ["x", "p", "f", "h₀", "α", "inst", "n"])

    def test_split_declaration(self):
        head, body = rh.split_declaration("theorem t (n : ℕ) : n = n := by\n  sorry")
        self.assertEqual(head, "theorem t (n : ℕ) : n = n := by")
        self.assertEqual(body.strip(), "sorry")
        head, _ = rh.split_declaration("theorem t : True := sorry")
        self.assertEqual(head, "theorem t : True := by")
        with self.assertRaises(rh.ReplayError):
            rh.split_declaration("theorem t : True := trivial")

    def test_revert_line_comes_first(self):
        tree = rh.build_tree(["G0"], CHAIN)
        out = rh.render(tree, revert=["x", "p"])
        self.assertEqual(out.body[0], "  revert x p")
        self.assertEqual(out.line_node[0], -1)


class ReadBackTest(unittest.TestCase):
    def setUp(self):
        self.tree = rh.build_tree(["p ∧ q"], BRANCH)
        self.probe = rh.render(self.tree, probe=True)
        self.first = 10
        self.trace_line = self.first + next(o for o, ln in enumerate(self.probe.body) if ln.strip() == "trace_state")

    def test_hole_goals_and_faithfulness(self):
        msgs = [{"severity": "information", "kind": "trace", "pos": {"line": self.trace_line}, "data": "case right\nhq : q\n⊢ q"}]
        goals = rh.holes_from_messages(msgs, self.probe, self.first)
        self.assertEqual(goals, {2: "case right\nhq : q\n⊢ q"})
        ok, rows = rh.faithful(self.tree, goals, self.probe.holes)
        self.assertTrue(ok, rows)
        ok, _ = rh.faithful(self.tree, {2: "⊢ r"}, self.probe.holes)
        self.assertFalse(ok)

    def test_search_side_support_names_compare_equal(self):
        self.assertEqual(
            rh.target_of("_wf_support_0_obstructionClass γ interfaces = 0"),
            rh.target_of("case mpr\nγ : Fin n → Fin n → ℕ\n⊢ obstructionClass γ interfaces = 0"),
        )

    def test_repair_an_error_on_a_tactic(self):
        line = self.first + next(o for o, ln in enumerate(self.probe.body) if "exact hp" in ln)
        targets, notes = rh.repair_targets(self.tree, [{"severity": "error", "pos": {"line": line}, "data": "type mismatch"}], self.probe, self.first)
        self.assertEqual(targets, {1})
        self.assertTrue(notes)

    def test_a_data_hole_gives_back_its_parent(self):
        line = self.first + next(o for o, ln in enumerate(self.probe.body) if "run_tac" in ln)
        msg = {"severity": "error", "pos": {"line": line}, "data": rh.DATA_HOLE_ERROR}
        self.assertEqual(rh.repair_targets(self.tree, [msg], self.probe, self.first)[0], {0})

    def test_a_failed_revert_abandons_the_form(self):
        out = rh.render(self.tree, revert=["x"], probe=True)
        msg = {"severity": "error", "pos": {"line": self.first}, "data": "unknown identifier 'x'"}
        targets, notes = rh.repair_targets(self.tree, [msg], out, self.first)
        self.assertEqual(targets, set())
        self.assertIn("revert line failed", notes[0])


class FinishingPassTest(unittest.TestCase):
    DECL = "theorem t (p q : Prop) (hp : p) (hq : q) : p ∧ q := by\n  constructor\n  · sorry\n  · sorry\n"

    def test_closers_unfold_only_what_is_named(self):
        with_defs = rh.closers(["obstructionClass"])
        self.assertEqual(with_defs[0], "simp_all [obstructionClass]")
        self.assertIn("(try simp only [obstructionClass] at *); aesop", with_defs)
        bare = rh.closers([])
        self.assertEqual(bare[0], "simp_all")
        self.assertIn("aesop", bare)
        self.assertFalse(any("[]" in c or "{d}" in c for c in bare))

    def test_division_goals_get_the_division_facts_and_grind_comes_last(self):
        out = rh.closers([], "r k : ℕ\nhk : 0 < k\n⊢ r ≤ k + k * (r / k)")
        self.assertIn("have := Nat.div_add_mod r k; have := Nat.mod_lt r (by omega); omega", out)
        self.assertEqual(out[-1], "grind")
        self.assertFalse(any("div_add_mod" in c for c in rh.closers([], "⊢ a ≤ b")))

    def test_defs_mentioned(self):
        goal = "a : obstructionClass γ s = 0\n⊢ γ x y = 0"
        self.assertEqual(rh.defs_mentioned(goal, ["sheafCondition", "obstructionClass"]), ["obstructionClass"])

    def test_triage_block_isolates_each_hole(self):
        probe = rh.triage_block(self.DECL, "t", [["simp_all"], ["omega", "tauto"]])
        self.assertEqual([(k, m) for k, m, _, _ in probe.ranges], [(0, "close"), (0, "refute"), (1, "close"), (1, "refute")])
        for name in ("t_wf_close_1", "t_wf_refute_1", "t_wf_close_2", "t_wf_refute_2"):
            self.assertIn(f"theorem {name} ", probe.text)
        self.assertNotIn("theorem t ", probe.text)
        lines = probe.text.splitlines()
        for _, _, lo, hi in probe.ranges:
            self.assertTrue(lines[lo].startswith("set_option maxHeartbeats"))
            self.assertTrue(lines[hi].strip())
        self.assertEqual(probe.text.count("plausible"), 2)

    def test_triage_outcomes(self):
        probe = rh.triage_block(self.DECL, "t", [["simp_all"], ["omega", "tauto"]])
        first = 10
        close1 = probe.ranges[0][2]
        refute2 = probe.ranges[3][2]
        msgs = [
            {"severity": "information", "pos": {"line": first + close1 + 3}, "data": "wf_local:0"},
            {"severity": "error", "pos": {"line": first + refute2 + 3}, "data": "Found a counter-example!\np := False"},
        ]
        out = rh.triage_outcomes(msgs, probe, first)
        self.assertEqual(out[0]["closed_by"], "simp_all")
        self.assertIsNone(out[0]["refuted"])
        self.assertIsNone(out[1]["closed_by"])
        self.assertIn("p := False", out[1]["refuted"])

    def test_a_filled_hole_is_rendered_as_its_closer(self):
        tree = rh.build_tree(["p ∧ q"], BRANCH)
        out = rh.render(tree, fills={2: "exact hq"})
        self.assertEqual(out.holes, [])
        self.assertEqual(out.filled, [2])
        self.assertIn(f"  -- {rh.LOCAL_MARK}", out.body)
        self.assertEqual(out.body[-1], "  · exact hq")


if __name__ == "__main__":
    unittest.main()
