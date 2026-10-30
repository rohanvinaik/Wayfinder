"""Filling one hole of the hand-off: where the fill goes, what it may contain, how a lifted
lemma's proof goes back in place, and how a model's answer is read."""

from __future__ import annotations

import unittest

import src.hole_fill as hf
from src.residual_handoff import HOLE_MARK

DECLARATION = f"""theorem t (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  -- {HOLE_MARK} 1: close this goal
  --   p q : Prop
  --   ⊢ p
  · sorry
  -- {HOLE_MARK} 2: close this goal
  --   ⊢ q
  · sorry
"""
FILE = "import Mathlib\nopen Finset\n\n" + DECLARATION
LINEAR = f"""theorem u (n : ℕ) : n + 0 = n := by
  -- {HOLE_MARK} 1: close this goal
  --   n : ℕ
  --   ⊢ n + 0 = n
  sorry
"""


class HoleSitesTest(unittest.TestCase):
    def test_every_open_hole_is_found_with_its_layout(self):
        sites = hf.hole_sites(DECLARATION)
        self.assertEqual(sorted(sites), [1, 2])
        self.assertTrue(sites[1].bullet)
        self.assertEqual(sites[1].indent, "  ")
        self.assertEqual(DECLARATION.splitlines()[sites[2].sorry].strip(), "· sorry")
        self.assertFalse(hf.hole_sites(LINEAR)[1].bullet)

    def test_a_filled_hole_is_no_longer_open(self):
        filled = hf.splice(DECLARATION, 1, "exact hp", note="claude")
        self.assertEqual(sorted(hf.hole_sites(filled)), [2])


class SpliceTest(unittest.TestCase):
    def test_a_bulleted_fill_continues_at_the_bullets_body(self):
        filled = hf.splice(DECLARATION, 1, "have h := hp\nexact h", note="a model")
        lines = filled.splitlines()
        at = lines.index("  · have h := hp")
        self.assertEqual(lines[at + 1], "    exact h")
        self.assertEqual(lines[at - 1], f"  -- {HOLE_MARK} 1: filled by a model")
        self.assertNotIn("--   ⊢ p", filled)  # the hole's goal comment goes with it
        self.assertIn("--   ⊢ q", filled)  # the other hole is untouched

    def test_a_linear_fill_takes_the_holes_indentation_and_keeps_its_own(self):
        filled = hf.splice(LINEAR, 1, "induction n with\n| zero => rfl\n| succ k ih => simp")
        self.assertIn("\n  induction n with\n  | zero => rfl\n  | succ k ih => simp\n", filled)

    def test_unknown_holes_and_empty_fills_are_refused(self):
        with self.assertRaises(KeyError):
            hf.splice(DECLARATION, 3, "trivial")
        with self.assertRaises(ValueError):
            hf.splice(DECLARATION, 1, "\n  \n")

    def test_helpers_go_directly_above_the_theorem(self):
        new_file, new_decl = hf.apply_fill(FILE, DECLARATION, 2, "exact helper hq", "lemma helper (h : q) : q := h")
        self.assertIn("open Finset\n\nlemma helper (h : q) : q := h\n\ntheorem t", new_file)
        self.assertTrue(new_file.endswith(new_decl))


class FindingsTest(unittest.TestCase):
    def test_placeholders_code_and_bypasses_are_refused(self):
        for bad in ("sorry", "exact sorryAx _", "native_decide", "run_tac pure ()", "#eval 1", "exact?",
                    "simp; #exit", "by_elab pure default", "set_option debug.skipKernelTC true in rfl"):  # fmt: skip
            self.assertTrue(hf.fill_findings(bad), bad)

    def test_a_line_that_would_end_the_proof_is_refused(self):
        self.assertTrue(hf.fill_findings("rfl\ntheorem other : True := trivial"))
        self.assertTrue(hf.fill_findings("· rfl\n· instance : Inhabited Nat := ⟨0⟩"))

    def test_ordinary_tactics_scoping_and_card_notation_pass(self):
        ok = "set_option maxHeartbeats 400000 in\nopen Nat in\nhave : #s ≤ #t := card_le_card h\nomega"
        self.assertEqual(hf.fill_findings(ok), [])

    def test_helpers_declare_lemmas_only(self):
        self.assertEqual(hf.helper_findings("set_option maxHeartbeats 400000 in\nlemma a : True := trivial"), [])
        self.assertTrue(hf.helper_findings("axiom cheat : False"))
        self.assertTrue(hf.helper_findings("instance : Inhabited Nat := ⟨0⟩"))
        self.assertTrue(hf.helper_findings("lemma a : True := by sorry"))

    def test_a_models_harmless_commands_do_not_sink_its_answer(self):
        self.assertEqual(hf.code_findings("import Mathlib\n#check Nat\ntheorem a : True := trivial"), [])
        self.assertTrue(hf.code_findings('#eval IO.println "x"'))
        self.assertTrue(hf.code_findings('macro "cheat" : tactic => `(tactic| sorry)'))


def _check(name: str, ty: str) -> dict:
    return {"severity": "information", "data": f"@{name} : {ty}"}


def _axioms(name: str, axioms: list[str]) -> dict:
    return {"severity": "information", "data": f"'{name}' depends on axioms: [{', '.join(axioms)}]"}


class PartialFindingsTest(unittest.TestCase):
    # the hand-off records the whole `#check @t` line as the statement's type
    def test_a_sound_partial_fill_passes_while_other_holes_are_open(self):
        msgs = [_check("t", "∀ (p q : Prop), p ∧ q"), _axioms("t", ["propext", "sorryAx"])]
        self.assertEqual(hf.partial_findings(msgs, "t", "@t : ∀ (p q : Prop), p ∧ q", holes_left=1), [])

    def test_sorry_is_only_tolerated_while_holes_remain(self):
        msgs = [_check("t", "P"), _axioms("t", ["sorryAx"])]
        self.assertTrue(hf.partial_findings(msgs, "t", "@t : P", holes_left=0))

    def test_errors_a_changed_statement_and_extra_axioms_are_caught(self):
        err = {"severity": "error", "pos": {"line": 3, "column": 2}, "data": "unsolved goals"}
        self.assertTrue(hf.partial_findings([err, _check("t", "P"), _axioms("t", [])], "t", "@t : P"))
        self.assertTrue(hf.partial_findings([_check("t", "Q"), _axioms("t", [])], "t", "@t : P"))
        self.assertTrue(hf.partial_findings([_check("t", "P"), _axioms("t", ["Lean.ofReduceBool"])], "t", "@t : P"))
        self.assertTrue(hf.partial_findings([_check("t", "P")], "t", "@t : P"))  # no axiom report


class ErrorReportTest(unittest.TestCase):
    def test_the_span_is_marked_and_the_message_follows(self):
        code = "a\nb\nc\nd\n  exact foo bar\ne"
        msg = {"severity": "error", "pos": {"line": 5, "column": 8}, "endPos": {"line": 5, "column": 11},
               "data": "unknown identifier 'foo'"}  # fmt: skip
        report = hf.error_report(code, [msg])
        self.assertIn("\nError 1:\n\nCorresponding Code:\n```lean4\na\nb\nc\nd\n  exact <error>foo</error> bar\ne\n", report)
        self.assertTrue(report.endswith("\nError Message: unknown identifier 'foo'\n"))

    def test_only_eight_errors_are_shown(self):
        msgs = [{"severity": "error", "pos": {"line": 1, "column": 0}, "endPos": None, "data": "x"}] * 10
        self.assertIn("[Omitted 2 more errors]", hf.error_report("theorem x : True := by\n  skip", msgs))


TOY_GOAL = "p q : Prop\nf : ℕ → ℕ\nleft✝ : p\nhq : q\nm✝ : ℕ\n⊢ f m✝ = f m✝"
TOY_LIFTED = "theorem toy_wf_hole_1 (p q : Prop) (f : ℕ → ℕ) (left : p) (hq : q) (m : ℕ) : f m = f m"
KAPPA_LIFTED = (
    "theorem kappa_additive_bound_wf_hole_1.{u_1} {α : Type u_1} [inst : DecidableEq α] {n : ℕ}\n"
    "  (localSuite : Fin n → Finset α) (κLocal : Fin n → ℕ) (hBound : ∀ (i : Fin n), #(localSuite i) ≤ κLocal i) :\n"
    "  #(univ.biUnion localSuite) ≤ univ.sum κLocal"
)


class LiftingTest(unittest.TestCase):
    def test_the_probe_asks_extract_goal_at_every_hole_plain_then_typed(self):
        probe = hf.lifting_probe(DECLARATION, "t")
        typed = (
            "set_option pp.numericTypes true in\n    set_option pp.funBinderTypes true in\n"
            "    set_option pp.coercions.types true in\n"
        )
        self.assertIn(f"  · extract_goal * using t_wf_hole_1\n    {typed}    extract_goal * using t_wf_hole_1_r\n    sorry", probe)
        self.assertIn("  · extract_goal * using t_wf_hole_2\n", probe)
        self.assertIn("  extract_goal * using u_wf_hole_1\n  set_option", hf.lifting_probe(LINEAR, "u"))

    def test_both_prints_are_read_back_and_the_typed_one_takes_the_holes_name(self):
        typed = "theorem t_wf_hole_1_r (h : ∑ k ∈ Finset.Icc (1 : ℕ) (35 : ℕ), f k = (0 : ℝ)) : True"
        msgs = [{"severity": "information", "data": "theorem t_wf_hole_1 (h : ∑ k ∈ Finset.Icc 1 35, f k = 0) : True := sorry"},
                {"severity": "information", "data": typed + " := sorry"}]  # fmt: skip
        plain, got = hf.lifted_variants(msgs, "t")[1]
        self.assertTrue(plain.startswith("theorem t_wf_hole_1 (h"))
        self.assertEqual(got, typed)
        self.assertTrue(hf.untyped_name(got, "t", 1).startswith("theorem t_wf_hole_1 (h : ∑ k ∈ Finset.Icc (1 : ℕ)"))
        self.assertEqual(hf.lifted_from_messages(msgs, "t"), {1: plain})

    def test_printed_statements_are_read_back_by_hole(self):
        msgs = [{"severity": "information", "data": KAPPA_LIFTED + " := sorry"},
                {"severity": "information", "data": "theorem other_wf_hole_1 : True := sorry"},
                {"severity": "warning", "data": "declaration uses `sorry`"}]  # fmt: skip
        self.assertEqual(hf.lifted_from_messages(msgs, "kappa_additive_bound"), {1: KAPPA_LIFTED})

    def test_a_statement_that_does_not_re_elaborate_is_named(self):
        block, ranges = hf.validation_block({1: "theorem a : True", 2: "theorem b : 1 = 1"})
        self.assertEqual(block.count(":= by\n  sorry"), 2)
        first = 10
        err = {"severity": "error", "pos": {"line": first + ranges[2][0]}, "data": "unknown constant"}
        self.assertEqual(hf.failing_ranges([err], ranges, first), {2: "unknown constant"})

    def test_binders_keep_order_brackets_and_anonymous_instances(self):
        self.assertEqual(
            hf.binders(KAPPA_LIFTED),
            [("α", "{"), ("inst", "["), ("n", "{"), ("localSuite", "("), ("κLocal", "("), ("hBound", "(")],
        )
        self.assertEqual(hf.binders("theorem x [DecidableEq α] (a b : ℕ) : a = a"),
                         [("", "["), ("a", "("), ("b", "(")])  # fmt: skip

    def test_context_names_read_the_printed_goal(self):
        goal = "case h\nα : Type u_1\ninst✝ : DecidableEq α\nhBound :\n  ∀ (i : Fin n), x i ≤ y i\na b : ℕ\n⊢ a = b"
        self.assertEqual(hf.context_names(goal), ["α", "inst✝", "hBound", "a", "b"])

    def test_rename_restores_the_lifted_names_in_place(self):
        lifted = hf.binders(TOY_LIFTED)
        self.assertEqual(hf.rename_line(hf.context_names(TOY_GOAL), lifted), "rename_i left m")
        self.assertEqual(hf.rename_line(["p", "q"], [("p", "("), ("q", "(")]), "")
        self.assertIsNone(hf.rename_line(["p"], lifted))  # the contexts do not line up
        self.assertIsNone(hf.rename_line(["x", "y"], [("p", "("), ("q", "(")]))
        self.assertEqual(hf.rename_line(["inst✝", "h✝"], [("", "["), ("h", "(")]), "rename_i _ h")

    def test_the_helper_is_applied_to_the_explicit_hypotheses(self):
        self.assertEqual(hf.helper_call("toy_wf_hole_1", hf.binders(TOY_LIFTED)),
                         "exact toy_wf_hole_1 p q f left hq m")  # fmt: skip
        self.assertEqual(hf.helper_call("k", hf.binders(KAPPA_LIFTED)), "exact k localSuite κLocal hBound")


ANSWER = """Plan: unfold, then bound each term.

```lean4
import Mathlib
open BigOperators Real

def obstructionClass (n : ℕ) : ℕ := n

lemma step (n : ℕ) : n ≤ n + 1 := Nat.le_succ n

theorem target_thm (n : ℕ) : n ≤ n + 1 := by
  exact step n
```"""


class ModernizeTest(unittest.TestCase):
    def test_old_big_operator_binders_are_respelled(self):
        code = "calc a ≤ ∑ i in Finset.univ, f i := h\n  _ = ∏ k in range n, g k := rfl"
        new, notes = hf.modernize(code)
        self.assertEqual(new, "calc a ≤ ∑ i ∈ Finset.univ, f i := h\n  _ = ∏ k ∈ range n, g k := rfl")
        self.assertEqual(len(notes), 1)

    def test_current_syntax_is_left_alone(self):
        code = "have : ∑ i ∈ s, f i ≤ ∑ i, g i := by\n  intro x hx\n  exact le_of_lt (lt_of_mem_in x)"
        self.assertEqual(hf.modernize(code), (code, []))

    def test_norm_num_at_a_hypothesis_and_the_goal_tries_the_hypotheses_first(self):
        # in this Mathlib, `norm_num at h ⊢` whose `h` closes the goal then fails on `⊢` (No goals)
        new, notes = hf.modernize("  interval_cases m <;> norm_num at h₅ ⊢")
        self.assertEqual(new, "  interval_cases m <;> (first | (norm_num at h₅; done) | norm_num at h₅ ⊢)")
        self.assertEqual(len(notes), 1)
        new, _ = hf.modernize("norm_num [Nat.dvd_iff_mod_eq_zero] at h₁ h₂ ⊢ <;> omega")
        self.assertEqual(new, "(first | (norm_num [Nat.dvd_iff_mod_eq_zero] at h₁ h₂; done) | "
                              "norm_num [Nat.dvd_iff_mod_eq_zero] at h₁ h₂ ⊢) <;> omega")

    def test_other_norm_num_locations_and_a_rewritten_call_are_left_alone(self):
        for code in ("norm_num at ⊢", "norm_num at h₅", "norm_num at *", "simp at h ⊢", "norm_num1 at h ⊢"):
            self.assertEqual(hf.modernize(code), (code, []), code)
        once, _ = hf.modernize("norm_num at h ⊢")
        self.assertEqual(hf.modernize(once), (once, []))


class AnswerTest(unittest.TestCase):
    def test_the_last_code_block_is_the_answer(self):
        raw = "draft:\n```lean4\ntheorem a : True := sorry\n```\nfinal:\n```lean4\ntheorem a : True := trivial\n```"
        self.assertEqual(hf.extract_code(raw), "theorem a : True := trivial")
        self.assertEqual(hf.extract_code("```lean\ntheorem a : True := trivial\n```"), "theorem a : True := trivial\n")
        self.assertIsNone(hf.extract_code("no code here"))

    def test_the_proof_its_new_lemmas_and_opens_are_kept(self):
        cand = hf.candidate(hf.extract_code(ANSWER), "target_thm", {"obstructionClass", "target_thm"})
        assert cand is not None
        self.assertEqual(cand.tactic_lines(), ["exact step n"])
        self.assertEqual(len(cand.helpers), 1)  # the restated `def` is ours, so it is dropped
        self.assertTrue(cand.helpers[0].startswith("lemma step"))
        self.assertEqual(cand.opens, ["BigOperators", "Real"])
        self.assertEqual(cand.scoped("lemma x : True := trivial"), "open BigOperators Real in\nlemma x : True := trivial")
        self.assertFalse(cand.renamed)

    def test_a_renamed_theorem_and_a_term_proof_still_give_a_fill(self):
        cand = hf.candidate("theorem my_name (n : ℕ) : n = n :=\n  rfl", "target_thm", set())
        assert cand is not None
        self.assertTrue(cand.renamed)
        self.assertEqual(cand.tactic_lines(), ["exact", "  rfl"])

    def test_a_proof_on_the_by_line_is_kept(self):
        cand = hf.candidate("theorem target_thm : True := by trivial", "target_thm", set())
        assert cand is not None
        self.assertEqual(cand.tactic_lines(), ["trivial"])
        self.assertIsNone(hf.candidate("lemma helper : True := trivial\n#check helper", "t", {"helper"}))


if __name__ == "__main__":
    unittest.main()
