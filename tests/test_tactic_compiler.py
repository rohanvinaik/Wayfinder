"""Intent tests for the pure tactic compiler (v1).

The lane-level characterization (tests/test_cosine_family_templates.py)
pins the same grammar through _try_ranked_family; these tests pin the
compiler's own contract directly, including the source-agnostic
compile_from_predictions entry point the deterministic lane uses.
"""

import unittest

from src.tactic_compiler import (
    compile_apply,
    compile_exact,
    compile_from_predictions,
    compile_rw,
    compile_rw_chain,
    compile_simp,
    compile_simp_set,
    compile_simpa,
    compile_tactic,
)


class CompileRwTest(unittest.TestCase):
    def test_backward_name_orders_backward_first(self):
        texts = [c.text for c in compile_rw(["Nat.add_comm"], "⊢ x + y = y + x")]
        self.assertEqual(texts, ["rw [← Nat.add_comm]", "rw [Nat.add_comm]"])

    def test_goal_visible_head_orders_forward_first(self):
        texts = [c.text for c in compile_rw(["Nat.succ_eq"], "Nat.succ n = n + 1")]
        self.assertEqual(texts, ["rw [Nat.succ_eq]", "rw [← Nat.succ_eq]"])

    def test_max_premises_truncates(self):
        cands = compile_rw(["a_comm", "b_comm", "c_comm"], "", max_premises=2)
        self.assertEqual(len(cands), 4)


class CompileExactApplyTest(unittest.TestCase):
    def test_exact_never_emits_wildcards(self):
        # `exact p _` is the documented premise_direct failure mode.
        for cand in compile_exact(["P1", "P2"]):
            self.assertNotIn("_", cand.text.replace(cand.premises_used[0], ""))

    def test_apply_ladder_order(self):
        texts = [c.text for c in compile_apply(["P1"])]
        self.assertEqual(
            texts,
            ["exact P1", "refine P1 ?_ ?_", "refine P1 ?_ ?_ ?_", "apply P1"],
        )


class CompileSimpTest(unittest.TestCase):
    def test_bare_first_single_hint_never_multihint(self):
        texts = [c.text for c in compile_simp(["P1", "P2", "P3"])]
        self.assertEqual(texts, ["simp", "simp [P1]"])

    def test_no_premises_still_offers_bare_simp(self):
        self.assertEqual([c.text for c in compile_simp([])], ["simp"])


class CompileTacticTest(unittest.TestCase):
    def test_unknown_family_returns_empty(self):
        self.assertEqual(compile_tactic("induction", ["P1"]), [])

    def test_confidence_is_stamped_on_all_candidates(self):
        for cand in compile_tactic("apply", ["P1"], confidence=0.7):
            self.assertEqual(cand.confidence, 0.7)


class CompileV2FormsTest(unittest.TestCase):
    """The composite forms the autopsy demanded (EXP-RF-005b §9):
    33/48 short-proof failures had in-beam premises whose true form
    (`rw [A, B]`, `simp [A, B]`, `simpa`) v1 could not utter."""

    def test_rw_chain_emits_ordered_pairs_rank_first(self):
        texts = [c.text for c in compile_rw_chain(["A", "B", "C"])]
        self.assertEqual(texts[0], "rw [A, B]")
        self.assertIn("rw [B, A]", texts)  # order matters — both emitted
        self.assertEqual(len(texts), 6)  # 3P2 ordered pairs, no self-pairs
        self.assertNotIn("rw [A, A]", texts)

    def test_rw_chain_needs_two_premises(self):
        self.assertEqual(compile_rw_chain(["A"]), [])
        self.assertEqual(compile_rw_chain([]), [])

    def test_simp_set_two_then_three_hints(self):
        texts = [c.text for c in compile_simp_set(["A", "B", "C", "D"])]
        self.assertEqual(texts, ["simp [A, B]", "simp [A, B, C]"])

    def test_simp_set_needs_two_premises(self):
        # The single-hint form belongs to v1's compile_simp — no duplication.
        self.assertEqual(compile_simp_set(["A"]), [])

    def test_simpa_bare_then_using_before_hint(self):
        texts = [c.text for c in compile_simpa(["A"])]
        self.assertEqual(texts, ["simpa", "simpa using A", "simpa [A]"])

    def test_simpa_without_premises_still_offers_bare(self):
        self.assertEqual([c.text for c in compile_simpa([])], ["simpa"])

    def test_dispatch_through_compile_tactic(self):
        for family, needle in (
            ("rw_chain", "rw [A, B]"),
            ("simp_set", "simp [A, B]"),
            ("simpa", "simpa using A"),
        ):
            texts = [c.text for c in compile_tactic(family, ["A", "B"])]
            self.assertIn(needle, texts, family)

    def test_families_are_their_own_censor_keys(self):
        # A fence mined on single-rw failures asserts nothing about chains.
        for cand in compile_rw_chain(["A", "B"]):
            self.assertEqual(cand.family, "rw_chain")
        for cand in compile_simpa(["A"]):
            self.assertEqual(cand.family, "simpa")


class CompileFromPredictionsTest(unittest.TestCase):
    def test_families_in_ranking_order_with_caps(self):
        cands = compile_from_predictions(
            [("exact", 0.9), ("simp", 0.5), ("rw", 0.3), ("apply", 0.1)],
            ["P1", "P2"],
            goal_target="",
            max_families=3,
            max_candidates_per_family=2,
        )
        texts = [c.text for c in cands]
        # Empty goal_target -> premise head not visible -> backward-first rw.
        self.assertEqual(
            texts, ["exact P1", "exact P2", "simp", "simp [P1]", "rw [← P1]", "rw [P1]"]
        )
        self.assertEqual([c.confidence for c in cands[:2]], [0.9, 0.9])

    def test_source_agnostic_uniform_prior_works(self):
        # The deterministic lane's degenerate case: uniform confidences.
        cands = compile_from_predictions(
            [("exact", 0.0), ("apply", 0.0)], ["P1"], max_families=2
        )
        self.assertTrue(cands)


if __name__ == "__main__":
    unittest.main()
