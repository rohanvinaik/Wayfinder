"""Intent tests for the bounded numeric-witness search bank.

Pins the design contract from the module docstring and the Dr. Ducky
paper's witness bank (§7.2): generative bounded enumeration for
existential Nat goals, emitted additively through `infer_bank_priors`
and `build_proof_skeletons`, with hole candidates served by
`_hole_candidates` in the executor.

Detective note: `is_bounded_nat_goal`'s unicode branches (∃ / ℕ) are
pinned here by hand because the `--input` argv channel mangles the
codepoints; the ASCII branches are pinned by the synthesized suite.
"""

import unittest

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_bounded_search import (
    append_bounded_search_skeletons,
    bounded_nat_literal_candidates,
    bounded_nat_search_bank_prior,
    bounded_nat_tuple_candidates,
    is_bounded_nat_goal,
)
from src.dr_ducky_executor import _hole_candidates


class IsBoundedNatGoalUnicodeTest(unittest.TestCase):
    def test_exists_nat_unicode_type(self):
        self.assertTrue(is_bounded_nat_goal("∃ x : ℕ, x > 3"))

    def test_exists_nat_ascii_type(self):
        self.assertTrue(is_bounded_nat_goal("∃ x : Nat, x > 3"))

    def test_exists_int_is_not_bounded_nat(self):
        self.assertFalse(is_bounded_nat_goal("∃ x : Int, x > 3"))

    def test_nat_without_existential_is_rejected(self):
        self.assertFalse(is_bounded_nat_goal("⊢ n : Nat"))


class CandidateEnumerationTest(unittest.TestCase):
    def test_literal_candidates_are_progressive_and_budgeted(self):
        candidates = bounded_nat_literal_candidates(max_candidates=20)
        self.assertLessEqual(len(candidates), 20)
        self.assertEqual(candidates[0], "0")
        self.assertIn("50", candidates)  # the 92ba6a answer region
        for token in candidates:
            self.assertTrue(token.isdigit())

    def test_literal_budget_truncates(self):
        self.assertEqual(len(bounded_nat_literal_candidates(max_candidates=5)), 5)

    def test_tuple_candidates_only_arity_four(self):
        self.assertEqual(bounded_nat_tuple_candidates(arity=3), [])
        four = bounded_nat_tuple_candidates(arity=4)
        self.assertIn("⟨10, 5, 10, 5⟩", four)  # the 92ba6a witness
        for tup in four:
            self.assertTrue(tup.startswith("⟨") and tup.endswith("⟩"))


class _SpecStub:
    def __init__(self, goal_bucket="exists", goal_text="∃ x : ℕ, x > 3"):
        self.goal_bucket = goal_bucket
        self.goal_text = goal_text


class _PriorStub:
    def __init__(self, name, suppressed=False):
        self.name = name
        self.suppressed = suppressed


class BankPriorEmissionTest(unittest.TestCase):
    def test_prior_emitted_for_exists_bucket(self):
        prior = bounded_nat_search_bank_prior(_SpecStub())
        self.assertIsNotNone(prior)
        self.assertEqual(prior.name, "bounded_nat_search")
        self.assertFalse(prior.suppressed)
        # Design contract: stays below witness_constructor (0.87) so the
        # extractive bank keeps precedence.
        self.assertLess(prior.weight, 0.87)

    def test_no_prior_for_equality_bucket(self):
        spec = _SpecStub(goal_bucket="equality", goal_text="⊢ a = b")
        self.assertIsNone(bounded_nat_search_bank_prior(spec))

    def test_disabled_flag_suppresses_emission(self):
        self.assertIsNone(bounded_nat_search_bank_prior(_SpecStub(), enabled=False))


class SkeletonEmissionTest(unittest.TestCase):
    def test_skeletons_require_enabled_bank_prior(self):
        spec = _SpecStub()
        no_prior = append_bounded_search_skeletons(spec, [])
        self.assertEqual(no_prior, [])
        suppressed = append_bounded_search_skeletons(
            spec, [_PriorStub("bounded_nat_search", suppressed=True)]
        )
        self.assertEqual(suppressed, [])

    def test_skeletons_emitted_when_bank_enabled(self):
        spec = _SpecStub()
        skeletons = append_bounded_search_skeletons(
            spec, [_PriorStub("bounded_nat_search")]
        )
        ids = {s.skeleton_id for s in skeletons}
        self.assertEqual(
            ids,
            {
                "bounded_nat_search_use",
                "bounded_nat_search_use_decide",
                "bounded_tuple_search_refine_4",
            },
        )
        for skeleton in skeletons:
            self.assertEqual(skeleton.bank, "bounded_nat_search")
            self.assertEqual(skeleton.skeleton_kind, "witness_instantiation")


class CapsuleIntegrationTest(unittest.TestCase):
    """End-to-end through the real entry point: build_goal_capsule."""

    def test_exists_nat_capsule_carries_bounded_bank(self):
        row = {
            "theorem_id": "canary_92ba6a",
            "last_goal": "⊢ ∃ A B a b : ℕ, A * 10 + B * 5 + a + b = 155",
            "last_goal_bucket": "exists",
        }
        capsule = build_goal_capsule(row)
        prior_names = {p.name for p in capsule.bank_priors if not p.suppressed}
        self.assertIn("bounded_nat_search", prior_names)
        bounded = [s for s in capsule.proof_skeletons if s.bank == "bounded_nat_search"]
        self.assertEqual(len(bounded), 3)

    def test_equality_capsule_stays_clean(self):
        row = {
            "theorem_id": "eq_case",
            "last_goal": "⊢ a + b = b + a",
            "last_goal_bucket": "equality",
        }
        capsule = build_goal_capsule(row)
        self.assertNotIn(
            "bounded_nat_search", {p.name for p in capsule.bank_priors}
        )
        self.assertFalse(
            any(s.bank == "bounded_nat_search" for s in capsule.proof_skeletons)
        )

    def test_executor_serves_bounded_hole_candidates(self):
        row = {
            "theorem_id": "canary_92ba6a",
            "last_goal": "⊢ ∃ A B a b : ℕ, A * 10 + B * 5 + a + b = 155",
            "last_goal_bucket": "exists",
        }
        capsule = build_goal_capsule(row)
        bounded = [s for s in capsule.proof_skeletons if s.bank == "bounded_nat_search"]
        seen_kinds = set()
        for skeleton in bounded:
            for hole in skeleton.holes:
                candidates = _hole_candidates(hole, None)
                self.assertTrue(candidates)
                self.assertLessEqual(len(candidates), hole.max_candidates)
                seen_kinds.add(hole.hole_kind)
        self.assertEqual(seen_kinds, {"bounded_nat_literal", "bounded_nat_tuple"})


if __name__ == "__main__":
    unittest.main()
