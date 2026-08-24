"""Intent tests for the structural-reads estimator (THE_REFOUNDING §4).

The estimator is pinned harder than any controller above it (Peitho DESIGN
§10): a confidently-wrong read propagates into every fence and coordinate
downstream. Detective synth suites pin the pure algebra; these tests pin the
INTENT — the bank-table semantics, the mined-zero discipline, the abstention
zeros, and the discrimination-guarantee machinery.
"""

import unittest

from src.structural_reads import (
    AMBIGUOUS,
    BANKS,
    CONSTRUCTIVE,
    DESTRUCTIVE,
    SILENT,
    ContextNorms,
    count_hypotheses,
    discriminates,
    goal_target,
    hamming,
    interference,
    mine_context_zero,
    read_context,
    read_goal,
    ternary,
)


class TernaryBoundaryTest(unittest.TestCase):
    def test_tolerance_band_is_inclusive(self):
        # Detective's suggested boundary pin: dev == tol sits INSIDE the band.
        self.assertEqual(ternary(0.0, 0.0), 0)
        self.assertEqual(ternary(1.0, 1.0), 0)
        self.assertEqual(ternary(-1.0, 1.0), 0)

    def test_poles(self):
        self.assertEqual(ternary(1.5, 1.0), 1)
        self.assertEqual(ternary(-1.5, 1.0), -1)


class MinedZeroTest(unittest.TestCase):
    def test_median_of_active_excludes_nonparticipants(self):
        # Zeros are the -1 pole by rule, so they must not drag the norm.
        self.assertEqual(mine_context_zero([0, 0, 0, 2, 4, 6]), 4.0)

    def test_even_active_count_averages_middle_pair(self):
        self.assertEqual(mine_context_zero([1, 3, 5, 7]), 4.0)

    def test_nothing_active_falls_back_to_documented_zero(self):
        self.assertEqual(mine_context_zero([0, 0]), ContextNorms().zero)


class ContextReadTest(unittest.TestCase):
    def test_bare_goal_is_the_pole_by_rule_not_deviation(self):
        read = read_context(0)
        self.assertEqual(read.sign, -1)
        self.assertIn("bare-goal pole", read.evidence)

    def test_moderate_context_reads_zero(self):
        self.assertEqual(read_context(4).sign, 0)

    def test_rich_context_reads_positive_against_norm(self):
        read = read_context(10)
        self.assertEqual(read.sign, 1)
        self.assertEqual(read.zero_state, ContextNorms().zero)
        self.assertEqual(read.depth, 6.0)  # magnitude retained in the row

    def test_remined_norms_shift_the_band(self):
        self.assertEqual(read_context(10, ContextNorms(zero=12.0, tol=3.0)).sign, 0)


class GoalReadsTest(unittest.TestCase):
    QUANTIFIED = "⊢ ∀ x : Nat, x + 0 = x"
    ABSTRACT = "F : CategoryTheory.Functor C D\n⊢ CategoryTheory.IsIso (F.map f)"
    BARE_ARITH = "x : ℕ\nh : x + 2 = 5\n⊢ x = 3"

    def test_quantified_goal_reads_structure_positive(self):
        reads = read_goal(self.QUANTIFIED)
        self.assertEqual(reads.directions()["structure"], 1)

    def test_abstract_goal_reads_automation_manual(self):
        reads = read_goal(self.ABSTRACT)
        self.assertEqual(reads.directions()["automation"], 1)
        self.assertIn("CategoryTheory", reads.evidence()["automation"])

    def test_depth_always_abstains_on_open_goals(self):
        for goal in (self.QUANTIFIED, self.ABSTRACT, self.BARE_ARITH):
            reads = read_goal(goal)
            self.assertEqual(reads.directions()["depth"], 0)
            self.assertIn("no opinion", reads.evidence()["depth"])

    def test_signature_is_bank_ordered_and_total(self):
        sig = read_goal(self.BARE_ARITH).signature()
        self.assertEqual(len(sig), len(BANKS))
        self.assertTrue(all(s in (-1, 0, 1) for s in sig))

    def test_every_read_carries_evidence(self):
        for text in read_goal(self.ABSTRACT).evidence().values():
            self.assertTrue(text)

    def test_discrimination_guarantee_machinery(self):
        sig_a = read_goal(self.QUANTIFIED).signature()
        sig_b = read_goal(self.ABSTRACT).signature()
        self.assertTrue(discriminates(sig_a, sig_b))
        self.assertEqual(hamming(sig_a, sig_a), 0)


class HypothesisParsingTest(unittest.TestCase):
    def test_counts_stop_at_turnstile(self):
        goal = "a : Nat\nb : Nat\n⊢ a = b"
        self.assertEqual(count_hypotheses(goal), 2)
        self.assertEqual(goal_target(goal), "a = b")

    def test_bare_goal(self):
        self.assertEqual(count_hypotheses("⊢ True"), 0)


class InterferenceVerdictTest(unittest.TestCase):
    def test_one_confident_opposition_vetoes_weak_supports(self):
        # Below-floor support loses to a floor-reaching opposition outright;
        # only when BOTH floors fire does the conflict escalate to AMBIGUOUS.
        self.assertEqual(interference(supports=1, opposes=1), DESTRUCTIVE)
        self.assertEqual(interference(supports=3, opposes=1), AMBIGUOUS)

    def test_clean_agreement_emits(self):
        self.assertEqual(interference(supports=2, opposes=0), CONSTRUCTIVE)

    def test_clean_opposition_eliminates(self):
        self.assertEqual(interference(supports=0, opposes=1), DESTRUCTIVE)

    def test_all_orthogonal_is_silent_not_zero_score(self):
        self.assertEqual(interference(supports=0, opposes=0), SILENT)


if __name__ == "__main__":
    unittest.main()
