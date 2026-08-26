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
    READ_DIMENSIONS,
    SILENT,
    TAIL_BOUNDED_WITNESS,
    TAIL_COMPACT_ARITH,
    TAIL_MINED_CELL,
    TAIL_NONE,
    ContextNorms,
    SurfaceNorms,
    atomic_sweep_tactics,
    count_hypotheses,
    deterministic_lane_order,
    discriminates,
    goal_tail_regime,
    goal_target,
    hamming,
    interference,
    mine_context_zero,
    mine_median,
    mine_top_tertile,
    read_casts,
    read_context,
    read_coupling,
    read_goal,
    read_numerals,
    read_size,
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

    def test_signature_spans_all_read_dimensions(self):
        sig = read_goal(self.BARE_ARITH).signature()
        self.assertEqual(len(sig), len(READ_DIMENSIONS))
        self.assertTrue(all(s in (-1, 0, 1) for s in sig))

    def test_directions_stay_six_bank_for_query_compat(self):
        # Surface dims have no entity-side positions; leaking them into a
        # StructuredQuery would silently score nothing.
        self.assertEqual(set(read_goal(self.BARE_ARITH).directions()), set(BANKS))

    def test_every_read_carries_evidence(self):
        for text in read_goal(self.ABSTRACT).evidence().values():
            self.assertTrue(text)

    def test_discrimination_guarantee_machinery(self):
        sig_a = read_goal(self.QUANTIFIED).signature()
        sig_b = read_goal(self.ABSTRACT).signature()
        self.assertTrue(discriminates(sig_a, sig_b))
        self.assertEqual(hamming(sig_a, sig_a), 0)


class SurfaceReadsTest(unittest.TestCase):
    """The Wave-1.5 dimensions — each off its mined zero (nav_eval, 2026-08-24)."""

    def test_size_bands_align_with_mined_tertiles(self):
        self.assertEqual(read_size(5).sign, -1)  # ≤6 compact
        self.assertEqual(read_size(11).sign, 0)  # the mined median
        self.assertEqual(read_size(16).sign, 1)  # ≥16 large
        self.assertEqual(read_size(30).zero_state, 11.0)

    def test_casts_one_sided_off_zero_mode(self):
        self.assertEqual(read_casts(0).sign, 0)
        self.assertEqual(read_casts(1).sign, 1)  # above the mined boundary 0
        # The −1 pole is deliberately unused: the mode IS "no casts".
        self.assertIn("corpus mode", read_casts(0).evidence)

    def test_numerals_boundary_is_the_mined_top_tertile(self):
        self.assertEqual(read_numerals(1).sign, 0)  # within mode (boundary 1)
        self.assertEqual(read_numerals(2).sign, 1)

    def test_coupling_poles(self):
        self.assertEqual(read_coupling(0, hyp_count=3).sign, -1)  # isolated target
        self.assertEqual(read_coupling(2, hyp_count=3).sign, 0)
        self.assertEqual(read_coupling(5, hyp_count=3).sign, 1)  # locally coupled

    def test_coupling_abstains_without_hypotheses(self):
        # The bare goal is CONTEXT's pole; coupling must not double-read it.
        read = read_coupling(0, hyp_count=0)
        self.assertEqual(read.sign, 0)
        self.assertIn("no opinion", read.evidence)

    def test_norms_are_remineable(self):
        self.assertEqual(mine_median([7, 11, 15]), 11.0)
        self.assertEqual(mine_top_tertile([0, 0, 0, 0, 1, 2]), 1.0)
        self.assertEqual(read_size(15, SurfaceNorms(size_zero=20.0, size_tol=4.0)).sign, -1)


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


class AtomicSweepTest(unittest.TestCase):
    """Close-before-unpack: the sweep exists exactly where decomposers would
    fire, and nowhere else (no risk → no duplicate attempts)."""

    def test_no_decomposition_risk_means_no_sweep(self):
        self.assertEqual(atomic_sweep_tactics("⊢ a = b"), [])
        self.assertEqual(atomic_sweep_tactics(""), [])

    def test_iff_target_gets_the_full_sweep(self):
        # The autopsy witness: constructor split `↔` goals whose true proof
        # was one bare simp. simp must lead the sweep.
        sweep = atomic_sweep_tactics("⊢ P ↔ Q")
        self.assertEqual(sweep[0], "simp")
        self.assertIn("omega", sweep)

    def test_forall_and_exists_targets_carry_risk(self):
        self.assertTrue(atomic_sweep_tactics("⊢ ∀ x : Nat, x = x"))
        self.assertTrue(atomic_sweep_tactics("⊢ ∃ n : ℕ, n > 3"))

    def test_numeric_suppression_removes_decision_procedures_only(self):
        sweep = atomic_sweep_tactics("⊢ P ↔ Q", suppress_numeric=True)
        self.assertIn("simp", sweep)
        for t in ("norm_num", "omega", "decide"):
            self.assertNotIn(t, sweep)

    def test_risk_is_read_on_the_target_not_the_hypotheses(self):
        # A hypothesis may mention ∀ while the target is atomic — no risk.
        goal = "h : ∀ x, P x\n⊢ Q"
        self.assertEqual(atomic_sweep_tactics(goal), [])


class TailRegimeTest(unittest.TestCase):
    """The §4.1b law-4 escalation read — named codes, abstract-domain veto."""

    def test_compact_concrete_arithmetic_is_tail_regime(self):
        self.assertEqual(goal_tail_regime("⊢ 2 + 2 = 4"), TAIL_COMPACT_ARITH)

    def test_bounded_nat_witness_is_tail_regime(self):
        self.assertEqual(
            goal_tail_regime("⊢ ∃ n : Nat, n * n = 16"), TAIL_BOUNDED_WITNESS
        )

    def test_abstract_domain_vetoes(self):
        self.assertEqual(
            goal_tail_regime("⊢ CategoryTheory.Functor.map f = 2 • g"), TAIL_NONE
        )

    def test_abstract_symbolic_goal_is_not_tail(self):
        # No numerals, no concrete-type surface: the bulk phase owns it.
        self.assertEqual(goal_tail_regime("⊢ f x = g x"), TAIL_NONE)
        self.assertEqual(goal_tail_regime(""), TAIL_NONE)

    def test_large_arith_target_stays_bulk(self):
        # Size is read against the MINED norm — a sprawling arithmetic
        # target is normalization territory, not a capsule close.
        big = "⊢ " + " + ".join(f"a{i} * 3" for i in range(20)) + " = 0"
        self.assertEqual(goal_tail_regime(big), TAIL_NONE)

    def test_mined_cell_routes_regardless_of_markers(self):
        # A REAL dr_ducky-close goal from the pass-1c traced corpus whose
        # (bucket, signature) sits in a mined concentration cell — no
        # concrete-type marker, no numeral, invisible to the marker
        # heuristics, routed by the population data (§10.7: cells are
        # mined, never hand-edited).
        self.assertEqual(goal_tail_regime("∃ x, f✝ ≤ 𝓝 x"), TAIL_MINED_CELL)


class DeterministicLaneOrderTest(unittest.TestCase):
    """The scheduler as a total pure function of the routing read."""

    def test_bulk_goal_keeps_rewriting_before_ducky(self):
        order = deterministic_lane_order(TAIL_NONE, False, True)
        self.assertEqual(
            order,
            [
                "structural_core",
                "symbolic_rw",
                "symbolic_exact",
                "symbolic_simp",
                "symbolic_simpa",
                "symbolic_rw_chain",
                "symbolic_simp_set",
                "symbolic_rw_at",
                "dr_ducky",
            ],
        )

    def test_composite_families_run_after_their_singles(self):
        # A single that closes makes its chain redundant — never vice versa.
        order = deterministic_lane_order(TAIL_NONE, False, True)
        self.assertLess(order.index("symbolic_rw"), order.index("symbolic_rw_chain"))
        self.assertLess(order.index("symbolic_simp"), order.index("symbolic_simp_set"))

    def test_tail_goal_reaches_ducky_before_any_rewriting(self):
        # The round-4 preemption fix: all four lost solves were dr_ducky
        # closes whose goals were rewritten out of capsule form first.
        order = deterministic_lane_order(TAIL_COMPACT_ARITH, False, True)
        self.assertLess(order.index("dr_ducky"), order.index("symbolic_rw"))
        self.assertEqual(order[0], "structural_core")

    def test_ducky_disabled_never_schedules_her(self):
        for code in (TAIL_NONE, TAIL_COMPACT_ARITH, TAIL_BOUNDED_WITNESS, TAIL_MINED_CELL):
            self.assertNotIn("dr_ducky", deterministic_lane_order(code, True, False))

    def test_mined_cell_code_routes_ducky_first(self):
        order = deterministic_lane_order(TAIL_MINED_CELL, False, True)
        self.assertLess(order.index("dr_ducky"), order.index("symbolic_rw"))

    def test_apply_lane_is_config_gated(self):
        self.assertIn("symbolic_apply", deterministic_lane_order(TAIL_NONE, True, True))
        self.assertNotIn("symbolic_apply", deterministic_lane_order(TAIL_NONE, False, True))


if __name__ == "__main__":
    unittest.main()
