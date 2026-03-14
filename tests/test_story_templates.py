"""Tests for story_templates — taxonomy, classification, bank signatures, indexing."""

import unittest

from src.nav_contracts import BANK_NAMES
from src.story_templates import (
    COMPLEX_TEMPLATES,
    SIMPLE_TEMPLATES,
    TEMPLATE_NAMES,
    TEMPLATE_TAXONOMY,
    _apply_structural_heuristics,
    _score_by_pattern_match,
    classify_tactic_sequence,
    compute_bank_signature,
    get_num_templates,
    get_template_index,
    get_template_name,
)


class TestTemplateTaxonomy(unittest.TestCase):
    """Tests for the TEMPLATE_TAXONOMY constant and its structure."""

    def test_taxonomy_has_nine_templates(self):
        self.assertEqual(len(TEMPLATE_TAXONOMY), 9)

    def test_taxonomy_keys_match_expected_names(self):
        expected = {
            "DECIDE",
            "REWRITE_CHAIN",
            "INDUCT_THEN_CLOSE",
            "DECOMPOSE_AND_CONQUER",
            "APPLY_CHAIN",
            "CASE_ANALYSIS",
            "CONTRAPOSITIVE",
            "EPSILON_DELTA",
            "HAMMER_DELEGATE",
        }
        self.assertEqual(set(TEMPLATE_TAXONOMY.keys()), expected)

    def test_template_names_list_length(self):
        self.assertEqual(len(TEMPLATE_NAMES), 9)

    def test_template_names_order(self):
        expected_order = [
            "DECIDE",
            "REWRITE_CHAIN",
            "INDUCT_THEN_CLOSE",
            "DECOMPOSE_AND_CONQUER",
            "APPLY_CHAIN",
            "CASE_ANALYSIS",
            "CONTRAPOSITIVE",
            "EPSILON_DELTA",
            "HAMMER_DELEGATE",
        ]
        self.assertEqual(TEMPLATE_NAMES, expected_order)

    def test_template_names_matches_taxonomy_keys(self):
        self.assertEqual(set(TEMPLATE_NAMES), set(TEMPLATE_TAXONOMY.keys()))

    def test_each_template_has_template_id_matching_key(self):
        for name, info in TEMPLATE_TAXONOMY.items():
            self.assertEqual(info.template_id, name)

    def test_each_template_has_nonempty_pattern(self):
        for name, info in TEMPLATE_TAXONOMY.items():
            self.assertIsInstance(info.pattern, str, msg=name)
            self.assertTrue(len(info.pattern) > 0, msg=name)

    def test_each_template_has_bank_signature(self):
        for name, info in TEMPLATE_TAXONOMY.items():
            self.assertIsInstance(info.bank_signature, dict, msg=name)
            for bank, direction in info.bank_signature.items():
                self.assertIn(bank, BANK_NAMES, msg=f"{name}: unknown bank {bank}")
                self.assertIn(direction, (-1, 0, 1), msg=f"{name}.{bank}")

    def test_each_template_has_tactic_patterns(self):
        for name, info in TEMPLATE_TAXONOMY.items():
            self.assertIsInstance(info.tactic_patterns, list, msg=name)
            self.assertGreater(len(info.tactic_patterns), 0, msg=name)


class TestSimpleComplexPartition(unittest.TestCase):
    """Tests that SIMPLE_TEMPLATES and COMPLEX_TEMPLATES partition all 9."""

    def test_simple_templates_contents(self):
        expected = {"DECIDE", "REWRITE_CHAIN", "APPLY_CHAIN", "HAMMER_DELEGATE"}
        self.assertEqual(SIMPLE_TEMPLATES, expected)

    def test_complex_templates_contents(self):
        expected = {
            "INDUCT_THEN_CLOSE",
            "DECOMPOSE_AND_CONQUER",
            "CASE_ANALYSIS",
            "CONTRAPOSITIVE",
            "EPSILON_DELTA",
        }
        self.assertEqual(COMPLEX_TEMPLATES, expected)

    def test_simple_and_complex_are_disjoint(self):
        self.assertEqual(SIMPLE_TEMPLATES & COMPLEX_TEMPLATES, set())

    def test_simple_and_complex_cover_all_templates(self):
        self.assertEqual(SIMPLE_TEMPLATES | COMPLEX_TEMPLATES, set(TEMPLATE_NAMES))

    def test_partition_size(self):
        self.assertEqual(len(SIMPLE_TEMPLATES) + len(COMPLEX_TEMPLATES), 9)


class TestClassifyTacticSequence(unittest.TestCase):
    """Tests for classify_tactic_sequence."""

    def test_empty_list_returns_decide(self):
        self.assertEqual(classify_tactic_sequence([]), "DECIDE")

    def test_single_omega_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["omega"]), "DECIDE")

    def test_single_simp_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["simp"]), "DECIDE")

    def test_single_decide_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["decide"]), "DECIDE")

    def test_single_norm_num_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["norm_num"]), "DECIDE")

    def test_single_ring_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["ring"]), "DECIDE")

    def test_single_linarith_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["linarith"]), "DECIDE")

    def test_single_aesop_returns_hammer_delegate(self):
        self.assertEqual(classify_tactic_sequence(["aesop"]), "HAMMER_DELEGATE")

    def test_single_tauto_returns_hammer_delegate(self):
        self.assertEqual(classify_tactic_sequence(["tauto"]), "HAMMER_DELEGATE")

    def test_single_apply_returns_apply_chain(self):
        self.assertEqual(classify_tactic_sequence(["apply"]), "APPLY_CHAIN")

    def test_single_exact_returns_apply_chain(self):
        self.assertEqual(classify_tactic_sequence(["exact"]), "APPLY_CHAIN")

    def test_single_rw_returns_rewrite_chain(self):
        self.assertEqual(classify_tactic_sequence(["rw"]), "REWRITE_CHAIN")

    def test_single_rewrite_returns_rewrite_chain(self):
        self.assertEqual(classify_tactic_sequence(["rewrite"]), "REWRITE_CHAIN")

    def test_single_unknown_tactic_returns_decide(self):
        self.assertEqual(classify_tactic_sequence(["some_unknown_tactic"]), "DECIDE")

    def test_tactic_args_ignored_uses_first_word(self):
        # "apply Nat.add_comm" should use "apply" only
        self.assertEqual(classify_tactic_sequence(["apply Nat.add_comm"]), "APPLY_CHAIN")

    def test_case_insensitive(self):
        self.assertEqual(classify_tactic_sequence(["OMEGA"]), "DECIDE")
        self.assertEqual(classify_tactic_sequence(["Aesop"]), "HAMMER_DELEGATE")

    def test_multi_tactic_induction_returns_induct_then_close(self):
        tactics = ["induction", "simp", "omega"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "INDUCT_THEN_CLOSE")

    def test_multi_tactic_by_contra_returns_contrapositive(self):
        tactics = ["by_contra", "apply", "contradiction"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "CONTRAPOSITIVE")

    def test_multi_tactic_have_suffices_returns_decompose_and_conquer(self):
        tactics = ["have", "suffices", "exact"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "DECOMPOSE_AND_CONQUER")

    def test_multi_tactic_cases_without_induction_returns_case_analysis(self):
        # "cases" + "rcases" strongly match CASE_ANALYSIS patterns,
        # and the heuristic boost of 0.4 pushes it over INDUCT_THEN_CLOSE.
        tactics = ["cases", "rcases", "exact"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "CASE_ANALYSIS")

    def test_multi_tactic_cases_with_induction_prefers_induct(self):
        # When both "induction" and "cases" are present, induction heuristic
        # should dominate, and "cases" heuristic is excluded by induction presence.
        tactics = ["induction", "cases", "simp", "omega"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "INDUCT_THEN_CLOSE")

    def test_multi_tactic_rewrite_chain(self):
        tactics = ["rw", "simp", "rw", "ring"]
        result = classify_tactic_sequence(tactics)
        self.assertEqual(result, "REWRITE_CHAIN")

    def test_classify_always_returns_valid_template(self):
        # Various inputs should all return something in TEMPLATE_NAMES
        cases = [
            [],
            ["omega"],
            ["foo"],
            ["apply", "exact"],
            ["induction", "cases", "simp"],
            ["by_contra", "exact"],
            ["have", "apply"],
        ]
        for tactics in cases:
            result = classify_tactic_sequence(tactics)
            self.assertIn(result, TEMPLATE_NAMES, msg=f"tactics={tactics}")


class TestScoreByPatternMatch(unittest.TestCase):
    """Tests for _score_by_pattern_match."""

    def test_returns_scores_for_all_templates(self):
        scores = _score_by_pattern_match(["omega", "simp"], 2)
        self.assertEqual(set(scores.keys()), set(TEMPLATE_NAMES))

    def test_scores_are_nonnegative(self):
        scores = _score_by_pattern_match(["apply", "exact", "refine"], 3)
        for name, score in scores.items():
            self.assertGreaterEqual(score, 0.0, msg=name)

    def test_scores_do_not_exceed_one(self):
        scores = _score_by_pattern_match(["omega", "simp", "decide"], 3)
        for name, score in scores.items():
            self.assertLessEqual(score, 1.0, msg=name)

    def test_perfect_match_for_apply_chain(self):
        # All tactics match APPLY_CHAIN patterns (apply, exact, refine)
        scores = _score_by_pattern_match(["apply", "exact", "refine"], 3)
        self.assertAlmostEqual(scores["APPLY_CHAIN"], 1.0)

    def test_no_match_gives_zero(self):
        # Tactics that match nothing in CONTRAPOSITIVE patterns
        scores = _score_by_pattern_match(["rw", "simp"], 2)
        self.assertAlmostEqual(scores["CONTRAPOSITIVE"], 0.0)

    def test_partial_match_fraction(self):
        # 1 out of 2 tactics matches DECIDE patterns (omega matches, intro doesn't)
        scores = _score_by_pattern_match(["omega", "intro"], 2)
        self.assertAlmostEqual(scores["DECIDE"], 0.5)


class TestApplyStructuralHeuristics(unittest.TestCase):
    """Tests for _apply_structural_heuristics in-place mutation."""

    def _zero_scores(self):
        return {name: 0.0 for name in TEMPLATE_NAMES}

    def test_induction_boosts_induct_then_close(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["induction", "simp"], scores)
        self.assertAlmostEqual(scores["INDUCT_THEN_CLOSE"], 0.5)

    def test_induct_keyword_also_boosts(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["induct", "omega"], scores)
        self.assertAlmostEqual(scores["INDUCT_THEN_CLOSE"], 0.5)

    def test_by_contra_boosts_contrapositive(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["by_contra", "exact"], scores)
        self.assertAlmostEqual(scores["CONTRAPOSITIVE"], 0.5)

    def test_have_boosts_decompose_and_conquer(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["have", "exact"], scores)
        self.assertAlmostEqual(scores["DECOMPOSE_AND_CONQUER"], 0.4)

    def test_suffices_boosts_decompose_and_conquer(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["suffices", "exact"], scores)
        self.assertAlmostEqual(scores["DECOMPOSE_AND_CONQUER"], 0.4)

    def test_obtain_boosts_decompose_and_conquer(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["obtain", "exact"], scores)
        self.assertAlmostEqual(scores["DECOMPOSE_AND_CONQUER"], 0.4)

    def test_cases_without_induction_boosts_case_analysis(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["cases", "simp"], scores)
        self.assertAlmostEqual(scores["CASE_ANALYSIS"], 0.4)

    def test_cases_with_induction_does_not_boost_case_analysis(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["cases", "induction", "simp"], scores)
        # CASE_ANALYSIS should NOT be boosted when induction is present
        self.assertAlmostEqual(scores["CASE_ANALYSIS"], 0.0)

    def test_calc_boosts_epsilon_delta(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["calc", "linarith"], scores)
        self.assertAlmostEqual(scores["EPSILON_DELTA"], 0.3)

    def test_rw_without_structural_tactics_boosts_rewrite_chain(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["rw", "simp"], scores)
        self.assertAlmostEqual(scores["REWRITE_CHAIN"], 0.3)

    def test_rw_with_induction_does_not_boost_rewrite_chain(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["rw", "induction"], scores)
        # REWRITE_CHAIN should NOT be boosted when induction is present
        self.assertAlmostEqual(scores["REWRITE_CHAIN"], 0.0)

    def test_rw_with_cases_does_not_boost_rewrite_chain(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["rw", "cases"], scores)
        # REWRITE_CHAIN excluded when cases is present
        self.assertAlmostEqual(scores["REWRITE_CHAIN"], 0.0)

    def test_no_triggers_leaves_scores_unchanged(self):
        scores = self._zero_scores()
        _apply_structural_heuristics(["intro", "exact"], scores)
        for name in TEMPLATE_NAMES:
            self.assertAlmostEqual(scores[name], 0.0, msg=name)

    def test_multiple_heuristics_can_fire(self):
        scores = self._zero_scores()
        # "have" triggers DECOMPOSE_AND_CONQUER, "calc" triggers EPSILON_DELTA
        _apply_structural_heuristics(["have", "calc", "exact"], scores)
        self.assertAlmostEqual(scores["DECOMPOSE_AND_CONQUER"], 0.4)
        self.assertAlmostEqual(scores["EPSILON_DELTA"], 0.3)


class TestComputeBankSignature(unittest.TestCase):
    """Tests for compute_bank_signature."""

    def test_empty_sequence_returns_all_zeros(self):
        sig = compute_bank_signature([])
        for bank in BANK_NAMES:
            self.assertEqual(sig[bank], 0, msg=bank)

    def test_empty_has_all_six_banks(self):
        sig = compute_bank_signature([])
        self.assertEqual(set(sig.keys()), set(BANK_NAMES))

    def test_single_step_identity(self):
        step = {
            "structure": 1,
            "domain": -1,
            "depth": 0,
            "automation": 1,
            "context": -1,
            "decomposition": 0,
        }
        sig = compute_bank_signature([step])
        for bank in BANK_NAMES:
            self.assertEqual(sig[bank], step[bank], msg=bank)

    def test_reinforcement_two_positive_steps(self):
        steps = [
            {
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            {
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
        ]
        sig = compute_bank_signature(steps)
        self.assertEqual(sig["structure"], 1)

    def test_cancellation_opposite_steps(self):
        steps = [
            {
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            {
                "structure": -1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
        ]
        sig = compute_bank_signature(steps)
        self.assertEqual(sig["structure"], 0)

    def test_dominant_direction_positive(self):
        steps = [
            {
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            {
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            {
                "structure": -1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
        ]
        sig = compute_bank_signature(steps)
        # sum = 1 + 1 + (-1) = 1 > 0, so sign = 1
        self.assertEqual(sig["structure"], 1)

    def test_dominant_direction_negative(self):
        steps = [
            {"automation": -1},
            {"automation": -1},
            {"automation": 1},
        ]
        sig = compute_bank_signature(steps)
        # sum = -1 + -1 + 1 = -1 < 0, so sign = -1
        self.assertEqual(sig["automation"], -1)

    def test_partial_bank_keys_in_steps(self):
        # Steps that only specify some banks; missing banks stay at 0
        steps = [{"structure": 1}, {"domain": -1}]
        sig = compute_bank_signature(steps)
        self.assertEqual(sig["structure"], 1)
        self.assertEqual(sig["domain"], -1)
        self.assertEqual(sig["depth"], 0)
        self.assertEqual(sig["automation"], 0)
        self.assertEqual(sig["context"], 0)
        self.assertEqual(sig["decomposition"], 0)

    def test_unknown_bank_in_step_ignored(self):
        # A bank key not in BANK_NAMES should be ignored
        steps = [{"structure": 1, "bogus_bank": 1}]
        sig = compute_bank_signature(steps)
        self.assertEqual(sig["structure"], 1)
        self.assertNotIn("bogus_bank", sig)

    def test_result_values_are_minus1_zero_or_plus1(self):
        steps = [
            {
                "structure": 1,
                "domain": -1,
                "depth": 1,
                "automation": -1,
                "context": 0,
                "decomposition": 1,
            },
            {
                "structure": 1,
                "domain": 1,
                "depth": -1,
                "automation": -1,
                "context": 0,
                "decomposition": 1,
            },
        ]
        sig = compute_bank_signature(steps)
        for bank in BANK_NAMES:
            self.assertIn(sig[bank], (-1, 0, 1), msg=bank)


class TestTemplateIndexing(unittest.TestCase):
    """Tests for get_template_index, get_template_name, get_num_templates."""

    def test_get_num_templates_returns_nine(self):
        self.assertEqual(get_num_templates(), 9)

    def test_get_template_index_decide(self):
        self.assertEqual(get_template_index("DECIDE"), 0)

    def test_get_template_index_hammer_delegate(self):
        self.assertEqual(get_template_index("HAMMER_DELEGATE"), 8)

    def test_get_template_name_zero(self):
        self.assertEqual(get_template_name(0), "DECIDE")

    def test_get_template_name_last(self):
        self.assertEqual(get_template_name(8), "HAMMER_DELEGATE")

    def test_index_name_roundtrip_all(self):
        for i, name in enumerate(TEMPLATE_NAMES):
            self.assertEqual(get_template_index(name), i)
            self.assertEqual(get_template_name(i), name)

    def test_get_template_index_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_template_index("NONEXISTENT_TEMPLATE")

    def test_get_template_name_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            get_template_name(99)


if __name__ == "__main__":
    unittest.main()
