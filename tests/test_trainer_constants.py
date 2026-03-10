"""Tests for trainer_constants — domain inference and constant tables."""

import unittest
from types import SimpleNamespace

from src.trainer_constants import (
    _DEFAULT_DOMAIN,
    _DOMAIN_KEYWORDS,
    _REPAIR_SEVERITY,
    _TACTIC_DOMAIN_HINTS,
    infer_domain,
)


class TestInferDomain(unittest.TestCase):

    def _make_example(self, goal_state="", theorem_statement="", tier1_tokens=None):
        return SimpleNamespace(
            goal_state=goal_state,
            theorem_statement=theorem_statement,
            tier1_tokens=tier1_tokens or [],
        )

    def test_algebra_from_goal(self):
        ex = self._make_example(goal_state="⊢ a * b = b * a in a commutative ring")
        self.assertEqual(infer_domain(ex), "algebra")

    def test_analysis_from_goal(self):
        ex = self._make_example(goal_state="continuous f → limit (f x) = L")
        self.assertEqual(infer_domain(ex), "analysis")

    def test_topology_from_goal(self):
        ex = self._make_example(goal_state="compact X → hausdorff X")
        self.assertEqual(infer_domain(ex), "topology")

    def test_number_theory_from_goal(self):
        ex = self._make_example(goal_state="prime p → divisible n p")
        self.assertEqual(infer_domain(ex), "number_theory")

    def test_linear_algebra_from_goal(self):
        ex = self._make_example(goal_state="matrix M has eigenvalue λ")
        self.assertEqual(infer_domain(ex), "linear_algebra")

    def test_domain_from_theorem_statement(self):
        ex = self._make_example(theorem_statement="supremum of a lattice")
        self.assertEqual(infer_domain(ex), "order_theory")

    def test_tactic_hint_fallback(self):
        ex = self._make_example(goal_state="⊢ x + y = z", tier1_tokens=["omega"])
        # Goal has "+" but not the keyword "add", so tactic hint wins
        self.assertEqual(infer_domain(ex), "number_theory")

    def test_tactic_hint_when_no_keyword_match(self):
        ex = self._make_example(goal_state="⊢ P", tier1_tokens=["ring"])
        self.assertEqual(infer_domain(ex), "algebra")

    def test_tactic_hint_omega(self):
        ex = self._make_example(goal_state="⊢ P", tier1_tokens=["omega"])
        self.assertEqual(infer_domain(ex), "number_theory")

    def test_tactic_hint_linarith(self):
        ex = self._make_example(goal_state="⊢ P", tier1_tokens=["linarith"])
        self.assertEqual(infer_domain(ex), "linear_algebra")

    def test_default_domain_when_no_match(self):
        ex = self._make_example(goal_state="⊢ True")
        self.assertEqual(infer_domain(ex), "general")

    def test_case_insensitive(self):
        ex = self._make_example(goal_state="COMPACT HAUSDORFF space")
        self.assertEqual(infer_domain(ex), "topology")

    def test_missing_goal_state_attribute(self):
        ex = SimpleNamespace()
        result = infer_domain(ex)
        self.assertEqual(result, _DEFAULT_DOMAIN)

    def test_none_goal_state(self):
        ex = SimpleNamespace(goal_state=None, theorem_statement=None)
        result = infer_domain(ex)
        self.assertEqual(result, _DEFAULT_DOMAIN)


class TestConstants(unittest.TestCase):

    def test_domain_keywords_has_all_domains(self):
        expected_domains = {
            "algebra", "analysis", "topology", "number_theory",
            "linear_algebra", "order_theory", "set_theory", "logic",
            "combinatorics", "category_theory",
        }
        self.assertEqual(set(_DOMAIN_KEYWORDS.keys()), expected_domains)

    def test_each_domain_has_keywords(self):
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            self.assertIsInstance(keywords, tuple, f"{domain} keywords should be tuple")
            self.assertGreater(len(keywords), 0, f"{domain} should have keywords")

    def test_tactic_domain_hints_values_are_valid_domains(self):
        for tactic, domain in _TACTIC_DOMAIN_HINTS.items():
            self.assertIn(
                domain, _DOMAIN_KEYWORDS,
                f"Tactic '{tactic}' maps to unknown domain '{domain}'",
            )

    def test_repair_severity_values_are_positive(self):
        for category, severity in _REPAIR_SEVERITY.items():
            self.assertGreater(severity, 0.0, f"{category} severity should be positive")

    def test_default_domain_is_general(self):
        self.assertEqual(_DEFAULT_DOMAIN, "general")


if __name__ == "__main__":
    unittest.main()
