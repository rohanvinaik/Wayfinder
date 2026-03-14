"""Tests for eval_retrieval — coverage-aware metrics (Stage 2)."""

import unittest

from scripts.eval_retrieval import (
    compute_conditional_recall_at_k,
    compute_recall_at_k,
    compute_universe_coverage,
)


class TestComputeRecallAtK(unittest.TestCase):
    """Predicate→effect tests for compute_recall_at_k."""

    def test_perfect_recall(self):
        self.assertAlmostEqual(compute_recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3), 1.0)

    def test_zero_recall(self):
        self.assertAlmostEqual(compute_recall_at_k(["x", "y"], ["a", "b"], k=2), 0.0)

    def test_partial_recall(self):
        self.assertAlmostEqual(compute_recall_at_k(["a", "x"], ["a", "b"], k=2), 0.5)

    def test_k_limits_retrieved(self):
        # Only first k=1 retrieved item counts
        self.assertAlmostEqual(compute_recall_at_k(["x", "a"], ["a"], k=1), 0.0)

    def test_empty_gt_returns_one(self):
        self.assertAlmostEqual(compute_recall_at_k(["a"], [], k=1), 1.0)

    def test_empty_retrieved_returns_zero(self):
        self.assertAlmostEqual(compute_recall_at_k([], ["a"], k=1), 0.0)

    def test_recall_is_fraction_of_gt(self):
        # 1 of 4 GT found → 0.25
        result = compute_recall_at_k(["a"], ["a", "b", "c", "d"], k=4)
        self.assertAlmostEqual(result, 0.25)


class TestComputeUniverseCoverage(unittest.TestCase):
    """Tests for compute_universe_coverage (Stage 2 metric)."""

    def test_full_coverage(self):
        cov, covered, uncovered = compute_universe_coverage(["a", "b"], {"a", "b", "c"})
        self.assertAlmostEqual(cov, 1.0)
        self.assertEqual(sorted(covered), ["a", "b"])
        self.assertEqual(uncovered, [])

    def test_zero_coverage(self):
        cov, covered, uncovered = compute_universe_coverage(["x", "y"], {"a", "b"})
        self.assertAlmostEqual(cov, 0.0)
        self.assertEqual(covered, [])
        self.assertEqual(sorted(uncovered), ["x", "y"])

    def test_partial_coverage(self):
        cov, covered, uncovered = compute_universe_coverage(["a", "x", "b"], {"a", "b", "c"})
        self.assertAlmostEqual(cov, 2.0 / 3.0, places=4)
        self.assertEqual(sorted(covered), ["a", "b"])
        self.assertEqual(uncovered, ["x"])

    def test_empty_gt_returns_one(self):
        cov, covered, uncovered = compute_universe_coverage([], {"a"})
        self.assertAlmostEqual(cov, 1.0)

    def test_empty_universe(self):
        cov, covered, uncovered = compute_universe_coverage(["a"], set())
        self.assertAlmostEqual(cov, 0.0)

    def test_preserves_order(self):
        _, covered, uncovered = compute_universe_coverage(["c", "a", "x"], {"a", "c"})
        self.assertEqual(covered, ["c", "a"])
        self.assertEqual(uncovered, ["x"])


class TestComputeConditionalRecallAtK(unittest.TestCase):
    """Tests for compute_conditional_recall_at_k (Stage 2 metric).

    This separates retrieval quality from extraction coverage by only
    counting ground-truth premises that exist in the entity universe.
    """

    def test_perfect_conditional_recall(self):
        # GT = [a, b], both in universe, both retrieved
        result = compute_conditional_recall_at_k(["a", "b"], ["a", "b"], {"a", "b"}, k=2)
        self.assertAlmostEqual(result, 1.0)

    def test_unreachable_gt_ignored(self):
        # GT = [a, x], x not in universe → only a counts
        # a is retrieved → conditional recall = 1.0
        result = compute_conditional_recall_at_k(["a"], ["a", "x"], {"a", "b"}, k=1)
        self.assertAlmostEqual(result, 1.0)

    def test_all_gt_unreachable(self):
        # All GT outside universe → vacuously correct
        result = compute_conditional_recall_at_k(["a"], ["x", "y"], {"a", "b"}, k=1)
        self.assertAlmostEqual(result, 1.0)

    def test_reachable_but_not_retrieved(self):
        # GT = [a, b], both reachable, neither retrieved
        result = compute_conditional_recall_at_k(["x", "y"], ["a", "b"], {"a", "b"}, k=2)
        self.assertAlmostEqual(result, 0.0)

    def test_partial_conditional_recall(self):
        # GT = [a, b, x], x unreachable → reachable GT = [a, b]
        # Only a retrieved → 1/2 = 0.5
        result = compute_conditional_recall_at_k(["a", "z"], ["a", "b", "x"], {"a", "b", "z"}, k=2)
        self.assertAlmostEqual(result, 0.5)

    def test_k_limits_apply(self):
        # b is at position 2, but k=1 → only first item checked
        result = compute_conditional_recall_at_k(["x", "b"], ["a", "b"], {"a", "b"}, k=1)
        self.assertAlmostEqual(result, 0.0)

    def test_differs_from_raw_recall(self):
        """Conditional recall should be higher than raw when coverage < 100%."""
        gt = ["a", "b", "x_missing"]
        retrieved = ["a"]
        universe = {"a", "b"}
        raw = compute_recall_at_k(retrieved, gt, k=1)
        cond = compute_conditional_recall_at_k(retrieved, gt, universe, k=1)
        # raw = 1/3, cond = 1/2 (x_missing excluded from denominator)
        self.assertAlmostEqual(raw, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(cond, 0.5)
        self.assertGreater(cond, raw)


if __name__ == "__main__":
    unittest.main()
