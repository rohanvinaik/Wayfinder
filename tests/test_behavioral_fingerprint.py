"""Tests for behavioral_fingerprint — pure numerical functions."""

import unittest

import numpy as np

from src.behavioral_fingerprint import (
    compute_action_distribution,
    compute_action_entropy,
    compute_discreteness,
    compute_variance_eigenvalues,
)


class TestComputeActionEntropy(unittest.TestCase):
    """Test Shannon entropy computation over action predictions."""

    def test_empty_returns_zero(self):
        self.assertEqual(compute_action_entropy([]), 0.0)

    def test_single_action_entropy_zero(self):
        # All same action → probability 1.0 → -1*log2(1) = 0
        result = compute_action_entropy(["simp", "simp", "simp"])
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_two_uniform_actions(self):
        # Two equally frequent actions → entropy = log2(2) = 1.0
        result = compute_action_entropy(["a", "b", "a", "b"])
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_four_uniform_actions(self):
        # Four equally frequent → entropy = log2(4) = 2.0
        result = compute_action_entropy(["a", "b", "c", "d"])
        self.assertAlmostEqual(result, 2.0, places=5)

    def test_skewed_distribution(self):
        # 3 of one, 1 of another → entropy between 0 and 1
        result = compute_action_entropy(["a", "a", "a", "b"])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_single_element_list(self):
        result = compute_action_entropy(["x"])
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_entropy_is_non_negative(self):
        result = compute_action_entropy(["a", "b", "c", "a", "b"])
        self.assertGreaterEqual(result, 0.0)


class TestComputeActionDistribution(unittest.TestCase):
    """Test normalized frequency distribution computation."""

    def test_empty_returns_empty_dict(self):
        self.assertEqual(compute_action_distribution([]), {})

    def test_single_action(self):
        result = compute_action_distribution(["simp"])
        self.assertEqual(result, {"simp": 1.0})

    def test_uniform_two_actions(self):
        result = compute_action_distribution(["a", "b", "a", "b"])
        self.assertEqual(result["a"], 0.5)
        self.assertEqual(result["b"], 0.5)

    def test_frequencies_sum_to_one(self):
        result = compute_action_distribution(["a", "b", "c", "a", "a"])
        self.assertAlmostEqual(sum(result.values()), 1.0, places=10)

    def test_keys_are_sorted(self):
        result = compute_action_distribution(["c", "a", "b"])
        self.assertEqual(list(result.keys()), ["a", "b", "c"])

    def test_exact_frequencies(self):
        result = compute_action_distribution(["x", "x", "y"])
        self.assertAlmostEqual(result["x"], 2 / 3, places=10)
        self.assertAlmostEqual(result["y"], 1 / 3, places=10)


class TestComputeVarianceEigenvalues(unittest.TestCase):
    """Test top eigenvalue extraction from output logits."""

    def test_1d_input_returns_empty(self):
        logits = np.array([1.0, 2.0, 3.0])
        self.assertEqual(compute_variance_eigenvalues(logits), [])

    def test_single_row_returns_empty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        self.assertEqual(compute_variance_eigenvalues(logits), [])

    def test_two_rows_returns_eigenvalues(self):
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = compute_variance_eigenvalues(logits)
        self.assertGreater(len(result), 0)

    def test_eigenvalues_are_sorted_descending(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(10, 5)
        result = compute_variance_eigenvalues(logits)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i], result[i + 1])

    def test_top_k_truncation(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(20, 15)
        result_default = compute_variance_eigenvalues(logits)
        result_k3 = compute_variance_eigenvalues(logits, top_k=3)
        self.assertLessEqual(len(result_default), 10)
        self.assertEqual(len(result_k3), 3)

    def test_returns_floats(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = compute_variance_eigenvalues(logits)
        for v in result:
            self.assertIsInstance(v, float)

    def test_identical_rows_eigenvalues_near_zero(self):
        logits = np.array([[1.0, 2.0]] * 5)
        result = compute_variance_eigenvalues(logits)
        for v in result:
            self.assertAlmostEqual(v, 0.0, places=5)


class TestComputeDiscreteness(unittest.TestCase):
    """Test discreteness score from output logits."""

    def test_1d_returns_zero(self):
        self.assertEqual(compute_discreteness(np.array([1.0, 2.0])), 0.0)

    def test_empty_rows_returns_zero(self):
        logits = np.zeros((0, 3))
        self.assertEqual(compute_discreteness(logits), 0.0)

    def test_peaked_logits_high_discreteness(self):
        # One very high logit → softmax peaked → large gap
        logits = np.array([[100.0, 0.0, 0.0]])
        result = compute_discreteness(logits)
        self.assertGreater(result, 0.9)

    def test_uniform_logits_low_discreteness(self):
        # All equal → softmax uniform → small gap
        logits = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = compute_discreteness(logits)
        self.assertLess(result, 0.1)

    def test_multiple_rows_averaged(self):
        # Mix of peaked and uniform
        logits = np.array([
            [100.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        result = compute_discreteness(logits)
        # Should be between peaked and uniform
        self.assertGreater(result, 0.1)
        self.assertLess(result, 0.95)

    def test_single_column(self):
        # One element per row → gap is the element itself (softmax=1.0)
        logits = np.array([[5.0], [3.0]])
        result = compute_discreteness(logits)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_result_between_zero_and_one(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(10, 5)
        result = compute_discreteness(logits)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
