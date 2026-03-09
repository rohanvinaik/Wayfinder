"""Tests for PAB metric computation functions."""

import unittest

import numpy as np

from src.pab_metrics import (
    classify_domain,
    compute_crystallization,
    compute_feature_importance,
    compute_generalization_gap,
    compute_predictability,
    compute_repr_evolution,
    compute_stability,
    find_tier_convergence,
    is_oscillating,
    linear_slope,
    monotonic_trend,
)


class TestComputeStability(unittest.TestCase):
    def test_no_change(self):
        self.assertAlmostEqual(compute_stability(1.0, 1.0), 0.0, places=5)

    def test_decreasing_loss(self):
        s = compute_stability(2.0, 1.0)
        self.assertGreater(s, 0.0)

    def test_increasing_loss(self):
        s = compute_stability(1.0, 2.0)
        self.assertGreater(s, 0.0)


class TestComputeGeneralizationGap(unittest.TestCase):
    def test_positive_gap(self):
        self.assertAlmostEqual(compute_generalization_gap(1.0, 1.5), 0.5)

    def test_negative_gap(self):
        self.assertAlmostEqual(compute_generalization_gap(1.5, 1.0), -0.5)


class TestComputeFeatureImportance(unittest.TestCase):
    def test_too_few_snapshots(self):
        self.assertEqual(compute_feature_importance([]), 0.0)
        self.assertEqual(compute_feature_importance([[1, 2]]), 0.0)

    def test_identical_snapshots(self):
        w = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(compute_feature_importance([w, w, w]), 0.0)

    def test_varying_snapshots(self):
        val = compute_feature_importance([[1.0, 0.0], [0.0, 1.0]])
        self.assertGreater(val, 0.0)


class TestComputePredictability(unittest.TestCase):
    def test_too_few(self):
        self.assertEqual(compute_predictability([1.0]), 0.0)

    def test_constant_loss(self):
        self.assertAlmostEqual(compute_predictability([1.0, 1.0, 1.0, 1.0]), 0.0)

    def test_varying_loss(self):
        losses = [2.0, 1.5, 1.8, 1.2, 1.6, 1.1]
        self.assertGreater(compute_predictability(losses), 0.0)

    def test_two_values(self):
        self.assertEqual(compute_predictability([1.0, 0.5]), 0.0)


class TestComputeReprEvolution(unittest.TestCase):
    def test_no_previous(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        r_t, new_mean = compute_repr_evolution(emb, None)
        self.assertEqual(r_t, 1.0)
        np.testing.assert_array_almost_equal(new_mean, [0.5, 0.5])

    def test_same_direction(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]])
        prev = np.array([1.0, 0.0])
        r_t, _ = compute_repr_evolution(emb, prev)
        self.assertAlmostEqual(r_t, 0.0, places=5)

    def test_orthogonal(self):
        emb = np.array([[0.0, 1.0]])
        prev = np.array([1.0, 0.0])
        r_t, _ = compute_repr_evolution(emb, prev)
        self.assertAlmostEqual(r_t, 1.0, places=5)


class TestComputeCrystallization(unittest.TestCase):
    def test_no_previous(self):
        self.assertEqual(compute_crystallization(np.array([1, -1]), None), 0.0)

    def test_length_mismatch(self):
        self.assertEqual(compute_crystallization(np.array([1, -1]), np.array([1])), 0.0)

    def test_identical_signs(self):
        signs = np.array([1, -1, 0, 1])
        c = compute_crystallization(signs, signs)
        self.assertGreater(c, 0.99)

    def test_changed_signs(self):
        s1 = np.array([1, -1, 0, 1])
        s2 = np.array([-1, 1, 0, -1])
        c = compute_crystallization(s2, s1)
        self.assertLess(c, 0.5)


class TestClassifyDomain(unittest.TestCase):
    def test_too_few(self):
        self.assertEqual(classify_domain([0.9, 0.9], 10), "unknown")

    def test_early_learner(self):
        accs = [0.85, 0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96]
        self.assertEqual(classify_domain(accs, 9), "early")

    def test_late_learner(self):
        accs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.90]
        self.assertEqual(classify_domain(accs, 9), "late")

    def test_unstable_oscillating(self):
        accs = [0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.2, 0.85, 0.3]
        self.assertEqual(classify_domain(accs, 9), "unstable")

    def test_never_reached(self):
        accs = [0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.19, 0.21, 0.20]
        result = classify_domain(accs, 9)
        self.assertIn(result, ("late", "unstable"))


class TestIsOscillating(unittest.TestCase):
    def test_too_short(self):
        self.assertEqual(is_oscillating([1.0, 2.0]), False)

    def test_monotonic_increasing(self):
        # Strictly increasing = 0 reversals
        self.assertEqual(is_oscillating([1.0, 2.0, 3.0, 4.0, 5.0]), False)

    def test_monotonic_decreasing(self):
        # Strictly decreasing = 0 reversals
        self.assertEqual(is_oscillating([5.0, 4.0, 3.0, 2.0, 1.0]), False)

    def test_oscillating(self):
        # [1,3,1,3,1,3] has 4 reversals >= default min_reversals=3
        self.assertEqual(is_oscillating([1, 3, 1, 3, 1, 3]), True)

    def test_exactly_at_threshold(self):
        # [1,3,1,3,1] has exactly 3 reversals == min_reversals=3
        self.assertEqual(is_oscillating([1, 3, 1, 3, 1]), True)

    def test_below_threshold(self):
        # [1,3,1,3] has 2 reversals < default min_reversals=3
        self.assertEqual(is_oscillating([1, 3, 1, 3]), False)

    def test_custom_min_reversals(self):
        # [1,3,1,3] has 2 reversals — True with min_reversals=2, False with 3
        self.assertEqual(is_oscillating([1, 3, 1, 3], min_reversals=2), True)
        self.assertEqual(is_oscillating([1, 3, 1, 3], min_reversals=3), False)

    def test_flat_segments_no_reversal(self):
        # Equal consecutive values: product of deltas = 0, not < 0
        self.assertEqual(is_oscillating([1.0, 1.0, 1.0, 1.0, 1.0]), False)

    def test_single_reversal(self):
        # V-shape: only 1 reversal
        self.assertEqual(is_oscillating([5.0, 3.0, 1.0, 3.0, 5.0]), False)


class TestMonotonicTrend(unittest.TestCase):
    def test_too_short(self):
        self.assertEqual(monotonic_trend([1.0]), 0.0)

    def test_all_increasing(self):
        self.assertAlmostEqual(monotonic_trend([1.0, 2.0, 3.0]), 1.0)

    def test_all_decreasing(self):
        self.assertAlmostEqual(monotonic_trend([3.0, 2.0, 1.0]), 0.0)

    def test_mixed(self):
        val = monotonic_trend([1.0, 2.0, 1.5, 3.0])
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)


class TestFindTierConvergence(unittest.TestCase):
    def test_never_converges(self):
        self.assertIsNone(find_tier_convergence([0.1, 0.2, 0.3], 0.8))

    def test_converges_at_index(self):
        self.assertEqual(find_tier_convergence([0.5, 0.7, 0.85, 0.9], 0.8), 2)

    def test_with_checkpoints(self):
        result = find_tier_convergence([0.5, 0.85, 0.9], 0.8, checkpoints=[100, 200, 300])
        self.assertEqual(result, 200)

    def test_immediate(self):
        self.assertEqual(find_tier_convergence([0.9, 0.95], 0.8), 0)


class TestLinearSlope(unittest.TestCase):
    def test_too_short(self):
        self.assertEqual(linear_slope([1.0]), 0.0)

    def test_positive_slope(self):
        self.assertGreater(linear_slope([1.0, 2.0, 3.0]), 0.0)

    def test_negative_slope(self):
        self.assertLess(linear_slope([3.0, 2.0, 1.0]), 0.0)

    def test_flat(self):
        self.assertAlmostEqual(linear_slope([5.0, 5.0, 5.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
