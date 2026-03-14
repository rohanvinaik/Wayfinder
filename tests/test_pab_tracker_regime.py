"""PAB tracker tests — regime classification, convergence, early stop, early exit."""

import unittest

from src.pab_tracker import CheckpointData, PABTracker


class TestClassifyRegime(unittest.TestCase):
    """Test _classify_regime returns correct regime string for each branch."""

    def _tracker_with_stability(self, stab_values):
        """Build a tracker whose stability list matches *stab_values*."""
        tracker = PABTracker(experiment_id="REG")
        for i, _ in enumerate(stab_values):
            tracker.record(CheckpointData(step=(i + 1) * 50, train_loss=1.0))
        tracker._core.stability = list(stab_values)
        return tracker

    def test_empty_stability_returns_unknown(self):
        tracker = PABTracker(experiment_id="REG")
        self.assertEqual(tracker._classify_regime(), "unknown")

    def test_low_mean_returns_stable(self):
        tracker = self._tracker_with_stability([0.05] * 5)
        self.assertEqual(tracker._classify_regime(), "stable")

    def test_high_mean_returns_chaotic(self):
        tracker = self._tracker_with_stability([0.40] * 5)
        self.assertEqual(tracker._classify_regime(), "chaotic")

    def test_phase_transition(self):
        # 10 points: first half mean > 0.25, second half mean < 0.15
        stab = [0.30] * 5 + [0.10] * 5
        tracker = self._tracker_with_stability(stab)
        self.assertEqual(tracker._classify_regime(), "phase_transition")

    def test_moderate_middle_values(self):
        tracker = self._tracker_with_stability([0.20] * 5)
        self.assertEqual(tracker._classify_regime(), "moderate")

    def test_phase_transition_needs_10_points(self):
        # 8 points with transition pattern -- too few for phase_transition
        stab = [0.30] * 4 + [0.10] * 4
        tracker = self._tracker_with_stability(stab)
        # mean = 0.20 which is between 0.15 and 0.30 -> moderate
        self.assertEqual(tracker._classify_regime(), "moderate")

    def test_boundary_stable_at_015(self):
        # mean exactly 0.149... < 0.15 -> stable
        tracker = self._tracker_with_stability([0.14] * 5)
        self.assertEqual(tracker._classify_regime(), "stable")

    def test_boundary_chaotic_at_031(self):
        # mean > 0.30 -> chaotic
        tracker = self._tracker_with_stability([0.31] * 5)
        self.assertEqual(tracker._classify_regime(), "chaotic")

    def test_phase_transition_first_half_not_high_enough(self):
        # 10 points, first half mean 0.20 (not > 0.25) => not phase_transition
        stab = [0.20] * 5 + [0.10] * 5
        # mean = 0.15, which is between 0.15 and 0.30 -> moderate
        tracker = self._tracker_with_stability(stab)
        self.assertEqual(tracker._classify_regime(), "moderate")


class TestDetectConvergence(unittest.TestCase):
    """Test _detect_convergence returns correct checkpoint or None."""

    def _tracker_with_stability_and_steps(self, stab_values, steps):
        tracker = PABTracker(experiment_id="CONV")
        for i, s in enumerate(steps):
            tracker.record(CheckpointData(step=s, train_loss=1.0))
        tracker._core.stability = list(stab_values)
        return tracker

    def test_too_few_checkpoints_returns_none(self):
        tracker = self._tracker_with_stability_and_steps([0.05] * 3, [50, 100, 150])
        self.assertIsNone(tracker._detect_convergence())

    def test_no_convergent_window_returns_none(self):
        # 6 values all above threshold
        stab = [0.20, 0.15, 0.30, 0.25, 0.11, 0.20]
        steps = [50, 100, 150, 200, 250, 300]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertIsNone(tracker._detect_convergence())

    def test_convergence_at_start(self):
        stab = [0.05, 0.03, 0.04, 0.02, 0.01, 0.30]
        steps = [50, 100, 150, 200, 250, 300]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 50)

    def test_convergence_in_middle(self):
        stab = [0.50, 0.30, 0.05, 0.03, 0.04, 0.02, 0.01]
        steps = [50, 100, 150, 200, 250, 300, 350]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        # Window [0.05, 0.03, 0.04, 0.02, 0.01] starts at index 2 -> step 150
        self.assertEqual(tracker._detect_convergence(), 150)

    def test_exactly_five_converged(self):
        stab = [0.09, 0.08, 0.07, 0.06, 0.05]
        steps = [50, 100, 150, 200, 250]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 50)

    def test_value_at_threshold_not_converged(self):
        # Values exactly at 0.10 are NOT < 0.10, so no convergence
        stab = [0.10, 0.10, 0.10, 0.10, 0.10]
        steps = [50, 100, 150, 200, 250]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertIsNone(tracker._detect_convergence())


class TestDetectEarlyStop(unittest.TestCase):
    """Test _detect_early_stop returns correct checkpoint or None."""

    def _tracker_with_val_losses(self, val_losses, val_steps, all_steps):
        tracker = PABTracker(experiment_id="ES")
        for s in all_steps:
            tracker.record(CheckpointData(step=s, train_loss=1.0))
        tracker._core.val_losses = list(val_losses)
        tracker._core.val_checkpoints = list(val_steps)
        return tracker

    def test_no_val_losses_returns_none(self):
        tracker = PABTracker(experiment_id="ES")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        # Override to ensure no val_losses
        tracker._core.val_losses = []
        self.assertIsNone(tracker._detect_early_stop())

    def test_single_val_loss_returns_none(self):
        tracker = self._tracker_with_val_losses([1.0], [50], [50])
        self.assertIsNone(tracker._detect_early_stop())

    def test_always_decreasing_returns_none(self):
        tracker = self._tracker_with_val_losses(
            [2.0, 1.5, 1.0, 0.5],
            [50, 100, 150, 200],
            [50, 100, 150, 200],
        )
        self.assertIsNone(tracker._detect_early_stop())

    def test_increase_at_specific_point(self):
        tracker = self._tracker_with_val_losses(
            [2.0, 1.5, 1.8],  # increase at index 2
            [50, 100, 150],
            [50, 100, 150],
        )
        self.assertEqual(tracker._detect_early_stop(), 150)

    def test_increase_at_index_1(self):
        tracker = self._tracker_with_val_losses(
            [1.0, 1.5],
            [50, 100],
            [50, 100],
        )
        self.assertEqual(tracker._detect_early_stop(), 100)

    def test_first_increase_returned_not_later(self):
        tracker = self._tracker_with_val_losses(
            [2.0, 1.5, 1.8, 1.4, 1.9],
            [50, 100, 150, 200, 250],
            [50, 100, 150, 200, 250],
        )
        # First increase at index 2 (1.8 > 1.5)
        self.assertEqual(tracker._detect_early_stop(), 150)

    def test_val_checkpoint_index_out_of_range_fallback(self):
        # val_checkpoints shorter than val_losses -- exercises the fallback
        tracker = PABTracker(experiment_id="ES")
        for s in [50, 100, 150]:
            tracker.record(CheckpointData(step=s, train_loss=1.0))
        tracker._core.val_losses = [2.0, 1.5, 1.8]
        # Only 2 val_checkpoints, but 3 val_losses
        tracker._core.val_checkpoints = [50, 100]
        result = tracker._detect_early_stop()
        # Fallback: self._checkpoints[min(2, 2)] = self._checkpoints[2] = 150
        self.assertEqual(result, 150)


class TestShouldEarlyExit(unittest.TestCase):
    """Test should_early_exit all branches."""

    def _make_tracker_with_data(
        self,
        n_checkpoints,
        tier1_values,
        stability_values,
        predictability_values,
        step_start=50,
        step_interval=50,
    ):
        """Build a tracker with controlled internal state."""
        tracker = PABTracker(experiment_id="EXIT")
        steps = [(i + 1) * step_interval + step_start - step_interval for i in range(n_checkpoints)]
        for s in steps:
            tracker.record(CheckpointData(step=s, train_loss=1.0))
        # Overwrite accumulators with controlled values
        tracker._tier.tier1 = list(tier1_values)
        tracker._core.stability = list(stability_values)
        tracker._core.predictability = list(predictability_values)
        return tracker

    def test_step_below_200_returns_false(self):
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.50] * 10,
            stability_values=[0.50] * 10,
            predictability_values=[0.50] * 10,
        )
        self.assertEqual(tracker.should_early_exit(100), False)

    def test_too_few_checkpoints_returns_false(self):
        tracker = self._make_tracker_with_data(
            n_checkpoints=3,
            tier1_values=[0.50] * 3,
            stability_values=[0.50] * 3,
            predictability_values=[0.50] * 3,
        )
        self.assertEqual(tracker.should_early_exit(500), False)

    def test_chaotic_low_accuracy_returns_true(self):
        # tier1 < 0.60, stability > 0.30, pred > 0.10
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.55] * 10,
            stability_values=[0.35] * 10,
            predictability_values=[0.15] * 10,
        )
        self.assertEqual(tracker.should_early_exit(300), True)

    def test_improving_returns_false(self):
        # tier1 < 0.70, stability < 0.15, trend > 0.5
        # Monotonically increasing tier1 gives trend = 1.0
        tier1 = [0.50 + i * 0.02 for i in range(10)]  # 0.50..0.68, all < 0.70
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=tier1,
            stability_values=[0.10] * 10,
            predictability_values=[0.05] * 10,
        )
        self.assertEqual(tracker.should_early_exit(300), False)

    def test_stagnant_returns_true(self):
        # step > 400, trend < 0.05, tier1 < 0.75
        # Flat tier1 gives trend = 0.0
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.70] * 10,
            stability_values=[0.20] * 10,
            predictability_values=[0.05] * 10,
        )
        self.assertEqual(tracker.should_early_exit(500), True)

    def test_very_unstable_returns_true(self):
        # stability > 0.50 (not caught by earlier branches)
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.80] * 10,
            stability_values=[0.55] * 10,
            predictability_values=[0.05] * 10,
        )
        self.assertEqual(tracker.should_early_exit(300), True)

    def test_default_path_returns_false(self):
        # Don't trigger any early exit condition
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.80] * 10,
            stability_values=[0.25] * 10,
            predictability_values=[0.05] * 10,
        )
        self.assertEqual(tracker.should_early_exit(300), False)

    def test_step_exactly_200_not_early(self):
        # step=200 is NOT < 200, so it passes the first guard
        tracker = self._make_tracker_with_data(
            n_checkpoints=10,
            tier1_values=[0.80] * 10,
            stability_values=[0.55] * 10,
            predictability_values=[0.05] * 10,
        )
        self.assertEqual(tracker.should_early_exit(200), True)

    def test_exactly_4_checkpoints_passes_guard(self):
        # 5 checkpoints, recent_stability = mean([0.55]*5) = 0.55 > 0.50 -> True
        tracker = self._make_tracker_with_data(
            n_checkpoints=5,
            tier1_values=[0.80] * 5,
            stability_values=[0.55] * 5,
            predictability_values=[0.05] * 5,
        )
        self.assertEqual(tracker.should_early_exit(300), True)

    def test_exactly_4_checkpoints_passes_len_guard(self):
        # len(checkpoints) == 4, recent_stability = 0.0 (needs >= 5)
        # -> default False
        tracker = self._make_tracker_with_data(
            n_checkpoints=4,
            tier1_values=[0.50] * 4,
            stability_values=[0.55] * 4,
            predictability_values=[0.05] * 4,
        )
        self.assertEqual(tracker.should_early_exit(300), False)

    def test_empty_tier1_uses_zero(self):
        tracker = PABTracker(experiment_id="EXIT")
        for i in range(5):
            tracker.record(CheckpointData(step=(i + 1) * 50, train_loss=1.0))
        # tier1 is empty -> tier1_acc = 0.0
        tracker._tier.tier1 = []
        tracker._core.stability = [0.35] * 5
        tracker._core.predictability = [0.15] * 5
        # tier1_acc=0.0 < 0.60, stability=0.35 > 0.30, pred=0.15 > 0.10
        self.assertEqual(tracker.should_early_exit(300), True)

    def test_recent_stability_needs_5_values(self):
        # With fewer than 5 stability values, recent_stability = 0.0
        tracker = self._make_tracker_with_data(
            n_checkpoints=4,
            tier1_values=[0.50] * 4,
            stability_values=[0.90] * 4,  # high but only 4 values
            predictability_values=[0.15] * 4,
        )
        # recent_stability = 0.0, so no branch triggers -> default False
        self.assertEqual(tracker.should_early_exit(300), False)


class TestClassifyRegimeExactValues(unittest.TestCase):
    """Exact-value assertions for _classify_regime with known stability inputs."""

    def _tracker_with_stability(self, stab_values):
        tracker = PABTracker(experiment_id="REG-EXACT")
        for i, _ in enumerate(stab_values):
            tracker.record(CheckpointData(step=(i + 1) * 50, train_loss=1.0))
        tracker._core.stability = list(stab_values)
        return tracker

    def test_stable_exact_mean(self):
        """mean([0.05, 0.10, 0.12, 0.08, 0.03]) = 0.076 < 0.15 -> stable."""
        stab = [0.05, 0.10, 0.12, 0.08, 0.03]
        tracker = self._tracker_with_stability(stab)
        import numpy as np

        self.assertAlmostEqual(float(np.mean(stab)), 0.076, places=6)
        self.assertEqual(tracker._classify_regime(), "stable")

    def test_chaotic_exact_mean(self):
        """mean([0.31, 0.35, 0.40, 0.28, 0.32]) = 0.332 > 0.30 -> chaotic."""
        stab = [0.31, 0.35, 0.40, 0.28, 0.32]
        tracker = self._tracker_with_stability(stab)
        import numpy as np

        self.assertAlmostEqual(float(np.mean(stab)), 0.332, places=6)
        self.assertEqual(tracker._classify_regime(), "chaotic")

    def test_moderate_exact_mean(self):
        """mean([0.20, 0.22, 0.18, 0.25, 0.16]) = 0.202, in [0.15, 0.30] -> moderate."""
        stab = [0.20, 0.22, 0.18, 0.25, 0.16]
        tracker = self._tracker_with_stability(stab)
        import numpy as np

        self.assertAlmostEqual(float(np.mean(stab)), 0.202, places=6)
        self.assertEqual(tracker._classify_regime(), "moderate")

    def test_phase_transition_exact_halves(self):
        """First half mean=0.30, second half mean=0.08, overall=0.19 -> phase_transition."""
        stab = [0.28, 0.30, 0.32, 0.30, 0.30, 0.08, 0.08, 0.08, 0.08, 0.08]
        tracker = self._tracker_with_stability(stab)
        import numpy as np

        first_half = stab[:5]
        second_half = stab[5:]
        self.assertAlmostEqual(float(np.mean(first_half)), 0.30, places=6)
        self.assertAlmostEqual(float(np.mean(second_half)), 0.08, places=6)
        overall = float(np.mean(stab))
        self.assertTrue(0.15 <= overall <= 0.30)
        self.assertEqual(tracker._classify_regime(), "phase_transition")

    def test_single_stability_value_exact(self):
        """Single value 0.05 -> mean=0.05 < 0.15 -> stable."""
        tracker = self._tracker_with_stability([0.05])
        self.assertEqual(tracker._classify_regime(), "stable")

    def test_boundary_exactly_015_is_moderate(self):
        """mean=0.15 is NOT < 0.15, so not stable. Not > 0.30. -> moderate."""
        tracker = self._tracker_with_stability([0.15] * 6)
        self.assertEqual(tracker._classify_regime(), "moderate")

    def test_boundary_exactly_030_is_moderate(self):
        """mean=0.30 is NOT > 0.30, so not chaotic. Not < 0.15. -> moderate."""
        tracker = self._tracker_with_stability([0.30] * 6)
        self.assertEqual(tracker._classify_regime(), "moderate")


class TestDetectConvergenceExactValues(unittest.TestCase):
    """Exact-value assertions for _detect_convergence with known inputs."""

    def _tracker_with_stability_and_steps(self, stab_values, steps):
        tracker = PABTracker(experiment_id="CONV-EXACT")
        for i, s in enumerate(steps):
            tracker.record(CheckpointData(step=s, train_loss=1.0))
        tracker._core.stability = list(stab_values)
        return tracker

    def test_exactly_4_values_returns_none(self):
        """Window size is 5; 4 values is too few regardless of values."""
        stab = [0.01, 0.02, 0.01, 0.03]
        steps = [50, 100, 150, 200]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertIsNone(tracker._detect_convergence())

    def test_five_converged_returns_first_step(self):
        """All 5 below 0.10 -> convergence at step index 0 = step 100."""
        stab = [0.09, 0.08, 0.07, 0.06, 0.05]
        steps = [100, 200, 300, 400, 500]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 100)

    def test_convergence_delayed_by_one_high_value(self):
        """[0.20, 0.05, 0.03, 0.04, 0.02, 0.01] -> window starts at index 1 -> step 200."""
        stab = [0.20, 0.05, 0.03, 0.04, 0.02, 0.01]
        steps = [100, 200, 300, 400, 500, 600]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 200)

    def test_convergence_delayed_to_end(self):
        """High values then converge only at the last 5 -> step at index 3."""
        stab = [0.50, 0.40, 0.30, 0.05, 0.04, 0.03, 0.02, 0.01]
        steps = [50, 100, 150, 200, 250, 300, 350, 400]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 200)

    def test_single_value_at_threshold_breaks_window(self):
        """0.10 is NOT < 0.10 -- breaks the window at index 2."""
        stab = [0.05, 0.03, 0.10, 0.02, 0.01, 0.04, 0.03, 0.02, 0.01, 0.005]
        steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        # Index 0: [0.05, 0.03, 0.10, ...] -- 0.10 not < 0.10, fails
        # Index 1: [0.03, 0.10, ...] -- 0.10 not < 0.10, fails
        # Index 2: [0.10, ...] -- fails immediately
        # Index 3: [0.02, 0.01, 0.04, 0.03, 0.02] all < 0.10 -> step at index 3 = 200
        self.assertEqual(tracker._detect_convergence(), 200)

    def test_no_convergence_all_above(self):
        """All values >= 0.10 -> None."""
        stab = [0.15, 0.12, 0.11, 0.10, 0.13]
        steps = [50, 100, 150, 200, 250]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertIsNone(tracker._detect_convergence())

    def test_convergence_with_nonstandard_steps(self):
        """Non-uniform step spacing works correctly."""
        stab = [0.01, 0.02, 0.03, 0.01, 0.02]
        steps = [10, 25, 77, 200, 999]
        tracker = self._tracker_with_stability_and_steps(stab, steps)
        self.assertEqual(tracker._detect_convergence(), 10)


if __name__ == "__main__":
    unittest.main()
