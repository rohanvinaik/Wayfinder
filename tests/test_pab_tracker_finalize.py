"""PAB tracker tests — domain data, finalize exact values, summary, edge cases."""

import unittest

import numpy as np

from src.pab_tracker import CheckpointData, PABTracker


class TestBuildDomainData(unittest.TestCase):
    """Test _build_domain_data classification through finalize()."""

    def test_empty_domains_returns_empty(self):
        tracker = PABTracker(experiment_id="DOM")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.domains.domain_progression, {})
        self.assertEqual(profile.domains.domain_classification, {})

    def test_domain_classification_populated(self):
        tracker = PABTracker(experiment_id="DOM")
        # Record enough checkpoints with domain data for classify_domain to work
        for i in range(6):
            tracker.record(CheckpointData(
                step=(i + 1) * 50, train_loss=2.0 - i * 0.2,
                domain_accuracies={"algebra": 0.85},
            ))
        profile = tracker.finalize()
        # "algebra" has 6 entries all at 0.85 >= 0.80
        self.assertEqual(profile.domains.domain_classification["algebra"], "early")

    def test_domain_classification_late(self):
        tracker = PABTracker(experiment_id="DOM")
        # First entries below 0.80, last entry at 0.85
        accs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.85]
        for i, a in enumerate(accs):
            tracker.record(CheckpointData(
                step=(i + 1) * 50, train_loss=2.0 - i * 0.2,
                domain_accuracies={"topology": a},
            ))
        profile = tracker.finalize()
        # third = 6//3 = 2, accs[:2] = [0.50, 0.55] -> not reached_early
        # reached_ever = True (0.85 >= 0.80)
        # not oscillating (monotonic) -> "late"
        self.assertEqual(profile.domains.domain_classification["topology"], "late")

    def test_tactic_progression_included_when_domains_present(self):
        tracker = PABTracker(experiment_id="DOM")
        for i in range(4):
            tracker.record(CheckpointData(
                step=(i + 1) * 50, train_loss=1.0,
                domain_accuracies={"algebra": 0.70},
                tactic_accuracies={"simp": 0.80},
            ))
        profile = tracker.finalize()
        self.assertEqual(
            profile.domains.tactic_progression["simp"], [0.80] * 4
        )


class TestFinalizeExactValues(unittest.TestCase):
    """Test finalize() returns exact expected values from known inputs."""

    def test_three_checkpoints_exact_profile(self):
        tracker = PABTracker(experiment_id="EXACT", config_hash="abc123")
        tracker.record(CheckpointData(
            step=50, train_loss=2.0, val_loss=2.2,
            tier_accuracies={"tier1": 0.40, "tier2": 0.30, "tier3": 0.20},
            loss_components={"ce": 1.5, "margin": 0.3, "repair": 0.2},
        ))
        tracker.record(CheckpointData(
            step=100, train_loss=1.0, val_loss=1.3,
            tier_accuracies={"tier1": 0.60, "tier2": 0.50, "tier3": 0.35},
            loss_components={"ce": 0.7, "margin": 0.2, "repair": 0.1},
        ))
        tracker.record(CheckpointData(
            step=150, train_loss=0.5, val_loss=0.8,
            tier_accuracies={"tier1": 0.75, "tier2": 0.65, "tier3": 0.50},
            loss_components={"ce": 0.3, "margin": 0.1, "repair": 0.1},
        ))
        profile = tracker.finalize()

        self.assertEqual(profile.experiment_id, "EXACT")
        self.assertEqual(profile.config_hash, "abc123")
        self.assertEqual(profile.checkpoints, [50, 100, 150])

        # Core series lengths
        self.assertEqual(len(profile.core.stability), 3)
        self.assertEqual(len(profile.core.predictability), 3)
        self.assertEqual(len(profile.core.generalization_gap), 3)
        self.assertEqual(len(profile.core.representation_evolution), 3)

        # First stability = 0.0 (only 1 loss)
        self.assertEqual(profile.core.stability[0], 0.0)

        # Second stability = |2.0 - 1.0| / (|2.0| + eps) = 0.5
        self.assertAlmostEqual(profile.core.stability[1], 0.5, places=5)

        # Third stability = |1.0 - 0.5| / (|1.0| + eps) = 0.5
        self.assertAlmostEqual(profile.core.stability[2], 0.5, places=5)

        # Gen gap = val_loss - train_loss
        self.assertAlmostEqual(profile.core.generalization_gap[0], 0.2, places=5)
        self.assertAlmostEqual(profile.core.generalization_gap[1], 0.3, places=5)
        self.assertAlmostEqual(profile.core.generalization_gap[2], 0.3, places=5)

        # Tier accuracies
        self.assertEqual(profile.tiers.tier1_accuracy, [0.40, 0.60, 0.75])
        self.assertEqual(profile.tiers.tier2_accuracy, [0.30, 0.50, 0.65])
        self.assertEqual(profile.tiers.tier3_accuracy, [0.20, 0.35, 0.50])

        # Loss components
        self.assertEqual(profile.losses.loss_ce, [1.5, 0.7, 0.3])
        self.assertEqual(profile.losses.loss_margin, [0.3, 0.2, 0.1])
        self.assertEqual(profile.losses.loss_repair, [0.2, 0.1, 0.1])

    def test_summary_stability_mean_and_std(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        tracker.record(CheckpointData(step=150, train_loss=0.5))
        profile = tracker.finalize()

        # stability = [0.0, 0.5, 0.5]
        expected_mean = float(np.mean([0.0, 0.5, 0.5]))
        expected_std = float(np.std([0.0, 0.5, 0.5]))
        self.assertAlmostEqual(
            profile.summary.stability_mean, expected_mean, places=5
        )
        self.assertAlmostEqual(
            profile.summary.stability_std, expected_std, places=5
        )

    def test_summary_predictability_final(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        tracker.record(CheckpointData(step=150, train_loss=0.5))
        profile = tracker.finalize()
        # predictability[-1] = var of loss deltas at step 3
        # deltas = [-1.0, -0.5], var = 0.0625
        expected = float(np.var([-1.0, -0.5]))
        self.assertAlmostEqual(
            profile.summary.predictability_final, expected, places=7
        )

    def test_summary_regime(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        tracker.record(CheckpointData(step=150, train_loss=0.5))
        profile = tracker.finalize()
        # stability = [0.0, 0.5, 0.5], mean ~ 0.333 > 0.30 -> "chaotic"
        self.assertEqual(profile.summary.stability_regime, "chaotic")

    def test_summary_convergence_none_too_few(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        profile = tracker.finalize()
        self.assertIsNone(profile.summary.convergence_epoch)

    def test_summary_early_stop_detected(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=2.0, val_loss=2.2))
        tracker.record(CheckpointData(step=100, train_loss=1.0, val_loss=1.3))
        tracker.record(CheckpointData(step=150, train_loss=0.5, val_loss=1.5))
        profile = tracker.finalize()
        # val_losses = [2.2, 1.3, 1.5], increase at index 2 -> step 150
        self.assertEqual(profile.summary.early_stop_epoch, 150)

    def test_summary_no_early_stop(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(step=50, train_loss=2.0, val_loss=2.2))
        tracker.record(CheckpointData(step=100, train_loss=1.0, val_loss=1.3))
        tracker.record(CheckpointData(step=150, train_loss=0.5, val_loss=0.8))
        profile = tracker.finalize()
        self.assertIsNone(profile.summary.early_stop_epoch)

    def test_summary_tier_convergence_steps(self):
        tracker = PABTracker(experiment_id="SUM")
        tracker.record(CheckpointData(
            step=50, train_loss=2.0,
            tier_accuracies={"tier1": 0.50, "tier2": 0.60, "tier3": 0.30},
        ))
        tracker.record(CheckpointData(
            step=100, train_loss=1.5,
            tier_accuracies={"tier1": 0.70, "tier2": 0.75, "tier3": 0.45},
        ))
        tracker.record(CheckpointData(
            step=150, train_loss=1.0,
            tier_accuracies={"tier1": 0.85, "tier2": 0.80, "tier3": 0.60},
        ))
        profile = tracker.finalize()
        self.assertEqual(profile.summary.tier1_convergence_step, 150)
        self.assertEqual(profile.summary.tier2_convergence_step, 100)

    def test_summary_tier_convergence_never_reached(self):
        tracker = PABTracker(experiment_id="SUM")
        for i in range(3):
            tracker.record(CheckpointData(
                step=(i + 1) * 50, train_loss=2.0,
                tier_accuracies={"tier1": 0.50, "tier2": 0.40, "tier3": 0.30},
            ))
        profile = tracker.finalize()
        self.assertIsNone(profile.summary.tier1_convergence_step)
        self.assertIsNone(profile.summary.tier2_convergence_step)


class TestComputeSummary(unittest.TestCase):
    """Test _compute_summary edge cases through finalize()."""

    def test_empty_tracker_summary_defaults(self):
        tracker = PABTracker(experiment_id="EMPTY")
        profile = tracker.finalize()
        self.assertEqual(profile.summary.stability_mean, 0.0)
        self.assertEqual(profile.summary.stability_std, 0.0)
        self.assertEqual(profile.summary.predictability_final, 0.0)
        self.assertIsNone(profile.summary.convergence_epoch)
        self.assertIsNone(profile.summary.early_stop_epoch)
        self.assertEqual(profile.summary.stability_regime, "unknown")
        self.assertIsNone(profile.summary.tier1_convergence_step)
        self.assertIsNone(profile.summary.tier2_convergence_step)
        self.assertEqual(profile.summary.crystallization_rate, 0.0)
        self.assertEqual(profile.summary.feature_importance_L, 0.0)

    def test_crystallization_rate_computed(self):
        tracker = PABTracker(experiment_id="CR")
        signs1 = np.array([1.0, -1.0, 0.0, 1.0])
        signs2 = np.array([1.0, -1.0, 0.0, 1.0])
        signs3 = np.array([1.0, -1.0, 0.0, 1.0])
        tracker.record(CheckpointData(
            step=50, train_loss=2.0, decoder_weight_signs=signs1
        ))
        tracker.record(CheckpointData(
            step=100, train_loss=1.5, decoder_weight_signs=signs2
        ))
        tracker.record(CheckpointData(
            step=150, train_loss=1.0, decoder_weight_signs=signs3
        ))
        profile = tracker.finalize()
        self.assertGreater(profile.summary.crystallization_rate, 0.0)

    def test_feature_importance_with_snapshots(self):
        tracker = PABTracker(experiment_id="FI")
        signs1 = np.array([1.0, -1.0, 0.0])
        signs2 = np.array([1.0, 1.0, 0.0])
        tracker.record(CheckpointData(
            step=50, train_loss=2.0, decoder_weight_signs=signs1
        ))
        tracker.record(CheckpointData(
            step=100, train_loss=1.5, decoder_weight_signs=signs2
        ))
        profile = tracker.finalize()
        # Two weight snapshots -> feature_importance_L > 0 (signs differ)
        self.assertGreater(profile.summary.feature_importance_L, 0.0)

    def test_feature_importance_identical_snapshots(self):
        tracker = PABTracker(experiment_id="FI")
        signs = np.array([1.0, -1.0, 0.0])
        tracker.record(CheckpointData(
            step=50, train_loss=2.0, decoder_weight_signs=signs
        ))
        tracker.record(CheckpointData(
            step=100, train_loss=1.5, decoder_weight_signs=signs.copy()
        ))
        profile = tracker.finalize()
        # Identical snapshots -> feature_importance_L = 0.0
        self.assertAlmostEqual(
            profile.summary.feature_importance_L, 0.0, places=7
        )


class TestEdgeCases(unittest.TestCase):
    """Miscellaneous edge cases."""

    def test_record_many_checkpoints(self):
        tracker = PABTracker(experiment_id="MANY")
        for i in range(100):
            tracker.record(CheckpointData(
                step=(i + 1) * 50,
                train_loss=2.0 * (0.99 ** i),
            ))
        profile = tracker.finalize()
        self.assertEqual(len(profile.checkpoints), 100)
        self.assertEqual(len(profile.core.stability), 100)

    def test_val_loss_only_some_checkpoints(self):
        tracker = PABTracker(experiment_id="SPARSE")
        tracker.record(CheckpointData(step=50, train_loss=2.0, val_loss=2.2))
        tracker.record(CheckpointData(step=100, train_loss=1.5))
        tracker.record(CheckpointData(step=150, train_loss=1.0, val_loss=1.1))
        profile = tracker.finalize()
        # gen_gap: [0.2, 0.2 (carry-forward), 0.1]
        self.assertAlmostEqual(
            profile.core.generalization_gap[0], 0.2, places=5
        )
        self.assertAlmostEqual(
            profile.core.generalization_gap[1], 0.2, places=5
        )
        self.assertAlmostEqual(
            profile.core.generalization_gap[2], 0.1, places=5
        )

    def test_partial_tier_accuracies(self):
        # Only tier1 provided, tier2/tier3 should default to 0.0
        tracker = PABTracker(experiment_id="PART")
        tracker.record(CheckpointData(
            step=50, train_loss=1.0,
            tier_accuracies={"tier1": 0.80},
        ))
        profile = tracker.finalize()
        self.assertEqual(profile.tiers.tier1_accuracy[0], 0.80)
        self.assertEqual(profile.tiers.tier2_accuracy[0], 0.0)
        self.assertEqual(profile.tiers.tier3_accuracy[0], 0.0)

    def test_partial_loss_components(self):
        # Only "ce" provided, margin/repair should default to 0.0
        tracker = PABTracker(experiment_id="PART")
        tracker.record(CheckpointData(
            step=50, train_loss=1.0,
            loss_components={"ce": 0.9},
        ))
        profile = tracker.finalize()
        self.assertEqual(profile.losses.loss_ce[0], 0.9)
        self.assertEqual(profile.losses.loss_margin[0], 0.0)
        self.assertEqual(profile.losses.loss_repair[0], 0.0)

    def test_stability_same_loss_values(self):
        tracker = PABTracker(experiment_id="SAME")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        profile = tracker.finalize()
        # |1.0 - 1.0| / (|1.0| + eps) = 0.0
        self.assertAlmostEqual(profile.core.stability[1], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
