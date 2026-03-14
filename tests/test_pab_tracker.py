"""Tests for PAB tracker -- accumulation and profile generation."""

import tempfile
import unittest

import numpy as np

from src.pab_tracker import CheckpointData, PABTracker


class TestPABTracker(unittest.TestCase):
    def _make_tracker(self) -> PABTracker:
        return PABTracker(experiment_id="TEST-1", checkpoint_interval=50)

    def test_record_single_checkpoint(self):
        tracker = self._make_tracker()
        data = CheckpointData(step=50, train_loss=2.0)
        tracker.record(data)
        self.assertEqual(len(tracker._checkpoints), 1)

    def test_finalize_produces_profile(self):
        tracker = self._make_tracker()
        for i in range(5):
            data = CheckpointData(
                step=(i + 1) * 50,
                train_loss=2.0 - i * 0.3,
                val_loss=2.1 - i * 0.25,
                tier_accuracies={"tier1": 0.5 + i * 0.1, "tier2": 0.4 + i * 0.05, "tier3": 0.3},
            )
            tracker.record(data)

        profile = tracker.finalize()
        self.assertEqual(profile.experiment_id, "TEST-1")
        self.assertEqual(len(profile.checkpoints), 5)
        self.assertEqual(len(profile.core.stability), 5)
        self.assertEqual(len(profile.tiers.tier1_accuracy), 5)

    def test_profile_save_load_roundtrip(self):
        tracker = self._make_tracker()
        for i in range(3):
            tracker.record(CheckpointData(step=(i + 1) * 50, train_loss=1.5 - i * 0.2))
        profile = tracker.finalize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            profile.save(f.name)
            from src.pab_profile import PABProfile

            loaded = PABProfile.load(f.name)
            self.assertEqual(loaded.experiment_id, "TEST-1")
            self.assertEqual(len(loaded.checkpoints), 3)

    def test_crystallization_tracking(self):
        tracker = self._make_tracker()
        signs1 = np.array([1.0, -1.0, 0.0, 1.0])
        signs2 = np.array([1.0, -1.0, 0.0, 1.0])  # same
        signs3 = np.array([1.0, 1.0, 0.0, -1.0])  # changed

        tracker.record(CheckpointData(step=50, train_loss=2.0, decoder_weight_signs=signs1))
        tracker.record(CheckpointData(step=100, train_loss=1.5, decoder_weight_signs=signs2))
        tracker.record(CheckpointData(step=150, train_loss=1.0, decoder_weight_signs=signs3))

        profile = tracker.finalize()
        self.assertEqual(len(profile.tiers.ternary_crystallization), 3)
        # Second checkpoint should have high crystallization (same signs)
        self.assertGreater(profile.tiers.ternary_crystallization[1], 0.9)
        # Third should have lower (signs changed)
        self.assertLess(profile.tiers.ternary_crystallization[2], 0.9)

    def test_should_early_exit_false_early(self):
        tracker = self._make_tracker()
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        self.assertFalse(tracker.should_early_exit(50))

    # --- P0 spec-gap closures below ---

    def test_make_tracker_initial_state(self):
        """Verify every field of a freshly constructed tracker has the expected value."""
        tracker = self._make_tracker()
        self.assertEqual(tracker.experiment_id, "TEST-1")
        self.assertEqual(tracker.config_hash, "")
        self.assertEqual(tracker.checkpoint_interval, 50)
        self.assertEqual(tracker._checkpoints, [])
        # Core accumulator defaults
        self.assertEqual(tracker._core.train_losses, [])
        self.assertEqual(tracker._core.val_losses, [])
        self.assertEqual(tracker._core.val_checkpoints, [])
        self.assertEqual(tracker._core.stability, [])
        self.assertEqual(tracker._core.predictability, [])
        self.assertEqual(tracker._core.gen_gap, [])
        self.assertEqual(tracker._core.repr_evolution, [])
        self.assertIsNone(tracker._core.prev_bottleneck_mean)
        # Tier accumulator defaults
        self.assertEqual(tracker._tier.tier1, [])
        self.assertEqual(tracker._tier.tier2, [])
        self.assertEqual(tracker._tier.tier3, [])
        self.assertEqual(tracker._tier.crystallization, [])
        self.assertIsNone(tracker._tier.prev_weight_signs)
        self.assertEqual(tracker._tier.weight_snapshots, [])
        # Aux accumulator defaults
        self.assertEqual(tracker._aux.domain_acc, {})
        self.assertEqual(tracker._aux.tactic_acc, {})
        self.assertEqual(tracker._aux.loss_ce, [])
        self.assertEqual(tracker._aux.loss_margin, [])
        self.assertEqual(tracker._aux.loss_repair, [])
        self.assertEqual(tracker._aux.loss_weights, [])

    def test_make_tracker_with_config_hash(self):
        """Verify config_hash is stored when provided."""
        tracker = PABTracker(experiment_id="T2", config_hash="abc123", checkpoint_interval=100)
        self.assertEqual(tracker.experiment_id, "T2")
        self.assertEqual(tracker.config_hash, "abc123")
        self.assertEqual(tracker.checkpoint_interval, 100)
        self.assertEqual(tracker._checkpoints, [])

    def test_profile_save_load_roundtrip_exact_values(self):
        """Verify every serialized field survives save/load with exact values."""
        tracker = self._make_tracker()
        # Record 3 checkpoints with known losses
        tracker.record(CheckpointData(step=50, train_loss=1.5))
        tracker.record(CheckpointData(step=100, train_loss=1.3))
        tracker.record(CheckpointData(step=150, train_loss=1.1))
        profile = tracker.finalize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            profile.save(f.name)
            from src.pab_profile import PABProfile

            loaded = PABProfile.load(f.name)

        # Top-level
        self.assertEqual(loaded.experiment_id, "TEST-1")
        self.assertEqual(loaded.config_hash, "")
        self.assertEqual(loaded.checkpoints, [50, 100, 150])
        # Core series lengths
        self.assertEqual(len(loaded.core.stability), 3)
        self.assertEqual(len(loaded.core.predictability), 3)
        self.assertEqual(len(loaded.core.generalization_gap), 3)
        self.assertEqual(len(loaded.core.representation_evolution), 3)
        # First stability = 0.0 (only 1 loss so far at record time)
        self.assertAlmostEqual(loaded.core.stability[0], 0.0)
        # Second stability: |1.5 - 1.3| / (|1.5| + 1e-8) = 0.2 / 1.5
        self.assertAlmostEqual(loaded.core.stability[1], 0.2 / 1.5, places=6)
        # Third stability: |1.3 - 1.1| / (|1.3| + 1e-8) = 0.2 / 1.3
        self.assertAlmostEqual(loaded.core.stability[2], 0.2 / 1.3, places=6)
        # Predictability: at step 1 only 1 loss -> 0.0; at step 2 only 1 delta -> 0.0
        # at step 3: deltas = [1.3-1.5, 1.1-1.3] = [-0.2, -0.2], var=0.0
        self.assertAlmostEqual(loaded.core.predictability[0], 0.0)
        self.assertAlmostEqual(loaded.core.predictability[1], 0.0)
        self.assertAlmostEqual(loaded.core.predictability[2], 0.0)
        # Generalization gap: no val_loss -> all 0.0
        self.assertEqual(loaded.core.generalization_gap, [0.0, 0.0, 0.0])
        # Repr evolution: no bottleneck -> all 0.0
        self.assertEqual(loaded.core.representation_evolution, [0.0, 0.0, 0.0])
        # Tiers: no tier_accuracies -> all 0.0
        self.assertEqual(loaded.tiers.tier1_accuracy, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.tiers.tier2_accuracy, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.tiers.tier3_accuracy, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.tiers.ternary_crystallization, [0.0, 0.0, 0.0])
        # Losses: no components -> all 0.0
        self.assertEqual(loaded.losses.loss_ce, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.losses.loss_margin, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.losses.loss_repair, [0.0, 0.0, 0.0])
        self.assertEqual(loaded.losses.loss_adaptive_weights, [{}, {}, {}])
        # Domains: empty
        self.assertEqual(loaded.domains.domain_progression, {})
        self.assertEqual(loaded.domains.domain_classification, {})
        self.assertEqual(loaded.domains.tactic_progression, {})
        # Summary exact values
        # stability = [0.0, 0.2/1.5, 0.2/1.3]; mean ~ 0.096 < 0.15 -> "stable"
        self.assertEqual(loaded.summary.stability_regime, "stable")
        self.assertIsNone(loaded.summary.early_stop_epoch)
        self.assertIsNone(loaded.summary.convergence_epoch)
        self.assertIsNone(loaded.summary.tier1_convergence_step)
        self.assertIsNone(loaded.summary.tier2_convergence_step)
        self.assertAlmostEqual(loaded.summary.crystallization_rate, 0.0)
        self.assertAlmostEqual(loaded.summary.feature_importance_L, 0.0)

    def test_record_single_checkpoint_exact_accumulators(self):
        """After recording one checkpoint, verify every accumulator value."""
        tracker = self._make_tracker()
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        # Checkpoints
        self.assertEqual(tracker._checkpoints, [50])
        # Core
        self.assertEqual(tracker._core.train_losses, [2.0])
        self.assertEqual(tracker._core.stability, [0.0])
        self.assertEqual(tracker._core.predictability, [0.0])
        self.assertEqual(tracker._core.gen_gap, [0.0])
        self.assertEqual(tracker._core.repr_evolution, [0.0])
        self.assertEqual(tracker._core.val_losses, [])
        self.assertEqual(tracker._core.val_checkpoints, [])
        # Tier
        self.assertEqual(tracker._tier.tier1, [0.0])
        self.assertEqual(tracker._tier.tier2, [0.0])
        self.assertEqual(tracker._tier.tier3, [0.0])
        self.assertEqual(tracker._tier.crystallization, [0.0])
        # Aux
        self.assertEqual(tracker._aux.loss_ce, [0.0])
        self.assertEqual(tracker._aux.loss_margin, [0.0])
        self.assertEqual(tracker._aux.loss_repair, [0.0])
        self.assertEqual(tracker._aux.loss_weights, [{}])
        self.assertEqual(tracker._aux.domain_acc, {})
        self.assertEqual(tracker._aux.tactic_acc, {})

    def test_record_two_checkpoints_stability_exact(self):
        """Verify exact stability computation after two records."""
        tracker = self._make_tracker()
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.5))
        # stability[0] = 0.0 (first checkpoint)
        self.assertAlmostEqual(tracker._core.stability[0], 0.0)
        # stability[1] = |2.0 - 1.5| / (|2.0| + 1e-8) = 0.5 / 2.0 = 0.25
        self.assertAlmostEqual(tracker._core.stability[1], 0.25, places=6)

    def test_record_with_val_loss_exact_gen_gap(self):
        """Verify generalization gap with known train/val losses."""
        tracker = self._make_tracker()
        tracker.record(CheckpointData(step=50, train_loss=1.0, val_loss=1.3))
        # gen_gap = val_loss - train_loss = 1.3 - 1.0 = 0.3
        self.assertAlmostEqual(tracker._core.gen_gap[0], 0.3, places=6)
        self.assertEqual(tracker._core.val_losses, [1.3])
        self.assertEqual(tracker._core.val_checkpoints, [50])

    def test_record_with_tier_accuracies_exact(self):
        """Verify tier values stored exactly."""
        tracker = self._make_tracker()
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                tier_accuracies={"tier1": 0.85, "tier2": 0.72, "tier3": 0.41},
            )
        )
        self.assertEqual(tracker._tier.tier1, [0.85])
        self.assertEqual(tracker._tier.tier2, [0.72])
        self.assertEqual(tracker._tier.tier3, [0.41])

    def test_record_with_loss_components_exact(self):
        """Verify loss component accumulation."""
        tracker = self._make_tracker()
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                loss_components={"ce": 0.6, "margin": 0.3, "repair": 0.1},
                adaptive_weights={"ce": 0.5, "margin": 0.3, "repair": 0.2},
            )
        )
        self.assertEqual(tracker._aux.loss_ce, [0.6])
        self.assertEqual(tracker._aux.loss_margin, [0.3])
        self.assertEqual(tracker._aux.loss_repair, [0.1])
        self.assertEqual(tracker._aux.loss_weights, [{"ce": 0.5, "margin": 0.3, "repair": 0.2}])


if __name__ == "__main__":
    unittest.main()
