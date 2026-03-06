"""Tests for PAB tracker -- accumulation and profile generation."""

import tempfile
import unittest
from pathlib import Path

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
        signs3 = np.array([1.0, 1.0, 0.0, -1.0])   # changed

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


if __name__ == "__main__":
    unittest.main()
