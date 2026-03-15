"""Tests for trainer_setup — pipeline construction helpers.

Focuses on functions testable without full model loading:
build_pab_tracker, setup_run_dirs, load_vocabs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.trainer_setup import build_pab_tracker, setup_run_dirs

# ---------------------------------------------------------------------------
# build_pab_tracker (σ=8, SWAP + VALUE)
# ---------------------------------------------------------------------------


class TestBuildPabTracker(unittest.TestCase):
    def test_disabled_returns_none(self):
        config = {"pab": {"enabled": False}}
        self.assertIsNone(build_pab_tracker(config, "run_001"))

    def test_missing_pab_returns_none(self):
        self.assertIsNone(build_pab_tracker({}, "run_001"))

    def test_enabled_returns_tracker(self):
        config = {"pab": {"enabled": True, "checkpoint_interval": 25}}
        tracker = build_pab_tracker(config, "run_001")
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.experiment_id, "run_001")
        self.assertEqual(tracker.checkpoint_interval, 25)

    def test_config_hash_deterministic(self):
        config = {"pab": {"enabled": True}, "model": {"x": 1}}
        t1 = build_pab_tracker(config, "a")
        t2 = build_pab_tracker(config, "b")
        self.assertEqual(t1.config_hash, t2.config_hash)

    def test_config_hash_changes_with_config(self):
        c1 = {"pab": {"enabled": True}, "lr": 0.001}
        c2 = {"pab": {"enabled": True}, "lr": 0.01}
        t1 = build_pab_tracker(c1, "run")
        t2 = build_pab_tracker(c2, "run")
        self.assertNotEqual(t1.config_hash, t2.config_hash)

    def test_default_checkpoint_interval(self):
        config = {"pab": {"enabled": True}}
        tracker = build_pab_tracker(config, "run")
        self.assertEqual(tracker.checkpoint_interval, 50)

    def test_swap_run_id(self):
        """SWAP: different run_ids produce different experiment_ids."""
        config = {"pab": {"enabled": True}}
        t1 = build_pab_tracker(config, "run_A")
        t2 = build_pab_tracker(config, "run_B")
        self.assertNotEqual(t1.experiment_id, t2.experiment_id)


# ---------------------------------------------------------------------------
# setup_run_dirs (σ=4, VALUE)
# ---------------------------------------------------------------------------


class TestSetupRunDirs(unittest.TestCase):
    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "logging": {
                    "run_dir": "runs",
                    "checkpoint_dir": "checkpoints",
                }
            }
            with patch("src.trainer_setup.PROJECT_ROOT", Path(tmp)):
                run_dir, ckpt_dir = setup_run_dirs(config, "test_run")
                self.assertTrue(run_dir.exists())
                self.assertTrue(ckpt_dir.exists())
                self.assertEqual(run_dir.name, "test_run")

    def test_returns_correct_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "logging": {
                    "run_dir": "my_runs",
                    "checkpoint_dir": "my_ckpts",
                }
            }
            with patch("src.trainer_setup.PROJECT_ROOT", Path(tmp)):
                run_dir, ckpt_dir = setup_run_dirs(config, "exp_42")
                self.assertEqual(run_dir, Path(tmp) / "my_runs" / "exp_42")
                self.assertEqual(ckpt_dir, Path(tmp) / "my_ckpts")

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"logging": {"run_dir": "r", "checkpoint_dir": "c"}}
            with patch("src.trainer_setup.PROJECT_ROOT", Path(tmp)):
                setup_run_dirs(config, "x")
                setup_run_dirs(config, "x")  # should not raise


if __name__ == "__main__":
    unittest.main()
