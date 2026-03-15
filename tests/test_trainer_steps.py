"""Tests for trainer_steps — pure functions and static methods.

Unlocks mutation profiling (DISCOVERY_FAILURE → testable).
Focuses on the 8 pure/static functions identified by spec_file_analyze.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.trainer_steps import TrainerStepsMixin, compute_batch_metrics

_UNK_TOKEN = "<UNK>"  # matches src/trainer_steps.py


# ---------------------------------------------------------------------------
# compute_batch_metrics (pure, σ=6, cacheable)
# ---------------------------------------------------------------------------


class TestComputeBatchMetrics(unittest.TestCase):
    """VALUE + SWAP for compute_batch_metrics."""

    def test_exact_perfect_accuracy(self):
        batch = [SimpleNamespace(goal_state="⊢ P", namespace="Algebra")]
        preds = np.array([0])
        targets = np.array([0])
        idx2token = {0: "ring"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertAlmostEqual(result["tier1_accuracy"], 1.0)

    def test_exact_zero_accuracy(self):
        batch = [SimpleNamespace(goal_state="⊢ P", namespace="Algebra")]
        preds = np.array([1])
        targets = np.array([0])
        idx2token = {0: "ring", 1: "simp"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertAlmostEqual(result["tier1_accuracy"], 0.0)

    def test_exact_half_accuracy(self):
        batch = [
            SimpleNamespace(goal_state="⊢ P", namespace="Algebra"),
            SimpleNamespace(goal_state="⊢ Q", namespace="Topology"),
        ]
        preds = np.array([0, 1])
        targets = np.array([0, 0])
        idx2token = {0: "ring", 1: "simp"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertAlmostEqual(result["tier1_accuracy"], 0.5)

    def test_domain_accuracies_populated(self):
        batch = [
            SimpleNamespace(goal_state="⊢ P", namespace="Algebra"),
            SimpleNamespace(goal_state="⊢ Q", namespace="Topology"),
        ]
        preds = np.array([0, 0])
        targets = np.array([0, 0])
        idx2token = {0: "ring"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertIn("domain_accuracies", result)
        self.assertTrue(len(result["domain_accuracies"]) > 0)

    def test_tactic_accuracies_populated(self):
        batch = [SimpleNamespace(goal_state="⊢ P", namespace="Algebra")]
        preds = np.array([0])
        targets = np.array([0])
        idx2token = {0: "ring"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertIn("tactic_accuracies", result)
        self.assertIn("ring", result["tactic_accuracies"])
        self.assertAlmostEqual(result["tactic_accuracies"]["ring"], 1.0)

    def test_unknown_token_mapped(self):
        batch = [SimpleNamespace(goal_state="⊢ P", namespace="Algebra")]
        preds = np.array([0])
        targets = np.array([99])  # not in idx2token
        idx2token = {0: "ring"}
        result = compute_batch_metrics(batch, preds, targets, idx2token)
        self.assertIn(_UNK_TOKEN, result["tactic_accuracies"])

    def test_empty_batch(self):
        result = compute_batch_metrics([], np.array([]), np.array([]), {})
        self.assertAlmostEqual(result["tier1_accuracy"], 0.0)

    def test_swap_predictions_targets(self):
        """SWAP: swapping preds and targets changes accuracy."""
        batch = [
            SimpleNamespace(goal_state="⊢ P", namespace="A"),
            SimpleNamespace(goal_state="⊢ Q", namespace="B"),
        ]
        preds = np.array([0, 1])
        targets = np.array([0, 0])
        idx2token = {0: "t0", 1: "t1"}
        r1 = compute_batch_metrics(batch, preds, targets, idx2token)
        r2 = compute_batch_metrics(batch, targets, preds, idx2token)
        # Both give 50% tier1 acc (symmetric), but tactic distribution differs
        self.assertEqual(set(r1["tactic_accuracies"].keys()), {"t0"})
        self.assertIn("t0", r2["tactic_accuracies"])


# ---------------------------------------------------------------------------
# _tier1_token (static, pure, σ=4)
# ---------------------------------------------------------------------------


class TestTier1Token(unittest.TestCase):
    """VALUE + BOUNDARY for _tier1_token."""

    def test_skips_bos(self):
        ex = SimpleNamespace(tier1_tokens=["BOS", "ring", "EOS"])
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), "ring")

    def test_single_token(self):
        ex = SimpleNamespace(tier1_tokens=["omega"])
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), "omega")

    def test_empty_returns_unk(self):
        ex = SimpleNamespace(tier1_tokens=[])
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), _UNK_TOKEN)

    def test_no_attribute_returns_unk(self):
        ex = SimpleNamespace()
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), _UNK_TOKEN)

    def test_boundary_two_tokens(self):
        """BOUNDARY: len > 1 triggers skip-BOS path."""
        ex = SimpleNamespace(tier1_tokens=["BOS", "simp"])
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), "simp")

    def test_boundary_one_token(self):
        """BOUNDARY: len == 1 returns first token directly."""
        ex = SimpleNamespace(tier1_tokens=["ring"])
        self.assertEqual(TrainerStepsMixin._tier1_token(ex), "ring")


if __name__ == "__main__":
    unittest.main()
