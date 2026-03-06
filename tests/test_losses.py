"""Tests for composite loss and OOD loss."""

import unittest

import torch

from src.losses import CompositeLoss, OODLoss


class TestCompositeLoss(unittest.TestCase):
    def test_forward_returns_required_keys(self):
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        negatives = torch.randint(0, 10, (4,))
        repair_weights = torch.ones(4)

        result = loss_fn(logits, targets, negative_targets=negatives, repair_weights=repair_weights)
        for key in ["L_total", "L_ce", "L_margin", "L_repair", "w_ce", "w_margin", "w_repair"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_total_loss_is_finite(self):
        loss_fn = CompositeLoss()
        logits = torch.randn(8, 20)
        targets = torch.randint(0, 20, (8,))
        result = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(result["L_total"]))

    def test_adaptive_weights_are_positive(self):
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        self.assertGreater(result["w_ce"].item(), 0)
        self.assertGreater(result["w_margin"].item(), 0)
        self.assertGreater(result["w_repair"].item(), 0)


class TestOODLoss(unittest.TestCase):
    def test_forward_returns_scalar(self):
        loss_fn = OODLoss()
        logits = torch.randn(6, 1)
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
        loss = loss_fn(logits, labels)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
