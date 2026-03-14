"""Tests for v3A censor additions: asymmetric loss and false-prune rate."""

import unittest

import torch

from src.censor import CensorNetwork, asymmetric_bce_loss, compute_false_prune_rate


class TestAsymmetricBceLoss(unittest.TestCase):
    def test_symmetric_matches_standard_bce(self):
        """With w_neg=1, w_pos=1, should approximate standard BCE."""
        preds = torch.tensor([[0.7], [0.3], [0.5]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])

        asym = asymmetric_bce_loss(preds, targets, w_neg=1.0, w_pos=1.0)
        standard = torch.nn.functional.binary_cross_entropy(preds, targets)
        self.assertAlmostEqual(asym.item(), standard.item(), places=4)

    def test_asymmetric_penalizes_missed_suppression_more(self):
        """Missed suppressions (FN) should cost more than false suppressions (FP)."""
        # FN: target=1 (failure), pred=0.1 (missed it)
        fn_pred = torch.tensor([[0.1]])
        fn_target = torch.tensor([[1.0]])

        # FP: target=0 (success), pred=0.9 (wrongly pruned)
        fp_pred = torch.tensor([[0.9]])
        fp_target = torch.tensor([[0.0]])

        fn_loss = asymmetric_bce_loss(fn_pred, fn_target, w_neg=2.0, w_pos=1.0)
        fp_loss = asymmetric_bce_loss(fp_pred, fp_target, w_neg=2.0, w_pos=1.0)

        # With w_neg=2.0, the missed suppression loss should be higher
        self.assertGreater(fn_loss.item(), fp_loss.item())

    def test_clamping_prevents_nan(self):
        """Extreme predictions should not produce NaN."""
        preds = torch.tensor([[0.0], [1.0]])
        targets = torch.tensor([[1.0], [0.0]])
        loss = asymmetric_bce_loss(preds, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_gradient_flows(self):
        preds = torch.tensor([[0.6]], requires_grad=True)
        targets = torch.tensor([[1.0]])
        loss = asymmetric_bce_loss(preds, targets, w_neg=2.0, w_pos=1.0)
        loss.backward()
        self.assertIsNotNone(preds.grad)


class TestComputeFalsePruneRate(unittest.TestCase):
    def test_no_false_prunes(self):
        """Valid tactics correctly kept → rate = 0."""
        preds = torch.tensor([0.1, 0.2, 0.8, 0.9])  # last two predicted as failures
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])  # first two valid, last two failures
        fpr = compute_false_prune_rate(preds, targets, threshold=0.5)
        self.assertEqual(fpr, 0.0)

    def test_all_valid_pruned(self):
        """All valid tactics incorrectly pruned → rate = 1."""
        preds = torch.tensor([0.9, 0.8])  # both predicted as failures
        targets = torch.tensor([0.0, 0.0])  # but both are valid
        fpr = compute_false_prune_rate(preds, targets, threshold=0.5)
        self.assertEqual(fpr, 1.0)

    def test_partial_false_prunes(self):
        """Some valid tactics pruned."""
        preds = torch.tensor([0.9, 0.3, 0.7, 0.1])
        targets = torch.tensor([0.0, 0.0, 0.0, 0.0])  # all valid
        fpr = compute_false_prune_rate(preds, targets, threshold=0.5)
        # 2 of 4 valid tactics pruned
        self.assertAlmostEqual(fpr, 0.5, places=4)

    def test_no_valid_tactics(self):
        """No valid tactics → rate = 0 (no false prunes possible)."""
        preds = torch.tensor([0.8, 0.9])
        targets = torch.tensor([1.0, 1.0])  # all failures
        fpr = compute_false_prune_rate(preds, targets, threshold=0.5)
        self.assertEqual(fpr, 0.0)

    def test_threshold_effect(self):
        """Higher threshold → fewer pruning decisions → lower false-prune rate."""
        preds = torch.tensor([0.6, 0.7, 0.8])
        targets = torch.tensor([0.0, 0.0, 0.0])  # all valid

        fpr_low = compute_false_prune_rate(preds, targets, threshold=0.5)
        fpr_high = compute_false_prune_rate(preds, targets, threshold=0.9)

        self.assertGreaterEqual(fpr_low, fpr_high)


class TestCensorNetworkIntegration(unittest.TestCase):
    def test_forward_shape(self):
        censor = CensorNetwork(goal_dim=32, tactic_dim=16, hidden_dim=32)
        goal = torch.randn(4, 32)
        tactic = torch.randn(4, 16)
        out = censor(goal, tactic)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    def test_predict_returns_censor_prediction(self):
        censor = CensorNetwork(goal_dim=32, tactic_dim=16, hidden_dim=32, threshold=0.5)
        goal = torch.randn(1, 32)
        tactic = torch.randn(1, 16)
        pred = censor.predict(goal, tactic)
        self.assertIsInstance(pred.failure_probability, float)
        self.assertIsInstance(pred.should_prune, bool)


if __name__ == "__main__":
    unittest.main()
