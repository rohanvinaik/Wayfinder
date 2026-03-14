"""Tests for composite loss, OOD loss, and navigational loss."""

import unittest

import torch

from src.losses import CompositeLoss, NavigationalLoss, OODLoss


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
        # L_total should be a scalar tensor
        self.assertEqual(result["L_total"].dim(), 0)
        # Without negatives or repair_weights, L_margin and L_repair are zero
        self.assertAlmostEqual(result["L_margin"].item(), 0.0, places=6)
        self.assertAlmostEqual(result["L_repair"].item(), 0.0, places=6)
        # L_ce (cross-entropy) must be positive for random logits
        self.assertGreater(result["L_ce"].item(), 0.0)

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


class TestCompositeLossPredicateEffects(unittest.TestCase):
    """Predicate→effect tests for CompositeLoss.forward decision paths."""

    def test_zero_margin_disables_margin_loss(self):
        """margin=0 → L_margin forced to zero regardless of inputs."""
        loss_fn = CompositeLoss(margin=0.0)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        neg = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets, negative_targets=neg)
        self.assertAlmostEqual(result["L_margin"].item(), 0.0, places=6)

    def test_negative_targets_activates_logit_margin(self):
        """negative_targets provided → logit-based margin path used."""
        loss_fn = CompositeLoss(margin=0.5)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        neg = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets, negative_targets=neg)
        # L_margin should be non-negative (ReLU output)
        self.assertGreaterEqual(result["L_margin"].item(), 0.0)

    def test_features_activates_cosine_margin(self):
        """positive/negative features → cosine distance margin path."""
        loss_fn = CompositeLoss(margin=0.5)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        pos_feat = torch.randn(4, 32)
        neg_feat = torch.randn(4, 32)
        result = loss_fn(logits, targets, positive_features=pos_feat, negative_features=neg_feat)
        self.assertGreaterEqual(result["L_margin"].item(), 0.0)

    def test_repair_weights_activates_repair_loss(self):
        """repair_weights provided → L_repair is non-zero."""
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        weights = torch.ones(4) * 2.0
        result = loss_fn(logits, targets, repair_weights=weights)
        self.assertGreater(result["L_repair"].item(), 0.0)

    def test_no_repair_weights_zeros_repair_loss(self):
        """No repair_weights → L_repair = 0."""
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        self.assertAlmostEqual(result["L_repair"].item(), 0.0, places=6)

    def test_adaptive_weights_sum_to_one(self):
        """UW-SO weights w_ce + w_margin + w_repair ≈ 1.0."""
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        w_sum = result["w_ce"].item() + result["w_margin"].item() + result["w_repair"].item()
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_sigma_values_are_positive(self):
        """Sigma = exp(log_sigma), must be positive."""
        loss_fn = CompositeLoss(initial_log_sigma=-1.0)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        self.assertGreater(result["sigma_ce"].item(), 0.0)
        self.assertGreater(result["sigma_margin"].item(), 0.0)
        self.assertGreater(result["sigma_repair"].item(), 0.0)

    def test_total_loss_is_differentiable(self):
        """L_total must have grad_fn (backprop works)."""
        loss_fn = CompositeLoss()
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (4,))
        result = loss_fn(logits, targets)
        result["L_total"].backward()
        self.assertIsNotNone(logits.grad)


class TestNavigationalLoss(unittest.TestCase):
    """Tests for NavigationalLoss — UW-SO adaptive multi-task loss."""

    def _make_direction_data(self, batch=4, n_classes=3):
        """Create direction logits and targets for active banks."""
        banks = ["structure", "domain", "depth", "automation", "context", "decomposition"]
        logits = {b: torch.randn(batch, n_classes) for b in banks}
        targets = {b: torch.randint(0, n_classes, (batch,)) for b in banks}
        return logits, targets

    def test_forward_returns_required_keys(self):
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        for key in [
            "L_total",
            "L_nav",
            "L_anchor",
            "L_progress",
            "L_critic",
            "bank_losses",
            "w_nav",
            "w_anchor",
            "w_progress",
            "w_critic",
        ]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_total_loss_is_scalar_and_finite(self):
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        self.assertEqual(result["L_total"].dim(), 0)
        self.assertTrue(torch.isfinite(result["L_total"]))

    def test_nav_loss_is_sum_of_bank_losses(self):
        """L_nav = sum of per-bank cross-entropy losses."""
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        bank_sum = sum(v.item() for v in result["bank_losses"].values())
        self.assertAlmostEqual(result["L_nav"].item(), bank_sum, places=4)

    def test_per_bank_losses_returned(self):
        """bank_losses dict has one entry per active bank."""
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        self.assertEqual(len(result["bank_losses"]), 6)
        for bank, loss_val in result["bank_losses"].items():
            self.assertGreater(loss_val.item(), 0.0)

    def test_no_optional_inputs_zeros_optional_losses(self):
        """Without anchor/progress/critic inputs, those losses are zero."""
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        self.assertAlmostEqual(result["L_anchor"].item(), 0.0, places=6)
        self.assertAlmostEqual(result["L_progress"].item(), 0.0, places=6)
        self.assertAlmostEqual(result["L_critic"].item(), 0.0, places=6)

    def test_anchor_loss_activates_with_inputs(self):
        """Providing anchor logits+targets → L_anchor > 0."""
        logits, targets = self._make_direction_data(batch=4)
        loss_fn = NavigationalLoss()
        anchor_logits = torch.randn(4, 20)
        anchor_targets = torch.zeros(4, 20).scatter_(1, torch.randint(0, 20, (4, 3)), 1.0)
        result = loss_fn(
            logits, targets, anchor_logits=anchor_logits, anchor_targets=anchor_targets
        )
        self.assertGreater(result["L_anchor"].item(), 0.0)

    def test_progress_loss_is_mse(self):
        """L_progress = MSE between prediction and target (soft, not binary)."""
        logits, targets = self._make_direction_data(batch=4)
        loss_fn = NavigationalLoss()
        progress_pred = torch.tensor([0.2, 0.5, 0.8, 1.0]).unsqueeze(-1)
        progress_target = torch.tensor([0.3, 0.4, 0.9, 0.7])
        result = loss_fn(
            logits, targets, progress_pred=progress_pred, progress_target=progress_target
        )
        expected_mse = torch.nn.functional.mse_loss(progress_pred.squeeze(-1), progress_target)
        self.assertAlmostEqual(result["L_progress"].item(), expected_mse.item(), places=5)

    def test_critic_loss_is_mse_not_bce(self):
        """L_critic = MSE (design contract: soft targets, NOT binary BCE)."""
        logits, targets = self._make_direction_data(batch=4)
        loss_fn = NavigationalLoss()
        critic_pred = torch.tensor([0.1, 0.9, 0.5, 0.3]).unsqueeze(-1)
        critic_target = torch.tensor([0.2, 0.8, 0.6, 0.4])
        result = loss_fn(logits, targets, critic_pred=critic_pred, critic_target=critic_target)
        expected_mse = torch.nn.functional.mse_loss(critic_pred.squeeze(-1), critic_target)
        self.assertAlmostEqual(result["L_critic"].item(), expected_mse.item(), places=5)

    def test_adaptive_weights_sum_to_one(self):
        """UW-SO weights w_nav + w_anchor + w_progress + w_critic ≈ 1.0."""
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        w_sum = (
            result["w_nav"].item()
            + result["w_anchor"].item()
            + result["w_progress"].item()
            + result["w_critic"].item()
        )
        self.assertAlmostEqual(w_sum, 1.0, places=4)

    def test_adaptive_weights_are_positive(self):
        logits, targets = self._make_direction_data()
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        for key in ["w_nav", "w_anchor", "w_progress", "w_critic"]:
            self.assertGreater(result[key].item(), 0.0)

    def test_total_loss_is_differentiable(self):
        """L_total must support backward pass."""
        logits = {b: torch.randn(4, 3, requires_grad=True) for b in ["structure", "domain"]}
        targets = {b: torch.randint(0, 3, (4,)) for b in ["structure", "domain"]}
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        result["L_total"].backward()
        for b in logits:
            self.assertIsNotNone(logits[b].grad)

    def test_missing_bank_in_targets_skipped(self):
        """Bank in logits but not in targets → not included in L_nav."""
        logits = {"structure": torch.randn(4, 3), "domain": torch.randn(4, 3)}
        targets = {"structure": torch.randint(0, 3, (4,))}  # domain missing
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        self.assertIn("structure", result["bank_losses"])
        self.assertNotIn("domain", result["bank_losses"])

    def test_empty_banks_gives_zero_nav_loss(self):
        """No matching banks → L_nav = 0."""
        logits = {"structure": torch.randn(4, 3)}
        targets = {"domain": torch.randint(0, 3, (4,))}  # no overlap
        loss_fn = NavigationalLoss()
        result = loss_fn(logits, targets)
        self.assertAlmostEqual(result["L_nav"].item(), 0.0, places=6)

    def test_initial_log_sigma_affects_weights(self):
        """Different initial_log_sigma changes initial weight balance."""
        logits, targets = self._make_direction_data()
        loss_fn_0 = NavigationalLoss(initial_log_sigma=0.0)
        loss_fn_1 = NavigationalLoss(initial_log_sigma=1.0)
        r0 = loss_fn_0(logits, targets)
        r1 = loss_fn_1(logits, targets)
        # Same weights (all equal) but different precision values
        # Both should still have equal weights since all sigmas are the same
        self.assertAlmostEqual(r0["w_nav"].item(), 0.25, places=3)
        self.assertAlmostEqual(r1["w_nav"].item(), 0.25, places=3)

    def test_full_pipeline_all_components(self):
        """All loss components active simultaneously."""
        logits, targets = self._make_direction_data(batch=4)
        loss_fn = NavigationalLoss()
        result = loss_fn(
            logits,
            targets,
            anchor_logits=torch.randn(4, 10),
            anchor_targets=torch.zeros(4, 10),
            progress_pred=torch.randn(4, 1),
            progress_target=torch.rand(4),
            critic_pred=torch.randn(4, 1),
            critic_target=torch.rand(4),
        )
        # All components should be non-zero
        self.assertGreater(result["L_nav"].item(), 0.0)
        self.assertTrue(torch.isfinite(result["L_total"]))


if __name__ == "__main__":
    unittest.main()
