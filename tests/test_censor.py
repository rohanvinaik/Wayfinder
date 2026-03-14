"""Tests for CensorNetwork — VERIFICATION enhancement failure predictor."""

import unittest

import torch

from src.censor import CensorNetwork
from src.som_contracts import CensorPrediction


class TestCensorNetworkConstruction(unittest.TestCase):
    """Test construction with default and custom parameters."""

    def test_default_construction(self):
        model = CensorNetwork()
        self.assertEqual(model.threshold, 0.7)

    def test_custom_construction(self):
        model = CensorNetwork(goal_dim=128, tactic_dim=32, hidden_dim=64, threshold=0.5)
        self.assertEqual(model.threshold, 0.5)


class TestCensorNetworkForward(unittest.TestCase):
    """Test forward pass output shapes, range, and finiteness."""

    def setUp(self):
        self.model = CensorNetwork()

    def test_forward_shape_batch_1(self):
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        out = self.model(goal, tactic)
        self.assertEqual(out.shape, (1, 1))

    def test_forward_shape_batch_4(self):
        goal = torch.randn(4, 256)
        tactic = torch.randn(4, 64)
        out = self.model(goal, tactic)
        self.assertEqual(out.shape, (4, 1))

    def test_forward_shape_batch_8(self):
        goal = torch.randn(8, 256)
        tactic = torch.randn(8, 64)
        out = self.model(goal, tactic)
        self.assertEqual(out.shape, (8, 1))

    def test_forward_output_in_unit_interval(self):
        goal = torch.randn(16, 256)
        tactic = torch.randn(16, 64)
        out = self.model(goal, tactic)
        self.assertTrue((out >= 0.0).all(), "Output has values below 0")
        self.assertTrue((out <= 1.0).all(), "Output has values above 1")

    def test_forward_output_is_finite(self):
        goal = torch.randn(8, 256)
        tactic = torch.randn(8, 64)
        out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_custom_dims(self):
        model = CensorNetwork(goal_dim=128, tactic_dim=32, hidden_dim=64)
        goal = torch.randn(3, 128)
        tactic = torch.randn(3, 32)
        out = model(goal, tactic)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue((out >= 0.0).all())
        self.assertTrue((out <= 1.0).all())


class TestCensorNetworkPredict(unittest.TestCase):
    """Test predict() returns correct CensorPrediction."""

    def setUp(self):
        self.model = CensorNetwork()

    def test_predict_returns_censor_prediction(self):
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = self.model.predict(goal, tactic)
        self.assertIsInstance(result, CensorPrediction)

    def test_predict_failure_probability_in_unit_interval(self):
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = self.model.predict(goal, tactic)
        self.assertGreaterEqual(result.failure_probability, 0.0)
        self.assertLessEqual(result.failure_probability, 1.0)

    def test_predict_should_prune_is_bool(self):
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = self.model.predict(goal, tactic)
        self.assertIsInstance(result.should_prune, bool)

    def test_predict_consistency(self):
        """should_prune must equal (failure_probability >= threshold)."""
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = self.model.predict(goal, tactic)
        expected = result.failure_probability >= self.model.threshold
        self.assertEqual(result.should_prune, expected)

    def test_predict_threshold_zero_always_prunes(self):
        model = CensorNetwork(threshold=0.0)
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = model.predict(goal, tactic)
        self.assertTrue(result.should_prune)

    def test_predict_threshold_one_never_prunes(self):
        """Sigmoid output is strictly less than 1, so threshold=1.0 never fires."""
        model = CensorNetwork(threshold=1.0)
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = model.predict(goal, tactic)
        self.assertFalse(result.should_prune)


class TestCensorNetworkShouldPrune(unittest.TestCase):
    """Test should_prune() convenience method."""

    def test_should_prune_returns_bool(self):
        model = CensorNetwork()
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        result = model.should_prune(goal, tactic)
        self.assertIsInstance(result, bool)

    def test_should_prune_matches_predict(self):
        model = CensorNetwork()
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        prediction = model.predict(goal, tactic)
        prune_result = model.should_prune(goal, tactic)
        self.assertEqual(prune_result, prediction.should_prune)


class TestCensorNetworkGradient(unittest.TestCase):
    """Test gradient flow through forward pass."""

    def test_backward_pass(self):
        model = CensorNetwork()
        goal = torch.randn(4, 256)
        tactic = torch.randn(4, 64)
        out = model(goal, tactic)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"Non-finite gradient for {name}",
            )


class TestCensorNetworkBatchIndependence(unittest.TestCase):
    """Test that outputs for one sample are independent of other samples."""

    def test_batch_independence(self):
        model = CensorNetwork()
        model.eval()

        goal_fixed = torch.randn(1, 256)
        tactic_fixed = torch.randn(1, 64)

        # Run sample alone
        with torch.no_grad():
            solo = model(goal_fixed, tactic_fixed)

        # Run sample as part of a batch with other random samples
        goal_batch = torch.cat([goal_fixed, torch.randn(3, 256)], dim=0)
        tactic_batch = torch.cat([tactic_fixed, torch.randn(3, 64)], dim=0)
        with torch.no_grad():
            batched = model(goal_batch, tactic_batch)

        self.assertTrue(
            torch.allclose(solo, batched[0:1], atol=1e-6),
            "Output for sample 0 changed when batched with other samples",
        )


class TestCensorBoundedProperty(unittest.TestCase):
    """Verify the bounded [0, 1] algebraic property of should_prune/forward.

    CensorNetwork.forward uses sigmoid as its final activation, so outputs
    MUST always lie in [0.0, 1.0]. This is the ONLY function in the codebase
    with the "bounded" algebraic property.  We verify it across many random
    inputs, extreme values, and varied network configurations.
    """

    def test_forward_bounded_random_loop(self):
        """100 random inputs — all outputs must be in [0, 1]."""
        model = CensorNetwork()
        model.eval()
        for _ in range(100):
            goal = torch.randn(1, 256)
            tactic = torch.randn(1, 64)
            with torch.no_grad():
                out = model(goal, tactic)
            self.assertGreaterEqual(out.item(), 0.0)
            self.assertLessEqual(out.item(), 1.0)

    def test_forward_bounded_large_batch_random(self):
        """Single large batch (512 samples) — all outputs in [0, 1]."""
        model = CensorNetwork()
        model.eval()
        goal = torch.randn(512, 256)
        tactic = torch.randn(512, 64)
        with torch.no_grad():
            out = model(goal, tactic)
        self.assertTrue(
            (out >= 0.0).all().item(),
            f"Found value below 0: {out.min().item()}",
        )
        self.assertTrue(
            (out <= 1.0).all().item(),
            f"Found value above 1: {out.max().item()}",
        )

    def test_forward_bounded_varied_configs(self):
        """Bounded property holds across different hidden/dim configurations."""
        configs = [
            dict(goal_dim=64, tactic_dim=16, hidden_dim=32),
            dict(goal_dim=512, tactic_dim=128, hidden_dim=256),
            dict(goal_dim=128, tactic_dim=128, hidden_dim=64),
            dict(goal_dim=32, tactic_dim=8, hidden_dim=16),
        ]
        for cfg in configs:
            model = CensorNetwork(**cfg)
            model.eval()
            goal = torch.randn(20, cfg["goal_dim"])
            tactic = torch.randn(20, cfg["tactic_dim"])
            with torch.no_grad():
                out = model(goal, tactic)
            self.assertTrue(
                (out >= 0.0).all().item(),
                f"Config {cfg}: value below 0 ({out.min().item()})",
            )
            self.assertTrue(
                (out <= 1.0).all().item(),
                f"Config {cfg}: value above 1 ({out.max().item()})",
            )

    def test_predict_failure_probability_bounded_loop(self):
        """50 random calls to predict — failure_probability in [0, 1]."""
        model = CensorNetwork()
        model.eval()
        for _ in range(50):
            goal = torch.randn(1, 256)
            tactic = torch.randn(1, 64)
            pred = model.predict(goal, tactic)
            self.assertGreaterEqual(pred.failure_probability, 0.0)
            self.assertLessEqual(pred.failure_probability, 1.0)


class TestCensorExtremeInputs(unittest.TestCase):
    """Output stays bounded and finite with extreme inputs."""

    def setUp(self):
        self.model = CensorNetwork()
        self.model.eval()

    def test_zero_inputs(self):
        goal = torch.zeros(1, 256)
        tactic = torch.zeros(1, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_ones_inputs(self):
        goal = torch.ones(1, 256)
        tactic = torch.ones(1, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_large_positive_inputs(self):
        goal = torch.full((1, 256), 1e6)
        tactic = torch.full((1, 64), 1e6)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_large_negative_inputs(self):
        goal = torch.full((1, 256), -1e6)
        tactic = torch.full((1, 64), -1e6)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_mixed_extreme_inputs(self):
        """Large positive goals + large negative tactic features."""
        goal = torch.full((1, 256), 1e4)
        tactic = torch.full((1, 64), -1e4)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_very_small_inputs(self):
        """Near-zero (subnormal-scale) inputs."""
        goal = torch.full((1, 256), 1e-30)
        tactic = torch.full((1, 64), 1e-30)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.item(), 0.0)
        self.assertLessEqual(out.item(), 1.0)

    def test_extreme_batch(self):
        """Batch with diverse extreme values — all outputs bounded."""
        goal = torch.cat(
            [
                torch.zeros(1, 256),
                torch.ones(1, 256),
                torch.full((1, 256), 1e6),
                torch.full((1, 256), -1e6),
                torch.randn(1, 256),
            ],
            dim=0,
        )
        tactic = torch.cat(
            [
                torch.zeros(1, 64),
                torch.ones(1, 64),
                torch.full((1, 64), 1e6),
                torch.full((1, 64), -1e6),
                torch.randn(1, 64),
            ],
            dim=0,
        )
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertEqual(out.shape, (5, 1))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out >= 0.0).all().item())
        self.assertTrue((out <= 1.0).all().item())


class TestCensorDeterminism(unittest.TestCase):
    """In eval mode, same input must produce identical output."""

    def test_forward_deterministic_eval_mode(self):
        model = CensorNetwork()
        model.eval()
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        with torch.no_grad():
            out1 = model(goal, tactic)
            out2 = model(goal, tactic)
        self.assertTrue(
            torch.equal(out1, out2),
            "Forward pass not deterministic in eval mode",
        )

    def test_should_prune_deterministic_eval_mode(self):
        """should_prune returns same bool for same input across 10 calls."""
        model = CensorNetwork()
        model.eval()
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        results = [model.should_prune(goal, tactic) for _ in range(10)]
        self.assertEqual(
            len(set(results)),
            1,
            f"should_prune returned inconsistent results: {results}",
        )

    def test_predict_deterministic_eval_mode(self):
        """predict returns identical failure_probability for same input."""
        model = CensorNetwork()
        model.eval()
        goal = torch.randn(1, 256)
        tactic = torch.randn(1, 64)
        probs = [model.predict(goal, tactic).failure_probability for _ in range(5)]
        for p in probs[1:]:
            self.assertEqual(
                probs[0],
                p,
                f"predict failure_probability not deterministic: {probs}",
            )

    def test_forward_deterministic_multiple_inputs(self):
        """Multiple different fixed inputs, each checked for determinism."""
        model = CensorNetwork()
        model.eval()
        torch.manual_seed(42)
        for _ in range(20):
            goal = torch.randn(1, 256)
            tactic = torch.randn(1, 64)
            with torch.no_grad():
                out1 = model(goal, tactic)
                out2 = model(goal, tactic)
            self.assertTrue(
                torch.equal(out1, out2),
                "Non-deterministic output in eval mode",
            )


class TestCensorForwardShapeEdgeCases(unittest.TestCase):
    """Additional shape tests for batch-size edge cases."""

    def setUp(self):
        self.model = CensorNetwork()
        self.model.eval()

    def test_forward_shape_batch_16(self):
        goal = torch.randn(16, 256)
        tactic = torch.randn(16, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertEqual(out.shape, (16, 1))

    def test_forward_shape_batch_32(self):
        goal = torch.randn(32, 256)
        tactic = torch.randn(32, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertEqual(out.shape, (32, 1))

    def test_forward_shape_batch_128(self):
        goal = torch.randn(128, 256)
        tactic = torch.randn(128, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertEqual(out.shape, (128, 1))

    def test_output_dtype_is_float(self):
        goal = torch.randn(4, 256)
        tactic = torch.randn(4, 64)
        with torch.no_grad():
            out = self.model(goal, tactic)
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
