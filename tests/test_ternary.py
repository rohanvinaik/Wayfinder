"""Tests for ternary quantization -- values in {-1, 0, +1} and STE backward finite."""

import unittest

import torch

from src.ternary_decoder import ternary_quantize


class TestTernaryQuantize(unittest.TestCase):
    def test_output_values_only_ternary(self):
        """Quantized weights must be exactly {-1, 0, +1}."""
        w = torch.randn(32, 64)
        q = ternary_quantize(w)
        unique = sorted(q.unique().tolist())
        for v in unique:
            self.assertIn(v, [-1.0, 0.0, 1.0])

    def test_large_positive_quantizes_to_plus_one(self):
        """Weights well above threshold should become +1."""
        w = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        q = ternary_quantize(w)
        # All values identical and positive => threshold = 0.7 * mean(abs) = 7.0
        # All > 7.0, so all should be +1
        self.assertEqual(q.tolist(), [[1.0, 1.0, 1.0, 1.0]])

    def test_large_negative_quantizes_to_minus_one(self):
        """Weights well below -threshold should become -1."""
        w = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
        q = ternary_quantize(w)
        self.assertEqual(q.tolist(), [[-1.0, -1.0, -1.0, -1.0]])

    def test_zero_weights_stay_zero(self):
        """All-zero weights should quantize to zero."""
        w = torch.zeros(4, 4)
        q = ternary_quantize(w)
        self.assertEqual(q.tolist(), torch.zeros(4, 4).tolist())

    def test_mixed_signs_correct(self):
        """Mixed large pos/neg values quantize to correct signs."""
        w = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
        q = ternary_quantize(w)
        # threshold = 0.7 * 5.0 = 3.5; all |values| > 3.5
        self.assertEqual(q.tolist(), [[1.0, -1.0, 1.0, -1.0]])

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        for shape in [(4, 8), (1, 1), (16, 32)]:
            w = torch.randn(*shape)
            q = ternary_quantize(w)
            self.assertEqual(q.shape, w.shape)

    def test_ste_backward_finite(self):
        """STE backward through quantization must produce finite gradients."""
        w = torch.randn(16, 32, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        self.assertIsNotNone(w.grad)
        self.assertEqual(torch.isnan(w.grad).any().item(), False)
        self.assertEqual(torch.isinf(w.grad).any().item(), False)

    def test_ste_gradient_is_identity(self):
        """STE gradient should be 1.0 for all weights (straight-through)."""
        w = torch.randn(8, 16, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        # STE: d(quantized)/d(weights) = 1.0 via straight-through
        self.assertEqual(w.grad.tolist(), torch.ones_like(w).tolist())

    def test_quantize_deterministic(self):
        """Same input should produce same output."""
        w = torch.randn(8, 16)
        q1 = ternary_quantize(w)
        q2 = ternary_quantize(w)
        self.assertEqual(q1.tolist(), q2.tolist())


if __name__ == "__main__":
    unittest.main()
