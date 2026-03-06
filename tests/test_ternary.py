"""Tests for ternary quantization -- values in {-1, 0, +1} and STE backward finite."""

import unittest

import torch

from src.ternary_decoder import ternary_quantize


class TestTernaryQuantize(unittest.TestCase):
    def test_output_values_only_ternary(self):
        """Quantized weights must be exactly {-1, 0, +1}."""
        w = torch.randn(32, 64)
        q = ternary_quantize(w)
        unique = set(q.unique().tolist())
        self.assertTrue(
            unique.issubset({-1.0, 0.0, 1.0}),
            f"Expected only {{-1, 0, 1}}, got {unique}",
        )

    def test_ste_backward_finite(self):
        """STE backward through quantization must produce finite gradients."""
        w = torch.randn(16, 32, requires_grad=True)
        q = ternary_quantize(w)
        loss = q.sum()
        loss.backward()
        assert w.grad is not None
        self.assertFalse(torch.isnan(w.grad).any(), "STE gradient contains NaN")
        self.assertFalse(torch.isinf(w.grad).any(), "STE gradient contains Inf")

    def test_quantize_deterministic(self):
        """Same input should produce same output."""
        w = torch.randn(8, 16)
        q1 = ternary_quantize(w)
        q2 = ternary_quantize(w)
        self.assertTrue(torch.equal(q1, q2))


if __name__ == "__main__":
    unittest.main()
