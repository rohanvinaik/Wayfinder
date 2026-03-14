"""Init tests for TernaryDecoder and TernaryLinear (mutation-prescribed)."""

import unittest

import torch

from src.ternary_decoder import TernaryDecoder, TernaryLinear


class TestTernaryLinearInit(unittest.TestCase):
    def test_stored_attributes(self):
        layer = TernaryLinear(16, 8, bias=True)
        self.assertEqual(layer.in_features, 16)
        self.assertEqual(layer.out_features, 8)

    def test_weight_shape(self):
        self.assertEqual(TernaryLinear(16, 8).weight.shape, (8, 16))

    def test_bias_present(self):
        self.assertIsNotNone(TernaryLinear(4, 2).bias)

    def test_bias_none(self):
        self.assertIsNone(TernaryLinear(4, 2, bias=False).bias)

    def test_swap(self):
        self.assertNotEqual(TernaryLinear(16, 8).weight.shape, TernaryLinear(8, 16).weight.shape)

    def test_weight_is_param(self):
        self.assertIsInstance(TernaryLinear(4, 2).weight, torch.nn.Parameter)


class TestTernaryDecoderInit(unittest.TestCase):
    def test_stored_attributes(self):
        dec = TernaryDecoder(input_dim=64, hidden_dim=128, tier1_vocab_size=10)
        self.assertEqual(dec.input_dim, 64)
        self.assertEqual(dec.hidden_dim, 128)
        self.assertEqual(dec.tier1_vocab_size, 10)
        self.assertTrue(dec.ternary_enabled)

    def test_swap(self):
        a = TernaryDecoder(input_dim=64, hidden_dim=128)
        b = TernaryDecoder(input_dim=128, hidden_dim=64)
        self.assertNotEqual(a.input_dim, b.input_dim)

    def test_ternary_disabled(self):
        dec = TernaryDecoder(ternary_enabled=False)
        self.assertFalse(any(isinstance(m, TernaryLinear) for m in dec.modules()))

    def test_ternary_enabled(self):
        dec = TernaryDecoder(ternary_enabled=True, tier1_vocab_size=5)
        self.assertTrue(any(isinstance(m, TernaryLinear) for m in dec.modules()))

    def test_boundary_zero_vocab(self):
        dec = TernaryDecoder(tier1_vocab_size=0, tier2_vocab_size=0)
        self.assertEqual(dec.tier1_vocab_size, 0)


if __name__ == "__main__":
    unittest.main()
