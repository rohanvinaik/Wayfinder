"""Init tests for DomainGate (mutation-prescribed)."""

import unittest

import torch

from src.domain_gate import DomainGate


class TestDomainGateInit(unittest.TestCase):
    def test_stored_attributes(self):
        dg = DomainGate(input_dim=384, hidden_dim=128)
        self.assertEqual(dg.input_dim, 384)
        self.assertEqual(dg.hidden_dim, 128)

    def test_swap(self):
        a = DomainGate(input_dim=384, hidden_dim=128)
        b = DomainGate(input_dim=128, hidden_dim=384)
        self.assertNotEqual(a.input_dim, b.input_dim)

    def test_sequential(self):
        self.assertIsInstance(DomainGate().net, torch.nn.Sequential)

    def test_single_logit_output(self):
        dg = DomainGate(input_dim=64, hidden_dim=32)
        self.assertEqual(dg(torch.randn(2, 64)).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
