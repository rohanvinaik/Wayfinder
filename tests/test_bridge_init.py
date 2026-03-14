"""Init tests for InformationBridge (mutation-prescribed)."""
import unittest
import torch
from src.bridge import InformationBridge

class TestInformationBridgeInit(unittest.TestCase):
    def test_stored_attributes(self):
        br = InformationBridge(input_dim=256, bridge_dim=128)
        self.assertEqual(br.input_dim, 256)
        self.assertEqual(br.bridge_dim, 128)
        self.assertEqual(br.history_dim, 0)
    def test_swap(self):
        a = InformationBridge(input_dim=256, bridge_dim=128)
        b = InformationBridge(input_dim=128, bridge_dim=256)
        self.assertNotEqual(a.input_dim, b.input_dim)
    def test_history_dim(self):
        br = InformationBridge(input_dim=256, bridge_dim=128, history_dim=64)
        self.assertEqual(br.history_dim, 64)
    def test_output_dim(self):
        br = InformationBridge(input_dim=256, bridge_dim=64)
        self.assertEqual(br(torch.randn(2, 256)).shape[-1], 64)

if __name__ == "__main__":
    unittest.main()
