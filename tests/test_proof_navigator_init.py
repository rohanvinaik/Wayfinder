"""Init tests for ProofNavigator (mutation-prescribed)."""
import unittest
from src.proof_navigator import ProofNavigator
from src.ternary_decoder import TernaryLinear

class TestProofNavigatorInit(unittest.TestCase):
    def test_stored_attributes(self):
        nav = ProofNavigator(input_dim=128, hidden_dim=256, num_anchors=300)
        self.assertEqual(nav.input_dim, 128)
        self.assertEqual(nav.hidden_dim, 256)
        self.assertEqual(nav.num_anchors, 300)
    def test_default_banks(self):
        self.assertEqual(len(ProofNavigator().navigable_banks), 6)
    def test_custom_banks(self):
        nav = ProofNavigator(navigable_banks=["structure", "domain"])
        self.assertEqual(len(nav.direction_heads), 2)
    def test_swap(self):
        a = ProofNavigator(input_dim=64, hidden_dim=128)
        b = ProofNavigator(input_dim=128, hidden_dim=64)
        self.assertNotEqual(a.input_dim, b.input_dim)
    def test_ternary_disabled(self):
        nav = ProofNavigator(ternary_enabled=False)
        self.assertFalse(any(isinstance(m, TernaryLinear) for m in nav.modules()))
    def test_direction_head_3_classes(self):
        nav = ProofNavigator(input_dim=64, hidden_dim=32, navigable_banks=["structure"])
        self.assertEqual(nav.direction_heads["structure"].out_features, 3)

if __name__ == "__main__":
    unittest.main()
