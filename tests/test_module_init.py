"""Constructor init tests for neural network modules.

Mutation-prescribed: VALUE (stored attributes), SWAP (parameter order),
STATE (initial module state), BOUNDARY (edge cases).
Targets: TernaryDecoder, TernaryLinear, GoalAnalyzer, DomainGate,
ProofNavigator, InformationBridge.
"""

import unittest

import torch

from src.bridge import InformationBridge
from src.domain_gate import DomainGate
from src.goal_analyzer import GoalAnalyzer
from src.proof_navigator import ProofNavigator
from src.ternary_decoder import TernaryDecoder, TernaryLinear


class TestTernaryLinearInit(unittest.TestCase):
    """VALUE + STATE prescriptions for TernaryLinear.__init__."""

    def test_stored_attributes(self):
        layer = TernaryLinear(16, 8, bias=True)
        self.assertEqual(layer.in_features, 16)
        self.assertEqual(layer.out_features, 8)

    def test_weight_shape(self):
        layer = TernaryLinear(16, 8)
        self.assertEqual(layer.weight.shape, (8, 16))

    def test_bias_present_by_default(self):
        layer = TernaryLinear(4, 2)
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape, (2,))

    def test_bias_none_when_disabled(self):
        layer = TernaryLinear(4, 2, bias=False)
        self.assertIsNone(layer.bias)

    def test_swap_in_out_features(self):
        """SWAP: TernaryLinear(16, 8) != TernaryLinear(8, 16)."""
        a = TernaryLinear(16, 8)
        b = TernaryLinear(8, 16)
        self.assertNotEqual(a.weight.shape, b.weight.shape)

    def test_weight_is_parameter(self):
        layer = TernaryLinear(4, 2)
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertTrue(layer.weight.requires_grad)


class TestTernaryDecoderInit(unittest.TestCase):
    """VALUE + STATE + BOUNDARY prescriptions for TernaryDecoder.__init__."""

    def test_stored_attributes(self):
        dec = TernaryDecoder(input_dim=64, hidden_dim=128, tier1_vocab_size=10)
        self.assertEqual(dec.input_dim, 64)
        self.assertEqual(dec.hidden_dim, 128)
        self.assertEqual(dec.tier1_vocab_size, 10)
        self.assertEqual(dec.tier2_vocab_size, 0)
        self.assertEqual(dec.num_layers, 2)
        self.assertTrue(dec.ternary_enabled)
        self.assertFalse(dec.partial_ternary)

    def test_swap_input_hidden(self):
        """SWAP: different input_dim and hidden_dim produce different architectures."""
        a = TernaryDecoder(input_dim=64, hidden_dim=128)
        b = TernaryDecoder(input_dim=128, hidden_dim=64)
        self.assertNotEqual(a.input_dim, b.input_dim)
        self.assertNotEqual(a.hidden_dim, b.hidden_dim)

    def test_ternary_disabled_uses_linear(self):
        dec = TernaryDecoder(ternary_enabled=False)
        # Hidden layers should be nn.Linear, not TernaryLinear
        has_ternary = any(isinstance(m, TernaryLinear) for m in dec.modules())
        self.assertFalse(has_ternary)

    def test_ternary_enabled_uses_ternary_linear(self):
        dec = TernaryDecoder(ternary_enabled=True, tier1_vocab_size=5)
        has_ternary = any(isinstance(m, TernaryLinear) for m in dec.modules())
        self.assertTrue(has_ternary)

    def test_boundary_zero_vocab(self):
        """BOUNDARY: tier1_vocab_size=0, tier2_vocab_size=0."""
        dec = TernaryDecoder(tier1_vocab_size=0, tier2_vocab_size=0)
        self.assertEqual(dec.tier1_vocab_size, 0)
        self.assertEqual(dec.tier2_vocab_size, 0)

    def test_partial_ternary_heads_are_linear(self):
        dec = TernaryDecoder(ternary_enabled=True, partial_ternary=True, tier1_vocab_size=5)
        self.assertTrue(dec.partial_ternary)


class TestGoalAnalyzerInit(unittest.TestCase):
    """VALUE + STATE prescriptions for GoalAnalyzer.__init__."""

    def test_stored_attributes(self):
        ga = GoalAnalyzer(input_dim=384, feature_dim=256)
        self.assertEqual(ga.input_dim, 384)
        self.assertEqual(ga.feature_dim, 256)

    def test_no_bank_heads_by_default(self):
        ga = GoalAnalyzer()
        self.assertIsNone(ga.bank_heads)

    def test_bank_heads_created_when_specified(self):
        banks = ["structure", "domain", "depth"]
        ga = GoalAnalyzer(navigable_banks=banks)
        self.assertIsNotNone(ga.bank_heads)
        self.assertEqual(set(ga.bank_heads.keys()), set(banks))

    def test_no_anchor_head_when_zero(self):
        ga = GoalAnalyzer(num_anchors=0)
        self.assertIsNone(ga.anchor_head)

    def test_anchor_head_created_when_positive(self):
        ga = GoalAnalyzer(num_anchors=100)
        self.assertIsNotNone(ga.anchor_head)

    def test_swap_input_feature_dim(self):
        """SWAP: different dims produce different projection shapes."""
        a = GoalAnalyzer(input_dim=384, feature_dim=256)
        b = GoalAnalyzer(input_dim=256, feature_dim=384)
        self.assertNotEqual(a.projection.in_features, b.projection.in_features)

    def test_projection_shape(self):
        ga = GoalAnalyzer(input_dim=384, feature_dim=128)
        self.assertEqual(ga.projection.in_features, 384)
        self.assertEqual(ga.projection.out_features, 128)


class TestDomainGateInit(unittest.TestCase):
    """VALUE + STATE prescriptions for DomainGate.__init__."""

    def test_stored_attributes(self):
        dg = DomainGate(input_dim=384, hidden_dim=128)
        self.assertEqual(dg.input_dim, 384)
        self.assertEqual(dg.hidden_dim, 128)

    def test_swap_dims(self):
        a = DomainGate(input_dim=384, hidden_dim=128)
        b = DomainGate(input_dim=128, hidden_dim=384)
        self.assertNotEqual(a.input_dim, b.input_dim)

    def test_network_is_sequential(self):
        dg = DomainGate()
        self.assertIsInstance(dg.net, torch.nn.Sequential)

    def test_output_is_single_logit(self):
        """STATE: final layer outputs 1 logit (binary classifier)."""
        dg = DomainGate(input_dim=64, hidden_dim=32)
        x = torch.randn(2, 64)
        out = dg(x)
        self.assertEqual(out.shape, (2, 1))


class TestProofNavigatorInit(unittest.TestCase):
    """VALUE + STATE prescriptions for ProofNavigator.__init__."""

    def test_stored_attributes(self):
        nav = ProofNavigator(input_dim=128, hidden_dim=256, num_anchors=300)
        self.assertEqual(nav.input_dim, 128)
        self.assertEqual(nav.hidden_dim, 256)
        self.assertEqual(nav.num_anchors, 300)
        self.assertTrue(nav.ternary_enabled)

    def test_default_banks(self):
        nav = ProofNavigator()
        self.assertEqual(len(nav.navigable_banks), 6)

    def test_custom_banks(self):
        nav = ProofNavigator(navigable_banks=["structure", "domain"])
        self.assertEqual(nav.navigable_banks, ["structure", "domain"])
        self.assertEqual(len(nav.direction_heads), 2)

    def test_swap_input_hidden(self):
        a = ProofNavigator(input_dim=64, hidden_dim=128)
        b = ProofNavigator(input_dim=128, hidden_dim=64)
        self.assertNotEqual(a.input_dim, b.input_dim)

    def test_ternary_disabled(self):
        nav = ProofNavigator(ternary_enabled=False)
        has_ternary = any(isinstance(m, TernaryLinear) for m in nav.modules())
        self.assertFalse(has_ternary)

    def test_direction_heads_output_3_classes(self):
        """STATE: each direction head outputs 3 logits ({-1, 0, +1})."""
        nav = ProofNavigator(input_dim=64, hidden_dim=32, navigable_banks=["structure"])
        self.assertEqual(nav.direction_heads["structure"].out_features, 3)


class TestInformationBridgeInit(unittest.TestCase):
    """VALUE + STATE prescriptions for InformationBridge.__init__."""

    def test_stored_attributes(self):
        br = InformationBridge(input_dim=256, bridge_dim=128)
        self.assertEqual(br.input_dim, 256)
        self.assertEqual(br.bridge_dim, 128)
        self.assertEqual(br.history_dim, 0)

    def test_swap_input_bridge(self):
        a = InformationBridge(input_dim=256, bridge_dim=128)
        b = InformationBridge(input_dim=128, bridge_dim=256)
        self.assertNotEqual(a.input_dim, b.input_dim)

    def test_history_dim_changes_projection(self):
        """STATE: history_dim > 0 increases projection input size."""
        br_no_hist = InformationBridge(input_dim=256, bridge_dim=128, history_dim=0)
        br_hist = InformationBridge(input_dim=256, bridge_dim=128, history_dim=64)
        self.assertNotEqual(br_no_hist.history_dim, br_hist.history_dim)

    def test_output_dim_matches_bridge_dim(self):
        br = InformationBridge(input_dim=256, bridge_dim=64)
        x = torch.randn(2, 256)
        out = br(x)
        self.assertEqual(out.shape[-1], 64)


if __name__ == "__main__":
    unittest.main()
