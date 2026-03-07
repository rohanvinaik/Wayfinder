"""Smoke test -- verify the full pipeline forward pass works end-to-end."""

import unittest

import torch

from src.bridge import InformationBridge
from src.domain_gate import DomainGate
from src.goal_analyzer import GoalAnalyzer
from src.losses import CompositeLoss
from src.ternary_decoder import TernaryDecoder


class TestPipelineSmoke(unittest.TestCase):
    def test_forward_pass(self):
        """Full pipeline: fake embeddings -> goal analyzer -> bridge -> decoder -> loss."""
        batch_size = 4
        embed_dim = 384
        feature_dim = 256
        bridge_dim = 128
        tier1_vocab = 50
        tier2_vocab = 100

        # Simulate encoder output
        embeddings = torch.randn(batch_size, embed_dim)

        # Domain gate
        gate = DomainGate(input_dim=embed_dim, hidden_dim=128)
        gate_logits = gate(embeddings)
        self.assertEqual(gate_logits.shape, (batch_size, 1))

        # Goal analyzer (returns features, bank_logits, anchor_logits)
        analyzer = GoalAnalyzer(input_dim=embed_dim, feature_dim=feature_dim)
        features, bank_logits, anchor_logits = analyzer(embeddings)
        self.assertEqual(features.shape, (batch_size, feature_dim))
        self.assertEqual(bank_logits, {})  # no banks configured
        self.assertIsNone(anchor_logits)  # no anchors configured

        # Bridge
        bridge = InformationBridge(input_dim=feature_dim, bridge_dim=bridge_dim)
        bridged = bridge(features)
        self.assertEqual(bridged.shape, (batch_size, bridge_dim))

        # Decoder
        decoder = TernaryDecoder(
            input_dim=bridge_dim,
            hidden_dim=256,
            tier1_vocab_size=tier1_vocab,
            tier2_vocab_size=tier2_vocab,
            num_layers=2,
            ternary_enabled=True,
        )
        output = decoder(bridged)
        self.assertIn("tier1_logits", output)
        self.assertEqual(output["tier1_logits"].shape, (batch_size, tier1_vocab))

        # Loss
        loss_fn = CompositeLoss()
        targets = torch.randint(0, tier1_vocab, (batch_size,))
        loss_dict = loss_fn(output["tier1_logits"], targets)
        self.assertTrue(torch.isfinite(loss_dict["L_total"]))

    def test_backward_pass(self):
        """Verify gradients flow through the full pipeline."""
        embeddings = torch.randn(2, 384)
        analyzer = GoalAnalyzer(input_dim=384, feature_dim=256)
        bridge = InformationBridge(input_dim=256, bridge_dim=128)
        decoder = TernaryDecoder(
            input_dim=128,
            hidden_dim=256,
            tier1_vocab_size=30,
            tier2_vocab_size=50,
            num_layers=1,
            ternary_enabled=True,
        )
        loss_fn = CompositeLoss()

        features, _, _ = analyzer(embeddings)
        bridged = bridge(features)
        output = decoder(bridged)
        targets = torch.randint(0, 30, (2,))
        loss_dict = loss_fn(output["tier1_logits"], targets)
        loss_dict["L_total"].backward()

        # Check gradients exist on decoder
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())
        self.assertTrue(has_grad, "No gradients flowed to decoder")


if __name__ == "__main__":
    unittest.main()
