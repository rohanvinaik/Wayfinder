"""
Information bridge — bottleneck between continuous encoder and ternary decoder.

Receives tensor features from GoalAnalyzer.forward(). Compresses to a
fixed-dim representation suitable for the ternary decoder's discrete
decision space.

Extended for Wayfinder with optional proof history concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBridge(nn.Module):
    """Bottleneck bridge from continuous to discrete space.

    When proof_history is provided, it is concatenated with the features
    before projection, enriching the bridge input with context from
    previously closed goals.

    Args:
        input_dim: Feature dimension from GoalAnalyzer.
        bridge_dim: Compressed representation dimension.
        history_dim: Proof history embedding dimension (0 = no history).
    """

    def __init__(
        self,
        input_dim: int = 256,
        bridge_dim: int = 128,
        history_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bridge_dim = bridge_dim
        self.history_dim = history_dim
        total_input = input_dim + history_dim
        self.projection = nn.Linear(total_input, bridge_dim)
        self.norm = nn.LayerNorm(bridge_dim)

    def forward(
        self,
        features: torch.Tensor,
        proof_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress features (+ optional history) to bridge representation.

        Args:
            features: [batch, input_dim] from GoalAnalyzer.
            proof_history: [batch, history_dim] mean-pooled history embeddings.
        """
        if self.history_dim > 0:
            if proof_history is not None:
                features = torch.cat([features, proof_history], dim=-1)
            else:
                zeros = torch.zeros(
                    features.shape[0],
                    self.history_dim,
                    device=features.device,
                    dtype=features.dtype,
                )
                features = torch.cat([features, zeros], dim=-1)
        return self.norm(self.projection(features))
