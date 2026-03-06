"""
Goal analyzer — maps encoder embeddings to latent proof-intent features.

Differentiable path: forward() returns features, bank logits, and anchor logits.
Non-differentiable path: analyze() returns GoalAnalysis for logging/debug.

Extended for Wayfinder with 6 bank direction heads and multi-label anchor head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class GoalAnalysis:
    """Structured analysis of a proof goal (non-differentiable, for logging).

    Attributes:
        math_domain: Detected domain (algebra, topology, analysis, etc.)
        goal_type: Classification of the goal structure.
        estimated_depth: Rough estimate of proof depth.
        hypotheses_count: Number of hypotheses in the context.
    """

    math_domain: str = "unknown"
    goal_type: str = "unknown"
    estimated_depth: str = "shallow"
    hypotheses_count: int = 0
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "math_domain": self.math_domain,
            "goal_type": self.goal_type,
            "estimated_depth": self.estimated_depth,
            "hypotheses_count": self.hypotheses_count,
            "entities": self.entities,
        }


class GoalAnalyzer(nn.Module):
    """Extracts structured proof intent from encoder embeddings.

    Two interfaces:
        forward(): Returns features + optional bank/anchor logits. Differentiable.
        analyze(): Returns GoalAnalysis. Non-differentiable, for logging.

    Args:
        input_dim: Embedding dimension (384).
        feature_dim: Latent feature dimension.
        num_anchors: Number of anchor labels (0 = no anchor head).
        navigable_banks: Bank names for direction heads (empty = no bank heads).
    """

    def __init__(
        self,
        input_dim: int = 384,
        feature_dim: int = 256,
        num_anchors: int = 0,
        navigable_banks: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.projection = nn.Linear(input_dim, feature_dim)

        # Bank direction heads: each predicts {-1, 0, +1} as 3-class softmax
        self.bank_heads: nn.ModuleDict | None = None
        if navigable_banks:
            self.bank_heads = nn.ModuleDict(
                {bank: nn.Linear(feature_dim, 3) for bank in navigable_banks}
            )

        # Anchor head: multi-label sigmoid prediction
        self.anchor_head: nn.Linear | None = None
        if num_anchors > 0:
            self.anchor_head = nn.Linear(feature_dim, num_anchors)

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Forward pass.

        Returns:
            Tuple of (features, bank_logits, anchor_logits):
                features: [batch, feature_dim]
                bank_logits: {bank_name: [batch, 3]} (empty dict if no bank heads)
                anchor_logits: [batch, num_anchors] or None
        """
        features = self.projection(embeddings)

        bank_logits: dict[str, torch.Tensor] = {}
        if self.bank_heads is not None:
            bank_logits = {bank: head(features) for bank, head in self.bank_heads.items()}

        anchor_logits = None
        if self.anchor_head is not None:
            anchor_logits = self.anchor_head(features)

        return features, bank_logits, anchor_logits

    def analyze(self, embeddings: torch.Tensor) -> list[GoalAnalysis]:
        with torch.no_grad():
            features, _, _ = self.forward(embeddings)
            return [GoalAnalysis() for _ in range(features.shape[0])]
