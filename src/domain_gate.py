"""
Domain gate — binary classifier for mathematical vs. non-mathematical goals.

Takes encoder embeddings and produces a scalar confidence that the input
is a valid mathematical proof goal. Trained separately with BCE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class GateDecision:
    """Domain gate classification result."""

    in_domain: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"in_domain": self.in_domain, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateDecision:
        return cls(in_domain=d["in_domain"], confidence=d["confidence"])


class DomainGate(nn.Module):
    """Binary math-domain classifier on top of encoder embeddings.

    Args:
        input_dim: Embedding dimension (384 for all-MiniLM-L6-v2).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)

    def predict(self, embeddings: torch.Tensor, threshold: float = 0.5) -> list[GateDecision]:
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = torch.sigmoid(logits)
            return [
                GateDecision(in_domain=p >= threshold, confidence=p)
                for p in probs.squeeze(-1).tolist()
            ]
