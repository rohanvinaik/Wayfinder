"""Residual executor — post-structural local tactic prediction.

A small classifier that predicts which tactic family to use on a
residual (post-intro) goal state. This is Task B from the post-EXP-3.2
plan: learn H(tactic | normalized_goal, hypotheses) not H(proof | theorem).

Architecture:
    goal_embedding (384d from frozen encoder) → 2-layer MLP → 6-class softmax

The 6 tactic families (covering 74% of residual steps):
    0: rw       (24% of residual)
    1: simp     (18%)
    2: exact    (15%)
    3: refine   (7%)
    4: apply    (5%)
    5: other    (31% — ext, simpa, rfl, etc.)

Secondary head: premise_needed (bool) — whether the tactic needs premise args.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Tactic family mapping — index → base tactic name
TACTIC_FAMILIES = ["rw", "simp", "exact", "refine", "apply", "other"]
TACTIC_TO_IDX = {t: i for i, t in enumerate(TACTIC_FAMILIES)}
NUM_FAMILIES = len(TACTIC_FAMILIES)


def tactic_to_family_idx(tactic_base: str) -> int:
    """Map a base tactic name to its family index."""
    return TACTIC_TO_IDX.get(tactic_base, TACTIC_TO_IDX["other"])


@dataclass
class ResidualPrediction:
    """Output of the residual executor."""

    family_logits: torch.Tensor  # [batch, NUM_FAMILIES]
    premise_needed_logit: torch.Tensor  # [batch, 1]
    top_family: int  # argmax family index
    top_family_name: str  # human-readable
    confidence: float  # softmax probability of top family
    premise_needed: bool  # sigmoid > 0.5


class ResidualExecutor(nn.Module):
    """Post-structural tactic family classifier.

    Takes a goal state embedding and predicts which tactic family
    (rw/simp/exact/refine/apply/other) should be used, plus whether
    the tactic needs premise arguments from retrieval.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_families: int = NUM_FAMILIES,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Primary head: tactic family classification
        self.family_head = nn.Linear(hidden_dim, num_families)
        # Secondary head: premise needed (binary)
        self.premise_head = nn.Linear(hidden_dim, 1)

    def forward(self, goal_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            goal_embedding: [batch, input_dim] from frozen encoder.

        Returns:
            family_logits: [batch, num_families]
            premise_logit: [batch, 1]
        """
        h = self.trunk(goal_embedding)
        return self.family_head(h), self.premise_head(h)

    def predict(self, goal_embedding: torch.Tensor) -> ResidualPrediction:
        """Single-example prediction with human-readable output."""
        self.eval()
        with torch.no_grad():
            family_logits, premise_logit = self.forward(goal_embedding.unsqueeze(0))
            probs = torch.softmax(family_logits, dim=-1).squeeze(0)
            top_idx = int(probs.argmax().item())
            return ResidualPrediction(
                family_logits=family_logits.squeeze(0),
                premise_needed_logit=premise_logit.squeeze(0),
                top_family=top_idx,
                top_family_name=TACTIC_FAMILIES[top_idx],
                confidence=float(probs[top_idx].item()),
                premise_needed=float(torch.sigmoid(premise_logit).item()) > 0.5,
            )
