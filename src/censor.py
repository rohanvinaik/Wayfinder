"""
Censor network — VERIFICATION enhancement (Slot 5) for Wayfinder v2/v3.

Learns to predict tactic failure BEFORE Lean kernel verification.
Trained on accumulated (goal_state, tactic, result) triples from search.
Prunes the candidate set before expensive Lean kernel calls.

This inverts the verification oracle: instead of only learning what works,
actively learn what does NOT work.

v3A adds asymmetric loss (MCA-motivated: missed suppressions penalized
more than false suppressions) and calibration metrics.

See DESIGN §10.5 and §12.3 for specification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.som_contracts import CensorPrediction


class CensorNetwork(nn.Module):
    """Predicts tactic failure from goal features + tactic features.

    Learns what does NOT work, inverting the verification oracle.
    Used to prune candidates before expensive Lean kernel verification.

    Args:
        goal_dim: Goal feature dimension (default 256).
        tactic_dim: Tactic embedding dimension (default 64).
        hidden_dim: Hidden layer dimension (default 128).
        threshold: Failure probability threshold for pruning (default 0.7).
    """

    def __init__(
        self,
        goal_dim: int = 256,
        tactic_dim: int = 64,
        hidden_dim: int = 128,
        threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.threshold = threshold

        self.network = nn.Sequential(
            nn.Linear(goal_dim + tactic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        goal_features: torch.Tensor,
        tactic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning failure probability.

        Args:
            goal_features: [batch, goal_dim]
            tactic_features: [batch, tactic_dim]

        Returns:
            Failure probability [batch, 1] in [0, 1].
        """
        combined = torch.cat([goal_features, tactic_features], dim=-1)
        return torch.sigmoid(self.network(combined))

    def predict(
        self,
        goal_features: torch.Tensor,
        tactic_features: torch.Tensor,
    ) -> CensorPrediction:
        """Predict whether a tactic will fail (inference mode).

        Args:
            goal_features: [1, goal_dim]
            tactic_features: [1, tactic_dim]
        """
        with torch.no_grad():
            failure_prob = self.forward(goal_features, tactic_features)

        prob = float(failure_prob[0, 0].item())
        return CensorPrediction(
            should_prune=prob >= self.threshold,
            failure_probability=prob,
        )

    def should_prune(
        self,
        goal_features: torch.Tensor,
        tactic_features: torch.Tensor,
    ) -> bool:
        """Quick check: should this tactic be pruned?"""
        return self.predict(goal_features, tactic_features).should_prune


def asymmetric_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    w_neg: float = 2.0,
    w_pos: float = 1.0,
) -> torch.Tensor:
    """Asymmetric binary cross-entropy for censor training (MCA-motivated).

    Missed suppressions (FN: failure classified as non-failure) are penalized
    more than false suppressions (FP: non-failure classified as failure).

    Rationale: a missed suppression wastes an expensive Lean call. A false
    suppression prunes a valid tactic (mitigated by safety net).

    Args:
        predictions: Predicted failure probabilities [batch, 1].
        targets: Ground truth labels [batch, 1]. 1 = failure, 0 = success.
        w_neg: Weight for missed suppressions (target=1, pred=low).
        w_pos: Weight for false suppressions (target=0, pred=high).

    Returns:
        Scalar loss.
    """
    eps = 1e-7
    predictions = predictions.clamp(eps, 1.0 - eps)
    loss = -(
        w_neg * targets * torch.log(predictions)
        + w_pos * (1 - targets) * torch.log(1 - predictions)
    )
    return loss.mean()


def compute_false_prune_rate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Fraction of valid tactics incorrectly pruned by the censor.

    Args:
        predictions: Predicted failure probabilities [N].
        targets: Ground truth [N]. 1 = failure, 0 = success.
        threshold: Pruning threshold.

    Returns:
        False prune rate (0 to 1). Target: < 0.05.
    """
    with torch.no_grad():
        valid_mask = targets == 0  # tactics that actually work
        if valid_mask.sum() == 0:
            return 0.0
        pruned = predictions[valid_mask] >= threshold
        return float(pruned.float().mean().item())
