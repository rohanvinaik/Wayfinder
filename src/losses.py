"""
Composite loss with adaptive UW-SO weighting.

L_total = L_ce + L_margin + L_repair, with uncertainty-weighted
soft-optimal (UW-SO) adaptive balancing. Each component has a
learnable log-variance parameter (sigma) that the optimizer discovers.

Separate L_ood for domain gate training.
NavigationalLoss for Wayfinder 6-bank navigational training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """Adaptive multi-task loss for Balanced Sashimi training.

    Components:
        L_ce: Cross-entropy over tier token predictions.
        L_margin: Contrastive margin loss from negative bank.
        L_repair: Verification-failure-weighted penalty.

    Each has a learnable log-sigma for UW-SO adaptive weighting.
    """

    def __init__(self, initial_log_sigma: float = 0.0, margin: float = 0.5) -> None:
        super().__init__()
        self.log_sigma_ce = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_margin = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_repair = nn.Parameter(torch.tensor(initial_log_sigma))
        self.margin = float(margin)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        positive_features: torch.Tensor | None = None,
        negative_features: torch.Tensor | None = None,
        repair_weights: torch.Tensor | None = None,
        negative_targets: torch.Tensor | None = None,
        margin: float | None = None,
    ) -> dict[str, Any]:
        L_ce = F.cross_entropy(logits, targets)

        margin_value = self.margin if margin is None else float(margin)
        if margin_value <= 0:
            L_margin = torch.tensor(0.0, device=logits.device)
        elif negative_targets is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            pos_logp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            neg_logp = log_probs.gather(1, negative_targets.unsqueeze(1)).squeeze(1)
            L_margin = F.relu(margin_value - pos_logp + neg_logp).mean()
        elif positive_features is not None and negative_features is not None:
            distance = 1.0 - F.cosine_similarity(positive_features, negative_features, dim=-1)
            L_margin = F.relu(margin_value - distance).mean()
        else:
            L_margin = torch.tensor(0.0, device=logits.device)

        if repair_weights is not None:
            per_sample = F.cross_entropy(logits, targets, reduction="none")
            L_repair = (per_sample * repair_weights).mean()
        else:
            L_repair = torch.tensor(0.0, device=logits.device)

        precision_ce = torch.exp(-self.log_sigma_ce)
        precision_margin = torch.exp(-self.log_sigma_margin)
        precision_repair = torch.exp(-self.log_sigma_repair)
        precision_sum = precision_ce + precision_margin + precision_repair + 1e-8

        L_total = (
            precision_ce * L_ce
            + self.log_sigma_ce
            + precision_margin * L_margin
            + self.log_sigma_margin
            + precision_repair * L_repair
            + self.log_sigma_repair
        )

        return {
            "L_total": L_total,
            "L_ce": L_ce,
            "L_margin": L_margin,
            "L_repair": L_repair,
            "sigma_ce": torch.exp(self.log_sigma_ce),
            "sigma_margin": torch.exp(self.log_sigma_margin),
            "sigma_repair": torch.exp(self.log_sigma_repair),
            "w_ce": precision_ce / precision_sum,
            "w_margin": precision_margin / precision_sum,
            "w_repair": precision_repair / precision_sum,
        }


class OODLoss(nn.Module):
    """Binary cross-entropy loss for domain gate training."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)


class NavigationalLoss(nn.Module):
    """Adaptive multi-task loss for Wayfinder navigational training.

    Components (UW-SO weighted):
        L_nav: CrossEntropy on each bank direction head (6 terms, summed)
        L_anchor: BCE on multi-label anchor prediction
        L_progress: MSE on normalized remaining steps (soft target)
        L_critic: MSE on solvability estimate (soft target, NOT binary)
    """

    def __init__(self, initial_log_sigma: float = 0.0) -> None:
        super().__init__()
        self.log_sigma_nav = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_anchor = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_progress = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_critic = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_move = nn.Parameter(torch.tensor(initial_log_sigma))

    def forward(
        self,
        direction_logits: dict[str, torch.Tensor],
        direction_targets: dict[str, torch.Tensor],
        anchor_logits: torch.Tensor | None = None,
        anchor_targets: torch.Tensor | None = None,
        progress_pred: torch.Tensor | None = None,
        progress_target: torch.Tensor | None = None,
        critic_pred: torch.Tensor | None = None,
        critic_target: torch.Tensor | None = None,
        move_logits: dict[str, torch.Tensor] | None = None,
        move_targets: dict[str, torch.Tensor] | None = None,
        move_masks: dict[str, torch.Tensor] | None = None,
        move_target_types: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        device = next(iter(direction_logits.values())).device

        # L_nav: sum of per-bank cross-entropy
        bank_losses: dict[str, torch.Tensor] = {}
        for bank, logits in direction_logits.items():
            if bank in direction_targets:
                bank_losses[bank] = F.cross_entropy(logits, direction_targets[bank])
        L_nav = sum(bank_losses.values()) if bank_losses else torch.tensor(0.0, device=device)

        # L_anchor: multi-label BCE
        if anchor_logits is not None and anchor_targets is not None:
            L_anchor = F.binary_cross_entropy_with_logits(anchor_logits, anchor_targets)
        else:
            L_anchor = torch.tensor(0.0, device=device)

        # L_progress: MSE on normalized remaining steps
        if progress_pred is not None and progress_target is not None:
            L_progress = F.mse_loss(progress_pred.squeeze(-1), progress_target)
        else:
            L_progress = torch.tensor(0.0, device=device)

        # L_critic: MSE on solvability (soft, not binary)
        if critic_pred is not None and critic_target is not None:
            L_critic = F.mse_loss(critic_pred.squeeze(-1), critic_target)
        else:
            L_critic = torch.tensor(0.0, device=device)

        move_losses: dict[str, torch.Tensor] = {}
        if move_logits is not None and move_targets is not None and move_masks is not None:
            target_types = move_target_types or {}
            for name, logits in move_logits.items():
                if name not in move_targets or name not in move_masks:
                    continue
                mask = move_masks[name]
                if mask.numel() == 0 or not bool(mask.any().item()):
                    continue
                if target_types.get(name) == "multiclass":
                    move_losses[name] = F.cross_entropy(logits[mask], move_targets[name][mask])
                elif target_types.get(name) == "multilabel":
                    move_losses[name] = F.binary_cross_entropy_with_logits(
                        logits[mask],
                        move_targets[name][mask],
                    )
        L_move = sum(move_losses.values()) if move_losses else torch.tensor(0.0, device=device)

        # UW-SO adaptive weighting
        p_nav = torch.exp(-self.log_sigma_nav)
        p_anchor = torch.exp(-self.log_sigma_anchor)
        p_progress = torch.exp(-self.log_sigma_progress)
        p_critic = torch.exp(-self.log_sigma_critic)
        p_move = torch.exp(-self.log_sigma_move)
        p_sum = p_nav + p_anchor + p_progress + p_critic + p_move + 1e-8

        L_total = (
            p_nav * L_nav
            + self.log_sigma_nav
            + p_anchor * L_anchor
            + self.log_sigma_anchor
            + p_progress * L_progress
            + self.log_sigma_progress
            + p_critic * L_critic
            + self.log_sigma_critic
            + p_move * L_move
            + self.log_sigma_move
        )

        return {
            "L_total": L_total,
            "L_nav": L_nav,
            "L_anchor": L_anchor,
            "L_progress": L_progress,
            "L_critic": L_critic,
            "L_move": L_move,
            "bank_losses": bank_losses,
            "move_losses": move_losses,
            "w_nav": p_nav / p_sum,
            "w_anchor": p_anchor / p_sum,
            "w_progress": p_progress / p_sum,
            "w_critic": p_critic / p_sum,
            "w_move": p_move / p_sum,
        }
