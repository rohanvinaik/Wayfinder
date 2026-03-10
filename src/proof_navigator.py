"""
6-bank ternary navigational decoder for proof search.

The core novel module: takes bridge output and produces navigational
coordinates (6 bank directions), anchor predictions, progress estimate,
and critic score. Reuses TernaryLinear from ternary_decoder.py.

Output is a NavOutput used by resolution.py to query the proof network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.nav_contracts import BANK_NAMES, NavOutput
from src.ternary_decoder import TernaryLinear

_DEFAULT_BANKS = list(BANK_NAMES)


class ProofNavigator(nn.Module):
    """6-bank ternary navigational decoder.

    Args:
        input_dim: Bridge output dimension.
        hidden_dim: Internal layer dimension.
        num_anchors: Number of anchor labels for multi-label prediction.
        num_layers: Number of TernaryLinear hidden layers.
        ternary_enabled: Use ternary weights (False = continuous baseline).
        navigable_banks: Which banks to navigate (default: all 6).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_anchors: int = 300,
        num_layers: int = 2,
        ternary_enabled: bool = True,
        navigable_banks: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        self.ternary_enabled = ternary_enabled
        self.navigable_banks = navigable_banks or list(_DEFAULT_BANKS)

        hidden_cls = TernaryLinear if ternary_enabled else nn.Linear

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(hidden_cls(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)

        # 6 direction heads: each outputs logits for {-1, 0, +1}
        self.direction_heads = nn.ModuleDict(
            {
                bank: TernaryLinear(hidden_dim, 3) if ternary_enabled else nn.Linear(hidden_dim, 3)
                for bank in self.navigable_banks
            }
        )

        # Anchor prediction (multi-label, sigmoid)
        self.anchor_head = nn.Linear(hidden_dim, num_anchors)

        # Progress head: estimates remaining proof steps (scalar)
        self.progress_head = nn.Linear(hidden_dim, 1)

        # Critic head: estimates solvability (scalar, sigmoid → [0, 1])
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, bridge_output: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing raw logits for all heads.

        Args:
            bridge_output: [batch, input_dim] from InformationBridge.

        Returns:
            Tuple of (direction_logits, anchor_logits, progress, critic):
                direction_logits: {bank_name: [batch, 3]} for each navigable bank
                anchor_logits: [batch, num_anchors]
                progress: [batch, 1]
                critic: [batch, 1]
        """
        hidden = self.hidden_layers(bridge_output)

        direction_logits = {bank: head(hidden) for bank, head in self.direction_heads.items()}

        return (
            direction_logits,
            self.anchor_head(hidden),
            self.progress_head(hidden),
            torch.sigmoid(self.critic_head(hidden)),
        )

    def predict(self, bridge_output: torch.Tensor) -> NavOutput:
        """Produce a NavOutput for resolution (inference mode).

        Takes argmax on direction heads and applies sigmoid to anchors.
        """
        with torch.no_grad():
            dir_logits, anchor_logits, progress, critic = self.forward(bridge_output)

        direction_map = {0: -1, 1: 0, 2: 1}
        directions: dict[str, int] = {}
        confidences: dict[str, float] = {}
        for bank, logits in dir_logits.items():
            probs = torch.softmax(logits[0], dim=-1)
            idx = int(probs.argmax().item())
            directions[bank] = direction_map[idx]
            confidences[bank] = float(probs[idx].item())

        anchor_probs = torch.sigmoid(anchor_logits[0])
        anchor_scores = {
            str(i): anchor_probs[i].item()
            for i in range(self.num_anchors)
            if anchor_probs[i].item() > 0.1
        }

        return NavOutput(
            directions=directions,
            direction_confidences=confidences,
            anchor_scores=anchor_scores,
            progress=progress[0, 0].item(),
            critic_score=critic[0, 0].item(),
        )
