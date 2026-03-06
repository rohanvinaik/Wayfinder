"""
Ternary structural decoder — {-1, 0, +1} weight layers for discrete decisions.

Uses Straight-Through Estimator (STE) for training: fp32 shadow weights with
quantized forward pass. Gradients flow through quantization via STE.

Training memory is dominated by fp32 shadow weights + Adam states,
NOT by the ternary weights themselves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Quantize continuous weights to {-1, 0, +1}.

    Uses threshold-based quantization with per-channel scaling.
    STE: gradient of this function is identity (straight-through).

    Args:
        weights: Continuous fp32 weights.
        eps: Small constant for numerical stability.

    Returns:
        Ternary tensor with values in {-1, 0, +1}.
    """
    threshold = 0.7 * weights.abs().mean(dim=-1, keepdim=True)

    quantized = torch.zeros_like(weights)
    quantized[weights > threshold] = 1.0
    quantized[weights < -threshold] = -1.0

    # STE: forward uses quantized, backward passes gradient through as identity
    return weights + (quantized - weights).detach()


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights via STE.

    Maintains fp32 shadow weights for gradient computation.
    Forward pass uses quantized {-1, 0, +1} weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = ternary_quantize(self.weight)
        return F.linear(x, q_weight, self.bias)


class TernaryDecoder(nn.Module):
    """Multi-layer ternary decoder for Tier 1 and Tier 2 token prediction.

    Args:
        input_dim: Bridge output dimension.
        hidden_dim: Internal layer dimension.
        tier1_vocab_size: Number of Tier 1 tactic/structural tokens.
        tier2_vocab_size: Number of Tier 2 premise/argument tokens.
        num_layers: Number of TernaryLinear layers.
        ternary_enabled: Use ternary weights (False = continuous baseline).
        partial_ternary: Ternary hidden layers, continuous heads.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        tier1_vocab_size: int = 0,
        tier2_vocab_size: int = 0,
        num_layers: int = 2,
        ternary_enabled: bool = True,
        partial_ternary: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tier1_vocab_size = tier1_vocab_size
        self.tier2_vocab_size = tier2_vocab_size
        self.num_layers = num_layers
        self.ternary_enabled = ternary_enabled
        self.partial_ternary = partial_ternary

        hidden_cls = TernaryLinear if ternary_enabled else nn.Linear
        head_cls = nn.Linear if (not ternary_enabled or partial_ternary) else TernaryLinear

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(hidden_cls(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)

        self.tier1_head: nn.Module | None = None
        self.tier2_head: nn.Module | None = None

        if tier1_vocab_size > 0:
            self.tier1_head = head_cls(hidden_dim, tier1_vocab_size)
        if tier2_vocab_size > 0:
            self.tier2_head = head_cls(hidden_dim, tier2_vocab_size)

    def forward(self, bridge_output: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.layers(bridge_output)
        result: dict[str, torch.Tensor] = {}
        if self.tier1_head is not None:
            result["tier1_logits"] = self.tier1_head(hidden)
        if self.tier2_head is not None:
            result["tier2_logits"] = self.tier2_head(hidden)
        return result
