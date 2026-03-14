"""
Bank-cluster specialist navigator — EXECUTION slot (Slot 4) for Wayfinder v2.

Decomposes the monolithic ProofNavigator into bank-cluster specialists,
each with its own bridge and hidden layers. Specialists only produce
direction heads for their assigned banks, eliminating the shared-bridge
composition gap γ that made v1 chaotic (NAV-001/002 stability_mean > 0.30).

Two default specialists (from NAV-002 bank difficulty analysis):
  Navigator-A (easy): DOMAIN, CONTEXT — Regime A, high symmetry
  Navigator-B (hard): STRUCTURE, AUTOMATION, DEPTH, DECOMPOSITION — Regime B

Fusion combines specialist outputs: max-pool anchors, confidence-weighted
progress, min critic. Directions are concatenated (no overlap between specialists).

See DESIGN §10.4 for specification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.bridge import InformationBridge
from src.nav_contracts import NavOutput
from src.proof_navigator import ProofNavigator
from src.som_contracts import ExecutionOutput

# Default specialist scopes from DESIGN §10.4
SPECIALIST_A_BANKS: list[str] = ["domain", "context"]
SPECIALIST_B_BANKS: list[str] = ["structure", "automation", "depth", "decomposition"]


class SpecialistNavigator(nn.Module):
    """Bank-cluster specialist with its own bridge and navigator.

    Each specialist has:
      - Own InformationBridge (eliminates shared-bridge γ)
      - Own ProofNavigator (hidden layers + direction heads for assigned banks only)
      - Shared: encoder (frozen), GoalAnalyzer (frozen)
      - Anchor logits, progress, critic produced independently then fused

    Args:
        name: Specialist identifier (e.g., "A", "B").
        banks: List of bank names this specialist is responsible for.
        feature_dim: Input feature dimension from GoalAnalyzer (default 256).
        bridge_dim: Bridge bottleneck dimension (default 128).
        hidden_dim: Navigator hidden dimension (default 256).
        num_anchors: Number of anchor labels (default 18729).
        num_layers: Number of TernaryLinear hidden layers (default 2).
        ternary_enabled: Use ternary weights (default True).
        history_dim: Proof history embedding dimension (default 64).
    """

    def __init__(
        self,
        name: str,
        banks: list[str],
        feature_dim: int = 256,
        bridge_dim: int = 128,
        hidden_dim: int = 256,
        num_anchors: int = 18729,
        num_layers: int = 2,
        ternary_enabled: bool = True,
        history_dim: int = 64,
    ) -> None:
        super().__init__()
        self.name = name
        self.banks = list(banks)

        self.bridge = InformationBridge(
            input_dim=feature_dim,
            bridge_dim=bridge_dim,
            history_dim=history_dim,
        )

        self.navigator = ProofNavigator(
            input_dim=bridge_dim,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            num_layers=num_layers,
            ternary_enabled=ternary_enabled,
            navigable_banks=banks,
        )

    def forward(
        self,
        features: torch.Tensor,
        proof_history: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through specialist bridge + navigator.

        Args:
            features: [batch, feature_dim] from GoalAnalyzer.
            proof_history: [batch, history_dim] optional proof history.

        Returns:
            Same as ProofNavigator.forward():
                (direction_logits, anchor_logits, progress, critic)
        """
        bridge_out = self.bridge(features, proof_history)
        return self.navigator.forward(bridge_out)

    def predict(
        self,
        features: torch.Tensor,
        proof_history: torch.Tensor | None = None,
    ) -> NavOutput:
        """Inference mode: produce NavOutput for resolution."""
        bridge_out = self.bridge(features, proof_history)
        return self.navigator.predict(bridge_out)


def fuse_specialist_outputs(outputs: dict[str, NavOutput]) -> ExecutionOutput:
    """Fuse outputs from multiple specialists into a single ExecutionOutput.

    Fusion rules (from DESIGN §10.4):
      - Directions: concatenate (no overlap — each specialist owns its banks)
      - Anchor scores: max-pool (if ANY specialist thinks it's relevant, consider it)
      - Progress: confidence-weighted average
      - Critic: min (conservative — if either thinks it's hard, trust that)

    Args:
        outputs: Dict mapping specialist name to NavOutput.

    Returns:
        Fused ExecutionOutput combining all specialist outputs.
    """
    all_directions: dict[str, int] = {}
    all_confidences: dict[str, float] = {}
    all_anchor_scores: dict[str, float] = {}
    progress_sum = 0.0
    confidence_sum = 0.0
    min_critic = 1.0

    for nav_out in outputs.values():
        # Directions: concatenate (no overlap)
        all_directions.update(nav_out.directions)
        all_confidences.update(nav_out.direction_confidences)

        # Anchors: max-pool
        for anchor, score in nav_out.anchor_scores.items():
            if anchor not in all_anchor_scores or score > all_anchor_scores[anchor]:
                all_anchor_scores[anchor] = score

        # Progress: confidence-weighted
        confs = nav_out.direction_confidences.values()
        mean_conf = sum(confs) / max(len(nav_out.direction_confidences), 1)
        progress_sum += nav_out.progress * mean_conf
        confidence_sum += mean_conf

        # Critic: min
        min_critic = min(min_critic, nav_out.critic_score)

    fused_progress = progress_sum / (confidence_sum + 1e-8)

    return ExecutionOutput(
        directions=all_directions,
        direction_confidences=all_confidences,
        anchor_logits=torch.zeros(1),  # Placeholder; scores are in NavOutput dict
        progress=fused_progress,
        critic=min_critic,
    )


def fuse_to_nav_output(outputs: dict[str, NavOutput]) -> NavOutput:
    """Fuse specialist NavOutputs into a single NavOutput for resolution compatibility.

    Preserves the NavOutput interface so fused output can be passed
    directly to resolve() without changes to the v1 resolution pipeline.
    """
    fused = fuse_specialist_outputs(outputs)

    # Merge all anchor scores with max-pool
    all_anchor_scores: dict[str, float] = {}
    for nav_out in outputs.values():
        for anchor, score in nav_out.anchor_scores.items():
            if anchor not in all_anchor_scores or score > all_anchor_scores[anchor]:
                all_anchor_scores[anchor] = score

    return NavOutput(
        directions=fused.directions,
        direction_confidences=fused.direction_confidences,
        anchor_scores=all_anchor_scores,
        progress=fused.progress,
        critic_score=fused.critic,
    )


class ExecutionSlot(nn.Module):
    """Manages specialist routing and fusion for the EXECUTION slot.

    Wraps multiple SpecialistNavigators, runs them all on the same input,
    and fuses their outputs. The key SoM property: specialists share
    data (features, proof history) not weights (Invariant #13).

    Args:
        specialists: Dict mapping specialist name to SpecialistNavigator.
    """

    def __init__(self, specialists: dict[str, SpecialistNavigator]) -> None:
        super().__init__()
        self.specialists = nn.ModuleDict(specialists)

    def _get_specialist(self, name: str) -> SpecialistNavigator:
        """Get a specialist by name with correct type."""
        module = self.specialists[name]
        assert isinstance(module, SpecialistNavigator)  # noqa: S101
        return module

    def forward(
        self,
        features: torch.Tensor,
        proof_history: torch.Tensor | None = None,
    ) -> dict[str, tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through all specialists (for training).

        Returns dict mapping specialist name to raw forward outputs.
        """
        results = {}
        for name in self.specialists:
            results[name] = self._get_specialist(name).forward(features, proof_history)
        return results

    def predict(
        self,
        features: torch.Tensor,
        proof_history: torch.Tensor | None = None,
    ) -> NavOutput:
        """Inference mode: run all specialists and fuse outputs.

        Returns a single fused NavOutput compatible with resolve().
        """
        outputs: dict[str, NavOutput] = {}
        for name in self.specialists:
            outputs[name] = self._get_specialist(name).predict(features, proof_history)
        return fuse_to_nav_output(outputs)

    @staticmethod
    def from_config(config: dict) -> ExecutionSlot:
        """Build ExecutionSlot from v2 config specialist definitions.

        Config format::

            specialists:
              A:
                banks: [domain, context]
                bridge_dim: 128
                hidden_dim: 256
              B:
                banks: [structure, automation, depth, decomposition]
                bridge_dim: 128
                hidden_dim: 256
        """
        specialist_configs = config.get("specialists", {})
        model_cfg = config.get("model", {})
        feature_dim = model_cfg.get("goal_analyzer", {}).get("feature_dim", 256)
        num_anchors = model_cfg.get("navigator", {}).get("num_anchors", 18729)
        ternary_enabled = model_cfg.get("navigator", {}).get("ternary_enabled", True)
        history_dim = model_cfg.get("bridge", {}).get("history_dim", 64)

        specialists: dict[str, SpecialistNavigator] = {}
        for name, spec_cfg in specialist_configs.items():
            specialists[name] = SpecialistNavigator(
                name=name,
                banks=spec_cfg["banks"],
                feature_dim=feature_dim,
                bridge_dim=spec_cfg.get("bridge_dim", 128),
                hidden_dim=spec_cfg.get("hidden_dim", 256),
                num_anchors=num_anchors,
                num_layers=spec_cfg.get("num_layers", 2),
                ternary_enabled=ternary_enabled,
                history_dim=history_dim,
            )

        return ExecutionSlot(specialists)
