"""Shared builders for Wayfinder navigational models.

Centralizes encoder-aware module construction so scripts do not need to guess
embedding dimensions, model family, or checkpoint-time model settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from src.bridge import InformationBridge
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.proof_navigator import ProofNavigator


def resolve_model_config(
    config: dict[str, Any], checkpoint: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Prefer the checkpoint's model section when reconstructing saved modules."""
    if checkpoint is not None:
        checkpoint_config = checkpoint.get("config")
        if isinstance(checkpoint_config, dict):
            checkpoint_model = checkpoint_config.get("model")
            if isinstance(checkpoint_model, dict):
                return deepcopy(checkpoint_model)
    return deepcopy(config["model"])


def build_navigational_modules(
    model_config: dict[str, Any], device: str
) -> dict[str, torch.nn.Module]:
    """Build encoder/analyzer/bridge/navigator from a resolved model config."""
    enc_cfg = model_config["encoder"]
    encoder = GoalEncoder.from_config(enc_cfg, device=device).to(device)
    encoder.ensure_loaded()

    ana_cfg = model_config["goal_analyzer"]
    br_cfg = model_config["bridge"]
    nav_cfg = model_config["navigator"]

    analyzer = GoalAnalyzer(
        input_dim=encoder.output_dim,
        feature_dim=ana_cfg["feature_dim"],
        num_anchors=ana_cfg.get("num_anchors", 300),
        navigable_banks=ana_cfg.get("navigable_banks"),
    ).to(device)
    bridge = InformationBridge(
        input_dim=ana_cfg["feature_dim"],
        bridge_dim=br_cfg["bridge_dim"],
        history_dim=br_cfg.get("history_dim", 0),
    ).to(device)
    navigator = ProofNavigator(
        input_dim=br_cfg["bridge_dim"],
        hidden_dim=nav_cfg["hidden_dim"],
        num_anchors=nav_cfg["num_anchors"],
        num_layers=nav_cfg["num_layers"],
        ternary_enabled=nav_cfg.get("ternary_enabled", True),
        navigable_banks=nav_cfg.get("navigable_banks"),
    ).to(device)

    return {
        "encoder": encoder,
        "analyzer": analyzer,
        "bridge": bridge,
        "navigator": navigator,
    }


def load_navigational_checkpoint(
    checkpoint_path,
    config: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, torch.nn.Module]]:
    """Load a navigational checkpoint and rebuild the matching module stack."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec B614 — trusted local checkpoints
    modules = build_navigational_modules(resolve_model_config(config, checkpoint), device)

    for name, module in modules.items():
        if name in checkpoint.get("modules", {}):
            module.load_state_dict(checkpoint["modules"][name])
        module.eval()

    return checkpoint, modules
