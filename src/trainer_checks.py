"""Per-step safety checks and logging helpers for the trainer.

All functions are stateless -- they operate on the pipeline/infra containers
passed in from the trainer.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np

from src.pab_tracker import CheckpointData

PABMetricsSnapshot = namedtuple(
    "PABMetricsSnapshot",
    "val_loss tier_accuracies domain_accuracies tactic_accuracies",
)
PABMetricsSnapshot.__new__.__defaults__ = (None, None, None, None)


def check_gradient_health(pipeline: Any) -> tuple[bool, str | None]:
    """Check all gradients for NaN/Inf. Returns (healthy, warning_message)."""
    import torch

    for name, module in [
        ("gate", pipeline.domain_gate),
        ("analyzer", pipeline.goal_analyzer),
        ("bridge", pipeline.bridge),
        ("decoder", pipeline.decoder),
    ]:
        for pname, p in module.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                return False, f"NaN/Inf gradient in {name}.{pname}"
    return True, None


def log_ternary_distribution(decoder: Any) -> dict[str, dict[str, float]]:
    """Log {-1, 0, +1} distribution per TernaryLinear layer."""
    from src.ternary_decoder import TernaryLinear

    dist: dict[str, dict[str, float]] = {}
    for name, module in decoder.named_modules():
        if isinstance(module, TernaryLinear):
            w = module.weight.data
            total = w.numel()
            neg = (w < -0.01).sum().item() / total * 100
            zero = ((w >= -0.01) & (w <= 0.01)).sum().item() / total * 100
            pos = (w > 0.01).sum().item() / total * 100
            dist[name] = {"neg_pct": neg, "zero_pct": zero, "pos_pct": pos}
    return dist


def collect_decoder_weight_signs(decoder: Any) -> np.ndarray | None:
    """Extract sign pattern from TernaryLinear layers."""
    from src.ternary_decoder import TernaryLinear

    signs = []
    for module in decoder.modules():
        if isinstance(module, TernaryLinear):
            signs.append(np.sign(module.weight.data.detach().cpu().numpy()).flatten())
    if not signs:
        return None
    return np.concatenate(signs)


def save_checkpoint(
    checkpoint_dir: Path,
    run_id: str,
    step: int,
    config: dict,
    pipeline: Any,
    infra: Any,
) -> Path:
    """Save model checkpoint with optimizer state."""
    import torch

    path = checkpoint_dir / f"{run_id}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "run_id": run_id,
            "config": config,
            "domain_gate": pipeline.domain_gate.state_dict(),
            "goal_analyzer": pipeline.goal_analyzer.state_dict(),
            "bridge": pipeline.bridge.state_dict(),
            "decoder": pipeline.decoder.state_dict(),
            "composite_loss": infra.composite_loss.state_dict(),
            "optimizer": infra.optimizer.state_dict(),
            "gate_optimizer": (
                infra.gate_optimizer.state_dict()
                if hasattr(infra, "gate_optimizer") and infra.gate_optimizer is not None
                else None
            ),
        },
        path,
    )
    return path


def check_gradient_abort(
    step: int,
    safety: dict,
    pipeline: Any,
    save_fn: Any,
) -> dict | None:
    """Check gradients and return abort dict if NaN/Inf detected."""
    if step % safety["gradient_check_every_n_steps"] != 0:
        return None
    healthy, msg = check_gradient_health(pipeline)
    if healthy:
        return None
    if msg:
        print(f"  WARNING: {msg}")
    if not safety["nan_abort"]:
        return None
    if safety.get("nan_checkpoint_before_abort", True):
        save_fn(step)
    print(f"ABORT: NaN/Inf gradient at step {step}")
    return {"status": "nan_abort", "step": step}


def build_checkpoint_data(
    step: int,
    loss_dict: dict,
    metrics: PABMetricsSnapshot | None,
    last_bridge_features: np.ndarray | None,
    decoder: Any,
) -> CheckpointData:
    """Pure construction of PAB checkpoint data from training state."""
    m = metrics or PABMetricsSnapshot()  # type: ignore[call-arg]
    return CheckpointData(
        step=step,
        train_loss=loss_dict["L_total"],
        val_loss=m.val_loss,
        loss_components={
            "ce": loss_dict.get("L_ce", 0.0),
            "margin": loss_dict.get("L_margin", 0.0),
            "repair": loss_dict.get("L_repair", 0.0),
        },
        adaptive_weights={
            k: v for k, v in loss_dict.items() if k.startswith("w_") and isinstance(v, float)
        },
        tier_accuracies=m.tier_accuracies or {},
        bottleneck_embeddings=last_bridge_features,
        decoder_weight_signs=collect_decoder_weight_signs(decoder),
        domain_accuracies=m.domain_accuracies or {},
        tactic_accuracies=m.tactic_accuracies or {},
    )


def record_pab_checkpoint(
    step: int,
    loss_dict: dict,
    config: dict,
    infra: Any,
    last_bridge_features: np.ndarray | None,
    decoder: Any,
    metrics: PABMetricsSnapshot | None = None,
) -> dict | None:
    """Record a PAB checkpoint if due. Returns early-exit dict or None."""
    tracker = infra.pab_tracker
    if tracker is None:
        return None
    pab_cfg = config.get("pab", {})
    interval = pab_cfg.get("checkpoint_interval", 50)
    if step % interval != 0:
        return None

    data = build_checkpoint_data(step, loss_dict, metrics, last_bridge_features, decoder)
    tracker.record(data)
    if pab_cfg.get("early_exit_enabled", False) and tracker.should_early_exit(step):
        print(f"  PAB early exit triggered at step {step}")
        return {"status": "pab_early_exit", "step": step}
    return None
