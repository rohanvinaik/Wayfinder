"""Train bank-cluster specialist navigators for Wayfinder v2.

Each specialist has its own bridge + navigator for assigned banks.
PAB stability is tracked per-specialist to guide decomposition.

Usage:
    python -m scripts.train_specialist \
        --config configs/wayfinder_v2.yaml --specialist A --run-id SPEC-A-001
    python -m scripts.train_specialist \
        --config configs/wayfinder_v2.yaml --specialist B --run-id SPEC-B-001
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.train_navigator import get_curriculum_phase, load_anchor_labels
from scripts.train_targets import (
    build_anchor_targets,
    build_critic_targets,
    build_direction_targets,
    build_progress_targets,
)
from src.data import NavigationalDataset
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.losses import NavigationalLoss
from src.pab_tracker import CheckpointData, PABTracker
from src.specialist_navigator import (
    SPECIALIST_A_BANKS,
    SPECIALIST_B_BANKS,
    SpecialistNavigator,
)

# Default specialist bank assignments
_SPECIALIST_BANKS: dict[str, list[str]] = {
    "A": SPECIALIST_A_BANKS,
    "B": SPECIALIST_B_BANKS,
}


@dataclass
class _TrainCtx:
    """Bundles training infrastructure to reduce argument passing."""

    encoder: GoalEncoder
    analyzer: GoalAnalyzer
    specialist: SpecialistNavigator
    banks: list[str]
    loss_fn: NavigationalLoss
    anchor_labels: list[str]
    optimizer: torch.optim.Optimizer
    all_params: list
    max_grad_norm: float
    device: str


def _build_specialist(
    specialist_name: str, config: dict, device: str
) -> tuple[GoalEncoder, GoalAnalyzer, SpecialistNavigator, list[str]]:
    """Build frozen perception + specialist from config."""
    specialist_cfg = config.get("specialists", {}).get(specialist_name, {})
    banks = specialist_cfg.get("banks", _SPECIALIST_BANKS.get(specialist_name, SPECIALIST_A_BANKS))

    enc_cfg = config.get("model", {}).get("encoder", {})
    encoder = GoalEncoder.from_config(enc_cfg, device=device)
    encoder.ensure_loaded()

    ana_cfg = config.get("model", {}).get("goal_analyzer", {})
    analyzer = GoalAnalyzer(
        input_dim=encoder.output_dim,
        feature_dim=ana_cfg.get("feature_dim", 256),
    ).to(device)
    for p in analyzer.parameters():
        p.requires_grad = False

    nav_cfg = config.get("model", {}).get("navigator", {})
    br_cfg = config.get("model", {}).get("bridge", {})
    specialist = SpecialistNavigator(
        name=specialist_name,
        banks=banks,
        feature_dim=ana_cfg.get("feature_dim", 256),
        bridge_dim=specialist_cfg.get("bridge_dim", br_cfg.get("bridge_dim", 128)),
        hidden_dim=specialist_cfg.get("hidden_dim", nav_cfg.get("hidden_dim", 256)),
        num_anchors=nav_cfg.get("num_anchors", 18729),
        num_layers=specialist_cfg.get("num_layers", nav_cfg.get("num_layers", 2)),
        ternary_enabled=nav_cfg.get("ternary_enabled", True),
        history_dim=specialist_cfg.get("history_dim", br_cfg.get("history_dim", 64)),
    ).to(device)

    return encoder, analyzer, specialist, banks


def _run_train_step(ctx: _TrainCtx, batch: list) -> dict:
    """Execute a single training step, return scalar loss dict."""
    ctx.optimizer.zero_grad()

    goal_states = [ex.goal_state for ex in batch]
    embeddings = ctx.encoder.encode(goal_states)
    features, _, _ = ctx.analyzer(embeddings)
    dir_logits, anchor_logits, progress_pred, critic_pred = ctx.specialist.forward(features)

    loss_dict = ctx.loss_fn(
        direction_logits=dir_logits,
        direction_targets=build_direction_targets(batch, ctx.banks, ctx.device),
        anchor_logits=anchor_logits,
        anchor_targets=build_anchor_targets(batch, ctx.anchor_labels, ctx.device),
        progress_pred=progress_pred,
        progress_target=build_progress_targets(batch, ctx.device),
        critic_pred=critic_pred,
        critic_target=build_critic_targets(batch, ctx.device),
    )

    loss_dict["L_total"].backward()
    torch.nn.utils.clip_grad_norm_(ctx.all_params, ctx.max_grad_norm)
    ctx.optimizer.step()

    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in loss_dict.items()
        if k != "bank_losses"
    }


def _should_abort(step: int, loss_val: dict, config: dict, dry_run: bool) -> str | None:
    """Return abort reason string, or None to continue."""
    if dry_run:
        return "dry_run"
    if config.get("safety", {}).get("nan_abort") and any(
        np.isnan(v) for v in loss_val.values() if isinstance(v, float)
    ):
        return f"NaN at step {step}"
    return None


def _log_step(step: int, max_iters: int, loss_val: dict, tracker: PABTracker) -> None:
    """Log progress and record PAB checkpoint every 50 steps."""
    if step % 50 != 0:
        return
    print(
        f"  Step {step}/{max_iters}: L={loss_val['L_total']:.4f} nav={loss_val.get('L_nav', 0):.4f}"
    )
    tracker.record(
        CheckpointData(
            step=step,
            train_loss=loss_val["L_total"],
            tier_accuracies={"tier1": 0.0, "tier2": 0.0, "tier3": 0.0},
        )
    )


def _build_train_ctx(
    encoder: GoalEncoder,
    analyzer: GoalAnalyzer,
    specialist: SpecialistNavigator,
    banks: list[str],
    config: dict,
    device: str,
) -> _TrainCtx:
    """Build training context with optimizer and loss function."""
    loss_fn = NavigationalLoss().to(device)
    train_cfg = config.get("training", {})

    all_params = [p for p in specialist.parameters() if p.requires_grad]
    all_params.extend(p for p in loss_fn.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        all_params,
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    return _TrainCtx(
        encoder=encoder,
        analyzer=analyzer,
        specialist=specialist,
        banks=banks,
        loss_fn=loss_fn,
        anchor_labels=load_anchor_labels(config),
        optimizer=optimizer,
        all_params=all_params,
        max_grad_norm=config.get("safety", {}).get("max_grad_norm", 1.0),
        device=device,
    )


def _train_loop(
    encoder: GoalEncoder,
    analyzer: GoalAnalyzer,
    specialist: SpecialistNavigator,
    banks: list[str],
    config: dict,
    run_id: str,
    device: str,
    dry_run: bool,
) -> tuple[int, list[dict], PABTracker]:
    """Run training loop, returning (final_step, losses, tracker)."""
    ctx = _build_train_ctx(encoder, analyzer, specialist, banks, config, device)
    train_cfg = config.get("training", {})
    nav_train_path = Path(config.get("data", {}).get("nav_train", "data/nav_train.jsonl"))
    max_iters = train_cfg.get("max_iterations", 5000)
    batch_size = train_cfg.get("batch_size", 16)
    tracker = PABTracker(experiment_id=run_id, checkpoint_interval=50)

    step = 0
    all_losses: list[dict] = []
    current_phase = ""
    dataset: NavigationalDataset | None = None

    while step < max_iters:
        phase_name, max_steps = get_curriculum_phase(step, config)
        if phase_name != current_phase:
            current_phase = phase_name
            dataset = NavigationalDataset(nav_train_path, max_steps=max_steps)
            print(f"\n  Phase {phase_name}: {len(dataset)} examples")
            if len(dataset) == 0:
                step += 1
                continue

        assert dataset is not None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: b,
            num_workers=0,
        )

        for batch in loader:
            if step >= max_iters:
                break
            loss_val = _run_train_step(ctx, batch)
            all_losses.append({"step": step, **loss_val})
            step += 1
            _log_step(step, max_iters, loss_val, tracker)

            abort = _should_abort(step, loss_val, config, dry_run)
            if abort:
                print(f"  {abort}, stopping.")
                return step, all_losses, tracker

    return step, all_losses, tracker


def train_specialist(
    config: dict,
    specialist_name: str,
    run_id: str,
    device: str,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """Train a single specialist navigator."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path(config.get("logging", {}).get("run_dir", "runs/")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.get("logging", {}).get("checkpoint_dir", "models/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    encoder, analyzer, specialist, banks = _build_specialist(specialist_name, config, device)

    print(f"Training specialist {specialist_name}: {run_id}")
    print(f"  Banks: {banks}")

    start = time.time()
    final_step, all_losses, tracker = _train_loop(
        encoder,
        analyzer,
        specialist,
        banks,
        config,
        run_id,
        device,
        dry_run,
    )
    elapsed = time.time() - start

    # Save checkpoint
    ckpt_path = ckpt_dir / f"{run_id}_specialist_{specialist_name}_step{final_step}.pt"
    torch.save(
        {
            "step": final_step,
            "specialist_name": specialist_name,
            "banks": banks,
            "specialist": specialist.state_dict(),
            "config": config,
        },
        ckpt_path,
    )

    # Save training log
    log_path = run_dir / f"{run_id}_training_log.jsonl"
    with open(log_path, "w") as f:
        for entry in all_losses:
            f.write(json.dumps(entry) + "\n")

    # PAB profile
    profile = tracker.finalize()
    profile_path = run_dir / f"{run_id}_pab_profile.json"
    profile.save(profile_path)

    print(f"\nSpecialist {specialist_name} training complete: {final_step} steps in {elapsed:.1f}s")
    print(f"Checkpoint: {ckpt_path}")
    print(f"PAB regime: {profile.summary.stability_regime}")

    return {
        "status": "complete",
        "specialist": specialist_name,
        "banks": banks,
        "steps": final_step,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "pab_regime": profile.summary.stability_regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train specialist navigator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--specialist", type=str, required=True, choices=["A", "B", "B1", "B2"])
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = train_specialist(
        config, args.specialist, args.run_id, args.device, args.seed, args.dry_run
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
