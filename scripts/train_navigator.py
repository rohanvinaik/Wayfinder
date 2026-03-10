"""
Train the Wayfinder navigational proof search pipeline.

Curriculum-based training with three phases:
  Phase A (warmup): 1-2 step proofs only
  Phase B (growth): ≤5 step proofs
  Phase C (full): all proofs, oversampling medium difficulty

Uses NavigationalLoss (UW-SO weighted), NavigationalDataset with
curriculum filtering, and PABTracker for process-aware benchmarking.

Usage:
    python scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-001
    python scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-001 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.train_targets import (
    build_anchor_targets,
    build_critic_targets,
    build_direction_targets,
    build_progress_targets,
    compute_nav_accuracy,
)
from src.bridge import InformationBridge
from src.data import NavigationalDataset
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.losses import NavigationalLoss
from src.nav_contracts import BANK_NAMES
from src.pab_tracker import CheckpointData, PABTracker
from src.proof_navigator import ProofNavigator


@dataclass
class TrainContext:
    """Bundles all shared training state to reduce argument passing."""

    modules: dict
    banks: list[str]
    anchor_labels: list[str]
    loss_fn: NavigationalLoss
    optimizer: torch.optim.Optimizer
    tracker: PABTracker
    device: str
    config: dict
    paths: dict = field(default_factory=dict)  # run_dir, ckpt_dir, log_path


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict, device: str) -> dict:
    """Build neural pipeline components from config."""
    enc_cfg = config["model"]["encoder"]
    encoder = GoalEncoder(
        model_name=enc_cfg.get("type", "byt5-small"),
        output_dim=enc_cfg.get("output_dim"),
        frozen=enc_cfg.get("frozen", True),
    )

    ana_cfg = config["model"]["goal_analyzer"]
    analyzer = GoalAnalyzer(
        input_dim=enc_cfg["output_dim"],
        feature_dim=ana_cfg["feature_dim"],
        num_anchors=ana_cfg.get("num_anchors", 300),
        navigable_banks=ana_cfg.get("navigable_banks"),
    )

    br_cfg = config["model"]["bridge"]
    bridge = InformationBridge(
        input_dim=ana_cfg["feature_dim"],
        bridge_dim=br_cfg["bridge_dim"],
        history_dim=br_cfg.get("history_dim", 0),
    )

    nav_cfg = config["model"]["navigator"]
    navigator = ProofNavigator(
        input_dim=br_cfg["bridge_dim"],
        hidden_dim=nav_cfg["hidden_dim"],
        num_anchors=nav_cfg["num_anchors"],
        num_layers=nav_cfg["num_layers"],
        ternary_enabled=nav_cfg.get("ternary_enabled", True),
        navigable_banks=nav_cfg.get("navigable_banks"),
    )

    modules = {
        "encoder": encoder.to(device),
        "analyzer": analyzer.to(device),
        "bridge": bridge.to(device),
        "navigator": navigator.to(device),
    }
    return modules


def load_anchor_labels(config: dict) -> list[str]:
    """Load anchor label list from config path or generate indices."""
    anchor_path = config.get("data", {}).get("anchor_labels")
    if anchor_path and Path(anchor_path).exists():
        with open(anchor_path) as f:
            return json.load(f)
    num_anchors = config["model"]["navigator"]["num_anchors"]
    return [str(i) for i in range(num_anchors)]


def train_step(batch: list, ctx: TrainContext, max_grad_norm: float) -> dict[str, float]:
    """Execute a single training step."""
    ctx.optimizer.zero_grad()

    goal_states = [ex.goal_state for ex in batch]
    embeddings = ctx.modules["encoder"].encode(goal_states)
    features, _, _ = ctx.modules["analyzer"](embeddings)
    bridge_out = ctx.modules["bridge"](features)
    dir_logits, anchor_logits, progress_pred, critic_pred = ctx.modules["navigator"](bridge_out)

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
    torch.nn.utils.clip_grad_norm_(
        [p for m in ctx.modules.values() for p in m.parameters() if p.requires_grad],
        max_grad_norm,
    )
    ctx.optimizer.step()

    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in loss_dict.items()
        if k != "bank_losses"
    }


def get_curriculum_phase(step: int, config: dict) -> tuple[str, int | None]:
    """Determine current curriculum phase and max_steps filter."""
    curriculum = config.get("training", {}).get("curriculum", {})
    phase_a = curriculum.get("phase_a", {})
    phase_b = curriculum.get("phase_b", {})

    a_end = phase_a.get("iterations", 500)
    b_end = a_end + phase_b.get("iterations", 1000)

    if step < a_end:
        return "A", phase_a.get("max_steps", 2)
    if step < b_end:
        return "B", phase_b.get("max_steps", 5)
    return "C", None


def _setup_training(config: dict, run_id: str, device: str, seed: int) -> TrainContext:
    """Initialize all training components."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path(config["logging"]["run_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    modules = build_pipeline(config, device)

    loss_fn = NavigationalLoss(
        initial_log_sigma=config["training"]["loss"].get("initial_log_sigma", 0.0)
    ).to(device)

    all_params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    all_params.extend(p for p in loss_fn.parameters() if p.requires_grad)
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        all_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    print(f"Training run: {run_id}")
    print(
        f"  Device: {device}, Max iters: {train_cfg['max_iterations']}"
        f", Batch: {train_cfg['batch_size']}"
    )

    return TrainContext(
        modules=modules,
        banks=config["model"]["navigator"].get("navigable_banks", BANK_NAMES),
        anchor_labels=load_anchor_labels(config),
        loss_fn=loss_fn,
        optimizer=optimizer,
        tracker=PABTracker(
            experiment_id=run_id,
            checkpoint_interval=config.get("pab", {}).get("checkpoint_interval", 50),
        ),
        device=device,
        config=config,
        paths={
            "run_dir": run_dir,
            "ckpt_dir": ckpt_dir,
            "log_path": run_dir / f"{run_id}{config['logging']['training_log_suffix']}",
        },
    )


def _run_step_checks(
    step: int,
    loss_dict: dict,
    ctx: TrainContext,
    dataset: NavigationalDataset,
) -> str | None:
    """Run per-step logging, PAB checkpoints, and NaN checks. Returns abort reason or None."""
    if step % 50 == 0:
        max_iters = ctx.config["training"]["max_iterations"]
        print(
            f"  Step {step}/{max_iters}: L={loss_dict['L_total']:.4f}"
            f" nav={loss_dict.get('L_nav', 0):.4f}"
            f" anchor={loss_dict.get('L_anchor', 0):.4f}"
            f" critic={loss_dict.get('L_critic', 0):.4f}"
        )

    pab_interval = ctx.config.get("pab", {}).get("checkpoint_interval", 50)
    if step % pab_interval == 0:
        nav_acc = compute_nav_accuracy(ctx.modules, dataset, ctx.banks, ctx.device)
        ctx.tracker.record(
            CheckpointData(
                step=step,
                train_loss=loss_dict["L_total"],
                tier_accuracies={"tier1": nav_acc["mean"]},
                domain_accuracies=nav_acc,
            )
        )
        if step % 200 == 0:
            print(f"    Nav accuracy: {nav_acc}")

    safety = ctx.config["safety"]
    if safety.get("nan_abort") and any(
        np.isnan(v) for v in loss_dict.values() if isinstance(v, float)
    ):
        return "nan_abort"

    return None


def _finalize_training(
    step: int,
    all_losses: list[dict],
    elapsed: float,
    run_id: str,
    ctx: TrainContext,
) -> dict:
    """Save checkpoint, training log, PAB profile, and return results."""
    ckpt_path = ctx.paths["ckpt_dir"] / f"{run_id}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "modules": {name: m.state_dict() for name, m in ctx.modules.items()},
            "loss_fn": ctx.loss_fn.state_dict(),
            "optimizer": ctx.optimizer.state_dict(),
            "config": ctx.config,
        },
        ckpt_path,
    )

    with open(ctx.paths["log_path"], "w") as f:
        for entry in all_losses:
            f.write(json.dumps(entry) + "\n")

    profile = ctx.tracker.finalize()
    profile_path = ctx.paths["run_dir"] / f"{run_id}_pab_profile.json"
    profile.save(profile_path)

    print(f"\nTraining complete: {step} steps in {elapsed:.1f}s")
    if all_losses:
        print(f"Final loss: {all_losses[-1]['L_total']:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"PAB profile: {profile_path}")
    print(f"PAB regime: {profile.summary.stability_regime}")

    return {
        "status": "complete",
        "steps": step,
        "elapsed_s": round(elapsed, 1),
        "final_loss": all_losses[-1] if all_losses else {},
        "checkpoint": str(ckpt_path),
        "pab_profile": str(profile_path),
    }


def _run_epoch(
    loader: DataLoader,
    ctx: TrainContext,
    state: dict,
    dry_run: bool,
) -> dict | None:
    """Run one epoch of batches. Returns early-exit result or None to continue."""
    max_grad_norm = ctx.config["safety"].get("max_grad_norm", 1.0)
    max_iters = ctx.config["training"]["max_iterations"]

    for batch in loader:
        if state["step"] >= max_iters:
            break

        loss_dict = train_step(batch, ctx, max_grad_norm)
        state["losses"].append({"step": state["step"], **loss_dict})
        state["step"] += 1

        abort = _run_step_checks(state["step"], loss_dict, ctx, state["dataset"])
        if abort:
            print(f"  {abort} at step {state['step']}, aborting")
            return {"status": abort, "step": state["step"]}

        if dry_run:
            print(f"  Dry run complete. Loss: {loss_dict['L_total']:.4f}")
            return {"status": "dry_run", "step": 1, "loss": loss_dict}

    return None


def train(config: dict, run_id: str, device: str, seed: int, dry_run: bool) -> dict:
    """Main training loop with curriculum phases."""
    ctx = _setup_training(config, run_id, device, seed)
    nav_train_path = Path(config["data"]["nav_train"])
    max_iters = config["training"]["max_iterations"]
    batch_size = config["training"]["batch_size"]

    state = {"step": 0, "losses": [], "phase": "", "dataset": None}
    start = time.time()

    while state["step"] < max_iters:
        phase_name, max_steps = get_curriculum_phase(state["step"], config)

        if phase_name != state["phase"]:
            state["phase"] = phase_name
            state["dataset"] = NavigationalDataset(nav_train_path, max_steps=max_steps)
            print(
                f"\n  Phase {phase_name}: {len(state['dataset'])} examples (max_steps={max_steps})"
            )
            if len(state["dataset"]) == 0:
                print(f"  WARNING: Phase {phase_name} has no examples, skipping")
                state["step"] += 1
                continue

        loader = DataLoader(
            state["dataset"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: b,
            num_workers=0,
        )
        result = _run_epoch(loader, ctx, state, dry_run)
        if result is not None:
            return result

    return _finalize_training(state["step"], state["losses"], time.time() - start, run_id, ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wayfinder navigator training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train(config, args.run_id, args.device, args.seed, args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
