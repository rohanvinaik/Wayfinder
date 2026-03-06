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
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.bridge import InformationBridge
from src.data import NavigationalDataset
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.losses import NavigationalLoss
from src.nav_contracts import BANK_NAMES
from src.pab_tracker import CheckpointData, PABTracker
from src.proof_navigator import ProofNavigator


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


def build_direction_targets(
    examples: list, banks: list[str], device: str
) -> dict[str, torch.Tensor]:
    """Build per-bank direction target tensors from a batch of examples."""
    direction_map = {-1: 0, 0: 1, 1: 2}
    targets: dict[str, torch.Tensor] = {}
    for bank in banks:
        vals = [direction_map[ex.nav_directions.get(bank, 0)] for ex in examples]
        targets[bank] = torch.tensor(vals, dtype=torch.long, device=device)
    return targets


def build_anchor_targets(examples: list, anchor_labels: list[str], device: str) -> torch.Tensor:
    """Build multi-label anchor target tensor."""
    label_to_idx = {label: i for i, label in enumerate(anchor_labels)}
    n_anchors = len(anchor_labels)
    targets = torch.zeros(len(examples), n_anchors, device=device)
    for i, ex in enumerate(examples):
        for label in ex.anchor_labels:
            if label in label_to_idx:
                targets[i, label_to_idx[label]] = 1.0
    return targets


def build_progress_targets(examples: list, device: str) -> torch.Tensor:
    """Build normalized progress targets (remaining / total)."""
    vals = [ex.remaining_steps / max(ex.total_steps, 1) for ex in examples]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def build_critic_targets(examples: list, device: str) -> torch.Tensor:
    """Build soft critic targets based on proof completion proximity."""
    vals = []
    for ex in examples:
        if not ex.solvable:
            vals.append(0.0)
        else:
            progress = 1.0 - (ex.remaining_steps / max(ex.total_steps, 1))
            vals.append(0.3 + 0.7 * progress)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def load_anchor_labels(config: dict) -> list[str]:
    """Load anchor label list from config path or generate indices."""
    anchor_path = config.get("data", {}).get("anchor_labels")
    if anchor_path and Path(anchor_path).exists():
        with open(anchor_path) as f:
            return json.load(f)
    num_anchors = config["model"]["navigator"]["num_anchors"]
    return [str(i) for i in range(num_anchors)]


def train_step(
    batch: list,
    modules: dict,
    loss_fn: NavigationalLoss,
    optimizer: torch.optim.Optimizer,
    banks: list[str],
    anchor_labels: list[str],
    device: str,
    max_grad_norm: float,
) -> dict[str, float]:
    """Execute a single training step."""
    optimizer.zero_grad()

    goal_states = [ex.goal_state for ex in batch]
    embeddings = modules["encoder"].encode(goal_states)
    features, _, _ = modules["analyzer"](embeddings)
    bridge_out = modules["bridge"](features)
    dir_logits, anchor_logits, progress_pred, critic_pred = modules["navigator"](bridge_out)

    dir_targets = build_direction_targets(batch, banks, device)
    anchor_targets = build_anchor_targets(batch, anchor_labels, device)
    progress_targets = build_progress_targets(batch, device)
    critic_targets = build_critic_targets(batch, device)

    loss_dict = loss_fn(
        direction_logits=dir_logits,
        direction_targets=dir_targets,
        anchor_logits=anchor_logits,
        anchor_targets=anchor_targets,
        progress_pred=progress_pred,
        progress_target=progress_targets,
        critic_pred=critic_pred,
        critic_target=critic_targets,
    )

    loss_dict["L_total"].backward()
    torch.nn.utils.clip_grad_norm_(
        [p for m in modules.values() for p in m.parameters() if p.requires_grad],
        max_grad_norm,
    )
    optimizer.step()

    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in loss_dict.items()
        if k != "bank_losses"
    }


def compute_nav_accuracy(
    modules: dict,
    dataset: NavigationalDataset,
    banks: list[str],
    device: str,
    max_samples: int = 200,
) -> dict[str, float]:
    """Compute per-bank direction accuracy on a subset."""
    n = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n, replace=False)
    examples = [dataset[int(i)] for i in indices]

    goal_states = [ex.goal_state for ex in examples]
    with torch.no_grad():
        embeddings = modules["encoder"].encode(goal_states)
        features, _, _ = modules["analyzer"](embeddings)
        bridge_out = modules["bridge"](features)
        dir_logits, _, _, _ = modules["navigator"](bridge_out)

    direction_map = {-1: 0, 0: 1, 1: 2}
    accuracies: dict[str, float] = {}
    for bank in banks:
        preds = dir_logits[bank].argmax(dim=-1).cpu().numpy()
        targets = np.array([direction_map[ex.nav_directions.get(bank, 0)] for ex in examples])
        accuracies[bank] = float(np.mean(preds == targets))

    accuracies["mean"] = float(np.mean(list(accuracies.values())))
    return accuracies


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


def _setup_training(config: dict, run_id: str, device: str, seed: int) -> dict:
    """Initialize all training components."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path(config["logging"]["run_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    modules = build_pipeline(config, device)
    banks = config["model"]["navigator"].get("navigable_banks", BANK_NAMES)
    anchor_labels = load_anchor_labels(config)

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

    tracker = PABTracker(
        experiment_id=run_id,
        checkpoint_interval=config.get("pab", {}).get("checkpoint_interval", 50),
    )

    log_path = run_dir / f"{run_id}{config['logging']['training_log_suffix']}"

    print(f"Training run: {run_id}")
    print(
        f"  Device: {device}, Max iters: {train_cfg['max_iterations']}"
        f", Batch: {train_cfg['batch_size']}"
    )

    return {
        "modules": modules,
        "banks": banks,
        "anchor_labels": anchor_labels,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "tracker": tracker,
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "log_path": log_path,
    }


def _run_step_checks(
    step: int,
    loss_dict: dict,
    config: dict,
    modules: dict,
    dataset: NavigationalDataset,
    banks: list[str],
    device: str,
    tracker: PABTracker,
) -> str | None:
    """Run per-step logging, PAB checkpoints, and NaN checks. Returns abort reason or None."""
    if step % 50 == 0:
        max_iters = config["training"]["max_iterations"]
        print(
            f"  Step {step}/{max_iters}: L={loss_dict['L_total']:.4f}"
            f" nav={loss_dict.get('L_nav', 0):.4f}"
            f" anchor={loss_dict.get('L_anchor', 0):.4f}"
            f" critic={loss_dict.get('L_critic', 0):.4f}"
        )

    pab_interval = config.get("pab", {}).get("checkpoint_interval", 50)
    if step % pab_interval == 0:
        nav_acc = compute_nav_accuracy(modules, dataset, banks, device)
        tracker.record(
            CheckpointData(
                step=step,
                train_loss=loss_dict["L_total"],
                tier_accuracies={"tier1": nav_acc["mean"]},
                domain_accuracies=nav_acc,
            )
        )
        if step % 200 == 0:
            print(f"    Nav accuracy: {nav_acc}")

    safety = config["safety"]
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
    config: dict,
    ctx: dict,
) -> dict:
    """Save checkpoint, training log, PAB profile, and return results."""
    modules, loss_fn = ctx["modules"], ctx["loss_fn"]
    optimizer, tracker = ctx["optimizer"], ctx["tracker"]
    ckpt_dir, log_path, run_dir = ctx["ckpt_dir"], ctx["log_path"], ctx["run_dir"]

    ckpt_path = ckpt_dir / f"{run_id}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "modules": {name: m.state_dict() for name, m in modules.items()},
            "loss_fn": loss_fn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        ckpt_path,
    )

    with open(log_path, "w") as f:
        for entry in all_losses:
            f.write(json.dumps(entry) + "\n")

    profile = tracker.finalize()
    profile_path = run_dir / f"{run_id}_pab_profile.json"
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


def train(config: dict, run_id: str, device: str, seed: int, dry_run: bool) -> dict:
    """Main training loop with curriculum phases."""
    ctx = _setup_training(config, run_id, device, seed)
    modules, banks, anchor_labels = ctx["modules"], ctx["banks"], ctx["anchor_labels"]
    loss_fn, optimizer, tracker = ctx["loss_fn"], ctx["optimizer"], ctx["tracker"]

    nav_train_path = Path(config["data"]["nav_train"])
    train_cfg = config["training"]
    max_iters = train_cfg["max_iterations"]
    safety = config["safety"]

    all_losses: list[dict] = []
    step = 0
    current_phase = ""
    dataset = None
    start = time.time()

    while step < max_iters:
        phase_name, max_steps = get_curriculum_phase(step, config)

        if phase_name != current_phase:
            current_phase = phase_name
            dataset = NavigationalDataset(nav_train_path, max_steps=max_steps)
            print(f"\n  Phase {phase_name}: {len(dataset)} examples (max_steps={max_steps})")
            if len(dataset) == 0:
                print(f"  WARNING: Phase {phase_name} has no examples, skipping")
                step += 1
                continue

        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

        for batch in loader:
            if step >= max_iters:
                break

            loss_dict = train_step(
                batch,
                modules,
                loss_fn,
                optimizer,
                banks,
                anchor_labels,
                device,
                safety.get("max_grad_norm", 1.0),
            )
            all_losses.append({"step": step, **loss_dict})
            step += 1

            abort = _run_step_checks(
                step,
                loss_dict,
                config,
                modules,
                dataset,
                banks,
                device,
                tracker,
            )
            if abort:
                print(f"  {abort} at step {step}, aborting")
                return {"status": abort, "step": step}

            if dry_run:
                print(f"  Dry run complete. Loss: {loss_dict['L_total']:.4f}")
                return {"status": "dry_run", "step": 1, "loss": loss_dict}

        new_phase, _ = get_curriculum_phase(step, config)
        if new_phase != current_phase:
            continue

    return _finalize_training(
        step,
        all_losses,
        time.time() - start,
        run_id,
        config,
        ctx,
    )


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
