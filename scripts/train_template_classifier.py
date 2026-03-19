"""Train the template classifier (RECOGNITION slot, Slot 2) for Wayfinder v2.

Trains a lightweight classifier to predict proof strategy templates from
GoalAnalyzer features. Uses augmented training data with template labels
from scripts/extract_templates.py.

Usage:
    python -m scripts.train_template_classifier --config configs/wayfinder.yaml --run-id TC-AUX-001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.move_supervision import (
    MoveSupervisionSpec,
    build_template_move_supervision_spec,
    build_template_move_targets,
    compute_move_metrics,
)
from src.pab_tracker import CheckpointData, PABTracker
from src.story_templates import get_num_templates
from src.template_classifier import TemplateClassifier


class TemplateDataset(Dataset):
    """Dataset for template classifier training.

    Loads augmented nav_train_templates.jsonl with template_id labels.
    """

    def __init__(self, path: Path) -> None:
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if "template_id" in d:
                        self.examples.append(d)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def _top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Compute top-k accuracy."""
    _, top_k_preds = logits.topk(k, dim=-1)
    correct = top_k_preds.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def _build_components(
    config: dict,
    device: str,
    move_supervision_spec: MoveSupervisionSpec | None = None,
) -> tuple[GoalEncoder, GoalAnalyzer, TemplateClassifier]:
    """Build encoder, analyzer, and classifier from config."""
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

    tc_cfg = config.get("model", {}).get("template_classifier", {})
    classifier = TemplateClassifier(
        input_dim=ana_cfg.get("feature_dim", 256),
        hidden_dim=tc_cfg.get("hidden_dim", 128),
        feature_dim=tc_cfg.get("feature_dim", 64),
        num_templates=get_num_templates(),
        auxiliary_heads=move_supervision_spec.head_sizes() if move_supervision_spec else None,
    ).to(device)

    return encoder, analyzer, classifier


def _build_move_supervision(
    config: dict,
    dataset: TemplateDataset,
    run_dir: Path,
) -> MoveSupervisionSpec | None:
    aux_cfg = config.get("training", {}).get("auxiliary", {})
    enabled_heads = aux_cfg.get(
        "template_heads",
        ["subtask_kind", "goal_target_head", "trigger_signature"],
    )
    if not aux_cfg.get("enabled", False):
        return None

    spec = build_template_move_supervision_spec(
        dataset.examples,
        enabled_heads=enabled_heads,
        subtask_min_support=aux_cfg.get("subtask_min_support", 25),
        goal_head_min_support=aux_cfg.get("goal_head_min_support", 100),
        max_goal_heads=aux_cfg.get("max_goal_heads", 64),
        trigger_min_support=aux_cfg.get("trigger_min_support", 250),
        max_trigger_signatures=aux_cfg.get("max_trigger_signatures", 128),
    )
    if not spec.has_any():
        return None

    spec_path = run_dir / "template_move_supervision_spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    return spec


def _compute_auxiliary_losses(
    batch: list[dict],
    aux_logits: dict[str, torch.Tensor],
    spec: MoveSupervisionSpec | None,
    device: str,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    if spec is None or not aux_logits:
        zero = torch.tensor(0.0, device=device)
        return zero, {}, {}

    targets, masks, target_types = build_template_move_targets(batch, spec, device)
    losses: dict[str, torch.Tensor] = {}
    for name, logits in aux_logits.items():
        if name not in targets or name not in masks:
            continue
        mask = masks[name]
        if mask.numel() == 0 or not bool(mask.any().item()):
            continue
        if target_types.get(name) == "multiclass":
            losses[name] = F.cross_entropy(logits[mask], targets[name][mask])
        elif target_types.get(name) == "multilabel":
            losses[name] = F.binary_cross_entropy_with_logits(logits[mask], targets[name][mask])

    if not losses:
        zero = torch.tensor(0.0, device=device)
        return zero, {}, {}

    aux_loss = torch.stack(list(losses.values())).mean()
    metrics = compute_move_metrics(aux_logits, targets, masks, target_types)
    loss_items = {f"L_aux_{name}": float(loss.item()) for name, loss in losses.items()}
    return aux_loss, loss_items, metrics


def train_template_classifier(config: dict, run_id: str, device: str, seed: int) -> dict:
    """Train the template classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path(config.get("logging", {}).get("run_dir", "runs/")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("training", {}).get("template_classifier", config.get("training", {}))
    lr = train_cfg.get("learning_rate", 1e-3)
    batch_size = train_cfg.get("batch_size", 32)
    max_iters = train_cfg.get("max_iterations", 2000)
    aux_loss_weight = train_cfg.get("auxiliary_loss_weight", 0.25)
    max_grad_norm = config.get("safety", {}).get("max_grad_norm", 1.0)

    data_path = Path(config.get("data", {}).get("template_train", "data/nav_train_templates.jsonl"))
    print(f"Loading template training data from {data_path}...")
    dataset = TemplateDataset(data_path)
    print(f"  {len(dataset)} examples")
    move_supervision_spec = _build_move_supervision(config, dataset, run_dir)
    if move_supervision_spec is not None:
        sizes = move_supervision_spec.head_sizes()
        print(
            "  auxiliary move heads:"
            f" subtask={sizes.get('subtask_kind', 0)}"
            f" goal_head={sizes.get('goal_target_head', 0)}"
            f" trigger={sizes.get('trigger_signature', 0)}"
        )

    encoder, analyzer, classifier = _build_components(config, device, move_supervision_spec)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)

    tracker = PABTracker(experiment_id=run_id, checkpoint_interval=50)

    print(f"\nTraining template classifier: {run_id}")
    print(
        f"  Device: {device}, LR: {lr}, Batch: {batch_size},"
        f" Max iters: {max_iters}, Aux weight: {aux_loss_weight}"
    )
    all_losses: list[dict] = []
    step = 0
    start = time.time()

    while step < max_iters:
        for batch in loader:
            if step >= max_iters:
                break

            optimizer.zero_grad()

            goal_states = [ex["goal_state"] for ex in batch]
            template_ids = torch.tensor(
                [ex["template_id"] for ex in batch], dtype=torch.long, device=device
            )

            embeddings = encoder.encode(goal_states)
            features, _, _ = analyzer(embeddings)
            logits, _, aux_logits = classifier.forward_with_aux(features)

            template_loss = F.cross_entropy(logits, template_ids)
            aux_loss, aux_loss_items, aux_metrics = _compute_auxiliary_losses(
                batch, aux_logits, move_supervision_spec, device
            )
            loss = template_loss + (aux_loss_weight * aux_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)
            optimizer.step()

            loss_val = loss.item()
            all_losses.append(
                {
                    "step": step,
                    "L_total": loss_val,
                    "L_template": float(template_loss.item()),
                    "L_move": float(aux_loss.item()),
                    **aux_loss_items,
                    **aux_metrics,
                }
            )

            if step % 50 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == template_ids).float().mean().item()
                    top3 = _top_k_accuracy(logits, template_ids, k=3)
                move_metric_str = ""
                if aux_metrics:
                    move_metric_str = " " + " ".join(
                        f"{name}={value:.3f}" for name, value in sorted(aux_metrics.items())
                    )
                print(
                    f"  Step {step}/{max_iters}: loss={loss_val:.4f}"
                    f" tmpl={template_loss.item():.4f} move={aux_loss.item():.4f}"
                    f" acc={acc:.3f} top3={top3:.3f}{move_metric_str}"
                )
                tracker.record(
                    CheckpointData(
                        step=step,
                        train_loss=loss_val,
                        tier_accuracies={"tier1": acc, "tier2": top3, "tier3": top3},
                    )
                )

            step += 1

    elapsed = time.time() - start

    ckpt_path = run_dir / f"{run_id}_template_classifier.pt"
    torch.save(
        {
            "step": step,
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "move_supervision_spec": (
                move_supervision_spec.to_dict() if move_supervision_spec is not None else None
            ),
        },
        ckpt_path,
    )

    profile = tracker.finalize()
    profile_path = run_dir / f"{run_id}_pab_profile.json"
    profile.save(profile_path)

    print(f"\nTemplate classifier training complete: {step} steps in {elapsed:.1f}s")
    print(f"Checkpoint: {ckpt_path}")
    print(f"PAB regime: {profile.summary.stability_regime}")

    return {
        "status": "complete",
        "steps": step,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "pab_regime": profile.summary.stability_regime,
        "move_supervision_spec": (
            str(run_dir / "template_move_supervision_spec.json")
            if move_supervision_spec is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train template classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = train_template_classifier(config, args.run_id, args.device, args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
