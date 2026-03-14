"""Train the template classifier (RECOGNITION slot, Slot 2) for Wayfinder v2.

Trains a lightweight classifier to predict proof strategy templates from
GoalAnalyzer features. Uses augmented training data with template labels
from scripts/extract_templates.py.

Usage:
    python -m scripts.train_template_classifier --config configs/wayfinder_v2.yaml --run-id TC-001
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
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
    config: dict, device: str
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
    ).to(device)

    return encoder, analyzer, classifier


def train_template_classifier(config: dict, run_id: str, device: str, seed: int) -> dict:
    """Train the template classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path(config.get("logging", {}).get("run_dir", "runs/")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder, analyzer, classifier = _build_components(config, device)

    train_cfg = config.get("training", {}).get("template_classifier", config.get("training", {}))
    lr = train_cfg.get("learning_rate", 1e-3)
    batch_size = train_cfg.get("batch_size", 32)
    max_iters = train_cfg.get("max_iterations", 2000)
    max_grad_norm = config.get("safety", {}).get("max_grad_norm", 1.0)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)

    data_path = Path(config.get("data", {}).get("template_train", "data/nav_train_templates.jsonl"))
    print(f"Loading template training data from {data_path}...")
    dataset = TemplateDataset(data_path)
    print(f"  {len(dataset)} examples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)

    tracker = PABTracker(experiment_id=run_id, checkpoint_interval=50)

    print(f"\nTraining template classifier: {run_id}")
    print(f"  Device: {device}, LR: {lr}, Batch: {batch_size}, Max iters: {max_iters}")
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
            logits, _ = classifier(features)

            loss = F.cross_entropy(logits, template_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)
            optimizer.step()

            loss_val = loss.item()
            all_losses.append({"step": step, "loss": loss_val})

            if step % 50 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == template_ids).float().mean().item()
                    top3 = _top_k_accuracy(logits, template_ids, k=3)
                print(
                    f"  Step {step}/{max_iters}: loss={loss_val:.4f} acc={acc:.3f} top3={top3:.3f}"
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
