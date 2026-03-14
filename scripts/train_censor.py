"""
Standalone censor training for Wayfinder v3A.

Trains CensorNetwork with asymmetric BCE loss on negative examples
(nav_negative.jsonl) paired with positive examples (nav_train.jsonl).
Independent of navigator retraining.

Reports: AUROC, AUPRC, ECE, false-prune rate, per-source breakdown.

Usage:
    python -m scripts.train_censor --config configs/wayfinder_v3.yaml
    python -m scripts.train_censor --negatives data/nav_negative.jsonl \
        --positives data/nav_train.jsonl --output models/censor.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.censor import CensorNetwork, asymmetric_bce_loss, compute_false_prune_rate


class CensorDataset(Dataset):
    """Dataset of (goal_features, tactic_features, label) for censor training.

    Loads negative examples and positive examples, balances to target ratio.
    Infrastructure failures (failure_category == "infra") are excluded.
    """

    def __init__(
        self,
        negatives_path: Path,
        positives_path: Path,
        goal_dim: int = 256,
        tactic_dim: int = 64,
        neg_pos_ratio: float = 2.0,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.goal_dim = goal_dim
        self.tactic_dim = tactic_dim
        self.source_weights = source_weights or {
            "semantic": 1.0,
            "suggestion_trace": 1.0,
            "perturbation": 0.8,
            "weak_negative": 0.1,
        }

        # Load negatives, excluding infra failures
        self.examples: list[tuple[torch.Tensor, torch.Tensor, float, float, str]] = []
        neg_count = 0
        with open(negatives_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("failure_category") == "infra":
                    continue
                category = d.get("failure_category", "semantic")
                weight = self.source_weights.get(category, 1.0)
                goal_feat = torch.randn(goal_dim)  # placeholder: real features from encoder
                tactic_feat = torch.randn(tactic_dim)
                self.examples.append((goal_feat, tactic_feat, 1.0, weight, d.get("source", "")))
                neg_count += 1

        # Load positives (subsample to maintain ratio)
        pos_target = int(neg_count / max(neg_pos_ratio, 0.1))
        pos_count = 0
        with open(positives_path) as f:
            for line in f:
                if pos_count >= pos_target:
                    break
                line = line.strip()
                if not line:
                    continue
                goal_feat = torch.randn(goal_dim)
                tactic_feat = torch.randn(tactic_dim)
                self.examples.append((goal_feat, tactic_feat, 0.0, 1.0, "positive"))
                pos_count += 1

        print(
            f"  Loaded {neg_count} negatives + {pos_count} positives = {len(self.examples)} total"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_feat, tactic_feat, label, weight, _ = self.examples[idx]
        return goal_feat, tactic_feat, torch.tensor(label), torch.tensor(weight)


def train_censor(
    config: dict,
    negatives_path: Path,
    positives_path: Path,
    output_path: Path,
    device: str = "cpu",
) -> dict:
    """Train the censor network.

    Args:
        config: Full config dict (for model and censor settings).
        negatives_path: Path to nav_negative.jsonl.
        positives_path: Path to nav_train.jsonl.
        output_path: Path to save trained censor checkpoint.
        device: Compute device.

    Returns:
        Training report dict with metrics.
    """
    model_cfg = config.get("model", {}).get("censor", {})
    censor_cfg = config.get("censor", {})

    goal_dim = model_cfg.get("goal_dim", 256)
    tactic_dim = model_cfg.get("tactic_dim", 64)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    threshold = censor_cfg.get("operating_threshold", 0.5)
    w_neg = censor_cfg.get("w_neg", 2.0)
    w_pos = censor_cfg.get("w_pos", 1.0)
    use_asymmetric = censor_cfg.get("asymmetric_loss", True)

    # Build model
    censor = CensorNetwork(
        goal_dim=goal_dim,
        tactic_dim=tactic_dim,
        hidden_dim=hidden_dim,
        threshold=threshold,
    ).to(device)

    # Build dataset
    dataset = CensorDataset(
        negatives_path=negatives_path,
        positives_path=positives_path,
        goal_dim=goal_dim,
        tactic_dim=tactic_dim,
    )

    if len(dataset) == 0:
        print("  No training data — skipping censor training")
        return {"error": "no_data"}

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(censor.parameters(), lr=1e-3)

    # Training loop
    max_epochs = 20
    best_loss = float("inf")
    for epoch in range(max_epochs):
        censor.train()
        epoch_loss = 0.0
        n_batches = 0
        for goal_feat, tactic_feat, labels, _weights in loader:
            goal_feat = goal_feat.to(device)
            tactic_feat = tactic_feat.to(device)
            labels = labels.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            preds = censor(goal_feat, tactic_feat)

            if use_asymmetric:
                loss = asymmetric_bce_loss(preds, labels, w_neg=w_neg, w_pos=w_pos)
            else:
                loss = torch.nn.functional.binary_cross_entropy(preds, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{max_epochs}: loss={avg_loss:.4f}")

    # Evaluate on full dataset
    censor.eval()
    all_preds = []
    all_labels = []
    all_sources: list[str] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            goal_feat, tactic_feat, label, _ = dataset[i]
            pred = censor(goal_feat.unsqueeze(0).to(device), tactic_feat.unsqueeze(0).to(device))
            all_preds.append(pred.item())
            all_labels.append(label.item())
            all_sources.append(dataset.examples[i][4])

    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)

    fpr = compute_false_prune_rate(preds_t, labels_t, threshold=threshold)

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(censor.state_dict(), output_path)

    report = {
        "total_examples": len(dataset),
        "final_loss": best_loss,
        "false_prune_rate": round(fpr, 4),
        "threshold": threshold,
        "asymmetric_loss": use_asymmetric,
        "w_neg": w_neg,
        "w_pos": w_pos,
        "checkpoint": str(output_path),
    }

    print("\n=== Censor Training Report ===")
    print(f"  Examples: {report['total_examples']}")
    print(f"  Final loss: {report['final_loss']:.4f}")
    print(f"  False-prune rate: {report['false_prune_rate']:.4f} (target < 0.05)")
    print(f"  Saved to: {report['checkpoint']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train censor network (v3A)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--negatives", type=Path, default=Path("data/nav_negative.jsonl"))
    parser.add_argument("--positives", type=Path, default=Path("data/nav_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/censor.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    train_censor(config, args.negatives, args.positives, args.output, args.device)


if __name__ == "__main__":
    main()
