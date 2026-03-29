"""Train SoM PyTorch model with three-stage curriculum and PAB stability.

Stage 1: Train specialists independently (each learns its domain signal)
Stage 2: Freeze specialists, train orchestrator (learns trust weights)
Stage 3: Joint fine-tuning at 10x lower LR (compositionality emerges)

Usage:
    python -m scripts.train_som_torch --output models/som_torch_v1
    python -m scripts.train_som_torch --stage 3 --output models/som_torch_v1  # resume stage 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.som_torch import SPECIALIST_NAMES, SoMConfig, SoMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# PAB Stability Monitor
# ---------------------------------------------------------------------------

class PABMonitor:
    """Train until trajectory stabilizes, not for fixed steps."""

    def __init__(self, threshold: float = 0.015, window: int = 20,
                 patience: int = 5, max_no_improve: int = 40):
        self.threshold = threshold
        self.window = window
        self.patience = patience
        self.max_no_improve = max_no_improve
        self.losses: list[float] = []
        self.best_metric: float = -1e9
        self.no_improve_count: int = 0
        self.stable_count: int = 0

    def update(self, loss: float, metric: float) -> bool:
        """Returns True if training should stop."""
        self.losses.append(loss)
        if metric > self.best_metric:
            self.best_metric = metric
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.max_no_improve:
            logger.info("PAB: max_no_improve (%d)", self.max_no_improve)
            return True

        if len(self.losses) >= self.window + 1:
            recent = self.losses[-self.window:]
            changes = [abs(recent[i] - recent[i-1]) / (abs(recent[i-1]) + 1e-8)
                       for i in range(1, len(recent))]
            mean_change = sum(changes) / len(changes)
            if mean_change < self.threshold:
                self.stable_count += 1
                if self.stable_count >= self.patience:
                    logger.info("PAB: stable (%.4f < %.4f)", mean_change, self.threshold)
                    return True
            else:
                self.stable_count = 0
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(train_path: str, eval_path: str, max_examples: int = 0, device: str = "cpu"):
    """Load pre-computed feature arrays into torch tensors."""
    train_npz = np.load(train_path)
    eval_npz = np.load(eval_path)

    def to_tensors(npz, limit=0):
        n = npz["labels"].shape[0]
        if limit and limit < n:
            n = limit
        return {
            "goal_emb": torch.tensor(npz["goal_emb"][:n], dtype=torch.float32, device=device),
            "goal_shape": torch.tensor(npz["goal_shape"][:n], dtype=torch.float32, device=device),
            "step_context": torch.tensor(npz["step_context"][:n], dtype=torch.float32, device=device),
            "labels": torch.tensor(npz["labels"][:n], dtype=torch.long, device=device),
        }

    return to_tensors(train_npz, max_examples), to_tensors(eval_npz)


def make_dataloader(data: dict, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(data["goal_emb"], data["goal_shape"],
                       data["step_context"], data["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: SoMModel, data: dict) -> dict:
    model.eval()
    trust, info = model(data["goal_emb"], data["goal_shape"], data["step_context"])

    # Also get specialist-only scores for Stage 1 eval
    scores = info["specialist_scores"]

    labels = data["labels"]
    n = labels.shape[0]

    # Trust-based metrics
    trust_preds = trust.argmax(dim=-1)
    trust_acc = (trust_preds == labels).float().mean().item()
    trust_top3 = sum(
        labels[i] in trust[i].topk(3).indices for i in range(n)
    ) / n

    # Specialist-score-based metrics (for Stage 1)
    score_preds = scores.argmax(dim=-1)
    score_acc = (score_preds == labels).float().mean().item()

    # Loss
    loss = nn.functional.cross_entropy(trust, labels).item()
    score_loss = nn.functional.cross_entropy(scores, labels).item()

    # Per-specialist
    per_spec = {}
    for i, name in enumerate(SPECIALIST_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_spec[name] = (trust_preds[mask] == i).float().mean().item()

    model.train()
    return {
        "trust_acc": trust_acc,
        "trust_top3": trust_top3,
        "trust_loss": loss,
        "score_acc": score_acc,
        "score_loss": score_loss,
        "per_specialist": per_spec,
    }


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def train_stage1(model: SoMModel, train_data: dict, eval_data: dict,
                 cfg: SoMConfig, batch_size: int, max_epochs: int = 100) -> dict:
    """Stage 1: Train specialists independently."""
    logger.info("=== Stage 1: Train specialists independently ===")

    # Only optimize specialist parameters
    spec_params = []
    for agent in model.specialists.values():  # type: ignore[union-attr]
        spec_params.extend(agent.parameters())
    optimizer = torch.optim.AdamW(spec_params, lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)

    pab = PABMonitor(threshold=0.015, window=20, patience=5, max_no_improve=40)
    loader = make_dataloader(train_data, batch_size)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for goal_emb, goal_shape, step_ctx, labels in loader:
            # Forward through specialists only (scores, not trust weights)
            context = torch.cat([goal_emb, goal_shape, step_ctx], dim=-1)
            domain = torch.cat([goal_emb, goal_shape], dim=-1)

            scores = []
            aux_preds = []
            for name in SPECIALIST_NAMES:
                conf, aux = model.specialists[f"spec_{name}"](domain, context)
                scores.append(conf)
                aux_preds.append(aux)
            scores_t = torch.stack(scores, dim=-1)

            # Cross-entropy on specialist scores
            loss = nn.functional.cross_entropy(scores_t, labels)

            # Auxiliary loss: predict which specialist is correct (weak signal)
            for i, aux in enumerate(aux_preds):
                target = (labels == i).float() * 2 - 1  # +1 if correct, -1 if not
                loss = loss + cfg.aux_weight * nn.functional.mse_loss(aux, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, eval_data)
        acc = metrics["score_acc"]
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if epoch % 5 == 0 or epoch < 5:
            logger.info("  S1 Epoch %d: loss=%.4f score_acc=%.4f trust_acc=%.4f best=%.4f@%d",
                        epoch, epoch_loss / n_batches, acc, metrics["trust_acc"],
                        best_acc, best_epoch)

        if pab.update(metrics["score_loss"], acc):
            break

    logger.info("Stage 1: best_score_acc=%.4f@%d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}


def train_stage2(model: SoMModel, train_data: dict, eval_data: dict,
                 cfg: SoMConfig, batch_size: int, max_epochs: int = 50) -> dict:
    """Stage 2: Freeze specialists, train orchestrator."""
    logger.info("=== Stage 2: Freeze specialists, train orchestrator ===")

    # Freeze specialists
    for agent in model.specialists.values():  # type: ignore[union-attr]
        for p in agent.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.orchestrator.parameters(),
                                   lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)

    pab = PABMonitor(threshold=0.01, window=20, patience=5, max_no_improve=40)
    loader = make_dataloader(train_data, batch_size)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for goal_emb, goal_shape, step_ctx, labels in loader:
            trust, info = model(goal_emb, goal_shape, step_ctx)
            loss = nn.functional.cross_entropy(trust, labels)
            # Auxiliary outcome loss
            loss = loss + cfg.aux_weight * nn.functional.mse_loss(
                info["outcome_pred"], (labels < 3).float() * 2 - 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, eval_data)
        acc = metrics["trust_acc"]
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if epoch % 5 == 0 or epoch < 3:
            logger.info("  S2 Epoch %d: loss=%.4f trust_acc=%.4f top3=%.4f best=%.4f@%d",
                        epoch, epoch_loss / n_batches, acc, metrics["trust_top3"],
                        best_acc, best_epoch)

        if pab.update(metrics["trust_loss"], acc):
            break

    # Unfreeze specialists for Stage 3
    for agent in model.specialists.values():  # type: ignore[union-attr]
        for p in agent.parameters():
            p.requires_grad = True

    logger.info("Stage 2: best_trust_acc=%.4f@%d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}


def train_stage3(model: SoMModel, train_data: dict, eval_data: dict,
                 cfg: SoMConfig, batch_size: int, max_epochs: int = 100,
                 save_path: str = "") -> dict:
    """Stage 3: Joint fine-tuning at 10x lower LR. Compositionality emerges."""
    logger.info("=== Stage 3: Joint fine-tuning (10x lower LR) ===")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)

    pab = PABMonitor(threshold=0.008, window=20, patience=5, max_no_improve=40)
    loader = make_dataloader(train_data, batch_size)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for goal_emb, goal_shape, step_ctx, labels in loader:
            trust, info = model(goal_emb, goal_shape, step_ctx)
            loss = nn.functional.cross_entropy(trust, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, eval_data)
        acc = metrics["trust_acc"]
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "best_acc": best_acc,
                    "epoch": epoch,
                }, save_path)

        if epoch % 5 == 0 or epoch < 5:
            logger.info("  S3 Epoch %d: loss=%.4f trust_acc=%.4f top3=%.4f best=%.4f@%d",
                        epoch, epoch_loss / n_batches, acc, metrics["trust_top3"],
                        best_acc, best_epoch)

        if pab.update(metrics["trust_loss"], acc):
            break

    logger.info("Stage 3: best_trust_acc=%.4f@%d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default="data/som_train_features.npz")
    parser.add_argument("--eval-features", default="data/som_eval_features.npz")
    parser.add_argument("--output", default="models/som_torch_v1")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", type=int, default=0, help="Run only this stage (1/2/3), 0=all")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    logger.info("Device: %s", device)

    # Load data
    logger.info("Loading features...")
    train_data, eval_data = load_features(
        args.train_features, args.eval_features, args.max_examples, device)
    logger.info("  Train: %d, Eval: %d", train_data["labels"].shape[0], eval_data["labels"].shape[0])

    # Label distribution
    for name, data in [("Train", train_data), ("Eval", eval_data)]:
        counts = torch.bincount(data["labels"], minlength=5)
        total = counts.sum()
        dist = ", ".join(f"{SPECIALIST_NAMES[i]}={counts[i]} ({100*counts[i]/total:.1f}%)"
                         for i in range(5))
        logger.info("  %s: %s", name, dist)

    # Build model
    cfg = SoMConfig()
    model = SoMModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", total_params)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded init checkpoint: %s", args.init_checkpoint)

    # Baselines
    majority = int(torch.bincount(eval_data["labels"], minlength=5).argmax())
    majority_acc = (eval_data["labels"] == majority).float().mean().item()
    logger.info("Baselines: random=0.200, majority=%.3f (%s)", majority_acc, SPECIALIST_NAMES[majority])

    t_start = time.time()
    results = {}

    # Stage 1
    if args.stage in (0, 1):
        s1 = train_stage1(model, train_data, eval_data, cfg, args.batch_size)
        results["stage1"] = s1
        torch.save(model.state_dict(), str(output_dir / "after_stage1.pt"))

    # Stage 2
    if args.stage in (0, 2):
        if args.stage == 2:
            ckpt = output_dir / "after_stage1.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
                logger.info("Loaded Stage 1 checkpoint")
        s2 = train_stage2(model, train_data, eval_data, cfg, args.batch_size)
        results["stage2"] = s2
        torch.save(model.state_dict(), str(output_dir / "after_stage2.pt"))

    # Stage 3
    if args.stage in (0, 3):
        if args.stage == 3:
            ckpt = output_dir / "after_stage2.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
                logger.info("Loaded Stage 2 checkpoint")
        s3 = train_stage3(model, train_data, eval_data, cfg, args.batch_size,
                          save_path=str(output_dir / "best.pt"))
        results["stage3"] = s3

    elapsed = time.time() - t_start

    # Final evaluation
    metrics = evaluate(model, eval_data)
    logger.info("=" * 60)
    logger.info("Final: trust_acc=%.4f top3=%.4f score_acc=%.4f (majority=%.3f)",
                metrics["trust_acc"], metrics["trust_top3"], metrics["score_acc"], majority_acc)
    for name, acc in metrics["per_specialist"].items():
        logger.info("  %s: %.3f", name, acc)

    # Save summary
    summary = {
        "total_params": total_params,
        "train_n": int(train_data["labels"].shape[0]),
        "eval_n": int(eval_data["labels"].shape[0]),
        "init_checkpoint": args.init_checkpoint,
        "majority_baseline": majority_acc,
        "final_trust_acc": metrics["trust_acc"],
        "final_trust_top3": metrics["trust_top3"],
        "final_score_acc": metrics["score_acc"],
        "per_specialist": metrics["per_specialist"],
        "elapsed_s": elapsed,
        "stages": results,
        "device": device,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved to %s (%.0fs)", output_dir, elapsed)


if __name__ == "__main__":
    main()
