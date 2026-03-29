"""Train the learned second-order SoM over frozen symbolic packet features.

Adapted from the first-order SoM regime:
- Stage 1: local executor-facing heads
- Stage 2: controller/orchestration heads over frozen local outputs
- Stage 3: low-LR joint finetune
- PAB-style stability stopping instead of fixed-step-only training
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.second_order_som_model import SecondOrderSoMConfig, SecondOrderSoMNet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PABMonitor:
    """Train until trajectory stabilizes, not just until a fixed epoch budget ends."""

    def __init__(self, threshold: float = 0.015, window: int = 10, patience: int = 3, max_no_improve: int = 12):
        self.threshold = threshold
        self.window = window
        self.patience = patience
        self.max_no_improve = max_no_improve
        self.losses: list[float] = []
        self.best_metric = -1e9
        self.no_improve_count = 0
        self.stable_count = 0

    def update(self, loss: float, metric: float) -> bool:
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
            recent = self.losses[-self.window :]
            changes = [
                abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-8)
                for i in range(1, len(recent))
            ]
            mean_change = sum(changes) / max(len(changes), 1)
            if mean_change < self.threshold:
                self.stable_count += 1
                if self.stable_count >= self.patience:
                    logger.info("PAB: stable (%.4f < %.4f)", mean_change, self.threshold)
                    return True
            else:
                self.stable_count = 0
        return False


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {name: data[name] for name in data.files}


def _to_loader(split: dict[str, np.ndarray], batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split["features"], dtype=torch.float32),
        torch.tensor(split["invoke_ducky"], dtype=torch.float32),
        torch.tensor(split["observed_progress"], dtype=torch.float32),
        torch.tensor(split["projector_rejection_seen"], dtype=torch.float32),
        torch.tensor(split["packet_kind"], dtype=torch.long),
        torch.tensor(split["engine_targets"], dtype=torch.float32),
        torch.tensor(split["backend_targets"], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _metrics(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
    _, invoke, progress, projector, packet_kind, engine_targets, backend_targets = batch
    invoke_pred = (torch.sigmoid(outputs["invoke_logit"]) >= 0.5).float()
    progress_pred = (torch.sigmoid(outputs["progress_logit"]) >= 0.5).float()
    projector_pred = (torch.sigmoid(outputs["projector_logit"]) >= 0.5).float()
    packet_pred = outputs["packet_kind_logits"].argmax(dim=-1)
    engine_pred = (torch.sigmoid(outputs["engine_logits"]) >= 0.5).float()
    backend_pred = (torch.sigmoid(outputs["backend_logits"]) >= 0.5).float()
    return {
        "invoke_acc": float((invoke_pred == invoke).float().mean().item()),
        "progress_acc": float((progress_pred == progress).float().mean().item()),
        "projector_acc": float((projector_pred == projector).float().mean().item()),
        "packet_kind_acc": float((packet_pred == packet_kind).float().mean().item()),
        "engine_micro_acc": float((engine_pred == engine_targets).float().mean().item()),
        "backend_micro_acc": float((backend_pred == backend_targets).float().mean().item()),
    }


def _stage1_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], aux_weight: float) -> torch.Tensor:
    _, invoke, progress, _projector, _packet_kind, engine_targets, backend_targets = batch
    invoke_loss = F.binary_cross_entropy_with_logits(outputs["invoke_logit"], invoke)
    progress_loss = F.binary_cross_entropy_with_logits(outputs["progress_logit"], progress)
    engine_loss = F.binary_cross_entropy_with_logits(outputs["engine_logits"], engine_targets)
    backend_loss = F.binary_cross_entropy_with_logits(outputs["backend_logits"], backend_targets)
    return invoke_loss + 0.8 * progress_loss + aux_weight * (engine_loss + backend_loss)


def _stage2_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], aux_weight: float) -> torch.Tensor:
    _, invoke, progress, projector, packet_kind, _engine_targets, _backend_targets = batch
    packet_loss = F.cross_entropy(outputs["packet_kind_logits"], packet_kind)
    projector_loss = F.binary_cross_entropy_with_logits(outputs["projector_logit"], projector)
    invoke_loss = F.binary_cross_entropy_with_logits(outputs["invoke_logit"], invoke)
    progress_loss = F.binary_cross_entropy_with_logits(outputs["progress_logit"], progress)
    return packet_loss + 0.6 * projector_loss + aux_weight * (invoke_loss + 0.5 * progress_loss)


def _joint_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], aux_weight: float) -> torch.Tensor:
    _, invoke, progress, projector, packet_kind, engine_targets, backend_targets = batch
    return (
        F.binary_cross_entropy_with_logits(outputs["invoke_logit"], invoke)
        + 0.6 * F.binary_cross_entropy_with_logits(outputs["progress_logit"], progress)
        + 0.4 * F.binary_cross_entropy_with_logits(outputs["projector_logit"], projector)
        + 0.5 * F.cross_entropy(outputs["packet_kind_logits"], packet_kind)
        + aux_weight * F.binary_cross_entropy_with_logits(outputs["engine_logits"], engine_targets)
        + aux_weight * F.binary_cross_entropy_with_logits(outputs["backend_logits"], backend_targets)
    )


@torch.no_grad()
def _evaluate(model: SecondOrderSoMNet, loader: DataLoader, device: str, *, stage: str, aux_weight: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    metrics: dict[str, float] = {
        "invoke_acc": 0.0,
        "progress_acc": 0.0,
        "projector_acc": 0.0,
        "packet_kind_acc": 0.0,
        "engine_micro_acc": 0.0,
        "backend_micro_acc": 0.0,
    }
    for batch in loader:
        batch = tuple(item.to(device) for item in batch)
        features = batch[0]
        outputs = model(features)
        if stage == "stage1":
            loss = _stage1_loss(outputs, batch, aux_weight)
        elif stage == "stage2":
            loss = _stage2_loss(outputs, batch, aux_weight)
        else:
            loss = _joint_loss(outputs, batch, aux_weight)
        total_loss += float(loss.item())
        count += 1
        batch_metrics = _metrics(outputs, batch)
        for key, value in batch_metrics.items():
            metrics[key] += value
    if count == 0:
        return {"loss": 0.0, **metrics}
    return {"loss": total_loss / count, **{key: value / count for key, value in metrics.items()}}


def _stage_score(stage: str, metrics: dict[str, float]) -> float:
    if stage == "stage1":
        return metrics["invoke_acc"] + metrics["progress_acc"] + metrics["engine_micro_acc"] + metrics["backend_micro_acc"]
    if stage == "stage2":
        return metrics["packet_kind_acc"] + metrics["projector_acc"] + 0.5 * metrics["invoke_acc"]
    return metrics["progress_acc"] + metrics["engine_micro_acc"] + metrics["backend_micro_acc"] + metrics["packet_kind_acc"]


def _set_requires_grad(params: list[torch.nn.Parameter], requires_grad: bool) -> None:
    for param in params:
        param.requires_grad = requires_grad


def _run_stage(
    *,
    stage_name: str,
    model: SecondOrderSoMNet,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    aux_weight: float,
) -> tuple[list[dict[str, float]], dict[str, Any], dict[str, torch.Tensor] | None]:
    if max_epochs <= 0:
        return [], {}, None
    pab = PABMonitor()
    history: list[dict[str, float]] = []
    best_metrics: dict[str, Any] = {}
    best_score = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = tuple(item.to(device) for item in batch)
            features = batch[0]
            outputs = model(features)
            if stage_name == "stage1":
                loss = _stage1_loss(outputs, batch, aux_weight)
            elif stage_name == "stage2":
                loss = _stage2_loss(outputs, batch, aux_weight)
            else:
                loss = _joint_loss(outputs, batch, aux_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        eval_metrics = _evaluate(model, eval_loader, device, stage=stage_name, aux_weight=aux_weight)
        score = _stage_score(stage_name, eval_metrics)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(batches, 1),
            "stage_score": score,
            **eval_metrics,
        }
        history.append(epoch_metrics)
        if score > best_score:
            best_score = score
            best_metrics = epoch_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        logger.info("%s epoch=%d train_loss=%.4f stage_score=%.4f", stage_name, epoch, epoch_metrics["train_loss"], score)
        if pab.update(eval_metrics["loss"], score):
            break
    return history, best_metrics, best_state


def train_second_order_som(
    feature_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 24,
    batch_size: int = 64,
    hidden_dim: int = 384,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    stage1_epochs: int = 0,
    stage2_epochs: int = 0,
    stage3_epochs: int = 0,
) -> dict[str, Any]:
    train = _load_npz(feature_dir / "train.npz")
    eval_split = _load_npz(feature_dir / "eval.npz")
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    feature_mean = train["features"].mean(axis=0)
    feature_std = train["features"].std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    train["features"] = ((train["features"] - feature_mean) / feature_std).astype(np.float32)
    eval_split["features"] = ((eval_split["features"] - feature_mean) / feature_std).astype(np.float32)

    s1_epochs = stage1_epochs or epochs
    s2_epochs = stage2_epochs or max(epochs // 2, 1)
    s3_epochs = stage3_epochs or max(epochs // 2, 1)

    cfg = SecondOrderSoMConfig(
        input_dim=int(train["features"].shape[1]),
        packet_kind_dim=int(len(metadata.get("packet_kind_vocab", []) or [])),
        engine_dim=int(train["engine_targets"].shape[1]),
        backend_dim=int(train["backend_targets"].shape[1]),
        hidden_dim=hidden_dim,
        lr_stage1=lr,
        lr_stage2=lr,
        lr_stage3=max(lr * 0.1, 1e-5),
        weight_decay=weight_decay,
    )
    model = SecondOrderSoMNet(cfg).to(device)
    train_loader = _to_loader(train, batch_size, shuffle=True)
    eval_loader = _to_loader(eval_split, batch_size, shuffle=False)

    history: dict[str, list[dict[str, float]]] = {}
    best_snapshots: dict[str, dict[str, Any]] = {}

    _set_requires_grad(model.local_parameters(), True)
    _set_requires_grad(model.controller_parameters(), False)
    optimizer_stage1 = torch.optim.AdamW(model.local_parameters(), lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
    h1, m1, s1 = _run_stage(
        stage_name="stage1",
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        optimizer=optimizer_stage1,
        max_epochs=s1_epochs,
        aux_weight=cfg.aux_weight,
    )
    history["stage1"] = h1
    best_snapshots["stage1"] = m1
    if s1 is not None:
        model.load_state_dict(s1)

    _set_requires_grad(model.local_parameters(), False)
    _set_requires_grad(model.controller_parameters(), True)
    optimizer_stage2 = torch.optim.AdamW(model.controller_parameters(), lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
    h2, m2, s2 = _run_stage(
        stage_name="stage2",
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        optimizer=optimizer_stage2,
        max_epochs=s2_epochs,
        aux_weight=cfg.aux_weight,
    )
    history["stage2"] = h2
    best_snapshots["stage2"] = m2
    if s2 is not None:
        model.load_state_dict(s2)

    _set_requires_grad(model.local_parameters(), True)
    _set_requires_grad(model.controller_parameters(), True)
    optimizer_stage3 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)
    h3, m3, s3 = _run_stage(
        stage_name="stage3",
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        optimizer=optimizer_stage3,
        max_epochs=s3_epochs,
        aux_weight=cfg.aux_weight,
    )
    history["stage3"] = h3
    best_snapshots["stage3"] = m3
    if s3 is not None:
        model.load_state_dict(s3)

    final_eval = _evaluate(model, eval_loader, device, stage="stage3", aux_weight=cfg.aux_weight)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "stage_history": history,
            "stage_best_metrics": best_snapshots,
        },
        checkpoint_path,
    )
    (output_dir / "metadata_snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        "feature_dir": str(feature_dir),
        "train_rows": int(train["features"].shape[0]),
        "eval_rows": int(eval_split["features"].shape[0]),
        "epochs_requested": epochs,
        "stage_epochs": {"stage1": s1_epochs, "stage2": s2_epochs, "stage3": s3_epochs},
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "pab": {"threshold": 0.015, "window": 10, "patience": 3, "max_no_improve": 12},
        "best_metrics": best_snapshots,
        "final_eval": final_eval,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--stage3-epochs", type=int, default=0)
    args = parser.parse_args()
    summary = train_second_order_som(
        Path(args.feature_dir).resolve(),
        Path(args.output_dir).resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
