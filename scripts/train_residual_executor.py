"""Train the residual executor — post-structural tactic family classifier.

Trains on data/residual_train.jsonl to predict which of 6 tactic families
(rw/simp/exact/refine/apply/other) should be used on a post-intro goal state.

Uses the frozen all-MiniLM-L6-v2 encoder (same as NAV-002) to embed goal states.

Usage:
    python -m scripts.train_residual_executor --run-id RES-001
    python -m scripts.train_residual_executor --run-id RES-001 --device mps --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.encoder import GoalEncoder
from src.residual_executor import (
    NUM_FAMILIES,
    TACTIC_FAMILIES,
    TACTIC_TO_IDX,
    ResidualExecutor,
    tactic_to_family_idx,
)


class ResidualDataset(Dataset):  # type: ignore[type-arg]
    """Dataset of residual goal states with tactic family labels."""

    def __init__(self, path: Path, max_examples: int | None = None) -> None:
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))
                if max_examples and len(self.examples) >= max_examples:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def _encode_batch(
    encoder: GoalEncoder, batch: list[dict], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a batch of goal states and extract labels."""
    goal_states = [ex["goal_state"] for ex in batch]
    embeddings = encoder.encode(goal_states)  # [batch, 384]

    family_labels = torch.tensor(
        [tactic_to_family_idx(ex["tactic_base"]) for ex in batch],
        dtype=torch.long,
        device=device,
    )
    premise_labels = torch.tensor(
        [1.0 if len(ex.get("ground_truth_premises", [])) > 0 else 0.0 for ex in batch],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    return embeddings, family_labels, premise_labels


def _collect_predictions(
    model: ResidualExecutor,
    encoder: GoalEncoder,
    eval_data: list[dict],
    device: str,
    batch_size: int,
) -> tuple[list[int], list[int], list[bool]]:
    """Run model on eval data. Returns (preds, labels, premise_deps)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_premise_deps: list[bool] = []

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size]
        with torch.no_grad():
            emb, fam_labels, _ = _encode_batch(encoder, batch, device)
            fam_logits, _ = model(emb)
            all_preds.extend(fam_logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(fam_labels.cpu().tolist())
            all_premise_deps.extend(len(ex.get("ground_truth_premises", [])) > 0 for ex in batch)
    return all_preds, all_labels, all_premise_deps


def _compute_top3(
    model: ResidualExecutor,
    encoder: GoalEncoder,
    eval_data: list[dict],
    device: str,
    batch_size: int,
) -> float:
    """Compute top-3 accuracy (needs logit access)."""
    correct = 0
    total = 0
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size]
        with torch.no_grad():
            emb, fam_labels, _ = _encode_batch(encoder, batch, device)
            fam_logits, _ = model(emb)
            top3 = fam_logits.topk(min(3, NUM_FAMILIES), dim=-1).indices
            for j, label in enumerate(fam_labels):
                if label in top3[j]:
                    correct += 1
            total += len(batch)
    return correct / max(total, 1)


def _per_class_stats(preds: list[int], labels: list[int]) -> tuple[dict, float, dict]:
    """Compute per-class P/R/F1, macro-F1, and confusion matrix."""
    per_class: dict[str, dict] = {}
    for idx, name in enumerate(TACTIC_FAMILIES):
        tp = sum(1 for p, l in zip(preds, labels) if p == idx and l == idx)
        fn = sum(1 for p, l in zip(preds, labels) if p != idx and l == idx)
        fp = sum(1 for p, l in zip(preds, labels) if p == idx and l != idx)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class[name] = {
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    class_f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = sum(class_f1s) / max(len(class_f1s), 1)

    confusion: dict[str, dict[str, int]] = {}
    for ti, tn in enumerate(TACTIC_FAMILIES):
        row = {
            TACTIC_FAMILIES[pi]: sum(1 for p, l in zip(preds, labels) if l == ti and p == pi)
            for pi in range(NUM_FAMILIES)
        }
        row = {k: v for k, v in row.items() if v > 0}
        if row:
            confusion[tn] = row

    return per_class, macro_f1, confusion


def _slice_accuracies(
    preds: list[int], labels: list[int], premise_deps: list[bool]
) -> dict[str, float]:
    """Compute split accuracies: premise-dep, local-only, arg-dep, rewrite."""
    other_idx = TACTIC_TO_IDX["other"]

    # Top-5 excluding "other"
    non_other = [(p, l) for p, l in zip(preds, labels) if l != other_idx]
    top5_exc = sum(p == l for p, l in non_other) / max(len(non_other), 1)

    # Premise-dep vs local
    pd_ok = sum(1 for p, l, d in zip(preds, labels, premise_deps) if p == l and d)
    pd_n = sum(1 for d in premise_deps if d)
    lo_ok = sum(1 for p, l, d in zip(preds, labels, premise_deps) if p == l and not d)
    lo_n = sum(1 for d in premise_deps if not d)

    # Arg-dep vs rewrite
    arg_idxs = {TACTIC_TO_IDX["exact"], TACTIC_TO_IDX["apply"], TACTIC_TO_IDX["refine"]}
    rw_idxs = {TACTIC_TO_IDX["simp"], TACTIC_TO_IDX["rw"]}
    arg_ok = sum(1 for p, l in zip(preds, labels) if p == l and l in arg_idxs)
    arg_n = sum(1 for l in labels if l in arg_idxs)
    rw_ok = sum(1 for p, l in zip(preds, labels) if p == l and l in rw_idxs)
    rw_n = sum(1 for l in labels if l in rw_idxs)

    return {
        "top5_exc_other": round(top5_exc, 4),
        "premise_dep_acc": round(pd_ok / max(pd_n, 1), 4),
        "local_only_acc": round(lo_ok / max(lo_n, 1), 4),
        "arg_dep_acc": round(arg_ok / max(arg_n, 1), 4),
        "rewrite_acc": round(rw_ok / max(rw_n, 1), 4),
    }


def _eval_metrics(
    model: ResidualExecutor,
    encoder: GoalEncoder,
    eval_data: list[dict],
    device: str,
    batch_size: int = 64,
) -> dict:
    """Comprehensive evaluation: top-1/3, macro-F1, per-class, confusion, splits."""
    preds, labels, premise_deps = _collect_predictions(
        model, encoder, eval_data, device, batch_size
    )
    n = len(labels)
    top1 = sum(p == l for p, l in zip(preds, labels)) / max(n, 1)
    top3 = _compute_top3(model, encoder, eval_data, device, batch_size)
    per_class, macro_f1, confusion = _per_class_stats(preds, labels)
    slices = _slice_accuracies(preds, labels, premise_deps)

    return {
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "macro_f1": round(macro_f1, 4),
        **slices,
        "per_class": per_class,
        "confusion": confusion,
        "total": n,
    }


def train(
    train_path: Path,
    eval_path: Path,
    run_id: str,
    device: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_train: int | None = None,
    max_eval: int = 5000,
) -> dict:
    """Train the residual executor."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder (frozen)
    encoder = GoalEncoder.from_config(
        {"type": "all-MiniLM-L6-v2", "output_dim": 384, "frozen": True},
        device=device,
    )
    encoder.ensure_loaded()

    # Build model
    model = ResidualExecutor(input_dim=encoder.output_dim, hidden_dim=256).to(device)
    print(f"ResidualExecutor: {sum(p.numel() for p in model.parameters())} params")

    # Load data
    print(f"Loading training data from {train_path}...")
    train_ds = ResidualDataset(train_path, max_examples=max_train)
    print(f"  Train: {len(train_ds)} examples")

    # Split eval from residual data (use last N examples)
    eval_data: list[dict] = []
    if eval_path.exists():
        eval_ds = ResidualDataset(eval_path, max_examples=max_eval)
        eval_data = eval_ds.examples
    else:
        # Use last max_eval from train as eval
        eval_data = train_ds.examples[-max_eval:]
        train_ds.examples = train_ds.examples[:-max_eval]
    print(f"  Eval: {len(eval_data)} examples")

    # Class-weighted loss — compensate for label imbalance
    # Distribution: rw=24%, simp=18%, exact=15%, refine=7%, apply=5%, other=31%
    # Weight = 1/freq (inverse frequency), normalized
    class_freqs = torch.zeros(NUM_FAMILIES)
    for ex in train_ds.examples:
        class_freqs[tactic_to_family_idx(ex["tactic_base"])] += 1
    class_weights = (1.0 / class_freqs.clamp(min=1)).to(device)
    class_weights = class_weights / class_weights.sum() * NUM_FAMILIES  # normalize
    print(f"  Class weights: {dict(zip(TACTIC_FAMILIES, class_weights.tolist()))}")

    family_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    premise_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
        num_workers=0,
    )

    # Training loop
    best_top1 = 0.0
    log_entries: list[dict] = []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_fam_loss = 0.0
        epoch_prem_loss = 0.0
        n_batches = 0

        for batch in loader:
            optimizer.zero_grad()
            emb, fam_labels, prem_labels = _encode_batch(encoder, batch, device)
            fam_logits, prem_logits = model(emb)

            l_fam = family_loss_fn(fam_logits, fam_labels)
            l_prem = premise_loss_fn(prem_logits, prem_labels)
            loss = l_fam + 0.3 * l_prem  # family is primary objective

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_fam_loss += l_fam.item()
            epoch_prem_loss += l_prem.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_fam = epoch_fam_loss / max(n_batches, 1)
        avg_prem = epoch_prem_loss / max(n_batches, 1)

        # Eval
        metrics = _eval_metrics(model, encoder, eval_data, device, batch_size)
        print(
            f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} "
            f"fam={avg_fam:.4f} | "
            f"top1={metrics['top1']:.3f} top3={metrics['top3']:.3f} "
            f"macF1={metrics['macro_f1']:.3f} "
            f"top5ex={metrics['top5_exc_other']:.3f}"
        )

        log_entries.append(
            {
                "epoch": epoch + 1,
                "loss": round(avg_loss, 4),
                "fam_loss": round(avg_fam, 4),
                "prem_loss": round(avg_prem, 4),
                **metrics,
            }
        )

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "metrics": metrics,
                },
                ckpt_dir / f"{run_id}_best.pt",
            )

    elapsed = time.time() - start

    # Save training log
    with open(run_dir / f"{run_id}_training_log.jsonl", "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    # Final eval with full breakdown
    final_metrics = _eval_metrics(model, encoder, eval_data, device, batch_size)
    print(f"\n=== Final Results ({elapsed:.0f}s) ===")
    print(f"  Top-1: {final_metrics['top1']:.3f}")
    print(f"  Top-3: {final_metrics['top3']:.3f}")
    print(f"  Macro-F1: {final_metrics['macro_f1']:.3f}")
    print(f"  Top-5 excl other: {final_metrics['top5_exc_other']:.3f}")
    print(f"  Premise-dep acc: {final_metrics['premise_dep_acc']:.3f}")
    print(f"  Local-only acc: {final_metrics['local_only_acc']:.3f}")
    print(f"  Arg-dep (exact/apply/refine): {final_metrics['arg_dep_acc']:.3f}")
    print(f"  Rewrite (simp/rw): {final_metrics['rewrite_acc']:.3f}")
    print("  Per-class:")
    for fam, stats in final_metrics["per_class"].items():
        print(
            f"    {fam:8s}: P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f} n={stats['support']}"
        )
    print("  Confusion matrix (true → pred):")
    for true_name, row in final_metrics["confusion"].items():
        top_preds = sorted(row.items(), key=lambda x: -x[1])[:3]
        preds_str = ", ".join(f"{p}:{c}" for p, c in top_preds)
        print(f"    {true_name:8s} → {preds_str}")
    print(f"  Best checkpoint: models/{run_id}_best.pt")

    return {
        "status": "complete",
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "best_top1": best_top1,
        "final_metrics": final_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual executor")
    parser.add_argument("--run-id", type=str, default="RES-001")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=5000)
    parser.add_argument("--train-data", type=Path, default=Path("data/residual_train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/residual_eval.jsonl"))
    args = parser.parse_args()

    result = train(
        args.train_data,
        args.eval_data,
        args.run_id,
        args.device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.max_train,
        args.max_eval,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
