"""
Train a binary executable selector for apply candidates.

Target: executable = accepted_with_goals ∪ closed  (label = 1)
        unification_mismatch / typeclass_missing / other  (label = 0)

Split: by theorem_full_name (no row-level leakage)

Model: pointwise MLP over [goal_emb; cand_emb; cosine_score; passed_static_filter]
       Input dim: 384 + 384 + 2 = 770
       Hidden: 256 -> 128
       Output: scalar logit (BCEWithLogitsLoss)

Loss: weighted BCE (pos_weight = neg/pos ratio)
Eval metrics:
  - PR-AUC (primary during training)
  - LeanAccepted@top-1 (reported at end using eval JSONL)

Usage:
    python -m scripts.train_apply_exec_selector \\
        --train data/apply_exec_train.jsonl \\
        --eval  data/apply_exec_dataset.jsonl \\
        --output models/apply_exec_selector_v1.pt \\
        --epochs 20 --lr 1e-3 --hidden 256
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score  # type: ignore[import]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ExecSelector(nn.Module):
    """Pointwise binary scorer: P(executable | goal, candidate)."""

    def __init__(self, emb_dim: int = 384, hidden: int = 256) -> None:
        super().__init__()
        in_dim = emb_dim * 2 + 2  # goal_emb + cand_emb + cosine_score + passed_filter
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Data loading + encoding
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def theorem_split(
    rows: list[dict],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split by theorem_full_name to prevent leakage."""
    theorems = list({r["theorem_full_name"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(theorems)
    n_val = max(1, int(len(theorems) * val_frac))
    val_set = set(theorems[:n_val])
    train = [r for r in rows if r["theorem_full_name"] not in val_set]
    val = [r for r in rows if r["theorem_full_name"] in val_set]
    return train, val


def encode_texts(
    encoder: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def build_features(
    rows: list[dict],
    goal_embs: np.ndarray,
    cand_embs: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate [goal_emb; cand_emb; cosine_score; passed_filter]."""
    n = len(rows)
    emb_dim = goal_embs.shape[1]
    X = np.zeros((n, emb_dim * 2 + 2), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        X[i, :emb_dim] = goal_embs[i]
        X[i, emb_dim : 2 * emb_dim] = cand_embs[i]
        X[i, -2] = float(r.get("cosine_score", 0.0))
        X[i, -1] = float(r.get("passed_static_filter", r.get("filter_passed", False)))
        y[i] = float(r.get("executable", 0))

    return torch.from_numpy(X), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: ExecSelector,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.train()
    idx = torch.randperm(len(X))
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start : start + batch_size]
        xb = X[batch_idx].to(device)
        yb = y[batch_idx].to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: ExecSelector,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    model.eval()
    scores = model.score(X.to(device)).cpu().numpy()
    labels = y.numpy()
    ap = float(average_precision_score(labels, scores))
    threshold = 0.5
    preds = (scores >= threshold).astype(float)
    acc = float((preds == labels).mean())
    return {"pr_auc": ap, "accuracy": acc}


# ---------------------------------------------------------------------------
# LeanAccepted@top-1 evaluation
# ---------------------------------------------------------------------------

def lean_accepted_top1(
    model: ExecSelector,
    eval_rows: list[dict],
    eval_goal_embs: np.ndarray,
    eval_cand_embs: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """
    Group eval rows by theorem+goal, rank by model score, check if top-1 is executable.
    Compare to cosine_top1 and static_filter_top1 baselines.
    """
    model.eval()
    emb_dim = eval_goal_embs.shape[1]

    # Build per-goal groups
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(eval_rows):
        key = r["theorem_full_name"]
        groups[key].append(i)

    model_top1_acc = 0
    cosine_top1_acc = 0
    filter_cosine_top1_acc = 0
    n_groups = 0

    X = np.zeros((len(eval_rows), emb_dim * 2 + 2), dtype=np.float32)
    for i, r in enumerate(eval_rows):
        X[i, :emb_dim] = eval_goal_embs[i]
        X[i, emb_dim : 2 * emb_dim] = eval_cand_embs[i]
        X[i, -2] = float(r.get("cosine_score", 0.0))
        X[i, -1] = float(r.get("passed_static_filter", r.get("filter_passed", False)))

    with torch.no_grad():
        all_scores = model.score(torch.from_numpy(X).to(device)).cpu().numpy()

    for key, indices in groups.items():
        if not indices:
            continue
        n_groups += 1
        group_scores = all_scores[indices]

        # Model top-1
        top_model = indices[int(np.argmax(group_scores))]
        model_top1_acc += int(eval_rows[top_model]["executable"] == 1)

        # Cosine top-1 (lowest cosine_rank)
        cosine_sorted = sorted(indices, key=lambda i: eval_rows[i].get("cosine_rank", 99))
        cosine_top1_acc += int(eval_rows[cosine_sorted[0]]["executable"] == 1)

        # Static filter + cosine top-1 (lowest cosine_rank among passed_static_filter)
        filtered = [i for i in indices if eval_rows[i].get("passed_static_filter", False)]
        if not filtered:
            filtered = indices  # fallback to all
        filter_sorted = sorted(filtered, key=lambda i: eval_rows[i].get("cosine_rank", 99))
        filter_cosine_top1_acc += int(eval_rows[filter_sorted[0]]["executable"] == 1)

    denom = float(max(n_groups, 1))
    return {
        "n_goals": float(n_groups),
        "model_top1": model_top1_acc / denom,
        "cosine_top1": cosine_top1_acc / denom,
        "filter_cosine_top1": filter_cosine_top1_acc / denom,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/apply_exec_train.jsonl")
    parser.add_argument("--eval",  default="data/apply_exec_dataset.jsonl")
    parser.add_argument("--output", default="models/apply_exec_selector_v1.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    logger.info("Loading train rows...")
    train_rows = load_rows(args.train)
    eval_rows  = load_rows(args.eval)
    logger.info("Train: %d rows, Eval: %d rows", len(train_rows), len(eval_rows))

    # Theorem-level split of train rows into train/val
    train_split, val_split = theorem_split(train_rows, val_frac=args.val_frac,
                                           seed=args.seed)
    n_pos_train = sum(r["executable"] for r in train_split)
    n_neg_train = len(train_split) - n_pos_train
    logger.info(
        "Train split: %d rows (%d pos / %d neg), Val split: %d rows",
        len(train_split), n_pos_train, n_neg_train, len(val_split),
    )

    # Encode
    logger.info("Loading encoder: %s", args.encoder)
    encoder = SentenceTransformer(args.encoder)

    logger.info("Encoding train goals + candidates...")
    train_goals = encode_texts(encoder, [r["goal_state"] for r in train_split])
    train_cands = encode_texts(encoder, [r["candidate"] for r in train_split])
    logger.info("Encoding val goals + candidates...")
    val_goals = encode_texts(encoder, [r["goal_state"] for r in val_split])
    val_cands = encode_texts(encoder, [r["candidate"] for r in val_split])
    logger.info("Encoding eval goals + candidates...")
    # apply_exec_dataset uses "goal_str"; apply_exec_train uses "goal_state"
    eval_goals = encode_texts(encoder, [r.get("goal_state", r.get("goal_str", "")) for r in eval_rows])
    eval_cands = encode_texts(encoder, [r["candidate"] for r in eval_rows])

    # Build tensors
    X_train, y_train = build_features(train_split, train_goals, train_cands)
    X_val,   y_val   = build_features(val_split,   val_goals,   val_cands)

    # Model + loss
    model = ExecSelector(emb_dim=384, hidden=args.hidden).to(device)
    pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    logger.info(
        "pos_weight=%.2f | model params=%d",
        pos_weight.item(),
        sum(p.numel() for p in model.parameters()),
    )

    # Training loop
    best_ap = 0.0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, X_train, y_train, optimizer, criterion,
            batch_size=args.batch_size, device=device,
        )
        val_metrics = evaluate(model, X_val, y_val, device=device)
        scheduler.step()

        ap = val_metrics["pr_auc"]
        logger.info(
            "Epoch %2d | loss=%.4f | val PR-AUC=%.4f | val acc=%.4f",
            epoch, loss, ap, val_metrics["accuracy"],
        )

        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # LeanAccepted@top-1 on holdout eval set
    la_metrics = lean_accepted_top1(
        model, eval_rows, eval_goals, eval_cands, device=device,
    )

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": best_state,
            "emb_dim": 384,
            "hidden": args.hidden,
            "encoder": args.encoder,
            "best_val_pr_auc": best_ap,
            "lean_accepted_top1": la_metrics,
        },
        args.output,
    )
    logger.info("Saved to %s", args.output)

    # Final report
    print("\n" + "=" * 60)
    print("ExecSelector v1 — Training complete")
    print("=" * 60)
    print(f"  Train rows        : {len(train_split)}  ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"  Val rows          : {len(val_split)}")
    print(f"  Best val PR-AUC   : {best_ap:.4f}")
    print()
    print(f"  LeanAccepted@top-1 (eval, {la_metrics['n_goals']} goals):")
    print(f"    model_top1        : {la_metrics['model_top1']:.3f}")
    print(f"    cosine_top1       : {la_metrics['cosine_top1']:.3f}")
    print(f"    filter+cosine_top1: {la_metrics['filter_cosine_top1']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
