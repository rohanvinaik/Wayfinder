"""Train a lightweight temporal controller: predict next progress lane.

Given the current search state (goal embedding, recent lanes, phase, step),
predict which lane will next make progress. This is the supervised imitation
learning baseline for the temporal controller.

Input: data/temporal_train_500.jsonl (from build_temporal_dataset.py)
Target: next_progress_lane (categorical)
Features: goal_emb(384) + recent_lanes_onehot(8) + phase_onehot(4) + scalars(4) = 400d

Usage:
    python -m scripts.train_temporal_controller \\
        --data data/temporal_train_500.jsonl \\
        --output models/temporal_controller_v1.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_LANES = [
    "automation", "interleaved_bootstrap", "cosine_rw", "cosine_rw_seq",
    "cosine_apply", "exec_selector_apply", "cosine_exact", "structural_core",
]
_LANE_INDEX = {l: i for i, l in enumerate(_LANES)}
_PHASES = ["structural_setup", "local_close", "automation_close", "repair_or_replan"]
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}


def _lane_onehot(lanes: list[str], max_recent: int = 5) -> list[float]:
    """Encode recent lanes as multi-hot over _LANES."""
    vec = [0.0] * len(_LANES)
    for lane in lanes[-max_recent:]:
        idx = _LANE_INDEX.get(lane, -1)
        if idx >= 0:
            vec[idx] = 1.0
    return vec


def _phase_onehot(phase: str) -> list[float]:
    vec = [0.0] * len(_PHASES)
    idx = _PHASE_INDEX.get(phase, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def load_data(
    path: str,
    encoder: SentenceTransformer,
    emb_cache: str = "",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load temporal training data. Returns (X, y, rows)."""
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            # Only train on steps with a known next progress lane
            if d.get("next_progress_lane"):
                rows.append(d)

    if not rows:
        raise ValueError(f"No rows with next_progress_lane in {path}")

    # Encode goals
    if emb_cache and Path(emb_cache).exists():
        goal_embs = np.load(emb_cache)
        if len(goal_embs) != len(rows):
            goal_embs = None
    else:
        goal_embs = None

    if goal_embs is None:
        logger.info("Encoding %d goals ...", len(rows))
        goals = [r.get("goal_state", "") for r in rows]
        goal_embs = encoder.encode(goals, show_progress_bar=True, normalize_embeddings=True, batch_size=64)
        if emb_cache:
            np.save(emb_cache, goal_embs)
            logger.info("Saved cache to %s", emb_cache)

    X_list = []
    y_list = []
    for i, row in enumerate(rows):
        feat = np.concatenate([
            goal_embs[i],                                                    # 384d
            _lane_onehot(row.get("recent_lanes", [])),                       # 8d
            _phase_onehot(row.get("phase", "")),                             # 4d
            [float(row.get("open_goals_count", 1)) / 10.0],                 # 1d
            [float(row.get("closed_goals_count", 0)) / 10.0],               # 1d
            [float(row.get("step", 0)) / 50.0],                             # 1d
            [float(row.get("eventual_theorem_success", False))],             # 1d
        ])
        X_list.append(feat)
        target_lane = row.get("next_progress_lane", "")
        y_list.append(_LANE_INDEX.get(target_lane, 0))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/temporal_train_500.jsonl")
    parser.add_argument("--output", default="models/temporal_controller_v1.pt")
    parser.add_argument("--emb-cache", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    X, y, rows = load_data(args.data, encoder, emb_cache=args.emb_cache)

    logger.info("Dataset: %d rows, %d classes", len(X), len(set(y.tolist())))
    logger.info("Class distribution: %s", Counter(y.tolist()).most_common())

    # Split by theorem
    rng = np.random.default_rng(42)
    theorems = list({r.get("theorem_id", ""): None for r in rows}.keys())
    rng.shuffle(theorems)
    n_val = max(1, int(len(theorems) * args.val_frac))
    val_set = set(theorems[:n_val])
    train_mask = np.array([r.get("theorem_id", "") not in val_set for r in rows])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # Model
    in_dim = X.shape[1]
    n_classes = len(_LANES)
    model = nn.Sequential(
        nn.Linear(in_dim, args.hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(args.hidden, args.hidden // 2),
        nn.ReLU(),
        nn.Linear(args.hidden // 2, n_classes),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    X_v = torch.tensor(X_val)
    y_v = torch.tensor(y_val)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(X_t), 256):
            idx = perm[i:i + 256]
            logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1)
            val_acc = float((val_preds == y_v).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info(
                "Epoch %2d | loss=%.4f | val_acc=%.3f",
                epoch, total_loss / max(steps, 1), val_acc,
            )

    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    with torch.no_grad():
        val_preds = model(X_v).argmax(dim=1).numpy()
        val_labels = y_v.numpy()

    # Per-class accuracy
    print("\n" + "=" * 60)
    print("Temporal Controller v1")
    print("=" * 60)
    print(f"  Data:     {args.data} ({len(X)} rows)")
    print(f"  Val acc:  {best_val_acc:.3f}")
    print(f"  Classes:  {n_classes}")
    print("\n  Per-class (val):")
    for lane_idx, lane_name in enumerate(_LANES):
        mask = val_labels == lane_idx
        if mask.sum() > 0:
            acc = (val_preds[mask] == lane_idx).mean()
            print(f"    {lane_name:30s}: {acc:.3f} (n={mask.sum()})")
    print("=" * 60)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden": args.hidden,
        "n_classes": n_classes,
        "lanes": _LANES,
        "phases": _PHASES,
        "val_acc": best_val_acc,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }, args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
