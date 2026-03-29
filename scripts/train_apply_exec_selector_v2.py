"""Train ExecSelector v2: compatibility-aware apply candidate scorer.

v1 used [goal_emb; cand_emb; cosine_score; filter_passed] — pure similarity.
v2 adds structural compatibility features derived from goal_shape_ir and
candidate_type_pp, targeting the gap between "semantically related" and
"unification-compatible."

New features (appended to v1's 770d):
  - goal_head / candidate_head match (1d)
  - goal_head one-hot (8d, top heads)
  - candidate conclusion head one-hot (8d)
  - goal binder count (1d)
  - candidate binder count (1d)
  - local-name overlap count (1d)
  - goal has_forall, has_equality, has_implication, has_exists, has_iff (5d)
  - candidate arity (1d)
  Total: 770 + 26 = 796d

Training data: step-0 + midstep probes combined (22K+ rows).
Eval grouping: by (theorem_full_name, goal_state) — not just theorem.
Loss: BCE with pos_weight (v1 baseline) + optional listwise ranking.

Usage:
    python -m scripts.train_apply_exec_selector_v2 \\
        --train data/apply_exec_train.jsonl data/apply_exec_midstep.jsonl \\
        --eval data/apply_exec_dataset.jsonl \\
        --output models/apply_exec_selector_v2.pt \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score  # type: ignore[import]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Top goal heads for one-hot encoding
_TOP_HEADS = ["eq", "le", "lt", "iff", "not", "and", "exists", "mem"]
_HEAD_INDEX = {h: i for i, h in enumerate(_TOP_HEADS)}
_BINDER_RE = re.compile(r"[({⦃][^:(){}⦃⦄]*:\s*[^(){}⦃⦄]+[)}⦄]")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_head(text: str) -> str:
    """Extract the head symbol from a type/goal target string."""
    text = text.strip()
    if text.startswith("("):
        text = text[1:]
    m = re.match(r"(\w[\w.]*)", text)
    return m.group(1).lower() if m else ""


def _head_onehot(head: str) -> list[float]:
    vec = [0.0] * len(_TOP_HEADS)
    idx = _HEAD_INDEX.get(head, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _count_binders(text: str) -> int:
    return len(_BINDER_RE.findall(text))


def _candidate_arity(cand_type: str) -> int:
    """Estimate arity from candidate type pp (count → arrows)."""
    return cand_type.count("→") + cand_type.count("->")


def _local_name_overlap(goal_locals: list[str], cand_type: str) -> int:
    """Count how many goal-local names appear in the candidate type."""
    if not goal_locals or not cand_type:
        return 0
    return sum(1 for name in goal_locals if name in cand_type)


def build_compat_features(row: dict) -> list[float]:
    """Build the 26d compatibility feature vector for one (goal, candidate) pair."""
    # Goal shape IR
    shape_ir = row.get("goal_shape_ir", {})
    goal_target = shape_ir.get("target", row.get("goal_state", ""))
    goal_head = shape_ir.get("target_head", _extract_head(goal_target))
    goal_locals = shape_ir.get("local_names", [])

    # Candidate type
    cand_type_obj = row.get("candidate_type_pp", "")
    if isinstance(cand_type_obj, dict):
        cand_type = cand_type_obj.get("pp", "")
    else:
        cand_type = str(cand_type_obj)

    # Extract candidate conclusion head (last part after all →)
    cand_parts = re.split(r"→|->", cand_type)
    cand_conclusion = cand_parts[-1].strip() if cand_parts else ""
    cand_head = _extract_head(cand_conclusion)

    # Features
    head_match = 1.0 if goal_head and cand_head and goal_head == cand_head else 0.0
    goal_head_oh = _head_onehot(goal_head)
    cand_head_oh = _head_onehot(cand_head)
    goal_binders = _count_binders(goal_target) / 10.0
    cand_binders = _count_binders(cand_type) / 10.0
    name_overlap = _local_name_overlap(goal_locals, cand_type) / 5.0
    cand_arity = _candidate_arity(cand_type) / 5.0

    # Boolean shape features from IR
    has_forall = float(shape_ir.get("has_forall", False))
    has_equality = float(shape_ir.get("has_equality", False))
    has_implication = float(shape_ir.get("has_implication", False))
    has_exists = float(shape_ir.get("has_exists", False))
    has_iff = float(shape_ir.get("has_iff", False))

    return [
        head_match,           # 1d
        *goal_head_oh,        # 8d
        *cand_head_oh,        # 8d
        goal_binders,         # 1d
        cand_binders,         # 1d
        name_overlap,         # 1d
        has_forall,           # 1d
        has_equality,         # 1d
        has_implication,      # 1d
        has_exists,           # 1d
        has_iff,              # 1d
        cand_arity,           # 1d
    ]  # 26d total


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ExecSelectorV2(nn.Module):
    def __init__(self, emb_dim: int = 384, compat_dim: int = 26, hidden: int = 256) -> None:
        super().__init__()
        in_dim = emb_dim * 2 + 2 + compat_dim
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
# Data loading
# ---------------------------------------------------------------------------


def load_rows(*paths: str) -> list[dict]:
    rows = []
    for path in paths:
        if not Path(path).exists():
            logger.warning("File not found: %s — skipping", path)
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        logger.info("Loaded %d rows from %s (total %d)", len(rows), path, len(rows))
    return rows


def theorem_goal_split(
    rows: list[dict],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split by theorem_full_name (grouped). Fixed from v1 which only grouped by theorem."""
    theorems = list({r["theorem_full_name"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(theorems)
    n_val = max(1, int(len(theorems) * val_frac))
    val_set = set(theorems[:n_val])
    train = [r for r in rows if r["theorem_full_name"] not in val_set]
    val = [r for r in rows if r["theorem_full_name"] in val_set]
    return train, val


def build_features(
    rows: list[dict],
    goal_embs: np.ndarray,
    cand_embs: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [goal_emb; cand_emb; cosine_score; filter; compat_features]."""
    n = len(rows)
    emb_dim = goal_embs.shape[1]
    compat_dim = 26
    X = np.zeros((n, emb_dim * 2 + 2 + compat_dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        X[i, :emb_dim] = goal_embs[i]
        X[i, emb_dim:2 * emb_dim] = cand_embs[i]
        X[i, 2 * emb_dim] = float(r.get("cosine_score", 0.0))
        X[i, 2 * emb_dim + 1] = float(r.get("passed_static_filter", r.get("filter_passed", False)))
        compat = build_compat_features(r)
        X[i, 2 * emb_dim + 2:] = compat
        y[i] = float(r.get("executable", 0))

    return torch.from_numpy(X), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: ExecSelectorV2,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    pos_weight_scale: float = 1.0,
) -> dict:
    pos = y_train.sum().item()
    neg = len(y_train) - pos
    pw = float(neg / max(pos, 1)) * pos_weight_scale
    logger.info("pos_weight=%.2f (pos=%d, neg=%d)", pw, int(pos), int(neg))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))

    best_ap = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        model.eval()
        with torch.no_grad():
            val_scores = model.score(X_val).numpy()
            val_labels = y_val.numpy()
            ap = float(average_precision_score(val_labels, val_scores))

        if ap > best_ap:
            best_ap = ap
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("Epoch %2d | loss=%.4f | val_PR-AUC=%.4f", epoch, total_loss / max(steps, 1), ap)

    if best_state:
        model.load_state_dict(best_state)
    return {"best_pr_auc": best_ap}


# ---------------------------------------------------------------------------
# Eval: LeanAccepted@top-k grouped by (theorem, goal)
# ---------------------------------------------------------------------------


def lean_accepted_topk(
    model: ExecSelectorV2,
    rows: list[dict],
    goal_embs: np.ndarray,
    cand_embs: np.ndarray,
    k_values: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Group by (theorem_full_name, goal_state), rank by model score, check top-k."""
    model.eval()
    X, _ = build_features(rows, goal_embs, cand_embs)

    with torch.no_grad():
        all_scores = model.score(X).numpy()

    # Group by (theorem, goal)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = f"{r['theorem_full_name']}::{r.get('goal_state', '')[:80]}"
        groups[key].append(i)

    results: dict[str, float] = {"n_groups": float(len(groups))}

    for k in k_values:
        model_topk_acc = 0
        cosine_topk_acc = 0
        for key, indices in groups.items():
            if not indices:
                continue
            # Model top-k
            scores = all_scores[indices]
            topk_idx = np.argsort(-scores)[:k]
            model_topk_acc += int(any(rows[indices[j]]["executable"] == 1 for j in topk_idx))
            # Cosine top-k
            cosine_sorted = sorted(indices, key=lambda i: rows[i].get("cosine_rank", 99))[:k]
            cosine_topk_acc += int(any(rows[j]["executable"] == 1 for j in cosine_sorted))

        denom = max(len(groups), 1)
        results[f"model_top{k}"] = model_topk_acc / denom
        results[f"cosine_top{k}"] = cosine_topk_acc / denom

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", default=["data/apply_exec_train.jsonl"])
    parser.add_argument("--eval", default="data/apply_exec_dataset.jsonl")
    parser.add_argument("--output", default="models/apply_exec_selector_v2.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    args = parser.parse_args()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load training data
    train_rows = load_rows(*args.train)
    train_rows, val_rows = theorem_goal_split(train_rows)
    logger.info("Train: %d, Val: %d", len(train_rows), len(val_rows))

    # Encode
    logger.info("Encoding train goals+candidates (%d texts)...", 2 * len(train_rows))
    train_goal_embs = encoder.encode(
        [r["goal_state"] for r in train_rows],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )
    train_cand_embs = encoder.encode(
        [r["candidate"] for r in train_rows],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )

    logger.info("Encoding val goals+candidates (%d texts)...", 2 * len(val_rows))
    val_goal_embs = encoder.encode(
        [r["goal_state"] for r in val_rows],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )
    val_cand_embs = encoder.encode(
        [r["candidate"] for r in val_rows],
        batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )

    X_train, y_train = build_features(train_rows, train_goal_embs, train_cand_embs)
    X_val, y_val = build_features(val_rows, val_goal_embs, val_cand_embs)

    logger.info("Feature dim: %d", X_train.shape[1])

    model = ExecSelectorV2(hidden=args.hidden)
    train_results = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        pos_weight_scale=args.pos_weight_scale,
    )

    # Eval: LeanAccepted@top-k
    val_topk = lean_accepted_topk(model, val_rows, val_goal_embs, val_cand_embs, k_values=(1, 3))

    print("\n" + "=" * 60)
    print("ExecSelector v2")
    print("=" * 60)
    print(f"  Train rows:     {len(train_rows)} ({int(y_train.sum())} pos)")
    print(f"  Val rows:       {len(val_rows)} ({int(y_val.sum())} pos)")
    print(f"  Feature dim:    {X_train.shape[1]}")
    print(f"  Val PR-AUC:     {train_results['best_pr_auc']:.4f}")
    print(f"  Val groups:     {int(val_topk['n_groups'])}")
    print(f"  Model top-1:    {val_topk['model_top1']:.3f}")
    print(f"  Cosine top-1:   {val_topk['cosine_top1']:.3f}")
    print(f"  Model top-3:    {val_topk['model_top3']:.3f}")
    print(f"  Cosine top-3:   {val_topk['cosine_top3']:.3f}")
    print("=" * 60)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "emb_dim": 384,
        "compat_dim": 26,
        "hidden": args.hidden,
        "encoder": "all-MiniLM-L6-v2",
        "version": 2,
        "val_pr_auc": train_results["best_pr_auc"],
        "val_topk": val_topk,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }, args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
