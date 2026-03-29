"""EXP-051: Train apply-trigger classifier.

Binary classifier: given a search residual state, should the controller
invoke the apply specialist?

Features:
  - goal_emb (384d MiniLM): current goal state
  - search_stage_onehot (4d): post_ib_fail / post_rw / post_auto / mid_search
  - scalar features (8d): open_goal_count, cosine/selector scores, candidate count, recent_lanes
  - goal_shape_features (14d): binder/equality/iff/arrow/forall/exists counts etc.
  - namespace_prefix hash (8d): hashed namespace bucket
  - step (1d): normalized search step
  Total: ~419d

Target: can_apply (0/1) — whether ANY candidate in the probed pool was
Lean-accepted. Auxiliary: selector_top1_accepted (multitask, not primary).

Split: by theorem_id (grouped), not by row.
Model selection: best val F1, not BCE loss.
Threshold: swept on val set, with per-stage calibration reported.

Usage:
    python -m scripts.train_trigger_classifier \\
        --data data/apply_trigger_train_full.jsonl \\
        --output models/apply_trigger_v1.pt \\
        --epochs 30 \\
        --min-positives 30

    # With cached embeddings (fast iteration):
    python -m scripts.train_trigger_classifier \\
        --data data/apply_trigger_train_full.jsonl \\
        --emb-cache data/trigger_goal_embs.npy \\
        --output models/apply_trigger_v2.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_STAGES = ["post_ib_fail", "post_rw", "post_auto", "mid_search"]
_STAGE_INDEX = {s: i for i, s in enumerate(_STAGES)}
_NS_BUCKETS = 8  # namespace hash buckets

# goal_shape_features keys (from collect_trigger_states.py)
_SHAPE_KEYS = [
    "char_len", "token_len", "binder_count", "forall_count", "exists_count",
    "arrow_count", "iff_count", "eq_count", "neq_count", "and_count",
    "or_count", "not_count", "typeclass_count", "type_count",
]
_SHAPE_NORMS = {
    "char_len": 500.0, "token_len": 100.0, "binder_count": 10.0,
    "forall_count": 5.0, "exists_count": 3.0, "arrow_count": 5.0,
    "iff_count": 3.0, "eq_count": 5.0, "neq_count": 3.0,
    "and_count": 3.0, "or_count": 3.0, "not_count": 3.0,
    "typeclass_count": 10.0, "type_count": 3.0,
}


def _stage_onehot(stage: str) -> list[float]:
    vec = [0.0] * len(_STAGES)
    idx = _STAGE_INDEX.get(stage, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _ns_hash(ns: str) -> list[float]:
    """Hash namespace prefix into a fixed-size bucket vector."""
    vec = [0.0] * _NS_BUCKETS
    if ns:
        h = int(hashlib.md5(ns.encode()).hexdigest(), 16) % _NS_BUCKETS
        vec[h] = 1.0
    return vec


def _shape_features(row: dict) -> list[float]:
    """Extract normalized goal_shape_features from row."""
    shape = row.get("goal_shape_features", {})
    return [float(shape.get(k, 0)) / _SHAPE_NORMS.get(k, 1.0) for k in _SHAPE_KEYS]


def load_dataset(
    path: str,
    encoder: Any | None = None,
    label_sources: tuple[str, ...] = ("lean_probe",),
    target_field: str = "can_apply",
    emb_cache: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load trigger dataset. Returns (X, y_primary, y_aux, meta).

    y_primary = can_apply, y_aux = selector_top1_accepted.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("row_type") != "trigger_state":
                continue
            if d.get("label_source") not in label_sources:
                continue
            if d.get(target_field) is None:
                continue
            rows.append(d)

    if not rows:
        raise ValueError(f"No trigger rows with label_source in {label_sources} found in {path}")

    # Goal embeddings: load from cache or encode
    if emb_cache and Path(emb_cache).exists():
        logger.info("Loading cached embeddings from %s", emb_cache)
        goal_embs = np.load(emb_cache)
        if len(goal_embs) != len(rows):
            logger.warning(
                "Cache size mismatch (%d vs %d rows) — re-encoding", len(goal_embs), len(rows)
            )
            goal_embs = None
        else:
            goal_embs = goal_embs  # noqa: PLW0127
    else:
        goal_embs = None

    if goal_embs is None:
        if encoder is None:
            raise ValueError("No encoder and no valid embedding cache")
        logger.info("Encoding %d goal states ...", len(rows))
        goals = [r["goal_state"] for r in rows]
        goal_embs = encoder.encode(
            goals, show_progress_bar=True, normalize_embeddings=True, batch_size=64
        )
        if emb_cache:
            np.save(emb_cache, goal_embs)
            logger.info("Saved embedding cache to %s", emb_cache)

    X_list = []
    y_list = []
    y_aux_list = []
    for i, row in enumerate(rows):
        feat = np.concatenate([
            goal_embs[i],                                                    # 384d
            _stage_onehot(row.get("search_stage", "")),                      # 4d
            [float(row.get("open_goal_count", 1)) / 10.0],                  # 1d
            [min(len(row.get("recent_lanes", [])), 5) / 5.0],               # 1d
            _shape_features(row),                                            # 14d
            _ns_hash(row.get("namespace_prefix", "")),                       # 8d
            [float(row.get("step", 0)) / 50.0],                             # 1d
            [1.0 if row.get("lane_provenance", "") == "none" else 0.0],      # 1d
        ])
        X_list.append(feat)
        y_list.append(float(row[target_field]))
        y_aux_list.append(
            float(row.get("selector_top1_accepted", 0) or 0)
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    y_aux = np.array(y_aux_list, dtype=np.float32)
    return X, y, y_aux, rows


def grouped_split(
    rows: list[dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split indices by theorem_id groups. All rows from a theorem go to train or val."""
    rng = np.random.default_rng(seed)

    theorem_rows: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        tid = row.get("theorem_id", f"unknown_{i}")
        theorem_rows.setdefault(tid, []).append(i)

    pos_theorems = []
    neg_theorems = []
    for tid, indices in theorem_rows.items():
        has_pos = any(rows[i].get("can_apply") == 1 for i in indices)
        if has_pos:
            pos_theorems.append(tid)
        else:
            neg_theorems.append(tid)

    rng.shuffle(pos_theorems)
    rng.shuffle(neg_theorems)

    n_val_pos = max(1, int(len(pos_theorems) * val_frac))
    n_val_neg = max(1, int(len(neg_theorems) * val_frac))
    val_theorems = set(pos_theorems[:n_val_pos]) | set(neg_theorems[:n_val_neg])

    val_idx = []
    train_idx = []
    for tid, indices in theorem_rows.items():
        if tid in val_theorems:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    logger.info(
        "Grouped split: %d theorems -> %d train (%d thm), %d val (%d thm)",
        len(theorem_rows), len(train_idx),
        len(theorem_rows) - len(val_theorems),
        len(val_idx), len(val_theorems),
    )
    return train_idx, val_idx


def _sweep_threshold(
    probs: np.ndarray, y: np.ndarray, rows: list[dict],
) -> dict:
    """Sweep thresholds for best global F1, plus per-stage thresholds."""
    # Global sweep
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(0.2, 0.8, 0.01):
        preds = (probs > thr).astype(float)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    # Per-stage sweep
    stage_thresholds: dict[str, float] = {}
    for stage in _STAGES:
        stage_mask = np.array([r.get("search_stage") == stage for r in rows])
        if stage_mask.sum() == 0:
            stage_thresholds[stage] = best_thr
            continue
        sp = probs[stage_mask]
        sy = y[stage_mask]
        best_sf1 = 0.0
        best_st = best_thr
        for thr in np.arange(0.2, 0.8, 0.01):
            preds = (sp > thr).astype(float)
            tp = ((preds == 1) & (sy == 1)).sum()
            fp = ((preds == 1) & (sy == 0)).sum()
            fn = ((preds == 0) & (sy == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_sf1:
                best_sf1 = f1
                best_st = float(thr)
        stage_thresholds[stage] = round(best_st, 2)

    return {
        "best_threshold": round(best_thr, 2),
        "best_f1": round(best_f1, 4),
        "stage_thresholds": stage_thresholds,
    }


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_aux_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_aux_val: np.ndarray,
    rows_val: list[dict],
    hidden: int = 128,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    pos_weight_scale: float = 1.0,
    aux_weight: float = 0.3,
) -> Any:
    import torch
    import torch.nn as nn

    in_dim = X_train.shape[1]
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight_val = float(neg_count / max(pos_count, 1)) * pos_weight_scale
    logger.info("pos_weight=%.2f (pos=%d, neg=%d, scale=%.2f)",
                pos_weight_val, int(pos_count), int(neg_count), pos_weight_scale)

    # Shared trunk + two heads
    trunk = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
    )
    head_primary = nn.Linear(hidden // 2, 1)   # can_apply
    head_aux = nn.Linear(hidden // 2, 1)        # selector_top1_accepted

    params = list(trunk.parameters()) + list(head_primary.parameters()) + list(head_aux.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    crit_primary = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32)
    )
    crit_aux = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train).unsqueeze(1)
    y_aux_t = torch.tensor(y_aux_train).unsqueeze(1)
    X_v = torch.tensor(X_val)

    n = len(X_t)
    best_val_f1 = 0.0
    best_state: dict[str, Any] = {}

    for epoch in range(epochs):
        trunk.train()
        head_primary.train()
        head_aux.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            features = trunk(xb)
            loss_p = crit_primary(head_primary(features), y_t[idx])
            loss_a = crit_aux(head_aux(features), y_aux_t[idx])
            loss = loss_p + aux_weight * loss_a
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        # Val F1 for model selection
        trunk.eval()
        head_primary.eval()
        with torch.no_grad():
            val_feats = trunk(X_v)
            val_probs = torch.sigmoid(head_primary(val_feats).squeeze(1)).numpy()
            val_preds = (val_probs > 0.5).astype(float)

        tp = ((val_preds == 1) & (y_val == 1)).sum()
        fp = ((val_preds == 1) & (y_val == 0)).sum()
        fn = ((val_preds == 0) & (y_val == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        val_f1 = float(2 * prec * rec / max(prec + rec, 1e-9))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "trunk": copy.deepcopy(trunk.state_dict()),
                "head_primary": copy.deepcopy(head_primary.state_dict()),
                "head_aux": copy.deepcopy(head_aux.state_dict()),
            }

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %2d | loss=%.4f | val_f1=%.3f | prec=%.3f | rec=%.3f",
                epoch, epoch_loss / max(steps, 1), val_f1, prec, rec,
            )

    if best_state:
        trunk.load_state_dict(best_state["trunk"])
        head_primary.load_state_dict(best_state["head_primary"])
        head_aux.load_state_dict(best_state["head_aux"])

    return trunk, head_primary, head_aux


def evaluate(
    trunk: Any, head: Any, X: np.ndarray, y: np.ndarray, rows: list[dict],
    threshold: float = 0.5,
) -> dict:
    import torch
    trunk.eval()
    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(trunk(torch.tensor(X))).squeeze(1)).numpy()
        preds = (probs > threshold).astype(float)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    acc = float((preds == y).mean())

    def _confusion(indices: list[int]) -> dict[str, int]:
        return {
            "tp": sum(1 for i in indices if y[i] == 1 and preds[i] == 1),
            "fp": sum(1 for i in indices if y[i] == 0 and preds[i] == 1),
            "fn": sum(1 for i in indices if y[i] == 1 and preds[i] == 0),
            "tn": sum(1 for i in indices if y[i] == 0 and preds[i] == 0),
        }

    stage_indices: dict[str, list[int]] = {}
    ns_indices: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        stage_indices.setdefault(row.get("search_stage", "?"), []).append(i)
        ns_indices.setdefault(row.get("namespace_prefix", "?"), []).append(i)

    stage_stats = {s: _confusion(idx) for s, idx in stage_indices.items()}
    top_ns = sorted(ns_indices.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    ns_stats = {ns: _confusion(idx) for ns, idx in top_ns}

    sel_rows = [i for i, r in enumerate(rows) if r.get("selector_top1_accepted") is not None]
    sel_acc = sum(1 for i in sel_rows if rows[i]["selector_top1_accepted"] == 1) / max(len(sel_rows), 1)

    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
        "stage_stats": stage_stats, "ns_stats": ns_stats,
        "selector_top1_accepted_rate": round(sel_acc, 4),
    }


def print_eval(label: str, metrics: dict) -> None:
    print(f"\n{label} (threshold={metrics.get('threshold', 0.5):.2f})")
    print(f"  Accuracy:  {metrics['acc']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  selector_top1_accepted rate: {metrics['selector_top1_accepted_rate']:.3f}")

    print("\n  Per-stage:")
    for stage, s in sorted(metrics["stage_stats"].items()):
        total = s["tp"] + s["fp"] + s["fn"] + s["tn"]
        pos = s["tp"] + s["fn"]
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        print(
            f"    [{stage:15s}] P={prec:.2f} R={rec:.2f} "
            f"TP={s['tp']:3d} FP={s['fp']:3d} FN={s['fn']:3d} TN={s['tn']:3d} "
            f"(pos={pos}/{total})"
        )

    if metrics.get("ns_stats"):
        print("\n  Top namespaces:")
        for ns, s in metrics["ns_stats"].items():
            total = s["tp"] + s["fp"] + s["fn"] + s["tn"]
            pos = s["tp"] + s["fn"]
            print(f"    [{ns:25s}] pos={pos}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/apply_trigger_train_full.jsonl")
    parser.add_argument("--output", default="models/apply_trigger_v1.pt")
    parser.add_argument("--emb-cache", default="",
                        help="Path to .npy embedding cache (saves re-encoding)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--min-positives", type=int, default=30)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.3,
                        help="Weight for selector_top1_accepted auxiliary loss")
    parser.add_argument("--also-proxy", action="store_true")
    args = parser.parse_args()

    # Load encoder only if no valid cache
    encoder = None
    if not args.emb_cache or not Path(args.emb_cache).exists():
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

    label_sources = ("lean_probe", "selector_proxy") if args.also_proxy else ("lean_probe",)

    X, y, y_aux, rows = load_dataset(
        args.data, encoder, label_sources=label_sources,
        target_field="can_apply", emb_cache=args.emb_cache,
    )

    pos_count = int(y.sum())
    logger.info("Dataset: %d rows, %d positives (%.1f%%), feature_dim=%d",
                len(X), pos_count, 100 * pos_count / max(len(X), 1), X.shape[1])

    if pos_count < args.min_positives:
        print(f"\nInsufficient positives: {pos_count} < {args.min_positives}.")
        return

    train_idx, val_idx = grouped_split(rows, val_frac=args.val_frac)

    X_train, y_train, y_aux_train = X[train_idx], y[train_idx], y_aux[train_idx]
    X_val, y_val, y_aux_val = X[val_idx], y[val_idx], y_aux[val_idx]
    rows_val = [rows[i] for i in val_idx]
    rows_train = [rows[i] for i in train_idx]

    logger.info("Train: %d (%d pos), Val: %d (%d pos)",
                len(X_train), int(y_train.sum()), len(X_val), int(y_val.sum()))

    trunk, head_primary, head_aux = train(
        X_train, y_train, y_aux_train,
        X_val, y_val, y_aux_val, rows_val,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, pos_weight_scale=args.pos_weight_scale,
        aux_weight=args.aux_weight,
    )

    # Threshold sweep
    import torch
    trunk.eval()
    head_primary.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(
            head_primary(trunk(torch.tensor(X_val))).squeeze(1)
        ).numpy()
    sweep = _sweep_threshold(val_probs, y_val, rows_val)
    best_thr = sweep["best_threshold"]

    # Evaluate at default and swept thresholds
    val_default = evaluate(trunk, head_primary, X_val, y_val, rows_val, threshold=0.5)
    val_swept = evaluate(trunk, head_primary, X_val, y_val, rows_val, threshold=best_thr)
    train_metrics = evaluate(trunk, head_primary, X_train, y_train, rows_train, threshold=best_thr)

    print_eval("Train metrics", train_metrics)
    print_eval("Val metrics (threshold=0.50)", val_default)
    print_eval(f"Val metrics (threshold={best_thr:.2f}, swept)", val_swept)

    print(f"\n  Threshold sweep: best_thr={best_thr:.2f}, best_f1={sweep['best_f1']:.4f}")
    print(f"  Per-stage thresholds: {sweep['stage_thresholds']}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trunk_state_dict": trunk.state_dict(),
        "head_primary_state_dict": head_primary.state_dict(),
        "head_aux_state_dict": head_aux.state_dict(),
        "in_dim": X.shape[1],
        "hidden": args.hidden,
        "encoder": "all-MiniLM-L6-v2",
        "stages": _STAGES,
        "target_field": "can_apply",
        "best_threshold": best_thr,
        "stage_thresholds": sweep["stage_thresholds"],
        "val_metrics_default": val_default,
        "val_metrics_swept": val_swept,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "pos_count": pos_count,
        "pos_weight_scale": args.pos_weight_scale,
        "aux_weight": args.aux_weight,
    }, args.output)
    logger.info("Saved to %s", args.output)

    print("\n" + "=" * 60)
    print("EXP-051: Apply Trigger Classifier v2")
    print("=" * 60)
    print(f"  Data:        {args.data} ({len(X)} rows, {pos_count} pos)")
    print(f"  Features:    {X.shape[1]}d (goal_emb + stage + scalars + shape + ns_hash + step)")
    print(f"  Target:      can_apply (aux: selector_top1_accepted, weight={args.aux_weight})")
    print(
        f"  Split:       theorem-grouped ({len(set(r['theorem_id'] for r in rows_train))} train / "
        f"{len(set(r['theorem_id'] for r in rows_val))} val)"
    )
    print("  Selection:   best val F1")
    print(f"  Best thr:    {best_thr:.2f}")
    print(f"  Val F1@0.50: {val_default['f1']:.3f} (P={val_default['precision']:.3f} R={val_default['recall']:.3f})")
    print(f"  Val F1@{best_thr:.2f}: {val_swept['f1']:.3f} (P={val_swept['precision']:.3f} R={val_swept['recall']:.3f})")
    print(f"  Checkpoint:  {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
