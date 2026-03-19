"""EXP-RERANK-045: Learned reranker over cosine top-5 for apply + refine_named.

Trains a small MLP reranker that takes cheap typed features and cosine score
for each top-5 candidate and scores them. Gold candidate (annotated_premise)
is the supervision signal.

Architecture:
  Input: 7 features per candidate
    head_compat    (0/1): name contains goal head symbol
    shape_compat   (0/1): name suffix matches conclusion shape
    conclusion_sfx (0/1): name ends in recognisable conclusion suffix
    namespace_match(0/1): shared top-level namespace with theorem
    local_overlap  (float): candidate token ∩ hypothesis token overlap
    cosine_score   (float): raw MiniLM cosine similarity
    rank_norm      (float): (5 - cosine_rank) / 5.0

  Model: Linear(7→32) → ReLU → Linear(32→1)
  Loss:  Listwise cross-entropy (softmax over top-5 scores vs gold position)

Training:
  5-fold CV on eligible examples (gold in top-5) for stable estimates
  Full-data model saved to models/reranker045.pt

Metrics (per fold + mean ± std):
  top1/eligible  — how often does reranker put gold at rank 1
  MRR@5         — mean reciprocal rank over top-5

Baselines:
  cosine_top1    — rank by cosine score only
  rule_head_shape — best hand-crafted rule from EXP-RANK-044

Usage:
    python -m scripts.train_reranker045 \\
        --db data/proof_network_v3.db \\
        --eval data/canonical/canonical_residual_eval.jsonl \\
        --output models/reranker045.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

from src.apply_scoper import _SHAPE_SUFFIXES, classify_goal_shape, extract_goal_head
from src.proof_network import get_accessible_premises


# ---------------------------------------------------------------------------
# Feature extraction (shared with EXP-RANK-044)
# ---------------------------------------------------------------------------

N_FEATURES = 7


def _tokens(name: str) -> set[str]:
    parts = re.split(r"[._]", name)
    tokens: set[str] = set()
    for p in parts:
        for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p):
            tokens.add(sub.lower())
    return tokens


def _hypothesis_names(goal_state: str) -> set[str]:
    names: set[str] = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w'✝]*)(?:\s+\w[\w'✝]*)*\s*:", line)
        if m and m.group(1) not in ("case", "⊢"):
            names.add(m.group(1))
    return names


def feature_vector(
    candidate: str,
    cosine_score: float,
    cosine_rank: int,
    goal_head: str,
    goal_shape: str,
    theorem_namespace: str,
    hyp_names: set[str],
) -> list[float]:
    name_lower = candidate.lower()

    head_compat = 0.0
    if goal_head and len(goal_head) >= 3:
        h = goal_head.lower()
        if h in name_lower or h in candidate.split(".")[-1].lower():
            head_compat = 1.0

    shape_compat = 0.0
    suffixes = _SHAPE_SUFFIXES.get(goal_shape, [])
    if any(suf in name_lower for suf in suffixes):
        shape_compat = 1.0

    _CONCL_SFXS = ["_iff", "_eq", "_le", "_lt", "_mem", "_surjective", "_injective",
                   "_continuous", "_tendsto", "_measurable", "_mono", "_nonneg"]
    conclusion_sfx = 1.0 if any(name_lower.endswith(s) or s + "_" in name_lower
                                 for s in _CONCL_SFXS) else 0.0

    cand_ns = ".".join(candidate.split(".")[:2]) if "." in candidate else ""
    thm_ns = ".".join(theorem_namespace.split(".")[:2]) if "." in theorem_namespace else ""
    namespace_match = 1.0 if cand_ns and cand_ns == thm_ns else 0.0

    cand_tokens = _tokens(candidate)
    hyp_tokens: set[str] = set()
    for h in hyp_names:
        hyp_tokens |= _tokens(h)
    local_overlap = len(cand_tokens & hyp_tokens) / max(len(cand_tokens), 1)

    rank_norm = (5 - cosine_rank) / 5.0  # higher = better cosine rank

    return [head_compat, shape_compat, conclusion_sfx, namespace_match,
            local_overlap, float(cosine_score), rank_norm]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class RerankExample:
    """One goal with top-5 candidates. gold_idx is the position of gold in top-5."""
    theorem_id: str
    subset: str
    features: np.ndarray   # shape (5, N_FEATURES)
    gold_idx: int          # 0-4, index of gold in top-5


def load_examples(eval_path: str, families: list[str]) -> list[dict]:
    examples = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("step_index", 0) == 0 and ex.get("family") in families:
                examples.append(ex)
    return examples


def label_subset(ex: dict) -> str:
    if ex.get("family") == "apply":
        return "apply"
    prem = ex.get("annotated_premise", "")
    ir = ex.get("canonical_action_ir", "")
    tac = ex.get("tactic_text", "")
    body = re.sub(r"^refine\s*", "", (ir or tac).strip())
    if prem and (body.startswith(prem) or body.startswith(f"({prem}")):
        return "refine_named"
    return "refine_anon"


def build_dataset(
    examples: list[dict],
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder,
    top_k: int = 5,
) -> list[RerankExample]:
    dataset: list[RerankExample] = []
    skipped = 0

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            logger.info("  building dataset [%d/%d]", i, len(examples))
        gold = ex.get("annotated_premise", "")
        subset = label_subset(ex)
        if not gold or subset == "refine_anon":
            skipped += 1
            continue

        theorem_id = ex["theorem_full_name"]
        goal_state = ex.get("goal_state_before", "")

        tid = name_to_id.get(theorem_id)
        if tid is None:
            skipped += 1
            continue
        pids = get_accessible_premises(conn, tid)
        accessible = [id_to_name[pid] for pid in pids if pid in id_to_name]
        if not accessible or gold not in accessible:
            skipped += 1
            continue

        # Cosine rank
        goal_emb = encoder.encode([goal_state], normalize_embeddings=True)
        cand_embs = encoder.encode(accessible, normalize_embeddings=True)
        scores = (goal_emb @ cand_embs.T).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        top5 = [(accessible[i], float(scores[i])) for i in ranked_idx]
        top5_names = [c for c, _ in top5]

        if gold not in top5_names:
            skipped += 1
            continue

        gold_idx = top5_names.index(gold)
        goal_head = extract_goal_head(goal_state)
        goal_shape = classify_goal_shape(goal_state)
        hyp_names = _hypothesis_names(goal_state)

        feats = []
        for rank, (name, score) in enumerate(top5):
            fv = feature_vector(name, score, rank, goal_head, goal_shape, theorem_id, hyp_names)
            feats.append(fv)
        # Pad to top_k if fewer than top_k accessible
        while len(feats) < top_k:
            feats.append([0.0] * N_FEATURES)

        dataset.append(RerankExample(
            theorem_id=theorem_id,
            subset=subset,
            features=np.array(feats, dtype=np.float32),
            gold_idx=gold_idx,
        ))

    logger.info("Dataset: %d examples, %d skipped", len(dataset), skipped)
    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CandidateReranker(nn.Module):
    """Small MLP that scores a single candidate given its feature vector."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, top_k, n_features) → scores: (batch, top_k)"""
        return self.net(x).squeeze(-1)  # (batch, top_k)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_fold(
    train_data: list[RerankExample],
    n_epochs: int = 50,
    lr: float = 3e-3,
    device: str = "cpu",
) -> CandidateReranker:
    model = CandidateReranker().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for ex in train_data:
            feats = torch.tensor(ex.features, dtype=torch.float32).unsqueeze(0).to(device)
            labels = torch.tensor([ex.gold_idx], dtype=torch.long).to(device)
            scores = model(feats)  # (1, top_k)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.debug("  epoch %d loss=%.4f", epoch + 1, total_loss / len(train_data))

    return model


def evaluate(
    model: CandidateReranker,
    data: list[RerankExample],
    device: str = "cpu",
) -> tuple[float, float]:
    """Returns (top1_rate, mrr)."""
    model.eval()
    top1 = 0
    mrr = 0.0
    with torch.no_grad():
        for ex in data:
            feats = torch.tensor(ex.features, dtype=torch.float32).unsqueeze(0).to(device)
            scores = model(feats).squeeze(0).cpu().numpy()  # (top_k,)
            pred_rank = int(np.argmax(scores))
            gold_rank_in_pred = int(np.argsort(-scores).tolist().index(ex.gold_idx))
            top1 += int(pred_rank == ex.gold_idx)
            mrr += 1.0 / (gold_rank_in_pred + 1)
    n = len(data)
    return top1 / max(n, 1), mrr / max(n, 1)


def baseline_cosine(data: list[RerankExample]) -> tuple[float, float]:
    """Baseline: cosine top-1 = just pick rank 0 (first in top-5)."""
    top1 = sum(1 for ex in data if ex.gold_idx == 0)
    mrr = sum(1.0 / (ex.gold_idx + 1) for ex in data)
    n = len(data)
    return top1 / max(n, 1), mrr / max(n, 1)


def baseline_head_shape(data: list[RerankExample]) -> tuple[float, float]:
    """Baseline: rule_head_shape from EXP-RANK-044 — head*20 + shape*10 + cosine_score."""
    top1 = 0
    mrr = 0.0
    for ex in data:
        scores = []
        for fv in ex.features:
            # fv = [head_compat, shape_compat, conclusion_sfx, namespace_match,
            #        local_overlap, cosine_score, rank_norm]
            s = fv[0] * 20 + fv[1] * 10 + fv[5]
            scores.append(s)
        pred_order = np.argsort(scores)[::-1].tolist()
        gold_pos = pred_order.index(ex.gold_idx)
        top1 += int(gold_pos == 0)
        mrr += 1.0 / (gold_pos + 1)
    n = len(data)
    return top1 / max(n, 1), mrr / max(n, 1)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    dataset: list[RerankExample],
    n_folds: int = 5,
    n_epochs: int = 80,
    lr: float = 3e-3,
    device: str = "cpu",
) -> dict:
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(dataset))
    folds = np.array_split(indices, n_folds)

    fold_results = []
    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx].tolist()
        train_idx = [i for j, f in enumerate(folds) if j != fold_idx for i in f.tolist()]

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        model = train_fold(train_data, n_epochs=n_epochs, lr=lr, device=device)
        top1, mrr = evaluate(model, val_data, device=device)
        fold_results.append({"top1": top1, "mrr": mrr, "n_val": len(val_data)})
        logger.info("  fold %d: top1=%.3f  MRR=%.3f  (n_val=%d)",
                    fold_idx + 1, top1, mrr, len(val_data))

    top1_scores = [r["top1"] for r in fold_results]
    mrr_scores = [r["mrr"] for r in fold_results]
    return {
        "fold_results": fold_results,
        "top1_mean": float(np.mean(top1_scores)),
        "top1_std": float(np.std(top1_scores)),
        "mrr_mean": float(np.mean(mrr_scores)),
        "mrr_std": float(np.std(mrr_scores)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _report_subset(
    label: str,
    train_data: list[RerankExample],
    eval_data: list[RerankExample],
    n_epochs: int,
    lr: float,
    device: str,
    n_folds: int,
) -> CandidateReranker | None:
    """Train on train_data, evaluate on eval_data. Returns trained model."""
    if not eval_data:
        return None
    cos_top1, cos_mrr = baseline_cosine(eval_data)
    hs_top1, hs_mrr = baseline_head_shape(eval_data)
    n = len(eval_data)

    print(f"\n  [{label}]  train={len(train_data)}, eval={n}")
    print(f"  {'Method':<24} {'top1/eval':>10} {'MRR@5':>8}")
    print("  " + "-" * 45)
    print(f"  {'cosine_top1':<24} {cos_top1*n:>5.0f}/{n:<4}  {cos_mrr:>7.3f}")
    print(f"  {'rule_head_shape':<24} {hs_top1*n:>5.0f}/{n:<4}  {hs_mrr:>7.3f}")

    if train_data:
        model = train_fold(train_data, n_epochs=n_epochs, lr=lr, device=device)
        top1, mrr = evaluate(model, eval_data, device=device)
        print(f"  {'reranker (held-out)':<24} {top1*n:>5.1f}/{n:<4}  {mrr:>7.3f}")
        return model
    else:
        # Fall back to CV on eval_data only
        cv = cross_validate(eval_data, n_folds=n_folds, n_epochs=n_epochs, lr=lr, device=device)
        top1_n = cv["top1_mean"] * n
        print(f"  {'reranker (CV mean)':<24} {top1_n:>5.1f}/{n:<4}  {cv['mrr_mean']:>7.3f}  "
              f"±(top1={cv['top1_std']:.3f})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-RERANK-045: learned reranker training")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--train", default="data/canonical/canonical_residual_train.jsonl",
                        help="Training set (empty string → CV on eval only)")
    parser.add_argument("--eval", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--output", default="models/reranker045.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=0,
                        help="Cap training examples per family (0=unlimited)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded MiniLM encoder")

    conn = sqlite3.connect(args.db)
    rows = conn.execute("SELECT id, name FROM entities").fetchall()
    id_to_name = {eid: name for eid, name in rows}
    name_to_id = {name: eid for eid, name in rows}

    # Build eval dataset
    logger.info("Building eval dataset from %s...", args.eval)
    raw_eval = load_examples(args.eval, ["apply", "refine"])
    eval_dataset = build_dataset(raw_eval, conn, id_to_name, name_to_id, encoder)

    # Build train dataset (if provided)
    train_dataset: list[RerankExample] = []
    if args.train:
        logger.info("Building train dataset from %s...", args.train)
        raw_train = load_examples(args.train, ["apply", "refine"])
        if args.limit_train > 0:
            import random
            random.seed(42)
            raw_train = random.sample(raw_train, min(args.limit_train, len(raw_train)))
        train_dataset = build_dataset(raw_train, conn, id_to_name, name_to_id, encoder)

    conn.close()

    # Split by family
    train_apply  = [ex for ex in train_dataset if ex.subset == "apply"]
    train_refine = [ex for ex in train_dataset if ex.subset == "refine_named"]
    eval_apply   = [ex for ex in eval_dataset  if ex.subset == "apply"]
    eval_refine  = [ex for ex in eval_dataset  if ex.subset == "refine_named"]
    logger.info("Train: apply=%d, refine_named=%d | Eval: apply=%d, refine_named=%d",
                len(train_apply), len(train_refine), len(eval_apply), len(eval_refine))

    print("\n" + "=" * 68)
    print("EXP-RERANK-045b: Scaled reranker (apply + refine_named)")
    print("=" * 68)

    model_apply  = _report_subset("apply",        train_apply,  eval_apply,
                                  args.epochs, args.lr, args.device, args.folds)
    model_refine = _report_subset("refine_named", train_refine, eval_refine,
                                  args.epochs, args.lr, args.device, args.folds)

    # Save per-family models
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for suffix, model, n_train in [("apply", model_apply, len(train_apply)),
                                    ("refine", model_refine, len(train_refine))]:
        if model is None:
            continue
        out = Path(args.output).with_stem(Path(args.output).stem + f"_{suffix}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_features": N_FEATURES,
            "hidden": 32,
            "n_train": n_train,
            "family": suffix,
        }, out)
        logger.info("Saved %s reranker to %s", suffix, out)

    print("=" * 68)
    print("=" * 68)


if __name__ == "__main__":
    main()
