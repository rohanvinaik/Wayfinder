"""Train residual executor from pre-computed embeddings.

Uses data/residual_embeddings.pt (pre-computed by the encoder on MPS)
so training only needs the MLP forward/backward — no encoder, fast on CPU.

Usage:
    python3 scripts/train_residual_cached.py --run-id RES-003 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Inline the model to avoid src imports on remote machines
TACTIC_FAMILIES = ["rw", "simp", "exact", "refine", "apply", "other"]
NUM_FAMILIES = len(TACTIC_FAMILIES)


class ResidualExecutor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_families=NUM_FAMILIES):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.family_head = nn.Linear(hidden_dim, num_families)
        self.premise_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.family_head(h), self.premise_head(h)


def train(run_id, data_path, epochs, batch_size, lr, eval_frac=0.02):
    torch.manual_seed(42)
    np.random.seed(42)

    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    print(f"Loading pre-computed embeddings from {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    emb = data["embeddings"]  # [N, 384]
    labels = data["family_labels"]  # [N]
    premise = data["premise_deps"].unsqueeze(1)  # [N, 1]
    n = emb.shape[0]
    print(f"  {n} examples, {emb.shape[1]}d")

    # Train/eval split
    n_eval = max(int(n * eval_frac), 1000)
    perm = torch.randperm(n)
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]

    train_ds = TensorDataset(emb[train_idx], labels[train_idx], premise[train_idx])
    eval_emb = emb[eval_idx]
    eval_labels = labels[eval_idx]
    eval_premise = premise[eval_idx]
    print(f"  Train: {len(train_ds)}, Eval: {n_eval}")

    # Class weights (inverse frequency)
    class_counts = torch.zeros(NUM_FAMILIES)
    for i in range(NUM_FAMILIES):
        class_counts[i] = (labels[train_idx] == i).sum().float()
    weights = (1.0 / class_counts.clamp(min=1))
    weights = weights / weights.sum() * NUM_FAMILIES
    print(f"  Class weights: {dict(zip(TACTIC_FAMILIES, [f'{w:.2f}' for w in weights.tolist()]))}")

    model = ResidualExecutor()
    print(f"  Model: {sum(p.numel() for p in model.parameters())} params")

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_top1 = 0.0
    log = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        nb = 0
        for batch_emb, batch_lab, batch_prem in loader:
            optimizer.zero_grad()
            fam_logits, prem_logits = model(batch_emb)
            loss = loss_fn(fam_logits, batch_lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        # Eval
        model.eval()
        with torch.no_grad():
            fam_logits, _ = model(eval_emb)
            preds = fam_logits.argmax(dim=-1)
            top1 = float((preds == eval_labels).float().mean())

            top3 = fam_logits.topk(3, dim=-1).indices
            top3_acc = float(sum(eval_labels[i] in top3[i] for i in range(n_eval)) / n_eval)

            # Per-class recall
            per_class = {}
            for idx, name in enumerate(TACTIC_FAMILIES):
                mask = eval_labels == idx
                if mask.sum() > 0:
                    per_class[name] = float((preds[mask] == idx).float().mean())
                else:
                    per_class[name] = 0.0

            # Macro-F1
            f1s = []
            for idx in range(NUM_FAMILIES):
                tp = float(((preds == idx) & (eval_labels == idx)).sum())
                fp = float(((preds == idx) & (eval_labels != idx)).sum())
                fn = float(((preds != idx) & (eval_labels == idx)).sum())
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                f1s.append(2 * p * r / max(p + r, 1e-8))
            macro_f1 = sum(f1s) / len(f1s)

            # Top-5 excluding other
            other_idx = TACTIC_FAMILIES.index("other")
            non_other_mask = eval_labels != other_idx
            if non_other_mask.sum() > 0:
                top5_exc = float((preds[non_other_mask] == eval_labels[non_other_mask]).float().mean())
            else:
                top5_exc = 0.0

        avg_loss = total_loss / max(nb, 1)
        print(
            f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
            f"top1={top1:.3f} top3={top3_acc:.3f} macF1={macro_f1:.3f} "
            f"top5ex={top5_exc:.3f} "
            f"[{', '.join(f'{n}:{v:.2f}' for n, v in per_class.items())}]"
        )
        log.append({"epoch": epoch+1, "loss": avg_loss, "top1": top1, "top3": top3_acc, "macro_f1": macro_f1, "top5_exc": top5_exc, "per_class": per_class})

        if top1 > best_top1:
            best_top1 = top1
            torch.save({"epoch": epoch+1, "model": model.state_dict(), "top1": top1, "macro_f1": macro_f1}, f"models/{run_id}_best.pt")

    elapsed = time.time() - t0

    # Final confusion matrix
    model.eval()
    with torch.no_grad():
        fam_logits, _ = model(eval_emb)
        preds = fam_logits.argmax(dim=-1)

    print(f"\n=== Final ({elapsed:.0f}s) ===")
    print(f"  Best top-1: {best_top1:.3f}")
    print(f"  Confusion (true → pred top-3):")
    for ti, tn in enumerate(TACTIC_FAMILIES):
        mask = eval_labels == ti
        if mask.sum() == 0:
            continue
        pred_counts = {}
        for pi in range(NUM_FAMILIES):
            c = int((preds[mask] == pi).sum())
            if c > 0:
                pred_counts[TACTIC_FAMILIES[pi]] = c
        top = sorted(pred_counts.items(), key=lambda x: -x[1])[:3]
        print(f"    {tn:8s} (n={int(mask.sum()):5d}) → {', '.join(f'{p}:{c}' for p, c in top)}")

    with open(run_dir / f"{run_id}_log.jsonl", "w") as f:
        for entry in log:
            f.write(json.dumps(entry, default=str) + "\n")

    print(f"\n  Checkpoint: models/{run_id}_best.pt")
    print(f"  Log: {run_dir}/{run_id}_log.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="RES-003")
    parser.add_argument("--data", default="data/residual_embeddings.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.run_id, args.data, args.epochs, args.batch_size, args.lr)
