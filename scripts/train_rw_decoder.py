"""Train the rw decoder — shape prediction + pointer over local vocabulary.

Uses canonical ActionIR supervision from data/rw_actionir_train.jsonl.
Pre-computes goal embeddings for fast training.

Stage 1 loss: CE on rewrite count + BCE on direction flags
Stage 2 loss: CE on pointer over local vocabulary (oracle premises)

Usage:
    python -m scripts.train_rw_decoder --run-id RW-001
    python -m scripts.train_rw_decoder --run-id RW-001 --epochs 20 --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.encoder import GoalEncoder
from src.rw_decoder import MAX_REWRITES, RwDecoder


class RwDataset(Dataset):  # type: ignore[type-arg]
    """Dataset of rw ActionIR examples with goal embeddings."""

    def __init__(self, examples: list[dict], goal_embeddings: torch.Tensor) -> None:
        self.examples = examples
        self.goal_embs = goal_embeddings

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        return self.examples[idx], self.goal_embs[idx]


def _build_vocab_for_example(
    ex: dict, all_symbols: list[str], sym_to_idx: dict
) -> tuple[list[int], int]:
    """Build local vocabulary indices and target index for one example.

    Returns (vocab_indices, target_leaf_idx_in_vocab).
    For now, the "local vocabulary" is the full symbol set (oracle).
    """
    # Target: first leaf symbol (simplification — predict first rewrite atom)
    target_sym = ex["leaf_symbols"][0] if ex["leaf_symbols"] else ""
    target_idx = sym_to_idx.get(target_sym, 0)
    return [], target_idx


def train(
    data_path: Path,
    run_id: str,
    device: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_examples: int | None = None,
) -> dict:
    """Train the rw decoder."""
    torch.manual_seed(42)
    np.random.seed(42)

    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # Load data
    print(f"Loading rw ActionIR data from {data_path}...")
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    print(f"  {len(examples)} examples")

    # Build symbol vocabulary (all unique leaf symbols)
    sym_counter: Counter[str] = Counter()
    for ex in examples:
        for s in ex["leaf_symbols"]:
            sym_counter[s] += 1
    # Top N symbols as vocabulary (rest mapped to UNK)
    vocab_size = min(len(sym_counter), 5000)
    top_symbols = [s for s, _ in sym_counter.most_common(vocab_size)]
    sym_to_idx = {s: i for i, s in enumerate(top_symbols)}
    print(f"  Vocabulary: {vocab_size} symbols (from {len(sym_counter)} unique)")

    # Encode goals
    print("Encoding goal states...")
    encoder = GoalEncoder.from_config(
        {"type": "all-MiniLM-L6-v2", "output_dim": 384, "frozen": True},
        device=device,
    )
    encoder.ensure_loaded()
    goal_embs = []
    enc_batch = 256
    for i in range(0, len(examples), enc_batch):
        batch_goals = [ex["goal_state"] for ex in examples[i : i + enc_batch]]
        emb = encoder.encode(batch_goals).cpu()
        goal_embs.append(emb)
    goal_emb_tensor = torch.cat(goal_embs, dim=0)
    print(f"  Goal embeddings: {goal_emb_tensor.shape}")

    # Encode vocabulary symbols (as pseudo-embeddings via encoder)
    print("Encoding vocabulary symbols...")
    vocab_embs = []
    for i in range(0, len(top_symbols), enc_batch):
        batch = top_symbols[i : i + enc_batch]
        emb = encoder.encode(batch).cpu()
        vocab_embs.append(emb)
    vocab_emb_tensor = torch.cat(vocab_embs, dim=0)  # [vocab_size, 384]
    print(f"  Vocabulary embeddings: {vocab_emb_tensor.shape}")

    # Train/eval split
    n_eval = min(1000, len(examples) // 10)
    perm = torch.randperm(len(examples))
    eval_idx = perm[:n_eval].tolist()
    train_idx = perm[n_eval:].tolist()

    train_examples = [examples[i] for i in train_idx]
    eval_examples = [examples[i] for i in eval_idx]
    train_embs = goal_emb_tensor[train_idx]
    eval_embs = goal_emb_tensor[eval_idx]

    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Build targets
    def build_targets(exs):
        counts = []  # rewrite count - 1 (0-indexed)
        dirs = []  # direction flags [batch, max_rewrites]
        leaf_indices = []  # first leaf symbol index in vocab

        for ex in exs:
            n = min(ex["n_rewrites"], MAX_REWRITES)
            counts.append(n - 1)

            dir_flags = [0.0] * MAX_REWRITES
            for i, d in enumerate(ex["directions"][:MAX_REWRITES]):
                dir_flags[i] = 1.0 if d == "backward" else 0.0
            dirs.append(dir_flags)

            first_sym = ex["leaf_symbols"][0] if ex["leaf_symbols"] else ""
            leaf_indices.append(sym_to_idx.get(first_sym, 0))

        return (
            torch.tensor(counts, dtype=torch.long),
            torch.tensor(dirs, dtype=torch.float32),
            torch.tensor(leaf_indices, dtype=torch.long),
        )

    train_counts, train_dirs, train_leaves = build_targets(train_examples)
    eval_counts, eval_dirs, eval_leaves = build_targets(eval_examples)

    # Model
    model = RwDecoder(goal_dim=384, hidden_dim=256).to(device)
    print(f"  RwDecoder: {sum(p.numel() for p in model.parameters())} params")

    # Loss
    count_loss_fn = nn.CrossEntropyLoss()
    dir_loss_fn = nn.BCEWithLogitsLoss()
    leaf_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Move shared vocab to device
    vocab_emb_dev = vocab_emb_tensor.to(device)

    # Training loop
    best_count_acc = 0.0
    log_entries = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm_t = torch.randperm(len(train_examples))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_examples), batch_size):
            idx = perm_t[i : i + batch_size]
            goal_batch = train_embs[idx].to(device)
            count_batch = train_counts[idx].to(device)
            dir_batch = train_dirs[idx].to(device)
            leaf_batch = train_leaves[idx].to(device)

            optimizer.zero_grad()

            # Stage 1: shape
            count_logits, dir_logits = model.forward_shape(goal_batch)
            l_count = count_loss_fn(count_logits, count_batch)
            l_dir = dir_loss_fn(dir_logits, dir_batch)

            # Stage 2: pointer (predict first leaf only for now)
            # Expand vocab for batch
            bsz = goal_batch.shape[0]
            vocab_expanded = vocab_emb_dev.unsqueeze(0).expand(bsz, -1, -1)
            positions = torch.zeros(bsz, 1, dtype=torch.long, device=device)
            pointer_logits = model.forward_leaves(goal_batch, vocab_expanded, positions)
            pointer_logits = pointer_logits.squeeze(1)  # [batch, vocab_size]
            l_leaf = leaf_loss_fn(pointer_logits, leaf_batch)

            loss = l_count + 0.5 * l_dir + l_leaf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Eval
        model.eval()
        with torch.no_grad():
            eval_goal = eval_embs.to(device)
            eval_count_t = eval_counts.to(device)
            eval_leaf_t = eval_leaves.to(device)

            c_logits, d_logits = model.forward_shape(eval_goal)
            count_preds = c_logits.argmax(dim=-1)
            count_acc = float((count_preds == eval_count_t).float().mean())

            dir_preds = (torch.sigmoid(d_logits) > 0.5).float()
            dir_acc = float((dir_preds[:, 0] == eval_dirs[:, 0].to(device)).float().mean())

            # Leaf accuracy
            bsz_e = eval_goal.shape[0]
            vocab_exp = vocab_emb_dev.unsqueeze(0).expand(bsz_e, -1, -1)
            pos_e = torch.zeros(bsz_e, 1, dtype=torch.long, device=device)
            ptr_logits = model.forward_leaves(eval_goal, vocab_exp, pos_e).squeeze(1)
            leaf_preds = ptr_logits.argmax(dim=-1)
            leaf_acc = float((leaf_preds == eval_leaf_t).float().mean())

            # Top-5 leaf accuracy
            top5 = ptr_logits.topk(5, dim=-1).indices
            leaf_top5 = float(sum(eval_leaf_t[j] in top5[j] for j in range(bsz_e)) / bsz_e)

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} "
                f"count_acc={count_acc:.3f} dir_acc={dir_acc:.3f} "
                f"leaf_top1={leaf_acc:.3f} leaf_top5={leaf_top5:.3f}"
            )

        log_entries.append(
            {
                "epoch": epoch + 1,
                "loss": round(avg_loss, 4),
                "count_acc": round(count_acc, 4),
                "dir_acc": round(dir_acc, 4),
                "leaf_top1": round(leaf_acc, 4),
                "leaf_top5": round(leaf_top5, 4),
            }
        )

        if count_acc > best_count_acc:
            best_count_acc = count_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "vocab": top_symbols,
                    "metrics": log_entries[-1],
                },
                f"models/{run_id}_best.pt",
            )

    elapsed = time.time() - t0

    with open(run_dir / f"{run_id}_log.jsonl", "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n=== Final ({elapsed:.0f}s) ===")
    print(f"  Best count_acc: {best_count_acc:.3f}")
    print(
        f"  Final: count={count_acc:.3f} dir={dir_acc:.3f} leaf_top1={leaf_acc:.3f} leaf_top5={leaf_top5:.3f}"
    )
    print(f"  Checkpoint: models/{run_id}_best.pt")

    return {
        "status": "complete",
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "best_count_acc": best_count_acc,
        "final_leaf_top1": leaf_acc,
        "final_leaf_top5": leaf_top5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rw decoder")
    parser.add_argument("--run-id", default="RW-001")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--data", type=Path, default=Path("data/rw_actionir_train.jsonl"))
    args = parser.parse_args()

    result = train(
        args.data,
        args.run_id,
        args.device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.max_examples,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
