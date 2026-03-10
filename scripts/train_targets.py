"""Target tensor builders and evaluation helpers for navigator training.

Pure functions that convert NavigationalExample batches into training target tensors
and compute evaluation metrics. Extracted from train_navigator.py.
"""

from __future__ import annotations

import numpy as np
import torch


def build_direction_targets(
    examples: list, banks: list[str], device: str
) -> dict[str, torch.Tensor]:
    """Build per-bank direction target tensors from a batch of examples."""
    direction_map = {-1: 0, 0: 1, 1: 2}
    targets: dict[str, torch.Tensor] = {}
    for bank in banks:
        vals = [direction_map[ex.nav_directions.get(bank, 0)] for ex in examples]
        targets[bank] = torch.tensor(vals, dtype=torch.long, device=device)
    return targets


def build_anchor_targets(examples: list, anchor_labels: list[str], device: str) -> torch.Tensor:
    """Build multi-label anchor target tensor."""
    label_to_idx = {label: i for i, label in enumerate(anchor_labels)}
    n_anchors = len(anchor_labels)
    targets = torch.zeros(len(examples), n_anchors, device=device)
    for i, ex in enumerate(examples):
        for label in ex.anchor_labels:
            if label in label_to_idx:
                targets[i, label_to_idx[label]] = 1.0
    return targets


def build_progress_targets(examples: list, device: str) -> torch.Tensor:
    """Build normalized progress targets (remaining / total)."""
    vals = [ex.remaining_steps / max(ex.total_steps, 1) for ex in examples]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def build_critic_targets(examples: list, device: str) -> torch.Tensor:
    """Build soft critic targets based on proof completion proximity."""
    vals = []
    for ex in examples:
        if not ex.solvable:
            vals.append(0.0)
        else:
            progress = 1.0 - (ex.remaining_steps / max(ex.total_steps, 1))
            vals.append(0.3 + 0.7 * progress)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def compute_nav_accuracy(
    modules: dict,
    dataset,
    banks: list[str],
    _device: str,
    max_samples: int = 200,
) -> dict[str, float]:
    """Compute per-bank direction accuracy on a subset."""
    rng = np.random.default_rng()
    n = min(len(dataset), max_samples)
    indices = rng.choice(len(dataset), n, replace=False)
    examples = [dataset[int(i)] for i in indices]

    goal_states = [ex.goal_state for ex in examples]
    with torch.no_grad():
        embeddings = modules["encoder"].encode(goal_states)
        features, _, _ = modules["analyzer"](embeddings)
        bridge_out = modules["bridge"](features)
        dir_logits, _, _, _ = modules["navigator"](bridge_out)

    direction_map = {-1: 0, 0: 1, 1: 2}
    accuracies: dict[str, float] = {}
    for bank in banks:
        preds = dir_logits[bank].argmax(dim=-1).cpu().numpy()
        targets = np.array([direction_map[ex.nav_directions.get(bank, 0)] for ex in examples])
        accuracies[bank] = float(np.mean(preds == targets))

    accuracies["mean"] = float(np.mean(list(accuracies.values())))
    return accuracies
