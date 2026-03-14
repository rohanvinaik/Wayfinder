"""Similarity metrics for encoder evaluation.

Extracted from encoder_backends.py for cohesion: these functions compute
intra-theorem vs inter-theorem cosine similarity to measure embedding quality.
"""

from __future__ import annotations

import numpy as np


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return embeddings / norms


def _compute_intra_sims(
    normed: np.ndarray,
    theorem_groups: dict[str, list[int]],
) -> list[float]:
    intra_sims: list[float] = []
    for indices in theorem_groups.values():
        if len(indices) < 2:
            continue
        group_embs = normed[indices]
        sim_matrix = group_embs @ group_embs.T
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                intra_sims.append(sim_matrix[i, j])
    return intra_sims


def _compute_inter_sims(
    normed: np.ndarray,
    theorem_groups: dict[str, list[int]],
    num_pairs: int,
) -> list[float]:
    rng = np.random.default_rng(42)
    all_tids = list(theorem_groups.keys())
    inter_sims: list[float] = []
    for _ in range(num_pairs):
        t1, t2 = rng.choice(len(all_tids), size=2, replace=False)
        i1 = rng.choice(theorem_groups[all_tids[t1]])
        i2 = rng.choice(theorem_groups[all_tids[t2]])
        inter_sims.append(float(normed[i1] @ normed[i2]))
    return inter_sims


def compute_similarity_metrics(
    embeddings: np.ndarray,
    theorem_groups: dict[str, list[int]],
) -> dict:
    """Compute intra-theorem vs inter-theorem cosine similarity."""
    normed = _normalize_embeddings(embeddings)
    intra_sims = _compute_intra_sims(normed, theorem_groups)
    inter_sims = _compute_inter_sims(normed, theorem_groups, min(len(intra_sims), 2000))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    separation = intra_mean - inter_mean

    return {
        "intra_theorem_sim": round(intra_mean, 4),
        "inter_theorem_sim": round(inter_mean, 4),
        "separation": round(separation, 4),
        "intra_pairs": len(intra_sims),
        "inter_pairs": len(inter_sims),
    }
