"""
Evaluate navigational retrieval vs dense retrieval baseline.

Compares the Wayfinder proof network navigation against a dense embedding
cosine-similarity baseline on premise recall@k metrics.

Produces a structured JSON report.

Usage:
    python scripts/eval_retrieval.py --config configs/wayfinder.yaml \
        --checkpoint models/NAV-001_step5000.pt --samples 500
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import load_nav_examples_jsonl
from src.nav_contracts import NavigationalExample
from src.nav_model_factory import load_navigational_checkpoint
from src.resolution import SearchContext, resolve


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path, config: dict, device: str) -> dict:
    """Load trained modules from checkpoint."""
    _, modules = load_navigational_checkpoint(path, config, device)
    return modules


def nav_retrieve(
    example: NavigationalExample,
    modules: dict,
    conn: sqlite3.Connection,
    limit: int,
) -> list[str]:
    """Retrieve premises using navigational pipeline."""
    with torch.no_grad():
        emb = modules["encoder"].encode([example.goal_state])
        feat, _, _ = modules["analyzer"](emb)
        br = modules["bridge"](feat)
        nav_output = modules["navigator"].predict(br)

    context = SearchContext()
    candidates = resolve(nav_output, conn, context, premise_limit=limit)
    seen: set[str] = set()
    premises: list[str] = []
    for c in candidates:
        for p in c.premises:
            if p not in seen:
                seen.add(p)
                premises.append(p)
            if len(premises) >= limit:
                return premises
    return premises


def dense_retrieve(
    example: NavigationalExample,
    modules: dict,
    all_premise_embeddings: torch.Tensor,
    all_premise_names: list[str],
    limit: int,
) -> list[str]:
    """Retrieve premises using dense cosine similarity baseline."""
    with torch.no_grad():
        emb = modules["encoder"].encode([example.goal_state])
        feat, _, _ = modules["analyzer"](emb)
        query_vec = feat[0]

    sims = torch.nn.functional.cosine_similarity(
        query_vec.unsqueeze(0), all_premise_embeddings, dim=-1
    )
    top_indices = sims.argsort(descending=True)[:limit]
    return [all_premise_names[i] for i in top_indices.cpu().numpy()]


def compute_recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Compute recall@k."""
    if not ground_truth:
        return 1.0
    gt_set = set(ground_truth)
    hits = sum(1 for p in retrieved[:k] if p in gt_set)
    return hits / len(gt_set)


def compute_universe_coverage(
    ground_truth: list[str], entity_names: set[str]
) -> tuple[float, list[str], list[str]]:
    """Compute what fraction of ground-truth premises exist in the DB.

    Returns:
        (coverage_fraction, covered_premises, uncovered_premises)
    """
    if not ground_truth:
        return 1.0, [], []
    covered = [p for p in ground_truth if p in entity_names]
    uncovered = [p for p in ground_truth if p not in entity_names]
    return len(covered) / len(ground_truth), covered, uncovered


def compute_conditional_recall_at_k(
    retrieved: list[str], ground_truth: list[str], entity_names: set[str], k: int
) -> float:
    """Compute recall@k conditioned on premises existing in the DB.

    Only counts ground-truth premises that are actually in the entity universe.
    This separates retrieval quality from extraction coverage.
    """
    reachable_gt = [p for p in ground_truth if p in entity_names]
    if not reachable_gt:
        return 1.0  # all GT premises are unreachable — vacuously correct
    gt_set = set(reachable_gt)
    hits = sum(1 for p in retrieved[:k] if p in gt_set)
    return hits / len(gt_set)


def _encode_all_premises(conn: sqlite3.Connection, modules: dict) -> tuple[torch.Tensor, list[str]]:
    """Pre-compute premise embeddings for dense baseline."""
    print("  Computing premise embeddings for dense baseline...")
    rows = conn.execute("SELECT name FROM entities WHERE entity_type = 'lemma'").fetchall()
    names = [r[0] for r in rows]

    if not names:
        return torch.zeros(0, 256), names

    batch_size = 64
    batched = []
    for i in range(0, len(names), batch_size):
        with torch.no_grad():
            emb = modules["encoder"].encode(names[i : i + batch_size])
            feat, _, _ = modules["analyzer"](emb)
            batched.append(feat)
    return torch.cat(batched, dim=0), names


def _compare_retrieval(
    examples: list,
    modules: dict,
    conn: sqlite3.Connection,
    premise_emb: torch.Tensor,
    premise_names: list[str],
    ks: list[int],
) -> dict:
    """Run nav vs dense retrieval on all examples, collecting recall and timing.

    Returns a dict with nav_recalls, dense_recalls, nav_cond_recalls,
    dense_cond_recalls, coverages, nav_times, dense_times.
    """
    entity_name_set = set(premise_names)
    nav_recalls: dict[int, list[float]] = {k: [] for k in ks}
    dense_recalls: dict[int, list[float]] = {k: [] for k in ks}
    nav_cond: dict[int, list[float]] = {k: [] for k in ks}
    dense_cond: dict[int, list[float]] = {k: [] for k in ks}
    coverages: list[float] = []
    nav_times: list[float] = []
    dense_times: list[float] = []

    print("  Running retrieval comparison...")
    for i, ex in enumerate(examples):
        gt = ex.ground_truth_premises
        if not gt:
            continue

        cov, _, _ = compute_universe_coverage(gt, entity_name_set)
        coverages.append(cov)

        t0 = time.perf_counter()
        nav_prem = nav_retrieve(ex, modules, conn, limit=16)
        nav_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        if len(premise_names) > 0:
            dense_prem = dense_retrieve(ex, modules, premise_emb, premise_names, limit=16)
        else:
            dense_prem = []
        dense_times.append(time.perf_counter() - t0)

        for k in ks:
            nav_recalls[k].append(compute_recall_at_k(nav_prem, gt, k))
            dense_recalls[k].append(compute_recall_at_k(dense_prem, gt, k))
            nav_cond[k].append(
                compute_conditional_recall_at_k(nav_prem, gt, entity_name_set, k)
            )
            dense_cond[k].append(
                compute_conditional_recall_at_k(dense_prem, gt, entity_name_set, k)
            )

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(examples)} done")

    return {
        "nav_recalls": nav_recalls,
        "dense_recalls": dense_recalls,
        "nav_cond_recalls": nav_cond,
        "dense_cond_recalls": dense_cond,
        "coverages": coverages,
        "nav_times": nav_times,
        "dense_times": dense_times,
    }


def _build_report(comparison: dict, ks: list[int]) -> dict:
    """Build and print the comparison report with coverage-aware metrics."""
    nav_recalls = comparison["nav_recalls"]
    dense_recalls = comparison["dense_recalls"]
    nav_cond = comparison["nav_cond_recalls"]
    dense_cond = comparison["dense_cond_recalls"]
    coverages = comparison["coverages"]
    nav_times = comparison["nav_times"]
    dense_times = comparison["dense_times"]

    nav_ret = {f"recall@{k}": round(float(np.mean(nav_recalls[k])), 4) for k in ks}
    dense_ret = {f"recall@{k}": round(float(np.mean(dense_recalls[k])), 4) for k in ks}
    nav_cond_ret = {f"cond_recall@{k}": round(float(np.mean(nav_cond[k])), 4) for k in ks}
    dense_cond_ret = {f"cond_recall@{k}": round(float(np.mean(dense_cond[k])), 4) for k in ks}
    timing = {
        "nav_avg_ms": round(float(np.mean(nav_times)) * 1000, 2),
        "dense_avg_ms": round(float(np.mean(dense_times)) * 1000, 2),
        "speedup": round(float(np.mean(dense_times)) / max(float(np.mean(nav_times)), 1e-9), 2),
    }
    coverage_stats = {
        "mean": round(float(np.mean(coverages)), 4),
        "min": round(float(np.min(coverages)), 4),
        "max": round(float(np.max(coverages)), 4),
        "fully_covered": sum(1 for c in coverages if c >= 1.0),
        "zero_covered": sum(1 for c in coverages if c <= 0.0),
    }
    report: dict = {
        "samples": len(nav_recalls[ks[0]]),
        "universe_coverage": coverage_stats,
        "nav_retrieval": nav_ret,
        "dense_retrieval": dense_ret,
        "nav_conditional_retrieval": nav_cond_ret,
        "dense_conditional_retrieval": dense_cond_ret,
        "timing": timing,
    }

    print(f"\n=== Universe Coverage ===")
    print(f"  mean={coverage_stats['mean']:.1%}"
          f"  fully_covered={coverage_stats['fully_covered']}/{len(coverages)}"
          f"  zero_covered={coverage_stats['zero_covered']}/{len(coverages)}")

    print("\n=== Retrieval Comparison (raw) ===")
    for k in ks:
        nr = nav_ret[f"recall@{k}"]
        dr = dense_ret[f"recall@{k}"]
        print(f"  recall@{k}: nav={nr:.4f} dense={dr:.4f} delta={nr - dr:+.4f}")

    print("\n=== Conditional Retrieval (only reachable premises) ===")
    for k in ks:
        nr = nav_cond_ret[f"cond_recall@{k}"]
        dr = dense_cond_ret[f"cond_recall@{k}"]
        print(f"  cond_recall@{k}: nav={nr:.4f} dense={dr:.4f} delta={nr - dr:+.4f}")

    print(
        f"\n  Timing: nav={timing['nav_avg_ms']:.1f}ms"
        f" dense={timing['dense_avg_ms']:.1f}ms"
        f" speedup={timing['speedup']:.1f}x"
    )
    return report


def evaluate(
    config: dict,
    checkpoint_path: Path,
    samples: int,
    device: str,
) -> dict:
    """Run the full retrieval comparison evaluation."""
    modules = load_checkpoint(checkpoint_path, config, device)
    for m in modules.values():
        m.eval()

    eval_path = Path(config["data"].get("nav_eval", "data/nav_eval.jsonl"))
    examples = load_nav_examples_jsonl(eval_path)
    if len(examples) > samples:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(examples), samples, replace=False)
        examples = [examples[int(i)] for i in indices]
    print(f"Evaluating on {len(examples)} examples")

    conn = sqlite3.connect(config["data"]["proof_network_db"])
    ks = [1, 4, 8, 16]

    premise_emb, premise_names = _encode_all_premises(conn, modules)
    comparison = _compare_retrieval(examples, modules, conn, premise_emb, premise_names, ks)
    conn.close()

    return _build_report(comparison, ks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    report = evaluate(config, args.checkpoint, args.samples, args.device)

    output = args.output or Path(f"runs/eval_retrieval_{args.samples}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
