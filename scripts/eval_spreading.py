"""
Evaluate spreading activation benefit in the proof network.

Measures whether spreading activation from seed entities improves
premise retrieval recall compared to navigation without spreading.

Produces a structured JSON report.

Usage:
    python scripts/eval_spreading.py --config configs/wayfinder.yaml \
        --checkpoint models/NAV-001_step5000.pt --samples 200
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


def load_modules(checkpoint_path: Path, config: dict, device: str) -> dict:
    """Load trained modules from checkpoint."""
    _, modules = load_navigational_checkpoint(checkpoint_path, config, device)
    return modules


def get_seed_entities(conn: sqlite3.Connection, premises_used: list[str]) -> list[int]:
    """Look up entity IDs for previously used premises (seed entities)."""
    if not premises_used:
        return []
    placeholders = ",".join("?" * len(premises_used))
    rows = conn.execute(
        f"SELECT id FROM entities WHERE name IN ({placeholders})",  # nosec B608
        premises_used,
    ).fetchall()
    return [r[0] for r in rows]


def retrieve_premises(
    example: NavigationalExample,
    modules: dict,
    conn: sqlite3.Connection,
    seed_ids: list[int],
    limit: int,
) -> list[str]:
    """Retrieve premises using the full pipeline with optional spreading."""
    with torch.no_grad():
        emb = modules["encoder"].encode([example.goal_state])
        feat, _, _ = modules["analyzer"](emb)
        br = modules["bridge"](feat)
        nav_output = modules["navigator"].predict(br)

    context = SearchContext(seed_entity_ids=seed_ids)
    candidates = resolve(nav_output, conn, context, premise_limit=limit)

    premises: list[str] = []
    for c in candidates:
        for p in c.premises:
            if p not in premises:
                premises.append(p)
            if len(premises) >= limit:
                break
        if len(premises) >= limit:
            break
    return premises[:limit]


def compute_recall(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    if not ground_truth:
        return 1.0
    gt_set = set(ground_truth)
    hits = sum(1 for p in retrieved[:k] if p in gt_set)
    return hits / len(gt_set)


def _compare_spreading(
    examples: list,
    modules: dict,
    conn: sqlite3.Connection,
    ks: list[int],
) -> tuple[dict[int, list[float]], dict[int, list[float]], list[float], list[float]]:
    """Run spreading vs no-spreading retrieval on all examples."""
    no_spread_recalls: dict[int, list[float]] = {k: [] for k in ks}
    with_spread_recalls: dict[int, list[float]] = {k: [] for k in ks}
    no_spread_times: list[float] = []
    with_spread_times: list[float] = []

    for i, ex in enumerate(examples):
        gt = ex.ground_truth_premises
        if not gt:
            continue

        t0 = time.perf_counter()
        no_spread = retrieve_premises(ex, modules, conn, seed_ids=[], limit=16)
        no_spread_times.append(time.perf_counter() - t0)

        seeds = get_seed_entities(conn, ex.proof_history[:5])
        t0 = time.perf_counter()
        with_spread = retrieve_premises(ex, modules, conn, seed_ids=seeds, limit=16)
        with_spread_times.append(time.perf_counter() - t0)

        for k in ks:
            no_spread_recalls[k].append(compute_recall(no_spread, gt, k))
            with_spread_recalls[k].append(compute_recall(with_spread, gt, k))

        if (i + 1) % 50 == 0:
            r16_ns = float(np.mean(no_spread_recalls[16]))
            r16_ws = float(np.mean(with_spread_recalls[16]))
            print(
                f"    {i + 1}/{len(examples)}: "
                f"no_spread recall@16={r16_ns:.3f}, "
                f"with_spread recall@16={r16_ws:.3f}"
            )

    return no_spread_recalls, with_spread_recalls, no_spread_times, with_spread_times


def _build_spreading_report(
    no_spread_recalls: dict[int, list[float]],
    with_spread_recalls: dict[int, list[float]],
    no_spread_times: list[float],
    with_spread_times: list[float],
    ks: list[int],
) -> dict:
    """Build and print the spreading evaluation report."""
    no_spread = {f"recall@{k}": round(float(np.mean(no_spread_recalls[k])), 4) for k in ks}
    with_spread = {f"recall@{k}": round(float(np.mean(with_spread_recalls[k])), 4) for k in ks}
    delta = {
        f"recall@{k}": round(
            float(np.mean(with_spread_recalls[k])) - float(np.mean(no_spread_recalls[k])),
            4,
        )
        for k in ks
    }
    timing = {
        "no_spread_avg_ms": round(float(np.mean(no_spread_times)) * 1000, 2),
        "with_spread_avg_ms": round(float(np.mean(with_spread_times)) * 1000, 2),
    }
    report: dict = {
        "samples": len(no_spread_recalls[ks[0]]),
        "multi_step_only": True,
        "no_spreading": no_spread,
        "with_spreading": with_spread,
        "delta": delta,
        "timing": timing,
    }

    print("\n=== Spreading Activation Evaluation ===")
    for k in ks:
        ns = no_spread[f"recall@{k}"]
        ws = with_spread[f"recall@{k}"]
        d = delta[f"recall@{k}"]
        print(f"  recall@{k}: no_spread={ns:.4f} with_spread={ws:.4f} delta={d:+.4f}")
    print(
        f"  Timing: no_spread={timing['no_spread_avg_ms']:.1f}ms"
        f" with_spread={timing['with_spread_avg_ms']:.1f}ms"
    )
    return report


def evaluate(
    config: dict,
    checkpoint_path: Path,
    samples: int,
    device: str,
) -> dict:
    """Run spreading activation evaluation."""
    modules = load_modules(checkpoint_path, config, device)

    eval_path = Path(config["data"].get("nav_eval", "data/nav_eval.jsonl"))
    all_examples = load_nav_examples_jsonl(eval_path)

    multi_step = [e for e in all_examples if e.step_index > 0 and e.proof_history]
    if len(multi_step) > samples:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(multi_step), samples, replace=False)
        multi_step = [multi_step[int(i)] for i in indices]
    print(f"Evaluating spreading on {len(multi_step)} multi-step examples")

    conn = sqlite3.connect(config["data"]["proof_network_db"])
    ks = [4, 8, 16]

    ns_recalls, ws_recalls, ns_times, ws_times = _compare_spreading(multi_step, modules, conn, ks)
    conn.close()

    return _build_spreading_report(ns_recalls, ws_recalls, ns_times, ws_times, ks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spreading activation evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    report = evaluate(config, args.checkpoint, args.samples, args.device)

    output = args.output or Path(f"runs/eval_spreading_{args.samples}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
