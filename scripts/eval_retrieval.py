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

from src.bridge import InformationBridge
from src.data import load_nav_examples_jsonl
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.nav_contracts import NavigationalExample
from src.proof_navigator import ProofNavigator
from src.resolution import SearchContext, resolve


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path, config: dict, device: str) -> dict:
    """Load trained modules from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # nosec B614 — trusted local checkpoints
    enc_cfg = config["model"]["encoder"]
    ana_cfg = config["model"]["goal_analyzer"]
    br_cfg = config["model"]["bridge"]
    nav_cfg = config["model"]["navigator"]

    encoder = GoalEncoder(
        model_name=enc_cfg.get("type", "byt5-small"),
        output_dim=enc_cfg.get("output_dim"),
        frozen=enc_cfg.get("frozen", True),
    ).to(device)
    analyzer = GoalAnalyzer(
        input_dim=enc_cfg["output_dim"],
        feature_dim=ana_cfg["feature_dim"],
        num_anchors=ana_cfg.get("num_anchors", 300),
        navigable_banks=ana_cfg.get("navigable_banks"),
    ).to(device)
    bridge = InformationBridge(
        input_dim=ana_cfg["feature_dim"],
        bridge_dim=br_cfg["bridge_dim"],
        history_dim=br_cfg.get("history_dim", 0),
    ).to(device)
    navigator = ProofNavigator(
        input_dim=br_cfg["bridge_dim"],
        hidden_dim=nav_cfg["hidden_dim"],
        num_anchors=nav_cfg["num_anchors"],
        num_layers=nav_cfg["num_layers"],
        ternary_enabled=nav_cfg.get("ternary_enabled", True),
        navigable_banks=nav_cfg.get("navigable_banks"),
    ).to(device)

    for name, module in [
        ("encoder", encoder),
        ("analyzer", analyzer),
        ("bridge", bridge),
        ("navigator", navigator),
    ]:
        if name in ckpt.get("modules", {}):
            module.load_state_dict(ckpt["modules"][name])

    return {"encoder": encoder, "analyzer": analyzer, "bridge": bridge, "navigator": navigator}


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


def _encode_all_premises(conn: sqlite3.Connection, modules: dict) -> tuple[torch.Tensor, list[str]]:
    """Pre-compute premise embeddings for dense baseline."""
    print("  Computing premise embeddings for dense baseline...")
    rows = conn.execute("SELECT name FROM entities WHERE type = 'lemma'").fetchall()
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
) -> tuple[dict[int, list[float]], dict[int, list[float]], list[float], list[float]]:
    """Run nav vs dense retrieval on all examples, collecting recall and timing."""
    nav_recalls: dict[int, list[float]] = {k: [] for k in ks}
    dense_recalls: dict[int, list[float]] = {k: [] for k in ks}
    nav_times: list[float] = []
    dense_times: list[float] = []

    print("  Running retrieval comparison...")
    for i, ex in enumerate(examples):
        gt = ex.ground_truth_premises
        if not gt:
            continue

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

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(examples)} done")

    return nav_recalls, dense_recalls, nav_times, dense_times


def _build_report(
    nav_recalls: dict[int, list[float]],
    dense_recalls: dict[int, list[float]],
    nav_times: list[float],
    dense_times: list[float],
    ks: list[int],
) -> dict:
    """Build and print the comparison report."""
    report = {
        "samples": len(nav_recalls[ks[0]]),
        "nav_retrieval": {f"recall@{k}": round(float(np.mean(nav_recalls[k])), 4) for k in ks},
        "dense_retrieval": {f"recall@{k}": round(float(np.mean(dense_recalls[k])), 4) for k in ks},
        "timing": {
            "nav_avg_ms": round(float(np.mean(nav_times)) * 1000, 2),
            "dense_avg_ms": round(float(np.mean(dense_times)) * 1000, 2),
            "speedup": round(float(np.mean(dense_times)) / max(float(np.mean(nav_times)), 1e-9), 2),
        },
    }

    print("\n=== Retrieval Comparison ===")
    for k in ks:
        nr = report["nav_retrieval"][f"recall@{k}"]
        dr = report["dense_retrieval"][f"recall@{k}"]
        print(f"  recall@{k}: nav={nr:.4f} dense={dr:.4f} delta={nr - dr:+.4f}")
    print(
        f"  Timing: nav={report['timing']['nav_avg_ms']:.1f}ms"
        f" dense={report['timing']['dense_avg_ms']:.1f}ms"
        f" speedup={report['timing']['speedup']:.1f}x"
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
        rng = np.random.default_rng()
        indices = rng.choice(len(examples), samples, replace=False)
        examples = [examples[int(i)] for i in indices]
    print(f"Evaluating on {len(examples)} examples")

    conn = sqlite3.connect(config["data"]["proof_network_db"])
    ks = [1, 4, 8, 16]

    premise_emb, premise_names = _encode_all_premises(conn, modules)
    nav_recalls, dense_recalls, nav_times, dense_times = _compare_retrieval(
        examples, modules, conn, premise_emb, premise_names, ks
    )
    conn.close()

    return _build_report(nav_recalls, dense_recalls, nav_times, dense_times, ks)


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
