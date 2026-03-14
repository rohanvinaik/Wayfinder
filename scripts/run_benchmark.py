"""
Benchmark runner — MiniF2F + Mathlib proof search evaluation.

Runs the full proof search pipeline on benchmark theorem sets and
produces structured metrics: theorems proved, avg budget consumed,
neural forward passes, wall-clock time.

Usage:
    python scripts/run_benchmark.py --config configs/wayfinder.yaml \
        --checkpoint models/NAV-001_step5000.pt
    python scripts/run_benchmark.py --config configs/wayfinder.yaml \
        --checkpoint models/NAV-001_step5000.pt --limit 50
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import yaml

from scripts.benchmark_lane_b import run_lane_b
from src.lean_interface import LeanConfig, LeanKernel
from src.nav_model_factory import load_navigational_checkpoint
from src.proof_search import Pipeline, SearchConfig, search

VALID_MODES = ("v1", "v2", "v3")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_modules(checkpoint_path: Path, config: dict, device: str) -> dict:
    """Load trained modules from checkpoint."""
    _, modules = load_navigational_checkpoint(checkpoint_path, config, device)
    return modules


def _load_theorems_from_file(path: Path, source_key: str) -> list[dict]:
    """Load theorem entries from a single JSONL file."""
    theorems: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            theorems.append(
                {
                    "theorem_id": d.get("theorem_id", d.get("name", "")),
                    "goal_state": d.get("goal_state", d.get("statement", "")),
                    "ground_truth_tactic": d.get("ground_truth_tactic", ""),
                    "source": source_key,
                }
            )
    return theorems


def load_benchmark_theorems(config: dict, limit: int | None) -> list[dict]:
    """Load benchmark theorems from configured paths."""
    theorems: list[dict] = []

    for key in ["benchmark_theorems", "mathlib_test_split"]:
        path_str = config.get("evaluation", {}).get(key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        theorems.extend(_load_theorems_from_file(path, key))

    if limit and len(theorems) > limit:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(theorems), limit, replace=False)
        theorems = [theorems[int(i)] for i in indices]

    return theorems


def _build_search_components(config: dict, checkpoint_path: Path, device: str) -> tuple:
    """Build pipeline, search config, lean kernel, and db connection."""
    modules = load_modules(checkpoint_path, config, device)
    pipeline = Pipeline(
        encoder=modules["encoder"],
        analyzer=modules["analyzer"],
        bridge=modules["bridge"],
        navigator=modules["navigator"],
    )

    search_cfg = config.get("search", {})
    cfg = SearchConfig(
        budget=search_cfg.get("budget", 600),
        hammer_delegation=search_cfg.get("hammer_delegation", True),
        accessible_premises=search_cfg.get("accessible_premises", True),
        max_candidates_per_step=search_cfg.get("max_candidates_per_step", 8),
        device=device,
    )

    lean_backend = config.get("lean", {}).get("backend", "stub")
    if lean_backend == "pantograph":
        raise RuntimeError(
            "Pantograph backend is not yet implemented (Phase 2+). "
            "Use --backend stub for offline testing or --backend replay for ground-truth matching. "
            "See docs/WAYFINDER_PLAN.md §3.1."
        )
    lean_cfg = LeanConfig(
        backend=lean_backend,
        hammer_timeout=search_cfg.get("hammer_timeout", 60),
    )
    lean = LeanKernel(lean_cfg)
    conn = sqlite3.connect(config["data"]["proof_network_db"])

    return pipeline, cfg, lean, lean_cfg, conn


def _build_theorem_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Build theorem name → DB integer ID map for accessible-premises lookup."""
    cursor = conn.execute("SELECT id, theorem_id FROM entities")
    return {name: eid for eid, name in cursor.fetchall()}


def _run_search_loop(
    theorems: list[dict],
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    cfg: SearchConfig,
) -> tuple[list[dict], int, int]:
    """Run proof search on all theorems. Returns (results, raw_proved, total_attempts)."""
    results: list[dict] = []
    raw_proved = 0
    total_attempts = 0

    # Build name→id map so search can filter to accessible premises
    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}

    for i, thm in enumerate(theorems):
        # Register ground-truth tactic for replay backend
        gt_tactic = thm.get("ground_truth_tactic", "")
        if gt_tactic and thm["goal_state"]:
            lean.register_ground_truth(thm["goal_state"], [gt_tactic])

        t0 = time.perf_counter()
        accessible_id = name_to_id.get(thm["theorem_id"]) if name_to_id else None
        result = search(
            theorem_id=thm["theorem_id"],
            initial_goal=thm["goal_state"],
            pipeline=pipeline,
            conn=conn,
            lean=lean,
            config=cfg,
            accessible_theorem_id=accessible_id,
        )
        elapsed = time.perf_counter() - t0

        results.append(
            {
                "theorem_id": thm["theorem_id"],
                "source": thm["source"],
                "success": result.success,
                "success_category": "raw_success" if result.success else "failed",
                "attempts": result.attempts,
                "goals_closed": result.goals_closed,
                "goals_remaining": result.goals_remaining,
                "tactics_used": result.tactics_used,
                "time_s": round(elapsed, 3),
            }
        )

        if result.success:
            raw_proved += 1
        total_attempts += result.attempts

        if (i + 1) % 50 == 0 or (i + 1) == len(theorems):
            rate = raw_proved / (i + 1)
            print(
                f"  {i + 1}/{len(theorems)}: raw_proved={raw_proved} "
                f"({100 * rate:.1f}%) avg_attempts={total_attempts / (i + 1):.0f}"
            )

    return results, raw_proved, total_attempts


def _build_specialists(config: dict) -> dict:
    """Build specialist navigators from config."""
    from src.specialist_navigator import SpecialistNavigator

    model_cfg = config.get("model", {})
    spec_cfg = config.get("specialists", {})
    specialists = {}
    for name, scfg in spec_cfg.items():
        specialists[name] = SpecialistNavigator(
            name=name,
            banks=scfg.get("banks", []),
            feature_dim=model_cfg.get("goal_analyzer", {}).get("feature_dim", 256),
            bridge_dim=scfg.get("bridge_dim", 128),
            hidden_dim=scfg.get("hidden_dim", 256),
            num_anchors=model_cfg.get("navigator", {}).get("num_anchors", 18729),
            num_layers=scfg.get("num_layers", 2),
        )
    return specialists


def _run_search_loop_v2(
    theorems: list[dict],
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    cfg: SearchConfig,
    config: dict,
) -> tuple[list[dict], int, int]:
    """Run v2 (SoM) proof search. Imports arbiter lazily to avoid circular deps."""
    from src.arbiter import SoMSlots, som_search
    from src.sketch_predictor import SketchPredictor
    from src.specialist_navigator import ExecutionSlot
    from src.template_classifier import TemplateClassifier

    # Build SoM slots from pipeline components + v2-specific modules
    model_cfg = config.get("model", {})
    specialists = _build_specialists(config)
    slots = SoMSlots(
        encoder=pipeline.encoder,
        analyzer=pipeline.analyzer,
        classifier=TemplateClassifier(
            input_dim=model_cfg.get("goal_analyzer", {}).get("feature_dim", 256),
            hidden_dim=model_cfg.get("template_classifier", {}).get("hidden_dim", 128),
            feature_dim=model_cfg.get("template_classifier", {}).get("feature_dim", 64),
        ),
        sketch_predictor=SketchPredictor(
            embedding_dim=model_cfg.get("encoder", {}).get("output_dim", 384),
            template_feature_dim=model_cfg.get("template_classifier", {}).get("feature_dim", 64),
            hidden_dim=model_cfg.get("sketch_predictor", {}).get("hidden_dim", 256),
        ),
        execution=ExecutionSlot(specialists=specialists),
        lean=lean,
    )

    results: list[dict] = []
    raw_proved = 0
    total_attempts = 0
    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}

    for i, thm in enumerate(theorems):
        gt_tactic = thm.get("ground_truth_tactic", "")
        if gt_tactic and thm["goal_state"]:
            lean.register_ground_truth(thm["goal_state"], [gt_tactic])

        t0 = time.perf_counter()
        accessible_id = name_to_id.get(thm["theorem_id"]) if name_to_id else None
        result = som_search(
            theorem_id=thm["theorem_id"],
            initial_goal=thm["goal_state"],
            slots=slots,
            conn=conn,
            config=cfg,
            accessible_theorem_id=accessible_id,
        )
        elapsed = time.perf_counter() - t0

        results.append(
            {
                "theorem_id": thm["theorem_id"],
                "source": thm["source"],
                "success": result.success,
                "success_category": "raw_success" if result.success else "failed",
                "attempts": result.attempts,
                "goals_closed": result.goals_closed,
                "goals_remaining": result.goals_remaining,
                "tactics_used": result.tactics_used,
                "time_s": round(elapsed, 3),
            }
        )

        if result.success:
            raw_proved += 1
        total_attempts += result.attempts

        if (i + 1) % 50 == 0 or (i + 1) == len(theorems):
            rate = raw_proved / (i + 1)
            print(
                f"  {i + 1}/{len(theorems)}: raw_proved={raw_proved} "
                f"({100 * rate:.1f}%) avg_attempts={total_attempts / (i + 1):.0f}"
            )

    return results, raw_proved, total_attempts


def _run_search_loop_v3(
    theorems: list[dict],
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    cfg: SearchConfig,
    config: dict,
) -> tuple[list[dict], int, int]:
    """Run v3 (boundary learning) proof search. Imports v3 runtime lazily."""
    from src.censor import CensorNetwork
    from src.sketch_predictor import SketchPredictor
    from src.specialist_navigator import ExecutionSlot
    from src.template_classifier import TemplateClassifier
    from src.v3_runtime import V3Config, V3Slots, v3_search
    from src.v3_scoring import compute_bank_idf

    model_cfg = config.get("model", {})
    specialists = _build_specialists(config)
    slots = V3Slots(
        encoder=pipeline.encoder,
        analyzer=pipeline.analyzer,
        classifier=TemplateClassifier(
            input_dim=model_cfg.get("goal_analyzer", {}).get("feature_dim", 256),
            hidden_dim=model_cfg.get("template_classifier", {}).get("hidden_dim", 128),
            feature_dim=model_cfg.get("template_classifier", {}).get("feature_dim", 64),
        ),
        sketch_predictor=SketchPredictor(
            embedding_dim=model_cfg.get("encoder", {}).get("output_dim", 384),
            template_feature_dim=model_cfg.get("template_classifier", {}).get("feature_dim", 64),
            hidden_dim=model_cfg.get("sketch_predictor", {}).get("hidden_dim", 256),
        ),
        execution=ExecutionSlot(specialists=specialists),
        lean=lean,
        censor=CensorNetwork(
            goal_dim=model_cfg.get("censor", {}).get("goal_dim", 256),
            tactic_dim=model_cfg.get("censor", {}).get("tactic_dim", 64),
            hidden_dim=model_cfg.get("censor", {}).get("hidden_dim", 128),
            threshold=config.get("censor", {}).get("operating_threshold", 0.5),
        ),
    )

    # Compute bank-IDF weights from proof network
    bank_idf = compute_bank_idf(conn)
    censor_cfg = config.get("censor", {})
    constraint_cfg = config.get("energy_refinement", {}).get("weights", {})
    v3_cfg = V3Config(
        bank_idf=bank_idf,
        censor_threshold=censor_cfg.get("operating_threshold", 0.5),
        safety_net_k=censor_cfg.get("safety_net_k", 3),
        constraint_weights=constraint_cfg
        if constraint_cfg
        else {
            "bank": 1.0,
            "critic": 0.5,
            "censor": 2.0,
            "anchor": 0.3,
        },
        energy_enabled=config.get("energy_refinement", {}).get("enabled", False),
    )

    results: list[dict] = []
    raw_proved = 0
    total_attempts = 0
    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}

    for i, thm in enumerate(theorems):
        gt_tactic = thm.get("ground_truth_tactic", "")
        if gt_tactic and thm["goal_state"]:
            lean.register_ground_truth(thm["goal_state"], [gt_tactic])

        t0 = time.perf_counter()
        accessible_id = name_to_id.get(thm["theorem_id"]) if name_to_id else None
        result = v3_search(
            theorem_id=thm["theorem_id"],
            initial_goal=thm["goal_state"],
            slots=slots,
            conn=conn,
            config=cfg,
            v3_config=v3_cfg,
            accessible_theorem_id=accessible_id,
        )
        elapsed = time.perf_counter() - t0

        results.append(
            {
                "theorem_id": thm["theorem_id"],
                "source": thm["source"],
                "success": result.success,
                "success_category": "raw_success" if result.success else "failed",
                "attempts": result.attempts,
                "goals_closed": result.goals_closed,
                "goals_remaining": result.goals_remaining,
                "tactics_used": result.tactics_used,
                "time_s": round(elapsed, 3),
                "mode": "v3",
            }
        )

        if result.success:
            raw_proved += 1
        total_attempts += result.attempts

        if (i + 1) % 50 == 0 or (i + 1) == len(theorems):
            rate = raw_proved / (i + 1)
            print(
                f"  {i + 1}/{len(theorems)}: raw_proved={raw_proved} "
                f"({100 * rate:.1f}%) avg_attempts={total_attempts / (i + 1):.0f}"
            )

    return results, raw_proved, total_attempts


def run_benchmark(
    config: dict,
    checkpoint_path: Path,
    device: str,
    limit: int | None,
    mode: str = "v1",
) -> dict:
    """Run proof search on benchmark theorems.

    Args:
        config: Loaded YAML config.
        checkpoint_path: Path to trained checkpoint.
        device: Compute device.
        limit: Max theorems to evaluate.
        mode: Runtime mode — "v1", "v2", or "v3".
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}, must be one of {VALID_MODES}")

    pipeline, cfg, lean, lean_cfg, conn = _build_search_components(config, checkpoint_path, device)

    theorems = load_benchmark_theorems(config, limit)
    print(f"Running benchmark on {len(theorems)} theorems (mode={mode})")
    print(f"  Budget: {cfg.budget}, Hammer: {cfg.hammer_delegation}")
    print(f"  Lean backend: {lean_cfg.backend}")

    start = time.time()
    if mode == "v1":
        results, raw_proved, total_attempts = _run_search_loop(theorems, pipeline, conn, lean, cfg)
    elif mode == "v2":
        results, raw_proved, total_attempts = _run_search_loop_v2(
            theorems,
            pipeline,
            conn,
            lean,
            cfg,
            config,
        )
    else:  # v3
        results, raw_proved, total_attempts = _run_search_loop_v3(
            theorems,
            pipeline,
            conn,
            lean,
            cfg,
            config,
        )
    conn.close()
    total_time = time.time() - start

    n = len(results)
    report = {
        "benchmark": {
            "total_theorems": n,
            "raw_success": raw_proved,
            "raw_success_rate": round(raw_proved / max(n, 1), 4),
            "axle_assisted_success": 0,
            "axle_repair_only": 0,
            "failed": n - raw_proved,
        },
        "efficiency": {
            "total_attempts": total_attempts,
            "avg_attempts_per_theorem": round(total_attempts / max(n, 1), 1),
            "avg_attempts_proved": round(
                sum(r["attempts"] for r in results if r["success"]) / max(raw_proved, 1),
                1,
            ),
            "avg_time_per_theorem_s": round(total_time / max(n, 1), 2),
            "total_time_s": round(total_time, 1),
        },
        "by_source": _group_by_source(results),
        "details": results,
        "config": {
            "checkpoint": str(checkpoint_path),
            "budget": cfg.budget,
            "hammer_delegation": cfg.hammer_delegation,
            "lean_backend": lean_cfg.backend,
            "axle_enabled": config.get("axle", {}).get("enabled", False),
            "device": device,
        },
    }

    axle_cfg = config.get("axle", {})
    if axle_cfg.get("enabled", False):
        report = run_lane_b(report, axle_cfg)

    _print_summary(report)
    return report


def _group_by_source(results: list[dict]) -> dict:
    """Group results by source dataset."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["source"], []).append(r)

    summary = {}
    for source, entries in groups.items():
        proved = sum(1 for e in entries if e["success"])
        summary[source] = {
            "total": len(entries),
            "proved": proved,
            "prove_rate": round(proved / max(len(entries), 1), 4),
        }
    return summary


def _print_summary(report: dict) -> None:
    """Print benchmark summary with metric separation."""
    bm = report["benchmark"]
    eff = report["efficiency"]
    n = bm["total_theorems"]
    print("\n=== Benchmark Results ===")
    print(f"  Theorems: {n}")
    print(f"  Raw success (Lane A only): {bm['raw_success']} ({100 * bm['raw_success_rate']:.1f}%)")
    if bm.get("axle_assisted_success"):
        assisted = bm["axle_assisted_success"]
        print(f"  Axle-assisted success:     {assisted} ({100 * assisted / max(n, 1):.1f}%)")
    if bm.get("axle_repair_only"):
        repair = bm["axle_repair_only"]
        print(f"  Axle repair-only:          {repair} ({100 * repair / max(n, 1):.1f}%)")
    print(f"  Failed: {bm['failed']}")
    print(f"  Avg attempts/theorem: {eff['avg_attempts_per_theorem']}")
    print(f"  Avg time/theorem: {eff['avg_time_per_theorem_s']:.2f}s")
    print(f"  Total time: {eff['total_time_s']:.1f}s")

    for source, data in report.get("by_source", {}).items():
        print(f"  {source}: {data['proved']}/{data['total']} ({100 * data['prove_rate']:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--backend", type=str, default=None, help="Lean backend: stub, pantograph")
    parser.add_argument("--mode", type=str, default="v1", choices=VALID_MODES, help="Runtime mode")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = load_config(args.config)
    if args.backend:
        config.setdefault("lean", {})["backend"] = args.backend
    report = run_benchmark(config, args.checkpoint, args.device, args.limit, mode=args.mode)

    output = args.output or Path("runs/benchmark_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    # Also save per-theorem results
    detail_path = output.with_suffix(".jsonl")
    with open(detail_path, "w") as f:
        for entry in report.get("details", []):
            f.write(json.dumps(entry) + "\n")

    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
