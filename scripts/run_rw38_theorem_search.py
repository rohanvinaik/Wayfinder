"""EXP-RW-038: Theorem-search integration with cosine_simp lane.

Three conditions on the same 50-theorem Mathlib split:
  baseline       — no cosine lanes (NAV-004, structural_core + automation only)
  +cosine_rw     — baseline + cosine_rw (additive policy, top-1 beam)
  +cosine_rw+simp — baseline + cosine_rw + cosine_simp (bare simp then simp [top1])

Metrics:
  proved theorems per condition
  no-regression delta (theorems lost vs baseline)
  unique theorems touched by simp lane
  subgoal closes per lane
  Lean calls / theorem (cost)

Usage:
    python -m scripts.run_rw38_theorem_search \\
        --checkpoint models/NAV-004_step5000.pt \\
        --theorems data/mathlib_benchmark_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

from scripts.run_benchmark import (
    _build_search_components,
    _resolve_initial_goal,
    load_benchmark_theorems,
)
from src.proof_search import SearchConfig, SearchResult, search


# ---------------------------------------------------------------------------
# Per-theorem result
# ---------------------------------------------------------------------------


@dataclass
class TheoremResult:
    theorem_id: str = ""
    source: str = ""

    baseline_success: bool = False
    baseline_attempts: int = 0
    baseline_time_s: float = 0.0
    baseline_close_lane: str = ""

    rw_success: bool = False
    rw_attempts: int = 0
    rw_time_s: float = 0.0
    rw_close_lane: str = ""
    rw_simp_subgoal_closes: int = 0

    rw_simp_success: bool = False
    rw_simp_attempts: int = 0
    rw_simp_time_s: float = 0.0
    rw_simp_close_lane: str = ""
    simp_subgoal_closes: int = 0


def _dominant_lane(result: SearchResult) -> str:
    prov = result.close_provenance
    if not result.success:
        return "failed"
    for label in ("learned", "cosine_simp", "cosine_rw_seq", "cosine_rw",
                  "solver_bootstrap", "structural_core", "automation"):
        if any(p == label or p.startswith(label) for p in prov):
            return label
    return prov[0] if prov else "unknown"


def _count_simp_closes(result: SearchResult) -> int:
    """Count goal closes attributed to cosine_simp lane."""
    return sum(1 for p in result.close_provenance if p == "cosine_simp" or p.startswith("cosine_simp"))


# ---------------------------------------------------------------------------
# Single-condition run
# ---------------------------------------------------------------------------


def _run_condition(
    condition: str,
    theorems: list[dict],
    cfg: SearchConfig,
    pipeline,
    lean,
    conn: sqlite3.Connection,
    encoder,
) -> list[TheoremResult]:
    name_to_id: dict = {}
    if cfg.accessible_premises:
        rows = conn.execute("SELECT id, name FROM entities").fetchall()
        name_to_id = {name: eid for eid, name in rows}

    results: list[TheoremResult] = []
    for thm in theorems:
        gt = thm.get("ground_truth_tactic", "")
        if gt and thm.get("goal_state"):
            lean.register_ground_truth(thm["goal_state"], [gt])

        t0 = time.perf_counter()
        initial_goal = _resolve_initial_goal(thm, lean)
        if initial_goal is None:
            results.append(TheoremResult(theorem_id=thm["theorem_id"], source=thm["source"]))
            continue

        accessible_id = name_to_id.get(thm["theorem_id"]) if name_to_id else None
        result: SearchResult = search(
            theorem_id=thm["theorem_id"],
            initial_goal=initial_goal,
            pipeline=pipeline,
            conn=conn,
            lean=lean,
            config=cfg,
            accessible_theorem_id=accessible_id,
            sentence_encoder=encoder,
        )
        elapsed = time.perf_counter() - t0
        close_lane = _dominant_lane(result)
        simp_closes = _count_simp_closes(result)

        res = TheoremResult(theorem_id=thm["theorem_id"], source=thm["source"])
        if condition == "baseline":
            res.baseline_success = result.success
            res.baseline_attempts = result.attempts
            res.baseline_time_s = elapsed
            res.baseline_close_lane = close_lane
        elif condition == "rw":
            res.rw_success = result.success
            res.rw_attempts = result.attempts
            res.rw_time_s = elapsed
            res.rw_close_lane = close_lane
            res.rw_simp_subgoal_closes = simp_closes
        else:  # rw_simp
            res.rw_simp_success = result.success
            res.rw_simp_attempts = result.attempts
            res.rw_simp_time_s = elapsed
            res.rw_simp_close_lane = close_lane
            res.simp_subgoal_closes = simp_closes
        results.append(res)

    return results


def _merge(combined: list[TheoremResult], cond_results: list[TheoremResult], condition: str) -> None:
    by_id = {r.theorem_id: r for r in cond_results}
    for res in combined:
        cr = by_id.get(res.theorem_id)
        if cr is None:
            continue
        if condition == "baseline":
            res.baseline_success = cr.baseline_success
            res.baseline_attempts = cr.baseline_attempts
            res.baseline_time_s = cr.baseline_time_s
            res.baseline_close_lane = cr.baseline_close_lane
        elif condition == "rw":
            res.rw_success = cr.rw_success
            res.rw_attempts = cr.rw_attempts
            res.rw_time_s = cr.rw_time_s
            res.rw_close_lane = cr.rw_close_lane
            res.rw_simp_subgoal_closes = cr.rw_simp_subgoal_closes
        else:  # rw_simp
            res.rw_simp_success = cr.rw_simp_success
            res.rw_simp_attempts = cr.rw_simp_attempts
            res.rw_simp_time_s = cr.rw_simp_time_s
            res.rw_simp_close_lane = cr.rw_simp_close_lane
            res.simp_subgoal_closes = cr.simp_subgoal_closes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[TheoremResult]) -> None:
    n = len(results)
    base_ok = sum(r.baseline_success for r in results)
    rw_ok = sum(r.rw_success for r in results)
    rw_simp_ok = sum(r.rw_simp_success for r in results)

    rw_lost = [r for r in results if r.baseline_success and not r.rw_success]
    rw_simp_won = [r for r in results if not r.baseline_success and r.rw_simp_success]
    rw_simp_lost = [r for r in results if r.baseline_success and not r.rw_simp_success]

    # simp lane activity
    simp_touched = sum(1 for r in results if r.simp_subgoal_closes > 0)
    simp_total_closes = sum(r.simp_subgoal_closes for r in results)

    base_calls = sum(r.baseline_attempts for r in results)
    rw_calls = sum(r.rw_attempts for r in results)
    rw_simp_calls = sum(r.rw_simp_attempts for r in results)
    base_time = sum(r.baseline_time_s for r in results)
    rw_time = sum(r.rw_time_s for r in results)
    rw_simp_time = sum(r.rw_simp_time_s for r in results)

    print("\n" + "=" * 70)
    print("EXP-RW-038: Theorem-Search Integration (cosine_rw + cosine_simp)")
    print("=" * 70)
    print(f"\nTheorems: {n}")
    print(f"\n{'Condition':<40} {'Proved':>8} {'Rate':>7} {'Delta':>8}")
    print("-" * 65)
    print(f"{'baseline':<40} {base_ok:>8} {100*base_ok/max(n,1):>6.1f}%        —")
    print(f"{'+ cosine_rw':<40} {rw_ok:>8} {100*rw_ok/max(n,1):>6.1f}%  {rw_ok - base_ok:>+6}")
    print(f"{'+ cosine_rw + cosine_simp':<40} {rw_simp_ok:>8} {100*rw_simp_ok/max(n,1):>6.1f}%  {rw_simp_ok - base_ok:>+6}")

    print(f"\nNo-regression check:")
    print(f"  baseline proved, +cosine_rw lost:      {len(rw_lost)}")
    print(f"  baseline proved, +cosine_rw+simp lost: {len(rw_simp_lost)}")

    print(f"\ncosine_simp lane activity (+cosine_rw+simp condition):")
    print(f"  theorems touched (≥1 simp subgoal close): {simp_touched}")
    print(f"  total simp subgoal closes:                {simp_total_closes}")
    print(f"  theorems won by +simp (vs +cosine_rw):    {len([r for r in results if not r.rw_success and r.rw_simp_success])}")

    print(f"\nLean calls / theorem:")
    print(f"  baseline:              {base_calls/max(n,1):.1f}  (total {base_calls})")
    print(f"  +cosine_rw:            {rw_calls/max(n,1):.1f}  (+{(rw_calls - base_calls)/max(n,1):.1f}/thm)")
    print(f"  +cosine_rw+cosine_simp:{rw_simp_calls/max(n,1):.1f}  (+{(rw_simp_calls - base_calls)/max(n,1):.1f}/thm)")

    print(f"\nTime / theorem:")
    print(f"  baseline:              {base_time/max(n,1):.1f}s  (total {base_time:.0f}s)")
    print(f"  +cosine_rw:            {rw_time/max(n,1):.1f}s")
    print(f"  +cosine_rw+cosine_simp:{rw_simp_time/max(n,1):.1f}s")

    if rw_simp_won:
        print(f"\nTheorems won by +cosine_rw+cosine_simp (vs baseline):")
        for r in rw_simp_won:
            print(f"  {r.theorem_id}  (lane={r.rw_simp_close_lane}, simp_closes={r.simp_subgoal_closes})")
    if rw_simp_lost:
        print(f"\nRegressions in +cosine_rw+cosine_simp vs baseline:")
        for r in rw_simp_lost:
            print(f"  {r.theorem_id}  (baseline_lane={r.baseline_close_lane})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-RW-038: theorem-search simp lane integration")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--output", default="runs/rw38_results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("evaluation", {})["benchmark_theorems"] = args.theorems
    config["evaluation"].pop("mathlib_test_split", None)
    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]

    pipeline, base_cfg, lean, _, conn = _build_search_components(
        config, Path(args.checkpoint), args.device
    )

    limit = args.limit if args.limit > 0 else None
    theorems = load_benchmark_theorems(config, limit)
    logger.info("Loaded %d theorems", len(theorems))

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded MiniLM encoder")
    except ImportError:
        logger.warning("sentence-transformers not available")

    if lean._backend == "pantograph":
        lean._ensure_server()

    # Condition 1: baseline — no cosine lanes
    cfg_baseline = deepcopy(base_cfg)
    cfg_baseline.search_mode = "no_learned"

    # Condition 2: +cosine_rw — additive policy, top-1 beam (proven +1 theorem in EXP-RW-037)
    cfg_rw = deepcopy(base_cfg)
    cfg_rw.search_mode = "no_learned"
    cfg_rw.cosine_rw_beam = 1

    # Condition 3: +cosine_rw +cosine_simp — bare simp then simp [top1]
    cfg_rw_simp = deepcopy(base_cfg)
    cfg_rw_simp.search_mode = "no_learned"
    cfg_rw_simp.cosine_rw_beam = 1
    cfg_rw_simp.cosine_simp_enabled = True

    combined = [TheoremResult(theorem_id=t["theorem_id"], source=t["source"]) for t in theorems]

    for condition, cfg, enc, label in [
        ("baseline", cfg_baseline, None,    "baseline (no cosine)"),
        ("rw",       cfg_rw,       encoder, "+cosine_rw"),
        ("rw_simp",  cfg_rw_simp,  encoder, "+cosine_rw +cosine_simp"),
    ]:
        logger.info("Condition: %s (%d theorems)", label, len(theorems))
        t0 = time.time()
        cond_results = _run_condition(condition, theorems, cfg, pipeline, lean, conn, enc)
        elapsed = time.time() - t0
        proved = sum(
            r.baseline_success if condition == "baseline" else
            r.rw_success if condition == "rw" else
            r.rw_simp_success
            for r in cond_results
        )
        logger.info("  %d/%d proved in %.0fs", proved, len(theorems), elapsed)
        _merge(combined, cond_results, condition)

    lean.close()
    conn.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "EXP-RW-038",
            "n_theorems": len(combined),
            "results": [asdict(r) for r in combined],
        }, f, indent=2)
    logger.info("Written to %s", output_path)

    print_report(combined)


if __name__ == "__main__":
    main()
