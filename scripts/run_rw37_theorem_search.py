"""Theorem-search integration experiment (EXP-RW-037).

Three conditions on the same 50-theorem Mathlib split used in EXP-3.2:

  baseline       — automation + structural_core (no cosine lanes)
  +cosine_rw     — baseline + single-step cosine_rw (top-5, both dirs)
  +cosine_rw_seq — baseline + sequential bare cosine_rw_seq (max_atoms=10)

Primary metrics:
  proved theorems per condition
  unique theorems won by rw lane
  rewrite progress events (lane fired ≥1 step without closing)
  Lean calls / theorem
  time / theorem
  no-regression check

Usage:
    python -m scripts.run_rw37_theorem_search \\
        --config configs/wayfinder.yaml \\
        --checkpoint models/NAV-002_step5000.pt \\
        --theorems data/mathlib_benchmark_50.jsonl \\
        --output runs/rw37_results.json
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
    rw_progress_events: int = 0

    rw_seq_success: bool = False
    rw_seq_attempts: int = 0
    rw_seq_time_s: float = 0.0
    rw_seq_close_lane: str = ""
    rw_seq_progress_events: int = 0


def _dominant_lane(result: SearchResult) -> str:
    prov = result.close_provenance
    if not result.success:
        return "failed"
    for label in ("learned", "cosine_rw_seq", "cosine_rw", "solver_bootstrap", "structural_core", "automation"):
        if any(p == label or p.startswith(label) for p in prov):
            return label
    return "unknown"


def _rw_progress(result: SearchResult) -> int:
    """Count rw lane tactics that fired but did not close all goals.

    tactics_used is a list of tactic strings; provenance is tracked via
    close_provenance (only goals that were actually closed). We use the
    presence of 'cosine_rw' in close_provenance as a proxy: if rw appears
    in provenance AND success=True that's a close; progress events are
    inferred from tactic strings that look like bare rewrites but where the
    theorem ultimately wasn't proved (or was proved by a later lane).
    """
    rw_tactics = sum(
        1 for t in result.tactics_used
        if isinstance(t, str) and t.startswith("rw [") or t.startswith("rw_seq(")
    )
    rw_closes = sum(1 for p in result.close_provenance if p.startswith("cosine_rw"))
    return max(0, rw_tactics - rw_closes)


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
        progress = _rw_progress(result)

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
            res.rw_progress_events = progress
        else:
            res.rw_seq_success = result.success
            res.rw_seq_attempts = result.attempts
            res.rw_seq_time_s = elapsed
            res.rw_seq_close_lane = close_lane
            res.rw_seq_progress_events = progress
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
            res.rw_progress_events = cr.rw_progress_events
        elif condition == "rw_seq":
            res.rw_seq_success = cr.rw_seq_success
            res.rw_seq_attempts = cr.rw_seq_attempts
            res.rw_seq_time_s = cr.rw_seq_time_s
            res.rw_seq_close_lane = cr.rw_seq_close_lane
            res.rw_seq_progress_events = cr.rw_seq_progress_events


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[TheoremResult]) -> None:
    n = len(results)
    base_ok = sum(r.baseline_success for r in results)
    rw_ok = sum(r.rw_success for r in results)
    rw_seq_ok = sum(r.rw_seq_success for r in results)

    rw_won = [r for r in results if not r.baseline_success and r.rw_success]
    rw_lost = [r for r in results if r.baseline_success and not r.rw_success]
    rw_seq_won = [r for r in results if not r.baseline_success and r.rw_seq_success]
    rw_seq_lost = [r for r in results if r.baseline_success and not r.rw_seq_success]

    rw_touched = sum(1 for r in results if r.rw_progress_events > 0 or (r.rw_success and not r.baseline_success))
    rw_seq_touched = sum(1 for r in results if r.rw_seq_progress_events > 0 or (r.rw_seq_success and not r.baseline_success))

    base_calls = sum(r.baseline_attempts for r in results)
    rw_calls = sum(r.rw_attempts for r in results)
    rw_seq_calls = sum(r.rw_seq_attempts for r in results)
    base_time = sum(r.baseline_time_s for r in results)
    rw_time = sum(r.rw_time_s for r in results)
    rw_seq_time = sum(r.rw_seq_time_s for r in results)

    print("\n" + "=" * 70)
    print("EXP-RW-037: Theorem-Search Integration")
    print("=" * 70)
    print(f"\nTheorems: {n}")
    print(f"\n{'Condition':<35} {'Proved':>8} {'Rate':>7} {'Delta':>8}")
    print("-" * 60)
    print(f"{'baseline':<35} {base_ok:>8} {100*base_ok/max(n,1):>6.1f}%        —")
    print(f"{'+ cosine_rw':<35} {rw_ok:>8} {100*rw_ok/max(n,1):>6.1f}%  {rw_ok - base_ok:>+6}")
    print(f"{'+ cosine_rw_seq':<35} {rw_seq_ok:>8} {100*rw_seq_ok/max(n,1):>6.1f}%  {rw_seq_ok - base_ok:>+6}")

    print(f"\nNo-regression check:")
    print(f"  baseline proved, +cosine_rw lost:     {len(rw_lost)}")
    print(f"  baseline proved, +cosine_rw_seq lost: {len(rw_seq_lost)}")

    print(f"\nUnique theorems touched by rw lane (won + progress-only):")
    print(f"  +cosine_rw:     {rw_touched}  (won={len(rw_won)}, progress_only={rw_touched - len(rw_won)})")
    print(f"  +cosine_rw_seq: {rw_seq_touched}  (won={len(rw_seq_won)}, progress_only={rw_seq_touched - len(rw_seq_won)})")

    print(f"\nLean calls / theorem:")
    print(f"  baseline:       {base_calls/max(n,1):.1f}  (total {base_calls})")
    print(f"  +cosine_rw:     {rw_calls/max(n,1):.1f}  (+{(rw_calls - base_calls)/max(n,1):.1f}/theorem)")
    print(f"  +cosine_rw_seq: {rw_seq_calls/max(n,1):.1f}  (+{(rw_seq_calls - base_calls)/max(n,1):.1f}/theorem)")

    print(f"\nTime / theorem:")
    print(f"  baseline:       {base_time/max(n,1):.1f}s  (total {base_time:.0f}s)")
    print(f"  +cosine_rw:     {rw_time/max(n,1):.1f}s")
    print(f"  +cosine_rw_seq: {rw_seq_time/max(n,1):.1f}s")

    if rw_seq_won:
        print(f"\nTheorems won by +cosine_rw_seq:")
        for r in rw_seq_won:
            print(f"  {r.theorem_id}  (lane={r.rw_seq_close_lane})")
    if rw_seq_lost:
        print(f"\nRegressions from +cosine_rw_seq:")
        for r in rw_seq_lost:
            print(f"  {r.theorem_id}  (baseline_lane={r.baseline_close_lane})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-RW-037: theorem-search rw lane integration")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-002_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--output", default="runs/rw37_results.json")
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

    # baseline: no encoder → cosine lanes never activate
    cfg_baseline = deepcopy(base_cfg)
    cfg_baseline.search_mode = "no_learned"

    # +cosine_rw: encoder present, cosine_rw_seq_enabled=False (default)
    cfg_rw = deepcopy(base_cfg)
    cfg_rw.search_mode = "no_learned"
    cfg_rw.cosine_rw_beam = 5

    # +cosine_rw_seq: encoder present, cosine_rw_seq_enabled=True
    cfg_rw_seq = deepcopy(base_cfg)
    cfg_rw_seq.search_mode = "no_learned"
    cfg_rw_seq.cosine_rw_beam = 5
    cfg_rw_seq.cosine_rw_seq_enabled = True

    combined = [TheoremResult(theorem_id=t["theorem_id"], source=t["source"]) for t in theorems]

    for condition, cfg, enc, label in [
        ("baseline", cfg_baseline, None,    "baseline (no cosine)"),
        ("rw",       cfg_rw,       encoder, "+cosine_rw"),
        ("rw_seq",   cfg_rw_seq,   encoder, "+cosine_rw_seq"),
    ]:
        logger.info("Condition: %s (%d theorems)", label, len(theorems))
        t0 = time.time()
        cond_results = _run_condition(condition, theorems, cfg, pipeline, lean, conn, enc)
        elapsed = time.time() - t0
        proved = sum(
            r.baseline_success if condition == "baseline" else
            r.rw_success if condition == "rw" else
            r.rw_seq_success
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
            "experiment": "EXP-RW-037",
            "n_theorems": len(combined),
            "results": [asdict(r) for r in combined],
        }, f, indent=2)
    logger.info("Written to %s", output_path)

    print_report(combined)


if __name__ == "__main__":
    main()
