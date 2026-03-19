"""EXP-APPLY-042: Gated cosine_apply — orchestration vs always-on.

Three conditions on the same 50-theorem Mathlib split:
  baseline        — interleaved_bootstrap + cosine_rw (frozen runtime policy)
  +apply_ungated  — baseline + cosine_apply always-on (top-5, cosine_apply_gated=False)
  +apply_gated    — baseline + cosine_apply gated by _apply_gate() (cosine_apply_gated=True)

Gate fires only when:
  - IB has already run and failed on this goal
  - exactly one open goal remains
  - goal is not automation-shaped (no pure arithmetic)
  - goal is not obviously rw/simp-shaped (simple symmetric eq)
  - goal head is a named predicate (apply target)

Primary comparison:
  marginal theorem wins (gated vs baseline, gated vs ungated)
  Lean calls / theorem (gate cost reduction)

Usage:
    python -m scripts.run_rw42_theorem_search \\
        --checkpoint models/NAV-004_step5000.pt
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
    theorem_id: str
    source: str

    baseline_success: bool = False
    baseline_attempts: int = 0
    baseline_time_s: float = 0.0
    baseline_close_lane: str = ""

    ungated_success: bool = False
    ungated_attempts: int = 0
    ungated_time_s: float = 0.0
    ungated_close_lane: str = ""
    ungated_apply_closes: int = 0

    gated_success: bool = False
    gated_attempts: int = 0
    gated_time_s: float = 0.0
    gated_close_lane: str = ""
    gated_apply_closes: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dominant_lane(result: SearchResult) -> str:
    prov = result.close_provenance
    if not result.success:
        return "failed"
    for label in ("learned", "cosine_apply", "cosine_simp", "cosine_rw_seq",
                  "cosine_rw", "interleaved_bootstrap", "solver_bootstrap",
                  "structural_core", "automation"):
        if label in prov:
            return label
    return prov[-1] if prov else "unknown"


def _count_prov(result: SearchResult, label: str) -> int:
    return sum(1 for p in result.close_provenance if p == label)


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
        apply_closes = _count_prov(result, "cosine_apply")

        res = TheoremResult(theorem_id=thm["theorem_id"], source=thm["source"])
        if condition == "baseline":
            res.baseline_success = result.success
            res.baseline_attempts = result.attempts
            res.baseline_time_s = elapsed
            res.baseline_close_lane = close_lane
        elif condition == "ungated":
            res.ungated_success = result.success
            res.ungated_attempts = result.attempts
            res.ungated_time_s = elapsed
            res.ungated_close_lane = close_lane
            res.ungated_apply_closes = apply_closes
        else:  # gated
            res.gated_success = result.success
            res.gated_attempts = result.attempts
            res.gated_time_s = elapsed
            res.gated_close_lane = close_lane
            res.gated_apply_closes = apply_closes
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
        elif condition == "ungated":
            res.ungated_success = cr.ungated_success
            res.ungated_attempts = cr.ungated_attempts
            res.ungated_time_s = cr.ungated_time_s
            res.ungated_close_lane = cr.ungated_close_lane
            res.ungated_apply_closes = cr.ungated_apply_closes
        else:  # gated
            res.gated_success = cr.gated_success
            res.gated_attempts = cr.gated_attempts
            res.gated_time_s = cr.gated_time_s
            res.gated_close_lane = cr.gated_close_lane
            res.gated_apply_closes = cr.gated_apply_closes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[TheoremResult]) -> None:
    n = len(results)
    base_ok  = sum(r.baseline_success for r in results)
    ung_ok   = sum(r.ungated_success for r in results)
    gated_ok = sum(r.gated_success for r in results)

    ung_lost   = [r for r in results if r.baseline_success and not r.ungated_success]
    gated_lost = [r for r in results if r.baseline_success and not r.gated_success]
    ung_won    = [r for r in results if not r.baseline_success and r.ungated_success]
    gated_won  = [r for r in results if not r.baseline_success and r.gated_success]

    ung_touched   = sum(1 for r in results if r.ungated_apply_closes > 0)
    gated_touched = sum(1 for r in results if r.gated_apply_closes > 0)
    ung_closes    = sum(r.ungated_apply_closes for r in results)
    gated_closes  = sum(r.gated_apply_closes for r in results)

    c_base  = sum(r.baseline_attempts for r in results)
    c_ung   = sum(r.ungated_attempts for r in results)
    c_gated = sum(r.gated_attempts for r in results)

    t_base  = sum(r.baseline_time_s for r in results)
    t_ung   = sum(r.ungated_time_s for r in results)
    t_gated = sum(r.gated_time_s for r in results)

    print("\n" + "=" * 72)
    print("EXP-APPLY-042: Gated cosine_apply vs ungated")
    print("=" * 72)
    print(f"\nTheorems: {n}")
    print(f"\n{'Condition':<45} {'Proved':>7} {'Rate':>7} {'Delta':>7}")
    print("-" * 68)
    print(f"{'baseline (+IB +cosine_rw)':<45} {base_ok:>7} {100*base_ok/max(n,1):>6.1f}%       —")
    print(f"{'+ cosine_apply ungated':<45} {ung_ok:>7} {100*ung_ok/max(n,1):>6.1f}%  {ung_ok-base_ok:>+6}")
    print(f"{'+ cosine_apply gated':<45} {gated_ok:>7} {100*gated_ok/max(n,1):>6.1f}%  {gated_ok-base_ok:>+6}")

    print(f"\nNo-regression check:")
    print(f"  baseline proved, ungated lost:  {len(ung_lost)}")
    print(f"  baseline proved, gated lost:    {len(gated_lost)}")

    print(f"\nGate effectiveness:")
    print(f"  ungated — theorems touched: {ung_touched}, subgoal closes: {ung_closes}")
    print(f"  gated   — theorems touched: {gated_touched}, subgoal closes: {gated_closes}")

    if gated_won:
        print(f"\n  Theorems won by gated (vs baseline):")
        for r in gated_won:
            print(f"    {r.theorem_id}  (lane={r.gated_close_lane}, apply_closes={r.gated_apply_closes})")
    if ung_won and not gated_won:
        print(f"\n  Theorems won by ungated only (gate may be too conservative):")
        for r in ung_won:
            print(f"    {r.theorem_id}  (lane={r.ungated_close_lane}, apply_closes={r.ungated_apply_closes})")

    saved_calls = c_ung - c_gated
    print(f"\nLean calls / theorem:")
    print(f"  baseline:  {c_base/max(n,1):.1f}  (total {c_base})")
    print(f"  ungated:   {c_ung/max(n,1):.1f}  (+{(c_ung-c_base)/max(n,1):.1f}/thm)")
    print(f"  gated:     {c_gated/max(n,1):.1f}  (+{(c_gated-c_base)/max(n,1):.1f}/thm)")
    print(f"  gate saved: {saved_calls} calls ({saved_calls/max(n,1):.1f}/thm)")

    print(f"\nTime / theorem:")
    print(f"  baseline:  {t_base/max(n,1):.1f}s")
    print(f"  ungated:   {t_ung/max(n,1):.1f}s")
    print(f"  gated:     {t_gated/max(n,1):.1f}s")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-APPLY-042: gated apply lane benchmark")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--output", default="runs/rw42_results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--apply-beam", type=int, default=5)
    parser.add_argument("--ib-max-depth", type=int, default=4)
    parser.add_argument("--ib-max-calls", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]
    config.setdefault("evaluation", {})["benchmark_theorems"] = args.theorems
    config["evaluation"].pop("mathlib_test_split", None)

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
        logger.warning("sentence-transformers not available — cosine lanes disabled")

    if lean._backend == "pantograph":
        lean._ensure_server()

    def _make_cfg(apply: bool = False, gated: bool = False) -> SearchConfig:
        cfg = deepcopy(base_cfg)
        cfg.search_mode = "no_learned"
        cfg.interleaved_bootstrap_enabled = True
        cfg.interleaved_bootstrap_max_depth = args.ib_max_depth
        cfg.interleaved_bootstrap_max_calls = args.ib_max_calls
        cfg.cosine_rw_beam = 1
        cfg.cosine_apply_beam = args.apply_beam
        if apply:
            cfg.cosine_apply_enabled = True
            cfg.cosine_apply_gated = gated
        return cfg

    cfg_baseline = _make_cfg(apply=False)
    cfg_ungated  = _make_cfg(apply=True, gated=False)
    cfg_gated    = _make_cfg(apply=True, gated=True)

    combined = [TheoremResult(theorem_id=t["theorem_id"], source=t["source"]) for t in theorems]

    for condition, cfg, enc, label in [
        ("baseline", cfg_baseline, None,    "baseline (+IB +cosine_rw, no apply)"),
        ("ungated",  cfg_ungated,  encoder, "+cosine_apply ungated"),
        ("gated",    cfg_gated,    encoder, "+cosine_apply gated"),
    ]:
        logger.info("Condition: %s (%d theorems)", label, len(theorems))
        t0 = time.time()
        cond_results = _run_condition(condition, theorems, cfg, pipeline, lean, conn, enc)
        elapsed = time.time() - t0
        proved = sum(
            r.baseline_success if condition == "baseline" else
            r.ungated_success if condition == "ungated" else
            r.gated_success
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
            "experiment": "EXP-APPLY-042",
            "apply_beam": args.apply_beam,
            "ib_max_depth": args.ib_max_depth,
            "ib_max_calls": args.ib_max_calls,
            "n_theorems": len(combined),
            "results": [asdict(r) for r in combined],
        }, f, indent=2)
    logger.info("Written to %s", output_path)

    print_report(combined)


if __name__ == "__main__":
    main()
