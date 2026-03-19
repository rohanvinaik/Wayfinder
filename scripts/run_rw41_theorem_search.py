"""EXP-APPLY-041: Theorem-search integration with cosine_apply lane.

Five conditions on the same 50-theorem Mathlib split:
  baseline               — interleaved_bootstrap only (new structural policy)
  +cosine_rw             — baseline + cosine_rw (top-1)
  +cosine_apply          — baseline + cosine_apply (top-5)
  +cosine_rw+apply       — baseline + cosine_rw + cosine_apply

Note: interleaved_bootstrap replaces structural_core in all conditions,
consistent with EXP-SIMP-039 freeze decision.

Metrics:
  proved theorems + delta vs baseline
  no-regression delta
  cosine_apply lane: theorems touched, subgoal closes, theorem wins
  Lean calls / theorem

Usage:
    python -m scripts.run_rw41_theorem_search \\
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

    rw_apply_success: bool = False
    rw_apply_attempts: int = 0
    rw_apply_time_s: float = 0.0
    rw_apply_close_lane: str = ""
    rw_apply_subgoal_closes: int = 0


def _dominant_lane(result: SearchResult) -> str:
    prov = result.close_provenance
    if not result.success:
        return "failed"
    for label in ("learned", "cosine_apply", "cosine_simp", "cosine_rw_seq",
                  "cosine_rw", "interleaved_bootstrap", "solver_bootstrap",
                  "structural_core", "automation"):
        if any(p == label or p.startswith(label) for p in prov):
            return label
    return prov[0] if prov else "unknown"


def _count_prov(result: SearchResult, prefix: str) -> int:
    return sum(1 for p in result.close_provenance if p == prefix or p.startswith(prefix))


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
        apply_closes = _count_prov(result, "cosine_apply")

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
        else:  # rw_apply
            res.rw_apply_success = result.success
            res.rw_apply_attempts = result.attempts
            res.rw_apply_time_s = elapsed
            res.rw_apply_close_lane = close_lane
            res.rw_apply_subgoal_closes = apply_closes
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
        else:  # rw_apply
            res.rw_apply_success = cr.rw_apply_success
            res.rw_apply_attempts = cr.rw_apply_attempts
            res.rw_apply_time_s = cr.rw_apply_time_s
            res.rw_apply_close_lane = cr.rw_apply_close_lane
            res.rw_apply_subgoal_closes = cr.rw_apply_subgoal_closes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[TheoremResult]) -> None:
    n = len(results)
    base_ok     = sum(r.baseline_success for r in results)
    rw_ok       = sum(r.rw_success for r in results)
    rw_apply_ok = sum(r.rw_apply_success for r in results)

    rw_apply_lost  = [r for r in results if r.baseline_success and not r.rw_apply_success]
    rw_apply_won   = [r for r in results if not r.baseline_success and r.rw_apply_success]

    rw_apply_touched   = sum(1 for r in results if r.rw_apply_subgoal_closes > 0)
    rw_apply_closes    = sum(r.rw_apply_subgoal_closes for r in results)

    c_base    = sum(r.baseline_attempts for r in results)
    c_rw      = sum(r.rw_attempts for r in results)
    c_rw_app  = sum(r.rw_apply_attempts for r in results)

    t_base   = sum(r.baseline_time_s for r in results)
    t_rw     = sum(r.rw_time_s for r in results)
    t_rwapp  = sum(r.rw_apply_time_s for r in results)

    print("\n" + "=" * 72)
    print("EXP-APPLY-041: Theorem-Search Integration (cosine_apply)")
    print("=" * 72)
    print(f"\nTheorems: {n}")
    print(f"\n{'Condition':<45} {'Proved':>7} {'Rate':>7} {'Delta':>7}")
    print("-" * 68)
    print(f"{'baseline (+interleaved_bootstrap)':<45} {base_ok:>7} {100*base_ok/max(n,1):>6.1f}%       —")
    print(f"{'+ cosine_rw':<45} {rw_ok:>7} {100*rw_ok/max(n,1):>6.1f}%  {rw_ok-base_ok:>+6}")
    print(f"{'+ cosine_rw + cosine_apply':<45} {rw_apply_ok:>7} {100*rw_apply_ok/max(n,1):>6.1f}%  {rw_apply_ok-base_ok:>+6}")

    print(f"\nNo-regression check:")
    print(f"  baseline proved, +rw+apply lost:     {len(rw_apply_lost)}")

    print(f"\ncosine_apply lane activity (+rw+apply condition):")
    print(f"  theorems touched:                    {rw_apply_touched}")
    print(f"  total apply subgoal closes:          {rw_apply_closes}")
    if rw_apply_won:
        print(f"\n  Theorems won by +rw+apply (vs baseline):")
        for r in rw_apply_won:
            print(f"    {r.theorem_id}  (lane={r.rw_apply_close_lane}, apply_closes={r.rw_apply_subgoal_closes})")

    print(f"\nLean calls / theorem:")
    print(f"  baseline:              {c_base/max(n,1):.1f}  (total {c_base})")
    print(f"  +cosine_rw:            {c_rw/max(n,1):.1f}  (+{(c_rw-c_base)/max(n,1):.1f}/thm)")
    print(f"  +cosine_rw+apply:      {c_rw_app/max(n,1):.1f}  (+{(c_rw_app-c_base)/max(n,1):.1f}/thm)")

    print(f"\nTime / theorem:")
    print(f"  baseline:              {t_base/max(n,1):.1f}s")
    print(f"  +cosine_rw:            {t_rw/max(n,1):.1f}s")
    print(f"  +cosine_rw+apply:      {t_rwapp/max(n,1):.1f}s")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-APPLY-041: apply lane theorem-search benchmark")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--output", default="runs/rw41_results.json")
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

    def _base(ib: bool = True, rw: bool = False, apply: bool = False) -> SearchConfig:
        cfg = deepcopy(base_cfg)
        cfg.search_mode = "no_learned"
        cfg.interleaved_bootstrap_enabled = ib
        cfg.interleaved_bootstrap_max_depth = args.ib_max_depth
        cfg.interleaved_bootstrap_max_calls = args.ib_max_calls
        cfg.cosine_rw_beam = 1
        cfg.cosine_apply_beam = args.apply_beam
        if rw:
            pass  # cosine_rw always active when encoder present
        if apply:
            cfg.cosine_apply_enabled = True
        return cfg

    # Condition 1: baseline = interleaved_bootstrap, no cosine (encoder=None)
    # Condition 2: +cosine_rw = IB + rw (encoder present, apply disabled)
    # Condition 3: +cosine_apply = IB + apply (encoder present, rw still active — cosine_rw always on)
    #   Note: cosine_rw is always appended when encoder is present; to isolate apply,
    #   we include both rw and apply in condition 3 (IB+rw+apply), and report the
    #   marginal gain vs condition 2. Pure apply isolation would require disabling rw.
    # Condition 4: +cosine_rw+apply = IB + rw + apply

    # Three real conditions:
    #   baseline:  IB only, no encoder (no cosine lanes)
    #   rw:        IB + cosine_rw (encoder present, apply disabled)
    #   rw_apply:  IB + cosine_rw + cosine_apply (encoder present)
    # Note: cosine_rw fires automatically whenever encoder is present,
    # so "apply" condition = rw+apply; there is no apply-only isolation.
    cfg_baseline  = _base(ib=True, apply=False)
    cfg_rw        = _base(ib=True, apply=False)
    cfg_rw_apply  = _base(ib=True, apply=True)

    combined = [TheoremResult(theorem_id=t["theorem_id"], source=t["source"]) for t in theorems]

    for condition, cfg, enc, label in [
        ("baseline", cfg_baseline, None,    "baseline (+IB, no cosine)"),
        ("rw",       cfg_rw,       encoder, "+cosine_rw"),
        ("rw_apply", cfg_rw_apply, encoder, "+cosine_rw +cosine_apply"),
    ]:
        logger.info("Condition: %s (%d theorems)", label, len(theorems))
        t0 = time.time()
        cond_results = _run_condition(condition, theorems, cfg, pipeline, lean, conn, enc)
        elapsed = time.time() - t0
        proved = sum(
            r.baseline_success if condition == "baseline" else
            r.rw_success if condition == "rw" else
            r.rw_apply_success
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
            "experiment": "EXP-APPLY-041",
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
