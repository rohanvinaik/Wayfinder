"""EXP-SIMP-039: Interleaved Structural Bootstrap theorem-search benchmark.

Four conditions on the same 50-theorem Mathlib split:
  baseline                    — structural_core + automation only (NAV-004)
  +cosine_rw                  — baseline + cosine_rw (top-1 beam)
  +interleaved_bootstrap      — baseline + interleaved intros→simp→aesop loop
  +cosine_rw+interleaved      — all three lanes

Metrics per condition:
  theorem closes + delta vs baseline
  no-regression delta
  interleaved_bootstrap lane activity:
    theorems touched (≥1 subgoal close)
    simp closes, simp→aesop closes, aesop-direct closes
  Lean calls / theorem

Usage:
    python -m scripts.run_rw39_theorem_search \\
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
    baseline_close_provenance: list[str] | None = None
    baseline_tactics_used: list[str] | None = None

    rw_success: bool = False
    rw_attempts: int = 0
    rw_time_s: float = 0.0
    rw_close_lane: str = ""
    rw_close_provenance: list[str] | None = None
    rw_tactics_used: list[str] | None = None

    ib_success: bool = False           # interleaved_bootstrap only
    ib_attempts: int = 0
    ib_time_s: float = 0.0
    ib_close_lane: str = ""
    ib_close_provenance: list[str] | None = None
    ib_tactics_used: list[str] | None = None
    ib_simp_closes: int = 0
    ib_simp_aesop_closes: int = 0
    ib_aesop_closes: int = 0

    rw_ib_success: bool = False        # cosine_rw + interleaved_bootstrap
    rw_ib_attempts: int = 0
    rw_ib_time_s: float = 0.0
    rw_ib_close_lane: str = ""
    rw_ib_close_provenance: list[str] | None = None
    rw_ib_tactics_used: list[str] | None = None
    rw_ib_simp_closes: int = 0
    rw_ib_simp_aesop_closes: int = 0


def _dominant_lane(result: SearchResult) -> str:
    prov = result.close_provenance
    if not result.success:
        return "failed"
    for label in ("learned", "interleaved_bootstrap", "cosine_simp",
                  "cosine_rw_seq", "cosine_rw", "solver_bootstrap",
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

        res = TheoremResult(theorem_id=thm["theorem_id"], source=thm["source"])
        if condition == "baseline":
            res.baseline_success = result.success
            res.baseline_attempts = result.attempts
            res.baseline_time_s = elapsed
            res.baseline_close_lane = close_lane
            res.baseline_close_provenance = list(result.close_provenance)
            res.baseline_tactics_used = list(result.tactics_used)
        elif condition == "rw":
            res.rw_success = result.success
            res.rw_attempts = result.attempts
            res.rw_time_s = elapsed
            res.rw_close_lane = close_lane
            res.rw_close_provenance = list(result.close_provenance)
            res.rw_tactics_used = list(result.tactics_used)
        elif condition == "ib":
            res.ib_success = result.success
            res.ib_attempts = result.attempts
            res.ib_time_s = elapsed
            res.ib_close_lane = close_lane
            res.ib_close_provenance = list(result.close_provenance)
            res.ib_tactics_used = list(result.tactics_used)
            res.ib_simp_closes = _count_prov(result, "interleaved_bootstrap/simp")
            res.ib_simp_aesop_closes = _count_prov(result, "interleaved_bootstrap/simp_aesop")
            res.ib_aesop_closes = _count_prov(result, "interleaved_bootstrap") - res.ib_simp_closes - res.ib_simp_aesop_closes
        else:  # rw_ib
            res.rw_ib_success = result.success
            res.rw_ib_attempts = result.attempts
            res.rw_ib_time_s = elapsed
            res.rw_ib_close_lane = close_lane
            res.rw_ib_close_provenance = list(result.close_provenance)
            res.rw_ib_tactics_used = list(result.tactics_used)
            res.rw_ib_simp_closes = _count_prov(result, "interleaved_bootstrap/simp")
            res.rw_ib_simp_aesop_closes = _count_prov(result, "interleaved_bootstrap/simp_aesop")
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
            res.baseline_close_provenance = cr.baseline_close_provenance
            res.baseline_tactics_used = cr.baseline_tactics_used
        elif condition == "rw":
            res.rw_success = cr.rw_success
            res.rw_attempts = cr.rw_attempts
            res.rw_time_s = cr.rw_time_s
            res.rw_close_lane = cr.rw_close_lane
            res.rw_close_provenance = cr.rw_close_provenance
            res.rw_tactics_used = cr.rw_tactics_used
        elif condition == "ib":
            res.ib_success = cr.ib_success
            res.ib_attempts = cr.ib_attempts
            res.ib_time_s = cr.ib_time_s
            res.ib_close_lane = cr.ib_close_lane
            res.ib_close_provenance = cr.ib_close_provenance
            res.ib_tactics_used = cr.ib_tactics_used
            res.ib_simp_closes = cr.ib_simp_closes
            res.ib_simp_aesop_closes = cr.ib_simp_aesop_closes
            res.ib_aesop_closes = cr.ib_aesop_closes
        else:  # rw_ib
            res.rw_ib_success = cr.rw_ib_success
            res.rw_ib_attempts = cr.rw_ib_attempts
            res.rw_ib_time_s = cr.rw_ib_time_s
            res.rw_ib_close_lane = cr.rw_ib_close_lane
            res.rw_ib_close_provenance = cr.rw_ib_close_provenance
            res.rw_ib_tactics_used = cr.rw_ib_tactics_used
            res.rw_ib_simp_closes = cr.rw_ib_simp_closes
            res.rw_ib_simp_aesop_closes = cr.rw_ib_simp_aesop_closes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[TheoremResult]) -> None:
    n = len(results)
    base_ok  = sum(r.baseline_success for r in results)
    rw_ok    = sum(r.rw_success for r in results)
    ib_ok    = sum(r.ib_success for r in results)
    rw_ib_ok = sum(r.rw_ib_success for r in results)

    ib_lost    = [r for r in results if r.baseline_success and not r.ib_success]
    rw_ib_lost = [r for r in results if r.baseline_success and not r.rw_ib_success]
    ib_won     = [r for r in results if not r.baseline_success and r.ib_success]
    rw_ib_won  = [r for r in results if not r.baseline_success and r.rw_ib_success]

    ib_touched      = sum(1 for r in results if r.ib_simp_closes + r.ib_simp_aesop_closes + r.ib_aesop_closes > 0)
    ib_simp_total   = sum(r.ib_simp_closes for r in results)
    ib_sa_total     = sum(r.ib_simp_aesop_closes for r in results)
    ib_aesop_total  = sum(r.ib_aesop_closes for r in results)

    rw_ib_touched   = sum(1 for r in results if r.rw_ib_simp_closes + r.rw_ib_simp_aesop_closes > 0)
    rw_ib_sa_total  = sum(r.rw_ib_simp_aesop_closes for r in results)

    def calls(attr: str) -> int:
        return sum(getattr(r, attr) for r in results)

    def t(attr: str) -> float:
        return sum(getattr(r, attr) for r in results)

    print("\n" + "=" * 72)
    print("EXP-SIMP-039: Interleaved Structural Bootstrap")
    print("=" * 72)
    print(f"\nTheorems: {n}")
    print(f"\n{'Condition':<45} {'Proved':>7} {'Rate':>7} {'Delta':>7}")
    print("-" * 68)
    print(f"{'baseline':<45} {base_ok:>7} {100*base_ok/max(n,1):>6.1f}%       —")
    print(f"{'+ cosine_rw':<45} {rw_ok:>7} {100*rw_ok/max(n,1):>6.1f}%  {rw_ok-base_ok:>+6}")
    print(f"{'+ interleaved_bootstrap':<45} {ib_ok:>7} {100*ib_ok/max(n,1):>6.1f}%  {ib_ok-base_ok:>+6}")
    print(f"{'+ cosine_rw + interleaved_bootstrap':<45} {rw_ib_ok:>7} {100*rw_ib_ok/max(n,1):>6.1f}%  {rw_ib_ok-base_ok:>+6}")

    print(f"\nNo-regression check:")
    print(f"  baseline proved, +ib lost:       {len(ib_lost)}")
    print(f"  baseline proved, +rw+ib lost:    {len(rw_ib_lost)}")

    print(f"\ninterleaved_bootstrap lane activity (+ib condition):")
    print(f"  theorems touched:                {ib_touched}")
    print(f"  simp closes (partial progress):  {ib_simp_total}")
    print(f"  simp→aesop closes (full chain):  {ib_sa_total}")
    print(f"  aesop-direct closes:             {ib_aesop_total}")
    if ib_won:
        print(f"\n  Theorems won by +ib (vs baseline):")
        for r in ib_won:
            print(f"    {r.theorem_id}  (lane={r.ib_close_lane})")

    print(f"\ninterleaved_bootstrap activity (+rw+ib condition):")
    print(f"  theorems touched:                {rw_ib_touched}")
    print(f"  simp→aesop closes:               {rw_ib_sa_total}")
    if rw_ib_won:
        print(f"\n  Theorems won by +rw+ib (vs baseline):")
        for r in rw_ib_won:
            print(f"    {r.theorem_id}  (lane={r.rw_ib_close_lane})")

    if ib_lost:
        print(f"\n  Regressions in +ib vs baseline:")
        for r in ib_lost:
            print(f"    {r.theorem_id}  (baseline_lane={r.baseline_close_lane})")

    c_base = calls("baseline_attempts")
    c_rw   = calls("rw_attempts")
    c_ib   = calls("ib_attempts")
    c_rwib = calls("rw_ib_attempts")
    print(f"\nLean calls / theorem:")
    print(f"  baseline:                        {c_base/max(n,1):.1f}  (total {c_base})")
    print(f"  +cosine_rw:                      {c_rw/max(n,1):.1f}  (+{(c_rw-c_base)/max(n,1):.1f}/thm)")
    print(f"  +interleaved_bootstrap:          {c_ib/max(n,1):.1f}  (+{(c_ib-c_base)/max(n,1):.1f}/thm)")
    print(f"  +cosine_rw+interleaved:          {c_rwib/max(n,1):.1f}  (+{(c_rwib-c_base)/max(n,1):.1f}/thm)")

    t_base = t("baseline_time_s")
    t_ib   = t("ib_time_s")
    t_rwib = t("rw_ib_time_s")
    print(f"\nTime / theorem:")
    print(f"  baseline:                        {t_base/max(n,1):.1f}s")
    print(f"  +interleaved_bootstrap:          {t_ib/max(n,1):.1f}s")
    print(f"  +cosine_rw+interleaved:          {t_rwib/max(n,1):.1f}s")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-SIMP-039: interleaved bootstrap benchmark")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--output", default="runs/rw39_results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    # Bootstrap tuning
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

    # Condition 1: baseline
    cfg_baseline = deepcopy(base_cfg)
    cfg_baseline.search_mode = "no_learned"

    # Condition 2: +cosine_rw
    cfg_rw = deepcopy(base_cfg)
    cfg_rw.search_mode = "no_learned"
    cfg_rw.cosine_rw_beam = 1

    # Condition 3: +interleaved_bootstrap
    cfg_ib = deepcopy(base_cfg)
    cfg_ib.search_mode = "no_learned"
    cfg_ib.interleaved_bootstrap_enabled = True
    cfg_ib.interleaved_bootstrap_max_depth = args.ib_max_depth
    cfg_ib.interleaved_bootstrap_max_calls = args.ib_max_calls

    # Condition 4: +cosine_rw +interleaved_bootstrap
    cfg_rw_ib = deepcopy(base_cfg)
    cfg_rw_ib.search_mode = "no_learned"
    cfg_rw_ib.cosine_rw_beam = 1
    cfg_rw_ib.interleaved_bootstrap_enabled = True
    cfg_rw_ib.interleaved_bootstrap_max_depth = args.ib_max_depth
    cfg_rw_ib.interleaved_bootstrap_max_calls = args.ib_max_calls

    combined = [TheoremResult(theorem_id=t["theorem_id"], source=t["source"]) for t in theorems]

    for condition, cfg, enc, label in [
        ("baseline", cfg_baseline, None,    "baseline"),
        ("rw",       cfg_rw,       encoder, "+cosine_rw"),
        ("ib",       cfg_ib,       None,    "+interleaved_bootstrap"),
        ("rw_ib",    cfg_rw_ib,    encoder, "+cosine_rw +interleaved_bootstrap"),
    ]:
        logger.info("Condition: %s (%d theorems)", label, len(theorems))
        t0 = time.time()
        cond_results = _run_condition(condition, theorems, cfg, pipeline, lean, conn, enc)
        elapsed = time.time() - t0
        proved = sum(
            r.baseline_success if condition == "baseline" else
            r.rw_success if condition == "rw" else
            r.ib_success if condition == "ib" else
            r.rw_ib_success
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
            "experiment": "EXP-SIMP-039",
            "ib_max_depth": args.ib_max_depth,
            "ib_max_calls": args.ib_max_calls,
            "n_theorems": len(combined),
            "results": [asdict(r) for r in combined],
        }, f, indent=2)
    logger.info("Written to %s", output_path)

    print_report(combined)


if __name__ == "__main__":
    main()
