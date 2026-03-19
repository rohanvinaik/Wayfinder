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
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

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
        search_mode=search_cfg.get("search_mode", "full"),
        no_learned_premises=search_cfg.get("no_learned_premises", False),
        temporal_mode=search_cfg.get("temporal_mode", "off"),
        cosine_rw_beam=search_cfg.get("cosine_rw_beam", 5),
        cosine_rw_seq_max_atoms=search_cfg.get("cosine_rw_seq_max_atoms", 10),
        cosine_rw_seq_max_calls=search_cfg.get("cosine_rw_seq_max_calls", 50),
        cosine_rw_seq_enabled=search_cfg.get("cosine_rw_seq_enabled", False),
    )

    lean_backend = config.get("lean", {}).get("backend", "stub")
    lean_section = config.get("lean", {})
    lean_cfg = LeanConfig(
        backend=lean_backend,
        hammer_timeout=search_cfg.get("hammer_timeout", 60),
        project_root=lean_section.get("project_root", ""),
        imports=lean_section.get("imports", ["Init"]),
    )
    lean = LeanKernel(lean_cfg)
    conn = sqlite3.connect(config["data"]["proof_network_db"])

    return pipeline, cfg, lean, lean_cfg, conn


def _build_theorem_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Build theorem name → DB integer ID map for accessible-premises lookup."""
    cursor = conn.execute("SELECT id, name FROM entities")
    return {name: eid for eid, name in cursor.fetchall()}


def _resolve_initial_goal(thm: dict, lean: LeanKernel) -> str | None:
    """Resolve the initial goal for proof search.

    For Pantograph with a Mathlib project, uses env_inspect to get the
    theorem's type, then goal_start (with load_sorry fallback for universe-
    polymorphic types). Falls back to raw goal_state if env_inspect fails.

    Returns None only if goal_start fails on both the inspected type AND
    the raw goal_state.
    """
    if lean._backend != "pantograph" or lean._server is None:
        return thm["goal_state"]

    tid = thm["theorem_id"]

    # Try env_inspect → goal_start (proper Lean type with load_sorry fallback)
    try:
        info = lean._server.env_inspect(tid)
        theorem_type = info["type"]["pp"]
        return lean.goal_start(theorem_type)
    except Exception:
        pass

    # env_inspect failed (theorem not in environment, or custom benchmark).
    # Fall back to raw goal_state — works for simple type expressions.
    raw_goal = thm["goal_state"]
    try:
        return lean.goal_start(raw_goal)
    except Exception as e:
        logger.debug("Could not start goal for %s: %s", tid, e)
        return None


def _run_search_loop(
    theorems: list[dict],
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    cfg: SearchConfig,
    sentence_encoder: object | None = None,
) -> tuple[list[dict], int, int]:
    """Run proof search on all theorems. Returns (results, raw_proved, total_attempts)."""
    results: list[dict] = []
    raw_proved = 0
    total_attempts = 0

    # Build name→id map so search can filter to accessible premises
    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}

    # Pre-initialize the Pantograph server before the loop
    if lean._backend == "pantograph":
        lean._ensure_server()

    for i, thm in enumerate(theorems):
        # Register ground-truth tactic for replay backend
        gt_tactic = thm.get("ground_truth_tactic", "")
        if gt_tactic and thm["goal_state"]:
            lean.register_ground_truth(thm["goal_state"], [gt_tactic])

        t0 = time.perf_counter()
        accessible_id = name_to_id.get(thm["theorem_id"]) if name_to_id else None

        # Resolve initial goal — uses env_inspect for Pantograph+Mathlib
        initial_goal = _resolve_initial_goal(thm, lean)

        if initial_goal is None:
            # Could not create goal state (universe polymorphism, missing theorem, etc.)
            results.append(
                {
                    "theorem_id": thm["theorem_id"],
                    "source": thm["source"],
                    "success": False,
                    "success_category": "failed",
                    "close_lane": "skipped",
                    "close_provenance": [],
                    "final_closer": "",
                    "attempts": 0,
                    "goals_closed": 0,
                    "goals_remaining": 1,
                    "tactics_used": [],
                    "time_s": 0.0,
                }
            )
            continue

        result = search(
            theorem_id=thm["theorem_id"],
            initial_goal=initial_goal,
            pipeline=pipeline,
            conn=conn,
            lean=lean,
            config=cfg,
            accessible_theorem_id=accessible_id,
            sentence_encoder=sentence_encoder,
        )
        elapsed = time.perf_counter() - t0

        # Determine dominant close lane for this theorem.
        # Priority: learned > solver_bootstrap > structural_core > automation > failed
        # "dominant" = the highest-priority lane that closed any goal.
        prov = getattr(result, "close_provenance", [])
        if not result.success:
            close_lane = "failed"
        elif any(p == "learned" for p in prov):
            close_lane = "learned"
        elif any(p.startswith("cosine_") for p in prov):
            close_lane = next(p for p in prov if p.startswith("cosine_"))
        elif any(p == "solver_bootstrap" for p in prov):
            close_lane = "solver_bootstrap"
        elif any(p == "structural_core" for p in prov):
            close_lane = "structural_core"
        elif any(p == "automation" for p in prov):
            close_lane = "automation"
        else:
            close_lane = "unknown"

        # Final closer (tactic that closed the last goal)
        final_closer = result.tactics_used[-1] if result.success and result.tactics_used else ""

        # Lane sequence summary: e.g. "structural_core→solver_bootstrap" or "automation"
        lane_sequence = "→".join(dict.fromkeys(prov)) if prov else ""

        results.append(
            {
                "theorem_id": thm["theorem_id"],
                "source": thm["source"],
                "success": result.success,
                "success_category": "raw_success" if result.success else "failed",
                "close_lane": close_lane,
                "lane_sequence": lane_sequence,
                "close_provenance": prov,
                "final_closer": final_closer,
                "temporal_trace": getattr(result, "temporal_trace", []),
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
    from src.arbiter import SoMSearchParams, SoMSlots, som_search
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
            params=SoMSearchParams(
                config=cfg,
                accessible_theorem_id=accessible_id,
            ),
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
    from src.v3_runtime import V3Config, V3SearchParams, V3Slots, v3_search
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
            params=V3SearchParams(
                config=cfg,
                v3_config=v3_cfg,
                accessible_theorem_id=accessible_id,
            ),
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
    cosine_rw: bool = False,
    cosine_rw_seq: bool = False,
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

    # Load sentence encoder for cosine_rw lane
    encoder = None
    if cosine_rw or cosine_rw_seq:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        msg = "  Cosine rw lane: enabled"
        if cosine_rw_seq:
            msg += " + cosine_rw_seq"
        print(f"{msg} (MiniLM encoder loaded)")

    theorems = load_benchmark_theorems(config, limit)
    print(f"Running benchmark on {len(theorems)} theorems (mode={mode})")
    print(f"  Budget: {cfg.budget}, Hammer: {cfg.hammer_delegation}")
    print(f"  Lean backend: {lean_cfg.backend}")

    start = time.time()
    if mode == "v1":
        results, raw_proved, total_attempts = _run_search_loop(
            theorems, pipeline, conn, lean, cfg, sentence_encoder=encoder)
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

    # Lane-separated provenance counts
    lane_counts: dict[str, int] = {}
    for r in results:
        lane = r.get("close_lane", "failed" if not r["success"] else "unknown")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

    report = {
        "benchmark": {
            "total_theorems": n,
            "raw_success": raw_proved,
            "raw_success_rate": round(raw_proved / max(n, 1), 4),
            "axle_assisted_success": 0,
            "axle_repair_only": 0,
            "failed": n - raw_proved,
            "by_close_lane": lane_counts,
            "lane_activity": _summarize_lane_activity(results),
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
            "search_mode": cfg.search_mode,
            "cosine_rw_enabled": bool(cosine_rw),
            "cosine_rw_seq_enabled": bool(cosine_rw_seq or cfg.cosine_rw_seq_enabled),
            "active_rewrite_lane": (
                "cosine_rw_seq"
                if (cosine_rw_seq or cfg.cosine_rw_seq_enabled)
                else ("cosine_rw" if cosine_rw else "none")
            ),
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


def _summarize_lane_activity(results: list[dict]) -> dict[str, dict[str, int]]:
    """Summarize lane activity beyond theorem-level final closers."""
    theorem_touches: dict[str, int] = {}
    subgoal_closes: dict[str, int] = {}
    tactic_rows = {"rw_rows": 0, "rw_seq_rows": 0}

    for r in results:
        touched = set()
        for lane in r.get("close_provenance", []):
            if not lane:
                continue
            subgoal_closes[lane] = subgoal_closes.get(lane, 0) + 1
            touched.add(lane)
        for lane in touched:
            theorem_touches[lane] = theorem_touches.get(lane, 0) + 1

        tactics = r.get("tactics_used", [])
        if any(isinstance(t, str) and t.startswith("rw [") for t in tactics):
            tactic_rows["rw_rows"] += 1
        if any(isinstance(t, str) and t.startswith("rw_seq(") for t in tactics):
            tactic_rows["rw_seq_rows"] += 1

    return {
        "theorem_touches": theorem_touches,
        "subgoal_closes": subgoal_closes,
        "tactic_rows": tactic_rows,
    }


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

    # Lane-separated provenance
    lanes = bm.get("by_close_lane", {})
    lane_activity = bm.get("lane_activity", {})
    if lanes:
        print("  --- Close lane breakdown ---")
        for lane in [
            "automation",
            "structural_core",
            "solver_bootstrap",
            "cosine_rw",
            "cosine_rw_seq",
            "learned",
            "failed",
        ]:
            count = lanes.get(lane, 0)
            if count > 0:
                print(f"    {lane}: {count} ({100 * count / max(n, 1):.1f}%)")
    if lane_activity:
        touches = lane_activity.get("theorem_touches", {})
        subgoal = lane_activity.get("subgoal_closes", {})
        tactic_rows = lane_activity.get("tactic_rows", {})
        if touches or subgoal:
            print("  --- Lane activity (touches / subgoal closes) ---")
            for lane in [
                "automation",
                "structural_core",
                "solver_bootstrap",
                "cosine_rw",
                "cosine_rw_seq",
                "learned",
            ]:
                if lane in touches or lane in subgoal:
                    print(
                        f"    {lane}: touches={touches.get(lane, 0)}"
                        f" subgoals={subgoal.get(lane, 0)}"
                    )
        if tactic_rows:
            print(
                "  --- Rewrite tactic rows ---"
                f" rw={tactic_rows.get('rw_rows', 0)}"
                f" rw_seq={tactic_rows.get('rw_seq_rows', 0)}"
            )

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
    parser.add_argument(
        "--theorems",
        type=Path,
        default=None,
        help="Override benchmark theorems file (JSONL)",
    )
    parser.add_argument(
        "--lean-project",
        type=str,
        default=None,
        help="Lean project root for Pantograph (enables Mathlib imports)",
    )
    parser.add_argument(
        "--lean-imports",
        type=str,
        nargs="+",
        default=None,
        help="Lean imports for Pantograph (e.g., Mathlib)",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        default=None,
        choices=["full", "learned_only", "learned_structural", "no_learned"],
        help="Search lane mode: full (all lanes), learned_only, learned_structural, no_learned",
    )
    parser.add_argument(
        "--no-learned-premises",
        action="store_true",
        help="Strip navigator premises from hammer calls (premise-value ablation)",
    )
    parser.add_argument(
        "--temporal",
        type=str,
        default=None,
        choices=["off", "shadow", "active"],
        help="Temporal controller mode: off (default), shadow (log only), active",
    )
    parser.add_argument(
        "--cosine-rw",
        action="store_true",
        help="Enable cosine rw lane (scope → MiniLM → top-k beam + Lean verify)",
    )
    parser.add_argument(
        "--cosine-rw-beam",
        type=int,
        default=5,
        help="Beam width for cosine rw lane (default: 5)",
    )
    parser.add_argument(
        "--cosine-rw-seq",
        action="store_true",
        help="Enable sequential bare rewrite lane; this replaces single-step cosine_rw in static lane order",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = load_config(args.config)
    if args.backend:
        config.setdefault("lean", {})["backend"] = args.backend
    if args.lean_project:
        config.setdefault("lean", {})["project_root"] = args.lean_project
    if args.lean_imports:
        config.setdefault("lean", {})["imports"] = args.lean_imports
    if args.search_mode:
        config.setdefault("search", {})["search_mode"] = args.search_mode
    if args.no_learned_premises:
        config.setdefault("search", {})["no_learned_premises"] = True
    if args.temporal:
        config.setdefault("search", {})["temporal_mode"] = args.temporal
    if args.theorems:
        config.setdefault("evaluation", {})["benchmark_theorems"] = str(args.theorems)
        config.setdefault("evaluation", {}).pop("mathlib_test_split", None)
    if args.cosine_rw_beam != 5:
        config.setdefault("search", {})["cosine_rw_beam"] = args.cosine_rw_beam
    if args.cosine_rw_seq:
        config.setdefault("search", {})["cosine_rw_seq_enabled"] = True
    report = run_benchmark(
        config,
        args.checkpoint,
        args.device,
        args.limit,
        mode=args.mode,
        cosine_rw=args.cosine_rw or args.cosine_rw_seq,
        cosine_rw_seq=args.cosine_rw_seq,
    )

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
