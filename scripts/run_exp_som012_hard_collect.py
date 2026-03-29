"""EXP-SOM-012 Stage 0: hard-set theorem collection with rich trace capture.

This runner is intentionally separate from `run_benchmark.py`. It is designed
for data acquisition rather than leaderboard-style evaluation:

- preserves full `step_trace` and `temporal_trace`
- records startability failures and file-context repairs
- optionally probes trigger states during the same pass
- writes JSONL incrementally so long runs are monitorable
- materializes a reusable hard-data bundle on completion
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_hard_collection_bundle import build_hard_collection_bundle
from scripts.collect_trigger_states import (
    _build_goal_start_failure_row,
    _goal_shape_features,
    _load_exec_selector,
    extract_trigger_states,
)
from scripts.run_benchmark import (
    _build_search_components,
    _build_theorem_id_map,
    _group_by_source,
    _resolve_initial_goal,
)
from src.benchmark_residuals import augment_result_entry, summarize_residual_structure
from src.hard_data_tags import (
    attempt_band,
    canonicalize_theorem_id,
    classify_goal_bucket,
    classify_reasoning_gap_family,
    goal_bucket_tags,
    trace_pathology_tags,
)
from src.hard_resolution_layer import load_jsonl as load_resolution_rows
from src.hard_resolution_layer import materialize_hard_resolution_layer
from src.proof_network import get_accessible_premises
from src.proof_search import SearchResult, search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _attach_file_logger(path: Path) -> None:
    root = logging.getLogger()
    target = str(path)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == target:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)


def load_hard_theorems(
    path: Path,
    limit: int | None = None,
    offset: int = 0,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = canonicalize_theorem_id(str(row.get("theorem_id", row.get("name", ""))))
            rows.append(
                {
                    "theorem_id": theorem_id,
                    "raw_theorem_id": row.get("theorem_id", row.get("name", "")),
                    "goal_state": row.get("goal_state", "") or row.get("theorem_statement", "") or row.get("statement", ""),
                    "ground_truth_tactic": row.get("ground_truth_tactic", ""),
                    "source": row.get("source", "hard_theorems"),
                    "namespace_prefix": row.get("namespace_prefix", ""),
                    "theorem_statement": row.get("theorem_statement", "") or row.get("goal_state", "") or row.get("statement", ""),
                    "template_id": row.get("template_id", ""),
                    "proof_steps": row.get("proof_steps", 0),
                    "unique_premises": row.get("unique_premises", 0),
                    "difficulty_band": row.get("difficulty_band", ""),
                    "hard_half": bool(row.get("hard_half", False)),
                    "split": row.get("split", ""),
                    "file_path": row.get("file_path", ""),
                    "module": row.get("module", ""),
                    "lean_path": row.get("lean_path", ""),
                    "theorem_line": int(row.get("theorem_line", 0) or 0),
                }
            )

    if shuffle and rows:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if offset:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def _final_open_goals(result: SearchResult) -> list[str]:
    step_trace = getattr(result, "step_trace", [])
    if not step_trace:
        return []
    last = step_trace[-1]
    open_goals = last.get("open_goals_after", [])
    if not isinstance(open_goals, list):
        return []
    return [str(goal) for goal in open_goals if isinstance(goal, str) and goal.strip()]


def _dominant_lane(result: SearchResult) -> str:
    prov = list(getattr(result, "close_provenance", []))
    if not result.success:
        return "failed"
    if any(p == "self_application" for p in prov):
        return "self_application"
    if any(p == "learned" for p in prov):
        return "learned"
    if any(p.startswith("cosine_") for p in prov):
        return next(p for p in prov if p.startswith("cosine_"))
    if any(p == "solver_bootstrap" for p in prov):
        return "solver_bootstrap"
    if any(p == "structural_core" for p in prov):
        return "structural_core"
    if any(p == "automation" for p in prov):
        return "automation"
    return prov[-1] if prov else "unknown"


def _startability_fields(start_row: dict[str, Any] | None) -> dict[str, Any]:
    if not start_row:
        return {
            "goal_start_status": "direct",
            "failure_category": "",
            "start_failure_family": "",
            "start_failure_tags": [],
            "feedback": None,
            "repair_attempted": False,
            "repair_success": False,
            "repair_goal_state": "",
            "repair_tier_used": "",
            "repair_failure_category": "",
            "repair_feedback": None,
            "module": "",
            "lean_path": "",
            "theorem_line": 0,
            "context_features": {},
            "context_unsupported_kinds": [],
        }
    return {
        "goal_start_status": start_row.get("goal_start_status", ""),
        "failure_category": start_row.get("failure_category", ""),
        "start_failure_family": start_row.get("start_failure_family", ""),
        "start_failure_tags": start_row.get("start_failure_tags", []),
        "feedback": start_row.get("feedback"),
        "repair_attempted": bool(start_row.get("repair_attempted")),
        "repair_success": bool(start_row.get("repair_success")),
        "repair_goal_state": start_row.get("repair_goal_state", ""),
        "repair_tier_used": start_row.get("repair_tier_used", ""),
        "repair_failure_category": start_row.get("repair_failure_category", ""),
        "repair_feedback": start_row.get("repair_feedback"),
        "module": start_row.get("module", ""),
        "lean_path": start_row.get("lean_path", ""),
        "theorem_line": int(start_row.get("theorem_line", 0) or 0),
        "context_features": start_row.get("context_features", {}),
        "context_unsupported_kinds": start_row.get("context_unsupported_kinds", []),
    }


def _refined_follow_on_stage(row: dict[str, Any]) -> str:
    if bool(row.get("success")):
        return "none"
    if not bool(row.get("started")):
        return "compiler_specialist"
    family = str(row.get("reasoning_gap_family", "") or "")
    if family == "compiler_specialist":
        return "compiler_specialist"
    if family == "theorem_replanner":
        return "theorem_replanner"
    return "hard_proof_solver"


def _theorem_row(
    thm: dict[str, Any],
    result: SearchResult,
    elapsed_s: float,
    initial_goal: str,
    start_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    close_provenance = list(getattr(result, "close_provenance", []))
    tactics_used = list(getattr(result, "tactics_used", []))
    remaining_goals = _final_open_goals(result)
    lane_sequence = "→".join(dict.fromkeys(close_provenance)) if close_provenance else ""
    final_closer = tactics_used[-1] if result.success and tactics_used else ""
    last_goal = remaining_goals[0] if remaining_goals else ""
    last_goal_bucket = classify_goal_bucket(last_goal)

    row = {
        "theorem_id": thm["theorem_id"],
        "theorem_full_name": thm["theorem_id"],
        "source": thm["source"],
        "file_path": thm.get("file_path", ""),
        "module": thm.get("module", ""),
        "lean_path": thm.get("lean_path", ""),
        "theorem_line": int(thm.get("theorem_line", 0) or 0),
        "namespace_prefix": thm.get("namespace_prefix", ""),
        "theorem_statement": thm.get("theorem_statement", ""),
        "template_id": thm.get("template_id", ""),
        "proof_steps": int(thm.get("proof_steps", 0) or 0),
        "unique_premises": int(thm.get("unique_premises", 0) or 0),
        "difficulty_band": thm.get("difficulty_band", ""),
        "hard_half": bool(thm.get("hard_half", False)),
        "split": thm.get("split", ""),
        "success": result.success,
        "success_category": "raw_success" if result.success else "failed",
        "close_lane": _dominant_lane(result),
        "lane_sequence": lane_sequence,
        "close_provenance": close_provenance,
        "final_closer": final_closer,
        "initial_goal": initial_goal,
        "initial_goal_shape_features": _goal_shape_features(initial_goal),
        "remaining_goals_snapshot": remaining_goals,
        "last_goal": last_goal,
        "last_goal_available": bool(last_goal),
        "last_goal_bucket": last_goal_bucket,
        "last_goal_tags": goal_bucket_tags(last_goal),
        "last_goal_shape_features": _goal_shape_features(last_goal) if last_goal else {},
        "temporal_trace": list(getattr(result, "temporal_trace", [])),
        "step_trace": list(getattr(result, "step_trace", [])),
        "attempts": int(getattr(result, "attempts", 0)),
        "attempt_band": attempt_band(int(getattr(result, "attempts", 0))),
        "goals_closed": int(getattr(result, "goals_closed", 0)),
        "goals_remaining": int(getattr(result, "goals_remaining", 0)),
        "tactics_used": tactics_used,
        "time_s": round(elapsed_s, 3),
    }
    row.update(_startability_fields(start_row))
    if not str(row.get("module", "") or "").strip():
        row["module"] = str(thm.get("module", "") or "")
    if not str(row.get("lean_path", "") or "").strip():
        row["lean_path"] = str(thm.get("lean_path", "") or thm.get("file_path", "") or "")
    if not int(row.get("theorem_line", 0) or 0):
        row["theorem_line"] = int(thm.get("theorem_line", 0) or 0)
    out = augment_result_entry(row)
    pathology_tags = trace_pathology_tags(list(getattr(result, "step_trace", [])), remaining_goals=remaining_goals)
    out["search_pathology_tags"] = pathology_tags
    out["reasoning_gap_family"] = classify_reasoning_gap_family(
        success=bool(out.get("success")),
        started=bool(out.get("started")),
        residual_bucket=str(out.get("residual_bucket", "")),
        last_goal_bucket=last_goal_bucket,
        goal_text=last_goal or initial_goal,
        remaining_goals=remaining_goals,
        pathology_tags=pathology_tags,
    )
    previous_stage = str(out.get("follow_on_stage", "") or "")
    refined_stage = _refined_follow_on_stage(out)
    if previous_stage and previous_stage != refined_stage:
        out["follow_on_stage_previous"] = previous_stage
    out["follow_on_stage"] = refined_stage
    return out


def _skipped_row(
    thm: dict[str, Any],
    start_row: dict[str, Any] | None,
) -> dict[str, Any]:
    theorem_goal = str(thm.get("goal_state", "") or thm.get("theorem_statement", ""))
    row = {
        "theorem_id": thm["theorem_id"],
        "theorem_full_name": thm["theorem_id"],
        "source": thm["source"],
        "file_path": thm.get("file_path", ""),
        "module": thm.get("module", ""),
        "lean_path": thm.get("lean_path", ""),
        "theorem_line": int(thm.get("theorem_line", 0) or 0),
        "namespace_prefix": thm.get("namespace_prefix", ""),
        "theorem_statement": thm.get("theorem_statement", ""),
        "template_id": thm.get("template_id", ""),
        "proof_steps": int(thm.get("proof_steps", 0) or 0),
        "unique_premises": int(thm.get("unique_premises", 0) or 0),
        "difficulty_band": thm.get("difficulty_band", ""),
        "hard_half": bool(thm.get("hard_half", False)),
        "split": thm.get("split", ""),
        "success": False,
        "success_category": "failed",
        "close_lane": "skipped",
        "lane_sequence": "",
        "close_provenance": [],
        "final_closer": "",
        "initial_goal": theorem_goal,
        "initial_goal_shape_features": _goal_shape_features(theorem_goal),
        "remaining_goals_snapshot": [],
        "last_goal": "",
        "last_goal_available": False,
        "last_goal_bucket": "empty",
        "last_goal_tags": [],
        "last_goal_shape_features": {},
        "temporal_trace": [],
        "step_trace": [],
        "attempts": 0,
        "attempt_band": "0",
        "goals_closed": 0,
        "goals_remaining": 1,
        "tactics_used": [],
        "time_s": 0.0,
    }
    row.update(_startability_fields(start_row))
    if not str(row.get("module", "") or "").strip():
        row["module"] = str(thm.get("module", "") or "")
    if not str(row.get("lean_path", "") or "").strip():
        row["lean_path"] = str(thm.get("lean_path", "") or thm.get("file_path", "") or "")
    if not int(row.get("theorem_line", 0) or 0):
        row["theorem_line"] = int(thm.get("theorem_line", 0) or 0)
    out = augment_result_entry(row)
    out["search_pathology_tags"] = []
    out["reasoning_gap_family"] = classify_reasoning_gap_family(
        success=False,
        started=bool(out.get("started")),
        residual_bucket=str(out.get("residual_bucket", "")),
        last_goal_bucket="empty",
        goal_text=theorem_goal,
        remaining_goals=[],
        pathology_tags=[],
    )
    previous_stage = str(out.get("follow_on_stage", "") or "")
    refined_stage = _refined_follow_on_stage(out)
    if previous_stage and previous_stage != refined_stage:
        out["follow_on_stage_previous"] = previous_stage
    out["follow_on_stage"] = refined_stage
    return out


def _write_monitor(
    path: Path,
    *,
    total: int,
    processed: int,
    raw_success: int,
    started: int,
    skipped_start: int,
    goal_start_failures: int,
    goal_start_repairs: int,
    hard_residuals: int,
    trigger_rows: int,
    total_attempts: int,
    started_at: float,
    current_theorem: str,
    details_path: Path | None = None,
    goal_start_path: Path | None = None,
    trigger_path: Path | None = None,
    log_path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    elapsed = time.time() - started_at
    lines = [
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Current theorem: {current_theorem or '(idle)'}",
        "",
        f"Processed: {processed}/{total}",
        f"Raw success: {raw_success}",
        f"Started: {started}",
        f"Skipped start: {skipped_start}",
        f"Goal-start failures seen: {goal_start_failures}",
        f"Goal-start repairs: {goal_start_repairs}",
        f"Hard residuals: {hard_residuals}",
        f"Trigger rows: {trigger_rows}",
        f"Avg attempts/theorem: {round(total_attempts / max(processed, 1), 1)}",
        f"Elapsed seconds: {round(elapsed, 1)}",
        "",
        "Artifacts",
        f"details.jsonl: {str(details_path) if details_path is not None else '(none)'}",
        f"details rows: {processed}",
        f"goal_start_failures.jsonl: {str(goal_start_path) if goal_start_path is not None else '(none)'}",
        f"goal-start rows: {goal_start_failures}",
        f"trigger_states.jsonl: {str(trigger_path) if trigger_path is not None else '(none)'}",
        f"trigger rows: {trigger_rows}",
        f"run.log: {str(log_path) if log_path is not None else '(none)'}",
        f"report.json: {str(report_path) if report_path is not None else '(none)'}",
        f"report exists: {bool(report_path and report_path.exists())}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _load_sentence_encoder(model_name: str) -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning("Could not load sentence encoder %s: %s", model_name, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument(
        "--theorems",
        default="data/hard_split_som_q75/hard_theorems_train.jsonl",
        help="Hard theorem JSONL file",
    )
    parser.add_argument("--output-dir", default="runs/exp_som012_hard_collect")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle theorem order deterministically before offset/limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --shuffle is enabled")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--search-mode", default=None)
    parser.add_argument("--temporal", default=None)
    parser.add_argument("--cosine-rw", action="store_true")
    parser.add_argument("--cosine-rw-seq", action="store_true")
    parser.add_argument("--cosine-rw-beam", type=int, default=0)
    parser.add_argument("--interleaved-bootstrap", action="store_true")
    parser.add_argument("--interleaved-bootstrap-max-depth", type=int, default=4)
    parser.add_argument("--interleaved-bootstrap-max-calls", type=int, default=20)
    parser.add_argument("--family-classifier-path", default="")
    parser.add_argument("--family-classifier-torch-path", default="")
    parser.add_argument("--no-norm-then-close", action="store_true")
    parser.add_argument("--sentence-encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--per-theorem-timeout", type=int, default=300)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--selector", default="", help="Optional ExecSelector checkpoint for trigger probing")
    parser.add_argument("--probe-lean", action="store_true")
    parser.add_argument("--probe-k", type=int, default=5)
    parser.add_argument("--min-trace-length", type=int, default=2)
    parser.add_argument("--min-strategy-support", type=int, default=3)
    parser.add_argument("--hard-resolution-candidate-limit", type=int, default=12)
    parser.add_argument("--hard-resolution-exemplar-limit", type=int, default=5)
    parser.add_argument("--no-materialize", action="store_true")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]
    if args.search_mode:
        config.setdefault("search", {})["search_mode"] = args.search_mode
    if args.temporal:
        config.setdefault("search", {})["temporal_mode"] = args.temporal
    if args.cosine_rw_seq:
        config.setdefault("search", {})["cosine_rw_seq_enabled"] = True
    if args.budget > 0:
        config.setdefault("search", {})["budget"] = args.budget

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    goal_start_path = output_dir / "goal_start_failures.jsonl"
    trigger_path = output_dir / "trigger_states.jsonl"
    monitor_path = output_dir / "MONITOR.txt"
    log_path = output_dir / "run.log"
    report_path = output_dir / "report.json"
    _attach_file_logger(log_path)

    theorems = load_hard_theorems(
        Path(args.theorems),
        limit=args.limit if args.limit > 0 else None,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info("Loaded %d hard theorems from %s", len(theorems), args.theorems)

    pipeline, cfg, lean, lean_cfg, conn = _build_search_components(
        config,
        Path(args.checkpoint),
        args.device,
    )
    cfg.collect_trace = True
    if args.budget > 0:
        cfg.budget = args.budget
    if args.cosine_rw_beam > 0:
        cfg.cosine_rw_beam = args.cosine_rw_beam
    if args.interleaved_bootstrap:
        cfg.interleaved_bootstrap_enabled = True
        cfg.interleaved_bootstrap_max_depth = args.interleaved_bootstrap_max_depth
        cfg.interleaved_bootstrap_max_calls = args.interleaved_bootstrap_max_calls
    if args.family_classifier_path:
        cfg.family_classifier_path = args.family_classifier_path
    if args.family_classifier_torch_path:
        cfg.family_classifier_torch_path = args.family_classifier_torch_path
    if args.no_norm_then_close:
        cfg.norm_then_close_enabled = False

    sentence_encoder = None
    if args.cosine_rw or args.cosine_rw_seq or args.selector:
        sentence_encoder = _load_sentence_encoder(args.sentence_encoder)

    selector = None
    selector_enc = None
    if args.selector:
        selector_result = _load_exec_selector(args.selector)
        if selector_result is not None:
            selector, selector_enc = selector_result

    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}
    id_to_name = {}
    if args.selector:
        id_to_name = {eid: name for eid, name in conn.execute("SELECT id, name FROM entities")}

    started_at = time.time()
    processed = 0
    raw_success = 0
    total_attempts = 0
    goal_start_failures = 0
    goal_start_repairs = 0
    trigger_rows_total = 0
    trigger_positive = 0
    started_count = 0
    skipped_start_count = 0
    hard_residual_count = 0
    summary_rows: list[dict[str, Any]] = []

    _write_monitor(
        monitor_path,
        total=len(theorems),
        processed=0,
        raw_success=0,
        started=0,
        skipped_start=0,
        goal_start_failures=0,
        goal_start_repairs=0,
        hard_residuals=0,
        trigger_rows=0,
        total_attempts=0,
        started_at=started_at,
        current_theorem="(startup)",
        details_path=details_path,
        goal_start_path=goal_start_path,
        trigger_path=trigger_path,
        log_path=log_path,
        report_path=report_path,
    )

    if lean._backend == "pantograph":
        lean._ensure_server()

    with details_path.open("w") as details_handle, goal_start_path.open("w") as start_handle, trigger_path.open("w") as trigger_handle:
        for idx, thm in enumerate(theorems):
            theorem_id = str(thm.get("theorem_id", ""))
            logger.info("[%d/%d] %s", idx + 1, len(theorems), theorem_id)
            _write_monitor(
                monitor_path,
                total=len(theorems),
                processed=processed,
                raw_success=raw_success,
                started=started_count,
                skipped_start=skipped_start_count,
                goal_start_failures=goal_start_failures,
                goal_start_repairs=goal_start_repairs,
                hard_residuals=hard_residual_count,
                trigger_rows=trigger_rows_total,
                total_attempts=total_attempts,
                started_at=started_at,
                current_theorem=theorem_id,
                details_path=details_path,
                goal_start_path=goal_start_path,
                trigger_path=trigger_path,
                log_path=log_path,
                report_path=report_path,
            )

            gt = thm.get("ground_truth_tactic", "")
            if gt and thm.get("goal_state"):
                lean.register_ground_truth(str(thm["goal_state"]), [str(gt)])

            accessible_id = name_to_id.get(theorem_id) if name_to_id else None
            initial_goal = _resolve_initial_goal(thm, lean)
            start_row: dict[str, Any] | None = None
            repaired_goal: str | None = None
            repair_result: Any | None = None

            if initial_goal is None:
                goal_start_failures += 1
                try:
                    repair_result = lean.goal_via_file_context(
                        theorem_id,
                        str(thm.get("file_path", "") or ""),
                        [],
                        "",
                        project_root=args.lean_project,
                    )
                    if repair_result.success:
                        repaired_goal = repair_result.goal_state
                except Exception:
                    repair_result = None
                    repaired_goal = None

                start_row = _build_goal_start_failure_row(
                    thm,
                    lean,
                    repaired_goal=repaired_goal,
                    repair_result=repair_result,
                )
                start_handle.write(json.dumps(start_row) + "\n")
                start_handle.flush()

                if repaired_goal is None:
                    theorem_row = _skipped_row(thm, start_row)
                    details_handle.write(json.dumps(theorem_row) + "\n")
                    details_handle.flush()
                    skipped_start_count += 1
                    processed += 1
                    summary_rows.append(
                        {
                            key: theorem_row.get(key)
                            for key in (
                                "theorem_id",
                                "source",
                                "success",
                                "close_lane",
                                "attempts",
                                "attempt_band",
                                "goals_closed",
                                "goals_remaining",
                                "started",
                                "progress_band",
                                "residual_bucket",
                                "follow_on_stage",
                                "reasoning_gap_family",
                                "honest_success",
                                "self_application_detected",
                                "time_s",
                            )
                        }
                    )
                    _write_monitor(
                        monitor_path,
                        total=len(theorems),
                        processed=processed,
                        raw_success=raw_success,
                        started=started_count,
                        skipped_start=skipped_start_count,
                        goal_start_failures=goal_start_failures,
                        goal_start_repairs=goal_start_repairs,
                        hard_residuals=hard_residual_count,
                        trigger_rows=trigger_rows_total,
                        total_attempts=total_attempts,
                        started_at=started_at,
                        current_theorem=theorem_id,
                        details_path=details_path,
                        goal_start_path=goal_start_path,
                        trigger_path=trigger_path,
                        log_path=log_path,
                        report_path=report_path,
                    )
                    continue

                goal_start_repairs += 1
                initial_goal = repaired_goal

            t0 = time.perf_counter()
            try:
                def _timeout_handler(signum: int, frame: object) -> None:  # noqa: ARG001
                    raise TimeoutError("theorem search exceeded per-theorem timeout")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.per_theorem_timeout)
                try:
                    result = search(
                        theorem_id=theorem_id,
                        initial_goal=initial_goal,
                        pipeline=pipeline,
                        conn=conn,
                        lean=lean,
                        config=cfg,
                        accessible_theorem_id=accessible_id,
                        sentence_encoder=sentence_encoder,
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except (Exception, TimeoutError) as exc:
                logger.warning("Search error on %s: %s", theorem_id, exc)
                result = SearchResult(
                    theorem_id=theorem_id,
                    success=False,
                    tactics_used=[],
                    attempts=0,
                    goals_closed=0,
                    goals_remaining=1,
                    close_provenance=[],
                )
            elapsed = time.perf_counter() - t0

            theorem_row = _theorem_row(
                thm=thm,
                result=result,
                elapsed_s=elapsed,
                initial_goal=str(initial_goal),
                start_row=start_row,
            )
            details_handle.write(json.dumps(theorem_row) + "\n")
            if (idx + 1) % max(args.flush_every, 1) == 0:
                details_handle.flush()

            processed += 1
            total_attempts += int(getattr(result, "attempts", 0))
            if result.success:
                raw_success += 1
            if theorem_row.get("started"):
                started_count += 1
            else:
                skipped_start_count += 1
            if theorem_row.get("follow_on_stage") == "hard_proof_solver":
                hard_residual_count += 1
            summary_rows.append(
                {
                    key: theorem_row.get(key)
                    for key in (
                        "theorem_id",
                        "source",
                        "success",
                        "close_lane",
                        "attempts",
                        "attempt_band",
                        "goals_closed",
                        "goals_remaining",
                        "started",
                        "progress_band",
                        "residual_bucket",
                        "follow_on_stage",
                        "reasoning_gap_family",
                        "honest_success",
                        "self_application_detected",
                        "time_s",
                    )
                }
            )

            if args.selector and selector is not None and selector_enc is not None and sentence_encoder is not None:
                premise_names: list[str] = []
                if accessible_id is not None:
                    premise_ids = get_accessible_premises(conn, accessible_id)
                    premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]
                trigger_rows = extract_trigger_states(
                    theorem_id=theorem_id,
                    step_trace=list(getattr(result, "step_trace", [])),
                    theorem_success=result.success,
                    accessible_premises=premise_names,
                    selector=selector,
                    selector_enc=selector_enc,
                    sentence_encoder=sentence_encoder,
                    lean=lean if args.probe_lean else None,
                    probe_lean=args.probe_lean,
                    probe_k=args.probe_k,
                )
                for row in trigger_rows:
                    row["template_id"] = thm.get("template_id", "")
                    row["difficulty_band"] = thm.get("difficulty_band", "")
                    row["proof_steps"] = thm.get("proof_steps", 0)
                    row["split"] = thm.get("split", "")
                    trigger_handle.write(json.dumps(row) + "\n")
                    trigger_rows_total += 1
                    trigger_positive += int(row.get("trigger_label") or row.get("can_apply") or 0)
                if trigger_rows:
                    trigger_handle.flush()

            _write_monitor(
                monitor_path,
                total=len(theorems),
                processed=processed,
                raw_success=raw_success,
                started=started_count,
                skipped_start=skipped_start_count,
                goal_start_failures=goal_start_failures,
                goal_start_repairs=goal_start_repairs,
                hard_residuals=hard_residual_count,
                trigger_rows=trigger_rows_total,
                total_attempts=total_attempts,
                started_at=started_at,
                current_theorem=theorem_id,
                details_path=details_path,
                goal_start_path=goal_start_path,
                trigger_path=trigger_path,
                log_path=log_path,
                report_path=report_path,
            )

    residual_structure = summarize_residual_structure(summary_rows)
    honest_success = sum(1 for row in summary_rows if row.get("honest_success"))
    self_application_successes = sum(
        1 for row in summary_rows if row.get("self_application_detected")
    )
    report = {
        "experiment": "EXP-SOM-012-stage0-hard-collect",
        "benchmark": {
            "total_theorems": len(summary_rows),
            "raw_success": raw_success,
            "raw_success_rate": round(raw_success / max(len(summary_rows), 1), 4),
            "honest_success": honest_success,
            "honest_success_rate": round(honest_success / max(len(summary_rows), 1), 4),
            "self_application_successes": self_application_successes,
            "failed": len(summary_rows) - raw_success,
            "started_theorems": residual_structure["started_theorems"],
            "skipped_start": residual_structure["skipped_start"],
            "started_success_rate": residual_structure["started_success_rate"],
            "residual_structure": residual_structure,
        },
        "efficiency": {
            "total_attempts": total_attempts,
            "avg_attempts_per_theorem": round(total_attempts / max(len(summary_rows), 1), 1),
            "avg_attempts_proved": round(
                sum(int(row.get("attempts", 0)) for row in summary_rows if row.get("success"))
                / max(raw_success, 1),
                1,
            ),
            "avg_time_per_theorem_s": round(
                sum(float(row.get("time_s", 0.0)) for row in summary_rows)
                / max(len(summary_rows), 1),
                2,
            ),
            "total_time_s": round(time.time() - started_at, 1),
        },
        "by_source": _group_by_source(summary_rows),
        "config": {
            "checkpoint": args.checkpoint,
            "budget": cfg.budget,
            "device": args.device,
            "lean_backend": lean_cfg.backend,
            "search_mode": cfg.search_mode,
            "temporal_mode": cfg.temporal_mode,
            "cosine_rw_enabled": bool(args.cosine_rw),
            "cosine_rw_seq_enabled": bool(args.cosine_rw_seq or cfg.cosine_rw_seq_enabled),
            "cosine_rw_beam": cfg.cosine_rw_beam,
            "interleaved_bootstrap_enabled": cfg.interleaved_bootstrap_enabled,
            "family_classifier_path": cfg.family_classifier_path,
            "family_classifier_torch_path": cfg.family_classifier_torch_path,
            "norm_then_close_enabled": cfg.norm_then_close_enabled,
            "collect_trace": True,
            "theorems": args.theorems,
        },
        "run_order": {
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "offset": int(args.offset),
            "limit": int(args.limit),
        },
        "collection": {
            "details_jsonl": str(details_path),
            "goal_start_failures_jsonl": str(goal_start_path),
            "trigger_states_jsonl": str(trigger_path) if args.selector else "",
            "goal_start_failures": goal_start_failures,
            "goal_start_repairs": goal_start_repairs,
            "trigger_rows": trigger_rows_total,
            "trigger_positive": trigger_positive,
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    if not args.no_materialize:
        bundle_summary = build_hard_collection_bundle(
            inputs=[details_path],
            output_dir=output_dir / "bundle",
            min_trace_length=args.min_trace_length,
            min_strategy_support=args.min_strategy_support,
        )
        report["collection"]["bundle_dir"] = str(output_dir / "bundle")
        report["collection"]["bundle_summary"] = bundle_summary
        bundle_rows = load_resolution_rows(output_dir / "bundle" / "collection_all.jsonl")
        hard_resolution_summary = materialize_hard_resolution_layer(
            rows=bundle_rows,
            output_dir=output_dir / "bundle" / "hard_resolution_layer",
            conn_or_db=conn,
            candidate_limit=args.hard_resolution_candidate_limit,
            exemplar_limit=args.hard_resolution_exemplar_limit,
        )
        report["collection"]["hard_resolution_layer_dir"] = str(
            output_dir / "bundle" / "hard_resolution_layer"
        )
        report["collection"]["hard_resolution_summary"] = hard_resolution_summary
        report_path.write_text(json.dumps(report, indent=2))

    _write_monitor(
        monitor_path,
        total=len(theorems),
        processed=processed,
        raw_success=raw_success,
        started=started_count,
        skipped_start=skipped_start_count,
        goal_start_failures=goal_start_failures,
        goal_start_repairs=goal_start_repairs,
        hard_residuals=hard_residual_count,
        trigger_rows=trigger_rows_total,
        total_attempts=total_attempts,
        started_at=started_at,
        current_theorem="(complete)",
        details_path=details_path,
        goal_start_path=goal_start_path,
        trigger_path=trigger_path,
        log_path=log_path,
        report_path=report_path,
    )

    lean.close()
    conn.close()

    print("=" * 72)
    print("EXP-SOM-012 Hard Collection")
    print("=" * 72)
    print(
        f"  raw_success={report['benchmark']['raw_success']}/"
        f"{report['benchmark']['total_theorems']}"
    )
    print(
        f"  started={report['benchmark']['started_theorems']} "
        f"skip={report['benchmark']['skipped_start']}"
    )
    print(
        f"  hard_residuals={report['benchmark']['residual_structure']['by_follow_on_stage'].get('hard_proof_solver', 0)}"
    )
    print(f"  goal_start_failures={goal_start_failures} repairs={goal_start_repairs}")
    if args.selector:
        print(f"  trigger_rows={trigger_rows_total} trigger_positive={trigger_positive}")
    print(f"  details={details_path}")
    print(f"  report={report_path}")


if __name__ == "__main__":
    main()
