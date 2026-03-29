"""EXP-SOM-012 Stage 1: fixed-budget depth ladder on hard residuals.

Runs the live proof-search stack on hard residual rows under a grid of:

- bounded attempt budgets (`128`, `256`, `512`, `1024`)
- bounded progress-depth caps (`2`, `4`, `8`, `12`, `16`)

The depth cap is enforced via `SearchConfig.max_progress_steps`, which limits
how many progress-making search steps the post-main solver may take before it
must hand off. This makes the ladder a genuine depth experiment rather than a
budget-only sweep.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark import _build_search_components, _build_theorem_id_map
from scripts.run_exp_som012_hard_collect import (
    _attach_file_logger,
    _theorem_row,
    _write_monitor,
)
from src.benchmark_residuals import summarize_residual_structure
from src.hard_resolution_layer import load_jsonl
from src.proof_search import SearchConfig, SearchResult, search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_csv_ints(spec: str) -> list[int]:
    out: list[int] = []
    for piece in (spec or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return out


def _condition_name(budget: int, depth: int) -> str:
    return f"b{budget}_d{depth}"


def load_hard_residual_rows(path: Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if offset:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def _dedupe_preserve_order(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _file_path_for_theorem(row: dict[str, Any], lean: Any) -> str:
    file_path = str(row.get("lean_path", "") or row.get("file_path", "") or "")
    if file_path:
        return file_path
    if getattr(lean, "_backend", "") != "pantograph" or getattr(lean, "_server", None) is None:
        return ""
    theorem_id = str(row.get("theorem_id", "") or "")
    if not theorem_id:
        return ""
    try:
        info = lean._server.env_inspect(theorem_id)
        module = info.get("module", "")
        if module:
            return module.replace(".", "/") + ".lean"
    except Exception:
        pass
    return ""


def _residual_goal_candidates(row: dict[str, Any]) -> list[tuple[str, str]]:
    candidates = [
        ("last_goal", str(row.get("last_goal", "") or "").strip()),
        ("goal_state", str(row.get("goal_state", "") or "").strip()),
        ("theorem_statement", str(row.get("theorem_statement", "") or "").strip()),
        ("initial_goal", str(row.get("initial_goal", "") or "").strip()),
    ]
    return [(kind, text) for kind, text in _dedupe_preserve_order(candidates) if text]


def _resolve_residual_goal(row: dict[str, Any], lean: Any) -> tuple[str | None, str, str]:
    theorem_id = str(row.get("theorem_id", "") or "")
    file_path = _file_path_for_theorem(row, lean)
    failures: list[str] = []
    for kind, text in _residual_goal_candidates(row):
        if getattr(lean, "_backend", "") != "pantograph":
            return text, kind, ""
        try:
            return lean.goal_start(text, theorem_name=theorem_id, file_path=file_path), kind, ""
        except Exception as exc:
            failures.append(f"{kind}:{type(exc).__name__}")
    return None, "", ",".join(failures[:4])


def _skipped_depth_row(
    row: dict[str, Any],
    *,
    initial_goal: str,
    initial_goal_kind: str,
    start_failure: str,
    budget_cap: int,
    depth_cap: int,
) -> dict[str, Any]:
    out = {
        "theorem_id": str(row.get("theorem_id", "")),
        "source": str(row.get("source", "hard_residuals")),
        "namespace_prefix": str(row.get("namespace_prefix", "")),
        "theorem_statement": str(row.get("theorem_statement", "")),
        "template_id": str(row.get("template_id", "")),
        "proof_steps": int(row.get("proof_steps", 0) or 0),
        "unique_premises": int(row.get("unique_premises", 0) or 0),
        "difficulty_band": str(row.get("difficulty_band", "")),
        "hard_half": bool(row.get("hard_half", False)),
        "split": str(row.get("split", "")),
        "success": False,
        "success_category": "failed",
        "close_lane": "skipped",
        "lane_sequence": "",
        "close_provenance": [],
        "final_closer": "",
        "initial_goal": initial_goal,
        "remaining_goals_snapshot": [],
        "last_goal": "",
        "last_goal_available": False,
        "last_goal_bucket": "empty",
        "last_goal_tags": [],
        "last_goal_shape_features": {},
        "temporal_trace": [],
        "step_trace": [],
        "attempts": 0,
        "goals_closed": 0,
        "goals_remaining": 1,
        "tactics_used": [],
        "time_s": 0.0,
        "goal_start_status": "failed",
        "failure_category": "residual_goal_start_fail",
        "start_failure_family": "residual_goal_start_fail",
        "start_failure_tags": ["failure_category:residual_goal_start_fail"],
        "budget_cap": budget_cap,
        "depth_cap": depth_cap,
        "progress_steps": 0,
        "initial_goal_kind": initial_goal_kind,
        "start_failure": start_failure,
        "input_residual_bucket": str(row.get("residual_bucket", "")),
        "input_hard_track": str(row.get("hard_track", "")),
        "input_reasoning_gap_family": str(row.get("reasoning_gap_family", "")),
        "input_last_goal_bucket": str(row.get("last_goal_bucket", "")),
        "input_last_goal": str(row.get("last_goal", "")),
        "depth_capped": False,
    }
    from src.benchmark_residuals import augment_result_entry

    return augment_result_entry(out)


def summarize_depth_condition(rows: list[dict[str, Any]], budget: int, depth: int) -> dict[str, Any]:
    residual_structure = summarize_residual_structure(rows)
    honest_success = sum(1 for row in rows if row.get("honest_success"))
    self_application_successes = sum(1 for row in rows if row.get("self_application_detected"))
    by_input_track = Counter(str(row.get("input_hard_track", "")) for row in rows)
    solved_by_input_track = Counter(
        str(row.get("input_hard_track", "")) for row in rows if row.get("honest_success")
    )
    by_input_family = Counter(str(row.get("input_reasoning_gap_family", "")) for row in rows)
    solved_by_input_family = Counter(
        str(row.get("input_reasoning_gap_family", "")) for row in rows if row.get("honest_success")
    )
    return {
        "budget": budget,
        "depth": depth,
        "total_theorems": len(rows),
        "raw_success": sum(1 for row in rows if row.get("success")),
        "honest_success": honest_success,
        "self_application_successes": self_application_successes,
        "started_theorems": residual_structure["started_theorems"],
        "depth_capped_rows": sum(1 for row in rows if row.get("depth_capped")),
        "mean_progress_steps": round(
            sum(int(row.get("progress_steps", 0) or 0) for row in rows) / max(len(rows), 1),
            2,
        ),
        "mean_attempts": round(
            sum(int(row.get("attempts", 0) or 0) for row in rows) / max(len(rows), 1),
            2,
        ),
        "by_input_hard_track": dict(by_input_track.most_common()),
        "solved_by_input_hard_track": dict(solved_by_input_track.most_common()),
        "by_input_reasoning_gap_family": dict(by_input_family.most_common()),
        "solved_by_input_reasoning_gap_family": dict(solved_by_input_family.most_common()),
        "residual_structure": residual_structure,
    }


class _TimeoutError(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _TimeoutError("theorem timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument(
        "--inputs",
        required=True,
        help="Hard residual JSONL input (e.g. hard_proof_local.jsonl or hard_proof_all.jsonl)",
    )
    parser.add_argument("--output-dir", default="runs/exp_som012_depth_ladder")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-project", default="data/lean_project")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--budgets", default="128,256,512,1024")
    parser.add_argument("--depths", default="2,4,8,12,16")
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
    parser.add_argument("--per-theorem-timeout", type=int, default=180)
    args = parser.parse_args()

    budgets = _parse_csv_ints(args.budgets)
    depths = _parse_csv_ints(args.depths)
    if not budgets or not depths:
        raise SystemExit("budgets and depths must be non-empty CSV integer lists")

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]
    if args.cosine_rw_seq:
        config.setdefault("search", {})["cosine_rw_seq_enabled"] = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    monitor_path = output_dir / "MONITOR.txt"
    _attach_file_logger(log_path)

    rows = load_hard_residual_rows(
        Path(args.inputs),
        limit=args.limit if args.limit > 0 else None,
        offset=args.offset,
    )
    logger.info("Loaded %d hard residual rows from %s", len(rows), args.inputs)

    pipeline, cfg, lean, _lean_cfg, conn = _build_search_components(
        config,
        Path(args.checkpoint),
        args.device,
    )
    cfg.collect_trace = True
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
    if args.cosine_rw or args.cosine_rw_seq:
        from scripts.run_exp_som012_hard_collect import _load_sentence_encoder

        sentence_encoder = _load_sentence_encoder(args.sentence_encoder)

    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}
    if lean._backend == "pantograph":
        lean._ensure_server()

    all_condition_summaries: dict[str, Any] = {}
    started_at = time.time()
    total_conditions = len(budgets) * len(depths)
    completed_conditions = 0

    for budget in budgets:
        for depth in depths:
            completed_conditions += 1
            condition = _condition_name(budget, depth)
            logger.info(
                "[%d/%d] running depth ladder condition %s",
                completed_conditions,
                total_conditions,
                condition,
            )
            cfg_condition = SearchConfig(**vars(cfg))
            cfg_condition.budget = budget
            cfg_condition.max_progress_steps = depth

            condition_dir = output_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            details_path = condition_dir / "details.jsonl"
            report_path = condition_dir / "report.json"
            condition_rows: list[dict[str, Any]] = []
            raw_success = 0
            started_count = 0
            skipped_count = 0
            total_attempts = 0

            with details_path.open("w") as handle:
                for idx, row in enumerate(rows):
                    theorem_id = str(row.get("theorem_id", "") or "")
                    _write_monitor(
                        monitor_path,
                        total=len(rows),
                        processed=idx,
                        raw_success=raw_success,
                        started=started_count,
                        skipped_start=skipped_count,
                        goal_start_failures=skipped_count,
                        goal_start_repairs=0,
                        hard_residuals=len(condition_rows),
                        trigger_rows=0,
                        total_attempts=total_attempts,
                        started_at=started_at,
                        current_theorem=f"{condition}:{theorem_id}",
                        details_path=details_path,
                        goal_start_path=None,
                        trigger_path=None,
                        log_path=log_path,
                        report_path=report_path,
                    )
                    logger.info("[%s %d/%d] %s", condition, idx + 1, len(rows), theorem_id)

                    initial_goal, initial_goal_kind, start_failure = _resolve_residual_goal(row, lean)
                    if initial_goal is None:
                        theorem_row = _skipped_depth_row(
                            row,
                            initial_goal=str(row.get("last_goal", "") or row.get("initial_goal", "") or ""),
                            initial_goal_kind=initial_goal_kind or "unresolved",
                            start_failure=start_failure,
                            budget_cap=budget,
                            depth_cap=depth,
                        )
                        skipped_count += 1
                    else:
                        accessible_id = name_to_id.get(theorem_id) if name_to_id else None
                        t0 = time.perf_counter()
                        try:
                            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                            signal.alarm(args.per_theorem_timeout)
                            try:
                                result = search(
                                    theorem_id=theorem_id,
                                    initial_goal=initial_goal,
                                    pipeline=pipeline,
                                    conn=conn,
                                    lean=lean,
                                    config=cfg_condition,
                                    accessible_theorem_id=accessible_id,
                                    sentence_encoder=sentence_encoder,
                                )
                            finally:
                                signal.alarm(0)
                                signal.signal(signal.SIGALRM, old_handler)
                        except (Exception, _TimeoutError) as exc:
                            logger.warning("Depth-ladder search error on %s: %s", theorem_id, exc)
                            result = SearchResult(
                                theorem_id=theorem_id,
                                success=False,
                                tactics_used=[],
                                attempts=0,
                                goals_closed=0,
                                goals_remaining=1,
                                progress_steps=0,
                                close_provenance=[],
                            )
                        elapsed = time.perf_counter() - t0
                        theorem_row = _theorem_row(
                            thm=row,
                            result=result,
                            elapsed_s=elapsed,
                            initial_goal=str(initial_goal),
                        )
                        theorem_row["budget_cap"] = budget
                        theorem_row["depth_cap"] = depth
                        theorem_row["progress_steps"] = int(getattr(result, "progress_steps", 0) or 0)
                        theorem_row["depth_capped"] = (
                            int(getattr(result, "progress_steps", 0) or 0) >= depth
                            and not bool(getattr(result, "success", False))
                        )
                        theorem_row["initial_goal_kind"] = initial_goal_kind
                        theorem_row["start_failure"] = start_failure
                        theorem_row["input_residual_bucket"] = str(row.get("residual_bucket", ""))
                        theorem_row["input_hard_track"] = str(row.get("hard_track", ""))
                        theorem_row["input_reasoning_gap_family"] = str(
                            row.get("reasoning_gap_family", "")
                        )
                        theorem_row["input_last_goal_bucket"] = str(row.get("last_goal_bucket", ""))
                        theorem_row["input_last_goal"] = str(row.get("last_goal", ""))
                        raw_success += int(bool(theorem_row.get("success")))
                        started_count += int(bool(theorem_row.get("started")))
                        skipped_count += int(not bool(theorem_row.get("started")))
                        total_attempts += int(theorem_row.get("attempts", 0) or 0)

                    condition_rows.append(theorem_row)
                    handle.write(json.dumps(theorem_row) + "\n")
                    handle.flush()

            condition_summary = summarize_depth_condition(condition_rows, budget, depth)
            all_condition_summaries[condition] = condition_summary
            report = {
                "experiment": "EXP-SOM-012-depth-ladder",
                "condition": condition,
                "benchmark": condition_summary,
                "details": condition_rows,
                "config": {
                    "budget": budget,
                    "depth_cap": depth,
                    "checkpoint": args.checkpoint,
                    "inputs": args.inputs,
                    "backend": args.backend,
                    "device": args.device,
                },
            }
            report_path.write_text(json.dumps(report, indent=2))

    summary = {
        "experiment": "EXP-SOM-012-depth-ladder",
        "inputs": args.inputs,
        "budgets": budgets,
        "depths": depths,
        "conditions": all_condition_summaries,
        "best_honest_condition": max(
            (
                {"condition": name, **cond}
                for name, cond in all_condition_summaries.items()
            ),
            key=lambda row: (row.get("honest_success", 0), -row.get("mean_attempts", 0.0)),
        )
        if all_condition_summaries
        else {},
    }
    (output_dir / "depth_ladder_summary.json").write_text(json.dumps(summary, indent=2))
    _write_monitor(
        monitor_path,
        total=len(rows),
        processed=len(rows),
        raw_success=0,
        started=0,
        skipped_start=0,
        goal_start_failures=0,
        goal_start_repairs=0,
        hard_residuals=0,
        trigger_rows=0,
        total_attempts=0,
        started_at=started_at,
        current_theorem="(complete)",
        details_path=output_dir / "depth_ladder_summary.json",
        goal_start_path=None,
        trigger_path=None,
        log_path=log_path,
        report_path=output_dir / "depth_ladder_summary.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
