"""Build a reusable artifact bundle from hard-set theorem collection rows.

The hard-set collector writes rich per-theorem rows with full `step_trace`,
residual labels, and theorem metadata. This script turns those raw rows into
the specific datasets needed by the current and planned SoM stages:

- `collection_all.jsonl`        : normalized theorem-level rows
- `hard_residuals.jsonl`        : failed rows routed to `hard_proof_solver`
- `hard_proof_all.jsonl`        : alias of all hard residual rows for postfreeze runners
- `last_goal_residuals.jsonl`   : hard residuals with a recoverable last goal
- `hard_proof_local.jsonl`      : one-goal and local near-miss residuals
- `hard_proof_planner.jsonl`    : small multi-goal residuals
- `goal_start_failures.jsonl`   : skipped / repaired-start diagnostics
- `dr_ducky_capsules.jsonl`     : symbolic Dr. Ducky capsules over hard residuals
- `dr_ducky_ledger_packets.jsonl`: projector/engine-ready ledger packets
- `temporal_dataset.jsonl`      : current temporal-controller training rows
- `strategy_memory.json`        : symbolic orchestration prior mined from solved traces
- `summary.json`
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_temporal_dataset import _build_temporal_rows
from scripts.mine_strategy_memory import _build_strategy_memory, _extract_strategy_observations
from src.benchmark_residuals import (
    augment_result_entry,
    detect_self_application,
    is_self_application_tactic,
)
from src.dr_ducky import build_goal_capsule, summarize_capsules
from src.hard_data_tags import (
    attempt_band,
    classify_goal_bucket,
    classify_reasoning_gap_family,
    goal_bucket_tags,
    sanitize_goal_text,
    trace_pathology_tags,
)

LOCAL_BUCKETS = {"single_goal_near_miss", "single_goal_stall"}
PLANNER_BUCKETS = {"multi_goal_small_progress", "multi_goal_small_stall"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _infer_remaining_goals(step_trace: list[dict[str, Any]]) -> list[str]:
    if not step_trace:
        return []
    final_open = step_trace[-1].get("open_goals_after", [])
    if not isinstance(final_open, list):
        return []
    return [str(goal) for goal in final_open if isinstance(goal, str) and goal.strip()]


def _infer_last_goal(row: dict[str, Any]) -> str:
    last_goal = row.get("last_goal")
    if isinstance(last_goal, str) and last_goal.strip():
        return last_goal
    remaining_goals = row.get("remaining_goals_snapshot")
    if isinstance(remaining_goals, list) and remaining_goals:
        goal = remaining_goals[0]
        if isinstance(goal, str) and goal.strip():
            return goal
    step_trace = row.get("step_trace", [])
    if isinstance(step_trace, list):
        inferred = _infer_remaining_goals(step_trace)
        if inferred:
            return inferred[0]
    return ""


def _hard_track(bucket: str) -> str:
    if bucket in LOCAL_BUCKETS:
        return "hard_proof_local"
    if bucket in PLANNER_BUCKETS:
        return "hard_proof_planner"
    return "hard_proof_other"


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


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = augment_result_entry(row)
    step_trace = out.get("step_trace", [])
    remaining_goals = out.get("remaining_goals_snapshot")
    if not isinstance(remaining_goals, list):
        remaining_goals = _infer_remaining_goals(step_trace if isinstance(step_trace, list) else [])
    out["remaining_goals_snapshot"] = remaining_goals
    out["remaining_goals_snapshot_count"] = len(remaining_goals)
    out["last_goal"] = _infer_last_goal(out)
    out["last_goal_available"] = bool(out["last_goal"])
    out["last_goal_bucket"] = out.get("last_goal_bucket") or classify_goal_bucket(str(out["last_goal"]))
    out["last_goal_tags"] = out.get("last_goal_tags") or goal_bucket_tags(str(out["last_goal"]))
    out["attempt_band"] = out.get("attempt_band") or attempt_band(int(out.get("attempts", 0) or 0))
    out["hard_track"] = _hard_track(str(out.get("residual_bucket", "")))
    pathology_tags = trace_pathology_tags(
        step_trace if isinstance(step_trace, list) else [],
        remaining_goals=remaining_goals,
    )
    out["search_pathology_tags"] = pathology_tags
    previous_family = str(out.get("reasoning_gap_family", "") or "")
    recomputed_family = classify_reasoning_gap_family(
        success=bool(out.get("success")),
        started=bool(out.get("started")),
        residual_bucket=str(out.get("residual_bucket", "")),
        last_goal_bucket=str(out.get("last_goal_bucket", "")),
        goal_text=str(out.get("last_goal") or out.get("initial_goal") or ""),
        remaining_goals=remaining_goals,
        pathology_tags=pathology_tags,
    )
    if previous_family and previous_family != recomputed_family:
        out["reasoning_gap_family_previous"] = previous_family
    out["reasoning_gap_family"] = recomputed_family
    previous_stage = str(out.get("follow_on_stage", "") or "")
    refined_stage = _refined_follow_on_stage(out)
    if previous_stage and previous_stage != refined_stage:
        out["follow_on_stage_previous"] = previous_stage
    out["follow_on_stage"] = refined_stage
    out["hard_track"] = _hard_track(str(out.get("residual_bucket", "")))
    return out


def _sanitize_step_trace(theorem_id: str, step_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for entry in step_trace:
        tactic = str(entry.get("tactic", "") or "")
        if is_self_application_tactic(tactic, theorem_id):
            continue
        cleaned = dict(entry)
        for field in ("goal_before", "attempted_goal"):
            if field in cleaned:
                cleaned[field] = sanitize_goal_text(str(cleaned.get(field, "") or ""))
        for field in ("open_goals_before", "open_goals_after"):
            if isinstance(cleaned.get(field), list):
                cleaned[field] = [sanitize_goal_text(str(goal or "")) for goal in cleaned[field]]
        sanitized.append(cleaned)
    return sanitized


def build_hard_collection_bundle(
    inputs: list[Path],
    output_dir: Path,
    min_trace_length: int = 2,
    min_strategy_support: int = 3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(_normalize_row(row) for row in _load_jsonl(path))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "collection_all.jsonl", rows)

    hard_residuals = [row for row in rows if row.get("follow_on_stage") == "hard_proof_solver"]
    last_goal_rows = [row for row in hard_residuals if row.get("last_goal_available")]
    local_rows = [row for row in hard_residuals if row.get("hard_track") == "hard_proof_local"]
    planner_rows = [row for row in hard_residuals if row.get("hard_track") == "hard_proof_planner"]
    goal_start_failures = [
        row
        for row in rows
        if str(row.get("close_lane")) == "skipped" or str(row.get("row_type")) == "goal_start_failure"
    ]

    _write_jsonl(output_dir / "hard_residuals.jsonl", hard_residuals)
    _write_jsonl(output_dir / "hard_proof_all.jsonl", hard_residuals)
    _write_jsonl(output_dir / "last_goal_residuals.jsonl", last_goal_rows)
    _write_jsonl(output_dir / "hard_proof_local.jsonl", local_rows)
    _write_jsonl(output_dir / "hard_proof_planner.jsonl", planner_rows)
    _write_jsonl(output_dir / "goal_start_failures.jsonl", goal_start_failures)

    ducky_rows = [row for row in hard_residuals if str(row.get("last_goal", "") or "").strip()]
    ducky_capsules = [build_goal_capsule(row) for row in ducky_rows]
    ducky_capsule_rows = [capsule.to_dict() for capsule in ducky_capsules]
    ducky_ledger_rows = [
        {
            "theorem_id": str(row.get("theorem_id", "") or ""),
            "residual_bucket": str(row.get("residual_bucket", "") or ""),
            "reasoning_gap_family": str(row.get("reasoning_gap_family", "") or ""),
            "goal_bucket": capsule.specification.goal_bucket,
            "ledger_seed": capsule.ledger_seed.to_dict(),
            "allowed_engines": list(capsule.allowed_engines),
            "projector_policy": dict(capsule.projector_policy),
            "execution_budgets": dict(capsule.execution_budgets),
            "projector_markers": list(capsule.specification.projector_markers),
            "residual_geometry": dict(capsule.specification.residual_geometry),
            "negative_geometry": {
                "suppression_hints": list(capsule.suppression_hints),
                "search_pathology_tags": list(capsule.specification.pathology_tags),
            },
        }
        for row, capsule in zip(ducky_rows, ducky_capsules)
    ]
    _write_jsonl(output_dir / "dr_ducky_capsules.jsonl", ducky_capsule_rows)
    _write_jsonl(output_dir / "dr_ducky_ledger_packets.jsonl", ducky_ledger_rows)
    ducky_summary = summarize_capsules(ducky_capsules) if ducky_capsules else {"total_capsules": 0}

    temporal_rows: list[dict[str, Any]] = []
    for row in rows:
        step_trace = row.get("step_trace", [])
        if not isinstance(step_trace, list) or len(step_trace) < min_trace_length:
            continue
        theorem_id = str(row.get("theorem_id", ""))
        sanitized_trace = _sanitize_step_trace(theorem_id, step_trace)
        if len(sanitized_trace) < min_trace_length:
            continue
        honest_success = row.get("honest_success")
        if honest_success is None:
            honest_success = bool(row.get("success")) and not detect_self_application(row)
        temporal_rows.extend(
            _build_temporal_rows(
                theorem_id=theorem_id,
                step_trace=sanitized_trace,
                theorem_success=bool(honest_success),
            )
        )
    _write_jsonl(output_dir / "temporal_dataset.jsonl", temporal_rows)

    observations: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("success"):
            continue
        honest_success = row.get("honest_success")
        if honest_success is False:
            continue
        if honest_success is None and detect_self_application(row):
            continue
        step_trace = row.get("step_trace", [])
        if not isinstance(step_trace, list) or not step_trace:
            continue
        theorem_id = str(row.get("theorem_id", ""))
        sanitized_trace = _sanitize_step_trace(theorem_id, step_trace)
        if not sanitized_trace:
            continue
        observations.extend(
            _extract_strategy_observations(
                theorem_id=theorem_id,
                step_trace=sanitized_trace,
                template=str(row.get("template_id", "")),
            )
        )
    strategy_memory = _build_strategy_memory(observations, min_support=min_strategy_support)
    (output_dir / "strategy_memory.json").write_text(json.dumps(strategy_memory, indent=2))

    summary = {
        "inputs": [str(path) for path in inputs],
        "total_theorems": len(rows),
        "raw_success": sum(1 for row in rows if row.get("success")),
        "honest_success": sum(1 for row in rows if row.get("honest_success")),
        "self_application_successes": sum(
            1 for row in rows if row.get("self_application_detected")
        ),
        "started": sum(1 for row in rows if row.get("started")),
        "goal_start_failures": len(goal_start_failures),
        "hard_residuals": len(hard_residuals),
        "last_goal_available": len(last_goal_rows),
        "dr_ducky_capsules": len(ducky_capsules),
        "dr_ducky_ledger_packets": len(ducky_ledger_rows),
        "temporal_rows": len(temporal_rows),
        "strategy_observations": len(observations),
        "strategy_entries": len(strategy_memory),
        "by_residual_bucket": dict(
            Counter(str(row.get("residual_bucket", "")) for row in rows).most_common()
        ),
        "by_hard_track": dict(
            Counter(str(row.get("hard_track", "")) for row in hard_residuals).most_common()
        ),
        "by_reasoning_gap_family": dict(
            Counter(str(row.get("reasoning_gap_family", "")) for row in hard_residuals).most_common()
        ),
        "by_search_pathology": dict(
            Counter(
                tag
                for row in hard_residuals
                for tag in row.get("search_pathology_tags", [])
            ).most_common()
        ),
        "dr_ducky_summary": ducky_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Theorem-level collection JSONL input(s)")
    parser.add_argument("--output-dir", required=True, help="Directory to write the hard collection bundle")
    parser.add_argument(
        "--min-trace-length",
        type=int,
        default=2,
        help="Minimum trace length for temporal rows",
    )
    parser.add_argument(
        "--min-strategy-support",
        type=int,
        default=3,
        help="Minimum support for mined strategy-memory entries",
    )
    args = parser.parse_args()

    summary = build_hard_collection_bundle(
        inputs=[Path(path) for path in args.inputs],
        output_dir=Path(args.output_dir),
        min_trace_length=args.min_trace_length,
        min_strategy_support=args.min_strategy_support,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
