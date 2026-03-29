"""Build staged hard-proof datasets from a postrun hard-bucket worklist.

This is the bridge from EXP-SOM-011 postrun outputs to the staged EXP-SOM-012
hard-proof program. It takes the typed `hard_proof_solver` worklist and
materializes:

- `last_goal_residuals.jsonl`
- `hard_proof_local.jsonl`
- `hard_proof_planner.jsonl`
- `summary.json`

When an older residual dataset already contains `last_goal` for a theorem
(`data/residual_exp058_started.jsonl`, for example), this script reattaches it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

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


def _load_residual_sources(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            theorem_id = row.get("theorem_id")
            if theorem_id:
                merged[str(theorem_id)] = row
    return merged


def _hard_track(bucket: str) -> str:
    if bucket in LOCAL_BUCKETS:
        return "hard_proof_local"
    if bucket in PLANNER_BUCKETS:
        return "hard_proof_planner"
    return "hard_proof_other"


def _best_last_goal(entry: dict[str, Any], residual_sources: dict[str, dict[str, Any]]) -> str:
    theorem_id = str(entry.get("theorem_id", ""))
    source_row = residual_sources.get(theorem_id, {})
    last_goal = source_row.get("last_goal")
    if isinstance(last_goal, str) and last_goal.strip():
        return last_goal
    goal_state = entry.get("goal_state")
    if isinstance(goal_state, str) and goal_state.strip():
        return goal_state
    return ""


def _augment_entry(
    entry: dict[str, Any],
    residual_sources: dict[str, dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    theorem_id = str(entry.get("theorem_id", ""))
    source_row = residual_sources.get(theorem_id, {})
    residual_bucket = str(entry.get("residual_bucket", ""))
    last_goal = _best_last_goal(entry, residual_sources)

    out = dict(entry)
    out["condition"] = condition
    out["hard_track"] = _hard_track(residual_bucket)
    out["last_goal"] = last_goal
    out["last_goal_available"] = bool(last_goal)
    if source_row:
        out["residual_source_type"] = source_row.get("residual_type", "")
        out["residual_source_attempts"] = source_row.get("attempts", 0)
        out["residual_source_remaining_goals"] = source_row.get("remaining_goals", 0)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_hard_proof_dataset(
    hard_bucket_path: Path,
    output_dir: Path,
    residual_source_paths: list[Path] | None = None,
    condition: str = "",
) -> dict[str, Any]:
    residual_sources = _load_residual_sources(residual_source_paths or [])
    rows = _load_jsonl(hard_bucket_path)
    augmented = [_augment_entry(row, residual_sources, condition=condition) for row in rows]

    local_rows = [row for row in augmented if row["hard_track"] == "hard_proof_local"]
    planner_rows = [row for row in augmented if row["hard_track"] == "hard_proof_planner"]
    last_goal_rows = [row for row in augmented if row["last_goal_available"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "hard_proof_all.jsonl", augmented)
    _write_jsonl(output_dir / "last_goal_residuals.jsonl", last_goal_rows)
    _write_jsonl(output_dir / "hard_proof_local.jsonl", local_rows)
    _write_jsonl(output_dir / "hard_proof_planner.jsonl", planner_rows)

    summary = {
        "input_path": str(hard_bucket_path),
        "condition": condition,
        "total": len(augmented),
        "last_goal_available": len(last_goal_rows),
        "last_goal_coverage": round(len(last_goal_rows) / max(len(augmented), 1), 4),
        "by_residual_bucket": dict(Counter(str(row.get("residual_bucket", "")) for row in augmented)),
        "by_hard_track": dict(Counter(str(row.get("hard_track", "")) for row in augmented)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hard-bucket",
        required=True,
        help="Path to postrun by_follow_on_stage/hard_proof_solver.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write staged hard-proof datasets",
    )
    parser.add_argument(
        "--residual-source",
        action="append",
        default=[],
        help="Optional residual JSONL source(s) that already contain last_goal fields",
    )
    parser.add_argument(
        "--condition",
        default="",
        help="Optional condition label to stamp onto all output rows",
    )
    args = parser.parse_args()

    summary = build_hard_proof_dataset(
        Path(args.hard_bucket),
        Path(args.output_dir),
        residual_source_paths=[Path(path) for path in args.residual_source],
        condition=args.condition,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
