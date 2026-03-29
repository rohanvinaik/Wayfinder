"""Sprint 2: Build supervised temporal-controller training data.

Converts per-step search traces into supervised examples for a compact
runtime temporal controller. Each row represents a decision point: given
the current search state, what lane/family will next make progress?

Input: benchmark result JSONL files with `step_trace` (from collect_trace=True).
Output: `data/temporal_train.jsonl`

Derives forward-looking labels from the trace:
  - next_progress_lane: which lane next makes progress (or "" if none)
  - next_progress_family: which tactic family next makes progress
  - next_progress_within_k_steps: how many steps until next progress (capped)
  - eventual_theorem_success: did this theorem eventually get proved?

Usage:
    python -m scripts.build_temporal_dataset \\
        --inputs runs/exp049_results/baseline.jsonl runs/mathlib-cosine-rw/benchmark_results.jsonl \\
        --output data/temporal_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.benchmark_residuals import detect_self_application, is_self_application_tactic
from src.hard_data_tags import sanitize_goal_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MAX_LOOKAHEAD = 20  # cap next_progress_within_k_steps


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


def _build_temporal_rows(
    theorem_id: str,
    step_trace: list[dict],
    theorem_success: bool,
) -> list[dict[str, Any]]:
    """Convert a single theorem's step trace into temporal training rows."""
    rows: list[dict[str, Any]] = []
    if not step_trace:
        return rows

    n = len(step_trace)

    for i, entry in enumerate(step_trace):
        # Find next progress step
        next_lane = ""
        next_family = ""
        steps_to_progress = _MAX_LOOKAHEAD
        for j in range(i + 1, min(i + 1 + _MAX_LOOKAHEAD, n)):
            if step_trace[j].get("progress"):
                next_lane = step_trace[j].get("lane", "")
                next_family = step_trace[j].get("closing_family", "")
                steps_to_progress = j - i
                break

        # Recent lanes (last 5 progress steps before this one)
        recent_lanes: list[str] = []
        recent_families: list[str] = []
        for k in range(i - 1, -1, -1):
            if step_trace[k].get("progress") and step_trace[k].get("lane"):
                recent_lanes.append(step_trace[k]["lane"])
                recent_families.append(step_trace[k].get("closing_family", ""))
                if len(recent_lanes) >= 5:
                    break
        recent_lanes.reverse()
        recent_families.reverse()

        row: dict[str, Any] = {
            "theorem_id": theorem_id,
            "step": entry.get("step", i),
            "goal_state": entry.get("goal_before", ""),
            "open_goals_count": len(entry.get("open_goals_before", [])),
            "closed_goals_count": entry.get("closed_goals_count", 0),
            "current_lane": entry.get("lane", ""),
            "current_family": entry.get("closing_family", ""),
            "current_progress": entry.get("progress", False),
            "recent_lanes": recent_lanes,
            "recent_families": recent_families,
            # TC decision context (if available)
            "phase": entry.get("phase", ""),
            "lane_order": entry.get("lane_order", []),
            "family_prior": entry.get("family_prior", []),
            "escalation_level": entry.get("escalation_level", 0),
            "replan": entry.get("replan", False),
            # Forward-looking labels
            "next_progress_lane": next_lane,
            "next_progress_family": next_family,
            "next_progress_within_k_steps": steps_to_progress,
            "eventual_theorem_success": theorem_success,
        }
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Benchmark result JSONL files with step_trace",
    )
    parser.add_argument("--output", default="data/temporal_train.jsonl")
    parser.add_argument(
        "--min-trace-length",
        type=int,
        default=2,
        help="Skip theorems with fewer than N trace steps",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_theorems = 0
    total_rows = 0
    progress_rows = 0
    success_theorems = 0

    with open(args.output, "w") as fout:
        for input_path in args.inputs:
            if not Path(input_path).exists():
                logger.warning("Input not found: %s — skipping", input_path)
                continue

            logger.info("Processing %s", input_path)
            with open(input_path) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                d = json.loads(line)
                trace = _sanitize_step_trace(
                    str(d.get("theorem_id", "")),
                    d.get("step_trace", []),
                )
                if len(trace) < args.min_trace_length:
                    continue

                theorem_id = d.get("theorem_id", "")
                success = d.get("honest_success")
                if success is None:
                    success = bool(d.get("success")) and not detect_self_application(d)
                total_theorems += 1
                if success:
                    success_theorems += 1

                    rows = _build_temporal_rows(theorem_id, trace, success)
                    for row in rows:
                        fout.write(json.dumps(row) + "\n")
                        total_rows += 1
                        if row["current_progress"]:
                            progress_rows += 1

    print("\n" + "=" * 60)
    print("Temporal Dataset Builder")
    print("=" * 60)
    print(f"  Inputs:         {len(args.inputs)} files")
    print(f"  Theorems:       {total_theorems} ({success_theorems} proved)")
    print(f"  Rows:           {total_rows}")
    print(f"  Progress rows:  {progress_rows} ({100 * progress_rows / max(total_rows, 1):.1f}%)")
    print(f"  Output:         {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
