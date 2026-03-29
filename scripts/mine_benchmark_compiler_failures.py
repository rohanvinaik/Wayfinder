"""Aggregate and probe benchmark theorem-start/compiler failures.

This mines prior benchmark outputs for architecture-level startability gaps,
instead of treating them as ordinary proof failures. It can optionally replay a
bounded sample of failures through the current theorem-site start path to
measure how many are already fixed by recent compiler-layer patches.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.lean_interface import LeanConfig, LeanKernel


def _iter_failure_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    goal_start_failures = run_dir / "goal_start_failures.jsonl"
    if goal_start_failures.exists():
        with goal_start_failures.open() as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                rows.append(
                    {
                        "run": run_dir.name,
                        "theorem_id": row.get("theorem_id", ""),
                        "file_path": row.get("file_path") or row.get("lean_path") or "",
                        "module": row.get("module", ""),
                        "failure_category": row.get("failure_category", ""),
                        "residual_bucket": row.get("residual_bucket", ""),
                        "goal_state": row.get("goal_state", ""),
                    }
                )
    details = run_dir / "details.jsonl"
    if details.exists():
        with details.open() as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if row.get("started", True):
                    continue
                rows.append(
                    {
                        "run": run_dir.name,
                        "theorem_id": row.get("theorem_id", ""),
                        "file_path": row.get("file_path") or row.get("lean_path") or "",
                        "module": row.get("module", ""),
                        "failure_category": row.get("failure_category", ""),
                        "residual_bucket": row.get("residual_bucket", ""),
                        "goal_state": row.get("goal_state", ""),
                    }
                )
    return rows


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row.get("run", ""), row.get("theorem_id", ""))


def _row_score(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(row.get("file_path"))),
        int(bool(row.get("module"))),
        int(bool(row.get("failure_category")) and row.get("failure_category") != "uncategorized"),
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        incumbent = deduped.get(key)
        if incumbent is None or _row_score(row) > _row_score(incumbent):
            deduped[key] = row
    return sorted(deduped.values(), key=_row_key)


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_metadata = 0
    for row in rows:
        run = row.get("run", "")
        category = row.get("failure_category", "") or "uncategorized"
        by_run[run][category] += 1
        if not row.get("file_path") and not row.get("module"):
            missing_metadata += 1
        if len(examples[category]) < 8:
            examples[category].append(
                {
                    "run": run,
                    "theorem_id": row.get("theorem_id", ""),
                    "file_path": row.get("file_path", ""),
                    "module": row.get("module", ""),
                    "residual_bucket": row.get("residual_bucket", ""),
                }
            )
    return {
        "total_failure_rows": len(rows),
        "rows_missing_file_and_module": missing_metadata,
        "by_run": {run: dict(counter) for run, counter in by_run.items()},
        "examples": examples,
    }


def _probe_failures(
    rows: list[dict[str, Any]],
    project_root: str,
    replay_limit_per_category: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    per_category: Counter[str] = Counter()
    seen_theorems: set[str] = set()
    for row in rows:
        theorem_id = row.get("theorem_id", "")
        category = row.get("failure_category", "") or "uncategorized"
        if not theorem_id or theorem_id in seen_theorems:
            continue
        if per_category[category] >= replay_limit_per_category:
            continue
        selected.append(row)
        per_category[category] += 1
        seen_theorems.add(theorem_id)

    lean = LeanKernel(
        LeanConfig(
            backend="pantograph",
            project_root=project_root,
            imports=["Mathlib"],
        )
    )
    recovered = 0
    by_category: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    try:
        for row in selected:
            theorem_id = row.get("theorem_id", "")
            category = row.get("failure_category", "") or "uncategorized"
            replay = lean.goal_via_file_context(
                theorem_full_name=theorem_id,
                file_path=row.get("file_path", ""),
                prefix_tactics=[],
                project_root=project_root,
                module_hint=row.get("module", ""),
                fallback_goal_pp=row.get("goal_state", ""),
            )
            if replay.success:
                recovered += 1
                by_category[category] += 1
            results.append(
                {
                    "theorem_id": theorem_id,
                    "run": row.get("run", ""),
                    "original_failure_category": category,
                    "recovered": replay.success,
                    "tier_used": replay.tier_used,
                    "replay_feedback": replay.feedback.category if replay.feedback else "",
                    "file_path": row.get("file_path", ""),
                    "module": row.get("module", ""),
                }
            )
    finally:
        lean.close()

    return {
        "probed_rows": len(selected),
        "recovered_rows": recovered,
        "recovered_by_failure_category": dict(by_category),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--project-root", default="/Users/rohanvinaik/Projects/Wayfinder/data/lean_project")
    parser.add_argument("--replay-limit-per-category", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        rows.extend(_iter_failure_rows(Path(run_dir)))

    rows = _dedupe_rows(rows)
    summary = _build_summary(rows)
    if args.replay_limit_per_category > 0:
        summary["patched_replay_audit"] = _probe_failures(
            rows,
            project_root=args.project_root,
            replay_limit_per_category=args.replay_limit_per_category,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
