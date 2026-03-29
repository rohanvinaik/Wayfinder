"""Audit a live or completed hard-run directory.

This script reads the hard-run JSONL artifacts directly and produces the
triage views that matter during a long-running evaluation:

- raw vs honest success
- top start-failure families
- top residual families
- top search pathology tags
- likely self-replay closures
- best partial-progress theorems for follow-on replanning

Example:
    python -m scripts.audit_hard_run \
        --run-dir runs/exp_som012_hard_eval_r2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_residuals import augment_result_entry, detect_self_application


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _likely_self_replay_rows(rows: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    suspects: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("success"):
            continue
        if detect_self_application(row):
            suspects.append(
                {
                    "theorem_id": row.get("theorem_id", ""),
                    "final_closer": row.get("final_closer", ""),
                    "lane_sequence": row.get("lane_sequence", ""),
                    "attempts": row.get("attempts", 0),
                    "proof_steps": row.get("proof_steps", 0),
                }
            )
    return suspects[:limit]


def _best_partials(rows: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    partials = [
        row
        for row in rows
        if not row.get("success") and row.get("progress_band") in {"near_miss", "partial_progress"}
    ]
    partials.sort(
        key=lambda row: (
            -int(row.get("goals_closed", 0) or 0),
            int(row.get("goals_remaining", 10**9) or 10**9),
            int(row.get("attempts", 10**9) or 10**9),
            str(row.get("theorem_id", "")),
        )
    )
    out: list[dict[str, Any]] = []
    for row in partials[:limit]:
        out.append(
            {
                "theorem_id": row.get("theorem_id", ""),
                "difficulty_band": row.get("difficulty_band", ""),
                "goals_closed": row.get("goals_closed", 0),
                "goals_remaining": row.get("goals_remaining", 0),
                "attempts": row.get("attempts", 0),
                "residual_bucket": row.get("residual_bucket", ""),
                "reasoning_gap_family": row.get("reasoning_gap_family", ""),
                "last_goal_bucket": row.get("last_goal_bucket", ""),
                "last_goal": row.get("last_goal", ""),
                "search_pathology_tags": row.get("search_pathology_tags", []),
            }
        )
    return out


def build_hard_run_audit(run_dir: Path) -> dict[str, Any]:
    details_path = run_dir / "details.jsonl"
    start_failures_path = run_dir / "goal_start_failures.jsonl"
    details = [augment_result_entry(row) for row in _load_jsonl(details_path)]
    start_failures = _load_jsonl(start_failures_path)

    raw_successes = [row for row in details if row.get("success")]
    honest_successes = [row for row in raw_successes if not row.get("self_application_detected")]
    partials = [
        row
        for row in details
        if not row.get("success") and row.get("progress_band") in {"near_miss", "partial_progress"}
    ]

    residual_buckets = Counter(str(row.get("residual_bucket", "")) for row in details if row.get("residual_bucket"))
    progress_bands = Counter(str(row.get("progress_band", "")) for row in details if row.get("progress_band"))
    reasoning_gaps = Counter(str(row.get("reasoning_gap_family", "")) for row in details if row.get("reasoning_gap_family"))
    pathologies = Counter(tag for row in partials for tag in (row.get("search_pathology_tags") or []))
    last_goal_buckets = Counter(str(row.get("last_goal_bucket", "")) for row in partials if row.get("last_goal_bucket"))
    success_lane_sequences = Counter(str(row.get("lane_sequence", "")) for row in raw_successes if row.get("lane_sequence"))

    start_failure_families = Counter(
        str(row.get("start_failure_family", "")) for row in start_failures if row.get("start_failure_family")
    )
    start_failure_categories = Counter(
        str(row.get("failure_category", "")) for row in start_failures if row.get("failure_category")
    )
    unsupported_context_kinds = Counter(
        kind for row in start_failures for kind in (row.get("context_unsupported_kinds") or [])
    )
    start_failure_modules = Counter(str(row.get("module", "")) for row in start_failures if row.get("module"))

    return {
        "run_dir": str(run_dir),
        "details_path": str(details_path),
        "goal_start_failures_path": str(start_failures_path),
        "processed": len(details),
        "raw_success": len(raw_successes),
        "honest_success": len(honest_successes),
        "self_application_successes": len(raw_successes) - len(honest_successes),
        "partial_progress": len(partials),
        "counts": {
            "residual_buckets": dict(residual_buckets.most_common()),
            "progress_bands": dict(progress_bands.most_common()),
            "reasoning_gap_families": dict(reasoning_gaps.most_common()),
            "search_pathology_tags": dict(pathologies.most_common()),
            "partial_last_goal_buckets": dict(last_goal_buckets.most_common()),
            "success_lane_sequences": dict(success_lane_sequences.most_common(20)),
        },
        "start_failures": {
            "count": len(start_failures),
            "families": dict(start_failure_families.most_common()),
            "categories": dict(start_failure_categories.most_common()),
            "unsupported_context_kinds": dict(unsupported_context_kinds.most_common()),
            "modules": dict(start_failure_modules.most_common(20)),
        },
        "likely_self_replay_rows": _likely_self_replay_rows(details),
        "best_partials": _best_partials(details),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_counter(counter: dict[str, int], limit: int = 10) -> None:
    for idx, (name, count) in enumerate(counter.items()):
        if idx >= limit:
            break
        print(f"  {name:36s} {count:5d}")


def _print_summary(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("Hard Run Audit")
    print("=" * 72)
    print(f"  Run dir: {report['run_dir']}")
    print(f"  Processed: {report['processed']}")
    print(f"  Raw success: {report['raw_success']}")
    print(f"  Honest success: {report['honest_success']}")
    print(f"  Self-application successes: {report['self_application_successes']}")
    print(f"  Partial-progress rows: {report['partial_progress']}")

    _print_section("Residual Buckets")
    _print_counter(report["counts"]["residual_buckets"])

    _print_section("Reasoning Gap Families")
    _print_counter(report["counts"]["reasoning_gap_families"])

    _print_section("Search Pathology Tags")
    _print_counter(report["counts"]["search_pathology_tags"])

    _print_section("Start Failure Families")
    _print_counter(report["start_failures"]["families"])

    _print_section("Unsupported Context Kinds")
    _print_counter(report["start_failures"]["unsupported_context_kinds"])

    _print_section("Likely Self-Replay Rows")
    for row in report["likely_self_replay_rows"][:10]:
        print(f"  {row['theorem_id']}: {row['final_closer']}")

    _print_section("Best Partials")
    for row in report["best_partials"][:10]:
        print(
            "  "
            f"{row['theorem_id']} | closed={row['goals_closed']} remaining={row['goals_remaining']} "
            f"| {row['reasoning_gap_family']} | {row['last_goal_bucket']}: {row['last_goal']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Hard-run directory containing JSONL artifacts")
    parser.add_argument("--output-json", help="Optional JSON output path")
    args = parser.parse_args()

    report = build_hard_run_audit(Path(args.run_dir))
    _print_summary(report)

    if args.output_json:
        _write_json(Path(args.output_json), report)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
