"""Analyze theorem-level benchmark reports as typed residual buckets.

This is the post-run bridge from a benchmark JSON report to the next SoM stage:

- `compiler_specialist`  -> skipped / goal-start failures
- `hard_proof_solver`    -> one-goal-left and small multi-goal residuals
- `theorem_replanner`    -> larger residual trees that need fresh planning

Examples:
    python -m scripts.analyze_benchmark_residuals \
        --report runs/exp_som011_paired_smoke/norm_then_close_torch.json

    python -m scripts.analyze_benchmark_residuals \
        --report runs/exp_som011_paired_2000/norm_then_close_torch.json \
        --theorems data/mathlib_test_2000.jsonl \
        --output-json runs/exp_som011_paired_2000/postrun/norm_then_close_torch.json \
        --bucket-dir runs/exp_som011_paired_2000/postrun/norm_then_close_torch
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_residuals import augment_result_entry, summarize_residual_structure


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_theorem_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open() as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                theorem_id = row.get("theorem_id") or row.get("name")
                if theorem_id:
                    rows[str(theorem_id)] = row
    return rows


def _augment_details(
    details: list[dict[str, Any]],
    theorem_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    theorem_rows = theorem_rows or {}
    for entry in details:
        augmented = augment_result_entry(entry)
        theorem_row = theorem_rows.get(augmented.get("theorem_id", ""))
        if theorem_row:
            augmented["goal_state"] = theorem_row.get("goal_state", theorem_row.get("statement", ""))
            augmented["file_path"] = theorem_row.get("file_path", "")
        enriched.append(augmented)
    return enriched


def _example_map(
    entries: list[dict[str, Any]],
    key: str,
    limit: int = 8,
) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        bucket = str(entry.get(key, ""))
        if bucket and len(examples[bucket]) < limit:
            examples[bucket].append(str(entry.get("theorem_id", "")))
    return dict(examples)


def _write_bucket_files(entries: list[dict[str, Any]], bucket_dir: Path) -> None:
    bucket_dir.mkdir(parents=True, exist_ok=True)

    grouped_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_progress: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        grouped_by_bucket[str(entry.get("residual_bucket", ""))].append(entry)
        grouped_by_stage[str(entry.get("follow_on_stage", ""))].append(entry)
        grouped_by_progress[str(entry.get("progress_band", ""))].append(entry)

    for folder_name, grouped in [
        ("by_residual_bucket", grouped_by_bucket),
        ("by_follow_on_stage", grouped_by_stage),
        ("by_progress_band", grouped_by_progress),
    ]:
        out_dir = bucket_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, grouped_entries in grouped.items():
            if not name:
                continue
            path = out_dir / f"{name}.jsonl"
            with path.open("w") as handle:
                for entry in grouped_entries:
                    handle.write(json.dumps(entry) + "\n")


def build_residual_report(
    report_path: Path,
    theorem_paths: list[Path] | None = None,
) -> dict[str, Any]:
    report = _load_json(report_path)
    theorem_rows = _load_theorem_rows(theorem_paths or []) if theorem_paths else {}
    details = _augment_details(report.get("details", []), theorem_rows)
    summary = summarize_residual_structure(details)

    failed = [entry for entry in details if not bool(entry.get("success"))]
    failed_lane_sequences = Counter(
        str(entry.get("lane_sequence", "")) for entry in failed if entry.get("lane_sequence")
    )

    return {
        "report_path": str(report_path),
        "raw_success": report.get("benchmark", {}).get("raw_success", 0),
        "honest_success": sum(1 for entry in details if bool(entry.get("honest_success"))),
        "self_application_successes": sum(
            1 for entry in details if bool(entry.get("self_application_detected"))
        ),
        "total_theorems": report.get("benchmark", {}).get("total_theorems", len(details)),
        "residual_structure": summary,
        "example_theorems": {
            "by_residual_bucket": _example_map(details, "residual_bucket"),
            "by_follow_on_stage": _example_map(details, "follow_on_stage"),
            "by_progress_band": _example_map(details, "progress_band"),
        },
        "failed_lane_sequences": dict(failed_lane_sequences.most_common(20)),
        "details": details,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _print_summary(report: dict[str, Any]) -> None:
    residual = report["residual_structure"]
    total = report["total_theorems"]
    raw = report["raw_success"]
    honest = report.get("honest_success", raw)
    print("=" * 72)
    print("Benchmark Residual Report")
    print("=" * 72)
    print(f"  Theorems: {total}")
    print(f"  Raw success: {raw}/{total}")
    if report.get("self_application_successes", 0):
        print(f"  Honest success: {honest}/{total}")
        print(f"  Self-application successes: {report['self_application_successes']}")
    print(
        f"  Started: {residual['started_theorems']}/{total} "
        f"(skip={residual['skipped_start']})"
    )
    print(f"  Progressed but unsolved: {residual['progressed_but_unsolved']}")
    print(f"  One-goal-left failures: {residual['one_goal_left_failures']}")
    print("\nResidual buckets:")
    for bucket, count in residual["by_residual_bucket"].items():
        print(f"  {bucket:28s} {count:5d}")
    print("\nFollow-on stages:")
    for stage, count in residual["by_follow_on_stage"].items():
        print(f"  {stage:28s} {count:5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Benchmark JSON report")
    parser.add_argument(
        "--theorems",
        action="append",
        default=[],
        help="Optional theorem JSONL file(s) to reattach goal_state/file_path metadata",
    )
    parser.add_argument("--output-json", help="Optional JSON output path")
    parser.add_argument("--bucket-dir", help="Optional directory for per-bucket JSONL files")
    args = parser.parse_args()

    report = build_residual_report(
        Path(args.report),
        [Path(path) for path in args.theorems],
    )
    _print_summary(report)

    if args.output_json:
        _write_json(Path(args.output_json), report)
        print(f"\nWrote {args.output_json}")
    if args.bucket_dir:
        _write_bucket_files(report["details"], Path(args.bucket_dir))
        print(f"Wrote bucket files to {args.bucket_dir}")


if __name__ == "__main__":
    main()
