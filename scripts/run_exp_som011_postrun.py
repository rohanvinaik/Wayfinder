"""Post-run analysis for EXP-SOM-011 paired theorem benchmark.

This script assumes the runtime conditions have finished and emitted:

- `norm_then_close.json`
- `norm_then_close_torch.json`

It turns those theorem-level reports into:

- residual bucket summaries
- follow-on worklists for `compiler_specialist`, `hard_proof_solver`,
  and `theorem_replanner`
- a condition-comparison summary aligned to the hard-proof protocol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_benchmark_residuals import build_residual_report
from scripts.build_hard_proof_dataset import build_hard_proof_dataset


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="runs/exp_som011_paired_2000",
        help="Directory containing paired condition JSON reports",
    )
    parser.add_argument(
        "--theorems",
        action="append",
        default=["data/mathlib_test_2000.jsonl"],
        help="Benchmark theorem JSONL(s) for goal-state metadata",
    )
    parser.add_argument(
        "--baseline-summary",
        default="runs/exp058_decisive_2000/summary.json",
        help="Optional EXP-058 baseline summary for theorem-count comparison",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir (defaults to <run-dir>/postrun)",
    )
    parser.add_argument(
        "--residual-source",
        action="append",
        default=["data/residual_exp058_started.jsonl"],
        help="Optional residual JSONL source(s) with last_goal fields",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "postrun"
    output_dir.mkdir(parents=True, exist_ok=True)

    theorem_paths = [Path(path) for path in args.theorems]
    residual_source_paths = [Path(path) for path in args.residual_source]
    condition_paths = {
        "norm_then_close": run_dir / "norm_then_close.json",
        "norm_then_close_torch": run_dir / "norm_then_close_torch.json",
    }

    reports: dict[str, dict[str, Any]] = {}
    for name, path in condition_paths.items():
        if not path.exists():
            continue
        report = build_residual_report(path, theorem_paths)
        reports[name] = report

        cond_dir = output_dir / name
        cond_dir.mkdir(parents=True, exist_ok=True)
        (cond_dir / "residual_report.json").write_text(json.dumps(report, indent=2))

        for folder_name, key in [
            ("by_residual_bucket", "residual_bucket"),
            ("by_follow_on_stage", "follow_on_stage"),
            ("by_progress_band", "progress_band"),
        ]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for entry in report["details"]:
                grouped.setdefault(str(entry.get(key, "")), []).append(entry)
            target_dir = cond_dir / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for bucket, entries in grouped.items():
                if not bucket:
                    continue
                out_path = target_dir / f"{bucket}.jsonl"
                with out_path.open("w") as handle:
                    for entry in entries:
                        handle.write(json.dumps(entry) + "\n")

        hard_bucket_path = cond_dir / "by_follow_on_stage" / "hard_proof_solver.jsonl"
        if hard_bucket_path.exists():
            build_hard_proof_dataset(
                hard_bucket_path=hard_bucket_path,
                output_dir=cond_dir / "hard_proof_stage",
                residual_source_paths=residual_source_paths,
                condition=name,
            )

    baseline_path = Path(args.baseline_summary)
    baseline = {}
    if baseline_path.exists():
        baseline = _load_json(baseline_path).get("cosine_rw_only", {})

    summary = {
        "run_dir": str(run_dir),
        "baseline": baseline,
        "conditions": {},
    }
    for name, report in reports.items():
        residual = report["residual_structure"]
        summary["conditions"][name] = {
            "raw_success": report["raw_success"],
            "total_theorems": report["total_theorems"],
            "started_theorems": residual["started_theorems"],
            "skipped_start": residual["skipped_start"],
            "progressed_but_unsolved": residual["progressed_but_unsolved"],
            "one_goal_left_failures": residual["one_goal_left_failures"],
            "by_residual_bucket": residual["by_residual_bucket"],
            "by_follow_on_stage": residual["by_follow_on_stage"],
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("EXP-SOM-011 Postrun")
    print("=" * 72)
    if baseline:
        print(
            f"  EXP-058 baseline: {baseline.get('proved')}/{baseline.get('total')} "
            f"success_rate={baseline.get('success_rate')}"
        )
    for name, data in summary["conditions"].items():
        print(
            f"  {name}: {data['raw_success']}/{data['total_theorems']} "
            f"started={data['started_theorems']} skip={data['skipped_start']} "
            f"one_goal_left={data['one_goal_left_failures']}"
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
