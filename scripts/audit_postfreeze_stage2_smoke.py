from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _count_validated_progress(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("started") and row.get("theorem_faithful") and row.get("progressed"):
            count += 1
    return count


def build_report(run_dir: Path, smoke_root: Path) -> dict[str, Any]:
    bundle = run_dir / "bundle"
    local_count = _count_jsonl(bundle / "hard_proof_local.jsonl")
    planner_count = _count_jsonl(bundle / "hard_proof_planner.jsonl")
    all_count = _count_jsonl(bundle / "hard_proof_all.jsonl")
    validated_progress_count = _count_validated_progress(
        bundle / "dr_ducky" / "executor_validation_stratified120_rows.jsonl"
    )

    smoke_runs = {
        "headroom": smoke_root / f"exp_som013a_headroom_full_from_{run_dir.name}",
        "dd014a": smoke_root / f"exp_dd014a_eqsat_full_from_{run_dir.name}",
        "dd014b": smoke_root / f"exp_dd014b_proof_dsl_full_from_{run_dir.name}",
        "dd014c": smoke_root / f"exp_dd014c_relational_full_from_{run_dir.name}",
        "dd014d": smoke_root / f"exp_dd014d_integrated_full_from_{run_dir.name}",
        "dd015_det": smoke_root / f"exp_dd015_integrated_bridge_seeded_from_{run_dir.name}",
    }

    summaries = {}
    for name, root in smoke_runs.items():
        summary = _load_json(root / "summary.json")
        if not summary and name == "headroom":
            summary = {
                "planner_summary_exists": (root / "depth_ladder_planner" / "depth_ladder_summary.json").exists(),
                "local_summary_exists": (root / "depth_ladder_local" / "depth_ladder_summary.json").exists(),
                "oracle_summary_exists": (root / "oracle_gap" / "summary.json").exists(),
            }
        summaries[name] = summary

    return {
        "source_run_dir": str(run_dir),
        "smoke_root": str(smoke_root),
        "bundle_counts": {
            "hard_proof_local": local_count,
            "hard_proof_planner": planner_count,
            "hard_proof_all": all_count,
            "validated_progress_seed_count": validated_progress_count,
        },
        "smoke_summaries": summaries,
        "recommended_full_env": {
            "LOCAL_LIMIT": str(local_count),
            "PLANNER_LIMIT": str(planner_count),
            "ORACLE_LIMIT": str(all_count),
            "DD014_LIMIT": str(all_count),
            "DD015_LIMIT": str(validated_progress_count or 40),
            "DD015_SELECTION_SOURCE": "validated_progress",
            "DD015_ALLOW_UNVALIDATED_BACKFILL": "false",
        },
        "recommended_complete_runs": {
            "headroom": str(run_dir.parent / f"exp_som013a_headroom_complete_from_{run_dir.name}"),
            "dd014a": str(run_dir.parent / f"exp_dd014a_eqsat_complete_from_{run_dir.name}"),
            "dd014b": str(run_dir.parent / f"exp_dd014b_proof_dsl_complete_from_{run_dir.name}"),
            "dd014c": str(run_dir.parent / f"exp_dd014c_relational_complete_from_{run_dir.name}"),
            "dd014d": str(run_dir.parent / f"exp_dd014d_integrated_complete_from_{run_dir.name}"),
            "dd015_det": str(run_dir.parent / f"exp_dd015_integrated_bridge_seeded_complete_from_{run_dir.name}"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    report = build_report(Path(args.run_dir), Path(args.smoke_root))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
