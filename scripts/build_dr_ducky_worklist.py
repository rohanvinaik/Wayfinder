"""Build Dr_Ducky symbolic packets from live or completed hard-run artifacts.

This script materializes the current Dr_Ducky bundle as a sidecar over the hard
benchmark outputs. It does not modify the running search process; it mines
local residuals into typed goal capsules, proof-shadow ledgers, deterministic
bank priors, projector policies, and symbolic execution contracts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.benchmark_residuals import augment_result_entry
from src.dr_ducky import build_goal_capsule, summarize_capsules

DEFAULT_RESIDUAL_BUCKETS = {
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def build_dr_ducky_worklist(
    *,
    run_dir: Path,
    output_dir: Path,
    residual_buckets: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    residual_buckets = residual_buckets or set(DEFAULT_RESIDUAL_BUCKETS)
    details_path = run_dir / "details.jsonl"
    details = [augment_result_entry(row) for row in _load_jsonl(details_path)]
    filtered = [row for row in details if str(row.get("residual_bucket", "")) in residual_buckets]
    capsules = [build_goal_capsule(row) for row in filtered]
    capsules.sort(key=lambda item: (-item.priority_score, item.specification.theorem_id))
    if limit is not None:
        capsules = capsules[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    capsules_path = output_dir / "dr_ducky_capsules.jsonl"
    ledgers_path = output_dir / "dr_ducky_ledger_packets.jsonl"
    with capsules_path.open("w") as handle:
        for capsule in capsules:
            handle.write(json.dumps(capsule.to_dict()) + "\n")
    with ledgers_path.open("w") as handle:
        for capsule in capsules:
            handle.write(
                json.dumps(
                    {
                        "theorem_id": capsule.specification.theorem_id,
                        "residual_bucket": capsule.specification.residual_bucket,
                        "goal_bucket": capsule.specification.goal_bucket,
                        "ledger_seed": capsule.ledger_seed.to_dict(),
                        "allowed_engines": list(capsule.allowed_engines),
                        "projector_policy": dict(capsule.projector_policy),
                        "execution_budgets": dict(capsule.execution_budgets),
                        "projector_markers": list(capsule.specification.projector_markers),
                        "residual_geometry": dict(capsule.specification.residual_geometry),
                    }
                )
                + "\n"
            )

    summary = summarize_capsules(capsules)
    summary.update(
        {
            "run_dir": str(run_dir),
            "details_path": str(details_path),
            "output_dir": str(output_dir),
            "residual_buckets": sorted(residual_buckets),
            "input_rows": len(details),
            "selected_rows": len(filtered) if limit is None else min(len(filtered), limit),
            "difficulty_band": dict(
                Counter(capsule.specification.difficulty_band for capsule in capsules).most_common()
            ),
            "reasoning_gap_family": dict(
                Counter(capsule.specification.reasoning_gap_family for capsule in capsules).most_common()
            ),
            "ledger_packets": len(capsules),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Hard-run directory containing details.jsonl")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write Dr_Ducky capsule artifacts",
    )
    parser.add_argument(
        "--residual-bucket",
        action="append",
        default=[],
        help="Residual bucket(s) to include. Defaults to the local Dr_Ducky repair regime.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of capsules written")
    args = parser.parse_args()

    summary = build_dr_ducky_worklist(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        residual_buckets=set(args.residual_bucket) if args.residual_bucket else set(DEFAULT_RESIDUAL_BUCKETS),
        limit=args.limit or None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
