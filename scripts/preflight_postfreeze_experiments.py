"""Preflight checks for post-freeze Wayfinder experiments.

These checks enforce the repo's run taxonomy:
- deterministic artifacts live under the frozen source run's bundle
- new experiments consume that bundle from sibling run directories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _required_paths(run_dir: Path, experiment: str) -> list[Path]:
    bundle_dir = run_dir / "bundle"
    if experiment == "som013a":
        return [
            bundle_dir / "hard_proof_planner.jsonl",
            bundle_dir / "hard_proof_local.jsonl",
            bundle_dir / "hard_proof_all.jsonl",
            bundle_dir / "hard_resolution_layer" / "resolution_packets.jsonl",
        ]
    if experiment in {"dd014a", "dd014b", "dd014c", "dd014d"}:
        return [
            run_dir / "details.jsonl",
            bundle_dir / "dr_ducky" / "summary.json",
            bundle_dir / "dr_ducky" / "dr_ducky_capsules.jsonl",
            bundle_dir / "dr_ducky" / "dr_ducky_ledger_packets.jsonl",
            bundle_dir / "hard_resolution_layer" / "resolution_packets.jsonl",
        ]
    if experiment == "dd015":
        return [
            run_dir / "details.jsonl",
            bundle_dir / "dr_ducky" / "summary.json",
            bundle_dir / "dr_ducky" / "dr_ducky_ledger_packets.jsonl",
            bundle_dir / "hard_resolution_layer" / "hard_som_packets.jsonl",
            bundle_dir / "second_order_som" / "second_order_packets.jsonl",
        ]
    if experiment == "som013c":
        return [
            bundle_dir / "second_order_som" / "second_order_packets.jsonl",
        ]
    if experiment == "som013d":
        return [
            bundle_dir / "second_order_som" / "features" / "train.npz",
            bundle_dir / "second_order_som" / "features" / "eval.npz",
            bundle_dir / "second_order_som" / "features" / "metadata.json",
        ]
    raise ValueError(f"unsupported experiment: {experiment}")


def _run_preflight(run_dir: Path, experiment: str) -> dict[str, Any]:
    required = _required_paths(run_dir, experiment)
    checks = [
        {
            "path": str(path),
            "exists": path.exists(),
        }
        for path in required
    ]
    missing = [item["path"] for item in checks if not item["exists"]]
    return {
        "run_dir": str(run_dir),
        "experiment": experiment,
        "ok": not missing,
        "required_count": len(required),
        "checks": checks,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=("som013a", "som013c", "som013d", "dd014a", "dd014b", "dd014c", "dd014d", "dd015"),
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    payload = _run_preflight(Path(args.run_dir), args.experiment)
    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    print(rendered)
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
