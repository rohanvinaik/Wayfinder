"""Analyze proof search traces to diagnose tactic prediction failures.

Reads benchmark results JSONL and produces diagnostic breakdowns:
- Three-lane provenance: automation / bootstrap / learned
- Search budget utilization patterns
- Per-theorem failure analysis with ground-truth comparison

Usage:
    python scripts/analyze_search_traces.py runs/EXP-3.1-init-logic/benchmark_results.jsonl
    python scripts/analyze_search_traces.py results.jsonl --theorems data/init_logic_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_ground_truth(path: Path | None) -> dict[str, dict]:
    """Load ground-truth theorem data keyed by theorem_id."""
    if not path or not path.exists():
        return {}
    gt = {}
    for d in _load_jsonl(path):
        gt[d["theorem_id"]] = d
    return gt


def _print_lane_breakdown(results: list[dict]) -> dict[str, int]:
    """Print three-lane provenance breakdown. Returns lane counts."""
    lane_counts: dict[str, int] = {}
    for r in results:
        lane = r.get("close_lane", "failed" if not r["success"] else "unknown")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

    n = len(results)
    if any(r.get("close_lane") for r in results):
        print("  --- Close lane breakdown ---")
        for lane in ["automation", "structural_core", "solver_bootstrap", "learned", "failed"]:
            count = lane_counts.get(lane, 0)
            if count > 0:
                print(f"    {lane}: {count} ({100 * count / max(n, 1):.1f}%)")
    return lane_counts


def _print_gt_categories(
    proved: list[dict], failed: list[dict], gt: dict[str, dict]
) -> tuple[int, int, int, int]:
    """Print ground-truth category breakdown. Returns (ap, af, np, nf) counts."""
    auto_p = [r for r in proved if gt.get(r["theorem_id"], {}).get("category") == "automation"]
    auto_f = [r for r in failed if gt.get(r["theorem_id"], {}).get("category") == "automation"]
    nav_p = [r for r in proved if gt.get(r["theorem_id"], {}).get("category") == "navigation"]
    nav_f = [r for r in failed if gt.get(r["theorem_id"], {}).get("category") == "navigation"]
    print(
        f"  Automation-category (GT): {len(auto_p)}/{len(auto_p) + len(auto_f)} "
        f"({100 * len(auto_p) / max(len(auto_p) + len(auto_f), 1):.1f}%)"
    )
    print(
        f"  Navigation-category (GT): {len(nav_p)}/{len(nav_p) + len(nav_f)} "
        f"({100 * len(nav_p) / max(len(nav_p) + len(nav_f), 1):.1f}%)"
    )
    return len(auto_p), len(auto_f), len(nav_p), len(nav_f)


def _print_tactic_summary(proved: list[dict]) -> dict[str, int]:
    """Print tactics used in proved theorems. Returns tactic counts."""
    counter: Counter[str] = Counter()
    for r in proved:
        for t in r.get("tactics_used", []):
            base = t.split()[0] if t.strip() else t
            counter[base] += 1
    print("\nTactics in proved theorems:")
    for tactic, count in counter.most_common(10):
        print(f"  {tactic}: {count}")
    return dict(counter)


def _print_failures(failed: list[dict], gt: dict[str, dict]) -> None:
    """Print per-theorem failure details."""
    print(f"\nFailed theorems ({len(failed)}):")
    for r in failed:
        gt_info = gt.get(r["theorem_id"], {})
        gt_proof = gt_info.get("ground_truth_proof", [])
        category = gt_info.get("category", "?")
        print(f"  {r['theorem_id']} [{category}]:")
        print(f"    Attempts: {r['attempts']}, Time: {r['time_s']}s")
        if gt_proof:
            print(f"    Ground truth: {' → '.join(gt_proof)}")
        print(f"    First tactic needed: {gt_info.get('ground_truth_tactic', '?')}")


def _print_lane_sequences(proved: list[dict]) -> None:
    """Print lane sequence distribution for proved theorems."""
    seqs: dict[str, int] = {}
    for r in proved:
        seq = r.get("lane_sequence", "→".join(dict.fromkeys(r.get("close_provenance", []))))
        if seq:
            seqs[seq] = seqs.get(seq, 0) + 1
    if seqs:
        print("\nLane sequences (proved theorems):")
        for seq, count in sorted(seqs.items(), key=lambda x: -x[1]):
            print(f"  {seq}: {count}")


def _print_budget(results: list[dict], proved: list[dict], failed: list[dict]) -> None:
    """Print budget utilization summary."""
    budget_exhausted = sum(1 for r in results if r["attempts"] >= 600)
    single_attempt = sum(1 for r in results if r["attempts"] == 1)
    print("\nBudget utilization:")
    print(f"  Single-attempt success: {single_attempt}")
    print(f"  Budget exhausted (600+): {budget_exhausted}")
    print(
        f"  Mean attempts (proved): {sum(r['attempts'] for r in proved) / max(len(proved), 1):.1f}"
    )
    print(
        f"  Mean attempts (failed): {sum(r['attempts'] for r in failed) / max(len(failed), 1):.1f}"
    )


def analyze(results_path: Path, theorems_path: Path | None = None) -> dict:
    """Analyze benchmark results with optional ground-truth comparison."""
    results = _load_jsonl(results_path)
    gt = _load_ground_truth(theorems_path)

    proved = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("=" * 60)
    print("SEARCH TRACE ANALYSIS")
    print("=" * 60)
    print(
        f"\nOverall: {len(proved)}/{len(results)} proved "
        f"({100 * len(proved) / max(len(results), 1):.1f}%)"
    )

    lane_counts = _print_lane_breakdown(results)
    auto_p, auto_f, nav_p, nav_f = _print_gt_categories(proved, failed, gt)
    tactic_counts = _print_tactic_summary(proved)
    _print_lane_sequences(proved)
    _print_failures(failed, gt)
    _print_budget(results, proved, failed)

    return {
        "total": len(results),
        "proved": len(proved),
        "failed": len(failed),
        "by_close_lane": lane_counts,
        "auto_proved": auto_p,
        "auto_failed": auto_f,
        "nav_proved": nav_p,
        "nav_failed": nav_f,
        "tactics_used": tactic_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search traces")
    parser.add_argument("results", type=Path, help="Benchmark results JSONL")
    parser.add_argument("--theorems", type=Path, default=None, help="Ground-truth theorems JSONL")
    args = parser.parse_args()

    analyze(args.results, args.theorems)


if __name__ == "__main__":
    main()
