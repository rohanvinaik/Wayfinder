"""Sprint 2: Mine strategy memory from successful search traces.

Builds a k-line-like symbolic orchestration prior: for each (template,
namespace, goal_shape_bucket, recent_lanes) key, what lane ordering and
family prior worked historically?

Input: benchmark result JSONL files with step_trace (proved theorems only)
Output: `data/strategy_memory.json`

Usage:
    python -m scripts.mine_strategy_memory \\
        --inputs runs/exp049_results/baseline.jsonl \\
        --output data/strategy_memory.json \\
        --min-support 3
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from src.benchmark_residuals import detect_self_application, is_self_application_tactic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _goal_shape_bucket(goal: str) -> str:
    """Cheap goal shape classifier for strategy keys."""
    if not goal:
        return "empty"
    if "∀" in goal and "→" in goal:
        return "forall_implication"
    if "∀" in goal:
        return "forall"
    if "∃" in goal:
        return "exists"
    if "↔" in goal:
        return "iff"
    if "=" in goal and "≤" not in goal and "≥" not in goal:
        return "equality"
    if "≤" in goal or "≥" in goal or "<" in goal or ">" in goal:
        return "inequality"
    if "∧" in goal or "∨" in goal:
        return "connective"
    return "other"


def _recent_lanes_key(recent: list[str]) -> str:
    """Normalize recent lanes into a hashable key."""
    if not recent:
        return "(none)"
    # Last 3 progress lanes, deduplicated consecutive
    deduped: list[str] = []
    for lane in recent[-3:]:
        if not deduped or deduped[-1] != lane:
            deduped.append(lane)
    return "+".join(deduped)


def _extract_strategy_observations(
    theorem_id: str,
    step_trace: list[dict],
    template: str = "",
) -> list[dict[str, Any]]:
    """Extract strategy observations from a proved theorem's trace.

    Each observation records: at a given state, which lane made progress.
    """
    observations: list[dict[str, Any]] = []
    namespace = theorem_id.split(".")[0] if "." in theorem_id else "(root)"

    recent_lanes: list[str] = []
    for entry in step_trace:
        tactic = str(entry.get("tactic", "") or "")
        if is_self_application_tactic(tactic, theorem_id):
            continue
        goal = entry.get("goal_before", "")
        lane = entry.get("lane", "")
        family = entry.get("closing_family", "")
        progress = entry.get("progress", False)

        if progress and lane:
            obs = {
                "namespace": namespace,
                "template": template,
                "goal_shape_bucket": _goal_shape_bucket(goal),
                "recent_lanes_key": _recent_lanes_key(recent_lanes),
                "progress_lane": lane,
                "progress_family": family,
            }
            observations.append(obs)
            recent_lanes.append(lane)

    return observations


def _build_strategy_memory(
    observations: list[dict[str, Any]],
    min_support: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate observations into strategy memory entries."""
    # Group by key
    buckets: dict[tuple, list[dict]] = {}
    for obs in observations:
        key = (
            obs["template"],
            obs["namespace"],
            obs["goal_shape_bucket"],
            obs["recent_lanes_key"],
        )
        buckets.setdefault(key, []).append(obs)

    entries: list[dict[str, Any]] = []
    for key, obs_list in buckets.items():
        if len(obs_list) < min_support:
            continue

        template, namespace, shape_bucket, recent_key = key

        # Aggregate lane preferences
        lane_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        for obs in obs_list:
            lane_counts[obs["progress_lane"]] += 1
            if obs["progress_family"]:
                family_counts[obs["progress_family"]] += 1

        preferred_lanes = [lane for lane, _ in lane_counts.most_common()]
        preferred_families = [fam for fam, _ in family_counts.most_common()]
        total = len(obs_list)
        top_lane_rate = lane_counts.most_common(1)[0][1] / total if lane_counts else 0

        entries.append(
            {
                "key": {
                    "template_id": template,
                    "namespace_prefix": namespace,
                    "goal_shape_bucket": shape_bucket,
                    "recent_lanes": recent_key,
                },
                "value": {
                    "preferred_lane_order": preferred_lanes,
                    "family_prior": preferred_families,
                    "support": total,
                    "top_lane_rate": round(top_lane_rate, 3),
                    "lane_distribution": dict(lane_counts.most_common()),
                },
            }
        )

    # Sort by support descending
    entries.sort(key=lambda e: e["value"]["support"], reverse=True)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Benchmark result JSONL files with step_trace",
    )
    parser.add_argument("--output", default="data/strategy_memory.json")
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument(
        "--templates",
        default="",
        help="Optional nav_train_templates.jsonl for template labels",
    )
    args = parser.parse_args()

    # Load template labels if available
    theorem_templates: dict[str, str] = {}
    if args.templates and Path(args.templates).exists():
        with open(args.templates) as f:
            for line in f:
                d = json.loads(line.strip())
                tid = d.get("theorem_id", "")
                tpl = d.get("template_name", "")
                if tid and tpl:
                    theorem_templates[tid] = tpl
        logger.info("Loaded %d template labels", len(theorem_templates))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_observations: list[dict[str, Any]] = []
    proved_count = 0
    total_count = 0

    for input_path in args.inputs:
        if not Path(input_path).exists():
            logger.warning("Input not found: %s — skipping", input_path)
            continue

        logger.info("Processing %s", input_path)
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                total_count += 1
                if not d.get("success"):
                    continue
                if d.get("honest_success") is False:
                    continue
                if d.get("honest_success") is None and detect_self_application(d):
                    continue

                proved_count += 1
                tid = d.get("theorem_id", "")
                trace = d.get("step_trace", [])
                template = theorem_templates.get(tid, "")

                obs = _extract_strategy_observations(tid, trace, template)
                all_observations.extend(obs)

    logger.info(
        "Extracted %d observations from %d proved theorems (of %d total)",
        len(all_observations),
        proved_count,
        total_count,
    )

    entries = _build_strategy_memory(all_observations, min_support=args.min_support)

    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)

    print("\n" + "=" * 60)
    print("Strategy Memory Miner")
    print("=" * 60)
    print(f"  Proved theorems:    {proved_count}/{total_count}")
    print(f"  Observations:       {len(all_observations)}")
    print(f"  Memory entries:     {len(entries)} (min_support={args.min_support})")
    if entries:
        print(f"  Top entry support:  {entries[0]['value']['support']}")
        print("\n  Top 5 entries:")
        for e in entries[:5]:
            k = e["key"]
            v = e["value"]
            print(
                f"    [{k['goal_shape_bucket']:20s}] "
                f"{k['recent_lanes']:20s} → {v['preferred_lane_order'][0]:20s} "
                f"(n={v['support']}, rate={v['top_lane_rate']:.0%})"
            )
    print(f"\n  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
