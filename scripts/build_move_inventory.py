"""Build a compact controller move inventory from SubtaskIR-annotated data.

Unlike `mine_move_schemas.py`, which keeps fine-grained trigger signatures,
this script intentionally coarsens the space for planning use:

  family -> subtask_kind -> common target heads / trigger kinds / examples

Primary use:
  - define the first explicit move inventory for apply/simp planning
  - keep only supported, repeatable move types

Usage:
    python -m scripts.build_move_inventory \
        --input data/canonical/subtask_train.jsonl \
        --families apply,simp \
        --min-support 25 \
        --output runs/move_inventory.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_row(row: dict) -> dict:
    goal_shape = row.get("goal_shape_ir", {})
    trigger = row.get("trigger_profile_ir", {})
    subtask = row.get("subtask_ir", {})
    return {
        "family": row.get("family", subtask.get("family", "")),
        "subtask_kind": row.get("subtask_kind", subtask.get("kind", "")),
        "subtask_summary": row.get("subtask_summary", subtask.get("summary", "")),
        "expected_effect": row.get("expected_effect", subtask.get("expected_effect", "")),
        "goal_target_head": row.get("goal_target_head", goal_shape.get("target_head", "")),
        "trigger_profile_ir": trigger,
        "tactic_text": row.get("tactic_text", ""),
        "theorem_full_name": row.get("theorem_full_name", ""),
        "step_index": row.get("step_index", 0),
        "primary_premise": row.get("primary_premise", subtask.get("primary_premise", "")),
    }


def build_inventory(
    rows: list[dict],
    families: set[str],
    min_support: int,
) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for raw in rows:
        row = _normalize_row(raw)
        family = row["family"]
        if families and family not in families:
            continue
        grouped[(family, row["subtask_kind"])].append(row)

    inventory: dict[str, list[dict]] = defaultdict(list)
    for (family, subtask_kind), items in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])
    ):
        if len(items) < min_support:
            continue
        target_heads = Counter(item["goal_target_head"] for item in items if item["goal_target_head"])
        trigger_kinds = Counter()
        trigger_values = Counter()
        premises = Counter()
        examples = []
        for item in items:
            trigger = item.get("trigger_profile_ir", {})
            for feature in trigger.get("features", []):
                kind = feature.get("kind", "")
                value = feature.get("value", "")
                if kind:
                    trigger_kinds[kind] += 1
                    trigger_values[f"{kind}={value}"] += 1
            if item["primary_premise"]:
                premises[item["primary_premise"]] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "theorem_full_name": item["theorem_full_name"],
                        "step_index": item["step_index"],
                        "tactic_text": item["tactic_text"],
                    }
                )

        inventory[family].append(
            {
                "subtask_kind": subtask_kind,
                "support": len(items),
                "summary": items[0]["subtask_summary"],
                "expected_effect": items[0]["expected_effect"],
                "top_target_heads": target_heads.most_common(10),
                "common_trigger_kinds": trigger_kinds.most_common(10),
                "common_trigger_values": trigger_values.most_common(10),
                "representative_premises": premises.most_common(10),
                "examples": examples,
            }
        )

    return {
        "families": dict(inventory),
        "min_support": min_support,
        "selected_families": sorted(families),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact move inventory from SubtaskIR data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/canonical/subtask_train.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--families",
        default="apply,simp",
        help="Comma-separated family filter",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=25,
        help="Minimum support to keep a move type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/move_inventory.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    families = {family.strip() for family in args.families.split(",") if family.strip()}
    rows = _load_jsonl(args.input)
    inventory = build_inventory(rows, families, args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(inventory, f, indent=2)

    print("Move Inventory")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Families: {', '.join(sorted(families))}")
    for family, subtasks in inventory["families"].items():
        print(f"  {family}: {len(subtasks)} move types")
        for subtask in subtasks[:5]:
            print(
                f"    {subtask['subtask_kind']}: support={subtask['support']} "
                f"heads={[head for head, _ in subtask['top_target_heads'][:3]]}"
            )


if __name__ == "__main__":
    main()
