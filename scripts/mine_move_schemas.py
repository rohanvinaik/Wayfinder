"""Mine reusable move schemas from canonical local-execution data.

This is a lightweight proof-schema miner over SubtaskIR/trigger annotations.
It aggregates successful proof steps into controller-facing move templates:

  (family, subtask_kind, trigger_signature) -> support, examples, common effect

The output is intended as:
  - a seed library for future controller rules
  - an audit surface for "motivated move" coverage
  - a compact classical-IR artifact derived from solved traces

Usage:
    python -m scripts.mine_move_schemas \
        --input data/canonical/canonical_residual_train.jsonl \
        --min-support 10 \
        --output runs/move_schemas.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.subtask_ir import derive_move_metadata


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _annotation(record: dict) -> tuple[dict, dict, dict]:
    goal_shape = record.get("goal_shape_ir")
    trigger = record.get("trigger_profile_ir")
    subtask = record.get("subtask_ir")
    if goal_shape and trigger and subtask and trigger.get("features"):
        return goal_shape, trigger, subtask
    goal_shape_obj, trigger_obj, subtask_obj = derive_move_metadata(
        goal_state=record.get("goal_state_before", ""),
        tactic_text=record.get("tactic_text", ""),
        family=record.get("family", "other"),
        canonical_action_ir=record.get("canonical_action_ir", ""),
        annotated_premise=record.get("annotated_premise", ""),
        step_index=int(record.get("step_index", 0)),
        prefix_tactics=record.get("prefix_tactics", []),
    )
    return goal_shape_obj.to_dict(), trigger_obj.to_dict(), subtask_obj.to_dict()


def _trigger_signature(trigger: dict) -> tuple[str, ...]:
    pairs = []
    for feature in trigger.get("features", []):
        kind = feature.get("kind", "")
        value = feature.get("value", "")
        if kind:
            pairs.append(f"{kind}={value}")
    return tuple(sorted(pairs))


def mine(records: list[dict], min_support: int = 1) -> list[dict]:
    buckets: dict[tuple[str, str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for record in records:
        _, trigger, subtask = _annotation(record)
        key = (
            subtask.get("family", record.get("family", "other")),
            subtask.get("kind", "unmodeled"),
            _trigger_signature(trigger),
        )
        buckets[key].append(record)

    schemas: list[dict] = []
    for (family, subtask_kind, signature), rows in sorted(
        buckets.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])
    ):
        if len(rows) < min_support:
            continue
        examples = []
        premises = []
        for row in rows[:5]:
            examples.append(
                {
                    "theorem_full_name": row.get("theorem_full_name", ""),
                    "step_index": row.get("step_index", 0),
                    "tactic_text": row.get("tactic_text", ""),
                }
            )
            premise = row.get("annotated_premise", "")
            if premise and premise not in premises:
                premises.append(premise)

        schemas.append(
            {
                "family": family,
                "subtask_kind": subtask_kind,
                "trigger_signature": list(signature),
                "support": len(rows),
                "representative_premises": premises[:5],
                "examples": examples,
            }
        )
    return schemas


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine move schemas from SubtaskIR data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/canonical/canonical_residual_train.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/move_schemas.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Discard schemas with support below this threshold",
    )
    args = parser.parse_args()

    schemas = mine(_load_jsonl(args.input), min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(schemas, f, indent=2)

    print("Move Schema Miner")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Schemas: {len(schemas)}")
    for schema in schemas[:10]:
        print(
            f"    {schema['family']}/{schema['subtask_kind']}: "
            f"{schema['support']}  [{', '.join(schema['trigger_signature'][:3])}]"
        )


if __name__ == "__main__":
    main()
