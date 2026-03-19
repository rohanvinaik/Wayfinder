"""Build a compact SubtaskIR training/eval dataset from canonical local data.

This projects the richer canonical records down to the fields needed for:
  - controller training
  - move-taxonomy audits
  - family-specific trigger analysis

Usage:
    python -m scripts.build_subtask_training_data \
        --input data/canonical/canonical_residual_train.jsonl \
        --output data/canonical/subtask_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.subtask_ir import derive_move_metadata


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _project_record(record: dict) -> dict:
    goal_shape = record.get("goal_shape_ir")
    trigger = record.get("trigger_profile_ir")
    subtask = record.get("subtask_ir")
    if (
        goal_shape is None
        or trigger is None
        or subtask is None
        or not trigger.get("features")
    ):
        goal_shape_obj, trigger_obj, subtask_obj = derive_move_metadata(
            goal_state=record.get("goal_state_before", ""),
            tactic_text=record.get("tactic_text", ""),
            family=record.get("family", "other"),
            canonical_action_ir=record.get("canonical_action_ir", ""),
            annotated_premise=record.get("annotated_premise", ""),
            step_index=int(record.get("step_index", 0)),
            prefix_tactics=record.get("prefix_tactics", []),
        )
        goal_shape = goal_shape_obj.to_dict()
        trigger = trigger_obj.to_dict()
        subtask = subtask_obj.to_dict()

    features = trigger.get("features", [])
    return {
        "theorem_full_name": record.get("theorem_full_name", ""),
        "file_path": record.get("file_path", ""),
        "step_index": int(record.get("step_index", 0)),
        "family": record.get("family", "other"),
        "goal_state_before": record.get("goal_state_before", ""),
        "canonical_action_ir": record.get("canonical_action_ir", ""),
        "annotated_premise": record.get("annotated_premise", ""),
        "prefix_len": len(record.get("prefix_tactics", [])),
        "subtask_kind": subtask.get("kind", ""),
        "subtask_summary": subtask.get("summary", ""),
        "expected_effect": subtask.get("expected_effect", ""),
        "primary_premise": subtask.get("primary_premise", ""),
        "local_inputs": subtask.get("local_inputs", []),
        "trigger_kinds": [f.get("kind", "") for f in features],
        "trigger_values": {f.get("kind", ""): f.get("value", "") for f in features},
        "goal_target_head": goal_shape.get("target_head", ""),
        "goal_local_count": len(goal_shape.get("local_names", [])),
        "goal_shape_ir": goal_shape,
        "trigger_profile_ir": trigger,
        "subtask_ir": subtask,
    }


def build(input_path: Path, output_path: Path) -> None:
    records = _load_jsonl(input_path)
    projected = [_project_record(record) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in projected:
            f.write(json.dumps(row) + "\n")

    family_counts = Counter(row["family"] for row in projected)
    subtask_counts = Counter(row["subtask_kind"] for row in projected)

    print("SubtaskIR Dataset")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Rows:   {len(projected)}")
    print("\n  Families:")
    for family, count in family_counts.most_common():
        print(f"    {family}: {count}")
    print("\n  Subtasks:")
    for kind, count in subtask_counts.most_common(12):
        print(f"    {kind}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact SubtaskIR dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/canonical/canonical_residual_train.jsonl"),
        help="Canonical JSONL with subtask annotations or raw canonical fields",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/canonical/subtask_train.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
