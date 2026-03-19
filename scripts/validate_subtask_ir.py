"""Validate SubtaskIR/trigger annotations on canonical or projected datasets.

Checks:
  - required fields present
  - family/subtask consistency
  - trigger profiles populated
  - family-specific trigger invariants for modeled local families

Usage:
    python -m scripts.validate_subtask_ir \
        --input data/canonical/canonical_residual_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.subtask_ir import derive_move_metadata


REQUIRED_TOP_LEVEL = {"family", "goal_state_before", "tactic_text", "step_index"}


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_annotation(record: dict) -> tuple[dict, dict, dict]:
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


def validate(records: list[dict]) -> dict[str, int]:
    stats: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    subtask_counts: Counter[str] = Counter()

    for record in records:
        stats["rows"] += 1
        missing = REQUIRED_TOP_LEVEL - set(record)
        if missing:
            stats["missing_required"] += 1
            continue

        goal_shape, trigger, subtask = _ensure_annotation(record)
        family = record.get("family", "other")

        if trigger.get("family") != family:
            stats["family_mismatch"] += 1
        if subtask.get("family") != family:
            stats["subtask_family_mismatch"] += 1
        if not trigger.get("features"):
            stats["empty_trigger_profile"] += 1

        features = trigger.get("features", [])
        feature_kinds = {feature.get("kind", "") for feature in features}
        for kind in feature_kinds:
            if kind:
                trigger_counts[kind] += 1

        kind = subtask.get("kind", "")
        if kind:
            subtask_counts[kind] += 1

        if family == "rw":
            if "rewrite_count" not in feature_kinds:
                stats["rw_missing_rewrite_count"] += 1
            if "direction_prior" not in feature_kinds:
                stats["rw_missing_direction"] += 1
        if family in {"apply", "exact", "refine"} and subtask.get("kind") == "unmodeled":
            stats["term_family_unmodeled"] += 1
        if family in {"simp", "simpa"} and "simplify" not in subtask.get("kind", ""):
            stats["simp_bad_subtask"] += 1
        if goal_shape.get("goal_count", 0) <= 0:
            stats["bad_goal_count"] += 1

    print("SubtaskIR Validation")
    print(f"  Rows: {stats['rows']}")
    print("\n  Errors / warnings:")
    for key, value in sorted(stats.items()):
        if key == "rows":
            continue
        print(f"    {key}: {value}")

    print("\n  Trigger coverage:")
    for kind, count in trigger_counts.most_common(12):
        print(f"    {kind}: {count}")

    print("\n  Subtask coverage:")
    for kind, count in subtask_counts.most_common(12):
        print(f"    {kind}: {count}")

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SubtaskIR annotations")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/canonical/canonical_residual_eval.jsonl"),
        help="Input JSONL path",
    )
    args = parser.parse_args()
    validate(_load_jsonl(args.input))


if __name__ == "__main__":
    main()
