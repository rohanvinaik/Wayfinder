"""Sprint 2: Build template narrative dataset for teacher models.

Aggregates per-theorem metadata into symbolicized training examples for
template audit, proof-story induction, and sketch teaching.

Input: `data/nav_train_templates.jsonl` (template-annotated nav training data)
Output: `data/template_narrative_train.jsonl`

Each output row is one theorem with:
  - template assignment and move profile
  - proof history summary (subtask sequence, family histogram)
  - trigger signatures across steps
  - context features and startability status

Usage:
    python -m scripts.build_template_narrative_dataset \\
        --input data/nav_train_templates.jsonl \\
        --output data/template_narrative_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _aggregate_theorem(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step rows into a single theorem-level narrative row."""
    first = steps[0]
    theorem_id = first.get("theorem_id", "")

    # Template info (should be consistent across steps)
    template_name = first.get("template_name", "")
    template_move_profile = first.get("template_move_profile", {})

    # Proof history: ordered sequence of families and subtask kinds
    family_sequence: list[str] = []
    family_histogram: Counter[str] = Counter()
    subtask_sequence: list[str] = []
    trigger_signatures: list[dict[str, Any]] = []
    premise_set: set[str] = set()

    for step in sorted(steps, key=lambda s: s.get("step_index", 0)):
        family = step.get("ground_truth_tactic", "")
        if family:
            family_sequence.append(family)
            family_histogram[family] += 1

        # Extract subtask kind from move profile if available
        move_profile = step.get("template_move_profile", {})
        subtask_kind = move_profile.get("dominant_subtask_kind", "")
        if subtask_kind:
            subtask_sequence.append(subtask_kind)

        # Premises used
        for p in step.get("ground_truth_premises", []):
            premise_set.add(p)

        # Trigger signature: compact per-step summary
        if step.get("step_index", 0) == 0 or family:
            trigger_signatures.append(
                {
                    "step": step.get("step_index", 0),
                    "family": family,
                    "subtask_kind": subtask_kind,
                    "has_premises": len(step.get("ground_truth_premises", [])) > 0,
                }
            )

    # Deduplicate consecutive subtask kinds
    deduped_subtasks: list[str] = []
    for sk in subtask_sequence:
        if not deduped_subtasks or deduped_subtasks[-1] != sk:
            deduped_subtasks.append(sk)

    # Namespace prefix
    namespace_prefix = theorem_id.split(".")[0] if "." in theorem_id else "(root)"

    return {
        "theorem_id": theorem_id,
        "namespace_prefix": namespace_prefix,
        "theorem_statement": first.get("goal_state", ""),
        "template_id": template_name,
        "template_move_profile": template_move_profile,
        "proof_history_summary": {
            "subtask_sequence": deduped_subtasks,
            "family_sequence": family_sequence,
            "family_histogram": dict(family_histogram.most_common()),
            "total_steps": len(steps),
            "unique_premises": len(premise_set),
        },
        "trigger_signatures": trigger_signatures,
        "solvable": first.get("solvable", True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/nav_train_templates.jsonl")
    parser.add_argument("--output", default="data/template_narrative_train.jsonl")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("Input not found: %s", args.input)
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Group steps by theorem_id
    theorem_steps: dict[str, list[dict]] = {}
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tid = d.get("theorem_id", d.get("theorem_key", ""))
            theorem_steps.setdefault(tid, []).append(d)

    logger.info("Loaded %d theorems from %s", len(theorem_steps), args.input)

    template_counts: Counter[str] = Counter()
    total_rows = 0

    with open(args.output, "w") as fout:
        for tid, steps in sorted(theorem_steps.items()):
            row = _aggregate_theorem(steps)
            fout.write(json.dumps(row) + "\n")
            total_rows += 1
            template_counts[row["template_id"]] += 1

    print("\n" + "=" * 60)
    print("Template Narrative Dataset")
    print("=" * 60)
    print(f"  Input:     {args.input}")
    print(f"  Theorems:  {total_rows}")
    print("\n  Template distribution:")
    for tpl, cnt in template_counts.most_common():
        print(f"    {tpl:30s}: {cnt:5d} ({100 * cnt / max(total_rows, 1):.1f}%)")
    print(f"\n  Output:    {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
