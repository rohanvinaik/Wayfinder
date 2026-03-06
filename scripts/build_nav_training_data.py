"""
Standalone worker: convert proof steps to NavigationalExample JSONL.

ModelAtlas worker pattern: reads JSONL (entity records from extract_proof_network),
writes JSONL (NavigationalExample records), zero Wayfinder src/ imports,
supports --resume. Deployable on CPU workers via parsync.

For each proof step in a theorem:
  - Goal state text → input
  - Tactic → 6-bank direction vector (from tactic_maps)
  - Goal + tactic + premises → ground-truth anchors
  - Remaining steps → progress label
  - Previously closed goals → proof history

Input:  data/proof_network_entities.jsonl (from extract_proof_network.py)
Output: data/nav_training.jsonl (NavigationalExample per proof step)

Usage:
  python scripts/build_nav_training_data.py --input data/proof_network_entities.jsonl \
      --output data/nav_training.jsonl [--resume] [--shard 0:2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.tactic_maps import DEFAULT_DIRECTION, TACTIC_ANCHORS, TACTIC_DIRECTIONS

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _build_nav_examples(entity: dict) -> list[dict]:
    """Convert one entity record into NavigationalExample dicts, one per step."""
    theorem_id = entity["theorem_id"]
    goal_states = entity.get("goal_states", [])
    tactic_names = entity.get("tactic_names", [])
    tactic_directions = entity.get("tactic_directions", [])
    premises = entity.get("premises", [])
    anchors = entity.get("anchors", [])
    positions = entity.get("positions", {})
    total_steps = len(tactic_names)

    if not goal_states or not tactic_names:
        return []

    examples: list[dict] = []
    proof_history: list[str] = []

    for step_idx in range(min(total_steps, len(goal_states))):
        goal_state = goal_states[step_idx]
        tactic = tactic_names[step_idx] if step_idx < len(tactic_names) else ""

        # Direction vector for this step's tactic
        if step_idx < len(tactic_directions):
            directions = tactic_directions[step_idx].get("directions", DEFAULT_DIRECTION)
        else:
            directions = TACTIC_DIRECTIONS.get(tactic, DEFAULT_DIRECTION)

        # Anchor labels: theorem anchors + tactic-specific anchors
        step_anchors = list(anchors)
        step_anchors.extend(TACTIC_ANCHORS.get(tactic, []))
        step_anchors = sorted(set(step_anchors))

        # Bank positions for training (sign * depth for each bank)
        bank_positions = {}
        for bank_name, pos_info in positions.items():
            sign = pos_info.get("sign", 0)
            depth = pos_info.get("depth", 0)
            bank_positions[bank_name] = [sign, depth]

        remaining_steps = total_steps - step_idx - 1

        example = {
            "goal_state": goal_state,
            "theorem_id": theorem_id,
            "step_index": step_idx,
            "total_steps": total_steps,
            "nav_directions": directions,
            "anchor_labels": step_anchors,
            "ground_truth_tactic": tactic,
            "ground_truth_premises": premises,
            "remaining_steps": remaining_steps,
            "solvable": True,
            "proof_history": list(proof_history),
            "bank_positions": bank_positions,
        }
        examples.append(example)

        # Add current goal to history for subsequent steps
        proof_history.append(goal_state)

    return examples


# ---------------------------------------------------------------------------
# Main: standalone worker with --resume
# ---------------------------------------------------------------------------


def _build_skip_set(output_path: Path) -> set[str]:
    """Load already-processed theorem IDs from existing output for --resume."""
    seen: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(json.loads(line).get("theorem_id", ""))
    return seen


def _process_entities(
    input_path: Path,
    output_path: Path,
    skip_ids: set[str],
    shard_idx: int,
    shard_total: int,
    append: bool,
) -> tuple[int, int, int]:
    """Process entity records into NavigationalExample JSONL.

    Returns (theorems_processed, theorems_skipped, examples_written).
    """
    mode = "a" if append else "w"
    processed = 0
    skipped = 0
    examples_written = 0
    unmapped = 0

    with open(input_path) as fin, open(output_path, mode) as fout:
        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if line_num % shard_total != shard_idx:
                continue

            entity = json.loads(line)
            tid = entity.get("theorem_id", "")
            if tid in skip_ids:
                skipped += 1
                continue

            examples = _build_nav_examples(entity)
            if not examples:
                unmapped += 1
                continue

            for ex in examples:
                fout.write(json.dumps(ex) + "\n")
                examples_written += 1

            processed += 1
            if processed % 5000 == 0:
                print(f"Processed {processed} theorems, {examples_written} examples...")

    if unmapped:
        print(f"Skipped {unmapped} theorems with no goal states or tactics")

    return processed, skipped, examples_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert entity records to NavigationalExample JSONL"
    )
    parser.add_argument("--input", required=True, help="Input JSONL (entities)")
    parser.add_argument("--output", required=True, help="Output JSONL (nav examples)")
    parser.add_argument("--resume", action="store_true", help="Skip processed")
    parser.add_argument("--shard", default=None, help="'idx:total' (e.g., '0:2')")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    skip_ids: set[str] = set()
    if args.resume:
        skip_ids = _build_skip_set(output_path)
        print(f"Resume: skipping {len(skip_ids)} already-processed theorems")

    shard_idx, shard_total = 0, 1
    if args.shard:
        parts = args.shard.split(":")
        shard_idx, shard_total = int(parts[0]), int(parts[1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed, skipped, examples = _process_entities(
        input_path, output_path, skip_ids, shard_idx, shard_total, append=args.resume
    )
    print(f"\nDone. Theorems: {processed}, Skipped: {skipped}, Examples: {examples}")
    if processed > 0:
        print(f"Avg examples/theorem: {examples / processed:.1f}")


if __name__ == "__main__":
    main()
