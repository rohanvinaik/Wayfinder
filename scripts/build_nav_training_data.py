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
from typing import NamedTuple

from scripts.tactic_maps import DEFAULT_DIRECTION, TACTIC_ANCHORS, TACTIC_DIRECTIONS


class ShardConfig(NamedTuple):
    """Shard/resume parameters for _process_entities."""

    skip_ids: set[str]
    shard_idx: int
    shard_total: int
    append: bool


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resolve_step_directions(entity: dict, step_idx: int, tactic: str) -> dict:
    """Resolve the 6-bank direction vector for a single proof step.

    Uses tactic-level supervision for banks where tactic identity is a
    valid signal (AUTOMATION, DECOMPOSITION, STRUCTURE, DEPTH, CONTEXT),
    but overrides DOMAIN with the entity-level namespace-derived position.
    This follows the bank-source alignment principle: DOMAIN represents
    semantic mathematical locality, which should come from theorem/premise
    content, not from which tactic was applied.
    """
    tactic_dirs = entity.get("tactic_directions", [])
    if step_idx < len(tactic_dirs):
        directions = dict(tactic_dirs[step_idx].get("directions", DEFAULT_DIRECTION))
    else:
        directions = dict(TACTIC_DIRECTIONS.get(tactic, DEFAULT_DIRECTION))

    # Override DOMAIN with entity-level position (namespace-derived)
    entity_positions = entity.get("positions", {})
    domain_pos = entity_positions.get("domain", {})
    directions["domain"] = domain_pos.get("sign", 0)

    return directions


def _resolve_step_anchors(anchors: list[str], tactic: str) -> list[str]:
    """Merge theorem-level and tactic-specific anchors."""
    merged = set(anchors)
    merged.update(TACTIC_ANCHORS.get(tactic, []))
    return sorted(merged)


def _encode_bank_positions(positions: dict) -> dict:
    """Convert positions dict to [sign, depth] training format."""
    return {bank: [info.get("sign", 0), info.get("depth", 0)] for bank, info in positions.items()}


def _build_nav_examples(entity: dict) -> list[dict]:
    """Convert one entity record into NavigationalExample dicts, one per step."""
    goal_states = entity.get("goal_states", [])
    tactic_names = entity.get("tactic_names", [])
    if not goal_states or not tactic_names:
        return []

    total_steps = len(tactic_names)
    premises = entity.get("premises", [])
    anchors = entity.get("anchors", [])
    bank_positions = _encode_bank_positions(entity.get("positions", {}))

    examples: list[dict] = []
    proof_history: list[str] = []

    for step_idx in range(min(total_steps, len(goal_states))):
        tactic = tactic_names[step_idx] if step_idx < len(tactic_names) else ""
        examples.append(
            {
                "goal_state": goal_states[step_idx],
                "theorem_id": entity["theorem_id"],
                "step_index": step_idx,
                "total_steps": total_steps,
                "nav_directions": _resolve_step_directions(entity, step_idx, tactic),
                "anchor_labels": _resolve_step_anchors(anchors, tactic),
                "ground_truth_tactic": tactic,
                "ground_truth_premises": premises,
                "remaining_steps": total_steps - step_idx - 1,
                "solvable": True,
                "proof_history": list(proof_history),
                "bank_positions": bank_positions,
            }
        )
        proof_history.append(goal_states[step_idx])

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


def _iter_shard_entities(input_path: Path, shard_idx: int, shard_total: int):
    """Yield parsed entity dicts for lines belonging to this shard."""
    with open(input_path) as fin:
        for line_num, raw_line in enumerate(fin):
            stripped = raw_line.strip()
            if stripped and line_num % shard_total == shard_idx:
                yield json.loads(stripped)


def _write_examples(fout, examples: list[dict]) -> None:
    """Write a batch of examples as JSONL lines."""
    for ex in examples:
        fout.write(json.dumps(ex) + "\n")


def _process_entities(
    input_path: Path,
    output_path: Path,
    shard_config: ShardConfig,
) -> tuple[int, int, int]:
    """Process entity records into NavigationalExample JSONL.

    Returns (theorems_processed, theorems_skipped, examples_written).
    """
    processed = 0
    skipped = 0
    examples_written = 0
    unmapped = 0

    mode = "a" if shard_config.append else "w"
    entities = _iter_shard_entities(input_path, shard_config.shard_idx, shard_config.shard_total)

    with open(output_path, mode) as fout:
        for entity in entities:
            if entity.get("theorem_id", "") in shard_config.skip_ids:
                skipped += 1
                continue

            examples = _build_nav_examples(entity)
            if not examples:
                unmapped += 1
                continue

            _write_examples(fout, examples)
            examples_written += len(examples)
            processed += 1

            if processed % 5000 == 0:
                print(f"Processed {processed} theorems, {examples_written} examples...")

    if unmapped:
        print(f"Skipped {unmapped} theorems with no goal states or tactics")

    return processed, skipped, examples_written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert entity records to NavigationalExample JSONL"
    )
    parser.add_argument("--input", required=True, help="Input JSONL (entities)")
    parser.add_argument("--output", required=True, help="Output JSONL (nav examples)")
    parser.add_argument("--resume", action="store_true", help="Skip processed")
    parser.add_argument("--shard", default=None, help="'idx:total' (e.g., '0:2')")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    skip_ids: set[str] = set()
    if args.resume:
        skip_ids = _build_skip_set(output_path)
        print(f"Resume: skipping {len(skip_ids)} already-processed theorems")

    shard_idx, shard_total = tuple(int(p) for p in args.shard.split(":")) if args.shard else (0, 1)

    shard_config = ShardConfig(
        skip_ids=skip_ids,
        shard_idx=shard_idx,
        shard_total=shard_total,
        append=args.resume,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed, skipped, examples = _process_entities(input_path, output_path, shard_config)
    print(f"\nDone. Theorems: {processed}, Skipped: {skipped}, Examples: {examples}")
    if processed > 0:
        print(f"Avg examples/theorem: {examples / processed:.1f}")


if __name__ == "__main__":
    main()
