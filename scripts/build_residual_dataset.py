"""Build residual-goal dataset from Mathlib training traces.

For each proof step in nav_train.jsonl:
  1. Classify as structural (intro/intros/constructor/cases) or non-structural
  2. For non-structural steps: record as residual training examples
  3. Output: goal_state, hypotheses, ground_truth_tactic, lane_category

This creates the training data for the local residual executor (Task B).
The key insight: train on post-structural goals, not raw theorem statements.

Usage:
    python scripts/build_residual_dataset.py \
        --input data/nav_train.jsonl \
        --output data/residual_train.jsonl \
        --samples 50000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Tactics that count as "structural" (setup, not closing)
STRUCTURAL_TACTICS = {
    "intro",
    "intros",
    "rintro",
    "intro_",
    "cases",
    "rcases",
    "obtain",
    "match",
    "induction",
    "induction'",
    "constructor",
    "exact_mod_cast",
    "have",
    "let",
    "suffices",
    "unfold",
    "dsimp",
    "change",
    "push_neg",
    "contrapose",
    "by_contra",
}


# Lane categories for non-structural tactics
def classify_lane(tactic_base: str) -> str:
    """Classify a tactic into a lane category."""
    automation = {
        "omega",
        "decide",
        "norm_num",
        "ring",
        "linarith",
        "positivity",
        "polyrith",
        "field_simp",
        "norm_cast",
    }
    solver = {
        "simp",
        "simp_all",
        "aesop",
        "tauto",
        "trivial",
        "rfl",
        "exact?",
        "apply?",
    }
    structural = {
        "intro",
        "intros",
        "rintro",
        "cases",
        "rcases",
        "obtain",
        "induction",
        "constructor",
        "have",
        "let",
        "suffices",
        "unfold",
        "dsimp",
    }

    if tactic_base in automation:
        return "automation"
    if tactic_base in solver:
        return "solver_bootstrap"
    if tactic_base in structural:
        return "structural_core"
    # Everything else: apply, exact, rw, calc, ext, etc.
    return "learned_local"


def build_residual_dataset(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> dict:
    """Build residual-goal dataset from training traces."""
    stats: dict[str, int] = Counter()
    residual_examples: list[dict] = []

    with open(input_path) as f:
        for i, line in enumerate(f):
            if max_samples and len(residual_examples) >= max_samples:
                break

            ex = json.loads(line)
            gt_tactic = ex.get("ground_truth_tactic", "")
            if not gt_tactic:
                continue

            tactic_base = gt_tactic.split()[0] if gt_tactic.strip() else ""
            stats["total"] += 1

            # Classify
            is_structural = tactic_base in STRUCTURAL_TACTICS
            lane = classify_lane(tactic_base)

            if is_structural:
                stats["structural"] += 1
                continue

            # This is a residual (non-structural) step — training target
            stats["residual"] += 1
            stats[f"lane:{lane}"] += 1

            residual_examples.append(
                {
                    "goal_state": ex["goal_state"],
                    "theorem_id": ex.get("theorem_id", ""),
                    "step_index": ex.get("step_index", 0),
                    "ground_truth_tactic": gt_tactic,
                    "tactic_base": tactic_base,
                    "lane_category": lane,
                    "ground_truth_premises": ex.get("ground_truth_premises", []),
                    "bank_positions": ex.get("bank_positions", {}),
                    "nav_directions": ex.get("nav_directions", {}),
                }
            )

            if (i + 1) % 50000 == 0:
                print(f"  Processed {i + 1} examples, {len(residual_examples)} residual")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in residual_examples:
            f.write(json.dumps(ex) + "\n")

    # Summary
    print("\n=== Residual Dataset Summary ===")
    print(f"  Total training examples: {stats['total']}")
    print(
        f"  Structural (skipped): {stats['structural']} ({100 * stats['structural'] / max(stats['total'], 1):.0f}%)"
    )
    print(
        f"  Residual (kept): {stats['residual']} ({100 * stats['residual'] / max(stats['total'], 1):.0f}%)"
    )
    print(f"  Output: {output_path} ({len(residual_examples)} examples)")
    print("\n  Lane distribution:")
    for key in sorted(stats):
        if key.startswith("lane:"):
            lane = key[5:]
            count = stats[key]
            print(f"    {lane}: {count} ({100 * count / max(stats['residual'], 1):.0f}%)")

    # Tactic base distribution (top 20)
    tactic_dist = Counter(ex["tactic_base"] for ex in residual_examples)
    print("\n  Top 20 residual tactics:")
    for tactic, count in tactic_dist.most_common(20):
        print(f"    {tactic}: {count} ({100 * count / len(residual_examples):.1f}%)")

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual-goal dataset")
    parser.add_argument("--input", type=Path, default=Path("data/nav_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/residual_train.jsonl"))
    parser.add_argument("--samples", type=int, default=None, help="Max residual examples")
    args = parser.parse_args()

    build_residual_dataset(args.input, args.output, args.samples)


if __name__ == "__main__":
    main()
