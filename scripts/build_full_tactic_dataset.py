"""Build residual dataset with FULL tactic text (including arguments).

Re-extracts from leandojo_mathlib.jsonl preserving the complete tactic
application string (e.g., "rw [mul_comm, mul_assoc]" not just "rw").

This is the training target for the constrained output executor:
  goal_state + premises → full tactic string (verified by Lean)

Usage:
    python -m scripts.build_full_tactic_dataset
    python -m scripts.build_full_tactic_dataset --samples 50000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Structural tactics (same as build_residual_dataset.py)
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


def classify_family(tactic_base: str) -> str:
    """Map tactic base to family."""
    if tactic_base in ("rw", "rw'", "simp_rw", "rwa"):
        return "rw"
    if tactic_base in ("simp", "simp_all", "simpa"):
        return "simp"
    if tactic_base in ("exact", "exact_mod_cast"):
        return "exact"
    if tactic_base == "refine":
        return "refine"
    if tactic_base == "apply":
        return "apply"
    return "other"


def build(
    raw_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> dict:
    """Build full-tactic residual dataset."""
    stats: Counter[str] = Counter()
    examples: list[dict] = []

    with open(raw_path) as f:
        for line_idx, line in enumerate(f):
            if max_samples and len(examples) >= max_samples:
                break

            thm = json.loads(line)
            theorem_id = thm.get("name", f"thm_{line_idx}")
            tactics = thm.get("tactics", [])
            goal_states = thm.get("states", thm.get("goal_states", []))

            for step_idx, tactic_text in enumerate(tactics):
                if not isinstance(tactic_text, str) or not tactic_text.strip():
                    continue

                stats["total"] += 1
                base = tactic_text.split()[0]

                # Skip structural
                if base in STRUCTURAL_TACTICS:
                    stats["structural"] += 1
                    continue

                family = classify_family(base)
                has_args = len(tactic_text.strip()) > len(base)

                stats["residual"] += 1
                stats[f"family:{family}"] += 1
                if has_args:
                    stats["with_args"] += 1

                # Get goal state for this step if available
                goal = ""
                if goal_states and step_idx < len(goal_states):
                    gs = goal_states[step_idx]
                    goal = gs if isinstance(gs, str) else str(gs)

                examples.append(
                    {
                        "theorem_id": theorem_id,
                        "step_index": step_idx,
                        "goal_state": goal,
                        "full_tactic": tactic_text.strip(),
                        "tactic_base": base,
                        "family": family,
                        "has_args": has_args,
                    }
                )

                if max_samples and len(examples) >= max_samples:
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print("\n=== Full Tactic Dataset ===")
    print(f"  Total steps: {stats['total']}")
    print(f"  Structural (skipped): {stats['structural']}")
    print(f"  Residual: {stats['residual']}")
    print(
        f"  With arguments: {stats['with_args']} ({100 * stats['with_args'] / max(stats['residual'], 1):.0f}%)"
    )
    print(f"  Output: {output_path} ({len(examples)} examples)")
    print("\n  By family:")
    for key in sorted(stats):
        if key.startswith("family:"):
            fam = key[7:]
            c = stats[key]
            print(f"    {fam}: {c} ({100 * c / max(stats['residual'], 1):.0f}%)")

    # Show examples with full text
    print("\n  Sample full tactics:")
    for fam in ["rw", "exact", "apply", "simp", "refine"]:
        fam_exs = [e for e in examples if e["family"] == fam and e["has_args"]][:2]
        for ex in fam_exs:
            print(f"    [{fam}] {ex['full_tactic'][:70]}")

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full tactic dataset")
    parser.add_argument("--input", type=Path, default=Path("data/leandojo_mathlib.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/residual_full_tactic.jsonl"))
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    build(args.input, args.output, args.samples)


if __name__ == "__main__":
    main()
