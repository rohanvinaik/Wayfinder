"""Build canonical local-execution datasets from raw LeanDojo traces.

This is §1 of PLAN_2: the single source of truth for local execution.

Produces three datasets:
  1. data/canonical_residual.jsonl — all non-structural steps with full context
  2. data/canonical_rw.jsonl — rw-family steps with ActionIR
  3. data/canonical_eval.jsonl — held-out replayable benchmark

Each record contains:
  - theorem_full_name (real Mathlib name, not synthetic)
  - file_path (for environment reconstruction)
  - step_index
  - goal_state_before (per-step goal)
  - tactic_text (full tactic with arguments)
  - tactic_base (first word)
  - family (rw/simp/exact/apply/refine/other)
  - annotated_premise (the premise used at this step)
  - prefix_tactics (all tactics before this step, for replay)
  - canonical_action_ir (lowered string from ActionIR parse, if parsable)
  - goal_shape_ir / trigger_profile_ir / subtask_ir (controller-facing move metadata)

Usage:
    python -m scripts.build_canonical_local_data
    python -m scripts.build_canonical_local_data --max-theorems 5000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.subtask_ir import derive_move_metadata
from src.tactic_canonicalizer import canonicalize

# Structural tactics — skipped for residual dataset
STRUCTURAL = {
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


def classify_family(base: str) -> str:
    if base in ("rw", "rw'", "simp_rw", "rwa"):
        return "rw"
    if base in ("simp", "simp_all", "simpa"):
        return "simp"
    if base in ("exact", "exact_mod_cast"):
        return "exact"
    if base == "refine":
        return "refine"
    if base == "apply":
        return "apply"
    return "other"


def build(
    raw_path: Path,
    output_dir: Path,
    max_theorems: int | None = None,
    eval_frac: float = 0.05,
    seed: int = 42,
) -> dict:
    """Build canonical datasets from raw LeanDojo traces."""
    import numpy as np

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    all_records: list[dict] = []

    print(f"Processing raw traces from {raw_path}...")
    with open(raw_path) as f:
        for thm_idx, line in enumerate(f):
            if max_theorems and thm_idx >= max_theorems:
                break

            thm = json.loads(line)
            full_name = thm.get("full_name", "")
            file_path = thm.get("file_path", "")
            tactics = thm.get("tactics", [])
            goal_states = thm.get("goal_states", [])
            premises = thm.get("premises", [])

            if not full_name or not tactics:
                stats["skip_no_name_or_tactics"] += 1
                continue

            stats["theorems"] += 1

            for step_idx, tactic_text in enumerate(tactics):
                if not isinstance(tactic_text, str) or not tactic_text.strip():
                    continue

                stats["total_steps"] += 1
                base = tactic_text.split()[0]

                # Goal state for this step
                goal_before = ""
                if goal_states and step_idx < len(goal_states):
                    goal_before = goal_states[step_idx]

                # Premise for this step
                premise = ""
                if isinstance(premises, list) and step_idx < len(premises):
                    p = premises[step_idx]
                    premise = p if isinstance(p, str) else str(p)

                # Prefix tactics and their intermediate goal states (for replay)
                prefix = [t for t in tactics[:step_idx] if isinstance(t, str)]
                prefix_goals = []
                for j in range(step_idx):
                    if goal_states and j < len(goal_states):
                        prefix_goals.append(goal_states[j])
                    else:
                        prefix_goals.append("")

                # Skip structural for residual dataset
                is_structural = base in STRUCTURAL
                if is_structural:
                    stats["structural"] += 1

                family = classify_family(base)
                has_args = len(tactic_text.strip()) > len(base)

                # Parse to ActionIR if applicable
                canonical_ir = ""
                if has_args and family != "other":
                    ir = canonicalize(tactic_text.strip(), family)
                    if ir is not None:
                        canonical_ir = ir.lower()
                        stats["ir_parsed"] += 1

                record = {
                    "theorem_full_name": full_name,
                    "file_path": file_path,
                    "step_index": step_idx,
                    "goal_state_before": goal_before,
                    "tactic_text": tactic_text.strip(),
                    "tactic_base": base,
                    "family": family,
                    "is_structural": is_structural,
                    "has_args": has_args,
                    "annotated_premise": premise,
                    "prefix_tactics": prefix,
                    "prefix_goal_states": prefix_goals,
                    "canonical_action_ir": canonical_ir,
                }

                goal_shape_ir, trigger_profile_ir, subtask_ir = derive_move_metadata(
                    goal_state=goal_before,
                    tactic_text=tactic_text.strip(),
                    family=family,
                    canonical_action_ir=canonical_ir,
                    annotated_premise=premise,
                    step_index=step_idx,
                    prefix_tactics=prefix,
                )
                record["goal_shape_ir"] = goal_shape_ir.to_dict()
                record["trigger_profile_ir"] = trigger_profile_ir.to_dict()
                record["subtask_ir"] = subtask_ir.to_dict()

                all_records.append(record)

            if (thm_idx + 1) % 10000 == 0:
                print(f"  {thm_idx + 1} theorems, {len(all_records)} records")

    print(f"  Total: {stats['theorems']} theorems, {len(all_records)} records")

    # Split: eval is a random subset of THEOREMS (not steps)
    theorem_names = sorted(set(r["theorem_full_name"] for r in all_records))
    n_eval_thms = max(int(len(theorem_names) * eval_frac), 100)
    eval_thm_set = set(rng.choice(theorem_names, n_eval_thms, replace=False))

    eval_records = [r for r in all_records if r["theorem_full_name"] in eval_thm_set]
    train_records = [r for r in all_records if r["theorem_full_name"] not in eval_thm_set]

    # Residual = non-structural
    residual_train = [r for r in train_records if not r["is_structural"]]
    residual_eval = [r for r in eval_records if not r["is_structural"]]

    # rw-specific
    rw_train = [r for r in residual_train if r["family"] == "rw" and r["has_args"]]
    rw_eval = [r for r in residual_eval if r["family"] == "rw" and r["has_args"]]

    # rw0: bare single rewrite (no direction, no applied args, no multi-atom)
    def is_rw0(r: dict) -> bool:
        ir = r.get("canonical_action_ir", "")
        if not ir or r["family"] != "rw":
            return False
        # rw0 = rw [single_name] with no ← and no comma
        return ir.startswith("rw [") and "←" not in ir and "," not in ir

    rw0_eval = [r for r in rw_eval if is_rw0(r)]

    # Write datasets
    def write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    write_jsonl(output_dir / "canonical_residual_train.jsonl", residual_train)
    write_jsonl(output_dir / "canonical_residual_eval.jsonl", residual_eval)
    write_jsonl(output_dir / "canonical_rw_train.jsonl", rw_train)
    write_jsonl(output_dir / "canonical_rw_eval.jsonl", rw_eval)
    write_jsonl(output_dir / "canonical_eval_replayable.jsonl", eval_records)
    write_jsonl(output_dir / "rw0_eval.jsonl", rw0_eval)

    # Summary
    print("\n=== Canonical Local-Execution Datasets ===")
    print(f"  Source: {raw_path}")
    print(f"  Theorems: {stats['theorems']} ({n_eval_thms} eval)")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Structural: {stats['structural']}")
    print(f"  IR parsed: {stats['ir_parsed']}")
    print("\n  Outputs:")
    print(f"    canonical_residual_train.jsonl: {len(residual_train)}")
    print(f"    canonical_residual_eval.jsonl:  {len(residual_eval)}")
    print(f"    canonical_rw_train.jsonl:       {len(rw_train)}")
    print(f"    canonical_rw_eval.jsonl:        {len(rw_eval)}")
    print(f"    canonical_eval_replayable.jsonl: {len(eval_records)}")
    print(f"    rw0_eval.jsonl:                {len(rw0_eval)}")

    # Family distribution
    fam_dist = Counter(r["family"] for r in residual_train)
    print("\n  Residual family distribution (train):")
    for fam, count in fam_dist.most_common():
        print(f"    {fam}: {count} ({100 * count / max(len(residual_train), 1):.0f}%)")

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical local-execution data")
    parser.add_argument("--raw", type=Path, default=Path("data/leandojo_mathlib.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--eval-frac", type=float, default=0.05)
    args = parser.parse_args()
    build(args.raw, args.output_dir, args.max_theorems, args.eval_frac)


if __name__ == "__main__":
    main()
