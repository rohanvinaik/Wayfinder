"""Build SoM multi-step training data from LeanDojo proof scripts.

Extracts (goal_state, tactic, specialist_family, step_context) tuples
for training specialist agents and the orchestrator.

Input: data/leandojo_benchmark_4/random/train.json (76K theorems with traced_tactics)
       data/canonical/subtask_train.jsonl (240K labeled steps with family tags)

Output: data/som_multistep_train.jsonl — step-level training data with:
  - goal_state_before: full goal context
  - tactic: the tactic applied
  - specialist: one of {rewrite, structural, solver, apply, closer}
  - goal_target: target expression (after ⊢)
  - goal_shape: structural features
  - step_index: position in the proof
  - proof_length: total steps in the proof
  - theorem_full_name: for grouping
  - state_after: goal after tactic (if available)

Usage:
    python -m scripts.build_som_training_data
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Tactic → Specialist mapping
# ---------------------------------------------------------------------------

def classify_specialist(tactic: str, family: str = "") -> str:
    """Map a tactic string to one of 5 SoM specialists.

    Priority order (first match wins):
    1. Family tag from subtask_train (if available)
    2. Tactic prefix matching
    """
    # Use family tag if available
    if family:
        _FAMILY_MAP = {
            "rw": "rewrite",
            "simp": "solver",
            "exact": "closer",
            "apply": "apply",
            "refine": "apply",
            "other": "structural",
        }
        return _FAMILY_MAP.get(family, "structural")

    # Prefix matching on tactic text
    tac = tactic.strip().lower()

    # Rewrite family
    if any(tac.startswith(p) for p in [
        "rw ", "rw[", "rw [", "rewrite", "conv", "simp only",
        "norm_cast", "push_cast", "ring_nf", "norm_num",
    ]):
        return "rewrite"

    # Solver family
    if any(tac.startswith(p) for p in [
        "simp", "omega", "linarith", "ring", "decide", "positivity",
        "field_simp", "norm_num", "aesop",
    ]):
        return "solver"

    # Closer family
    if any(tac.startswith(p) for p in [
        "exact ", "exact?", "exact(", "exact!",
        "trivial", "assumption", "rfl",
    ]):
        return "closer"

    # Apply family
    if any(tac.startswith(p) for p in [
        "apply ", "apply(", "refine ", "refine'", "refine(",
        "have ", "let ", "obtain ", "use ",
    ]):
        return "apply"

    # Structural family (default)
    # intro, cases, induction, constructor, ext, funext, rcases, etc.
    return "structural"


def extract_goal_shape(goal_state: str) -> dict:
    """Extract structural features from a goal state."""
    target = ""
    if "⊢" in goal_state:
        target = goal_state.split("⊢")[-1].strip()
    else:
        target = goal_state.strip()

    # Count hypotheses
    lines = goal_state.split("\n")
    hyp_count = sum(1 for l in lines if ":" in l and "⊢" not in l)

    # Target head
    target_head = target.split()[0] if target.split() else ""

    return {
        "target_head": target_head,
        "hyp_count": hyp_count,
        "target_len": len(target),
        "has_forall": "∀" in target,
        "has_exists": "∃" in target,
        "has_eq": "=" in target and "≤" not in target and "≥" not in target,
        "has_ineq": any(s in target for s in ["≤", "≥", "<", ">"]),
        "has_iff": "↔" in target,
        "has_and": "∧" in target,
        "has_or": "∨" in target,
        "has_neg": "¬" in target,
        "has_fun": "fun " in target or "λ" in target,
    }


# ---------------------------------------------------------------------------
# Build from LeanDojo traced tactics
# ---------------------------------------------------------------------------

def build_from_leandojo(train_path: str) -> list[dict]:
    """Extract step-level training data from LeanDojo traced tactics."""
    with open(train_path) as f:
        theorems = json.load(f)

    examples = []
    for thm in theorems:
        tactics = thm.get("traced_tactics", [])
        if not tactics:
            continue

        full_name = thm["full_name"]
        proof_length = len(tactics)

        for step_idx, step in enumerate(tactics):
            tactic = step.get("tactic", "")
            state_before = step.get("state_before", "")
            state_after = step.get("state_after", "")

            if not tactic or not state_before:
                continue

            specialist = classify_specialist(tactic)
            shape = extract_goal_shape(state_before)

            # Extract target (after last ⊢)
            target = ""
            if "⊢" in state_before:
                target = state_before.split("⊢")[-1].strip()

            examples.append({
                "theorem_full_name": full_name,
                "step_index": step_idx,
                "proof_length": proof_length,
                "goal_state_before": state_before,
                "goal_target": target[:500],  # truncate long targets
                "tactic": tactic,
                "specialist": specialist,
                "goal_shape": shape,
                "state_after": state_after[:500] if state_after else "",
                "source": "leandojo",
            })

    return examples


# ---------------------------------------------------------------------------
# Build from subtask_train (already labeled)
# ---------------------------------------------------------------------------

def build_from_subtask(subtask_path: str) -> list[dict]:
    """Extract step-level training data from subtask_train.jsonl."""
    examples = []
    with open(subtask_path) as f:
        for line in f:
            d = json.loads(line)

            family = d.get("family", "other")
            specialist = classify_specialist(d.get("canonical_action_ir", ""), family)
            state_before = d.get("goal_state_before", "")

            if not state_before:
                continue

            target = ""
            if "⊢" in state_before:
                target = state_before.split("⊢")[-1].strip()

            shape = extract_goal_shape(state_before)

            examples.append({
                "theorem_full_name": d.get("theorem_full_name", ""),
                "step_index": d.get("step_index", 0),
                "proof_length": d.get("prefix_len", 0) + 1,
                "goal_state_before": state_before,
                "goal_target": target[:500],
                "tactic": d.get("canonical_action_ir", ""),
                "specialist": specialist,
                "goal_shape": shape,
                "state_after": "",  # not available in subtask_train
                "source": "subtask",
                "family": family,
                "primary_premise": d.get("primary_premise", ""),
                "subtask_kind": d.get("subtask_kind", ""),
            })

    return examples


def main() -> None:
    leandojo_path = "data/leandojo_benchmark_4/random/train.json"
    subtask_path = "data/canonical/subtask_train.jsonl"
    output_path = "data/som_multistep_train.jsonl"

    # Build from both sources
    logger.info("Extracting from LeanDojo traced tactics...")
    leandojo_examples = build_from_leandojo(leandojo_path)
    logger.info("  %d examples from LeanDojo", len(leandojo_examples))

    logger.info("Extracting from subtask_train...")
    subtask_examples = build_from_subtask(subtask_path)
    logger.info("  %d examples from subtask_train", len(subtask_examples))

    # Deduplicate by (theorem_full_name, step_index)
    # Prefer subtask_train entries (they have richer labels)
    seen = set()
    final = []
    for ex in subtask_examples:
        key = (ex["theorem_full_name"], ex["step_index"])
        if key not in seen:
            seen.add(key)
            final.append(ex)
    for ex in leandojo_examples:
        key = (ex["theorem_full_name"], ex["step_index"])
        if key not in seen:
            seen.add(key)
            final.append(ex)

    logger.info("Total after dedup: %d", len(final))

    # Specialist distribution
    spec_counts = {}
    for ex in final:
        s = ex["specialist"]
        spec_counts[s] = spec_counts.get(s, 0) + 1
    logger.info("Specialist distribution:")
    for s, n in sorted(spec_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d (%.1f%%)", s, n, 100 * n / len(final))

    # Write
    with open(output_path, "w") as f:
        for ex in final:
            f.write(json.dumps(ex) + "\n")
    logger.info("Written to %s", output_path)

    # Also create eval split (last 10% of theorems by name)
    theorem_names = sorted(set(ex["theorem_full_name"] for ex in final))
    eval_cutoff = int(len(theorem_names) * 0.9)
    eval_names = set(theorem_names[eval_cutoff:])
    train_examples = [ex for ex in final if ex["theorem_full_name"] not in eval_names]
    eval_examples = [ex for ex in final if ex["theorem_full_name"] in eval_names]

    with open("data/som_multistep_eval.jsonl", "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    logger.info("Train: %d, Eval: %d (%.1f%%)",
                len(train_examples), len(eval_examples),
                100 * len(eval_examples) / len(final))


if __name__ == "__main__":
    main()
