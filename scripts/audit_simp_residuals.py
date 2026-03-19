"""Post-simp residual audit for EXP-RW-038.

For each simp-touched theorem (5 total), this script:
  1. Creates the initial goal via Pantograph
  2. Tries bare simp — captures goal state before and after
  3. If simp makes progress, tries a family classification probe
     on each remaining goal: apply, rw, omega, ring, exact, aesop
  4. Reports the classification and the remaining goal text

Usage:
    python -m scripts.audit_simp_residuals
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

from scripts.run_benchmark import _build_search_components, _resolve_initial_goal, load_benchmark_theorems
from src.proof_search import SearchConfig

TOUCHED_THEOREMS = {
    "update_le_update_iff",
    "DirectSum.map_eq_iff",
    "HasFDerivWithinAt.cpow",
    "Submodule.exists_fg_le_subset_range_rTensor_inclusion",
    "Surreal.Multiplication.P4_neg_right",
}

# Probe tactics grouped by family — ordered cheap to expensive
PROBES: list[tuple[str, str]] = [
    # family,         tactic
    ("automation",    "rfl"),
    ("automation",    "decide"),
    ("automation",    "omega"),
    ("automation",    "ring"),
    ("automation",    "norm_num"),
    ("automation",    "simp"),
    ("automation",    "aesop"),
    ("rw",            "rw [mul_comm]"),       # dummy — just checks rw receptivity
    ("apply",         "exact?"),              # skipped if too slow
]

# Cheap probes only (no exact? — too slow per goal)
FAST_PROBES: list[tuple[str, str]] = [
    ("automation",    "rfl"),
    ("automation",    "decide"),
    ("automation",    "omega"),
    ("automation",    "ring"),
    ("automation",    "norm_num"),
    ("automation",    "simp"),
    ("automation",    "aesop"),
]


def classify_goal_shape(goal_text: str) -> str:
    """Heuristic shape classification from goal text."""
    t = goal_text.lower()
    if "∀" in goal_text or "→" in goal_text:
        return "forall/implication → intro"
    if "∃" in goal_text:
        return "exists → constructor/use"
    if "=" in goal_text and "≤" not in goal_text and "<" not in goal_text:
        return "equality → rw/ring/simp/apply"
    if "≤" in goal_text or "<" in goal_text:
        return "inequality → omega/linarith/apply"
    if "∧" in goal_text:
        return "conjunction → constructor+apply"
    if "∨" in goal_text:
        return "disjunction → left/right+apply"
    if "⊢ True" in goal_text:
        return "trivial → trivial"
    if "mem" in t or "∈" in goal_text:
        return "membership → apply/simp"
    return "unknown → apply?"


def probe_goal(lean, goal_state: str) -> dict:
    """Try fast probes on a goal state. Returns {family: bool, ...}."""
    results = {}
    for family, tactic in FAST_PROBES:
        try:
            r = lean.try_tactic(goal_state, tactic)
            if r.success:
                results[tactic] = True
                break  # first success is enough
            results[tactic] = False
        except Exception:
            results[tactic] = False
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-simp residual audit")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("evaluation", {})["benchmark_theorems"] = args.theorems
    config["evaluation"].pop("mathlib_test_split", None)
    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]

    pipeline, base_cfg, lean, _, conn = _build_search_components(
        config, Path(args.checkpoint), args.device
    )

    theorems = load_benchmark_theorems(config, None)
    touched = [t for t in theorems if t["theorem_id"] in TOUCHED_THEOREMS]
    logger.info("Auditing %d simp-touched theorems", len(touched))

    if lean._backend == "pantograph":
        lean._ensure_server()

    print("\n" + "=" * 70)
    print("Post-simp Residual Audit — EXP-RW-038")
    print("=" * 70)

    for thm in touched:
        tid = thm["theorem_id"]
        print(f"\n{'─'*60}")
        print(f"Theorem: {tid}")

        initial_goal = _resolve_initial_goal(thm, lean)
        if initial_goal is None:
            print("  SKIP: goal creation failed")
            continue

        print(f"\nInitial goal:\n  {initial_goal[:200]}")

        # Try bare simp
        try:
            r = lean.try_tactic(initial_goal, "simp")
            print(f"\nsimp result: success={r.success}, error={r.error_message[:80] if r.error_message else ''}")
            if r.success:
                remaining = r.new_goals
                print(f"  Remaining goals after simp: {len(remaining)}")
                for i, g in enumerate(remaining):
                    gstr = str(g)
                    print(f"\n  Goal {i+1}:")
                    print(f"    {gstr[:300]}")
                    shape = classify_goal_shape(gstr)
                    print(f"    Shape heuristic: {shape}")

                    # Probe
                    probes = probe_goal(lean, gstr)
                    closers = [t for t, ok in probes.items() if ok]
                    if closers:
                        print(f"    Fast closers found: {closers}")
                    else:
                        print(f"    No fast closer found → needs apply/rw/learned")
            else:
                print("  simp did not succeed on initial goal directly")
                print("  (simp may have fired mid-search on a subgoal after other tactics)")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    lean.close()
    conn.close()


if __name__ == "__main__":
    main()
