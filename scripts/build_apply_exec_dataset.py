"""
Build a per-candidate executable-validity dataset from EXP-APPLY-047 probe results.

One row per (goal, candidate) probe, labeled with 6-class outcome:
  closed               — apply accepted, all goals closed
  accepted_with_goals  — apply accepted, subgoals remain
  unification_mismatch — Lean rejected: conclusion didn't unify with goal
  typeclass_missing    — Lean rejected: typeclass elaboration failed
  unknown_identifier   — Lean rejected: name not in scope
  other                — other failure or no feedback

Features per row:
  theorem_full_name, goal_str, goal_head, candidate,
  candidate_namespace, candidate_conclusion_shape (from error text),
  cosine_rank, filter_passed, outcome_class, accepted, closed

Usage:
    python -m scripts.build_apply_exec_dataset \
        --probes runs/apply047_results.jsonl \
        --eval   data/canonical/canonical_residual_eval.jsonl \
        --output data/apply_exec_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Goal head extraction
# ---------------------------------------------------------------------------

_HEAD_RE = re.compile(r"⊢\s*(\S+)")


def goal_head(goal_str: str) -> str:
    """Extract the leading symbol of the goal conclusion."""
    m = _HEAD_RE.search(goal_str)
    if not m:
        return ""
    tok = m.group(1)
    # Strip trailing punctuation
    tok = tok.rstrip(".,;:")
    return tok


# ---------------------------------------------------------------------------
# Conclusion shape extraction from Lean error text
# ---------------------------------------------------------------------------

_CONCL_RE = re.compile(
    r"could not unify the conclusion of `[^`]+`\s*\n\s*(.+?)\nwith the goal",
    re.DOTALL,
)


def extract_candidate_conclusion(error_text: str) -> str:
    """Pull the candidate conclusion shape out of a unification_mismatch error."""
    m = _CONCL_RE.search(error_text)
    if not m:
        return ""
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Outcome class
# ---------------------------------------------------------------------------

_VALID_CLASSES = {
    "closed",
    "accepted_with_goals",
    "unification_mismatch",
    "typeclass_missing",
    "unknown_identifier",
    "other",
}


def outcome_class(probe: dict[str, Any]) -> str:
    if probe.get("crashed"):
        return "other"
    accepted = probe.get("accepted", False)
    closed = probe.get("closed", False)
    fb = probe.get("feedback") or {}
    cat = fb.get("category", "")
    if closed:
        return "closed"
    if accepted:
        return "accepted_with_goals"
    if cat in _VALID_CLASSES:
        return cat
    if cat == "none" and accepted:
        return "closed"
    return "other"


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def load_goal_map(eval_path: str) -> dict[str, str]:
    """Map theorem_full_name -> goal_state_before (step-0 apply examples)."""
    goal_map: dict[str, str] = {}
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("family") != "apply":
                continue
            if ex.get("step_index", 0) != 0:
                continue
            name = ex.get("theorem_full_name", "")
            goal = ex.get("goal_state_before", "")
            if name and goal:
                goal_map[name] = goal
    return goal_map


def build_dataset(
    probes_path: str,
    eval_path: str,
    output_path: str,
    probe_set: str = "filtered",
) -> None:
    goal_map = load_goal_map(eval_path)
    print(f"Loaded {len(goal_map)} step-0 apply goals from eval", file=sys.stderr)

    rows_written = 0
    skipped_no_goal = 0
    skipped_no_probes = 0

    # Per-class counts
    class_counts: dict[str, int] = {c: 0 for c in _VALID_CLASSES}

    probe_key = "filtered_top5_probes" if probe_set == "filtered" else "raw_top5_probes"

    with open(probes_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            result = json.loads(line)
            if not result.get("goal_started"):
                continue

            name = result["theorem_full_name"]
            goal_str = goal_map.get(name, "")
            if not goal_str:
                skipped_no_goal += 1
                continue

            probes = result.get(probe_key) or []
            if not probes:
                skipped_no_probes += 1
                continue

            g_head = goal_head(goal_str)

            # Build cosine rank lookup from raw probes (order = cosine rank)
            raw_probes = result.get("raw_top5_probes") or []
            cosine_rank: dict[str, int] = {
                p["candidate"]: i for i, p in enumerate(raw_probes)
            }
            filtered_set = {
                p["candidate"] for p in (result.get("filtered_top5_probes") or [])
            }

            for probe in probes:
                cand = probe["candidate"]
                fb = probe.get("feedback") or {}
                error_text = fb.get("raw_error", "")
                messages = fb.get("messages") or []
                # Prefer message data for conclusion extraction
                msg_text = " ".join(
                    m.get("data", "") for m in messages if isinstance(m, dict)
                )
                concl_shape = extract_candidate_conclusion(msg_text or error_text)

                oc = outcome_class(probe)
                class_counts[oc] = class_counts.get(oc, 0) + 1

                # Candidate namespace = first dotted component
                cand_ns = cand.split(".")[0] if "." in cand else ""

                acc = probe.get("accepted", False)
                clo = probe.get("closed", False)
                row = {
                    "theorem_full_name": name,
                    "goal_str": goal_str,
                    "goal_head": g_head,
                    "candidate": cand,
                    "candidate_namespace": cand_ns,
                    "candidate_conclusion_shape": concl_shape,
                    "cosine_rank": cosine_rank.get(cand, -1),
                    "filter_passed": cand in filtered_set,
                    "outcome_class": oc,
                    "executable": 1 if (acc or clo) else 0,
                    "accepted": acc,
                    "closed": clo,
                    "feedback_category": fb.get("category", ""),
                    "feedback_stage": fb.get("stage", ""),
                    "feedback_raw_error": error_text[:512],
                }
                fout.write(json.dumps(row) + "\n")
                rows_written += 1

    print(f"Written {rows_written} rows to {output_path}", file=sys.stderr)
    print(f"Skipped: {skipped_no_goal} no goal text, {skipped_no_probes} no probes",
          file=sys.stderr)
    print("Outcome class distribution:", file=sys.stderr)
    total = max(rows_written, 1)
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {cls:<25} {count:4d}  ({100*count/total:.1f}%)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes", default="runs/apply047_results.jsonl",
        help="JSONL output from run_apply047_compat_filter",
    )
    parser.add_argument(
        "--eval", default="data/canonical/canonical_residual_eval.jsonl",
        help="Canonical residual eval JSONL (for goal text)",
    )
    parser.add_argument(
        "--output", default="data/apply_exec_dataset.jsonl",
        help="Output per-candidate dataset",
    )
    parser.add_argument(
        "--probe-set", choices=["filtered", "raw"], default="filtered",
        help="Which probe set to flatten (default: filtered)",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    build_dataset(args.probes, args.eval, args.output, probe_set=args.probe_set)


if __name__ == "__main__":
    main()
