"""
Negative example collection pipeline for Wayfinder v3A.

Three collectors + weak negative extraction:
  1. SorryHoleCollector: Failed tactics at sorry sites in Mathlib history
  2. PerturbationCollector: Single-tactic perturbations of working proofs
  3. SuggestionTraceCollector: Lean elaborator search trace failures
  4. Weak negatives: Unchosen tactics from LeanDojo traces (free, 0.1x weight)

Output: data/nav_negative.jsonl with NegativeExample schema.
Split hygiene: inherits train/eval by theorem_id from nav_train/nav_eval.

Usage:
    python -m scripts.collect_negatives --source sorry_holes,perturbation,suggestion_trace
    python -m scripts.collect_negatives --source weak --train-data data/nav_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v3_contracts import NegativeExample
from src.v3_scoring import compute_otp_dimensionality


def _load_train_ids(path: Path) -> set[str]:
    """Load theorem IDs from training data for split hygiene."""
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids.add(d.get("theorem_id", ""))
    return ids


def _classify_failure(error_message: str) -> str:
    """Classify a Lean error message as semantic or infra failure.

    Semantic: tactic genuinely wrong (type mismatch, no rewrite rules, etc.)
    Infra: timeout, environment mismatch, memory limit.
    """
    infra_keywords = ["timeout", "memory", "out of memory", "interrupted", "connection"]
    lower = error_message.lower()
    for keyword in infra_keywords:
        if keyword in lower:
            return "infra"
    return "semantic"


class SorryHoleCollector:
    """Collect negatives from sorry sites in Mathlib development history.

    Requires Axle API access to verify_proof and sorry2lemma.
    """

    def __init__(self, axle_client: object = None) -> None:
        self.axle_client = axle_client

    def collect(self, limit: int = 10000) -> list[NegativeExample]:
        """Collect negative examples from sorry holes.

        In production, this queries Mathlib git history for sorry sites,
        extracts goal states, and tests all mapped tactics via Axle.
        Returns placeholder structure for now — implementation requires
        Axle API integration (Phase 7.2).
        """
        # Placeholder: production implementation requires Axle API
        print(f"  SorryHoleCollector: limit={limit} (requires Axle API)")
        return []


class PerturbationCollector:
    """Collect negatives by perturbing working proofs.

    Single-tactic changes produce clean boundary signal:
    the (perturbation, repair) pair gives a (negative, positive) example.
    """

    def __init__(self, axle_client: object = None) -> None:
        self.axle_client = axle_client

    def collect(
        self,
        train_data_path: Path,
        limit: int = 50000,
    ) -> list[NegativeExample]:
        """Collect perturbation negatives from working proofs.

        In production, this generates single-tactic perturbations of
        working proofs and submits to Axle verify_proof for validation.
        Returns placeholder structure for now.
        """
        print(f"  PerturbationCollector: limit={limit} (requires Axle API)")
        return []


class SuggestionTraceCollector:
    """Collect negatives from Lean elaborator search traces.

    The strongest negatives: the Lean elaborator itself determined
    invalidity, not human choice.
    """

    def collect(
        self,
        eval_data_path: Path,
        limit: int = 20000,
    ) -> list[NegativeExample]:
        """Collect suggestion trace negatives.

        In production, this runs suggestion tactics (exact?, apply?, simp?)
        and captures the search trace. Returns placeholder for now.
        """
        print(f"  SuggestionTraceCollector: limit={limit} (requires Lean server)")
        return []


def collect_weak_negatives(
    train_data_path: Path,
    tactic_vocabulary: list[str] | None = None,
    subsample: int = 500000,
) -> list[NegativeExample]:
    """Extract weak negatives: unchosen tactics from LeanDojo traces.

    For each training example, the unchosen tactics from the 72 mapped
    tactics are candidate weak negatives. These are FREE — no Lean
    interaction required. Use 0.1x loss weight.

    Args:
        train_data_path: Path to nav_train.jsonl.
        tactic_vocabulary: List of mapped tactic names. If None, uses
            a minimal default set.
        subsample: Max examples to emit (random subsample for balance).
    """
    if tactic_vocabulary is None:
        tactic_vocabulary = [
            "simp",
            "rfl",
            "exact",
            "apply",
            "intro",
            "rw",
            "cases",
            "induction",
            "constructor",
            "ext",
            "funext",
            "linarith",
            "omega",
            "norm_num",
            "ring",
            "decide",
            "trivial",
            "assumption",
            "contradiction",
            "exfalso",
            "have",
            "obtain",
            "rcases",
            "refine",
            "calc",
            "conv",
            "push_neg",
            "by_contra",
        ]

    negatives: list[NegativeExample] = []
    count = 0

    with open(train_data_path) as f:
        for line in f:
            if count >= subsample:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gt_tactic = d.get("ground_truth_tactic", "")
            theorem_id = d.get("theorem_id", "")
            goal_state = d.get("goal_state", "")
            directions = d.get("nav_directions", {})
            history = d.get("proof_history", [])

            # For each tactic in vocabulary that isn't the ground truth
            gt_base = gt_tactic.split()[0] if gt_tactic else ""
            for tactic in tactic_vocabulary:
                if tactic == gt_base:
                    continue
                if count >= subsample:
                    break

                negatives.append(
                    NegativeExample(
                        goal_state=goal_state,
                        theorem_id=theorem_id,
                        step_index=d.get("step_index", 0),
                        failed_tactic=tactic,
                        failure_reason="unchosen_in_trace",
                        failure_category="weak_negative",
                        source="unchosen_weak",
                        proof_history=history,
                        paired_positive_tactic=gt_tactic,
                        paired_positive_premises=d.get("ground_truth_premises", []),
                        bank_directions=directions,
                        otp_dimensionality=compute_otp_dimensionality(directions),
                    )
                )
                count += 1

    return negatives


def write_negatives(negatives: list[NegativeExample], output_path: Path) -> None:
    """Write negative examples to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for neg in negatives:
            f.write(json.dumps(neg.to_dict()) + "\n")
    print(f"  Wrote {len(negatives)} negatives to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect negative examples for v3A training")
    parser.add_argument(
        "--source",
        type=str,
        default="weak",
        help="Comma-separated sources: sorry_holes,perturbation,suggestion_trace,weak",
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/nav_train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/nav_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/nav_negative.jsonl"))
    parser.add_argument("--limit", type=int, default=500000)
    args = parser.parse_args()

    sources = [s.strip() for s in args.source.split(",")]
    all_negatives: list[NegativeExample] = []

    for source in sources:
        print(f"Collecting from source: {source}")
        if source == "sorry_holes":
            collector = SorryHoleCollector()
            all_negatives.extend(collector.collect(limit=args.limit))
        elif source == "perturbation":
            collector_p = PerturbationCollector()
            all_negatives.extend(collector_p.collect(args.train_data, limit=args.limit))
        elif source == "suggestion_trace":
            collector_s = SuggestionTraceCollector()
            all_negatives.extend(collector_s.collect(args.eval_data, limit=args.limit))
        elif source == "weak":
            all_negatives.extend(collect_weak_negatives(args.train_data, subsample=args.limit))
        else:
            print(f"  Unknown source: {source}, skipping")

    print(f"\nTotal negatives: {len(all_negatives)}")
    write_negatives(all_negatives, args.output)


if __name__ == "__main__":
    main()
