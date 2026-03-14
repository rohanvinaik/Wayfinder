"""Convert LeanDojo Benchmark 4 JSON to the flat JSONL format expected by extract_proof_network.py.

Input:  data/leandojo_benchmark_4/random/{train,val,test}.json
Output: data/leandojo_mathlib.jsonl

Each LeanDojo theorem has traced_tactics with {tactic, annotated_tactic, state_before, state_after}.
This script converts them to flat records with goal_states[], tactics[], premises[], etc.

Usage:
    python -m scripts.convert_leandojo \
        [--input-dir data/leandojo_benchmark_4/random] \
        [--output data/leandojo_mathlib.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_theorem_statement(traced_tactics: list[dict]) -> str:
    """Extract the theorem statement from the first goal state.

    The initial state_before typically has the goal after ⊢.
    We extract the full first state_before as the theorem type context.
    """
    if not traced_tactics:
        return ""
    state = traced_tactics[0]["state_before"]
    # Find the turnstile — the goal statement follows it
    turnstile_idx = state.rfind("⊢")
    if turnstile_idx >= 0:
        return state[turnstile_idx + 1 :].strip()
    return state.strip()


def _extract_premises(traced_tactics: list[dict]) -> list[str]:
    """Extract unique premise names from annotated_tactic references."""
    seen: set[str] = set()
    premises: list[str] = []
    for tt in traced_tactics:
        ann = tt.get("annotated_tactic", [])
        if not isinstance(ann, list) or len(ann) <= 1:
            continue
        refs = ann[1]
        if not isinstance(refs, list):
            continue
        for ref in refs:
            name = ref.get("full_name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            premises.append(name)
    return premises


def convert_theorem(thm: dict) -> dict | None:
    """Convert a single LeanDojo theorem to the flat extract format.

    Returns None for theorems with no traced_tactics (term-mode proofs
    or definitions — no tactic-level data to learn from).
    """
    traced = thm.get("traced_tactics", [])
    if not traced:
        return None

    full_name = thm["full_name"]
    goal_states = [tt["state_before"] for tt in traced]
    tactics = [tt["tactic"] for tt in traced]
    premises = _extract_premises(traced)

    return {
        "full_name": full_name,
        "theorem_statement": _extract_theorem_statement(traced),
        "goal_states": goal_states,
        "tactics": tactics,
        "premises": premises,
        "file_path": thm.get("file_path", ""),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LeanDojo Benchmark 4 → flat JSONL")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/leandojo_benchmark_4/random"),
        help="Directory containing train.json, val.json, test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/leandojo_mathlib.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to include (default: all)",
    )
    return parser.parse_args()


def _process_split(split: str, input_dir: Path, fout) -> tuple[int, int, int]:  # type: ignore[type-arg]
    path = input_dir / f"{split}.json"
    if not path.exists():
        print(f"  Skipping {path} (not found)", file=sys.stderr)
        return 0, 0, 0

    with open(path) as f:
        data = json.load(f)

    total = 0
    converted = 0
    skipped = 0
    print(f"  Processing {split}: {len(data)} theorems", file=sys.stderr)
    for thm in data:
        total += 1
        record = convert_theorem(thm)
        if record is None:
            skipped += 1
            continue
        fout.write(json.dumps(record) + "\n")
        converted += 1
    return total, converted, skipped


def main() -> None:
    args = _parse_args()

    total = 0
    converted = 0
    skipped = 0

    with open(args.output, "w") as fout:
        for split in args.splits:
            s_total, s_converted, s_skipped = _process_split(split, args.input_dir, fout)
            total += s_total
            converted += s_converted
            skipped += s_skipped

    print(
        f"Done: {converted} converted, {skipped} skipped (no tactics), {total} total",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
