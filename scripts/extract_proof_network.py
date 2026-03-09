"""
Standalone worker: extract proof network entities from LeanDojo JSONL export.

ModelAtlas worker pattern: reads JSONL, writes JSONL, zero Wayfinder src/ imports,
supports --resume. Deployable on CPU workers (macpro, homebridge) via parsync.

For each theorem, assigns:
  - 6-bank positions (STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION)
  - Anchors from type content, tactic usage, namespace
  - Accessible premises from the premise list

Input:  data/leandojo_mathlib.jsonl (one JSON object per theorem)
Output: data/proof_network_entities.jsonl (one JSON object per entity record)

Usage:
  python scripts/extract_proof_network.py --input data/leandojo_mathlib.jsonl \
      --output data/proof_network_entities.jsonl [--resume] [--shard 0:2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.tactic_maps import (
    AUTO_TACTICS,
    BUILDER_TACTICS,
    CLOSER_TACTICS,
    CONTEXT_ENRICHERS,
    CONTEXT_REDUCERS,
    DEFAULT_DIRECTION,
    DOMAIN_PATTERNS,
    MANUAL_TACTICS,
    SIMPLIFIER_TACTICS,
    SPLITTER_TACTICS,
    TACTIC_ANCHORS,
    TACTIC_DIRECTIONS,
)

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_tactic_name(tactic_text: str) -> str:
    """Extract the base tactic name from a tactic application string."""
    text = tactic_text.strip()
    if not text:
        return ""
    match = re.match(r"(\w+['?]?)", text)
    return match.group(1) if match else text.split()[0]


def _classify_domain(namespace: str) -> tuple[int, list[str]]:
    """Classify theorem domain from namespace. Returns (sign, anchors)."""
    for prefix, sign, anchors in DOMAIN_PATTERNS:
        if namespace.startswith(prefix):
            return sign, anchors
    return 0, ["general"]


def _compute_structure_position(theorem_type: str, tactic_names: list[str]) -> tuple[int, int]:
    """Compute STRUCTURE bank position from type complexity and tactics used."""
    arrows = theorem_type.count("→") + theorem_type.count("->")
    foralls = theorem_type.count("∀") + theorem_type.lower().count("forall")
    complexity = arrows + foralls

    simplifiers = sum(1 for t in tactic_names if t in SIMPLIFIER_TACTICS)
    builders = sum(1 for t in tactic_names if t in BUILDER_TACTICS)

    if simplifiers > builders:
        sign = -1
    elif builders > simplifiers:
        sign = 1
    else:
        sign = 0

    return sign, min(complexity, 3)


def _compute_depth_position(proof_length: int) -> tuple[int, int]:
    """Compute DEPTH bank position from proof length."""
    if proof_length <= 1:
        return -1, 2
    if proof_length <= 3:
        return -1, 1
    if proof_length <= 8:
        return 0, 0
    if proof_length <= 15:
        return 1, 1
    return 1, 2


def _compute_automation_position(tactic_names: list[str]) -> tuple[int, int]:
    """Compute AUTOMATION bank position."""
    auto = sum(1 for t in tactic_names if t in AUTO_TACTICS)
    manual = sum(1 for t in tactic_names if t in MANUAL_TACTICS)
    total = max(auto + manual, 1)
    ratio = auto / total

    if ratio > 0.7:
        sign = -1
    elif ratio < 0.3:
        sign = 1
    else:
        sign = 0

    return sign, min(total, 3)


def _compute_context_position(hypothesis_count: int, tactic_names: list[str]) -> tuple[int, int]:
    """Compute CONTEXT bank position."""
    enrichers = sum(1 for t in tactic_names if t in CONTEXT_ENRICHERS)
    reducers = sum(1 for t in tactic_names if t in CONTEXT_REDUCERS)

    if enrichers > reducers:
        sign = 1
    elif reducers > enrichers:
        sign = -1
    else:
        sign = 0

    return sign, min(hypothesis_count, 3)


def _compute_decomposition_position(tactic_names: list[str]) -> tuple[int, int]:
    """Compute DECOMPOSITION bank position."""
    splitters = sum(1 for t in tactic_names if t in SPLITTER_TACTICS)
    closers = sum(1 for t in tactic_names if t in CLOSER_TACTICS)

    if splitters > closers:
        sign = 1
    elif closers > splitters:
        sign = -1
    else:
        sign = 0

    return sign, min(splitters, 3)


def _extract_type_anchors(theorem_type: str) -> list[str]:
    """Extract semantic anchors from theorem type signature."""
    anchors: list[str] = []
    type_lower = theorem_type.lower()

    # fmt: off
    checks: list[tuple[bool, str]] = [
        ("→" in theorem_type or "->" in theorem_type, "implication"),
        ("∀" in theorem_type or "forall" in type_lower, "universal-quantifier"),
        ("∃" in theorem_type or "exists" in type_lower, "existential"),
        ("iff" in type_lower or "↔" in theorem_type, "biconditional"),
        ("=" in theorem_type, "equality"),
        ("≤" in theorem_type or "<=" in theorem_type, "inequality"),
        ("<" in theorem_type and "←" not in theorem_type, "strict-inequality"),
        ("+" in theorem_type, "addition"),
        ("*" in theorem_type or "·" in theorem_type, "multiplication"),
        ("ℕ" in theorem_type or "Nat" in theorem_type, "nat-arithmetic"),
        ("ℤ" in theorem_type or "Int" in theorem_type, "int-arithmetic"),
        ("ℝ" in theorem_type or "Real" in theorem_type, "real-analysis"),
        ("Set" in theorem_type, "set-theory"),
        ("List" in theorem_type, "list"),
        ("Finset" in theorem_type, "finite"),
    ]
    # fmt: on

    for condition, anchor in checks:
        if condition:
            anchors.append(anchor)
    return anchors


def _infer_namespace(theorem_id: str) -> str:
    """Infer namespace from fully qualified theorem name."""
    parts = theorem_id.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else ""


def _count_hypotheses(goal_state: str) -> int:
    """Count hypotheses in a goal state string."""
    if not goal_state:
        return 0
    count = 0
    for line in goal_state.strip().split("\n"):
        if "⊢" in line or "|-" in line:
            break
        if ":" in line and line.strip():
            count += 1
    return count


def _compute_all_positions(
    theorem_type: str, tactic_names: list[str], namespace: str, goal_states: list[str]
) -> tuple[dict, list[str], int]:
    """Compute all 6-bank positions. Returns (positions, domain_anchors, hyp_count)."""
    struct_sign, struct_depth = _compute_structure_position(theorem_type, tactic_names)
    domain_sign, domain_anchors = _classify_domain(namespace)
    depth_sign, depth_depth = _compute_depth_position(len(tactic_names))
    auto_sign, auto_depth = _compute_automation_position(tactic_names)
    hyp_count = _count_hypotheses(goal_states[0] if goal_states else "")
    ctx_sign, ctx_depth = _compute_context_position(hyp_count, tactic_names)
    decomp_sign, decomp_depth = _compute_decomposition_position(tactic_names)

    positions = {
        "structure": {"sign": struct_sign, "depth": struct_depth},
        "domain": {"sign": domain_sign, "depth": 1},
        "depth": {"sign": depth_sign, "depth": depth_depth},
        "automation": {"sign": auto_sign, "depth": auto_depth},
        "context": {"sign": ctx_sign, "depth": ctx_depth},
        "decomposition": {"sign": decomp_sign, "depth": decomp_depth},
    }
    return positions, domain_anchors, hyp_count


def _collect_anchors(
    domain_anchors: list[str], theorem_type: str, tactic_names: list[str]
) -> list[str]:
    """Collect and deduplicate all anchors for an entity."""
    anchors: list[str] = list(domain_anchors)
    anchors.extend(_extract_type_anchors(theorem_type))
    for tname in set(tactic_names):
        anchors.extend(TACTIC_ANCHORS.get(tname, []))
    return sorted(set(anchors))


def extract_entity(theorem: dict) -> dict:
    """Extract a proof network entity record from a LeanDojo theorem entry."""
    theorem_id = theorem.get("theorem_id", theorem.get("full_name", ""))
    theorem_type = theorem.get("theorem_statement", theorem.get("type", ""))
    goal_states = theorem.get("goal_states", [])
    namespace = theorem.get("namespace", _infer_namespace(theorem_id))

    tactic_names = [t for t in (_extract_tactic_name(s) for s in theorem.get("tactics", [])) if t]
    positions, domain_anchors, hyp_count = _compute_all_positions(
        theorem_type, tactic_names, namespace, goal_states
    )

    return {
        "theorem_id": theorem_id,
        "entity_type": "lemma",
        "namespace": namespace,
        "file_path": theorem.get("file_path", ""),
        "positions": positions,
        "anchors": _collect_anchors(domain_anchors, theorem_type, tactic_names),
        "premises": theorem.get("premises", []),
        "tactic_names": tactic_names,
        "tactic_directions": [
            {"tactic": t, "directions": TACTIC_DIRECTIONS.get(t, DEFAULT_DIRECTION)}
            for t in tactic_names
        ],
        "goal_states": goal_states,
        "proof_length": len(tactic_names),
        "hypothesis_count": hyp_count,
    }


# ---------------------------------------------------------------------------
# Main: standalone worker with --resume
# ---------------------------------------------------------------------------


def _build_skip_set(output_path: Path) -> set[str]:
    """Load already-processed theorem IDs from existing output for --resume."""
    skip_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    skip_ids.add(json.loads(line).get("theorem_id", ""))
    return skip_ids


def _iter_shard_lines(input_path: Path, shard_idx: int, shard_total: int):
    """Yield parsed JSON dicts for lines belonging to this shard."""
    with open(input_path) as fin:
        for line_num, raw_line in enumerate(fin):
            stripped = raw_line.strip()
            if stripped and line_num % shard_total == shard_idx:
                yield json.loads(stripped)


def _process_theorems(
    input_path: Path,
    output_path: Path,
    skip_ids: set[str],
    shard_idx: int,
    shard_total: int,
    append: bool,
) -> tuple[int, int, set[str]]:
    """Process theorems from input JSONL, write entity records to output."""
    processed = 0
    skipped = 0
    unmapped: set[str] = set()

    with open(output_path, "a" if append else "w") as fout:
        for theorem in _iter_shard_lines(input_path, shard_idx, shard_total):
            tid = theorem.get("theorem_id", theorem.get("full_name", ""))
            if tid in skip_ids:
                skipped += 1
                continue

            entity = extract_entity(theorem)
            unmapped.update(
                t for t in entity["tactic_names"] if t not in TACTIC_DIRECTIONS
            )
            fout.write(json.dumps(entity) + "\n")
            processed += 1
            if processed % 5000 == 0:
                print(f"Processed {processed} theorems...")

    return processed, skipped, unmapped


def _print_summary(processed: int, skipped: int, unmapped: set[str]) -> None:
    """Print extraction summary statistics."""
    total = len(TACTIC_DIRECTIONS)
    pct = len(unmapped) / max(len(unmapped) + total, 1) * 100
    print(f"\nDone. Processed: {processed}, Skipped (resume): {skipped}")
    print(f"Mapped tactics: {total}, Unmapped: {len(unmapped)} ({pct:.1f}%)")
    if unmapped:
        print(f"Unmapped: {sorted(unmapped)[:20]}")
        if pct > 10:
            print("WARNING: >10% unmapped tactics — review mapping coverage")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract proof network entities from LeanDojo JSONL"
    )
    parser.add_argument("--input", required=True, help="Input JSONL (LeanDojo)")
    parser.add_argument("--output", required=True, help="Output JSONL (entities)")
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
    processed, skipped, unmapped = _process_theorems(
        input_path, output_path, skip_ids, shard_idx, shard_total, append=args.resume
    )
    _print_summary(processed, skipped, unmapped)


if __name__ == "__main__":
    main()
