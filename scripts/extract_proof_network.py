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
import functools
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
    text = tactic_text.strip().lstrip("(")
    if not text:
        return ""
    match = re.match(r"(\w+['?]?)", text)
    return match.group(1) if match else text.split()[0]


@functools.lru_cache(maxsize=1024)
def _classify_domain(namespace: str) -> tuple[int, tuple[str, ...]]:
    """Classify theorem domain from namespace. Returns (sign, anchors).

    Tries matching against the raw namespace first, then with common
    prefixes stripped (Mathlib., Lean., Init.) to handle both
    file-path-derived and Lean-4-module-derived namespace formats.
    """
    candidates = [namespace]
    for strip_prefix in ("Mathlib.", "Lean.", "Init."):
        if namespace.startswith(strip_prefix):
            candidates.append(namespace[len(strip_prefix) :])

    for ns in candidates:
        for prefix, sign, anchors in DOMAIN_PATTERNS:
            if ns.startswith(prefix):
                return sign, tuple(anchors)
    return 0, ("general",)


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


@functools.lru_cache(maxsize=4096)
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
) -> tuple[dict, tuple[str, ...], int]:
    """Compute all 6-bank positions. Returns (positions, domain_anchors, hyp_count)."""
    positions: dict[str, dict[str, int]] = {}

    sign, depth = _compute_structure_position(theorem_type, tactic_names)
    positions["structure"] = {"sign": sign, "depth": depth}

    domain_sign, domain_anchors = _classify_domain(namespace)
    positions["domain"] = {"sign": domain_sign, "depth": 1}

    sign, depth = _compute_depth_position(len(tactic_names))
    positions["depth"] = {"sign": sign, "depth": depth}

    sign, depth = _compute_automation_position(tactic_names)
    positions["automation"] = {"sign": sign, "depth": depth}

    hyp_count = _count_hypotheses(goal_states[0] if goal_states else "")
    sign, depth = _compute_context_position(hyp_count, tactic_names)
    positions["context"] = {"sign": sign, "depth": depth}

    sign, depth = _compute_decomposition_position(tactic_names)
    positions["decomposition"] = {"sign": sign, "depth": depth}

    return positions, domain_anchors, hyp_count


def _extract_namespace_anchors(namespace: str) -> list[str]:
    """Extract hierarchical namespace anchors for fine-grained domain matching."""
    if not namespace:
        return []
    anchors: list[str] = []
    parts = namespace.split(".")
    # Add each level: ns:Mathlib, ns:Mathlib.Order, ns:Mathlib.Order.Basic
    for i in range(1, min(len(parts) + 1, 5)):  # up to 4 levels
        anchors.append("ns:" + ".".join(parts[:i]))
    return anchors


def _extract_file_anchors(file_path: str) -> list[str]:
    """Extract file path anchors for locality-based matching.

    Premises are overwhelmingly from the same or nearby files.
    """
    if not file_path:
        return []
    # Normalize: Mathlib/Order/Basic.lean → Mathlib/Order/Basic
    path = file_path.replace(".lean", "").replace("\\", "/")
    parts = path.split("/")
    anchors: list[str] = []
    # Full file path as anchor (strongest locality signal)
    anchors.append(f"file:{path}")
    # Directory levels for broader matching
    for i in range(1, min(len(parts), 4)):
        anchors.append("dir:" + "/".join(parts[: i + 1]))
    return anchors


def _extract_name_anchors(theorem_id: str) -> list[str]:
    """Extract semantic anchors from the theorem name's fragments."""
    # Split camelCase and underscores: "mul_comm_of_ne_zero" → {mul, comm, ne, zero}
    name = theorem_id.rsplit(".", 1)[-1] if "." in theorem_id else theorem_id
    fragments: set[str] = set()
    for part in re.split(r"[_.]", name):
        # Split camelCase
        tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", part)
        for tok in tokens:
            tok_lower = tok.lower()
            if len(tok_lower) >= 3:  # skip tiny fragments
                fragments.add(tok_lower)
    return [f"name:{f}" for f in sorted(fragments)]


_TYPE_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:\.[A-Z][A-Za-z]+)*)\b")


def _extract_type_token_anchors(theorem_type: str) -> list[str]:
    """Extract capitalized type tokens as anchors (e.g., Finset, Module, Group)."""
    # Match capitalized identifiers (Lean type names)
    tokens = _TYPE_TOKEN_RE.findall(theorem_type)
    unique: set[str] = set()
    for tok in tokens:
        # Use the leaf name for qualified names
        leaf = tok.rsplit(".", 1)[-1]
        if len(leaf) >= 3:
            unique.add(leaf)
    return [f"type:{t}" for t in sorted(unique)]


def _collect_anchors(
    domain_anchors: list[str],
    theorem_type: str,
    tactic_names: list[str],
    entity_fields: dict,
) -> list[str]:
    """Collect and deduplicate all anchors for an entity.

    Args:
        domain_anchors: Anchors from domain classification.
        theorem_type: The theorem's type signature string.
        tactic_names: List of tactic names used in the proof.
        entity_fields: Dict with 'namespace', 'theorem_id', 'file_path' keys.
    """
    anchors: list[str] = list(domain_anchors)
    anchors.extend(_extract_type_anchors(theorem_type))
    for tname in set(tactic_names):
        anchors.extend(TACTIC_ANCHORS.get(tname, []))
    # Content-based anchors for fine-grained retrieval
    anchors.extend(_extract_namespace_anchors(entity_fields.get("namespace", "")))
    anchors.extend(_extract_name_anchors(entity_fields.get("theorem_id", "")))
    anchors.extend(_extract_type_token_anchors(theorem_type))
    anchors.extend(_extract_file_anchors(entity_fields.get("file_path", "")))
    return sorted(set(anchors))


def extract_entity(theorem: dict) -> dict:
    """Extract a proof network entity record from a LeanDojo theorem entry."""
    theorem_id = theorem.get("theorem_id", theorem.get("full_name", ""))
    theorem_type = theorem.get("theorem_statement", theorem.get("type", ""))
    goal_states = theorem.get("goal_states", [])
    namespace = theorem.get("namespace", _infer_namespace(theorem_id))
    file_path = theorem.get("file_path", "")

    tactic_names = [t for t in (_extract_tactic_name(s) for s in theorem.get("tactics", [])) if t]
    positions, domain_anchors, hyp_count = _compute_all_positions(
        theorem_type, tactic_names, namespace, goal_states
    )

    entity_fields = {
        "theorem_id": theorem_id,
        "namespace": namespace,
        "file_path": file_path,
    }

    return {
        "theorem_id": theorem_id,
        "entity_type": "lemma",
        "namespace": namespace,
        "file_path": file_path,
        "positions": positions,
        "anchors": _collect_anchors(domain_anchors, theorem_type, tactic_names, entity_fields),
        "premises": theorem.get("premises", []),
        "tactic_names": tactic_names,
        "tactic_directions": [
            {"tactic": t, "directions": TACTIC_DIRECTIONS.get(t, DEFAULT_DIRECTION)}
            for t in tactic_names
        ],
        "goal_states": goal_states,
        "proof_length": len(tactic_names),
        "hypothesis_count": hyp_count,
        "provenance": "traced",
    }


# ---------------------------------------------------------------------------
# Main: standalone worker with --resume
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 1 expansion: premise-only entities from LeanDojo corpus
# ---------------------------------------------------------------------------


def _extract_type_from_code(code: str) -> str:
    """Extract the type signature from a Lean 4 declaration's code.

    Handles: theorem X : TYPE, def X : TYPE, lemma X : TYPE, etc.
    Returns the type portion after the first ':' and before ':=' or 'where'.
    """
    # Find first colon that's not inside braces/parens
    depth = 0
    colon_pos = -1
    for i, c in enumerate(code):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == ":" and depth == 0 and colon_pos == -1:
            colon_pos = i
            break

    if colon_pos == -1:
        return ""

    rest = code[colon_pos + 1 :]
    # Trim at ':=' or 'where' (handle various whitespace patterns)
    for marker in [":=", " where\n", " where ", "\nwhere", " where\r"]:
        idx = rest.find(marker)
        if idx != -1:
            rest = rest[:idx]
    return rest.strip()


def extract_premise_entity(
    full_name: str,
    code: str,
    kind: str,
    file_path: str,
) -> dict:
    """Extract a premise-only entity from a LeanDojo corpus entry.

    These entities have no tactic traces, so bank positions are partial:
    - DOMAIN: from namespace classification (same as traced entities)
    - STRUCTURE: from type signature complexity (arrows, foralls)
    - DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION: zero (Informational Zero)

    Provenance is 'premise_only' to distinguish from 'traced' entities.
    """
    namespace = _infer_namespace(full_name)
    type_text = _extract_type_from_code(code)

    # Partial bank positions — only what we can infer without tactic traces
    positions: dict[str, dict[str, int]] = {}

    domain_sign, domain_anchors = _classify_domain(namespace)
    positions["domain"] = {"sign": domain_sign, "depth": 1}

    # Structure from type signature (no tactic info)
    arrows = type_text.count("→") + type_text.count("->")
    foralls = type_text.count("∀") + type_text.lower().count("forall")
    complexity = arrows + foralls
    # Without tactics, we can't determine simplifier vs builder direction
    # Use zero (Informational Zero) for sign, but record complexity as depth
    positions["structure"] = {"sign": 0, "depth": min(complexity, 3)}

    # These banks require tactic traces — set to zero (Informational Zero)
    positions["depth"] = {"sign": 0, "depth": 0}
    positions["automation"] = {"sign": 0, "depth": 0}
    positions["context"] = {"sign": 0, "depth": 0}
    positions["decomposition"] = {"sign": 0, "depth": 0}

    # Anchors from available metadata (no tactic anchors)
    anchors = list(domain_anchors)
    if type_text:
        anchors.extend(_extract_type_anchors(type_text))
        anchors.extend(_extract_type_token_anchors(type_text))
    anchors.extend(_extract_namespace_anchors(namespace))
    anchors.extend(_extract_name_anchors(full_name))
    anchors.extend(_extract_file_anchors(file_path))
    anchors = sorted(set(anchors))

    return {
        "theorem_id": full_name,
        "entity_type": "lemma",
        "namespace": namespace,
        "file_path": file_path,
        "positions": positions,
        "anchors": anchors,
        "premises": [],
        "tactic_names": [],
        "tactic_directions": [],
        "goal_states": [],
        "proof_length": 0,
        "hypothesis_count": 0,
        "provenance": "premise_only",
        "declaration_kind": kind,
    }


def _load_corpus_premises(corpus_path: Path) -> dict[str, dict]:
    """Load all declarations from LeanDojo corpus, keyed by full_name.

    Returns dict mapping full_name → {code, kind, file_path}.
    """
    premises: dict[str, dict] = {}
    with open(corpus_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            file_path = entry.get("path", "")
            for p in entry.get("premises", []):
                name = p.get("full_name", "")
                if name and name not in premises:
                    premises[name] = {
                        "code": p.get("code", ""),
                        "kind": p.get("kind", ""),
                        "file_path": file_path,
                    }
    return premises


def _process_premise_entities(
    corpus_path: Path,
    output_path: Path,
    existing_ids: set[str],
) -> int:
    """Extract premise-only entities from corpus and append to output.

    Only processes premises not already in existing_ids (traced entities).
    """
    print(f"Loading corpus from {corpus_path}...")
    corpus = _load_corpus_premises(corpus_path)
    print(f"  {len(corpus)} unique declarations in corpus")

    new_ids = set(corpus.keys()) - existing_ids
    print(f"  {len(new_ids)} premise-only entities to extract")

    count = 0
    with open(output_path, "a") as fout:
        for name in sorted(new_ids):
            info = corpus[name]
            entity = extract_premise_entity(
                full_name=name,
                code=info["code"],
                kind=info["kind"],
                file_path=info["file_path"],
            )
            fout.write(json.dumps(entity) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Extracted {count} premise-only entities...")

    print(f"  Done: {count} premise-only entities appended to {output_path}")
    return count


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
    shard: tuple[int, int],
    append: bool,
) -> tuple[int, int, set[str]]:
    """Process theorems from input JSONL, write entity records to output.

    Args:
        shard: (shard_idx, shard_total) for distributed processing.
    """
    processed = 0
    skipped = 0
    unmapped: set[str] = set()

    with open(output_path, "a" if append else "w") as fout:
        for theorem in _iter_shard_lines(input_path, shard[0], shard[1]):
            tid = theorem.get("theorem_id", theorem.get("full_name", ""))
            if tid in skip_ids:
                skipped += 1
                continue

            entity = extract_entity(theorem)
            unmapped.update(t for t in entity["tactic_names"] if t not in TACTIC_DIRECTIONS)
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the extraction worker."""
    parser = argparse.ArgumentParser(
        description="Extract proof network entities from LeanDojo JSONL"
    )
    parser.add_argument("--input", required=True, help="Input JSONL (LeanDojo)")
    parser.add_argument("--output", required=True, help="Output JSONL (entities)")
    parser.add_argument("--resume", action="store_true", help="Skip processed")
    parser.add_argument("--shard", default=None, help="'idx:total' (e.g., '0:2')")
    parser.add_argument(
        "--corpus",
        default=None,
        help="LeanDojo corpus.jsonl for premise-only entity expansion (Stage 1)",
    )
    return parser.parse_args()


def _parse_shard(shard_arg: str | None) -> tuple[int, int]:
    """Parse shard argument into (index, total). Defaults to (0, 1)."""
    if not shard_arg:
        return 0, 1
    parts = shard_arg.split(":")
    return int(parts[0]), int(parts[1])


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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed, skipped, unmapped = _process_theorems(
        input_path, output_path, skip_ids, _parse_shard(args.shard), append=args.resume
    )
    _print_summary(processed, skipped, unmapped)

    # Stage 1 expansion: add premise-only entities from corpus
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"Error: corpus not found: {corpus_path}", file=sys.stderr)
            sys.exit(1)

        # Collect all traced entity IDs from the output so far
        traced_ids = _build_skip_set(output_path)
        print("\n--- Stage 1: Premise-only entity expansion ---")
        print(f"Traced entities: {len(traced_ids)}")
        premise_count = _process_premise_entities(corpus_path, output_path, traced_ids)
        print(f"Total entities in {output_path}: {len(traced_ids) + premise_count}")
        print(f"  traced: {len(traced_ids)}, premise_only: {premise_count}")


if __name__ == "__main__":
    main()
