"""
Deterministic lowering — converts ProofExample tier tokens to Lean tactic proof text.

This is the final pipeline stage: takes the three-tier decomposition
and produces a tactic proof string that can be verified by Lean's kernel.

No ML involved — purely deterministic template expansion.
"""

from __future__ import annotations

from collections.abc import Callable

from src.contracts import ProofExample, Tier2Block

# Tactics that take a premise argument directly
_PREMISE_TACTICS = frozenset(
    {
        "apply",
        "exact",
        "rw",
        "rewrite",
        "simp",
        "have",
        "use",
        "specialize",
        "refine",
        "calc",
        "conv",
        "unfold",
    }
)

# Tactics that are self-contained (no arguments needed)
_NULLARY_TACTICS = frozenset(
    {
        "ring",
        "omega",
        "linarith",
        "norm_num",
        "decide",
        "trivial",
        "assumption",
        "contradiction",
        "exfalso",
        "done",
        "rfl",
        "positivity",
        "norm_cast",
        "push_neg",
        "field_simp",
        "aesop",
        "tauto",
        "simp_all",
    }
)

# Structural tokens
_STRUCTURAL = frozenset(
    {
        "BOS",
        "EOS",
        "PAD",
        "SEP",
        "by",
        "sorry",
    }
)


def _build_premise_map(example: ProofExample) -> dict[int, Tier2Block]:
    """Build tactic_index → Tier2Block mapping."""
    return {block.tactic_index: block for block in example.tier2_blocks}


def _build_term_map(example: ProofExample) -> dict[str, str | int | float]:
    """Build slot_id → value mapping from tier3 slots."""
    return {slot.slot_id: slot.value for slot in example.tier3_slots}


def _lower_intro(block: Tier2Block | None, _term_map: dict, _index: int) -> str:
    if block and block.tokens:
        return f"intro {' '.join(block.tokens)}"
    return "intro"


def _lower_cases(block: Tier2Block | None, _term_map: dict, _index: int) -> str:
    if block and block.tokens:
        return f"cases {block.tokens[0]}"
    return "cases _"


def _lower_induction(block: Tier2Block | None, _term_map: dict, _index: int) -> str:
    if block and block.tokens:
        target = block.tokens[0]
        rest = f" with {' '.join(block.tokens[1:])}" if len(block.tokens) > 1 else ""
        return f"induction {target}{rest}"
    return "induction _"


def _lower_rw(block: Tier2Block | None, _term_map: dict, _index: int) -> str:
    if block and block.tokens:
        return f"rw [{', '.join(block.tokens)}]"
    return "rw []"


def _lower_simp(block: Tier2Block | None, _term_map: dict, _index: int) -> str:
    if block and block.tokens:
        return f"simp [{', '.join(block.tokens)}]"
    return "simp"


def _lower_have(block: Tier2Block | None, term_map: dict, index: int) -> str:
    if block and block.tokens:
        name = block.tokens[0]
        type_slot = term_map.get(f"have_{index}_type", "")
        if type_slot:
            return f"have {name} : {type_slot} := by"
        return f"have {name} := by"
    return "have h := by"


_TACTIC_HANDLERS: dict[str, Callable] = {
    "intro": _lower_intro,
    "cases": _lower_cases,
    "induction": _lower_induction,
    "rw": _lower_rw,
    "rewrite": _lower_rw,
    "simp": _lower_simp,
    "have": _lower_have,
}


def _lower_tactic(
    tactic: str,
    index: int,
    premise_map: dict[int, Tier2Block],
    term_map: dict[str, str | int | float],
) -> str | None:
    """Lower a single tactic token to Lean tactic syntax."""
    if tactic in _STRUCTURAL:
        return None
    if tactic in _NULLARY_TACTICS:
        return tactic

    block = premise_map.get(index)
    handler = _TACTIC_HANDLERS.get(tactic)
    if handler:
        return handler(block, term_map, index)

    if tactic in _PREMISE_TACTICS:
        if block and block.tokens:
            return f"{tactic} {' '.join(block.tokens)}"
        return f"{tactic} _"

    return tactic


def lower_proof_to_lean(example: ProofExample) -> str:
    """Convert a ProofExample to Lean tactic proof text.

    Uses tier1_tokens for tactic sequence, tier2_blocks for premises,
    and tier3_slots for free-form terms.
    """
    premise_map = _build_premise_map(example)
    term_map = _build_term_map(example)

    lines: list[str] = []
    for i, tactic in enumerate(example.tier1_tokens):
        lowered = _lower_tactic(tactic, i, premise_map, term_map)
        if lowered is not None:
            lines.append(f"  {lowered}")

    if not lines:
        return "  sorry"

    return "\n".join(lines)


def lower_to_theorem(example: ProofExample) -> str:
    """Convert a ProofExample to a complete Lean theorem block."""
    proof_body = lower_proof_to_lean(example)
    return f"{example.theorem_statement} := by\n{proof_body}"


def roundtrip_validate(example: ProofExample) -> tuple[bool, str]:
    """Lower to Lean syntax and do basic structural checks.

    Full verification requires a running Lean server — this only checks
    that lowering produces syntactically plausible output.
    """
    try:
        text = lower_proof_to_lean(example)
    except Exception as e:
        return False, f"Lowering failed: {e}"

    if not text.strip():
        return False, "Empty proof text"

    return True, ""
