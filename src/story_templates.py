"""
Template taxonomy and extraction for Wayfinder v2.

Defines the 9 canonical proof strategy templates and provides
extraction logic to map tactic sequences to template labels.

Templates are the key to narrative regime conversion: raw proof structure
(Regime B, |G_μ| ≈ 1) becomes template classification (Regime A, |G_μ| >> 1).
See RESEARCH §2.9.5 for theoretical basis.
"""

from __future__ import annotations

from src.nav_contracts import BANK_NAMES
from src.som_contracts import TemplateInfo

# The 9 canonical proof strategy templates from DESIGN §10.2
TEMPLATE_TAXONOMY: dict[str, TemplateInfo] = {
    "DECIDE": TemplateInfo(
        template_id="DECIDE",
        pattern="Single automation tactic closes goal",
        bank_signature={"automation": -1, "depth": -1},
        tactic_patterns=["omega", "simp", "decide", "norm_num", "ring", "linarith"],
        is_simple=True,
    ),
    "REWRITE_CHAIN": TemplateInfo(
        template_id="REWRITE_CHAIN",
        pattern="Sequence of rewrites reaching normal form",
        bank_signature={"structure": 0, "automation": 0},
        tactic_patterns=["rw", "simp", "ring", "conv"],
        is_simple=True,
    ),
    "INDUCT_THEN_CLOSE": TemplateInfo(
        template_id="INDUCT_THEN_CLOSE",
        pattern="Induction + base/step each closed by automation",
        bank_signature={"structure": 1, "depth": 1, "automation": -1},
        tactic_patterns=["induction", "cases", "simp", "omega"],
        is_simple=False,
    ),
    "DECOMPOSE_AND_CONQUER": TemplateInfo(
        template_id="DECOMPOSE_AND_CONQUER",
        pattern="Split into independent subgoals via have/suffices",
        bank_signature={"decomposition": 1, "depth": 1},
        tactic_patterns=["have", "suffices", "obtain", "rcases"],
        is_simple=False,
    ),
    "APPLY_CHAIN": TemplateInfo(
        template_id="APPLY_CHAIN",
        pattern="Sequence of apply/exact targeting specific lemmas",
        bank_signature={"structure": 0, "automation": 1},
        tactic_patterns=["apply", "exact", "refine"],
        is_simple=True,
    ),
    "CASE_ANALYSIS": TemplateInfo(
        template_id="CASE_ANALYSIS",
        pattern="Split on data constructor or hypothesis",
        bank_signature={"structure": 1, "decomposition": 1},
        tactic_patterns=["cases", "rcases", "match", "split"],
        is_simple=False,
    ),
    "CONTRAPOSITIVE": TemplateInfo(
        template_id="CONTRAPOSITIVE",
        pattern="Negate goal, derive contradiction",
        bank_signature={"structure": 1, "context": 1},
        tactic_patterns=["by_contra", "contradiction", "absurd", "exfalso"],
        is_simple=False,
    ),
    "EPSILON_DELTA": TemplateInfo(
        template_id="EPSILON_DELTA",
        pattern="Introduce witnesses, bound distances",
        bank_signature={"domain": 1, "depth": 1},
        tactic_patterns=["use", "intro", "calc", "linarith", "norm_num"],
        is_simple=False,
    ),
    "HAMMER_DELEGATE": TemplateInfo(
        template_id="HAMMER_DELEGATE",
        pattern="Entirely delegated to ATP",
        bank_signature={"automation": -1},
        tactic_patterns=["aesop", "decide", "tauto", "omega"],
        is_simple=True,
    ),
}

# Ordered list of template names (canonical order)
TEMPLATE_NAMES: list[str] = list(TEMPLATE_TAXONOMY.keys())

# Simple templates get deterministic sketches (no learning needed)
SIMPLE_TEMPLATES: set[str] = {name for name, info in TEMPLATE_TAXONOMY.items() if info.is_simple}

COMPLEX_TEMPLATES: set[str] = {
    name for name, info in TEMPLATE_TAXONOMY.items() if not info.is_simple
}


_SINGLE_TACTIC_MAP: dict[str, str] = {}
for _tac in ("aesop", "tauto"):
    _SINGLE_TACTIC_MAP[_tac] = "HAMMER_DELEGATE"
for _tac in ("omega", "simp", "decide", "norm_num", "ring", "linarith"):
    _SINGLE_TACTIC_MAP[_tac] = "DECIDE"
for _tac in ("apply", "exact", "refine"):
    _SINGLE_TACTIC_MAP[_tac] = "APPLY_CHAIN"
for _tac in ("rw", "rewrite"):
    _SINGLE_TACTIC_MAP[_tac] = "REWRITE_CHAIN"

# Structural heuristic rules: (tactic_set, template_boost, boost_value, exclude_set)
_STRUCTURAL_HEURISTICS: list[tuple[frozenset[str], str, float, frozenset[str]]] = [
    (frozenset({"induction", "induct"}), "INDUCT_THEN_CLOSE", 0.5, frozenset()),
    (
        frozenset({"cases", "rcases", "match"}),
        "CASE_ANALYSIS",
        0.4,
        frozenset({"induction", "induct"}),
    ),
    (frozenset({"have", "suffices", "obtain"}), "DECOMPOSE_AND_CONQUER", 0.4, frozenset()),
    (
        frozenset({"by_contra", "contradiction", "absurd", "exfalso"}),
        "CONTRAPOSITIVE",
        0.5,
        frozenset(),
    ),
    (frozenset({"calc", "use"}), "EPSILON_DELTA", 0.3, frozenset()),
    (
        frozenset({"rw", "rewrite", "conv"}),
        "REWRITE_CHAIN",
        0.3,
        frozenset({"induction", "induct", "cases", "rcases", "match"}),
    ),
]


def classify_tactic_sequence(tactics: list[str]) -> str:
    """Classify a tactic sequence into a template by dominant pattern matching.

    Uses a scoring approach: for each template, count how many tactics in the
    sequence match the template's tactic_patterns. Structural heuristics
    break ties (e.g., presence of ``induction`` strongly signals INDUCT_THEN_CLOSE).

    Args:
        tactics: List of tactic names (e.g., ["induction", "simp", "omega"]).

    Returns:
        Template name (e.g., "INDUCT_THEN_CLOSE").
    """
    if not tactics:
        return "DECIDE"

    # Normalize: take first word of each tactic (ignore arguments)
    tac_lower = [t.lower().split()[0] for t in tactics]

    # Single-tactic proofs: direct lookup
    if len(tac_lower) == 1:
        return _SINGLE_TACTIC_MAP.get(tac_lower[0], "DECIDE")

    # Multi-tactic: score each template by pattern match ratio
    scores = _score_by_pattern_match(tac_lower, len(tactics))
    _apply_structural_heuristics(tac_lower, scores)
    return max(scores, key=lambda k: scores[k])


def _score_by_pattern_match(tac_lower: list[str], total: int) -> dict[str, float]:
    """Score each template by fraction of tactics matching its patterns."""
    scores: dict[str, float] = {}
    for name, info in TEMPLATE_TAXONOMY.items():
        pattern_set = {p.lower() for p in info.tactic_patterns}
        matches = sum(1 for t in tac_lower if t in pattern_set)
        scores[name] = matches / total
    return scores


def _apply_structural_heuristics(tac_lower: list[str], scores: dict[str, float]) -> None:
    """Apply structural heuristics to boost template scores based on key tactics."""
    tac_set = set(tac_lower)
    for trigger_set, template, boost, exclude_set in _STRUCTURAL_HEURISTICS:
        if tac_set & trigger_set and not (tac_set & exclude_set):
            scores[template] += boost


def compute_bank_signature(directions_sequence: list[dict[str, int]]) -> dict[str, int]:
    """Compute the bank-signature centroid for a sequence of per-step directions.

    For each bank, takes the sign of the sum of directions across all steps.
    This gives the dominant direction per bank for the entire proof.

    Args:
        directions_sequence: List of per-step direction dicts (bank_name -> {-1, 0, +1}).

    Returns:
        Bank signature dict: bank_name -> {-1, 0, +1} (dominant direction).
    """
    if not directions_sequence:
        return {bank: 0 for bank in BANK_NAMES}

    sums: dict[str, int] = {bank: 0 for bank in BANK_NAMES}
    for dirs in directions_sequence:
        for bank, direction in dirs.items():
            if bank in sums:
                sums[bank] += direction

    def _sign(x: int) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    return {bank: _sign(s) for bank, s in sums.items()}


def get_template_index(template_name: str) -> int:
    """Get the integer index of a template name."""
    return TEMPLATE_NAMES.index(template_name)


def get_template_name(index: int) -> str:
    """Get the template name from its integer index."""
    return TEMPLATE_NAMES[index]


def get_num_templates() -> int:
    """Return the number of templates in the taxonomy."""
    return len(TEMPLATE_NAMES)
