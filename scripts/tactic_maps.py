"""
Tactic-to-bank direction mappings, anchor maps, and domain classification.

Extracted from WAYFINDER_DESIGN.md Section 5.1. Shared between extraction
and training data scripts. No Wayfinder src/ imports (standalone worker data).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 6-bank tactic-to-direction mapping (from WAYFINDER_DESIGN.md §5.1)
# Order: structure, automation, decomposition, domain, depth, context
# ---------------------------------------------------------------------------

# fmt: off
_DIR = dict  # alias for compactness

TACTIC_DIRECTIONS: dict[str, dict[str, int]] = {
    # Decision procedures: -1, -1, -1, -1, -1, 0
    "omega":    _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=0),
    "decide":   _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=0),
    "norm_num": _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=0),
    # Simplification: -1, -1, 0, 0, -1, 0
    "simp":      _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "simp_all":  _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "ring":      _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "field_simp":_DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    # Rewriting: 0, 0, 0, 0, 0, 0
    "rw":      _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "rewrite": _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "conv":    _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "unfold":  _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    # Linear arithmetic: -1, -1, -1, -1, -1, +1
    "linarith":   _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=1),
    "positivity": _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=1),
    # Introduction: 0, +1, 0, 0, 0, +1
    "intro":  _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "rintro": _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    # Elimination: +1, +1, +1, 0, +1, +1
    "cases":  _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "rcases": _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "obtain": _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=1),
    # Induction: +1, +1, +1, 0, +1, 0
    "induction":  _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=0),
    "induction'": _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=0),
    # Application: 0, +1, 0, 0, 0, 0
    "apply":  _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
    "exact":  _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
    "refine": _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
    # Subgoal creation: 0, +1, +1, 0, +1, +1
    "have":     _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "suffices": _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "calc":     _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=1, context=1),
    # Assumption: -1, -1, -1, 0, -1, -1
    "assumption": _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    "exact?":     _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    "trivial":    _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    # Contradiction: +1, +1, 0, 0, 0, +1
    "contradiction": _DIR(structure=1, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "exfalso":       _DIR(structure=1, automation=1, decomposition=0, domain=0, depth=0, context=1),
    # Automation with hints: -1, 0, 0, 0, 0, 0
    "aesop":    _DIR(structure=-1, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "tauto":    _DIR(structure=-1, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "push_neg": _DIR(structure=-1, automation=0, decomposition=0, domain=0, depth=0, context=0),
    # Type coercion: 0, -1, 0, -1, -1, 0
    "norm_cast": _DIR(structure=0, automation=-1, decomposition=0, domain=-1, depth=-1, context=0),
    "push_cast": _DIR(structure=0, automation=-1, decomposition=0, domain=-1, depth=-1, context=0),
    # Additional common tactics
    "rfl":        _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    "done":       _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    "specialize": _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "use":        _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
}
# fmt: on

DEFAULT_DIRECTION: dict[str, int] = {
    "structure": 0,
    "automation": 0,
    "decomposition": 0,
    "domain": 0,
    "depth": 0,
    "context": 0,
}

BANK_NAMES: list[str] = [
    "structure",
    "domain",
    "depth",
    "automation",
    "context",
    "decomposition",
]

# ---------------------------------------------------------------------------
# Tactic anchor mapping
# ---------------------------------------------------------------------------

TACTIC_ANCHORS: dict[str, list[str]] = {
    "omega": ["omega-solvable", "nat-arithmetic", "int-arithmetic", "decidable"],
    "decide": ["decidable", "bool-valued"],
    "norm_num": ["norm-num-solvable", "numerical", "decidable"],
    "simp": ["simp-solvable", "rewriting", "simplification"],
    "simp_all": ["simp-solvable", "rewriting", "simplification"],
    "ring": ["ring-solvable", "algebraic"],
    "field_simp": ["field-simplification", "algebraic"],
    "rw": ["rewriting", "equality"],
    "rewrite": ["rewriting", "equality"],
    "conv": ["rewriting", "targeted-rewrite"],
    "unfold": ["rewriting", "definition-unfolding"],
    "linarith": ["linarith-solvable", "inequality", "arithmetic"],
    "positivity": ["positivity-solvable", "inequality"],
    "intro": ["needs-manual-intro", "universal-quantifier", "implication"],
    "rintro": ["needs-manual-intro", "pattern-matching"],
    "cases": ["needs-cases", "multi-branch", "inductive-type"],
    "rcases": ["needs-cases", "pattern-matching"],
    "obtain": ["needs-cases", "existential"],
    "induction": ["needs-induction", "structural-recursion", "nat-arithmetic"],
    "apply": ["needs-manual-apply", "implication"],
    "exact": ["needs-manual-exact", "one-liner"],
    "refine": ["needs-manual-refine", "partial-term"],
    "have": ["needs-have-chain", "multi-step", "decomposition"],
    "suffices": ["needs-suffices", "backward-reasoning"],
    "calc": ["needs-calc", "chain-of-equalities"],
    "assumption": ["assumption-discharge", "context-use"],
    "trivial": ["trivially-solvable"],
    "contradiction": ["contradiction-method", "context-use"],
    "exfalso": ["contradiction-method"],
    "aesop": ["aesop-solvable", "automation-hint"],
    "tauto": ["tautology-solvable"],
    "push_neg": ["negation-pushing"],
    "norm_cast": ["type-coercion", "cast-normalization"],
    "push_cast": ["type-coercion", "cast-pushing"],
    "rfl": ["reflexivity", "one-liner"],
    "done": ["one-liner"],
    "specialize": ["hypothesis-specialization", "context-use"],
    "use": ["existential-witness"],
}

# ---------------------------------------------------------------------------
# Domain classification by Mathlib namespace
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: list[tuple[str, int, list[str]]] = [
    # (namespace prefix, domain_sign, domain anchors)
    # -1 = concrete/computational, +1 = abstract/structural
    ("Mathlib.Algebra", 0, ["algebra"]),
    ("Mathlib.Analysis", 0, ["analysis"]),
    ("Mathlib.CategoryTheory", 1, ["category-theory", "abstract"]),
    ("Mathlib.Combinatorics", -1, ["combinatorics", "concrete"]),
    ("Mathlib.Computability", -1, ["computability", "concrete"]),
    ("Mathlib.Data.Nat", -1, ["nat-arithmetic", "concrete"]),
    ("Mathlib.Data.Int", -1, ["int-arithmetic", "concrete"]),
    ("Mathlib.Data.Real", 0, ["real-analysis"]),
    ("Mathlib.Data.Fin", -1, ["finite", "concrete"]),
    ("Mathlib.Data.List", -1, ["list", "concrete"]),
    ("Mathlib.Data.Set", 0, ["set-theory"]),
    ("Mathlib.Data", -1, ["data-structures"]),
    ("Mathlib.GroupTheory", 1, ["group-theory", "abstract"]),
    ("Mathlib.LinearAlgebra", 0, ["linear-algebra"]),
    ("Mathlib.Logic", 0, ["logic"]),
    ("Mathlib.MeasureTheory", 1, ["measure-theory", "abstract"]),
    ("Mathlib.NumberTheory", -1, ["number-theory", "concrete"]),
    ("Mathlib.Order", 0, ["order-theory"]),
    ("Mathlib.RingTheory", 1, ["ring-theory", "abstract"]),
    ("Mathlib.SetTheory", 0, ["set-theory"]),
    ("Mathlib.Tactic", 0, ["tactic-library"]),
    ("Mathlib.Topology", 1, ["topology", "abstract"]),
]

# ---------------------------------------------------------------------------
# Tactic category sets (for bank position computation)
# ---------------------------------------------------------------------------

AUTO_TACTICS: frozenset[str] = frozenset(
    {
        "simp",
        "simp_all",
        "ring",
        "omega",
        "norm_num",
        "decide",
        "linarith",
        "positivity",
        "aesop",
        "tauto",
        "trivial",
        "assumption",
        "rfl",
        "done",
        "norm_cast",
        "push_neg",
        "field_simp",
    }
)

MANUAL_TACTICS: frozenset[str] = frozenset(
    {
        "apply",
        "exact",
        "refine",
        "have",
        "cases",
        "induction",
        "intro",
        "rintro",
        "obtain",
        "rcases",
        "calc",
        "suffices",
        "use",
        "specialize",
        "contradiction",
        "exfalso",
    }
)

SIMPLIFIER_TACTICS: frozenset[str] = frozenset(
    {
        "simp",
        "simp_all",
        "ring",
        "omega",
        "norm_num",
        "decide",
        "linarith",
        "trivial",
        "assumption",
        "rfl",
        "done",
    }
)

BUILDER_TACTICS: frozenset[str] = frozenset(
    {
        "cases",
        "induction",
        "have",
        "suffices",
        "calc",
        "obtain",
        "rcases",
        "contradiction",
        "exfalso",
    }
)

CONTEXT_ENRICHERS: frozenset[str] = frozenset(
    {
        "intro",
        "rintro",
        "have",
        "obtain",
        "rcases",
        "cases",
        "specialize",
        "suffices",
    }
)

CONTEXT_REDUCERS: frozenset[str] = frozenset(
    {
        "assumption",
        "exact",
        "rfl",
        "done",
        "trivial",
    }
)

SPLITTER_TACTICS: frozenset[str] = frozenset(
    {
        "cases",
        "rcases",
        "obtain",
        "induction",
        "have",
        "suffices",
        "calc",
        "constructor",
    }
)

CLOSER_TACTICS: frozenset[str] = frozenset(
    {
        "assumption",
        "exact",
        "apply",
        "rfl",
        "done",
        "omega",
        "simp",
        "ring",
        "norm_num",
        "decide",
        "linarith",
        "trivial",
    }
)
