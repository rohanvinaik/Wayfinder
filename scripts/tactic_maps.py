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
    # Simplification variants
    "simpa":      _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "simp_rw":    _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "dsimp":      _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "rwa":        _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "erw":        _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    # Extension / congruence
    "ext":        _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=0, context=1),
    "ext1":       _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=0, context=1),
    "congr":      _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=0, context=0),
    "gcongr":     _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=0, context=0),
    "convert":    _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
    # Structure / goal management
    "constructor":  _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=0),
    "split_ifs":    _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "by_cases":     _DIR(structure=1, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "by_contra":    _DIR(structure=1, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "contrapose":   _DIR(structure=1, automation=1, decomposition=0, domain=0, depth=0, context=1),
    # Context manipulation
    "let":         _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "letI":        _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "haveI":       _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "set":         _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "change":      _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
    "replace":     _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "subst":       _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=-1),
    "classical":   _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=1),
    # Existential / witness
    "choose":         _DIR(structure=0, automation=1, decomposition=1, domain=0, depth=1, context=1),
    "exacts":         _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=0),
    "infer_instance": _DIR(structure=-1, automation=-1, decomposition=-1, domain=0, depth=-1, context=-1),
    # Automation
    "fun_prop":        _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "norm_num_ext":    _DIR(structure=-1, automation=-1, decomposition=-1, domain=-1, depth=-1, context=0),
    "filter_upwards":  _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=1),
    "lift":            _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "intros":          _DIR(structure=0, automation=1, decomposition=0, domain=0, depth=0, context=1),
    "exact_mod_cast":  _DIR(structure=0, automation=-1, decomposition=0, domain=-1, depth=-1, context=0),
    "Abel":            _DIR(structure=-1, automation=-1, decomposition=0, domain=0, depth=-1, context=0),
    "abs_of_nonneg":   _DIR(structure=0, automation=0, decomposition=0, domain=0, depth=0, context=0),
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
    "simpa": ["simp-solvable", "simplification"],
    "simp_rw": ["simp-solvable", "rewriting", "simplification"],
    "dsimp": ["simp-solvable", "simplification", "definition-unfolding"],
    "rwa": ["rewriting", "equality"],
    "erw": ["rewriting", "equality"],
    "ext": ["extensionality", "decomposition"],
    "ext1": ["extensionality", "decomposition"],
    "congr": ["congruence", "decomposition"],
    "gcongr": ["congruence", "decomposition", "inequality"],
    "convert": ["needs-manual-apply", "partial-term"],
    "constructor": ["needs-cases", "multi-branch"],
    "split_ifs": ["needs-cases", "multi-branch"],
    "by_cases": ["needs-cases", "multi-branch"],
    "by_contra": ["contradiction-method", "context-use"],
    "contrapose": ["contradiction-method", "context-use"],
    "let": ["needs-have-chain", "context-use"],
    "letI": ["needs-have-chain", "context-use"],
    "haveI": ["needs-have-chain", "multi-step", "decomposition"],
    "set": ["needs-have-chain", "context-use"],
    "change": ["rewriting"],
    "replace": ["hypothesis-specialization", "context-use"],
    "subst": ["rewriting", "context-use"],
    "classical": ["logic"],
    "choose": ["existential", "decomposition"],
    "exacts": ["needs-manual-exact", "one-liner"],
    "infer_instance": ["type-class-inference", "one-liner"],
    "fun_prop": ["automation-hint", "function-properties"],
    "filter_upwards": ["measure-theory", "context-use"],
    "lift": ["type-coercion", "context-use"],
    "intros": ["needs-manual-intro", "universal-quantifier"],
    "exact_mod_cast": ["type-coercion", "one-liner"],
}

# ---------------------------------------------------------------------------
# Domain classification by Mathlib namespace
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: list[tuple[str, int, list[str]]] = [
    # (namespace prefix, domain_sign, domain anchors)
    # -1 = concrete/computational, +1 = abstract/structural
    #
    # Patterns match against Lean 4 namespace (e.g. "CategoryTheory.Limits")
    # NOT "Mathlib.CategoryTheory.Limits". Both forms are checked via
    # _classify_domain's Mathlib-prefix stripping logic.
    #
    # Concrete (sign=-1): objects you can compute with
    ("Nat", -1, ["nat-arithmetic", "concrete"]),
    ("Int", -1, ["int-arithmetic", "concrete"]),
    ("Fin", -1, ["finite", "concrete"]),
    ("Finset", -1, ["finite", "combinatorics", "concrete"]),
    ("Multiset", -1, ["combinatorics", "concrete"]),
    ("List", -1, ["list", "concrete"]),
    ("Array", -1, ["array", "concrete"]),
    ("Vector", -1, ["vector", "concrete"]),
    ("Polynomial", -1, ["polynomial", "concrete"]),
    ("Matrix", -1, ["matrix", "concrete"]),
    ("SimpleGraph", -1, ["graph-theory", "concrete"]),
    ("ProbabilityTheory", -1, ["probability", "concrete"]),
    ("NumberTheory", -1, ["number-theory", "concrete"]),
    ("Combinatorics", -1, ["combinatorics", "concrete"]),
    ("Computability", -1, ["computability", "concrete"]),
    ("Data.Nat", -1, ["nat-arithmetic", "concrete"]),
    ("Data.Int", -1, ["int-arithmetic", "concrete"]),
    ("Data.Fin", -1, ["finite", "concrete"]),
    ("Data.List", -1, ["list", "concrete"]),
    ("Data", -1, ["data-structures", "concrete"]),
    # Abstract (sign=+1): structural, categorical, topological
    ("CategoryTheory", 1, ["category-theory", "abstract"]),
    ("MeasureTheory", 1, ["measure-theory", "abstract"]),
    ("GroupTheory", 1, ["group-theory", "abstract"]),
    ("RingTheory", 1, ["ring-theory", "abstract"]),
    ("FieldTheory", 1, ["field-theory", "abstract"]),
    ("Topology", 1, ["topology", "abstract"]),
    ("AlgebraicGeometry", 1, ["algebraic-geometry", "abstract"]),
    ("AlgebraicTopology", 1, ["algebraic-topology", "abstract"]),
    ("RepresentationTheory", 1, ["representation-theory", "abstract"]),
    ("Geometry", 1, ["geometry", "abstract"]),
    ("Filter", 1, ["filter", "abstract"]),
    ("Ideal", 1, ["ring-theory", "abstract"]),
    ("Module", 1, ["module-theory", "abstract"]),
    ("Submodule", 1, ["module-theory", "abstract"]),
    # Neutral (sign=0): spans concrete-abstract boundary
    ("Algebra", 0, ["algebra"]),
    ("Analysis", 0, ["analysis"]),
    ("LinearAlgebra", 0, ["linear-algebra"]),
    ("Logic", 0, ["logic"]),
    ("Order", 0, ["order-theory"]),
    ("Set", 0, ["set-theory"]),
    ("SetTheory", 0, ["set-theory"]),
    ("Data.Real", 0, ["real-analysis"]),
    ("Data.Set", 0, ["set-theory"]),
    ("Real", 0, ["real-analysis"]),
    ("Complex", 0, ["complex-analysis"]),
    ("Tactic", 0, ["tactic-library"]),
]

# ---------------------------------------------------------------------------
# Tactic category sets (for bank position computation)
# ---------------------------------------------------------------------------

AUTO_TACTICS: frozenset[str] = frozenset(
    {
        "simp",
        "simp_all",
        "simpa",
        "simp_rw",
        "dsimp",
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
        "fun_prop",
        "infer_instance",
        "exact_mod_cast",
    }
)

MANUAL_TACTICS: frozenset[str] = frozenset(
    {
        "apply",
        "exact",
        "exacts",
        "refine",
        "have",
        "haveI",
        "cases",
        "induction",
        "induction'",
        "intro",
        "intros",
        "rintro",
        "obtain",
        "rcases",
        "calc",
        "suffices",
        "use",
        "specialize",
        "contradiction",
        "exfalso",
        "convert",
        "ext",
        "ext1",
        "congr",
        "gcongr",
        "constructor",
        "by_cases",
        "by_contra",
        "contrapose",
        "choose",
        "let",
        "letI",
        "set",
        "replace",
        "lift",
    }
)

SIMPLIFIER_TACTICS: frozenset[str] = frozenset(
    {
        "simp",
        "simp_all",
        "simpa",
        "simp_rw",
        "dsimp",
        "ring",
        "omega",
        "norm_num",
        "decide",
        "linarith",
        "trivial",
        "assumption",
        "rfl",
        "done",
        "fun_prop",
        "infer_instance",
    }
)

BUILDER_TACTICS: frozenset[str] = frozenset(
    {
        "cases",
        "induction",
        "induction'",
        "have",
        "haveI",
        "suffices",
        "calc",
        "obtain",
        "rcases",
        "contradiction",
        "exfalso",
        "constructor",
        "by_cases",
        "split_ifs",
        "choose",
        "ext",
        "ext1",
        "congr",
        "gcongr",
    }
)

CONTEXT_ENRICHERS: frozenset[str] = frozenset(
    {
        "intro",
        "intros",
        "rintro",
        "have",
        "haveI",
        "obtain",
        "rcases",
        "cases",
        "specialize",
        "suffices",
        "let",
        "letI",
        "set",
        "replace",
        "by_cases",
        "by_contra",
        "contrapose",
        "split_ifs",
        "choose",
        "ext",
        "ext1",
        "classical",
        "filter_upwards",
        "lift",
    }
)

CONTEXT_REDUCERS: frozenset[str] = frozenset(
    {
        "assumption",
        "exact",
        "exacts",
        "rfl",
        "done",
        "trivial",
        "infer_instance",
        "subst",
    }
)

SPLITTER_TACTICS: frozenset[str] = frozenset(
    {
        "cases",
        "rcases",
        "obtain",
        "induction",
        "induction'",
        "have",
        "haveI",
        "suffices",
        "calc",
        "constructor",
        "by_cases",
        "split_ifs",
        "choose",
        "ext",
        "ext1",
    }
)

CLOSER_TACTICS: frozenset[str] = frozenset(
    {
        "assumption",
        "exact",
        "exacts",
        "apply",
        "rfl",
        "done",
        "omega",
        "simp",
        "simpa",
        "ring",
        "norm_num",
        "decide",
        "linarith",
        "trivial",
        "infer_instance",
        "fun_prop",
        "exact_mod_cast",
    }
)
