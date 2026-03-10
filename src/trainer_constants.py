"""Domain inference constants and helpers for the proof training pipeline."""

from __future__ import annotations

from typing import Any

_DEFAULT_DOMAIN = "general"

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "algebra": ("group", "ring", "field", "monoid", "abelian", "commutative", "mul", "add"),
    "analysis": ("continuous", "limit", "derivative", "integral", "epsilon", "delta", "converge"),
    "topology": ("open", "closed", "compact", "hausdorff", "connected", "homeomorphism"),
    "number_theory": ("prime", "divisible", "mod", "gcd", "coprime", "congruent"),
    "linear_algebra": ("matrix", "vector", "linear", "span", "basis", "dimension", "eigenvalue"),
    "order_theory": ("lattice", "supremum", "infimum", "monotone", "partial_order"),
    "set_theory": ("subset", "union", "intersection", "element", "empty_set", "powerset"),
    "logic": ("implies", "iff", "not", "forall", "exists", "contradiction"),
    "combinatorics": ("permutation", "combination", "binomial", "counting", "pigeonhole"),
    "category_theory": ("functor", "morphism", "natural_transformation", "adjoint"),
}

_TACTIC_DOMAIN_HINTS: dict[str, str] = {
    "ring": "algebra",
    "omega": "number_theory",
    "linarith": "linear_algebra",
    "norm_num": "number_theory",
    "field_simp": "algebra",
    "positivity": "analysis",
}

_REPAIR_SEVERITY: dict[str, float] = {
    "sorry_proof": 1.5,
    "wrong_tactic": 1.4,
    "wrong_premise": 1.2,
    "type_mismatch": 1.1,
    "missing_hypothesis": 1.0,
    "incomplete_proof": 0.9,
    "unnecessary_tactic": 0.7,
    "style_issue": 0.4,
}


def infer_domain(example: Any) -> str:
    """Infer a coarse math domain label for domain-wise progression tracking."""
    goal = (getattr(example, "goal_state", "") or "").lower()
    stmt = (getattr(example, "theorem_statement", "") or "").lower()
    text = f"{goal} {stmt}"

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(k in text for k in keywords):
            return domain

    tactics = getattr(example, "tier1_tokens", [])
    if tactics:
        for tactic in tactics:
            tactic_domain = _TACTIC_DOMAIN_HINTS.get(tactic)
            if tactic_domain:
                return tactic_domain

    return _DEFAULT_DOMAIN
