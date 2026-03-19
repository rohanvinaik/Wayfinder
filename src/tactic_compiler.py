"""Tactic compiler — deterministic lowering from family + premises to Lean syntax.

The neural system predicts: family (6 classes) + premise ranking.
This module compiles that prediction to valid Lean tactic strings.
The Lean kernel verifies.

This is the GSE sentence-architecture pattern applied to proof tactics:
- Family = word-scale primitive (constrained classification)
- Premises = argument slots (constrained selection from context)
- Compilation = deterministic template (no neural inference)
- Verification = exact oracle (Lean kernel)

The grammar per family is tiny and well-defined:
    rw    → "rw [p1, p2, ...]" or "rw [← p1, p2, ...]"
    exact → "exact p1" or "exact p1 arg1 arg2"
    apply → "apply p1"
    simp  → "simp [p1, p2, ...]" or "simp" or "simp only [p1, ...]"
    refine → "refine p1 ?_ ?_"
    other → pass-through (no template, use raw tactic name)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TacticCandidate:
    """A compiled tactic candidate ready for Lean verification."""

    text: str
    family: str
    premises_used: list[str] = field(default_factory=list)
    confidence: float = 0.0


def compile_rw(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """Compile rw candidates from ranked premises.

    Generates multiple candidates with different premise combinations
    and directionality (forward and backward rewrite).
    """
    candidates = []
    if not premises:
        return candidates

    # Single-premise rewrites (most common)
    for p in premises[:max_premises]:
        candidates.append(
            TacticCandidate(
                text=f"rw [{p}]",
                family="rw",
                premises_used=[p],
            )
        )
        # Backward rewrite
        candidates.append(
            TacticCandidate(
                text=f"rw [← {p}]",
                family="rw",
                premises_used=[p],
            )
        )

    # Multi-premise rewrite (top 2-3)
    if len(premises) >= 2:
        prem_str = ", ".join(premises[:3])
        candidates.append(
            TacticCandidate(
                text=f"rw [{prem_str}]",
                family="rw",
                premises_used=premises[:3],
            )
        )

    return candidates


def compile_exact(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """Compile exact candidates — each premise as a direct term."""
    candidates = []
    for p in premises[:max_premises]:
        candidates.append(
            TacticCandidate(
                text=f"exact {p}",
                family="exact",
                premises_used=[p],
            )
        )
        # With wildcard arguments (for applied lemmas)
        candidates.append(
            TacticCandidate(
                text=f"exact {p} _",
                family="exact",
                premises_used=[p],
            )
        )
    return candidates


def compile_apply(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """Compile apply candidates."""
    candidates = []
    for p in premises[:max_premises]:
        candidates.append(
            TacticCandidate(
                text=f"apply {p}",
                family="apply",
                premises_used=[p],
            )
        )
    return candidates


def compile_simp(premises: list[str], max_premises: int = 6) -> list[TacticCandidate]:
    """Compile simp candidates — with and without premise hints."""
    candidates = [
        TacticCandidate(text="simp", family="simp"),
    ]
    if premises:
        prem_str = ", ".join(premises[:max_premises])
        candidates.append(
            TacticCandidate(
                text=f"simp [{prem_str}]",
                family="simp",
                premises_used=premises[:max_premises],
            )
        )
        candidates.append(
            TacticCandidate(
                text=f"simp only [{prem_str}]",
                family="simp",
                premises_used=premises[:max_premises],
            )
        )
    return candidates


def compile_refine(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """Compile refine candidates — premises with hole placeholders."""
    candidates = []
    for p in premises[:max_premises]:
        candidates.append(
            TacticCandidate(
                text=f"refine {p} ?_",
                family="refine",
                premises_used=[p],
            )
        )
        candidates.append(
            TacticCandidate(
                text=f"refine {p}",
                family="refine",
                premises_used=[p],
            )
        )
    return candidates


# Family → compiler function
_COMPILERS = {
    "rw": compile_rw,
    "simp": compile_simp,
    "exact": compile_exact,
    "apply": compile_apply,
    "refine": compile_refine,
}


def compile_tactic(
    family: str,
    premises: list[str],
    confidence: float = 0.0,
) -> list[TacticCandidate]:
    """Compile tactic candidates from family + ranked premises.

    Returns a list of candidates ordered by expected likelihood.
    Each candidate is a valid Lean tactic string ready for verification.
    """
    compiler = _COMPILERS.get(family)
    if compiler is None:
        return []

    candidates = compiler(premises)
    for c in candidates:
        c.confidence = confidence
    return candidates


def compile_from_predictions(
    family_ranking: list[tuple[str, float]],
    premises: list[str],
    max_families: int = 3,
    max_candidates_per_family: int = 6,
) -> list[TacticCandidate]:
    """Compile candidates from a ranked list of family predictions.

    Args:
        family_ranking: [(family_name, confidence), ...] sorted by confidence desc
        premises: ranked premise names from retrieval
        max_families: how many top families to try
        max_candidates_per_family: max candidates per family

    Returns:
        Flat list of TacticCandidate, ordered by family confidence then template order.
    """
    all_candidates = []
    for family, conf in family_ranking[:max_families]:
        candidates = compile_tactic(family, premises, confidence=conf)
        all_candidates.extend(candidates[:max_candidates_per_family])
    return all_candidates
