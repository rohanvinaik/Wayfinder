"""Simp-aware scoping — build the local vocabulary for simp hint ranking.

Unlike rw_scoper, simp operates over a broader vocabulary:
  - local hypotheses (names directly usable as simp lemmas)
  - derived local projections (.1, .2, .symm for equality-like hyps)
  - accessible global premises (no rewrite-pattern filter)

The combined scope is then ranked purely by cosine similarity at search time;
no rewrite-direction heuristics are applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SimpScope:
    """Scoped vocabulary for simp hint selection."""

    hypotheses: list[str] = field(default_factory=list)
    derived_locals: list[str] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)
    all_symbols: list[str] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)  # "hyp" | "derived" | "premise"


def extract_hypotheses(goal_state: str) -> list[str]:
    """Extract hypothesis names from a Lean goal state (before ⊢)."""
    hyps: list[str] = []
    seen: set[str] = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w\'✝]*)\s*:", line)
        if m and m.group(1) not in ("case", "⊢") and m.group(1) not in seen:
            hyps.append(m.group(1))
            seen.add(m.group(1))
    return hyps


def _hyp_type(line: str) -> str:
    """Extract the type annotation of a hypothesis line (after the colon)."""
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return ""


def derive_local_projections(goal_state: str, hyps: list[str]) -> list[str]:
    """Produce simple derived local projections useful as simp hints.

    For each hypothesis h, adds:
      - h.1, h.2   if the type looks like a product/conjunction (∧, ×, And, Prod)
      - h.symm     if the type contains = or ↔ (equality/iff)
    Keeps derived list small — only projections likely to simplify.
    """
    derived: list[str] = []
    lines_by_name: dict[str, str] = {}
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w\'✝]*)\s*:(.*)", line)
        if m:
            lines_by_name[m.group(1)] = m.group(2).strip()

    for h in hyps:
        typ = lines_by_name.get(h, "")
        if not typ:
            continue
        if re.search(r"\b(And|∧|Prod|×)\b", typ):
            derived.extend([f"{h}.1", f"{h}.2"])
        if re.search(r"=|↔", typ):
            derived.append(f"{h}.symm")

    return derived


def scope_for_simp(
    goal_state: str,
    available_premises: list[str],
    max_premises: int = 30,
) -> SimpScope:
    """Build a scoped vocabulary for simp hint selection.

    1. Extract hypothesis names from goal state
    2. Derive local projections (.1, .2, .symm) for structured/equality hyps
    3. Include accessible global premises without rewrite-pattern filtering
       (simp can use any lemma as a simplification rule — no name heuristics)
    4. Cap global premises at max_premises (cosine ranking happens externally)
    5. Return combined scope with source tags
    """
    hyps = extract_hypotheses(goal_state)
    derived = derive_local_projections(goal_state, hyps)
    top_premises = available_premises[:max_premises]

    all_symbols = list(hyps) + derived + top_premises
    source_tags = (
        ["hyp"] * len(hyps)
        + ["derived"] * len(derived)
        + ["premise"] * len(top_premises)
    )

    return SimpScope(
        hypotheses=hyps,
        derived_locals=derived,
        premises=top_premises,
        all_symbols=all_symbols,
        source_tags=source_tags,
    )


def gold_hints_in_scope(
    gold_lemmas: list[str],
    scope: SimpScope,
) -> tuple[int, int]:
    """Return (n_gold_in_scope, n_gold_total) for recall diagnostics."""
    scope_set = set(scope.all_symbols)
    in_scope = sum(1 for g in gold_lemmas if g in scope_set)
    return in_scope, len(gold_lemmas)
