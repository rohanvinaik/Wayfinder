"""Rewrite-aware scoping — build the local vocabulary for rw decoding.

For each goal state, builds a scoped vocabulary by:
1. Extracting local hypotheses from the goal text
2. Extracting head symbols from the goal target (redex candidates)
3. Filtering premises to rewrite-compatible ones (equality/iff names)
4. Ranking by head-symbol overlap with goal redexes

This replaces the generic "all accessible premises" approach with
a family-specific, redex-aware scoper that produces scope ~15-30.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ScopedVocabulary:
    """Scoped local vocabulary for a single goal state."""

    hypotheses: list[str] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)
    all_symbols: list[str] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)  # "hyp" | "premise" per symbol


def extract_hypotheses(goal_state: str) -> list[str]:
    """Extract hypothesis names from a Lean goal state."""
    hyps = []
    seen = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w\'✝]*)\s*:", line)
        if m and m.group(1) not in ("case", "⊢") and m.group(1) not in seen:
            hyps.append(m.group(1))
            seen.add(m.group(1))
    return hyps


def extract_goal_target(goal_state: str) -> str:
    """Extract the goal target (after ⊢) from a Lean goal state."""
    for line in goal_state.split("\n"):
        if "⊢" in line:
            return line.split("⊢", 1)[1].strip()
    return ""


def extract_head_symbols(target: str) -> set[str]:
    """Extract head symbols from a goal target for redex matching.

    Collects identifiers that could be the head of a rewritable subexpression.
    """
    # Find all identifiers (including qualified names like Nat.add)
    symbols = set(re.findall(r"\b(\w[\w\.]*)\b", target))
    # Filter out very short variable-like names and numbers
    return {s for s in symbols if len(s) > 1 or s.isupper()}


# Common rewrite-indicating name patterns
_REWRITE_PATTERNS = re.compile(
    r"(comm|assoc|zero|one|mul|add|sub|div|neg|inv|pow|succ|pred|"
    r"eq|iff|le|lt|ge|gt|not|and|or|imp|"
    r"cast|coe|norm|abs|dist|"
    r"map|comp|id|apply|ext|funext|congr|"
    r"mem|subset|inter|union|image|preimage|range|"
    r"sum|prod|sup|inf|max|min|"
    r"symm|trans|refl|rfl)",
    re.IGNORECASE,
)


def is_rewrite_candidate(premise_name: str) -> bool:
    """Heuristic: is this premise name likely useful for rewriting?

    True for names containing patterns like _comm, _assoc, _zero, eq_, etc.
    """
    return bool(_REWRITE_PATTERNS.search(premise_name))


def scope_for_rw(
    goal_state: str,
    available_premises: list[str],
    max_premises: int = 20,
) -> ScopedVocabulary:
    """Build a scoped vocabulary for rw tactic decoding.

    1. Extract hypotheses from goal state
    2. Extract head symbols from goal target
    3. Filter premises by rewrite compatibility
    4. Rank by head-symbol overlap
    5. Return scoped vocab with source tags
    """
    hyps = extract_hypotheses(goal_state)
    target = extract_goal_target(goal_state)
    goal_heads = extract_head_symbols(target)

    # Score premises by relevance
    scored: list[tuple[float, str]] = []
    for prem in available_premises:
        score = 0.0

        # Rewrite-pattern match
        if is_rewrite_candidate(prem):
            score += 1.0

        # Head symbol overlap (premise name contains a goal head symbol)
        prem_parts = set(prem.replace(".", "_").split("_"))
        overlap = len(prem_parts & goal_heads)
        score += overlap * 0.5

        # Namespace match with goal symbols
        if "." in prem:
            ns = prem.rsplit(".", 1)[0]
            if any(ns in s for s in goal_heads):
                score += 0.3

        scored.append((score, prem))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: -x[0])
    top_premises = [name for _, name in scored[:max_premises]]

    # Build combined vocabulary
    all_symbols = list(hyps) + top_premises
    source_tags = ["hyp"] * len(hyps) + ["premise"] * len(top_premises)

    return ScopedVocabulary(
        hypotheses=hyps,
        premises=top_premises,
        all_symbols=all_symbols,
        source_tags=source_tags,
    )


def infer_direction(premise_name: str, goal_target: str) -> str:
    """Infer rw direction from symbolic redex matching.

    Heuristics (ordered by specificity):
    1. Premise name contains directional suffix (_symm, _comm, _swap)
       → more likely backward (these convert one form to its symmetric)
    2. Premise base name components appear in the goal target
       → forward (the LHS pattern is visible in the goal)
    3. Premise name ends with a function name NOT in the goal
       → backward (we're rewriting TO this function, not FROM it)
    4. Default: ambiguous (caller should beam both)

    Returns "forward", "backward", or "ambiguous".
    """
    base = premise_name.rsplit(".", 1)[-1] if "." in premise_name else premise_name
    base_lower = base.lower()

    # Directional name patterns that suggest backward application
    _BACKWARD_SUFFIXES = {"_symm", "_comm", "_swap", "_flip", "_reverse",
                          "_inv", "_neg", "_sub", "_div"}
    if any(base_lower.endswith(s) for s in _BACKWARD_SUFFIXES):
        return "backward"

    # Check if the base name's head function appears in the goal
    base_parts = set(base.replace("_", " ").split())
    goal_words = set(re.findall(r"\b\w+\b", goal_target))

    # Strong overlap → forward (LHS is visible in goal)
    overlap = base_parts & goal_words
    if len(overlap) >= 2:
        return "forward"

    # Single overlap is weak — could go either way
    if len(overlap) == 1:
        return "forward"

    # No overlap → the premise's head isn't in the goal,
    # suggesting we're rewriting TO a different form (backward)
    if not overlap and base_parts:
        return "backward"

    return "ambiguous"
