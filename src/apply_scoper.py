"""Apply-aware scoping — build local vocabulary for apply/refine hint ranking.

Unlike rw_scoper (direction heuristics) and simp_scoper (broad vocabulary),
apply scope is head-symbol aware:

  1. Local hypotheses — direct apply targets
  2. Head-match tier — accessible premises whose name contains the goal head
     (e.g., goal head "Tendsto" → premises with "tendsto" in the name)
  3. Conclusion-shape tier — premises whose suffix encodes typical concluders
     (e.g., _iff, _le, _lt, _eq, _surjective, _injective matching goal shape)
  4. Cosine fallback — remaining accessible premises ranked externally

Rationale: apply needs the *conclusion* of the lemma to unify with the goal.
Name-pattern matching on the head symbol gives a cheap first-pass filter
that outperforms blind cosine on apply prediction (conclusion shape ≠ premise
shape, so cosine on full text is misleading).

Source tags: "local_hyp" | "head_match" | "shape_match" | "premise"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ApplyScope:
    """Scoped vocabulary for apply/refine hint selection."""

    hypotheses: list[str] = field(default_factory=list)
    head_matches: list[str] = field(default_factory=list)   # name contains goal head
    shape_matches: list[str] = field(default_factory=list)  # suffix encodes goal shape
    premises: list[str] = field(default_factory=list)       # remaining cosine fallback
    all_symbols: list[str] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)    # per-symbol tier label
    goal_head: str = ""
    goal_shape: str = ""                                    # "eq" | "le" | "iff" | ...


# ---------------------------------------------------------------------------
# Goal analysis
# ---------------------------------------------------------------------------

def extract_goal_head(goal_state: str) -> str:
    """Extract the top-level head symbol from the ⊢ target.

    Examples:
        '⊢ Tendsto f atTop (nhds x)'  → 'Tendsto'
        '⊢ a ≤ b'                     → 'LE.le'   (relation normalised)
        '⊢ f x = g x'                 → 'Eq'
        '⊢ ∃ n, P n'                  → 'Exists'
        '⊢ P ∧ Q'                     → 'And'
    """
    for line in goal_state.split("\n"):
        if "⊢" in line:
            target = line.split("⊢", 1)[1].strip()
            # Logical connectives → canonical names
            if target.startswith("∀") or "→" in target[:6]:
                return "forall"
            if "↔" in target[:30]:
                return "Iff"
            if "∧" in target[:30]:
                return "And"
            if "∨" in target[:30]:
                return "Or"
            if "∃" in target[:6]:
                return "Exists"
            # Relational operators
            m_rel = re.search(r"\s(≤|<|≥|>|=|≠)\s", target[:60])
            if m_rel:
                op = m_rel.group(1)
                return {"≤": "LE.le", "<": "LT.lt", "≥": "GE.ge",
                        ">": "GT.gt", "=": "Eq", "≠": "Ne"}[op]
            # First identifier token (function application head)
            m = re.match(r"\(?(\w[\w\.]*)", target)
            if m:
                return m.group(1)
    return ""


def classify_goal_shape(goal_state: str) -> str:
    """Coarse shape label from the goal target for suffix-matching.

    Returns one of: eq | le | lt | iff | mem | surjective | injective |
                    tendsto | continuous | measurable | forall | exists | other
    """
    for line in goal_state.split("\n"):
        if "⊢" in line:
            t = line.split("⊢", 1)[1].strip().lower()
            if "↔" in t:
                return "iff"
            if " = " in t:
                return "eq"
            if "≤" in t or " le " in t:
                return "le"
            if " < " in t or " lt " in t:
                return "lt"
            if "∈" in t or "mem" in t:
                return "mem"
            if "surjective" in t or "bijective" in t:
                return "surjective"
            if "injective" in t:
                return "injective"
            if "tendsto" in t:
                return "tendsto"
            if "continuous" in t:
                return "continuous"
            if "measurable" in t:
                return "measurable"
            if t.startswith("∃") or t.startswith("exists"):
                return "exists"
            if t.startswith("∀") or "→" in t[:10]:
                return "forall"
    return "other"


# ---------------------------------------------------------------------------
# Head-match filtering
# ---------------------------------------------------------------------------

# Suffixes that indicate a conclusion-type match for the given shape
_SHAPE_SUFFIXES: dict[str, list[str]] = {
    "eq":          ["_eq", "_eq_", "eq_", ".eq", "_comm", "_assoc", "_congr", "_sub_eq", "_add_eq"],
    "le":          ["_le", "_le_", "le_", ".le", "_mono", "_nonneg", "_pos", "_bound"],
    "lt":          ["_lt", "_lt_", "lt_", ".lt"],
    "iff":         ["_iff", "_iff_", "iff_", ".iff", "_congr_iff"],
    "mem":         ["_mem", "_mem_", "mem_", ".mem", "_in_", "_subset"],
    "surjective":  ["_surjective", "surjective_", "_bijective"],
    "injective":   ["_injective", "injective_", "_inj"],
    "tendsto":     ["tendsto_", "_tendsto", "Tendsto"],
    "continuous":  ["continuous_", "_continuous", "Continuous"],
    "measurable":  ["measurable_", "_measurable", "Measurable"],
}


def _name_contains_head(name: str, head: str) -> bool:
    """Check if a lemma name contains the goal head symbol (case-insensitive)."""
    if not head or len(head) < 3:
        return False
    head_lower = head.lower()
    name_lower = name.lower()
    # Strip namespace prefix for matching (last component)
    last = name.split(".")[-1].lower()
    return head_lower in name_lower or head_lower in last


def _name_matches_shape(name: str, shape: str) -> bool:
    """Check if a lemma name has a suffix matching the goal's conclusion shape."""
    suffixes = _SHAPE_SUFFIXES.get(shape, [])
    name_lower = name.lower()
    return any(suf in name_lower for suf in suffixes)


# ---------------------------------------------------------------------------
# Main scoping function
# ---------------------------------------------------------------------------

def scope_for_apply(
    goal_state: str,
    available_premises: list[str],
    max_head_matches: int = 10,
    max_shape_matches: int = 10,
    max_premises: int = 20,
) -> ApplyScope:
    """Build a scoped vocabulary for apply/refine hint selection.

    Tier order (highest priority first):
      1. Local hypotheses — directly applicable as `apply h`
      2. Head-match tier — premises whose name contains the goal head symbol
      3. Shape-match tier — premises whose name suffix matches the goal shape
      4. Remaining accessible premises (cosine fallback, capped)

    Total scope size ≈ hyps + max_head_matches + max_shape_matches + max_premises.
    The tiers are de-duplicated: a premise appearing in head_match is not
    repeated in shape_match or premises.
    """
    goal_head = extract_goal_head(goal_state)
    goal_shape = classify_goal_shape(goal_state)

    # Tier 1: local hypotheses
    hyps: list[str] = []
    seen: set[str] = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w'✝]*)\s*:", line)
        if m and m.group(1) not in ("case", "⊢") and m.group(1) not in seen:
            hyps.append(m.group(1))
            seen.add(m.group(1))

    # Tier 2: head-match
    head_matches: list[str] = []
    for p in available_premises:
        if p in seen:
            continue
        if _name_contains_head(p, goal_head):
            head_matches.append(p)
            seen.add(p)
            if len(head_matches) >= max_head_matches:
                break

    # Tier 3: shape-match (from remaining premises not already in head tier)
    shape_matches: list[str] = []
    for p in available_premises:
        if p in seen:
            continue
        if _name_matches_shape(p, goal_shape):
            shape_matches.append(p)
            seen.add(p)
            if len(shape_matches) >= max_shape_matches:
                break

    # Tier 4: remaining premises (cosine fallback)
    remaining: list[str] = []
    for p in available_premises:
        if p in seen:
            continue
        remaining.append(p)
        if len(remaining) >= max_premises:
            break

    all_symbols = list(hyps) + head_matches + shape_matches + remaining
    source_tags = (
        ["local_hyp"] * len(hyps)
        + ["head_match"] * len(head_matches)
        + ["shape_match"] * len(shape_matches)
        + ["premise"] * len(remaining)
    )

    return ApplyScope(
        hypotheses=hyps,
        head_matches=head_matches,
        shape_matches=shape_matches,
        premises=remaining,
        all_symbols=all_symbols,
        source_tags=source_tags,
        goal_head=goal_head,
        goal_shape=goal_shape,
    )


def gold_in_scope(
    gold_premise: str,
    scope: ApplyScope,
) -> tuple[bool, str]:
    """Return (in_scope, tier) for a gold premise. Tier is the source tag or ''."""
    for sym, tag in zip(scope.all_symbols, scope.source_tags):
        if sym == gold_premise:
            return True, tag
    return False, ""
