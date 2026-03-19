"""Tactic canonicalizer — parse Lean tactic strings into ActionIR.

Converts raw tactic text from proof traces into the typed ActionIR
representation. This is the bridge between LeanDojo extraction and
the constrained output training pipeline.

Supports the 82% of residual tactics that are structurally parsable:
    rw [lemma1, ← lemma2, ...]
    exact term
    apply term
    simp [lemma1, ...] / simp only [...] / simpa [...] using term
    refine term_with_holes
"""

from __future__ import annotations

import re

from src.tactic_ir import (
    ActionIR,
    Direction,
    RewriteAtom,
    TermExpr,
    TermKind,
    const,
    hole,
    var,
)


def _parse_term(text: str) -> TermExpr:
    """Parse a term expression from tactic argument text.

    Handles:
        bare_name               → CONST
        h / hp / hx             → VAR (if lowercase single-segment)
        name arg1 arg2          → APP
        (expr).method           → CHAIN
        expr.field              → PROJ (.1, .2, .symm, .trans)
        _                       → HOLE
        ?_                      → HOLE (named)
        ⟨a, b, c⟩              → CTOR
    """
    text = text.strip()
    if not text:
        return hole()

    # Holes
    if text == "_":
        return hole()
    if text == "?_":
        return hole(named=True)

    # Anonymous constructor ⟨...⟩
    if text.startswith("⟨") and text.endswith("⟩"):
        inner = text[1:-1]
        parts = _split_top_level(inner, ",")
        return TermExpr(
            kind=TermKind.CTOR,
            args=[_parse_term(p) for p in parts],
        )

    # Parenthesized expression with method chain: (expr).method (args)
    chain_match = re.match(r"\((.+)\)\.(\w+)\s*(.*)", text, re.DOTALL)
    if chain_match:
        base = _parse_term(chain_match.group(1))
        method = chain_match.group(2)
        rest = chain_match.group(3).strip()
        args = [base]
        if rest:
            args.append(_parse_term(rest))
        return TermExpr(kind=TermKind.CHAIN, args=args, field_name=method)

    # Field projection: name.field (but not name.subname which is a qualified name)
    proj_match = re.match(r"(\w[\w\']*)\.(1|2|symm|trans|mp|mpr|left|right|elim)\b(.*)", text)
    if proj_match:
        base_name = proj_match.group(1)
        field_name = proj_match.group(2)
        rest = proj_match.group(3).strip()
        base = (
            var(base_name) if base_name[0].islower() and "." not in base_name else const(base_name)
        )
        result = TermExpr(kind=TermKind.PROJ, args=[base], field_name=field_name)
        if rest:
            # Chained: h.symm.trans ...
            return TermExpr(kind=TermKind.APP, head="", args=[result, _parse_term(rest)])
        return result

    # Application: head arg1 arg2 ...
    # Split on whitespace, but respect parentheses
    parts = _split_top_level(text, " ")
    if len(parts) == 1:
        name = parts[0]
        # Single token: VAR if looks like hypothesis, else CONST
        if name[0].islower() and len(name) <= 3 and "." not in name:
            return var(name)
        return const(name)

    # Multi-token: first is head, rest are args
    head = parts[0]
    args = [_parse_term(p) for p in parts[1:]]
    return TermExpr(kind=TermKind.APP, head=head, args=args)


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split text on separator, respecting parentheses and brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([⟨{":
            depth += 1
        elif char in ")]⟩}":
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    remainder = "".join(current).strip()
    if remainder:
        parts.append(remainder)
    return parts


def canonicalize(tactic_text: str, family: str) -> ActionIR | None:
    """Parse a tactic string into ActionIR.

    Returns None if the tactic cannot be parsed into the IR.
    """
    text = tactic_text.strip()

    if family == "rw":
        return _parse_rw(text)
    if family == "exact":
        return _parse_exact(text)
    if family == "apply":
        return _parse_apply(text)
    if family == "simp":
        return _parse_simp(text)
    if family == "refine":
        return _parse_refine(text)
    return None


def _parse_rw(text: str) -> ActionIR | None:
    """Parse rw [...] or rw [← ...] into ActionIR."""
    m = re.search(r"\[([^\]]*)\]", text)
    if not m:
        return None

    raw_atoms = _split_top_level(m.group(1), ",")
    rewrites: list[RewriteAtom] = []

    for atom_str in raw_atoms:
        atom_str = atom_str.strip()
        if not atom_str:
            continue
        if atom_str.startswith("←") or atom_str.startswith("← "):
            direction = Direction.BACKWARD
            expr_str = atom_str.lstrip("← ").strip()
        else:
            direction = Direction.FORWARD
            expr_str = atom_str

        rewrites.append(
            RewriteAtom(
                direction=direction,
                expr=_parse_term(expr_str),
            )
        )

    return ActionIR(family="rw", rewrites=rewrites)


def _parse_exact(text: str) -> ActionIR | None:
    """Parse exact <term>."""
    parts = text.split(None, 1)
    if len(parts) < 2:
        return ActionIR(family="exact")
    return ActionIR(family="exact", term=_parse_term(parts[1]))


def _parse_apply(text: str) -> ActionIR | None:
    """Parse apply <term>."""
    parts = text.split(None, 1)
    if len(parts) < 2:
        return ActionIR(family="apply")
    return ActionIR(family="apply", term=_parse_term(parts[1]))


def _parse_simp(text: str) -> ActionIR | None:
    """Parse simp/simpa [lemmas] [using term]."""
    family = "simpa" if text.startswith("simpa") else "simp"
    only = "only" in text.split("[")[0] if "[" in text else "only" in text

    # Extract [lemmas]
    lemmas: list[TermExpr] = []
    m = re.search(r"\[([^\]]*)\]", text)
    if m:
        raw_lemmas = _split_top_level(m.group(1), ",")
        lemmas = [_parse_term(l) for l in raw_lemmas if l.strip()]

    # Extract "using term"
    using_term = None
    using_match = re.search(r"\busing\s+(.+)$", text)
    if using_match:
        using_term = _parse_term(using_match.group(1))

    return ActionIR(
        family=family,
        simp_lemmas=lemmas,
        using_term=using_term,
        only=only,
    )


def _parse_refine(text: str) -> ActionIR | None:
    """Parse refine <term_with_holes>."""
    parts = text.split(None, 1)
    if len(parts) < 2:
        return ActionIR(family="refine")
    return ActionIR(family="refine", term=_parse_term(parts[1]))
