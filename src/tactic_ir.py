"""Tactic intermediate representation — typed AST between family prediction and Lean syntax.

The neural system produces family + constrained selections from local vocabulary.
This IR captures the semantic structure. Deterministic lowering compiles to Lean.

Three regimes (v1 supports 1 and 2):
    1. Template-only: bare premise names + direction flags
    2. Local term synthesis: application, projection, holes
    3. Micro-programs: tactic combinators, sequences (deferred)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class TermKind(Enum):
    """Kind of term expression in the IR."""

    VAR = "var"  # local hypothesis (h, hp, hx)
    CONST = "const"  # premise / lemma name (mul_comm, le_antisymm)
    APP = "app"  # function application: head arg1 arg2
    PROJ = "proj"  # field projection: term.field (.1, .2, .trans)
    HOLE = "hole"  # placeholder: _ or ?_
    CHAIN = "chain"  # method chaining: a.trans b
    CTOR = "ctor"  # anonymous constructor: ⟨a, b⟩


@dataclass
class TermExpr:
    """A term expression in the tactic IR."""

    kind: TermKind
    head: str = ""  # name for VAR/CONST, empty for APP/HOLE
    args: list[TermExpr] = field(default_factory=list)  # children for APP/CTOR
    field_name: str = ""  # for PROJ: .1, .2, .trans, .symm

    def lower(self) -> str:
        """Compile to Lean syntax."""
        if self.kind == TermKind.VAR:
            return self.head
        if self.kind == TermKind.CONST:
            return self.head
        if self.kind == TermKind.HOLE:
            return "?_" if self.head == "?" else "_"
        if self.kind == TermKind.APP:
            parts = [self.head] + [a.lower() for a in self.args]
            return " ".join(parts)
        if self.kind == TermKind.PROJ:
            base = self.args[0].lower() if self.args else ""
            return f"{base}.{self.field_name}"
        if self.kind == TermKind.CHAIN:
            base = self.args[0].lower() if self.args else ""
            method = self.field_name
            chain_args = " ".join(a.lower() for a in self.args[1:])
            if chain_args:
                return f"({base}).{method} ({chain_args})"
            return f"({base}).{method}"
        if self.kind == TermKind.CTOR:
            inner = ", ".join(a.lower() for a in self.args)
            return f"⟨{inner}⟩"
        return self.head

    def iter_var_names(self) -> Iterator[str]:
        """Yield local variable names referenced inside this term."""
        if self.kind == TermKind.VAR and self.head:
            yield self.head
        for arg in self.args:
            yield from arg.iter_var_names()

    def iter_const_names(self) -> Iterator[str]:
        """Yield global constant names referenced inside this term."""
        if self.kind == TermKind.CONST and self.head:
            yield self.head
        if self.kind == TermKind.APP and self.head:
            yield self.head
        for arg in self.args:
            yield from arg.iter_const_names()


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class RewriteAtom:
    """A single rewrite directive: direction + expression."""

    direction: Direction
    expr: TermExpr

    def lower(self) -> str:
        prefix = "← " if self.direction == Direction.BACKWARD else ""
        return f"{prefix}{self.expr.lower()}"


@dataclass
class ActionIR:
    """Typed intermediate representation of a tactic application.

    This is the output of the family-specific decoder.
    Deterministic lowering compiles it to a Lean tactic string.
    """

    family: str  # rw | simp | exact | apply | refine

    # rw-specific
    rewrites: list[RewriteAtom] = field(default_factory=list)

    # exact / apply / refine
    term: TermExpr | None = None

    # simp-specific
    simp_lemmas: list[TermExpr] = field(default_factory=list)
    using_term: TermExpr | None = None
    only: bool = False

    def lower(self) -> str:
        """Compile to Lean tactic string."""
        if self.family == "rw":
            if not self.rewrites:
                return "rw []"
            atoms = ", ".join(r.lower() for r in self.rewrites)
            return f"rw [{atoms}]"

        if self.family in ("exact", "apply", "refine"):
            if self.term is None:
                return self.family
            return f"{self.family} {self.term.lower()}"

        if self.family == "simp":
            base = "simp only" if self.only else "simp"
            if self.simp_lemmas:
                lemmas = ", ".join(l.lower() for l in self.simp_lemmas)
                base = f"{base} [{lemmas}]"
            if self.using_term:
                base = f"{base} using {self.using_term.lower()}"
            return base

        if self.family == "simpa":
            base = "simpa only" if self.only else "simpa"
            if self.simp_lemmas:
                lemmas = ", ".join(l.lower() for l in self.simp_lemmas)
                base = f"{base} [{lemmas}]"
            if self.using_term:
                base = f"{base} using {self.using_term.lower()}"
            return base

        return self.family

    def local_var_names(self) -> list[str]:
        """Collect local variable names referenced by this action."""
        seen: list[str] = []

        def _push(name: str) -> None:
            if name and name not in seen:
                seen.append(name)

        for rewrite in self.rewrites:
            for name in rewrite.expr.iter_var_names():
                _push(name)
        if self.term is not None:
            for name in self.term.iter_var_names():
                _push(name)
        for lemma in self.simp_lemmas:
            for name in lemma.iter_var_names():
                _push(name)
        if self.using_term is not None:
            for name in self.using_term.iter_var_names():
                _push(name)
        return seen

    def const_names(self) -> list[str]:
        """Collect constant/premise names referenced by this action."""
        seen: list[str] = []

        def _push(name: str) -> None:
            if name and name not in seen:
                seen.append(name)

        for rewrite in self.rewrites:
            for name in rewrite.expr.iter_const_names():
                _push(name)
        if self.term is not None:
            for name in self.term.iter_const_names():
                _push(name)
        for lemma in self.simp_lemmas:
            for name in lemma.iter_const_names():
                _push(name)
        if self.using_term is not None:
            for name in self.using_term.iter_const_names():
                _push(name)
        return seen

    def primary_premise_name(self) -> str:
        """Best-effort primary global symbol for this action."""
        if self.family == "rw" and self.rewrites:
            names = list(self.rewrites[0].expr.iter_const_names())
            return names[0] if names else ""
        if self.family in ("apply", "exact", "refine") and self.term is not None:
            names = list(self.term.iter_const_names())
            return names[0] if names else ""
        if self.family in ("simp", "simpa") and self.simp_lemmas:
            names = list(self.simp_lemmas[0].iter_const_names())
            return names[0] if names else ""
        names = self.const_names()
        return names[0] if names else ""


# --- Convenience constructors ---


def var(name: str) -> TermExpr:
    """Local hypothesis reference."""
    return TermExpr(kind=TermKind.VAR, head=name)


def const(name: str) -> TermExpr:
    """Premise / lemma constant."""
    return TermExpr(kind=TermKind.CONST, head=name)


def app(head: str, *args: TermExpr) -> TermExpr:
    """Function application."""
    return TermExpr(kind=TermKind.APP, head=head, args=list(args))


def proj(base: TermExpr, field_name: str) -> TermExpr:
    """Field projection: base.field."""
    return TermExpr(kind=TermKind.PROJ, args=[base], field_name=field_name)


def hole(named: bool = False) -> TermExpr:
    """Placeholder: _ or ?_."""
    return TermExpr(kind=TermKind.HOLE, head="?" if named else "_")


def ctor(*args: TermExpr) -> TermExpr:
    """Anonymous constructor ⟨a, b, ...⟩."""
    return TermExpr(kind=TermKind.CTOR, args=list(args))


def chain(base: TermExpr, method: str, *args: TermExpr) -> TermExpr:
    """Method chaining: base.method(args)."""
    return TermExpr(kind=TermKind.CHAIN, args=[base, *args], field_name=method)


def rw_forward(premise: str) -> RewriteAtom:
    """Forward rewrite atom."""
    return RewriteAtom(direction=Direction.FORWARD, expr=const(premise))


def rw_backward(premise: str) -> RewriteAtom:
    """Backward rewrite atom."""
    return RewriteAtom(direction=Direction.BACKWARD, expr=const(premise))
