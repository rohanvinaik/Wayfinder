"""LeanExpr — warranted Lean emission (THE_REFOUNDING §5, Wave 3).

The Detective `SourceExpr` homology carried to Lean: one object that is
simultaneously the live derivation lineage (the verified tactic
applications, in order) and the source that reconstructs it — with the two
fields Python emission never needed and Lean emission cannot do without:

- ``environment`` — a Lean artifact elaborates only *relative to* a
  toolchain + project pin. It is a first-class field, never
  reconstruction-time inference.
- ``warrant`` — every emitted artifact carries its provenance as a Lean
  block comment (the analogue of Detective's warrant docstring on every
  generated test): which lanes closed it, which engine/bank/certificate
  produced it, from which run. A reconstructed proof file is
  self-documenting evidence, not bare text.

``repr(LeanExpr)`` is the source seam, exactly as in ``SourceExpr``: any
render site that writes ``repr(e)`` emits the tactic block.

The emission is typed assembly with one boundary — the projector discipline
(Dr. Ducky §5.7) applied to the artifact layer: internal lineage on one
side, compilable Lean on the other, nothing crossing except through these
constructors. String luck is what this module retires; EXP-RF-006 measures
the difference at the v4.28 gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_INDENT = "  "


@dataclass(frozen=True)
class EmissionWarrant:
    """The provenance an emitted artifact carries — engine-side facts only."""

    theorem_id: str
    provenance: tuple[str, ...] = ()  # lane provenance of the closing sequence, in order
    engine: str = ""  # Ducky engine name, when the emission came through the VM
    bank: str = ""
    certificate_id: str = ""
    skeleton_id: str = ""
    source_run: str = ""  # the run/session the lineage was verified in
    note: str = ""


@dataclass(frozen=True)
class LeanExpr:
    """A dual live/source Lean emission object.

    ``value`` is the lineage — the tactic applications exactly as verified
    against the kernel, in order. ``expr`` is the rendered tactic block that
    reconstructs the derivation. The two are built together by the
    constructors below so they cannot drift; consumers replay ``value`` and
    emit ``expr``.
    """

    expr: str  # the rendered `by`-block body (the SOURCE)
    value: tuple[str, ...]  # the verified tactic lineage (the LIVE side)
    imports: tuple[str, ...] = ()
    opens: tuple[str, ...] = ()
    environment: str = ""  # toolchain + project pin the expr elaborates against
    warrant: EmissionWarrant = field(default_factory=lambda: EmissionWarrant(theorem_id=""))

    def __repr__(self) -> str:  # the source seam: repr(e) -> tactic block
        return self.expr


def tactic_block(tactics: tuple[str, ...] | list[str], indent: int = 1) -> str:
    """Render a tactic lineage as an indented `by`-block body.

    An empty lineage renders the explicit ``sorry`` placeholder — emission
    never invents content, and the sorryAx gate downstream rejects it, so
    the placeholder is visible, never silently load-bearing.
    """
    pad = _INDENT * indent
    if not tactics:
        return f"{pad}sorry"
    return "\n".join(f"{pad}{t}" for t in tactics)


def warrant_header(warrant: EmissionWarrant, environment: str) -> str:
    """The warrant as a Lean block comment — the emitted artifact's evidence."""
    rows = [f"theorem_id: {warrant.theorem_id}"]
    if environment:
        rows.append(f"environment: {environment}")
    if warrant.provenance:
        rows.append(f"provenance: {' -> '.join(warrant.provenance)}")
    for label, value in (
        ("engine", warrant.engine),
        ("bank", warrant.bank),
        ("certificate", warrant.certificate_id),
        ("skeleton", warrant.skeleton_id),
        ("source_run", warrant.source_run),
        ("note", warrant.note),
    ):
        if value:
            rows.append(f"{label}: {value}")
    body = "\n".join(f"  {row}" for row in rows)
    return f"/- wayfinder warrant\n{body}\n-/"


def render_declaration(decl_head: str, expr: LeanExpr, include_warrant: bool = True) -> str:
    """The full compilable artifact: warrant, imports, opens, head, body.

    ``decl_head`` is the declaration text up to and including ``:= by``
    (caller-owned — the head belongs to the theorem row, not the emission).
    """
    sections = []
    if include_warrant:
        sections.append(warrant_header(expr.warrant, expr.environment))
    if expr.imports:
        sections.append("\n".join(expr.imports))
    if expr.opens:
        sections.append("\n".join(expr.opens))
    sections.append(f"{decl_head}\n{expr.expr}")
    return "\n\n".join(sections) + "\n"


def from_tactics(
    tactics: list[str] | tuple[str, ...],
    warrant: EmissionWarrant,
    imports: tuple[str, ...] = (),
    opens: tuple[str, ...] = (),
    environment: str = "",
) -> LeanExpr:
    """Build a LeanExpr from a verified tactic lineage (the search-result seam)."""
    lineage = tuple(tactics)
    return LeanExpr(
        expr=tactic_block(lineage),
        value=lineage,
        imports=tuple(imports),
        opens=tuple(opens),
        environment=environment,
        warrant=warrant,
    )


def from_ducky_program(
    program,
    theorem_id: str,
    imports: tuple[str, ...] = (),
    opens: tuple[str, ...] = (),
    environment: str = "",
    source_run: str = "",
) -> LeanExpr:
    """Build a LeanExpr from a ProjectedProofProgram (the projector seam).

    Direct attribute access on purpose — a renamed field must break loudly
    here, never be absorbed by a default (the Detective renderer lesson).
    """
    warrant = EmissionWarrant(
        theorem_id=theorem_id,
        provenance=("dr_ducky",),
        engine=program.specialist,
        bank=program.bank,
        certificate_id=program.certificate_id,
        skeleton_id=program.skeleton_id,
        source_run=source_run,
        note=program.rationale,
    )
    return from_tactics(
        list(program.tactics), warrant, imports=imports, opens=opens, environment=environment
    )


def read_environment_pin(project_dir: str | Path) -> str:
    """The environment string for a Lean project: its toolchain pin + path.

    The one impure helper in this module. A missing toolchain file yields an
    explicit ``unpinned:`` marker rather than a guess — an artifact emitted
    against an unknown environment must say so.
    """
    project = Path(project_dir)
    toolchain_file = project / "lean-toolchain"
    if toolchain_file.exists():
        toolchain = toolchain_file.read_text().strip()
        return f"{toolchain} @ {project.name}"
    return f"unpinned: @ {project.name}"
