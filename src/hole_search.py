"""Wayfinder's search on the holes of a residual, where they sit.

A hole is searched from the goal Lean has at that `sorry` — its own hypotheses, its place in the
proof — never from a lifted lemma. To get that goal, the hand-off file is loaded with only that
hole left as `sorry` and every other hole closed by a placeholder axiom: Lean then returns exactly
one goal, the hole's. (A hand-off's holes are propositions — the residual builder refuses data
holes — so how the other holes are closed cannot change this one's context.)

What the search builds comes back as a tactic block for the hole: a complete proof when it closed
the goal, otherwise its committed steps with a marked `sorry` at each goal still open — partial
progress kept, not thrown away.
"""

from __future__ import annotations

import re
import textwrap

from src import hole_fill as hf
from src import residual_handoff as rh

PLACEHOLDER = "wf_hole_placeholder"
PLACEHOLDER_AXIOM = f"axiom {PLACEHOLDER}.{{u}} {{α : Sort u}} : α"


def only_hole(declaration: str, k: int) -> str:
    """The declaration with every marked hole but `k` closed by the placeholder axiom."""
    lines = declaration.split("\n")
    for j, site in hf.hole_sites(declaration).items():
        if j != k:
            lines[site.sorry] = site.indent + ("· " if site.bullet else "") + f"exact {PLACEHOLDER}"
    return "\n".join(lines)


def _without_imports(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("import "))


def hole_source(file_text: str, declaration: str, k: int) -> str:
    """The source to load for hole `k`: the hand-off file (imports dropped — the server has them),
    the placeholder axiom above its declaration, the declaration with hole `k` its only `sorry`."""
    at = file_text.rindex(declaration)
    above = _without_imports(file_text[:at]).rstrip("\n")
    below = file_text[at + len(declaration) :]
    return f"{above}\n\n{PLACEHOLDER_AXIOM}\n\n{only_hole(declaration, k)}{below}"


def withdrawn(tree: rh.Tree) -> set[int]:
    """The committing steps that must not stand: each with a goal still open under it (withdrawing
    one makes its own goal a hole, which can leave a commitment above it open in turn)."""
    forced: set[int] = set()
    while True:
        holes = [i for i, n in enumerate(tree.nodes) if n.closure < 0 or n.tactic is None]
        more = rh.unvalidated_commitments(tree, holes + sorted(forced), forced)
        if not more:
            return forced
        forced |= more


def hole_block(initial_goals: list, initial_ids: list | None, close_log: list[dict]) -> tuple[str, int]:
    """The search's committed tree from the hole's goal, as the tactic block that goes in the hole,
    and how many goals it leaves open (each a marked `sorry` with its goal under it). Validated
    progress only: a commitment with a goal open under it is withdrawn, what it tried noted above
    its hole."""
    tree = rh.build_tree(list(initial_goals), list(close_log), initial_ids)
    forced = withdrawn(tree)
    tried = {nid: rh.attempted(tree, nid) for nid in forced}
    rendered = rh.render(tree, forced_holes=forced, tried=tried)
    block = textwrap.dedent("\n".join(rendered.body)).strip("\n")
    return block, len(rendered.holes)


def splice(declaration: str, k: int, block: str) -> str:
    """The declaration with hole `k` replaced by `block` (its own holes renumbered with the rest)."""
    n = len(hf.hole_sites(declaration))
    fills: list[str | None] = [None] * n
    fills[k - 1] = block
    text = hf.fill_holes(declaration, fills)
    return hf.renumber_holes(text)[0]


def open_holes(declaration: str) -> int:
    return len(hf.hole_sites(declaration))


def probe_declaration(declaration: str) -> tuple[str, dict[int, int]]:
    """The declaration with `trace_state` just before each hole's `sorry` (Lean prints the goal it
    has there), and which 0-based line of it reports which hole."""
    lines = declaration.split("\n")
    sites = {site.sorry: (k, site) for k, site in hf.hole_sites(declaration).items()}
    out: list[str] = []
    at: dict[int, int] = {}
    for i, line in enumerate(lines):
        if i in sites:
            k, site = sites[i]
            at[len(out)] = k
            if site.bullet:
                out += [f"{site.indent}· trace_state", f"{site.indent}  sorry"]
            else:
                out += [f"{site.indent}trace_state", f"{site.indent}sorry"]
            continue
        out.append(line)
    return "\n".join(out), at


def with_goals(declaration: str, goals: dict[int, str]) -> str:
    """Each hole's goal comment lines replaced by the goal Lean printed there (`goals` by hole)."""
    lines = declaration.split("\n")
    for k, site in sorted(hf.hole_sites(declaration).items(), reverse=True):
        if k not in goals:
            continue
        pad = lines[site.marker][: len(lines[site.marker]) - len(lines[site.marker].lstrip())]
        lines[site.marker + 1 : site.sorry] = [f"{pad}--   {g}" for g in goals[k].strip("\n").split("\n")]
    return "\n".join(lines)


_SORRY_TOKEN = re.compile(r"\bsorry\b")


def unmarked_sorries(declaration: str) -> int:
    """`sorry`s that are not marked holes (each marked hole has exactly one)."""
    return len(_SORRY_TOKEN.findall(declaration)) - open_holes(declaration)
