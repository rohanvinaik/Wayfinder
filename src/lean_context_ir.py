"""Lean source-context extraction for theorem-faithful wrapper generation.

This module provides a conservative parser for the subset of Lean source
context that matters for replayable local execution:

- namespace / section / noncomputable section scope stack
- open / open scoped
- variable / variables
- universe / universes
- local notation / scoped notation / notation-style declarations
- local attributes (notably [local instance] / [local simp])
- include / omit

The output is a small IR that can be rendered back into prefix/suffix wrapper
text and audited for unsupported forms. The goal is not to parse all of Lean;
it is to compile enough source-context structure that theorem-faithful start
states and replay stop depending on ad hoc header heuristics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_DIRECTIVE_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("open_scoped", re.compile(r"^open scoped\b")),
    ("open", re.compile(r"^open\b")),
    ("set_option", re.compile(r"^set_option\b")),
    ("variable", re.compile(r"^(?:variable|variables)\b")),
    ("universe", re.compile(r"^(?:universe|universes)\b")),
    ("local_macro", re.compile(r"^local\s+macro(?::[A-Za-z_][A-Za-z0-9_]*)?\b")),
    ("local_notation", re.compile(r"^local notation\b")),
    (
        "scoped_notation",
        re.compile(r"^(?:scoped\[|scoped\s+(?:notation|infix[lr]?|prefix|postfix|macro)\b)"),
    ),
    ("notation", re.compile(r"^(?:notation|infix[lr]?|prefix|postfix)\b")),
    (
        "local_attribute",
        re.compile(r"^(?:attribute\s+\[[^\]]*local[^\]]*\]|local\s+attribute\b|local\s+instance\b)"),
    ),
    ("include", re.compile(r"^include\b")),
    ("omit", re.compile(r"^omit\b")),
)

_DECL_MODIFIERS = r"(?:(?:noncomputable|private|protected)\s+)*"
_DECL_REGEX = re.compile(
    rf"^(?:@\[[^\]]+\]\s*)*{_DECL_MODIFIERS}(?:theorem|lemma|def|instance|abbrev|alias)\s+([^\s\(:\[]+)"
)


def _normalize_decl_name(name: str) -> str:
    """Normalize declaration names for fuzzy source matching.

    This deliberately ignores dots/underscores/case so that benchmark theorem
    ids remain recoverable across harmless spelling drift such as
    `foo_not_mem` vs `foo_notMem`.
    """
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()


def _decl_name_candidates(name: str) -> list[str]:
    """Return exact suffix candidates for a theorem full name."""
    if not name:
        return []
    parts = [part for part in name.split(".") if part]
    candidates: list[str] = []
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


@dataclass(frozen=True)
class ContextDirective:
    """One context-bearing source directive."""

    kind: str
    text: str
    line_no: int
    inline_only: bool = False
    renderable: bool = True
    reason: str = ""


@dataclass
class ContextFrame:
    """One active lexical scope frame."""

    kind: str
    name: str
    start_line: int
    directives: list[ContextDirective] = field(default_factory=list)


@dataclass
class ContextIR:
    """Active source context at a theorem declaration site."""

    lean_path: str
    theorem_line: int
    top_level_directives: list[ContextDirective]
    scope_stack: list[ContextFrame]
    inline_directives: list[ContextDirective] = field(default_factory=list)
    unsupported: list[ContextDirective] = field(default_factory=list)

    @property
    def prefix_lines(self) -> list[str]:
        """Scope-level context lines emitted before the theorem wrapper."""
        lines = [d.text for d in self.top_level_directives if d.renderable and not d.inline_only]
        for frame in self.scope_stack:
            if frame.kind == "namespace":
                lines.append(f"namespace {frame.name}")
            elif frame.kind == "public section":
                lines.append(f"public section {frame.name}" if frame.name else "public section")
            elif frame.kind == "noncomputable section":
                lines.append(
                    f"noncomputable section {frame.name}" if frame.name else "noncomputable section"
                )
            else:
                lines.append(f"section {frame.name}" if frame.name else "section")
            lines.extend(d.text for d in frame.directives if d.renderable and not d.inline_only)
        return lines

    @property
    def inline_lines(self) -> list[str]:
        """Inline `... in` directives emitted immediately before the theorem decl.

        These are forms like `open Classical in` or `variable (M) in` that
        apply only to the next declaration. They don't need matching `end`
        closers — the `in` keyword scopes them to the following declaration.
        """
        return [directive.text for directive in self.inline_directives]

    @property
    def suffix_lines(self) -> list[str]:
        """Scope-closing lines emitted after the theorem wrapper."""
        suffix: list[str] = []
        for frame in reversed(self.scope_stack):
            suffix.append(f"end {frame.name}" if frame.name else "end")
        return suffix

    @property
    def active_directives(self) -> list[ContextDirective]:
        directives = list(self.top_level_directives)
        for frame in self.scope_stack:
            directives.extend(frame.directives)
        return directives

    def feature_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for directive in self.active_directives:
            counts[directive.kind] += 1
        for directive in self.unsupported:
            counts[f"unsupported:{directive.kind}"] += 1
        return counts


def classify_context_directive(stripped: str) -> tuple[str | None, bool]:
    """Classify a stripped line as a context directive.

    Returns (kind, inline_only). `inline_only` marks `... in` declarations that
    apply only to the next declaration and should not be replayed as active scope.
    """
    inline_only = stripped.endswith(" in")
    for kind, pattern in _DIRECTIVE_REGEXES:
        if pattern.match(stripped):
            return kind, inline_only
    return None, False


def find_decl_line(lean_path: str | Path, short_name: str) -> int:
    """Find the first declaration line for a theorem/lemma/def name.

    Supports exact suffix matches and small benchmark-id spelling drift.
    """
    path = Path(lean_path)
    candidates = _decl_name_candidates(short_name)
    candidate_priority = {candidate: len(candidate) for candidate in candidates}
    normalized_priority = {
        _normalize_decl_name(candidate): len(candidate) for candidate in candidates
    }
    fallback_line = -1
    fallback_priority = -1
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            match = _DECL_REGEX.match(line.strip())
            if not match:
                continue
            decl_name = match.group(1)
            priority = candidate_priority.get(decl_name)
            if priority is None:
                priority = normalized_priority.get(_normalize_decl_name(decl_name), -1)
            if priority <= 0:
                continue
            if "." in decl_name:
                return line_no
            if priority > fallback_priority:
                fallback_line = line_no
                fallback_priority = priority
    return fallback_line


def extract_context_ir(lean_path: str | Path, theorem_line: int) -> ContextIR:
    """Extract active source context at a theorem line.

    This is intentionally conservative. Unsupported or ambiguous directives are
    recorded in `unsupported` so validation scripts can measure the remaining
    gap instead of silently dropping them.
    """
    path = Path(lean_path)
    with path.open() as handle:
        all_lines = handle.readlines()

    scope_stack: list[ContextFrame] = []
    top_level: list[ContextDirective] = []
    unsupported: list[ContextDirective] = []
    inline_directives: list[ContextDirective] = []

    inline_start = theorem_line - 1
    while inline_start > 0:
        stripped = all_lines[inline_start - 1].strip()
        if not stripped or stripped.startswith("--") or stripped.startswith("/-") or stripped == "-/":
            inline_start -= 1
            continue
        kind, inline_only = classify_context_directive(stripped)
        if kind and inline_only:
            inline_start -= 1
            continue
        break
    inline_line_numbers = set(range(inline_start + 1, theorem_line))

    for idx, raw in enumerate(all_lines[: max(theorem_line - 1, 0)], start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("--") or stripped.startswith("/-") or stripped == "-/":
            continue
        if stripped.startswith("import "):
            continue

        if stripped.startswith("namespace "):
            name = (
                stripped[len("namespace ") :].split()[0]
                if len(stripped) > len("namespace ")
                else ""
            )
            scope_stack.append(ContextFrame(kind="namespace", name=name, start_line=idx))
            continue

        if stripped == "section" or stripped.startswith("section "):
            rest = stripped[len("section") :].strip()
            scope_stack.append(
                ContextFrame(kind="section", name=rest.split()[0] if rest else "", start_line=idx)
            )
            continue

        # @[expose] public section / public section — Mathlib v4.27+ scope construct
        if "public section" in stripped:
            # Strip attribute prefix: @[expose] public section Name -> Name
            ps_match = re.search(r"public\s+section\s*(.*)", stripped)
            rest = ps_match.group(1).strip() if ps_match else ""
            scope_stack.append(
                ContextFrame(
                    kind="public section",
                    name=rest.split()[0] if rest else "",
                    start_line=idx,
                )
            )
            continue

        if stripped.startswith("noncomputable section"):
            rest = stripped[len("noncomputable section") :].strip()
            scope_stack.append(
                ContextFrame(
                    kind="noncomputable section",
                    name=rest.split()[0] if rest else "",
                    start_line=idx,
                )
            )
            continue

        if stripped == "end" or stripped.startswith("end "):
            close_name = stripped[len("end") :].strip() if len(stripped) > 3 else ""
            if scope_stack:
                if close_name:
                    while scope_stack and scope_stack[-1].name != close_name:
                        scope_stack.pop()
                    if scope_stack:
                        scope_stack.pop()
                else:
                    scope_stack.pop()
            continue

        kind, inline_only = classify_context_directive(stripped)
        if not kind:
            continue

        renderable = True
        reason = ""

        # Conservative multiline detection. If the next line is indented and the
        # current directive is notation/attribute/variable-like, treat it as
        # unsupported until the compiler handles block forms explicitly.
        if idx < theorem_line - 1:
            nxt = all_lines[idx]
            if nxt.startswith("  ") or nxt.startswith("\t"):
                if kind in {"local_notation", "notation", "local_attribute"}:
                    renderable = False
                    reason = reason or "probable_multiline"
                elif kind == "variable":
                    # Capture multiline variable blocks: continuation lines
                    # that start with [ or ( are typeclass/binder extensions
                    if nxt.strip().startswith("[") or nxt.strip().startswith("(") or nxt.strip().startswith("{"):
                        block_lines = [stripped]
                        j = idx
                        while j < theorem_line - 1:
                            cont = all_lines[j]
                            cont_s = cont.strip()
                            if (cont.startswith("  ") or cont.startswith("\t")) and (
                                cont_s.startswith("[") or cont_s.startswith("(") or cont_s.startswith("{")
                            ):
                                block_lines.append(cont.rstrip())
                                j += 1
                            else:
                                break
                        stripped = "\n".join(block_lines)
                    else:
                        renderable = False
                        reason = reason or "probable_multiline"
                elif kind == "local_macro":
                    block_lines = [stripped]
                    j = idx
                    while j < theorem_line - 1:
                        cont = all_lines[j]
                        if cont.startswith("  ") or cont.startswith("\t"):
                            block_lines.append(cont.rstrip())
                            j += 1
                            continue
                        break
                    stripped = "\n".join(block_lines)

        directive = ContextDirective(
            kind=kind,
            text=stripped,
            line_no=idx,
            inline_only=inline_only,
            renderable=renderable,
            reason=reason,
        )

        if inline_only:
            if idx in inline_line_numbers:
                inline_directives.append(directive)
            continue

        if not renderable:
            unsupported.append(directive)

        if scope_stack:
            scope_stack[-1].directives.append(directive)
        else:
            top_level.append(directive)

    return ContextIR(
        lean_path=str(path),
        theorem_line=theorem_line,
        top_level_directives=top_level,
        scope_stack=scope_stack,
        inline_directives=inline_directives,
        unsupported=unsupported,
    )
