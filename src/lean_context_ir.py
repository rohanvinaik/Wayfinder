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

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re


_DIRECTIVE_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("open_scoped", re.compile(r"^open scoped\b")),
    ("open", re.compile(r"^open\b")),
    ("variable", re.compile(r"^(?:variable|variables)\b")),
    ("universe", re.compile(r"^(?:universe|universes)\b")),
    ("local_notation", re.compile(r"^local notation\b")),
    ("scoped_notation", re.compile(r"^scoped\[")),
    ("notation", re.compile(r"^(?:notation|infix[lr]?|prefix|postfix)\b")),
    ("local_attribute", re.compile(r"^attribute\s+\[[^\]]*local[^\]]*\]")),
    ("include", re.compile(r"^include\b")),
    ("omit", re.compile(r"^omit\b")),
)

_DECL_REGEX = re.compile(r"^(?:theorem|lemma|def|instance)\s+([^\s\(:\[]+)")


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
    unsupported: list[ContextDirective] = field(default_factory=list)

    @property
    def prefix_lines(self) -> list[str]:
        """Scope-level context lines emitted before the theorem wrapper."""
        lines = [d.text for d in self.top_level_directives if d.renderable and not d.inline_only]
        for frame in self.scope_stack:
            if frame.kind == "namespace":
                lines.append(f"namespace {frame.name}")
            elif frame.kind == "noncomputable section":
                lines.append(
                    f"noncomputable section {frame.name}" if frame.name else "noncomputable section"
                )
            else:
                lines.append(f"section {frame.name}" if frame.name else "section")
            lines.extend(
                d.text for d in frame.directives if d.renderable and not d.inline_only
            )
        return lines

    @property
    def inline_lines(self) -> list[str]:
        """Inline `... in` directives emitted immediately before the theorem decl.

        These are forms like `open Classical in` or `variable (M) in` that
        apply only to the next declaration. They don't need matching `end`
        closers — the `in` keyword scopes them to the following declaration.
        """
        lines: list[str] = []
        all_directives = list(self.top_level_directives)
        for frame in self.scope_stack:
            all_directives.extend(frame.directives)
        for d in all_directives:
            if d.inline_only:
                lines.append(d.text)
        return lines

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
    """Find the first declaration line for a short theorem/lemma/def name."""
    path = Path(lean_path)
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            match = _DECL_REGEX.match(line.strip())
            if match and match.group(1) == short_name:
                return line_no
    return -1


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

    for idx, raw in enumerate(all_lines[: max(theorem_line - 1, 0)], start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("--") or stripped.startswith("/-") or stripped == "-/":
            continue
        if stripped.startswith("import "):
            continue

        if stripped.startswith("namespace "):
            name = stripped[len("namespace ") :].split()[0] if len(stripped) > len("namespace ") else ""
            scope_stack.append(ContextFrame(kind="namespace", name=name, start_line=idx))
            continue

        if stripped == "section" or stripped.startswith("section "):
            rest = stripped[len("section") :].strip()
            scope_stack.append(ContextFrame(kind="section", name=rest.split()[0] if rest else "", start_line=idx))
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

        if inline_only:
            renderable = False
            reason = "inline_next_decl_only"

        # Conservative multiline detection. If the next line is indented and the
        # current directive is notation/attribute/variable-like, treat it as
        # unsupported until the compiler handles block forms explicitly.
        if idx < theorem_line - 1:
            nxt = all_lines[idx]
            if nxt.startswith("  ") or nxt.startswith("\t"):
                if kind in {"variable", "local_notation", "notation", "local_attribute"}:
                    renderable = False
                    reason = reason or "probable_multiline"

        directive = ContextDirective(
            kind=kind,
            text=stripped,
            line_no=idx,
            inline_only=inline_only,
            renderable=renderable,
            reason=reason,
        )

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
        unsupported=unsupported,
    )
