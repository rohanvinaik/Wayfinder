"""Filling the residual's holes — one hole at a time, each fill checked in place before it is kept.

The hand-off skeleton (src/residual_handoff.py) marks each goal Wayfinder left open with
`-- WAYFINDER RESIDUAL k: close this goal`, the goal Lean printed there, and a `sorry`. Any
prover may propose a fill for hole k: tactic text that replaces that `sorry`, plus optional
helper declarations placed above the theorem. A fill is kept only if the file still compiles
with it, the theorem's elaborated statement is unchanged, and nothing in it runs code at
elaboration time or reaches past the kernel. Once the last hole is filled, the whole file goes
through the acceptance gate (src/proof_gate.py).

For a model, a hole is lifted to a standalone lemma: `extract_goal *` prints the goal with its
whole context as a theorem Lean re-parses, inaccessible names (`h✝`) given accessible ones in
context order. A proof of the lifted lemma goes back in place behind a `rename_i` that gives the
hole those same names, or, when the two contexts do not line up, as a helper lemma that the hole
applies.

Pure stdlib (the MCP host and the local-prover worker both import it): text in, text out, and
Lean's `--json` messages read back. The callers compile.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field

import src.proof_gate as gate
from src.residual_handoff import HOLE_MARK, TRIED_MARK

_MARKER = re.compile(r"^(?P<indent>\s*)-- " + re.escape(HOLE_MARK) + r" (?P<k>\d+): close this goal\s*$")
_GOAL_COMMENT = re.compile(r"^\s*--   ")
_SORRY = re.compile(r"^(?P<indent>\s*)(?P<bullet>· )?sorry\s*$")


@dataclass(frozen=True)
class HoleSite:
    index: int  # k of `-- WAYFINDER RESIDUAL k`
    marker: int  # 0-based line of the marker in the declaration
    sorry: int  # 0-based line of its `sorry`
    indent: str  # indentation of the `sorry` line
    bullet: bool  # the `sorry` opens a `· ` bullet


def hole_sites(declaration: str) -> dict[int, HoleSite]:
    """Every open hole of a hand-off declaration, by its index."""
    lines = declaration.splitlines()
    out: dict[int, HoleSite] = {}
    i = 0
    while i < len(lines):
        m = _MARKER.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and _GOAL_COMMENT.match(lines[j]):
            j += 1
        s = _SORRY.match(lines[j]) if j < len(lines) else None
        if s is None:
            i += 1
            continue
        k = int(m.group("k"))
        out[k] = HoleSite(index=k, marker=i, sorry=j, indent=s.group("indent"), bullet=bool(s.group("bullet")))
        i = j + 1
    return out


# ── putting a fill in place ───────────────────────────────────────────────────


def fill_lines(fill: str) -> list[str]:
    """A fill's tactic lines, dedented as a block, blank lines dropped."""
    text = textwrap.dedent(fill.expandtabs(2).strip("\n"))
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def splice(declaration: str, k: int, fill: str, note: str = "") -> str:
    """The declaration with hole k's `sorry` replaced by `fill`, its marker recording the fill.

    Under a `· ` bullet the fill's first line takes the bullet and the rest sit two columns in,
    in the bullet's body; elsewhere every line takes the hole's indentation. The hole's goal
    comment goes; the marker stays, saying who filled it.
    """
    site = hole_sites(declaration).get(k)
    if site is None:
        raise KeyError(f"no open hole {k}")
    body = fill_lines(fill)
    if not body:
        raise ValueError("empty fill")
    lines = declaration.splitlines()
    if site.bullet:
        new = [f"{site.indent}· {body[0]}"] + [f"{site.indent}  {ln}" for ln in body[1:]]
    else:
        new = [f"{site.indent}{ln}" for ln in body]
    mark_indent = re.match(r"^(\s*)", lines[site.marker]).group(1)  # type: ignore[union-attr]
    marker = f"{mark_indent}-- {HOLE_MARK} {k}: filled" + (f" by {note}" if note else "")
    out = lines[: site.marker] + [marker] + new + lines[site.sorry + 1 :]
    return "\n".join(out) + ("\n" if declaration.endswith("\n") else "")


def apply_fill(
    file_text: str, declaration: str, k: int, fill: str, helpers: str = "", note: str = ""
) -> tuple[str, str]:
    """(file, declaration) with hole k filled and `helpers` placed directly above the theorem."""
    new_decl = splice(declaration, k, fill, note)
    at = file_text.rindex(declaration)
    block = helpers.strip("\n")
    head = file_text[:at] + (block + "\n\n" if block.strip() else "")
    return head + new_decl + file_text[at + len(declaration) :], new_decl


# ── what a fill may never contain ─────────────────────────────────────────────

# Anywhere in a fill or helper: placeholders, code run at elaboration time, the parser
# extensions that would let later text run code, kernel bypasses, and the library searches
# whose answer is not part of the proof. A command keyword is denied even mid-line: after a
# parse error Lean resumes at the next command token and runs it.
_DENY_TOKENS = (
    ("sorry", "admit", "sorryAx", "native_decide", "ofReduceBool", "Lean.ofReduceBool", "run_tac", "by_elab",
     "exact?", "apply?", "rw?", "unsafe", "implemented_by", "@[extern", "@[implemented_by", "axiom", "opaque",
     "import", "#exit", "#print", "#check", "#reduce", "#lint", "#synth", "#guard_msgs", "#help", "#find",
     "#simp", "#norm_num", "#conv", "#where")
    + gate._EXECUTING_COMMANDS
    + gate._SYNTAX_COMMANDS
)  # fmt: skip
# Declaration-level commands: denied at the start of a fill line (they would end the proof).
# `open … in` and `set_option … in` have tactic forms and are allowed.
_LINE_COMMANDS = tuple(c for c in gate._COMMANDS if c not in ("open", "set_option"))
# `set_option` names a fill or helper may set; everything else (`debug.skipKernelTC` above all)
# is refused.
_SAFE_OPTIONS = (
    "maxHeartbeats", "maxRecDepth", "synthInstance.", "pp.", "linter.", "trace.", "simprocs", "aesop.",
    "push_neg.", "tactic.", "Elab.async", "autoImplicit", "relaxedAutoImplicit", "warn.",
)  # fmt: skip
_SET_OPTION = re.compile(r"\bset_option\s+([\w.]+)")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[#@]?\[?[\w.'!?]+", gate.strip_comments(text))


def _option_findings(text: str) -> list[str]:
    out = []
    for name in _SET_OPTION.findall(gate.strip_comments(text)):
        if not name.startswith(_SAFE_OPTIONS):
            out.append(f"`set_option {name}` is not an option a proof may set")
    return out


def fill_findings(fill: str) -> list[str]:
    """Why a fill must not be compiled, as named findings (empty = clean)."""
    findings = [f"`{t}` is not allowed in a fill" for t in dict.fromkeys(_tokens(fill)) if t in _DENY_TOKENS]
    for line in gate.strip_comments(fill).splitlines():
        head = line.strip().removeprefix("· ").split(None, 1)
        if head and head[0] in _LINE_COMMANDS:
            findings.append(f"a fill line starts the command `{head[0]}`")
    return findings + _option_findings(fill)


def helper_findings(helpers: str) -> list[str]:
    """Why helper declarations must not be compiled (empty = clean).

    Helpers are top-level declarations (theorem / lemma / def / abbrev, optionally scoped by a
    `set_option … in` or `open … in`); the gate's static check runs on each one.
    """
    if not helpers.strip():
        return []
    findings = list(gate.static_findings(helpers, ""))
    findings += [f"`{t}` is not allowed in a helper" for t in dict.fromkeys(_tokens(helpers))
                 if t in _DENY_TOKENS and f"`{t}` remains" not in findings]  # fmt: skip
    for chunk in gate.top_level_chunks(helpers):
        if _SCOPE_PREFIX.match(chunk):
            continue
        kind = gate.declaration_kind(chunk)
        if kind not in ("theorem", "lemma", "def", "abbrev") and not chunk.startswith("universe "):
            findings.append(f"helpers may only declare theorems, lemmas and definitions, not `{chunk.split()[0]}`")
    return findings + _option_findings(helpers)


# In a model's whole answer: what runs code or rewrites the parser. Anything milder (`#check`, a
# stray `import`) is dropped with the commands around the theorem, never compiled.
_CODE_DENY = ("run_tac", "by_elab", "unsafe", "implemented_by", "@[extern", "@[implemented_by") + (
    gate._EXECUTING_COMMANDS + gate._SYNTAX_COMMANDS
)


def code_findings(code: str) -> list[str]:
    """Why a model's code block is refused before anything from it is compiled (empty = clean)."""
    findings = [f"`{t}` is not allowed" for t in dict.fromkeys(_tokens(code)) if t in _CODE_DENY]
    return findings + _option_findings(code)


def declared_names(text: str) -> set[str]:
    """The names a Lean text declares at top level."""
    return {gate.declaration_name(ch).rstrip(".") for ch in gate.top_level_chunks(text) if gate.declaration_kind(ch)}


# ── reading a filled file back ────────────────────────────────────────────────


def partial_findings(
    messages: list[dict],
    target: str,
    submitted_type: str | None,
    allowed_axioms: tuple[str, ...] = (),
    holes_left: int = 0,
) -> list[str]:
    """Why a hand-off file with a fill in place is not acceptable (empty = acceptable).

    The file was compiled with `gate.probe_suffix(target)`. It must elaborate with no error,
    the theorem's elaborated statement must be the hand-off's, and its axioms must be the
    standard three (plus the row's allowed ones, plus `sorryAx` while other holes are open).
    """
    reasons = [f"error {e}" for e in gate.errors_in(messages)[:5]]
    got = gate.checked_type(messages, target)
    if submitted_type is None:
        reasons.append("no recorded statement type to compare against")
    elif got is None or got != " ".join(submitted_type.split()):
        reasons.append("the theorem's elaborated statement differs from the hand-off's")
    axioms = gate.axioms_from_messages(messages, target)
    allowed = set(gate.STANDARD_AXIOMS) | set(allowed_axioms) | ({"sorryAx"} if holes_left else set())
    if axioms is None:
        reasons.append("`#print axioms` produced no report")
    else:
        extra = [a for a in axioms if a not in allowed]
        if extra:
            reasons.append(f"non-allowlisted axioms: {', '.join(extra)}")
    if not holes_left and gate.uses_sorry(messages):
        reasons.append("Lean reports `declaration uses sorry`")
    return reasons


def error_report(code: str, messages: list[dict], limit: int = 8) -> str:
    """Lean's errors as Goedel-Prover-V2's pipeline shows them to the model: each error's code,
    four lines before it and one after, the span between `<error>` and `</error>`, then the
    message — the format its self-correction rounds were trained on."""
    errors = [m for m in messages if m.get("severity") == "error"]
    lines = code.split("\n")
    out = ""
    for i, e in enumerate(errors[:limit]):
        pos = e.get("pos") or {}
        start_line = min(max(int(pos.get("line", 1)) - 1, 0), len(lines) - 1)
        start_col = int(pos.get("column", 0))
        end = e.get("endPos")
        if end is None:
            end_line, end_col = start_line, len(lines[start_line])
        else:
            end_line = min(max(int(end.get("line", 1)) - 1, start_line), len(lines) - 1)
            end_col = int(end.get("column", 0))
        out += f"\nError {i + 1}:\n"
        out += "\nCorresponding Code:\n```lean4\n"
        snippet = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                snippet += f"{lines[start_line + ii]}\n"
        if start_line != end_line:
            snippet += lines[start_line][:start_col] + "<error>" + lines[start_line][start_col:] + "\n"
            show = 6
            j = start_line
            for j in range(start_line + 1, min(end_line, start_line + show)):
                snippet += f"{lines[j]}\n"
            if end_line > start_line + show:
                lead = len(lines[j]) - len(lines[j].lstrip(" "))
                snippet += "\n" + " " * lead + "... --[Truncated]-- ...\n"
            snippet += lines[end_line][:end_col] + "</error>" + lines[end_line][end_col:] + "\n"
        else:
            row = lines[start_line]
            snippet += row[:start_col] + "<error>" + row[start_col:end_col] + "</error>" + row[end_col:] + "\n"
        if end_line + 1 < len(lines):
            snippet += f"{lines[end_line + 1]}\n"
        out += snippet
        out += "\n```\n"
        out += f"\nError Message: {e.get('data', '')}\n"
    if len(errors) > limit:
        out += f"\n... [Omitted {len(errors) - limit} more errors] ...\n"
    return out


# ── lifting a hole to a standalone lemma ──────────────────────────────────────


def lifted_name(target: str, k: int) -> str:
    return f"{target}_wf_hole_{k}"


# The second print of each hole: numerals, binders and coercions with their types written out. The
# plain print reads better but can re-elaborate differently — `Finset.Icc 1 35` over a goal whose
# index was ℕ re-reads at ℝ — or not at all: `↑n / ↑N ^ (3 : ℝ)⁻¹ < …` with nothing to say the casts
# land in ℝ fails to find `HPow ℕ ℝ _` (amc12a_2021_p25), so it is used only when it elaborates back.
_TYPED = (
    "set_option pp.numericTypes true in",
    "set_option pp.funBinderTypes true in",
    "set_option pp.coercions.types true in",
)


def lifting_probe(declaration: str, target: str) -> str:
    """The declaration with `extract_goal *` before every open hole's `sorry`, twice — plain, then
    with types written out (`<name>_r`); compiled, Lean prints each hole as a standalone theorem."""
    lines = declaration.splitlines()
    for s in sorted(hole_sites(declaration).values(), key=lambda s: s.sorry, reverse=True):
        name = lifted_name(target, s.index)
        lead, body = (f"{s.indent}· ", f"{s.indent}  ") if s.bullet else (s.indent, s.indent)
        lines[s.sorry : s.sorry + 1] = (
            [f"{lead}extract_goal * using {name}"]
            + [f"{body}{opt}" for opt in _TYPED]
            + [f"{body}extract_goal * using {name}_r", f"{body}sorry"]
        )
    return "\n".join(lines) + "\n"


def lifted_from_messages(messages: list[dict], target: str) -> dict[int, str]:
    """hole index -> the plain statement `extract_goal` printed for it (its `:= sorry` dropped)."""
    return {k: v[0] for k, v in lifted_variants(messages, target).items() if v[0]}


def lifted_variants(messages: list[dict], target: str) -> dict[int, tuple[str, str]]:
    """hole index -> (plain statement, typed statement still named `<name>_r`); "" where Lean printed none."""
    head = re.compile(r"^theorem " + re.escape(target) + r"_wf_hole_(\d+)(_r)?(?=[.\s({\[⦃:])")
    out: dict[int, list[str]] = {}
    for m in messages:
        if m.get("severity") != "information":
            continue
        data = str(m.get("data", "")).strip()
        hit = head.match(data)
        if hit and data.endswith(":= sorry"):
            pair = out.setdefault(int(hit.group(1)), ["", ""])
            pair[1 if hit.group(2) else 0] = data[: -len(":= sorry")].rstrip()
    return {k: (v[0], v[1]) for k, v in out.items()}


def untyped_name(statement: str, target: str, k: int) -> str:
    """A typed statement under the hole's own lifted name (`<name>_r` → `<name>`)."""
    name = lifted_name(target, k)
    return re.sub(r"^(theorem )" + re.escape(name) + r"_r(?=[.\s({\[⦃:])", r"\1" + name, statement, count=1)


def validation_block(statements: dict) -> tuple[str, dict]:
    """Each statement as its own `:= by sorry` declaration, to check it re-elaborates.

    Returns (text, key -> (first, last) line of its declaration, 0-based in the text); keys are any
    sortable labels (a hole, or a hole and which print of it).
    """
    chunks, ranges, cursor = [], {}, 0
    for k in sorted(statements):
        chunk = f"{statements[k]} := by\n  sorry\n"
        n = chunk.count("\n")
        ranges[k] = (cursor, cursor + n - 1)
        chunks.append(chunk)
        cursor += n + 1  # the blank line between declarations
    return "\n".join(chunks), ranges


def failing_ranges(messages: list[dict], ranges: dict, first_line: int) -> dict:
    """key -> the first error inside its range (`first_line` is the 1-based file line of the block)."""
    out: dict = {}
    for m in messages:
        if m.get("severity") != "error":
            continue
        line = int((m.get("pos") or {}).get("line", -1)) - first_line
        for k, (lo, hi) in ranges.items():
            if lo <= line <= hi and k not in out:
                out[k] = " ".join(str(m.get("data", "")).split())[:200]
    return out


# `theorem name.{u_1}` — the name, then the universe list `extract_goal` prints for a
# universe-polymorphic goal.
_THEOREM_HEAD = re.compile(
    r"\s*(?:@\[[^\]]*\]\s*)*(?:\w+\s+)*?(?:theorem|lemma)\s+[^\s({\[⦃:.]+(?:\.[^\s({\[⦃:.]+)*(?:\.\{[^}]*\})?"
)


def binders(statement: str) -> list[tuple[str, str]]:
    """(name, bracket) for each binder of a theorem statement, in order, up to its colon.

    An anonymous instance binder `[C]` gives ("", "["). Groups bind several names: `(p q : Prop)`.
    """
    m = _THEOREM_HEAD.match(statement)
    rest = statement[m.end() :] if m else statement
    out: list[tuple[str, str]] = []
    opens, closes = "({[⦃", ")}]⦄"
    i, n = 0, len(rest)
    while i < n:
        c = rest[i]
        if c.isspace():
            i += 1
            continue
        if c not in opens:
            break  # the colon
        depth, j = 0, i
        while j < n:
            if rest[j] in opens:
                depth += 1
            elif rest[j] in closes:
                depth -= 1
                if depth == 0:
                    break
            j += 1
        group = rest[i + 1 : j]
        if ":" in group and not group.lstrip().startswith(":"):
            names = group.split(":", 1)[0].split()
            out += [(t, c) for t in names]
        elif c == "[":
            out.append(("", c))
        i = j + 1
    return out


def context_names(goal_text: str) -> list[str]:
    """The hypothesis names of a printed goal, in context order (`a b : T` gives both)."""
    names: list[str] = []
    for line in goal_text.splitlines():
        if line.lstrip().startswith("⊢"):
            break
        if not line or line[0].isspace() or line.startswith("case "):
            continue
        # a long type wraps: `h :` ends the line and the type follows, indented
        m = re.match(r"^(.+?) :(?: |$)", line)
        if m:
            names += m.group(1).split()
    return names


def rename_line(context: list[str], lifted: list[tuple[str, str]]) -> str | None:
    """`rename_i …` giving the hole's inaccessible hypotheses the lifted lemma's names; "" when
    none needs one; None when the two contexts do not line up one-to-one."""
    if len(context) != len(lifted):
        return None
    for have, (want, _) in zip(context, lifted):
        if "✝" not in have and want and have != want:
            return None
    names = [want or "_" for have, (want, _) in zip(context, lifted) if "✝" in have]
    if all(n == "_" for n in names):
        return ""
    return "rename_i " + " ".join(names)


def helper_call(name: str, lifted: list[tuple[str, str]]) -> str:
    """The tactic that closes a hole with its lifted lemma, passing the explicit hypotheses."""
    return " ".join([f"exact {name}"] + [b for b, bracket in lifted if bracket == "(" and b])


# ── a model's answer ──────────────────────────────────────────────────────────

_CODE_BLOCKS = (
    re.compile(r"```lean4\n(.*?)\n```", re.S),
    re.compile(r"```lean4\n(.*?)```", re.S),
    re.compile(r"```lean\n(.*?)```", re.S),
)
_SCOPE_PREFIX = re.compile(r"^(set_option\s+\S+\s+\S+|open\s+[^\n]+?)\s+in$")


# `∑ x in s, f x` (and `∏`): the big-operator syntax a model trained on 2024 Mathlib writes. This
# Mathlib spells it `∑ x ∈ s, f x` and no longer parses the old form.
_OLD_BIG_OP = re.compile(r"([∑∏]\s*[\w'₀-₉]+)\s+in\s+")


# `norm_num at h ⊢` whose `h` becomes False closes the goal, then fails on `⊢` with "No goals to be
# solved" in this Mathlib (older ones stopped quietly, and Goedel writes it constantly, often after
# `interval_cases x <;>`). Hypotheses first, finishing if that closes the goal, else the call as
# written: wherever the original succeeds this does too. `| ` before it marks one already rewritten.
_NORM_NUM_AT_GOAL = re.compile(r"(?<!\| )\bnorm_num(\s*\[[^\]]*\])?\s+at\s+((?:[^\s⊢;<>|()\[\]]+\s+)+)⊢")


def _hypotheses_first(m: re.Match) -> str:
    call = f"norm_num{m.group(1) or ''} at {m.group(2).strip()}"
    return f"(first | ({call}; done) | {call} ⊢)"


def modernize(code: str) -> tuple[str, list[str]]:
    """A model's code with older-Mathlib syntax and tactic behaviour rewritten to what this Mathlib
    accepts.

    The proof says the same thing, and every candidate is still compiled.
    Returns (code, what was rewritten).
    """
    new, n = _OLD_BIG_OP.subn(r"\1 ∈ ", code)
    notes = [f"{n} big-operator binder(s): `∑ x in s` → `∑ x ∈ s`"] if n else []
    new, n = _NORM_NUM_AT_GOAL.subn(_hypotheses_first, new)
    if n:
        notes.append(f"{n} `norm_num at h ⊢` call(s): hypotheses first, so closing the goal does not fail on `⊢`")
    return new, notes


def extract_code(raw: str) -> str | None:
    """The last Lean code block of a model's answer (Goedel-Prover-V2's own extraction)."""
    for pat in _CODE_BLOCKS:
        found = pat.findall(raw)
        if found:
            return found[-1]
    return None


# ── a rejected proof's skeleton ───────────────────────────────────────────────
# Goedel-Prover-V2 §5: the model's own unsolved steps, lifted and re-solved. A rejected proof
# usually fails at a few steps of a sound plan: its `have` claims, case splits and witnesses are
# kept, each step Lean rejected becomes a hole, and the holes go to Wayfinder's closers first.


@dataclass
class Skeleton:
    text: str  # the lemma, each failing step's proof replaced by a marked `sorry`
    holes: int
    replaced: list[str]  # per hole, the first line of what the model had there


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _opens_block(line: str) -> bool:
    t = line.rstrip()
    return bool(re.search(r"(^|\s)by$", t)) or t.endswith("=>") or line.lstrip().startswith("· ")


def _block_end(lines: list[str], k: int, stop: int) -> int:
    """The first line after the block opened at `k` (lines indented past it), or `stop`."""
    j = k + 1
    while j < stop and (not lines[j].strip() or _indent(lines[j]) > _indent(lines[k])):
        j += 1
    return j


def _hole(indent: int, k: int, bullet: bool = False) -> list[str]:
    pad = " " * indent
    return [f"{pad}-- {HOLE_MARK} {k}: close this goal", f"{pad}{'· ' if bullet else ''}sorry"]


def skeleton(lemma: str, errors: list[tuple[int, int, str]], max_holes: int = 4) -> Skeleton | None:
    """The model's proof with each step Lean rejected replaced by a marked `sorry` (`hole_sites`
    reads them). `lemma` is the lemma as compiled — statement, `:= by`, tactics, one per line or
    indented under their block — and `errors` Lean's errors in it, as (0-based line, column,
    message). A `have` whose proof failed keeps its statement and its proof becomes the hole; a
    failed bullet becomes `· sorry`; a failure in the main block makes the rest of it the hole;
    unsolved goals at a block's end add a hole per goal; a step that ran with no goals is dropped.
    None when the proof is not a tactic block, an error lies outside it, or holes exceed `max_holes`.
    """
    lines = lemma.split("\n")
    main = next((i for i, ln in enumerate(lines) if re.search(r":=\s*by\s*$", ln)), None)
    if main is None:
        return None
    stop = len(lines)
    while stop > main + 1 and not lines[stop - 1].strip():
        stop -= 1
    body = _indent(lines[main + 1]) if main + 1 < stop else 2
    spans: list[tuple[int, int, int, bool]] = []  # replace lines[a:b] with a hole at an indent (bullet?)
    inline: dict[int, int] = {}  # a line whose proof follows `:= by` on it -> its continuation end
    appends: dict[int, list[int]] = {}  # before line e: holes for goals a block left, at these indents
    drops: set[int] = set()
    for li, col, msg in errors:
        # the statement's own line reports only the main block's unsolved goals; anything else there
        # (or above it) is an error in the statement, not in the proof
        if not (main < li < stop or (li == main and msg.startswith("unsolved goals"))):
            return None
        line = lines[li]
        if "No goals" in msg:
            drops.add(li)
        elif msg.startswith("unsolved goals"):  # reported at the block's opener
            end, at = (stop, body) if li == main else (_block_end(lines, li, stop), _indent(line) + 2)
            appends.setdefault(end, []).extend([at] * max(1, msg.count("⊢")))
        elif line.lstrip().startswith("· "):
            spans.append((li, _block_end(lines, li, stop), _indent(line), True))
        elif (m := re.search(r":=\s*by\s+\S", line)) and col >= m.end() - 1:
            inline[li] = _block_end(lines, li, stop)
        else:
            k = li - 1
            while k > main and not (_indent(lines[k]) < _indent(line) and _opens_block(lines[k])):
                k -= 1
            if k > main:
                spans.append((k + 1, _block_end(lines, k, stop), _indent(lines[k]) + 2, False))
            else:
                spans.append((li, stop, _indent(line), False))
    spans.sort()
    kept: list[tuple[int, int, int, bool]] = []
    for sp in spans:  # an outer span covers the ones inside it
        if kept and sp[0] < kept[-1][1]:
            continue
        kept.append(sp)
    out: list[str] = []
    replaced: list[str] = []
    i = 0
    while i < len(lines):
        for at in appends.get(i, []):
            out += _hole(at, len(replaced) + 1)
            replaced.append("(goals left at the end of the block)")
        sp = next((x for x in kept if x[0] == i), None)
        if sp:
            out += _hole(sp[2], len(replaced) + 1, sp[3])
            replaced.append(lines[i].strip())
            i = sp[1]
            continue
        if i in inline:
            m = re.search(r":=\s*by\s+", lines[i])
            out.append(lines[i][: m.start()] + ":= by")  # type: ignore[union-attr]
            out += _hole(_indent(lines[i]) + 2, len(replaced) + 1)
            replaced.append(lines[i][m.end() :].strip())  # type: ignore[union-attr]
            i = inline[i]
            continue
        if i not in drops:
            out.append(lines[i])
        i += 1
    for at in appends.get(len(lines), []):
        out += _hole(at, len(replaced) + 1)
        replaced.append("(goals left at the end of the block)")
    if not replaced or len(replaced) > max_holes:
        return None
    return Skeleton(text="\n".join(out), holes=len(replaced), replaced=replaced)


def closed_type(statement: str) -> str | None:
    """`theorem n (x : A) (h : P x) : Q x` as the type `∀ (x : A) (h : P x), Q x` — how the search
    harness starts a goal no file declares (a lifted step)."""
    flat = " ".join(statement.split())
    m = re.match(r"^(?:theorem|lemma)\s+\S+\s*", flat)
    if not m:
        return None
    rest, depth = flat[m.end() :], 0
    for i, ch in enumerate(rest):
        if ch in "([{⦃":
            depth += 1
        elif ch in ")]}⦄":
            depth -= 1
        elif ch == ":" and depth == 0 and rest[i : i + 2] != ":=":
            binders, concl = rest[:i].strip(), rest[i + 1 :].strip()
            return f"∀ {binders}, {concl}" if binders else concl
    return None


def tried_at(declaration: str) -> dict[int, str]:
    """Per hole, the steps Wayfinder tried there and withdrew (the hand-off's comment above it)."""
    lines, out = declaration.splitlines(), {}
    for k, site in hole_sites(declaration).items():
        above = lines[site.marker - 1].strip() if site.marker else ""
        if above.startswith(f"-- {TRIED_MARK}:"):
            out[k] = above.split(":", 1)[1].strip()
    return out


def renumber_holes(text: str) -> tuple[str, int]:
    """The hole markers numbered 1.. in order (a skeleton of a skeleton repeats numbers)."""
    lines, k = text.split("\n"), 0
    for i, ln in enumerate(lines):
        if m := _MARKER.match(ln):
            k += 1
            lines[i] = f"{m.group('indent')}-- {HOLE_MARK} {k}: close this goal"
    return "\n".join(lines), k


def fill_holes(text: str, fills: list[str | None], left: str = "") -> str:
    """Each marked hole (in order) replaced by its tactic, at the hole's indentation; a hole whose
    fill is None stays a `sorry`, under the comment `left` (or its marker, when `left` is empty)."""
    lines = text.split("\n")
    for k, site in sorted(hole_sites(text).items(), reverse=True):
        lead = site.indent + ("· " if site.bullet else "")
        if fills[k - 1] is not None:
            # a fill of several lines (a search's tactic block) keeps its own shape under the hole
            first, *more = textwrap.dedent(str(fills[k - 1])).strip("\n").split("\n")
            under = site.indent + ("  " if site.bullet else "")
            lines[site.marker : site.sorry + 1] = [lead + first] + [under + ln for ln in more]
        elif left:
            lines[site.marker : site.sorry + 1] = [f"{site.indent}-- {left}", lead + "sorry"]
    return "\n".join(lines)


def claims_at_holes(text: str) -> dict[int, str]:
    """The `have` claim each hole proves, where the hole is a `have`'s proof."""
    lines, out = text.split("\n"), {}
    for k, site in hole_sites(text).items():
        above = lines[site.marker - 1].strip() if site.marker else ""
        if re.search(r":=\s*by$", above):
            out[k] = re.sub(r"\s*:=\s*by$", "", above)
    return out


@dataclass
class Candidate:
    proof: str  # the main theorem's proof as written, from its `:=` on (` by …` or a term)
    helpers: list[str] = field(default_factory=list)  # declarations it stated first, in order
    opens: list[str] = field(default_factory=list)  # namespaces its code opened
    renamed: bool = False  # the main theorem came back under another name

    def tactic_lines(self) -> list[str]:
        """The proof as a tactic block (a term proof becomes `exact` of it)."""
        text = self.proof.strip("\n")
        stripped = text.lstrip()
        if re.match(r"by\b", stripped):
            rest = stripped[2:]
            first, _, tail = rest.partition("\n")
            lines = ([first.strip()] if first.strip() else []) + fill_lines(tail)
            return [ln for ln in lines if ln.strip()]
        return ["exact"] + ["  " + ln for ln in fill_lines(text)]

    def scoped(self, chunk: str) -> str:
        """A helper (or the lifted lemma) under the namespaces the model's code opened."""
        return (f"open {' '.join(self.opens)} in\n" if self.opens else "") + chunk


def candidate(code: str, name: str, known: set[str]) -> Candidate | None:
    """The model's proof of `name` from its code block, with the declarations it added.

    The model may restate the file's own definitions, add lemmas, open namespaces, or rename the
    theorem; what is kept is the proof text after the main theorem's `:=` (its statement is
    replaced by ours, as Goedel-Prover-V2's own pipeline does) and the new declarations above it.
    Returns None when the block holds no theorem.
    """
    chunks = gate.top_level_chunks(code)
    out = Candidate(proof="")
    main: str | None = None
    pending: list[str] = []
    last_theorem: tuple[str, list[str]] | None = None
    helpers: list[tuple[str, list[str]]] = []
    for ch in chunks:
        head = ch.split(None, 1)[0] if ch.split() else ""
        if _SCOPE_PREFIX.match(ch.strip()):
            pending.append(ch.strip())
            continue
        if head == "open":
            out.opens += [t for t in ch.split()[1:] if re.match(r"^[\w.]+$", t)]
            continue
        kind = gate.declaration_kind(ch)
        decl = gate.declaration_name(ch).rstrip(".")
        if kind in ("theorem", "lemma"):
            if decl == name:
                main, pending = ch, []
                continue
            if decl not in known:  # a restatement of the file's own lemma is never the answer
                last_theorem = (ch, pending)
        if kind and decl not in known:
            helpers.append((ch, pending))
        pending = []
    if main is None:
        if last_theorem is None:
            return None
        main = last_theorem[0]
        helpers = [h for h in helpers if h[0] is not main]
        out.renamed = True
    _, proof = gate.split_header(main)
    if not proof.strip():
        return None
    out.proof = proof
    out.helpers = ["\n".join(prefix + [ch]) for ch, prefix in helpers]
    return out
