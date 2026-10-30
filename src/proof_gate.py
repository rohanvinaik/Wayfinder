"""The acceptance gate — a proof is accepted only as the proof the row asked for.

A returned file is a *candidate*, never a result. Counting `sorry` tokens is not
a check: a candidate can be free of `sorry` and still fail to compile, rest on a
declared axiom, prove a different statement, or quietly change a definition the
statement depends on. The gate answers one question — is this the correct,
compiling proof of the submitted statement? — with checks that each name the
failure they rule out:

1. **static** — before anything is compiled (elaboration runs code): no new
   `axiom`, no `unsafe`/`extern`/`implemented_by`, no `#eval`/`run_cmd`-style
   commands, no new notation or macros, no `sorry`/`admit`.
2. **preserved** — every declaration that was submitted is still there,
   text-identical (comments and whitespace aside). Additions are allowed
   (helper lemmas); modifications are not.
3. **statement** — the target theorem's header is text-identical, and its
   elaborated type (``pp.all``) equals the one recorded at submission: an added
   instance or `open` that changes what identical text means is caught here.
4. **compiles** — the file elaborates in the pinned environment with no error
   and no `sorry`.
5. **axioms** — ``#print axioms`` names only the standard three, plus any the
   row explicitly allows.

Finishing versus re-proving is measured beside the verdict (``mode``): did the
returned proof keep the tactic lines that were submitted? A correct re-proof is
still a correct proof; it is accepted unless the caller asks for strict
finishing, and it is always reported.

Pure stdlib (the MCP host imports it). The one impure helper is
:func:`run_lean_json`, which compiles a source text in a temp file under a
project's environment — never inside the project tree.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

STANDARD_AXIOMS: tuple[str, ...] = ("propext", "Classical.choice", "Quot.sound")

_MODIFIERS = ("private", "protected", "noncomputable", "partial", "unsafe", "nonrec", "scoped", "local")
_COMMANDS = (
    "theorem", "lemma", "def", "abbrev", "instance", "structure", "inductive", "class", "axiom",
    "opaque", "example", "attribute", "open", "namespace", "section", "end", "variable", "universe",
    "set_option", "import", "macro", "macro_rules", "syntax", "notation", "infix", "infixl", "infixr",
    "prefix", "postfix", "elab", "elab_rules", "declare_syntax_cat", "initialize", "builtin_initialize",
    "#eval", "#exec", "#check", "#print", "#reduce", "#exit", "#lint", "run_cmd", "run_elab", "run_meta",
    "mutual", "export", "alias", "irreducible_def", "deriving",
)  # fmt: skip
# Commands that extend the parser: a new one can change what unchanged text means.
_SYNTAX_COMMANDS = (
    "notation", "infix", "infixl", "infixr", "prefix", "postfix", "macro", "macro_rules", "syntax",
    "elab", "elab_rules", "declare_syntax_cat",
)  # fmt: skip
# Commands that run code at elaboration time, or bypass the kernel.
_EXECUTING_COMMANDS = ("#eval", "#exec", "run_cmd", "run_elab", "run_meta", "initialize", "builtin_initialize")
_BYPASS_MARKERS = ("unsafe", "@[extern", "@[implemented_by", "implemented_by")

_CHUNK_START = re.compile(
    r"^(?:@\[|(?:" + "|".join(re.escape(m) for m in _MODIFIERS) + r")\b|(?:"
    + "|".join(re.escape(c) for c in _COMMANDS) + r")(?![\w.'!?]))"
)
_DECL_NAME = re.compile(
    r"^(?:@\[[^\]]*\]\s*)*(?:(?:" + "|".join(_MODIFIERS) + r")\s+)*"
    r"(theorem|lemma|def|abbrev|instance|structure|inductive|class|axiom|opaque)\s+([^\s:({\[⦃]+)"
)
_AXIOMS_LINE = re.compile(r"'(?P<name>[^']+)' depends on axioms: \[(?P<axioms>[^\]]*)\]", re.S)
_NO_AXIOMS_LINE = re.compile(r"'(?P<name>[^']+)' does not depend on any axioms")


# ── text normalization ────────────────────────────────────────────────────────


def strip_comments(text: str) -> str:
    """Lean source with comments blanked out; line structure is kept.

    Handles `--` line comments, nested `/- -/` block comments (doc and module
    comments included) and string literals, so a `--` inside a string survives.
    """
    out: list[str] = []
    i, n, depth = 0, len(text), 0
    in_string = False
    while i < n:
        c = text[i]
        two = text[i : i + 2]
        if depth:
            if two == "/-":
                depth += 1
                i += 2
            elif two == "-/":
                depth -= 1
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if two == "/-":
            depth = 1
            i += 2
            continue
        if two == "--":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == '"':
            in_string = True
        out.append(c)
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    """Comments stripped, whitespace collapsed — the form two texts are compared in."""
    return " ".join(strip_comments(text).split())


def top_level_chunks(text: str) -> list[str]:
    """Split a Lean file into its top-level commands (comment-stripped, raw spacing).

    A command starts at column 0 with a command keyword, a modifier, or an
    attribute; everything indented beneath it belongs to it.
    """
    chunks: list[list[str]] = []
    for line in strip_comments(text).splitlines():
        if not line.strip():
            if chunks:
                chunks[-1].append(line)
            continue
        if not line[0].isspace() and _CHUNK_START.match(line):
            chunks.append([line])
        elif chunks:
            chunks[-1].append(line)
        else:
            chunks.append([line])
    return ["\n".join(c).strip() for c in chunks if "\n".join(c).strip()]


def declaration_name(chunk: str) -> str:
    """The declared name of a theorem/def-like chunk, or "" for other commands."""
    m = _DECL_NAME.match(chunk.strip())
    return m.group(2) if m else ""


def declaration_kind(chunk: str) -> str:
    m = _DECL_NAME.match(chunk.strip())
    return m.group(1) if m else ""


def name_from_declaration(full_declaration: str) -> str:
    """The Lean name a row's declaration declares (which can differ from its theorem_id)."""
    for chunk in top_level_chunks(full_declaration):
        if declaration_kind(chunk) in ("theorem", "lemma"):
            return declaration_name(chunk)
    return ""


def find_target(text: str, name: str) -> str | None:
    """The chunk declaring theorem `name`, or None."""
    for chunk in top_level_chunks(text):
        if declaration_kind(chunk) in ("theorem", "lemma") and declaration_name(chunk) == name:
            return chunk
    return None


def split_header(chunk: str) -> tuple[str, str]:
    """(statement header, proof body) at the first `:=` outside any bracket."""
    depth = 0
    opens, closes = "([{⟨⦃", ")]}⟩⦄"
    for i, c in enumerate(chunk):
        if c in opens:
            depth += 1
        elif c in closes:
            depth = max(0, depth - 1)
        elif c == ":" and depth == 0 and chunk[i : i + 2] == ":=":
            return chunk[:i], chunk[i + 2 :]
    return chunk, ""


# ── static checks ─────────────────────────────────────────────────────────────


def _code_tokens(text: str) -> list[str]:
    return re.findall(r"[#@]?\[?[\w.'!?]+", strip_comments(text))


def _command_head(norm_chunk: str) -> str:
    """The command keyword of a normalized chunk, past attributes and modifiers."""
    rest = norm_chunk
    while rest.startswith("@["):
        close = rest.find("]")
        rest = rest[close + 1 :].lstrip() if close >= 0 else ""
    tokens = rest.split()
    while tokens and tokens[0] in _MODIFIERS:
        tokens = tokens[1:]
    return tokens[0] if tokens else ""


def static_findings(returned: str, submitted: str) -> list[str]:
    """What must never be compiled or accepted, as named findings (empty = clean)."""
    findings: list[str] = []
    sub_chunks = {normalize(c) for c in top_level_chunks(submitted)}
    for chunk in top_level_chunks(returned):
        norm = normalize(chunk)
        if norm in sub_chunks:
            continue  # unchanged from the submission: whatever it is, it was ours
        head = _command_head(norm)
        kind = declaration_kind(chunk)
        label = declaration_name(chunk) or head
        if kind in ("axiom", "opaque"):
            findings.append(f"new {kind}: {label}")
        if head in _EXECUTING_COMMANDS:
            findings.append(f"code-executing command: {head}")
        if head in _SYNTAX_COMMANDS:
            findings.append(f"new syntax extension: {head}")
        bypass = sorted(set(_code_tokens(chunk)) & set(_BYPASS_MARKERS))
        if bypass:
            findings.append(f"kernel bypass ({bypass[0]}): {label}")
    tokens = _code_tokens(returned)
    for bad in ("sorry", "admit"):
        if bad in tokens:
            findings.append(f"`{bad}` remains")
    return findings


def preserved_findings(returned: str, submitted: str, target: str) -> list[str]:
    """Submitted declarations that are missing or modified in the returned file."""
    have = {normalize(c) for c in top_level_chunks(returned)}
    missing = []
    for chunk in top_level_chunks(submitted):
        if declaration_kind(chunk) in ("theorem", "lemma") and declaration_name(chunk) == target:
            continue  # the target's body is the part that is allowed to change
        if normalize(chunk) not in have:
            label = declaration_name(chunk) or normalize(chunk)[:60]
            missing.append(f"submitted declaration changed or removed: {label}")
    return missing


def header_findings(returned: str, submitted: str, target: str) -> list[str]:
    sub = find_target(submitted, target)
    ret = find_target(returned, target)
    if sub is None:
        return [f"target `{target}` not found in the submitted file"]
    if ret is None:
        return [f"target `{target}` not found in the returned file"]
    if normalize(split_header(sub)[0]) != normalize(split_header(ret)[0]):
        return [f"statement of `{target}` changed"]
    return []


# ── finish vs re-prove ────────────────────────────────────────────────────────


def _proof_lines(chunk: str) -> list[str]:
    _, body = split_header(chunk)
    lines = [" ".join(ln.split()) for ln in strip_comments(body).splitlines()]
    return [ln for ln in lines if ln and ln not in ("by", "sorry", "· sorry")]


def prefix_kept(returned: str, submitted: str, target: str) -> tuple[bool, list[str], int]:
    """Did the returned proof keep the submitted tactic lines, in order?

    Returns (kept, lines-not-found, number of submitted lines). With no
    submitted prefix (a bare `sorry`) there was nothing to keep.
    """
    sub, ret = find_target(submitted, target), find_target(returned, target)
    if sub is None or ret is None:
        return False, ["target missing"], 0
    want, have = _proof_lines(sub), _proof_lines(ret)
    j, missing = 0, []
    for line in want:
        k = next((idx for idx in range(j, len(have)) if have[idx] == line), None)
        if k is None:
            missing.append(line)
        else:
            j = k + 1
    return not missing, missing, len(want)


# ── Lean output ───────────────────────────────────────────────────────────────


def parse_lean_json(stdout: str) -> list[dict]:
    """`lean --json` messages, one JSON object per line; other lines are dropped."""
    messages = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict):
            messages.append(msg)
    return messages


def axioms_from_messages(messages: list[dict], name: str) -> list[str] | None:
    """The axioms `#print axioms name` reported, or None if it never reported."""
    for msg in messages:
        data = str(msg.get("data", ""))
        m = _AXIOMS_LINE.search(data)
        if m and m.group("name") == name:
            return [a.strip() for a in m.group("axioms").replace("\n", " ").split(",") if a.strip()]
        m = _NO_AXIOMS_LINE.search(data)
        if m and m.group("name") == name:
            return []
    return None


def checked_type(messages: list[dict], name: str) -> str | None:
    """The `#check @name` output (whitespace-normalized), or None.

    A universe-polymorphic theorem prints its universe list after the name
    (`@foo.{u_1} : ...`), so the name is matched with an optional `.{...}`.
    """
    head = re.compile(r"^@?" + re.escape(name) + r"(?:\.\{[^}]*\})?\s+:")
    for msg in messages:
        data = str(msg.get("data", ""))
        if head.match(data):
            return " ".join(data.split())
    return None


def errors_in(messages: list[dict]) -> list[str]:
    return [
        f"{m.get('pos', {}).get('line', '?')}:{m.get('pos', {}).get('column', '?')}: {str(m.get('data', ''))[:300]}"
        for m in messages
        if m.get("severity") == "error"
    ]


def uses_sorry(messages: list[dict]) -> bool:
    return any(m.get("kind") == "hasSorry" or "declaration uses `sorry`" in str(m.get("data", "")) for m in messages)


def probe_suffix(name: str) -> str:
    """What the gate appends to a candidate to read back its type and axioms."""
    return f"\n\nset_option pp.all true in\n#check @{name}\n\n#print axioms {name}\n"


# ── the verdict ───────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    accepted: bool
    target: str
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    axioms: list[str] | None = None
    mode: str = "n/a"  # "finished" | "reproved" | "n/a"
    prefix_missing: list[str] = field(default_factory=list)
    environment: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "target": self.target,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "axioms": self.axioms,
            "mode": self.mode,
            "prefix_missing": list(self.prefix_missing),
            "environment": self.environment,
        }


def judge(
    *,
    returned: str,
    submitted: str,
    target: str,
    compile_run: dict | None,
    submitted_type: str | None = None,
    allowed_axioms: tuple[str, ...] = (),
    require_finish: bool = False,
    environment: str = "",
) -> GateResult:
    """Combine the checks into one verdict.

    `compile_run` is :func:`run_lean_json` of `returned + probe_suffix(target)`;
    None means nothing was compiled (the static checks failed first).
    `submitted_type` is the target's ``pp.all`` type recorded at submission. When
    it is absent the type check fails: an unrecorded statement cannot be confirmed.
    """
    res = GateResult(accepted=False, target=target, environment=environment)
    static = static_findings(returned, submitted)
    res.checks["static"] = not static
    res.reasons += static
    preserved = preserved_findings(returned, submitted, target)
    res.checks["preserved"] = not preserved
    res.reasons += preserved
    header = header_findings(returned, submitted, target)
    res.checks["statement_text"] = not header
    res.reasons += header

    kept, missing, n_submitted = prefix_kept(returned, submitted, target)
    res.mode = "n/a" if n_submitted == 0 else ("finished" if kept else "reproved")
    res.prefix_missing = missing
    strict_miss = require_finish and res.mode == "reproved"
    if strict_miss:
        res.reasons.append("re-proved instead of finishing (strict mode)")

    if compile_run is None:
        res.checks["compiles"] = False
        res.reasons.append("not compiled (static check failed)")
        return res

    messages = compile_run.get("messages") or []
    errors = errors_in(messages)
    if compile_run.get("timed_out"):
        res.reasons.append("compile timed out")
    elif compile_run.get("returncode") != 0 and not errors:
        res.reasons.append(f"lean exited {compile_run.get('returncode')}: {str(compile_run.get('tail', ''))[-200:]}")
    res.checks["compiles"] = (
        not compile_run.get("timed_out")
        and compile_run.get("returncode") == 0
        and not errors
        and not uses_sorry(messages)
    )
    if errors:
        res.reasons += [f"error {e}" for e in errors[:5]]
    if uses_sorry(messages):
        res.reasons.append("Lean reports `declaration uses sorry`")

    got_type = checked_type(messages, target)
    if submitted_type is None:
        res.checks["statement_type"] = False
        res.reasons.append("no recorded statement type to compare against")
    else:
        res.checks["statement_type"] = got_type is not None and got_type == " ".join(submitted_type.split())
        if not res.checks["statement_type"]:
            res.reasons.append("elaborated statement differs from the submitted one")

    axioms = axioms_from_messages(messages, target)
    res.axioms = axioms
    allowed = set(STANDARD_AXIOMS) | set(allowed_axioms)
    if axioms is None:
        res.checks["axioms"] = False
        res.reasons.append("`#print axioms` produced no report")
    else:
        extra = [a for a in axioms if a not in allowed]
        res.checks["axioms"] = not extra
        if extra:
            res.reasons.append(f"non-allowlisted axioms: {', '.join(extra)}")

    res.accepted = all(res.checks.values()) and not strict_miss
    return res


# ── the one impure helper ─────────────────────────────────────────────────────


def run_lean_json(lake_bin: str | Path, project_dir: str | Path, source: str, timeout: int = 600) -> dict:
    """Compile `source` under `project_dir`'s environment, from a temp file outside it.

    Returns {"returncode", "messages", "timed_out", "tail"}. The project tree is
    never written to.
    """
    tmp = Path(tempfile.mkdtemp(prefix="wf_gate_"))
    path = tmp / "Candidate.lean"
    try:
        path.write_text(source)
        proc = subprocess.run(
            [str(lake_bin), "env", "lean", "--json", str(path)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "messages": parse_lean_json(proc.stdout),
            "timed_out": False,
            "tail": (proc.stdout + proc.stderr)[-1500:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": None, "messages": [], "timed_out": True, "tail": f"timeout after {timeout}s"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
