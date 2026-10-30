"""The residual hand-off — Wayfinder's stopping state as a Lean proof with holes.

When the search stops, what it has is a tree: the goal it started from, the
tactics it committed, the subgoals each one opened, and the goals still open.
This module turns that tree back into Lean — the declaration, Wayfinder's
verified tactics in place, and one `sorry` per open goal — so the next prover
is handed the *remaining* problem, not the original theorem.

Pure stdlib (the MCP host imports it). Lean is never called here: the caller
compiles what :func:`render` emits, hands the messages back to
:func:`holes_from_messages` / :func:`repair_targets`, and repeats until the
skeleton compiles. A skeleton is only presented as Wayfinder's stopping state
when the goals Lean prints at its holes are the goals the search left open
(:func:`faithful`); otherwise it is reported as diverged.

The tree comes from `SearchResult.close_log` — one record per committed
closure: the goal, its index in the open list at the time, the open list
itself, the replayable Lean text, the goals it opened, and whether they were
prepended. A record whose Lean text is None (self-application, a Dr. Ducky
program with no script) is a hole by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INDENT = "  "
HOLE_MARK = "WAYFINDER RESIDUAL"
LOCAL_MARK = "closed locally by Wayfinder's finishing pass"
TRIED_MARK = "Wayfinder tried here and withdrew it (nothing under it closed)"
DATA_HOLE_ERROR = "wayfinder: data hole"
# The compile-time instrument at each hole: refuse a hole whose goal is data
# (a witness), then print the goal Lean has there.
_DATA_PROBE = (
    "run_tac do",
    "  let t ← Lean.Elab.Tactic.getMainTarget",
    f'  unless (← Lean.Meta.isProp t) do throwError "{DATA_HOLE_ERROR}"',
    "trace_state",
)
_REVERT_LINE = -1  # line_node marker for the `revert` line
_SUPPORT_PREFIX = re.compile(r"\b_wf_support_\d+_")


class ReplayError(ValueError):
    """The closure log does not describe a consistent tree, or the row can't host one."""


@dataclass
class Node:
    goal: str  # the goal as the search tracked it (target text)
    parent: int = -1
    tactic: str | None = None  # replayable Lean text of the closure applied here
    descriptor: str = ""  # the closure's `tactics_used` entry
    provenance: str = ""
    closure: int = -1  # index of the closure in the log; -1 = never closed
    children: list[int] = field(default_factory=list)


@dataclass
class Tree:
    nodes: list[Node]
    roots: list[int]
    open_ids: list[int]  # what the search left open, in its final order


def _resync(open_ids: list[int], nodes: list[Node], before: list[str]) -> list[int]:
    """Reorder our open list to the search's (it rotates goals); fail on divergence."""
    if [nodes[i].goal for i in open_ids] == list(before):
        return open_ids
    pool, out = list(open_ids), []
    for text in before:
        k = next((j for j, i in enumerate(pool) if nodes[i].goal == text), None)
        if k is None:
            raise ReplayError(f"open goals diverged: {text[:80]!r} is not in the reconstructed open list")
        out.append(pool.pop(k))
    if pool:
        raise ReplayError(f"open goals diverged: {len(pool)} reconstructed goal(s) the search no longer had")
    return out


def build_tree(initial_goals: list[str], close_log: list[dict]) -> Tree:
    """Rebuild the committed proof tree from the closure log."""
    nodes = [Node(goal=str(g)) for g in initial_goals]
    roots = list(range(len(nodes)))
    open_ids = list(roots)
    for k, rec in enumerate(close_log):
        if rec.get("open_before") is not None:
            open_ids = _resync(open_ids, nodes, list(rec["open_before"]))
        idx = int(rec.get("goal_idx", 0))
        if not 0 <= idx < len(open_ids):
            raise ReplayError(f"closure {k}: goal index {idx} outside {len(open_ids)} open goals")
        nid = open_ids.pop(idx)
        node = nodes[nid]
        if node.goal != str(rec.get("goal", node.goal)):
            raise ReplayError(f"closure {k}: closed {str(rec.get('goal'))[:60]!r}, open goal is {node.goal[:60]!r}")
        node.tactic = rec.get("lean_tactic")
        node.descriptor = str(rec.get("tactic", ""))
        node.provenance = str(rec.get("provenance", ""))
        node.closure = k
        kids = []
        for g in rec.get("new_goals") or []:
            nodes.append(Node(goal=str(g), parent=nid))
            kids.append(len(nodes) - 1)
        node.children = kids
        open_ids = kids + open_ids if rec.get("prepend") else open_ids + kids
    return Tree(nodes=nodes, roots=roots, open_ids=open_ids)


# ── what the residual may keep ────────────────────────────────────────────────

# Steps that can turn a true goal into a false one: they choose (a disjunct, a witness, a middle
# term, a lemma to apply, a claim to prove first) or give something up (a hypothesis, generality).
# One stands in a residual only if everything under it closed; otherwise it is withdrawn and its
# own goal is the hole. Rewriting, case analysis and introduction steps cannot lose the goal's
# truth, so they stand as validated forward progress either way.
COMMITTING = frozenset({
    "left", "right", "use", "use!", "exists", "existsi", "refine", "refine'", "apply", "congr", "congr!",
    "gcongr", "exfalso", "trans", "calc", "specialize", "clear", "generalize", "have", "suffices", "wlog",
    "apply_fun", "constructorm",
})  # fmt: skip
_STEP_SPLIT = re.compile(r";|<;>|\n")
_HEAD = re.compile(r"[·\s(]*([A-Za-z_][A-Za-z_0-9'!?]*)")


def commits(node: Node) -> bool:
    """Whether the step at `node` can lose the goal's truth (a choice or a weakening)."""
    for part in _STEP_SPLIT.split(node.tactic or ""):
        m = _HEAD.match(part)
        if not m:
            continue
        head = m.group(1)
        # `constructor` splits a conjunction or an iff without loss, but on `∃` it commits a witness
        if head in COMMITTING or (head == "constructor" and target_of(node.goal).startswith("∃")):
            return True
    return False


def subtree(tree: Tree, nid: int) -> list[int]:
    """`nid` and every node under it, in pre-order."""
    out, stack = [], [nid]
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(reversed(tree.nodes[k].children))
    return out


def unvalidated_commitments(tree: Tree, open_holes: list[int] | set[int], forced: set[int]) -> set[int]:
    """The committing steps with a hole still open under them — none of what they chose is proven,
    so none of it may stand in the residual. (A step already given back is not counted again.)"""
    holes = set(open_holes)
    out = set()
    for nid, node in enumerate(tree.nodes):
        if nid in forced or node.closure < 0 or not commits(node):
            continue
        if any(k in holes for k in subtree(tree, nid)[1:]):
            out.add(nid)
    # an outer withdrawal drops everything under it, inner ones included
    return {n for n in out if not any(m != n and n in subtree(tree, m) for m in out)}


def attempted(tree: Tree, nid: int) -> list[str]:
    """The steps Wayfinder tried from `nid` down (pre-order), each on one line."""
    steps = []
    for k in subtree(tree, nid):
        if tree.nodes[k].tactic and tree.nodes[k].closure >= 0:
            steps.append(" ".join(tree.nodes[k].tactic.split()))
    return steps


# ── the declaration ───────────────────────────────────────────────────────────


def split_declaration(declaration: str) -> tuple[str, str]:
    """(header ending in `:= by`, original body) of a theorem declaration.

    A term-mode `:= sorry` is read as an empty tactic proof.
    """
    m = re.search(r":=\s*by\b", declaration)
    if m:
        return declaration[: m.end()], declaration[m.end() :]
    m = re.search(r":=\s*sorry\s*$", declaration)
    if m:
        return declaration[: m.start()].rstrip() + " := by", "\n  sorry"
    raise ReplayError("declaration is neither a `:= by` proof nor `:= sorry`")


def binder_names(header: str) -> list[str]:
    """Names bound before the statement's colon: `(x y : T)`, `{a : T}`, `[i : C]`.

    Anonymous instance binders (`[C]`) have no name to revert by; reverting the
    variables they depend on takes them along.
    """
    m = re.match(r"\s*(?:@\[[^\]]*\]\s*)*(?:\w+\s+)*?(?:theorem|lemma)\s+\S+", header)
    rest = header[m.end() :] if m else header
    names: list[str] = []
    opens, closes = "({[⦃", ")}]⦄"
    i, n = 0, len(rest)
    while i < n:
        c = rest[i]
        if c.isspace():
            i += 1
            continue
        if c not in opens:
            break  # the colon, or syntax this reader does not handle
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
            left = group.split(":", 1)[0]
            names += [t for t in left.split() if t != "_" and re.match(r"^[^\W\d][\w'.!?₀-₉]*$", t)]
        i = j + 1
    return names


# ── rendering ─────────────────────────────────────────────────────────────────


@dataclass
class Rendered:
    body: list[str]  # the tactic-block lines, indented under `:= by`
    holes: list[int]  # node ids that are holes, in source order
    line_node: dict[int, int]  # body line offset -> node id (or _REVERT_LINE)
    kept: list[int]  # node ids whose tactic is in the skeleton
    filled: list[int] = field(default_factory=list)  # holes the local finishing pass closed


def _tactic_lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()] or [text.strip()]


def render(
    tree: Tree,
    *,
    revert: list[str] | None = None,
    forced_holes: set[int] | frozenset[int] = frozenset(),
    probe: bool = False,
    hole_goals: dict[int, str] | None = None,
    fills: dict[int, str] | None = None,
    tried: dict[int, list[str]] | None = None,
) -> Rendered:
    """The tactic block for the tree, with a `sorry` at every open (or forced) goal.

    `revert` names the binders to revert first (the search started from the
    closed type). `probe=True` emits the compile-time instrument at each hole;
    otherwise each hole carries a comment with its goal — `hole_goals[nid]`,
    the goal Lean printed there — for the prover that finishes it. `fills`
    maps a hole to the tactic the local finishing pass closed it with; a filled
    hole is rendered as that tactic, marked, and is no longer a hole. `tried` maps a hole to the
    steps Wayfinder tried there and withdrew (unvalidated), written above it as comments.
    """
    out = Rendered(body=[], holes=[], line_node={}, kept=[])

    def is_hole(nid: int) -> bool:
        node = tree.nodes[nid]
        return node.closure < 0 or node.tactic is None or nid in forced_holes

    def emit(line: str, nid: int, depth: int) -> None:
        out.line_node[len(out.body)] = nid
        out.body.append(_INDENT * depth + line)

    def bullet(start: int, depth: int) -> None:
        # A hole's comment lines sit at the bullet's own level, just above `· sorry`,
        # so they read as belonging to that goal rather than to the bullet before it.
        for k in range(start, len(out.body)):
            if out.body[k].lstrip().startswith("--"):
                out.body[k] = _INDENT * depth + out.body[k].lstrip()
                continue
            out.body[k] = _INDENT * depth + "· " + out.body[k].lstrip()
            return

    def block(nid: int, depth: int) -> None:
        node = tree.nodes[nid]
        if is_hole(nid) and nid in (fills or {}):
            out.filled.append(nid)
            emit(f"-- {LOCAL_MARK}", nid, depth)
            for line in _tactic_lines((fills or {})[nid]):
                emit(line, nid, depth)
            return
        if is_hole(nid):
            out.holes.append(nid)
            if probe:
                for line in _DATA_PROBE:
                    emit(line, nid, depth)
            else:
                steps = (tried or {}).get(nid) or []
                if steps:
                    emit(f"-- {TRIED_MARK}: {' ; '.join(steps)}"[:400], nid, depth)
                emit(f"-- {HOLE_MARK} {len(out.holes)}: close this goal", nid, depth)
                for gl in ((hole_goals or {}).get(nid) or node.goal).splitlines():
                    emit(f"--   {gl}", nid, depth)
            emit("sorry", nid, depth)
            return
        out.kept.append(nid)
        for line in _tactic_lines(node.tactic or ""):
            emit(line, nid, depth)
        if len(node.children) == 1:
            block(node.children[0], depth)
        elif len(node.children) > 1:
            for kid in node.children:
                start = len(out.body)
                block(kid, depth + 1)
                bullet(start, depth)

    if revert:
        out.line_node[0] = _REVERT_LINE
        out.body.append(_INDENT + "revert " + " ".join(revert))
    if len(tree.roots) == 1:
        block(tree.roots[0], 1)
    else:
        for r in tree.roots:
            start = len(out.body)
            block(r, 2)
            bullet(start, 1)
    return out


def assemble(header: str, rendered: Rendered) -> str:
    """The declaration: the original header, then the rendered block."""
    return header.rstrip() + "\n" + "\n".join(rendered.body) + "\n"


def body_first_line(file_text: str, header: str) -> int:
    """1-based file line where the rendered block starts (the line after the header)."""
    end = file_text.rindex(header.rstrip()) + len(header.rstrip())
    return file_text[:end].count("\n") + 2


# ── reading Lean back ─────────────────────────────────────────────────────────


def target_of(goal_text: str) -> str:
    """The goal's target: the text after its last turnstile, whitespace-collapsed.

    The search kernel loads a file's local declarations under collision-proof
    names (`_wf_support_<k>_<name>`, LeanKernel._rename_support_declarations); the
    prefix is removed so its goals compare with the ones Lean prints in the
    hand-off file, where the declarations keep their own names.
    """
    text = goal_text.rsplit("⊢", 1)[-1] if "⊢" in goal_text else goal_text
    return " ".join(_SUPPORT_PREFIX.sub("", text).split())


def holes_from_messages(messages: list[dict], rendered: Rendered, first_line: int) -> dict[int, str]:
    """node id -> the goal Lean printed at that hole (the probe's `trace_state`)."""
    trace_lines = {
        first_line + off: nid for off, nid in rendered.line_node.items() if rendered.body[off].strip() == "trace_state"
    }
    out: dict[int, str] = {}
    for m in messages:
        line = int((m.get("pos") or {}).get("line", -1))
        if line in trace_lines and m.get("severity") == "information":
            out[trace_lines[line]] = str(m.get("data", ""))
    return out


def repair_targets(tree: Tree, messages: list[dict], rendered: Rendered, first_line: int) -> tuple[set[int], list[str]]:
    """Which nodes to turn into holes so the next compile gets further.

    - an error on a tactic's lines: that node becomes a hole (its subtree goes
      with it; siblings stay);
    - a data hole: its parent becomes the hole — a `sorry` standing for a
      witness would leave the siblings proving things about an opaque term;
    - any other error at a hole (e.g. "no goals"): the parent did more than the
      search recorded, so the parent becomes the hole;
    - an error the block can't place (unsolved goals at the declaration): the
      root chain's last kept step becomes the hole;
    - an error on the `revert` line: nothing — this form is abandoned.

    Returns (nodes, notes). An empty set means this form can't be repaired.
    """
    targets: set[int] = set()
    notes: list[str] = []
    for m in messages:
        if m.get("severity") != "error":
            continue
        line = int((m.get("pos") or {}).get("line", -1))
        data = " ".join(str(m.get("data", "")).split())
        nid = rendered.line_node.get(line - first_line)
        if nid == _REVERT_LINE:
            return set(), [f"the revert line failed: {data[:100]}"]
        if nid is not None and nid in rendered.holes:
            parent = tree.nodes[nid].parent
            what = "data hole" if DATA_HOLE_ERROR in data else f"error at a hole ({data[:60]})"
            if parent >= 0:
                targets.add(parent)
                notes.append(f"{what}: closure {tree.nodes[parent].closure} becomes the hole")
            else:
                notes.append(f"{what} at the root: nothing above it to give back")
            continue
        if nid is not None:
            targets.add(nid)
            tactic = (tree.nodes[nid].tactic or "")[:40]
            notes.append(f"closure {tree.nodes[nid].closure} `{tactic}` failed: {data[:100]}")
            continue
        chain = _root_chain(tree, rendered)
        if chain:
            targets.add(chain[-1])
            closure = tree.nodes[chain[-1]].closure
            notes.append(f"unplaced error at line {line} ({data[:60]}): closure {closure} becomes the hole")
    return targets, notes


def _root_chain(tree: Tree, rendered: Rendered) -> list[int]:
    if not tree.roots:
        return []
    chain, nid = [], tree.roots[0]
    while nid in rendered.kept:
        chain.append(nid)
        kids = tree.nodes[nid].children
        if len(kids) != 1:
            break
        nid = kids[0]
    return chain


def faithful(tree: Tree, hole_goals: dict[int, str], holes: list[int]) -> tuple[bool, list[dict]]:
    """Do the goals Lean printed at the holes equal the goals the search left there?"""
    rows = []
    for nid in holes:
        expected = target_of(tree.nodes[nid].goal)
        got = target_of(hole_goals.get(nid, ""))
        rows.append({"node": nid, "expected": expected, "lean": got, "match": bool(got) and got == expected})
    return all(r["match"] for r in rows), rows


# ── the local finishing pass: close what can be closed here, refute what is false ──

# Deterministic closers, cheapest first; `{d}` is the file's own definitions the hole
# mentions. `grind` is last: it is the heaviest, and a run that exhausts the heartbeat
# budget only costs this hole's attempt (each hole is its own declaration). try? is left
# out: on the T7.18 holes it exhausted a 1M-heartbeat budget without an answer.
_CLOSERS = (
    "simp_all [{d}]",
    "omega",
    "decide",
    "norm_num [{d}] at *",
    "linarith",
    "nlinarith",
    "positivity",
    "tauto",
    "(try simp only [{d}] at *); aesop",
)
_HEAVY_CLOSERS = ("grind",)
# `a / b` between plain names: the goal is about Euclidean division (ℕ).
_DIV_TERM = re.compile(r"(?<![\w.'])([^\W\d][\w']*) / ([^\W\d][\w']*)(?![\w'])")
_LOCAL_TRACE = "wf_local:"
COUNTEREXAMPLE = "Found a counter-example!"


def closers(defs: list[str], goal: str = "") -> list[str]:
    """The closer battery for one hole, with the file's definitions it mentions unfolded.

    A goal about `a / b` also gets the division facts — `b * (a / b) + a % b = a` and
    `a % b < b` — as hypotheses for `omega`, which treats the products as atoms: with them
    the Euclidean-division bounds (`a ≤ b + b * (a / b)`) are linear.
    """
    d = ", ".join(defs)
    out = []
    for c in _CLOSERS:
        if d:
            out.append(c.format(d=d))
        else:
            out.append(c.replace(" [{d}]", "").replace("(try simp only at *); ", ""))
    pairs = list(dict.fromkeys(_DIV_TERM.findall(goal.rsplit("⊢", 1)[-1] if "⊢" in goal else goal)))
    if pairs:
        facts = "; ".join(f"have := Nat.div_add_mod {a} {b}; have := Nat.mod_lt {a} (by omega)" for a, b in pairs)
        out.append(f"{facts}; omega")
    return out + list(_HEAVY_CLOSERS)


def defs_mentioned(goal_text: str, defs: list[str]) -> list[str]:
    """The definitions (of the hand-off file) a hole's goal names, in file order."""
    tokens = set(re.findall(r"[\w'.]+", goal_text))
    return [d for d in defs if d in tokens]


def _rename_target(declaration: str, name: str, new_name: str) -> str:
    head = re.compile(r"\b(theorem|lemma)\s+" + re.escape(name) + r"(?=[\s:({\[⦃])")
    return head.sub(r"\1 " + new_name, declaration, count=1)


@dataclass
class TriageProbe:
    text: str  # the declarations block (2 copies of the theorem per hole)
    ranges: list[tuple[int, str, int, int]]  # (hole index, "close"|"refute", first line, last line), 0-based in text
    battery: list[list[str]]  # the closers tried at each hole


def triage_block(declaration: str, name: str, battery: list[list[str]], heartbeats: int = 400000) -> TriageProbe:
    """Two renamed copies of the declaration per hole — one trying the closers there, one
    trying to refute it — each its own declaration, so each has its own heartbeat budget and
    a runaway tactic at one hole cannot take the others down. Other holes stay `sorry`.
    """
    lines = declaration.rstrip("\n").splitlines()
    hole_rows = [i for i, ln in enumerate(lines) if re.match(r"^\s*(· )?sorry$", ln)]
    chunks: list[str] = []
    ranges: list[tuple[int, str, int, int]] = []
    cursor = 0
    for k, row in enumerate(hole_rows):
        indent = re.match(r"^(\s*)", lines[row]).group(1)  # type: ignore[union-attr]
        bullet = "· " if lines[row].strip().startswith("·") else ""
        for mode in ("close", "refute"):
            body = list(lines)
            if mode == "close":
                alts = [f"{indent}{bullet}first"] + [
                    f"{indent}    | ({c}; done; trace \"{_LOCAL_TRACE}{j}\")" for j, c in enumerate(battery[k])
                ]
                body[row : row + 1] = alts
            else:
                body[row] = f"{indent}{bullet}plausible"
            decl = _rename_target("\n".join(body), name, f"{name}_wf_{mode}_{k + 1}")
            chunk = f"set_option maxHeartbeats {heartbeats} in\n{decl}\n"
            n_lines = chunk.count("\n")
            ranges.append((k, mode, cursor, cursor + n_lines - 1))
            chunks.append(chunk)
            cursor += n_lines + 1  # the blank line between chunks
    return TriageProbe(text="\n".join(chunks), ranges=ranges, battery=battery)


def triage_outcomes(messages: list[dict], probe: TriageProbe, first_line: int) -> list[dict]:
    """Per hole: which closer closed it (if any), and the counterexample that refutes it (if any).

    `first_line` is the 1-based file line where the probe block starts.
    """
    n = len(probe.battery)
    out = [{"closed_by": None, "refuted": None, "refute_note": ""} for _ in range(n)]
    for k, mode, lo, hi in probe.ranges:
        span = [m for m in messages if lo <= int((m.get("pos") or {}).get("line", -1)) - first_line <= hi]
        errors = [m for m in span if m.get("severity") == "error"]
        if mode == "close":
            marks = [str(m.get("data", "")) for m in span if str(m.get("data", "")).startswith(_LOCAL_TRACE)]
            if marks and not errors:
                j = int(marks[0][len(_LOCAL_TRACE) :].strip())
                out[k]["closed_by"] = probe.battery[k][j]
        else:
            for m in errors:
                data = str(m.get("data", ""))
                if COUNTEREXAMPLE in data:
                    out[k]["refuted"] = data.strip()
                    break
                out[k]["refute_note"] = " ".join(data.split())[:160]
    return out

