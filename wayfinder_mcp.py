"""FastMCP server for Wayfinder — runs the ACTUAL neural proof-search pipeline.

Three steps, mirroring how proofs are actually produced:
  * `prove(theorem_id)` — run the real Wayfinder prover (below), then replay the tree it
    committed as Lean: the declaration, Wayfinder's verified tactics in place, a `sorry` on
    each goal it left open — compiled at v4.28 until it elaborates (the hand-off skeleton).
  * `send_to_aristotle(theorem_id)` — hand Aristotle, a SEPARATE hosted prover, that
    RESIDUAL: the skeleton, with the instruction to fill exactly its holes. Aristotle
    finishes Wayfinder's proof; it is not asked to re-prove the theorem.
  * `aristotle_result(project_id)` — collect and JUDGE: the returned file is accepted only
    if it is the correct, compiling proof of the submitted statement (src/proof_gate.py).

`prove()` invokes `scripts/run_benchmark.py` — the real pipeline: one neural forward
pass per proof state (encoder → navigator → ternary 6-bank direction) → proof-network
premises (bank alignment / anchor IDF / spreading) → ordered lanes {structural, cosine,
hammer, Dr. Ducky}, each Lean-verified through Pantograph — on a single theorem.

This deliberately replaces the earlier reduced tools (a "triage" profiler and a
replay-only "dr_ducky_solve"). Those did NOT run the prover. This does.

Path discipline (the "no cwd errors in the database / model weights" requirement):
  * checkpoint, config, and lean-project are passed as ABSOLUTE paths;
  * the premise DB is rewritten to an ABSOLUTE path in a resolved temp config, so
    a relative `sqlite3.connect` can never silently open an empty DB and drop
    premise retrieval (gotcha-1);
  * the subprocess runs with cwd = Wayfinder root and PYTHONPATH set.

Wayfinder and Aristotle are separate, unrelated provers. This tool runs Wayfinder
only. If Wayfinder leaves a residual goal open, that residual is what would be sent
to Aristotle — this tool does not call it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WF = Path("/Users/rohanvinaik/Projects/Wayfinder")

# The warranted-emission seam (THE_REFOUNDING §5) is pure stdlib, so it is
# safe to import into the MCP host environment directly.
import sys  # noqa: E402

if str(WF) not in sys.path:
    sys.path.insert(0, str(WF))
import src.proof_gate as gate  # noqa: E402
import src.residual_handoff as rh  # noqa: E402
from src.cheat_sheet import render_cheat_sheet  # noqa: E402
from src.lean_expr import (  # noqa: E402
    EmissionWarrant,
    from_tactics,
    read_environment_pin,
    render_declaration,
    warrant_header,
)

# The goal ledger (THE_REFOUNDING §4.1b/§6): prove() persists each theorem's
# augmented detail row here; send_to_aristotle renders the cheat sheet from it
# so the hand-off carries Wayfinder's computed understanding (Gotcha 3 closed).
GOAL_LEDGER = WF / "runs/goal_ledger"

VENV_PY = WF / ".venv/bin/python"
BENCH = WF / "scripts/run_benchmark.py"
CONFIG = WF / "configs/wayfinder.yaml"
# Newest move-supervised checkpoint; 42% on mathlib_benchmark_50
# (NAV-002 crashed Pantograph mid-run).
CHECKPOINT = WF / "models/NAV-004_step5000.pt"
# v4.27.0 — Pantograph search env (REPL + premise DB match here)
LEAN_PROJECT = WF / "data/lean_project"
# v4.28.0 — soundness gate: reconstructed proofs re-verified here
VERIFY_PROJECT = WF / "data/lean_project_v428"
# The environment pin every emitted artifact carries (THE_REFOUNDING §5).
VERIFY_ENV = read_environment_pin(VERIFY_PROJECT)
DB = WF / "data/proof_network_v3.db"  # absolute — kills the cwd DB error
DEFAULT_MANIFEST = WF / "data/rerun_v427_manifest.jsonl"  # the 9 SSL/Ansatz/Socratic theorems
LAKE_BIN = Path("/Users/rohanvinaik/.elan/bin/lake")  # elan shim (picks toolchain from lean-toolchain)

# Aristotle (separate hosted fallback prover; aristotlelib 2.1.0 = v3 API on the MCP host)
ARISTOTLE_ENV = Path("/Users/rohanvinaik/semantic_probing/.env")  # ARISTOTLE_API_KEY lives here
LEAN_VERSION = "v4.28.0"  # Aristotle solves at v4.28.0; submissions + saved proofs match it
# Where finished Aristotle proofs land — one clean proofs/ project per owning repo.
SSL_PROOFS = Path(
    "/Users/rohanvinaik/Projects/rohan-vinaik.github.io/papers/Core Documents/Semantic_Specification_Learning/proofs"
)
ANSATZ_PROOFS = Path("/Users/rohanvinaik/Projects/Ansatz/proofs")
ANSATZ_THEOREMS = {"depletion_time", "pesin_band_rate_lower_bound"}  # Pesin/physics-bridge cores; rest → SSL
ARISTOTLE_INDEX = WF / "runs" / "aristotle" / "index.json"  # project_id → theorem context
# Building the hand-off skeleton: at most this many repair compiles per form, within this budget.
HANDOFF_MAX_REPAIRS = 6
HANDOFF_BUDGET_S = 420
# The local finishing pass: rounds of close-or-refute over the holes, within this budget.
HANDOFF_FINISH_ROUNDS = 6
HANDOFF_FINISH_BUDGET_S = 600


def _proof_home(row: dict) -> Path:
    """Where a finished proof lands: the manifest row's own `proof_home`.

    Each calling repo declares its proofs/ project on its manifest rows (an absolute path), so
    every repo gets its own home. The legacy per-theorem routes (SSL / Ansatz) apply only to the
    pre-existing manifests whose rows carry no `proof_home`.
    """
    declared = str(row.get("proof_home") or "").strip()
    if declared:
        return Path(declared).expanduser()
    return ANSATZ_PROOFS if row.get("theorem_id") in ANSATZ_THEOREMS else SSL_PROOFS


mcp = FastMCP("Wayfinder")


def _resolved_config(collect_trace: bool = False, dest: Path | None = None) -> Path:
    """Copy the canonical config, forcing the premise DB path absolute.

    Text patch (no YAML dependency on the MCP host python); preserves comments.
    collect_trace: also enable per-step step_trace collection (the corpus-measurement observable).
    dest: write to this PERSISTENT path instead of a temp file (a long background run must keep it).
    """
    text = CONFIG.read_text()
    patched, n = re.subn(
        r"(?m)^(\s*proof_network_db:\s*).*$",
        lambda m: f"{m.group(1)}{DB}",
        text,
    )
    if n == 0:  # key absent — inject under a data: block defensively
        patched = text + f"\ndata:\n  proof_network_db: {DB}\n"
    if collect_trace and "collect_trace" not in patched:
        patched2, m = re.subn(r"(?m)^(search:\s*)$", r"\1\n  collect_trace: true", patched, count=1)
        patched = patched2 if m else patched + "\nsearch:\n  collect_trace: true\n"
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(patched)
        return dest
    fd = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fd.write(patched)
    fd.close()
    return Path(fd.name)


def _lean_imports(rows: list[dict]) -> list[str]:
    """The Lean modules the search kernel must import for these manifest rows: the union of
    every row's `imports` (`"import Foo"` → `Foo`), in first-seen order, `Mathlib` first when
    present and as the fallback when no row declares any. Each calling repo's definitions live
    in a module of the search project (built with `lake build`), named on the row's `imports`;
    nothing is hard-coded here, so a project's own module reaches the kernel."""
    seen: list[str] = []
    for row in rows:
        for line in row.get("imports") or []:
            name = str(line).strip()
            if name.startswith("import "):
                name = name[len("import ") :].strip()
            if name and name not in seen:
                seen.append(name)
    if not seen:
        return ["Mathlib"]
    if "Mathlib" in seen:
        seen.remove("Mathlib")
        seen.insert(0, "Mathlib")
    return seen


def _manifest_ids(man: Path) -> list[str]:
    return [json.loads(ln)["theorem_id"] for ln in man.read_text().splitlines() if ln.strip()]


def _def_preamble(row: dict) -> str:
    """The local context the theorem's statement needs, pulled from its source file, in source order:
    its `def`/`abbrev` blocks, and the `variable`/`universe` commands in scope at the theorem.

    The manifest packages only the theorem statement; theorems like survivingDOF_antitone reference a
    local def (survivingDOF) that must be in scope for BOTH Wayfinder and Aristotle, and statements
    like T7.17's use a section's `variable {α : Type*} [DecidableEq α]` (without it `Finset.biUnion`
    has no instance). Variables of a section closed before the theorem do not apply to it and are
    left out. Returns "" if there is nothing.
    """
    src = LEAN_PROJECT / row.get("file_path", "")
    if not src.exists():
        return ""
    lines = src.read_text().splitlines()
    stop = int(row.get("theorem_line") or 0) or len(lines) + 1  # 1-based line of the theorem
    items: list[list] = []  # [text, scope id, is a variable/universe command]
    scopes = [0]
    next_scope = 1
    closed: set[int] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        is_def = re.match(r"^(noncomputable\s+)?(def|abbrev)\s", line)
        is_var = re.match(r"^(variable|universe)\b", line) and i + 1 < stop
        if re.match(r"^(section|namespace)\b", line):
            scopes.append(next_scope)
            next_scope += 1
        elif re.match(r"^end\b", line) and len(scopes) > 1:
            closed.add(scopes.pop())
        if is_def or is_var:
            blk = [line]
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                blk.append(lines[i])
                i += 1
            items.append(["\n".join(blk), scopes[-1], bool(is_var)])
            continue
        i += 1
    if stop <= len(lines):
        # scopes still open at the theorem; a variable outside them was closed before it
        closed_before = {sid for sid in closed if sid not in scopes}
    else:
        closed_before = set()
    return "\n\n".join(text for text, sid, is_var in items if not (is_var and sid in closed_before))


def _augmented_declaration(row: dict) -> str:
    """full_declaration with any needed local defs prepended (so it elaborates standalone)."""
    pre = _def_preamble(row)
    return f"{pre}\n\n{row['full_declaration']}" if pre else row["full_declaration"]


def _reconstruct_proof(
    row: dict,
    tactics_used: list[str],
    provenance: tuple[str, ...] = (),
    source_run: str = "",
) -> str:
    """Assemble Wayfinder's closing tactic sequence into a WARRANTED candidate .lean proof.

    Wave-3 emission (THE_REFOUNDING §5): the artifact goes through the typed
    LeanExpr seam — imports/opens/environment first-class, the warrant
    (lane provenance, source run, verify environment) as a Lean comment
    header — instead of bare string assembly. The emitted file is
    self-documenting evidence; the compile surface is unchanged.
    """
    head = _augmented_declaration(row).split(":= by", 1)[0].rstrip() + " := by"
    warrant = EmissionWarrant(
        theorem_id=str(row.get("theorem_id", "")),
        provenance=tuple(provenance),
        source_run=source_run,
        note="reconstructed from the verified search lineage",
    )
    expr = from_tactics(
        tactics_used,
        warrant,
        imports=tuple(row.get("imports") or ["import Mathlib"]),
        opens=tuple(row.get("opens") or ()),
        environment=VERIFY_ENV,
    )
    return render_declaration(head, expr)


def _elaboration_probe(row: dict) -> tuple[bool, str]:
    """Compile the theorem STATEMENT (sorry included) in the SEARCH project to surface real Lean errors.

    A `close_lane=skipped, attempts=0` result means `run_benchmark._resolve_initial_goal` returned None
    — i.e. Pantograph could not build the initial goal. The overwhelmingly common cause is that the
    STATEMENT ITSELF does not elaborate (a missing instance / typo / undefined name), NOT a harness-setup
    or onboarding problem, and NOT a residual. That real Lean error is otherwise swallowed, so an opaque
    "skipped" gets misread as "send to Aristotle". This probe re-surfaces it so the failure self-explains.

    Returns (elaborates, error_tail). `elaborates` is True iff `lake env lean` reports no `error:` (a bare
    `sorry` yields only a warning, which is fine — we are checking the statement, not the proof).
    """
    imports = "\n".join(row.get("imports") or ["import Mathlib"])
    opens = "\n".join(row.get("opens") or [])
    probe = "\n".join(x for x in [imports, opens, "", _augmented_declaration(row)] if x) + "\n"
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(row.get("theorem_id", "thm")))
    fpath = LEAN_PROJECT / f"_elab_probe_{safe}.lean"
    try:
        fpath.write_text(probe)
        p = subprocess.run(
            [str(LAKE_BIN), "env", "lean", str(fpath)],
            cwd=str(LEAN_PROJECT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = p.stdout + p.stderr
        elaborates = p.returncode == 0 and "error:" not in out
        return elaborates, out[-1000:]
    except subprocess.TimeoutExpired:
        return True, "elaboration probe timed out (>300s) — inconclusive"
    finally:
        try:
            fpath.unlink()
        except OSError:
            pass


def _verify_at_v428(lean_text: str, thm_name: str, allowed_axioms: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Compile `lean_text` in the v4.28 environment; True iff it elaborates with no error and
    no `sorry`, and `#print axioms` names only the standard three (plus `allowed_axioms`).

    The soundness gate for a Wayfinder-closed proof — a declared axiom no longer passes just
    because it is not `sorryAx`. Compiled from a temp file; the project tree is never written to.
    """
    run = gate.run_lean_json(LAKE_BIN, VERIFY_PROJECT, lean_text.rstrip() + gate.probe_suffix(thm_name), timeout=420)
    if run["timed_out"]:
        return False, "verify timeout (>420s)"
    msgs = run["messages"]
    errors = gate.errors_in(msgs)
    axioms = gate.axioms_from_messages(msgs, thm_name)
    allowed = set(gate.STANDARD_AXIOMS) | set(allowed_axioms)
    ok = (
        run["returncode"] == 0
        and not errors
        and not gate.uses_sorry(msgs)
        and axioms is not None
        and set(axioms) <= allowed
    )
    detail = errors[:3] or ([f"axioms: {axioms}"] if axioms is not None else ["`#print axioms` gave no report"])
    return ok, "; ".join(detail)[-800:]


# ── the residual hand-off: Wayfinder's stopping state as Lean with holes ────────


def _lean_name(row: dict) -> str:
    """The name the row's declaration declares (the theorem_id can differ, e.g. minif2f ids)."""
    return gate.name_from_declaration(str(row.get("full_declaration", ""))) or str(row.get("theorem_id", ""))


def _statement_sha(row: dict) -> str:
    header, _ = rh.split_declaration(str(row.get("full_declaration", "")))
    return hashlib.sha256(gate.normalize(header).encode()).hexdigest()


def _file_with_declaration(row: dict, declaration: str) -> str:
    """The standalone hand-off file (`_proof_lean`) with the row's declaration replaced."""
    base = _proof_lean(row)
    original = str(row["full_declaration"])
    at = base.rindex(original)
    return base[:at] + declaration + base[at + len(original) :]


def _compile_v428(text: str, timeout: int = 420) -> dict:
    return gate.run_lean_json(LAKE_BIN, VERIFY_PROJECT, text, timeout=timeout)


def _under_forced(tree: rh.Tree, nid: int, forced: set[int]) -> bool:
    """Is some ancestor of `nid` a forced hole (so `nid`'s subtree is no longer rendered)?"""
    p = tree.nodes[nid].parent
    while p >= 0:
        if p in forced:
            return True
        p = tree.nodes[p].parent
    return False


def _skeleton_attempt(
    row: dict,
    header: str,
    tree: rh.Tree,
    revert: list[str] | None,
    deadline: float,
    forced: set[int] | None = None,
    fills: dict[int, str] | None = None,
) -> dict | None:
    """Compile-repair loop for one form. Returns the compiling candidate, or None.

    Starts from `forced` holes and `fills` (holes the finishing pass closed) when given; a fill
    that breaks the combined file goes back to being a plain hole.
    """
    forced = set(forced or ())
    fills = dict(fills or {})
    notes: list[str] = []
    for _ in range(HANDOFF_MAX_REPAIRS + 1):
        if time.monotonic() > deadline:
            notes.append("hand-off build budget spent")
            return None
        probe = rh.render(tree, revert=revert, forced_holes=forced, probe=True, fills=fills)
        text = _file_with_declaration(row, rh.assemble(header, probe))
        first = rh.body_first_line(text, header)
        run = _compile_v428(text)
        if run["timed_out"]:
            return None
        if not gate.errors_in(run["messages"]):
            goals = rh.holes_from_messages(run["messages"], probe, first)
            ok, rows = rh.faithful(tree, goals, probe.holes)
            return {"revert": revert, "forced": forced, "fills": fills, "goals": goals, "faithful": ok,
                    "rows": rows, "kept": len(probe.kept), "notes": notes}  # fmt: skip
        targets, more = rh.repair_targets(tree, run["messages"], probe, first)
        notes += more
        bad_fills = targets & set(fills)
        for nid in bad_fills:
            fills.pop(nid)
        targets -= bad_fills
        if not targets - forced and not bad_fills:
            return None
        forced |= targets
        fills = {nid: t for nid, t in fills.items() if not _under_forced(tree, nid, forced)}
    return None


def _file_defs(row: dict) -> list[str]:
    """The definitions the hand-off file itself declares — what a closer may need to unfold."""
    names = []
    for chunk in gate.top_level_chunks(_proof_lean(row)):
        if gate.declaration_kind(chunk) in ("def", "abbrev"):
            names.append(gate.declaration_name(chunk))
    return [n for n in dict.fromkeys(names) if n]


def _finish_locally(row: dict, header: str, tree: rh.Tree, best: dict, deadline: float) -> tuple[dict, dict]:
    """The local finishing pass — as much as can be done here is done here.

    Each hole is tried with a battery of deterministic closers (unfolding the file's own
    definitions it mentions); a hole that closes is filled. Each hole that does not close is
    tried for a counterexample (`plausible`): a refuted hole means the step that produced it
    committed to something false (a `left`, a witness), so that step is given back — its
    parent becomes the hole — and the new hole is tried in turn. A refutation that reaches
    the root means the statement itself is false. Returns (candidate, report).
    """
    target = _lean_name(row)
    defs = _file_defs(row)
    report: dict = {"closed": [], "backed_off": [], "refuted_statement": None, "notes": []}
    fills = dict(best.get("fills") or {})
    for _ in range(HANDOFF_FINISH_ROUNDS):
        if time.monotonic() > deadline:
            report["notes"].append("finishing pass budget spent")
            break
        current = rh.render(tree, revert=best["revert"], forced_holes=best["forced"], fills=fills,
                            hole_goals=best["goals"])  # fmt: skip
        if not current.holes:
            break
        battery = [
            rh.closers(rh.defs_mentioned(best["goals"].get(nid, ""), defs), best["goals"].get(nid, ""))
            for nid in current.holes
        ]
        probe = rh.triage_block(rh.assemble(header, current), target, battery, heartbeats=1000000)
        text = _file_with_declaration(row, probe.text)
        first = text[: text.rindex(probe.text)].count("\n") + 1
        run = _compile_v428(text, timeout=600)
        if run["timed_out"]:
            report["notes"].append("finishing probe timed out")
            break
        outcomes = rh.triage_outcomes(run["messages"], probe, first)
        refuted = []
        for nid, o in zip(current.holes, outcomes):
            goal = rh.target_of(best["goals"].get(nid, "") or tree.nodes[nid].goal)
            if o["closed_by"]:
                fills[nid] = o["closed_by"]
                report["closed"].append({"goal": goal, "by": o["closed_by"]})
            elif o["refuted"]:
                refuted.append((nid, goal, o["refuted"]))
        if not refuted:
            break
        forced = set(best["forced"])
        for nid, goal, cex in refuted:
            parent = tree.nodes[nid].parent
            step = tree.nodes[parent].tactic if parent >= 0 else None
            report["backed_off"].append({"hole_goal": goal, "counterexample": cex, "step_given_back": step})
            if parent < 0 or tree.nodes[parent].closure < 0:
                report["refuted_statement"] = cex
            else:
                forced.add(parent)
        if report["refuted_statement"]:
            break
        fills = {nid: t for nid, t in fills.items() if not _under_forced(tree, nid, forced)}
        cand = _skeleton_attempt(row, header, tree, best["revert"], deadline, forced=forced, fills=fills)
        if cand is None:
            report["notes"].append("the skeleton did not compile after giving a step back")
            break
        best, fills = cand, cand["fills"]
    return {**best, "fills": fills}, report


def _build_handoff(row: dict, d: dict) -> dict:
    """Wayfinder's stopping state as a verified Lean skeleton, finished locally as far as it goes.

    status: "closed" (no holes — Wayfinder's proof, with the finishing pass's closes),
    "residual" (holes left for the next prover), "bare" (no replayable prefix and nothing
    closed: the residual is the whole statement), "refuted" (the statement itself has a
    counterexample — nothing to send), or "failed" (the statement does not elaborate).
    `faithful` / `matches_stopping_state` describe Wayfinder's stopping state, before the
    finishing pass; `local` reports what the finishing pass closed and gave back.
    """
    target = _lean_name(row)
    header, _ = rh.split_declaration(str(row["full_declaration"]))
    out: dict = {"target": target, "statement_sha256": _statement_sha(row), "verified_env": VERIFY_ENV,
                 "initial_goal_tier": d.get("initial_goal_tier", ""), "notes": []}  # fmt: skip
    tree: rh.Tree | None = None
    if d.get("close_log") is None or not d.get("initial_goals"):
        out["notes"].append("no closure log in the search result (run predates the instrumentation)")
    else:
        try:
            tree = rh.build_tree(list(d["initial_goals"]), list(d["close_log"]))
        except rh.ReplayError as e:
            out["notes"].append(f"closure log did not rebuild: {e}")
    deadline = time.monotonic() + HANDOFF_BUDGET_S
    best = None
    if tree is not None and any(n.closure >= 0 for n in tree.nodes):
        names = rh.binder_names(header)
        closed_type = out["initial_goal_tier"] in ("A", "env_inspect", "raw")
        forms = [names or None, None] if closed_type else [None, names or None]
        for revert in dict.fromkeys(tuple(f) if f else None for f in forms):
            cand = _skeleton_attempt(row, header, tree, list(revert) if revert else None, deadline)
            if cand and (best is None or (cand["faithful"], cand["kept"]) > (best["faithful"], best["kept"])):
                best = cand
            if best and best["faithful"]:
                break
    if best is None:
        # No replayable prefix: the residual is the whole statement. One probe
        # compile still reads back the goal Lean shows (and that it elaborates).
        start_goal = tree.nodes[0].goal if tree is not None and tree.nodes else str(row.get("goal_state", ""))
        tree = rh.Tree(nodes=[rh.Node(goal=start_goal)], roots=[0], open_ids=[0])
        empty = {"revert": None, "forced": set(), "fills": {}, "goals": {}, "faithful": False, "rows": [],
                 "kept": 0, "notes": []}  # fmt: skip
        best = _skeleton_attempt(row, header, tree, None, deadline + 120) or empty
        out["notes"].append("no replayable prefix: the residual is the whole statement")
    assert tree is not None  # the bare fallback above always leaves a tree

    # Wayfinder's stopping state, as the search left it (before anything is finished here).
    stop = rh.render(tree, revert=best["revert"], forced_holes=best["forced"], hole_goals=best["goals"])
    out["faithful"] = bool(best["faithful"])
    # The stronger claim: the holes are exactly what the search left open (de-duplicated —
    # the kernel can report a goal twice — and with harness names stripped). A repair that
    # dropped a real step fails this even when every hole matches its own node.
    if d.get("final_goals") is None:
        out["matches_stopping_state"] = None
    else:
        left_open = {rh.target_of(g) for g in d.get("final_goals") or []}
        at_holes = {rh.target_of(best["goals"].get(nid, "")) for nid in stop.holes if best["goals"].get(nid)}
        out["matches_stopping_state"] = len(at_holes) == len(stop.holes) and at_holes == left_open
    out["stopping_holes"] = len(stop.holes)

    best, local = _finish_locally(row, header, tree, best, time.monotonic() + HANDOFF_FINISH_BUDGET_S)
    out["local"] = local
    fills = dict(best.get("fills") or {})
    for _ in range(3):  # the combined file must still compile; a fill that breaks it is dropped
        final = rh.render(tree, revert=best["revert"], forced_holes=best["forced"], hole_goals=best["goals"],
                          fills=fills)  # fmt: skip
        declaration = rh.assemble(header, final)
        text = _file_with_declaration(row, declaration)
        run = _compile_v428(text + gate.probe_suffix(target))
        errors = gate.errors_in(run["messages"])
        first = rh.body_first_line(text, header)
        broken = {
            final.line_node.get(int((m.get("pos") or {}).get("line", -1)) - first)
            for m in run["messages"]
            if m.get("severity") == "error"
        } & set(fills)
        if not errors or not broken:
            break
        for nid in broken:
            fills.pop(nid)
        local["notes"].append(f"{len(broken)} local close(s) dropped: they did not compile together")
    out["notes"] += best["notes"]
    out["file"] = text
    out["declaration"] = declaration
    out["pp_type"] = gate.checked_type(run["messages"], target)
    out["axioms"] = gate.axioms_from_messages(run["messages"], target)
    out["holes"] = [
        {"index": k + 1, "lean_goal": best["goals"].get(nid, ""), "search_goal": rh.target_of(tree.nodes[nid].goal)}
        for k, nid in enumerate(final.holes)
    ]
    out["prefix"] = [tree.nodes[nid].tactic for nid in final.kept] + [fills[nid] for nid in final.filled]
    out["kept_provenance"] = [tree.nodes[nid].provenance for nid in final.kept]
    out["closures_total"] = len(d.get("close_log") or [])
    out["closures_kept"] = len(final.kept)
    out["closed_locally"] = len(final.filled)
    out["form"] = "revert" if best["revert"] else "direct"
    if local.get("refuted_statement"):
        out["status"] = "refuted"
    elif errors or out["pp_type"] is None:
        out["status"] = "failed"
        out["errors"] = errors[:5]
    elif not final.holes:
        out["status"] = "closed"
    elif not final.kept and not final.filled:
        out["status"] = "bare"
    else:
        out["status"] = "residual"
    return out


def _handoff_prompt(theorem_id: str, name: str, handoff: dict) -> str:
    """The finish instruction: Wayfinder's verified prefix stays; only the holes are filled."""
    target = handoff["target"]
    holes = handoff.get("holes") or []
    if handoff.get("status") == "residual":
        lines = [
            f"`Residual.lean` contains theorem `{target}` ({name}) with a partial proof. Every tactic line already in "
            f"the proof was produced and kernel-checked by Wayfinder; what remains is exactly {len(holes)} `sorry` "
            f"placeholder(s), each marked `-- {rh.HOLE_MARK} k` with the goal it must close.",
            "Finish the proof: replace each marked `sorry` with a proof of its goal. Do not change the theorem "
            "statement, any other declaration in the file, or any existing tactic line, and do not restructure the "
            "proof. You may add helper lemmas above the theorem. Do not add axioms, `sorry`, or `admit`.",
            "",
            "The goals to close:",
        ]
        for h in holes:
            lines.append(f"{rh.HOLE_MARK} {h['index']}:\n{h.get('lean_goal') or h.get('search_goal')}")
    else:
        lines = [
            f"Prove theorem `{target}` ({name}) in `Residual.lean` by replacing its `sorry`. Wayfinder made no "
            "replayable progress on it, so the residual is the whole statement.",
            "Do not change the theorem statement or any other declaration in the file. You may add helper lemmas "
            "above the theorem. Do not add axioms, `sorry`, or `admit`.",
        ]
    return "\n".join(lines)


def _handoff_sheet(detail: dict, row: dict, handoff: dict) -> str:
    """The cheat sheet as the hand-off's companion: it describes the file it travels with.

    The search's own records name things the file does not (descriptor tactics such as
    `rw_seq(3)`, harness-renamed definitions, goals the kernel reported twice), so the sheet is
    rendered from a view whose progress is the skeleton's kept Lean prefix and whose residual is
    the goals Lean prints at its holes — with a closing line that agrees with the instruction.
    """
    view = {
        **detail,
        "step_trace": [],
        "tactics_used": list(handoff.get("prefix") or []),
        # the Lean-printed goal (hypotheses and all) — the reads use the context
        "final_goals": [h.get("lean_goal") or h.get("search_goal", "") for h in handoff.get("holes") or []],
    }
    line = (
        "hand-off: the file holds Wayfinder's verified partial proof; fill only the marked `sorry` holes, "
        "keep every existing line, and do not add axioms or `sorry`."
        if handoff.get("status") == "residual"
        else "hand-off: prove the statement by replacing its `sorry`; do not add axioms or `sorry`."
    )
    return render_cheat_sheet(view, row, handoff_line=line)


def _legacy_ids() -> set[str]:
    """Theorems routed before `proof_home` existed, whose fixed routes stay: the default
    manifest, the Ansatz cores, and every theorem the Aristotle index already filed somewhere."""
    ids = set(ANSATZ_THEOREMS)
    try:
        ids |= set(_manifest_ids(DEFAULT_MANIFEST))
    except (OSError, ValueError, KeyError):
        pass
    ids |= {str(rec.get("theorem_id")) for rec in _load_index().values() if rec.get("home") and rec.get("theorem_id")}
    return ids


def _accept_home(row: dict) -> Path | None:
    """Where an accepted proof may be saved: the row's declared home, or a legacy route.

    A row with neither gets None — its proof stays in quarantine rather than
    falling through to another paper's proofs/ project.
    """
    if str(row.get("proof_home") or "").strip():
        return _proof_home(row)
    if str(row.get("theorem_id")) in _legacy_ids():
        return _proof_home(row)
    return None


def _acceptance_header(res: gate.GateResult, provenance: str) -> str:
    rows = [
        f"theorem: {res.target}",
        f"verified in: {res.environment}",
        f"axioms: {', '.join(res.axioms or []) or 'none'}",
        "statement: unchanged (text and elaborated type)",
        f"provenance: {provenance}",
        f"mode: {res.mode}",
    ]
    body = "\n".join(f"  {r}" for r in rows)
    return f"/- wayfinder acceptance\n{body}\n-/\n"


@mcp.tool()
def prove(
    theorem_id: str,
    manifest: str = str(DEFAULT_MANIFEST),
    device: str = "mps",
    timeout: int = 900,
) -> str:
    """Prove ONE theorem with the real Wayfinder pipeline (Pantograph + Dr. Ducky, all lanes).

    Runs `run_benchmark.py` end-to-end on the single theorem's manifest row: the neural
    navigator predicts a direction, the proof network resolves premises/tactics, and the
    lane search (structural / cosine / hammer / Dr. Ducky) attempts to close every goal
    against a live Lean (Pantograph) kernel. Returns a JSON summary: whether Wayfinder
    proved it, the closing lane, and any residual goal (which is what would go to Aristotle).

    Args:
        theorem_id: the theorem to prove (must be a row in `manifest`).
        manifest: JSONL manifest containing the theorem's row (theorem_id, file_path,
                  full_declaration, imports, opens). Default: the 9-theorem rerun manifest.
        device: torch device ("mps", "cpu", "cuda").
        timeout: seconds (torch load + Pantograph + Mathlib + search).
    """
    man = Path(manifest).expanduser().resolve()
    if not man.exists():
        return f"error: manifest not found: {man}"

    rows = [ln for ln in man.read_text().splitlines() if ln.strip() and json.loads(ln).get("theorem_id") == theorem_id]
    if not rows:
        return f"error: '{theorem_id}' not in {man.name}. Available: {_manifest_ids(man)}"

    row = json.loads(rows[0])
    cfg = _resolved_config()
    with tempfile.TemporaryDirectory() as td:
        one = Path(td) / "one_theorem.jsonl"
        # Send run_benchmark a row whose declaration carries any local defs, so goal resolution
        # (load_sorry) sees them. Keep `row` original so _reconstruct_proof augments exactly once.
        run_row = {**row, "full_declaration": _augmented_declaration(row)}
        one.write_text(json.dumps(run_row) + "\n")
        out = Path(td) / "result.json"
        cmd = [
            str(VENV_PY),
            str(BENCH),
            "--config",
            str(cfg),
            "--checkpoint",
            str(CHECKPOINT),
            "--backend",
            "pantograph",
            "--lean-project",
            str(LEAN_PROJECT),
            "--lean-imports",
            *_lean_imports([row]),
            "--theorems",
            str(one),
            "--limit",
            "1",
            "--search-mode",
            "full",
            "--cosine-rw",  # scope → MiniLM → top-k beam + Lean verify (the lemma-retrieval lane)
            "--cosine-rw-seq",  # sequential bare-rewrite lane — the KEY closers in real runs
            "--device",
            device,
            "--output",
            str(out),
        ]
        env = {**os.environ, "PYTHONPATH": str(WF)}
        try:
            p = subprocess.run(
                cmd,
                cwd=str(WF),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s] {theorem_id} — Wayfinder still searching"
        finally:
            try:
                cfg.unlink()
            except OSError:
                pass

        if not out.exists():
            return f"no result file (exit {p.returncode}).\n{(p.stdout + p.stderr)[-1800:]}"
        r = json.loads(out.read_text())
        d = (r.get("details") or [{}])[0]
        proved = bool(d.get("success")) and bool(d.get("honest_success"))
        skipped = (not proved) and d.get("close_lane") == "skipped" and not d.get("attempts")
        name = _lean_name(row)
        allowed = tuple(row.get("allowed_axioms") or ())

        # A skip means the initial goal never built (attempts=0). DIAGNOSE it — almost always the
        # STATEMENT doesn't elaborate. This is NEVER a residual and NEVER an Aristotle candidate; it
        # is a declaration to fix or a harness issue to debug. Surfacing it here (not in my memory)
        # is what stops the recurring "attempts=0 → onboard the harness / send to Aristotle" misread.
        statement_elaborates = None
        statement_error = None
        handoff: dict | None = None
        if skipped:
            statement_elaborates, statement_error = _elaboration_probe(row)
        else:
            # The stopping state as Lean: the committed tree replayed with a `sorry` at every open
            # goal, compiled until it elaborates. A closed proof is the zero-hole case.
            try:
                handoff = _build_handoff(row, d)
            except (rh.ReplayError, ValueError) as e:
                handoff = {"status": "failed", "notes": [f"hand-off not built: {e}"]}

        # Wayfinder closed it → the tree's zero-hole replay, else the linear trace; saved only if
        # the gate accepts it (compiles, no sorry, only allowlisted axioms).
        verified = None
        saved_to = None
        verify_detail = None
        closed_here = bool(handoff and handoff.get("status") == "closed")
        if proved or closed_here:
            candidates = []
            if closed_here:
                finished = (handoff or {}).get("closed_locally") or 0
                warrant = EmissionWarrant(
                    theorem_id=str(theorem_id),
                    provenance=tuple((handoff or {}).get("kept_provenance") or ())
                    + (("local_finish",) if finished else ()),
                    source_run=str(out),
                    note=(
                        "replayed from the committed closure tree"
                        + (f"; {finished} hole(s) closed by the local finishing pass" if finished else "")
                        if (handoff or {}).get("closures_kept")
                        else "closed entirely by the local finishing pass (no search prefix was kept)"
                    ),
                )
                candidates.append(warrant_header(warrant, VERIFY_ENV) + "\n\n" + (handoff or {})["file"])
            if proved:
                candidates.append(
                    _reconstruct_proof(
                        row,
                        d.get("tactics_used") or [],
                        provenance=tuple(d.get("close_provenance") or ()),
                        source_run=str(out),
                    )
                )
            for candidate in candidates:
                verified, verify_detail = _verify_at_v428(candidate, name, allowed)
                if verified:
                    home = _accept_home(row)
                    saved_to = str(_save_solved_proof(row, candidate)) if home else None
                    break

        # Persist the goal-ledger row: the accreting per-theorem source of truth the hand-off
        # (cheat sheet + skeleton) renders from at send time.
        try:
            GOAL_LEDGER.mkdir(parents=True, exist_ok=True)
            (GOAL_LEDGER / f"{theorem_id}.json").write_text(
                json.dumps({"detail": d, "manifest_row": row, "handoff": handoff}, indent=2)
            )
        except OSError:
            pass  # the ledger is an artifact, never a gate on the prove result

        h = handoff or {}
        status = h.get("status")
        if proved:
            how = "proved by Wayfinder's search"
        elif h.get("closures_kept"):
            how = "closed locally (Wayfinder's search prefix, finished by its local finishing pass)"
        else:
            how = "closed by Wayfinder's local finishing pass alone (the search kept no step)"
        if verified and saved_to:
            nxt = f"{how}, verified @v4.28 → saved to {saved_to}. No Aristotle needed."
        elif verified:
            nxt = f"{how} and verified @v4.28, but the row declares no proof_home: not saved."
        elif status == "refuted":
            cex = " ".join(str((h.get("local") or {}).get("refuted_statement", "")).split())[:400]
            nxt = (
                "REFUTED: the statement as written is false — a counterexample was found locally "
                f"({cex}). Fix the statement; nothing is sent to Aristotle."
            )
        elif skipped and statement_elaborates is False:
            nxt = (
                "SKIPPED: the STATEMENT DOES NOT ELABORATE (see statement_error). This is a broken "
                "declaration (missing instance / typo / undefined name) — FIX THE .lean + manifest and "
                "re-run prove(). NOT a residual, NOT a harness/onboarding bug, do NOT send to Aristotle."
            )
        elif skipped:
            nxt = (
                "SKIPPED though the statement elaborates → goal-resolution/harness issue "
                "(debug run_benchmark._resolve_initial_goal). NOT a residual, do NOT send to Aristotle."
            )
        elif status == "residual":
            nxt = (
                f"residual: Wayfinder's verified prefix ({h.get('closures_kept')} of {h.get('closures_total')} "
                f"closures, {h.get('closed_locally', 0)} hole(s) closed locally) with {len(h.get('holes') or [])} "
                "hole(s) left → send_to_aristotle hands Aristotle exactly these."
            )
            if _load_service_status().get("lean_only") == "refused":
                nxt += " NOTE: Aristotle currently refuses Lean-only projects; the residual stays in the ledger."
            if h.get("matches_stopping_state") is False or not h.get("faithful"):
                nxt += (
                    " NOTE: the holes are not exactly where Wayfinder stopped"
                    " (replay diverged or repair dropped steps)."
                )
        elif status == "bare":
            nxt = "no replayable progress: the residual is the whole statement → send_to_aristotle sends it as such."
            if _load_service_status().get("lean_only") == "refused":
                nxt += " NOTE: Aristotle currently refuses Lean-only projects; the residual stays in the ledger."
        else:
            nxt = "the hand-off skeleton did not compile (see handoff.notes): nothing to send until it does."

        summary = {
            "theorem_id": theorem_id,
            "proved_by_wayfinder": bool(d.get("success")),
            "honest_success": bool(d.get("honest_success")),
            "reconstructed_verified_v428": verified,
            "saved_to": saved_to,
            "close_lane": d.get("close_lane"),
            "goals_closed": d.get("goals_closed"),  # subgoals Wayfinder cleared
            "goals_remaining": d.get("goals_remaining"),  # what the search left open
            "attempts": d.get("attempts"),
            "time_s": r.get("efficiency", {}).get("avg_time_per_theorem_s"),
            "next_step": nxt,
        }
        if handoff:
            summary["handoff"] = {
                "status": status,
                "holes": len(handoff.get("holes") or []),
                "closures_kept": handoff.get("closures_kept"),
                "closures_total": handoff.get("closures_total"),
                "faithful": handoff.get("faithful"),
                "matches_stopping_state": handoff.get("matches_stopping_state"),
                "stopping_holes": handoff.get("stopping_holes"),
                "closed_locally": handoff.get("closed_locally"),
                "local": {k: v for k, v in (handoff.get("local") or {}).items() if v},
                "form": handoff.get("form"),
                "notes": (handoff.get("notes") or [])[:6],
                "hole_goals": [h.get("lean_goal", "")[-300:] for h in (handoff.get("holes") or [])[:3]],
            }
        if verified is False:
            summary["verify_tail"] = verify_detail
        if skipped:
            summary["statement_elaborates"] = statement_elaborates
            summary["statement_error"] = statement_error
        return json.dumps(summary, indent=2)


def _aristotle_key() -> str | None:
    """Read the Aristotle API key from $ARISTOTLE_API_KEY, else semantic_probing/.env. Never hardcoded."""
    k = os.environ.get("ARISTOTLE_API_KEY")
    if k and k.strip():
        return k.strip()
    if ARISTOTLE_ENV.exists():
        for ln in ARISTOTLE_ENV.read_text().splitlines():
            ln = ln.strip()
            if ln.startswith("ARISTOTLE_API_KEY") and "=" in ln:
                return ln.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _sane_project_name(theorem_id: str, row: dict) -> str:
    """Resolvable Aristotle project name: provenance + theorem name + proof number.

    Aristotle derives the project's identity from the submitted directory name
    (`create_from_directory` → `project_dir.name`), so this is what shows in the
    dashboard and lets it be found later.
    """
    stem = Path(row.get("file_path", "")).stem or "thm"  # e.g. proof_30
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"Wayfinder-{theorem_id}-{stem}")


def _lakefile(package: str, lib: str, roots_or_globs: str) -> str:
    """A lakefile.toml requiring mathlib at LEAN_VERSION. No manifest (lake generates a consistent one)."""
    return (
        f'name = "{package}"\nversion = "0.1.0"\ndefaultTargets = ["{lib}"]\n\n'
        '[[require]]\nname = "mathlib"\n'
        'git = "https://github.com/leanprover-community/mathlib4.git"\n'
        f'rev = "{LEAN_VERSION}"\n\n'
        f'[[lean_lib]]\nname = "{lib}"\n{roots_or_globs}\n'
    )


def split_imports(import_lines: list[str], local_modules: list[str]) -> dict[str, list[str]]:
    """Partition a row's `import X` lines into the ones a standalone Mathlib project can satisfy
    ("external", as import lines) and the calling repo's own modules ("local", bare names, in order).

    Pure: `local_modules` is the list of module names that resolve to a file of the search project.
    Blank entries and a missing `import` keyword are tolerated, as `_lean_imports` tolerates them.
    """
    external: list[str] = []
    local: list[str] = []
    for line in import_lines or []:
        name = str(line or "").strip()
        if name.startswith("import "):
            name = name[len("import ") :].strip()
        if not name:
            continue
        if name in local_modules:
            if name not in local:
                local.append(name)
        else:
            external.append(f"import {name}")
    return {"external": external, "local": local}


def strip_import_lines(source: str) -> str:
    """A module's source with its `import` lines removed, so it can be inlined below other imports."""
    kept = [ln for ln in source.splitlines() if not ln.strip().startswith("import ")]
    return "\n".join(kept).strip("\n")


def _local_module_file(name: str) -> Path | None:
    """The search project's source file for a bare module name, if the module is local to it."""
    for cand in (LEAN_PROJECT / "lib" / f"{name}.lean", LEAN_PROJECT / f"{name}.lean"):
        if cand.exists():
            return cand
    return None


def _proof_lean(row: dict) -> str:
    """The Lean source for one theorem, standalone: external imports, then the calling repo's
    definition modules inlined (a hosted prover's project cannot `import` them), then opens, local
    defs and the (sorry-bearing) declaration.
    """
    lines = [str(x) for x in (row.get("imports") or ["import Mathlib"])]
    names = [ln.strip().removeprefix("import ").strip() for ln in lines]
    local_files = {n: f for n in names if n for f in [_local_module_file(n)] if f is not None}
    parts = split_imports(lines, list(local_files))
    imports = "\n".join(parts["external"] or ["import Mathlib"])
    inlined = "\n\n".join(strip_import_lines(local_files[n].read_text()) for n in parts["local"])
    opens = "\n".join(row.get("opens") or [])
    return "\n".join(x for x in [imports, inlined, opens, "", _augmented_declaration(row)] if x) + "\n"


def proofs_roots(modules: list[str]) -> str:
    """The `roots = [...]` line for a proofs/ project whose files sit flat in the directory.

    Lake's `globs` take module patterns (`Foo.*`), never a bare `*`, so a flat proof home lists
    its modules explicitly; regenerated from the directory on every save. Sorted, de-duplicated,
    blanks dropped, so the lakefile is deterministic in the directory's contents.
    """
    names = sorted({str(m).strip() for m in modules or [] if str(m).strip()})
    return "roots = [" + ", ".join(f'"{n}"' for n in names) + "]"


def _write_aristotle_project(proj_dir: Path, source: str) -> Path:
    """Write a minimal standalone Lean project (v4.28.0, no manifest) into `proj_dir`.

    `source` is the hand-off file (the residual skeleton). `proj_dir`'s basename names the project
    on SDKs that name it after the directory. No lake-manifest is written: a v4.27 manifest would
    poison the v4.28 build, and Aristotle resolves mathlib itself.
    """
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "Residual.lean").write_text(source)
    (proj_dir / "lakefile.toml").write_text(_lakefile("AristotleResidual", "Residual", 'roots = ["Residual"]'))
    (proj_dir / "lean-toolchain").write_text(f"leanprover/lean4:{LEAN_VERSION}\n")
    return proj_dir


# ── Backgrounded-project index (project_id → theorem context) ───────────────────


def _load_index() -> dict:
    if ARISTOTLE_INDEX.exists():
        try:
            return json.loads(ARISTOTLE_INDEX.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_index(idx: dict) -> None:
    ARISTOTLE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    ARISTOTLE_INDEX.write_text(json.dumps(idx, indent=2))


# ── what the hosted service currently accepts ──────────────────────────────────
# Aristotle 3.0 (2026-10-27) verifies Verilog, C and Rust; a Lean-only project is refused at
# run time ("Lean-only projects aren't supported right now"). The refusal is recorded so the
# next hand-off is declined locally, with the reason, instead of spending the account's one
# concurrent run on a predictable refusal; an accepted proof clears the record.
ARISTOTLE_SERVICE = WF / "runs" / "aristotle" / "service_status.json"
_REFUSAL_MARKERS = (
    "aren't supported",
    "are not supported",
    "not supported right now",
    "no source files in a supported language",
)


def _service_refusal(summary: str | None) -> bool:
    """Is a run's summary the service refusing the project (not a proof attempt that failed)?"""
    text = (summary or "").lower()
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _load_service_status() -> dict:
    try:
        return json.loads(ARISTOTLE_SERVICE.read_text())
    except (OSError, ValueError):
        return {}


def _save_service_status(status: dict) -> None:
    ARISTOTLE_SERVICE.parent.mkdir(parents=True, exist_ok=True)
    ARISTOTLE_SERVICE.write_text(json.dumps(status, indent=2))


def _save_solved_proof(row: dict, solved_lean: str) -> Path:
    """Save a finished proof into the calling repo's clean proofs/ project (v4.28.0 env).

    One proofs/ Lean project per repo (shared toolchain + lakefile, mathlib globbed), chosen by
    the row's `proof_home`; the solved proof lands as `<theorem_id>.lean`. Verify with
    `lake env lean <file>` + `#print axioms` (after `lake update && lake exe cache get` once per
    repo).
    """
    theorem_id = str(row.get("theorem_id") or "thm")
    home = _proof_home(row)
    home.mkdir(parents=True, exist_ok=True)
    lib = "Proofs"
    toolchain = home / "lean-toolchain"
    if not toolchain.exists():
        toolchain.write_text(f"leanprover/lean4:{LEAN_VERSION}\n")
    proof_path = home / f"{theorem_id}.lean"
    proof_path.write_text(solved_lean)
    # the lakefile lists every module in the home, regenerated so the project always builds
    modules = [p.stem for p in home.glob("*.lean")]
    (home / "lakefile.toml").write_text(_lakefile(f"{home.parent.name}_proofs", lib, proofs_roots(modules)))
    return proof_path


def _create_kwargs(aristotlelib, name: str) -> dict:
    """Unattended-run options for `Project.create_from_directory`, as the installed SDK takes them.

    Questions are always DISABLED: nobody answers an MCP-submitted run, and since 3.0 the SDK's
    default lets Aristotle stop and wait 15 minutes per question. The project is named explicitly
    where the SDK accepts a name (3.x); 2.x names it after the uploaded directory.
    """
    params = inspect.signature(aristotlelib.Project.create_from_directory).parameters
    kwargs: dict = {}
    if "agent_questions_setting" in params:
        kwargs["agent_questions_setting"] = aristotlelib.AgentQuestionsSetting.DISABLED
    if "name" in params:
        kwargs["name"] = name
    return kwargs


@mcp.tool()
async def send_to_aristotle(
    theorem_id: str,
    manifest: str = str(DEFAULT_MANIFEST),
    prompt: str = "",
    force: bool = False,
) -> str:
    """Hand Aristotle the RESIDUAL Wayfinder left — its verified partial proof, `sorry` only on the
    goals still open — and return immediately with the project id.

    Requires a fresh `prove(theorem_id)` run: the hand-off skeleton it built (and compiled at v4.28)
    is what gets submitted, with an instruction to fill exactly those holes and keep everything else.
    If Wayfinder made no replayable progress, the residual is the whole statement and is sent as such.
    Collect with `aristotle_result(project_id)`; the result is accepted only through the gate there.

    METERED hosted call. Key from $ARISTOTLE_API_KEY or semantic_probing/.env (never hardcoded).

    Args:
        theorem_id: theorem to hand off (a row in `manifest`, already run through `prove`).
        manifest: JSONL manifest. Default: the rerun manifest.
        prompt: optional text appended to the finish instruction.
        force: send even though Aristotle last refused Lean-only projects (to check whether it
            takes them again).
    """
    import aristotlelib  # lazy: the MCP host's SDK

    key = _aristotle_key()
    if not key:
        return (
            "error: no Aristotle API key. Set $ARISTOTLE_API_KEY or add "
            f"ARISTOTLE_API_KEY=... to {ARISTOTLE_ENV}, then retry."
        )

    man = Path(manifest).expanduser().resolve()
    if not man.exists():
        return f"error: manifest not found: {man}"
    rows = [ln for ln in man.read_text().splitlines() if ln.strip() and json.loads(ln).get("theorem_id") == theorem_id]
    if not rows:
        return f"error: '{theorem_id}' not in {man.name}. Available: {_manifest_ids(man)}"
    row = json.loads(rows[0])
    name = _sane_project_name(theorem_id, row)

    # The residual must exist, be current, and compile — or nothing is sent.
    ledger_path = GOAL_LEDGER / f"{theorem_id}.json"
    try:
        ledger = json.loads(ledger_path.read_text())
    except (OSError, ValueError):
        return f"error: no goal-ledger entry for '{theorem_id}'. Run prove('{theorem_id}') first."
    handoff = ledger.get("handoff") or {}
    status = handoff.get("status")
    if status not in ("residual", "bare"):
        refutation = " ".join(str((handoff.get("local") or {}).get("refuted_statement", "")).split())[:300]
        why = {
            "closed": "Wayfinder already closed it — nothing to hand off",
            "refuted": f"the statement is false as written (counterexample found locally: {refutation})",
            "failed": f"its hand-off skeleton did not compile ({'; '.join(handoff.get('notes') or [])[:300]})",
            None: "the ledger entry predates the residual hand-off; re-run prove()",
        }.get(status, f"hand-off status is {status!r}")
        return f"error: not sent — {why}."
    if handoff.get("statement_sha256") != _statement_sha(row):
        return f"error: not sent — the manifest's statement changed since prove(); re-run prove('{theorem_id}')."

    service = _load_service_status()
    if service.get("lean_only") == "refused" and not force:
        return (
            f"error: not sent — Aristotle refused a Lean-only project on {service.get('seen_at')} "
            f"(project {service.get('project_id')}): {service.get('message')!r}. Aristotle 3.x verifies Verilog, "
            "C and Rust code; a Lean theorem is not a submission it takes. The residual stays in the goal ledger. "
            "Pass force=True to check whether it takes Lean again."
        )

    instruction = _handoff_prompt(theorem_id, name, handoff)
    try:
        sheet = _handoff_sheet(ledger.get("detail") or {}, ledger.get("manifest_row") or row, handoff)
        instruction += "\n\nContext from Wayfinder's own search (computed, deterministic):\n\n" + sheet
    except (ValueError, KeyError, TypeError):
        pass  # a malformed ledger never blocks the submission; the finish instruction stands
    if prompt:
        instruction += "\n\n" + prompt

    aristotlelib.set_api_key(key)
    server_toolchain = None
    try:
        from aristotlelib import local_file_utils

        server_toolchain = await local_file_utils.get_expected_toolchain()
    except Exception:  # noqa: BLE001 — informational only
        pass
    base = Path(tempfile.mkdtemp(prefix="aristotle_"))
    try:
        proj_dir = _write_aristotle_project(base / name, handoff["file"])
        try:
            project = await aristotlelib.Project.create_from_directory(
                instruction, proj_dir, **_create_kwargs(aristotlelib, name)
            )
        except aristotlelib.AristotleAPIError as e:
            return f"error: Aristotle refused the submission ({type(e).__name__}): {e}"
    finally:
        shutil.rmtree(base, ignore_errors=True)  # the archive is built; the result lives server-side

    # Everything the gate needs to judge the result, recorded at submit time — the exact
    # submitted text, its elaborated statement, the holes — so nothing is reconstructed later.
    idx = _load_index()
    idx[project.project_id] = {
        "theorem_id": theorem_id,
        "project_name": name,
        "home": str(_accept_home(row) or ""),
        "row": row,
        "target": handoff.get("target"),
        "submitted_text": handoff["file"],
        "submitted_type": handoff.get("pp_type"),
        "statement_sha256": handoff.get("statement_sha256"),
        "handoff_status": status,
        "holes": handoff.get("holes"),
        "closures_kept": handoff.get("closures_kept"),
        "faithful": handoff.get("faithful"),
        "sdk": getattr(aristotlelib, "__version__", "") or _sdk_version(),
    }
    _save_index(idx)

    return json.dumps(
        {
            "theorem_id": theorem_id,
            "project_name": name,
            "project_id": project.project_id,
            "status": _status_name(project.status),
            "sent": f"{status}: {len(handoff.get('holes') or [])} hole(s), "
            f"{handoff.get('closures_kept', 0)} verified Wayfinder closure(s) kept",
            "saves_to": str(_accept_home(row) or "nowhere (no proof_home declared) — accepted proofs stay quarantined"),
            "server_lean_toolchain": server_toolchain,
            "submitted_lean_toolchain": f"leanprover/lean4:{LEAN_VERSION}",
            "note": "Fired and backgrounded. Collect with aristotle_result(project_id); the result is accepted "
            "only if it compiles at v4.28, keeps the statement and every submitted declaration, and uses "
            "only allowlisted axioms.",
        },
        indent=2,
    )


def _status_name(status) -> str:
    """A project status as its name (`RUNNING`, `IDLE`) — 3.x enums print as their number."""
    return str(getattr(status, "name", status))


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("aristotlelib")
    except Exception:  # noqa: BLE001 — informational only
        return ""


@mcp.tool()
async def aristotle_result(project_id: str, destination: str = "", require_finish: bool = False) -> str:
    """Collect a backgrounded Aristotle project by id and judge it. Non-blocking: one status check.

    A returned file is a candidate, never a result. It is accepted only if it passes the gate
    (src/proof_gate.py): nothing that executes code or bypasses the kernel, every submitted
    declaration unchanged, the statement unchanged (text and elaborated type), compiles at v4.28
    with no `sorry`, and `#print axioms` names only the standard three (plus the row's
    `allowed_axioms`). Only an accepted proof is saved to its proofs/ project; everything else
    stays in quarantine under the download directory with the gate's reasons.

    Args:
        project_id: id returned by `send_to_aristotle`.
        destination: optional download directory (absolute); defaults to runs/aristotle/<project_id>/.
        require_finish: reject a correct proof that rebuilt Wayfinder's prefix instead of finishing it
            (default: accept it, and report it as "reproved").
    """
    import aristotlelib  # lazy: the MCP host's SDK

    key = _aristotle_key()
    if not key:
        return f"error: no Aristotle API key. Set $ARISTOTLE_API_KEY or add it to {ARISTOTLE_ENV}."

    aristotlelib.set_api_key(key)
    project = await aristotlelib.Project.from_id(project_id)
    await project.refresh()
    run_status = getattr(getattr(project, "latest_run_status", None), "name", None)
    if project.status == aristotlelib.ProjectStatus.RUNNING:
        return json.dumps(
            {"project_id": project_id, "status": "RUNNING", "run": run_status,
             "note": "still proving on Aristotle — call again later."},  # fmt: skip
            indent=2,
        )
    summary = None
    get_tasks = getattr(project, "get_tasks", None)
    if get_tasks is not None:
        try:
            tasks, _ = await get_tasks(limit=1)
            summary = tasks[0].output_summary if tasks else None
        except Exception:  # noqa: BLE001 — informational only
            summary = None
    if _service_refusal(summary):
        rec = _load_index().get(project_id, {})
        _save_service_status(
            {
                "lean_only": "refused",
                "seen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "project_id": project_id,
                "message": " ".join(str(summary).split())[:400],
            }
        )
        return json.dumps(
            {"project_id": project_id, "theorem_id": rec.get("theorem_id"), "status": _status_name(project.status),
             "run": run_status, "accepted": False, "outcome": "service_refused",
             "aristotle_summary": summary,
             "next_step": "Aristotle refused the project itself — no proof was attempted, so there is nothing to "
             "judge. The residual stays in the goal ledger; send_to_aristotle now declines Lean-only hand-offs "
             "(force=True to retry)."},  # fmt: skip
            indent=2,
        )
    if not getattr(project, "has_files", True):
        # get_files would hand back our own submission here; never judge the input as a result.
        return json.dumps(
            {"project_id": project_id, "status": _status_name(project.status), "run": run_status,
             "note": "Aristotle produced no result files for this project."},  # fmt: skip
            indent=2,
        )

    dest = Path(destination).expanduser().resolve() if destination else WF / "runs" / "aristotle" / project_id
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "aristotle_result.tar.gz"
    await project.get_files(tar_path)  # get_files writes the result tar.gz to this FILE path
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")  # no absolute paths, no escaping links
    leans = sorted(dest.rglob("Residual.lean"))
    rec = _load_index().get(project_id, {})
    if not leans:
        return json.dumps(
            {"project_id": project_id, "status": _status_name(project.status), "run": run_status, "accepted": False,
             "reasons": ["the returned archive has no Residual.lean"], "downloaded_to": str(dest)},  # fmt: skip
            indent=2,
        )
    returned = leans[0].read_text()

    theorem_id = rec.get("theorem_id")
    row = rec.get("row") or ({"theorem_id": theorem_id} if theorem_id else {})
    reconstructed = "submitted_text" not in rec
    if row.get("full_declaration"):
        submitted = rec.get("submitted_text") or _proof_lean(row)
        target = rec.get("target") or _lean_name(row)
    else:
        submitted, target = "", str(theorem_id or "")
    submitted_type = rec.get("submitted_type")
    if submitted_type is None and submitted:
        probe = _compile_v428(submitted + gate.probe_suffix(target))
        submitted_type = gate.checked_type(probe["messages"], target)

    compile_run = None
    if submitted and not gate.static_findings(returned, submitted):
        compile_run = _compile_v428(returned + gate.probe_suffix(target))
    res = gate.judge(
        returned=returned,
        submitted=submitted,
        target=target,
        compile_run=compile_run,
        submitted_type=submitted_type,
        allowed_axioms=tuple(row.get("allowed_axioms") or ()),
        require_finish=require_finish,
        environment=VERIFY_ENV,
    )
    if not submitted:
        res.accepted = False
        res.reasons.insert(0, "no submission on record for this project id")

    # Aristotle's own per-declaration verdicts (SDK 3.x) — reported beside ours, never instead of it.
    verdicts = None
    get_declarations = getattr(project, "get_declarations", None)
    if get_declarations is not None:
        try:
            verdicts = [
                {"name": dcl.name, "status": dcl.status.name,
                 "failure": dcl.failure_reason.name if dcl.failure_reason else None}  # fmt: skip
                for dcl in await get_declarations()
                if dcl.name == target or dcl.name.endswith(f".{target}")
            ]
        except Exception as e:  # noqa: BLE001 — informational only
            verdicts = [{"error": f"{type(e).__name__}: {e}"[:200]}]

    saved_to = None
    home = _accept_home(row) if row else None
    if res.accepted and home is not None:
        provenance = (
            f"Wayfinder ({rec.get('closures_kept', 0)} verified closures, {rec.get('handoff_status', '?')}) + "
            f"Aristotle project {project_id}"
        )
        saved_to = str(_save_solved_proof(row, _acceptance_header(res, provenance) + returned))
    (dest / "gate.json").write_text(
        json.dumps({**res.to_dict(), "aristotle_verdicts": verdicts, "aristotle_summary": summary}, indent=2)
    )
    if res.accepted and _load_service_status().get("lean_only") == "refused":
        _save_service_status({"lean_only": "accepted", "seen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "project_id": project_id})  # fmt: skip
    idx = _load_index()
    if project_id in idx:
        idx[project_id]["gate"] = {**res.to_dict(), "saved_to": saved_to}
        _save_index(idx)

    if res.accepted and saved_to:
        nxt = f"ACCEPTED → saved to {saved_to}"
        if res.mode == "reproved":
            nxt += " — NOTE: Aristotle rebuilt Wayfinder's prefix instead of finishing it (see prefix_missing)."
    elif res.accepted:
        nxt = "ACCEPTED, but the row declares no proof_home: kept in quarantine at the download directory."
    else:
        nxt = "REJECTED — not saved. The candidate and gate.json stay in quarantine at the download directory."
    out = {
        "project_id": project_id,
        "theorem_id": theorem_id,
        "status": _status_name(project.status),
        "run": run_status,
        "accepted": res.accepted,
        "reasons": res.reasons,
        "checks": res.checks,
        "axioms": res.axioms,
        "mode": res.mode,
        "aristotle_verdicts": verdicts,
        "aristotle_summary": summary,
        "saved_to": saved_to,
        "downloaded_to": str(dest),
        "next_step": nxt,
    }
    if res.prefix_missing and res.mode == "reproved":
        out["prefix_missing"] = res.prefix_missing[:10]
    if reconstructed:
        out["warning"] = "submitted before the gate recorded submissions: compared against a reconstruction"
    return json.dumps(out, indent=2)


@mcp.tool()
def prove_corpus(
    theorems: str,
    limit: int | None = None,
    checkpoint: str = str(CHECKPOINT),
    device: str = "mps",
    resume: bool = False,
    collect_trace: bool = True,
    per_theorem_timeout: int = 300,
    output: str = "",
) -> str:
    """Launch a LONG, large-scale proof-search run over a whole corpus, in the BACKGROUND.

    The COMPLETE Wayfinder instantiation — NAV-004 navigator + all lanes (structural / cosine
    retrieval / hammer / Dr. Ducky) — run over every theorem in `theorems` (a JSONL corpus).
    Fire-and-return: spawns a DETACHED subprocess and returns immediately with the output paths + pid;
    it keeps running after this call. Crash-safe and resumable — each theorem's result is appended to
    the .jsonl AS it completes, a per-theorem SIGALRM watchdog skips a hung Lean kernel instead of
    freezing the whole run, and `resume=True` continues an interrupted run from where it stopped.
    Poll progress / collect the result with `prove_corpus_status(output)`.

    Args:
        theorems: JSONL corpus of theorem rows (theorem_id + goal_state/full_declaration). The
            search kernel imports the union of the rows' `imports`; a calling repo's definitions
            must be a module of the search project (`lake build` it there) named on its rows.
        limit: cap the number of theorems (None = all in the corpus).
        checkpoint: navigator checkpoint (default NAV-004 — the full-pipeline one).
        device: torch device ("mps", "cpu", "cuda").
        resume: skip theorem_ids already recorded in the output .jsonl (continue a parked run).
        collect_trace: record per-step apply_candidates + goal-stack (the SSL candidate-narrowing observable).
        per_theorem_timeout: SIGALRM watchdog seconds per theorem (hung kernel → skip, not freeze).
        output: output .json path (default runs/corpus/<corpus-stem>.json).
    """
    man = Path(theorems).expanduser().resolve()
    if not man.exists():
        return f"error: theorems corpus not found: {man}"
    ckpt = Path(checkpoint).expanduser().resolve()
    if not ckpt.exists():
        return f"error: checkpoint not found: {ckpt}"
    manifest_rows = [json.loads(ln) for ln in man.read_text().splitlines() if ln.strip()]
    n_total = len(manifest_rows)
    out = Path(output).expanduser().resolve() if output else (WF / "runs" / "corpus" / f"{man.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    # Persistent resolved config (a multi-hour run must keep it) with collect_trace enabled.
    cfg = _resolved_config(collect_trace=collect_trace, dest=out.with_suffix(".config.yaml"))

    cmd = [
        str(VENV_PY),
        "-u",
        str(BENCH),
        "--config",
        str(cfg),
        "--checkpoint",
        str(ckpt),
        "--backend",
        "pantograph",
        "--lean-project",
        str(LEAN_PROJECT),
        "--lean-imports",
        *_lean_imports(manifest_rows),
        "--theorems",
        str(man),
        "--search-mode",
        "full",
        "--cosine-rw",
        "--cosine-rw-seq",
        "--device",
        device,
        "--per-theorem-timeout",
        str(per_theorem_timeout),
        "--output",
        str(out),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd += ["--resume"]

    env = {**os.environ, "PYTHONPATH": str(WF), "PYTHONUNBUFFERED": "1"}
    logf = open(log, "a" if resume else "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WF),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so the run survives this MCP call returning
    )
    return json.dumps(
        {
            "status": "launched",
            "pid": proc.pid,
            "theorems": str(man),
            "n_theorems": n_total if limit is None else min(limit, n_total),
            "checkpoint": ckpt.name,
            "output_json": str(out),
            "output_jsonl": str(out.with_suffix(".jsonl")),
            "log": str(log),
            "note": "Background corpus run started (detached, resumable, watchdog-protected; cosine + "
            "collect_trace ON). Poll with prove_corpus_status(output_json). It runs after this "
            "call returns; resume an interrupted run with prove_corpus(..., resume=True).",
        },
        indent=2,
    )


@mcp.tool()
def prove_corpus_status(output: str) -> str:
    """Non-blocking progress / collection for a `prove_corpus` run — reads the incremental .jsonl.

    Returns theorems-done, solve rate, close-lane breakdown, and whether the final .json report is
    written (complete). Safe to poll repeatedly. If a run was interrupted, resume it with
    `prove_corpus(theorems=..., resume=True, output=<same output>)`.

    Args:
        output: the output_json path returned by prove_corpus (its .jsonl carries live progress).
    """
    out = Path(output).expanduser().resolve()
    jsonl = out.with_suffix(".jsonl")
    if not jsonl.exists():
        return f"no run found at {jsonl} — not started yet, or wrong output path."
    rows = []
    for ln in jsonl.read_text().splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    done = len(rows)
    solved = sum(1 for r in rows if r.get("honest_success") or r.get("success"))
    from collections import Counter

    lanes = Counter(r.get("close_lane") for r in rows)
    with_trace = sum(1 for r in rows if r.get("step_trace"))
    return json.dumps(
        {
            "output_json": str(out),
            "complete": out.exists(),  # the .json report is written only at completion
            "theorems_done": done,
            "solved": solved,
            "solve_rate": round(solved / max(1, done), 3),
            "with_step_trace": with_trace,
            "last_theorem": rows[-1].get("theorem_id") if rows else None,
            "close_lanes": dict(lanes),
            "note": "complete=True once the final .json report lands. Re-call to poll. "
            "Resume an interrupted run: prove_corpus(theorems=..., resume=True, output=this path).",
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
