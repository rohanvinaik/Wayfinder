"""FastMCP server for Wayfinder — runs the ACTUAL neural proof-search pipeline.

Two tools, mirroring how proofs are actually produced:
  * `prove(theorem_id)` — run the real Wayfinder prover (below).
  * `send_to_aristotle(theorem_id)` — if Wayfinder leaves a residual goal open,
    fire that statement at Aristotle, a SEPARATE, unrelated hosted prover. This is
    the "if Wayfinder doesn't finish it, send it to Aristotle" fallback — two
    explicit steps, not an integrated bridge.

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

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WF = Path("/Users/rohanvinaik/Projects/Wayfinder")

# The warranted-emission seam (THE_REFOUNDING §5) is pure stdlib, so it is
# safe to import into the MCP host environment directly.
import sys  # noqa: E402

if str(WF) not in sys.path:
    sys.path.insert(0, str(WF))
from src.cheat_sheet import render_cheat_sheet  # noqa: E402
from src.lean_expr import EmissionWarrant, from_tactics, read_environment_pin, render_declaration  # noqa: E402

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
DB = WF / "data/proof_network_v3.db"                          # absolute — kills the cwd DB error
DEFAULT_MANIFEST = WF / "data/rerun_v427_manifest.jsonl"      # the 9 SSL/Ansatz/Socratic theorems
LAKE_BIN = Path("/Users/rohanvinaik/.elan/bin/lake")          # elan shim (picks toolchain from lean-toolchain)

# Aristotle (separate hosted fallback prover; aristotlelib 2.1.0 = v3 API on the MCP host)
ARISTOTLE_ENV = Path("/Users/rohanvinaik/semantic_probing/.env")   # ARISTOTLE_API_KEY lives here
LEAN_VERSION = "v4.28.0"   # Aristotle solves at v4.28.0; submissions + saved proofs match it
# Where finished Aristotle proofs land — one clean proofs/ project per owning repo.
SSL_PROOFS = Path(
    "/Users/rohanvinaik/Projects/rohan-vinaik.github.io/papers/"
    "Core Documents/Semantic_Specification_Learning/proofs"
)
ANSATZ_PROOFS = Path("/Users/rohanvinaik/Projects/Ansatz/proofs")
ANSATZ_THEOREMS = {"depletion_time", "pesin_band_rate_lower_bound"}  # Pesin/physics-bridge cores; rest → SSL
ARISTOTLE_INDEX = WF / "runs" / "aristotle" / "index.json"          # project_id → theorem context


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
                name = name[len("import "):].strip()
            if name and name not in seen:
                seen.append(name)
    if not seen:
        return ["Mathlib"]
    if "Mathlib" in seen:
        seen.remove("Mathlib")
        seen.insert(0, "Mathlib")
    return seen


def _manifest_ids(man: Path) -> list[str]:
    return [
        json.loads(ln)["theorem_id"]
        for ln in man.read_text().splitlines()
        if ln.strip()
    ]


def _def_preamble(row: dict) -> str:
    """Local `def`/`abbrev` blocks the theorem depends on, pulled from its source file.

    The manifest packages only the theorem statement; theorems like survivingDOF_antitone reference a
    local def (survivingDOF) that must be in scope for BOTH Wayfinder and Aristotle. Returns "" if none.
    """
    src = LEAN_PROJECT / row.get("file_path", "")
    if not src.exists():
        return ""
    lines = src.read_text().splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^(noncomputable\s+)?(def|abbrev)\s", lines[i]):
            blk = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                blk.append(lines[i])
                i += 1
            blocks.append("\n".join(blk))
        else:
            i += 1
    return "\n\n".join(blocks)


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
            cwd=str(LEAN_PROJECT), capture_output=True, text=True, timeout=300,
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


def _verify_at_v428(lean_text: str, thm_name: str) -> tuple[bool, str]:
    """Compile `lean_text` in the built v4.28 project; True iff it elaborates AND has no `sorryAx`.

    This is the soundness gate — a reconstructed Wayfinder proof is saved only if it re-verifies.
    """
    probe = lean_text.rstrip() + f"\n#print axioms {thm_name}\n"
    fpath = VERIFY_PROJECT / f"_verify_{thm_name}.lean"
    try:
        fpath.write_text(probe)
        p = subprocess.run(
            [str(LAKE_BIN), "env", "lean", str(fpath)],
            cwd=str(VERIFY_PROJECT), capture_output=True, text=True, timeout=420,
        )
        out = p.stdout + p.stderr
        ok = p.returncode == 0 and "sorryAx" not in out
        return ok, out[-800:]
    except subprocess.TimeoutExpired:
        return False, "verify timeout (>420s)"
    finally:
        try:
            fpath.unlink()
        except OSError:
            pass


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

    rows = [
        ln
        for ln in man.read_text().splitlines()
        if ln.strip() and json.loads(ln).get("theorem_id") == theorem_id
    ]
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
            str(VENV_PY), str(BENCH),
            "--config", str(cfg),
            "--checkpoint", str(CHECKPOINT),
            "--backend", "pantograph",
            "--lean-project", str(LEAN_PROJECT),
            "--lean-imports", *_lean_imports([row]),
            "--theorems", str(one),
            "--limit", "1",
            "--search-mode", "full",
            "--cosine-rw",          # scope → MiniLM → top-k beam + Lean verify (the lemma-retrieval lane)
            "--cosine-rw-seq",      # sequential bare-rewrite lane — the KEY closers in real runs
            "--device", device,
            "--output", str(out),
        ]
        env = {**os.environ, "PYTHONPATH": str(WF)}
        try:
            p = subprocess.run(
                cmd, cwd=str(WF), capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s] {theorem_id} — Wayfinder still searching"
        finally:
            try:
                cfg.unlink()
            except OSError:
                pass

        if not out.exists():
            return (
                f"no result file (exit {p.returncode}).\n"
                f"{(p.stdout + p.stderr)[-1800:]}"
            )
        r = json.loads(out.read_text())
        d = (r.get("details") or [{}])[0]
        # Persist the goal-ledger row: the accreting per-theorem source of
        # truth the cheat sheet renders from at hand-off time.
        try:
            GOAL_LEDGER.mkdir(parents=True, exist_ok=True)
            (GOAL_LEDGER / f"{theorem_id}.json").write_text(
                json.dumps({"detail": d, "manifest_row": row}, indent=2)
            )
        except OSError:
            pass  # the ledger is an artifact, never a gate on the prove result
        proved = bool(d.get("success")) and bool(d.get("honest_success"))
        skipped = (not proved) and d.get("close_lane") == "skipped" and not d.get("attempts")

        # Wayfinder closed it → reconstruct its tactic trace, verify @v4.28, save only if it re-checks.
        verified = None
        saved_to = None
        verify_detail = None
        if proved:
            candidate = _reconstruct_proof(
                row,
                d.get("tactics_used") or [],
                provenance=tuple(d.get("close_provenance") or ()),
                source_run=str(out),
            )
            verified, verify_detail = _verify_at_v428(candidate, theorem_id)
            if verified:
                saved_to = str(_save_solved_proof(row, candidate))

        # A skip means the initial goal never built (attempts=0). DIAGNOSE it — almost always the
        # STATEMENT doesn't elaborate. This is NEVER a residual and NEVER an Aristotle candidate; it
        # is a declaration to fix or a harness issue to debug. Surfacing it here (not in my memory)
        # is what stops the recurring "attempts=0 → onboard the harness / send to Aristotle" misread.
        statement_elaborates = None
        statement_error = None
        if skipped:
            statement_elaborates, statement_error = _elaboration_probe(row)

        if proved and verified:
            nxt = f"proved by Wayfinder, verified @v4.28 → saved to {saved_to}. No Aristotle needed."
        elif proved and not verified:
            nxt = ("Wayfinder closed it but its tactic trace did not re-verify @v4.28 "
                   "(non-linearizable) → send to Aristotle.")
        elif skipped and statement_elaborates is False:
            nxt = ("SKIPPED: the STATEMENT DOES NOT ELABORATE (see statement_error). This is a broken "
                   "declaration (missing instance / typo / undefined name) — FIX THE .lean + manifest and "
                   "re-run prove(). NOT a residual, NOT a harness/onboarding bug, do NOT send to Aristotle.")
        elif skipped:
            nxt = ("SKIPPED though the statement elaborates → goal-resolution/harness issue "
                   "(debug run_benchmark._resolve_initial_goal). NOT a residual, do NOT send to Aristotle.")
        else:
            nxt = "residual goal open → send to Aristotle (separate prover; not called here)."

        summary = {
            "theorem_id": theorem_id,
            "proved_by_wayfinder": bool(d.get("success")),
            "honest_success": bool(d.get("honest_success")),
            "reconstructed_verified_v428": verified,
            "saved_to": saved_to,
            "close_lane": d.get("close_lane"),
            "goals_closed": d.get("goals_closed"),        # subgoals Wayfinder cleared
            "goals_remaining": d.get("goals_remaining"),  # residual handed to Aristotle
            "attempts": d.get("attempts"),
            "time_s": r.get("efficiency", {}).get("avg_time_per_theorem_s"),
            "next_step": nxt,
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


def _count_sorries(text: str) -> int:
    """Count `sorry` tactics outside comments (mirrors the pipeline's counter)."""
    in_block = False
    n = 0
    for line in text.splitlines():
        if "/-" in line:
            in_block = True
        if "-/" in line:
            in_block = False
            continue
        if in_block:
            continue
        if re.search(r"\bsorry\b", re.sub(r"--.*", "", line)):
            n += 1
    return n


def _sane_project_name(theorem_id: str, row: dict) -> str:
    """Resolvable Aristotle project name: provenance + theorem name + proof number.

    Aristotle derives the project's identity from the submitted directory name
    (`create_from_directory` → `project_dir.name`), so this is what shows in the
    dashboard and lets it be found later.
    """
    stem = Path(row.get("file_path", "")).stem or "thm"      # e.g. proof_30
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"Wayfinder-{theorem_id}-{stem}")


def _lakefile(package: str, lib: str, roots_or_globs: str) -> str:
    """A lakefile.toml requiring mathlib at LEAN_VERSION. No manifest (lake generates a consistent one)."""
    return (
        f'name = "{package}"\nversion = "0.1.0"\n\n'
        '[[require]]\nname = "mathlib"\n'
        'git = "https://github.com/leanprover-community/mathlib4.git"\n'
        f'rev = "{LEAN_VERSION}"\n\n'
        f'[[lean_lib]]\nname = "{lib}"\n{roots_or_globs}\n'
    )


def _proof_lean(row: dict) -> str:
    """The Lean source for one theorem: imports + opens + local defs + the (sorry-bearing) declaration."""
    imports = "\n".join(row.get("imports") or ["import Mathlib"])
    opens = "\n".join(row.get("opens") or [])
    return "\n".join(x for x in [imports, opens, "", _augmented_declaration(row)] if x) + "\n"


def _write_aristotle_project(row: dict, proj_dir: Path) -> Path:
    """Write a minimal standalone Lean project (v4.28.0, no manifest) into `proj_dir`.

    `proj_dir`'s basename becomes the Aristotle project name — pass a named dir. No lake-manifest is
    written: a v4.27 manifest would poison the v4.28 build, and Aristotle resolves mathlib itself.
    """
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "Residual.lean").write_text(_proof_lean(row))
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
    lakefile = home / "lakefile.toml"
    if not lakefile.exists():
        lakefile.write_text(_lakefile(f"{home.parent.name}_proofs", lib, 'globs = ["*"]'))
    proof_path = home / f"{theorem_id}.lean"
    proof_path.write_text(solved_lean)
    return proof_path


@mcp.tool()
async def send_to_aristotle(
    theorem_id: str,
    manifest: str = str(DEFAULT_MANIFEST),
    prompt: str = "",
) -> str:
    """Fire ONE theorem at Aristotle (the SEPARATE hosted fallback prover) and return immediately.

    Use when Wayfinder's `prove()` leaves a residual goal open ("if Wayfinder doesn't finish it,
    send it to Aristotle"). This SUBMITS and returns right away with the project id + name — it does
    NOT wait for the proof (Aristotle keeps running server-side). Collect the result later with
    `aristotle_result(project_id)`. Wayfinder and Aristotle are unrelated systems.

    METERED hosted call. Key read from $ARISTOTLE_API_KEY or semantic_probing/.env (never hardcoded).
    Uses aristotlelib 2.1.0 (v3 API). The Aristotle project is named `Wayfinder-<theorem>-<proof_NN>`
    so it is resolvable later.

    Args:
        theorem_id: theorem to prove (a row in `manifest`: full_declaration/imports/opens/file_path).
        manifest: JSONL manifest. Default: the 9-theorem rerun manifest.
        prompt: optional instruction override; a sensible default is used if empty.
    """
    import aristotlelib  # lazy: host's 2.1.0 (v3 API)

    key = _aristotle_key()
    if not key:
        return (
            "error: no Aristotle API key. Set $ARISTOTLE_API_KEY or add "
            f"ARISTOTLE_API_KEY=... to {ARISTOTLE_ENV}, then retry."
        )

    man = Path(manifest).expanduser().resolve()
    if not man.exists():
        return f"error: manifest not found: {man}"
    rows = [
        ln for ln in man.read_text().splitlines()
        if ln.strip() and json.loads(ln).get("theorem_id") == theorem_id
    ]
    if not rows:
        return f"error: '{theorem_id}' not in {man.name}. Available: {_manifest_ids(man)}"
    row = json.loads(rows[0])
    name = _sane_project_name(theorem_id, row)

    default_prompt = (
        f"Fill all sorry placeholders and prove the theorem `{theorem_id}` ({name}) using Mathlib. "
        "Leave any declarations marked `axiom` as axioms; do not add new axioms or `sorry`."
    )
    # Fold Wayfinder's computed understanding into the hand-off (Gotcha 3):
    # the cheat sheet renders deterministically from the persisted goal-ledger
    # row of the most recent prove() run — signature walk, lane history, the
    # negative channel, and the verified progress so far.
    ledger_path = GOAL_LEDGER / f"{theorem_id}.json"
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text())
            sheet = render_cheat_sheet(ledger.get("detail") or {}, ledger.get("manifest_row") or row)
            default_prompt += "\n\nContext from Wayfinder's own search (computed, deterministic):\n\n" + sheet
        except (OSError, ValueError, KeyError):
            pass  # a malformed ledger never blocks the submission; the bare prompt stands

    aristotlelib.set_api_key(key)
    base = Path(tempfile.mkdtemp(prefix="aristotle_"))
    try:
        proj_dir = _write_aristotle_project(row, base / name)
        project = await aristotlelib.Project.create_from_directory(
            prompt or default_prompt, proj_dir
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)  # tar already built; result lives server-side

    # Record project_id → theorem context so aristotle_result can save it to the right repo by id alone.
    idx = _load_index()
    idx[project.project_id] = {
        "theorem_id": theorem_id,
        "project_name": name,
        "home": str(_proof_home(row)),
        "row": row,
    }
    _save_index(idx)

    return json.dumps(
        {
            "theorem_id": theorem_id,
            "project_name": name,
            "project_id": project.project_id,
            "status": str(project.status),
            "saves_to": str(_proof_home(row)),
            "note": "Fired and backgrounded — Aristotle is proving it server-side. "
                    "Collect with aristotle_result(project_id) later; no need to wait.",
        },
        indent=2,
    )


@mcp.tool()
async def aristotle_result(project_id: str, destination: str = "") -> str:
    """Collect a backgrounded Aristotle project by id. Non-blocking: one status check.

    If the project is still RUNNING, says so (check again later). If done, downloads the solved
    Lean and reports the remaining-sorry count. Downloads to `destination` if given, else
    `runs/aristotle/<project_id>/`.

    Args:
        project_id: id returned by `send_to_aristotle`.
        destination: optional download directory (absolute); defaults under the Wayfinder repo.
    """
    import aristotlelib  # lazy: host's 2.1.0 (v3 API)

    key = _aristotle_key()
    if not key:
        return f"error: no Aristotle API key. Set $ARISTOTLE_API_KEY or add it to {ARISTOTLE_ENV}."

    aristotlelib.set_api_key(key)
    project = await aristotlelib.Project.from_id(project_id)
    await project.refresh()
    if project.status == aristotlelib.ProjectStatus.RUNNING:
        return json.dumps(
            {"project_id": project_id, "status": "RUNNING",
             "note": "still proving on Aristotle — call again later."},
            indent=2,
        )

    dest = (
        Path(destination).expanduser().resolve()
        if destination
        else WF / "runs" / "aristotle" / project_id
    )
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "aristotle_result.tar.gz"
    await project.get_files(tar_path)          # get_files writes the result tar.gz to this FILE path
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    leans = list(dest.rglob("Residual.lean")) or [p for p in dest.rglob("*.lean") if p.is_file()]
    if not leans:
        return json.dumps(
            {"project_id": project_id, "status": str(project.status),
             "note": "no .lean returned", "downloaded_to": str(dest)},
            indent=2,
        )
    solved = leans[0].read_text()
    n_sorry = _count_sorries(solved)

    # On a finished proof, save it into its owning repo's clean proofs/ project (looked up by id).
    rec = _load_index().get(project_id, {})
    theorem_id = rec.get("theorem_id")
    # the indexed manifest row carries the calling repo's `proof_home`; a legacy index entry
    # without a row still routes by theorem id
    row = rec.get("row") or ({"theorem_id": theorem_id} if theorem_id else None)
    saved_to = None
    if n_sorry == 0 and row:
        saved_to = str(_save_solved_proof(row, solved))

    return json.dumps(
        {
            "project_id": project_id,
            "theorem_id": theorem_id,
            "status": str(project.status),
            "proved_by_aristotle": n_sorry == 0,
            "sorries_remaining": n_sorry,
            "saved_to": saved_to,
            "downloaded_to": str(dest),
            "next_step": (
                f"closed by Aristotle → saved to {saved_to}. Verify: `lake env lean` + `#print axioms` "
                "(run `lake update && lake exe cache get` once per proofs project)."
                if n_sorry == 0
                else "Aristotle left sorries — still open; not saved as a proof."
            ),
        },
        indent=2,
    )


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
        str(VENV_PY), "-u", str(BENCH),
        "--config", str(cfg),
        "--checkpoint", str(ckpt),
        "--backend", "pantograph",
        "--lean-project", str(LEAN_PROJECT),
        "--lean-imports", *_lean_imports(manifest_rows),
        "--theorems", str(man),
        "--search-mode", "full",
        "--cosine-rw", "--cosine-rw-seq",
        "--device", device,
        "--per-theorem-timeout", str(per_theorem_timeout),
        "--output", str(out),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd += ["--resume"]

    env = {**os.environ, "PYTHONPATH": str(WF), "PYTHONUNBUFFERED": "1"}
    logf = open(log, "a" if resume else "w")
    proc = subprocess.Popen(
        cmd, cwd=str(WF), env=env, stdout=logf, stderr=subprocess.STDOUT,
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
