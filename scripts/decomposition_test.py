"""EXP-DEC-001 — the decomposition test: the same local prover on the whole statement and on
Wayfinder's residual, paired, on MiniF2F-test.

Pre-registered in docs/Research_Paper/EXPERIMENT_RESULTS.md (EXP-DEC-001) before any model sample
was drawn. Four phases, each resumable, run one at a time (the model phase sizes its servers to
the memory free when each arm starts, so nothing else heavy may run beside it):

    search    the deterministic search over the corpus — one run_benchmark process
    handoffs  the residual hand-off (and the local finishing pass) for every row the search left open
    pairs     for the frozen sample of residual theorems: arm A (the whole statement) and arm B
              (Wayfinder's holes), the same model, sampling and budget, each arm a local-prover
              worker subprocess; a theorem counts as closed only when the completed file passes the
              acceptance gate
    report    the paired table

Budgets are matched on total attempts: B gets k attempts per hole, A gets k x (B's hole count).
Infrastructure outcomes (a run that did not finish, server errors, memory yields, a full context,
a lifted statement that does not re-elaborate, a verified proof that would not go back in place)
are never scored as model losses: the arm is rerun once, then the pair is excluded and counted.

Run on the MCP host python (it imports wayfinder_mcp), from the repo root:
    python scripts/decomposition_test.py {search,handoffs,pairs,report} [--dir runs/exp_dec_001]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WF))

import src.hole_fill as hf  # noqa: E402
import src.local_prover as lp  # noqa: E402
import src.proof_gate as gate  # noqa: E402
import src.residual_handoff as rh  # noqa: E402
import wayfinder_mcp as m  # noqa: E402

CORPUS = WF / "data" / "minif2f_test.jsonl"
SEED = 20261029  # the sample of residual theorems, drawn once and frozen
N = 40
K = 8  # attempts per hole (arm B) and per hole of B's residual (arm A: K x holes)
ARM_TIMEOUT_S = 4 * 3600
INFRA = {"server_error", "memory_yield", "stopped", "context_full", "in_place_failed"}
# Amendment 1 (EXPERIMENT_RESULTS.md): every attempt gets exactly CAP new tokens; a slot holds
# CTX, room for the prompt beside them, so eight slots fit beside the 8B and a one-hole arm's
# eight attempts run in one wave. The recheck reruns a discordant pair's losing arm at RECHECK_CAP.
CAP, CTX, SLOTS = 8192, 16384, 8
RECHECK_CAP, RECHECK_CTX, RECHECK_SLOTS = 32768, 40960, 4


def _rows() -> dict[str, dict]:
    return {r["theorem_id"]: r for r in map(json.loads, CORPUS.read_text().splitlines()) if r}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ── phase 1: the deterministic search ─────────────────────────────────────────


CHUNK = 20  # theorems per search process: a fresh Lean server for each chunk
RETRIES = 2  # fresh-process retries for a theorem that did not start


def search(exp: Path) -> None:
    """The deterministic search, in chunks of CHUNK theorems, each chunk its own process.

    A Pantograph server that dies mid-run is not always brought back, and every theorem after it
    is then skipped at once (the first full-corpus attempt skipped 57 of its last 61 that way). So a
    theorem that did not start is retried at the front of a later chunk, on a fresh server, up to
    RETRIES times: a statement that genuinely cannot start skips again; one the dead kernel skipped
    starts.
    """
    rows = _rows()
    master = exp / "search.jsonl"
    retries_path = exp / "search_retries.json"
    retries: dict[str, int] = json.loads(retries_path.read_text()) if retries_path.exists() else {}
    chunk_dir = exp / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    cfg = m._resolved_config(dest=exp / "config.yaml")
    env = {**os.environ, "PYTHONPATH": str(WF), "PYTHONUNBUFFERED": "1"}
    n = len(list(chunk_dir.glob("chunk_*.jsonl")))
    while True:
        done = {r["theorem_id"]: r for r in _jsonl(master)}
        redo = [t for t, r in done.items() if r.get("disposition") == "skipped_start" and retries.get(t, 0) < RETRIES]
        fresh = [t for t in rows if t not in done]
        batch = (redo + fresh)[:CHUNK]
        if not batch:
            break
        for t in batch:
            if t in done:
                retries[t] = retries.get(t, 0) + 1
        retries_path.write_text(json.dumps(retries, indent=2))
        n += 1
        corpus = chunk_dir / f"chunk_{n:03d}_corpus.jsonl"
        corpus.write_text(
            "".join(json.dumps({**rows[t], "full_declaration": m._augmented_declaration(rows[t])}) + "\n" for t in batch)
        )
        out = chunk_dir / f"chunk_{n:03d}.json"
        cmd = [
            str(m.VENV_PY), "-u", str(m.BENCH), "--config", str(cfg), "--checkpoint", str(m.CHECKPOINT),
            "--backend", "pantograph", "--lean-project", str(m.LEAN_PROJECT), "--lean-imports",
            *m._lean_imports([rows[t] for t in batch]), "--theorems", str(corpus), "--device", "mps",
            "--per-theorem-timeout", "300", "--output", str(out),
        ]  # fmt: skip
        with open(exp / "search.log", "a") as log:
            subprocess.run(cmd, cwd=str(WF), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        for r in _jsonl(out.with_suffix(".jsonl")):
            done[r["theorem_id"]] = r
        master.write_text("".join(json.dumps(done[t]) + "\n" for t in rows if t in done))
        started = sum(1 for t in batch if done.get(t, {}).get("disposition") != "skipped_start")
        print(f"chunk {n}: {len(batch)} theorems ({len([t for t in batch if retries.get(t)])} retries), "
              f"{started} started; {len(done)}/{len(rows)} recorded", flush=True)  # fmt: skip


# ── phase 2: the hand-offs ────────────────────────────────────────────────────


def _one_handoff(tid: str, row: dict, d: dict, dest: str) -> str:
    """Build one hand-off (in a worker process) and write it; returns its status."""
    if not d.get("started") or not d.get("attempts"):
        rec = {"theorem_id": tid, "status": "skipped_start"}
    elif d.get("success") and d.get("honest_success"):
        rec = {"theorem_id": tid, "status": "search_closed", "close_lane": d.get("close_lane")}
    else:
        try:
            h = m._build_handoff(row, d)
            rec = {"theorem_id": tid, "status": h.get("status"), "handoff": h}
        except Exception as e:  # noqa: BLE001 — recorded, never silently dropped
            rec = {"theorem_id": tid, "status": "handoff_error", "error": f"{type(e).__name__}: {e}"[:400]}
    Path(dest).write_text(json.dumps(rec, indent=2))
    return str(rec["status"])


def handoffs(exp: Path, workers: int) -> None:
    rows = _rows()
    details = {d["theorem_id"]: d for d in _jsonl(exp / "search.jsonl")}
    out = exp / "handoffs"
    out.mkdir(exist_ok=True)
    todo = [t for t in rows if t in details and not (out / f"{t}.json").exists()]
    print(f"{len(todo)} hand-offs to build ({len(details)} searched, {len(rows)} in the corpus)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one_handoff, t, rows[t], details[t], str(out / f"{t}.json")): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(todo)}] {futs[fut]}: {fut.result()}", flush=True)


# ── phase 3: the paired arms ──────────────────────────────────────────────────


def _population(exp: Path) -> list[str]:
    """Residual theorems: the finishing pass left ≥ 1 hole and the skeleton keeps ≥ 1 search step."""
    out = []
    for f in sorted((exp / "handoffs").glob("*.json")):
        rec = json.loads(f.read_text())
        h = rec.get("handoff") or {}
        if rec.get("status") == "residual" and h.get("holes") and (h.get("closures_kept") or 0) >= 1:
            out.append(rec["theorem_id"])
    return out


def _sample(exp: Path) -> list[str]:
    frozen = exp / "sample.json"
    if frozen.exists():
        return json.loads(frozen.read_text())["theorems"]
    pop = _population(exp)
    pick = sorted(random.Random(SEED).sample(pop, min(N, len(pop))))
    frozen.write_text(json.dumps({"seed": SEED, "n": len(pick), "population": len(pop), "theorems": pick}, indent=2))
    return pick


def _job(
    run_id: str, row: dict, handoff: dict, model: str, *, whole: bool, samples: int,
    cap: int = CAP, ctx: int = CTX, slots: int = SLOTS,
) -> dict:
    target = handoff["target"]
    header, _ = rh.split_declaration(str(row["full_declaration"]))
    if whole:
        goal = str(row.get("goal_state") or "")  # the theorem itself, as the search started from it
        declaration = header.rstrip() + f"\n  -- {rh.HOLE_MARK} 1: close this goal\n  sorry\n"
        file_text = m._file_with_declaration(row, declaration)
        holes = [{"index": 1, "lean_goal": goal}]
        status, statement = "bare", re.sub(r":=\s*by\s*$", "", header).rstrip()
    else:
        declaration, file_text, holes = handoff["declaration"], handoff["file"], handoff["holes"]
        status, statement = "residual", ""
    return {
        "run_id": run_id, "theorem_id": row["theorem_id"], "row": row, "target": target, "status": status,
        "file": file_text, "declaration": declaration, "statement": statement, "holes": holes,
        "pp_type": handoff.get("pp_type"), "allowed_axioms": list(row.get("allowed_axioms") or []),
        "defs": m._file_defs(row), "as_built_sha256": m._sha256(file_text), "model": model, "samples": samples,
        "correction_rounds": 0, "fix_attempts": 0, "fix_samples": 0, "max_new_tokens": cap, "ctx": ctx,
        "reserve_gb": 6.0, "max_slots": slots, "lake_bin": str(m.LAKE_BIN), "verify_project": str(m.VERIFY_PROJECT),
        "gguf_dir": str(m.GGUF_DIR), "llama_server": str(m.LLAMA_SERVER),
    }  # fmt: skip


def _alive(pid: object) -> bool:
    """Whether the stream that wrote a claim is still running."""
    try:
        os.kill(int(pid), 0)  # type: ignore[call-overload]
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _closed(job: dict, fills: list[dict]) -> tuple[bool, list[str]]:
    """Apply every fill to the arm's file, then the acceptance gate on the completed file."""
    accepted, reasons, _, _ = _complete(job, fills)
    return accepted, reasons


def _complete(job: dict, fills: list[dict]) -> tuple[bool, list[str], str | None, gate.GateResult | None]:
    """(accepted, reasons, the completed file, the gate's verdict) for an arm's verified fills."""
    by_hole = {int(f["hole"]): f for f in fills}
    holes = [int(h["index"]) for h in job["holes"]]
    if any(k not in by_hole for k in holes):
        return False, [f"{sum(k in by_hole for k in holes)}/{len(holes)} holes verified"], None, None
    name = re.compile(r"(?<![\w.'])" + re.escape(job["target"]) + r"(?![\w'])")
    file_text, decl = job["file"], job["declaration"]
    for k in holes:
        f = by_hole[k]
        if name.search(f["fill"]) or name.search(f["helpers"]):
            return False, [f"hole {k}: the fill names the theorem under test (self-citation)"], None, None
        file_text, decl = hf.apply_fill(file_text, decl, k, f["fill"], f["helpers"], note=f["source"])
    run = m._compile_v428(file_text + gate.probe_suffix(job["target"]))
    res = gate.judge(returned=file_text, submitted=job["file"], target=job["target"], compile_run=run,
                     submitted_type=job["pp_type"], allowed_axioms=tuple(job["allowed_axioms"]),
                     environment=m.VERIFY_ENV)  # fmt: skip
    return res.accepted, res.reasons, file_text, res


def _run_arm(arm_dir: Path, job: dict, *, lenient: bool = False, timeout: int = ARM_TIMEOUT_S) -> dict:
    arm_dir.mkdir(parents=True, exist_ok=True)
    for f in ("status.json", "fills.json", "samples.jsonl"):
        (arm_dir / f).unlink(missing_ok=True)
    (arm_dir / "job.json").write_text(json.dumps(job, indent=2))
    # only servers a dead worker left behind: a live one may be another stream's
    stray = lp.orphaned_servers()
    if stray:
        subprocess.run(["kill", *stray], check=False)
        time.sleep(5)
    start = time.monotonic()
    with open(arm_dir / "worker.log", "w") as log:
        try:
            subprocess.run([sys.executable, "-m", "src.local_prover", str(arm_dir / "job.json")], cwd=str(WF),
                           env={**os.environ, "PYTHONPATH": str(WF)}, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, check=False)  # fmt: skip
        except subprocess.TimeoutExpired:
            pass
    status = json.loads((arm_dir / "status.json").read_text()) if (arm_dir / "status.json").exists() else {}
    fills = json.loads((arm_dir / "fills.json").read_text()) if (arm_dir / "fills.json").exists() else []
    samples = _jsonl(arm_dir / "samples.jsonl")
    outcomes: dict[str, int] = {}
    for s in samples:
        outcomes[s["outcome"]] = outcomes.get(s["outcome"], 0) + 1
    lifting = [
        note for h in (status.get("holes") or {}).values()
        if not h.get("plan") and ("re-elaborate" in (note := h.get("note") or "") or "printed no statement" in note)
    ]  # fmt: skip
    # every attempt that ran must have had the full budget (a long prompt must not shorten it); in an
    # escalation tier only first attempts must: a revision carries its failed answer in the prompt
    full = [s for s in samples if not lenient or s.get("round") == 0]
    short = [s["id"] for s in full if s.get("budget") is not None and int(s["budget"]) != int(job["max_new_tokens"])]
    # an arm that produced no answer at all (no server answered, a remote-only run whose machine went
    # away) tested nothing: infrastructure, to be rerun — never the model's failure
    unserved = not samples and not fills
    infra = (status.get("state") != "done" or any(outcomes.get(o) for o in INFRA) or bool(lifting) or bool(short)
             or unserved)  # fmt: skip
    if lenient and fills:  # an escalation tier's proof the gate accepts stands, whatever else went wrong
        closed, why = _closed(job, fills)
        infra = infra and not closed
    else:
        closed, why = (False, ["infrastructure"]) if infra else _closed(job, fills)
    model = (status.get("models") or {}).get(job["model"]) or {}
    return {
        "state": status.get("state"), "infra": infra, "closed": closed, "why": why, "lifting": lifting,
        "budget_short": short, "truncated": outcomes.get("truncated", 0), "cap": job["max_new_tokens"],
        "holes": len(job["holes"]), "holes_verified": len({f["hole"] for f in fills}), "budget": job["samples"],
        "outcomes": outcomes, "tokens": model.get("tokens"), "gen_seconds": model.get("gen_seconds"),
        "wall_s": round(time.monotonic() - start), "verified_samples": [f["sample"] for f in fills],
    }  # fmt: skip


def pairs(exp: Path, model: str, limit: int | None) -> None:
    rows = _rows()
    sample = _sample(exp)[:limit] if limit else _sample(exp)
    out = exp / "pairs" / model
    out.mkdir(parents=True, exist_ok=True)
    for i, tid in enumerate(sample):
        rec_path = out / f"{tid}.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"theorem_id": tid, "arms": {}, "reruns": {}}
        handoff = json.loads((exp / "handoffs" / f"{tid}.json").read_text())["handoff"]
        holes = len(handoff["holes"])
        rec.update(holes=holes, kept=handoff.get("closures_kept"))
        jobs = {"A": _job(f"{tid}-A", rows[tid], handoff, model, whole=True, samples=K * holes),
                "B": _job(f"{tid}-B", rows[tid], handoff, model, whole=False, samples=K)}
        for job in jobs.values():  # EXP-DEC-001's protocol: no repair loop, as pre-registered
            job["repair"] = False
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]  # alternate which arm goes first
        for arm in order:
            done = rec["arms"].get(arm)
            if done and (not done["infra"] or rec["reruns"].get(arm, 0) >= 1):
                continue
            if done and done["infra"]:
                rec["reruns"][arm] = rec["reruns"].get(arm, 0) + 1
            print(f"[{i + 1}/{len(sample)}] {tid} arm {arm} (holes {holes}, budget {jobs[arm]['samples']})", flush=True)
            rec["arms"][arm] = _run_arm(exp / "arms" / model / tid / arm, jobs[arm])
            rec_path.write_text(json.dumps(rec, indent=2))
            a = rec["arms"][arm]
            print(f"    {arm}: closed={a['closed']} infra={a['infra']} holes {a['holes_verified']}/{a['holes']} "
                  f"outcomes={a['outcomes']} wall={a['wall_s']}s", flush=True)  # fmt: skip


def recheck(exp: Path, model: str) -> None:
    """Amendment 1's robustness check: in every scored discordant pair whose losing arm truncated an
    attempt, rerun that arm alone at RECHECK_CAP tokens. Reported beside the primary, never instead."""
    rows = _rows()
    for rec_path in sorted((exp / "pairs" / model).glob("*.json")):
        rec = json.loads(rec_path.read_text())
        arms = rec.get("arms") or {}
        if {"A", "B"} - set(arms) or arms["A"]["infra"] or arms["B"]["infra"]:
            continue
        if arms["A"]["closed"] == arms["B"]["closed"]:
            continue
        loser = "B" if arms["A"]["closed"] else "A"
        if not arms[loser].get("truncated") or loser in (rec.get("recheck") or {}):
            continue
        tid = rec["theorem_id"]
        handoff = json.loads((exp / "handoffs" / f"{tid}.json").read_text())["handoff"]
        job = _job(f"{tid}-{loser}-recheck", rows[tid], handoff, model, whole=(loser == "A"),
                   samples=K * len(handoff["holes"]) if loser == "A" else K,
                   cap=RECHECK_CAP, ctx=RECHECK_CTX, slots=RECHECK_SLOTS)  # fmt: skip
        print(f"recheck {tid} arm {loser} at {RECHECK_CAP} tokens", flush=True)
        rec.setdefault("recheck", {})[loser] = _run_arm(exp / "arms" / model / tid / f"{loser}_recheck", job)
        rec_path.write_text(json.dumps(rec, indent=2))
        print(f"    {loser} recheck: closed={rec['recheck'][loser]['closed']} infra={rec['recheck'][loser]['infra']}",
              flush=True)  # fmt: skip


DEMO_SEED = 20261030


def demo(exp: Path, model: str, limit: int | None, remote_only: bool = False, reverse: bool = False,
         pool_key: str = "", start: int = 0, escalate_from: str = "", corrections: int = 0,
         cap: int = CAP, capped: str = "any", tier: str = "", revise: bool = False, samples: int = K,
         fresh_uncapped: int = K, ctx: int = 0, fix: tuple[int, int] = (4, 2), whole: bool = False,
         theorems: str = "open") -> None:  # fmt: skip
    """The capacity run: the production lane on every residual — Wayfinder's holes to the local model,
    the completed file through the gate, each accepted proof saved with its provenance — in a seeded
    random order, so the tally at any point is an unbiased sample of the residual population. It
    takes every theorem not yet proven: the residuals into the loop, and the theorems with no
    usable residual (the search never started, or left nothing replayable) on the whole statement.
    Scored leniently (a proof the gate accepts stands), with the repair loop and the REPL on."""
    rows = _rows()
    order = sorted(_population(exp)) + (sorted(_unstarted(exp)) if not escalate_from else [])
    random.Random(DEMO_SEED).shuffle(order)
    order = order[:limit] if limit else order
    if reverse:  # a second stream works the same order from the end; records keep the streams apart
        order = order[::-1]
    if start:  # a third stream starts part-way in
        order = order[start:] + order[:start]
    out = exp / "demo" / model
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "proofs").mkdir(parents=True, exist_ok=True)
    residuals = set(_population(exp))
    if escalate_from:
        return _escalate(exp, model, order, escalate_from, corrections, rows, cap, capped, tier, revise, samples,
                         fresh_uncapped, ctx, fix, whole, theorems)
    for i, tid in enumerate(order, 1):
        rec_path = out / "records" / f"{tid}.json"
        prior = json.loads(rec_path.read_text()) if rec_path.exists() else None
        if prior and prior.get("in_progress"):
            if _alive(prior.get("pid")):
                continue  # another stream is on it
            prior = None  # a claim left by a stream that stopped: the theorem is free
        if prior and (not prior["arm"]["infra"] or prior.get("reruns", 0) >= 1):
            continue
        whole = tid not in residuals
        handoff = _whole_handoff(exp, tid, rows[tid]) if whole else json.loads(
            (exp / "handoffs" / f"{tid}.json").read_text())["handoff"]  # fmt: skip
        if whole and handoff.get("status") in ("closed", "refuted", "failed"):
            continue  # settled on the whole statement already (or not a statement to prove)
        job = _job(f"{tid}-demo", rows[tid], handoff, model, whole=whole, samples=K)
        job["repl"] = True
        if pool_key:
            job["pool_key"] = pool_key
        if remote_only:
            job["remote_only"] = True
        # every stream claims its theorem, so none starts one another live stream is working
        claim = {"theorem_id": tid, "in_progress": True, "pid": os.getpid(), "arm": {"infra": False, "closed": False}}
        rec_path.write_text(json.dumps(claim))
        arm = _run_arm(out / "arms" / tid, job, lenient=True)
        rec = {"theorem_id": tid, "holes": len(handoff["holes"]), "kept": handoff.get("closures_kept"), "arm": arm,
               "whole": whole}  # fmt: skip
        if prior:
            rec["reruns"] = prior.get("reruns", 0) + 1
        if arm["closed"]:
            fills = json.loads((out / "arms" / tid / "fills.json").read_text())
            ok, _, completed, res = _complete(job, fills)
            if ok and completed and res:
                provenance = (f"Wayfinder deterministic search ({handoff.get('closures_kept', 0)} kept closures) + "
                              + "; ".join(f"hole {f['hole']}: {f['source']}" for f in fills))  # fmt: skip
                (out / "proofs" / f"{tid}.lean").write_text(m._acceptance_header(res, provenance) + completed)
        current = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        if not current.get("in_progress") and (current.get("arm") or {}).get("closed") and not arm["closed"]:
            continue  # another stream closed it meanwhile: a closure is never overwritten
        rec_path.write_text(json.dumps(rec, indent=2))
        done = [r for f in (out / "records").glob("*.json") if not (r := json.loads(f.read_text())).get("in_progress")]
        closed = sum(1 for r in done if r["arm"]["closed"])
        infra = sum(1 for r in done if r["arm"]["infra"])
        print(f"[{i}/{len(order)}] {tid}: {'CLOSED' if arm['closed'] else 'open'} "
              f"(holes {arm['holes_verified']}/{arm['holes']}, {arm['outcomes']}, {arm['wall_s']}s) — "
              f"running tally {closed} closed of {len(done) - infra} scored, {infra} infra", flush=True)  # fmt: skip


def _hit_cap(arm_dir: Path) -> bool:
    """Whether any of an arm's answers stopped at the token cap. Read from the stop type, not the
    label: a cut-off answer's last code block is often a draft with `sorry`, labelled `denied`."""
    f = arm_dir / "samples.jsonl"
    return f.exists() and any(json.loads(x).get("stop_type") == "limit" for x in f.read_text().splitlines() if x)


def _closeness(arm_dir: Path) -> int:
    """How close the lower tier came: the fewest errors Lean reported on one of its proofs."""
    wrong = [s for s in _jsonl(arm_dir / "samples.jsonl") if s.get("outcome") == "wrong_proof"]
    return min((len(s.get("errors") or []) for s in wrong), default=99)


def _closed_anywhere(exp: Path) -> set[str]:
    """Theorems some demo tier has closed (a later tier never redoes one)."""
    out = set()
    for f in (exp / "demo").glob("*/records/*.json"):
        r = json.loads(f.read_text())
        if not r.get("in_progress") and (r.get("arm") or {}).get("closed"):
            out.add(r["theorem_id"])
    return out


def _unstarted(exp: Path) -> list[str]:
    """Theorems the search never started (it crashed on them): no residual exists to hand over."""
    out = []
    for f in sorted((exp / "handoffs").glob("*.json")):
        rec = json.loads(f.read_text())
        h = rec.get("handoff") or {}
        if rec.get("status") in ("skipped_start", "handoff_error", "bare", "failed") or (
            rec.get("status") == "residual" and not (h.get("closures_kept") or 0)
        ):
            out.append(rec["theorem_id"])
    return out


def _whole_handoff(exp: Path, tid: str, row: dict) -> dict:
    """What a whole-statement attempt needs (target, checked type): the residual's hand-off, or for
    a theorem the search never started a bare one, built once — the finishing pass's closers try
    the whole statement first, so it may come back closed by Wayfinder alone."""
    rec = json.loads((exp / "handoffs" / f"{tid}.json").read_text())
    if rec.get("handoff"):
        return rec["handoff"]
    cache = exp / "handoffs_whole" / f"{tid}.json"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(m._build_handoff(row, {"initial_goal_tier": ""}), indent=2))
    return json.loads(cache.read_text())


def _escalate(exp: Path, model: str, order: list[str], below: str, corrections: int, rows: dict,
              cap: int = CAP, capped: str = "any", tier: str = "", revise: bool = False, samples: int = K,
              fresh_uncapped: int = K, ctx: int = 0, fix: tuple[int, int] = (4, 2), whole: bool = False,
              theorems: str = "open") -> None:  # fmt: skip
    """The next tier: `model` on every residual the `below` model tried and left open, with
    `corrections` self-correction rounds, rescanning while `below`'s streams still have theorems in
    progress — the ladder Wayfinder → small model → large model, all on this machine. `capped`
    splits the open residuals by whether any of `below`'s answers ran out of tokens ("only": a
    longer-cap retry; "none": everything else); a tier with its own `cap` gets its own directory.
    `whole` is the standard treatment: the model on the theorem itself, as it would work alone — for
    the theorems the search never started (`theorems="unstarted"`) and the residuals no tier closed
    (`"open"`, infrastructure failures included). No tier redoes a theorem another has closed."""
    name = tier or (model if cap == CAP else f"{model}-cap{cap}")
    if name == below:
        raise SystemExit(f"tier {name!r} would write over the tier it escalates from: name it with --tier")
    src, out = exp / "demo" / below / "records", exp / "demo" / name
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "proofs").mkdir(parents=True, exist_ok=True)
    ctx = ctx or max(CTX, min(40960, cap + 12288))  # room for a revision's prompt beside the full cap
    timeout = max(ARM_TIMEOUT_S, 12 * 3600 if revise or cap > CAP else 0)
    done: set[str] = set()
    no_residual = set(_unstarted(exp))
    order = order + [t for t in sorted(no_residual) if t not in order]  # the lower tier took these too
    quiet = 0
    while True:
        below_recs = {t: json.loads((src / f"{t}.json").read_text()) for t in order if (src / f"{t}.json").exists()}
        open_ = [t for t in order if t in below_recs and not below_recs[t].get("in_progress")
                 and not below_recs[t]["arm"]["closed"] and not below_recs[t]["arm"]["infra"]]  # fmt: skip
        if whole:  # the standard treatment also takes what failed for infrastructure
            open_ = [t for t in order if t in below_recs and not below_recs[t].get("in_progress")
                     and not below_recs[t]["arm"]["closed"]]  # fmt: skip
        if capped != "any":
            open_ = [t for t in open_ if _hit_cap(exp / "demo" / below / "arms" / t) == (capped == "only")]
        if theorems == "unstarted":
            open_ = []
        if theorems in ("unstarted", "all"):
            open_ += _unstarted(exp)
        closed_elsewhere = _closed_anywhere(exp)
        todo = [t for t in open_ if t not in done and t not in closed_elsewhere
                and not (out / "records" / f"{t}.json").exists()]  # fmt: skip
        # revising is cheap where the lower tier's answers ran to completion: those first, closest first
        todo.sort(key=lambda t: (_hit_cap(exp / "demo" / below / "arms" / t), _closeness(exp / "demo" / below / "arms" / t)))
        if not todo:
            if theorems == "unstarted":
                return  # nothing the lower tier is still finishing can add to these
            # wait while a lower-tier claim is live; leave after two quiet looks (a stream may be
            # between theorems), so a stopped stream's leftover claim cannot hold this tier forever
            live = any(r.get("in_progress") and _alive(r.get("pid")) for r in below_recs.values())
            quiet = 0 if live else quiet + 1
            if quiet < 2:
                time.sleep(300)
                continue
            return
        quiet = 0
        tid = todo[0]
        # a theorem with no usable residual is on its whole statement in every tier
        on_whole = whole or tid in no_residual
        handoff = (_whole_handoff(exp, tid, rows[tid]) if on_whole
                   else json.loads((exp / "handoffs" / f"{tid}.json").read_text())["handoff"])  # fmt: skip
        hit_cap = _hit_cap(exp / "demo" / below / "arms" / tid)
        fresh = samples if (hit_cap or not revise or whole) else fresh_uncapped
        if on_whole and handoff.get("status") in ("closed", "refuted", "failed"):
            # the finishing pass settled it on the whole statement (closed by Wayfinder alone, false, or
            # not elaborating): nothing for a model to do
            closed = handoff["status"] == "closed"
            rec = {"theorem_id": tid, "whole": True, "settled_by_wayfinder": handoff["status"],
                   "arm": {"closed": closed, "infra": handoff["status"] == "failed", "outcomes": {}, "wall_s": 0}}  # fmt: skip
            (out / "records" / f"{tid}.json").write_text(json.dumps(rec, indent=2))
            done.add(tid)
            print(f"[{out.name}] {tid}: the finishing pass on the whole statement: {handoff['status']}", flush=True)
            continue
        job = _job(f"{tid}-{out.name}", rows[tid], handoff, model, whole=on_whole, samples=fresh, cap=cap, ctx=ctx)
        # repair: between revision rounds, rejected proofs' skeletons to Wayfinder's closers (§5)
        job.update(correction_rounds=corrections, fix_attempts=fix[0], fix_samples=fix[1], repair=True, repl=True)
        if revise and not whole:  # the lower tier's rejected proofs (Goedel's self-correction); seeds answered
            # the same statement either way: the lower tier gave a no-residual theorem its whole statement too
            below_arm = exp / "demo" / below / "arms" / tid
            seeds = [{k: s.get(k) for k in ("id", "quant", "server", "model", "stop_type")}
                     for s in _jsonl(below_arm / "samples.jsonl") if s.get("outcome") == "wrong_proof"]  # fmt: skip
            job["seed"] = {"arm": str(below_arm), "samples": seeds}
        arm = _run_arm(out / "arms" / tid, job, lenient=True, timeout=timeout)
        rec = {"theorem_id": tid, "holes": len(handoff["holes"]), "kept": handoff.get("closures_kept"), "arm": arm,
               "escalated_from": below, "revised": revise, "fresh": fresh, "hit_cap_below": hit_cap,
               "whole": whole}  # fmt: skip
        if arm["closed"]:
            fills = json.loads((out / "arms" / tid / "fills.json").read_text())
            ok, _, completed, res = _complete(job, fills)
            if ok and completed and res:
                how = ("the whole statement (standard treatment)" if whole else
                       f"Wayfinder deterministic search ({handoff.get('closures_kept', 0)} kept closures)")
                provenance = f"{how} + " + "; ".join(f"hole {f['hole']}: {f['source']}" for f in fills)
                (out / "proofs" / f"{tid}.lean").write_text(m._acceptance_header(res, provenance) + completed)
        (out / "records" / f"{tid}.json").write_text(json.dumps(rec, indent=2))
        done.add(tid)
        tally = [json.loads(f.read_text()) for f in (out / "records").glob("*.json")]
        print(f"[{out.name}] {tid}: {'CLOSED' if arm['closed'] else 'open'} ({arm['outcomes']}, {arm['wall_s']}s) — "
              f"{sum(1 for r in tally if r['arm']['closed'])} closed of {len(tally)} the {below} left open", flush=True)


# ── reconcile: Wayfinder's own closes, as the v4.28 gate sees them ─────────────


def reconcile(exp: Path) -> None:
    """A theorem counts as Wayfinder's only once its proof passes the gate the models' proofs pass
    (v4.28, from scratch; proofs in `wayfinder_proofs/`, checked beforehand). The search runs on
    v4.27 through Pantograph, and a close whose replay does not pass at v4.28 goes back into the
    pipeline: its hand-off rebuilt from the search's tree (the old record archived)."""
    rows = _rows()
    details = {d["theorem_id"]: d for d in _jsonl(exp / "search.jsonl")}
    proofs, archive = exp / "wayfinder_proofs", exp / "handoffs_archive_pre_reconcile"
    for f in sorted((exp / "handoffs").glob("*.json")):
        rec = json.loads(f.read_text())
        tid = rec["theorem_id"]
        if rec.get("status") not in ("search_closed", "closed") or (proofs / f"{tid}.lean").exists():
            continue
        archive.mkdir(exist_ok=True)
        (archive / f.name).write_text(f.read_text())
        h = m._build_handoff(rows[tid], details[tid])
        if h.get("status") == "closed":  # a closed replay the gate refused is not closed: the whole statement
            h = m._build_handoff(rows[tid], {"initial_goal_tier": ""})
        f.write_text(json.dumps({"theorem_id": tid, "status": h.get("status"), "handoff": h,
                                 "reconciled_from": f"{rec.get('status')} (its proof did not pass the v4.28 gate)"},
                                indent=2))  # fmt: skip
        print(f"{tid}: {rec.get('status')} -> {h.get('status')}", flush=True)


# ── the capacity run's ledger: what each theorem cost ─────────────────────────


def ledger(exp: Path) -> dict:
    """Per theorem, across every demo tier: the answers the models generated (a revised-from seed is
    not a new answer), their tokens (a cut-off answer has no count from the server: estimated from
    its text, ~3.6 characters a token), and which tier closed it. Wayfinder's own closes cost none.
    Goedel-Prover-V2's published MiniF2F figures are pass@32: 32 answers a theorem (up to 96
    generations with its two revision rounds)."""
    demo = exp / "demo"
    tiers = sorted(d.name for d in demo.iterdir() if (d / "records").is_dir())
    statuses = {json.loads(f.read_text())["theorem_id"]: json.loads(f.read_text()).get("status")
                for f in (exp / "handoffs").glob("*.json")}  # fmt: skip
    rows: dict[str, dict] = {}
    for tid, st in statuses.items():
        row = {"answers": 0, "tokens": 0, "estimated_tokens": 0, "closed_by": None, "tiers": []}
        verified = exp / "wayfinder_proofs"
        if st in ("closed", "search_closed") and (not verified.is_dir() or (verified / f"{tid}.lean").exists()):
            row["closed_by"] = "wayfinder"  # once the v4.28 gate has checked its proof, when that check ran
        for tier in tiers:
            arm = demo / tier / "arms" / tid
            rec_path = demo / tier / "records" / f"{tid}.json"
            if not (arm / "samples.jsonl").exists():
                continue
            row["tiers"].append(tier)
            for s in _jsonl(arm / "samples.jsonl"):
                out = arm / "outputs" / f"{s['id']}.txt"
                if "-seed-" in str(s["id"]) or not out.exists():
                    continue
                row["answers"] += 1
                if s.get("tokens"):
                    row["tokens"] += int(s["tokens"])
                else:
                    row["estimated_tokens"] += round(len(out.read_text()) / 3.6)
            rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
            if not row["closed_by"] and not rec.get("in_progress") and (rec.get("arm") or {}).get("closed"):
                row["closed_by"] = tier
        rows[tid] = row
    closed = [r for r in rows.values() if r["closed_by"]]
    by_model = [r for r in closed if r["closed_by"] != "wayfinder"]
    spent = sorted(r["answers"] for r in by_model)
    total_answers = sum(r["answers"] for r in rows.values())
    summary = {
        "theorems": len(rows), "closed": len(closed), "closed_by": {}, "answers_total": total_answers,
        "answers_per_theorem": round(total_answers / max(1, len(rows)), 1),
        "answers_per_model_closure": {"median": spent[len(spent) // 2] if spent else None,
                                      "max": max(spent) if spent else None},
        "tokens_total": sum(r["tokens"] + r["estimated_tokens"] for r in rows.values()),
        "goedel_published": "pass@32 = 32 answers a theorem (≤ 96 generations with two revision rounds)",
    }  # fmt: skip
    for r in closed:
        summary["closed_by"][r["closed_by"]] = summary["closed_by"].get(r["closed_by"], 0) + 1
    (exp / "ledger.json").write_text(json.dumps({"summary": summary, "theorems": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ── phase 4: the report ───────────────────────────────────────────────────────


def report(exp: Path) -> dict:
    counts: dict[str, int] = {}
    for f in (exp / "handoffs").glob("*.json"):
        s = json.loads(f.read_text()).get("status")
        counts[s] = counts.get(s, 0) + 1
    out: dict = {"corpus": len(_rows()), "searched": len(_jsonl(exp / "search.jsonl")), "handoff_status": counts,
                 "population": len(_population(exp)), "models": {}}  # fmt: skip
    for model_dir in sorted((exp / "pairs").glob("*")):
        recs = [json.loads(f.read_text()) for f in sorted(model_dir.glob("*.json"))]
        scored = [r for r in recs if {"A", "B"} <= set(r["arms"]) and not r["arms"]["A"]["infra"]
                  and not r["arms"]["B"]["infra"]]  # fmt: skip
        cell = {"A+B+": 0, "A+B-": 0, "A-B+": 0, "A-B-": 0}
        for r in scored:
            cell[f"A{'+' if r['arms']['A']['closed'] else '-'}B{'+' if r['arms']['B']['closed'] else '-'}"] += 1
        holes = sum(r["arms"]["B"]["holes"] for r in scored)
        holes_ok = sum(r["arms"]["B"]["holes_verified"] for r in scored)

        def tot(arm: str, key: str) -> float:
            return round(sum(float(r["arms"][arm].get(key) or 0) for r in scored), 1)

        checked = dict(cell)
        for r in scored:
            for arm, res in (r.get("recheck") or {}).items():
                if res["infra"] or not res["closed"]:
                    continue
                before = f"A{'+' if r['arms']['A']['closed'] else '-'}B{'+' if r['arms']['B']['closed'] else '-'}"
                checked[before] -= 1
                checked["A+B+"] += 1  # the losing arm closed it given the longer budget: concordant

        out["models"][model_dir.name] = {
            "pairs_run": len(recs), "pairs_scored": len(scored), "excluded_infra": len(recs) - len(scored),
            "cells_after_32k_recheck": checked,
            "truncated_attempts": {"A": tot("A", "truncated"), "B": tot("B", "truncated")},
            "cells": cell, "A_closed": cell["A+B+"] + cell["A+B-"], "B_closed": cell["A+B+"] + cell["A-B+"],
            "B_holes_verified": f"{holes_ok}/{holes}",
            "tokens": {"A": tot("A", "tokens"), "B": tot("B", "tokens")},
            "gen_seconds": {"A": tot("A", "gen_seconds"), "B": tot("B", "gen_seconds")},
        }  # fmt: skip
    (exp / "report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("phase", choices=["search", "handoffs", "pairs", "recheck", "demo", "ledger", "reconcile", "report"])
    ap.add_argument("--dir", default=str(WF / "runs" / "exp_dec_001"))
    ap.add_argument("--workers", type=int, default=3, help="hand-off builders in parallel")
    ap.add_argument("--model", default=lp.LIGHT, choices=sorted(lp.MODELS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N of the frozen sample (the pilot)")
    ap.add_argument("--remote-only", action="store_true", help="demo: sample only on the pool file's remote servers")
    ap.add_argument("--reverse", action="store_true", help="demo: work the order from the end (a second stream)")
    ap.add_argument("--pool-key", default="", help="demo: the pool-file key whose servers this stream uses")
    ap.add_argument("--start", type=int, default=0, help="demo: start this far into the order (a third stream)")
    ap.add_argument("--escalate-from", default="", help="demo: only the residuals this lower-tier model left open")
    ap.add_argument("--corrections", type=int, default=0, help="demo --escalate-from: self-correction rounds")
    ap.add_argument("--cap", type=int, default=CAP, help="demo --escalate-from: tokens per answer")
    ap.add_argument("--capped", default="any", choices=["any", "only", "none"],
                    help="demo --escalate-from: the residuals where the lower tier ran out of tokens, or the rest")
    ap.add_argument("--tier", default="", help="demo --escalate-from: this tier's directory under demo/")
    ap.add_argument("--revise", action="store_true",
                    help="demo --escalate-from: start from the lower tier's rejected proofs and Lean's errors")
    ap.add_argument("--samples", type=int, default=K, help="demo --escalate-from: fresh first attempts per hole")
    ap.add_argument("--fresh-uncapped", type=int, default=K,
                    help="demo --escalate-from --revise: fresh attempts where the lower tier never ran out of tokens")
    ap.add_argument("--ctx", type=int, default=0, help="demo --escalate-from: tokens of context per slot (0: cap + 12K)")
    ap.add_argument("--whole", action="store_true",
                    help="demo --escalate-from: the standard treatment — the model on the theorem itself")
    ap.add_argument("--theorems", default="open", choices=["open", "unstarted", "all"],
                    help="demo --escalate-from: the lower tier's open residuals, the theorems the search never started, or both")
    ap.add_argument("--fix", type=int, nargs=2, default=[4, 2], metavar=("PROOFS", "TRIES"),
                    help="demo --escalate-from: per correction round, how many failed proofs and tries at each")
    args = ap.parse_args()
    exp = Path(args.dir)
    exp.mkdir(parents=True, exist_ok=True)
    m.GOAL_LEDGER = exp / "ledger"  # nothing here writes the real ledger
    if args.phase == "search":
        search(exp)
    elif args.phase == "handoffs":
        handoffs(exp, args.workers)
    elif args.phase == "pairs":
        pairs(exp, args.model, args.limit)
    elif args.phase == "recheck":
        recheck(exp, args.model)
    elif args.phase == "ledger":
        ledger(exp)
    elif args.phase == "reconcile":
        reconcile(exp)
    elif args.phase == "demo":
        demo(exp, args.model, args.limit, remote_only=args.remote_only, reverse=args.reverse,
             pool_key=args.pool_key, start=args.start, escalate_from=args.escalate_from,
             corrections=args.corrections, cap=args.cap, capped=args.capped, tier=args.tier, revise=args.revise,
             samples=args.samples, fresh_uncapped=args.fresh_uncapped, ctx=args.ctx, fix=tuple(args.fix),
             whole=args.whole, theorems=args.theorems)
    else:
        report(exp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
