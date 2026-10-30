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
INFRA = {"server_error", "memory_yield", "context_full", "in_place_failed"}


def _rows() -> dict[str, dict]:
    return {r["theorem_id"]: r for r in map(json.loads, CORPUS.read_text().splitlines()) if r}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ── phase 1: the deterministic search ─────────────────────────────────────────


def search(exp: Path) -> None:
    rows = list(_rows().values())
    corpus = exp / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({**r, "full_declaration": m._augmented_declaration(r)}) + "\n" for r in rows))
    cfg = m._resolved_config(dest=exp / "config.yaml")
    out = exp / "search.json"
    cmd = [
        str(m.VENV_PY), "-u", str(m.BENCH), "--config", str(cfg), "--checkpoint", str(m.CHECKPOINT),
        "--backend", "pantograph", "--lean-project", str(m.LEAN_PROJECT), "--lean-imports", *m._lean_imports(rows),
        "--theorems", str(corpus), "--device", "mps", "--per-theorem-timeout", "300", "--output", str(out),
    ]  # fmt: skip
    if out.with_suffix(".jsonl").exists():
        cmd.append("--resume")
    env = {**os.environ, "PYTHONPATH": str(WF), "PYTHONUNBUFFERED": "1"}
    with open(exp / "search.log", "a") as log:
        subprocess.run(cmd, cwd=str(WF), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)


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


def _job(run_id: str, row: dict, handoff: dict, model: str, *, whole: bool, samples: int) -> dict:
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
        "correction_rounds": 0, "fix_attempts": 0, "fix_samples": 0, "max_new_tokens": 32768, "reserve_gb": 6.0,
        "max_slots": 4, "lake_bin": str(m.LAKE_BIN), "verify_project": str(m.VERIFY_PROJECT),
        "gguf_dir": str(m.GGUF_DIR), "llama_server": str(m.LLAMA_SERVER),
    }  # fmt: skip


def _stray_servers() -> list[str]:
    """Our llama-servers left running (never the squishos router on port 8080: it serves no Goedel GGUF)."""
    out = subprocess.run(["pgrep", "-f", "llama-server.*models/gguf/Goedel"], capture_output=True, text=True)
    return out.stdout.split()


def _closed(job: dict, fills: list[dict]) -> tuple[bool, list[str]]:
    """Apply every fill to the arm's file, then the acceptance gate on the completed file."""
    by_hole = {int(f["hole"]): f for f in fills}
    holes = [int(h["index"]) for h in job["holes"]]
    if any(k not in by_hole for k in holes):
        return False, [f"{sum(k in by_hole for k in holes)}/{len(holes)} holes verified"]
    name = re.compile(r"(?<![\w.'])" + re.escape(job["target"]) + r"(?![\w'])")
    file_text, decl = job["file"], job["declaration"]
    for k in holes:
        f = by_hole[k]
        if name.search(f["fill"]) or name.search(f["helpers"]):
            return False, [f"hole {k}: the fill names the theorem under test (self-citation)"]
        file_text, decl = hf.apply_fill(file_text, decl, k, f["fill"], f["helpers"], note=f["source"])
    run = m._compile_v428(file_text + gate.probe_suffix(job["target"]))
    res = gate.judge(returned=file_text, submitted=job["file"], target=job["target"], compile_run=run,
                     submitted_type=job["pp_type"], allowed_axioms=tuple(job["allowed_axioms"]),
                     environment=m.VERIFY_ENV)  # fmt: skip
    return res.accepted, res.reasons


def _run_arm(arm_dir: Path, job: dict) -> dict:
    arm_dir.mkdir(parents=True, exist_ok=True)
    for f in ("status.json", "fills.json", "samples.jsonl"):
        (arm_dir / f).unlink(missing_ok=True)
    (arm_dir / "job.json").write_text(json.dumps(job, indent=2))
    stray = _stray_servers()
    if stray:
        subprocess.run(["kill", *stray], check=False)
        time.sleep(5)
    start = time.monotonic()
    with open(arm_dir / "worker.log", "w") as log:
        try:
            subprocess.run([sys.executable, "-m", "src.local_prover", str(arm_dir / "job.json")], cwd=str(WF),
                           env={**os.environ, "PYTHONPATH": str(WF)}, stdout=log,
                           stderr=subprocess.STDOUT, timeout=ARM_TIMEOUT_S, check=False)  # fmt: skip
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
    infra = status.get("state") != "done" or any(outcomes.get(o) for o in INFRA) or bool(lifting)
    closed, why = (False, ["infrastructure"]) if infra else _closed(job, fills)
    model = (status.get("models") or {}).get(job["model"]) or {}
    return {
        "state": status.get("state"), "infra": infra, "closed": closed, "why": why, "lifting": lifting,
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

        out["models"][model_dir.name] = {
            "pairs_run": len(recs), "pairs_scored": len(scored), "excluded_infra": len(recs) - len(scored),
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
    ap.add_argument("phase", choices=["search", "handoffs", "pairs", "report"])
    ap.add_argument("--dir", default=str(WF / "runs" / "exp_dec_001"))
    ap.add_argument("--workers", type=int, default=3, help="hand-off builders in parallel")
    ap.add_argument("--model", default=lp.LIGHT, choices=sorted(lp.MODELS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N of the frozen sample (the pilot)")
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
    else:
        report(exp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
