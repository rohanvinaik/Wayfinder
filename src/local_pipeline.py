"""The local prover's pipeline for one theorem: the batch run's ladder, in order, stopping at the
first closure.

1. the residual — the 8B at 8K tokens, with revisions and the repair loop (skeletons of rejected
   proofs and drafts, Wayfinder's closers and search on their failing steps, completions);
2. where its answers ran out of tokens — the 8B at 30K, revising stage 1's rejected proofs;
3. the whole statement (the standard treatment) — the 8B at 16K, with revisions;
4. the 32B, revising the 8B's best rejected proofs, with fresh tries of its own.

Each stage is one `local_prover` worker run in its own directory under the run's. The run's own
directory mirrors the stage under way (`status.json`) and, once a stage closes the theorem, the
winning stage (`job.json`, `fills.json`; a whole-statement fill is marked `whole`) — so
`local_prover_result` reads a pipeline as it reads a single worker run.

    python -m src.local_pipeline <run_dir>      # run_dir/pipeline.json, written by send_to_local_prover
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WF = Path(__file__).resolve().parents[1]
STAGE_TIMEOUT_S = 6 * 3600


def stages(spec: dict) -> list[dict]:
    """The stage list: (name, job overrides, when it runs). The 32B comes last: it is the slowest."""
    base = {"repl": True, "repair": True, "correction_rounds": 2, "fix_attempts": 4, "fix_samples": 2}
    out = [
        {"name": "8b", "job": {**base, "model": "goedel-v2-8b", "samples": 8, "max_new_tokens": 8192, "ctx": 20480}},
        {"name": "8b-long", "when": "capped", "seed_from": ["8b"],
         "job": {**base, "model": "goedel-v2-8b", "samples": 8, "max_new_tokens": 30000, "ctx": 40960}},
    ]  # fmt: skip
    if spec.get("whole_job") and spec["job"].get("status") != "bare":
        whole = {**base, "model": "goedel-v2-8b", "samples": 8, "max_new_tokens": 16384, "ctx": 28672}
        out.append({"name": "whole", "whole": True, "job": whole})
    out.append({"name": "32b", "seed_from": ["8b", "8b-long"],
                "job": {**base, "model": "goedel-v2-32b", "samples": 4, "max_new_tokens": 8192, "ctx": 16384,
                        "fix_attempts": 2, "fix_samples": 2}})  # fmt: skip
    return out


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def _hit_cap(stage_dir: Path) -> bool:
    return any(s.get("stop_type") == "limit" for s in _jsonl(stage_dir / "samples.jsonl"))


def _seeds(run_dir: Path, names: list[str]) -> dict | None:
    """The rejected proofs of the named earlier stages — fewest errors first — to revise from."""
    best: list[tuple[int, str, dict]] = []
    for name in names:
        d = run_dir / "stages" / name
        for s in _jsonl(d / "samples.jsonl"):
            if s.get("outcome") == "wrong_proof" and (d / "outputs" / f"{s['id']}.txt").exists():
                best.append((len(s.get("errors") or []), str(d), s))
    if not best:
        return None
    best.sort(key=lambda t: t[0])
    arm = best[0][1]  # one arm per seed set: the worker reads outputs from it
    return {"arm": arm, "samples": [{k: s.get(k) for k in ("id", "quant", "server", "model", "stop_type")}
                                    for _, a, s in best if a == arm]}  # fmt: skip


def _mirror(run_dir: Path, status: dict) -> None:
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))


def run(run_dir: Path) -> None:
    spec = json.loads((run_dir / "pipeline.json").read_text())
    trail: list[dict] = []
    status = {"run_id": spec["job"]["run_id"], "theorem_id": spec["job"]["theorem_id"], "pid": os.getpid(),
              "state": "starting", "pipeline": trail, "notes": []}  # fmt: skip
    _mirror(run_dir, status)
    ladder = stages(spec)
    names = [st["name"] for st in ladder]
    begin = names.index(spec["start_at"]) if spec.get("start_at") in names else 0
    for i, st in enumerate(ladder):
        d = run_dir / "stages" / st["name"]
        if i < begin:  # started further down the ladder (a forced stage, or a test of one)
            continue
        if st.get("when") == "capped" and i > begin and not _hit_cap(run_dir / "stages" / "8b"):
            trail.append({"stage": st["name"], "skipped": "stage 1's answers ran to completion"})
            continue
        base = spec["whole_job"] if st.get("whole") else spec["job"]
        job = {**base, **st["job"], "run_id": f"{spec['job']['run_id']}-{st['name']}"}
        if st.get("seed_from") and not st.get("whole"):
            seed = _seeds(run_dir, st["seed_from"])
            if seed:
                job["seed"] = seed
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job, indent=2))
        status.update(state="sampling", stage=st["name"])
        _mirror(run_dir, status)
        started = time.monotonic()
        with open(d / "worker.log", "w") as log:
            try:
                subprocess.run([sys.executable, "-m", "src.local_prover", str(d / "job.json")], cwd=str(WF),
                               env={**os.environ, "PYTHONPATH": str(WF), "PYTHONUNBUFFERED": "1"}, stdout=log,
                               stderr=subprocess.STDOUT, timeout=STAGE_TIMEOUT_S, check=False)  # fmt: skip
            except subprocess.TimeoutExpired:
                status["notes"].append(f"stage {st['name']} timed out after {STAGE_TIMEOUT_S} s")
        worker = _read(d / "status.json", {})
        fills = _read(d / "fills.json", [])
        trail.append({"stage": st["name"], "state": worker.get("state"), "closed": bool(fills),
                      "wall_s": round(time.monotonic() - started), "models": worker.get("models")})  # fmt: skip
        with open(run_dir / "samples.jsonl", "a") as out:  # every stage's answers, for the result's counts
            for s in _jsonl(d / "samples.jsonl"):
                out.write(json.dumps({**s, "stage": st["name"]}) + "\n")
        if fills:  # the winning stage becomes what the run's directory says
            shutil.copy2(d / "job.json", run_dir / "job.json")
            (run_dir / "fills.json").write_text(json.dumps([{**f, "whole": bool(st.get("whole"))} for f in fills],
                                                           indent=2))  # fmt: skip
            status.update(state="done", stage=st["name"], holes=worker.get("holes"), models=worker.get("models"),
                          verified=[f["hole"] for f in fills])  # fmt: skip
            _mirror(run_dir, status)
            return
        status.update(holes=worker.get("holes"), models=worker.get("models"))
        _mirror(run_dir, status)
    status.update(state="done", stage=None, verified=[])
    _mirror(run_dir, status)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m src.local_pipeline <run_dir>", file=sys.stderr)
        return 2
    run(Path(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
