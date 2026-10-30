"""Re-run the acceptance gate over every Aristotle result already downloaded.

Offline: no Aristotle calls. For each project in runs/aristotle/index.json whose
result is on disk, the submitted file is rebuilt from the indexed manifest row
(the same `_proof_lean` the hand-off used), its statement type is read back from
Lean, and the returned `Residual.lean` is judged by `src.proof_gate`.

Caveat for results submitted before the gate existed: the submitted text is a
reconstruction. If a row's source file or an inlined module changed since it was
submitted, a "submitted declaration changed" finding can be our drift, not
Aristotle's edit — the report marks every such finding for a human look.

Run with the MCP host's python (it imports wayfinder_mcp):
    ~/miniconda3/bin/python3 scripts/regate_aristotle_results.py [--workers 3] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.proof_gate as gate  # noqa: E402
import wayfinder_mcp as wf  # noqa: E402


def _submitted_type(text: str, target: str, cache: dict) -> str | None:
    key = hashlib.sha256((target + "\0" + text).encode()).hexdigest()
    if key not in cache:
        run = gate.run_lean_json(wf.LAKE_BIN, wf.VERIFY_PROJECT, text + gate.probe_suffix(target))
        cache[key] = gate.checked_type(run["messages"], target)
    return cache[key]


def regate_one(project_id: str, rec: dict, cache: dict) -> dict:
    row = rec.get("row")
    out = {"project_id": project_id, "theorem_id": rec.get("theorem_id")}
    if not row:
        return {**out, "status": "no indexed row"}
    result_dir = ROOT / "runs" / "aristotle" / project_id
    leans = sorted(result_dir.rglob("Residual.lean")) if result_dir.exists() else []
    if not leans:
        return {**out, "status": "not downloaded"}
    returned = leans[0].read_text()
    submitted = wf._proof_lean(row)
    target = gate.name_from_declaration(row.get("full_declaration", "")) or str(row.get("theorem_id"))
    static = gate.static_findings(returned, submitted)
    run = None if static else gate.run_lean_json(wf.LAKE_BIN, wf.VERIFY_PROJECT, returned + gate.probe_suffix(target))
    res = gate.judge(
        returned=returned,
        submitted=submitted,
        target=target,
        compile_run=run,
        submitted_type=_submitted_type(submitted, target, cache),
        allowed_axioms=tuple(row.get("allowed_axioms") or ()),
        environment=wf.VERIFY_ENV,
    )
    # A declaration the returned file never mentions at all is more likely one
    # added to our source after submission than one Aristotle deleted.
    absent = [
        r.rsplit(": ", 1)[-1]
        for r in res.reasons
        if r.startswith("submitted declaration changed or removed: ")
        and r.rsplit(": ", 1)[-1] not in gate.strip_comments(returned)
    ]
    status = "accepted" if res.accepted else "rejected"
    only_drift = absent and all(
        r.startswith("submitted declaration changed or removed: ") for r in res.reasons
    ) and all(r.rsplit(": ", 1)[-1] in absent for r in res.reasons)
    if not res.accepted and only_drift:
        status = "unverifiable (submission drift)"
    return {**out, "status": status, "drift_suspects": absent, "home": rec.get("home"), **res.to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "aristotle" / "regate_report.json"))
    args = ap.parse_args()
    index = json.loads(wf.ARISTOTLE_INDEX.read_text())
    cache: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda kv: regate_one(kv[0], kv[1], cache), index.items()))
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(Counter(r["status"] for r in rows))
    failed = Counter(name for r in rows if r["status"] == "rejected" for name, ok in r["checks"].items() if not ok)
    print("failed checks:", dict(failed))
    print("modes:", Counter(r.get("mode") for r in rows if r["status"] in ("accepted", "rejected")))
    print("report:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
