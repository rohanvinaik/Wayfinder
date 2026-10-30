#!/usr/bin/env python3
"""Wayfinder's deterministic search on a batch of lifted goals, in one kernel session.

The local prover's repair loop lifts the steps a model's proof failed at (and Wayfinder's closers
could not close) to standalone lemmas; this runs the search on all of them at once — one
Pantograph start for the batch — and reports each lemma's proof when the search closes it.

    python scripts/search_goals.py <request.json>

request: {"rows": [manifest rows: theorem_id, full_declaration, imports, opens, ...],
          "budget": tactic attempts per goal, "timeout": seconds for the batch, "out": path}
out: {theorem_id: {"proved": bool, "declaration": the lemma with the search's proof | null}}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WF))

import wayfinder_mcp as m  # noqa: E402


def _config(budget: int) -> Path:
    cfg = m._resolved_config()
    cfg.write_text(re.sub(r"(?m)^(\s*budget:\s*)\d+", lambda g: f"{g.group(1)}{budget}", cfg.read_text(), count=1))
    return cfg


def search(rows: list[dict], budget: int, timeout: int) -> dict[str, dict]:
    out: dict[str, dict] = {str(r["theorem_id"]): {"proved": False, "declaration": None} for r in rows}
    if not rows:
        return out
    cfg = _config(budget)
    with tempfile.TemporaryDirectory() as td:
        manifest, result = Path(td) / "goals.jsonl", Path(td) / "result.json"
        manifest.write_text("".join(json.dumps({**r, "full_declaration": m._augmented_declaration(r)}) + "\n" for r in rows))
        cmd = [str(m.VENV_PY), str(m.BENCH), "--config", str(cfg), "--checkpoint", str(m.CHECKPOINT),
               "--backend", "pantograph", "--lean-project", str(m.LEAN_PROJECT), "--lean-imports",
               *m._lean_imports(rows), "--theorems", str(manifest), "--limit", str(len(rows)),
               "--device", "cpu", "--output", str(result)]  # fmt: skip
        try:
            subprocess.run(cmd, cwd=str(m.WF), capture_output=True, text=True, timeout=timeout,
                           env={**__import__("os").environ, "PYTHONPATH": str(m.WF)})  # fmt: skip
        except subprocess.TimeoutExpired:
            pass  # a partial result file still reports the goals it finished
        finally:
            cfg.unlink(missing_ok=True)
        if not result.exists():
            return out
        by_id = {str(r["theorem_id"]): r for r in rows}
        for d in json.loads(result.read_text()).get("details") or []:
            tid = str(d.get("theorem_id", ""))
            if tid not in by_id or not (d.get("success") and d.get("honest_success")):
                continue
            row = by_id[tid]
            try:  # the committed tree replayed (with the finishing pass's closes), else the linear trace
                h = m._build_handoff(row, d)
                decl = h.get("declaration") if h.get("status") == "closed" else None
            except Exception:  # noqa: BLE001 — fall back to the linear trace
                decl = None
            if decl is None:
                text = m._reconstruct_proof(row, d.get("tactics_used") or [])
                start = text.find(f"theorem {m._lean_name(row)}")
                decl = text[start:] if start >= 0 else None
            out[tid] = {"proved": decl is not None, "declaration": decl}
    return out


def main(argv: list[str]) -> int:
    req = json.loads(Path(argv[0]).read_text())
    res = search(req["rows"], int(req.get("budget", 200)), int(req.get("timeout", 420)))
    Path(req["out"]).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
