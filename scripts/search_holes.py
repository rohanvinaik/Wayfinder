#!/usr/bin/env python3
"""Wayfinder's deterministic search on a residual's holes, where they sit (src/hole_search.py), in one
run_benchmark session: each hole's goal is the one Lean has at that `sorry`, and what the search
builds comes back as the hole's tactic block — complete, or its validated steps with the goals still
open marked.

    python scripts/search_holes.py <request.json>

request: {"file": the hand-off file, "declaration": its declaration with marked holes, "holes": [k, …],
          "row": the theorem's manifest row, "budget": tactic attempts per hole, "timeout": seconds, "out": path}
out: {k: {"block": the hole's tactic block, "open": goals it leaves open, "attempts": n} | {"error": …}}
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WF))

import wayfinder_mcp as m  # noqa: E402
from src import hole_search as hs  # noqa: E402


def _config(budget: int) -> Path:
    cfg = m._resolved_config()
    cfg.write_text(re.sub(r"(?m)^(\s*budget:\s*)\d+", lambda g: f"{g.group(1)}{budget}", cfg.read_text(), count=1))
    return cfg


def hole_rows(file_text: str, declaration: str, holes: list[int], row: dict) -> list[dict]:
    """One run_benchmark row per hole: the theorem's own row (its imports, file, context) with the
    hole's source to load, under an id of its own; premises stay those accessible to the theorem."""
    return [{**row, "theorem_id": f"{row['theorem_id']}__hole{k}", "accessible_as": row["theorem_id"],
             "hole_source": hs.hole_source(file_text, declaration, k)} for k in holes]  # fmt: skip


def search(file_text: str, declaration: str, holes: list[int], row: dict, budget: int, timeout: int) -> dict[int, dict]:
    out: dict[int, dict] = {k: {"error": "no result"} for k in holes}
    if not holes:
        return out
    rows = hole_rows(file_text, declaration, holes, row)
    cfg = _config(budget)
    with tempfile.TemporaryDirectory() as td:
        manifest, result = Path(td) / "holes.jsonl", Path(td) / "result.json"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cmd = [str(m.VENV_PY), str(m.BENCH), "--config", str(cfg), "--checkpoint", str(m.CHECKPOINT),
               "--backend", "pantograph", "--lean-project", str(m.LEAN_PROJECT), "--lean-imports",
               *m._lean_imports(rows), "--theorems", str(manifest), "--limit", str(len(rows)),
               "--device", "cpu", "--output", str(result)]  # fmt: skip
        try:
            subprocess.run(cmd, cwd=str(m.WF), capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONPATH": str(m.WF)})  # fmt: skip
        except subprocess.TimeoutExpired:
            pass  # a partial result file still reports the holes it finished
        finally:
            cfg.unlink(missing_ok=True)
        if not result.exists():
            return out
        by_id = {f"{row['theorem_id']}__hole{k}": k for k in holes}
        for d in json.loads(result.read_text()).get("details") or []:
            k = by_id.get(str(d.get("theorem_id", "")))
            if k is None:
                continue
            if d.get("initial_goal_tier") != "hole" or not d.get("initial_goals"):
                out[k] = {"error": f"no goal at the hole (tier {d.get('initial_goal_tier')!r})"}
                continue
            try:
                block, n_open = hs.hole_block(d["initial_goals"], d.get("initial_goal_ids"), d.get("close_log") or [])
            except Exception as e:  # noqa: BLE001 — one hole's tree failing leaves the others
                out[k] = {"error": f"{type(e).__name__}: {e}"[:300]}
                continue
            out[k] = {"block": block, "open": n_open, "attempts": d.get("attempts")}
    return out


def main(argv: list[str]) -> int:
    req = json.loads(Path(argv[0]).read_text())
    res = search(req["file"], req["declaration"], [int(k) for k in req["holes"]], req["row"],
                 int(req.get("budget", 200)), int(req.get("timeout", 600)))  # fmt: skip
    Path(req["out"]).write_text(json.dumps({str(k): v for k, v in res.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
