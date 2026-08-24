#!/usr/bin/env python3
"""EXP-RF-006 — warranted emission vs legacy string assembly at the v4.28 gate.

THE_REFOUNDING §9's emission experiment. For each proved theorem in a
benchmark result (details JSON with `tactics_used` + `close_provenance`),
emit BOTH artifacts from the same manifest row and lineage:

  legacy    the pre-Wave-3 string assembly (reproduced here verbatim so the
            comparison outlives the code it replaced)
  warranted the LeanExpr path (warrant header, environment pin)

Default is --dry: emit both, assert the compile SURFACE is identical modulo
the warrant comment (the Wave-3 claim: evidence added, behavior preserved),
and write the artifact pairs for inspection. With --verify, each artifact is
compiled in the v4.28 project (`lake env lean`) and the compile/sorryAx
verdicts are paired — the live half, hours not seconds, run deliberately.

Usage:
  python -m scripts.exp_rf006_emission_gate --details <benchmark.json> \
      --manifest data/rerun_v427_manifest.jsonl [--verify]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.lean_expr import EmissionWarrant, from_tactics, read_environment_pin, render_declaration

VERIFY_PROJECT = Path("data/lean_project_v428")
LAKE_BIN = Path("/Users/rohanvinaik/.elan/bin/lake")
OUT_DIR = Path("runs/exp_rf006_artifacts")
REPORT = Path("runs/exp_rf006_report.json")


def legacy_reconstruct(row: dict, tactics_used: list[str], head: str) -> str:
    """The pre-Wave-3 assembly, reproduced verbatim for the paired comparison."""
    imports = "\n".join(row.get("imports") or ["import Mathlib"])
    opens = "\n".join(row.get("opens") or [])
    body = "\n".join("  " + t for t in tactics_used) if tactics_used else "  sorry"
    return "\n".join(x for x in [imports, opens, "", head, body] if x) + "\n"


def declaration_head(row: dict) -> str:
    return row["full_declaration"].split(":= by", 1)[0].rstrip() + " := by"


def strip_comments_and_blanks(text: str) -> list[str]:
    """The compile surface: non-blank lines outside block comments."""
    lines: list[str] = []
    in_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/-"):
            in_comment = True
        if not in_comment and stripped:
            lines.append(stripped)
        if in_comment and stripped.endswith("-/"):
            in_comment = False
    return lines


def compile_at_gate(text: str, name: str) -> tuple[bool, str]:
    probe = text.rstrip() + f"\n#print axioms {name}\n"
    fpath = VERIFY_PROJECT / f"_rf006_{name}.lean"
    try:
        fpath.write_text(probe)
        proc = subprocess.run(
            [str(LAKE_BIN), "env", "lean", str(fpath)],
            cwd=str(VERIFY_PROJECT), capture_output=True, text=True, timeout=420,
        )
        out = proc.stdout + proc.stderr
        return proc.returncode == 0 and "sorryAx" not in out, out[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        fpath.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True, help="benchmark JSON with a details list")
    parser.add_argument("--manifest", type=Path, default=Path("data/rerun_v427_manifest.jsonl"))
    parser.add_argument("--verify", action="store_true", help="compile both artifacts at the v4.28 gate")
    args = parser.parse_args()

    manifest = {}
    with args.manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            manifest[row["theorem_id"]] = row

    payload = json.loads(args.details.read_text())
    details = payload.get("details", payload if isinstance(payload, list) else [])
    environment = read_environment_pin(VERIFY_PROJECT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    surface_identical = 0
    for d in details:
        if not d.get("success") or d.get("theorem_id") not in manifest:
            continue
        row = manifest[d["theorem_id"]]
        head = declaration_head(row)
        tactics = d.get("tactics_used") or []
        legacy = legacy_reconstruct(row, tactics, head)
        warrant = EmissionWarrant(
            theorem_id=d["theorem_id"],
            provenance=tuple(d.get("close_provenance") or ()),
            source_run=str(args.details),
            note="EXP-RF-006 paired emission",
        )
        warranted = render_declaration(
            head,
            from_tactics(
                tactics, warrant,
                imports=tuple(row.get("imports") or ["import Mathlib"]),
                opens=tuple(row.get("opens") or ()),
                environment=environment,
            ),
        )
        same_surface = strip_comments_and_blanks(legacy) == strip_comments_and_blanks(warranted)
        surface_identical += same_surface
        name = d["theorem_id"].replace(".", "_")
        (OUT_DIR / f"{name}.legacy.lean").write_text(legacy)
        (OUT_DIR / f"{name}.warranted.lean").write_text(warranted)
        entry = {"theorem_id": d["theorem_id"], "surface_identical": same_surface}
        if args.verify:
            entry["legacy_compiles"], _ = compile_at_gate(legacy, d["theorem_id"])
            entry["warranted_compiles"], _ = compile_at_gate(warranted, d["theorem_id"])
        pairs.append(entry)

    report = {
        "experiment": "EXP-RF-006",
        "mode": "verify" if args.verify else "dry",
        "environment": environment,
        "pairs": pairs,
        "surface_identical": surface_identical,
        "total": len(pairs),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(
        f"EXP-RF-006 ({report['mode']}) · {len(pairs)} proved theorem(s) · "
        f"surface identical modulo warrant: {surface_identical}/{len(pairs)}"
    )
    if args.verify:
        both = sum(1 for p in pairs if p.get("legacy_compiles") and p.get("warranted_compiles"))
        print(f"  both compile at v4.28: {both}/{len(pairs)}")
    print(f"artifacts → {OUT_DIR} · report → {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
