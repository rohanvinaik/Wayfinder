"""EXP-RF-007 stage B — submit the paired sample to Aristotle (METERED).

Consumes runs/exp_rf007_sample.jsonl (stage A: validated statements + both
arms' prompts). For each theorem, submits TWO projects — arm "bare" and arm
"sheet" — mirroring wayfinder_mcp's project shape (v4.28 toolchain, mathlib
require, no manifest). Async: submissions return immediately; project ids
are recorded to runs/exp_rf007_submissions.json for later collection by
exp_rf007_collect.py. Paced to be polite to the service.

RUNS UNDER THE ARISTOTLELIB ENV (miniconda), not the repo venv:
    ~/miniconda3/bin/python -m scripts.exp_rf007_submit [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

LEAN_VERSION = "v4.28.0"
ARISTOTLE_ENV = Path("/Users/rohanvinaik/semantic_probing/.env")


def _key() -> str:
    import os

    if os.environ.get("ARISTOTLE_API_KEY"):
        return os.environ["ARISTOTLE_API_KEY"]
    for line in ARISTOTLE_ENV.read_text().splitlines():
        if line.startswith("ARISTOTLE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _lakefile() -> str:
    return (
        'name = "AristotleResidual"\nversion = "0.1.0"\n\n'
        '[[require]]\nname = "mathlib"\n'
        'git = "https://github.com/leanprover-community/mathlib4.git"\n'
        f'rev = "{LEAN_VERSION}"\n\n'
        '[[lean_lib]]\nname = "Residual"\nroots = ["Residual"]\n'
    )


def _project_dir(base: Path, name: str, statement: str) -> Path:
    proj = base / name
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "Residual.lean").write_text(
        f"import Mathlib\n\ntheorem wayfinder_residual : {statement} := by sorry\n"
    )
    (proj / "lakefile.toml").write_text(_lakefile())
    (proj / "lean-toolchain").write_text(f"leanprover/lean4:{LEAN_VERSION}\n")
    return proj


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("runs/exp_rf007_sample.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/exp_rf007_submissions.json"))
    parser.add_argument("--limit", type=int, default=0, help="pairs to submit (0 = all)")
    parser.add_argument("--pace", type=float, default=5.0, help="seconds between submissions")
    args = parser.parse_args()

    import aristotlelib

    key = _key()
    if not key:
        raise SystemExit("no ARISTOTLE_API_KEY available")
    aristotlelib.set_api_key(key)

    pairs = [json.loads(line) for line in args.sample.read_text().splitlines()]
    if args.limit:
        pairs = pairs[: args.limit]

    ledger: dict = {"submissions": []}
    if args.out.exists():
        ledger = json.loads(args.out.read_text())
    already = {(s["theorem_id"], s["arm"]) for s in ledger["submissions"]}

    for i, pair in enumerate(pairs):
        tid = pair["theorem_id"]
        for arm, prompt in (("bare", pair["prompt_bare"]), ("sheet", pair["prompt_sheet"])):
            if (tid, arm) in already:
                print(f"  skip {tid}/{arm}: already submitted")
                continue
            name = f"WF-RF007-{arm}-{tid}".replace(".", "_")[:80]
            base = Path(tempfile.mkdtemp(prefix="rf007_"))
            try:
                proj_dir = _project_dir(base, name, pair["statement"])
                project = None
                # Aristotle enforces a CONCURRENT-project cap (plus burst
                # 429s): the submitter is therefore a slot scheduler —
                # wait for in-flight proofs to finish, then fire the next.
                for attempt in range(48):  # up to ~8h of patient slot-waiting
                    try:
                        project = await aristotlelib.Project.create_from_directory(prompt, proj_dir)
                        break
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc)
                        if "429" in msg or "too many projects" in msg.lower():
                            wait = 600
                            print(f"  slots full ({tid}/{arm}) — waiting {wait}s "
                                  f"(attempt {attempt + 1}/48)", flush=True)
                            await asyncio.sleep(wait)
                            continue
                        raise
                if project is None:
                    print(f"  giving up on {tid}/{arm} after ~8h without a free slot; resume later")
                    return
            finally:
                shutil.rmtree(base, ignore_errors=True)
            ledger["submissions"].append(
                {
                    "theorem_id": tid,
                    "arm": arm,
                    "project_id": project.project_id,
                    "project_name": name,
                    "status_at_submit": str(project.status),
                }
            )
            args.out.write_text(json.dumps(ledger, indent=2))
            print(f"  fired {tid}/{arm} → {project.project_id} ({i + 1}/{len(pairs)})")
            await asyncio.sleep(args.pace)

    print(f"\n{len(ledger['submissions'])} total submissions recorded → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
