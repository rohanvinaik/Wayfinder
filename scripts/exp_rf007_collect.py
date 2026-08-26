"""EXP-RF-007 stage C — collect Aristotle verdicts and tally the paired result.

Polls each recorded submission (runs/exp_rf007_submissions.json), downloads
finished proofs, and AUDITS them: a proof that cites the original Mathlib
name (these are restated Mathlib theorems) is a self-citation — counted
raw, excluded from honest, exactly the standing metric discipline. The
paired tally is per-theorem: (bare, sheet) ∈ {solved, failed, pending}²;
the experiment's question is the sheet arm's honest lift.

RUNS UNDER THE ARISTOTLELIB ENV (miniconda):
    ~/miniconda3/bin/python -m scripts.exp_rf007_collect
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

ARISTOTLE_ENV = Path("/Users/rohanvinaik/semantic_probing/.env")


def _key() -> str:
    import os

    if os.environ.get("ARISTOTLE_API_KEY"):
        return os.environ["ARISTOTLE_API_KEY"]
    for line in ARISTOTLE_ENV.read_text().splitlines():
        if line.startswith("ARISTOTLE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def cites_original(proof_text: str, theorem_id: str) -> bool:
    """Self-citation audit: the original name (or its final segment as a
    bare token) appearing in the proof body."""
    if theorem_id in proof_text:
        return True
    leaf = theorem_id.rsplit(".", 1)[-1]
    return bool(re.search(rf"(?<![A-Za-z0-9_.']){re.escape(leaf)}(?![A-Za-z0-9_'])", proof_text))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=Path, default=Path("runs/exp_rf007_submissions.json"))
    parser.add_argument("--proof-dir", type=Path, default=Path("runs/exp_rf007_proofs"))
    parser.add_argument("--out", type=Path, default=Path("runs/exp_rf007_results.json"))
    args = parser.parse_args()

    import aristotlelib

    key = _key()
    if not key:
        raise SystemExit("no ARISTOTLE_API_KEY available")
    aristotlelib.set_api_key(key)
    args.proof_dir.mkdir(parents=True, exist_ok=True)

    ledger = json.loads(args.submissions.read_text())
    results: list[dict] = []
    for sub in ledger["submissions"]:
        pid, tid, arm = sub["project_id"], sub["theorem_id"], sub["arm"]
        entry = dict(sub)
        try:
            project = await aristotlelib.Project.from_id(pid)
            status = str(project.status)
            entry["status"] = status
            if "complete" in status.lower() or "solved" in status.lower():
                dest = args.proof_dir / f"{sub['project_name']}.lean"
                try:
                    await project.download_files(str(dest.parent), overwrite=True)
                except TypeError:
                    await project.download_files(str(dest.parent))
                texts = [
                    p.read_text()
                    for p in dest.parent.glob(f"{sub['project_name']}*")
                    if p.is_file()
                ] or [
                    p.read_text() for p in dest.parent.glob("*.lean") if p.is_file()
                ]
                proof = "\n".join(texts)
                entry["has_sorry"] = "sorry" in proof
                entry["self_citation"] = cites_original(proof, tid)
                entry["solved_raw"] = bool(proof) and not entry["has_sorry"]
                entry["solved_honest"] = entry["solved_raw"] and not entry["self_citation"]
        except Exception as exc:  # noqa: BLE001 — per-submission isolation
            entry["status"] = f"poll_error:{type(exc).__name__}"
        results.append(entry)
        print(f"  {tid}/{arm}: {entry.get('status')}"
              + (f" raw={entry.get('solved_raw')} honest={entry.get('solved_honest')}"
                 if "solved_raw" in entry else ""))

    by_thm: dict[str, dict] = {}
    for r in results:
        by_thm.setdefault(r["theorem_id"], {})[r["arm"]] = r
    tally = {"pairs": len(by_thm),
             "bare_raw": sum(1 for v in by_thm.values() if v.get("bare", {}).get("solved_raw")),
             "bare_honest": sum(1 for v in by_thm.values() if v.get("bare", {}).get("solved_honest")),
             "sheet_raw": sum(1 for v in by_thm.values() if v.get("sheet", {}).get("solved_raw")),
             "sheet_honest": sum(1 for v in by_thm.values() if v.get("sheet", {}).get("solved_honest")),
             "pending": sum(1 for r in results if "solved_raw" not in r)}
    args.out.write_text(json.dumps({"tally": tally, "results": results}, indent=2))
    print("\n=== EXP-RF-007 tally ===")
    print(json.dumps(tally, indent=1))
    print(f"→ {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
