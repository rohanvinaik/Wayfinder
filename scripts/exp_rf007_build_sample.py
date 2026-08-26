"""EXP-RF-007 stage A — build the validated, paired submission sample.

The cheat-sheet fold (THE_REFOUNDING §9): Aristotle with vs without the
rendered sheet, on the pass-2 residual. This stage spends NO metered
calls: it selects a deterministic stride sample of the residual, resolves
each theorem's kernel-elaborated statement (env_inspect type.pp), VALIDATES
that the statement elaborates locally (goal_start at v4.27) so no broken
project ever reaches the metered service, and renders both arms' prompts
(bare / + cheat sheet from the pass-2 control ledger row). Output:
runs/exp_rf007_sample.jsonl, consumed by exp_rf007_submit.py (stage B,
which runs under the aristotlelib environment).

Both arms carry the SAME anti-self-citation instruction — these are
Mathlib theorems, so the original name is excluded and results are
audited for self-citation post-hoc (raw vs honest, as always).

Usage:
    .venv/bin/python -m scripts.exp_rf007_build_sample [--n 20] [--oversample 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cheat_sheet import render_cheat_sheet


def stride_sample(ids: list[str], n: int) -> list[str]:
    """Deterministic, order-stable stride over the sorted residual."""
    ids = sorted(ids)
    if len(ids) <= n:
        return ids
    stride = len(ids) / n
    return [ids[int(i * stride)] for i in range(n)]


def build_prompts(theorem_id: str, detail: dict, manifest_row: dict) -> tuple[str, str]:
    """(bare, sheet) prompts — identical base, the sheet fold is the only delta."""
    base = (
        "Fill the sorry placeholder and prove the theorem `wayfinder_residual` using Mathlib. "
        f"This statement is Mathlib's `{theorem_id}` restated under a fresh name: do NOT prove it "
        f"by citing `{theorem_id}` itself or a direct alias of it — produce the proof from "
        "underlying lemmas. Do not add axioms or `sorry`."
    )
    sheet = render_cheat_sheet(detail, manifest_row)
    with_sheet = base + "\n\nContext from Wayfinder's own search (computed, deterministic):\n\n" + sheet
    return base, with_sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--oversample", type=int, default=40)
    parser.add_argument("--control", type=Path, default=Path("runs/loop_pass2_control.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("runs/loop_pass2_residual_manifest.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("runs/exp_rf007_sample.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.control.read_text().splitlines()]
    man = {json.loads(line)["theorem_id"]: json.loads(line)
           for line in args.manifest.read_text().splitlines()}
    residual = [
        r for r in rows
        if not r.get("success") and r.get("disposition") in ("progressed", "blocked_unproven")
    ]
    by_id = {r["theorem_id"]: r for r in residual}
    candidates = stride_sample(list(by_id), args.oversample)
    print(f"residual={len(residual)}; candidate stride sample={len(candidates)}; target n={args.n}")

    # One Pantograph kernel for resolution + validation (v4.27, the search env).
    import yaml

    from scripts.run_benchmark import canonicalize_theorem_id
    from src.lean_interface import LeanKernel, LeanConfig

    cfg = yaml.safe_load(open("configs/wayfinder_deterministic.yaml"))["lean"]
    lean = LeanKernel(LeanConfig(backend="pantograph", project_root=cfg["project_root"],
                                 imports=cfg.get("imports") or ["Mathlib"]))
    lean._ensure_server()

    picked: list[dict] = []
    for tid in candidates:
        if len(picked) >= args.n:
            break
        canon = canonicalize_theorem_id(tid)
        try:
            info = lean._server.env_inspect(canon)
            statement = info["type"]["pp"]
        except Exception as exc:  # noqa: BLE001 — per-theorem isolation
            print(f"  skip {tid}: env_inspect failed ({type(exc).__name__})")
            continue
        if not statement or "⋯" in statement:  # pretty-printer ellipsis = unsound to submit
            print(f"  skip {tid}: unusable pp (empty or elided)")
            continue
        try:
            goal = lean.goal_start(statement, theorem_name=canon)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {tid}: goal_start crashed ({type(exc).__name__})")
            continue
        if not goal:
            print(f"  skip {tid}: statement does not elaborate locally")
            continue
        bare, sheet = build_prompts(tid, by_id[tid], man.get(tid, {}))
        picked.append(
            {
                "theorem_id": tid,
                "statement": statement,
                "prompt_bare": bare,
                "prompt_sheet": sheet,
            }
        )
        print(f"  ok   {tid} ({len(picked)}/{args.n})")

    args.out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked))
    print(f"\n{len(picked)} validated pairs → {args.out}")


if __name__ == "__main__":
    main()
