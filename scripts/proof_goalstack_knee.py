#!/usr/bin/env python3
"""Phase 2 of the Wayfinder DYNAMIC proof knee — the STRUCTURAL goal-stack collapse.

The navigator's uncertainty is the PAC / statistical pole (excluded here). The STRUCTURAL H is the
open-goal stack: how many subgoals (unresolved `sorry`s) remain. Phase 1 (`run_benchmark` full search)
solved the self-contained minif2f competition theorems and recorded each `tactics_used`. This phase
replays each solved tactic sequence through the live Pantograph kernel and logs the FULL open-goal
count per step (Pantograph exposes all goals, not just the focused one) — the real "resolve open
uncertainty each epoch, track completion to H→0" trajectory.

  H(t)      = # open goals after applying tactic t          (structural uncertainty remaining)
  closed(t) = max(0, H(t-1) - H(t))                          (subgoals resolved this epoch = amplification)
  knee H*   = where cluster-closers (a lemma/tactic that discharges many goals) give way to the
              one-at-a-time tail. Bulk = closes-per-step early; tail = late.

Honest: many competition proofs close in ONE powerful step (nlinarith/omega/decide) — a [1,0] stack,
= a degenerate "all bulk, no tail". That is itself a finding (competition math is often one-shot), and
is reported separately from the genuinely multi-step (branching) proofs where a knee is well-defined.

Usage: python scripts/proof_goalstack_knee.py [--phase1 data/minif2f_goalstack_phase1.json]
                                               [--manifest data/minif2f_tractable_rerun.jsonl]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _knee_curvature(cum):
    n = len(cum)
    if n < 3 or cum[-1] == 0:
        return n - 1
    y = [c / cum[-1] for c in cum]
    x = [i / (n - 1) for i in range(n)]
    return max(range(n), key=lambda i: y[i] - x[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default=str(ROOT / "data" / "minif2f_goalstack_phase1.json"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "minif2f_tractable_rerun.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "proof_goalstack_knee.json"))
    ap.add_argument("--per-proof-out", default=str(ROOT / "data" / "proof_goalstack_trajectories.jsonl"))
    args = ap.parse_args()

    import logging
    logging.disable(logging.CRITICAL)
    from src.lean_interface import LeanConfig, LeanKernel

    manifest = {}
    for line in Path(args.manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            manifest[r["theorem_id"]] = r

    p1 = json.loads(Path(args.phase1).read_text())
    details = p1.get("details") or p1.get("results") or []
    solved = [d for d in details if d.get("success") and (d.get("tactics_used"))]
    print(f"Phase-1: {len(details)} attempted, {len(solved)} solved with a tactic trace to replay.\n")

    cfg = LeanConfig(backend="pantograph", hammer_timeout=60,
                     project_root=str(ROOT / "data" / "lean_project"), imports=["Mathlib"])
    lean = LeanKernel(cfg)
    lean._ensure_server()
    srv = lean._server

    def replay_stack(row, tactics):
        opens = row.get("opens") or []
        pre = "\n".join(str(o) for o in opens) + "\n" if opens else ""
        decl = str(row["full_declaration"])
        try:
            targets = srv.load_sorry(f"{pre}{decl}")
        except Exception as e:
            return None, f"load-fail: {str(e)[:40]}"
        if not targets or not targets[0].goal_state.goals:
            return None, "no-initial-goal"
        stt = targets[0].goal_state
        traj = [len(stt.goals)]
        for t in tactics:
            try:
                stt = srv.goal_tactic(stt, tactic=t)
            except Exception as e:
                return traj, f"tac-fail@{len(traj)}: {str(e)[:30]}"
            traj.append(len(stt.goals))
            if not stt.goals:
                return traj, "CLOSED"
        return traj, ("CLOSED" if not stt.goals else "incomplete")

    per = open(args.per_proof_out, "w")
    knee_fracs, bulk_tail, chain_lens, maxstacks = [], [], [], []
    one_shot = multi = replay_ok = 0
    for d in solved:
        tid = d["theorem_id"]
        row = manifest.get(tid)
        if not row:
            continue
        traj, status = replay_stack(row, d["tactics_used"])
        if not traj or len(traj) < 2:
            continue
        replay_ok += 1
        closed = [max(0, traj[i] - traj[i + 1]) for i in range(len(traj) - 1)]
        total_closed = sum(closed)
        maxstacks.append(max(traj))
        per.write(json.dumps({"theorem_id": tid, "status": status, "stack": traj,
                              "closed_per_step": closed, "n_tactics": len(d["tactics_used"])}) + "\n")
        # one-shot = the whole obligation discharged in a single step (stack goes to 0 at step 1)
        if traj[1] == 0 or len([c for c in closed if c > 0]) <= 1:
            one_shot += 1
            continue
        if max(traj) <= 1 and total_closed <= 1:
            one_shot += 1
            continue
        multi += 1
        cum, acc = [], 0
        for c in closed:
            acc += c
            cum.append(acc)
        if cum and cum[-1] > 0:
            knee = _knee_curvature(cum)
            knee_fracs.append(knee / max(1, len(closed) - 1))
            k = max(1, knee)
            bulk = sum(closed[:k]) / k
            tail = sum(closed[k:]) / max(1, len(closed) - k)
            if tail > 0:
                bulk_tail.append(bulk / tail)
            chain_lens.append(len(closed))
    per.close()

    res = {
        "solved_replayed": replay_ok,
        "one_shot_proofs": one_shot, "multi_step_proofs": multi,
        "one_shot_fraction": round(one_shot / max(1, replay_ok), 3),
        "max_stack_mean": round(st.mean(maxstacks), 2) if maxstacks else None,
        "max_stack_max": max(maxstacks) if maxstacks else None,
        "multi_step_knee_frac_mean": round(st.mean(knee_fracs), 3) if knee_fracs else None,
        "multi_step_knee_frac_median": round(st.median(knee_fracs), 3) if knee_fracs else None,
        "multi_step_bulk_tail_median": round(st.median(bulk_tail), 2) if bulk_tail else None,
        "multi_step_mean_chain": round(st.mean(chain_lens), 1) if chain_lens else None,
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[Wayfinder goal-stack knee — {replay_ok} replayed proofs]\n")
    print(f"  ONE-SHOT (discharged in a single closing step): {one_shot}/{replay_ok} "
          f"({100*res['one_shot_fraction']:.0f}%)  — competition math is often one-hammer")
    print(f"  MULTI-STEP (branching, a knee is well-defined): {multi}")
    print(f"  open-goal stack: mean-max {res['max_stack_mean']}, max {res['max_stack_max']}")
    if knee_fracs:
        print(f"  MULTI-STEP knee at {100*(res['multi_step_knee_frac_mean'] or 0):.0f}% of steps "
              f"(median {100*(res['multi_step_knee_frac_median'] or 0):.0f}%), "
              f"bulk→tail median {res['multi_step_bulk_tail_median']}×, mean chain {res['multi_step_mean_chain']}")
    print(f"\n  per-proof trajectories → {args.per_proof_out}\n  summary → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
