#!/usr/bin/env python3
"""Wayfinder DYNAMIC proof knee with the REAL trained SoM navigator (not a proxy).

The goal-state-SIZE proxy failed (Lean states are non-monotone under branching tactics). The
domain-principled H is the trained navigator's per-state BRANCHING ENTROPY — which Wayfinder computes.
This loads the real NAV-001 checkpoint (encoder → analyzer → bridge → navigator) and, for each goal
state along a proof, reads the navigator's action distribution:

  H_dir(t)   = Σ over the 6 ternary direction heads of the softmax entropy   (how many ways forward)
  branch(t)  = # anchors (premises) the navigator scores viable (sigmoid > τ)  (lemma-network branching)
  progress(t)= the navigator's own [0,1] progress estimate                     (its "how done" signal)

As a proof closes, a well-trained navigator gets CERTAIN → H_dir, branch descend. H(t) is that descent;
C(t) = H(t) − H(t+1) is the resolution a closed step (a replaced `sorry`) adds. Knee H* = where the
bulk (big-drop, structure-amplified steps) gives way to the routine tail. This is the within-proof
"learning by replacing sorry" curve, measured by the model itself — no cosine, no size proxy.

Usage: python scripts/proof_knee_som.py [--min-steps 20] [--max-proofs 200] [--device mps]
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
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--max-proofs", type=int, default=200)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--tau", type=float, default=0.5, help="anchor viability threshold")
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "proof_knee_som.json"))
    args = ap.parse_args()

    import torch
    import yaml

    from src.nav_model_factory import load_navigational_checkpoint

    config = yaml.safe_load((ROOT / "configs" / "wayfinder.yaml").read_text())
    ckpt = ROOT / "models" / "NAV-001_step5000.pt"
    print(f"[loading SoM navigator {ckpt.name} on {args.device}]")
    _, mods = load_navigational_checkpoint(ckpt, config, args.device)
    enc, analyzer, bridge, nav = mods["encoder"], mods["analyzer"], mods["bridge"], mods["navigator"]
    for m in (analyzer, bridge, nav):
        if hasattr(m, "eval"):
            m.eval()

    def state_branching(goal_state: str):
        """(H_dir, viable_premise_count, progress) from the navigator at this goal state."""
        emb = enc.encode([goal_state])
        feats, _, _ = analyzer(emb)
        bout = bridge(feats)
        with torch.no_grad():
            dir_logits, anchor_logits, progress, _critic = nav.forward(bout)
        Hdir = 0.0
        for _bank, lg in dir_logits.items():
            p = torch.softmax(lg[0], dim=-1)
            Hdir += float(-(p * torch.log(p + 1e-9)).sum())
        ap = torch.sigmoid(anchor_logits[0])
        branch = int((ap > args.tau).sum().item())
        return Hdir, branch, float(progress.reshape(-1)[0].item())

    src = ROOT / "data" / "leandojo_mathlib.jsonl"
    knee_fracs, bulk_tail, chain_lens = [], [], []
    hdir_first_last, branch_first_last = [], []
    used = 0
    with src.open() as f:
        for line in f:
            if used >= args.max_proofs:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            states = r.get("goal_states") or []
            if len(states) < args.min_steps:
                continue
            # H(t) via the navigator's branching entropy along the real proof states, + a closed 0.
            traj = [state_branching(s) for s in states]
            H = [hd for hd, _b, _p in traj] + [0.0]           # direction-entropy descent
            B = [b for _hd, b, _p in traj]                     # premise-branching
            deltas = [max(0.0, H[t] - H[t + 1]) for t in range(len(H) - 1)]
            cum, acc = [], 0.0
            for d in deltas:
                acc += d
                cum.append(acc)
            if not cum or cum[-1] <= 0:
                continue
            used += 1
            chain_lens.append(len(deltas))
            knee = _knee_curvature(cum)
            knee_fracs.append(knee / max(1, len(deltas) - 1))
            k = max(1, knee)
            bulk = sum(deltas[:k]) / k
            tail = sum(deltas[k:]) / max(1, len(deltas) - k)
            if tail > 0:
                bulk_tail.append(bulk / tail)
            hdir_first_last.append((H[0], H[-2]))
            branch_first_last.append((B[0], B[-1]))
            if used % 25 == 0:
                print(f"  {used}/{args.max_proofs} proofs scored ...")

    res = {
        "corpus": "leandojo_mathlib", "model": "NAV-001_step5000",
        "proofs_used": used, "min_steps": args.min_steps,
        "mean_chain_len": round(st.mean(chain_lens), 1) if chain_lens else 0,
        "max_chain_len": max(chain_lens) if chain_lens else 0,
        "knee_frac_mean": round(st.mean(knee_fracs), 3) if knee_fracs else None,
        "knee_frac_median": round(st.median(knee_fracs), 3) if knee_fracs else None,
        "bulk_tail_ratio_median": round(st.median(bulk_tail), 2) if bulk_tail else None,
        "Hdir_start_mean": round(st.mean([a for a, _ in hdir_first_last]), 2) if hdir_first_last else None,
        "Hdir_end_mean": round(st.mean([b for _, b in hdir_first_last]), 2) if hdir_first_last else None,
        "branch_start_mean": round(st.mean([a for a, _ in branch_first_last]), 1) if branch_first_last else None,
        "branch_end_mean": round(st.mean([b for _, b in branch_first_last]), 1) if branch_first_last else None,
    }
    Path(args.out).write_text(json.dumps(res, indent=2))

    print("\n[Wayfinder DYNAMIC knee — REAL SoM navigator branching entropy]\n")
    print(f"proofs ≥{args.min_steps} steps: {used}, mean chain {res['mean_chain_len']} (max {res['max_chain_len']})")
    print(f"  navigator direction-entropy: start {res['Hdir_start_mean']} → end {res['Hdir_end_mean']} "
          f"({'DESCENDS (navigator resolves)' if (res['Hdir_end_mean'] or 9) < (res['Hdir_start_mean'] or 0) else 'flat/rises'})")
    print(f"  viable premises (branching): start {res['branch_start_mean']} → end {res['branch_end_mean']}")
    print(f"  KNEE at {100*(res['knee_frac_mean'] or 0):.0f}% of steps (median {100*(res['knee_frac_median'] or 0):.0f}%), "
          f"bulk→tail ratio median {res['bulk_tail_ratio_median']}×")
    print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
