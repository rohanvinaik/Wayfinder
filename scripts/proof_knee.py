#!/usr/bin/env python3
"""Wayfinder instance of the Semantic Completeness knee — the THIRD domain (formal proofs).

The SSL completeness equation is domain-agnostic. GSE (NLP-WSD) gave a near-uniform-coverage knee at
~3% of the curriculum. This measures the SAME structure on Wayfinder's home turf — Lean/mathlib proofs —
where the coverage relation is EXPLICIT (every proof records its premises; no attribution gap, unlike the
LintGate attempt).

Two coupled readings of the equation, both from the real LeanDojo traced corpus (`data/leandojo_mathlib.jsonl`,
per proof: `tactics`, `goal_states`, `premises`):

  DYNAMIC (within-proof "learning by replacing sorry"):  the proof is the curriculum; closing step t
    (revealing tactic t) resolves part of the goal. H(t) = remaining-resolution complexity of goal_states[t]
    (goal-term size × open-subgoal count — a PROXY for the SoM navigator's per-state branching entropy).
    C(t) = H(t) − H(t+1) = resolution a closed step adds. Knee H* = where the bulk (big-drop, structure-
    amplified steps) gives way to the linear tail (routine cleanup). Aggregated over proofs with ≥ MIN steps.

  NETWORK ("information in the lemma network"):  cover(premise p) = {theorems whose proof cites p}. This is
    the direct GSE-analog cover(v). Its degree distribution answers the contrast hypothesis: GSE coverage was
    near-uniform (Gini 0.11, sharp knee from SATURATION); math lemma-coverage is expected SCALE-FREE (a few
    workhorse lemmas cover thousands) — a heavy tail, the OTHER mechanism for the same bounded-bulk H*.

No cosine, no re-running the prover — reads the traced corpus. Composes real Wayfinder data.

Usage: python scripts/proof_knee.py [--min-steps 10] [--max-proofs 4000] [--out data/proof_knee.json]
"""
import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gini(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def _goal_complexity(state: str) -> float:
    """Remaining-resolution proxy: (#open subgoals) × (mean goal-term token size).

    Uses the goal term(s) after ⊢ (not the hypothesis context, which is ambient). A closed state → 0.
    """
    if not state or "⊢" not in state:
        return 0.0
    goals = [g.strip() for g in state.split("⊢")[1:]]  # each ⊢ opens one goal term
    if not goals:
        return 0.0
    sizes = [len(re.findall(r"\S+", g.splitlines()[0] if g else "")) for g in goals]
    return len(goals) * (sum(sizes) / len(sizes))


def _knee_curvature(cum):
    """Kneedle: normalized cum curve, index of max distance above the chord."""
    n = len(cum)
    if n < 3 or cum[-1] == 0:
        return n - 1
    y = [c / cum[-1] for c in cum]
    x = [i / (n - 1) for i in range(n)]
    return max(range(n), key=lambda i: y[i] - x[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-steps", type=int, default=10, help="min tactics/proof to have room for a knee")
    ap.add_argument("--max-proofs", type=int, default=4000, help="cap proofs scanned for the DYNAMIC curve")
    ap.add_argument("--premise-scan", type=int, default=60000, help="proofs scanned for the NETWORK distribution")
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "proof_knee.json"))
    args = ap.parse_args()

    src = ROOT / "data" / "leandojo_mathlib.jsonl"
    if not src.exists():
        print(f"missing {src}")
        return 1

    # ---- DYNAMIC: per-proof H(t) descent over the ≥MIN-step chains ----
    knee_fracs, bulk_tail, chain_lens = [], [], []
    n_dyn = 0
    # ---- NETWORK: cover(premise) = theorems citing it ----
    prem_cover = Counter()
    n_theorems = 0

    with src.open() as f:
        for i, line in enumerate(f):
            if i >= args.premise_scan:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prems = r.get("premises") or []
            if prems:
                n_theorems += 1
                for p in set(prems):
                    prem_cover[p] += 1

            if n_dyn >= args.max_proofs:
                continue
            states = r.get("goal_states") or []
            tactics = r.get("tactics") or []
            if len(tactics) < args.min_steps or len(states) < args.min_steps:
                continue
            # H trajectory: complexity before each tactic, then 0 at close.
            H = [_goal_complexity(s) for s in states] + [0.0]
            H0 = H[0]
            if H0 <= 0:
                continue
            deltas = [max(0.0, H[t] - H[t + 1]) for t in range(len(H) - 1)]
            cum = []
            acc = 0.0
            for d in deltas:
                acc += d
                cum.append(acc)
            if not cum or cum[-1] <= 0:
                continue
            n_dyn += 1
            chain_lens.append(len(deltas))
            knee = _knee_curvature(cum)
            knee_fracs.append(knee / max(1, len(deltas) - 1))
            # bulk = mean drop up to knee; tail = mean drop after
            k = max(1, knee)
            bulk = sum(deltas[:k]) / k
            tail = sum(deltas[k:]) / max(1, len(deltas) - k)
            if tail > 0:
                bulk_tail.append(bulk / tail)

    # NETWORK distribution stats
    covs = list(prem_cover.values())
    covs_sorted = sorted(covs, reverse=True)
    top_decile = covs_sorted[: max(1, len(covs) // 10)]
    top_share = sum(top_decile) / max(1, sum(covs))
    gini = _gini(covs)
    workhorses = [(p, c) for p, c in prem_cover.most_common(12)]

    res = {
        "corpus": "leandojo_mathlib",
        "dynamic": {
            "proofs_used": n_dyn, "min_steps": args.min_steps,
            "mean_chain_len": round(st.mean(chain_lens), 1) if chain_lens else 0,
            "max_chain_len": max(chain_lens) if chain_lens else 0,
            "knee_frac_mean": round(st.mean(knee_fracs), 3) if knee_fracs else None,
            "knee_frac_median": round(st.median(knee_fracs), 3) if knee_fracs else None,
            "bulk_tail_ratio_median": round(st.median(bulk_tail), 1) if bulk_tail else None,
        },
        "network": {
            "theorems": n_theorems, "distinct_premises": len(prem_cover),
            "cover_max": max(covs) if covs else 0, "cover_mean": round(st.mean(covs), 2) if covs else 0,
            "top_decile_coverage_share": round(top_share, 3), "gini": round(gini, 3),
            "workhorse_lemmas": workhorses,
        },
    }
    Path(args.out).write_text(json.dumps(res, indent=2))

    d, nw = res["dynamic"], res["network"]
    print("[Wayfinder proof knee — LeanDojo mathlib; DYNAMIC H(t) descent + NETWORK lemma coverage]\n")
    print(f"DYNAMIC (within-proof, ≥{args.min_steps} steps): {d['proofs_used']} proofs, "
          f"mean chain {d['mean_chain_len']} (max {d['max_chain_len']})")
    print(f"  knee at {100*(d['knee_frac_mean'] or 0):.0f}% of steps (median {100*(d['knee_frac_median'] or 0):.0f}%), "
          f"bulk→tail ratio median {d['bulk_tail_ratio_median']}×")
    print(f"\nNETWORK (lemma coverage over {nw['theorems']} theorems, {nw['distinct_premises']} distinct premises):")
    print(f"  cover(premise): max={nw['cover_max']} mean={nw['cover_mean']} "
          f"top-decile share={100*nw['top_decile_coverage_share']:.0f}% Gini={nw['gini']}")
    heavy = nw["top_decile_coverage_share"] > 0.5 and nw["gini"] > 0.5
    print(f"  → lemma network is {'SCALE-FREE / heavy-tailed (workhorse hubs)' if heavy else 'light-tailed'} "
          f"— {'CONTRAST' if heavy else 'similar'} to GSE's near-uniform (Gini 0.11)")
    print(f"  workhorse lemmas: {', '.join(f'{p}({c})' for p,c in nw['workhorse_lemmas'][:6])}")
    print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
