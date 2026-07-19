# Wayfinder

[![CI](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml)
[![Mutation Score](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rohanvinaik/52acfed02cbb67c88482b37074809fc2/raw/wayfinder-mutation-badge.json)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/mutation.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)

**Proof search as navigation, not prediction.**

`1,277 / 2,000 Mathlib theorems (63.8%) · one neural pass per state · 22M-param frozen encoder · runs on a laptop · Lean-verified`

Every current neural theorem prover runs a multi-billion-parameter model at *every node* of the search — thousands of forward passes to prove one lemma. Wayfinder runs the network **once** per proof state. That single pass turns the goal into six ternary coordinates — `{-1, 0, +1}` on six orthogonal axes — and from there it is pure arithmetic: a precomputed semantic network of Mathlib resolves those coordinates to concrete premises and tactics by set intersection and IDF-weighted overlap, with **no embedding queried at search time.** The neural net does the one genuinely hard part — understand the goal — and deterministic search does the structured part — find the relevant mathematics. They never compete for the same compute.

On a frozen 2,000-theorem Mathlib benchmark it proves **1,277 (63.8%)**, extending to ~68.6% with offline residual work, on a **22M-parameter frozen encoder** at 617 goals/sec on an Apple laptop — and Lean's kernel checks every proof.

## The part that matters most: it compresses the hard residual

Wayfinder is not trying to replace cluster-scale end-to-end math models. The trade is different, and better. A normal prover that fails hands the theorem back as raw text, and a bigger model then has to reason over *the entirety of mathematics.* Wayfinder does not fail flat. Even an unsolved theorem comes out **structured** — which lane it belongs to, a constrained candidate set, the branches that already failed, residual diagnostics — so the hard tail is no longer "all of math." It is a small, better-conditioned frontier, and a stronger model or a human works from a partial solution state with an actual sense of the answer space instead of a cold goal.

- **Make the routine majority cheap** — discharge the easy and medium obligations with a small trainable stack, symbolic search, and exact Lean verification, on commodity hardware.
- **Compress the hard minority** — collapse what's left into a narrowed, legible region, so a larger solver spends compute only where it is actually needed.

Solved proofs become cheap local search; unsolved proofs still produce usable structure. The claim is not "beat the giants" — it is "spend their compute only on the tail, and hand them a map of it."

## How it works

```
Proof state → [Encoder] → Continuous embedding
                              ↓
                     [Information Bottleneck]
                              ↓
                     [Ternary Decoder] → 6-bank direction vector {-1, 0, +1}^6
                              ↓
              [Proof Network: SQLite semantic network]
                    ↓              ↓              ↓
            Bank alignment   Anchor IDF    Spreading activation
                    ↓              ↓              ↓
              [Multiplicative composition] → Ranked premises + tactics
```

### The six banks

Proof entities are positioned along six orthogonal signed dimensions. Zero sits at the mode — the most common mathematical situation.

| Bank | Negative | Zero | Positive |
|------|----------|------|----------|
| **STRUCTURE** | decidable, arithmetic | equality / rewrite | quantified, dependent |
| **DOMAIN** | concrete (ℕ, ℤ, ℚ) | general algebra | abstract (topology, category) |
| **DEPTH** | trivial (1 tactic) | 2-3 tactic proof | deep (10+ tactics) |
| **AUTOMATION** | fully auto (omega, simp) | partially automated | manual reasoning |
| **CONTEXT** | no hypotheses | moderate (3-5 hyps) | rich (10+ hyps) |
| **DECOMPOSITION** | atomic (no splits) | single goal | multi-subgoal |

3⁶ = 729 direction bins. A vector like `(-1, +1, 0, -1, 0, +1)` reads: *decidable lemmas in abstract domains, at typical depth, solvable by automation, typical context, needs subgoal decomposition.* The proof network resolves it to a ranked list of matching premises — every score auditable, traceable to the banks and anchors that produced it.

## Dr. Ducky — the hard closes, symbolically

The theorems the navigator cannot discharge get a proof-local execution engine of their own: **Dr. Ducky**, a *pure symbolic* VM (a `ProofShadowLedger`, a portfolio of proof-bearing math engines — equality saturation, a proof-DSL, bounded relational search, an arithmetic runtime — and a `ProofProjector` that lowers its certificates into Lean-valid proof programs). No neural network runs in that path, by invariant, so every close it produces is auditable down to the kernel. Honest progress on the frozen hard slice is real; converting progress into closure is the active engineering frontier.

## What this is not

- **Not an LLM wrapper.** No autoregressive generation, no token-level prediction, no prompt engineering.
- **Not embedding retrieval.** Premises are found by structured coordinate matching and IDF-weighted anchor overlap, not dense vector similarity.
- **Not a standalone prover.** Wayfinder produces tactic suggestions and premise rankings; Lean's kernel verifies, and `AUTOMATION = -1` goals are delegated to hammer tactics (LeanHammer, Aesop).
- **Not a headline-inflator.** Reports separate `raw_success` from `honest_success` by invariant — self-application is telemetry, never counted as a solve.

## Getting started

```bash
pip install -e .

# Build the proof network from Mathlib, then train the navigator
python scripts/extract_proof_network.py --config configs/wayfinder.yaml
python scripts/build_nav_training_data.py --config configs/wayfinder.yaml
python scripts/train_navigator.py         --config configs/wayfinder.yaml

# Theorem-search benchmark
python -m scripts.run_benchmark --config configs/wayfinder.yaml \
  --checkpoint models/NAV-AUX-001_step5000.pt --mode v1 --search-mode full --cosine-rw
```

Requires Python ≥ 3.11, PyTorch ≥ 2.1, and a LeanDojo-compatible Lean 4 + Mathlib installation. Full documentation at **[rohanvinaik.github.io/Wayfinder](https://rohanvinaik.github.io/Wayfinder/)**.

## Lineage

Wayfinder integrates three prior systems:

1. **[ModelAtlas](https://github.com/rohanvinaik/ModelAtlas)** — structured semantic navigation for ML-model discovery. Wayfinder adapts the bank / anchor / IDF architecture from models to mathematical entities.
2. **Balanced Sashimi** — the hybrid continuous-ternary architecture; Wayfinder uses the `{-1, 0, +1}` information bottleneck for navigational coordinates.
3. **Mutation Theory** (formalized in Lean 4) — convergence guarantees for structured search: trajectory monotonicity, phase transitions, fixed-point partitions.

## Status

**63.8% Mathlib baseline (`EXP-058`), architecture wired end to end.** The first-order navigator is the frozen, tuned baseline; the symbolic hard-tail stack (Dr. Ducky, the second-order controller, the integrated bridge) is built and under active measurement, aimed at closing a nontrivial share of the remaining tail and compressing the rest. Baselines: ReProver (LeanDojo), DeepSeek-Prover-V2, HTPS.

---

*Rohan Vinaik, with Claude, Anthropic.*
