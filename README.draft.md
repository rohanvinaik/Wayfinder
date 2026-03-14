# Wayfinder

**Proof search as navigation, not prediction.**

[![CI](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)

DeepSeek threw 671 billion parameters at theorem proving. I asked: what if proof space has structure you can navigate?

Wayfinder is a neural theorem prover for Lean 4 / Mathlib that runs the neural network *once* per proof state. All subsequent search — premise retrieval, tactic resolution, proof tree expansion — is deterministic arithmetic on a precomputed semantic network. Same paradigm that navigates [800K ML models](https://github.com/rohanvinaik/ModelAtlas), now applied to 100K Mathlib lemmas.

---

## The six banks

Proof entities sit along six orthogonal signed dimensions. Zero is the mode — the most common mathematical situation.

| Bank | Negative | Zero | Positive |
|------|----------|------|----------|
| **STRUCTURE** | decidable, arithmetic | equality / rewrite | quantified, dependent |
| **DOMAIN** | concrete (N, Z, Q) | general algebra | abstract (topology, category) |
| **DEPTH** | trivial (1 tactic) | 2-3 tactic proof | deep (10+ tactics) |
| **AUTOMATION** | fully auto (omega, simp) | partially automated | manual reasoning |
| **CONTEXT** | no hypotheses | moderate (3-5 hyps) | rich (10+ hyps) |
| **DECOMPOSITION** | atomic (no splits) | single goal | multi-subgoal |

A ternary vector like `(-1, +1, 0, -1, 0, +1)` means: "decidable lemma, abstract domain, typical depth, automation-friendly, typical context, needs subgoal decomposition." The proof network resolves this to a ranked list of matching premises. 3^6 = 729 direction bins. Enough to navigate; few enough to learn.

## How it works

```
Proof state → [Encoder] → Continuous embedding
                               ↓
                      [Information Bottleneck]
                               ↓
                      [Ternary Decoder] → 6-bank direction {-1, 0, +1}^6
                               ↓
               [Proof Network: SQLite semantic network]
                     ↓              ↓              ↓
             Bank alignment   Anchor IDF    Spreading activation
                     ↓              ↓              ↓
               [Multiplicative composition] → Ranked premises + tactics
```

The encoder handles the hard part — understanding the goal state. The proof network handles the structured part — finding relevant mathematics. They don't compete for the same compute. One neural forward pass, then arithmetic.

**Scoring:** `bank_alignment × anchor_relevance × spreading_activation`. Multiplicative — a premise that matches structurally but lives in the wrong domain scores zero, not fifty percent.

**Three-lane verification:** Pantograph (step-wise tactic search), Axle (proof repair), lean4checker (high-assurance). Each lane catches what the others miss.

## What this is not

- **Not an LLM wrapper.** No autoregressive generation, no token-level prediction, no prompt engineering.
- **Not embedding retrieval.** Premises are found by structured coordinate matching and IDF-weighted anchor overlap, not dense vector similarity.
- **Not a standalone prover.** Wayfinder produces tactic suggestions and premise rankings. Lean's kernel verifies. The system delegates `AUTOMATION=-1` goals to hammer tactics.

## Status

**Early experimental.** 29 source modules, 14 scripts, 88% test coverage. Encoder selection complete (15 models evaluated). First training runs pass all phase gates — peak navigational accuracy 72% across six banks. Proof network construction and benchmark evaluation are next.

Baselines: ReProver (26% MiniF2F), DeepSeek-Prover-V2 (88.9% MiniF2F, 671B parameters). Wayfinder targets >= 20% MiniF2F with a single forward pass per proof state — competitive with ReProver at a fraction of the inference cost, not with DeepSeek's scale. The honest bet: navigational search matches dense retrieval. The moonshot: it exceeds it on efficiency-normalized metrics.

Full experiment ledger: [EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)

## Getting started

```bash
pip install -e .

# Build proof network from Mathlib
python scripts/extract_proof_network.py --config configs/wayfinder.yaml

# Train the navigator
python scripts/train_navigator.py --config configs/wayfinder.yaml

# Evaluate
python scripts/run_benchmark.py --config configs/wayfinder.yaml
```

Requires Python >= 3.11, PyTorch >= 2.1, and a LeanDojo-compatible Lean 4 + Mathlib installation.

## Documentation

Full docs at **[rohanvinaik.github.io/Wayfinder](https://rohanvinaik.github.io/Wayfinder/)**.

| Document | What it covers |
|----------|---------------|
| [Research](https://rohanvinaik.github.io/Wayfinder/abstract/) | Theory, claims, related work, evaluation design |
| [Design](https://rohanvinaik.github.io/Wayfinder/1-design-thesis/) | Banks, anchors, scoring, module architecture |
| [Experiment Plan](https://rohanvinaik.github.io/Wayfinder/plan-overview/) | Phases 0-5, stop/go criteria, hardware distribution |
| [Results](docs/EXPERIMENT_RESULTS.md) | Experiment ledger |

---

Part of a research program on structured navigation through constrained semantic spaces — the same paradigm applied to [ML model discovery](https://github.com/rohanvinaik/ModelAtlas) and [code quality supervision](https://github.com/rohanvinaik/LintGate).

*Rohan Vinaik, with Claude (Opus 4.6). March 2026.*
