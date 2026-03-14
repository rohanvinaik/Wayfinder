# Wayfinder

[![CI](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml)
[![Mutation Testing](https://github.com/rohanvinaik/Wayfinder/actions/workflows/mutation.yml/badge.svg)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/mutation.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wayfinder&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wayfinder)
[![Security Lite](https://github.com/rohanvinaik/Wayfinder/actions/workflows/security-lite.yml/badge.svg)](https://github.com/rohanvinaik/Wayfinder/actions/workflows/security-lite.yml)

**Proof search as navigation, not prediction.**

Wayfinder is a neural theorem prover for Lean 4 / Mathlib that runs the neural network *once* per proof state. All subsequent search — premise retrieval, tactic resolution, proof tree expansion — is deterministic arithmetic on a precomputed semantic network.

Current neural provers (ReProver, DeepSeek-Prover-V2, HTPS) run a multi-billion parameter model at every search node. Wayfinder inverts this: a lightweight encoder produces ternary coordinates `{-1, 0, +1}` across six navigational dimensions, and a structured semantic network resolves those coordinates to concrete tactics and premises via set intersection, IDF weighting, and spreading activation. No embeddings are queried at search time.

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

The neural network handles the hard part (understanding the goal). Search handles the structured part (finding relevant mathematics). They don't compete for the same compute.

## The six banks

Proof entities are positioned along six orthogonal signed dimensions. Zero sits at the mode — the most common mathematical situation.

| Bank | Negative | Zero | Positive |
|------|----------|------|----------|
| **STRUCTURE** | decidable, arithmetic | equality / rewrite | quantified, dependent |
| **DOMAIN** | concrete (N, Z, Q) | general algebra | abstract (topology, category) |
| **DEPTH** | trivial (1 tactic) | 2-3 tactic proof | deep (10+ tactics) |
| **AUTOMATION** | fully auto (omega, simp) | partially automated | manual reasoning |
| **CONTEXT** | no hypotheses | moderate (3-5 hyps) | rich (10+ hyps) |
| **DECOMPOSITION** | atomic (no splits) | single goal | multi-subgoal |

3^6 = 729 direction bins. A ternary vector like `(-1, +1, 0, -1, 0, +1)` means: "look for decidable lemmas in abstract domains, at typical depth, solvable by automation, with typical context, requiring subgoal decomposition." The proof network resolves this to a ranked list of matching premises.

## What this is not

- **Not an LLM wrapper.** No autoregressive generation, no token-level prediction, no prompt engineering.
- **Not embedding retrieval.** Premises are found by structured coordinate matching and IDF-weighted anchor overlap, not dense vector similarity.
- **Not a standalone prover.** Wayfinder produces tactic suggestions and premise rankings. Lean's kernel verifies. The system delegates `AUTOMATION=-1` goals to hammer tactics (LeanHammer, Aesop).

## Project structure

```
src/                        # 29 modules
  encoder.py                # Dual-backend goal encoder (SentenceTransformer / T5)
  ternary_decoder.py        # Continuous → ternary {-1, 0, +1} bottleneck
  proof_network.py          # SQLite semantic network (banks, anchors, IDF)
  resolution.py             # Coordinate → premise/tactic resolution
  proof_search.py           # Navigational best-first search
  bridge.py                 # Continuous-ternary bridge with proof history
  proof_navigator.py        # End-to-end: goal → ranked actions
  lean_interface.py         # Lean 4 interaction (Pantograph / Axle / lean4checker)
  trainer.py                # Training loop with PAB tracking
  pab_*.py                  # Process-Aware Benchmarking metrics and profiling
  verification.py           # 3-lane verification (Pantograph, Axle, lean4checker)
  ...

scripts/                    # Extraction, training, evaluation
  extract_proof_network.py  # Mathlib → proof_network.db
  build_nav_training_data.py# Proof traces → nav_training.jsonl
  anchor_gap_analysis.py    # Iterative anchor refinement (target: top-16 recall >= 70%)
  train_navigator.py        # Train encoder + ternary decoder
  run_benchmark.py          # MiniF2F / Mathlib evaluation

configs/
  wayfinder.yaml            # All hyperparameters and paths

docs/
  WAYFINDER_RESEARCH.md     # Theory and claims
  WAYFINDER_DESIGN.md       # Engineering specification
  WAYFINDER_PLAN.md         # Experiment phases 0-5
  EXPERIMENT_RESULTS.md     # Results ledger
```

## Getting started

```bash
pip install -e .

# Phase 0: Build the proof network from Mathlib
python scripts/extract_proof_network.py --config configs/wayfinder.yaml

# Phase 0: Generate navigation training data
python scripts/build_nav_training_data.py --config configs/wayfinder.yaml

# Phase 0: Anchor gap analysis (iterate until recall >= 70%)
python scripts/anchor_gap_analysis.py --config configs/wayfinder.yaml

# Phase 1: Train the navigator
python scripts/train_navigator.py --config configs/wayfinder.yaml

# Evaluate
python scripts/run_benchmark.py --config configs/wayfinder.yaml
```

Requires Python >= 3.11, PyTorch >= 2.1, and a LeanDojo-compatible Lean 4 + Mathlib installation.

## Documentation

Full documentation at **[rohanvinaik.github.io/Wayfinder](https://rohanvinaik.github.io/Wayfinder/)** — structured reading rails across research theory, engineering design, experiment plan, and results.

| Document | What it covers |
|----------|---------------|
| [Research](https://rohanvinaik.github.io/Wayfinder/abstract/) | Theory, claims, related work, evaluation design |
| [Design](https://rohanvinaik.github.io/Wayfinder/1-design-thesis/) | Banks, anchors, scoring, module architecture, config |
| [Experiment Plan](https://rohanvinaik.github.io/Wayfinder/plan-overview/) | Phases 0-5, stop/go criteria, hardware distribution |
| [Results](docs/EXPERIMENT_RESULTS.md) | Experiment ledger (pre-experimental) |

## Lineage

Wayfinder integrates three prior systems:

1. **ModelAtlas** (Vinaik, 2025) — Structured semantic navigation for ML model discovery. Wayfinder adapts the bank/anchor/IDF architecture from models to mathematical entities.
2. **Balanced Sashimi** (Vinaik & Claude, 2026) — Hybrid continuous-ternary architecture. Wayfinder uses the `{-1, 0, +1}` information bottleneck for navigational coordinates.
3. **Mutation Theory** (formalized in Lean 4) — Convergence guarantees for structured search: trajectory monotonicity, phase transitions, fixed-point partitions.

## Status

**v1.1 — Implementation complete, pre-experimental.** All source modules implemented and import-chain verified. Phase 0 (proof network construction from Mathlib) is the next execution step.

Baselines: ReProver (LeanDojo), LeanProgress, DeepSeek-Prover-V2.

---

*Rohan Vinaik, with Claude (Opus 4.6), Anthropic. March 2026.*
