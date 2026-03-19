# Wayfinder Documentation Index

**Project:** Navigational theorem proving for Lean 4 / Mathlib
**Status:** v1 trained (NAV-001/002). v2 SoM implemented (Phase 6.0). v3 planned (Phase 7).

---

## Program Shape

| Version | What | Status | Orchestrator |
|---------|------|--------|-------------|
| **v1** | Monolithic navigator. 6-bank ternary decoder, IDF-weighted anchors, spreading activation, best-first search. | Trained. Chaotic PAB confirmed. Frozen baseline. | `src/proof_search.py` |
| **v2** | Society of Mind. 5-slot temporal pipeline (Perception→Recognition→Planning→Execution→Verification). Theorem-level temporal orchestration, specialist decomposition, sketch prediction, local residual execution, and a stateful temporal controller in the Arbiter. | Implemented (Phase 6.0). Temporal-controller module exists; runtime wiring and eval pending. | `src/arbiter.py` |
| **v3** | Boundary learning, guidance distillation, and energy refinement. OTP scoring, negative data, asymmetric censor, multi-lens guidance, contrastive training. Parallel runtime — does not modify v1/v2. | Planned (Phase 7). Gated on v2 ≥ v1. | `src/v3_runtime.py` (planned) |

v3 is split by maturity:
- **v3A (committed)**: Negative data, censor, OTP scoring reforms, active boundary learning, guidance-layer distillation over collapsed retrieval frontiers.
- **v3B (experimental, gated on v3A)**: Energy function, continuous ternary relaxation, sketch refinement.

## Value Proposition

Wayfinder is not trying to replace cluster-scale end-to-end math models. The practical target is different:

- **Make the routine majority cheap**: discharge the easy and medium proof obligations with a small trainable stack, symbolic search structure, and exact Lean verification on commodity hardware.
- **Compress the hard residual**: when a theorem is not solved outright, the system should still collapse the search space into a much smaller, better-conditioned frontier rather than leaving a larger model to reason over "the entirety of math."

Operationally, that means:

- solved proofs become essentially cheap local search problems;
- unsolved proofs still produce useful structure: lane attribution, constrained candidate sets, failed branch evidence, and residual diagnostics;
- larger models or stronger solvers can then spend compute only on the hard tail, with a much better "vibe" of the solution space than raw theorem text alone.

Current evidence for this value proposition:

- the end-to-end Pantograph path is operational;
- the Init logic smoke benchmark reached `27/30` proved (`90%`) with clear lane separation;
- post-structural residual execution is learnable with a tiny specialist, and oracle premises improve family prediction (`28.0% → 32.2%` top-1; `73.8% → 77.4%` top-3);
- the remaining failures are localized to the learned local tactic prior, not to verification, automation, or basic proof-state control.

---

## Canonical Documents

| Document | Authoritative for | Current state |
|----------|------------------|---------------|
| [`WAYFINDER_RESEARCH.md`](WAYFINDER_RESEARCH.md) | Theory, claims, intellectual foundations. §2.1-2.10 (v1/v2), §2.11 (v3 OTP/COEC/guidance/EBM). | Complete through v3 theory. |
| [`WAYFINDER_DESIGN.md`](WAYFINDER_DESIGN.md) | Engineering: interfaces, modules, invariants. §1-9 (v1), §10 (v2 SoM), §12 (v3 runtime). | Complete through v3 architecture. |
| [`WAYFINDER_PLAN.md`](WAYFINDER_PLAN.md) | Execution: phases, stop/go gates, ablation matrix. Phases 0-5 (v1), 6 (v2), 7 (v3). | Complete. Phase 7 uses v3A/v3B wave structure. |
| [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) | Results ledger. Cycles 0-6 (v1/v2), Cycle 7 (v3). Hypothesis tracker (H1-H38). | ~40% populated. Cycle 7 templates added. |

## Reference Documents (not canonical architecture)

| Document | Purpose |
|----------|---------|
| [`MODEL_SELECTION.md`](MODEL_SELECTION.md) | Decision log for encoder selection (Phase 0.6). 15 models evaluated. |
| [`SPEC_COVERAGE_PLAN.md`](SPEC_COVERAGE_PLAN.md) | LintGate-side reference for test coverage analysis. |

---

## Quick Start

```bash
# Enhanced controller data pipeline
./scripts/run_enhanced_controller_pipeline.sh

# Full main experiment path: data -> template classifier -> navigator
./scripts/run_main_experiment.sh

# Template classifier with planning/recognition move supervision
python -m scripts.train_template_classifier --config configs/wayfinder.yaml --run-id TC-AUX-001

# Navigator with descriptive move supervision only
python scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-AUX-001

# Theorem-search benchmark with additive rewrite executor
python -m scripts.run_benchmark \
  --config configs/wayfinder.yaml \
  --checkpoint models/NAV-AUX-001_step5000.pt \
  --mode v1 \
  --search-mode full \
  --cosine-rw \
  --cosine-rw-seq
```

## Key Invariants

1. `data/nav_eval.jsonl` is frozen. Never modify.
2. Neural inference happens once per proof state.
3. All retrieval scores are auditable (trace to anchors + banks).
4. v1 and v2 code paths are frozen once v3 development begins.
5. Censor safety net: never prune ALL candidates.
6. Negative labels: semantic vs infra separation is mandatory.
7. Primary metric is always `raw_success` (Lane A only).
