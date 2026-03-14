# Wayfinder Documentation Index

**Project:** Navigational theorem proving for Lean 4 / Mathlib
**Status:** v1 trained (NAV-001/002). v2 SoM implemented (Phase 6.0). v3 planned (Phase 7).

---

## Program Shape

| Version | What | Status | Orchestrator |
|---------|------|--------|-------------|
| **v1** | Monolithic navigator. 6-bank ternary decoder, IDF-weighted anchors, spreading activation, best-first search. | Trained. Chaotic PAB confirmed. Frozen baseline. | `src/proof_search.py` |
| **v2** | Society of Mind. 5-slot temporal pipeline (Perception→Recognition→Planning→Execution→Verification). Template classification, specialist decomposition, sketch prediction. | Implemented (Phase 6.0). Training pending (6.1-6.5). | `src/arbiter.py` |
| **v3** | Boundary learning + energy refinement. OTP scoring, negative data, asymmetric censor, contrastive training. Parallel runtime — does not modify v1/v2. | Planned (Phase 7). Gated on v2 ≥ v1. | `src/v3_runtime.py` (planned) |

v3 is split by maturity:
- **v3A (committed)**: Negative data, censor, OTP scoring reforms, active boundary learning.
- **v3B (experimental, gated on v3A)**: Energy function, continuous ternary relaxation, sketch refinement.

---

## Canonical Documents

| Document | Authoritative for | Current state |
|----------|------------------|---------------|
| [`WAYFINDER_RESEARCH.md`](WAYFINDER_RESEARCH.md) | Theory, claims, intellectual foundations. §2.1-2.10 (v1/v2), §2.11 (v3 OTP/EBM/COEC). | Complete through v3 theory. |
| [`WAYFINDER_DESIGN.md`](WAYFINDER_DESIGN.md) | Engineering: interfaces, modules, invariants. §1-9 (v1), §10 (v2 SoM), §12 (v3 runtime). | Complete through v3 architecture. |
| [`WAYFINDER_PLAN.md`](WAYFINDER_PLAN.md) | Execution: phases, stop/go gates, ablation matrix. Phases 0-5 (v1), 6 (v2), 7 (v3). | Complete. Phase 7 uses v3A/v3B wave structure. |
| [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) | Results ledger. Cycles 0-6 (v1/v2), Cycle 7 (v3). Hypothesis tracker (H1-H34). | ~40% populated. Cycle 7 templates added. |

## Reference Documents (not canonical architecture)

| Document | Purpose |
|----------|---------|
| [`MODEL_SELECTION.md`](MODEL_SELECTION.md) | Decision log for encoder selection (Phase 0.6). 15 models evaluated. |
| [`SPEC_COVERAGE_PLAN.md`](SPEC_COVERAGE_PLAN.md) | LintGate-side reference for test coverage analysis. |

---

## Quick Start

```bash
# v1 training
python -m scripts.train_navigator --config configs/wayfinder.yaml

# v1 benchmark
python -m scripts.run_benchmark --config configs/wayfinder.yaml --mode v1 --checkpoint runs/<run-id>/checkpoint.pt

# v2 benchmark (after Phase 6 training)
python -m scripts.run_benchmark --config configs/wayfinder_v2.yaml --mode v2 --checkpoint runs/<som-run>/checkpoint.pt

# v3 benchmark (after Phase 7, planned)
python -m scripts.run_benchmark --config configs/wayfinder_v3.yaml --mode v3 --checkpoint runs/<v3-run>/checkpoint.pt
```

## Key Invariants

1. `data/nav_eval.jsonl` is frozen. Never modify.
2. Neural inference happens once per proof state.
3. All retrieval scores are auditable (trace to anchors + banks).
4. v1 and v2 code paths are frozen once v3 development begins.
5. Censor safety net: never prune ALL candidates.
6. Negative labels: semantic vs infra separation is mandatory.
7. Primary metric is always `raw_success` (Lane A only).
