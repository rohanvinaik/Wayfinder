# Wayfinder Documentation Index

**Project:** Navigational theorem proving for Lean 4 / Mathlib
**Status:** Publishable main baseline remains **1,277/2,000 Mathlib theorems proved (63.8%)** under `EXP-058`, with validated offline residual extension to **~1,371/2,000 (68.6%)**. The original / first-order SoM has already been tuned deterministically, including `EXP-SOM-010` and `models/som_torch_v1/best.pt`. The hard benchmark `exp_som012_hard_eval_r2` is now frozen at `3440` processed theorems and has already been materialized into the second-order corpus: `386` hard residual packets, `588` compiler/startability packets, `2332` Dr. Ducky ledger packets, and `974` second-order controller packets. Dr. Ducky vNext is now implemented as a **pure symbolic proof-local execution VM** built around a `ProofShadowLedger`, proof-bearing `MathEngine` portfolio, and `ProofProjector`; the frozen stratified closure benchmark shows `69/120` honest progress and `0/120` honest closures, while the backend runtime smoke pass at [runtime_smoke_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/runtime_smoke_summary.json) shows `5/5` backend cases progressing with `4/5` backend-local closures. The project now also has the missing integrated bridge runtime: `Ducky pass 1 -> first-order proof-closing search -> post-Ducky symbolic closer layer (solve_by_elim / apply? / exact?) -> second-order packet policy/controller -> Ducky pass 2 -> post-Ducky symbolic closer layer -> rarified proof-gap packet`. That bridge lives in [hardtail_bridge.py](/Users/rohanvinaik/Projects/Wayfinder/src/hardtail_bridge.py), [second_order_controller.py](/Users/rohanvinaik/Projects/Wayfinder/src/second_order_controller.py), and [run_exp_dd015_integrated_bridge.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd015_integrated_bridge.py). The learned second-order SoM training/runtime path is now implemented through [train_second_order_som.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/train_second_order_som.py), [second_order_som_model.py](/Users/rohanvinaik/Projects/Wayfinder/src/second_order_som_model.py), and [run_exp_som013d_train_second_order_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_som013d_train_second_order_guarded.sh). That trainer now follows the adapted first-order SoM discipline: staged optimization plus PAB-style stability stopping. The full architecture is therefore wired from frozen first-order hard residuals through deterministic bridge to learned-controller rerun. The canonical final first-order benchmark collector is now `EXP-SOM-016`: a frozen random 2,000-theorem sample from the full Mathlib corpus, run with the integrated first-order stack plus Dr. Ducky, crash-stable JSONL logging, `MONITOR.txt`, trigger-probe supervision, hard-resolution packetization, and second-order packet/feature materialization for the final training stage.

---

## Program Shape

| Version | What | Status | Orchestrator |
|---------|------|--------|-------------|
| **v1** | Monolithic navigator. 6-bank ternary decoder, IDF-weighted anchors, spreading activation, best-first search. | Trained. Chaotic PAB confirmed. Frozen baseline. | `src/proof_search.py` |
| **v2A** | Original / first-order SoM. Deterministic and supervised temporal control, specialist routing, strategy arbitration, and multistep family prediction over the live proof-search stack. | Tuned deterministically and component-validated. `EXP-SOM-010` and `models/som_torch_v1/best.pt` belong here. Not by themselves a settled replacement for the `EXP-058` runtime. | `src/temporal_controller.py`, `src/strategy_arbiter.py`, `src/som_model.py`, `src/som_torch.py` |
| **v2B** | Dr. Ducky. Typed residual capsules, `ProofShadowLedger`, deterministic proof skeletons, proof-bearing symbolic engines, and a `ProofProjector` that lowers internal certificates into Lean-valid proof programs. Backend families now include in-repo `egglog_eqsat`, `rosette_proof_dsl`, `kodkod_relational`, `symbolic_rewrite_vm`, and `lean_arith` runtimes. | Implemented as the canonical symbolic residual layer; routing, backend runtime smoke, and Lean-checked progress are validated. Frozen-corpus closure lift is the active engineering target. | `src/dr_ducky.py`, `src/dr_ducky_executor.py` |
| **v2C** | Second-order SoM. Hard-run-driven orchestration over symbolic packets: compiler/startability routing, Dr. Ducky invocation, residual-stage budgeting, and post-main escalation. | Deterministic packet policy and learned-controller training/runtime are implemented. The learned trainer uses the adapted first-order SoM regime: local heads -> controller heads -> joint finetune, with PAB-style stability stopping. | `src/second_order_controller.py`, `src/second_order_som_model.py`, `scripts/build_second_order_packet_freeze.py`, `scripts/build_second_order_feature_dataset.py`, `scripts/train_second_order_som.py` |
| **v2D** | Integrated hard-tail bridge. Deterministic closure path over frozen hard residuals: theorem-faithful replay, Dr. Ducky pass 1, first-order proof-closing search, explicit symbolic closer layer (`solve_by_elim` / `apply?` / `exact?`), second-order policy/controller decision, Dr. Ducky pass 2, second symbolic closer layer, rarified proof-gap packet. | Implemented. Active experiment surface for the full architecture claim before learned second-order training. | `src/hardtail_bridge.py`, `scripts/run_exp_dd015_integrated_bridge.py` |
| **v3** | Boundary learning, guidance distillation, and energy refinement. OTP scoring, negative data, asymmetric censor, multi-lens guidance, contrastive training. Parallel runtime — does not modify v1/v2A/v2B/v2C. | Planned (Phase 7). Gated on second-order SoM + Dr. Ducky results. | `src/v3_runtime.py` (planned) |

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
- the remaining failures are localized to the learned local tactic prior, not to verification, automation, or basic proof-state control;
- ExecSelector v1 achieves 38.5% LeanAccepted on apply steps (+133% vs cosine), validating that a small learned executable selector works — the bottleneck has shifted to orchestration/triggering (EXP-049: apply gate fired 0 times on 50-theorem benchmark).
- the frozen hard-run now materializes a real second-order training corpus: `386` hard residual packets, `588` compiler/startability packets, and `974` controller packets in [summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/second_order_som/summary.json);
- Dr. Ducky has been rebuilt against that corpus into `2332` ledger packets, with `320` observed theorem-level outcome packets and `181` theorem-level progress cases, giving the second-order program a real symbolic executor surface rather than raw residual text.

---

## Canonical Documents

| Document | Authoritative for | Current state |
|----------|------------------|---------------|
| [`WAYFINDER_RESEARCH.md`](Research_Paper/WAYFINDER_RESEARCH.md) | Theory, claims, intellectual foundations. §2.1-2.10 (v1/v2), §2.11 (v3 OTP/COEC/guidance/EBM). | Complete through v3 theory. |
| [`WAYFINDER_DESIGN.md`](Research_Paper/WAYFINDER_DESIGN.md) | Engineering: interfaces, modules, invariants. Main runtime, original / first-order SoM, Dr. Ducky symbolic VM (`GoalSpecification` → `ProofShadowLedger` → `MathEngine` → `ProofProjector`), compiler/startability, and second-order SoM boundary. | Canonical architecture reference. |
| [`WAYFINDER_PLAN.md`](Research_Paper/WAYFINDER_PLAN.md) | Execution: phases, stop/go gates, hard-run program, and post-run second-order training order. | Canonical execution plan. |
| [`EXPERIMENT_RESULTS.md`](Research_Paper/EXPERIMENT_RESULTS.md) | Results ledger. Cycles 0-6 (v1/v2), Cycle 7 (v3). EXP-044→050 (apply/trigger arc). | ~60% populated. Apply/trigger arc added. |
| [`PLAN.md`](PLAN.md) | rw-first formal buildout plan. Locked `rw` contract, benchmark infra, family expansion. | Complete. rw family collapsed to cosine. |
| [`PLAN_2.md`](PLAN_2.md) | Formal local-execution buildout. Canonical data, goal creation, symbolic Dr. Ducky VM, hard-run artifact emission, and benchmark-first second-order preparation. | Active. |
| [`SoM/EXPERIMENT_SET.md`](Research_Paper/SoM/EXPERIMENT_SET.md) | Concrete SoM experiment order: original / first-order SoM results plus second-order hard-run gates. | Active. |
| [`Ducky/DR_DUCKY_RESEARCH_PAPER.md`](Research_Paper/Ducky/DR_DUCKY_RESEARCH_PAPER.md) | Dr. Ducky theory, implementation, and validation. | Active. |

The post-`EXP-SOM-012` experiment slate is now defined explicitly rather than implicitly:

- [WAYFINDER_PLAN.md](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/WAYFINDER_PLAN.md) is the canonical cross-stack sequence.
- [SoM/EXPERIMENT_SET.md](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/SoM/EXPERIMENT_SET.md) defines the second-order SoM gates and packet requirements.
- [Ducky/EXECUTION_PLAN.md](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/Ducky/EXECUTION_PLAN.md) defines the ready-to-run Dr. Ducky closure experiments.

## Path To Publication

The publication path is now explicit:

1. complete the frozen-corpus headroom and backend-family closure-lift stage (`EXP-SOM-013A`, `EXP-DD-014A/B/C/D`)
2. run the full integrated hard-tail bridge (`EXP-DD-015`) so the real architecture is measured, not just the components
   - default bridge seeding now comes from the frozen validated Ducky replay set, not raw hard-tail row order
   - `EXP-DD-015` writes `selected_theorems.json` and a startup `summary.json` before heavy model/Lean initialization so the run is observable immediately
3. identify which remaining deficits are engine-limited, projector-limited, compiler-limited, controller-limited, or true rarified-tail gaps
4. freeze second-order SoM training packets over first-order telemetry, bridge-stage traces, and both Ducky passes
5. train the second-order SoM (`EXP-SOM-013D`) and rerun the integrated bridge with a learned controller
6. run paired theorem-search comparisons against the static strong stack
7. measure whether the combined stack closes a nontrivial share of the current hard tail and compresses the remainder into a much smaller rarified-gap regime
8. lock the publication bundle: code revision, artifacts, ablations, case studies, and reproducibility scripts

## Conditions For Hard-Tail Closure

The research program is now explicitly designed to test and create the conditions needed to close the remaining hard theorems. Those conditions are:

1. **Strong first-order baseline**
   - already satisfied by `EXP-058` and the tuned first-order SoM program
   - this establishes the `~64-69%` solved regime and generates structured hard residuals
2. **Frozen second-order data geometry**
   - already satisfied by the frozen hard-run packet surfaces
   - this gives the project a stable symbolic corpus instead of ad hoc residual text
3. **Executable local symbolic progress**
   - partially satisfied by Dr. Ducky
   - current evidence: honest progress on the frozen hard slice is real, but closure is still weak
4. **Backend-family closure lift**
   - this is the next active experiment target
   - `EXP-DD-014A/B/C/D` are the component experiments that test whether equality saturation, proof-DSL transport, bounded relational search, and the integrated symbolic VM can convert progress into closure on theorem-faithful replay
5. **Integrated theorem-level bridge**
   - `EXP-DD-015` is the first experiment that actually wires the whole closure path together:
     - theorem-faithful replay
     - Dr. Ducky pass 1
     - first-order proof-producing search with `solve_by_elim` / `apply?` / `exact?` / interleaved bootstrap / automation
     - second-order controller decision
     - Dr. Ducky pass 2
     - rarified proof-gap analysis
6. **Second-order orchestration over symbolic packets**
   - this is the learned step after backend-family and bridge behavior are stable
   - the second-order SoM is supposed to learn invocation, budgeting, escalation, and post-search Ducky2 targeting over real bridge outcomes
7. **Paired theorem-search lift**
   - the final claim is not a component claim
   - it is a theorem-search comparison across:
     - static strong baseline
     - baseline + Dr. Ducky
     - baseline + Dr. Ducky + second-order SoM

If these six conditions are met, the project has created the right environment to test whether a nontrivial share of the current hard tail can actually be closed.

For unattended post-freeze Dr. Ducky runs, the canonical launcher is now
[scripts/run_dr_ducky_overnight_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_dr_ducky_overnight_guarded.sh),
which adds row timeouts, periodic Pantograph restarts, streamed partial summaries, and validation-safe suppression of unstable arithmetic solvers.

For the second-order SoM freeze stage, the canonical packet builder is now
[scripts/build_second_order_packet_freeze.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/build_second_order_packet_freeze.py),
which consolidates hard-resolution packets, compiler packets, Dr. Ducky ledger packets, and observed Ducky outcomes into
`bundle/second_order_som/`.

The next canonical second-order preparation step is now
[scripts/build_second_order_feature_dataset.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/build_second_order_feature_dataset.py),
which converts the frozen second-order packets into trainable dense arrays plus multi-target supervision for controller training.

The canonical learned second-order training launcher is now
[scripts/run_exp_som013d_train_second_order_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_som013d_train_second_order_guarded.sh),
which consumes the frozen feature surface and emits a checkpoint, metadata snapshot, and training summary in a sibling run directory.

For guarded post-freeze headroom runs, the canonical launcher is now
[scripts/run_exp_som013a_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_som013a_guarded.sh),
which wraps planner/local depth ladders and the oracle-gap audit into one bounded post-freeze pass.
Its default output path is a **sibling experiment run** under `runs/`, not a nested child of the frozen source run.

For guarded backend-family Dr. Ducky runs, the canonical launchers are now:
- [scripts/run_exp_dd014a_eqsat_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014a_eqsat_guarded.sh)
- [scripts/run_exp_dd014b_proof_dsl_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014b_proof_dsl_guarded.sh)
- [scripts/run_exp_dd014c_relational_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014c_relational_guarded.sh)
- [scripts/run_exp_dd014d_integrated_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014d_integrated_guarded.sh)

They all preflight the frozen `EXP-SOM-012` bundle before launch and write to sibling run directories under `runs/`.

For the full hard-tail bridge experiment, the canonical launcher is now:
- [scripts/run_exp_dd015_integrated_bridge_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd015_integrated_bridge_guarded.sh)

It consumes the frozen `EXP-SOM-012` bundle, the second-order packet freeze, the Phase-1 proof-producing runtime, and emits:
- theorem-level bridge rows
- controller decisions
- rarified proof-gap packets
- theorem-level closure/progress summaries

If `SECOND_ORDER_MODEL` and `SECOND_ORDER_METADATA` are supplied, the same launcher reruns the bridge with the learned second-order controller while keeping the rest of the closure path fixed.

The canonical end-to-end post-freeze protocol runner is now
[scripts/run_postfreeze_closure_protocol.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_postfreeze_closure_protocol.sh),
which sequences feature regeneration, second-order training, deterministic bridge evaluation, and learned-controller bridge rerun from a frozen source run.

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
4. v1, v2A, v2B, and v2C code paths are frozen once v3 development begins.
5. Censor safety net: never prune ALL candidates.
6. Negative labels: semantic vs infra separation is mandatory.
7. Theorem-search reports must separate `raw_success` from `honest_success`; self-application is telemetry, not a headline solve.
8. Dr. Ducky runtime is pure symbolic. Any future neural residual must be documented as non-canonical and out-of-band.
9. `EXP-SOM-016` is the canonical final first-order benchmark collector. It must freeze a random 2,000-theorem Mathlib sample and emit crash-stable theorem rows, trigger states, hard-resolution packets, and second-order packet/features from the same run.
