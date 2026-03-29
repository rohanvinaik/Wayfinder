# Dr_Ducky

This folder contains the Dr_Ducky research and implementation bundle.

Canonical location:
- `/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/Ducky`

Current validation scope:
- routing, prescription, and residual-family alignment over live benchmark artifacts
- theorem-faithful replay of benchmark residuals into Lean
- `ProofShadowLedger` seeding from row surfaces and cached goal state
- proof-bearing engine/program generation through the symbolic `MathEngine` interface
- in-repo production backends for equality saturation, proof-DSL transport/witness search, bounded relational search, recursive rewrite control, and arithmetic normalization
- `ProofProjector` lowering into Lean-valid tactic programs
- honest progress validation on targeted and local residual slices
- closure lift is still limited; theorem-recovery validation is not yet strong

Validation is now split into six categories:
- routing validation
- theorem-faithful replay validation
- certificate generation
- projector compilation
- honest progress lift
- honest closure lift

Architecture role:
- Dr. Ducky is the deterministic **1.5th-order** executor in the full Wayfinder stack.
- She sits between the first-order theorem-search and first-order SoM control stack, and the second-order SoM controller.
- In the full bridge architecture she appears twice:
  - `Ducky pass 1` contracts the initial hard residual before the core proof-closing stack runs
  - the contracted frontier is then handed to the cheap pre-existing Lean symbolic closer layer (`solve_by_elim` / `apply?` / `exact?`) inside the first-order proof-producing runtime
  - `Ducky pass 2` runs after second-order control on whatever remains
  - the same symbolic closer layer runs again before any rarified proof-gap packet is emitted
- Canonical runtime shape:
  - `GoalSpecification`
  - `ProofShadowLedger`
  - `BankPrior` / `GoalPrescription`
  - `ProofSkeleton`
  - `MathEngine`
  - `EngineCertificate`
  - `ProofProjector`
  - Lean verifier / replay loop
- Canonical backend families:
  - `egglog_eqsat`
  - `rosette_proof_dsl`
  - `kodkod_relational`

This is a deliberate design choice, not a fallback accident. Wayfinder is supposed to do the expensive decomposition work once, then use standard symbolic systems liberally on the simplified local proof state. In the current Lean 4 / Mathlib runtime the operational symbolic closer layer is `solve_by_elim`, `apply?`, and `exact?`.
  - `symbolic_rewrite_vm`
- The second-order SoM should learn when and how to invoke Dr. Ducky from symbolic benchmark packets, not replace her deterministic execution path.

Ready-to-run post-freeze experiments:
- [EXECUTION_PLAN.md](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/Ducky/EXECUTION_PLAN.md) now carries the concrete `EXP-DD-013A` / `EXP-DD-013B` / `EXP-DD-013C` freeze slate, the `EXP-DD-014A/B/C/D` backend-runtime benchmarks, the integrated bridge `EXP-DD-015`, and the path into learned second-order training and final paired theorem-search evaluation.

Primary paper:
- [DR_DUCKY_RESEARCH_PAPER.md](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/Ducky/DR_DUCKY_RESEARCH_PAPER.md)

Implementation entry points:
- [src/dr_ducky.py](/Users/rohanvinaik/Projects/Wayfinder/src/dr_ducky.py)
- [scripts/build_dr_ducky_worklist.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/build_dr_ducky_worklist.py)
- [src/dr_ducky_executor.py](/Users/rohanvinaik/Projects/Wayfinder/src/dr_ducky_executor.py)
- [scripts/run_dr_ducky_executor_validation.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_dr_ducky_executor_validation.py)
- [src/proof_search.py](/Users/rohanvinaik/Projects/Wayfinder/src/proof_search.py)

Validation artifacts from the live hard run:
- [audit.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/audit.json)
- [summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/summary.json)
- [dr_ducky_capsules.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/dr_ducky_capsules.jsonl)
- [dr_ducky_ledger_packets.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/dr_ducky_ledger_packets.jsonl)
- [validation.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/validation.json)
- [runtime_smoke_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/runtime_smoke_summary.json)
- [dr_ducky_engine_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/dr_ducky_engine_outcomes.jsonl)
- [dr_ducky_projector_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/dr_ducky_projector_outcomes.jsonl)
- [dr_ducky_closure_report.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/dr_ducky_closure_report.json)
- [executor_validation_local20_vnext_rows.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_rows.jsonl)
- [executor_validation_local20_vnext_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_summary.json)
- [executor_validation_local20_vnext_engine_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_engine_outcomes.jsonl)
- [executor_validation_local20_vnext_projector_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_projector_outcomes.jsonl)
- [executor_validation_local20_vnext_closure_report.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_closure_report.json)
- [executor_validation_targeted_v6.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6.json)
- [executor_validation_targeted_v6_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6_summary.json)

Tests:
- [test_dr_ducky.py](/Users/rohanvinaik/Projects/Wayfinder/tests/test_dr_ducky.py)
- [test_build_dr_ducky_worklist.py](/Users/rohanvinaik/Projects/Wayfinder/tests/test_build_dr_ducky_worklist.py)
- [test_audit_hard_run.py](/Users/rohanvinaik/Projects/Wayfinder/tests/test_audit_hard_run.py)
- [test_dr_ducky_executor.py](/Users/rohanvinaik/Projects/Wayfinder/tests/test_dr_ducky_executor.py)
