# Dr. Ducky vNext Execution Plan

## Goal

Make Dr. Ducky the canonical deterministic 1.5th-order executor for local proof residue:

- no LLM runtime,
- no free-form tactic generation,
- no raw residual-text learning below the compiler boundary.

The runtime contract is:

1. build a typed `GoalSpecification`
2. seed a `ProofShadowLedger`
3. derive `BankPrior` and `GoalPrescription`
4. synthesize `ProofSkeleton`s
5. run proof-bearing `MathEngine`s
6. lower `EngineCertificate`s through a `ProofProjector`
7. verify every `ProjectedProofProgram` in Lean

This is the proof-search analogue of the LintGate specification/prescription/skeleton pipeline.

## Canonical Object Model

Implemented or partially implemented in
[dr_ducky.py](/Users/rohanvinaik/Projects/Wayfinder/src/dr_ducky.py) and
[dr_ducky_executor.py](/Users/rohanvinaik/Projects/Wayfinder/src/dr_ducky_executor.py):

- `GoalSpecification`
- `BankPrior`
- `GoalPrescription`
- `ProofHoleSpec`
- `ProofSkeleton`
- `ProofShadowLedger`
- `LedgerFact`
- `MathEngineRequest`
- `EngineCertificate`
- `ProjectedProofProgram`
- `ProjectorDecision`
- `GoalCapsule`

## Runtime Architecture

### 1. Capsule Construction

Every hard residual is converted into a self-contained `GoalCapsule` containing:

- structural signals
- residual geometry
- search-pathology geometry
- engine eligibility
- projector markers
- suppression hints
- bank priors
- prescriptions
- proof skeletons
- seeded ledger facts
- execution budgets

### 2. ProofShadowLedger

The ledger is the internal symbolic state of the executor. It stores:

- normalized local facts
- accessible premises
- candidate witness terms
- candidate rewrite facts
- engine outcomes
- rejected branches with provenance

The ledger is intentionally not a user-facing artifact. It is the proof-local VM state that powers deterministic execution.

### 3. MathEngine Portfolio

The current vNext architecture recognizes these engine families:

- `EqSatEngine`
- `ArithEngine`
- `WitnessEngine`
- `RecursiveInvariantEngine`
- `FiniteFilterEngine`
- `ContextTransportEngine`

Canonical backend mapping:

- `EqSatEngine` -> `egglog_eqsat`
- `ArithEngine` -> `lean_arith`
- `WitnessEngine` -> `rosette_proof_dsl`
- `RecursiveInvariantEngine` -> `symbolic_rewrite_vm`
- `FiniteFilterEngine` -> `kodkod_relational`
- `ContextTransportEngine` -> `rosette_proof_dsl`

Current runtime status:

- `EqSatEngine` now executes an in-repo bounded rewrite-saturation runtime and emits extracted rewrite/calc certificates.
- `ArithEngine` now emits arithmetic normalization and solver-oriented certificates through the symbolic runtime, not just generic tactic templates.
- `WitnessEngine` now performs bounded witness search over local terms and emits exact witness constructions when a supporting proof fact exists.
- `RecursiveInvariantEngine` now runs a bounded symbolic rewrite VM over recursive surfaces instead of only acting as a routing marker.
- `FiniteFilterEngine` now performs bounded relational local search for membership / structural closure cases.
- `ContextTransportEngine` now performs proof-DSL-style local fact transport, including derived fact replay and intrinsic iff bridges.

Default bank-to-engine mapping:

- `eq_sat`, `transport_normalizer` -> `EqSatEngine`
- `arith_nf`, `solver_dispatch` -> `ArithEngine`
- `witness_constructor`, `canonical_witness` -> `WitnessEngine`
- `recursive_unfold_one`, `loop_breaker` -> `RecursiveInvariantEngine`
- `membership_exposure`, `set_pointwise`, `eventual_filter_normalizer`, `structural_close` -> `FiniteFilterEngine`
- `context_forward`, `local_fact_selector`, `binder_instantiation`, `iff_splitter`, `diagram_transport` -> `ContextTransportEngine`

### 4. ProofProjector

The projector is the architecture boundary mandated by the data-geometry design:

- internal symbolic state and certificates stay inside the ledger/engine layer
- the only scorer-facing/runtime-facing output is a Lean-valid `ProjectedProofProgram`

Current projector lowering modes:

- rewrite-chain lowering
- `have`-chain synthesis
- `calc`-style lowering
- direct proof-term / exact close
- witness construction
- local fact transport

### 5. Lean Verifier

Every projected program is executed in Lean under theorem-faithful replay when available.

Accepted programs are separated into:

- compiled but no-op
- honest progress
- honest closure

Accepted programs that leave the theorem-faithful goal list unchanged are rejected as no-op.

## Artifact Contract

The current architecture requires these postrun artifacts:

- `dr_ducky_capsules.jsonl`
- `dr_ducky_ledger_packets.jsonl`
- `dr_ducky_engine_outcomes.jsonl`
- `dr_ducky_projector_outcomes.jsonl`
- `dr_ducky_closure_report.json`

The capsule/runtime contract now also carries:

- `proof_dsl_programs`
- `solver_constraints`
- `eqsat_plan`
- `relational_search_specs`
- `backend_preferences`

And the hard-resolution / SoM packets must surface:

- `ducky_specialist_target`
- `engine_family`
- `certificate_shape`
- `projector_status`
- `compile_status`
- `progress_status`
- `closure_status`
- `negative_geometry`

## Rollout Order

### Phase 1. Docs and Contracts

Freeze terminology and packet schemas before extending the runtime.

Status: in progress and partially implemented.

### Phase 2. Ledger + Projector Scaffolding

Add the typed symbolic VM state and explicit projector boundary.

Status: implemented.

Current evidence:

- `ProofShadowLedger` and `LedgerFact` are live
- projector/certificate metadata is emitted by the executor
- hard-resolution packets now expose Ducky surface geometry

### Phase 3. EqSat + Context Transport

First fully working end-to-end symbolic lane:

- equality-heavy near misses
- arithmetic side conditions
- local fact transport via `mp` / `mpr` / symmetry / projections

Status: implemented as a real in-repo symbolic runtime and smoke-tested.

### Phase 4. Recursive Invariant + Witness Paths

Cover:

- UnionFind/infinite-mirror style recursion
- existential and membership witness paths

Status: implemented in baseline form; frozen-corpus closure lift remains to be measured.

### Phase 5. Finite / Filter / Structural Property Path

Cover:

- `Finite`
- `Filter`
- eventuality / structural-property residue

Status: implemented in baseline bounded-relational form; frozen-corpus closure strength remains the key experiment.

### Phase 6. Closure Benchmark Pass

Run theorem-faithful replay slices and report:

- faithful start rate
- certificate generation rate
- projector success rate
- compile rate
- honest progress rate
- honest closure rate
- closure lift by residual family

Status: ready to run on the frozen corpus. Pre-freeze smoke evidence is now available at [runtime_smoke_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/runtime_smoke_summary.json).

### Phase 7. Second-order SoM Data Integration

Once the hard benchmark freezes:

1. materialize Ducky symbolic artifacts
2. join them with first-order SoM telemetry and hard-resolution packets
3. run the integrated bridge so those packets include theorem-level handoff behavior
4. train the second-order SoM on those symbolic packets

This is the architectural handoff point: the second-order SoM learns when and how to invoke Dr. Ducky, when to hand off to the core first-order proof-closing runtime, and when a second Ducky pass plus rarified-gap packet is the right final action.

## Ready-To-Run Post-Freeze Experiments

Assume:

```bash
RUN_DIR=runs/exp_som012_hard_eval_r2_frozen
DUCKY_DIR=$RUN_DIR/bundle/dr_ducky
```

### `EXP-DD-013A`: Capsule + Ledger Freeze

**Run:**

```bash
python -m scripts.build_dr_ducky_worklist \
  --run-dir "$RUN_DIR" \
  --output-dir "$DUCKY_DIR"
```

**Primary outputs:**

- `dr_ducky_capsules.jsonl`
- `dr_ducky_ledger_packets.jsonl`
- `summary.json`

**Gate:** capsule counts, ledger counts, and summary counts reconcile against the frozen run.

### `EXP-DD-013B`: Closure Benchmark

Sampling policy:
- `executor_validation_stratified120` is a guarded round-robin sample across the four local residual buckets after the targeted theorem set is injected.
- targeted rows are capped by `--limit`; they do not silently overrun the requested slice size.

**Run:**

```bash
python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 20 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --output-json "$DUCKY_DIR/executor_validation_local20_vnext_summary.json" \
  --output-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_rows.jsonl" \
  --engine-outcomes-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_projector_outcomes.jsonl" \
  --closure-report-json "$DUCKY_DIR/executor_validation_local20_vnext_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 120 \
  --row-timeout-seconds 180 \
  --restart-every 12 \
  --disable-tactic linarith \
  --disable-tactic nlinarith \
  --output-json "$DUCKY_DIR/executor_validation_stratified120_summary.json" \
  --output-jsonl "$DUCKY_DIR/executor_validation_stratified120_rows.jsonl" \
  --engine-outcomes-jsonl "$DUCKY_DIR/executor_validation_stratified120_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$DUCKY_DIR/executor_validation_stratified120_projector_outcomes.jsonl" \
  --closure-report-json "$DUCKY_DIR/executor_validation_stratified120_closure_report.json"
```

For unattended post-freeze runs, use the checked-in guarded launcher:

```bash
caffeinate -dimsu /Users/rohanvinaik/Projects/Wayfinder/scripts/run_dr_ducky_overnight_guarded.sh \
  > /Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/overnight_closure_sweep_guarded.log 2>&1
```

The guarded launcher is the canonical execution path because it adds:

- row-level timeouts for Pantograph-heavy theorem starts
- periodic Pantograph restarts
- incremental `summary.json` / `closure_report.json` writes during execution
- validation-safe suppression of `linarith` / `nlinarith`, which currently create noisy low-value arithmetic failures on frozen replay slices

**Primary metrics:**

- theorem-faithful starts
- certificate generation count
- projector success / rejection
- Lean compile proxy count
- honest progress
- honest closure

**Gate:** theorem-faithful replay is stable and honest progress remains non-zero on both slices.

### `EXP-DD-013C`: Residual-Bucket Ablation

**Run:** repeat `scripts.run_dr_ducky_executor_validation` with one bucket at a time:

- `single_goal_near_miss`
- `single_goal_stall`
- `multi_goal_small_progress`
- `multi_goal_large_progress`

Use the same guarded flags as `EXP-DD-013B` for every ablation pass.

**Outputs:** one closure report family per residual bucket.

**Gate:** the per-bucket summaries are discriminative enough to tell whether the next closure lift should come from:

- stronger certificates,
- stronger projector lowering,
- stronger local fact transport,
- or second-order orchestration.

## Backend Implementation Track

These are the next concrete backend experiments after the frozen `013` slate.

### `EXP-DD-014A`: egglog EqSat backend

**Question:** Does a real egg/egglog-style equality-saturation backend improve equality-heavy closure?

**Canonical launcher:** [run_exp_dd014a_eqsat_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014a_eqsat_guarded.sh)

**Readouts:**

- `summary.json -> by_backend_preference.egglog_eqsat`
- executor `by_backend_family.egglog_eqsat`
- closure lift on equality-heavy replay slices

### `EXP-DD-014B`: Rosette-style proof DSL backend

**Question:** Does a solver-aided proof DSL improve local fact transport, witness construction, and binder drilldown?

**Canonical launcher:** [run_exp_dd014b_proof_dsl_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014b_proof_dsl_guarded.sh)

**Readouts:**

- `proof_dsl_programs`
- `solver_constraints`
- executor `by_backend_family.rosette_proof_dsl`
- honest progress and closure on theorem-faithful replay

### `EXP-DD-014C`: Kodkod-style bounded relational backend

**Question:** Does bounded relational search improve membership, `Finite`, and filter/eventuality closure?

**Canonical launcher:** [run_exp_dd014c_relational_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014c_relational_guarded.sh)

**Readouts:**

- `relational_search_specs`
- executor `by_backend_family.kodkod_relational`
- closure lift on membership / witness / structural-property slices

### `EXP-DD-014D`: Integrated symbolic closing suite

**Question:** What is the closure lift of the full symbolic Dr. Ducky VM once the egglog, Rosette-style DSL, and Kodkod-style bounded relational backends are all present?

**Canonical launcher:** [run_exp_dd014d_integrated_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd014d_integrated_guarded.sh)

**Ablations:**

- current Ducky
- `+ egglog_eqsat`
- `+ rosette_proof_dsl`
- `+ kodkod_relational`
- full integrated symbolic VM

**Current pre-freeze smoke evidence:** `5/5` backend cases progress and `4/5` close in [runtime_smoke_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/runtime_smoke_summary.json). The remaining task is to convert that into honest frozen-corpus closure lift.

**Architectural note:** these backend runs are component tests, not the final theorem-closure claim. Dr. Ducky is supposed to contract the residual into a form the core first-order proof-producing architecture can finish. Final theorem-closure evidence therefore comes from the later paired theorem-search experiments, not from Ducky-only replay in isolation.

### `EXP-DD-015`: Integrated bridge and rarified-tail pass

**Question:** After Dr. Ducky contracts a theorem-faithful hard residual, can the core first-order proof-closing architecture plus the explicit symbolic closer layer (`solve_by_elim` / `apply?` / `exact?`) finish the theorem, and if not, does a second Ducky pass after second-order control plus the same symbolic closer layer produce a rarified proof-gap packet that makes the remaining tail significantly more tractable?

**Canonical launcher:** [run_exp_dd015_integrated_bridge_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_dd015_integrated_bridge_guarded.sh)

**Runtime path:**

1. theorem-faithful replay
2. Dr. Ducky pass 1
3. first-order proof-closing search
4. post-Ducky symbolic closer layer (`solve_by_elim` / `apply?` / `exact?`)
5. second-order packet policy / controller refinement
6. Dr. Ducky pass 2
7. post-Ducky symbolic closer layer (`solve_by_elim` / `apply?` / `exact?`)
8. rarified proof-gap packet emission

**Primary outputs:**

- `rows.jsonl`
- `controller_decisions.jsonl`
- `rarified_gap_packets.jsonl`
- `summary.json`
- `selected_theorems.json`

**Execution guardrails:**

- default theorem selection should come from the frozen validated Ducky replay set (`executor_validation_stratified120_rows.jsonl`) and prioritize theorem-faithful started/progressed rows
- single-goal tractable rows should be front-loaded ahead of multigoal recursive/pathological rows
- `selected_theorems.json` and startup `summary.json` should be written before heavy encoder/model/Lean initialization

## Conditions Required For Ducky To Contribute To Hard-Tail Closure

Dr. Ducky is not meant to be an analysis sidecar. She is meant to be the deterministic local symbolic executor that converts first-order residual progress into theorem-level closure. For that to happen, the following conditions must be created and tested:

1. **The residual must be theorem-faithfully replayable**
   - already strong on the frozen benchmark slices
2. **The symbolic engines must produce proof-bearing certificates**
   - already true at the progress level
   - still weak at the closure level
3. **The projector must lower those certificates into Lean-valid proof programs**
   - already strong enough for honest progress measurement
   - not yet strong enough for broad closure
4. **The integrated bridge must hand Ducky contraction back into the actual proof-producing runtime**
   - this is the point of `EXP-DD-015`
5. **A second Ducky pass must produce a rarified final-gap packet when theorem closure still fails**
   - this turns the last tail into a tractable study/specialist regime rather than an undifferentiated hard bucket
6. **Ducky outcomes must be packetized for controller training**
   - already satisfied structurally by the frozen second-order packet surfaces and the new bridge outputs

If these six conditions are met, Dr. Ducky becomes a real closure engine rather than only a progress engine.

## From Here To Publication

The remaining path from the current frozen state to a publication-ready Dr. Ducky subsystem is:

1. complete the frozen-corpus closure measurements already in flight:
   - `EXP-DD-013B`
   - `EXP-DD-013C`
2. run `EXP-DD-014A/B/C/D` to isolate backend-family lift
3. run `EXP-DD-015` so Ducky contraction, first-order closure, second-order policy, and the second Ducky pass are all measured together
4. promote only backend/runtime combinations and bridge behaviors that show honest theorem-level lift or durable rarification of the tail
5. patch the weak closure families using the backend/bridge evidence, not by ad hoc tactic expansion
6. freeze Ducky packet schemas and outcome formats for second-order SoM training
7. join Ducky executor outcomes, bridge decisions, and rarified-gap packets with first-order SoM control traces and compiler/startability packets
8. train and benchmark the second-order SoM on that frozen symbolic corpus
9. run final paired theorem-search experiments with:
   - static strong baseline
   - static strong baseline + Dr. Ducky
   - static strong baseline + Dr. Ducky + second-order SoM
-10. lock the publication bundle:
   - frozen artifact bundle
   - ablation tables
   - case studies
   - reproducibility commands

## Validation Requirements

Dr. Ducky is not “done” until the repo reports these separately:

- routing validation
- theorem-faithful replay validation
- certificate generation
- projector compilation
- honest progress lift
- honest closure lift

Required benchmark slices:

- targeted stress slice
- local 20-row slice
- larger stratified residual slice from `EXP-SOM-012`
- by-family slices for equality, inequality, membership, witness, recursive invariant, filter/finiteness, and small multigoal

## Current Bottleneck

The remaining gap is not routing. It is proof-bearing local closure:

- better fact-graph saturation
- stronger domain micro-theories
- stronger certificate extraction
- stronger projector lowering for structural bridges

That is the correct next engineering target before second-order controller training.
