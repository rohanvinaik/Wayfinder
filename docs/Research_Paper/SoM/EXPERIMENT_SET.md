# Wayfinder SoM Experiment Set

**Version:** 1.3  
**Date:** 2026-03-25  
**Scope:** Concrete experiment slate for the full SoM program: the tuned original / first-order SoM plus the later second-order Society of Mind architecture built on top of the strong theorem-search baseline and the deterministic Dr. Ducky residual executor.

## 0. Current Architecture State

The current project state is:

1. **Main theorem-search stack**
   - `EXP-058` remains the publishable theorem-search baseline.
   - validated 2-step residual search remains the strongest cheap post-main extension.
2. **Original / first-order SoM**
   - deterministically tuned control substrate over the main solver
   - includes `EXP-SOM-010`, `models/som_torch_v1/best.pt`, temporal-control, and strategy-arbiter work
   - real component evidence and runtime-adjacent infrastructure, but not the same thing as the later second-order controller
3. **Dr. Ducky**
   - implemented deterministic 1.5th-order residual executor
   - consumes local residual buckets and executes typed symbolic proof programs in Lean
4. **Second-order SoM**
   - not yet the canonical theorem-search runtime
   - current hard benchmark has generated the residual/gap corpus that will train it
5. **Integrated hard-tail bridge**
   - now implemented in deterministic form
   - runs `Ducky1 -> first-order closure (+ explicit symbolic closer layer: solve_by_elim / apply? / exact?) -> second-order policy/controller -> Ducky2 -> post-Ducky symbolic closer layer -> rarified gap`
   - this is the architecture that must be measured before the learned second-order controller claim is made
6. **Learned second-order SoM training/runtime**
   - now implemented as a guarded training path over frozen packet features
   - uses the adapted first-order SoM protocol:
     - local heads
     - controller heads
     - low-LR joint finetune
     - PAB-style stability stopping
   - reruns the integrated bridge with the learned controller while keeping the rest of the closure architecture fixed

Live hard-run snapshot (`exp_som012_hard_eval_r2`, 2026-03-25):
- `2907` rows surfaced in the latest Ducky validation snapshot
- `475` skipped starts
- `1253` `single_goal_near_miss`
- `323` `multi_goal_small_progress`

Latest Dr. Ducky snapshot on that run:
- `1975` local residual capsules
- `90.38%` recursive circuit-breaker routing on targeted validation
- `4/5` honest progress on targeted executor replay
- `10/20` honest progress on a theorem-faithful local 20-row replay slice
- closure lift remains weak

So the operative order is:
- preserve the original / first-order SoM as the current control substrate and feature source,
- finish the hard benchmark,
- materialize the symbolic residual and Dr. Ducky artifacts,
- then train and benchmark the second-order SoM on those artifacts.

The concrete post-freeze experiment slate is now:

- `EXP-SOM-016`: formal final first-order random Mathlib benchmark + full packet collection
- `EXP-DD-013A`: frozen hard-corpus materialization
- `EXP-DD-013B`: Dr. Ducky closure benchmark
- `EXP-DD-013C`: Dr. Ducky residual-bucket ablation
- `EXP-SOM-013A`: frozen hard-bucket headroom audit
- `EXP-SOM-013B`: second-order packet freeze and launch gate
- `EXP-SOM-013D`: learned second-order training
- `EXP-DD-015`: integrated hard-tail bridge

These are not optional side analyses. They are the canonical bridge from `EXP-SOM-012` into second-order SoM training.

`EXP-SOM-016` is the other canonical input to the same training program: it is the formal final first-order benchmark on a frozen random 2,000-theorem Mathlib sample, and it must emit:

- theorem-level rows (`details.jsonl`)
- goal-start failure rows
- trigger/selector supervision rows
- hard-resolution packets
- second-order packet freeze
- second-order feature arrays

That benchmark is therefore both:

- the final first-order headline measurement
- the largest clean training corpus for the final second-order SoM pass

The canonical learned-controller training launcher is:
- [run_exp_som013d_train_second_order_guarded.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_exp_som013d_train_second_order_guarded.sh)

The canonical end-to-end post-freeze protocol runner is:
- [run_postfreeze_closure_protocol.sh](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_postfreeze_closure_protocol.sh)

## 1. Current Baseline

The project is no longer at the stage of asking whether decomposition helps at all. That question is already answered.

Current theorem-search baselines:

| Run | Proved | Rate | Proved\|started | Time |
|---|---:|---:|---:|---:|
| No `exact?` / pre-finisher stack ([`exp033_test2000_v2`](../../../runs/exp033_test2000_v2/summary.json)) | 557/2000 | 27.9% | 31.7% | 6320s |
| `+ exact?` ([`exp055_exact_q_2000`](../../../runs/exp055_exact_q_2000/summary.json)) | 1113/2000 | 55.7% | 63.3% | 8040s |
| `+ exact? + shape + induction + micro` ([`exp058_decisive_2000`](../../../runs/exp058_decisive_2000/summary.json)) | 1277/2000 | 63.8% | 72.6% | 4008s |

Implication:

- The strongest current system is a **dominance-ordered finisher stack**.
- The next SoM question is not "add another always-on lane."
- The next SoM question is:
  - can an Arbiter route between specialists better than the fixed strong stack,
  - can symbolic memory help that routing,
  - can teacher-distilled planning improve over typed-state-only control,
  - can residual specialists recover the post-`EXP-058` remainder without regressing the dominant finisher regime.

## 2. SoM Thesis To Test

The current SoM program should test four claims, in order:

1. **Dominance-aware orchestration:** a typed Arbiter can outperform or sparsify the fixed strong finisher stack.
2. **Symbolic memory helps control:** k-line-like `strategy_memory` provides useful priors before any learned controller is trusted.
3. **Compact typed controllers are enough:** runtime SoM control should be handled by small typed models over symbolic packets, not by large raw-Lean models.
4. **Residual specialists belong on the residual:** `apply`, `refine`, and startability repair should be invoked where the dominant finisher stack fails, not globally.

## 3. Hard Design Rules

These rules govern every SoM experiment in this file.

1. The comparison baseline is now `EXP-058`, not the old `cosine_rw` or `+apply` baselines.
2. Every orchestration run must emit typed temporal packets and route traces, not only theorem-level outcomes.
3. Large story/reasoning models are **offline teachers only** and operate on symbolicized packets, not raw Lean traces.
4. `apply` is a residual specialist until it shows positive theorem-level value on the post-`EXP-058` residual.
5. A SoM experiment is not complete unless it reports:
   - theorem-level metrics,
   - process/routing metrics,
   - cost metrics,
   - regression analysis.
6. Paired theorem benchmarks must emit typed residual-bucket metrics, not only solved/failed counts:
   - `skipped_start`
   - `single_goal_near_miss`
   - `progressed_but_unsolved`
   - follow-on stage assignments (`compiler_specialist`, `hard_proof_solver`, `theorem_replanner`)
7. Skipped-start theorems and started-but-unsolved reasoning residuals are separate regimes.
   - compiler-specialist wins and hard-proof wins must be reported separately before any integrated claim.
8. Concurrently executed paired runtime conditions are diagnostic only.
   - headline theorem-count claims must come from one-condition-at-a-time runs or from runs with explicit throughput normalization.
9. Hard-proof work must be benchmarked on residual states, not only on full theorem restarts.
   - `last_goal` and small-residual states are first-class data products.
10. Every post-main comparison must report bounded-budget curves.
   - required attempt budgets: `128`, `256`, `512`, `1024`
11. Before scaling a hard-proof model class, run oracle-gap audits on:
   - startability / goal creation
   - residual-family routing
   - residual-state premise or unfolding selection
   - final closer selection
12. Self-application is diagnostic, not headline benchmark success.
   - every theorem-search report must separate `raw_success` from `honest_success`
   - any trace closed by applying the target theorem itself remains useful for structural telemetry, but must be excluded from zero-shot claims, k-line memory, and hard-SoM supervision
13. Crystallized memory is allowed at the level of proof language, not full proofs.
   - permitted memory: lemma names, rewrite patterns, tactic-family snippets, lane-order tendencies, residual-goal homologies, proof-plan geometries
   - prohibited memory: direct target-theorem replay, exact-proof retrieval, or any packet that turns theorem identity itself into the executable answer
14. Trace-derived hard specialist families should remain explicit in both data and evaluation.
   - current expected post-main families: `local_eq_close`, `local_ineq_close`, `membership_close`, `witness_construction_close`, `forward_context_close`, `small_multigoal_side_conditions`, `small_multigoal_planner`, `theorem_replanner`
   - these should be treated as separate architectural targets rather than as one generic hard bucket
15. The Arbiter must support domain-aware lane suppression when the trace family already makes a mismatch obvious.
   - example: `CategoryTheory`, `AlgebraicGeometry`, and abstract algebraic-geometry / discriminant-style residuals should demote numeric solver tactics such as `norm_num`, `ring`, `omega`, and `linarith` while preserving rewrite/exact/extensionality style moves
16. The symbolic hard layer should make "close before you unpack" explicit for abstract structural theorems.
   - if a residual still contains high-level structural tokens (`IsOpenMap`, `FormallyUnramified`, `HasRingHomProperty`, `IsIso`, `essImage`), the controller should try structural closure / lemma retrieval before definitional expansion
   - equality residuals with local bridge hypotheses (`IsIntegral`, `traceMatrix`, `Matrix.det`, `discr`) should surface `hypothesis_injection` and `symmetric_unfolding` rather than being treated as generic rewrite-only equalities
17. Membership walls are their own post-main regime even when they remain single-goal residuals.
   - goals of the form `x ∈ carrier ...` or membership into opaque `Ideal` / `Submodule` / `PrimeSpectrum` structures should surface `membership_specialist` packets with carrier-exposure and closure-lemma priors
18. Canonical existential witnesses should be modeled explicitly when the domain makes them obvious.
   - `exists_close` states in abstract domains such as `CategoryTheory` should surface canonical-witness priors before generic search
   - example: existentials over injective kernels / splittings should expose zero-object / zero-morphism witness hypotheses as symbolic priors for the witness-instantiation specialist
19. Search-pathology traces are first-class residual signals.
   - required pathologies to track: metavariable corruption, bare type side-goals, fold/unfold loops, goal explosions, and backward-rewrite corruption
   - these are replanner or branch-pruning signals, not ordinary local-closer supervision
20. Goal text must be sanitized before it reaches the embedding or routing stack.
   - replay-local artifacts such as `_wayfinder_replay__...` or `_wayfinder_decl_*` are compiler leakage, not theorem content
   - abstract-domain duplicate-goal pseudo-progress after tactics like `norm_num`/`simp` should be rejected rather than counted as real controller progress
21. Lean metaprogramming wrappers should be stripped before residual routing.
   - wrappers such as `autoParam` and `optParam` should not be allowed to change the goal-shape bucket or block structural tactics like `intros`
   - the runtime should attempt a tiny wrapper-removal janitor phase before assigning `forall_close` / `exists_close` / local-close families
22. Scoped-context failures should be surfaced explicitly in compiler-specialist data.
   - if goal creation fails under missing or inline-only `open scoped` context, treat that as `scoped_context_missing` rather than as a generic theorem-start failure
23. Plateau traces should become explicit hard-resolution geometry, not just benchmark anecdotes.
   - hard packets should carry search-control geometry (`no_progress_ratio`, blank-lane plateau streaks, rewrite-cycle counts, plateau goal signatures)
   - repeated identical-goal no-progress tails should produce negative-k-line packets for avoidance learning and replanner control
24. Compiler-specialist packets should carry reconstruction actions rather than only labels.
   - required actions include theorem-site lookup, file-context replay, symbol-name canonicalization, and source-header replay (`variable`, `open`, `open scoped`, `include/omit`, `local notation`, `local attribute`)
25. Structural-property and eventual/filter residuals should be surfaced as dedicated symbolic priors before learned closure.
   - abstract structural goals (`IsIso`, `Injective`, `IsOpenMap`, `FormallyUnramified`, `essImage`, etc.) should bias toward `close_before_unpack`
   - eventual/filter goals (`=ᶠ`, `Filter.atTop`, `Tendsto`, `isLittleO`, `isBigO`) should form their own residual geometry rather than being flattened into generic equality or replanner buckets
26. Dr. Ducky is the mandatory deterministic pre-learning stage for local residuals.
   - learned hard-proof or second-order SoM training must consume Dr. Ducky capsules, prescriptions, proof skeletons, and executor outcomes rather than bypassing that symbolic layer
   - the controller should learn when and how to invoke Dr. Ducky, not relearn her local symbolic work from raw residual text
27. Backend-family telemetry is part of the controller-visible symbolic surface.
   - second-order SoM packets should expose whether a residual prefers `egglog_eqsat`, `rosette_proof_dsl`, `kodkod_relational`, or `symbolic_rewrite_vm`
   - controller learning should use those backend-family surfaces for invocation and budget decisions

## 3A. Ready-To-Run Post-Freeze Experiment Sequence

Assume the live hard run has been frozen into:

```bash
RUN_DIR=runs/exp_som012_hard_eval_r2_frozen
BUNDLE_DIR=$RUN_DIR/bundle
DUCKY_DIR=$BUNDLE_DIR/dr_ducky
```

### `EXP-DD-013A`: Frozen hard-corpus materialization

```bash
python -m scripts.audit_hard_run \
  --run-dir "$RUN_DIR" \
  --output-json "$RUN_DIR/audit.json"

python -m scripts.build_hard_collection_bundle \
  --inputs "$RUN_DIR/details.jsonl" \
  --output-dir "$BUNDLE_DIR"

python -m scripts.build_hard_resolution_layer \
  --bundle-dir "$BUNDLE_DIR" \
  --db data/proof_network_v3.db
```

Purpose:
- freeze the data-geometry surface for the hard run
- materialize `hard_som_packets.jsonl`, `compiler_specialist_packets.jsonl`, and Ducky-ready residual packets

Success gate:
- packet counts reconcile with the frozen `details.jsonl`

### `EXP-DD-013B`: Dr. Ducky closure benchmark

Sampling policy:
- `executor_validation_stratified120` is a guarded round-robin sample across the four local residual buckets after the targeted theorem set is injected.
- targeted rows are capped by `--limit`; they do not silently overrun the requested slice size.

```bash
python -m scripts.build_dr_ducky_worklist \
  --run-dir "$RUN_DIR" \
  --output-dir "$DUCKY_DIR"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 20 \
  --output-json "$DUCKY_DIR/executor_validation_local20_vnext_summary.json" \
  --output-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_rows.jsonl" \
  --engine-outcomes-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$DUCKY_DIR/executor_validation_local20_vnext_projector_outcomes.jsonl" \
  --closure-report-json "$DUCKY_DIR/executor_validation_local20_vnext_closure_report.json"

python -m scripts.run_dr_ducky_executor_validation \
  --run-dir "$RUN_DIR" \
  --limit 120 \
  --output-json "$DUCKY_DIR/executor_validation_stratified120_summary.json" \
  --output-jsonl "$DUCKY_DIR/executor_validation_stratified120_rows.jsonl" \
  --engine-outcomes-jsonl "$DUCKY_DIR/executor_validation_stratified120_engine_outcomes.jsonl" \
  --projector-outcomes-jsonl "$DUCKY_DIR/executor_validation_stratified120_projector_outcomes.jsonl" \
  --closure-report-json "$DUCKY_DIR/executor_validation_stratified120_closure_report.json"
```

Purpose:
- measure theorem-faithful start rate, certificate generation, projector success, honest progress, and honest closure on frozen residuals

Success gate:
- non-trivial honest progress on both slices, with closure reported separately

### `EXP-DD-013C`: Dr. Ducky residual-bucket ablation

Run `scripts.run_dr_ducky_executor_validation` four times with:

- `--residual-bucket single_goal_near_miss`
- `--residual-bucket single_goal_stall`
- `--residual-bucket multi_goal_small_progress`
- `--residual-bucket multi_goal_large_progress`

Purpose:
- identify which residual buckets are engine-limited, projector-limited, or genuinely replanner-limited

Success gate:
- per-bucket `by_engine`, `by_certificate_shape`, and `by_projector_status` summaries are discriminative enough to train invocation policy

### `EXP-SOM-013A`: Frozen hard-bucket headroom audit

```bash
python -m scripts.run_exp_som012_depth_ladder \
  --inputs "$BUNDLE_DIR/hard_proof_local.jsonl" \
  --output-dir "$RUN_DIR/depth_ladder_local" \
  --cosine-rw \
  --cosine-rw-seq

python -m scripts.run_exp_som012_depth_ladder \
  --inputs "$BUNDLE_DIR/hard_proof_planner.jsonl" \
  --output-dir "$RUN_DIR/depth_ladder_planner" \
  --cosine-rw \
  --cosine-rw-seq

python -m scripts.run_exp_som012_oracle_gap \
  --inputs "$BUNDLE_DIR/hard_proof_all.jsonl" \
  --resolution-packets "$BUNDLE_DIR/hard_resolution_layer/resolution_packets.jsonl" \
  --output-dir "$RUN_DIR/oracle_gap"
```

Run-layout rule:
- the frozen `EXP-SOM-012` directory is the **source run**
- deterministic freeze artifacts stay under `$BUNDLE_DIR`
- post-freeze evaluation runs such as `EXP-SOM-013A` should be created as sibling directories under `runs/`, not nested inside the source run

Purpose:
- quantify remaining headroom before launching learned second-order control

Success gate:
- depth-ladder and oracle-gap summaries cleanly separate routing, premise, closer, and replanner deficits

### `EXP-SOM-013B`: Second-order packet freeze

Purpose:
- freeze the full symbolic training surface for second-order SoM

Run:

```bash
python -m scripts.build_second_order_packet_freeze \
  --run-dir "$RUN_DIR"
```

Primary outputs:
- `second_order_som/second_order_packets.jsonl`
- `second_order_som/ducky_outcome_packets.jsonl`
- `second_order_som/summary.json`
- `dr_ducky_engine_outcomes.jsonl`
- `dr_ducky_projector_outcomes.jsonl`
- `dr_ducky_closure_report.json`

Required inputs:
- `hard_som_packets.jsonl`
- `compiler_specialist_packets.jsonl`
- `dr_ducky_ledger_packets.jsonl`
- `dr_ducky_engine_outcomes.jsonl`
- `dr_ducky_projector_outcomes.jsonl`
- `dr_ducky_closure_report.json`
- first-order SoM telemetry from the hard bundle

Launch gate:
- `EXP-DD-013A` through `EXP-SOM-013A` complete without packet-schema churn
- Ducky and hard-resolution counts reconcile
- the next code change is a packet-to-feature adapter for second-order controller training, not more raw benchmark collection

### `EXP-SOM-013C`: Second-order feature dataset build

Purpose:
- convert the frozen symbolic packet manifold into trainable controller arrays

Run:

```bash
python -m scripts.build_second_order_feature_dataset \
  --packets "$BUNDLE_DIR/second_order_som/second_order_packets.jsonl" \
  --output-dir "$BUNDLE_DIR/second_order_som/features"
```

Primary outputs:
- `second_order_som/features/train.npz`
- `second_order_som/features/eval.npz`
- `second_order_som/features/metadata.json`
- `second_order_som/features/summary.json`

Launch gate:
- `EXP-SOM-013B` complete
- controller labels and backend targets are represented as trainable arrays
- the next step is controller training, not more packet munging

## Conditions Required Before Second-Order Success Claims

The second-order SoM is not supposed to replace the symbolic layers below it. It is supposed to learn orchestration over them. Before the project can make a serious second-order success claim, the following conditions must hold:

1. **The first-order baseline must already solve the majority regime**
   - already true
2. **The hard tail must already be frozen into symbolic packet geometry**
   - already true
3. **Dr. Ducky must already emit real executor outcomes on theorem-faithful replay**
   - already true at the progress level
4. **Backend-family outcome surfaces must be discriminative**
   - this is what `EXP-DD-014A/B/C/D` tests
5. **The second-order packet schema must be stable**
   - this is what `EXP-SOM-013B` enforces
6. **The trained controller must then be measured at theorem-search level**
   - not merely by packet classification metrics

This means the next second-order experiments are not “train now and hope.” They are gated on the symbolic closure experiments.

### `EXP-DD-014A/B/C/D`: Backend implementation track

Before the second-order controller is trained, the frozen corpus should also support explicit backend-family ablations inside Dr. Ducky:

- `egglog_eqsat`
- `rosette_proof_dsl`
- `kodkod_relational`

These backend families now exist as in-repo production runtimes. The second-order controller should therefore consume their actual outcome surfaces, not just their names. These backend families should enter second-order SoM packets as ordinary symbolic features, not hidden implementation details.

The gating rule is strict:

1. run `EXP-DD-014A/B/C/D`
2. run `EXP-DD-015` so backend-local Ducky progress is measured inside the actual theorem-closing architecture
3. freeze only backend families and bridge behaviors that show honest closure lift or clear honest progress lift
4. expose those backend-family and bridge outcomes as controller-visible packet features
5. only then train the second-order SoM

## Path From The Current Freeze To The Full Research End-State

From the current frozen benchmark state, the full program path is:

1. quantify hard-tail headroom with `EXP-SOM-013A`
2. measure backend-family closure lift with `EXP-DD-014A/B/C/D`
3. run the fully integrated hard-tail bridge with `EXP-DD-015`
4. patch only the closure-critical symbolic deficits that the backend and bridge results expose
5. freeze the second-order packet schema with `EXP-SOM-013B`
6. build the trainable feature dataset with `EXP-SOM-013C`
7. train the second-order SoM
8. rerun the integrated bridge with the learned second-order controller in place
9. run paired theorem-search comparisons:
   - static strong baseline
   - baseline + Dr. Ducky
   - baseline + Dr. Ducky + second-order SoM
10. convert the best condition into the publication bundle with frozen artifacts, ablations, case studies, and reproducibility commands

That is the route from the current first-order majority-solver to the full layered architecture capable of testing genuine hard-tail recovery.

## 4. Required Data Products

These are the minimum data artifacts required before strong SoM claims.

### 4.1 Required before learned-controller claims

1. `data/temporal_train.jsonl`
   - source: `EXP-058`-style traces, not just the old 500-theorem slice
   - content:
     - `TemporalState` snapshot
     - lane order / family prior / phase / escalation
     - proof-history summary
     - `SubtaskIR` / trigger summaries
     - exact Lean-backed next-progress labels

2. `data/template_narrative_train.jsonl`
   - theorem-level narrative packets
   - theorem statement / namespace
   - proof-history summary
   - theorem-level `SubtaskIR` / trigger profile summaries
   - startability / repair status

3. `data/strategy_memory.json`
   - symbolic orchestration priors mined from successful traces
   - keyed by template, namespace, goal-shape bucket, recent lane history, repair status

4. `data/routing_hard_negatives.jsonl`
   - states where one lane clearly dominates another
   - regressions from bad lane choice
   - wasted-budget examples
   - failed replan examples

### 4.2 Required before residual-specialist claims

5. `data/residual_exp058_started.jsonl`
   - the 482 started-but-unproved theorems under `EXP-058`
   - residual classification and trace summaries

6. `data/residual_exp058_skipped.jsonl`
   - the 241 skipped-start theorems under `EXP-058`
   - goal-start failure class, repair attempts, and `LeanFeedback`

7. `runs/<paired_run>/postrun/residual_report.json`
   - theorem-level paired benchmark decomposed into residual buckets
   - includes `started_theorems`, `skipped_start`, `progressed_but_unsolved`, `one_goal_left_failures`, `raw_success`, `honest_success`, and `self_application_successes`

8. `runs/<paired_run>/postrun/by_follow_on_stage/compiler_specialist.jsonl`
   - skipped / goal-start failures only
   - worklist for ContextIR / source-context / formatting repair analysis

9. `runs/<paired_run>/postrun/by_follow_on_stage/hard_proof_solver.jsonl`
   - started theorems with one-goal-left or small-multi-goal residuals
   - the semi-isolated post-main input set for a hard-proof SoM

10. `runs/<paired_run>/postrun/last_goal_residuals.jsonl`
   - one-goal-left and small residuals with the best available reduced goal text
   - canonical benchmark for post-main theorem patching

11. `runs/<paired_run>/postrun/hard_proof_local.jsonl`
   - `single_goal_near_miss` and `single_goal_stall`
   - input set for local closer/setup-chain specialists

12. `runs/<paired_run>/postrun/hard_proof_planner.jsonl`
   - `multi_goal_small_progress` and `multi_goal_small_stall`
   - input set for small-multi-goal planners

13. `runs/<paired_run>/postrun/depth_ladder_summary.json`
   - fixed-budget depth sweep over the hard bucket
   - minimum budgets: `128`, `256`, `512`, `1024`

14. `runs/<paired_run>/postrun/oracle_gap_summary.json`
   - upper bounds for startability, routing, premise selection, and final closure on the hard bucket

15. `runs/<paired_run>/postrun/hard_proof_combined_analysis.json`
   - structural summary of the hard bucket
   - namespace, geometry, template, lane-sequence, and tactic-prefix distributions

16. `runs/<hard_run>/bundle/hard_resolution_layer/resolution_packets.jsonl`
   - one symbolic prior packet per `hard_proof_solver` theorem
   - includes proof-graph candidate lemmas, k-line solved exemplars, residual skeleton geometry, proof-plan geometry, prior-graph geometry, and multigoal dependency geometry

17. `runs/<hard_run>/bundle/hard_resolution_layer/summary.json`
   - live family counts for the symbolic hard-resolution layer
   - top closing features, prior lemmas, and solved exemplars

18. `runs/<hard_run>/bundle/hard_resolution_layer/hard_som_packets.jsonl`
   - compact training/inference packets for the hard SoM
   - canonical schema for the learned hard-proof controller

19. `runs/<hard_run>/bundle/hard_resolution_layer/compiler_specialist_packets.jsonl`
   - startability/context packets kept off the hard-proof manifold
   - canonical schema for the compiler specialist track

20. `runs/<hard_run>/bundle/hard_resolution_layer/surface_inventory.json`
   - aggregated representation pressures, plan methods, domain hints, lane-suppression hints, and specialist targets
   - used to refine the hard SoM architecture before training
21. `runs/<hard_run>/bundle/summary.json`
   - must also expose `by_reasoning_gap_family` and `by_search_pathology`
   - used to separate clean residual reasoning from branch-corruption traces before training
22. `runs/<hard_run>/bundle/dr_ducky/dr_ducky_capsules.jsonl`
   - typed Dr. Ducky capsules over local hard residuals
   - includes ledger seeds, allowed engines, projector policy, and residual geometry

23. `runs/<hard_run>/bundle/dr_ducky/dr_ducky_ledger_packets.jsonl`
   - engine-ready symbolic packets for second-order SoM training
   - exposes negative geometry, projector markers, and execution budgets

24. `runs/<hard_run>/bundle/dr_ducky/dr_ducky_engine_outcomes.jsonl`
   - certificate-generation and engine-family telemetry

25. `runs/<hard_run>/bundle/dr_ducky/dr_ducky_projector_outcomes.jsonl`
   - projector success/rejection telemetry

26. `runs/<hard_run>/bundle/dr_ducky/dr_ducky_closure_report.json`
   - theorem-faithful starts, certificate generation, projector success, compile/progress/closure counts

27. `data/hard_split_som_q75/hard_theorems_train.jsonl`
   - theorem-level hard training corpus
   - default criterion: top quartile by proof-step complexity with `min_steps >= 5`
   - must carry `file_path` / `module` metadata and canonicalized theorem IDs so file-context replay can recover starts even when `env_inspect` fails

28. `data/hard_split_som_q75/hard_theorems_eval.jsonl`
   - theorem-level hard evaluation corpus
   - deterministic holdout split from the same hard pool
   - same metadata contract as train: theorem-site metadata is infrastructure, not optional garnish
   - collection runs over this corpus should use deterministic shuffle (`seed` recorded) to avoid namespace-order skew in partial-run diagnostics

## 5. Experiment Slate

`EXP-SOM-001` through `EXP-SOM-011` are primarily original / first-order SoM and residual-harvest experiments.

`EXP-SOM-012` onward define the benchmark-first data program for the second-order SoM.

## EXP-SOM-001: Materialize SoM Data From The Strong Regime

**Question:** Do we actually have the right data products for SoM training and audit, derived from the current strong search stack rather than old smoke-test traces?

**Build:**

1. Rebuild `data/temporal_train.jsonl` from `EXP-058`-style traces.
2. Rebuild `data/template_narrative_train.jsonl`.
3. Mine `data/strategy_memory.json`.
4. Build `data/routing_hard_negatives.jsonl`.

**Important constraint:** `temporal_train.jsonl` must be **decision-point sampled**, not an undifferentiated all-steps dump.

**Primary outputs:**

- counts by packet type
- theorem coverage
- decision-point coverage
- per-lane class balance
- route-conflict examples

**Stop/go:**

- `temporal_train.jsonl` covers at least 1000 proved theorems from the strong regime
- route-conflict / lane-choice states are nontrivial
- hard-negative bank contains at least 500 high-confidence route mistakes

## EXP-SOM-002: Rule Arbiter vs Static `EXP-058`

**Question:** Can a typed rule-based Arbiter beat or sparsify the fixed `EXP-058` schedule?

**Conditions:**

1. `EXP-058` static baseline
2. rule Arbiter using current `TemporalController` contracts
3. rule Arbiter with goal ordering only
4. rule Arbiter with lane-order control only

**Primary metrics:**

- proved
- proved\|started
- Lean calls / theorem
- time / theorem
- route regret
- lane hit rate

**Success criteria:**

- at least match `EXP-058` proved count
- reduce Lean calls / theorem or wall time
- nontrivial route signal over the static schedule

**Failure modes to audit:**

- bad suppression of dominant finisher tactics
- poor goal ordering
- budget starvation from misallocated `budget_slice`

## EXP-SOM-003: Strategy Memory Ablation

**Question:** Does symbolic k-line-like memory improve routing before any learned controller is deployed?

**Conditions:**

1. static `EXP-058`
2. rule Arbiter only
3. rule Arbiter + `strategy_memory`

**Primary metrics:**

- theorem count
- time
- Lean calls
- early-close rate
- lane hit rate

**Success criteria:**

- equal or higher prove rate than rule-only Arbiter
- lower cost or lower route regret

**Why this matters:** this is the cheapest direct test of the Minsky/Winston-style memory thesis inside Wayfinder.

## EXP-SOM-004: Recognition and Planning Over Symbolic Packets

**Question:** Can template recognition and theorem-level planning over symbolic packets predict productive finisher regimes better than static heuristics?

**Inputs:**

- `template_narrative_train.jsonl`
- theorem statement / namespace
- proof-history summary
- theorem-level `SubtaskIR`
- trigger-profile summaries
- startability status

**Targets:**

- active template
- induction-needed flag
- shape-tactic-needed flag
- next productive family
- replan-needed flag

**Evaluation:**

- top-1 / top-3 template accuracy
- calibration
- improvement in family-prior prediction
- downstream benefit when fed into Arbiter decisions

**Success criteria:**

- meaningful lift over namespace-only or goal-shape-only heuristics
- signal strong enough to justify feeding template/planning into routing

## EXP-SOM-005: Compact Learned Temporal Controller

**Question:** Can a compact typed controller beat the rule-based Arbiter on the strong regime?

**Candidates:**

- MLP
- GRU
- other small recurrent controller

**Inputs:**

- `temporal_train.jsonl`
- typed temporal packets only

**Labels:**

- next goal
- next productive lane
- next productive family
- replan
- escalation level

**Primary metrics:**

- theorem count in active benchmark mode
- lane hit rate
- route regret
- Lean calls / theorem
- time / theorem

**Success criteria:**

- beat rule Arbiter on paired benchmark
- no theorem-count regression vs static `EXP-058` unless cost reduction is substantial

## EXP-SOM-006: Teacher Distillation Over Symbolic Packets

**Question:** Do offline narrative/reasoning teachers improve the compact controller or planner when restricted to symbolicized packets?

**Teacher role only:**

- template audit
- proof-story induction
- lane-order proposals
- replan proposals
- strategy-summary generation

**Allowed inputs:**

- `template_narrative_train.jsonl`
- `temporal_train.jsonl`
- `strategy_memory.json`

**Not allowed:**

- raw Lean proof scripts as the default teacher input
- direct runtime theorem-search control by a large story/reasoning model

**Primary comparison:**

1. compact runtime trained on typed supervision only
2. compact runtime trained with teacher soft labels / distilled targets

**Success criteria:**

- distilled runtime beats typed-supervision-only runtime
- teacher labels improve routing without increasing runtime complexity

## EXP-SOM-007: Residual Specialist Program

**Question:** Which specialists are actually warranted on the post-`EXP-058` residual?

**Residual sets:**

- 482 started-but-unproved under `EXP-058`
- 241 skipped-start under `EXP-058`

**Specialist tracks:**

1. residual `apply`
2. residual `refine`
3. startability/context-repair specialist
4. subgoal-local premise rescoping

**Hard rule:** no specialist returns to always-on global deployment without proving value on the post-`EXP-058` residual.

**Primary metrics:**

- solved residual theorems
- added theorem count over `EXP-058`
- cost / added theorem
- interaction with dominant finisher stack

## EXP-SOM-008: Full SoM Integration Benchmark

**Question:** Does the full SoM stack beat the fixed strong finisher system?

**System under test:**

- Recognition
- Planning
- Temporal controller
- Strategy memory
- Dominance-aware routing
- Residual specialists
- startability specialist

**Benchmark conditions:**

1. `EXP-058` static baseline
2. SoM rule + memory
3. SoM learned controller
4. SoM learned controller + teacher distillation

**Primary metrics:**

- proved / total
- proved\|started
- started
- Lean calls / theorem
- time / theorem
- regressions vs baseline

**Hard gate:**

- no claim of SoM validation unless the integrated SoM stack matches or beats `EXP-058` on paired theorem sets

## EXP-SOM-009: 2-Step Search on Single-Goal Residual

**Question:** How many of the 256 single-goal stalls are exactly one setup tactic away from a closable form?

**Motivation:** EXP-SOM-007 proved that no SINGLE tactic closes any residual theorem. But many single-goal stalls may be one intermediate rewrite, simplification, or normalization away from a state where `exact?` succeeds. This is the "2-ply look-ahead" for proofs — analogous to Yami's 2-ply mate detection, but applied to tactic search.

**Architecture:** Progressive 2-step pipeline (setup; closer):
1. Load each single-goal stall's `last_goal` and create a Lean goal state
2. Generate setup candidates:
   - Top-10 cosine-ranked accessible premises as `rw [premise]` and `rw [← premise]` (20 attempts)
   - Normalization tactics: `push_cast`, `norm_cast`, `simp only []` (empty simp), `ring_nf`, `norm_num` (5 attempts)
   - Structural setup: `push_neg`, `contrapose`, `by_contra` (3 attempts)
3. For each setup that **changes** the goal (progress without closing):
   - Try `exact?` on the modified goal
   - Try `simp_all` on the modified goal
   - Try `linarith` / `omega` / `ring` on the modified goal
4. Record: which setup+closer pairs succeed, what the intermediate goal looks like

**Conditions:**

1. `rw_then_exact` — cosine-ranked `rw [premise]; exact?` (primary)
2. `norm_then_exact` — normalization tactics then exact? (secondary)
3. `structural_then_exact` — push_neg/contrapose/by_contra then exact? (tertiary)
4. `combined` — all setup sources, all closers

**Primary metrics:**

- theorems closed / 256 single-goal stalls
- theorems closed / 482 total residual
- setup tactic distribution (which setups work?)
- closer distribution (which closers work on modified goals?)
- cost: Lean calls per closed theorem

**Stop/go:**

- If >=30/256 close: multi-step SoM has large headroom. Proceed to Level 2 (learned step planner).
- If 10-30 close: moderate headroom. Proceed but calibrate expectations.
- If <10 close: residual is genuinely deep. Skip Level 2, go directly to LLM-guided search or deeper planning.

**Hard rule:** Goal creation must use the same Tier B file-context path as EXP-058. No proxy goals.

## EXP-SOM-010: Original / First-Order Multi-Step SoM Training (Contingent on SOM-009)

**Question:** Can the original three-stage first-order SoM with specialist agents (rewrite, structural, solver, apply, closer) and a learned orchestrator improve multi-step proof search beyond the 2-step baseline?

**Contingent on:** EXP-SOM-009 showing >=30/256 closures (evidence that multi-step has headroom).

**Current interpretation (2026-03-25):** the multistep router is strong original / first-order SoM component evidence. It should remain in that role until the hard-run corpus and Dr. Ducky outputs are folded into the later second-order training/evaluation program.

**Architecture:** Adapts the Yami SoM training methodology (SOM_TRAINING_ANALYSIS §2) to proof search:
- 5 specialists: Rewrite Agent, Structural Agent, Solver Agent, Apply Agent, Closer Agent
- Each specialist: domain signals at high capacity + full goal context at low capacity
- Orchestrator: trust-weight softmax over specialists, override mechanism for clear cases
- Three-stage training: specialists independently → orchestrator frozen → joint fine-tuning
- PAB stability for all stages

**Training data:** LeanDojo proof scripts (78K theorems) → (goal_state, tactic, next_goal_state) triples, labeled by specialist ownership.

**Success criteria:** Beat EXP-058 + 2-step baseline on the residual. No regression on the 1277 already-proved.

## EXP-SOM-011: Paired Runtime Benchmark + Residual Bucket Audit

**Question:** On the same 2000-theorem benchmark as `EXP-058`, do live `norm_then_close` deployment and torch routing improve theorem count and, just as importantly, reshape the remaining residual into cleaner post-main buckets?

**Conditions:**

1. `EXP-058` paired baseline
2. `+ norm_then_close`
3. `+ norm_then_close + torch routing`

**Protocol additions:**

1. Run paired theorem-search conditions on the identical theorem list.
2. Analyze each finished report with `scripts/analyze_benchmark_residuals.py`.
3. Materialize worklists with `scripts/run_exp_som011_postrun.py`.
4. Record concurrency level and effective throughput for each condition.
5. If conditions are run concurrently on the same machine, do not treat the raw theorem-count delta as a clean replacement for the single-condition `EXP-058` baseline.
   - in practice, parallel paired runs on one laptop can cut per-condition effective compute budget by at least about one third

**Primary metrics:**

- proved / total
- started / total
- skipped-start count
- progressed-but-unsolved count
- one-goal-left failures
- residual bucket distribution
- follow-on stage distribution
- cost / theorem

**Success criteria:**

- theorem-count lift vs `EXP-058`, or
- no theorem-count lift but a cleaner residual distribution:
  - fewer `skipped_start`
  - more `single_goal_near_miss`
  - more residual mass assigned to `hard_proof_solver` rather than `theorem_replanner`

**Hard rule:** no routing claim from `EXP-SOM-011` is complete without residual bucket migration analysis.

## EXP-SOM-012: Semi-Isolated Hard-Proof Program

**Question:** Can a second-stage hard-proof program, invoked only after the dominant main solver exits unsolved, close the `hard_proof_solver` bucket without perturbing the main proof stack?

**Input set:** `runs/<paired_run>/postrun/by_follow_on_stage/hard_proof_solver.jsonl`

**Theorems admitted to this stage:**

- `single_goal_near_miss`
- `single_goal_stall`
- `multi_goal_small_progress`
- `multi_goal_small_stall`

**Architecture constraints:**

1. This stage runs only after the main solver finishes.
2. It is not an always-on lane.
3. It may use its own SoM decomposition:
   - local closer specialist
   - local inequality specialist
   - witness / instantiation specialist
   - forward-context specialist
   - side-condition sweeper
   - setup-chain specialist
   - small-multi-goal planner
   - theorem replanner / branch-pruner
   - theorem-local memory / hard-negative bank
4. `compiler_specialist` theorems remain out of scope for this stage.

**Mandatory gates before learned hard-proof training:**

### Stage 0: Residual-state materialization

Build:

- theorem-level collection rows with full trace (`details.jsonl`)
- goal-start diagnostics (`goal_start_failures.jsonl`)
- optional trigger probes (`trigger_states.jsonl`)
- `last_goal_residuals.jsonl`
- `hard_proof_local.jsonl`
- `hard_proof_planner.jsonl`
- `hard_proof_combined_analysis.json`
- hard theorem split (`hard_theorems_train`, `hard_theorems_eval`)
- current-SoM bundle artifacts (`temporal_dataset.jsonl`, `strategy_memory.json`)
- symbolic hard-resolution bundle (`hard_resolution_layer/resolution_packets.jsonl`, `summary.json`)

Purpose:

- convert theorem-level failures into benchmarkable residual states
- isolate one-goal and small-multi-goal subproblems before adding new learning
- build a theorem-level hard corpus for future data collection runs, so the next trace harvest is biased toward the right part of proof space
- ensure the hard-set trace harvest carries both current-SoM supervision and second-stage residual-state supervision from the same run
- precompute theorem-local structural priors so the learned hard SoM starts from explicit candidate lemmas, solved exemplars, and family-specific closing features rather than from raw residual text alone

### Stage 0.5: Symbolic hard-resolution layer

**Question:** Can a proof-graph plus k-line symbolic layer turn raw hard residuals into structured prior packets before any hard-proof learning is trained?

**Build:**

1. retrieve candidate prior lemmas from the proof network using theorem-local graph position, accessible premises, and residual-goal symbols
2. attach solved-theorem exemplars from the same hard collection as k-line-like prior traces
3. emit family-specific closing features (`rewrite_chain`, `extensionality`, `subgoal_ordering`, `bridge_lemma_synthesis`, etc.)
4. encode residual skeleton geometry (`goal bucket`, binder/connective counts, transport/extensionality/normalization pressure)
5. encode proof-plan geometry (lane history, tactic-family profile, candidate method abstractions, specialist targets, and lane-suppression hints)
6. encode prior-graph geometry (bank signature, anchor surface, accessible-premise neighborhood, candidate-locality statistics)
7. encode search-pathology geometry (metavariable spill, fold/unfold loop, goal explosion, backward-rewrite corruption)
8. encode search-control geometry and negative-k-line geometry (plateau streaks, blank-lane tails, rewrite-cycle counts, avoid-tactic prefixes)
9. encode compiler reconstruction surfaces (theorem-site metadata plus source-header replay actions)
10. group packets by `resolution_family` for specialist consumption

**Primary outputs:**

- `hard_resolution_layer/resolution_packets.jsonl`
- `hard_resolution_layer/hard_som_packets.jsonl`
- `hard_resolution_layer/compiler_specialist_packets.jsonl`
- `hard_resolution_layer/by_resolution_family/*.jsonl`
- `hard_resolution_layer/closing_feature_inventory.json`
- `hard_resolution_layer/candidate_prior_inventory.json`
- `hard_resolution_layer/kline_exemplar_inventory.json`
- `hard_resolution_layer/surface_inventory.json`

**Why this stage exists:**

- it preserves the discovery/evaluation separation from the broader data-geometry architecture
- it gives the learned hard SoM structural priors instead of forcing it to rediscover obvious lemma neighborhoods from scratch
- it turns the current run into direct supervision for both theorem-local memory and prior-work discovery
- it creates a cleaner interface between symbolic decomposition and learned residual control
- it makes representation-change pressure explicit, so the hard SoM can learn over `normalization`, `transport`, `extensionality`, `bridge`, and `subgoal coordination` as first-class geometric signals instead of trying to infer them only from text
- it lets the trace-derived hard families become explicit architectural surfaces, e.g. inequality closure, witness construction, side-condition sweep, and domain-aware suppression
- it isolates replanner-worthy traces from clean local-closer supervision by making branch corruption and state loops explicit
- it gives the learned hard SoM explicit negative-memory surfaces for plateau avoidance while keeping branch bailout and source-header replay as symbolic controller logic

**Stop/go:**

- at least 80% of hard residuals should receive nonempty candidate-prior packets
- at least 60% should receive solved exemplar matches
- family inventories should remain concentrated rather than fragment into many tiny one-off clusters

### Stage 0.75: Dr. Ducky deterministic residual program

**Question:** Can the deterministic residual executor turn the hard-run local bucket into executable symbolic supervision before any learned second-order controller is trained?

**Build:**

1. materialize Dr. Ducky capsules from:
   - `single_goal_near_miss`
   - `single_goal_stall`
   - `multi_goal_small_progress`
   - `multi_goal_large_progress`
2. validate routing / suppression / bank selection
3. replay targeted and sampled residuals theorem-faithfully in Lean
4. record:
   - started / theorem-faithful start
   - progress
   - closure
   - accepted no-op rejection
   - winning proof skeleton / program family

**Primary outputs:**

- `bundle/dr_ducky/dr_ducky_capsules.jsonl`
- `bundle/dr_ducky/dr_ducky_ledger_packets.jsonl`
- `bundle/dr_ducky/summary.json`
- `bundle/dr_ducky/validation.json`
- `bundle/dr_ducky/dr_ducky_engine_outcomes.jsonl`
- `bundle/dr_ducky/dr_ducky_projector_outcomes.jsonl`
- `bundle/dr_ducky/dr_ducky_closure_report.json`
- `bundle/dr_ducky/executor_validation_*.json`

**Why this stage exists:**

- it converts “local human-style reasoning gap” into a concrete executable subsystem
- it supplies the second-order SoM with symbolic routing targets and real executor outcomes
- it separates deterministic symbolic failure from orchestration failure

**Stop/go:**

- local residual coverage is substantial (`single_goal_near_miss` should remain a major bucket)
- routing alignment is strong on the targeted families
- theorem-faithful replay is reliable enough to generate training labels
- honest progress exists on benchmark residuals even if closure remains weak
- the artifact split is rich enough to report:
  - theorem-faithful replay validation
  - certificate generation
  - projector compilation
  - honest progress lift
  - honest closure lift

### Stage 1: Depth ladder

**Question:** How much headroom comes from deeper post-main search before any new learned hard-proof SoM is introduced?

**Conditions:**

1. `depth=2`
2. `depth=4`
3. `depth=8`
4. `depth=12`
5. `depth=16`

All conditions must be reported at bounded attempt budgets: `128`, `256`, `512`, `1024`.

**Primary metrics:**

- solved / hard bucket
- solved / `hard_proof_local`
- solved / `hard_proof_planner`
- bucket migration vs depth
- cost / added theorem

**Decision rule:** If deeper search does not materially reduce the hard bucket under fixed budgets, do not assume that a learned hard-proof SoM will rescue the same regime.

**Reference harness:** `scripts/run_exp_som012_depth_ladder.py`

### Stage 2: Oracle-gap audit

**Question:** On the hard bucket, what is the real next bottleneck: startability, routing, premise choice, or final closure?

**Required oracle-style audits:**

1. startability / reduced-goal creation
2. residual-family routing
3. premise or unfolding choice on the reduced goal
4. final closer choice

**Purpose:** measure the upper bound of each subproblem before scaling model size or depth.

**Reference harness:** `scripts/run_exp_som012_oracle_gap.py`

### Stage 3: Learned hard-proof SoM

Only after Stages 0-2 pass should the learned hard-proof SoM be trained and evaluated.

**Primary metrics:**

- hard-bucket theorems solved
- added theorem count over the paired main-solver condition
- cost / added theorem
- bucket shrinkage inside the hard-proof set

**Success criteria:**

- positive closures on the `hard_proof_solver` bucket
- no regression on already-proved theorems, because this stage is post-main and semi-isolated
- evidence that at least one hard-proof subfamily can be learned as a specialist rather than treated as generic search
- improvement must persist on bounded-budget curves, not only at one unconstrained endpoint

## 6. Recommended Order

This is the execution order that best matches current theory and code reality.

### Wave A: Make SoM real in the data layer

1. `EXP-SOM-001`
2. `EXP-SOM-002`
3. `EXP-SOM-003`

### Wave B: Learn control only after the process signal is real

4. `EXP-SOM-004`
5. `EXP-SOM-005`
6. `EXP-SOM-006`

### Wave C: Residual specialists and full integration

7. `EXP-SOM-007`
8. `EXP-SOM-008`

### Wave D: Multi-step proof planning

9. `EXP-SOM-009` — 2-step search baseline (infrastructure, no learning)
10. `EXP-SOM-010` — original / first-order multi-step SoM training (contingent on SOM-009 >=30/256)
11. `EXP-SOM-011` — paired runtime benchmark + residual bucket audit

### Wave E: Post-main hard-proof solving

12. `EXP-SOM-012` — staged hard-proof program:
    - residual-state materialization
    - symbolic hard-resolution layer
    - Dr. Ducky deterministic residual program
    - depth ladder
    - oracle-gap audit
    - learned hard-proof SoM

## 7. Balanced Sashimi Lessons To Carry Over

These are the process rules that should govern SoM work.

1. **Dual validation**
   - every SoM experiment must report both:
     - architecture result
     - process/routing result

2. **PAB-style trajectory discipline**
   - controller/routing traces must be logged from the beginning
   - do not retrofit process metrics after the fact

3. **Hard-negative banks early**
   - route mistakes, dominated-lane choices, failed replans, and wasted-budget states are first-class supervision

4. **Strict stop/go gates**
   - do not train the learned controller until the rule+memory regime shows real signal
   - do not claim teacher value until distillation beats compact typed supervision
   - do not redeploy residual specialists globally unless they win on the post-`EXP-058` residual

## 8. Recommended Immediate Work

If only one branch of work is funded right now, it should be:

1. finish the paired `EXP-SOM-011` runtime benchmark
2. materialize the postrun residual buckets (`compiler_specialist`, `hard_proof_solver`, `theorem_replanner`)
3. build `last_goal_residuals`, `hard_proof_local`, `hard_proof_planner`, and `hard_proof_combined_analysis` from the finished hard bucket
4. derive a theorem-level hard corpus (`q75`, `min_steps >= 5`) for the next training and benchmark trace harvest
5. run `scripts/run_exp_som012_hard_collect.py` on `hard_theorems_train` / `hard_theorems_eval`, then materialize the bundle with `scripts/build_hard_collection_bundle.py`
6. materialize the symbolic hard-resolution layer (`scripts/build_hard_resolution_layer.py`) and use it to define the initial hard specialist families
7. run the `EXP-SOM-012` depth ladder (`scripts/run_exp_som012_depth_ladder.py`) and oracle-gap audit (`scripts/run_exp_som012_oracle_gap.py`) under fixed budgets
8. only then launch the learned hard-proof SoM on the hard bucket

That is the shortest path from the current strong theorem-search result to a real Society of Mind validation that respects the actual structure of the remaining failures.
