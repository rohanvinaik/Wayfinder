# Formal Local-Execution Buildout Plan

## Summary
Rebuild the local-execution stack around one canonical source of truth, then deploy verifier-backed cosine local lanes and expand family coverage.

Order of work:
1. **rebuild canonical data** from raw LeanDojo traces — ✅ DONE (`data/canonical/`)
2. **theorem-faithful goal creation** — ✅ DONE (declaration-faithful Tier B, GoalStart 91%, B→A cascade)
3. **rw as first formal local lane** — ✅ DONE (cosine_rw beam=1 deployed, +1/50 theorem lift, EXP-RW-015/016)
4. **rw0 semantic benchmark** — ✅ DONE (oracle 62.8%, cosine top-5 61.5%, cosine top-1 33.3% on n=78 overlap set)
5. **extend cosine rewrite lane to rw1** — ✅ DONE (tier-default direction is enough)
6. **rewrite-family collapse through rw3** — ✅ DONE on current started step-0 slices (`cosine_rw_seq`)
7. **source-context compiler (`ContextIR`)** — IN PROGRESS (parallel infra track for replay coverage + future families)
8. **step>0 replay hardening** — USABLE (26% gate cleared; continue only if it raises denominator or unlocks families)
9. **motivated move layer (`SubtaskIR` + trigger profiles)** — NEW (controller-facing move typing + schema mining)
10. **multi-family expansion** — IN PROGRESS (global lane expansion paused; next target is executable action selection for `apply` / `refine`)
11. only then re-evaluate temporal-controller gains in `local_close`

### Key result update (2026-03-17)
**rw0 is solved by cosine geometry.** Cosine top-5 recovers 98% of oracle ceiling (61.5% vs 62.8%) on the same-denominator overlap set. The learned decoder (RW-001) does not beat cosine at matched Lean-call budget (38.5% vs 61.5% top-5). Cosine_rw beam=1 deployed in theorem search: +1/50 theorem, zero regressions, 4 Lean calls/thm overhead.

**Learning's value is no longer on rewrite-family local execution.** On the current measured slices,
`rw0`, `rw1`, `rw2`, and both bare/with-args `rw3` all collapse to a cheap scoped cosine executor.
The open frontier for learning is now:
- **Residual-conditioned orchestration** (when to invoke helper/specialist lanes after `rw` / bootstrap)
- **Executable action selection** over cosine shortlists where scoped retrieval already puts a plausible candidate in top-5
- **Controller-visible move selection** via `SubtaskIR` / trigger profiles
- **Family-specific decoders** only for families that still fail after scoping plus reranking

This keeps the plan aligned with the theory:
- theorem-level navigation is temporal/frontier orchestration
- local execution is a constrained DSL problem, not raw tactic generation
- deterministic lowering + Lean verification are part of the architecture, not post hoc checks
- theorem-site source context is itself a compiler problem (`ContextIR`), not just a header heuristic
- local proof steps should also be typed as **motivated moves** (`SubtaskIR`), not only executable syntax (`ActionIR`)
- cosine geometry is surprisingly strong on easy local families — learning should target what cosine can't do

## Key Changes

### 1. Canonical data pipeline — ✅ DONE

**Status:** Completed. `scripts/build_canonical_local_data.py` builds from raw LeanDojo traces.

**Outputs** (in `data/canonical/`):
- `canonical_residual_train.jsonl`: 240,848 non-structural steps
- `canonical_residual_eval.jsonl`: 12,505 held-out residual steps
- `canonical_rw_train.jsonl`: 67,239 rw-family steps with ActionIR
- `canonical_rw_eval.jsonl`: 3,508 held-out rw benchmark
- `canonical_eval_replayable.jsonl`: 16,242 full eval steps with prefix replay

**Record shape:**
- `theorem_full_name` (real Mathlib name)
- `file_path` (for environment reconstruction)
- `step_index`
- `goal_state_before`
- `tactic_text` (full tactic with arguments)
- `tactic_base`, `family`
- `annotated_premise`
- `prefix_tactics` (all tactics before this step, for replay)
- `canonical_action_ir` (lowered from ActionIR parse)
- `goal_shape_ir`, `trigger_profile_ir`, `subtask_ir`

**Hard rule:** all local-execution training and benchmarking uses this data. Synthetic `thm_N` datasets are disposable.

### 2. Theorem-faithful goal creation — ✅ DONE

**Status (EXP-RW-009, 2026-03-17):**
- GoalStart: **91%** (declaration-faithful Tier B + Tier A fallback, n=100 seed=42)
- Tier B: 54/91 (primary — uses env_inspect module + sourceStart/sourceEnd + active context wrapper)
- Tier A: 37/91 (fallback — env_inspect → goal_start)
- LeanValid|started: **22.0%** (oracle canonical), **62.8%** (oracle qualified)
- Dominant remaining failure: identifier_scope (54% of started failures — short names needing `open` directives)

**Three-tier goal creation:**

**Tier A — Direct start (fast path, ~80% coverage):**
- `env_inspect(full_name)` → theorem type
- `goal_start(type)` with `_goal_via_sorry` fallback (namespace-open from theorem name)
- Already implemented and working

**Tier B — Theorem-specific wrapper (faithful path):**
- Load theorem module/header context from `file_path`
- Generate a local wrapper theorem in the same environment:
  - derive `open` directives from file path and theorem namespace
  - include module-level scoped notation
- Use `load_sorry` on the theorem-specific wrapper
- Use `check_compile` as preflight validation before goal extraction
- This is the main improvement target

**Tier C — Replay-derived fallback (high-fidelity):**
- For local steps (step_index > 0): reconstruct by replaying `prefix_tactics` from theorem start
- Apply each prefix tactic sequentially via `goal_tactic`
- Then apply the candidate tactic at the actual intermediate state
- Slowest but most faithful — handles all scoping and elaboration correctly

**`check_compile` role:**
- Preflight / hygiene for Tier B wrappers
- Failure classification (environment vs tactic vs elaboration)
- NOT the main runtime oracle — `goal_start`/`goal_tactic` remain the step execution path

**Failure taxonomy (explicit in benchmark output):**
- `goal_creation_fail` — Tier A+B both failed
- `identifier_scope` — goal created but lemma name not resolved
- `elaboration_fail` — type mismatch / implicit argument failure
- `rw_inapplicable` — tactic semantically wrong for this goal
- `prefix_replay_fail` — Tier C replay broke on an intermediate step
- `success` — tactic accepted by Lean
- `success_with_goals` — tactic accepted, remaining subgoals

**Benchmark modes:**
- **Type-start mode** (fast): Tier A only — good for iteration, reports type_start_success
- **Replay mode** (faithful): Tier A → B → C cascade — reports replay_success, replay_only_success

**Acceptance gates (split):**
- Fast-path gate: `env_inspect` → `goal_start` success ≥ 80%
- Faithful benchmark gate: after Tier A+B+C cascade, unrecoverable goal creation failure < 5%
- Identifier/scope failure < 5% (after theorem-specific wrapper)
- Gold `ActionIR` lowering: compile-valid ~100% on benchmark harness
- Infra failures never enter model-negative accounting

### 3. `rw` as the first formal local lane — ✅ DONE (rw0 tier)
Cosine_rw beam=1 deployed in theorem search. rw0 (bare single forward rewrite) is solved by cosine geometry.

Fixed `rw` stack:
1. residual family gate includes `rw`
2. source gate chooses `hyp` vs `premise`
3. redex-aware scoped vocabulary is built from:
   - local hypotheses
   - rewrite-capable accessible premises
   - small fixed combinator set
4. pointer/reranker ranks scoped symbols
5. `ActionIR` is built
6. deterministic lowering emits Lean
7. Lean verifies with a small beam

Hard decisions:
- scoping is family-specific and redex-aware
- no theorem-sibling proxy scope in deployable eval
- no unconditional global fallback; only adaptive fallback when scope is empty or below minimum size
- direction is symbolic-first in v1:
  - lhs-only redex match -> forward
  - rhs-only redex match -> backward
  - ambiguous -> beam both
- exact string match is not a primary metric; canonical `ActionIR` and Lean-valid are primary

Ranking work:
- train inside-scope hard negatives
- include geometry features in reranking:
  - cosine / symbol similarity
  - source type
  - head-symbol / redex compatibility
  - theorem-level frontier bias as a weak prior only
- keep the cosine-only system as a permanent baseline

Acceptance gates (rw0 — all met):
- scoped coverage `>=95%` — ✅ 100% (EXP-RW-013)
- mean scope size `<=20` — ✅ 11.3 (EXP-RW-013)
- Lean-valid rw top-5 on replayable benchmark `>=20%` — ✅ cosine top-5 = 61.5% (EXP-RW-014)
- learned `rw` produces non-zero progress and non-zero closes in theorem search — ✅ cosine_rw: +1 theorem, 17 progress events (EXP-RW-015)

Acceptance gates (rw0 — not met, reclassified):
- beat cosine-only baseline — ❌ learned top-5 (38.5%) < cosine top-5 (61.5%). Learning does not add value on rw0 symbol selection. Cosine is the deployed baseline.
- improve in-scope top-1 over 43.6% — N/A (cosine top-1 = 45%, learned top-1 = 5%). Learning deferred to harder tiers.

### 3b. Rewrite tier roadmap — NEXT

**rw1 (backward rewrites) is now operational, not speculative.**
- Oracle: 58.1%
- Cosine top-5 (both dirs): 60.5%
- Direction is carried mostly by the tier itself:
  - rw0 defaults forward
  - rw1 defaults backward
  - a standalone learned direction head is not justified on the current split

**Deployment rule:**
- extend the runtime rewrite lane from `rw0` to `rw0 + rw1`
- use tier-conditioned default direction
- beam both only on ambiguity / low-confidence cases

**Where learning's value proposition now lives:**
- mixed-family orchestration once `simp` / `apply` are productive
- optional reranking over cosine shortlists, not standalone premise replacement
- controller-facing move selection over `SubtaskIR` / trigger profiles

Acceptance gates for learned decoder on rw2-rw3:
- beat cosine shortlist + Lean beam on `LeanValid@k|started` at matched Lean-call budget
- produce nonzero theorem-level gain beyond the deployed cosine rewrite lane

### 4. Search integration and theorem-level evaluation — ✅ DONE (rw0-rw1), NEXT (unified `cosine_rw_seq`)

Cosine_rw deployed in `proof_search.py` as a verifier-backed local lane.

**Results (EXP-RW-015/016/017):**

| Config | Proved | Gained | Attempts/thm | Time/thm |
|--------|--------|--------|-------------|----------|
| Baseline (no cosine) | 12/50 | — | 48 | 77s |
| **cosine_rw beam=1** | **13/50** | **+1** | **52** | **75s** |
| cosine_rw beam=5 | 13/50 | +1 | 56 | 78s |
| All 4 cosine lanes | 13/50 | +1 | 99 | 144s |

- cosine_rw is purely additive (zero regressions)
- beam=1 is optimal (wider beams add cost, not proves)
- Multi-family (apply/simp) adds progress events but no extra theorem closes at 2x cost

**Immediate next theorem-level experiment:**
- deploy unified sequential `cosine_rw_seq` over the rewrite family
- keep the existing single-step `cosine_rw` as an ablation/baseline
- compare:
  - baseline
  - baseline + cosine_rw(single-step)
  - baseline + cosine_rw_seq(sequential rewrite family)
- primary metric: theorem closes
- secondary: unique theorems touched, rewrite-lane provenance events, Lean calls/theorem

### 5. Family expansion — IN PROGRESS (after rewrite-family freeze)
Generalize cosine-beam-verify carefully, but do not confuse local component signal with theorem-search deployment value.

**Current status (2026-03-19):**
- `simp` is a supportive/helper lane: real local progress, no theorem-level lift yet
- `apply` is component-real but theorem-dormant when run globally
- `refine_named` is promising only as a reranking target
- `refine_anon` is a distinct harder regime not captured by current typed features

Family order and approach:
1. **executable action selection for `apply` and `refine`**
   - `apply`: name-ranking lift does not transfer to `LeanAccepted`; next target is unification-aware candidate selection
   - `refine_named`: ranking lift also does not transfer; next target is structured skeleton / action selection
   - next step is not theorem-search integration of the current reranker
2. **simp** — keep as a helper lane and compiler stress-test
   - useful for local simplification and subgoal cleanup
   - not yet justified as a theorem-winning default lane
3. **apply** — keep gated/off-by-default until executable candidate selection shows acceptable cost/yield
4. **exact / refine_anon** — distinct harder scoping/skeleton problems, not current deployment targets

Generalization rules:
- reuse the same canonical data pipeline
- each family gets its own scope builder and hard legality filters
- cosine-beam-verify is the proven deployment pattern — apply it first, learn second
- acceptance gate: nonzero theorem-level gain at acceptable Lean-call budget
- when gold is already often in top-5 but top-1 is weak, do not assume ranking is the right target; check `LeanAccepted` before integrating

Immediate milestone:
- freeze `NAV-004` as the aligned checkpoint
- freeze runtime around `cosine_rw` plus interleaved bootstrap
- keep `simp` as a helper lane and `apply` as a gated dormant specialist
- build the next executable-selection track on:
  - `apply` unification-aware candidate filtering/scoring
  - `refine` structured skeleton/action selection

### 6. Source-Context Compiler (`ContextIR`) — IN PROGRESS (parallel track)

Mathlib source context is the next bridge layer between theorem-faithful start states and reliable step>0 replay.

Current evidence:
- Whole-Mathlib census: `open scoped` 2370, `local_notation` 496, `local_attribute` 915, `include` 1064, `omit` 422, inline-only `... in` forms 6742
- Benchmark audit on `rw0_eval`: active `open` on 693 processed examples, `open_scoped` on 262, `local_notation` on 42, `local_attribute` on 42
- Common unsupported patterns are exactly the next-declaration forms the current wrapper path does not compile: `open Classical in`, `variable (M) in`, `include Q in`

Implementation target:
- parse theorem-site lexical context into `ContextIR`
- support first-pass rendering of:
  - `open`, `open scoped`
  - `variable(s)`, `universe(s)`
  - `local notation`, `scoped[...]`, notation declarations
  - `attribute [local instance]`, `attribute [local simp]`
  - `include`, `omit`
- track inline-only `... in` forms explicitly and add executable support in a second pass

Validation scripts:
- `python -m scripts.context_ir_census`
- `python -m scripts.context_ir_benchmark_audit --dataset data/canonical/rw0_eval.jsonl`

Acceptance gates:
- benchmark audit processes most examples without silent drops
- unsupported context forms are explicit and ranked by frequency
- Tier B wrapper generation consumes `ContextIR` instead of ad hoc header extraction
- only keep spending on `ContextIR` if it:
  - raises `ReplaySuccess | base_state`
  - reduces index-0 replay failures
  - or unlocks theorem-level value for `simp` / `apply`

### 6b. Motivated Move Layer (`SubtaskIR`) — NEW

This is a controller-facing layer above `ActionIR`, derived from canonical proof data rather than
directly executed in Lean.

Implemented path:
- `src/subtask_ir.py`
- `scripts/build_subtask_training_data.py`
- `scripts/validate_subtask_ir.py`
- `scripts/mine_move_schemas.py`
- `scripts/build_move_inventory.py`

Outputs:
- `GoalShapeIR`: coarse local state shape
- `TriggerProfileIR`: why a move is admissible here
- `SubtaskIR`: what local transformation the step attempts

Role in the plan:
- make move triggers explicit instead of implicit in lane ordering
- give the controller a typed move space richer than raw tactic strings
- mine reusable `(family, subtask_kind, trigger_signature)` schemas from successful traces
- collapse those schemas into a compact move inventory for `apply` / `simp` planning

Alignment rule:
- `SubtaskIR` is a planning/controller contract, not a navigator-bank label
- navigator training may use only descriptive local-state metadata (`goal_target_head`,
  `trigger_signature`) as auxiliary regularization
- template recognition / planning may consume `subtask_kind`

This is the direct import from the motivated-proof / Human-Oriented ATP line that is worth keeping:
not their full framework, but the constraint that local proof steps should have explicit triggers
and explicit local objectives.

### 7. Step>0 replay — USABLE (26%, gate cleared)

Tier C sequential `goal_tactic` replay progression: 0% → 26% (EXP-RW-007 → EXP-RW-024).

**Step>0 metric contract (frozen, EXP-RW-028):**
- **Primary**: `LeanValid@k | replayed` — does any premise in scope produce a Lean-valid rw at the replayed state
- **Secondary**: `ReplaySuccess`, `gold_in_scope`, scope size
- **Diagnostic only**: oracle qualified (not a ceiling — 50% of oracle failures are stale labels or replay drift)
- **Noise bucket**: examples where `annotated_premise` is not equality/iff or not executable as rw atom — excluded from oracle ceiling, tracked separately

**Replay drift tracking**: `state_drift_oracle_fail` — cases where cosine succeeds but oracle fails due to replayed state not matching the trace state exactly. Prevents hiding replay problems inside model failure on future tiers.

**Dataset hygiene for future rw tiers:**
- Mark examples where `annotated_premise` fails `rw` validity check (not equality/iff, equation compiler fails) as `label_noise`
- Exclude from oracle comparisons; keep in LeanValid@k denominators
- Apply this filter before any rw1/rw2/rw3 oracle benchmark

**Step>0 semantic results (EXP-RW-027, n=21):**
- Cosine top-5: 71.4% [50%, 86%] — comparable to step-0's 61.5%
- Oracle qualified: 42.9% (but inflated by label noise — real ceiling is higher)
- Step>0 is not a harder regime for cosine selection

**Implication:** step>0 is no longer a blocker for moving the rewrite lane forward. It is now a parallel fidelity track. The next semantic experiments should use it, but not wait on it.

### 8. Harder rewrite tiers (rw2-rw3) — ✅ COLLAPSED ON CURRENT STEP-0 SLICES

**rw1 conclusion (EXP-RW-029/030):**
- cosine premise selection remains strong
- tier-conditioned default direction carries almost all of the signal
- do not build a standalone direction head now

**Current state:**
- `rw2`: `args_necessary` is only 4% on the settled step-0 slice; bare-premise cosine is the correct executor
- `rw3_bare`: full sequential cosine reaches 70.7%; composition is not the bottleneck
- `rw3_with_args`: `args_necessary` is 3%; sequential bare cosine still dominates oracle

**Decision:** do not build rewrite-specific learned decoders now. The rewrite family is operationally
collapsed to the cheap sequential cosine executor. The remaining rewrite gap is state/lowering
fidelity, not premise ranking, direction prediction, argument construction, or composition control.

### 9. Temporal controller after executor competence

Do not spend more cycles on TC until `local_close` has at least one more productive family beyond cosine_rw. The TC is correctly routing (EXP-3.2d), but executor quality is the bottleneck.

Trigger: re-evaluate TC after executable-action selection or another local family produces nonzero theorem-level
gain beyond `cosine_rw`. The nearest candidate is no longer a new global lane; it is structured
`apply` / `refine` action selection over cosine shortlists.

### 9. Learned decoder repositioning

Learning's value is no longer on rw0 symbol selection or standalone rw1 direction. Reposition to:
- **Executable selection**: unification-aware / elaboration-aware selection for `apply`, structured action selection for `refine`
- **Controller-visible move selection**: choose among `SubtaskIR`-typed local transformations
- **Residual-conditioned orchestration**: route among finisher, scaffold, helper, and future specialist lanes
- **Family-specific decoders**: only where scoping plus reranking still fails

Hard bar: no learned decoder work unless it beats cosine at matched Lean-call budget.

## Assumptions and Defaults
- Canonical source: raw LeanDojo traces
- Cosine-beam-verify is the proven deployment pattern for local lanes
- `ActionIR` is the authoritative target; tactic strings are lowering artifacts
- theorem-level retrieval is temporal orchestration, not local step resolution
- temporal-controller benefit is deferred until local executor coverage broadens
- learning targets what cosine still does not do: residual-conditioned orchestration, planner-visible move selection, and executable action selection
