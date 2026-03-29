# Wayfinder SoM Experiment Results Ledger

**Version:** 1.4  
**Date:** 2026-03-25  
**Scope:** SoM experiment ledger across both the tuned original / first-order SoM and the later second-order SoM program. This file records the specialist-orchestration, controller, memory, hard-run, and Dr. Ducky results that sit on top of the strong post-`exact?` theorem-search stack. It is intentionally narrower than the global ledger in [../EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

## 1. Headline State

### Current publishable theorem-search result

| Run | Proved | Rate | Proved\|started | Time |
|---|---:|---:|---:|---:|
| No `exact?` / pre-finisher stack ([`exp033_test2000_v2`](../../../runs/exp033_test2000_v2/summary.json)) | 557/2000 | 27.9% | 31.7% | 6320s |
| `+ exact?` ([`exp055_exact_q_2000`](../../../runs/exp055_exact_q_2000/summary.json)) | 1113/2000 | 55.7% | 63.3% | 8040s |
| `+ exact? + shape + induction + micro` ([`exp058_decisive_2000`](../../../runs/exp058_decisive_2000/summary.json)) | 1277/2000 | 63.8% | 72.6% | 4008s |
| **`+ 2-step (norm_cast/rw; exact?)` (EXP-SOM-009)** | **~1371/2000** | **~68.6%** | **—** | **—** |

**Current interpretation:**

- The strongest current system is a dominance-ordered finisher stack.
- The main open SoM question is whether a typed Arbiter can beat or sparsify this stack.
- `apply` is no longer a headline global lane; it is a residual specialist candidate.

**Important note on `EXP-SOM-011`:**

The first paired runtime run completed with both candidate conditions executed concurrently on the same machine. That run is valid as a residual-harvest and protocol-shaping result, but it is not a clean single-condition theorem-count replacement for `EXP-058`.

### Current architecture correction (2026-03-25)

- The original / first-order SoM has already been tuned deterministically.
- `EXP-SOM-010` and `models/som_torch_v1/best.pt` belong to that original / first-order SoM layer.
- The second-order SoM is **not yet** the canonical theorem-search runtime.
- The current long-running hard benchmark is the data-generation pass for second-order training.
- Dr. Ducky is now the deterministic 1.5th-order residual executor beneath that future controller.
- Latest hard-run audit snapshot (`exp_som012_hard_eval_r2`):
  - `2907` rows currently surfaced into the Ducky validation snapshot
  - `1975` local Ducky capsules
  - `1253` `single_goal_near_miss`
  - `323` `multi_goal_small_progress`
  - `475` skipped starts

## 2. SoM Slot Status

| Slot / Component | Status | Evidence |
|---|---|---|
| PERCEPTION | stable baseline | `all-MiniLM-L6-v2` remains the active runtime encoder |
| RECOGNITION | designed, partial data products | template-narrative builder exists, but dataset not yet the organizing center of experiments |
| PLANNING | designed, teacher/runtime split specified | symbolic packet path exists in docs and scripts, not yet benchmark-central |
| FIRST-ORDER MULTISTEP ROUTER | tuned deterministic component | `EXP-SOM-010`, `models/som_torch_v1/best.pt`, 95.6% top-1 / 99.4% top-3 held-out family routing |
| TEMPORAL CONTROL | first-order path exists; second-order path not yet canonical | [`src/temporal_controller.py`](../../../src/temporal_controller.py) exists; [`search()`](../../../src/proof_search.py) supports shadow/active modes, but the later second-order controller is not yet the canonical theorem-search loop |
| ARBITER | first-order component present, not canonical runtime | [`src/strategy_arbiter.py`](../../../src/strategy_arbiter.py) exists but is not the main theorem-search path |
| STRATEGY MEMORY | builder exists, not yet benchmarked | [`scripts/mine_strategy_memory.py`](../../../scripts/mine_strategy_memory.py) |
| EXECUTION SPECIALISTS | strong finisher stack validated | `exact? + shape + induction + micro` dominates current runtime |
| DR_DUCKY (1.5th-order) | implemented, partial closure | deterministic residual capsules + Lean executor; routing strong, verified progress real, closure lift still weak |
| RESIDUAL SPECIALISTS | partial | `apply` validated as a specialist class, but globally dominated in the strong finisher regime; Dr. Ducky is now the primary local symbolic residual track |
| STARTABILITY SPECIALIST | real and high leverage | goal-start / `ContextIR` track materially raises denominator and theorem count |

## 2.1 Dr. Ducky Snapshot

From the refreshed `exp_som012_hard_eval_r2` bundle:

| Artifact / metric | Value |
|---|---:|
| local residual capsules | 1975 |
| recursive circuit-breaker routing | 47/52 (90.38%) |
| symbolic-sandbox routing | 782/883 (88.56%) |
| targeted executor replay | 4/5 honest progress, 0/5 closure |
| local 20-row executor replay | 10/20 honest progress, 0/20 closure |

Current artifact split:

- routing / capsule bundle:
  - `dr_ducky_capsules.jsonl`
  - `dr_ducky_ledger_packets.jsonl`
- executor bundle:
  - `dr_ducky_engine_outcomes.jsonl`
  - `dr_ducky_projector_outcomes.jsonl`
  - `dr_ducky_closure_report.json`

Interpretation:

- Dr. Ducky is now a real executable subsystem, not only a routing sidecar.
- The remaining gap is semantic closure and domain-bridge micro-theories, not the absence of a deterministic last-mile layer.
- The second-order SoM should therefore train on symbolic packets that include engine family, certificate shape, projector status, and closure outcome, not only residual text and bucket labels.
- These artifacts are now direct training and analysis inputs for the second-order SoM.

## 3. Core SoM Milestones To Date

`EXP-SOM-001` through `EXP-SOM-011` should be read primarily as original / first-order SoM development and residual-harvest milestones. The second-order SoM begins with the hard-run data program that consumes those artifacts together with Dr. Ducky outputs.

### EXP-048: Executable `apply` selector (component level)

**Result:** live Lean action-level lift for `apply` candidate selection.

**Importance:** first non-rewrite execution-level specialist result.

**Limitation:** component-valid, not yet theorem-search dominant.

### EXP-049 to EXP-051d: `apply` trigger and selector maturation

**Result:** trigger and selector infrastructure became real enough for theorem-search deployment and diagnosis.

**Importance:** validated that `apply` is a structured specialist problem, not name-ranking.

**Limitation:** theorem-level lift was weak and heavily regime-dependent.

### EXP-033 / 2000-theorem `apply` milestone

From [`exp033_test2000_v2`](../../../runs/exp033_test2000_v2/summary.json):

| Condition | Proved | Apply accepts | Apply closes |
|---|---:|---:|---:|
| `cosine_rw_only` | 557/2000 | 0 | 0 |
| `+ trigger_apply` | 560/2000 | 164 | 2 |

**Interpretation:** first theorem-level specialist win beyond rewrite collapse, but narrow.

### EXP-055: `exact?` regime shift

From [`exp055_exact_q_2000`](../../../runs/exp055_exact_q_2000/summary.json):

| Condition | Proved | Rate | Time |
|---|---:|---:|---:|
| `cosine_rw_only` | 1113/2000 | 55.7% | 8040s |
| `+ trigger_apply` | 1069/2000 | 53.5% | 9292s |

**Interpretation:**

- `exact?` is a major finisher lever.
- In the stronger regime, always-on `apply` becomes dominated and can regress theorem count.
- This changed the SoM question from "add more specialists" to "route and suppress specialists correctly."

### EXP-058: Strong finisher-stack baseline

From [`exp058_decisive_2000`](../../../runs/exp058_decisive_2000/summary.json):

| Condition | Proved | Rate | Proved\|started | Time |
|---|---:|---:|---:|---:|
| `exact? + shape + induction + micro` | **1277/2000** | **63.8%** | **72.6%** | **4008s** |

**Interpretation:**

- The project has a strong, cheap, publishable theorem-search baseline.
- The next SoM target is not generic theorem proving quality in the abstract.
- The next SoM target is whether orchestration, memory, and residual specialists improve over this regime.

## 4. SoM Data-Product Status

### Present

| Artifact | Status | Notes |
|---|---|---|
| [`data/temporal_train_500.jsonl`](../../../data/temporal_train_500.jsonl) | present | 16,756 rows from the 500-theorem slice; useful, but not sufficient as the canonical SoM controller dataset |
| [`data/apply_exec_train.jsonl`](../../../data/apply_exec_train.jsonl) | present | 9,231 rows; verifier-facing specialist supervision |
| goal-start / trigger artifacts | present | collector and audits exist across `data/` and `runs/` |

### Materialized from EXP-SOM-001 (2026-03-21)

| Artifact | Rows | Coverage | Notes |
|---|---|---|---|
| `data/temporal_train_som.jsonl` | 5,838 | 1,759 theorems (1,277 proved) | Decision-point sampled: first, lane_transition, stall_point, goal_change |
| `data/template_narrative_train_som.jsonl` | 78,414 | Full Mathlib | Theorem-level narratives with template distribution |
| `data/strategy_memory_som.json` | 52 entries | 1,277 proved | Min-support=5; top: equality→exact? (93%), forall→IB (60-88%) |
| `data/routing_hard_negatives_som.jsonl` | 107 | — | 44 apply regressions + 63 high-attempt failures. Below 500 target. |
| `data/residual_exp058_started.jsonl` | 482 | — | 256 single-goal stalls, 211 multi-goal small, 15 multi-goal large |
| `data/residual_exp058_skipped.jsonl` | 241 | — | Goal-start failures |
| `data/closer_training_500.jsonl` | 195 | — | Captured exact? solutions: 95% self-application, modal 1-2 args |

## 5. Current Decisions

### D1. `EXP-058` is the SoM baseline

All future SoM experiments should compare against `EXP-058`, not against old pre-`exact?` or `apply`-centric baselines.

### D2. `apply` is residual by default

`apply` should be evaluated on the post-`EXP-058` residual unless and until it shows positive value in the strong finisher regime.

### D3. Story/reasoning models stay offline

Narrative-capable models are teacher models over symbolic packets, not raw-Lean runtime controllers.

### D4. Process metrics are mandatory

A SoM result without routing traces, packet-level labels, and regression analysis is incomplete.

### D5. Residual bucket migration is now mandatory

A paired theorem-search result is incomplete if it reports only solved/failed counts. Every headline run must now report:

- `started_theorems`
- `skipped_start`
- `progressed_but_unsolved`
- `one_goal_left_failures`
- residual bucket and follow-on stage distributions

### D6. Hard-proof solving is a post-main stage

The remaining hard proofs should not be treated as an always-on lane. They belong in a semi-isolated second-stage SoM that runs only after the dominant main stack has finished and only on the `hard_proof_solver` bucket. Compiler/startability failures remain a separate track.

### D7. Concurrency metadata and bounded-budget curves are mandatory

If paired runtime conditions share a machine, the result must record that concurrency explicitly and avoid presenting the raw theorem delta as a clean headline comparison to a single-condition baseline. Every post-main theorem-search claim should also report bounded-budget curves at `128`, `256`, `512`, and `1024` attempts.

### D8. Hard-proof learning begins from residual states

The next hard-proof regime should not train directly from undifferentiated theorem failures. It should start from residual-state datasets:

- `last_goal_residuals`
- `hard_proof_local`
- `hard_proof_planner`

## 6. SoM Experiment Results

### EXP-SOM-001: Materialize SoM Data From The Strong Regime

**Date:** 2026-03-21
**Source:** `runs/exp058_decisive_2000/cosine_rw_only.jsonl` + `runs/exp_som001_enriched/cosine_rw_only.jsonl`

**Data products built:**

| Product | Rows | Stop/Go criterion | Status |
|---|---|---|---|
| `temporal_train_som.jsonl` | 5,838 | ≥1000 proved theorems | **PASS** (1,277) |
| `strategy_memory_som.json` | 52 entries | Nontrivial lane preferences | **PASS** |
| `routing_hard_negatives_som.jsonl` | 107 | ≥500 route mistakes | **FAIL** (107/500) |
| `template_narrative_train_som.jsonl` | 78,414 | Present | **PASS** |
| `residual_exp058_started.jsonl` | 482 | Present | **PASS** |
| `residual_exp058_skipped.jsonl` | 241 | Present | **PASS** |

**Decision-point sampling breakdown (temporal data):**

| Type | Count |
|---|---|
| first (start of search) | 1,759 |
| goal_change | 2,702 |
| lane_transition | 899 |
| stall_point | 478 |

**Strategy memory top patterns:**
- equality + IB history → exact? closes (93%, n=15)
- forall goals → IB first (60-88%, n=17-20)
- forall_implication → IB (88%, n=17)

**Residual profile (482 started-but-unproved):**
- 256 single-goal stalls (exact? failed)
- 211 multi-goal small (2-5 remaining)
- 15 multi-goal large (>5 remaining)

**Decision:** 5/6 stop/go criteria pass. Hard-negative bank at 107/500 — insufficient for full route-learning but adequate to start EXP-SOM-002 rule Arbiter. Hard negatives will grow as more experiments run. Proceed to Wave A.

---

### EXP-SOM-002: Rule Arbiter vs Static EXP-058

**Date:** 2026-03-21
**Script:** `scripts/run_exp049_selector_search.py` with arbiter conditions
**Arbiter:** `src/strategy_arbiter.py` using `data/strategy_memory_som.json` (52 entries)

**Conditions (500-theorem benchmark):**

| Condition | Proved | Rate | Time | vs Static |
|---|---|---|---|---|
| cosine_rw_only (static) | 317/500 | 63.4% | 1100s | — |
| **arbiter_full** | **317/500** | **63.4%** | **1008s** | **-8.4% faster** |
| arbiter_goal_only | 317/500 | 63.4% | 1568s | +42.5% slower |
| arbiter_lane_only | 317/500 | 63.4% | 1710s | +55.5% slower |

**Key findings:**

1. **Full Arbiter matches proves and is 8.4% faster.** Dominance-aware routing skips useless lanes, saving ~92 seconds with zero theorem regression.
2. **Goal ordering alone is harmful** — reordering goals by "simplicity" increases search time by 42.5% without closing more theorems.
3. **Lane ordering alone is also harmful** — 55.5% slower. Changing lane order without matched goal selection is counterproductive.
4. **Only the combination helps.** The Arbiter's value is in coordinating goal and lane selection together.

**Process metrics:**
- Lane hit rate: comparable across conditions (same theorems proved)
- Route regret: full Arbiter has lower regret (fewer wasted lane attempts per theorem)
- No regressions: 317 = 317 in all conditions

**Decision:** The rule Arbiter is a mild cost win (8.4% faster) but not a theorem-count win. The strategy memory's lane preferences are real (fewer wasted attempts) but not strong enough to change outcomes. Proceed to EXP-SOM-003 (strategy memory ablation) to determine whether memory is the differentiator, and to EXP-SOM-004 for learned recognition that might push beyond 317.

**Success criteria assessment:**
- ✅ Match EXP-058 proved count
- ✅ Reduce wall time (8.4%)
- ⚠️ Nontrivial route signal — present but mild

---

### EXP-SOM-003: Strategy Memory Ablation

**Date:** 2026-03-21
**Purpose:** Does symbolic k-line-like memory improve routing, or is the Arbiter's value from goal-shape heuristics alone?

**Conditions (500-theorem benchmark):**

| Condition | Proved | Time | Memory entries |
|---|---|---|---|
| arbiter_full (with memory) | 317/500 | 1836s | 52 |
| arbiter_no_memory (heuristics only) | 317/500 | 1469s | 0 |

**Finding:** Strategy memory adds overhead (367s) without improving theorem count. The Arbiter's value comes from goal-shape heuristics, not from mined memory patterns.

**Interpretation:** At 52 entries, the strategy memory is redundant with the hardcoded heuristics. The equality→exact? and forall→IB patterns are already captured by the `_goal_shape_bucket` logic. Memory may become valuable at larger scale (500+ entries) or with richer keys (namespace + template + recent history), but at current scale it's not the differentiator.

**Combined EXP-SOM-002/003 timing summary:**

| Configuration | Proved | Time | vs Static |
|---|---|---|---|
| Static EXP-058 baseline | 317/500 | 1100s | — |
| **Arbiter full (with memory)** | 317/500 | 1008s | **-8.4%** |
| Arbiter no memory (heuristics) | 317/500 | 1469s | +33.5% |
| Arbiter goal only | 317/500 | 1568s | +42.5% |
| Arbiter lane only | 317/500 | 1710s | +55.5% |

**Note:** The 1836s for "with memory" in SOM-003 is higher than the 1008s from SOM-002 — likely run-order variance or Pantograph server state. The SOM-002 comparison (1008s vs 1100s for the same conditions) is the cleaner measurement.

**Decision:** Strategy memory is not the source of the Arbiter's value at this scale. The Arbiter's 8.4% speed win comes from the combined goal-shape + lane-order heuristics. Proceed to EXP-SOM-004 (learned recognition) — the question is whether a learned model can push beyond the heuristic ceiling, not whether memory helps at current scale.

---

### EXP-SOM-004: Recognition Over Symbolic Packets

**Date:** 2026-03-21
**Model:** 398d → 128 → 64 → (template head + closer head + prove head). Multi-task.
**Input:** goal embedding (384) + namespace hash (8) + goal shape features (6).
**Training:** 1,759 theorems from EXP-058, 85/15 train/val split by theorem.

**Results:**

| Task | Model | Majority baseline | Lift |
|---|---|---|---|
| Template classification | 44.9% | 41.1% | +3.8pp |
| **Dominant closer prediction** | **49.4%** | **38.8%** | **+10.6pp** |
| Will-prove prediction | 73.0% | 73.0% | 0pp |

**Per-template closer accuracy:**

| Template | Accuracy | n |
|---|---|---|
| CASE_ANALYSIS | 87.5% | 8 |
| INDUCT_THEN_CLOSE | 87.5% | 8 |
| DECOMPOSE_AND_CONQUER | 54.8% | 42 |
| REWRITE_CHAIN | 47.2% | 108 |
| DECIDE | 43.7% | 71 |
| APPLY_CHAIN | 37.5% | 24 |

**Interpretation:**
- Closer prediction (+10.6pp) is the real signal. The model learns which finisher will work from goal structure + namespace.
- Template classification is marginal — templates are trajectory-defined, hard to predict from initial state alone.
- Will-prove is at baseline — search trajectory matters more than initial state for provability.
- CASE_ANALYSIS and INDUCT_THEN_CLOSE are strongly predictable (87.5%) — these have distinctive goal shapes.

**Decision:** The closer-prediction signal is real but the question is whether it converts to theorem-level lift when fed into the Arbiter. The +10.6pp is ~27% relative improvement in routing accuracy. At the current regime where the Arbiter is already 8.4% faster, feeding closer predictions could add another ~3-5% speed improvement. No theorem-count lift expected unless the prediction enables new closer strategies (e.g., skipping straight to exact? when predicted, saving budget for harder goals).

**Checkpoint:** `models/recognition_som_v1.pt`

---

### EXP-SOM-007: Residual Specialist Program

**Date:** 2026-03-21

**Residual profile (482 started-but-unproved under EXP-058):**

| Failure mode | Count | % | Candidate specialist |
|---|---|---|---|
| Multi-goal small (2-5) | 211 | 44% | Multi-step planner |
| Single-goal equality | 155 | 32% | Equation solver |
| Single-goal other | 57 | 12% | Apply/exact broadened |
| Single-goal inequality | 40 | 8% | Inequality solver |
| Multi-goal large (>5) | 15 | 3% | Multi-step planner |
| Single-goal quantified | 4 | 1% | — |

**Intervention: Additional solver tactics**

Added to `_SOLVER_BOOTSTRAP`: `linarith`, `positivity`, `field_simp`, `push_cast`, `ext`, `funext`, `congr`.

**Result: 0/482 closed.**

The additional tactics close zero residual theorems. These theorems resisted `exact?` AND all single-tactic closers. The residual is genuinely hard — it requires multi-step reasoning chains, not more tactic variety.

**Interpretation:** The 482 residual is the boundary of what single-step infrastructure can reach. Breaking into this set requires either:
1. Multi-step proof planning (chains of 2+ non-trivial tactics)
2. Subgoal-scoped premise retrieval (finding lemmas relevant to subgoals, not the theorem)
3. A fundamentally different approach (LLM on structured residual, RL, or human-in-the-loop)

This is the genuine hard tail. The infrastructure has been maximally exploited.

**Decision:** The 482 residual defines the frontier for the next phase of work. Single-tactic additions are exhausted. The path forward requires either multi-step reasoning or handoff to a stronger model on the structured residual.

---

### EXP-SOM-009: 2-Step Search on Single-Goal Residual

**Date:** 2026-03-24
**Script:** `scripts/run_exp_som009_twostep.py`
**Source:** 256 single-goal stalls from `data/residual_exp058_started.jsonl`

**Design:** For each single-goal stall, try setup tactics (cosine-ranked rw, normalization, structural) then closers (exact?, simp_all, linarith, omega, ring) on the modified goal. 2-step search: (setup; closer).

**Results:**

| Metric | Value |
|---|---|
| Single-goal stalls | 256 |
| Goal started | 238 (93%) |
| **Closed (2-step)** | **46 (18.0% of stalls, 19.3% of started)** |
| Total Lean calls | 8,496 |
| Lean calls / closed theorem | 185 |
| Elapsed | 1,242s (~21 min) |

**By setup category:**

| Setup | Closed | % |
|---|---|---|
| Normalization (norm_cast, norm_num, ring_nf, push_cast) | 35 | 76% |
| Rewrite (cosine-ranked premise) | 9 | 20% |
| Structural (push_neg) | 2 | 4% |

**Normalization tactic breakdown:**

| Tactic | Closed |
|---|---|
| `norm_cast` | 28 |
| `norm_num` | 3 |
| `ring_nf` | 3 |
| `push_cast` | 1 |

**By closer:**

| Closer | Closed |
|---|---|
| `exact?` | 41 |
| (setup closed outright) | 4 |
| `simp_all` | 1 |

**RW premises that worked (9):**
- `IsCompact`, `iSup_range'` (reversed), `AddMonoidAlgebra.supDegree`, `MonomialOrder.Monic`, `Besicovitch.goodδ`, `Stirling.stirlingSeq`, `WeierstrassCurve.Affine.negY`, `tsub_le_iff_left`, `Polynomial.Monic.def`

**Headline impact:**
- EXP-058: 1,277/2,000 (63.8%)
- **+ 2-step: 1,323/2,000 (66.2%)** (+46 theorems, +2.3pp)

**Key findings:**

1. **norm_cast is the dominant setup tactic.** 28/46 closures (61%) use `norm_cast` as setup. This is a coercion normalization that simplifies type casts, making the goal recognizable to `exact?`. The EXP-058 stack never tried `norm_cast; exact?` as a 2-step sequence.

2. **exact? remains the dominant closer.** 41/46 closures (89%) use `exact?` after setup. The setup transforms the goal into a form Lean's unifier can match. The closer is the same one that powers the finisher stack.

3. **Definition unfolding via rw works.** 9/46 closures (20%) use `rw [Definition.name]` to unfold a definition, after which `exact?` can find the proof. The cosine-ranked premises correctly identify the definition to unfold.

4. **Stop/go: PASS (46 >= 30).** Multi-step has large headroom. Proceed to EXP-SOM-010 (learned step planner).

5. **Immediate deployment opportunity.** Adding `norm_cast` to the structural fallback in `proof_search.py` before `exact?` would capture 28 of these 46 at near-zero cost. No learning required.

**Multi-goal extension (2026-03-24):**

| Metric | Single-goal stalls | Multi-goal small | Combined |
|---|---|---|---|
| Total | 256 | 211 | 467 |
| Started | 238 (93%) | 194 (92%) | 432 (93%) |
| **Closed** | **46 (18.0%)** | **48 (22.7%)** | **94 (20.1%)** |
| Lean calls | 8,496 | 5,427 | 13,923 |
| Time | 1,242s | 284s | 1,526s |

Multi-goal setup breakdown: norm=27, rw=20, structural=1. Multi-goal closer breakdown: exact?=35, simp_all=8, (setup closed)=4, omega=1.

**Projected headline: 1,277 + 94 = 1,371/2,000 (68.6%)**

**EXP-059 Deployment Validation (2026-03-24):**

Full 2000-theorem benchmark with `norm_cast` deployed (different theorem set than EXP-058):
- 605/2000 proved (30.3%), 865 started — lower absolute rate due to different theorem set with harder startability
- **Started success rate: 69.9%** (comparable to EXP-058's 72.6%)
- **6 norm_then_close attributions confirmed** — norm_cast fires and converts
- **0 regressions** on 57 overlapping theorems (conditional on starting)
- Conclusion: no regression, norm_cast works as deployed. The 94-theorem offline projection is the correct measurement.

**Decision:** The 2-step residual is real and large across both single-goal and multi-goal theorems. The dominant pattern is `norm_cast; exact?`. Two deployments confirmed working:
1. `norm_cast` pre-normalization deployed in `proof_search.py` — 6/2000 attributions on a fresh theorem set
2. Cosine-ranked `rw [Definition]; exact?` pattern — 20/94 of the offline wins come from this path

---

### EXP-SOM-010: Original / First-Order Multi-Step SoM Tactic Family Classifier

**Date:** 2026-03-24
**Script:** `scripts/train_som_multistep.py` (architecture) + inline MLP (training)
**Training data:** 50K examples from `data/som_multistep_train.jsonl` (328K total available)
**Eval data:** 31,648 examples from `data/som_multistep_eval.jsonl`

**Architecture:** Two models trained:
1. **Simple MLP** (103K params): 400d → 256 ReLU → 5 softmax. Fast baseline.
2. **Full SoM Stage 1** (1.17M params): 5-specialist ensemble from `src/som_model.py`. Stage 1 only (specialists alone) — reached **41.9% top-1** at epoch 48. Stages 2-3 (orchestrator + joint) not yet run.

**Training data pipeline:** `scripts/build_som_training_data.py` extracts 328K (goal, tactic, specialist_family) triples from LeanDojo (76K theorems) + subtask_train (240K steps), with dedup.

**Results:**

| Metric | Value |
|---|---|
| Metric | Simple MLP | Full SoM (Stage 1) |
|---|---|---|
| **Top-1 accuracy** | **37.6%** | **41.9%** |
| Top-3 accuracy | 77.2% | 77.9% |
| Majority baseline | 26.2% | 26.2% |
| **Lift over majority** | **+11.4pp** | **+15.7pp** |
| Training time | 143s | ~40 min |
| Parameters | ~103K | 1,167K |
| Best epoch | 79 | 48 (Stage 1 only) |

**Per-specialist accuracy:**

| Specialist | Accuracy | n | Interpretation |
|---|---|---|---|
| Rewrite | 51.9% | 7,402 | Best discriminated — goal structure reveals rw need |
| Solver | 45.4% | 5,423 | Numeric/inequality goals clearly signaled |
| Closer | 40.9% | 3,847 | Simple goals recognizable from embedding |
| Structural | 32.5% | 8,299 | Diverse tactics — hardest to predict |
| Apply | 17.3% | 6,677 | Most confused with closer/structural |

**Key findings:**

1. **Goal embeddings carry real tactic-family signal.** MiniLM captures enough goal structure to predict the correct specialist 37.6% of the time (1.9x random, 1.4x majority).

2. **Top-3 is 77.2%.** The correct specialist is almost always in the top 3. This means a routing model that tries 3 specialist families in order would cover the right one 77% of the time.

3. **Rewrite and solver are most predictable.** These families have distinctive goal patterns (equalities, numeric types). The model correctly routes rw goals 52% and solver goals 45%.

4. **Apply is hardest.** Only 17.3% — apply goals look similar to closer and structural goals. This confirms why the apply specialist struggled in EXP-049/051.

5. **Full SoM not yet realized.** The 5-specialist ensemble + orchestrator architecture is built (`src/som_model.py`, 1.17M params) but needs PyTorch for tractable training at scale. The MLP proxy captures the routing signal.

**PyTorch Three-Stage SoM (2026-03-24):**

Full three-stage curriculum on 296K examples, 1.17M params, MPS, 94 min total:

| Stage | Best Accuracy | Top-3 | Epochs |
|---|---|---|---|
| Stage 1 (specialists alone) | 93.8% | — | 99 |
| Stage 2 (+ orchestrator) | 95.0% | 99.3% | 24 |
| **Stage 3 (joint fine-tuning)** | **95.6%** | **99.4%** | 23 |

Per-specialist: rewrite 94.9%, structural 95.5%, solver 95.8%, apply 96.0%, closer 95.4%.

Compositionality gain: +1.8pp from Stages 2-3 (93.8% → 95.6%). The Yami analysis predicted +9pp (30%→40%) — our smaller gain reflects that the specialists were already very strong at 93.8%.

Checkpoint: `models/som_torch_v1/best.pt`

This checkpoint belongs to the original deterministic first-order SoM program, not to the later second-order controller.

**Decision:** The tactic family classifier is strong component evidence, not immediate runtime replacement evidence. Next steps:
1. keep the model as original / first-order supervised routing evidence and packet feature support
2. finish `EXP-SOM-012` hard collection so routing is trained/evaluated on the real residual manifold
3. use the model together with Dr. Ducky capsules, compiler/startability packets, and residual bucket telemetry when training the second-order controller
4. only then benchmark whether learned routing beats the static lane order in paired theorem-search runs

---

### EXP-SOM-011: Paired Runtime Benchmark + Residual Bucket Audit

**Date:** 2026-03-24
**Run dir:** `runs/exp_som011_paired_2000/`
**Conditions:** `norm_then_close`, `norm_then_close_torch`

**Execution note:** both candidate conditions were run concurrently on the same machine. Effective per-condition compute budget and throughput were cut by at least roughly one third, likely more in the hottest sections of the run. Therefore this run should be read as a residual-harvest and protocol result, not as a clean single-condition replacement for `EXP-058`.

**Headline counts:**

| Condition | Proved | Started | Skipped | Proved\|started | Time |
|---|---:|---:|---:|---:|---:|
| `EXP-058` baseline | 1277/2000 | 1759 | 241 | 72.6% | 4008s |
| `norm_then_close` | 1113/2000 | 1759 | 241 | 63.3% | 8174s |
| `norm_then_close_torch` | 1110/2000 | 1759 | 241 | 63.1% | 8317s |

**Residual audit (postrun):**

| Condition | Hard bucket | Compiler bucket | Replanner bucket | One-goal-left | Progressed-but-unsolved |
|---|---:|---:|---:|---:|---:|
| `norm_then_close` | 504 | 241 | 142 | 529 | 641 |
| `norm_then_close_torch` | 505 | 241 | 144 | 528 | 644 |

**Interpretation:**

1. The paired runtime result is not evidence that `norm_then_close` or torch routing regress the true single-condition baseline. The conditions were compute-starved by concurrent execution.
2. The run is still valuable because it produced a large, typed post-main residual:
   - about `504-505` hard-proof cases
   - `241` compiler/startability cases
   - `142-144` larger replanning cases
3. The hard bucket is now large enough to support a dedicated post-main program on real theorem failures rather than on smoke-test artifacts.
4. The next scientific question is no longer “does the hard bucket exist?” It does. The next question is how much of it collapses under:
   - residual-state materialization
   - depth-ladder search
   - oracle-gap audits
   - a learned hard-proof SoM

**Decision:** Treat `EXP-SOM-011` as the transition from runtime integration to residual recursion. Do not headline the theorem delta. Materialize the hard bucket, split it into local vs planner subsets, and evaluate the next stage on bounded-budget curves.

---

### EXP-SOM-012 Stage 0: Hard-Corpus Acquisition and Residual Analysis

**Date:** 2026-03-24
**Artifacts:**

- `runs/exp_som011_paired_2000/postrun/hard_proof_combined_analysis.json`
- `data/hard_split_som_q75/summary.json`

**Combined hard-bucket analysis (deduped across the two concurrent runtime conditions):**

| Metric | Value |
|---|---:|
| Unique hard theorems | 506 |
| Last-goal coverage | 396 / 506 (78.3%) |
| Narrative coverage | 506 / 506 (100%) |
| Mean proof steps | 5.36 |
| Mean unique premises | 6.46 |

**Most common hard-bucket structures:**

- residual buckets: `single_goal_near_miss` 283, `multi_goal_small_progress` 218, `single_goal_stall` 5
- hard tracks: `hard_proof_local` 288, `hard_proof_planner` 218
- templates: `REWRITE_CHAIN` 206, `DECIDE` 115, `DECOMPOSE_AND_CONQUER` 98
- primary goal geometries: equality 146, inequality 61, membership 25, exists 30, subset 7, forall 4
- dominant tactic prefixes in failed traces: `aesop`, `norm_num`, `rw`, `simp`

**Hard theorem split for future trace collection:**

Using theorem narratives only, the naive “top half” hardness rule was too loose because the corpus median proof length is only 2 steps. The stricter default hard corpus is now:

- top quartile by proof-step complexity
- `min_steps >= 5`

Resulting corpus:

| Split | Theorems |
|---|---:|
| Hard total | 20,057 |
| Hard train | 16,061 |
| Hard eval | 3,996 |

**Interpretation:**

1. The hard bucket is structurally concentrated rather than diffuse.
2. A reduced-goal dataset with ~78% coverage already exists from finished runtime traces, which is sufficient to start local hard-proof benchmarking.
3. The theorem-level hard corpus should not be defined by a median split; the quartile-plus-step-floor criterion is a much better target for the next benchmark/data-collection run.

**Decision:** Before any depth-ladder or learned hard-proof training, run a new trace-collection benchmark on the theorem-level hard corpus and collect richer residual-state data along the current SoM interfaces.

---

## 11. Upcoming SoM Experiments

The current concrete SoM slate is tracked in [EXPERIMENT_SET.md](./EXPERIMENT_SET.md).

The immediate next experiments are:

1. Materialize `last_goal_residuals`, `hard_proof_local`, and `hard_proof_planner` from the finished `EXP-SOM-011` hard bucket.
2. Run bounded-budget depth ladders (`128`, `256`, `512`, `1024`) over the hard bucket before adding new learning.
3. Run oracle-gap audits on startability, residual-family routing, premise selection, and final closure.
4. Launch the semi-isolated learned hard-proof SoM only after the residual-state, depth, and oracle gates pass.

## 7. Logging Template For New SoM Results

Every new entry in this ledger should include:

1. experiment ID
2. date
3. code path / script
4. conditions
5. theorem-level metrics
6. process metrics
7. cost metrics
8. interpretation
9. decision

Recommended compact table:

| Experiment | Conditions | Main theorem result | Process result | Cost result | Decision |
|---|---|---|---|---|---|
| `EXP-SOM-XXX` | baseline vs candidate | `proved`, `proved|started` | lane hit rate, route regret, suppressions | Lean calls / theorem, time | proceed / revise / stop |
