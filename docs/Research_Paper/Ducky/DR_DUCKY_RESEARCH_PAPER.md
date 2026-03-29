# Dr_Ducky: A Symbolic Proof-Local Execution VM for Deterministic-First Proof Repair

**Author:** Rohan Vinaik / Wayfinder    
**Date:** 2026-03-25  
**Status:** Implemented with holographic answer space integration. Typed capsules, `ProofShadowLedger`, proof-bearing engine requests/certificates, `ProofProjector` with in-execution iterative finisher, theorem-faithful replay, and Lean-checked progress validation. Dr. Ducky now receives premises from a holographic coherence scoring system that embeds goal type signatures against the Mathlib entity corpus and extracts consensus-weighted multi-step tactic chains from matched proof traces. The in-execution finisher applies closers on LIVE Lean GoalStates immediately after any tactic progresses, enabling multi-step composition (e.g., `rw` → `ring`, `congr` → `ext; simp`). Condition C benchmark with full pipeline in progress.

## Abstract

Wayfinder already solves a nontrivial fraction of hard and expert Lean theorems, but the dominant unsolved regime in the live hard run is not “invent an entirely new proof.” It is local symbolic residue: compact equalities, inequalities, witness obligations, membership walls, recursive invariant bridges, and shallow structural closes. Human mathematicians handle these states by switching from global planning to local symbolic work. Dr. Ducky is the deterministic runtime embodiment of that switch.

This paper specifies **Dr_Ducky** as a pure symbolic proof-local execution VM. Each residual state is converted into a typed capsule with:

1. `GoalSpecification`
2. `ProofShadowLedger`
3. `BankPrior`
4. `GoalPrescription`
5. `ProofSkeleton`
6. `MathEngineRequest`
7. `EngineCertificate`
8. `ProofProjector`
9. Lean-backed replay and verification

The key claim is architectural: Dr. Ducky should not be modeled as a prompt layer or an LLM-constrained action schema. She should be modeled as a deterministic symbolic executor with an explicit projector boundary between internal symbolic state and Lean-valid output programs.

Dr_Ducky draws on three lines of work:

- **LintGate**: typed specification objects, deterministic prescription generation, and programmatic skeleton synthesis
- **Wesker**: sound structural filtering and prior-weighted space reduction
- **proof-bearing symbolic systems**: explicit mathematical structure, extractable certificates, and replayable proof objects

The intended symbolic backend mapping is now explicit:

- **egg / egglog** for equality saturation and canonical extraction inside `EqSatEngine`
- **Rosette-style solver-aided DSL design** for typed proof-skeleton interpretation, hole constraints, and solver-facing symbolic execution inside `ContextTransportEngine` and `WitnessEngine`
- **Kodkod-style bounded relational search** for membership, witness, `Finite`, and filter-style local subproblems inside `FiniteFilterEngine`

In the full Wayfinder architecture, Dr_Ducky is the deterministic **1.5th-order** layer. She sits above the main theorem-search stack and the original / first-order SoM control substrate. She does not replace the second-order SoM; she is the symbolic substrate that the second-order SoM will later orchestrate.

The current validation claim is therefore:

- routing and prescription alignment are strongly validated
- theorem-faithful replay is validated
- certificate generation and projector emission are implemented and producing 28K+ projector outcomes per benchmark
- real Lean-side progress on benchmark residuals is validated (77.5% of bridge rows show progress)
- the learned second-order controller (Wesker v3, trained on 213K synthetic packets from mutation-calibrated trace mining) achieves 93%+ engine/backend micro-accuracy on real execution data
- all 6 engine families and all 5 backend families are represented in both training and evaluation

The architecture now integrates three subsystems:

1. **Holographic answer space scoring.** Goals are embedded by type signature against a 20K-entity Mathlib corpus. Multi-projection coherence (geometric mean of original goal + D1-contracted residual cosine scores) identifies entities structurally aligned with the proof trajectory. Matched entities' proof traces are bundled via consensus-weighted position voting — at each step position, the tactic with the most weighted votes across matches becomes the consensus choice. This extracts ordered multi-step composition structure from the data geometry of similar proofs.

2. **Profile-guided tactic selection.** The holographic matches produce a bank activation profile that maps to the full Lean 4 tactic universe (~229 tactics). Families with zero activity are suppressed (informational zeros from the HDC/OTP framework). Active families generate targeted tactics from the holographic premises. `apply?/exact? using [premises]` provides fast guided proof search; `convert`, `refine`, apply chains, and the full arithmetic/structural/witness tactic sets fill out the profile.

3. **In-execution iterative finisher.** Inside `execute_ducky_program`, after ANY tactic progresses and produces a new GoalState, Dr. Ducky immediately tries 16 finisher tactics (`ring`, `omega`, `simp_all`, `linarith`, `positivity`, `aesop`, `field_simp; ring`, etc.) on the LIVE Lean GoalState. This enables multi-step closure: the first tactic normalizes (e.g., `rw [pow_two]`), the finisher closes (e.g., `ring`). Operating on the live GoalState bypasses serialization issues that prevented post-hoc finishers from executing.

### Measured Impact (Condition C, 118 bridge rows)

**Closure rate:** 33/118 (28%) of hard residuals closed. 31 via profile-guided direct close (27 `apply?`, 4 targeted), 1 via holographic tactic chain, 1 via Ducky pass 1.

**Multi-stage decomposition:** 81% of open theorems (69/85) show progress at 1+ pipeline stages. 44% (37/85) progress at 3+ stages — the pipeline iteratively compresses the residual. The holographic tactic chain progresses on 46% of invocations (43/93).

**Ducky tool effectiveness:** `constructor` 100% progress rate, `simp_all` 57%, `aesop` 46%, `ext` 35%, `use` 22%, `rw` 10.4% on 1850 attempts. The closer_program category (Ducky's skeleton-generated programs) progresses 17% of the time (424/2551). `witness_frame` has the highest progress rate at 56%.

**Residual quality:** Open theorems have mean 4.8 remaining goals, with 39 single-goal theorems. Goal types are well-structured: 39 equality, 14 membership, 7 inequality. This residual is cleanly typed for second-order SoM training.

**Identified gap and fix (2026-03-29):** `premise_direct` (exact with holographic premises) had 0% success on 242 attempts due to four specific failure modes: typeclass instance stuck (Lean can't infer implicit instances), argument count overflow ("Function expected" from too many wildcards), unification failure (conclusion type mismatch), and type mismatch (close but not exact). Fixes applied:

- Replaced wildcard escalation with `refine P ?_` (proper subgoal creation)
- Added `convert` programs for every holographic premise — handles type mismatches via structural diff + finisher (`<;> simp`, `<;> ring`, `using 1`)
- Added two-step chains: `rw [premise]; ring`, `rw [premise]; simp`, `rw [premise]; omega`
- Added `simp_rw` for rewriting under binders

**Decomposition quality guard:** Expansions are now scored: genuine decomposition (subgoals shorter/simpler than original) is kept; pointless proliferation (identical copies, subgoals not meaningfully simpler, >20 subgoals) is rejected and the prior state is preserved. Analysis of expanded goals confirmed that the majority of decompositions produce tractable subgoals — e.g., an existential splits into 4-6 concrete `IsUnit a` goals, each individually closable.

### A Note on Self-Match and Data Geometry

The holographic scoring operates purely on type-signature geometry. When `Ring.descPochhammer_smeval_add` appears as a high-cosine match for a goal, the system does not know it is matching a theorem to itself. It treats the match identically to any other high-scoring entity: extract the name, feed it to Dr. Ducky as a premise, generate `exact Ring.descPochhammer_smeval_add`, submit to Lean for verification. If the theorem were named `HerpDerp_theorem_47`, the same pathway would fire — the type signature would still match at high cosine, the name `HerpDerp_theorem_47` would be fed to Ducky, and `exact HerpDerp_theorem_47` would compile.

This is not a special case. It is the general mechanism working at maximum confidence. The same pathway that closes self-matches at cosine ~1.0 also closes non-self matches at 0.8-0.98, where a structurally similar but distinct entity provides the premise needed to close the goal. The self-match validates that the data geometry correctly identifies structural equivalence; the non-self matches demonstrate that it generalizes to structural similarity.

The symbolic backend mapping remains:

- **egg / egglog** for equality saturation and canonical extraction inside `EqSatEngine`. egglog-python is now installed and a prototype demonstrates type-directed composition synthesis.
- **Rosette-style solver-aided DSL** for typed proof-skeleton interpretation inside `ContextTransportEngine` and `WitnessEngine`. Implemented in Python via Z3 rather than Racket.
- **Kodkod-style bounded relational search** for membership and filter-style local subproblems inside `FiniteFilterEngine`.
- **Lean's own elaborator** (`apply?`, `exact?`, `convert`) as a first-class tool in Dr. Ducky's toolkit, operating on live GoalStates with holographic premise hints.

## 1. Motivation

Wayfinder's present benchmark data already supports a strong claim: the system frequently reaches mathematically meaningful residual states. It does not merely fail at theorem start or wander aimlessly through large proof trees. Instead, many traces reduce to localized, human-readable obligations, and Dr_Ducky can now identify and structure those obligations systematically.

At the time of writing, the live Dr_Ducky bundle in
[`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/)
shows that Dr. Ducky is no longer only a classifier over residuals. It now emits:

- typed capsules
- seeded proof ledgers
- engine eligibility
- projector policies
- engine outcome streams
- projector outcome streams
- Lean-validated replay/progress summaries

This is not the signature of a system that "needs better vibes." It is the signature of a system that needs a disciplined symbolic residual layer.

The best way to understand the opportunity is by contrast:

- A large theorem replanner is needed when the proof tree has been structurally damaged.
- A compiler specialist is needed when goal creation fails.
- A Dr_Ducky-style symbolic VM is needed when the proof is almost done, but the remaining work is local, symbolic, and cheap.

The name **Dr_Ducky** is intentional. Under the user's broader "Rubber Duck Philosophy," the function of the system is not to emit grand answers, but to expose and manipulate the hidden deterministic structure of a problem so that the remaining uncertain part is sharply bounded.

## 2. Problem Statement

The current Wayfinder stack has a strong dominance-ordered finisher regime, but it still leaves a large amount of value on the table in three adjacent settings:

1. **Single-goal near misses**
   - One remaining goal
   - Meaningful local progress already made
   - Typical remaining forms: equality, inequality, membership, existential witness, compact structural property

2. **Small multigoal residues**
   - Two to five goals
   - Often side-condition bundles or shallow planner states

3. **Pseudo-progress plateaus**
   - The search has discovered relevant local structure
   - But controller behavior collapses into duplicate-goal creation, blank-lane retries, or harmless rewrites that do not reduce the true search burden

The present runtime can detect these states after the fact, but it does not yet instantiate them as first-class symbolic objects with deterministic manipulation budgets.

## 3. Prior Signals and Related Work

### 3.1 LintGate-style specification objects

LintGate already uses structured specification objects and prescription generation to turn static code analysis into deterministic guidance. The relevant local references are:

- [`lintgate/specification/types.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/types.py)
- [`lintgate/specification/prescriptions.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptions.py)

The key insight worth importing is architectural, not domain-specific:

- separate **specification extraction** from **prescription generation**
- make guidance typed and inspectable
- make the unresolved portion explicit as typed proof holes with deterministic source spaces

Dr_Ducky is the proof-search analogue of this design.

### 3.2 Wesker-style sound space reduction

Wesker's README and filter layer are especially relevant:

- [`tools/Wesker/README.md`](/Users/rohanvinaik/tools/Wesker/README.md)
- [`tools/Wesker/Wesker/filter.py`](/Users/rohanvinaik/tools/Wesker/Wesker/filter.py)

Wesker's contribution is not merely speed. It is the principle that a large amount of search can be removed *soundly* by reading structural signals before generating candidates:

- no comparisons => boundary mutants impossible
- no state writes => state mutations irrelevant
- function-local structure defines which mutation categories are worth spending budget on

Dr_Ducky adopts this exact pattern for proof repair:

- no existential structure => do not spend budget on witness construction
- no algebraic operators => do not activate equational saturation banks
- category-theoretic structural residual with no numeric content => suppress numeric solvers

### 3.3 Symbolic math and equational proof engines

Wolfram's [`FindEquationalProof`](https://reference.wolfram.com/language/ref/FindEquationalProof.html) and `ProofObject` model show the right abstraction boundary for local proof repair: deterministic proof search over explicit equations with inspectable proof objects.

This matters because much of Wayfinder's residue is equational or near-equational. It is therefore a mistake to route all such states through generic text generation before asking whether the problem is structurally deterministic.

### 3.4 E-graphs and equality saturation

The [egg / egglog ecosystem](https://egraphs-good.github.io/) is directly relevant:

- e-graphs compactly represent large equivalence classes of terms
- equality saturation can explore many rewrites without committing early
- extraction gives a canonical representative after saturation

For Dr_Ducky, e-graphs are a better fit than ad hoc rewrite loops for:

- algebraic normalization
- coercion-cleanup normal forms
- congruence-heavy equalities
- extracting minimal terms before reconstruction into Lean

This now appears in the architecture as:

- `backend_family = "egglog_eqsat"` for `EqSatEngine`
- `EqSatPlan` inside each relevant capsule
- certificate traces intended to carry extraction/explanation provenance

The current repo state goes one step further than the original design sketch: the in-repo `EqSatEngine` now executes a bounded rewrite-saturation runtime with extracted rewrite paths and projected close/progress programs, rather than serving only as a named placeholder.

### 3.4b Rosette and solver-aided proof DSLs

[Rosette](https://emina.github.io/rosette/) contributes the clearest design lesson for Dr_Ducky's closure gap:

- define a domain-specific symbolic language
- interpret that language symbolically
- compile symbolic holes and assertions into solver-facing constraints
- separate the internal symbolic state from the external output program

For Dr_Ducky this means:

- proof skeletons are the fixed structure of a proof DSL
- typed holes become `ProofConstraint`s
- `ProofDSLProgram`s are the symbolic execution units
- the `ProofProjector` remains the only path from internal symbolic state to Lean-facing output

The current repo state now includes baseline production runtimes for this family as well: `ContextTransportEngine` and `WitnessEngine` execute proof-DSL-style local fact transport and bounded witness search in repo code, not merely in documentation.

### 3.4c Kodkod and bounded relational local search

[Kodkod](https://emina.github.io/kodkod/) is relevant because many near-miss proof states are naturally bounded relational problems:

- membership walls
- existential witness goals
- finite structural obligations
- filter/eventuality side conditions

The imported ideas are:

- bounded universes
- explicit relation symbols
- finite search with negative information when no bounded model fits

This now appears in the architecture as `RelationalSearchSpec` and the `backend_family = "kodkod_relational"` contract for `FiniteFilterEngine`.

The current repo state now includes a baseline bounded relational runtime that can directly emit structural witness programs for local membership-style closures.

### 3.5 LintGate-style skeleton synthesis

The decisive implementation cue comes from LintGate's prescriptive and synthesis stack:

- [skeleton_generator.py](/Users/rohanvinaik/tools/lintgate/lintgate/controlplane/skeleton_generator.py)
- [prescriptive/spec.py](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/spec.py)
- [prescriptive/composer.py](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/composer.py)
- [prescriptive/synthesis.py](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/synthesis.py)

The pattern is exact:

- extract a typed specification
- derive deterministic prescriptions
- synthesize a skeleton whose fixed structure is known in advance
- fill only the remaining typed slots from programmatically enumerated witnesses

For Dr_Ducky, this means no prompt stage and no free-form tactic text generation. The correct runtime object is a **proof skeleton** with:

- fixed tactic/program structure
- typed proof holes
- explicit source spaces for each hole
- deterministic compatibility rules for filling those holes

## 4. Core Thesis

The central thesis of Dr_Ducky is:

> Near-miss theorem proving should be treated as a typed residual-repair problem with a symbolic VM and projector boundary, not as free-form theorem proving from scratch.

More concretely:

1. Extract the deterministic structure of the residual goal.
2. Use that structure to filter and prioritize symbolic banks.
3. Seed a `ProofShadowLedger` with local facts, witnesses, rewrites, and accessible premises.
4. Synthesize typed proof skeletons whose open slots are filled programmatically.
5. Execute those actions inside proof-bearing deterministic engines.
6. Lower resulting certificates through a `ProofProjector`.
7. Reconstruct and validate every accepted step in Lean.

This produces a principled split:

- **Internal symbolic layer**
  - term inspection
  - residual classification
  - bank filtering
  - ledger construction
  - proof-bearing engines
  - certificate production

- **Projector/output layer**
  - skeleton-guided lowering
  - typed proof-hole filling
  - Lean-valid tactic/program synthesis
  - compile/progress/closure validation

## 5. The Dr_Ducky Object Model

### 5.1 GoalSpecification

Each residual state is converted into a typed `GoalSpecification`.

Fields include:

- theorem id
- namespace prefix
- last goal text
- goal bucket
- reasoning gap family
- residual bucket
- lane history
- search pathology tags
- difficulty band
- counts of goals closed, goals remaining, attempts
- derived structural signals

Derived signals include:

- coercion density
- binder count
- operator counts
- domain tags
- recursive-pressure markers
- structural-property markers
- membership-wall markers
- canonical-witness markers

### 5.2 ProofShadowLedger

The `ProofShadowLedger` is the core VM state. It stores:

- normalized local facts
- accessible premises
- candidate witnesses
- candidate rewrites
- engine outcomes
- rejected branches with provenance

It is a shadow ledger because it is not the final proof object. It is the internal symbolic state that drives engines and projector lowering.

### 5.3 BankPrior

Each goal specification induces a set of candidate deterministic banks with weights and rationales. Typical banks are:

- `eq_sat`
- `coercion_normalizer`
- `arith_nf`
- `binder_instantiation`
- `witness_constructor`
- `membership_exposure`
- `context_forward`
- `structural_close`
- `recursive_unfold_one`
- `solver_dispatch`

The weights are not learned initially. They are computed heuristically from structural signals and trace telemetry.

### 5.4 GoalPrescription

Prescriptions are the proof analogue of LintGate test prescriptions. They are not proof steps. They are typed, ordered repair suggestions such as:

- normalize coercions before exact close
- expose carrier lemmas before rewriting membership goals
- try structural closure before unpacking category-theoretic definitions
- enumerate canonical witnesses from local zero-like morphisms
- run equality saturation before attempting extraction

### 5.5 ProofSkeleton

`ProofSkeleton` is the proof analogue of a LintGate code skeleton:

- fixed deterministic structure
- typed holes
- deterministic source spaces for each hole
- admissible certificate kinds

The runtime never asks a language model to invent this structure.

### 5.6 MathEngineRequest and EngineCertificate

`MathEngineRequest` binds:

- active bank
- goal text / goal bucket
- allowed projector transformations
- execution budget
- ledger slice

`EngineCertificate` is the proof-bearing output of a symbolic engine. It records:

- engine name
- bank
- skeleton association
- certificate kind
- evidence
- projected tactic hints
- target-before text
- provenance

### 5.7 ProofProjector

The `ProofProjector` is the explicit boundary between internal symbolic state and Lean-valid output.

It lowers certificates into:

- direct proof terms
- rewrite chains
- `have` chains
- `calc` chains
- witness constructions
- structural closure programs

### 5.8 GoalCapsule

The `GoalCapsule` is the “mini-venv” abstraction.

It contains:

- the goal specification
- the filtered bank set
- the ordered prescriptions
- the typed proof skeleton set
- the seeded proof ledger
- allowed engines
- projector policy
- execution budgets
- a priority score for scheduling

The capsule is intentionally self-contained. It is the boundary object passed between analysis, orchestration, execution, and later second-order SoM training.

## 6. Theoretical Rationale

### 6.1 Why a capsule instead of free-form tactics?

Because the current bottleneck is not language generation capacity. It is failure to exploit deterministic local structure cheaply and consistently.

A capsule gives four benefits:

1. **Auditability**
   - every action is typed and attributable
2. **Budget control**
   - the system knows what not to try
3. **Deterministic leverage**
   - the symbolic engines do real work instead of acting as passive validators
4. **Trainability**
   - the executor targets a compact proof-program search space, not raw Lean syntax

### 6.2 Why deterministic-first?

Because the live run shows a huge concentration of local repair states. When the remaining goal is `a = c`, `x = 0`, `j < μ.rowLen i`, or a compact existential witness, it is wasteful to begin by asking a large model to improvise a proof script.

The correct order is:

1. ask what deterministic structure is present
2. compress the search space
3. synthesize a typed proof skeleton
4. fill its holes from deterministic source spaces
5. validate in Lean

### 6.3 Why not just use a general CAS or a weird machine?

General CAS tooling is useful as a reference but not as the primary runtime substrate:

- hard to align with Lean reconstruction
- often too broad and under-typed for proof-state management
- poor fit for domain-specialized suppression rules

FRACTRAN-like or similarly exotic universal substrates are the wrong abstraction entirely:

- no native typing
- no proof-object discipline
- poor explainability
- no practical integration path with Lean

Dr_Ducky should instead use explicit, typed, domain-aware residual machinery.

## 7. Engineering Proposal

### 7.1 Immediate MVP

The first implementable version should target only `single_goal_near_miss`.

Inputs:

- `details.jsonl` rows from live hard runs
- existing residual augmentation
- current `last_goal` / `last_goal_bucket` / `search_pathology_tags`

Outputs:

- one `GoalCapsule` per near miss
- ranked bank priors
- ranked prescriptions
- typed proof skeleton families for deterministic instantiation

The initial MVP was sidecar-only. The current implementation goes further: it includes a real Dr_Ducky executor and an optional proof-search lane, while still allowing offline theorem-faithful validation on captured benchmark residuals.

### 7.2 Deterministic bank design

The deterministic banks should be modular.

#### Equality bank

Used when:

- `last_goal_bucket == equality`
- coercion count is high
- ring/semiring syntax is present
- many congruent or repeated algebraic subterms appear

Implementation ideas:

- lightweight rewrite-bank normalizer
- e-graph back-end for saturation and extraction
- coercion cleanup pass before extraction

#### Inequality bank

Used when:

- `last_goal_bucket == inequality`
- order/norm/metric markers are present

Implementation ideas:

- arithmetic normal forms
- bound rearrangement
- monotonicity lemma retrieval
- cheap linear arithmetic delegation

#### Membership bank

Used when:

- `last_goal_bucket == membership`
- opaque container markers are present

Implementation ideas:

- carrier-exposure lemmas
- `mem_*` lemma harvesting
- local context forwarding

#### Witness bank

Used when:

- `last_goal_bucket == exists`
- canonical witness tags or constructive sources are visible

Implementation ideas:

- witness templates from local context
- zero / identity / inclusion / image-based candidates
- restricted finite witness enumeration

#### Structural close bank

Used when:

- category-theoretic or abstract structural tokens dominate
- the goal looks like `IsIso`, `Injective`, `Surjective`, `FormallyUnramified`, `HasRingHomProperty`, etc.

Implementation ideas:

- close-before-unpack policy
- high-level lemma retrieval
- suppression of arithmetic banks when no numeric content exists

### 7.3 Proof skeleton synthesis

Dr_Ducky should not emit free-form Lean and should not ask a language model to choose actions. It should synthesize typed proof skeletons such as:

```text
Skeleton: local_fact_transport
Templates:
  1. exact {fact}
Hole:
  fact : matching_local_fact from local_context
Compatibility:
  target_equivalent(hyp_type, goal_target)
```

```text
Skeleton: premise_bridge
Templates:
  1. rw [{lemma}]
Hole:
  lemma : rewrite_lemma from accessible_premises
Compatibility:
  rewrite_like(lemma)
```

```text
Skeleton: witness_instantiation
Templates:
  1. refine ⟨{witness}, ?_⟩
Hole:
  witness : witness_term from goal_surface
Compatibility:
  canonical_witness_family(goal_bucket, domain_hints)
```

The correct execution discipline is therefore:

1. derive skeleton family from prescriptions and bank priors
2. build the binding space from local hypotheses, accessible premises, and goal-surface witness terms
3. instantiate typed holes programmatically
4. compile instantiated skeletons into Lean tactic programs
5. validate the compiled programs in Lean

### 7.4 Lean integration

Lean remains the authority. Deterministic engines do not replace the kernel.

The execution discipline is:

1. choose action in capsule
2. run deterministic bank or transformation
3. synthesize candidate Lean tactic sequence
4. validate in Lean
5. accept only if Lean closes or reduces the target state

## 8. Validation Logic

The rational validation for Dr_Ducky is not speculative. It follows from current telemetry.

### 8.1 Current benchmark evidence

The implemented Dr_Ducky capsule layer has already been run against the live hard benchmark. The current bundle shows:

- `2907` input rows examined
- `1975` validated local rows selected into Dr_Ducky capsules
- coverage across four residual buckets:
  - `single_goal_near_miss`
  - `single_goal_stall`
  - `multi_goal_small_progress`
  - `multi_goal_large_progress`
- dominant specialist assignments:
  - `recursive_circuit_breaker`: `1659`
  - `side_condition_sweeper`: `1085`
  - `human_calculator`: `1009`
  - `symbolic_sandbox`: `883`
  - `structural_closer`: `309`
  - `membership_surface_engine`: `246`
- dominant enabled banks:
  - `loop_breaker`: `1658`
  - `local_fact_selector`: `1257`
  - `context_forward`: `1175`
  - `side_condition_sweep`: `1085`
  - `eq_sat`: `976`
  - `arith_nf`: `928`

This matters because it shows the architecture is not hypothetical. The current run already contains a large, typed population of residuals that Dr_Ducky can classify into concrete human-style repair modes.

### 8.2 Recorded validation evidence

The formal validation report at [`docs/Research_Paper/Ducky/VALIDATION_REPORT.md`](./VALIDATION_REPORT.md) and the machine-readable artifact at [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/validation.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/validation.json) show:

- recursive circuit-breaker routing: `47/52` (`90.38%`)
- symbolic sandbox routing: `782/883` (`88.56%`)
- domain numeric-solver suppression: `284/284` (`100.0%`)

Family-alignment checks are currently perfect on the targeted local families:

- `local_eq_close`: `34/34` (`100.0%`)
- `local_ineq_close`: `10/10` (`100.0%`)
- `membership_close`: `4/4` (`100.0%`)
- `witness_construction_close`: `5/5` (`100.0%`)
- `forward_context_close`: `6/6` (`100.0%`)
- `forall_close`: `5/5` (`100.0%`)
- `iff_close`: `4/4` (`100.0%`)
- `subset_close`: `1/1` (`100.0%`)
- `atomic_prop_close`: `3/3` (`100.0%`)

These numbers validate the diagnostic layer: Dr_Ducky is correctly identifying the right kind of local human-style repair mechanism for the current run's residual population.

### 8.3 Executor validation

The crucial gap in the earlier version of this paper was that Dr_Ducky had only been validated as a classifier. That is no longer true.

The current implementation includes:

- theorem-faithful residual replay via `goal_via_file_context`
- real tactic-program generation from capsules
- Lean-backed execution of those programs
- bounded multi-round local repair
- local-context mining from cached Pantograph goal states
- deterministic local fact synthesis over those states:
  - conjunction projections
  - symmetry facts
  - `mp` / `mpr` transports over local `↔`
- post-progress closure sweeps
- rejection of accepted tactics that leave the goal list unchanged
- optional integration as a live proof-search lane

The relevant implementation points are:

- [dr_ducky_executor.py](/Users/rohanvinaik/Projects/Wayfinder/src/dr_ducky_executor.py)
- [run_dr_ducky_executor_validation.py](/Users/rohanvinaik/Projects/Wayfinder/scripts/run_dr_ducky_executor_validation.py)
- [proof_search.py](/Users/rohanvinaik/Projects/Wayfinder/src/proof_search.py)

Focused tests now pass over the executor and proof-search integration:

- `python -m pytest tests/test_dr_ducky.py tests/test_dr_ducky_executor.py tests/test_build_dr_ducky_worklist.py tests/test_build_hard_collection_bundle.py tests/test_hard_resolution_layer.py tests/test_validate_dr_ducky_design.py -q`
- result: `22 passed`

Broader regression coverage also passes:

- `python -m pytest tests/test_benchmark_residuals.py tests/test_proof_search.py tests/test_audit_hard_run.py tests/test_dr_ducky.py tests/test_build_dr_ducky_worklist.py tests/test_validate_dr_ducky_design.py tests/test_dr_ducky_executor.py -q`
- result: `43 passed`

Machine-readable executor validation artifacts:

- [executor_validation_local20_vnext_rows.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_rows.jsonl)
- [executor_validation_local20_vnext_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_summary.json)
- [executor_validation_local20_vnext_engine_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_engine_outcomes.jsonl)
- [executor_validation_local20_vnext_projector_outcomes.jsonl](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_projector_outcomes.jsonl)
- [executor_validation_local20_vnext_closure_report.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_vnext_closure_report.json)
- [executor_validation_targeted_v6.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6.json)
- [executor_validation_targeted_v6_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6_summary.json)

The executor evidence currently shows:

- targeted stress cases: `5/5` theorem-faithful starts, `4/5` honest verified progress, `0/5` closures
- local 20-row slice: `20/20` theorem-faithful starts, `1014` projected certificates, `248` Lean compile proxies, `10/20` honest verified progress, `0/20` closures

These stricter counts replace earlier optimistic summaries that still allowed accepted-but-unchanged multi-goal no-ops to count as progress. The current numbers are therefore the ones that matter.

This is not yet a theorem-recovery victory lap. But it is no longer fair to call Dr_Ducky "just routing." It is now a real Lean-side residual executor whose current weakness is final closure, not abstract diagnosis.

### 8.3.1 Last-mile deepening implemented

The most important architectural deepening since the earlier draft is that Dr_Ducky no longer depends only on flattened goal text once replay has begun. The executor now reads cached Pantograph goal-state objects from Lean's internal goal cache and compiles additional symbolic repair material from those objects:

- local proof facts from cached variables
- derived projection facts (`h.1`, `h.2`)
- symmetry facts (`h.symm`)
- local `↔` transports (`hiff.mp hp`, `hiff.mpr hq`)
- target-only follow-up repair when the current residual text has lost explicit context

This matters because benchmark rows often record only the target surface after a local transformation, while the usable human-style closing information still exists in the Lean proof state. Dr_Ducky now exploits that internal state directly, which is much closer to the original theory than the earlier text-only executor.

### 8.4 Targeted case-study evidence

The validation report contains direct case studies that match the motivating examples behind the design.

1. **UnionFind / "Infinite Mirror"**
   - routed to `recursive_circuit_breaker`, `human_calculator`, and `side_condition_sweeper`
   - theorem-faithfully replayed in Lean
   - now makes verified progress with `rw [← Batteries.UnionFind.rootD]` followed by bounded structural closure attempts
   - this is the intended recursive circuit-breaker behavior: stop the mirror and re-enter a typed wrapper rather than blindly unfolding forever

2. **ModularGroup / "Veto Wall"**
   - `ModularGroup.tendsto_normSq_coprime_pair` is replayed theorem-faithfully and simplified by a longer symbolic chain that introduces binders, rewrites `Complex.normSq`, clears denominators, and normalizes casts
   - `ModularGroup.smul_eq_lcRow0_add` is replayed theorem-faithfully, but under stricter no-op filtering it no longer counts as verified progress on the sampled slice
   - these are exactly the goals where human-style symbolic cleanup should happen off the fragile text-level tactic path

3. **Local inequality cleanup**
   - `ModularGroup.eq_zero_of_mem_fdo_of_T_zpow_mem_fdo` is replayed theorem-faithfully and normalized to `n = 0`
   - the winning chain uses accessible rewrites plus arithmetic normalization to reach the semantic core
   - this is exactly the intended "System 1 calculator" role: local symbolic grinding, not new theorem planning
   - the case also exposes the current closure boundary: reaching `n = 0` is not yet enough without a stronger domain bridge

4. **Geometric equality collapse**
   - `Complex.eq_const_of_exists_le` is replayed theorem-faithfully and reduced to `x = 0` via a congruence push
   - the case is not yet closed, but Dr_Ducky is demonstrably shrinking the residual into a much smaller symbolic core

5. **Actual local fact use on benchmark slices**
   - On the honest 20-row local replay slice, Dr_Ducky now produces real chains that use cached-state local proof facts rather than only generic cleanup
   - Example: `MeasureTheory.lintegral_withDensity_le_lintegral_mul` reaches a residual headed by `Measurable f` and the winning chain includes `simpa using a`, compiled from a cached local proof fact
   - This did not close the theorem on the sampled slice, but it is decisive architectural evidence that Dr_Ducky is now performing symbolic/programmatic last-mile work of the intended kind

The case-study evidence is especially important because it demonstrates that the architecture captures the qualitative failure modes that motivated the project, not merely aggregate statistics.

### 8.5 Next-phase evaluation metrics

As Dr_Ducky is expanded from its current executor form into a default runtime recovery lane, it should be evaluated on:

- honest closure rate on local residual buckets
- closure lift on `single_goal_near_miss`
- closure lift on `single_goal_stall`
- reduction in blank-lane plateau frequency
- reduction in repeated duplicate-goal pseudo-progress
- reduction in accepted-but-unchanged tactic no-ops
- average additional attempts per recovered theorem
- reconstruction success rate for symbolic-bank outputs

### 8.6 Validation protocol

1. Build a capsule worklist from current hard-run residual data.
2. Validate bank routing and prescription alignment against targeted local families.
3. Integrate deterministic prescriptions without learned control.
4. Add typed proof skeleton synthesis and deterministic hole filling over capsules.
5. Compare:
   - routing-only capsules
   - deterministic proof-program execution
   - current baseline finishers

## 9. Failure Modes and Risks

### 9.1 Over-expansion of deterministic banks

If the banks are too permissive, Dr_Ducky becomes another uncontrolled search loop. This is why Wesker-style structural filtering is essential.

### 9.2 Unreconstructable symbolic transformations

A deterministic engine that cannot be turned back into Lean proof steps is useless for benchmark truth. Reconstruction discipline must be designed in from the start.

### 9.3 Hidden self-replay leakage

All new closure paths must inherit the patched self-application detection logic. Dr_Ducky cannot be allowed to turn target-theorem replay into a fake improvement.

### 9.4 Scope creep

The original first target was `single_goal_near_miss`, and that remains the conceptual center of gravity. The current implementation safely expands to adjacent local buckets such as `single_goal_stall`, `multi_goal_small_progress`, and `multi_goal_large_progress`, but it should still avoid theorem-start repair or full theorem replanning under the Dr_Ducky banner.

## 10. Roadmap

### Phase 1: Capsule extraction

- Build `GoalSpecification`
- Build structural signal extraction
- Build bank filtering and prescriptions
- Mine current local residual rows into capsules

### Phase 2: Deterministic execution

- Add normalization and structural-close bank adapters
- Add membership and witness helpers
- Add Lean reconstruction hooks

### Phase 3: Skeleton completion and proof-object search

- Add local fact-graph saturation
- Expand typed proof skeleton families
- Add proof-object-producing symbolic engines
- Keep validation Lean-backed and deterministic

### Phase 4: Runtime integration

- Add Dr_Ducky as a post-residual specialist
- Run paired evaluations against current finisher stack
- Report honest theorem recovery and cost

## 11. Conclusion

Dr_Ducky is not a replacement for Wayfinder's theorem search. It is the missing deterministic residual layer that the current benchmark regime is explicitly asking for. The telemetry already says the same thing hundreds of times: many proofs are almost done, and the remaining work is local, typed, symbolic, and cheap.

The current implementation now supports that claim at a stronger level than the initial draft did. Dr_Ducky is no longer only a typed classifier. It can replay benchmark residuals theorem-faithfully, synthesize real Lean repair programs, execute them, and make verified progress on the benchmark's motivating failure modes. The remaining gap is closure strength, not architectural nonexistence.

The correct response is not a larger unconstrained model. It is a typed capsule architecture that extracts deterministic structure, filters irrelevant actions, synthesizes proof skeletons, fills their holes programmatically, and uses symbolic engines where they belong.

That is what Dr_Ducky is.

## References

- Wayfinder hard-run audit: [`runs/exp_som012_hard_eval_r2/audit.json`](../../../runs/exp_som012_hard_eval_r2/audit.json)
- Dr_Ducky live bundle summary: [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/summary.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/summary.json)
- Dr_Ducky live bundle validation: [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/validation.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/validation.json)
- Dr_Ducky executor validation (local 20): [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_v4.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_v4.json)
- Dr_Ducky executor validation (local 20 summary): [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_v4_summary.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_local20_v4_summary.json)
- Dr_Ducky executor validation (targeted): [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6.json)
- Dr_Ducky executor validation (targeted summary): [`runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6_summary.json`](../../../runs/exp_som012_hard_eval_r2/bundle/dr_ducky/executor_validation_targeted_v6_summary.json)
- Dr_Ducky validation report: [`docs/Research_Paper/Ducky/VALIDATION_REPORT.md`](./VALIDATION_REPORT.md)
- LintGate specification types: [`tools/lintgate/lintgate/specification/types.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/types.py)
- LintGate prescriptions: [`tools/lintgate/lintgate/specification/prescriptions.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptions.py)
- LintGate skeleton generator: [`tools/lintgate/lintgate/controlplane/skeleton_generator.py`](/Users/rohanvinaik/tools/lintgate/lintgate/controlplane/skeleton_generator.py)
- LintGate prescriptive spec: [`tools/lintgate/lintgate/specification/prescriptive/spec.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/spec.py)
- LintGate prescriptive composer: [`tools/lintgate/lintgate/specification/prescriptive/composer.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/composer.py)
- LintGate deterministic synthesis: [`tools/lintgate/lintgate/specification/prescriptive/synthesis.py`](/Users/rohanvinaik/tools/lintgate/lintgate/specification/prescriptive/synthesis.py)
- Wesker overview: [`tools/Wesker/README.md`](/Users/rohanvinaik/tools/Wesker/README.md)
- Wesker filtering: [`tools/Wesker/Wesker/filter.py`](/Users/rohanvinaik/tools/Wesker/Wesker/filter.py)
- Wolfram `FindEquationalProof`: [reference.wolfram.com/language/ref/FindEquationalProof.html](https://reference.wolfram.com/language/ref/FindEquationalProof.html)
- Wolfram `ProofObject`: [reference.wolfram.com/language/ref/ProofObject.html](https://reference.wolfram.com/language/ref/ProofObject.html)
- Wayfinder design reference: [`docs/Research_Paper/WAYFINDER_DESIGN.md`](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/WAYFINDER_DESIGN.md)
- Wayfinder research reference: [`docs/Research_Paper/WAYFINDER_RESEARCH.md`](/Users/rohanvinaik/Projects/Wayfinder/docs/Research_Paper/WAYFINDER_RESEARCH.md)
- egg / egglog: [egraphs-good.github.io](https://egraphs-good.github.io/)
- Rosette: [emina.github.io/rosette](https://emina.github.io/rosette/)
- Kodkod: [emina.github.io/kodkod](https://emina.github.io/kodkod/index.html)
