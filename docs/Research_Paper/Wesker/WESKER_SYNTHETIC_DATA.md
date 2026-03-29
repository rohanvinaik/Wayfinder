# Wesker: Mutation-Driven Synthetic Data for Proof Orchestration

**Status:** Research idea → active implementation
**Date:** March 27, 2026
**Parent project:** Wayfinder (navigational theorem prover for Lean 4 / Mathlib)
**Scope:** Synthetic training data generation for the second-order Society of Mind controller

---

## The Problem

Wayfinder's second-order SoM controller learns routing, budgeting, and engine selection over symbolic residual packets. It decides: given a hard residual, which Dr. Ducky engine family (EqSat, ContextTransport, Arith, FiniteFilter, RecursiveInvariant, Witness) and which backend (egglog, rosette_proof_dsl, kodkod_relational, lean_arith, symbolic_rewrite_vm) will make progress or close the goal?

The current training corpus has **89 examples** (20 eval). The model converges in under a second and memorizes the training set. Eval metrics are reasonable (invoke=100%, engine=85%, backend=86%) but confidence intervals are wide and generalization is untested. The model needs orders of magnitude more training data.

Real data is expensive: each training example requires a full Lean theorem-search run (~45 seconds/theorem) followed by Dr. Ducky execution on the residual. Scaling to 10K examples would take ~125 hours of Lean compute.

## The Insight

The project already has **240,000 labeled proof steps** in `data/canonical/canonical_residual_train.jsonl`, each with a goal state, tactic, family, and premise. Each step is an implicit (goal_state, oracle_engine_family) pair. The second-order SoM's prediction targets — engine and backend selection — are derivable from tactic family labels that already exist.

Additionally, the 682 proved theorems from the current benchmark contain complete tactic traces. These can be reverse-engineered into synthetic hard residuals by selectively deleting closing tactics — creating states with known ground-truth solutions.

Finally, LintGate's mutation testing infrastructure provides a principled framework for generating hard negatives: proof mutations that are structurally plausible but semantically wrong, forcing the model to learn fine-grained engine/backend distinctions.

## Three Strategies

### Strategy 1: LeanDojo Trace Mining

**Source:** `data/canonical/canonical_residual_train.jsonl` (240K steps)
**Cost:** Zero Lean calls. Pure parsing.
**Expected yield:** 10K–50K second-order packets after filtering.

Each canonical residual step contains:
```
theorem_full_name, goal_state_before, tactic_text, tactic_base, family,
annotated_premise, goal_shape_ir, trigger_profile_ir, subtask_ir
```

The tactic family maps deterministically to engine/backend targets:

| Tactic family | Engine | Backend | Confidence |
|---|---|---|---|
| `rw`, `rw_seq` | EqSatEngine | egglog_eqsat | High |
| `exact`, `apply` | ContextTransportEngine | rosette_proof_dsl | High |
| `simp`, `norm_num`, `ring`, `field_simp` | ArithEngine | lean_arith | High |
| `omega`, `linarith`, `nlinarith` | ArithEngine | lean_arith | High |
| `ext`, `funext`, `congr` | ContextTransportEngine | rosette_proof_dsl | Medium |
| `cases`, `induction`, `rcases` | RecursiveInvariantEngine | symbolic_rewrite_vm | Medium |
| `use`, `refine`, `obtain` | WitnessEngine | rosette_proof_dsl | Medium |
| `decide`, `norm_cast` | FiniteFilterEngine | kodkod_relational | Medium |
| `aesop`, `tauto` | ContextTransportEngine | rosette_proof_dsl | Low |

**Filtering for quality:**
- Exclude structural/setup tactics (intro, constructor, left/right) — these aren't engine-relevant
- Prioritize "closing" steps (last tactic in a goal branch) — most similar to residual resolution
- Prioritize steps with non-trivial premises — these carry engine-selection signal
- Weight by goal complexity (hypothesis count, nested quantifiers, coercion depth)

**Packet construction:** For each filtered step, build a synthetic second-order packet:
- `residual_bucket`: derive from goal shape (single_goal if 1 remaining goal, multi_goal otherwise)
- `goal_bucket`: derive from goal_shape_ir (equality, membership, inequality, etc.)
- `resolution_family`: derive from the goal's dominant type
- `second_order_labels.engine_family_budget_targets`: from tactic family mapping
- `second_order_labels.backend_budget_targets`: from tactic family mapping
- `second_order_labels.invoke_ducky`: True (all these are non-structural steps)
- `second_order_labels.observed_progress`: True (these steps succeeded in the original proof)

### Strategy 2: Tactic-Deletion Residuals

**Source:** 682 proved theorems from EXP-SOM-016 r6
**Cost:** Lean replay required (Tier C prefix replay). ~20-60s/theorem.
**Expected yield:** 682–2K high-fidelity synthetic residuals.

For each proved theorem with trace `[t_1, t_2, ..., t_n]`:

1. **Identify closing steps**: steps where a goal transitions from open to closed
2. **Delete the last K closing steps** (K=1,2,3) to create partial-proof states
3. **Replay prefix `[t_1, ..., t_{n-K}]`** in Lean to get the real goal state at the deletion point
4. **Label**: the deleted tactic tells you the oracle engine/backend

Deletion depth calibrates difficulty:
- K=1: near-miss residual (one closing step away)
- K=2: medium residual (needs two-step resolution)
- K=3: hard residual (needs multi-step planning)

**Advantages over trace mining:**
- Goal states are real Lean states, not parsed approximations
- Residual structure is realistic (matches what the second-order SoM actually sees)
- Multi-goal interactions are preserved (deleting a step in a branching proof creates realistic multi-goal states)

**Disadvantages:**
- Requires Lean server (Pantograph) for replay
- Slower than pure parsing
- Some replays will fail (scope issues, elaboration drift)

### Strategy 3: Proof Mutation (Wesker-style)

**Source:** 240K canonical steps + LintGate mutation operators
**Cost:** Mutation generation is cheap; validation requires Lean calls.
**Expected yield:** 5K–20K hard negatives after Wesker filtering.

**Connection to LintGate/Wesker:** LintGate generates program mutations and measures test survival. Wesker adds *sound structural filtering* — rejecting mutations that are structurally impossible before testing. The same framework applied to proof tactics:

**Mutation operators:**

| Operator | Example | What it tests |
|---|---|---|
| Premise swap | `rw [add_comm]` → `rw [mul_comm]` | Engine-correct, premise-wrong |
| Direction flip | `rw [foo]` → `rw [← foo]` | Backend-correct, orientation-wrong |
| Family swap | `simp [foo]` → `rw [foo]` | Same premise, different engine needed |
| Weakening | `exact foo` → `apply foo` | Creates multi-goal from single-goal |
| Hypothesis deletion | Remove a `have` step | Missing local fact → transport engine |
| Scope perturbation | Remove an `open` qualifier | Namespace-scoped premise becomes invisible |

**Wesker structural filter (3 stages):**

1. **Syntax filter**: Does the mutated tactic parse as valid Lean syntax?
2. **Scope filter**: Is the substituted premise accessible in the theorem's import scope? (Reuse the existing scoping pipeline from `src/rw_scoper.py`)
3. **Type filter**: Is the goal's type signature compatible with the mutated tactic family? (e.g., don't apply `omega` to a non-numeric goal)

Mutations that pass all three filters but fail Lean verification are **maximally informative hard negatives**. They're the kind of near-miss the second-order SoM most needs to distinguish from correct decisions.

**Hard negative packet construction:**
- Same packet format as Strategy 1
- But `observed_progress = False` and `observed_close = False`
- Engine/backend labels reflect what the mutation *tried*, not what would work
- The model learns: "this goal shape + this engine = failure" as well as "this goal shape + that engine = success"

**The LintGate mutation survival principle:** A mutation that changes the engine/backend target while preserving goal-state similarity is maximally informative. The SoM learns "these two goals look almost identical but one needs eqsat and the other needs proof DSL."

## Implementation Plan

### Phase 1: Trace Mining (no Lean calls)

```
canonical_residual_train.jsonl
    → filter (closing steps, non-trivial premises, complex goals)
    → map tactic family → engine/backend targets
    → construct second-order packets
    → merge with real r6 packets
    → retrain second-order SoM
```

**Script:** `scripts/build_wesker_trace_mining_dataset.py`

**Inputs:**
- `data/canonical/canonical_residual_train.jsonl`
- `runs/exp_som016_final_random2000_r6_repair_v1/bundle/second_order_som/second_order_packets.jsonl` (real packets)

**Outputs:**
- `data/wesker/trace_mining_packets.jsonl`
- `data/wesker/trace_mining_features/train.npz`, `eval.npz`, `metadata.json`

### Phase 2: Tactic Deletion (Lean replay required)

```
r6 proved theorems (682)
    → extract tactic traces from details.jsonl
    → select deletion points (last 1-3 closing tactics)
    → replay prefix in Lean (Tier C)
    → extract real goal state at deletion point
    → label with oracle engine/backend from deleted tactic
    → construct second-order packets
```

**Script:** `scripts/build_wesker_tactic_deletion_dataset.py`

### Phase 3: Proof Mutation (Lean validation required)

```
canonical steps (240K)
    → apply mutation operators
    → Wesker structural filter (syntax, scope, type)
    → validate in Lean (pass/fail)
    → construct hard-negative packets from filtered failures
```

**Script:** `scripts/build_wesker_mutation_dataset.py`

### Evaluation Protocol

Train second-order SoM under four conditions:

| Condition | Training data | Expected size |
|---|---|---|
| A: Baseline | Real r6 packets only | 89 |
| B: + Trace mining | A + LeanDojo trace-mined packets | ~10K–50K |
| C: + Tactic deletion | B + deletion-derived packets | +682–2K |
| D: + Mutations | C + Wesker hard negatives | +5K–20K |

**Metrics:**
- Eval accuracy on held-out real r6 packets (the 20 eval examples)
- Engine micro-accuracy
- Backend micro-accuracy
- Cross-validation stability (does the model generalize or memorize?)
- Downstream: bridge closure rate when the trained SoM drives DD-015

## Theoretical Connection

This work connects three threads:

1. **Wayfinder's bimodal structure claim**: most of theorem proving is structural, the hard tail is small and typed. Synthetic data generation exploits this by mining the structural majority for engine/backend labels.

2. **LintGate's mutation testing**: the principle that *controlled perturbation reveals specification boundaries*. Applied to proofs: which mutations change the required engine family? Those boundaries are exactly what the orchestrator needs to learn.

3. **Wesker's sound structural filtering**: not all mutations are informative. The Wesker filter (syntax, scope, type) eliminates structurally impossible mutations before wasting compute, concentrating the training signal on genuinely hard distinctions.

The synthesis: **proof-step data is abundant, orchestration labels are derivable, and mutation testing surfaces the decision boundaries the second-order controller needs to learn.**
