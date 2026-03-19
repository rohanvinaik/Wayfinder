# rw-First Formal System Buildout Plan

## Summary
Build the formal local-execution system in a strict order: first make Lean-valid `rw` metrics trustworthy, then turn `rw` into a real learned lane, then generalize the same `ActionIR` pattern to the other families. The current evidence already justifies this order:

- theorem-level orchestration is no longer the bottleneck
- `ActionIR` is semantically correct enough to serve as the target (`100%` parse, `70%` exact round-trip)
- deployable `rw` scoping exists (`96%` gold-in-scope, mean scope `15.4`)
- within-scope ranking is already useful (`43.6%` top-1, `86.6%` top-5)
- the remaining blocker is trustworthy Lean-valid execution, not architectural uncertainty

Default chosen:
- **scope**: `rw`-first proving
- **ordering**: infra cleanup before decoder lift

## Implementation Plan

### 1. Lock the `rw` formal contract before more model work
Use `rw` as the first fully formal local lane and treat it as the reference design for the rest of `ActionIR`.

Required contracts:
- `ActionIR` stays the sole supervision target; tactic strings are lowering artifacts only
- `rw` v1 grammar is fixed to:
  - rewrite list
  - per-item direction bit
  - per-item `TermExpr`
- `LocalSymbol` / scoped vocabulary contract is explicit:
  - source: `hyp` | `premise` | `combinator`
  - name
  - typed eligibility flags (`rewrite_capable`, `local_hyp`, etc.)
  - optional embedding / scope scores
- `RwScopeResult` records:
  - scoped symbols
  - source-gate decision
  - redex features
  - whether gold was in scope
- `ActionCandidate` always carries:
  - lowered tactic text
  - optional `ActionIR`
  - scope provenance
  - rank score
  - Lean outcome / failure category

Hard decisions:
- do **not** optimize exact string match further
- use canonical `ActionIR` match, compile-valid, and Lean-valid as the only authoritative metrics
- `rw` direction is **symbolic-first** in v1:
  - if only lhs matches a redex, use forward
  - if only rhs matches, use backward
  - if ambiguous, beam both
  - learned direction remains auxiliary only, not a gate

### 2. Remove infrastructure false negatives from the `rw` Lean-valid benchmark
Before improving the model, make the benchmark trustworthy.

Work to frontload:
- fix Pantograph goal creation for universe-polymorphic goals in the actual benchmark path
- standardize theorem environment setup so imported names and scoped notation are available before `goal_start`
- classify Lean failures into:
  - goal creation failure
  - identifier out of scope
  - elaboration / implicit-argument failure
  - rewrite inapplicable
  - successful tactic but no theorem close
- ensure benchmark output preserves these categories per candidate and per theorem

Acceptance gates for this phase:
- goal creation failure rate on the held-out `rw` benchmark: `<5%`
- unknown-identifier / missing-scope failure rate: `<5%`
- canonical lowering on gold `ActionIR` examples: `100%` compile-valid on the benchmark harness
- no infrastructure failures are counted as model negatives in training

### 3. Turn `rw` into a real learned lane
Use the deployable source-gated redex-aware scope as fixed infrastructure and optimize ranking inside that scope.

Search stack for `family == rw`:
1. residual family gate includes `rw`
2. source gate predicts `hyp` vs `premise`
3. build redex-aware scoped vocabulary
4. pointer reranker scores scoped symbols
5. build `ActionIR`
6. lower to Lean
7. verify with small beam

Modeling decisions:
- freeze the scoper shape in v1:
  - hypotheses from goal text
  - redex-aware accessible rewrite premises
  - no theorem-sibling proxy scope in deployable eval
  - no unconditional global fallback; only adaptive fallback if scope is empty or below minimum size
- train hard negatives **inside scope**, not against the full corpus
- add cosine score and scoper features as reranker inputs; keep geometry as a first-class signal
- use beam search at inference (`k=5` default) because top-5 is already strong

Primary metrics:
- gold-in-scope coverage
- in-scope top-1 / top-5 / MRR
- exact canonical `ActionIR` match
- Lean-valid top-1 / top-5
- theorem-step success on first-tactic `rw` benchmark

Acceptance gates:
- maintain scope coverage `>=95%`
- maintain mean scope size `<=20`
- improve in-scope top-1 over current `43.6%`
- Lean-valid `rw` top-5 on first-tactic sample: `>=20%`
- beat cosine-only exact-IR baseline (`25%`) clearly on the same sample

### 4. Integrate `rw` into live proof search as the first learned local lane
Once `rw` is Lean-valid at component level, wire it into the benchmarked search path before expanding family coverage.

Runtime behavior:
- only activate learned `rw` when residual family gate includes `rw`
- run `rw` after structural normalization and before generic learned candidate expansion
- keep hammer/bootstrap unchanged
- preserve lane attribution:
  - `learned_rw`
  - `solver_bootstrap`
  - `automation`
  - `structural_core`
- record per-theorem:
  - whether `rw` was attempted
  - whether it advanced the goal
  - whether it closed the theorem
  - Lean calls spent in `rw`

Paired theorem experiments:
- baseline: current search without learned `rw`
- variant A: cosine-only `rw`
- variant B: learned-pointer `rw`
- same theorem set, same budget

Acceptance gates:
- learned `rw` contributes non-zero theorem progress and non-zero closes on Mathlib
- learned `rw` reduces wasted local attempts or Lean calls on `rw`-applicable goals
- no regression greater than noise on total prove rate

### 5. Generalize only after `rw` is real
Do not build all families in parallel. Use `rw` as the formal reference implementation.

Next family order:
1. shared `exact/apply/refine` term decoder
2. `simp` decoder
3. leave tactic-combinator scripts out of scope until the single-action families are productive

Generalization rules:
- reuse `ActionIR` / `TermExpr`
- keep family-specific scoping
- keep typed hard constraints
- oracle-premise first, retrieved-premise second
- only revisit theorem-level retrieval after oracle→retrieved gap is measured on the local family tasks

### 6. Re-stage temporal control after local executor competence exists
Do not spend more cycles on TC active-mode gains until `local_close` has a real executor to route to.

Temporal-controller plan:
- keep collecting shadow/active traces during `rw` experiments
- once learned `rw` produces real local-close wins, rerun:
  - `TC1-goal`
  - `TC2-lane`
- only after `rw` and one more family are productive, train a learned temporal controller

Decision:
- temporal control remains a routing layer, not a blocker for local formalization

## Test Plan
Component tests:
- canonicalizer/lowering round-trip on gold `ActionIR`
- scoper correctness:
  - source gating
  - redex filter eligibility
  - gold-in-scope coverage
- pointer reranker:
  - in-scope ranking
  - hard-negative discrimination
- Lean harness:
  - universe-polymorphic goal creation
  - import/scoped-notation handling
  - failure categorization

Benchmarks:
- held-out first-tactic `rw` benchmark:
  - exact `ActionIR`
  - Lean-valid top-1/top-5
- paired theorem benchmark:
  - baseline vs cosine `rw` vs learned `rw`
- per-family local task eval:
  - oracle vs retrieved premise gap once `exact/apply/refine` starts

Reporting:
- component-level metrics and theorem-level metrics are always separated
- infrastructure failures are reported separately from semantic/model failures
- `Research_Paper` results ledger is updated after each milestone, not only at the end

## Assumptions and Defaults
- Internal audience: this is an implementation-facing research plan, not a paper-only outline
- `rw` is the first formal lane because it has the strongest current evidence and the cleanest deployable scoping
- theorem-sibling symbol pools are treated as oracle/proxy tools only; they are excluded from deployable claims
- canonical `ActionIR` is the ground truth; string identity is non-authoritative
- family-specific constrained decoding is the core execution path; raw tactic-string generation is out of scope
- no further investment in theorem-level step retrieval until the local oracle→retrieved gap is measured on real family decoders
