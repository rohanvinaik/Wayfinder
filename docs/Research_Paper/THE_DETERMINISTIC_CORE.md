# The Deterministic Core

**Machine intelligence as principled inference, with statistics as bounded intuition**

*Status: implemented and pinned (2026-08). The deterministic path is complete, runnable
end-to-end via `configs/wayfinder_deterministic.yaml`, and protected by tests
(`tests/test_deterministic_lane.py`) that fail if any statistical component leaks into it.*

---

## 1. The thesis

Wayfinder wears the guise of a theorem prover, and its benchmark numbers are real. But the
project's purpose is not to solve proofs well. It is a testbed and demonstration of a thesis
about artificial intelligence and information theory:

> **A genuinely intelligent system is an inferential, deterministic system. The statistical
> layer above it is a genuine but limited learning/wiring/intuition/efficiency hack — never
> the substrate of the intelligence itself.**

The distinction this chapter defends is architectural, not rhetorical. In most contemporary
systems the neural network *is* the reasoner, and symbolic machinery survives only as
scaffolding around it — a parser here, a verifier there. Wayfinder inverts that: the
deterministic machinery *is* the reasoner — it represents, retrieves, composes, and verifies
— and the neural components are permitted to do exactly one class of work: **narrow
selection among options the principled layer already generated and will independently
verify.** A statistical component may make the system faster. It is never allowed to make
the system *possible*.

That claim is only meaningful if it is falsifiable, so the architecture enforces it
structurally: the entire statistical layer is **removable**, and what remains is not a
degraded fallback but a complete, functional, auditable proof-search system. This chapter
documents that system — the deterministic core — as a first-class artifact.

## 2. The two layers, component by component

**The principled layer** (deterministic, inferential, auditable — every step traces to a
rule):

| Component | Mechanism | Where |
|---|---|---|
| Proof network | SQLite semantic network: entities, 6-bank ternary positions, anchor dictionary, IDF table, accessible-premise closure | `src/proof_network.py` |
| Symbolic query | goal-text tokens × anchor dictionary — exact set intersection, neutral banks | `src/resolution.py::build_symbolic_query` |
| Premise ranking | IDF-weighted anchor intersection over the accessible pool, stable tie-break | `src/resolution.py::symbolic_rank_premises` |
| Rewrite scoping | source-gated, redex-aware premise filtering and direction inference — string/structure analysis, no model | `src/rw_scoper.py` |
| Tactic grammar | the pure family→tactic compiler (§4) | `src/tactic_compiler.py` |
| Structural bootstrap | fixed intro/simp/aesop setup ladder | `src/proof_search.py` (structural lanes) |
| Symbolic closers | Lean's own elaborator search: `exact?`, `apply?`, `solve_by_elim` — deterministic given the environment | `src/proof_search.py` |
| Dr. Ducky | the symbolic proof-local execution VM: typed capsules, `ProofShadowLedger`, proof-bearing engines, `ProofProjector` (invariant #8: pure symbolic, by construction) | `src/dr_ducky.py`, `src/dr_ducky_executor.py` |
| CAS tier | SymPy-derived obstruction-clearing chains, domain-gated | `src/cas_engine.py` |
| Verification | the Lean kernel. The one oracle; never statistical | `src/lean_interface.py` |

**The statistical layer** (bounded intuition — selection efficiency over the same
option space):

| Component | What it selects | What it can never do |
|---|---|---|
| Encoder + ternary navigator | which region of the network to open first | generate a tactic; validate anything |
| Cosine premise ranking | order within the (deterministically scoped) accessible pool | add a premise the pool doesn't contain |
| Family classifier / SoM routers | which lane to try first | add a lane; change a lane's grammar |
| ExecSelector, apply trigger | which `apply` candidates are structurally promising | bypass Lean verification |
| Second-order SoM (planned) | orchestration over symbolic packets: invocation, budgeting, escalation | produce proof content |

The pattern is uniform: every statistical component sits at a **ranking or routing seam**
between deterministic stages. Remove it and the seam falls back to a principled ordering
(IDF rank, goal-bucket order, FIFO). Nothing downstream changes, because nothing downstream
ever consumed a statistical *product* — only a statistical *ordering* of deterministic
products.

## 3. The deterministic path, end to end

`search_mode: deterministic` runs the full search with the statistical layer absent — not
disabled-but-loaded: `run_benchmark` never loads the navigator checkpoint, no encoder is
constructed, and the pinned test asserts no `cosine_*` or `learned` lane is ever scheduled.

```
goal text
   │  tokenize (identifiers + dotted components)
   ▼
anchor dictionary ─── set intersection ──► matched anchors
   │                                          (proof_network.anchors)
   ▼
accessible-premise pool ── IDF-weighted anchor overlap ──► ranked premises
   │  (accessible_premises closure,           (anchor_idf table;
   │   rw/simp scope filter)                   stable tie-break)
   ▼
tactic grammar ── pure compilation ──► ordered Lean candidates
   │  (src/tactic_compiler: direction-aware rw,
   │   apply escalation ladder, bare-simp-first)
   ▼
Lean kernel ── verify ──► closed goal, provenance "symbolic_<family>"
```

Around that spine run the other deterministic lanes: the structural bootstrap for setup,
the symbolic closers (`exact?`/`apply?`/`solve_by_elim`) for the last mile, the
norm-then-close normalization pass, and optionally the Dr. Ducky residual VM for the local
symbolic tail. Goal selection is FIFO (no critic). Hammer delegation is off (its gate reads
navigator output). Every closure carries a lane provenance, so the run's every decision can
be read back to the rule that produced it.

Run it:

```bash
python scripts/run_benchmark.py --config configs/wayfinder_deterministic.yaml
```

## 4. The grammar is knowledge, not weights

The pure tactic compiler deserves its own section, because its history is the thesis in
miniature.

The compilation grammar — how a chosen (family, premise) pair becomes Lean text — was
first sketched as static templates (the v0 compiler, March 2026). Benchmarks then improved
the templates: bare `simp` before hinted `simp` (multi-hint measurably hurts acceptance);
`exact → refine ?_ ?_ → refine ?_ ?_ ?_ → apply` instead of bare `apply` (from the
"Function expected" failure analysis); per-premise rewrite direction inference; the
elimination of `exact p _` wildcard escalation (the documented 0%-success `premise_direct`
failure mode).

Note what kind of improvements those are. **Every one is a fact about Lean's elaborator, not
about a training distribution.** The learning that upgraded the grammar produced *rules*,
not weights — and so the upgraded grammar is exactly as deterministic as the sketch it
replaced. This is the intended relationship between the layers: the statistical apparatus
(benchmarks, telemetry, learned experiments) *discovers* structure, and the discovered
structure is *installed in the principled layer* as inspectable rules. Intelligence
accumulates in the deterministic core; statistics is how some of it was found.

`src/tactic_compiler.py` is therefore deliberately a total function —
`(family, ranked premises, goal target) → ordered candidates` — with no model, no I/O, and
no state. The learned lanes and the symbolic lanes compile through the *same* grammar
(`proof_search._try_ranked_family`); the layers differ only in who ordered the premises.

## 5. The evidence that the core carries the system

The claim "the deterministic layer is the intelligence" is not aesthetic; it is what the
project's own ledger shows:

- **EXP-3.2** (50-theorem Mathlib, 3 modes): `learned_only` proved **0/50**;
  `learned_structural` proved 11/50 — all closures attributed to the deterministic
  bootstrap; the full stack added one more via hammer. The learned local tactic prior
  contributed zero direct closures.
- **NAV-004** (aligned navigator freeze): the navigational signal contributed zero proved
  theorems beyond what the deterministic infrastructure provides; its validated value is
  routing.
- **EXP-058**, the published 1,277/2,000 baseline, is a *finisher-stack* result — `exact? +
  shape + induction + micro`: Lean's own symbolic search plus deterministic structural
  policy. The validated 2-step residual extension (~1,371) deploys a discovered
  deterministic pattern (`norm_cast; exact?`).
- **Dr. Ducky** is pure symbolic by invariant, and makes Lean-verified progress on the
  frozen hard slice (honest progress 69/120 on the stratified closure benchmark).
- The statistical layer's *validated* wins are exactly where the thesis puts them:
  selection efficiency (ExecSelector 38.5% LeanAccepted on apply steps vs cosine's 16.5%;
  the family router's 95.6% top-1 routing accuracy) — orderings over deterministically
  generated, kernel-verified options.

The honest caveat, stated as prominently as the evidence: the deterministic path's
*theorem-level* solve rate under this exact configuration has not yet been benchmarked as
its own condition. That run — the deterministic path on the frozen 2,000-theorem split,
reported beside `EXP-058` — is the natural next experiment this chapter defines
(**EXP-DET-001**), and the paired comparison `deterministic` vs `deterministic + statistical
layer` is the cleanest measurement of what the intuition layer is actually worth.

## 6. Removability as an invariant

Specification-complexity theory (RESEARCH §2.9) gives the design rule: composition is safe
when the cross term γ vanishes — when components do not entangle. The two layers are built
to keep γ at zero by construction:

1. **Interface discipline.** The statistical layer communicates with the core only through
   rankings of core-generated options. There is no channel through which a learned
   component can inject content.
2. **Fallback totality.** Every seam has a principled default (IDF rank, bucket order,
   FIFO), so deleting the statistical layer leaves total functions everywhere.
3. **Pinned separation.** `tests/test_deterministic_lane.py` runs a full search step with
   `pipeline=None` and asserts (a) a goal closes with provenance `symbolic_exact`, and
   (b) no statistical lane is ever scheduled. `tests/test_cosine_family_templates.py` pins
   the shared grammar so neither layer can drift it unilaterally.

This is also why the layering is stated as a project invariant (docs/README.md): the
deterministic path must remain complete and runnable at all times. A change that makes the
core depend on the statistical layer is not an optimization; it is a violation of the
thesis the system exists to demonstrate.

## 7. Relation to the rest of the program

The Society-of-Mind program (first- and second-order) does not weaken this chapter's claim
— it instantiates it at the next level up. The SoM controllers are statistical *orchestration*
over symbolic packets: they decide when to invoke Dr. Ducky, how to budget, when to
escalate. The packets themselves — capsules, ledgers, certificates, gap analyses — are
deterministic objects, and the actions taken remain kernel-verified. The same is true of
Phase 7 (v3): energy scoring and censors shape *search order*, never search *content*.

The program's long arc, in this chapter's terms: **grow the principled layer's coverage
(the compiler, the network, the residual VM), use the statistical layer to find the next
rule worth installing, and keep the boundary between them a tested invariant.** That is
what machine intelligence actually looks like on this thesis — and Wayfinder is the
demonstration that it proves theorems.
