---
title: "Navigation as Exact Identification: The Two-Sign Re-Founding of Wayfinder"
subtitle: "A diff thesis — from vibe-estimated coordinates to computed structural reads, admissible censors, and warranted emission"
author: "Rohan Vinaik"
date: "2026-08-24"
status: >
  re-founding specification (a diff thesis against the 2026-08 state of the repository; strip-and-replace
  design, not a reset — the deterministic path, the pure grammar, the Ducky invariant, and the metric
  discipline are load-bearing and stay). Every claim carries a per-claim status tag; the ledger is §11.
priors_do_not_rederive:
  - "σ = teaching dimension (SC Thm 2.7 / Lean T5_17); the greedy dynamics: proportional progress, exponential decay, bulk→tail phase transition (SC §3); regime = symmetry (SC Thm 4.1/T6.15); composition gap γ + interface-mutant bound (SC Thm 3.15/3.16)"
  - "Two-sign σ(P, μ ∪ μ⁻); channel isolation (NEG_SPEC Thm 5.2); censor admissibility = spine-sourced + retained plurality (NEG_SPEC Def. 9.3, Prop. 14.7, machine-checked `self_confirming_cannot_certify` / `falsifiability_pivot`); κ = marginal coverage; censors-are-bridges (MEASURED, NEG_SPEC Thm 14.4/Cor. 14.5); bounded curvature `bridge_curvature_bound` (proved 2026-08-24, `sorries_remaining: 0`, `#print axioms` audit owed)"
  - "The semantic transport license and the Semantic Completeness Equation dH/dt = −(N + C(H)) with its bulk knee (SSL §1.4/§2.5; the knee measured on the LeanDojo corpus by scripts/proof_knee*.py — the third domain)"
  - "Categorical exclusion is lossless (Wesker README Layer 2 — the Monty Hall filter); SC(f)=1 ⟺ H(f|tests)=0"
  - "Value-kill ≠ run-kill; candidate-equivalent UNPROVEN never promoted; certificate standing consumed, never re-derived (Detective ARCHITECTURE §0, §5; NEG_SPEC Def. 1.4, 12.4)"
  - "The deterministic core is the system; the statistical layer is bounded intuition above it, removable by invariant (THE_DETERMINISTIC_CORE, docs/README.md invariant #10)"
---

# Navigation as Exact Identification

## The Two-Sign Re-Founding of Wayfinder

### Abstract

Wayfinder was founded on a thesis it implemented by intuition: that proof search is navigation —
deterministic filtering through a structured semantic network — and that the neural component is an
efficiency layer, not the reasoner. The intuition was right and the realization was primitive. The
6-bank ternary coordinate, the correct object, was *estimated by a 22M-parameter encoder* because no
principled estimator existed; navigation ran on **one sign** (retrieval of what may work) with its
negative knowledge scattered as ungoverned heuristics; the projector lowered internal certificates to
tactic *strings* and trusted the parser; and a failed search discarded precisely the structure it had
computed. The theory that grounds the principled forms now exists, machine-checked where it matters:
specification complexity identifies navigation with **exact identification** (σ = teaching dimension =
Angluin query complexity), the two-sign construction supplies the **negative channel** navigation was
missing and proves it non-redundant (channel isolation), the censor/κ machinery governs
population-derived negative constraints with a tractability bound **this system itself proved**
(`bridge_curvature_bound`, closed locally by the pre-refit prover), and the Detective/Peitho/AIMO3
implementations demonstrate every mechanism at production scale in wider semantic domains than formal
mathematics. This document is the **diff thesis**: a formal specification of the gap between the
current system and the theory-compliant design, module by module — what stands, what is stripped,
what replaces it, and the experimental program that measures the difference. The ideas are heavy; the
code is light. Each replacement is a few thousand lines, and each has a running prototype in a sibling
project.

---

## 1. The Object and the Gap

### 1.1 What Wayfinder is

Wayfinder is a testbed and demonstration of a thesis about machine intelligence: **a genuinely
intelligent system is an inferential, deterministic system; the statistical layer above it is a
genuine but bounded learning/intuition/efficiency hack — never the substrate**
(THE_DETERMINISTIC_CORE §1; project invariant #10). The proof-search guise is the demonstration
domain, chosen because the external oracle (the Lean kernel) is incorruptible. The benchmark numbers
are real and are not the point.

### 1.2 The current realization, honestly characterized

The 2026-08 system already carries theory-compliant load-bearing structure, and this document does
not touch it:

| Element | Status | Why it stands |
|---|---|---|
| The deterministic path (`search_mode: deterministic`; symbolic IDF lanes; pipeline-free search) | **compliant** | the removability invariant, pinned by `tests/test_deterministic_lane.py` |
| The pure tactic grammar (`src/tactic_compiler.py` v1) | **compliant** | knowledge-not-weights: every template is a Lean-elaborator fact installed as a rule |
| Dr. Ducky as pure symbolic VM (capsules, ledger, engines, projector *boundary*) | **compliant** | invariant #8; the projector's *output form* is the non-compliant part (§5) |
| Lane provenance; `raw_success`/`honest_success` separation; infra-vs-model ledgers | **compliant** | the metric discipline is the honesty layer and extends (§7.3) |
| The proof network (entities, anchors, IDF, accessible-premise closure) | **compliant** | the substrate all reads and censors are computed against |

Against that, the primitive elements — each the *shape* of a theory object realized by intuition:

| Element | Current mechanism | What it actually is | Theory-compliant form |
|---|---|---|---|
| 6-bank coordinates | 22M encoder → bottleneck → ternary decoder (NAV-002/004) | a *statistical estimator of a computable object*; measured contribution to closures: **zero** (EXP-3.2, NAV-004) | computed structural reads against corpus-mined norms (§4) |
| Negative knowledge | scattered: Ducky suppression hints, domain lane policy, `_NUMERIC_SOLVER_TACTICS` suppression, the apply-lane freeze | real censors, *ungoverned*: no admissibility condition, no κ, no derivation record, not reported as a channel | admissible censor store + Monty Hall gate (§3) |
| `src/censor.py` | `CensorNetwork` — a learned failure predictor with asymmetric BCE (v3 plan) | the one place the design reached for a model where the theory supplies a fence | deterministic censor store; the network demoted to optional ordering (§3.6) |
| Emission | `ProjectedProofProgram.tactics: list[str]`; reconstruction re-parses strings at v4.28 | string lowering — compiles by parser luck; warrant lineage present (`rationale`, `certificate_id`) but no value/source duality, no environment | `LeanExpr` warranted emission (§5) |
| The failed-search residual | `details.jsonl` rows + capsule bundles, consumed only by offline analysis | computed understanding, discarded at the moment of hand-off (Aristotle receives the bare statement — the measured Gotcha 3) | the cheat-sheet artifact (§6) |
| Search-loop verdicts | success bool + lane label + pathology tags | dispositions exist but are not a calculus: no UNPROVEN discipline, no certificate standing, next actions re-derived per formatter | disposition calculus + consumed-signal controller (§7) |

### 1.3 The re-founding principle

**Strip-and-replace, not reset.** Each primitive element is replaced by its principled form *behind
the same seam it already occupies*, so the compliant structure and the frozen baselines
(EXP-058, the eval sets, the v1 code paths per invariant #4) are undisturbed. The statistical
components are never deleted — they are *demoted* to the layer the thesis assigns them: optional
ordering over a space the principled layer has already generated and fenced. Every replacement lands
with its paired measurement (§9); nothing is asserted that the ledger does not carry.

---

## 2. σ_search: Navigation as Exact Identification

### 2.1 The transport license

SSL established that the specification-complexity apparatus survives noun translation when the
neighborhood is finitely enumerable and faithfully generated (SSL §6.1, with the enumeration cores
machine-checked). This document performs the **third transport** — the first was program identity,
the second a text's meaning, and `scripts/proof_knee.py` names proofs as the third domain explicitly.
The table, term by term:

| Specification Complexity (proved prior) | σ_search (this transport) |
|---|---|
| program identity | the goal's closing derivation |
| surviving mutant (unconstrained behavioral DOF) | a candidate action (tactic × premise) the current evidence leaves admissible |
| a test that kills a mutant | a structural read of the goal (bucket, domain markers, operator counts, binder shape) — or a Lean probe |
| σ = min tests to SC=1 | σ_search(g) = min structural evidence to isolate the closing action class |
| bulk phase (correlated kills) | the structural majority — one read eliminates whole tactic families |
| tail phase (independent DOF) | the hard residual — one targeted constraint per goal; Dr. Ducky's regime |
| regime = symmetry of μ | regime = symmetry of the tactic policy over goal space (§10, open) |
| composition gap γ | the multi-goal coupling cost — interface obligations between subgoals |

[Status: the transport is ASSERTED under SSL's license; the enumeration-fidelity requirement — the
candidate space is finitely enumerated by the grammar × the accessible pool, faithfully — is
satisfied *by construction* here, which is stronger ground than SSL's own case: the tactic grammar is
a closed operator dictionary exactly as Python's mutation policy is (NEG_SPEC Rem. 1.3b).]

**Definition 2.1 (the goal identification problem).** For a goal state $g$ with accessible pool
$A(g)$ and tactic grammar $\Gamma$, the candidate space is
$\mathrm{Cand}(g) = \{\gamma(p) : \gamma \in \Gamma,\ p \in A(g)\} \cup \Gamma_0$ (premise-free
closers). A *structural read* $r$ is a total computable predicate over goal text/state whose value
partitions $\mathrm{Cand}(g)$ into (still-admissible, eliminated). $\sigma_{\mathrm{search}}(g)$ is
the minimum set of reads/probes after which the admissible set is exactly the closing class.

### 2.2 The dynamics, and their existing measurement

The transported dynamics predict what the ledger already shows. The structural majority — EXP-058's
63.8% falling to `exact? + shape + induction + micro` — is the **bulk phase**: cheap correlated
elimination, structure resolving goals "for free" (SSL's $L(D)$). The hard residual decomposing into
typed families (FUTURE_DIRECTIONS' eight regimes) is the **tail**: independent degrees of freedom,
each demanding its own targeted constraint — which is *why* Dr. Ducky's per-goal typed-capsule
discipline is the correct architecture for exactly that phase and no earlier one. The knee between
them is not hypothesized: `proof_knee.py` measures the completeness knee on the traced LeanDojo
corpus, where the coverage relation is explicit. [Status: dynamics TRANSPORTED; the knee MEASURED
(third domain); the identification of EXP-058's margin with the bulk is ASSERTED, testable by
EXP-RF-002 (§9).]

### 2.3 What the encoder was

Under Definition 2.1 the encoder's role is precisely characterized: it *estimated* the outcome of
structural reads it could not name, at the cost of 22M parameters and per-state inference, with
measured direct contribution of zero closures beyond infrastructure (EXP-3.2; NAV-004). This is not
an indictment — it was the correct scaffold while the reads were unknown. The re-founding replaces
the estimator with the computed object and retires the scaffold to the statistical layer, where its
legitimate residual role is *ordering within the fenced space* (§4.3).

---

## 3. The Two-Sign Navigation Principle

### 3.1 The sign asymmetry, located in the current system

Current navigation is **positive-sign only**: retrieval (cosine or symbolic IDF) ranks what *may*
work. Its negative knowledge exists but is scattered and ungoverned — Ducky's
`suppress_numeric_solvers` hints, the domain lane policy, the regression-audit apply freeze — each a
real fence, none with a derivation record, an admissibility check, or a reporting channel. NEG_SPEC
Thm 5.2 (channel isolation) says the negative sign carries information the positive channel *cannot
derive*; the current system's own history says the same empirically: the apply-lane freeze (44 lost,
0 gained) was negative knowledge discovered by paired measurement, then hard-coded — a censor minted
without the machinery to govern or generalize it.

### 3.2 Censors for proof search

**Definition 3.1 (search censor).** A censor is a pair $c = (\phi, F)$ — a goal-shape predicate
$\phi$ (over the same structural-read vocabulary as §2) and a fenced action class $F \subseteq
\Gamma \times \mathcal{P}$ — asserted as: *no goal satisfying $\phi$ is closed by an action in $F$*,
and **derived from the observed population of search traces**, never authored a priori. The
derivation source is the spine: the 240,848-step residual corpus, the benchmark failure traces, and
rejected-program records from the Ducky executor (`engine_outcomes`/`projector_outcomes` streams).

**Definition 3.2 (admissibility; transported intact).** $c$ is admissible iff (i) **spine-sourced**
— carved from observed near-misses, structurally incapable of confirmation from the engine's own
derived output (a censor justified by "the censored lane then never fired" is self-confirming and
inadmissible; machine-checked guard: `self_confirming_cannot_certify`, `falsifiability_pivot`); and
(ii) **retained plurality** — the admissible action set for every reachable goal shape stays
non-empty. Clause (ii) is project invariant #5 ("never prune ALL candidates"), which the project
wrote by instinct years before the theory named it a well-definedness condition. [Status:
TRANSPORTED; the guards PROVED; the identification with invariant #5 is exact.]

### 3.3 The Monty Hall gate

The censor store is consulted **before any Lean call**, Wesker-Layer-2 style: compute the goal's
structural reads once, intersect with the censor store, and the fenced families are *not generated*.
The gate is lossless by the same argument as Wesker's: an admissible censor asserts the fenced
actions cannot close this goal shape, so skipping them loses nothing — and clause (ii) guarantees the
gate never empties the space. This is the information-theoretic payoff: each read's eliminations are
the "free" implicit information of the Monty Hall structure, C(H) realized at search time. The win is
measured in **budget** (attempts not spent), which is the resource-bounded axis FUTURE_DIRECTIONS
already fixed as the honest evaluation frame.

### 3.4 κ, bridges, and the bound this system proved

A candidate censor's worth is its marginal coverage $\kappa(c \mid S)$ over the obligation graph —
how many observed failure clusters it fences beyond the adopted set. Ranking is greedy by κ; the
stopping rule is the κ→0 knee, not a cap. The load-bearing negative result transports with the
machinery: the *valuable* censors — spanning goal families, fencing cross-cluster near-misses — are
**bridges**, super-additive, exactly where submodular greedy fails (NEG_SPEC Thm 14.4, measured).
The correct tractability object is bounded supermodular degree $d$, with the guarantee
$\mathrm{opt} - \mathrm{gap}_k \ge (1 - e^{-1/(d+1)})\,\mathrm{opt}$ — **`bridge_curvature_bound`,
proved in Lean by this system, locally, before its own executor was refit** (2026-08-24;
`sorries_remaining: 0`; `#print axioms` audit owed). The instrument proved the tractability theorem
of its own control theory; the re-founding closes the loop by *using* it. Obligation: measure $d$ on
the proof-network obligation graph before importing any dense-graph constant (NEG_SPEC Caveat 10.5 —
the ~3% knee and 28× drop were dense-graph numbers; the sparse rule graph kneed 3–12× later).

### 3.5 Prototypes

The entire censor stack exists as running code in Detective — `censor.py` (spine-source + plurality
guards, both censor sources), `kappa.py` (coverage, marginal-κ, graph build), `promotion_ledger.py`
(rank/promote/demote to the κ→0 fixpoint, conservative-empty on clean data by construction) — ported
there from Regenesis where the loop was first measured. The Wayfinder port changes the graph (proof
network obligations for call graph) and the spine (search traces for call sites); the mechanism
transports unchanged (NEG_SPEC Rem. 14.8).

### 3.6 The fate of `CensorNetwork`

`src/censor.py`'s learned failure predictor is the v3 plan's one model-shaped answer to a
fence-shaped question. Under the re-founding it is **demoted, not deleted**: the deterministic censor
store is the gate; the network may live in the statistical layer as an *ordering* signal over
non-fenced candidates (a prediction of failure is a prior, never a prohibition). Its asymmetric-BCE
design goal — never over-prune — is subsumed by admissibility clause (ii), which achieves it by
construction rather than by loss shaping.

---

## 4. Computed Coordinates: Banks as Structural Reads

### 4.1 The Peitho model

Peitho demonstrates the form at production scale: independent signed axes, each reading
$\{+1, -1, 0\}$ **against norms mined from the data itself**, where $0$ is a first-class verdict
("this axis has no opinion here") and a decision is *what survives elimination* — the signature is
simultaneously the decision and its complete justification. The six banks are re-founded as exactly
this: STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION each become a total, pure
function of (goal state, corpus norms) — norms extracted once from the proof network (e.g. the
depth-band and hypothesis-count distributions the DB already stores as entity positions), not learned
weights. The read vocabulary already exists in scattered deterministic form: `classify_goal_bucket`,
`goal_bucket_tags`, the marker constants of `dr_ducky.py`, `_is_arith_goal`, the domain-gate hints.
The re-founding *consolidates* them into one bank-read module with declared norms, rather than
inventing new signals. [Status: the model MEASURED in Peitho; the consolidation is BUILD; the claim
that computed reads match or beat encoder-estimated coordinates on routing is EXP-RF-003's question,
not an assumption.]

### 4.1b The Peitho geometry, transported precisely (interrogated 2026-08-24; code + design)

The Peitho probe (its `position`/`otp`/`network`/`resolve` core and DESIGN §§2–6, noting the
multi-network layer is *specified beyond the build*) sharpens §4.1 into seven transported laws:

1. **The position is five-tuple, not a sign.** `DimensionPosition(dimension, sign, depth,
   zero_state, path)`: the ternary sign for the signature, the *magnitude* retained as depth, the
   *mined zero the deviation was read against*, and the provenance path. The ledger row keeps the
   raw fact AND the read AND its norm — "the flat number is the fact; the signed position is what
   the geometry knows about it; both sit in the same row." `BankRead` adopts this shape verbatim.
2. **The mined zero is median-of-active.** Non-participants are excluded so they cannot drag the
   norm (`mine_sales_zero`: the median of *sellers*). Norms are data, re-mined when the corpus
   changes — never authored thresholds; a tolerance band around zero defines the `0` read.
3. **The discrimination guarantee is an executable check.** `discriminates(sig_a, sig_b)` =
   Hamming > 0. Two goal populations with different closing behavior sharing one signature is a
   **structural bug whose fix is a new orthogonal read off its own mined zero — never a tuned
   threshold**. This check is a first-class deliverable of EXP-RF-003, not commentary.
4. **The verdict operator is set-valued interference, not a score.** `interference(supports,
   opposes, support_min, oppose_min)` → CONSTRUCTIVE / DESTRUCTIVE / AMBIGUOUS / SILENT; one
   confident opposition vetoes a pile of weak supports; banks never fuse or average. The routing
   translation: DESTRUCTIVE = censor-grade elimination of an action family; AMBIGUOUS = the
   *escalate/disambiguate* signal (route to Ducky or a probe); SILENT = no structural opinion
   (fall through to the ranking layer).
5. **Two gate laws bind the censor layer.** From `resolve.gate`: *no signature → conservative
   keep* (elimination requires positive fence evidence, never absence of evidence), and the
   *full-signature escape hatch* (a protected class — Peitho's core inventory; here, e.g., a goal
   family under active study — is never struck by a single read's dip, only by full-signature
   decision). Both are admissibility clause (ii) made operational.
6. **The multi-network substrate, with the proof step as the transaction.** Peitho's deepest
   design element: several geometries over the *same* data, coupled by atomic primitives, the
   transaction being the seam that touches all networks at once. The proof network already holds
   five latent networks — the anchor/lens geometry (five anchor categories), the
   accessible-premises DAG, the 339K `depends_on` premise graph, the namespace hierarchy, and the
   trace-derived tactic co-occurrence structure — and **the proof step is the transaction**: one
   step touches a goal, a tactic, premises, and an outcome. The obligation graph of §3.4 is one of
   these networks (the `depends_on` geometry), not a new artifact.
7. **The cross-network resolution operator.** Where one network's read under-resolves (an
   AMBIGUOUS interference verdict), the residual is resolved by *comparing independent orthogonal
   traversals* — agreement between structurally-derived readings is itself a signal, a dimension
   beyond any literal encoding. The in-repo primitives are `compute_lens_scores`'s geometric-mean
   lens coherence and the holographic multi-projection coherence (the latter now recognizable as
   this operator's vibe-era form); Detective's parsimony consensus ("fused by lens *agreement*,
   never a weighted sum") is the same operator in its third domain. [Status: laws 1–5 BUILT and
   mutation-pinned in Peitho — transport is a port; laws 6–7 SPECIFIED in Peitho's design with
   in-repo primitives here — their full form lands with Waves 2 and 4.]

The goal-ledger consequence unifies §4 with §6: the per-goal **ledger row** (raw goal text, bucket,
the five-tuple reads, the interference verdicts, eliminations with warrants, lane history) is the
single accreting source of truth — Peitho's shadow ledger for proof states — and the cheat sheet of
§6 is a *projection* of it, exactly as Peitho's operator reports are projections of its ledgers.

### 4.2 Reads, fences, and coordinates are one vocabulary

The same structural reads serve three consumers: they *are* the σ_search tests of §2, the censor
predicates $\phi$ of §3, and the bank coordinates consumed by the network's `navigate()`. One
vocabulary, three uses — which is what makes the whole system auditable: every elimination, every
fence, and every retrieval bias traces to a named, computable read with a stored norm.

### 4.3 The statistical layer's residual role

Unchanged from THE_DETERMINISTIC_CORE, sharpened by the two-sign frame: statistical components rank
**within** the space the reads have fenced. The encoder, the family classifiers, ExecSelector — all
legitimate as ordering, all prohibited as content, and now additionally prohibited as *fencing*: a
learned signal may demote a candidate, never eliminate one. Elimination requires a warrant
(an admissible censor or a structural impossibility), because elimination is the operation that
carries the soundness burden.

---

## 5. Warranted Emission: the LeanExpr Projector

### 5.1 The Detective homology

Detective's emission stack is the model: `ExecutableProperty` (setup, assertion, category, mutant id,
confidence) → `_render_test`, where **every generated test carries its warrant as its docstring** —
and `SourceExpr`, the dual object whose `repr()` *is* its constructor source while its `value` is the
live object, imports threaded to the header. Everything that runs can be written down
round-trippably; everything written down compiles by construction; the artifact is then *verified by
running it* (`verify_under_pytest`) and folded back into the live session. The emission is a typed
pipeline with one boundary — never string assembly.

### 5.2 The upgrade

**Definition 5.1 (LeanExpr).** The emission object of the re-founded projector:
`value` — the live derivation lineage (the goal-state chain and tactic applications as replayed
against Pantograph); `expr` — the Lean source that reconstructs it (tactic block or term); `imports`
— the modules the source needs; `environment` — the toolchain/Mathlib pin the expr elaborates
against (the field Python never needed: a Lean artifact compiles only *relative to an environment*);
`warrant` — engine, bank, certificate id, skeleton lineage, and the replay evidence. `repr(LeanExpr)`
is the source seam, exactly as in `SourceExpr`.

The current `ProjectedProofProgram` already carries the warrant *lineage* (`rationale`,
`certificate_id`, `derivation_path`) — the diff is the duality and the environment: `tactics:
list[str]` becomes structured emission whose reconstruction at the v4.28 verify project
(`wayfinder_mcp._reconstruct_proof`, `_augmented_declaration`) consumes typed objects with their
imports and environment attached, instead of re-parsing strings and re-deriving preambles. Emitted
artifacts carry the warrant as a comment header — the Lean analogue of Detective's warrant docstring
— so a reconstructed proof file is self-documenting evidence, not bare text. [Status: homology
MEASURED (Detective, production); the LeanExpr object is BUILD; prototype surface:
`Detective/equivalence.py::SourceExpr` + `synthesis/writer.py` + the existing
`ProjectedProofProgram`/`_reconstruct_proof` seams.]

### 5.3 The named open edge

Lean-checked is a strictly harder target than Python-compiles, and the gap has structure the Python
prototypes never faced: elaboration is environment- and instance-dependent (an expr that "parses"
can fail on typeclass synthesis alone); universe metavariables have no Python analogue; **idiomatic
presentation** matters to the checker's practical success (term-mode vs tactic-mode forms of the
same derivation elaborate differently under time budgets); and **cross-expression definition
coupling** — `have`-chain names, local notation, `variable` blocks — means expressions are not
independently emittable the way pytest functions are: the unit of compilation is the declaration
plus its preamble closure, not the expression. These are experimental-clearing items (§10), standard
rigor-application work, and they are *why* `environment` and the preamble closure are first-class
LeanExpr fields rather than reconstruction-time inference.

---

## 6. The Residual as Artifact: the Cheat Sheet

### 6.1 The discipline

AIMO3's contribution is not the renderer but the **ledger discipline behind it**: one source of
truth per stage — every extractor emits into the ledger, every downstream reader pulls from it,
"there's nowhere else to stash information, so nothing can drift" — and the model, if consulted at
all, receives the *rendered digest*, never raw text. The rendering is deterministic
(`cheat_sheet_renderer.render_cheat_sheet`, no model in the loop).

### 6.2 The Wayfinder alphabet

The internal form already exists: the capsule (`GoalSpecification` + priors + prescriptions +
skeletons + ledger seed), the hard-resolution packets, the bridge rows, the lane history. The
re-founding adds the **renderer**: a deterministic projection of that alphabet into a transferable
artifact — the goal's structural signature (the bank reads, §4), what was eliminated and by which
censor (the negative channel, reported for the first time), what progressed and stalled per lane,
the surviving admissible frontier, and the residual goals with their typed families. A failed search
stops discarding its computed understanding at the hand-off.

### 6.3 Consumers

(i) **The Aristotle lane** — the measured Gotcha 3 (WAYFINDER_MCP_NOTES): `submit` currently proves
bare statements because the triage context is never folded into the prompt. The cheat sheet is that
fold, and `wayfinder_mcp.send_to_aristotle` is the seam. (ii) **The second-order SoM** — the packets
it trains on become *rendered, uniform* rather than ad hoc. (iii) **The human**, for whom the sheet
is the audit trail. [Status: discipline MEASURED (AIMO3); renderer BUILD; prototypes
`cheat_sheet_renderer.py` + `stage2_alphabet_instance.py`.]

---

## 7. The Search Loop as Converge

### 7.1 Disposition calculus for goals

Detective's calculus transports onto the search loop's verdicts, with the same never-promote
discipline: **closed-value** (kernel-verified, honest — the value kill); **closed-suspect**
(self-application or replay-artifact closure — the run kill: reported, never banked; the
`raw`/`honest` split already implements this instinct and is hereby named); **progressed**
(goal-state changed, decomposition-quality guard passed); **blocked — UNPROVEN** (no admissible
action closed it; *never* promoted to "unprovable" — the exact analogue of candidate-equivalent,
resolved only by a richer read, a new premise, or explicit `flag`-style judgment); **abstained —
certified** (the censor gate or an expressibility boundary fenced the whole family, with the warrant
named — Detective's internal-condition abstention); **infra** (crash/timeout, the separate ledger).

### 7.2 The consumed-signal controller

`converge_next_action`'s doctrine transports verbatim: the named defect class is *the signal was
measured and the decision did not consume it* — Wayfinder's own EXP-049 (the trigger gate that never
fired) is this defect class in its history. The re-founded loop's per-goal next action is a total,
ordered resolution over certificate standing (stale replay / ungateable measurement / unverified
reconstruction outrank everything), then dispositions — one imperative, never a menu, every branch
reading the same computed state the report prints.

### 7.3 The metric extension

`raw_success` stays the headline. The two-sign re-founding adds the axes the theory makes primary:
**budget accounting** (attempts per closure; attempts *saved* by the censor gate, reported as the
negative channel's measured worth) and the **abstention ledger** (what was fenced, under which
censor, with what κ) — so the negative channel is auditable exactly as the positive one is, and a
degenerate censor regime (over-fencing) is visible in the report rather than silent in a solve-rate
drop.

---

## 8. The Module Diff

### 8.1 Keep (compliant; untouched by the re-founding)

`tactic_compiler.py` · the deterministic lanes and `search_mode: deterministic` ·
`resolution.build_symbolic_query`/`symbolic_rank_premises` · the proof network and its schema ·
`rw_scoper.py` · `hard_data_tags.py` · the Ducky capsule/engine/projector *architecture* ·
`lean_interface.py` · the metric/reporting discipline · all frozen v1/v2 baselines (invariant #4).

### 8.2 Replace (the strip-and-replace core)

| Seam | Current occupant | Replacement | Prototype | Warrant |
|---|---|---|---|---|
| Bank coordinates | encoder→bottleneck→ternary decoder as *default* navigation input | `structural_reads.py`: computed signed reads vs corpus-mined norms; encoder demoted to optional ordering | Peitho axes; the scattered read fns already in-repo | §2.3, §4; EXP-3.2/NAV-004 zero-contribution |
| Negative knowledge | scattered suppressions + `CensorNetwork` | `censor_store.py` + `kappa.py` + `censor_ledger.py`: admissible fences, κ-ranked, Monty-Hall-gated pre-Lean | Detective `censor.py`/`kappa.py`/`promotion_ledger.py` | §3; Thm 5.2 isolation; the proved guards; `bridge_curvature_bound` |
| Emission | `ProjectedProofProgram.tactics: list[str]` + string re-parse at verify | `lean_expr.py`: dual value/source object, imports + environment + warrant; projector and `_reconstruct_proof` consume it | `SourceExpr` + `writer.py` | §5; compile-by-construction |
| Residual hand-off | raw rows/bundles; bare statements to Aristotle | `cheat_sheet.py`: deterministic alphabet renderer; wired into `send_to_aristotle` and packet freeze | AIMO3 renderer + alphabet | §6; Gotcha 3 (measured) |
| Loop verdicts | bool + tags; per-formatter next actions | disposition calculus + certificate standing + one-action controller in `proof_search`/reporting | Detective `converge_next_action`, disposition chain | §7; EXP-049 as the in-house defect witness |

### 8.3 Strip / demote (never delete)

The encoder/bridge/navigator inference path as *default* (demoted: statistical layer, opt-in);
`CensorNetwork` as gate (demoted: ordering prior); ungoverned hard-coded suppressions (absorbed:
re-derived as admissible censors with recorded spines, or retained with an explicit
`ungoverned_legacy` tag until re-derived — no silent fence survives the re-founding).

### 8.4 Size honesty

Each replacement is O(10²–10³) lines against running prototypes: the Detective censor stack is ~1.5K
lines ported across a graph change; `structural_reads` consolidates existing predicates plus a norm
miner; `lean_expr` is a dataclass + two consumption seams; the renderer is a projection. The heavy
objects are the ideas, already paid for.

---

## 9. The Experimental Program

Each experiment is paired, budget-reported, and gates the next wave. Names are canonical.

- **EXP-DET-001** (defined in THE_DETERMINISTIC_CORE): the deterministic path, bare, on the frozen
  2,000 split — the pre-refounding baseline of record.
- **EXP-RF-002 — the bulk/tail identification**: re-analyze EXP-058 + hard-run traces as a
  σ-trajectory; locate the knee; confirm the structural-majority = bulk reading against
  `proof_knee`'s corpus measurement. Analysis-only.
- **EXP-RF-003 — reads vs estimator**: computed bank reads vs NAV-004 coordinates on routing
  agreement and downstream lane-ordering quality. Gate for demoting the encoder by default.
- **EXP-RF-004 — d on the obligation graph**: build the proof-network obligation graph, measure
  supermodular degree on candidate censor sets. Determines the greedy regime honestly before any
  censor constant is trusted.
- **EXP-RF-005 — censors, paired** (the headline): deterministic path vs deterministic + censor
  gate. Primary metrics: attempts saved, solve-rate delta (must be ≥ 0), abstention ledger audit.
  The negative channel's measured worth, with the statistical layer entirely out of the room.
  **005a (offline, out-of-sample) MEASURED 2026-08:** corpus split by theorem-hash parity, store
  mined on half A (20 fences, standard floors), evaluated on unseen half B: **4,000 of 23,552
  failed lane-attempts fenced (17.0%) at ZERO fenced successes (0 of 2,010)** — the fences
  generalize exactly as channel isolation + admissibility predict. `scripts/exp_rf005_offline_split.py`.
  **005b (live Lean, paired) MEASURED 2026-08** — four-arm battery (MiniF2F ×100 cross-corpus
  store; Mathlib half-B ×150 half-A store), run twice retrieval-dark and once retrieval-live:
  **solve delta exactly 0 in every pair, every round** (the safety constraint, live: MiniF2F 40/100
  both arms; Mathlib½B 71/150 both arms retrieval-live), with **1,682 lane-removals under warrant**
  in the live round (304 + 1,378) paying **65 + 224 attempts saved (1.4% / 2.7% of budget)** — the
  negative channel's first live payment, on the budget axis §3.3 predicted and at zero solve cost.
  Two methodological findings the run itself produced: (i) the `premise_pool_counts` ledger — a
  consumed signal per §7.2 — exposed two successive retrieval starvations (a site-only pool
  disposition, then encoder-gated entity maps) that two green paired rounds could not distinguish
  from a working channel; the dark rounds still measured gate safety honestly (801 removals, delta
  0), but budget movement required the channel live. (ii) Retrieval-live perturbs the trajectory:
  the symbolic lanes' *progress* (rewrites) preempts the Ducky tail regime — all four Mathlib
  solves lost vs the dark round were prior `dr_ducky` closes whose goals were rewritten out of
  capsule form before Ducky saw them (one theorem gained the same way; net −3 on the control arm,
  orthogonal to the censor gate). The principled fix is §4.1b law 4's routing: reads that identify
  a tail-regime goal should escalate to Ducky *before* bulk rewriting — a lane-ordering question
  now carried by the ledger, not a heuristic patch.
- **EXP-RF-006 — warranted emission**: LeanExpr reconstruction vs string reconstruction at v4.28:
  compile rate, repair rate, environment-mismatch failures. The §5.3 quirks get their data here.
  **Harness BUILT + dry-mode smoked 2026-08** (`scripts/exp_rf006_emission_gate.py`): paired
  legacy/warranted emission from the same lineage, compile-surface identity modulo the warrant
  comment verified mechanically (the behavior-preservation claim); `--verify` runs the live v4.28
  gate. Wave 3 shipped: `src/lean_expr.py` (value/expr coupled by construction, environment pin,
  warrant headers) consumed by `wayfinder_mcp._reconstruct_proof`; `from_ducky_program` provides
  the projector seam for the packet writers.
- **EXP-RF-007 — the cheat-sheet fold**: Aristotle lane with vs without the rendered sheet on the
  residual set — closes the loop on the oldest measured gap in the project's notes.
- **EXP-RF-021 — the pre/post-refit proof batch**: rerun the 21-theorem SSL/Ansatz batch (closed
  2026-08-24 by the pre-refit prover) through the refit stack — bounded-nat search, CAS tier, and
  successively the re-founded elements. A fixed baseline on *novel mathematics*, free of Mathlib
  replay concerns; the cleanest longitudinal instrument the project owns.

---

## 10. Honest Boundaries and Open Problems

1. **The tactic-policy μ question** (the transport's §2.3 inheritance): σ_search is
   grammar-relative, and no canonical tactic policy exists — the same open problem the parent theory
   names for mutation policies. The grammar's closed dictionary makes the relativity explicit, not
   solved.
2. **Regime = symmetry for goal space**: which symmetry group governs the tactic policy over goal
   space, and hence when random vs greedy probing separates, is unformalized. Working guess parallel
   to NEG_SPEC Q2: typed goal-shape classes.
3. **The Lean emission quirks** (§5.3): elaboration-environment dependence, universe metavariables,
   idiomatic-form sensitivity, and preamble-closure coupling are experimental-clearing items under
   EXP-RF-006 — standard rigor application, but unfenced until measured.
4. **Censor keying granularity**: a censor keyed below the goal's semantic-equivalence class
   over-fences ("rhyming" goals trapped); above it, under-fences. Same shape as NEG_SPEC Q2;
   wants a derivation, not a heuristic.
5. **Dense-graph constants do not transfer** (standing): every κ/knee constant in this document's
   priors was measured elsewhere; nothing quantitative is asserted for the proof network until
   EXP-RF-004/005 report.
6. **`#print axioms` audit** on the 2026-08-24 Lean batch (including `bridge_curvature_bound`) is
   owed; until it runs, those results are cited proved-modulo-audit, never audited.
7. **The residual is evidence, never the specification** (the anti-overfitting discipline for the
   iterative loop). A failure autopsy *prioritizes*; it does not *warrant*. A design change is
   admissible only with all three: (i) a mechanistic justification independent of the residual —
   a native production of the tactic language, an information-preservation ordering argument, a
   read off a mined norm; (ii) a population-level measurement over the full corpus, not the
   failure set (grammar v2's warrant: 94.1% of 62,233 traced rw steps are multi-premise, 79.1%
   of simp steps multi-hint, simpa 3.3% of all 338,888 steps — the 48 autopsy rows merely
   surfaced these); (iii) out-of-sample validation (the 005a split discipline: changes derived
   from Mathlib½B failures are validated cross-corpus and on untouched splits). Parameterization
   is always by rank/read/norm, never by theorem identity. The loop's universe-redefinition step
   is population re-estimation over the residual distribution (norms, censors, IDF re-mined) —
   by construction never a per-theorem patch. Corollary boundary: have/refine/obtain (≈17% of
   corpus steps) are term-construction forms deliberately OUTSIDE the premise-pool grammar —
   Ducky/LeanExpr territory, not productions to be chased by form-mining.

---

## 11. Status Ledger

**Proved (machine-checked priors; cite, never re-derive).** σ = TD; the greedy dynamics; regime =
symmetry; γ + interface bound; channel isolation's guards (`self_confirming_cannot_certify`,
`falsifiability_pivot`); `coverage_submodular` / `marginal_antitone` / `greedy_coverage_bound`;
`bridge_curvature_bound` (proved-modulo-audit, closed locally by this system).

**Measured (elsewhere; mechanism evidence, constants non-transferable).** Censors-are-bridges;
the SSL knee (dense graph) and its sparse-graph softening; Detective's disposition calculus,
channel-propagation leaks and closures, and the theorem/prompt surface split; Peitho's
elimination architecture; AIMO3's ledger discipline; extreme-mutation prevalence (the field's own
scale data).

**Measured (in-repo).** EXP-3.2 / NAV-004 (learned lane zero direct closures); EXP-058 and the
residual structure; the apply-lane regression (the ungoverned censor); EXP-049 (the unconsumed
signal); Gotcha 3 (the discarded residual); the `proof_knee` corpus knee; **EXP-RF-003** (reads:
62→497 distinct signatures after the mined surface dimensions; contradiction 2–8%); **EXP-RF-005a**
(the negative channel, out-of-sample: 17.0% of failed lane-attempts fenced at zero fenced
successes — the two-sign claim's first in-repo measurement); **EXP-RF-005b** (the negative channel,
live kernel, paired: solve delta exactly 0 across all rounds and both corpora; 1,682 warranted
lane-removals paying 1.4%/2.7% of attempt budget with the retrieval channel live; plus the
Ducky-preemption trajectory finding and the consumed-signal starvation catches — §9).

**Asserted (this document's moves; each with its named test).** The σ_search transport (EXP-RF-002);
reads replace the estimator (EXP-RF-003); censors pay in budget at non-negative solve delta
(EXP-RF-005); warranted emission raises compile rate (EXP-RF-006); the cheat-sheet fold pays on the
Aristotle lane (EXP-RF-007).

**Build (prototyped elsewhere, to be ported).** `structural_reads` · `censor_store`/`kappa`/ledger ·
`lean_expr` · `cheat_sheet` · the disposition controller.

---

## 12. Build Order

**Wave 1 — the vocabulary** (`structural_reads.py` + norm miner; EXP-RF-002/003). Everything else
consumes the reads; nothing else can land first. **SHIPPED 2026-08** (+ Wave 1.5 surface
dimensions off the collapsed-signature audit: 62→497 distinct signatures).
**Wave 2 — the negative channel** (censor stack port; obligation graph + EXP-RF-004; mining run;
EXP-RF-005). The headline wave. **SHIPPED 2026-08** (37 censors mined; EXP-RF-005a: 17.0% budget
at zero fenced successes out-of-sample; live gate wired, paired configs staged).
**Wave 3 — warranted emission** (`lean_expr.py`; projector + reconstruction seams; EXP-RF-006).
**SHIPPED 2026-08** (MCP reconstruction emits warranted artifacts; EXP-RF-006 dry: surface
identical modulo warrant).
**Wave 4 — the artifact** (`cheat_sheet.py`; Aristotle fold; EXP-RF-007; packet-freeze
integration). **SHIPPED 2026-08** (goal ledger persisted by prove(); send_to_aristotle folds the
rendered sheet — Gotcha 3 closed; packet-freeze integration remains open).
**Wave 5 — the calculus** (dispositions + controller into the loop and reports; EXP-RF-021 runs
across all waves as the longitudinal instrument). **SHIPPED 2026-08** (seven-verdict calculus
consumed by every row via augment_result_entry; EXP-RF-021 and the live 005b/006/007 runs are the
open experimental program).

Each wave lands with its paired experiment, its Detective-pinned pure decisions, and its ledger
entry; a wave that cannot show its measurement does not merge its claims into this document — the
document is re-issued per wave with the ledger updated, exactly as the parent corpus maintains
itself.

---

*The ideas are heavy; the code is light. The theory was proved on programs, measured on meaning,
demonstrated on retail stock and competition mathematics, and — in the one closure that makes the
loop a loop — its tractability theorem was proved by the very system this document re-founds. What
remains is the porting of a few thousand lines into the seams that were built, this month, to
receive them.*
