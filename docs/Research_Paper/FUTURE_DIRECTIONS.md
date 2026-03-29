# Future Directions: From 64-69% to 80%+ with Classical AI Techniques

## Current State (March 24, 2026)

Wayfinder's current paired 2000-theorem baseline is **1277/2000 (63.8%)** under `EXP-058` (`exact? + shape + induction + micro`). Residual 2-step search now closes **94/467** started residual theorems offline, projecting **1371/2000 (68.6%)**, and the dominant `norm_cast; exact?` pattern has already been validated in the live runtime. The remaining 31-36% is still a concrete, profiled set of failures with known structural characteristics.

## Architecture Correction (2026-03-25)

The current project stack is now explicitly four-layered:

1. **Main theorem-search baseline**
   - strong cheap finisher stack (`EXP-058` + validated 2-step residual extension)
2. **Original / first-order SoM**
   - deterministically tuned control substrate
   - includes `EXP-SOM-010`, `som_torch_v1`, temporal-control, and strategy-arbiter work
3. **Dr. Ducky**
   - deterministic 1.5th-order local residual executor
   - typed capsules, proof skeletons, Lean-backed replay
4. **Second-order SoM**
   - post-benchmark orchestration layer trained on residual/gap/capsule/executor data
   - not yet the canonical theorem-search runtime

So the immediate path to 80% is no longer “just train a bigger controller.” It is:
- preserve and reuse the tuned original / first-order SoM as control substrate,
- finish the hard residual benchmark,
- deepen Dr. Ducky on local closure gaps,
- then train the second-order SoM on the benchmark’s structured residual corpus.

This document argues that classical AI techniques — most of which already exist as formal components in the broader research program — can push the system to 80%+ without requiring frontier-scale models.

---

## The Theoretical Framework

### What the Data Shows

The 44% residual decomposes into structured failure categories:

| Category | % of failures | Character |
|----------|--------------|-----------|
| Goal-start failures | ~12% | Compiler/context reconstruction |
| Single-goal stalls (exact? fails) | ~15% | Need different lemma or reasoning chain |
| Multi-goal stalls | ~10% | Need better multi-step composition |
| No-strategy failures | ~7% | Need first-move strategy selection |

Each category maps onto a specific classical AI technique that the project's broader research program has already formalized in other domains.

The experimental protocol should now treat these as separate post-benchmark worklists, not as one undifferentiated "failure" pool. From this point on, theorem-search runs should be analyzed into at least three follow-on stages: `compiler_specialist` for skipped / goal-start failures, `hard_proof_solver` for one-goal-left and small-multi-goal started residuals, and `theorem_replanner` for larger residual trees. That change is theory-aligned with COEC: constrain the post-main search space by routing each residual class into the narrowest specialist regime that still has headroom.

The competitive evaluation axis should also become explicitly resource-bounded. For Wayfinder, the meaningful question is not whether a small specialist stack matches frontier models at unconstrained scale, but how much theorem-solving quality it can recover at fixed attempt budgets and modest hardware. Every post-main stage should therefore report solve-rate curves at `128`, `256`, `512`, and `1024` attempts, and any runtime comparison produced by concurrently executed conditions should be treated as diagnostic rather than as a clean headline theorem-count result.

That implies a new data-collection strategy as well: future trace harvests should bias toward a theorem-level hard corpus rather than uniformly sampling from the full proof library. A practical default is to filter narratives into a harder quartile with a minimum proof-step floor, then collect richer residual-state traces across the current SoM interfaces. This creates the right ablation matrix for the next stage: search budget vs residual geometry vs specialist architecture.

### The Core Theory: Constraint-Oriented Proof Compilation

The project's theoretical foundation — COEC (Constraint-Oriented Emergent Computation) — frames computation as trajectory through constrained state space. Instead of specifying what a system should DO, specify what it CANNOT do. Behavior emerges from the interaction of constraints.

In proof search, this means: instead of training a model to generate the right tactic, progressively constrain the search space until the right tactic is the only remaining option. The current finisher stack already does this for roughly 64-69% of the benchmark, depending on whether one counts only the paired `EXP-058` baseline or the validated 2-step residual extension. Extending it to the remainder requires applying the same principle more aggressively.

That also means the proposed "hard proof solver" should not be a monolithic replacement for the main system. It should be a semi-isolated second-stage Society of Mind that runs only after the dominant main stack finishes, only on the `hard_proof_solver` bucket, and only after compiler/startability failures have been separated out. Otherwise the project would lose the very decomposition signal that currently makes the remaining gaps tractable.

The next layer above the raw hard bucket should also stay symbolic before it becomes learned. In practice that means a dedicated hard-resolution layer: build theorem-local prior packets from the proof graph, accessible premises, residual-goal symbols, solved-trace exemplars, residual skeleton geometry, proof-plan geometry, and multigoal dependency geometry, then let the learned hard SoM operate over those packets. That is more theory-aligned than asking a small residual network to rediscover lemma neighborhoods, representation-change pressure, and proof templates from raw residual strings alone.

That symbolic layer should now also carry two new surfaces. First, `search_control_geometry` and negative-k-line packets for plateau avoidance: repeated blank-lane identical-goal tails are not ordinary “hard proofs,” they are search-control failures that should inform replanning and learned avoidance. Second, compiler-specialist reconstruction actions: source-header replay (`variable`, `open`, `open scoped`, `include/omit`, local notation / attributes), theorem-site lookup, and symbol-name canonicalization should be explicit outputs of the compiler track rather than latent assumptions.

Early hard-trace analysis also suggests that the post-main layer should not be monolithic even internally. The live residual families are already splitting into at least eight reusable regimes: `local_eq_close` for structured equalities that need symmetric unfolding or hypothesis injection, `local_ineq_close` for geometric/order bounds, `membership_close` for opaque carrier / subobject membership walls, `exists_close` for constructive existential closure with canonical witnesses, `witness_construction_close` for existential and supremum/set-builder witnesses, `forward_context_close` for local-hypothesis / diagram-chase closures, `small_multigoal_side_conditions` for cheap side-goal sweeps, `small_multigoal_planner` for the genuinely coupled residual trees that remain after those sweeps, and `theorem_replanner` for branch-corruption cases such as metavariable spill or fold/unfold loops. The symbolic hard-resolution packets should therefore surface `specialist_targets`, `lane_suppression_hints`, and `search_pathology_tags` explicitly instead of leaving those distinctions implicit in free-form text.

The newest category-theoretic traces also suggest that `exists_close` should not stay generic. Some existential residuals are not open-ended search problems; they are canonical-witness problems. When the reduced state already exposes patterns such as `Injective I`, `Nonempty ... Splitting`, and zero morphisms in an Abelian or categorical context, the hard layer should surface zero-object or zero-morphism witness priors explicitly and let a witness-instantiation specialist try those canonical objects before broader search.

Another controller fix is now unavoidable: sanitize replay and compiler-local namespace artifacts before they reach the embedding stack. Goals containing prefixes such as `_wayfinder_replay__...` are pipeline errors, not hard mathematics, and should be routed to the compiler-specialist / goal-sanitizer track. Similarly, abstract-domain pseudo-progress that only duplicates goals after `norm_num` or `simp` should be rejected as bad progress rather than accepted into the search tree.

The same sanitation layer should also strip Lean metaprogramming wrappers such as `autoParam` and `optParam` before routing or residual labeling. Those wrappers are useful to Lean's elaborator, but they are not part of the mathematical content of the goal. In practice they should trigger a tiny Janitor-phase simplification step (`dsimp` / `simpa` over the wrappers) before the controller asks whether a state is a `forall_close`, `exists_close`, or abstract structural residual.

On the compiler-specialist side, `open scoped`-dependent failures should be promoted from vague typeclass errors into an explicit scoped-context repair regime. When a theorem-start failure combines typeclass synthesis breakdown with missing or inline-only `open scoped` context, that is strong evidence that environment reconstruction failed rather than that the theorem itself is currently out of reach.

Two concrete controller rules follow immediately from the traces. First, the Arbiter should become domain-aware enough to suppress obviously bad numeric solver moves in abstract domains such as `CategoryTheory`, `AlgebraicGeometry`, and discriminant / integrality style algebraic goals. Second, the runtime should treat newly introduced metavariables, bare type side-goals, and short-cycle state loops as replanner signals rather than as ordinary progress. A third rule is now equally clear: abstract structural theorems should prefer "close before you unpack." If the residual still exposes tokens like `IsOpenMap`, `HasRingHomProperty`, `FormallyUnramified`, `IsIso`, or `essImage`, structural retrieval should be tried before the controller unfolds them into lower-level set-topology obligations. Those are routing refinements, not a new proof theory, and they should be handled at the symbolic-control layer before any learned hard SoM is asked to rediscover them.

One benchmark hygiene rule now needs to be explicit: self-application is diagnostic only. If the runtime closes a theorem by ultimately applying the theorem itself after goal normalization or binder stripping, that trace is still useful evidence about structural setup quality, but it does not count as an honest zero-shot theorem solve. Future reports should therefore carry both `raw_success` and `honest_success`, plus a separate `self_application_successes` count. Those traces should also be excluded from k-line memory and hard-proof supervision, otherwise the second-stage learner will train on theorem leakage instead of genuine residual reasoning.

The deeper architectural version of the same rule is: learn from proof morphology, not from proof identity. A strong second-stage system should absolutely reuse the shared language of proofs — rewrite motifs, extensionality reductions, binder-handling patterns, local lemma neighborhoods, and multigoal coordination templates. What it should not do is crystallize a theorem name into a direct executable answer. That is the boundary between learned experience and theorem replay.

---

## Five Classical AI Strategies for the Remaining 44%

### Strategy 1: Negative Learning / Censors for Search Pruning

**Source concept:** Negative Learning (Minsky's censors/suppressors formalized). Learning what NOT to do is as information-dense as learning what to do — achieving 22x sample efficiency in prior experiments.

**Application to proof search:**

The current system tries tactics and records whether they succeed. It does NOT systematically learn from failures. The 164 failed apply attempts, the 484 trigger fires that didn't lead to proves, the exact? calls that fail — these are all negative training signal that the system currently discards.

**Concrete implementation:**

1. **Tactic censors**: For each goal shape, learn which tactic families are guaranteed to fail. A goal that's pure arithmetic never needs `apply`; a goal that's a set containment never needs `omega`. The censor fires first, suppresses the useless lanes, and the search only tries what might work.

2. **Premise censors**: For each goal, learn which premises are guaranteed to fail unification. Instead of trying all 20 cosine-ranked premises, suppress the ones whose type signature is structurally incompatible. The current 34.6% accept rate means 65.4% of attempts are wasted — a censor trained on those negatives could cut that to <30%.

3. **Strategy censors**: At the theorem level, learn which proof templates are guaranteed to fail for a given goal shape. REWRITE_CHAIN on a theorem that requires induction is always wrong. A template-level censor would redirect the search to the right template from the start.

**Expected impact:** Reducing wasted search budget by 50-70% means more budget available for productive exploration. Estimated +5-10% prove rate from budget reallocation alone.

### Strategy 2: Near-Miss Learning for Exact? Distillation

**Source concept:** Winston's structural learning from almost-correct examples. A near-miss identifies exactly which feature matters.

**Application to proof search:**

The 29 near-miss theorems (1 goal remaining, apply progress but not proved) and the 153 `exact?` solutions are the richest training signal the project has ever produced. Each `exact?` success says: "for this goal, the correct closing tactic is `exact TheoremName arg1 arg2`."

**Concrete implementation:**

1. **Harvest exact? solutions at scale**: Run the 2000-theorem benchmark with `exact?` and capture all solutions. This produces ~600-800 (goal, closing_tactic) pairs — a labeled dataset of final-goal closures.

2. **Learn the closing pattern cautiously**: Many captured `exact?` solutions reduce to self-application (`exact TheoremName local_args`). These are useful as diagnostics for binder handling and normalization quality, but they should be tracked separately from honest theorem solving. The right use is as structural telemetry or as a local theorem-environment audit, not as a headline closer metric.

3. **Learn from near-misses**: The 20 near-miss theorems where `exact?` fails are the hardest cases. These need a *different* lemma, not the theorem itself. Use the failure diagnosis (which `exact?` tactic was tried and why it failed) as negative supervision — the near-miss tells you exactly what the right answer is NOT, which constrains the search.

4. **Hard-negative contrastive training**: Following the Balanced Sashimi approach (hard-negative contrastive learning = Winston's near-miss learning applied to neural training), train the closing model on (correct_tactic, near_miss_tactic) pairs. The almost-correct tactic is the most informative training example.

**Expected impact:** Replace `exact?` with a learned closer that runs in <0.1s instead of 5-10s, while preserving 80-90% of its accuracy. Estimated +3-5% prove rate from faster closing (more budget for exploration) plus +5-8% from learning from near-misses.

### Strategy 3: Society of Mind Specialist Decomposition

**Source concept:** Minsky's SoM. Intelligence emerges from the interaction of many simple, specialized agents — not from a single sophisticated reasoner.

**Application to proof search:**

The current system uses a strong lane-separated execution stack plus residual SoM components. The PyTorch SoM family router is now trained and reaches **95.6%** top-1 / **99.4%** top-3 tactic-family accuracy on held-out step data, but it is not yet the canonical theorem-search router.

The new data changes this: `EXP-058` traces, the 328K-example multi-step dataset, the residual audits, and the validated 2-step closures provide enough signal to train and benchmark theorem-search routing rather than only discussing it architecturally.

**Concrete implementation:**

1. **Template-conditioned first-move selection**: Instead of always starting with IB + structural, use the template classifier to predict the proof template (REWRITE_CHAIN, APPLY_CHAIN, INDUCT_THEN_CLOSE, etc.) and start with the appropriate lane order. The template classifier already works at 100% coverage over 9 templates.

2. **K-line memory for proved theorems**: Minsky's K-lines re-activate the agents involved in a prior solution. Implementation: for each proved theorem, store the (template, lane_sequence, closing_tactic) triple. When a new theorem has similar goal structure, activate the same agent configuration. The 560+ proved theorems (2000-theorem baseline) + 279 (with exact?) provide the K-line database.

3. **Specialist executors**: Train separate small models for each template family:
   - REWRITE_CHAIN specialist (37% of proofs): already solved by cosine_rw
   - APPLY_CHAIN specialist (8%): the ExecSelector v2 + trigger
   - INDUCT_THEN_CLOSE specialist (3%): needs induction first-move + structural closing
   - DECOMPOSE_AND_CONQUER specialist (34%): needs case splitting + parallel subgoal work

4. **Arbiter agent**: The temporal controller, retrained on the 279+ proved traces, acts as Minsky's arbiter — selecting which specialist to deploy at each step.

**Expected impact:** Template-conditioned search should reduce wasted first-move attempts by 30-50%. Specialist executors for induction and case analysis are new lanes that directly address category 4 failures (no-strategy). Estimated +8-15% prove rate.

### Strategy 4: COEC Constraint Propagation for Multi-Step Planning

**Source concept:** COEC (Constraint-Oriented Emergent Computation). Behavior emerges from the interaction of constraints, not from explicit instruction.

**Application to proof search:**

The multi-goal stalls (category 3) fail because the search doesn't plan — it tries lanes on each goal independently, without considering how closing one goal affects another. COEC's constraint propagation principle says: propagate the constraints from each subgoal to the others, and the solution emerges from their intersection.

**Concrete implementation:**

1. **Subgoal dependency analysis**: After apply opens N subgoals, analyze their dependencies. Some subgoals share variables — closing one may instantiate variables needed by another. Build a constraint graph over subgoals.

2. **Constraint-directed goal ordering**: Instead of round-robin or priority-queue goal selection, order goals by constraint propagation: close the goal that most constrains the remaining goals first. This is the COEC principle applied to proof planning.

3. **Backward constraint reasoning**: From the remaining unclosed goal, propagate constraints backward through the proof tree. If the unclosed goal needs `x : Nat`, and a closed sibling provides `x`, the constraint tells us to try the sibling's proof output as input.

4. **Energy-based search**: The Phase 7 energy-constrained navigation framework (OTP scoring + COEC constraints + energy minimization) provides the formal foundation. Each proof state has an "energy" based on how constrained it is. The search follows the energy gradient — the most constrained state is the most solvable.

**Expected impact:** Directly addresses the 10% multi-goal stall category. Estimated +5-8% prove rate from better multi-step composition.

### Strategy 5: Geometric Semantic Encoding for Proof-Space Navigation

**Source concept:** GSE (Geometric Semantic Encoding). Meaning arises from position in structured geometric space, not from learned embeddings.

**Application to proof search:**

The current 6-bank ternary navigation (STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION) was designed as a GSE instantiation. But it's been underutilized — the navigational signal contributes zero proved theorems beyond what infrastructure provides (NAV-004 result).

The insight from the ARC solver work: GSE's power is not in predicting the answer, but in **constraining the search space geometrically**. The 6-bank positions define a 729-bin direction space. Each bin contains ~330 entities (242K / 729). The navigation should not predict the exact premise — it should constrain the search to the right bin, then let cosine/exact? handle the remaining selection.

**Concrete implementation:**

1. **Navigational constraint propagation**: Use the 6-bank position of the current goal to constrain which premises are considered. Instead of searching all 242K entities, search only the ~330 in the predicted direction bin. This is a 700x reduction — enough to make global `exact?`-style search feasible on every goal, not just the last one.

2. **Multi-scale lens architecture**: Following the ARC solver's multi-scale approach, evaluate each goal at multiple resolution levels:
   - Coarse: which template family? (9 bins)
   - Medium: which direction bin? (729 bins)
   - Fine: which premises? (~11 candidates)

   Each level constrains the next. The current system only uses the fine level.

3. **Proof trajectory as geometric path**: A proof is a path through the 6-dimensional ternary space. Successful proofs follow characteristic trajectories for each template. The K-line memory (Strategy 3) stores these trajectories. When navigating a new proof, the system follows the nearest successful trajectory, deviating only when constraints force it.

4. **OTP scoring for goal assessment**: The ternary structure {support, contradict, irrelevant} maps onto {+1, -1, 0}. For each candidate tactic, compute the OTP score: how many constraints does it satisfy (+1), violate (-1), or leave unchanged (0)? Choose the tactic with the best OTP score. This is Negative Learning applied to tactic selection.

**Expected impact:** Navigational constraints reduce the search space by orders of magnitude. Combined with strategy 1 (censors), the effective search space per goal drops from 242K to ~50 entities. Estimated +5-10% prove rate from more efficient search.

---

## Combined Projection

| Strategy | Estimated lift | Effort | Dependencies |
|----------|---------------|--------|-------------|
| 1. Negative Learning / Censors | +5-10% | Medium | Training data from failures |
| 2. Near-Miss Exact? Distillation | +8-13% | Medium | Exact? harvest (running) |
| 3. SoM Specialist Decomposition | +8-15% | Large | Template classifier + TC |
| 4. COEC Constraint Propagation | +5-8% | Medium | Subgoal analysis infrastructure |
| 5. GSE Navigation Constraints | +5-10% | Medium | Bank position improvements |

**Conservative estimate (no overlap):** +15-20% → **71-76%**
**Optimistic estimate (synergy):** +25-30% → **81-86%**

The strategies are complementary, not competing. Censors (Strategy 1) reduce wasted search. Near-miss learning (Strategy 2) improves the closer. Specialists (Strategy 3) improve first-move selection. Constraints (Strategy 4) improve multi-step planning. Navigation (Strategy 5) improves premise retrieval.

---

## The Path to 80%

### Phase 1: Harvest and Distill (1-2 months)
- Complete exact? harvest on 2000 theorems
- Train exact? distillation model from harvested solutions
- Implement basic tactic censors from failure data
- Deploy template-conditioned first-move selection

### Phase 2: Specialize and Compose (2-3 months)
- Train template-specific specialists (induction, case analysis)
- Retrain temporal controller on 600+ proved traces
- Implement K-line memory for proof trajectories
- Deploy constraint-directed goal ordering

### Phase 3: Navigate and Refine (2-3 months)
- Improve 6-bank navigational constraints
- Implement multi-scale lens evaluation
- Deploy OTP scoring for tactic selection
- Energy-based search integration

### Phase 4: Validate at Scale (1 month)
- Full Mathlib test split (5000+ theorems)
- Ablation studies for each strategy
- Publication-ready results

**Total estimated timeline:** 6-9 months to reach 80%+ on full Mathlib benchmark.

---

## Why This Is Tractable

Every strategy described above uses techniques that have been formalized and validated in the broader research program:

- Negative Learning achieved 22x sample efficiency in prior experiments
- Near-Miss Learning (Winston) is a 50-year-old technique with proven efficacy
- SoM specialist decomposition is the core of the ARC solver architecture
- COEC has a formal proof of Turing universality and 9 computational classes
- GSE/OTP has been implemented in 3 separate domains (theorem proving, code quality, genomics)

The 44% residual is not "hard AI" in the general sense. It's a structured set of failures with known characteristics, addressed by known techniques that have been validated in related domains. The infrastructure that handles the first 56% provides the scaffolding — the structured residual output, the lane attribution, the failure diagnostics — that makes each successive strategy cheaper to implement and validate.

The honest uncertainty: the strategies may not compose linearly. Some theorems may resist all five approaches. The 80% target is achievable but not guaranteed. The 70% target (conservative estimate) is high-confidence.

---

*Future Directions document v1.0 — March 2026. Draws on the broader research program: Society of Mind, Negative Learning, Near-Miss Learning, COEC, GSE/OTP, ARC solver architecture.*
