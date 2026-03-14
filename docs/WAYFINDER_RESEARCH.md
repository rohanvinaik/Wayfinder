# Wayfinder: Navigational Proof Search Through Structured Semantic Networks

**Principal Investigator:** Rohan Vinaik
**Research Partner:** Claude (Opus 4.6), Anthropic
**Date:** March 10, 2026
**Status:** v2.0 — Society of Mind architecture. v1.1 monolithic navigator trained (NAV-001/002). Pivoting to specialist decomposition.
**Project:** Wayfinder — Navigational theorem proving for Lean 4

---

## Abstract

We propose a novel approach to neural theorem proving that treats proof search as *navigation through structured semantic space* rather than token prediction. The system, **Wayfinder**, combines three architectural ideas that have not previously been integrated:

**Navigational claim.** A structured semantic network — where mathematical entities (lemmas, tactics, proof states) are positioned along orthogonal signed dimensions and connected through a shared anchor dictionary — enables premise selection and proof search via deterministic symbolic operations (IDF-weighted set intersection, Bellman-Ford spreading activation, multiplicative bank alignment) that are faster, more auditable, and more precise than dense embedding retrieval.

**Architectural claim.** A hybrid continuous-ternary architecture — with a continuous goal encoder, a learned information bottleneck, and a ternary {-1, 0, +1} navigational decoder — produces *directional coordinates in proof space* rather than vocabulary indices. These coordinates resolve to concrete tactics and premises through the semantic network. The neural network runs once per proof state; all subsequent search operations are pure arithmetic on structured data.

**Decomposition claim (v2).** Specification complexity theory provides a formal criterion for decomposing a monolithic proof search system into a *Society of Mind* architecture — multiple specialist models operating in typed temporal slots (PERCEPTION → RECOGNITION → PLANNING → EXECUTION → VERIFICATION). The composition gap theorem (σ(A∘B) ≤ σ(A) + σ(B) + γ(A,B)) guarantees that when γ vanishes (independent specialists), total specification complexity is additive, not multiplicative. PAB stability per specialist serves as the empirical proxy for specification complexity, guiding decomposition until every component reaches the "stable" regime. Narrative construction converts hard proof structure prediction (Regime B, low symmetry, exponential σ) into tractable template classification (Regime A, high symmetry, polynomial σ).

**Epistemological claim.** The architecture produces measurable learning trajectories, enabling Process-Aware Benchmarking (PAB) to evaluate *how* it learns to navigate, not just whether proofs close. Progress prediction — estimating remaining proof steps — connects PAB's trajectory analysis to actionable training signals and search heuristics.

The system draws on three distinct intellectual traditions: (1) the navigational semantic network architecture of ModelAtlas (Vinaik, 2025), which positions ML models in a structured coordinate system for deterministic similarity queries; (2) the hybrid continuous-ternary decoder of Balanced Sashimi (Vinaik & Claude, 2026), which uses {-1, 0, +1} weights for categorical decisions; and (3) the formal convergence guarantees of Mutation Theory (formalized in Lean 4), which proves trajectory monotonicity, phase transitions, and fixed-point partitions applicable to structured search processes.

The experimental target is Lean 4 proof generation over Mathlib, evaluated against ReProver (Lean-Dojo), LeanProgress, and DeepSeek-Prover-V2 baselines.

---

## 1. Introduction

### 1.1 The Navigation Problem in Proof Search

Every competitive neural theorem prover uses the same fundamental approach: encode the proof state as a dense vector, predict the next tactic token from a vocabulary, and embed the prediction in a tree search (BFS, MCTS, or best-first). Premise selection — choosing which lemmas from a library of ~100k candidates are relevant — is handled by a learned retriever that computes dot-product similarity between the proof state embedding and pre-computed premise embeddings.

This approach works. ReProver (Yang et al., 2024) achieves state-of-the-art premise retrieval via an encoder that jointly embeds proof states and premises. DeepSeek-Prover-V2 (Ren et al., 2025) achieves 88.9% on MiniF2F-test via subgoal decomposition and reinforcement learning. HTPS (Lample et al., 2022) uses AlphaZero-inspired MCTS with a policy-critic architecture.

But the approach has a structural limitation: **the intelligence is in the wrong place.** The neural network must run at every search node — every candidate tactic, every candidate premise, every expansion of the proof tree. Search budgets are dominated by neural forward passes. A single MiniF2F proof attempt may require thousands of forward passes through a multi-billion parameter model.

Meanwhile, the *structure* of mathematical knowledge is rich, regular, and well-understood. Mathematicians don't search for premises by computing similarity in a continuous embedding space. They navigate structured relationships: "this is an inequality, so I need monotonicity lemmas; this is about finite sets, so I need the Finset API; I already used `linarith` here, so the premises should be arithmetic." This navigation is *combinatorial* — it operates on discrete properties, categorical relationships, and structured hierarchies.

What if we could build a system where the neural network handles the hard part (understanding the proof state, predicting which *direction* to go) while search and retrieval are handled by deterministic symbolic operations on structured data?

### 1.2 The ModelAtlas Paradigm

ModelAtlas (Vinaik, 2025) solves an analogous problem in a different domain: navigating a space of ~40,000 ML models on HuggingFace to find the ones most relevant to a user's needs. Its core insight is that structured navigation through signed semantic coordinates outperforms both flat database queries and dense embedding retrieval for this task.

The key mechanisms:

1. **Signed orthogonal banks.** Eight independent dimensions (ARCHITECTURE, CAPABILITY, EFFICIENCY, etc.), each with a **zero state** placed at the most common query target. Models are positioned as signed distances from zero. A 3B model is at EFFICIENCY = (-1, 2), meaning "two steps in the negative direction from the 7B zero." Positions enable gradient scoring: a query for "small" gives full marks to negative-efficiency models, partial marks to zero-efficiency, and decaying marks to positive.

2. **Anchor dictionary.** A shared vocabulary of ~130+ semantic labels (e.g., "instruction-following", "GGUF-available", "Llama-family"). Models link to their applicable anchors. Similarity is *emergent* — two models sharing 15 anchors are related without an explicit edge. No embeddings are computed or stored.

3. **IDF-weighted scoring.** Rare anchors (present on few models) count far more than ubiquitous ones. Sharing the anchor "proof-assistant" (12 models) matters more than sharing "decoder-only" (17,000 models). This prevents common features from dominating retrieval.

4. **Multiplicative composition.** The final score is `bank_alignment × anchor_relevance × seed_similarity`. A zero in any component kills the result. A model that perfectly matches efficiency but completely fails capability scores 0, not 0.5. This prevents averaging away bad matches.

5. **Spreading activation.** Bellman-Ford propagation from seed nodes through both explicit links (fine_tuned_from, variant_of) and implicit connections (shared anchors). Bank scoping prevents semantic bleeding — spreading through irrelevant dimensions is suppressed.

6. **Zero neural inference at query time.** The entire query engine runs on SQLite, integer arithmetic, set intersection, and logarithmic decay. Full semantic decomposition was done once at ingestion. Queries execute in milliseconds.

This paper asks: **can the same paradigm be applied to mathematical theorem proving?**

### 1.3 The Lean 4 / Mathlib Testbed

Lean 4 with Mathlib provides an ideal testbed for navigational proof search:

- **Formally verified output**: The Lean kernel provides unambiguous pass/fail verification. No proxy metrics, no subjective scores.
- **Structured mathematical entities**: Mathlib's ~100k lemmas have rich type information, namespace hierarchies, and dependency graphs — natural structure for bank positioning and anchor assignment.
- **Finite tactic vocabulary**: The set of commonly-used Lean 4 tactics is bounded (~50 core tactics, ~200 with variants). This is a navigational decision, not open-ended text generation.
- **Hierarchical proof structure**: Proofs are trees of subgoals, not flat sequences. Each subgoal has a structured goal state with hypotheses, target type, and local context.
- **Existing benchmarks**: MiniF2F (488 problems), Mathlib test split (~10k theorems), and full Mathlib (~100k theorems) provide standardized evaluation at multiple scales.

### 1.4 Research Questions

This work addresses five interconnected questions:

1. **Navigational (Q1)**: Can structured semantic navigation — bank positions, IDF-weighted anchors, and spreading activation — match or exceed dense embedding retrieval for premise selection from Mathlib?

2. **Architectural (Q2)**: Can a ternary {-1, 0, +1} decoder that produces navigational coordinates (directions in proof space) outperform classification-based tactic prediction, by leveraging the semantic network for resolution?

3. **Search efficiency (Q3)**: Does separating the neural network (runs once per proof state) from the search engine (runs deterministic symbolic operations) enable faster proof search with equivalent or better success rates?

4. **Process (Q4)**: Does the architecture's learning trajectory satisfy PAB's quality criteria? Can progress prediction (remaining proof steps) serve as both a training signal and a search heuristic?

5. **Convergence (Q5)**: Do the formal guarantees from Mutation Theory (trajectory monotonicity, phase transitions, fixed-point partitions) apply to this navigational search process, providing formal convergence bounds?

### 1.5 Scope and Stance

This is a research project with three streams:

- **Primary objective**: A working theorem prover that navigates proof space via structured semantic networks, evaluated on standard Lean 4 benchmarks against ReProver, LeanProgress, and DeepSeek-Prover-V2 baselines.
- **Co-equal objective**: Empirical validation that navigational search (symbolic operations on structured data) can compete with or exceed learned retrieval (neural forward passes) for premise selection.
- **Tertiary objective**: Empirical validation of PAB's trajectory evaluation framework in the theorem-proving domain, with progress prediction as a novel PAB metric.

**Non-goal**: Beating DeepSeek-Prover-V2's 88.9% on MiniF2F. That system uses a 671B parameter model and reinforcement learning. Our contribution is architectural: demonstrating that a small model (sub-100M trainable parameters) with structured navigation can achieve competitive results by spending compute on symbolic search rather than neural inference.

---

## 2. Theoretical Framework

### 2.1 Proof Search as Spatial Navigation

We propose a fundamental reframing: **proof search is navigation through a structured mathematical space, not sequential token prediction.**

In token prediction, the model outputs a sequence of tactic tokens conditioned on the goal state. Each token is selected from a flat vocabulary via softmax. The model must learn, from training data alone, which tactics are applicable in which situations — the structure of mathematical knowledge is implicit in the weights.

In navigational search, the model outputs *directions* in a structured coordinate system. The directions resolve to concrete tactics and premises through deterministic lookup in a semantic network that *explicitly encodes* the structure of mathematical knowledge. The model's job is narrower: predict which region of mathematical space contains the answer. The symbolic system handles the last mile.

This reframing has three theoretical consequences:

**Consequence 1: Separation of concerns.** The neural network handles understanding (what kind of mathematical situation is this?) while the symbolic system handles retrieval (given this kind of situation, which specific lemmas and tactics are relevant?). Neither system needs to do the other's job. This is the same separation that makes ModelAtlas work: the LLM decomposes the user's natural language query into structured parameters; ModelAtlas does deterministic math.

**Consequence 2: Auditable search.** Every premise retrieval traces back to specific shared anchors and bank alignments. "This lemma scored 0.87 because it shares the rare anchors `monotone-convergence` and `finset-sum` with the current goal, and its DOMAIN bank position aligns with the goal's." Compare to dense retrieval: "the embedding similarity was 0.87" (uninspectable).

**Consequence 3: Amortized computation.** The semantic network is built once from Mathlib's type information, namespace structure, and dependency graph. At query time, navigation is pure arithmetic on precomputed structures. The neural network runs once per proof state to produce coordinates; the search engine can explore thousands of candidate lemmas in the time it takes to run one forward pass.

### 2.2 Signed Semantic Banks for Mathematical Entities

We define six orthogonal **proof banks** — signed dimensions with zero states placed at the mode of mathematical activity:

```
Bank           Zero State                  Negative                   Positive
────────────── ─────────────────────────── ────────────────────────── ─────────────────────────
STRUCTURE      equality / rewrite goal     arithmetic, decidable      quantified, higher-order
DOMAIN         general algebra             concrete (ℕ, ℤ, ℚ)        abstract (topology, category)
DEPTH          2-3 tactic proof            trivial (1 tactic)         deep (10+ tactics)
AUTOMATION     partially automated         fully auto (omega, simp)   requires manual insight
CONTEXT        moderate hypotheses (3-5)   no hypotheses              rich context (10+ hyps)
DECOMPOSITION  single-goal                 atomic (no subgoals)       multi-subgoal
```

**Why these banks are orthogonal:** A theorem can be structurally simple (STRUCTURE=0) but in an abstract domain (DOMAIN=+2); it can be deeply nested (DEPTH=+2) but fully automated (AUTOMATION=-2, e.g., a long `simp` chain). The banks capture independent dimensions of variation.

**Zero placement rationale:** Zero is "the most common proof goal a working mathematician encounters." The majority of Mathlib lemmas are about algebraic equalities or inequalities, require 2-3 tactics, and use a mix of automation and manual reasoning. Queries resolve near the origin.

**Positioning examples:**

| Lemma | STRUCTURE | DOMAIN | DEPTH | AUTOMATION | CONTEXT | DECOMP |
|-------|-----------|--------|-------|------------|---------|--------|
| `Nat.add_comm` | (0,0) | (-1,1) | (-1,2) | (-1,2) | (-1,1) | (-1,1) |
| `Set.Finite.induction_on` | (+1,2) | (+1,1) | (+1,1) | (+1,2) | (+1,1) | (+1,2) |
| `MonoidHom.map_mul` | (0,0) | (0,0) | (-1,1) | (-1,1) | (0,0) | (-1,1) |
| `TopologicalSpace.isOpen_iUnion` | (+1,1) | (+1,2) | (0,0) | (+1,1) | (+1,1) | (+1,1) |

Bank positions are extracted deterministically from Lean's type information:
- **STRUCTURE**: Analyze the goal type — quantifier nesting, dependent types, propositional vs. data.
- **DOMAIN**: Namespace hierarchy (`Mathlib.Algebra` → DOMAIN=0, `Mathlib.Topology` → DOMAIN=+2).
- **DEPTH**: Proof length from ground-truth proofs (when available) or heuristic estimation from type complexity.
- **AUTOMATION**: Whether the lemma's proof (or similar lemmas' proofs) use automation tactics.
- **CONTEXT**: Number and complexity of hypotheses in the local context.
- **DECOMPOSITION**: Whether the proof requires multiple `have` statements or case splits.

### 2.3 Anchor Dictionaries as Mathematical Vocabulary

An **anchor** is a semantic label shared across mathematical entities. Anchors create emergent similarity — two lemmas sharing 12 anchors are related without an explicit edge. The anchor dictionary is the semantic vocabulary of the system.

**Proof anchors by bank:**

| Bank | Example Anchors |
|------|----------------|
| STRUCTURE | `equality`, `inequality`, `membership`, `universal-quantifier`, `existential`, `iff`, `implication`, `negation`, `dependent-type` |
| DOMAIN | `nat-arithmetic`, `group-theory`, `ring-theory`, `topology`, `measure-theory`, `linear-algebra`, `order-theory`, `set-theory`, `category-theory` |
| DEPTH | `one-liner`, `multi-step`, `deep-induction`, `structural-recursion` |
| AUTOMATION | `omega-solvable`, `simp-solvable`, `decide-solvable`, `norm_num-solvable`, `needs-manual-intro`, `needs-manual-cases` |
| CONTEXT | `hypothesis-rich`, `hypothesis-free`, `uses-local-def`, `type-class-instance` |
| DECOMPOSITION | `needs-cases`, `needs-induction`, `needs-have-chain`, `single-goal`, `multi-branch` |

**Cross-cutting anchors** (not bank-specific):
`monotonicity`, `commutativity`, `associativity`, `distributivity`, `idempotency`, `transitivity`, `injectivity`, `surjectivity`, `continuity`, `compactness`, `finiteness`, `convergence`, `bounded`, `well-founded`

**Anchor extraction:**
- **Tier 1 (deterministic)**: From Lean's type checker — `Nat` in the type → `nat-arithmetic`; `∀` in the goal → `universal-quantifier`; namespace `Mathlib.Topology` → `topology`.
- **Tier 2 (pattern matching)**: From proof text — `omega` in proof → `omega-solvable`; `induction` → `needs-induction`; `rw [mul_comm]` → `commutativity`.
- **Tier 3 (optional)**: LLM-based annotation for subtle mathematical properties not captured by syntax.

**IDF weighting:** The anchor `equality` appears on thousands of lemmas. The anchor `Lyapunov-monotonicity` appears on maybe 5. IDF = `log(N / count_lemmas_with_anchor)`. When a goal state shares both anchors with a candidate lemma, the IDF weighting correctly focuses on the rare, informative one. This mirrors how human mathematicians search: they key off the unusual, distinctive features of a problem.

**Anchor gap analysis (critical bootstrap step):** The ~300 initial anchors will have blind spots. The system's retrieval ceiling is bounded by anchor expressiveness — if a relevant mathematical property isn't captured by any anchor, it's invisible to `navigate()`. Known gaps in the initial dictionary include:

- **Type coercion anchors**: Many Mathlib proofs struggle with coercion between types (`↑n`, `(n : ℤ)`, `Nat.cast`). This is a major source of proof difficulty and needs dedicated anchors: `needs-cast`, `nat-int-coercion`, `subtype-coercion`, `coe-simp`.
- **Algebraic hierarchy anchors**: The Mathlib typeclass hierarchy (Monoid → Group → Ring → Field) creates proof obligations that are structurally similar but domain-specific. Anchors should track position in the hierarchy: `monoid-level`, `group-level`, `ring-level`, `field-level`.
- **Proof *pattern* anchors**: Beyond tactic names, recurring proof patterns need labels: `split-and-recombine`, `contrapositive-argument`, `epsilon-delta`, `sequence-limit`, `diagonal-argument`.

**Anchor gap analysis procedure** (executed iteratively in Phase 0-1):
1. For 500 randomly selected proof steps, attempt to resolve the ground-truth tactic and premises via `navigate()` using only the existing anchor dictionary.
2. For each failure (correct premise not in top-16), examine what anchors *would have* connected the goal to the premise. These are gap anchors.
3. Cluster the gap anchors by mathematical theme. Add the most frequent clusters as new anchors.
4. Re-run the analysis. Iterate until top-16 recall on the 500-step sample exceeds 70%.

This iterative refinement is the single highest-leverage activity in Phase 0. The proof network's quality caps the entire system.

### 2.4 Premise Retrieval as Structured Navigation

With lemmas positioned in banks and connected through anchors, premise retrieval becomes a `navigate()` query — the same operation ModelAtlas uses to find ML models:

```
retrieve(goal) =
    bank_alignment(goal_position, lemma_position)      # Are we in the right region?
  × anchor_relevance(goal_anchors, lemma_anchors, IDF)  # Do semantic labels match?
  × seed_similarity(context_lemmas, lemma_anchors, IDF) # Similar to what we've already used?
```

**Three signals, multiplicative:**

1. **Bank alignment.** For each bank where the goal has a position, score the candidate lemma's alignment:
   - Same direction: 1.0 (fully aligned)
   - At zero: 0.5 (neutral)
   - Opposite direction: `1 / (1 + |distance|)` (decaying penalty)
   - Multiplicative across banks — a lemma that matches DOMAIN but opposes STRUCTURE gets penalized.

2. **Anchor relevance (IDF-weighted).** Three anchor lists:
   - **require**: Hard filter. The goal has `nat-arithmetic` and `inequality` → only lemmas with both anchors survive.
   - **prefer**: Soft boost. The goal also has `monotonicity` → lemmas with this anchor get IDF-weighted bonus.
   - **avoid**: Penalty. The search has already tried `omega` and failed → penalize `omega-solvable` lemmas.

3. **Seed similarity.** If the proof has already successfully applied certain lemmas, find candidates with overlapping anchor sets. IDF-weighted Jaccard between the seed's anchors and the candidate's anchors.

**Why multiplicative scoring matters for premises:** In dense retrieval, a lemma about natural numbers that happens to have similar embedding structure to a topology proof can score highly — the embedding conflates all dimensions. With multiplicative scoring, if the DOMAIN bank doesn't align, the score goes to zero. `Nat.add_comm` is never relevant to a topology proof, no matter how textually similar the goal states look.

**Scoring mechanism — open design question**: Pure multiplicative scoring (`Π bank_score_i`) is both the system's precision advantage and its fragility. A zero in any bank kills the total score. When bank positions are noisy (and they will be, especially early), this means a misclassified STRUCTURE position can exclude correct premises that match perfectly on every other dimension.

Several alternative scoring mechanisms should be evaluated empirically:

| Mechanism | Formula | Tradeoff |
|-----------|---------|----------|
| Pure multiplicative (ModelAtlas) | `Π score_i` | Maximum precision, zero tolerance for noise |
| Confidence-weighted multiplicative | `Π score_i^(confidence_i)` | Banks with uncertain positions contribute less; requires per-bank confidence estimation |
| Geometric mean | `(Π score_i)^(1/k)` | Softer than product; a single low score reduces but doesn't kill |
| Log-additive with learned weights | `Σ w_i · log(score_i)` | Equivalent to weighted geometric mean; weights learned per bank |
| Soft-floor multiplicative | `Π max(score_i, ε)` | Floor prevents hard zeros; ε tunable per bank |
| Hybrid: multiplicative filter + additive rank | Filter by hard bank constraints, then rank by additive anchor+seed score | Separates coarse filtering from fine ranking |

The **confidence-weighted** variant is theoretically most principled: if the GoalAnalyzer's bank head outputs a softmax distribution where the winning class has only 40% probability (uncertain), that bank should influence the final score less than one where the winning class has 95% probability. This naturally degrades to pure multiplicative when confidence is high and to no-bank-influence when confidence is low.

The right mechanism likely depends on proof network maturity. Early (noisy bank positions): confidence-weighted or soft-floor. Late (refined positions): pure multiplicative. This could be a training schedule parameter.

**Accessible-premises filtering (free lunch from ReProver).** ReProver demonstrated that restricting retrieval to premises accessible via Lean's import structure (~33k of ~130k total) gives a ~2% recall improvement at zero compute cost. Wayfinder's proof network should encode this constraint: each entity stores its import-accessible set, and `navigate()` pre-filters to accessible premises before scoring. This is a hard filter in SQL (`WHERE entity_id IN accessible_set`), applied before any bank alignment or anchor scoring. LeanDojo's extraction already provides this accessibility information.

**Performance:** Batch SQL replaces per-lemma neural forward passes. Pre-filter by accessible premises AND required anchors (SQL `HAVING`), batch-fetch positions and anchor sets, score in Python from in-memory dicts. Four indexed queries instead of 100k dot products.

### 2.5 Ternary Weights as Navigational Coordinates

ModelAtlas's bank positions are signed integers. The Balanced Sashimi decoder uses ternary {-1, 0, +1} weights. These are the same conceptual operation: discretizing a continuous space into signed directions relative to a meaningful zero.

We propose that the ternary decoder does not predict tactic tokens from a vocabulary. Instead, it predicts **navigational directions in proof space:**

- **STRUCTURE direction** {-1, 0, +1}: Should the next step simplify (toward decidable/arithmetic), maintain (rewrite), or complexify (introduce quantifiers)?
- **AUTOMATION direction** {-1, 0, +1}: Should the next step use full automation (omega, simp), mixed, or manual reasoning?
- **DECOMPOSITION direction** {-1, 0, +1}: Should the goal be decomposed into subgoals (via `have`, `cases`, `induction`), maintained as-is, or atomized?

These ternary coordinates, combined with predicted anchor logits, form a `StructuredQuery` that resolves to concrete tactics and premises through the semantic network:

```python
# The ternary decoder outputs:
#   structure_direction:  {-1, 0, +1}  (simplify / maintain / complexify)
#   automation_direction: {-1, 0, +1}  (auto / mixed / manual)
#   decompose_direction:  {-1, 0, +1}  (atomize / maintain / decompose)
#   anchor_logits:        ℝ^|anchors|  (which semantic labels apply)
#   progress:             ℝ^1          (estimated remaining steps)

# Resolution via navigate():
query = StructuredQuery(
    structure=structure_direction,
    automation=automation_direction,
    decomposition=decompose_direction,
    prefer_anchors=top_k(sigmoid(anchor_logits)),
    avoid_anchors=recently_failed_anchors,
)
tactic = navigate(tactic_network, query)[0]
premises = navigate(premise_network, query)[:top_k]
```

**Why ternary is natural here:** A ternary navigational coordinate says "go toward more automation / stay neutral / go toward less automation." This is a directional choice, not a precise numerical value. The fine-grained resolution (which *specific* tactic?) comes from the anchor dictionary, which is symbolic and deterministic. The model doesn't need to learn each tactic independently; it learns the *dimensions along which tactics vary*, and the structured network handles resolution.

**Why this is better than vocabulary classification:** A classification head treats `simp` and `linarith` as equally distant tokens (both are indices in a vocab). In navigational space, they're very close — both have AUTOMATION = -1 (fully automated) and share anchors like `arithmetic`, `automated-decision`. The model learns the abstract *kind* of tactic needed, not the specific token.

**External validation — activation steering (arXiv 2502.15507).** Recent work on activation steering for tactic prediction found that "the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately." Steering toward the right *reasoning category* improves tactic selection. This is direct empirical support for the navigational approach: predict the category (direction), then resolve the specific tactic through structured lookup. The steering vectors in that work correspond to what Wayfinder makes explicit as ternary bank directions.

**Precedent — TacticToe (HOL4).** TacticToe abstracts tactics to templates (`apply <thm>`, `rewrite <thm>`, `induction <var>`) — roughly 100 base tactics — then separately predicts arguments. The key finding: *argument prediction for abstracted tactics has the highest impact on success*. This validates the two-level decomposition in Wayfinder: ternary directions select the tactic template (coarse), anchor logits + proof network resolution select the arguments (fine). No prior system has developed TacticToe's abstraction into a full navigation paradigm.

**The many-to-one problem and expanded navigation**: The initial design navigates three banks (STRUCTURE, AUTOMATION, DECOMPOSITION), giving 3^3 = 27 possible direction vectors. Several semantically distinct tactics collapse to the same direction. For example, `intro` (STRUCTURE=0, AUTOMATION=+1, DECOMPOSE=0) and `apply` share identical coordinates despite being very different operations. `cases` and `induction` likewise both map to (+1, +1, +1).

This is partially by design — the anchor logits provide fine-grained disambiguation. But 27 coarse bins places excessive load on the anchor channel, especially when the anchor dictionary is still maturing.

**Resolution: expand the navigable bank set.** The proof network has 6 banks, but the v1 design only navigates 3. Adding DOMAIN, DEPTH, and CONTEXT as navigable dimensions expands the direction space to 3^6 = 729 possible vectors — a 27× increase in coarse resolution:

| Tactic | STRUCT | AUTO | DECOMP | DOMAIN | DEPTH | CONTEXT | Unique? |
|--------|--------|------|--------|--------|-------|---------|---------|
| `intro` | 0 | +1 | 0 | 0 | 0 | +1 | Yes |
| `apply` | 0 | +1 | 0 | 0 | 0 | 0 | Yes |
| `cases` | +1 | +1 | +1 | 0 | +1 | +1 | Yes |
| `induction` | +1 | +1 | +1 | 0 | +1 | 0 | Yes (vs cases) |
| `omega` | -1 | -1 | -1 | -1 | -1 | 0 | Yes |
| `simp` | -1 | -1 | 0 | 0 | -1 | 0 | Yes |

With 6 navigable banks, `intro` and `apply` are distinguished by CONTEXT (intro creates hypotheses → CONTEXT=+1; apply consumes them → CONTEXT=0). `cases` and `induction` are distinguished by CONTEXT (cases typically adds multiple hypotheses; induction does not).

**Implementation**: The ProofNavigator gains three additional ternary direction heads (6 total). The loss function gains three additional cross-entropy terms. The StructuredQuery uses all 6 banks for `navigate()`. The training data mapping (Section 5.1 of the Design doc) is extended with DOMAIN/DEPTH/CONTEXT direction labels.

**Fallback**: If 6-bank navigation proves too hard to learn (too many degrees of freedom for the available training signal), the system can gracefully degrade to 3-bank navigation by setting the additional bank weights to zero in the loss function. The architecture supports both configurations via the `navigable_banks` config parameter.

### 2.6 Spreading Activation as Proof Exploration

ModelAtlas's `spread()` function does Bellman-Ford propagation from seed nodes through two channels:

1. **Link channel (Layer 1):** Explicit relationships — `lemma A depends_on lemma B`, `tactic T commonly_precedes tactic S`.
2. **Anchor channel (Layer 2):** Shared semantic labels — if the current goal and a candidate lemma share rare anchors, activation spreads.

In proof search, spreading activation replaces the learned value function used in MCTS/HTPS:

```
proof_spread(current_goals, already_used_lemmas) → {lemma_id: relevance_score}
```

**Seeds:** Current open goals + previously successful lemmas.

**Link channel:** Mathlib's dependency graph provides explicit lemma-to-lemma relationships. If lemma A's proof uses lemma B, there is a directed link with weight proportional to how directly B is used. If tactic T commonly follows tactic S (learned from proof corpus), there is a tactic-to-tactic link.

**Anchor channel:** Find lemmas that share anchors with the current goals. Weight by fraction of shared anchors. IDF means rare shared anchors propagate stronger activation.

**Bank scoping:** If the current goal is about algebra (DOMAIN ≤ 0), only spread through algebra-bank anchors. This prevents semantic bleeding — topology lemmas don't activate during an algebra proof, even if they share generic anchors like `equality`.

**Advantages over learned value functions:**
1. **No neural inference at search time.** Spreading activation is priority-queue Bellman-Ford on SQLite. Milliseconds, not seconds.
2. **Auditable.** Every activation trace is inspectable: "this lemma activated because it shares anchors X, Y with the current goal and is one link from lemma Z which we already used."
3. **Dynamic.** As the proof progresses (goals close, new goals open, lemmas are used), the seed set changes and activation recomputes — no need to re-run a neural forward pass.

### 2.7 Connections to Mutation Theory

The Mutation Theory corpus (200+ Lean 4 theorems across 7 phases) provides formal guarantees that connect directly to navigational proof search. Of all application domains for machine learning, mathematical theorem proving is arguably the *most* naturally aligned with formal learning theory — the state space is well-defined, the verification oracle is exact, and the mathematical structure of the domain mirrors the mathematical structure of the learning theory. The connections below are not loose analogies; they are structural correspondences between formalized theorems and concrete system behaviors.

**Trajectory monotonicity (Phase 7, T7.0).** The `killed_trajectory_mono` theorem proves that killed sets grow monotonically along well-structured search trajectories. In ModelAtlas terms: as you add constraints (require/prefer/avoid anchors), the candidate set monotonically shrinks. In proof search terms: each successful tactic application monotonically reduces the survivor set (remaining open goals).

The Lyapunov monotonicity theorem (T7.5) guarantees convergence when the search process has a decreasing potential function. The progress prediction head estimates this potential — remaining proof steps serve as a Lyapunov function candidate. If the progress estimate decreases at each proof step, convergence is formally guaranteed. This is stronger than the heuristic role LeanProgress assigns to step prediction; it connects progress prediction to formal convergence through a chain: `progress_head → Lyapunov estimate → T7.5 → convergence guarantee`.

**Research needed**: Existing work on proof search termination and Lyapunov-style arguments in interactive theorem provers (e.g., well-founded recursion in Lean itself, termination metrics in Isabelle's Sledgehammer) may provide tighter bounds than the general Mutation Theory guarantees. The connection between proof complexity measures (e.g., proof tree depth, cut-rank) and potential functions is an active area that should inform the progress head's training targets.

**Phase transitions (Phase 7, T7.3).** Greedy kill trajectories exhibit a structural transition: initial rapid progress (easy tactics close many goals) followed by a harder regime (remaining goals require specific premises). The `phase_transition` theorem gives an explicit transition-step witness. In navigational terms: early search moves near the origin (common tactics like `intro`, `unfold`, basic rewrites); the transition point is where search must venture into specialized regions of the semantic network. The phase transition is precisely where navigational retrieval matters most — after the easy tactics, the model must navigate to specific, rare anchors.

**Fixed-point partition (Phase 4, T4).** The `fixedPointPartition` theorem proves that the mutant space partitions into three disjoint sets: Killed × Equivalent × DistinguishableSurvivors. The proof space partitions identically:

- **Killed** = subgoals closed by a tactic
- **Equivalent** = subgoals that can be restated without changing provability (via `rw`, `conv`, `simp`)
- **Distinguishable Survivors** = genuinely hard subgoals requiring new tactics

This partition maps directly to the three-signal scoring architecture — not as a metaphor, but as a structural correspondence:

| Mutation Theory Partition | Scoring Signal | Role |
|--------------------------|---------------|------|
| Killed vs. Survivor | `bank_alignment` | Are we in the right region of proof space? Bank misalignment → the lemma is in the wrong partition. |
| Equivalent detection | `anchor_relevance` | Do the semantic labels match? Equivalent subgoals share anchors with the original — IDF-weighted anchor overlap detects them. |
| Distinguishable → navigate to known | `seed_similarity` | Is this similar to something we've already solved? For genuine survivors, spreading activation from already-used lemmas navigates toward related, previously successful patterns. |

The three-signal multiplicative structure is not an arbitrary design choice; it mirrors the three-way partition that Mutation Theory proves is exhaustive and disjoint.

**Decomposition (Phase 3, T3).** The `separableIndependentTesting` theorem proves that when a system decomposes into independent components, each can be tested independently and the cost is additive (not multiplicative). This is the formal basis for the SubgoalDecomposer: when the ternary DECOMPOSITION coordinate is +1, the system predicts `have` subgoals, and each subgoal gets its own independent navigational query in the premise network. The cost is `Σ solve_cost(subgoal_i)`, not `Π`. The theorem guarantees this decomposition is optimal when subgoals are genuinely independent — which the system can verify by checking whether the subgoals share anchors (shared anchors → likely dependent → don't decompose).

**Teaching dimension (Phase 5, T5.17).** Teaching dimension equals specification complexity: `TD = κ`. The minimum set of premises needed to uniquely specify a proof is bounded by the specification complexity of the target theorem. This provides a formal lower bound on the `premise_candidates` parameter: if a theorem's specification complexity is 4, retrieving 3 premises is provably insufficient. The proof network's anchor set for a theorem approximates its specification complexity — a theorem with 12 unique anchors is more complex to specify than one with 3.

### 2.8 Why Theorem Proving is Uniquely Aligned with Learning Theory

The Mutation Theory connections above are stronger than typical "ML meets formal methods" claims, for a structural reason: **mathematical theorem proving is the domain where learning theory's assumptions are most naturally satisfied.**

1. **Exact verification oracle.** PAC learning, teaching dimension, and specification complexity all assume access to a membership oracle or equivalence oracle. The Lean kernel *is* that oracle — it provides binary, deterministic, zero-noise feedback on whether a tactic application is correct. No other ML application domain has this property. Image classification has label noise; NLP has subjective ground truth; even code generation has ambiguous specifications. Proof verification is exact.

2. **Well-defined state space.** The space of Lean goal states is a formal language with known grammar. The space of tactics is finite and enumerable. The dependency graph is a DAG. These are the conditions under which learning bounds are tight, not loose.

3. **Decomposability.** Mathematical proofs decompose into independent subgoals (formalized in T3). This is the condition under which additive (not multiplicative) sample complexity bounds hold. Most real-world domains don't decompose cleanly; theorem proving does, by construction.

4. **Monotonic progress.** Proof search has a natural Lyapunov function: the number of remaining open goals (or more precisely, the sum of their complexities). This is the condition under which trajectory convergence theorems (T7.0, T7.5) apply with their tightest bounds.

5. **Structured knowledge.** The mathematical knowledge needed for proof search (Mathlib's 100k lemmas) is organized into a formal hierarchy with types, namespaces, and explicit dependencies. This is the condition under which navigational approaches (structured retrieval) have their strongest theoretical advantage over unstructured approaches (dense retrieval).

The implication is practical: formal learning theory should not be treated as an aspirational connection but as an *engineering tool* for this domain. Teaching dimension bounds should inform retrieval parameters. Convergence theorems should inform search budgets. Decomposition theorems should inform subgoal strategies. Wayfinder is designed to close this loop.

### 2.9 Specification Complexity and the Society of Mind Architecture

NAV-001 and NAV-002 training runs revealed a fundamental limitation of monolithic proof search: the PAB stability regime remains "chaotic" (stability_mean > 0.30) even with full instrumentation and cosine LR scheduling. This is not a training failure — it is the expected behavior when a single model attempts to jointly optimize six bank dimensions with heterogeneous difficulty. Analysis through the lens of specification complexity theory reveals why, and prescribes the architectural response.

#### 2.9.1 Specification Complexity σ(P, μ)

Specification complexity (Vinaik, 2026; see `/Users/rohanvinaik/resume/Specification_Complexity_Paper/`) defines σ(P, μ) as the minimum number of tests required to achieve complete specification of a program P under mutation operator μ. It satisfies the Blum axioms (Theorem 2.5), admits exponential separation between programs with identical mutation counts (Theorem 2.6), and connects to five independent fields via identification theorems:

- **Teaching dimension** (Theorem 4.2): TD(P) = σ(P) — the minimum teaching set equals specification complexity
- **Query complexity** (Theorem 4.3): σ(P) = Q_exact(P) — the minimum queries for exact identification
- **Identity testing** (Theorem 4.4): σ(P) = IT(P) — the minimum tests for identity verification
- **Local testability** (Theorem 4.5): σ(P) ≤ LT(P) — specification complexity lower-bounds local test count
- **SpecP** (Theorem 4.6): σ(P) = SpecP(P) — specification complexity equals SpecP

These identifications mean that specification complexity is not merely an analogy for proof search — it is the exact mathematical quantity that determines the minimum information needed to uniquely identify a proof strategy.

#### 2.9.2 Composition Gap γ and the Formal Basis for Decomposition

**Theorem (Composition Gap, 3.15).** For composed programs A∘B:

```
σ(A∘B) ≤ σ(A) + σ(B) + γ(A,B)
```

where γ(A,B) counts "interface mutants" — mutations that are undetectable by testing A or B in isolation but are detectable when tested in composition. The interface mutant bound (Theorem 3.16) gives:

```
|γ(A,B)| ≤ |M_A^interface| × |M_B^interface|
```

**Critical insight:** γ vanishes for independent components. When A and B share no interface — when they can be tested (trained) independently — specification complexity is strictly additive: σ(A∘B) = σ(A) + σ(B). This is the formal justification for the Society of Mind architecture: decompose proof search into independent specialists until γ ≈ 0, making total training complexity additive rather than multiplicative.

**Application to NAV-001/002:** The monolithic 6-bank navigator has high γ because all banks share the bridge bottleneck. The bridge is the interface — mutations (weight perturbations) in one bank's representation interact with all other banks through the shared 128-dim bridge. PAB stability_mean = 0.34 is the empirical measurement of |γ|. Decomposing into per-bank (or per-bank-cluster) specialists eliminates the bridge as shared interface, driving γ → 0.

#### 2.9.3 Mutation Symmetry and Bank Difficulty Regimes

**Theorem (Symmetry Determines Regime, 4.1).** The mutation symmetry group G_μ determines the specification complexity regime:

- **|G_μ| large → Regime A**: polynomial σ, easy to specify
- **|G_μ| small → Regime B**: exponential σ, hard to specify

NAV-002 empirically confirms this partition across the six banks:

| Bank | Accuracy Range | |G_μ| | Regime | Interpretation |
|------|---------------|-------|--------|----------------|
| DOMAIN | 0.95+ | Large | A | Permuting theorems within a domain preserves label → many equivalent mutations → easy |
| CONTEXT | 0.73-0.88 | Large | A | Context patterns are locally similar → moderate symmetry |
| DECOMPOSITION | 0.63-0.88 | Medium | A/B | Some structural variation but recurring patterns |
| STRUCTURE | 0.43-0.66 | Small | B | Each proof tree is nearly unique → few equivalent mutations → hard |
| AUTOMATION | 0.25-0.60 | Small | B | Automation decisions are highly specific → minimal symmetry |
| DEPTH | 0.24-0.60 | Small | B | Depth is a continuous property poorly captured by ternary bins |

**The difficulty hierarchy is not noise — it is a structural consequence of mutation symmetry.** Easy banks (DOMAIN, CONTEXT) have massive symmetry groups: permuting theorems within the same domain doesn't change the domain label. Hard banks (STRUCTURE, AUTOMATION, DEPTH) have minimal symmetry: each proof tree is nearly unique, each automation decision is highly specific.

#### 2.9.4 Bulk-to-Tail Phase Transition

**Theorem (Bulk-to-Tail, 3.4).** There exists a critical step i* in any specification trajectory where the process transitions from correlated mutant elimination (bulk phase — cheap, high symmetry exploited) to individual mutant targeting (tail phase — expensive, each remaining mutant requires a dedicated test).

NAV-002 exhibits this transition empirically:
- **Phase A (steps 0-500)**: Bulk regime — easy banks saturate rapidly (DOMAIN reaches 0.95+), representation evolves quickly
- **Phase B (steps 500-1500)**: Transition — bridge representations begin to freeze
- **Phase C (steps 1500-5000)**: Tail regime — hard banks destabilize (DEPTH crashes 0.60→0.24), accuracy on easy banks degrades as the model contorts to improve hard banks

The Phase C accuracy crash on DEPTH is the tail phase beginning. The remaining low-symmetry errors (Regime B banks) resist elimination by the monolithic architecture because each requires a specialized representation that conflicts with the representations needed for other banks through the shared bridge.

#### 2.9.5 Narrative Construction: Regime B → Regime A Conversion

The deepest architectural insight from specification complexity theory is that the regime classification is not fixed — it depends on the mutation operator. A problem that is Regime B under one μ may be Regime A under a different μ that introduces symmetry.

**Story templates introduce symmetry.** Raw proof structure prediction is Regime B: each proof tree is nearly unique, giving |G_μ| ≈ 1. But *proof strategy templates* — recurring narrative patterns like "induction on the primary variable, with base case handled by simp and inductive step by rewriting" — have high symmetry: many proofs instantiate the same template, giving |G_μ| >> 1.

This is the computational content of Winston's Strong Story Hypothesis: stories are compression mechanisms that convert low-symmetry instances into high-symmetry classes. A proof's raw structure has exponential specification complexity; its *narrative type* (which story template it instantiates) has polynomial specification complexity.

**Cross-project evidence for narrative regime conversion:**

| Project | Low-Symmetry Input (Regime B) | Story Template (Regime A) | Mechanism |
|---------|------------------------------|--------------------------|-----------|
| **ARC** | Raw grid transformations | 16+ story templates (PERCEIVE→IDENTIFY→TRANSFORM→VERIFY) | Compass typed-zero goal encoding |
| **ARC-AGI-3** | Arbitrary puzzle solutions | RECOGNIZE→PROBE→PLAY methodology | 9 specialist networks with Kuramoto sync |
| **Ralph Loop** | Unconstrained LLM outputs | StoryFrame types (COMPARISON, CREATION, TRANSFORMATION, etc.) | 8-network SoM voting, Noether invariants as frozen specs |
| **Relational-AI** | Specialist disagreements | 6-lens narrative pipeline (dramatic, worldbuilding, structural, causal, inferential, stylistic) | 3-model ensemble, collision dynamics |
| **ShortcutForge** | Raw user macros | 6-phase deterministic linter, 31 templates | LALR(1) grammar + execution planner complexity tiers |
| **Wayfinder** | Raw proof structure | Proof strategy templates (induction-then-simp, decompose-and-conquer, etc.) | Narrative constructor (NEW) |

#### 2.9.6 Five Typed Temporal Slots

The Society of Mind architecture for proof search consists of five typed temporal slots, each operating at a different specification complexity:

```
Slot 1: PERCEPTION (Symbolic Representation)
  σ ≈ O(1). Deterministic encoding of goal state.
  Analogous to ShortcutForge's DSL parsing.
  Implementation: Goal encoder (already implemented, frozen).

Slot 2: RECOGNITION (Story Template Selection)
  σ ≈ O(log k) where k = number of templates.
  Winston-style story matching: classify the proof into a narrative template.
  Converts Regime B (raw structure) → Regime A (template class).
  ARC's step-zero hypothesis collapse. Ralph's StoryFrame selection.
  Implementation: Template classifier over proof features (NEW).

Slot 3: PLANNING (Narrative Construction)
  σ ≈ O(poly(n)). Story-writing model instantiates the selected template.
  Produces a proof sketch: sequence of abstract steps, key lemma targets.
  Relational-AI's 6-lens narrative pipeline.
  Implementation: Proof sketch generator, possibly story-writing LLM (NEW).

Slot 4: EXECUTION (Bank-Specific Navigation)
  σ varies by bank. Specialist models per bank cluster.
  PAB stability per specialist must reach "stable".
  This is the existing v1 Wayfinder pipeline, decomposed.
  Implementation: Bank-cluster specialists (v1 navigator → multiple specialists).

Slot 5: VERIFICATION (Constraint Satisfaction)
  σ ≈ O(1). Lean kernel is an exact verification oracle.
  Censor network learns negatives (what NOT to try).
  Implementation: Lane A/B/C verification (already implemented).
```

**Key property:** Each slot has bounded specification complexity. The total system σ is additive across slots (composition gap γ ≈ 0 because slots communicate through typed interfaces, not shared weights). The monolithic NAV-001/002 navigator tried to learn all five functions simultaneously through a shared bridge — producing high γ and chaotic PAB dynamics.

#### 2.9.7 PAB as Decomposition Optimization Signal

The connection between PAB stability and specification complexity provides a practical optimization loop:

1. Train a specialist model for a candidate scope (e.g., "STRUCTURE + AUTOMATION banks")
2. Measure PAB stability_regime
3. If "stable" → scope is correctly sized; σ is bounded for this specialist
4. If "chaotic" → scope too broad; γ between internal components is high; decompose further
5. Iterate until every specialist reaches "stable"

The free energy bound (Theorem 3.10 of the specification complexity paper) guarantees monotonic improvement: each decomposition step that reduces γ also reduces total specification complexity. There is no risk of over-decomposition making things worse, because σ is additive for independent components and sub-additive for dependent ones.

**Cross-project convergence.** All user projects independently converge on this pattern:

| Project | "Mutants" | "Tests" | "σ" | "γ" |
|---------|-----------|---------|-----|-----|
| Spec Complexity Paper | Program variants | Test cases | min tests for SC=1 | Interface mutants |
| Wayfinder PAB | Training checkpoints | Eval examples | Stability regime | Bridge coupling |
| ARC | Candidates | Constraints | Typed-zero count | Scale×Lens cross-talk |
| Ralph Loop | LLM outputs | Validation layers | Noether invariant count | Cross-modal binding cost |
| Relational-AI | Specialist outputs | Collision dynamics | Agreement threshold | Destructive interference |
| ShortcutForge | LLM DSL | Linter phases | Lint repair count | Phase interaction |

### 2.10 PAC Learning and Process-Aware Evaluation

Process-Aware Benchmarking (PAB), introduced by Pal (2025), evaluates *how* models learn, not just what they achieve. PAB defines quality functions over learning trajectories: stability (smoothness of loss), predictability (variance of loss deltas), generalization efficiency, and class-wise progression.

In Wayfinder, PAB is extended with two domain-specific metrics:

**Progress prediction accuracy.** The progress head estimates remaining proof steps. PAB tracks how this estimate improves during training — a model that learns accurate progress prediction early has a better internal model of proof structure than one that predicts well only on familiar theorem types.

**Navigational coherence.** PAB tracks whether the ternary decoder's navigational coordinates become consistent during training. A well-trained model should predict similar directions for similar proof states (measured by anchor overlap of the proof states). Inconsistent navigation — predicting "simplify" for one arithmetic goal and "complexify" for a near-identical one — indicates the model hasn't learned the structure of proof space.

**Ternary-PAB synergy (from Balanced Sashimi).** Ternary weights force discrete commitment. A weight that crystallizes to +1 is *provably* committed to that direction. PAB can track crystallization directly: the fraction of weights whose ternary sign is stable across checkpoints. As crystallization increases, the decoder's navigational output stabilizes, and PAB can measure whether stable navigation correlates with better proof search performance.

### 2.11 Orthogonal Ternary Projection, Constraint-Oriented Computation, and Energy-Based Refinement

*This section synthesizes three theoretical streams that ground Phase 7 (v3). §2.11a-b are mature theory informing v3A. §2.11c is speculative, informing v3B.*

#### 2.11a Orthogonal Ternary Projection (OTP) [v3A — mature]

**Source**: Vinaik (2025), formalized in the data geometry research program.

**Core claim**: The ternary alphabet {-1, 0, +1} is not a quantization approximation — it is a geometrically complete projection system. The three values have distinct informational roles:

- **+1 (support)**: Positive alignment along the observation axis. "This direction contributes."
- **-1 (contradiction)**: Negative alignment. "This direction opposes." The censor's explicit rejection.
- **0 (orthogonality)**: Not "unknown" or "absent" but *transparent* — the position is orthogonal to the observation axis. The **Informational Zero**: a third informational state that creates geometric structure making projection meaningful.

**Direct application to Wayfinder**: The 6-bank ternary decoder already IS an OTP projection. A proof step with bank target `[+1, 0, 0, -1, 0, 0]` operates in a 2-dimensional subspace (STRUCTURE and AUTOMATION), with the other four banks orthogonal/transparent. This is not a limitation — it is the mechanism. Most proof steps don't need all six navigational dimensions simultaneously.

**Minority Channel Advantage (MCA)**: When most of a ternary vector is zero, the few non-zero positions carry maximum information density. The rare signal is the valuable signal. This maps directly to IDF weighting — banks that rarely fire non-zero carry more information per activation. MCA predicts that STRUCTURE and AUTOMATION (sparse activators) should carry disproportionate retrieval value vs DOMAIN (frequent activator).

**Zero-Sparsity as Dimensionality**: The count of zeros in a ternary target is a direct measure of the proof step's OTP dimensionality. Steps with more zeros are "simpler" — fewer simultaneous navigational decisions. This grounds the curriculum training strategy: train on low-dimensionality (high-zero) examples first, progressively reveal higher-dimensional steps.

#### 2.11b Constraint-Oriented Emergent Computation (COEC) [v3A — mature]

**Source**: Vinaik (2025), formalized as a 7-tuple framework.

**Core claim**: Specify what a system CANNOT do; behavior emerges from the interaction of constraints. Rather than designing positive behavior, define a constraint system and let valid behavior emerge as the feasible region.

**Application to Wayfinder**: Wayfinder's bank scores, critic, and censor already form a constraint system:
- **Bank alignment constraints**: Each bank score constrains the tactic to align with a navigational direction.
- **Critic constraint**: The distance-to-completion estimate constrains which goal states are worth pursuing.
- **Censor constraint**: The failure predictor (OTP -1 channel) constrains which tactics to avoid.
- **Anchor matching constraint**: IDF-weighted anchor overlap constrains premises to pattern-relevant candidates.

The v3A Censor is the explicit realization of COEC — it specifies what the system CANNOT do (invalid tactics), and valid proof behavior emerges from the remaining feasible region. The asymmetric loss (MCA-motivated: missed suppressions penalized 2× vs false suppressions) reflects that constraint violations carry more information than satisfactions.

#### 2.11c Energy-Based Refinement (EBM) [v3B — speculative, gated on v3A]

**Source**: Logical Intelligence/Kona (2026), energy-based models for constraint satisfaction.

**Core claim**: Define a scalar energy function over entire solutions, then minimize via gradient-based refinement in continuous latent space. Avoids the autoregressive failure mode where sequential commitment prevents revision of earlier decisions.

**Empirical validation**: Kona achieves 96.2% on Sudoku at 313ms vs 2% for frontier LLMs — because it evaluates all constraints holistically rather than committing token-by-token. The 50× performance gap demonstrates the cost of sequential commitment in constraint-rich domains.

**Application to Wayfinder (speculative)**: The four constraint channels (bank + critic + censor + anchor) can be composed into a differentiable energy function: `E(sketch) = α·E_bank + β·E_critic + γ·E_censor + δ·E_anchor`. Continuous ternary relaxation via Gumbel-softmax bridges the discrete OTP decoder to gradient-based optimization. The temperature annealing schedule (τ: 1.0→0.1) IS OTP's Progressive Revelation in the energy landscape — high temperature favors Informational Zeros; low temperature hardens non-zero activations.

**Learning-theoretic framing**: Navigation (Slots 1-4) is PAC learning. Verification (Slot 5 + Lean kernel) is an exact oracle. Negative learning exploits the oracle to construct version-space boundaries from both sides. The energy function (v3B) unifies all constraint channels into a single differentiable objective, enabling whole-sketch revision rather than step-by-step commitment.

**Maturity gate**: Energy refinement requires a trained censor (v3A 7.3) to provide E_censor. It ships only after v3A demonstrates value on real proof outcomes. If discrete v3A scoring matches or exceeds energy-refined scoring, the energy function remains a theoretical tool for analysis, not a shipping component.

**Convergence with external programs**: AMI Labs (LeCun, 2026) validates structured representations + constrained prediction + planning over autoregressive generation. Logical Intelligence validates energy-based constraint satisfaction over sequential search. Both confirm the same architectural bet Wayfinder makes — but Wayfinder integrates these ideas in a domain (theorem proving) with an exact verification oracle, making empirical validation possible.

---

## 3. Architecture Specification

### 3.1 System Overview

**Wayfinder v2 (Society of Mind)** decomposes the monolithic v1 pipeline into five typed temporal slots, each with bounded specification complexity. The v1 architecture becomes the EXECUTION slot.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      WAYFINDER v2 (Society of Mind)                   │
│                                                                      │
│  [Lean Goal State Text]                                              │
│          │                                                           │
│          ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  SLOT 1: PERCEPTION          σ ≈ O(1)                    │        │
│  │  Goal Encoder (frozen) + Domain Gate                     │        │
│  │  Output: h_enc ∈ ℝ^384, in_domain: bool                │        │
│  └──────────────────────────┬──────────────────────────────┘        │
│                              │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐        │
│  │  SLOT 2: RECOGNITION        σ ≈ O(log k)                │        │
│  │  Story Template Classifier                               │        │
│  │  Classifies proof into k narrative templates             │        │
│  │  (induction-then-simp, decompose-and-conquer, etc.)     │        │
│  │  Converts Regime B (raw structure) → Regime A (template) │        │
│  │  Output: template_id, template_confidence                │        │
│  └──────────────────────────┬──────────────────────────────┘        │
│                              │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐        │
│  │  SLOT 3: PLANNING           σ ≈ O(poly(n))              │        │
│  │  Narrative Constructor (Proof Sketch Generator)          │        │
│  │  Instantiates template with proof-specific details       │        │
│  │  Output: proof_sketch (ordered subgoal sequence),        │        │
│  │          key_lemma_targets, estimated_depth               │        │
│  └──────────────────────────┬──────────────────────────────┘        │
│                              │ (per subgoal in sketch)               │
│  ┌──────────────────────────▼──────────────────────────────┐        │
│  │  SLOT 4: EXECUTION          σ varies by bank cluster     │        │
│  │  Bank-Cluster Specialists (decomposed v1 navigator)      │        │
│  │  ┌─────────────────────┐  ┌───────────────────────┐     │        │
│  │  │ Specialist A        │  │ Specialist B           │     │        │
│  │  │ (DOMAIN, CONTEXT)   │  │ (STRUCTURE, AUTOMATION │     │        │
│  │  │ PAB: stable         │  │  DEPTH, DECOMPOSITION) │     │        │
│  │  │ Regime A banks      │  │ PAB: → stable          │     │        │
│  │  └─────────┬───────────┘  └──────────┬────────────┘     │        │
│  │            └──────────┬──────────────┘                   │        │
│  │                       ▼                                  │        │
│  │  Structured Resolution (navigate + spread, deterministic)│        │
│  │  Output: ranked (tactic, premise_list) pairs             │        │
│  └──────────────────────────┬──────────────────────────────┘        │
│                              │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐        │
│  │  SLOT 5: VERIFICATION      σ ≈ O(1)                     │        │
│  │  Lane A: Pantograph (local, step-wise)                   │        │
│  │  Lane B: Axle API (cloud, verify/repair/decompose)       │        │
│  │  Lane C: lean4checker (high-assurance, leaderboard)      │        │
│  │  Censor: learns negatives (what NOT to try next)         │        │
│  └──────────────────────────┬──────────────────────────────┘        │
│                              │                                       │
│                      [Arbiter / Orchestrator]                        │
│                  Selects next goal, routes to specialist,            │
│                  updates proof state, terminates search              │
└──────────────────────────────────────────────────────────────────────┘
```

**v1 architecture (EXECUTION slot detail):**

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WAYFINDER v1                                  │
│                                                                      │
│  [Lean Goal State Text]                                              │
│          │                                                           │
│          ▼                                                           │
│  ┌─────────────────────────────────────────────┐                     │
│  │  GOAL ENCODER (math-native, see §3.2)       │                    │
│  │  Model: TBD (Qwen3.5/DeepSeek-Math/pruned)  │                    │
│  │  Output: h_enc ∈ ℝ^embed_dim                │                    │
│  └─────────────┬───────────────────────────────┘                     │
│                │                                                      │
│       ┌────────┴────────┐                                            │
│       │                 │                                             │
│  ┌────▼─────┐   ┌──────▼───────────────────────────┐                │
│  │ DOMAIN   │   │  GOAL ANALYZER                    │                │
│  │ GATE     │   │  (continuous, learnable)           │                │
│  │          │   │  Output: features ∈ ℝ^256         │                │
│  │ Binary:  │   │  + bank positions (6 banks)       │                │
│  │ in/out   │   │  + anchor logits                   │                │
│  └──────────┘   └──────────┬───────────────────────┘                │
│                            │                                          │
│  ┌─────────────────────────▼───────────────────────────────┐         │
│  │  SUBGOAL DECOMPOSER                                      │        │
│  │  Ternary decision: decompose? {-1, 0, +1}               │        │
│  │  If +1: predict 'have' subgoals, recurse                │        │
│  │  If 0/-1: proceed with current goal                      │        │
│  └─────────────────────────┬───────────────────────────────┘         │
│                            │ (per subgoal)                            │
│  ┌─────────────────────────▼───────────────────────────────┐         │
│  │  INFORMATION BRIDGE                                       │        │
│  │  Continuous → discrete transition                         │        │
│  │  Compression to bridge_dim (128)                          │        │
│  │  LayerNorm + projection                                   │        │
│  └─────────────────────────┬───────────────────────────────┘         │
│                            │                                          │
│  ┌─────────────────────────▼───────────────────────────────┐         │
│  │  PROOF NAVIGATOR (ternary decoder, 6-bank)                 │        │
│  │  Weights ∈ {-1, 0, +1}, trained via STE                  │        │
│  │                                                           │        │
│  │  Output heads (3^6 = 729 coarse direction bins):          │        │
│  │    • structure_direction    ∈ {-1, 0, +1}                │        │
│  │    • domain_direction       ∈ {-1, 0, +1}                │        │
│  │    • depth_direction        ∈ {-1, 0, +1}                │        │
│  │    • automation_direction   ∈ {-1, 0, +1}                │        │
│  │    • context_direction      ∈ {-1, 0, +1}                │        │
│  │    • decompose_direction    ∈ {-1, 0, +1}                │        │
│  │    • anchor_logits          ∈ ℝ^|anchors|   (sigmoid)    │        │
│  │    • progress               ∈ ℝ^1           (remaining)  │        │
│  │    • critic                 ∈ ℝ^1           (solvable?)  │        │
│  │                                                           │        │
│  │  Loss: L_nav + λ₁·L_anchor + λ₂·L_progress + λ₃·L_critic│        │
│  │  (adaptive λ via learned log-σ uncertainty)               │        │
│  └─────────────────────────┬───────────────────────────────┘         │
│                            │                                          │
│  ┌─────────────────────────▼───────────────────────────────┐         │
│  │  STRUCTURED RESOLUTION (deterministic, no NN)             │        │
│  │                                                           │        │
│  │  navigate(tactic_network, directions + anchors) → tactic  │        │
│  │  navigate(premise_network, directions + anchors) → lemmas │        │
│  │  spread(proof_network, current_goals) → search priority   │        │
│  └─────────────────────────┬───────────────────────────────┘         │
│                            │                                          │
│  ┌─────────────────────────▼───────────────────────────────┐         │
│  │  DETERMINISTIC LOWERING                                   │        │
│  │  Tactic + premises → Lean syntax string                   │        │
│  └─────────────────────────┬───────────────────────────────┘         │
│                            │                                          │
│                            ▼                                          │
│  [Lean 4 Kernel: verify tactic, return new goals or error]           │
│          │                                                           │
│          ▼                                                           │
│  [Update proof network: new goals → new seeds for spread()]          │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Goal Encoder

**Status**: Encoder selection is an active design decision. The original prototype used `all-MiniLM-L6-v2`, but the encoder is the system's perceptual bottleneck and deserves more aggressive investment.

**The problem with small general-purpose encoders**: Lean goal states are formal syntax with type annotations, unicode math symbols, and dependent type structure — not natural language. A sentence-transformer trained on English text will produce mediocre embeddings for `⊢ ∀ (x : α) [inst : TopologicalSpace α], IsOpen (⋃ i, s i)`. If two goal states are semantically different in math but look similar as strings, the encoder conflates them, and no amount of downstream projection can recover the lost information.

**Encoder strategy — four tiers under evaluation**:

1. **Byte-level models (strong candidate)**. ByT5-small (299M) is the de facto standard for Lean, used by ReProver, LeanAgent, and Lean Copilot. Its byte-level operation handles Lean's Unicode symbols (∀, ⊢, →, ℕ, ⋃) without tokenization artifacts — a critical advantage over subword tokenizers that fragment mathematical notation. ByT5-small produces 1472-dimensional embeddings via average pooling. Training requires only 120 GPU hours. A Lean-specific retrained tokenizer variant (arXiv 2501.13959) achieves 30.74% vs ReProver's 28.28% on MiniF2F. However, ByT5 has no mathematical pretraining — it is a generic byte-level model.

2. **Math-native models (preferred if byte-level gap is addressed)**. Recent releases like Qwen 3.5, DeepSeek-Math, and InternLM-Math are trained on mathematical corpora including formal languages. These models understand Lean-like syntax natively. A math-native encoder (even at 1-7B parameters) will produce dramatically better embeddings for goal states than any general-purpose sentence transformer. The risk: most math-native models use subword tokenizers that may fragment Lean syntax. Evaluate both byte-level and subword-based math models.

3. **Ternary/BitNet encoder (architecturally distinctive)**. BitNet b1.58 (ternary weights, 1.58 bits/parameter) at 2B matches full-precision models on mathematical reasoning benchmarks with 2.7× speedup and 3.5× memory reduction — but no one has applied ternary architectures to formal theorem proving. Since Wayfinder already uses ternary weights in the decoder (TernaryLinear with STE), extending ternary quantization to the encoder would be architecturally consistent and could enable real-time premise retrieval. A ternary encoder is the most distinctive option and would make the full pipeline ternary-native.

4. **Aggressively pruned large models**. Recent work on structured pruning demonstrates that ~95-99% of nodes can be removed from large generalist models while retaining domain-specific performance, especially when followed by brief fine-tuning on in-domain data. Quantization to 4-bit introduces up to 32% accuracy degradation on math benchmarks, but fine-tuning on just 545 task-specific examples for 3 minutes can recover full performance. A 4-bit quantized 14B still outperforms its dense 7B counterpart. Techniques: structured pruning (SparseGPT, Wanda), layer dropping, distillation, and post-pruning fine-tuning on Mathlib goal states.

**Key design constraint**: The encoder may be frozen *or* fine-tuned, but the downstream architecture (GoalAnalyzer, Bridge, Navigator) must work with either. The encoder's output dimensionality is a configuration parameter (`embed_dim` in `configs/base.yaml`), not hardcoded.

**Evaluation criterion**: Encoder quality is measured by navigational accuracy at Phase 1 (smoke test). If navigational accuracy on 1-2 step proofs plateaus below 60% after 500 training steps, the encoder is the prime suspect and should be upgraded before debugging other components.

### 3.3 Goal Analyzer

**Parameters**: ~100K (learnable)
**Input**: 384-dim embedding from encoder
**Output**: 256-dim feature vector + bank position estimates + anchor logits

The GoalAnalyzer serves three functions:

1. **Feature projection**: Linear projection from 384 to 256 dimensions. Extracts proof-relevant features from the general-purpose sentence embedding.
2. **Bank position estimation**: Six classification heads, one per proof bank. Each predicts (sign, depth) for the goal state. Used to position the current goal in the semantic network.
3. **Anchor extraction**: Multi-label classification over the anchor dictionary. Predicts which semantic labels apply to the current goal state.

Bank positions and anchors enable the navigational queries that follow. The analyzer bridges the continuous embedding space and the structured semantic network.

### 3.4 Subgoal Decomposer

**Parameters**: ~50K (learnable)
**Input**: 256-dim features from GoalAnalyzer
**Output**: Ternary decomposition decision + subgoal projections

Inspired by DeepSeek-Prover-V2's proof sketching, the SubgoalDecomposer decides whether to decompose the current goal into subgoals via `have` statements:

- **-1 (atomize)**: The goal is already atomic. Proceed directly to tactic prediction.
- **0 (neutral)**: No decomposition signal. Proceed with current goal.
- **+1 (decompose)**: The goal should be broken into subgoals. Predict subgoal count and project features for each subgoal.

When decomposing, each subgoal gets its own feature vector (via learned projection heads) and is processed independently through the rest of the pipeline. This implements Mutation Theory Phase 3's `separableIndependentTesting`: independent subgoals can be solved independently.

### 3.5 Information Bridge

**Parameters**: ~33K (learnable)
**Input**: 256-dim features
**Output**: 128-dim compressed representation

The bottleneck between continuous and discrete space. LayerNorm + linear projection compresses the feature representation to a dimensionality suitable for the ternary decoder.

**Information-theoretic role**: The bridge tests the hypothesis that the mutual information between goal understanding and tactic selection is low-dimensional. If a 128-dim bridge suffices for accurate navigation, the goal→tactic mapping has low intrinsic dimensionality.

### 3.6 Proof Navigator (Ternary Decoder)

**Parameters**: ~200K (learnable, ternary hidden layers)
**Input**: 128-dim bridge output
**Output**: 6 navigational directions + anchor logits + progress + critic

The core innovation. Unlike a traditional decoder that classifies into a tactic vocabulary, the Proof Navigator outputs *directions in proof space*. The expanded 6-bank design (see Section 2.5) gives 3^6 = 729 coarse direction bins, resolving the many-to-one problem where semantically distinct tactics collapsed to the same coordinates.

**Ternary direction heads** (hidden layers use TernaryLinear with STE, one per navigable bank):
- `structure_direction ∈ {-1, 0, +1}`: Simplify / maintain / complexify
- `domain_direction ∈ {-1, 0, +1}`: Concrete / general / abstract
- `depth_direction ∈ {-1, 0, +1}`: Shallow / moderate / deep
- `automation_direction ∈ {-1, 0, +1}`: Automate / mixed / manual
- `context_direction ∈ {-1, 0, +1}`: Reduce context / maintain / enrich context
- `decompose_direction ∈ {-1, 0, +1}`: Atomize / maintain / decompose

**Continuous heads** (standard linear, on top of ternary hidden layers):
- `anchor_logits ∈ ℝ^|anchors|`: Which semantic labels should the next tactic/premises have
- `progress ∈ ℝ^1`: Estimated remaining proof steps (MSE loss against ground truth). Connects to Mutation Theory's Lyapunov convergence guarantee (T7.5) — when progress decreases monotonically, convergence is formally guaranteed. LeanProgress showed that including proof history boosts accuracy from 61.8% to 75.1%. We incorporate this by concatenating embeddings of previously closed goals into the bridge input (see Section 3.10 for details).
- `critic ∈ ℝ^1`: Estimated **distance to proof completion** (MSE loss), NOT binary solvability. HTPS found that hard binary critic targets (solvable/not-solvable) are *worse than no critic at all* — even worse than random. Soft targets dramatically outperform: 78.1% vs 63.1% on their benchmark. AlphaProof similarly trains its value function to predict remaining tactic count rather than binary provability, because all proven states have probability 1.0, destroying the training signal. Our critic head predicts a continuous distance metric (normalized remaining steps / budget consumed), trained with MSE loss.

**Training loss** (UW-SO adaptive weighting via learned log-σ):

```
L_total = (1/2σ₁²)·L_nav + (1/2σ₂²)·L_anchor + (1/2σ₃²)·L_progress
        + (1/2σ₄²)·L_critic + log(σ₁σ₂σ₃σ₄)
```

Where:
- `L_nav`: Cross-entropy on navigational directions (ternary classification, 6 banks, summed)
- `L_anchor`: Binary cross-entropy on anchor predictions
- `L_progress`: MSE on remaining-step prediction
- `L_critic`: MSE on normalized distance-to-completion (soft target, NOT binary BCE — per HTPS finding)

The `navigable_banks` configuration parameter controls which banks are active. Setting `navigable_banks: [STRUCTURE, AUTOMATION, DECOMPOSITION]` reverts to 3-bank navigation (27 bins) if the full 6-bank version proves too hard to learn.

### 3.7 Proof Network (Semantic Database)

**Storage**: SQLite database (single file, ~100MB for full Mathlib)
**Entities**: Lemmas, tactics, proof patterns
**Schema** (adapted from ModelAtlas):

```sql
-- Mathematical entities (lemmas, definitions, instances)
CREATE TABLE entities (
    entity_id   TEXT PRIMARY KEY,
    kind        TEXT,  -- 'lemma', 'def', 'instance', 'tactic'
    namespace   TEXT,
    display     TEXT
);

-- Bank positions (6 proof banks)
CREATE TABLE entity_positions (
    entity_id   TEXT REFERENCES entities(entity_id),
    bank        TEXT,
    sign        INTEGER,
    depth       INTEGER,
    PRIMARY KEY (entity_id, bank)
);

-- Anchor dictionary
CREATE TABLE anchors (
    anchor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT UNIQUE,
    bank        TEXT,
    source      TEXT DEFAULT 'deterministic'
);

-- Entity ↔ anchor links
CREATE TABLE entity_anchors (
    entity_id   TEXT REFERENCES entities(entity_id),
    anchor_id   INTEGER REFERENCES anchors(anchor_id),
    confidence  REAL DEFAULT 1.0,
    PRIMARY KEY (entity_id, anchor_id)
);

-- Explicit links (dependency, co-occurrence)
CREATE TABLE entity_links (
    source_id   TEXT REFERENCES entities(entity_id),
    target_id   TEXT REFERENCES entities(entity_id),
    relation    TEXT,  -- 'depends_on', 'commonly_precedes', 'same_namespace'
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation)
);

-- IDF cache
CREATE TABLE anchor_idf (
    anchor_id   INTEGER PRIMARY KEY,
    idf_value   REAL
);
```

**Population**: Deterministic extraction from Mathlib's Lean source:
1. Parse all theorem/lemma declarations → entities table
2. Extract types and namespaces → bank positions
3. Analyze proof text for tactic usage → anchors
4. Walk import graph → dependency links
5. Compute IDF from anchor frequencies

### 3.8 Structured Resolution

The resolution layer converts the Proof Navigator's output into concrete tactics and premises. **No neural inference** — pure symbolic operations:

```python
def resolve(nav_output, proof_network, current_context):
    """Convert navigational coordinates to tactic + premises."""

    # Build structured query from ternary directions + anchor logits
    query = StructuredQuery(
        structure=nav_output.structure_direction,
        automation=nav_output.automation_direction,
        decomposition=nav_output.decompose_direction,
        prefer_anchors=top_k_anchors(nav_output.anchor_logits, k=8),
        avoid_anchors=current_context.recently_failed,
    )

    # Navigate tactic space
    tactic_candidates = navigate(proof_network.tactic_table, query, limit=5)

    # Navigate premise space
    premise_candidates = navigate(proof_network.lemma_table, query, limit=16)

    # Spreading activation from current proof context
    context_activation = spread(
        proof_network,
        seeds=current_context.used_lemmas + current_context.open_goals,
        banks=["DOMAIN", "STRUCTURE"],
        max_depth=3,
    )

    # Combine: navigate score × spreading activation
    for p in premise_candidates:
        p.score *= context_activation.get(p.entity_id, 0.1)

    return tactic_candidates, sorted(premise_candidates, key=lambda p: p.score, reverse=True)
```

### 3.9 Deterministic Lowering

Template-based conversion from (tactic, premises) to Lean syntax:

```python
def lower_step(tactic, premises, context):
    if tactic.name in NULLARY_TACTICS:  # omega, ring, linarith, ...
        return tactic.name
    if tactic.name in PREMISE_TACTICS:  # apply, exact, rw, ...
        return f"{tactic.name} {' '.join(p.name for p in premises)}"
    if tactic.name == "have":
        return f"have {premises[0].name} := by"
    # ... etc
```

This is the existing `lowering.py` module, adapted to accept navigational resolution output instead of tier tokens.

### 3.10 Lean Kernel Interaction

The outer proof search loop interacts with the Lean kernel via Pantograph (LeanDojo's interaction protocol):

```python
while open_goals:
    goal = select_goal(open_goals, critic_scores, progress_estimates)

    # Neural forward pass (ONCE per goal)
    embedding = encoder.encode(goal.state)
    features = analyzer(embedding)
    nav_output = navigator(bridge(features))

    # Symbolic resolution (fast, no NN)
    tactics, premises = resolve(nav_output, proof_network, context)

    # Try candidates (interaction with Lean kernel)
    for tactic, premise_set in product(tactics[:3], premise_combinations(premises)):
        lean_text = lower_step(tactic, premise_set, context)
        result = lean_kernel.try_tactic(goal.state, lean_text)

        if result.success:
            open_goals.remove(goal)
            open_goals.extend(result.new_goals)
            context.record_success(tactic, premise_set)
            break
        else:
            context.record_failure(tactic, premise_set)
```

**Goal selection** uses the critic head (estimated distance-to-completion) and progress head (estimated remaining steps). Prioritize goals with lowest combined distance estimate.

**Proof history input.** LeanProgress demonstrated a 13-point accuracy improvement (61.8% → 75.1%) from including proof history in progress prediction. The spreading activation mechanism captures this symbolically (seeds from already-used lemmas), but the neural side also benefits. At each goal, the encoder receives not just the current goal state but also a summary of proof context: embeddings of previously closed goals are mean-pooled and concatenated to the bridge input. This gives the progress and critic heads access to "where we've been," not just "where we are."

---

## 4. Training

### 4.1 Data Preparation

**Training corpus**: LeanDojo's extracted Mathlib dataset (~98,734 theorems with proof traces). Each example provides:
- Theorem statement and goal state
- Tactic proof with intermediate goal states
- Premise information (which lemmas are referenced)

**Conversion to navigational training data**: For each proof step, extract:
- Ground-truth navigational directions (from the tactic used, mapped to bank coordinates)
- Ground-truth anchors (from the goal state and tactic/premise analysis)
- Ground-truth progress (number of remaining proof steps)
- Ground-truth solvability (1.0 for all goals in successful proofs)

**Negative examples**: Failed proof attempts, wrong tactic selections, incorrect premises — sourced from proof search logs and synthetic perturbation.

### 4.2 Curriculum Learning

Inspired by LeanAgent's complexity ordering (`complexity = e^S` where S = proof steps) and the navigational wavefront concept:

1. **Phase A (warmup)**: Train on 1-2 step proofs only. The model learns basic navigation near the origin.
2. **Phase B (expansion)**: Gradually increase proof depth. 3-5 step proofs.
3. **Phase C (full)**: All proof lengths, with oversampling of medium-difficulty theorems.

This follows the Mutation Theory's phase-transition insight: start with easy kills (short proofs near the origin), then venture into harder regions.

**Navigational wavefront curriculum (Phase C refinement).** During expert iteration (Phase 5), the curriculum should be navigational, not just complexity-based: order theorems by their *distance in the proof network from already-proven results*, creating a wavefront of learnability that expands outward through mathematical space. Theorems whose premises are 1-2 hops from already-proven theorems are trained first; theorems requiring navigation into unexplored regions of the semantic network are deferred. This addresses the saturation problem observed across systems — up to 98.5% of generated proofs during expert iteration are incorrect, and remaining unproven theorems become exponentially harder. STP (Self-play Theorem Prover) found that 47% of self-generated conjectures at the "cusp of provability" are successfully proved, vs only 11.4% of real unproved statements. The wavefront curriculum targets this cusp by definition.

### 4.3 PAB Integration

PAB metrics are collected at every checkpoint (every 50 training steps):
- Loss components (L_nav, L_anchor, L_progress, L_critic)
- Ternary crystallization rate
- Per-bank navigational accuracy
- Progress prediction correlation with actual remaining steps
- Critic calibration (predicted solvability vs. actual proof success)
- Domain-wise accuracy (by Mathlib namespace)

---

## 5. Evaluation

### 5.1 Endpoint Metrics

| Metric | Benchmark | Comparison |
|--------|-----------|------------|
| Theorem proved rate | MiniF2F-test (488 problems) | ReProver, DeepSeek-Prover-V2, HTPS |
| Theorem proved rate | Mathlib test split (~10k) | ReProver |
| Premise retrieval recall@k | Mathlib (ground-truth premises) | ReProver's retriever |
| Proof search nodes expanded | MiniF2F | All baselines |
| Wall-clock time per proof | MiniF2F | All baselines |

### 5.2 Trajectory Metrics (PAB)

| Metric | What it measures |
|--------|-----------------|
| Navigational consistency | Do similar goals produce similar directions? |
| Crystallization curve | How quickly do ternary weights commit? |
| Progress prediction accuracy | Does the model learn to estimate remaining steps? |
| Domain progression order | Which Mathlib namespaces are learned first? |
| Bank-wise accuracy | Which navigational dimensions are learned first? |

### 5.3 Ablation Studies

| Variant | What it tests |
|---------|--------------|
| Dense retrieval (no proof network) | Is navigational retrieval better than embedding similarity? |
| Tactic classification (no navigation) | Are directional coordinates better than vocabulary indices? |
| No spreading activation | Does spreading activation improve search? |
| No progress head | Does progress prediction improve goal selection? |
| Continuous decoder (no ternary) | Are ternary weights beneficial for navigation? |
| No IDF weighting | Does IDF improve premise retrieval? |

---

## 6. Related Work

A comprehensive survey of the external landscape is in `docs/Wayfinder_External_Research.md`. This section summarizes the most architecturally relevant systems and their implications for Wayfinder.

### 6.1 ReProver / LeanDojo (Yang et al., NeurIPS 2023)

Retrieval-augmented tactic generation for Lean 4. ByT5-small (299M) dual-encoder retriever with 1472-dim embeddings, MSE-based contrastive loss with 3 negatives per example. Restricts retrieval to accessible premises only (~33k of ~130k), giving ~2% recall improvement for free. On premises never seen during training, recall@10 drops from 38.4% to 27.6% — a 30% degradation revealing the fundamental limitation of dense retrieval for unseen entities. **Key difference from Wayfinder**: Dense retrieval (learned embeddings) vs structured navigation (bank positions + IDF-weighted anchors). ReProver requires neural forward passes for every retrieval; Wayfinder resolves via SQL queries. **Design influence**: Accessible-premises filtering adopted directly; ByT5-small as encoder candidate.

### 6.2 AlphaProof (DeepMind, 2024)

~3B encoder-decoder transformer with AND/OR tree search and AlphaZero-style MCTS. Critically, the value function predicts *remaining tactics to completion*, not binary provability — because all proven states have probability 1.0, destroying the training signal. Test-time RL (TTRL): for each hard problem, generates millions of variants and trains a specialist agent, requiring hundreds of TPU-days per IMO problem. Backpropagates the *value of the hardest branch* rather than the product. **Design influence**: Soft critic targets (distance, not binary) adopted directly. Encoder-decoder pattern with dual policy/value output is the efficiency model for search-heavy systems.

### 6.3 HTPS (Lample et al., NeurIPS 2022)

AlphaZero-inspired proof search with shared 600M encoder-decoder as policy and critic. Critic restricts decoder to two tokens (PROVABLE/UNPROVABLE). Critical finding: **soft critic targets dramatically outperform hard targets** — 78.1% vs 63.1% on Equations benchmark. Hard critic is worse than *no critic at all*. Value backpropagation through AND nodes uses product of children's values. Online training (continuous refresh every 5 minutes) outperforms batch expert iteration. **Design influence**: Soft critic targets adopted; our spreading activation replaces MCTS for search prioritization.

### 6.4 BFS-Prover (ByteDance, 2025)

Achieved 72.95% on MiniF2F-test with plain best-first search plus expert iteration, DPO from Lean compiler feedback, and length normalization. At fixed inference budget, outperforms InternLM2.5-StepProver's value-guided search (65.9%) and DeepSeek-Prover-V1.5's MCTS (63.5%). **Implication for Wayfinder**: With a strong enough policy, sophisticated search may be unnecessary. If Wayfinder's navigational directions are accurate, simple best-first search with spreading activation may suffice — no MCTS needed.

### 6.5 LeanProgress (2025)

Fine-tuned DeepSeek Coder 1.3B to predict remaining proof steps. Including proof history boosts accuracy from 61.8% to 75.1%. Combined score `C(sᵢ) = α·N(sᵢ) + (1-α)·P(sᵢ)` yields 3.8% improvement on Mathlib4. **Design influence**: Progress head inspired by this work; proof history input adopted (mean-pooled embeddings of previously closed goals concatenated to bridge input).

### 6.6 DeepSeek-Prover-V2 (Ren et al., 2025)

Two-model architecture: DeepSeek-V3 (671B MoE, ~21B activated) for `have`-chain decomposition, 7B prover for individual subgoals. GRPO with binary Lean compiler rewards. 88.9% on MiniF2F-test. **Design influence**: SubgoalDecomposer inspired by this, but as ternary decision within a small model. The division of labor (reasoning about proof structure vs executing individual steps) maps to our separation: neural network reasons about direction, proof network handles resolution.

### 6.7 Aristotle (Harmonic, 2025)

200B+ transformer with Monte Carlo Graph Search (MCTS extended to hypergraphs with state equivalence classes). Lemma-based informal reasoning: natural-language proof sketches → formal lemmas → iterative refinement. Test-time training on search traces. Disproof augmentation: each goal gets a negation transition to prune impossible branches. **Implication for Wayfinder**: State equivalence merging (if two proof states have identical goals) could reduce the search graph; worth exploring in Phase 5.

### 6.8 LeanHammer (2025) and the Hammer Tradition

Combines neural premise selection (LeanPremise) with symbolic methods (MePo relevance filter) and routes to both Aesop and ATP translation (Zipperposition). Achieves 37.3% cumulative proof rate on Mathlib, outperforming any single method by 21%. Magnushammer's two-stage SELECT+RERANK pipeline achieves 59.5% on PISA vs Sledgehammer's 38.3%.

**Hammer complementarity.** The broader hammer tradition (Sledgehammer for Isabelle, CoqHammer for Coq, LeanHammer for Lean) translates ITP goals into first-order logic for external ATP solvers. Wayfinder does not replace hammers; it handles the cases they cannot. Hammers dominate the AUTOMATION = -1 region of proof space (fully decidable, translatable to first-order logic). Wayfinder navigates the AUTOMATION ≥ 0 region (structured manual reasoning, lemma selection, subgoal decomposition). The optimal system is a **hybrid**: Wayfinder's navigational retrieval feeds premise candidates to hammers for the decidable portion, while handling the non-decidable portion via structured proof search. Integration point: when the navigator predicts AUTOMATION = -1 for a goal, delegate directly to LeanHammer/Aesop rather than attempting navigational resolution.

### 6.9 Knowledge Graphs for Mathematics

AutoMathKG (2025): 13,388 entities, 29,459 edges, SBERT embeddings, evaluated with TransE/DistMult/R-GCN. A separate Lean 4 Mathlib graph in Neo4j captures theorem dependencies. GNN-augmented ReProver (arXiv 2510.23637) layers an RGCN over ByT5 embeddings. Tree-Based Premise Selection (NeurIPS 2025) proposes training-free selection using Weisfeiler-Lehman kernels.

**Critical gap**: Not a single published system uses a knowledge graph to guide proof search or premise selection. The integration stops at basic dependency filtering. No system implements spreading activation over mathematical knowledge structures. No system positions entities in structured coordinate systems beyond standard embeddings. **This is Wayfinder's most distinctive design opportunity** — the proof network architecture is genuinely novel in this landscape.

### 6.10 TacticToe and Structured Tactic Prediction

TacticToe (HOL4) abstracts tactics to ~100 templates and separately predicts arguments. Graph2Tac (Coq, ICML 2024) uses GNN-based classification over ~100 tactic categories. ASTactic generates tactics as programs via grammar-controlled RNN. Proof transformation prediction (FroCoS 2023) guesses the correct tactic 74% of the time given before-and-after states.

**No existing system cleanly separates tactic category prediction from specific tactic/argument generation.** TacticToe's abstraction is the closest precedent, and its finding about argument prediction's importance validates the decomposition. Wayfinder's two-level architecture (ternary directions → structured resolution) is the first to develop this separation into a full navigation paradigm.

### 6.11 BitNet and Ternary Architectures

BitNet b1.58 (ternary weights) at 2B matches full-precision models on mathematical reasoning with 2.7× speedup and 3.5× memory reduction. No one has applied ternary architectures to formal theorem proving. **Wayfinder opportunity**: A ternary encoder (BitNet-style) feeding into a ternary decoder (existing STE infrastructure) would create a fully ternary pipeline — maximally efficient for real-time proof search.

### 6.12 ModelAtlas (Vinaik, 2025)

Semantic network of ~40k ML models positioned across 8 orthogonal signed banks with anchor-based similarity. IDF-weighted structured navigation outperforms keyword search and embedding-based similarity. **Connection to Wayfinder**: The entire navigational paradigm — banks, anchors, IDF scoring, multiplicative composition, spreading activation — is adapted from ModelAtlas.

### 6.13 Mutation Theory (Vinaik, 2025-2026; formalized in Lean 4)

200+ Lean 4 theorems formalizing specification complexity, exact learning, teaching dimension, decomposition, trajectory convergence, and phase transitions. **Connection to Wayfinder**: Provides formal convergence guarantees. See Sections 2.7-2.8.

---

## 7. Discussion

### 7.1 The Intelligence Budget Argument

The key insight driving Wayfinder is about *where to spend intelligence*. Current theorem provers spend enormous compute on neural inference during search — every candidate tactic, every candidate premise, every tree node expansion requires a forward pass through a large model. The model must encode *all* mathematical knowledge in its weights because search is parametric.

Wayfinder inverts this: mathematical knowledge is encoded *explicitly* in a structured semantic network (the proof network). The neural model handles only the genuinely hard part — understanding the current proof state and predicting which *direction* to go. Search and retrieval are outsourced to deterministic symbolic operations that are orders of magnitude cheaper per query.

This is the same design philosophy as ModelAtlas: "Don't waste tokens on problems that have been solved for 50 years — waste them on *your* terms." Set intersection, IDF computation, and Bellman-Ford are solved problems. Understanding a Lean goal state is not. Spend the neural compute where it matters.

### 7.2 Limitations

1. **Proof network construction requires Mathlib-specific engineering.** Bank positions and anchors must be extracted from Lean's type system, namespace hierarchy, and proof corpus. This is significant upfront work. The anchor gap analysis procedure (Section 2.3) mitigates but does not eliminate this cost.

2. **Scoring mechanism sensitivity.** The scoring mechanism (multiplicative, confidence-weighted, or hybrid — see Section 2.4) must balance precision against noise tolerance. The optimal mechanism likely varies with proof network maturity and may need to evolve during training. This is an active design question with multiple viable alternatives under evaluation.

3. **Anchor dictionary coverage.** The system can only retrieve premises whose relevant properties are captured by anchors. Subtle mathematical relationships not expressible as discrete labels will be missed. The iterative anchor gap analysis (Section 2.3) is designed to systematically close these gaps, but long-tail mathematical concepts will always be harder to anchor than common ones.

4. **Encoder quality.** The encoder is the system's perceptual bottleneck (Section 3.2). The strategy of evaluating math-native models, aggressively pruned large models, and fine-tuned sentence transformers mitigates this risk, but the optimal encoder may not be known until empirical evaluation in Phase 1.

5. **Comparison fairness.** DeepSeek-Prover-V2 uses a 671B model. Direct comparison on theorem-proved rate is unfair. Our contribution is architectural efficiency (proofs found per compute unit), not raw performance.

### 7.3 Hammer Complementarity

Wayfinder is not a replacement for ATP-based hammer tools (Sledgehammer, CoqHammer, LeanHammer). It is complementary. The AUTOMATION bank cleanly separates their domains:

- **AUTOMATION ≤ -1 (fully decidable)**: Goals that can be translated to first-order logic and solved by external ATPs. These are the hammer's domain. When the navigator predicts AUTOMATION = -1, delegate to LeanHammer/Aesop directly. This is the "known region" of proof space — no navigation needed.
- **AUTOMATION ≥ 0 (requires structured reasoning)**: Goals needing specific lemma selection, manual case analysis, induction, or multi-step decomposition. These are Wayfinder's domain. Hammers fail here because the goals aren't translatable to first-order logic, or the right premises aren't in the hammer's premise set.

LeanHammer (2025) shows that hybrid neural+symbolic achieves 37.3% — outperforming any single method by 21%. Wayfinder + hammer is a natural hybrid: navigational retrieval feeds premise candidates to hammers for the decidable portion, while handling the non-decidable portion via structured proof search. The AUTOMATION bank direction is the routing decision.

### 7.4 The Broader Vision

If navigational proof search works — if structured semantic networks can compete with dense retrieval for mathematical reasoning — the implications extend beyond theorem proving:

- **Any domain with structured output**: Code generation (API calls from a known catalog), molecular design (reactions from known reaction types), circuit synthesis (components from known libraries).
- **Any domain with a verification oracle**: The Lean kernel provides binary pass/fail. Any domain with an equivalent (compiler, simulator, SAT solver) can use the same navigational architecture.
- **Any domain where knowledge structure is known a priori**: Mathematics has a rich, well-understood structure (algebraic hierarchies, topological properties, order relations). Any domain with similar structure can be encoded in a semantic network.

The thesis of Wayfinder is not specific to theorem proving. It is a general claim about the relationship between structured knowledge and neural prediction: **when the knowledge structure is known, encode it explicitly and let the neural network navigate it. Don't force the neural network to learn what you already know.**

### 7.5 The Society of Mind Vision

NAV-001/002 results demonstrate that the navigational paradigm works (v1 thesis validated) but that a monolithic navigator cannot simultaneously master all six bank dimensions (v2 motivation). The specification complexity analysis (§2.9) explains why: the composition gap γ through the shared bridge forces multiplicative rather than additive complexity, and the mutation symmetry hierarchy (§2.9.3) means that Regime B banks (STRUCTURE, AUTOMATION, DEPTH) require fundamentally different representations than Regime A banks (DOMAIN, CONTEXT).

The Society of Mind architecture (§2.9.6) is the response: decompose proof search into typed temporal slots where each specialist operates at bounded σ, communicating through typed interfaces rather than shared weights. The composition gap theorem guarantees that total complexity is additive when γ ≈ 0 — which holds when specialists share only structured data (navigational coordinates, proof sketches, template IDs), not learned representations.

**Three concrete architectural advances over v1:**

1. **Narrative regime conversion.** The RECOGNITION slot classifies proofs into story templates, converting Regime B structure prediction into Regime A template classification. This is the single highest-leverage change: it introduces massive symmetry (|G_μ| >> 1) into a problem that otherwise has |G_μ| ≈ 1. Evidence from ARC (16+ templates), Ralph Loop (StoryFrame types), and Relational-AI (6-lens narrative pipeline) demonstrates that narrative templates are learnable and generalizable across diverse domains.

2. **PAB-guided specialist decomposition.** Rather than hand-designing the specialist scope, use PAB stability as the optimization signal. Train a candidate specialist, measure stability_regime, decompose if "chaotic." The free energy bound (Theorem 3.10) guarantees monotonic improvement. This creates a principled, data-driven decomposition procedure rather than an architectural guess.

3. **Proof sketch planning.** The PLANNING slot produces explicit proof sketches (ordered subgoal sequences with key lemma targets) before the EXECUTION slot attempts navigation. This separates *what to prove* (strategic, narrative) from *how to prove it* (tactical, navigational). DeepSeek-Prover-V2 demonstrated the power of this separation with its 671B decomposer + 7B prover architecture. Wayfinder achieves it at orders of magnitude smaller scale via story templates.

**The fundamental claim remains the same as v1** — navigate, don't predict — but now applied at the meta-level: the system navigates not just through proof space, but through *the space of proof strategies*, using narrative templates as the coordinate system for strategy selection.

---

## 8. Communication Architecture

Informed by personal semiotics framework (Winston's communication theory). The paper and experimental output are designed for mechanistic clarity.

### 8.1 Winston's Star for Wayfinder

| Element | Content |
|---------|---------|
| **Symbol** | Navigation through mathematical space -- the compass/map |
| **Slogan** | "Navigate, don't predict" -- or -- "One inference, then symbolic search" |
| **Surprise** | Zero neural inference at retrieval time; a knowledge graph replaces dense embedding |
| **Salience** | The 6-bank coordinate system: three ternary digits locate a tactic family |
| **Story** | "We built a system for navigating ML models through semantic coordinates. What if mathematical proofs worked the same way?" |

### 8.2 Fencing

- **IS**: Navigational proof search through structured semantic networks
- **IS NOT**: An LLM predicting tactic tokens
- **Differs from nearest thing** (ReProver): Neural embeddings are replaced by symbolic navigation from a single-inference coordinate. Neural network runs once; everything after is a database query.

### 8.3 Three Laws Applied

1. **Alignment precedes argument**: Open with the problem (proof search is expensive because every candidate requires neural inference), not the solution. Let the reader arrive at "what if we could do this symbolically?"
2. **Minimize surprise except in content**: Schema, SQLite, bank positions should feel familiar. The surprise: this works for theorem proving.
3. **The cheese contains the medicine**: mlx-vis visualizations ARE the argument. 98K theorems clustered by domain = thesis understood before it's stated.

### 8.4 Assertion-Evidence Figure Convention

Every figure title is a claim, not a topic:
- "Bank positions separate mathematical domains without training" (not "Bank position distribution")
- "AUTOMATION and STRUCTURE banks crystallize before DOMAIN and CONTEXT" (not "Training dynamics")
- "Navigational retrieval matches dense retrieval at 1/50th the neural compute" (not "Retrieval comparison")

### 8.5 Three-Stream Cycling

The three experimental streams cycle through one thesis from different angles:
- **Stream 1 (Navigation)**: The proof network enables retrieval without neural inference
- **Stream 2 (Architecture)**: Ternary coordinates generalize better than token indices
- **Stream 3 (PAB Process)**: Training dynamics reveal structured learning invisible to endpoint metrics

A reader who misses one angle catches it in another. Each stream feels like progression, not repetition.

---

## References

- AlphaProof Team (2024). Solving IMO Problems with AlphaProof. *DeepMind*.
- Azerbayev, Z., et al. (2024). Llemma: An Open Language Model for Mathematics. *ICLR*.
- First, E., et al. (2023). Baldur: Whole-Proof Generation and Repair with LLMs. *ESEC/FSE*.
- Gauthier, T. (2021). TacticToe: Learning to Reason with HOL4 Tactics. *LPAR*.
- Kumarappan, A., et al. (2025). LeanAgent: Lifelong Learning for Formal Theorem Proving. *ICLR*.
- Lample, G., Lacroix, T., et al. (2022). HyperTree Proof Search for Neural Theorem Proving. *NeurIPS*.
- Ma, S., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *arXiv:2402.17764*.
- Mikuła, M., et al. (2024). Magnushammer: A Transformer-Based Approach to Premise Selection. *ICLR*.
- Pal, P. (2025). Process-Aware Benchmarking (PAB). *PABKit*.
- Ren, Z., et al. (2025). DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Subgoal Decomposition. *arXiv*.
- Tishby, N., Pereira, F., Bialek, W. (2000). The Information Bottleneck Method. *arXiv:physics/0004057*.
- Trinh, T., et al. (2024). Solving Olympiad Geometry without Human Demonstrations. *Nature*.
- Valiant, L. (1984). A Theory of the Learnable. *Communications of the ACM*.
- Vinaik, R. (2025). ModelAtlas: A Navigable Semantic Network of ML Models.
- Vinaik, R. (2025). Orthogonal Ternary Projection: Informational Zeros and the Minority Channel Advantage. Data geometry research program.
- Vinaik, R. (2025). Constraint-Oriented Emergent Computation (COEC): A 7-Tuple Framework.
- Vinaik, R. (2025-2026). Mutation Theory. Formalized in Lean 4.
- Vinaik, R. (2026). Specification Complexity: A Five-Field Identification Theorem. Composition gap, regime classification, bulk-to-tail phase transitions.
- Winston, P. H. (2011). The Strong Story Hypothesis and the Directed Perception Hypothesis. *AAAI Fall Symposium*.
- Xin, H., et al. (2024). BFS-Prover: Scalable Best-First Proof Search for LLMs. *arXiv*.
- Yang, K., et al. (2024). LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. *NeurIPS*.
- Zhang, Y., et al. (2025). LeanHammer: An Automated Reasoning Tool for Lean 4. *arXiv*.
