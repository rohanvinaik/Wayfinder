# Wayfinder: Technical Design Reference

**Version:** 2.1
**Date:** March 19, 2026
**Corresponding documents:** `WAYFINDER_RESEARCH.md` (theory), `WAYFINDER_PLAN.md` (operational plan), `EXPERIMENT_RESULTS.md` (results ledger)

---

## 1. Design Thesis

**Navigation, not prediction.** The central design decision in Wayfinder is that proof search should be *spatial navigation through structured mathematical space* rather than *sequential token prediction*. This has a concrete architectural consequence: the neural network runs once per proof state to produce navigational coordinates, and all subsequent operations — premise retrieval, tactic resolution, search prioritization — are deterministic symbolic operations on a precomputed semantic network.

This design is directly adapted from ModelAtlas (Vinaik, 2025), which demonstrated that structured navigation over signed semantic coordinates outperforms both flat database queries and dense embedding retrieval for finding ML models on HuggingFace. We apply the same paradigm to mathematical entities: lemmas, tactics, and proof states are positioned in a structured coordinate system; retrieval is multiplicative bank alignment × IDF-weighted anchor relevance × seed similarity.

**Decompose, then navigate (v2).** NAV-001/002 training runs demonstrated that a monolithic navigator produces chaotic PAB dynamics (stability_mean > 0.30) because the six bank dimensions have heterogeneous difficulty — Regime A banks (DOMAIN, CONTEXT) saturate early while Regime B banks (STRUCTURE, AUTOMATION, DEPTH) never converge. The composition gap theorem (σ(A∘B) ≤ σ(A) + σ(B) + γ(A,B)) explains why: the shared bridge creates high γ between bank representations. The v2 architecture decomposes proof search into a Society of Mind — six typed temporal slots (PERCEPTION → RECOGNITION → PLANNING → TEMPORAL ORCHESTRATION → EXECUTION → VERIFICATION) with independent specialists. Each specialist operates at bounded specification complexity, communicating through typed interfaces rather than shared weights. PAB stability per specialist serves as the empirical proxy for σ, guiding decomposition until every component is "stable." See `WAYFINDER_RESEARCH.md` §2.9 for the full theoretical argument.

The thesis is falsifiable. If dense retrieval consistently outperforms structured navigation for premise selection (Phase 2.2 of the plan), the navigational paradigm is wrong for this domain. If tactic classification outperforms navigational coordinates (Phase 4.4 ablation), the ternary decoder design is wrong. If a monolithic navigator with sufficient capacity achieves stable PAB dynamics (Phase 6 comparison), the SoM decomposition is unnecessary. All negative results would be informative.

**Post-EXP-048 update.** The first execution-level result beyond rewrite collapse is now in hand:
for step-0 `apply`, a small executable selector trained on Lean-generated feedback improves live
`LeanAccepted` from `15/91 (16.5%)` under cosine top-1 to `35/91 (38.5%)` on held-out goals. This
sharpens the design claim. The right learned object is not "the best theorem name." It is the
best **structured executable action boundary** inside a compiler-checked interface.

---

## 2. The Proof Network

The proof network is a SQLite database encoding mathematical structure. It is the architectural equivalent of ModelAtlas's `network.db` — the precomputed semantic space through which all search navigates.

### 2.1 Proof Banks

Six orthogonal signed dimensions. Each bank has a **zero state** at the mode — the most common mathematical situation a working mathematician encounters.

| Bank | Zero State | Negative Direction | Positive Direction | Extraction Source |
|------|-----------|-------------------|-------------------|-------------------|
| STRUCTURE | equality / rewrite | decidable, arithmetic | quantified, dependent | Goal type analysis |
| DOMAIN | general algebra | concrete (ℕ, ℤ, ℚ) | abstract (topology, category) | Namespace hierarchy |
| DEPTH | 2-3 tactic proof | trivial (1 tactic) | deep (10+ tactics) | Proof length |
| AUTOMATION | partially automated | fully auto (omega, simp) | manual reasoning | Tactic analysis |
| CONTEXT | moderate context (3-5 hyps) | no hypotheses | rich context (10+) | Goal state parsing |
| DECOMPOSITION | single goal | atomic (no splits) | multi-subgoal | Proof structure |

**Signed position encoding**: Each entity stores `(sign, depth)` per bank. `sign ∈ {-1, 0, +1}`, `depth ∈ {0, 1, 2, 3}`. Signed position = `sign × depth`.

**Why signed hierarchies instead of categories**: A categorical "domain" field with values {algebra, topology, analysis} can't express proximity. Signed positions give gradient scoring — a linear algebra lemma (DOMAIN = 0, between concrete and abstract) scores better against an algebra query than a topology lemma, without binary mismatch. Zero placement at the mode means most queries resolve near the origin.

### 2.2 Anchor Dictionary

A shared vocabulary of semantic labels. Target: ~300 anchors at launch, growing organically as the system discovers mathematical patterns.

**Bootstrap anchors** (created before training):

**STRUCTURE bank:**
`equality`, `inequality`, `membership`, `universal-quantifier`, `existential-quantifier`, `iff-statement`, `implication`, `negation`, `dependent-type`, `pi-type`, `sigma-type`, `propositional`, `data-type`, `function-type`

**DOMAIN bank:**
`nat-arithmetic`, `int-arithmetic`, `rat-arithmetic`, `real-analysis`, `complex-analysis`, `group-theory`, `ring-theory`, `field-theory`, `module-theory`, `linear-algebra`, `topology`, `metric-space`, `measure-theory`, `probability`, `order-theory`, `lattice-theory`, `set-theory`, `finset`, `multiset`, `category-theory`, `homological-algebra`, `number-theory`, `combinatorics`, `graph-theory`

**DEPTH bank:**
`one-liner`, `two-step`, `multi-step`, `deep-induction`, `structural-recursion`, `nested-proof`

**AUTOMATION bank:**
`omega-solvable`, `simp-solvable`, `decide-solvable`, `norm-num-solvable`, `linarith-solvable`, `ring-solvable`, `field-simp-solvable`, `aesop-solvable`, `tauto-solvable`, `needs-manual-intro`, `needs-manual-cases`, `needs-manual-induction`, `needs-manual-apply`, `needs-manual-exact`

**CONTEXT bank:**
`hypothesis-free`, `hypothesis-light`, `hypothesis-rich`, `uses-local-def`, `uses-instance`, `uses-classical`, `uses-choice`

**DECOMPOSITION bank:**
`single-goal`, `needs-cases`, `needs-induction`, `needs-have-chain`, `needs-constructor`, `multi-branch`, `needs-rcases`, `needs-obtain`

**Cross-cutting anchors** (mathematical properties, not bank-specific):
`monotonicity`, `commutativity`, `associativity`, `distributivity`, `idempotency`, `transitivity`, `reflexivity`, `symmetry`, `antisymmetry`, `injectivity`, `surjectivity`, `bijectivity`, `continuity`, `differentiability`, `integrability`, `compactness`, `connectedness`, `finiteness`, `countability`, `convergence`, `boundedness`, `well-foundedness`, `decidability`, `cancellation`, `absorption`

**Known anchor gaps** (to be addressed in bootstrap):
- **Type coercion**: `needs-cast`, `nat-int-coercion`, `subtype-coercion`, `coe-simp` — Mathlib proofs frequently struggle with coercion between numeric types
- **Algebraic hierarchy**: `monoid-level`, `group-level`, `ring-level`, `field-level` — position in the typeclass hierarchy
- **Proof patterns**: `split-and-recombine`, `contrapositive-argument`, `epsilon-delta`, `sequence-limit`, `diagonal-argument` — recurring structural patterns beyond individual tactics

**Anchor lifecycle**:
1. **Bootstrap**: ~300 anchors from the lists above, plus known gap anchors.
2. **Deterministic extraction**: Type analysis and namespace matching assign anchors with confidence 1.0.
3. **Pattern matching**: Proof text analysis assigns anchors with confidence 0.8.
4. **Discovery**: New anchors may be minted when the extraction pipeline encounters unclassifiable patterns. These start at confidence 0.5 and are validated by checking co-occurrence with existing anchors.
5. **Gap analysis** (iterative, Phase 0-1): For 500 random proof steps, attempt resolution with current anchors. For each failure (correct premise not in top-16), identify what anchors *would have* connected goal to premise. Cluster gaps by theme, add as new anchors, re-run. Iterate until top-16 recall exceeds 70% on the sample. This is the highest-leverage Phase 0 activity.

**IDF computation**: `idf(anchor) = log(N / count_entities_with_anchor)`. Cached in `anchor_idf` table. Invalidated and recomputed after any batch update to entity-anchor links.

### 2.3 Entity Links

Explicit relationships between mathematical entities:

| Relation | Weight | Extraction Source |
|----------|--------|-------------------|
| `depends_on` | 0.9 | Proof references this lemma |
| `same_namespace` | 0.7 | Share Mathlib namespace (e.g., `Nat.`) |
| `same_declaration_block` | 0.8 | Adjacent in source file |
| `commonly_co_occurs` | 0.6 | Frequently used together in proofs |
| `tactic_precedes` | 0.5 | Tactic T commonly follows tactic S |
| `generalization_of` | 0.85 | One lemma generalizes another |

Links enable the spreading activation channel. When a proof has already used lemma A, activation spreads through `depends_on` and `commonly_co_occurs` links to find related lemmas.

### 2.4 Tactic Entities

Tactics are entities in the proof network, not just string labels. Each tactic has bank positions and anchors:

| Tactic | STRUCTURE | AUTOMATION | Anchors |
|--------|-----------|------------|---------|
| `omega` | (-1, 2) | (-1, 2) | `omega-solvable`, `nat-arithmetic`, `int-arithmetic`, `decidable` |
| `simp` | (-1, 1) | (-1, 1) | `simp-solvable`, `rewriting`, `simplification` |
| `intro` | (0, 0) | (+1, 1) | `needs-manual-intro`, `universal-quantifier`, `implication` |
| `cases` | (+1, 1) | (+1, 1) | `needs-cases`, `multi-branch`, `inductive-type` |
| `induction` | (+1, 2) | (+1, 2) | `needs-induction`, `structural-recursion`, `nat-arithmetic` |
| `apply` | (0, 0) | (+1, 1) | `needs-manual-apply`, `implication` |
| `exact` | (-1, 1) | (+1, 1) | `needs-manual-exact`, `one-liner` |
| `have` | (+1, 1) | (+1, 1) | `needs-have-chain`, `multi-step`, `decomposition` |
| `rw` | (0, 0) | (0, 0) | `rewriting`, `equality` |
| `linarith` | (-1, 2) | (-1, 2) | `linarith-solvable`, `inequality`, `arithmetic` |

When the Proof Navigator produces navigational coordinates, `navigate()` over the tactic entity table finds the tactics whose bank positions align with the predicted direction.

**External validation**: This two-level resolution (coarse tactic category → fine argument selection) is validated by TacticToe's finding that *argument prediction for abstracted tactics has the highest impact on proof success*. The hard part isn't selecting `apply` vs `cases` — it's selecting *which lemma* to apply. Navigation handles the coarse level; the proof network's anchor-based premise retrieval handles the fine level. See `WAYFINDER_RESEARCH.md` Section 2.5 for the activation steering evidence that further supports this decomposition.

---

## 3. Scoring System

### 3.1 Bank Alignment (adapted from ModelAtlas)

For each bank where the query specifies a direction:

```python
def bank_score(model_signed_pos, query_direction):
    if query_direction == 0:
        return 1.0 / (1.0 + abs(model_signed_pos))  # want zero: penalize distance
    alignment = model_signed_pos * query_direction
    if alignment > 0:
        return 1.0      # on the right side
    elif alignment == 0:
        return 0.5      # at zero (neutral)
    else:
        return 1.0 / (1.0 + abs(alignment))  # wrong side: decay
```

**Scoring composition — configurable mechanism.** The default is multiplicative (`bank_alignment = Π bank_score(pos_i, dir_i)`), but the system supports alternative composition strategies selected via `scoring.mechanism` in config:

```python
def compose_bank_scores(scores: dict[str, float], confidences: dict[str, float],
                        mechanism: str) -> float:
    if mechanism == "multiplicative":
        return math.prod(scores.values())
    elif mechanism == "confidence_weighted":
        # Banks with uncertain positions contribute less
        return math.prod(s ** c for s, c in zip(scores.values(), confidences.values()))
    elif mechanism == "soft_floor":
        epsilon = config.scoring.floor_epsilon  # default 0.1
        return math.prod(max(s, epsilon) for s in scores.values())
    elif mechanism == "geometric_mean":
        return math.prod(scores.values()) ** (1.0 / len(scores))
    elif mechanism == "log_additive":
        weights = config.scoring.bank_weights  # learned or tuned per bank
        return math.exp(sum(w * math.log(max(s, 1e-6))
                            for w, s in zip(weights, scores.values())))
```

**Recommended starting point**: `confidence_weighted`. The GoalAnalyzer's bank heads output softmax distributions — the max probability serves as a natural confidence estimate. Banks with uncertain positions (max prob ~0.4) automatically contribute less to the final score than banks with confident positions (max prob ~0.95). This degrades gracefully: high confidence → pure multiplicative; low confidence → near-neutral.

**Missing bank penalty**: If an entity lacks a position for a queried bank, score = 0.3 (not zero, but penalized). This is intentional — some entities genuinely don't have a meaningful position on some banks.

### 3.2 Anchor Relevance (IDF-weighted)

Three anchor lists:

- **require** (hard filter): `SELECT entity_id ... HAVING COUNT(DISTINCT anchor_id) = len(require)`. Missing any required anchor → entity excluded.
- **prefer** (IDF-weighted boost): `score = Σ idf(matched) / Σ idf(all_preferred)`. Rare anchors count more.
- **avoid** (decay): Each avoided anchor present halves the score: `0.5^count`.

### 3.3 Seed Similarity (IDF-weighted Jaccard)

When the proof context includes already-used lemmas:

```
seed_similarity = Σ idf(shared_anchors) / Σ idf(union_anchors)
```

Standard Jaccard treats all anchors equally. IDF weighting means sharing `Lyapunov-monotonicity` (rare) matters more than sharing `equality` (ubiquitous).

### 3.4 Final Score

```
final_score = bank_alignment × anchor_relevance × seed_similarity
```

All three components are in [0, 1]. The product ensures no averaging away bad matches. A lemma that perfectly matches domain but completely fails structure scores 0, not 0.5.

---

## 4. Spreading Activation

Priority-queue Bellman-Ford from seed nodes, adapted from ModelAtlas.

### 4.1 Two Channels

**Link channel (Layer 1):** Traverse `entity_links` bidirectionally. Activation decays by link weight × hop decay (0.8 per hop). A `depends_on` link (weight 0.9) transmits 72% of activation per hop. A `commonly_co_occurs` link (weight 0.6) transmits 48%.

**Anchor channel (Layer 2):** Find entities sharing anchors with the current node. Activation weighted by fraction of shared anchors. A entity sharing 8/10 of the current node's anchors receives higher activation than one sharing 2/10.

### 4.2 Bank Scoping

If the current proof is about algebra (DOMAIN ≤ 0), only spread through DOMAIN-bank anchors. This prevents topology lemmas from activating during algebra proofs via shared generic anchors like `equality`.

### 4.3 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_depth` | 3 | Beyond 3 hops, activation is too diluted to be meaningful |
| `decay` | 0.8 | 20% loss per hop — aggressive enough to focus, gentle enough to explore |
| `neighbor_slice` | 20 | Max link neighbors per node — prevents explosion |
| `anchor_slice` | 30 | Max anchor co-occurrences per node |

### 4.4 Dynamic Seeds

Seeds change as the proof progresses:
- **Initially**: Just the theorem's goal state entity.
- **After each successful tactic**: Add the used lemmas and the new subgoal states as seeds.
- **After each failure**: Add the failed tactic/premise to an "avoid" set that dampens their activation.

This creates a feedback loop: successful proof steps refine the search space, making subsequent steps more likely to find relevant lemmas.

---

## 5. Navigational Training Data Format

Each training example maps a proof step to navigational labels:

```json
{
  "goal_state": "⊢ ∀ x : ℕ, 0 + x = x",
  "theorem_id": "Nat.zero_add",
  "step_index": 0,
  "total_steps": 1,

  "nav_labels": {
    "structure": -1,
    "automation": -1,
    "decompose": -1
  },

  "anchor_labels": [
    "nat-arithmetic", "equality", "universal-quantifier",
    "omega-solvable", "one-liner"
  ],

  "ground_truth": {
    "tactic": "omega",
    "premises": [],
    "remaining_steps": 0,
    "solvable": true
  },

  "bank_positions": {
    "STRUCTURE": [0, 0],
    "DOMAIN": [-1, 1],
    "DEPTH": [-1, 2],
    "AUTOMATION": [-1, 2],
    "CONTEXT": [-1, 1],
    "DECOMPOSITION": [-1, 1]
  }
}
```

### 5.1 Tactic-to-Direction Mapping (6-Bank)

The v1 design navigated 3 banks (STRUCTURE, AUTOMATION, DECOMPOSE), giving only 27 direction vectors. Several distinct tactics collapsed to the same coordinates (e.g., `intro` and `apply` both at (0, +1, 0)). The expanded 6-bank mapping adds DOMAIN, DEPTH, and CONTEXT directions, giving 729 possible vectors and disambiguating nearly all common tactics.

| Tactic Category | STRUCT | AUTO | DECOMP | DOMAIN | DEPTH | CONTEXT | Representative Tactics |
|----------------|--------|------|--------|--------|-------|---------|----------------------|
| Decision procedures | -1 | -1 | -1 | -1 | -1 | 0 | `omega`, `decide`, `norm_num` |
| Simplification | -1 | -1 | 0 | 0 | -1 | 0 | `simp`, `simp_all`, `ring`, `field_simp` |
| Rewriting | 0 | 0 | 0 | 0 | 0 | 0 | `rw`, `conv`, `unfold` |
| Linear arithmetic | -1 | -1 | -1 | -1 | -1 | +1 | `linarith`, `positivity` (uses hyps) |
| Introduction | 0 | +1 | 0 | 0 | 0 | +1 | `intro`, `rintro` (creates hyps) |
| Elimination | +1 | +1 | +1 | 0 | +1 | +1 | `cases`, `rcases`, `obtain` |
| Induction | +1 | +1 | +1 | 0 | +1 | 0 | `induction`, `induction'` |
| Application | 0 | +1 | 0 | 0 | 0 | 0 | `apply`, `exact`, `refine` |
| Subgoal creation | 0 | +1 | +1 | 0 | +1 | +1 | `have`, `suffices`, `calc` |
| Assumption | -1 | -1 | -1 | 0 | -1 | -1 | `assumption`, `exact?`, `trivial` |
| Contradiction | +1 | +1 | 0 | 0 | 0 | +1 | `contradiction`, `exfalso` (uses hyps) |
| Automation with hints | -1 | 0 | 0 | 0 | 0 | 0 | `aesop`, `tauto`, `push_neg` |
| Type coercion | 0 | -1 | 0 | -1 | -1 | 0 | `norm_cast`, `push_cast` |

**Key disambiguations from 6-bank expansion:**
- `intro` (CONTEXT=+1, creates hypotheses) vs `apply` (CONTEXT=0, consumes the goal)
- `cases` (CONTEXT=+1, adds hypotheses per case) vs `induction` (CONTEXT=0, structural)
- `linarith` (CONTEXT=+1, uses hypotheses) vs `omega` (CONTEXT=0, self-contained)
- `assumption` (CONTEXT=-1, discharges from context) vs other closers

**Bank semantics:**
- **STRUCTURE -1** = reduces structural complexity (decides, simplifies)
- **STRUCTURE +1** = increases structural complexity (introduces cases, subgoals)
- **AUTOMATION -1** = requires no human insight
- **AUTOMATION +1** = requires choosing what to apply/introduce
- **DECOMPOSE +1** = creates new subgoals
- **DOMAIN -1** = concrete/computational domain (naturals, integers)
- **DOMAIN +1** = abstract/structural domain (topology, categories)
- **DEPTH -1** = shallow step (proof gets shorter)
- **DEPTH +1** = deep step (proof branches or lengthens)
- **CONTEXT -1** = reduces or ignores context (discharges, clears)
- **CONTEXT +1** = enriches or uses context (introduces, applies hypotheses)

**Graceful degradation**: The system supports a `navigable_banks` config parameter. Setting `navigable_banks: [STRUCTURE, AUTOMATION, DECOMPOSITION]` reverts to 3-bank navigation if the full 6-bank version proves too hard to learn.

---

## 6. The Ternary-Navigation Correspondence

The deepest architectural insight in Wayfinder is the correspondence between ModelAtlas's signed bank positions and the ternary decoder's {-1, 0, +1} weights.

### 6.1 Structural Isomorphism

ModelAtlas: `bank_position = sign × depth`, where `sign ∈ {-1, 0, +1}`.
Ternary decoder: `weight ∈ {-1, 0, +1}`, via STE quantization.

Both enforce the same computational regime:
- **+1**: "This dimension is positively relevant."
- **-1**: "This dimension is negatively relevant."
- **0**: "This dimension is irrelevant."

In ModelAtlas, this means: "This model is positively positioned on the CAPABILITY axis."
In Wayfinder's decoder, this means: "The next proof step should go in the positive direction on the STRUCTURE axis."

### 6.2 Why Ternary Beats Continuous for Navigation

A continuous decoder could output navigational coordinates as real-valued vectors in [-1, +1]^6. But continuous coordinates suffer from:

1. **Ambiguity at zero**: Is 0.001 "slightly positive" or "noise"? Ternary forces commitment — the weight is exactly 0 (irrelevant) or exactly ±1 (committed). This matches the navigational semantics: you're either heading toward more automation, away from it, or you don't care.

2. **Gradient dilution**: Continuous weights spread gradient across all values. Ternary weights via STE concentrate gradient on the sign decision, which is the only thing that matters for navigation.

3. **Interpretability**: A ternary navigational vector like (+1, -1, 0, +1, 0, -1) is immediately readable: "increase structural complexity, maximize automation, don't care about decomposition, ..." A continuous vector (0.23, -0.87, 0.02, 0.91, -0.11, -0.45) requires thresholding to interpret.

4. **Crystallization as learning signal**: PAB can track when each navigational dimension crystallizes (commits to a stable ternary value). A dimension that crystallizes early is one the model has "figured out." A dimension that oscillates is one where the model is uncertain. This is invisible in continuous coordinates.

### 6.3 Resolution Granularity

With the expanded 6-bank navigation, the ternary decoder gives *coarse* navigation (3^6 = 729 possible direction vectors). The anchor logits give *fine* navigation (~300+ possible labels, top-k selected). Together they form a `StructuredQuery` that resolves to specific tactics and premises through the proof network.

The 6-bank expansion resolves the many-to-one problem from the original 3-bank design (3^3 = 27 bins). Nearly all common tactics now have unique direction vectors, reducing the disambiguation burden on the anchor channel. See Section 5.1 for the full mapping.

This two-level resolution mirrors ModelAtlas's query structure:
- Bank directions (ternary) → coarse filtering ("small code models")
- Anchor preferences (continuous, IDF-weighted) → fine ranking ("with tool-calling support, not embedding models")

---

## 7. Connection to Mutation Theory

The Mutation Theory corpus provides formal guarantees applicable to Wayfinder's navigational search. These connections are structural correspondences, not loose analogies — theorem proving is the ML application domain most naturally aligned with formal learning theory (exact verification oracle, well-defined state space, decomposable structure). See `WAYFINDER_RESEARCH.md` Section 2.7-2.8 for the full argument.

### 7.1 Convergence Guarantees

**Theorem (Trajectory Monotonicity, T7.0):** Killed sets grow monotonically along well-structured trajectories.

**Application:** In ModelAtlas terms: as you add constraints (require/prefer/avoid anchors), the candidate set monotonically shrinks. In proof search terms: each successful tactic application monotonically reduces the survivor set (remaining open goals). The Lyapunov monotonicity theorem (T7.5) guarantees convergence when the progress head's estimate (remaining steps) decreases at each step. This connects progress prediction → Lyapunov estimate → T7.5 → formal convergence guarantee.

**Research direction:** Existing work on proof complexity measures (proof tree depth, cut-rank, quantifier depth) may provide better Lyapunov function candidates than raw step count. See Phase 2.0 literature review.

### 7.2 Phase Transition

**Theorem (Phase Transition, T7.3):** Greedy kill trajectories exhibit a structural transition step: initial rapid progress followed by a harder regime.

**Application:** The first 1-3 tactics in most proofs are easy (intro, unfold, basic rewrites — near the origin in proof space). The hard part is finding the right deep lemma (far from origin, requiring specific anchors). The phase transition is precisely where navigational retrieval matters most — after the easy tactics, the model must navigate to specific, rare anchors in specialized regions of the semantic network.

### 7.3 Fixed-Point Partition

**Theorem (fixedPointPartition, T4):** The mutant space partitions into Killed × Equivalent × DistinguishableSurvivors.

**Application:** The proof space partitions identically:

| Mutation Theory Partition | Scoring Signal | Role |
|--------------------------|---------------|------|
| Killed vs. Survivor | `bank_alignment` | Right region of proof space? |
| Equivalent detection | `anchor_relevance` | Semantic labels match? (IDF-weighted) |
| Distinguishable → navigate to known | `seed_similarity` | Similar to already-solved patterns? |

The three-signal multiplicative scoring structure mirrors this three-way partition.

### 7.4 Decomposition Optimality

**Theorem (Separable Independent Testing, T3):** When a system decomposes into independent components, each can be tested independently with additive (not multiplicative) cost.

**Application:** The SubgoalDecomposer's decision to create `have` subgoals is formally justified when the subgoals are independent. Independent subgoals can be solved with separate navigational queries, and the total compute is `Σ solve_cost(subgoal_i)`, not `Π`. Independence can be verified by checking whether subgoals share anchors — shared anchors suggest dependency.

### 7.5 Teaching Dimension

**Theorem (TD = κ, T5.17):** Teaching dimension equals specification complexity.

**Application:** The minimum number of premises needed to uniquely determine a proof is bounded by the specification complexity of the theorem. This provides a formal lower bound on the `premise_candidates` parameter: if a theorem's specification complexity is 4, retrieving 3 premises is provably insufficient. The proof network's anchor count per theorem approximates specification complexity.

---

## 8. Module Architecture

All modules are implemented. The codebase consists of 28 source files and 11 scripts.

### 8.0 Three-Lane Verification Architecture

Proof verification operates across three lanes at different granularities:

| Lane | Purpose | Backend | Latency | Metric Category |
|------|---------|---------|---------|-----------------|
| **A** | Step-wise tactic search (inner loop) | Pantograph (local) | ~1ms/tactic | `raw_success` |
| **B** | Proof audit: verify, repair, decompose | Axle API (cloud) | ~100-200ms/call | `axle_assisted_success`, `axle_repair_only` |
| **C** | High-assurance recheck (leaderboard only) | lean4checker / Comparator / SafeVerify | seconds | independent validation |

**Lane A** is the core search loop (`proof_search.py` → `lean_interface.py`). It applies tactics one at a time within a 600-step budget.

**Lane B** is the `ProofAuditor` service (`proof_auditor.py` → Axle API). It operates on complete or near-complete proofs after Lane A finishes. Operations: `verify_proof` (authoritative check), `repair_proofs` (close remaining goals with terminal tactics), `sorry2lemma` (decompose failures into subgoals for retry). Lane B never inflates Lane A metrics — results are tagged with strict `SuccessCategory`.

**Lane C** is optional and reserved for results submitted to benchmarks/leaderboards. Axle itself notes that `verify_proof` trusts the Lean environment and is not adversarial-safe (see [Axle docs](https://axle.axiommath.ai/v1/docs/)). For high-assurance, use [lean4checker](https://github.com/leanprover/lean4checker), [Comparator](https://github.com/leanprover/comparator), or [SafeVerify](https://github.com/GasStationManager/SafeVerify).

**Cost controls (Lane B):**
- Content-hash cache (SHA-256) avoids redundant Axle API calls
- Caller filters to top-N candidates or high critic score before invoking
- Bounded async concurrency (`max_concurrency` in config)
- Graceful degradation on timeout/429/503 (returns failed `AuditResult`, search continues)

**Metric separation** prevents inflated claims. Benchmark reports always show:
- `raw_success`: proved by Lane A alone, no Axle involvement
- `axle_assisted_success`: Lane A partial + Axle decompose/repair
- `axle_repair_only`: Axle `repair_proofs` closed remaining goals

**Scope caveat:** Axle is designed for simple imports/theorems/definitions. Tool count and API surface are evolving (v1.0.0, March 2026). Pin `environment` aggressively in config; update only with Mathlib version bumps.

```
src/
├── encoder.py               GoalEncoder (math-native model, see Research §3.2)
├── goal_analyzer.py          GoalAnalyzer (features + bank positions + anchor logits)
├── bridge.py                 InformationBridge (continuous → discrete bottleneck)
├── proof_navigator.py        ProofNavigator (ternary navigational decoder)
├── proof_network.py          Proof network database interface
├── resolution.py             Structured resolution (navigate + spread)
├── proof_search.py           Lane A: outer search loop with Lean interaction
├── lean_interface.py         Lane A: Pantograph-based Lean kernel interaction
├── proof_auditor.py          Lane B: Axle-based verification/repair/decomposition
├── nav_contracts.py          Navigational data contracts (NavigationalExample, NavOutput, etc.)
├── ternary_decoder.py        TernaryLinear, ternary_quantize (shared)
├── domain_gate.py            OOD detection gate
├── losses.py                 NavigationalLoss (UW-SO adaptive)
├── lowering.py               Deterministic tactic lowering
├── contracts.py              Core data contracts (ProofExample, etc.)
├── data.py                   Dataset loading (NavigationalDataset, etc.)
├── verification.py           Proof verification
├── pab_profile.py            PAB profile artifact
├── pab_metrics.py            PAB metric functions
├── pab_tracker.py            Checkpoint accumulation
├── behavioral_fingerprint.py Behavioral signatures
├── trainer.py                Training loop
├── trainer_steps.py          Train step mixin
├── trainer_checks.py         Gradient health, logging
├── trainer_constants.py      Domain inference
├── trainer_setup.py          Pipeline construction
├── trainer_validation.py     Proof validation
└── evaluate.py               Evaluation harness

scripts/
├── extract_proof_network.py    Populate proof network from Mathlib
├── build_nav_training_data.py  Convert proof traces to nav labels + `SubtaskIR` metadata
├── build_proof_network_db.py   Load entities JSONL → SQLite proof network
├── convert_leandojo.py         Bridge LeanDojo Benchmark 4 → flat JSONL
├── anchor_gap_analysis.py      Iterative anchor gap analysis
├── eval_encoders.py            Phase 0.6 encoder evaluation (10 candidates, 4 tiers)
├── tactic_maps.py              Tactic-to-direction mapping tables
├── train_navigator.py          Training script with curriculum
├── eval_retrieval.py           Nav vs dense retrieval comparison
├── eval_spreading.py           Spreading activation evaluation
└── run_benchmark.py            MiniF2F + Mathlib benchmark runner

configs/
└── wayfinder.yaml              Full Wayfinder experiment configuration

data/  (generated/downloaded at runtime)
├── proof_network.db            SQLite semantic network
├── nav_training.jsonl          Navigational training data + controller-facing move metadata
├── nav_eval.jsonl              Navigational eval data (frozen)
├── nav_train_templates.jsonl   Template-labeled nav data + theorem-level move profiles
├── template_taxonomy.json      Template counts + aggregated move profiles
└── leandojo_mathlib.jsonl      LeanDojo extracted dataset
```

### 8.1 Navigational Modules

**`proof_navigator.py`**: Replaces `ternary_decoder.py`'s vocabulary classification heads with 6 navigational direction heads (one per navigable bank). Inherits TernaryLinear for hidden layers. Outputs 6 ternary directions (3^6 = 729 coarse bins) + continuous anchor logits + progress + critic. Configurable via `navigable_banks` for graceful degradation to 3-bank.

**`proof_network.py`**: SQLite interface for the semantic network. Adapted from ModelAtlas's `db.py`. Schema: entities, entity_positions, anchors, entity_anchors, entity_links, anchor_idf. Core functions: `navigate()`, `spread()`, `get_anchor_set()`, `compute_idf()`, `batch_get_positions()`.

**`resolution.py`**: Converts ProofNavigator output to concrete tactics and premises. Builds a StructuredQuery from ternary directions + top-k anchors. Calls `navigate()` on the tactic and premise entity tables. Combines with `spread()` scores from proof context. Returns ranked (tactic, premise_list) pairs.

**`proof_search.py`**: Outer search loop. Manages open goals, proof context, search budget. Calls the neural pipeline once per goal, resolution symbolically, and the Lean kernel for verification. Goal selection uses critic (solvability estimate) and progress (remaining steps). Uses `Pipeline` dataclass bundling encoder/analyzer/bridge/navigator.

**`lean_interface.py`** (Lane A): Wrapper around LeanDojo's Pantograph interaction protocol. Provides `try_tactic(goal_state, tactic_text) -> TacticResult` with success/failure and new goal states. Supports stub backend for offline testing. Configurable via `lean.backend` in config.

**`proof_auditor.py`** (Lane B): Axle API wrapper for proof verification, repair, and subgoal decomposition. `ProofAuditor` service with `verify()`, `repair()`, `decompose()`, `check()`, and `extract_theorems()`. Content-hash cache, bounded concurrency, graceful degradation on API errors. All results tagged with `SuccessCategory` (raw/assisted/repair_only). Optional dependency: `pip install wayfinder[axle]`.

**`nav_contracts.py`**: Navigational data contracts including `NavigationalExample`, `NavOutput`, `StructuredQuery`, `Candidate`, `SearchContext`, `BANK_NAMES`.

### 8.2 Retained Modules

All modules from the BalancedSashimi port are retained. The GoalAnalyzer is extended (not replaced) with bank and anchor heads. The InformationBridge is unchanged. The TernaryLinear and STE infrastructure is shared between the existing TernaryDecoder and the new ProofNavigator.

---

## 9. Configuration

```yaml
# configs/wayfinder.yaml — Wayfinder configuration (reference; see actual file for current values)

# Proof network
proof_network:
  db_path: data/proof_network.db
  banks: [STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION]
  navigable_banks: [STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION]  # all 6
  max_spread_depth: 3
  spread_decay: 0.8
  missing_bank_penalty: 0.3
  avoid_decay: 0.5
  navigate_limit: 20
  accessible_premises: true  # filter to import-accessible premises (ReProver: ~2% free recall gain)

# Scoring
scoring:
  mechanism: confidence_weighted  # multiplicative | confidence_weighted | soft_floor | geometric_mean | log_additive
  floor_epsilon: 0.1  # only used by soft_floor
  # bank_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # only used by log_additive

# Goal encoder — active design decision, see WAYFINDER_RESEARCH.md Section 3.2
encoder:
  model_name: TBD  # evaluate: ByT5-small, Qwen-3.5-math, DeepSeek-Math, BitNet-ternary, pruned-7B
  embed_dim: 1472  # ByT5-small default; will change with encoder selection
  frozen: false  # default to fine-tunable; freeze only if compute-constrained
  byte_level: true  # critical for Lean Unicode (∀, ⊢, →, ℕ); ByT5 and BitNet candidates

# Goal analyzer
goal_analyzer:
  feature_dim: 256
  num_banks: 6
  num_anchors: 300  # bootstrap dictionary size, grows via gap analysis

# Information bridge
bridge:
  bridge_dim: 128

# Proof navigator
proof_navigator:
  hidden_dim: 256
  num_layers: 2
  num_direction_heads: 6  # one per navigable bank
  ternary_enabled: true
  partial_ternary: false  # ternary hidden, continuous heads

# Training
training:
  batch_size: 32
  learning_rate: 1e-3
  warmup_steps: 100
  total_steps: 5000
  checkpoint_interval: 50
  curriculum:
    phase_a_steps: 500    # 1-2 step proofs
    phase_b_steps: 1500   # ≤5 step proofs
    # phase_c: remainder  # all proofs

# Loss weights (UW-SO adaptive)
loss:
  nav_weight_init: 1.0
  anchor_weight_init: 0.5
  progress_weight_init: 0.3
  critic_weight_init: 0.3
  critic_target: soft  # soft (MSE on normalized distance) | binary (BCE, NOT recommended per HTPS)

# Proof history
proof_history:
  enabled: true  # concatenate mean-pooled embeddings of closed goals to bridge input
  max_history: 10  # max previously closed goals to include

# Proof search
search:
  budget: 600           # max tactic attempts per theorem
  tactic_candidates: 5  # top-k tactics from navigate()
  premise_candidates: 16 # top-k premises from navigate()
  goal_selection: critic_progress  # use critic + progress for goal ordering
  hammer_delegation: true  # when AUTOMATION = -1, delegate to LeanHammer/Aesop
  hammer_budget: 30  # max seconds per hammer attempt

# PAB
pab:
  checkpoint_interval: 50
  experiment_id: wayfinder-v1
```

---

## 10. Society of Mind Architecture (v2)

The v2 architecture decomposes the monolithic v1 pipeline into six typed temporal slots. Each slot has bounded specification complexity and communicates with adjacent slots through typed data contracts, not shared weights. The composition gap γ ≈ 0 because slots share structured data (navigational coordinates, template IDs, proof sketches, temporal control state), not learned representations.

### 10.1 Slot 1: PERCEPTION (σ ≈ O(1))

**Function:** Deterministic encoding of goal state text into a fixed embedding.

**Implementation:** The existing Goal Encoder (`src/encoder.py`). Frozen `all-MiniLM-L6-v2` (384d, 617 goals/sec). Domain Gate (`src/domain_gate.py`) provides OOD detection. No changes from v1.

**Output contract:** `PerceptionOutput { embedding: Tensor[384], in_domain: bool }`

### 10.2 Slot 2: RECOGNITION (σ ≈ O(log k))

**Function:** Classify the proof into one of k narrative templates. This is the key regime conversion — raw proof structure (Regime B, |G_μ| ≈ 1) becomes template classification (Regime A, |G_μ| >> 1).

**Template taxonomy (initial, to be refined empirically):**

| Template ID | Pattern | Bank Signature | Example |
|-------------|---------|----------------|---------|
| DECIDE | Single automation tactic closes goal | AUTO=-1, DEPTH=-1 | `omega`, `simp`, `decide` |
| REWRITE_CHAIN | Sequence of rewrites reaching normal form | STRUCT=0, AUTO=0 | `rw [h1, h2]; ring` |
| INDUCT_THEN_CLOSE | Induction + base/step each closed by automation | STRUCT=+1, DEPTH=+1, AUTO=-1 | `induction n; simp; omega` |
| DECOMPOSE_AND_CONQUER | Split into independent subgoals via `have`/`suffices` | DECOMP=+1, DEPTH=+1 | `have h1 := ...; have h2 := ...; exact ...` |
| APPLY_CHAIN | Sequence of `apply`/`exact` targeting specific lemmas | STRUCT=0, AUTO=+1 | `apply foo; exact bar` |
| CASE_ANALYSIS | Split on data constructor or hypothesis | STRUCT=+1, DECOMP=+1 | `cases h; · ...; · ...` |
| CONTRAPOSITIVE | Negate goal, derive contradiction | STRUCT=+1, CONTEXT=+1 | `by_contra h; ...` |
| EPSILON_DELTA | Introduce witnesses, bound distances | DOMAIN=+1, DEPTH=+1 | Analysis-style ε-δ proofs |
| HAMMER_DELEGATE | Entirely delegated to ATP | AUTO=-1 | `aesop`, LeanHammer |

**Implementation:** A lightweight classifier over the GoalAnalyzer features (256d → k classes). Trained on proof corpus with template labels extracted from tactic sequences. Template assignment is deterministic post-classification: argmax over softmax logits.

**Output contract:** `RecognitionOutput { template_id: int, template_confidence: float, template_features: Tensor[64] }`

**Training data:** Extract template labels from existing `nav_training.jsonl` by clustering tactic sequences. Each proof's tactic sequence maps to its dominant template via the bank signature table above.

### 10.3 Slot 3: PLANNING (σ ≈ O(poly(n)))

**Function:** Given a template and goal state, produce a concrete proof sketch — an ordered sequence of abstract subgoals with key lemma targets and estimated depth per subgoal.

**Implementation options (evaluated in Phase 6):**

1. **Template instantiation (deterministic).** For simple templates (DECIDE, REWRITE_CHAIN), the sketch is the template itself — no learning needed. The template specifies the tactic sequence; the EXECUTION slot fills in arguments.

2. **Sketch predictor (learned).** For complex templates (DECOMPOSE_AND_CONQUER, INDUCT_THEN_CLOSE), train a small model to predict the subgoal sequence. Input: goal embedding + template features. Output: ordered list of (abstract_subgoal_type, estimated_difficulty, key_anchor_targets).

3. **Story-writing LLM (deferred, Phase 6.3).** For the hardest proofs, use a fine-tuned story-writing model that generates natural-language proof narratives, which are then parsed into structured sketches. This is the Relational-AI pattern: a narrative model produces the strategy, and specialist models execute it. Requires integration with a small LLM (e.g., Qwen 3.5, DeepSeek-Math 1.3B).

**Output contract:** `PlanningOutput { sketch: list[SubgoalSpec], total_estimated_depth: int }` where `SubgoalSpec { subgoal_type: str, anchor_targets: list[str], estimated_steps: int, bank_hints: dict[str, int] }`

### 10.4 Slot 4: EXECUTION (σ varies by specialist)

**Function:** Slot 4 is a two-level execution system:

1. **Theorem-level execution guidance** uses the old navigational machinery to decide lane, coarse family bias, and a bounded premise frontier for each subgoal.
2. **Residual local execution** solves the post-structural local step inside that frontier.
3. **Premise grounding** is invoked only for premise-sensitive tactic families.

The current experiments force this correction. The theorem-level network is good at temporal orchestration and frontier shaping; it is not the right granularity for direct post-structural premise-step retrieval.

**Specialist decomposition (guided by PAB stability):**

| Specialist | Banks | Regime | Rationale |
|-----------|-------|--------|-----------|
| **Navigator-A** (easy) | DOMAIN, CONTEXT | A | High symmetry, saturates early. Shared encoder features suffice. |
| **Navigator-B** (hard) | STRUCTURE, AUTOMATION, DEPTH, DECOMPOSITION | B | Low symmetry, needs dedicated capacity. Separate bridge and hidden layers. |

This is the initial decomposition. PAB stability measurement on each specialist determines whether further splitting is needed:
- If Navigator-B remains "chaotic" → split into Navigator-B1 (STRUCTURE, DECOMPOSITION) and Navigator-B2 (AUTOMATION, DEPTH)
- If Navigator-A remains "stable" → scope is correct, no further decomposition

Each theorem-level guidance specialist has its own bridge (eliminates the shared-bridge γ that made v1 chaotic), its own hidden layers, and its own output heads for its assigned banks.

**Fusion mechanism for shared heads.** Anchor logits, progress, and critic are produced by each specialist independently, then fused:

```python
def fuse_specialist_outputs(outputs_a, outputs_b):
    # Direction heads: concatenate (no fusion needed — each specialist owns its banks)
    directions = {**outputs_a.directions, **outputs_b.directions}

    # Anchor logits: max-pool across specialists
    # Rationale: if ANY specialist thinks an anchor is relevant, it should be considered.
    # Max preserves strong signals without averaging them away.
    anchor_logits = torch.max(outputs_a.anchor_logits, outputs_b.anchor_logits)

    # Progress: confidence-weighted average
    # Each specialist's critic confidence weights its progress estimate.
    total_conf = outputs_a.critic_confidence + outputs_b.critic_confidence + 1e-8
    progress = (outputs_a.progress * outputs_a.critic_confidence +
                outputs_b.progress * outputs_b.critic_confidence) / total_conf

    # Critic: min (conservative — if either specialist thinks the goal is hard, trust that)
    critic = min(outputs_a.critic, outputs_b.critic)

    return ExecutionGuidance(directions, anchor_logits, progress, critic)
```

**Fusion alternatives to ablate in Phase 6:**
- Max-pool anchors vs mean-pool vs learned attention
- Min critic vs mean critic vs max critic
- Confidence-weighted progress vs simple average

**Output contract (theorem-level guidance):** `ExecutionGuidance { directions: dict[str, int], anchor_logits: Tensor, progress: float, critic: float, lane_hint: str | None, family_hint: list[str] | None }`

**Residual local executor.** After one-shot structural normalization (`intro`, `intros`, `assumption`, `constructor`, `rfl`, `trivial`), the local layer does not behave like a flat always-on tactic menu. The measured runtime roles are:

- **Finisher**: `cosine_rw`
- **Scaffolder**: interleaved bootstrap
- **Helper**: `simp`
- **Executable selector**: `apply`
- **Next structured-action regime**: `refine_named`

The long-run interface can still be expressed as a family gate over a restricted vocabulary:

- `rw`
- `simp`
- `exact`
- `refine`
- `apply`
- `other`

But the concrete design consequence is now sharper: each family needs its own executable contract.
Search should only explore candidates consistent with those contracts. This uses theorem-level
guidance as context, but it is trained on the normalized residual goal, not on raw theorem states.

**Candidate grounding.** Candidate grounding is family-sensitive:

- `rw` is largely solved by scoped cosine retrieval plus deterministic lowering.
- `simp` is mostly local and often does not need a specific premise embedding.
- `apply` depends on whether a candidate is actually executable against the current goal; name
  relevance alone is insufficient.
- `refine_named` depends on structured action / skeleton choice over a short scoped candidate set.
- `refine_anon` is a distinct harder regime and should not be forced into the same selector.

The concrete architecture is therefore:

1. theorem-level guidance → coarse frontier;
2. residual family prediction on the normalized local goal;
3. family-conditioned candidate grounding inside the frontier;
4. family-specific executable selection before lowering.

This is the intended decomposition, not a fallback. The theorem-level system handles temporal orchestration; the residual executor handles step-level local choice.

**Constrained action synthesis.** The family predictor should not emit raw Lean text. It should emit a family-specific intermediate representation that can be lowered deterministically:

```python
@dataclass
class TermExpr:
    kind: str              # var | const | app | proj | hole | chain | ctor | lambda
    head: str | None
    args: list["TermExpr"]
    field: str | None = None

@dataclass
class RewriteAtom:
    direction: str         # forward | backward
    expr: TermExpr

@dataclass
class ActionIR:
    family: str            # rw | simp | exact | apply | refine
    term: TermExpr | None = None
    rewrites: list[RewriteAtom] | None = None
    simp_lemmas: list[TermExpr] | None = None
    using_term: TermExpr | None = None
    only: bool = False
```

The decoder's job is therefore:

1. choose a tactic family;
2. choose symbols and local applications from the frontier and context;
3. emit `ActionIR`;
4. let a deterministic lowering layer format Lean syntax.

For `apply`-like families, that lowering step now has an additional verified precondition:
the chosen candidate must first survive executable selection against the current goal. In other
words, the learned selector sits between scoped retrieval and ActionIR lowering; it is not a
second theorem-name reranker. The selector is trained on Lean-generated feedback such as:
- `accepted_with_goals`
- `closed`
- `unification_mismatch`
- `typeclass_missing`
- goal-start failures

This is the first concrete case where the design's "small model over a structured executable
interface" claim is already validated in live Lean.

Current residual-tactic analysis shows that `82%` of examples are parsable into family + structured arguments, while only `32%` are covered by the first simple template subset. The implication is to grow the IR and canonicalizer, not to revert to raw tactic-string generation.

### 10.5 Slot 5: VERIFICATION (σ ≈ O(1))

**Function:** Verify tactic applications and learn from failures. Unchanged from v1's three-lane architecture.

**New addition — Censor network.** A small classifier that learns to predict tactic failure *before* Lean kernel verification, trained on accumulated (goal_state, tactic, result) triples from search. This inverts the verification oracle: instead of only learning what works, actively learn what does NOT work. The censor prunes the candidate set before expensive Lean kernel calls.

**Output contract:** `VerificationOutput { success: bool, new_goals: list[GoalState], feedback: LeanFeedback, failure_reason: Optional[str] }`

`LeanFeedback` is now part of the design contract, not just logging. The verifier should preserve:
- stage (`goal_creation`, `tactic_parse`, `elaboration`, `tactic_exec`)
- category (`unification_mismatch`, `typeclass_missing`, `unknown_identifier`, `accepted_with_goals`, `closed`, `other`)
- raw compiler/elaborator messages when available

This feedback serves two roles:
1. runtime diagnosis / repair;
2. generated supervision for executable selectors.

### 10.6 Arbiter (Orchestrator)

The Arbiter manages the proof search loop at the SoM level. It has two distinct responsibilities:

1. **Template-level orchestration** across the six SoM slots.
2. **Online temporal control** over subgoal ordering, lane order, escalation, and replanning.

The temporal controller is a first-class Arbiter component, not an extension of RECOGNITION. It models proof-solving dynamics conditioned on prior progress.

**Core temporal contracts**:

```python
@dataclass
class TemporalState:
    theorem_id: str
    active_template: str | None
    open_goals: list[str]
    closed_goals: list[str]
    current_goal_id: str | None
    prior_lanes: list[str]
    prior_families: list[str]
    successful_tactics: list[str]
    failed_tactics: list[str]
    phase: str
    escalation_level: int
    budget_remaining: int
    total_attempts: int

@dataclass
class OrchestrationDecision:
    next_goal_id: str
    phase: str
    lane_order: list[str]
    family_prior: list[str]
    escalation_level: int
    budget_slice: int
    replan: bool
```

**Current implementation status**:
- `src/temporal_controller.py` implements a tested rule-based `TemporalController` v0.
- The benchmarked `search()` path in `src/proof_search.py` now supports both `shadow` and `active` temporal modes and records `temporal_trace`.
- `src/arbiter.py` / `som_search()` still do not consume the controller.
- `budget_slice` and `replan` are tracked in the controller contract, but hammer still runs as an unconditional pre-check and full budget enforcement is not yet operative.

The intended Arbiter loop is:

1. Receives initial goal from PERCEPTION
2. Routes to RECOGNITION for template classification
3. Routes to PLANNING for proof sketch generation
4. Builds or updates `TemporalState`
5. Uses the temporal controller to choose `next_goal_id`, `phase`, `lane_order`, `family_prior`, and `escalation_level`
6. Routes to the appropriate theorem-level EXECUTION specialist
7. Runs one-shot structural normalization, then residual family prediction and family-conditioned premise grounding
8. Routes bounded action candidates to VERIFICATION
9. On verification failure: updates censor and `TemporalState`, possibly escalating or replanning
10. On verification success: advances to next subgoal, updates proof history and closed-goal context
11. On sketch failure (all subgoals attempted, proof incomplete): re-routes to RECOGNITION/PLANNING with `replan=True`

**Goal selection today** is mixed-mode. In the benchmarked `search()` path, shadow mode logs decisions and active mode consumes `next_goal_id` and `lane_order`. In the full SoM `arbiter.py` path, critic/progress-based selection remains the baseline until the controller is wired there as well.

### 10.6a Temporal Controller (Arbiter Implementation)

The Arbiter's core is a `TemporalController` that produces `OrchestrationDecision` from `TemporalState` each search step.

**Contracts** (from `src/temporal_controller.py`):

```python
@dataclass
class TemporalState:
    theorem_id: str
    active_template: str | None
    open_goals: list[str]
    closed_goals: list[str]
    current_goal_id: str | None
    prior_lanes: list[str]
    prior_families: list[str]
    successful_tactics: list[str]
    failed_tactics: list[str]
    phase: str  # structural_setup | local_close | automation_close | repair_or_replan
    escalation_level: int
    budget_remaining: int
    goal_attempt_counts: dict[str, int]
    goal_lane_failures: dict[str, set[str]]

@dataclass
class OrchestrationDecision:
    next_goal_id: str
    phase: str
    lane_order: list[str]
    family_prior: list[str]
    escalation_level: int
    budget_slice: int
    replan: bool
```

**Phases** (4-state FSM):
1. `structural_setup`: intros, constructors, trivial normalization
2. `local_close`: residual executor — exact/apply/refine/rw/simp family execution
3. `automation_close`: hammer/aesop/decide/omega when state is favorable
4. `repair_or_replan`: lane exhausted, residual stagnant, failures clustered

**Implementation status (2026-03-16):**
- Rule-based v0 controller: `src/temporal_controller.py` (18 unit tests)
- Shadow mode: logs decisions without changing behavior (TC0-log)
- Active mode: consumes `next_goal_id` and `lane_order` from decisions
- `StepOutcome` provides clean goal attribution for TC state updates
- Hammer fires as pre-check independent of TC lane ordering
- `budget_slice` and `replan` are not yet full control surfaces in the operative search loop

**Key empirical finding:** The current theorem-level gains are still dominated by escalation=0
structural/bootstrap behavior plus `cosine_rw` as the active finisher. The TC's value depends on
the quality of the residual specialists in `local_close`. `apply` executable selection is now the
first specialist beyond rewrite collapse to show a live execution-level gain, but theorem-search
deployment is still the next benchmark rather than an already-proven theorem-level win.

### 10.6b ActionIR Decoder (Constrained Tactic Construction)

The ActionIR decoder sits inside the EXECUTION slot. It takes the residual executor's family prediction + local vocabulary and emits typed `ActionIR` nodes that deterministic lowering compiles to Lean syntax.

**Modules:**
- `src/tactic_ir.py` — `TermExpr`, `RewriteAtom`, `ActionIR` typed AST with `lower()` method
- `src/tactic_canonicalizer.py` — parse Lean tactic strings into canonical ActionIR
- `src/tactic_compiler.py` — compile family + premises into tactic candidates (template-based v0)

**Architecture (GSE-aligned two-stage):**

Stage 1 — Shape prediction:
- rw: rewrite count + direction flags per atom
- exact/apply/refine: coarse term skeleton (symbol | app | projection | chain | hole | ctor)
- simp: mode (bare | list | only | using)

Stage 2 — Leaf filling:
- Pointer attention over dynamic local vocabulary
- Hard grammar masks per family
- Hard type masks where Lean type info is available

**Dynamic local vocabulary per goal:**
- Local hypotheses (from goal state text)
- Accessible premises (from proof network, scoped to theorem)
- Fixed combinators: `.trans`, `.symm`, `_`, `?_`, `.1`, `.2`, `←`
- Each entry: source_kind, type_features, family_eligibility, embedding

**Empirical status (2026-03-16):**
- ActionIR parse rate: 100% of residual tactics with arguments
- Round-trip fidelity: 70% exact (IR → lower → identical string)
- Parsable output space: 82% of residual tactics
- Not yet trained — architecture and IR are specified, decoder model is next

### 10.6c Source-Context Compiler (ContextIR)

ActionIR is only half of the deterministic compiler. The local-execution harness also has to
reconstruct the **theorem-site source context** in which tactics are meaningful. Mathlib source
relies heavily on scoped surface constructs that change executable meaning without changing the
theorem statement itself:

- `open` / `open scoped`
- `variable(s)` / `universe(s)`
- `local notation`, `scoped[...]` notation, and notation declarations
- `attribute [local instance]`, `attribute [local simp]`
- `include` / `omit`
- inline next-declaration forms such as `open Classical in` or `variable (R) in`

These should be treated as a compiled DSL, not as header text.

**Contract:**

```python
@dataclass
class ContextDirective:
    kind: str
    text: str
    line_no: int
    inline_only: bool
    renderable: bool

@dataclass
class ContextFrame:
    kind: str
    name: str
    start_line: int
    directives: list[ContextDirective]

@dataclass
class ContextIR:
    top_level_directives: list[ContextDirective]
    scope_stack: list[ContextFrame]
    unsupported: list[ContextDirective]
```

**Compiler stages:**
1. Parse theorem-site lexical context into `ContextIR`
2. Render wrapper prefix/suffix from the supported subset
3. Keep unsupported forms explicit in logs and validation reports
4. Reuse the same `ContextIR` in:
   - Tier B declaration-faithful wrapper generation
   - Tier C step>0 replay normalization
   - future family-specific scopers (`simp`, `apply`, `exact`)

**Current implementation status (2026-03-17):**
- `src/lean_context_ir.py` provides the parser/IR and conservative renderer
- `src/lean_interface.py` now uses the `ContextIR`-backed extractor for Tier B wrapper prefix/suffix
- validation scripts:
  - `scripts/context_ir_census.py`
  - `scripts/context_ir_benchmark_audit.py`

**Design consequence:** future executor work is now explicitly a two-compiler stack:
- **source-context compiler** (`ContextIR`) for theorem-site environment reconstruction
- **action compiler** (`ActionIR`) for family-specific local tactic lowering

**Updated execution consequence (2026-03-17):**
- `rw0` and `rw1` are now operationally served by scoped cosine retrieval plus Lean verification.
- `ContextIR` remains the coverage/faithfulness track for replay and future families (`simp`, `apply`), not the reason to delay extending the runtime rewrite lane from `rw0` to `rw1`.
- The rewrite family has since collapsed further: on current started step-0 slices, `rw2` and `rw3` are also served well enough by the same cheap sequential cosine executor. The next learned targets are therefore above rewrite-family local execution: residual-conditioned orchestration, controller-visible move typing, and executable action selection over cosine shortlists.

### 10.6d Motivated Move Layer (`SubtaskIR` + Trigger Profiles)

The Human-Oriented ATP / motivated-proof line suggests a useful refinement for Wayfinder: local
execution should not be represented only as a tactic string or even only as `ActionIR`. It should
also carry an explicit answer to two controller-facing questions:

1. **Why is this move admissible here?** (`TriggerProfileIR`)
2. **What local subtask is this move trying to accomplish?** (`SubtaskIR`)

This layer is intentionally lossy and sits **above** `ActionIR`. It does not participate in Lean
lowering; it exists to support planning, move typing, auditability, and proof-schema mining.

**Contract:**

```python
@dataclass
class GoalShapeIR:
    goal_count: int
    target: str
    target_head: str
    local_names: list[str]
    has_forall: bool
    has_implication: bool
    has_exists: bool
    has_equality: bool
    has_iff: bool

@dataclass
class TriggerFeatureIR:
    kind: str
    value: str
    source: str

@dataclass
class TriggerProfileIR:
    family: str
    primary_premise: str
    features: list[TriggerFeatureIR]

@dataclass
class SubtaskIR:
    family: str
    kind: str
    summary: str
    primary_premise: str
    local_inputs: list[str]
    expected_effect: str
```

**Examples:**
- `rw [h]` on an equality goal:
  - `SubtaskIR.kind = normalize_target_forward`
  - triggers include `target_shape=equality`, `rewrite_count=1`, `direction_prior=forward`
- `rw [foo h]`:
  - `SubtaskIR.kind = specialize_rewrite`
  - triggers include `argument_mode=applied`, `local_inputs=h`
- `apply foo`:
  - `SubtaskIR.kind = reduce_goal_by_lemma`
- `exact h`:
  - `SubtaskIR.kind = close_with_term`

**Why this layer exists:**
- `ActionIR` is the **execution compiler** target
- `SubtaskIR` is the **planning / move-type** target
- the controller should reason over subtasks such as:
  - normalize target
  - continue rewrite chain
  - reduce goal by lemma
  - close with term
  - simplify with context
  rather than over raw tactic strings

**Placement rule (alignment correction):**
- `goal_target_head` and `trigger_signature` are **descriptive** local-state labels and may be
  used as auxiliary supervision on routing-oriented models such as the navigator.
- `subtask_kind` is a **planning/controller** label. It should feed RECOGNITION / PLANNING /
  temporal-control modules, not be treated as a required target for the bank-direction decoder.
- Therefore, a collapsed `subtask_kind` head on the navigator is not evidence against
  `SubtaskIR`; it is evidence that the label belongs in a different slot.

**Pipeline integration (implemented 2026-03-18):**
- `src/subtask_ir.py`
  - derives `GoalShapeIR`, `TriggerProfileIR`, and `SubtaskIR` from
    `(goal_state_before, family, tactic_text, canonical_action_ir, annotated_premise)`
- `scripts/build_canonical_local_data.py`
  - now writes `goal_shape_ir`, `trigger_profile_ir`, and `subtask_ir` into canonical datasets
- `scripts/build_subtask_training_data.py`
  - projects canonical data to a compact controller-facing SubtaskIR dataset
- `scripts/validate_subtask_ir.py`
  - validates family/subtask/trigger invariants over a dataset
- `scripts/mine_move_schemas.py`
  - aggregates successful steps into reusable `(family, subtask_kind, trigger_signature)` schemas
- `scripts/build_move_inventory.py`
  - coarsens SubtaskIR data into a compact planner inventory, especially for `apply` / `simp`
- `src/move_supervision.py`
  - builds compact auxiliary vocabularies over `subtask_kind`, `goal_target_head`, and
    `trigger_signature`
  - supports head filtering so each slot only trains on conceptually appropriate targets
- `scripts/train_navigator.py`
  - uses only **descriptive** move vocabularies (`goal_target_head`, `trigger_signature`) as
    auxiliary supervision heads on the navigator hidden state
  - persists `move_supervision_spec.json` in the run directory
- `scripts/train_template_classifier.py`
  - uses theorem-level `template_move_profile` summaries as auxiliary supervision on the
    recognition hidden state (`subtask_kind`, `goal_target_head`, `trigger_signature`)
  - persists `template_move_supervision_spec.json` in the run directory
- `scripts/run_enhanced_controller_pipeline.sh`
  - rebuilds the controller-facing data products end-to-end
- `scripts/run_main_experiment.sh`
  - canonical end-to-end experiment driver for the enhanced controller stack

**Design consequence:** Wayfinder now has a three-layer local stack:
- **source-context compiler** (`ContextIR`)
- **action compiler** (`ActionIR`)
- **motivated move layer** (`SubtaskIR` + trigger profiles) for controller-visible move typing

**Runtime boundary:** these additions do not change the current execution DSL or Lean-lowering
contracts. The benchmarked search path still executes through:
- `ContextIR` for theorem-site wrapper reconstruction
- `ActionIR` / rewrite-family executors for deterministic local execution
- `proof_search.py` consuming `NavOutput` for routing

The motivated-move layer is currently a data/training/planning contract, not a runtime executor.

**Stable theorem identity:** downstream theorem-level aggregation now keys on
`theorem_key = file_path::theorem_id`, not on the display name alone. This prevents auxiliary
theorems such as `aux` from collapsing into the same template bucket across files.

This layer is especially relevant for the future non-rewrite families. Rewrite-family experiments
show that raw local execution often collapses to cheap geometry + verification. The remaining
difficult question is therefore not "which string do we emit?" but "which local transformation
should we attempt next?" `SubtaskIR` is the contract for that question.

**Post-experiment role split (2026-03-19):**
- **Finisher lanes**: theorem-winning local executors
  - currently: `cosine_rw`
- **Scaffolder lanes**: structural/bootstrap policies that simplify the goal and expose the next residual
  - currently: interleaved bootstrap
- **Helper / transformer lanes**: locally useful but not yet theorem-winning on their own
  - currently: `simp`
- **Executable specialist lanes**: locally real selectors over scoped candidates; not automatically global lanes
  - currently: `apply`
- **Unresolved structured-action regimes**: families whose bottleneck is still skeleton choice rather than executable scoring
  - currently: `refine_named`, `refine_anon`

This matters because the next improvement is no longer "add another always-on lane."

Current family evidence now splits cleanly:
- `apply`: ranking the annotated premise is not enough; the real bottleneck is whether a candidate can actually unify/elaborate against the current goal. A Lean-feedback-trained selector now improves live `LeanAccepted`, which validates executable candidate selection as the correct interface.
- `refine_named`: better premise ranking barely changes `LeanAccepted`, so the real bottleneck is structured action / skeleton choice
- `refine_anon`: typed features are neutral-to-harmful, so it is a different computational regime

**Design implication:** the architecture should prefer:
- deterministic scoping to produce a small auditable candidate set
- executable action filtering and scoring over that shortlist
- residual-conditioned orchestration rather than globally enabled specialist lanes
- provenance-aware training sets so specialist models do not conflate canonical step-0 data with
  post-bootstrap or mid-search residual regimes

That means the learned object is no longer "the best theorem name."
It is the best **structured executable action boundary**:
- for `apply`: unification-aware / elaboration-aware candidate selection
- for `refine_named`: skeleton / structured action selection
- for `refine_anon`: not yet collapsed into the same selector regime

This preserves the central factorization:
- broad easy local mass collapses into deterministic compilers + verifier-backed execution
- the remaining hard residue becomes small-candidate executable-action selection rather than open-ended tactic generation

**Immediate design-order consequence:** do not jump from the first `apply` selector win straight to
full SoM specialist training. The next order is:
1. expand provenance-aware executable datasets (`canonical_step0`, `canonical_midstep`,
   `post_bootstrap_residual`, `mid_search_residual`);
2. train source-aware executable selectors;
3. integrate those selectors into theorem search;
4. then extend the same regime to `refine_named`;
5. only then train the full SoM over multiple validated specialist contracts.

### 10.7 Module Map Update (v2)

New modules for v2 (implemented or planned within Phase 6):

```
src/
├── ... (all existing v1 modules retained)
├── story_templates.py          Template taxonomy and extraction (Slot 2)
├── template_classifier.py      Recognition classifier (Slot 2)
├── sketch_predictor.py         Proof sketch generator (Slot 3)
├── temporal_controller.py      Temporal controller (Arbiter core, 4-phase FSM)
├── specialist_navigator.py     Bank-cluster specialist wrapper (Slot 4)
├── residual_executor.py        Residual tactic family classifier (6-class)
├── lean_context_ir.py          Source-context parser/compiler for theorem-site wrappers
├── action_ir.py                Family-specific local tactic IR + typed terms
├── tactic_ir.py                ActionIR typed AST (TermExpr, RewriteAtom, ActionIR)
├── tactic_canonicalizer.py     Parse Lean tactics into canonical ActionIR
├── subtask_ir.py               Derive GoalShapeIR / TriggerProfileIR / SubtaskIR
├── tactic_compiler.py          Template-based tactic compilation (v0)
├── tactic_lowering.py          Deterministic ActionIR -> Lean tactic lowering
├── censor.py                   Failure prediction network (Slot 5)
└── arbiter.py                  SoM orchestrator

scripts/
├── ... (all existing scripts retained)
├── extract_templates.py        Extract template labels from proof corpus
├── context_ir_census.py        Whole-Mathlib context-feature census
├── context_ir_benchmark_audit.py  Benchmark-targeted ContextIR needs audit
├── build_subtask_training_data.py  Project canonical data to controller-facing SubtaskIR dataset
├── validate_subtask_ir.py      Validate TriggerProfileIR/SubtaskIR annotations
├── mine_move_schemas.py        Mine reusable move schemas from successful proof steps
├── train_specialist.py         Train individual specialists with PAB monitoring
└── train_template_classifier.py  Train the RECOGNITION slot
```

### 10.8 PAB-Guided Decomposition Protocol

The decomposition protocol is iterative and data-driven:

```
1. Start with candidate specialist scope (e.g., all 6 banks)
2. Train specialist for N steps (e.g., 2000)
3. Measure PAB stability_regime
4. If "stable" (stability_mean < 0.15):
     → Scope is correct. Lock specialist, proceed.
5. If "transitional" (0.15 ≤ stability_mean < 0.30):
     → Scope is borderline. Try longer training (2x steps).
     → If still transitional after 2x: decompose.
6. If "chaotic" (stability_mean ≥ 0.30):
     → Scope too broad. Decompose into sub-specialists.
     → Split banks by difficulty tier (Regime A vs Regime B).
     → Recurse on each sub-specialist.
7. Verify: total system σ = Σ specialist σ_i (additive, no γ)
   by running end-to-end eval and checking that combined performance
   ≥ monolithic performance.
```

The free energy bound (Theorem 3.10 of the specification complexity paper) guarantees that each decomposition step that reduces γ also reduces total specification complexity. Over-decomposition is self-correcting: a specialist with scope too narrow will have σ ≈ O(1) and trivially reach "stable," contributing its small additive cost without harming the total.

---

## 11. The Name

**Wayfinder** refers to the Polynesian navigators who crossed the Pacific Ocean — the largest navigable space on Earth — without instruments. They read structured patterns: star positions (fixed reference points, like bank zero states), ocean swells (propagation patterns, like spreading activation), and bird flight paths (semantic signals, like anchors). They navigated by understanding the *structure* of the space, not by memorizing routes.

Wayfinder navigates proof space the same way: structured coordinates (banks), semantic patterns (anchors), and propagation signals (spreading activation). The neural network is the navigator who reads these signals. The symbolic network is the ocean — vast, structured, and navigable by those who understand its patterns.

---

## 12. v3 Runtime Architecture: Boundary Learning and Energy Refinement

*v3 is a parallel orchestration path added alongside v1 (monolithic) and v2 (SoM). It does NOT modify v1 or v2 code paths. All three runtimes produce the same top-level metrics schema for A/B/C/D comparison.*

### 12.1 Runtime Modes

Benchmark runs require explicit mode selection: `--mode v1|v2|v3`. Each mode owns its orchestration logic; shared infrastructure (encoder, proof network, Lean interface) is reused.

| Mode | Orchestrator | Scoring | Pruning | Sketch |
|------|-------------|---------|---------|--------|
| **v1** | `proof_search.py` (sequential best-first) | `confidence_weighted` | None | None (tactic-by-tactic) |
| **v2** | `arbiter.py` (SoM 5-slot pipeline) | Specialist fusion (§10.4) | None | Template-based (sketch_predictor) |
| **v3** | `v3_runtime.py` (new) | OTP bank-IDF + ConstraintReport | Censor (asymmetric) | Template-based + optional energy refinement |

### 12.2 Shared Interfaces

All v3 data flows through these dataclasses, defined in `src/v3_contracts.py`:

```python
@dataclass
class GoalContext:
    theorem_id: str
    goal_text: str
    proof_history: list[str]          # closed goals so far
    accessible_premises: list[str]     # import-accessible premise IDs
    source_split: str                  # "train" | "eval"

@dataclass
class ActionCandidate:
    tactic: str
    premises: list[str]
    provenance: str                    # "navigate" | "spread" | "hammer"
    navigational_scores: dict[str, float]  # per-bank scores
    template_provenance: str | None    # template ID if from sketch

@dataclass
class NegativeExample:
    goal_state: str
    theorem_id: str
    step_index: int
    failed_tactic: str
    failure_reason: str
    failure_category: str              # "semantic" | "infra" | "weak_negative"
    source: str                        # "sorry_hole" | "perturbation" | "suggestion_trace" | "unchosen_weak"
    proof_history: list[str]
    paired_positive_tactic: str | None
    paired_positive_premises: list[str]
    bank_directions: dict[str, int]
    otp_dimensionality: int            # 6 - count(zeros)

@dataclass
class ConstraintReport:
    bank_scores: dict[str, float]      # per-bank alignment
    critic_distance: float             # estimated steps remaining
    censor_score: float                # P(failure)
    anchor_alignment: float            # IDF-weighted Jaccard
    total_score: float                 # composite (v3A: weighted sum)
    energy: float | None               # v3B only: differentiable energy

@dataclass
class SketchProposal:
    template_id: str
    proposed_steps: list[ActionCandidate]
    latent_form: 'torch.Tensor | None'  # v3B only: continuous representation
    total_constraint_score: float

@dataclass
class SearchTrace:
    theorem_id: str
    mode: str                          # "v1" | "v2" | "v3"
    steps: list[dict]                  # per-step decisions
    pruning_decisions: list[dict]      # censor prune log (v3)
    lean_calls: int
    result: str                        # "proved" | "failed" | "timeout"
    constraint_reports: list[ConstraintReport]  # v3 only
```

### 12.3 v3 Subsystem Boundaries

The v3 runtime is organized by function, not phase:

```
src/
├── v3_runtime.py          v3 orchestrator (imports below)
├── v3_contracts.py         Shared interfaces above
├── v3_scoring.py           OTP bank-IDF scoring, ConstraintReport composition
├── lens_guidance.py        GuidancePacket / ResolutionDecision contracts
├── lens_models.py          Rule-based or learned lens specialists over collapsed frontiers
├── coherence_engine.py     Typed-vote fusion, abstention handling, action routing
├── censor.py               Standalone censor (existing from Phase 6.0, retrained with asymmetric loss)
├── energy.py               [v3B] Energy function, Gumbel-softmax, refinement loop
│
scripts/
├── collect_negatives.py    First-class data pipeline (3 collectors + weak negatives)
├── train_censor.py         Standalone censor training (independent of navigator)
└── run_benchmark.py        --mode v1|v2|v3, SearchTrace output
```

### 12.4 v3A Pipeline (Boundary Learning)

```
GoalContext
  → Encoder (shared, frozen)
  → OTP-scored navigation (bank-IDF weights from v3_scoring.py)
  → Deterministic collapse (landmark selection, freeze/residual, typed expansion)
  → Guidance modulation over collapsed frontier (optional, modulate-only by default)
  → Censor pruning (asymmetric, safety-net: never prune ALL)
  → Template classification (shared with v2)
  → Discrete sketch scoring via ConstraintReport
  → Lean verification (Lane A)
  → SearchTrace output
```

### 12.4a v3A Guidance Layer (Wave 1.5)

This layer sits between deterministic retrieval collapse and later v3B refinement. Its purpose is to spend learned or symbolic specialist capacity only on the ambiguous remainder after the proof universe has already been reduced to a small auditable frontier.

**Core contracts**:
- **`GuidancePacket`**: collapsed frontier, committed landmarks, residual anchor/category mass, conflict clusters, phase signal, negative evidence, candidate summaries.
- **`LensVote`**: typed specialist output over a candidate or branch: `support`, `oppose`, or `abstain`, with confidence and provenance.
- **`LensCommitteeState`**: fused view of support/oppose/abstain patterns, informativeness, and conflict structure across lenses.
- **`ResolutionDecision`**: committee action (`modulate`, `trust_lens`, `bifurcate`, `expand_more`, `noop`) plus bounded score adjustments or routing decisions.

**Default invariants**:
- Deterministic collapse remains the primary retrieval mechanism.
- Guidance modulates ranking inside the collapsed frontier by default.
- `replace` mode is ablation-only until it beats modulation and no-lens retrieval on paired evaluation.
- Abstention is first-class and logged; it is not collapsed into low confidence.
- Lean verification remains the final authority.

This is a v3A component, not a v3B luxury. It formalizes the bridge between symbolic navigation and later learned refinement/distillation without requiring energy-based optimization.

### 12.5 v3B Pipeline Extension (Energy Refinement — gated on v3A)

```
[v3A pipeline through template classification]
  → Continuous sketch in latent space (Gumbel-softmax relaxation)
  → Energy minimization loop (energy.py, ~20 gradient steps)
  → Temperature annealing (τ: 1.0 → 0.1)
  → Snap to discrete ternary
  → Lean verification (Lane A)
  → SearchTrace output (includes energy trajectory)
```

### 12.6 Config Separation

v3 config lives in its own namespace to avoid overloading v1/v2 settings:

```yaml
# In configs/wayfinder_v3.yaml (extends wayfinder_v2.yaml)
runtime:
  mode: v3                    # v1 | v2 | v3

otp_scoring:
  bank_idf_enabled: true
  zero_sparsity_curriculum: true
  curriculum_phases: [3, 5, 6]  # max active banks per phase

negative_data:
  path: data/nav_negative.jsonl
  semantic_weight: 1.0
  weak_negative_weight: 0.1
  infra_exclude: true

censor:
  asymmetric_loss: true
  w_neg: 2.0                 # missed suppression penalty
  w_pos: 1.0                 # false suppression penalty
  operating_threshold: 0.5
  safety_net_k: 3            # minimum candidates after pruning

contrastive:
  lambda: 0.05
  margin_start: 0.1
  margin_end: 0.3
  margin_anneal_steps: 5000
  hard_negative_ratio: 0.5

guidance:
  enabled: false
  mode: modulate              # modulate | replace (ablation only)
  specialist_source: rule_based  # rule_based | learned
  min_informativeness: 0.15
  max_downweight: 0.10

# v3B (disabled by default until v3A demonstrates value)
energy_refinement:
  enabled: false
  refine_steps: 20
  refine_lr: 0.01
  tau_start: 1.0
  tau_end: 0.1
  energy_threshold: 0.1
  weights:
    bank: 1.0
    critic: 0.5
    censor: 2.0
    anchor: 0.3
```

### 12.7 Invariants (v3-specific, supplements §Invariants in PLAN)

1. **v1 and v2 code paths are frozen.** v3 must not modify `proof_search.py` or `arbiter.py`. Shared infrastructure changes require v1/v2 regression tests.
2. **SearchTrace is mandatory.** Every v3 benchmark run emits per-theorem SearchTrace for auditability.
3. **Censor safety net is non-negotiable.** `safety_net_k ≥ 1`. The censor may never prune all candidates.
4. **Negative label hygiene.** `failure_category: "infra"` examples never enter training loss. Enforced at data loading, not filtering.
5. **Energy refinement defaults to off.** `energy_refinement.enabled: false` in shipping config. Switched on only after v3A gate passes.
