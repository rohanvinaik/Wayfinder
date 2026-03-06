# Wayfinder: Technical Design Reference

**Version:** 1.1
**Date:** March 6, 2026
**Corresponding documents:** `WAYFINDER_RESEARCH.md` (theory), `WAYFINDER_PLAN.md` (operational plan), `EXPERIMENT_RESULTS.md` (results ledger)

---

## 1. Design Thesis

**Navigation, not prediction.** The central design decision in Wayfinder is that proof search should be *spatial navigation through structured mathematical space* rather than *sequential token prediction*. This has a concrete architectural consequence: the neural network runs once per proof state to produce navigational coordinates, and all subsequent operations — premise retrieval, tactic resolution, search prioritization — are deterministic symbolic operations on a precomputed semantic network.

This design is directly adapted from ModelAtlas (Vinaik, 2025), which demonstrated that structured navigation over signed semantic coordinates outperforms both flat database queries and dense embedding retrieval for finding ML models on HuggingFace. We apply the same paradigm to mathematical entities: lemmas, tactics, and proof states are positioned in a structured coordinate system; retrieval is multiplicative bank alignment × IDF-weighted anchor relevance × seed similarity.

The thesis is falsifiable. If dense retrieval consistently outperforms structured navigation for premise selection (Phase 2.2 of the plan), the navigational paradigm is wrong for this domain. If tactic classification outperforms navigational coordinates (Phase 4.4 ablation), the ternary decoder design is wrong. Both negative results would be informative.

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

All modules are implemented. The codebase consists of 28 source files and 8 scripts.

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
├── build_nav_training_data.py  Convert proof traces to nav labels
├── anchor_gap_analysis.py      Iterative anchor gap analysis
├── tactic_maps.py              Tactic-to-direction mapping tables
├── train_navigator.py          Training script with curriculum
├── eval_retrieval.py           Nav vs dense retrieval comparison
├── eval_spreading.py           Spreading activation evaluation
└── run_benchmark.py            MiniF2F + Mathlib benchmark runner

configs/
└── wayfinder.yaml              Full Wayfinder experiment configuration

data/  (generated/downloaded at runtime)
├── proof_network.db            SQLite semantic network
├── nav_training.jsonl             Navigational training data
├── nav_eval.jsonl              Navigational eval data (frozen)
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

## 10. The Name

**Wayfinder** refers to the Polynesian navigators who crossed the Pacific Ocean — the largest navigable space on Earth — without instruments. They read structured patterns: star positions (fixed reference points, like bank zero states), ocean swells (propagation patterns, like spreading activation), and bird flight paths (semantic signals, like anchors). They navigated by understanding the *structure* of the space, not by memorizing routes.

Wayfinder navigates proof space the same way: structured coordinates (banks), semantic patterns (anchors), and propagation signals (spreading activation). The neural network is the navigator who reads these signals. The symbolic network is the ocean — vast, structured, and navigable by those who understand its patterns.
