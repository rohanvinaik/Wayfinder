# Wayfinder: Operational Research Plan

**Version:** 1.1
**Date:** March 6, 2026
**Corresponding documents:** `WAYFINDER_RESEARCH.md` (theory), `WAYFINDER_DESIGN.md` (engineering), `EXPERIMENT_RESULTS.md` (results ledger)
**Status:** Implementation complete. Ready for data pipeline and experiments.

---

## Overview

This document specifies the sequence of work, tooling, compute decisions, and stop/go criteria for running the Wayfinder experiments. All source code and scripts are implemented (27 src files, 8 scripts, 1 config). This plan covers data pipeline execution and experimental evaluation.

This plan produces three categories of results:

- **Stream 1 (Navigation)**: Does structured semantic navigation — bank positions, IDF-weighted anchors, spreading activation — outperform dense embedding retrieval for premise selection and proof search?

- **Stream 2 (Architecture)**: Does a ternary navigational decoder that produces directional coordinates, resolved through a semantic network, outperform tactic-token classification?

- **Stream 3 (Process Evaluation)**: Does PAB's trajectory evaluation reveal information about the navigational learning process that endpoint metrics cannot?

Every phase states what it validates in each stream. The experimental pipeline is designed so that a single set of experiments tests all three claims. Results are recorded in `docs/EXPERIMENT_RESULTS.md`.

**Time horizon:** 6-10 weeks for Phase 0-4. Phase 5+ contingent on results.

**Compute budget:** Apple Silicon (M-series Mac) as primary target. The architecture is designed for efficiency: small learnable navigational components (~400K trainable params) and symbolic search (SQLite). Encoder selection (Phase 0.6) will determine whether the encoder is frozen, fine-tuned, or aggressively pruned from a larger model — this is the main variable in compute requirements. A pruned math-native model (70-350M effective params) may require brief cloud GPU time for the pruning/fine-tuning step but runs locally thereafter.

---

## Hardware & Distribution Strategy

Adapted from the proven ModelAtlas pipeline pattern: standalone workers, JSONL interchange, `scp`-deployable, `--resume` support.

| Machine | Role | Specs | Wayfinder Use |
|---------|------|-------|---------------|
| **MLX laptop** | Primary dev + training | Apple Silicon M-series, MLX/Metal | Encoder eval, training, proof search, benchmarks |
| **macpro** | Persistent CPU worker | Xeon 6C 3.5GHz, 12GB ECC, Ollama | LeanDojo extraction, proof network construction, anchor gap analysis, data conversion |
| **homebridge** | Persistent CPU worker | i7 4C 2.6GHz, 16GB | Parallel extraction shards, accessible-premises crawl |

**Distribution pattern:**
- CPU-heavy extraction/analysis scripts are standalone workers with zero Wayfinder src/ imports
- All data exchange via JSONL (one JSON object per line, UTF-8)
- Workers support `--resume` (read output, build skip set, append)
- File sync via **parsync** (parallel SSH, resumable, chunked) instead of `scp`
- Merge results back into the main DB on the laptop

**Key parallelism**: While the laptop runs encoder evaluation (Phase 0.6, GPU-bound), macpro and homebridge run proof network extraction and anchor gap analysis (Phase 0.1-0.5, CPU-only). Training never blocks on data preparation.

---

## Tools & Extensions

| Tool | Verdict | Use |
|------|---------|-----|
| **mlx-tune** | Adopted | Encoder LoRA fine-tuning on Apple Silicon. QLoRA, GGUF export. |
| **mlx-vis** | Adopted | GPU-accelerated UMAP/t-SNE/PaCMAP on Metal. 70K points in ~3s. Visualization backbone. |
| **parsync** | Adopted | Parallel SSH file sync for laptop-macpro-homebridge workflow. |
| **LintGate** | Used throughout | Code quality, theory extraction, compass alignment during implementation. |
| **Axle** (axiom-axle) | Adopted (Lane B) | Cloud Lean 4 verification/repair/decomposition. `pip install wayfinder[axle]`. See DESIGN §8.0. |

---

## Phase 0: Proof Network Construction (Week 1-2)

*Goal: Build the semantic network of mathematical entities from Mathlib, and establish the data pipeline.*
*Code status: All scripts implemented (`extract_proof_network.py`, `build_nav_training_data.py`, `anchor_gap_analysis.py`). Awaiting execution on data.*

### 0.1 Obtain LeanDojo Benchmark Data

**What**: Download LeanDojo's extracted Mathlib dataset — ~98,734 theorems with proof traces, intermediate goal states, and premise information.

**How**:
```bash
# LeanDojo provides extracted data via their Python package
pip install lean-dojo
python -c "from lean_dojo import LeanGitRepo, trace; ..."
```

Or use their pre-extracted dataset:
```bash
# ~98k theorems from Mathlib4, pre-traced
# Each entry: theorem_statement, goal_states[], tactics[], premises[]
```

**Output**: `data/leandojo_mathlib.jsonl` — one line per theorem with full proof trace.

**Validation**: Spot-check 100 random entries: goal states are valid Lean, tactics are real Lean tactics, premises reference real Mathlib lemmas.

**Stop/go**: If <90k theorems extract cleanly, investigate and fix. The benchmark dataset must be complete.

### 0.2 Build Proof Network Schema

**What**: Create the SQLite semantic network schema (adapted from ModelAtlas) and populate with Mathlib entities.

**How**:
```python
# src/proof_network.py
# Adapts ModelAtlas's db.py schema for mathematical entities
#
# Tables: entities, entity_positions, anchors, entity_anchors, entity_links, anchor_idf
# Banks: STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION
# Bootstrap ~200 anchors from mathematical vocabulary
```

**Output**: `data/proof_network.db` — SQLite database.

**Validation**:
- All ~100k Mathlib lemmas inserted as entities
- Bank positions assigned to ≥95% of entities (deterministic extraction from types/namespaces)
- ≥3 anchors per entity on average
- IDF computed and cached
- Unit tests for navigate(), spread(), and scoring functions

**Compute**: Local, CPU-only, <2 hours.

**Stop/go**: If bank position coverage <90% or average anchors per entity <2, the extraction pipeline needs refinement.

### 0.3 Implement Proof Network Extraction Pipeline

**What**: Deterministic extraction of bank positions and anchors from Lean type information.

**How**:
```python
# scripts/extract_proof_network.py
#
# For each Mathlib lemma:
#   1. Parse theorem type → STRUCTURE bank position
#      - Count quantifiers, check for dependent types, identify prop vs data
#   2. Namespace → DOMAIN bank position
#      - Mathlib.Algebra → DOMAIN (0,0)
#      - Mathlib.Topology → DOMAIN (+1,2)
#      - Mathlib.Data.Nat → DOMAIN (-1,1)
#   3. Proof length → DEPTH bank position
#      - 1 tactic → DEPTH (-1,2)
#      - 2-3 tactics → DEPTH (0,0)
#      - 10+ tactics → DEPTH (+1,2)
#   4. Tactic analysis → AUTOMATION bank position + anchors
#      - Uses omega/simp/decide → AUTOMATION (-1,*)
#      - Manual intro/cases/induction → AUTOMATION (+1,*)
#   5. Context analysis → CONTEXT bank position
#      - Count hypotheses in goal state
#   6. Proof structure → DECOMPOSITION bank position
#      - Count 'have' statements, case splits
#   7. Assign anchors based on type content, tactic usage, namespace
```

**Output**: Populated `data/proof_network.db`.

**Validation**: Round-trip test — for 100 randomly selected lemmas, verify that:
- Bank positions are plausible (spot-check against manual judgment)
- Anchors are relevant (no false positives from pattern matching)
- IDF weights differentiate rare vs. common anchors
- `navigate()` returns lemmas from the right namespace/domain when queried with domain-specific anchors

### 0.3b Extract Accessible Premises

**What**: For each theorem in Mathlib, extract the set of premises accessible via Lean's import structure. ReProver demonstrated a ~2% recall improvement from this filtering at zero compute cost.

**How**:
```python
# Part of scripts/extract_proof_network.py
#
# For each theorem T in Mathlib:
#   1. Walk T's transitive imports
#   2. Collect all lemma/def entity_ids in imported files
#   3. Store as accessible_premises set in proof_network.db
#   4. navigate() pre-filters to accessible set before scoring
```

**Output**: `accessible_premises` table in `data/proof_network.db` mapping (theorem_id → set of accessible entity_ids).

**Validation**: For 100 random theorems, verify that ground-truth premises are all in the accessible set. If any ground-truth premise is inaccessible, the extraction is wrong.

**Compute**: Local, CPU-only, <30 minutes. LeanDojo already provides this data.

### 0.4 Build Navigational Training Data

**What**: Convert LeanDojo proof traces to navigational training examples.

**How**:
```python
# scripts/build_nav_training_data.py
#
# For each proof step in a proof trace:
#   1. Goal state text → input
#   2. Map tactic used → navigational directions:
#      - omega → structure=-1, automation=-1, decompose=-1
#      - intro → structure=0, automation=+1, decompose=0
#      - cases → structure=+1, automation=+1, decompose=+1
#      - simp [lemma1, lemma2] → structure=-1, automation=-1, decompose=0
#      - have h : T := by → structure=0, automation=0, decompose=+1
#   3. Extract ground-truth anchors from goal + tactic + premises
#   4. Count remaining proof steps → progress label
#   5. Solvability = 1.0 (all steps in successful proofs)
#   6. Write to JSONL
```

**Output**: `data/nav_training.jsonl`, `data/nav_eval.jsonl`

**Format**:
```json
{
  "goal_state": "⊢ ∀ x : ℕ, x + 0 = x",
  "theorem_id": "Nat.add_zero",
  "nav_directions": {"structure": -1, "automation": -1, "decompose": -1},
  "ground_truth_anchors": ["nat-arithmetic", "equality", "omega-solvable"],
  "ground_truth_tactic": "omega",
  "ground_truth_premises": [],
  "remaining_steps": 0,
  "solvable": true
}
```

**Validation**: Verify that navigational direction labels are consistent — similar tactics should produce similar directions. Verify that anchor labels match the proof network's anchor dictionary.

**Stop/go**: If >10% of proof steps can't be mapped to navigational directions (unmapped tactics), extend the mapping table.

### 0.5 Anchor Gap Analysis (Iterative)

**What**: Validate that the proof network can actually retrieve correct premises before any training begins. This is the single highest-leverage Phase 0 activity.

**How**:
```python
# scripts/anchor_gap_analysis.py
#
# For 500 randomly selected proof steps:
#   1. Build a "perfect" navigational query from ground-truth:
#      - Bank positions from the ground-truth tactic mapping
#      - Anchors from the ground-truth anchor labels
#   2. Run navigate(proof_network, perfect_query, limit=16)
#   3. Check: is the ground-truth premise in the top-16 results?
#   4. For each miss: identify what anchors WOULD have connected
#      the goal to the correct premise
#   5. Cluster gap anchors by mathematical theme
#   6. Add top clusters as new anchors, re-populate, re-run
```

**Output**: Iteratively refined anchor dictionary. Gap analysis report per iteration.

**Validation**: Top-16 recall on the 500-step sample using perfect navigational queries.

**Stop/go**: Iterate until top-16 recall ≥ 70%. If recall plateaus below 50% after 3 iterations, the bank positioning scheme (not just anchors) needs rethinking.

**Known gap areas to watch for**:
- Type coercion patterns (↑n, Nat.cast, etc.)
- Algebraic hierarchy position (monoid → group → ring → field)
- Recurring proof patterns (epsilon-delta, contrapositive, diagonal arguments)

### 0.6 Encoder Selection Research

**What**: Evaluate encoder candidates before committing. The encoder is the system's perceptual bottleneck — a bad choice here caps everything downstream.

**Candidates to evaluate**:

| Candidate | Size | Math capability | Unicode handling | Cost |
|-----------|------|----------------|-----------------|------|
| ByT5-small (de facto Lean standard) | 299M | Generic, proven for Lean | Byte-level, native | Low (120 GPU-hrs) |
| ByT5-small + contrastive fine-tuning on Mathlib | 299M | Learned from Lean data | Byte-level, native | Low-Medium |
| Qwen 3.5 (embedding mode) | ~1-7B | Strong math training | Subword tokenizer | Medium |
| DeepSeek-Math (encoder layers) | ~1-7B | Formal math specific | Subword tokenizer | Medium |
| BitNet b1.58 ternary encoder | ~2B effective, ternary | Matches FP on math reasoning | Depends on impl | Medium |
| Aggressively pruned 7B (SparseGPT/Wanda) | ~70-350M effective | Retains math capability | Inherited from base | High initial, low runtime |
| Lean-retrained tokenizer variant (arXiv 2501.13959) | ~299M | Lean-specific | Custom tokenizer | Low-Medium |

**Critical insight from external research**: Byte-level encoding is important for Lean's Unicode-heavy syntax (∀, ⊢, →, ℕ, ⋃). Subword tokenizers fragment mathematical notation. ByT5-small is the proven standard used by ReProver, LeanAgent, and Lean Copilot. Math-native models (Qwen, DeepSeek-Math) may have better math understanding but worse Unicode handling. The BitNet ternary encoder is the most architecturally distinctive option — a fully ternary pipeline from encoder to decoder.

**How**: For each candidate:
1. Encode 1000 Mathlib goal states
2. Measure: do goal states with shared premises/tactics cluster together? (nearest-neighbor premise overlap)
3. Measure: are alpha-equivalent goals (same theorem, different variable names) close?
4. Measure: are goals from different domains (algebra vs topology) separable?

**Output**: Encoder comparison report with quantitative clustering metrics.

**Stop/go**: Select the encoder with best premise-clustering score that fits within compute budget. If no candidate achieves >50% nearest-neighbor premise overlap at k=10, consider fine-tuning the best candidate on Mathlib goal state pairs before proceeding.

**Note on aggressive pruning**: Recent work shows that structured pruning can remove 95-99% of a model's parameters while retaining domain-specific performance when followed by brief fine-tuning. A 7B model pruned to ~70-350M parameters and fine-tuned on Mathlib goal states could provide math-native embeddings at small-model compute cost. This is a high-reward research direction worth a dedicated 2-3 day spike.

### 0.7 Implement PAB Profile Infrastructure

**What**: Adapt the existing PABTracker from BalancedSashimi for navigational metrics.

**How**: Extend `src/pab_tracker.py` with:
- Per-bank navigational accuracy (6 separate accuracy curves)
- Anchor prediction F1 (multi-label classification quality)
- Progress prediction correlation (Spearman ρ with actual remaining steps)
- Navigational consistency (do similar goals produce similar directions?)

**Validation**: Unit tests with synthetic training data, metric hand-computation verification.

**Compute**: Local, CPU-only, <1 hour.

---

## Phase 1: Core Pipeline (Week 2-3)

*Goal: Full forward pass from goal state to navigational output, with proof network resolution.*
*Code status: All modules implemented (`proof_navigator.py`, `proof_network.py`, `resolution.py`, `proof_search.py`, `lean_interface.py`, `nav_contracts.py`, `losses.py`). Awaiting integration testing with real data.*

### 1.1 Implement GoalAnalyzer with Bank/Anchor Heads

**What**: Extend the existing GoalAnalyzer to predict bank positions and anchors alongside features.

**How**:
```python
# src/goal_analyzer.py — extend existing module
#
# Add:
#   self.bank_heads: nn.ModuleDict — one 3-class classifier per bank
#     (maps features to {-1, 0, +1} for each bank)
#   self.anchor_head: nn.Linear(feature_dim, num_anchors)
#     (multi-label sigmoid classification)
#
# forward() returns:
#   features: Tensor [batch, 256]
#   bank_positions: dict[str, Tensor]  # per-bank logits
#   anchor_logits: Tensor [batch, num_anchors]
```

**Validation**: Smoke test — synthetic embeddings produce plausible bank positions and anchor logits. Test that gradients flow through all heads.

### 1.2 Implement ProofNavigator (Ternary Navigational Decoder)

**What**: Replace the existing TernaryDecoder's vocabulary heads with navigational heads. Uses expanded 6-bank navigation to resolve the many-to-one tactic mapping problem (see WAYFINDER_DESIGN.md Section 5.1).

**How**:
```python
# src/proof_navigator.py — new module
#
# class ProofNavigator(nn.Module):
#   Hidden layers: TernaryLinear (STE)
#   Output heads (6 direction heads, one per navigable bank):
#     structure_head: TernaryLinear(hidden, 3)  — 3 classes for {-1, 0, +1}
#     domain_head: TernaryLinear(hidden, 3)
#     depth_head: TernaryLinear(hidden, 3)
#     automation_head: TernaryLinear(hidden, 3)
#     context_head: TernaryLinear(hidden, 3)
#     decompose_head: TernaryLinear(hidden, 3)
#     anchor_head: nn.Linear(hidden, num_anchors)  — continuous, sigmoid
#     progress_head: nn.Linear(hidden, 1)  — continuous, regression
#     critic_head: nn.Linear(hidden, 1)  — continuous, sigmoid
#
#   navigable_banks configured via config; graceful degradation to 3-bank
```

**Validation**: Forward + backward pass smoke test. Verify ternary quantization on hidden layers. Verify that navigational heads produce valid {-1, 0, +1} outputs after argmax. Verify that 6-bank direction vectors disambiguate `intro` vs `apply` and `cases` vs `induction` (the key many-to-one collisions from the 3-bank design).

### 1.3 Implement Structured Resolution

**What**: The deterministic resolution layer that converts navigational output to tactics and premises via the proof network.

**How**:
```python
# src/resolution.py — new module
#
# def resolve(nav_output, proof_network_conn, context):
#   query = build_query(nav_output)
#   tactics = navigate_tactics(proof_network_conn, query)
#   premises = navigate_premises(proof_network_conn, query)
#   spread_scores = spread(proof_network_conn, context.seeds)
#   return combine(tactics, premises, spread_scores)
#
# Adapts ModelAtlas's navigate() and spread() for the proof domain.
```

**Validation**: Given a hand-crafted navigational output (e.g., structure=-1, automation=-1, anchors=["nat-arithmetic", "omega-solvable"]), verify that `resolve()` returns `omega` as the top tactic and appropriate arithmetic lemmas as premises.

### 1.4 Full Pipeline Smoke Test

**What**: End-to-end test: goal state text → encoder → analyzer → bridge → navigator → resolution → tactic + premises.

**Validation**:
- Forward pass completes without error for batch of 4 goal states
- Backward pass produces gradients on all learnable parameters
- Resolution produces real Mathlib tactic names (not garbage)
- Progress head outputs scalar in reasonable range
- Critic head outputs probability in [0, 1]

**Stop/go**: If the full pipeline doesn't produce gradients on all components, debug before proceeding.

---

## Phase 2: Navigation Training (Week 3-5)

*Goal: Train the navigational components and validate that the proof network enables meaningful retrieval.*
*Code status: Training script (`train_navigator.py`), eval scripts (`eval_retrieval.py`, `eval_spreading.py`), and config (`wayfinder.yaml`) all implemented. Awaiting Phase 0-1 data.*

### 2.0 Literature Review: Lean4 Proof Search Algorithms

**What**: Survey existing work on proof search termination, progress estimation, and potential functions in interactive theorem provers. The Mutation Theory connections (trajectory monotonicity, Lyapunov convergence) may have tighter domain-specific analogs in the ITP literature.

**Already completed**: Initial survey in `docs/Wayfinder_External_Research.md` covers 8 major systems. Key findings already incorporated:
- AlphaProof uses remaining-tactic-count as value target (adopted for critic head)
- HTPS: soft critic targets >> hard binary (adopted)
- LeanProgress: proof history gives 13-point improvement (adopted)
- BFS-Prover: best-first search competitive with MCTS when policy is strong
- No system uses knowledge graph navigation for proof search (validates novelty)
- LeanHammer hybrid achieves 37.3% (hammer integration adopted)

**Remaining areas to investigate**:
- Well-founded recursion and termination metrics in Lean 4 itself
- Proof complexity measures (proof tree depth, cut-rank, quantifier depth) as alternative Lyapunov function candidates — raw step count may not be the best training target for the progress head
- QEDCartographer's reward-free RL for state values — compare to our Lyapunov approach
- Crouse et al.'s GNN-based proof length prediction in first-order logic
- Formal connection between proof complexity and teaching dimension (T5.17)

**Output**: Annotated bibliography in `docs/RELATED_PROOF_SEARCH.md`. Focus on: (1) better progress head training targets, (2) formal convergence criteria, (3) any existing structured retrieval approaches we may have missed.

**Time**: 2-3 days concurrent with Phase 1/2 implementation.

### 2.1 Navigational Direction Training

**What**: Train the ProofNavigator to predict correct navigational directions from goal state embeddings.

**How**:
```python
# scripts/train_navigator.py
#
# Training loop:
#   1. Encode goal states (frozen encoder)
#   2. GoalAnalyzer extracts features + bank positions + anchor logits
#   3. Bridge compresses
#   4. ProofNavigator predicts directions, anchors, progress, critic
#   5. Loss = L_nav + L_anchor + L_progress + L_critic (UW-SO weighted)
#      Note: L_critic uses SOFT targets (MSE on normalized distance-to-completion),
#      NOT binary BCE. HTPS found hard binary targets worse than no critic.
#      AlphaProof also trains value on remaining tactic count, not binary.
#   5b. Progress and critic heads receive proof history (mean-pooled embeddings
#       of previously closed goals, concatenated to bridge input). LeanProgress
#       showed 61.8% → 75.1% accuracy from including history.
#   6. Backprop through all learnable components
#
# Curriculum:
#   Phase A (steps 0-500): 1-2 step proofs only
#   Phase B (steps 500-1500): ≤5 step proofs
#   Phase C (steps 1500+): all proofs, oversampling medium difficulty
```

**Metrics (per checkpoint)**:
- Navigational accuracy: % correct {-1, 0, +1} per bank (6 banks, tracked independently)
- Per-bank learning order: which banks crystallize first? (hypothesis: AUTOMATION and STRUCTURE first, DOMAIN and CONTEXT last)
- Anchor F1: multi-label F1 on anchor predictions
- Progress MAE: mean absolute error on remaining steps
- Critic AUC: area under ROC for solvability prediction
- Ternary crystallization: % of stable weight signs
- Scoring mechanism comparison: run retrieval eval with each scoring variant (multiplicative, confidence-weighted, soft-floor) at each checkpoint

**PAB tracking**: All metrics recorded via PABTracker at every 50-step checkpoint.

**Stop/go**:
- After Phase A (step 500): navigational accuracy ≥ 60% on 1-2 step proofs. If not, the navigational labeling or the architecture has a fundamental issue.
- After Phase B (step 1500): navigational accuracy ≥ 50% on ≤5 step proofs. Progress MAE < 2.0 steps.
- After Phase C (step 3000): navigational accuracy ≥ 45% on all proofs. Anchor F1 ≥ 0.3.

### 2.2 Proof Network Retrieval Validation

**What**: Evaluate whether navigational premise retrieval matches or exceeds dense retrieval quality.

**How**:
```python
# scripts/eval_retrieval.py
#
# For each theorem in eval set:
#   1. Get ground-truth premises from LeanDojo data
#   2. Run navigational retrieval: navigate(proof_network, predicted_query)
#   3. Run dense retrieval baseline: dot_product(goal_embedding, premise_embeddings)
#   4. Compare recall@k for k in [1, 4, 8, 16, 32]
```

**Target**: Navigational retrieval recall@16 ≥ 80% of dense retrieval recall@16. If navigational retrieval is within 80% of the learned baseline using *zero neural inference* at retrieval time, the paradigm is validated.

**Ablations**:
- Without IDF weighting → does IDF matter?
- Without bank alignment → are bank positions useful?
- Without spreading activation → does spreading help?
- Only anchors (no banks) → is the anchor dictionary sufficient alone?

**Stop/go**: If navigational retrieval recall@16 < 50% of dense retrieval, the proof network needs enrichment (more anchors, better bank positions, or hybrid retrieval).

### 2.3 Spreading Activation Validation

**What**: Evaluate spreading activation as a proof search heuristic.

**How**:
```python
# scripts/eval_spreading.py
#
# For each multi-step proof in eval set:
#   1. Initialize spread() from step 0's goal state
#   2. At each step, check whether the ground-truth premise was in the
#      top-k activated entities
#   3. Compare to: random baseline, dense retrieval, navigational retrieval alone
```

**Target**: Spreading activation should improve premise recall for later proof steps (where context from earlier steps is informative). Specifically: for proof steps 3+, spreading activation should add ≥5% recall@16 over navigational retrieval alone.

---

## Phase 3: Proof Search Integration (Week 5-7)

*Goal: Close the loop with the Lean kernel. Full proof search with navigational guidance.*
*Code status: Search loop (`proof_search.py`), Lean interface (`lean_interface.py`), proof auditor (`proof_auditor.py`), and benchmark runner (`run_benchmark.py`) all implemented.*
*Architecture: 3-lane verification — Lane A (Pantograph, local), Lane B (Axle, cloud), Lane C (lean4checker, high-assurance). See DESIGN §8.0.*

### 3.1 Lean Kernel Integration

**What**: Set up Pantograph-based interaction with the Lean 4 kernel for tactic verification.

**How**:
```python
# src/lean_interface.py
#
# class LeanKernel:
#   def try_tactic(self, goal_state, tactic_text) -> TacticResult:
#       """Send tactic to Lean kernel, get success/failure + new goals."""
#       ...
#
# Uses LeanDojo's Pantograph server for goal state interaction.
```

**Validation**: Send 100 known-good tactic applications, verify all succeed. Send 100 known-bad applications, verify all fail with appropriate error.

**Stop/go**: If Lean kernel interaction is unreliable (>5% false negatives on known-good tactics), fix before proceeding.

### 3.2 Proof Search Loop

**What**: Implement the outer search loop: neural prediction → symbolic resolution → Lean verification → update context.

**How**:
```python
# src/proof_search.py
#
# def search(theorem, navigator, proof_network, lean_kernel, budget=600):
#   open_goals = [initial_goal]
#   context = SearchContext()
#   attempts = 0
#
#   while open_goals and attempts < budget:
#       goal = select_goal(open_goals, navigator)  # critic + progress
#       tactics, premises = predict_and_resolve(goal, navigator, proof_network, context)
#
#       for tactic, premise_set in candidates(tactics, premises):
#           result = lean_kernel.try_tactic(goal, lower(tactic, premise_set))
#           attempts += 1
#           if result.success:
#               open_goals.remove(goal)
#               open_goals.extend(result.new_goals)
#               context.record_success(...)
#               break
#           else:
#               context.record_failure(...)
#
#   return len(open_goals) == 0  # proved?
```

**Validation**: Run on 50 easy theorems (1-2 step proofs). Target: ≥80% proved within budget of 100 attempts.

### 3.2b Hammer Integration

**What**: Integrate LeanHammer/Aesop as a complementary solver for goals the navigator identifies as fully decidable (AUTOMATION = -1).

**How**:
```python
# In src/proof_search.py, modify the search loop:
#
# if nav_output.automation_direction == -1:
#     # Delegate to hammer — this goal is in the decidable region
#     hammer_result = lean_kernel.try_hammer(goal.state, premises[:16], timeout=30)
#     if hammer_result.success:
#         open_goals.remove(goal)
#         context.record_success(hammer_result.tactic, hammer_result.premises)
#         continue
#     # If hammer fails, fall through to navigational resolution
```

**Rationale**: LeanHammer (2025) shows hybrid neural+symbolic achieves 37.3% — outperforming any single method by 21%. The AUTOMATION bank direction naturally routes between Wayfinder (AUTOMATION ≥ 0) and hammers (AUTOMATION = -1). Wayfinder's navigational premise retrieval feeds candidates to the hammer, combining structured selection with ATP power.

**Validation**: Compare proof rate with and without hammer delegation on 200 test theorems. Measure what fraction of proved theorems use the hammer path.

### 3.3 Benchmark Evaluation

**What**: Evaluate on standard benchmarks.

**Benchmarks**:
1. **MiniF2F-test** (488 problems): Primary comparison benchmark.
2. **Mathlib test split** (~2k held-out theorems): Larger-scale evaluation.

**Metrics**:
- Theorems proved (within search budget)
- Average search budget consumed per proof
- Neural forward passes per proof (should be dramatically lower than baselines)
- Wall-clock time per proof

**Baselines for comparison**:
- ReProver (published numbers)
- LeanProgress (published numbers)
- DeepSeek-Prover-V2 (published numbers, noting parameter count difference)
- Ablated Wayfinder variants (from Phase 2 ablations)

**Stop/go**:
- MiniF2F: ≥20% proved (competitive with ReProver-scale systems, given our much smaller model)
- Mathlib test: ≥15% proved
- Neural forward passes: ≤50% of ReProver per proof (demonstrating the efficiency thesis)

If proved rates are <10%, analyze failure modes before Phase 4.

### 3.4 Lane B: Axle Proof Audit Integration

**What**: Wire the `ProofAuditor` (Lane B) as a post-search verification and repair layer.

**How**:
1. After Lane A search completes (success or failure), optionally pass the proof to Lane B.
2. For **successful** Lane A proofs: `auditor.verify()` provides authoritative confirmation. Tagged `raw_success`.
3. For **failed** Lane A proofs with partial progress: `auditor.repair()` attempts to close remaining goals with terminal tactics (`grind`, `aesop`, `simp`, `omega`, `decide`). Tagged `axle_repair_only`.
4. For **failed** Lane A proofs: `auditor.decompose()` via `sorry2lemma` extracts subgoals. Navigator retries each subgoal independently. Tagged `axle_assisted_success` if all subgoals close.

**Cost controls**:
- Content-hash cache (SHA-256) avoids redundant API calls
- Only invoke Axle on top-N candidates or theorems with high critic score
- Bounded async concurrency (`axle.max_concurrency` in config)
- Fallback policy: on timeout/429/503, Lane B returns failed `AuditResult`; Lane A result stands as-is

**Metric separation** (enforced in `run_benchmark.py`):
- `raw_success`: Lane A proved, no Axle involvement
- `axle_assisted_success`: Lane A partial + Axle decompose/repair completed the proof
- `axle_repair_only`: Axle `repair_proofs` alone closed remaining goals

**Validation**: Run 200 benchmark theorems with Lane B enabled. Report all three metric categories separately. Verify cache hit rate >50% on repeated proofs.

**Scope caveat**: Axle is designed for simple imports/theorems/definitions. Pin `axle.environment` in config; update only with Mathlib version bumps. For leaderboard submissions, additionally verify with Lane C (lean4checker/Comparator/SafeVerify).

### 3.5 Axle-Enriched Proof Network (Optional)

**What**: Use Axle's `extract_theorems` to enrich the proof network with dependency metadata not available from LeanDojo extraction alone.

**How**: `auditor.extract_theorems()` returns per-theorem `proof_length`, `tactic_counts`, and 6 dependency flavors (`local_type_dependencies`, `local_value_dependencies`, `external_type_dependencies`, `external_value_dependencies`, `local_syntactic_dependencies`, `external_syntactic_dependencies`). Write an enrichment script that merges these fields into `proof_network.db` entity records.

**Value**: Richer dependency graph for spreading activation. `tactic_counts` directly validates tactic direction labels. `proof_length` correlates with DEPTH bank positions.

---

## Phase 4: Evaluation and Ablation (Week 7-9)

*Goal: Deep analysis of results across all three streams.*

### 4.1 Stream 1 Analysis: Navigation vs. Dense Retrieval

**Questions to answer**:
- Where does navigational retrieval outperform dense retrieval? (Which domains, which proof types?)
- Where does it underperform? (Is the deficit in bank positions, anchors, or IDF weighting?)
- What is the compute profile? (Neural forward passes saved per proof)
- Does the multiplicative scoring prevent good matches that additive scoring would find?

### 4.2 Stream 2 Analysis: Navigational Decoder vs. Token Classification

**Questions to answer**:
- Do navigational coordinates generalize to unseen theorem types better than vocabulary indices?
- Does the ternary decoder's crystallization pattern reveal structured learning?
- Which navigational bank is learned first? Does the order match architectural expectations?
- Does the anchor prediction quality correlate with proof success?

### 4.3 Stream 3 Analysis: PAB Trajectory Evaluation

**Questions to answer**:
- Does navigational accuracy trajectory predict final proof success better than loss trajectory?
- Does crystallization rate predict navigational consistency?
- Does progress prediction accuracy improve monotonically? Does it plateau?
- Does PAB detect failure modes that endpoint metrics miss?

### 4.4 Ablation Matrix

Run full evaluation for each ablation variant from Phase 2:

| Variant | MiniF2F | Mathlib | Retrieval R@16 | FP/proof | Notes |
|---------|---------|---------|----------------|----------|-------|
| Full Wayfinder | — | — | — | — | Primary |
| Dense retrieval (no proof network) | — | — | — | — | Navigation thesis |
| Tactic classification (no navigation) | — | — | — | — | Architecture thesis |
| No spreading activation | — | — | — | — | Spreading thesis |
| No progress head | — | — | — | — | Progress thesis |
| Continuous decoder (no ternary) | — | — | — | — | Ternary thesis |
| No IDF weighting | — | — | — | — | IDF thesis |
| No bank alignment (anchors only) | — | — | — | — | Bank thesis |
| Binary critic (BCE, not soft MSE) | — | — | — | — | HTPS soft-target thesis |
| No proof history input | — | — | — | — | LeanProgress history thesis |
| No hammer delegation | — | — | — | — | Hammer complementarity thesis |
| No accessible-premises filter | — | — | — | — | ReProver filtering thesis |
| 3-bank navigation (original design) | — | — | — | — | 6-bank expansion thesis |

### 4.5 Write-up

**Deliverables**:
- Updated `WAYFINDER_RESEARCH.md` with experimental results
- `docs/EXPERIMENT_RESULTS.md` — raw results ledger
- PAB profile comparisons across all variants
- Failure analysis with navigational interpretation

---

## Phase 5: Scale and Extend (Week 9+, contingent on results)

### 5.1 Enriched Proof Network

If navigational retrieval underperforms dense retrieval:
- Add LLM-based anchor extraction (Tier 3) for subtle mathematical properties
- Add cross-namespace links from proof co-occurrence analysis
- Experiment with finer-grained bank positions (depth 3-4 instead of 1-2)

### 5.2 Online Learning with Navigational Wavefront Curriculum

If proof search works but slowly improves:
- Update proof network links based on successful proofs (HTPS-inspired online training)
- Add new anchors discovered from proof patterns
- Re-compute IDF as the network evolves
- **Navigational wavefront curriculum**: Order new training theorems by distance in the proof network from already-proven results. Spread activation from the set of proven theorems; the frontier (theorems 1-2 hops away with highest activation) defines the next training batch. This addresses the saturation problem: up to 98.5% of proofs during expert iteration are incorrect. By targeting the "cusp of provability" (STP found 47% success on cusp-generated conjectures vs 11.4% on random unproved), training signal is dramatically denser.

### 5.3 Hybrid Retrieval

If navigational and dense retrieval have complementary strengths:
- Use navigational retrieval for initial candidate set (fast, high recall)
- Re-rank with dense retrieval for top-k precision (one neural forward pass for k candidates)
- Compare to pure-dense and pure-navigational

### 5.4 Subgoal Decomposition Training

If the SubgoalDecomposer shows promise in ablations:
- Train on DeepSeek-Prover-V2-style `have`-chain examples
- Evaluate decomposition quality independently
- Measure impact on longer proofs (5+ steps)

---

## Definition of Ready to Run

The primary Wayfinder config works end-to-end:

- [ ] `proof_network.db` populated with ~98k entities
- [ ] Anchor gap analysis: top-16 recall >= 70% on perfect queries
- [ ] Encoder chosen and integrated
- [ ] Full pipeline forward+backward works on real data
- [ ] Training runs 3000+ steps with decreasing loss, improving nav accuracy
- [ ] Proof search proves >= 5 trivial theorems end-to-end
- [ ] Benchmark runner produces metrics on MiniF2F subset
- [ ] Core tests pass (proof network, navigator, resolution, search, training)
- [ ] Reproducible within tolerance (same seed -> metrics within +/- 1%)

**Post-readiness hardening:**
- [ ] All 13 ablation configs load, train, and evaluate
- [ ] PAB trajectory comparison tooling
- [ ] Full test coverage for edge cases

---

## Module Map

| Task | Start here |
|------|-----------|
| Understand the navigational paradigm | `docs/WAYFINDER_RESEARCH.md` Section 2 |
| Understand scoring alternatives | `docs/WAYFINDER_RESEARCH.md` Section 2.4, `docs/WAYFINDER_DESIGN.md` Section 3.1 |
| Understand 6-bank tactic mapping | `docs/WAYFINDER_DESIGN.md` Section 5.1 |
| Understand encoder strategy | `docs/WAYFINDER_RESEARCH.md` Section 3.2 |
| Understand anchor gap analysis | `docs/WAYFINDER_RESEARCH.md` Section 2.3, Phase 0.5 |
| Understand Mutation Theory connections | `docs/WAYFINDER_RESEARCH.md` Section 2.7-2.8 |
| Understand the proof network | `src/proof_network.py` |
| Understand navigation resolution | `src/resolution.py` |
| Understand the ternary navigator | `src/proof_navigator.py` |
| Understand proof search | `src/proof_search.py` |
| Understand proof auditor (Lane B) | `src/proof_auditor.py` |
| Understand navigational contracts | `src/nav_contracts.py` |
| Understand PAB metrics | `src/pab_tracker.py` |
| Understand training | `scripts/train_navigator.py` |
| Understand evaluation | `scripts/eval_retrieval.py`, `scripts/eval_spreading.py` |
| Understand data format | `src/nav_contracts.py`, `src/contracts.py` |
| Run anchor gap analysis | `python -m scripts.anchor_gap_analysis` |
| Run training | `python -m scripts.train_navigator --config configs/wayfinder.yaml` |
| Run retrieval eval | `python -m scripts.eval_retrieval --config configs/wayfinder.yaml` |
| Run spreading eval | `python -m scripts.eval_spreading --config configs/wayfinder.yaml` |
| Run benchmark | `python -m scripts.run_benchmark --config configs/wayfinder.yaml` |
| Record results | `docs/EXPERIMENT_RESULTS.md` |

## Key Data Files

| File | Purpose |
|------|---------|
| `data/leandojo_mathlib.jsonl` | LeanDojo extracted Mathlib dataset (~98k theorems) |
| `data/proof_network.db` | SQLite semantic network of mathematical entities |
| `data/nav_training.jsonl` | Navigational training data |
| `data/nav_eval.jsonl` | Navigational evaluation data (frozen) |
| `configs/wayfinder.yaml` | Primary experiment configuration |

## Invariants

1. **`data/nav_eval.jsonl` is frozen.** Never modify the evaluation set.
2. **The proof network is populated and gap-analyzed before training begins.** Bank positions, anchors, IDF, accessible-premises sets, and anchor gap analysis (Phase 0.5) must be complete before the first training step. Top-16 recall on perfect queries must exceed 70%.
3. **Neural inference happens once per proof state.** The resolution and search layers must not call neural forward passes. (Exception: hammer delegation calls external ATPs, which is allowed.)
4. **All retrieval scores are auditable.** Every premise retrieval must trace to specific shared anchors and bank alignments.
5. **PAB tracking is mandatory from step 0.** Retrofit trajectory analysis produces incomplete data.
6. **Scoring mechanism is configurable.** The scoring composition (multiplicative, confidence-weighted, etc.) is a config parameter, not hardcoded. Default: `confidence_weighted`.
7. **Encoder selection is an explicit decision.** Do not default to MiniLM. Evaluate byte-level and math-native candidates (Phase 0.6) before committing. ByT5-small is the proven baseline; BitNet ternary is the distinctive option.
8. **6-bank navigation is default.** All 6 banks are navigable. Graceful degradation to 3-bank is available via `navigable_banks` config but should not be the starting point.
9. **Critic uses soft targets.** MSE on normalized distance-to-completion, NOT binary BCE. This is a hard constraint per HTPS and AlphaProof findings. Binary critic is an ablation variant only.
10. **Accessible-premises filtering is always on.** Pre-filter to import-accessible premises before scoring. Free ~2% recall gain from ReProver.
11. **Hammer delegation for AUTOMATION = -1.** When the navigator predicts fully decidable, delegate to LeanHammer/Aesop before attempting navigational resolution.
