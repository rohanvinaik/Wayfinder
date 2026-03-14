# Wayfinder: Operational Research Plan

**Version:** 2.0
**Date:** March 10, 2026
**Corresponding documents:** `WAYFINDER_RESEARCH.md` (theory), `WAYFINDER_DESIGN.md` (engineering), `EXPERIMENT_RESULTS.md` (results ledger)
**Status:** v1 implementation complete. Phase 0 data pipeline complete. NAV-001/002 trained (monolithic navigator, chaotic PAB). Pivoting to Society of Mind architecture (v2) guided by specification complexity theory.

---

## Overview

This document specifies the sequence of work, tooling, compute decisions, and stop/go criteria for running the Wayfinder experiments. All source code and scripts are implemented (36 src files, 18 scripts, 3 configs). This plan covers data pipeline execution and experimental evaluation.

This plan produces five categories of results:

- **Stream 1 (Navigation)**: Does structured semantic navigation — bank positions, IDF-weighted anchors, spreading activation — outperform dense embedding retrieval for premise selection and proof search?

- **Stream 2 (Architecture)**: Does a ternary navigational decoder that produces directional coordinates, resolved through a semantic network, outperform tactic-token classification?

- **Stream 3 (Process Evaluation)**: Does PAB's trajectory evaluation reveal information about the navigational learning process that endpoint metrics cannot?

- **Stream 4 (Decomposition)**: Does a Society of Mind architecture — multiple specialists with PAB-guided scope, narrative template classification, and typed temporal slots — outperform a monolithic navigator? Does specification complexity theory correctly predict when and how to decompose?

- **Stream 5 (Boundary Learning + Energy Refinement)**: Does incorporating structured negative examples — failed tactics, incomplete proofs, and actively-generated boundary cases — transform positive-only imitation into boundary-aware reasoning? Does an energy-based refinement loop, grounded in Orthogonal Ternary Projection theory (Vinaik, 2025) and energy-based constraint satisfaction (cf. Logical Intelligence/Kona, 2026), enable holistic proof sketch optimization that outperforms sequential candidate scoring?

Every phase states what it validates in each stream. The experimental pipeline is designed so that a single set of experiments tests all five claims. Results are recorded in `docs/EXPERIMENT_RESULTS.md`.

**Time horizon:** 6-10 weeks for Phase 0-4 (v1). Phase 5 extensions. Phase 6 Society of Mind (v2, 4-6 additional weeks). Phase 7 Energy-Constrained Navigation (v2.1, 4-6 additional weeks).

**Compute budget:** Apple Silicon (M-series Mac) as primary target. The architecture is designed for efficiency: small learnable navigational components (~400K trainable params) and symbolic search (SQLite). Encoder selection (Phase 0.6) is complete: `all-MiniLM-L6-v2` (22M, 384d) selected as primary encoder — frozen, no fine-tuning needed. 617 goals/sec on MPS, minimal memory. LeanDojo retriever (`kaiyuy/leandojo-lean4-retriever-byt5-small`, 218M, 1472d) reserved as ablation candidate for Phase 2.2 encoder comparison.

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

## Phase 0: Proof Network Construction (Week 1-2) ✅ COMPLETE

*Goal: Build the semantic network of mathematical entities from Mathlib, and establish the data pipeline.*
*Status: All phases complete. 78,414 entities extracted, proof network populated, nav training data built (321K examples), anchor gap analysis passed (recall@16=100%), encoder selected (MiniLM primary, LeanDojo + pplx-embed ablation candidates from 15-model evaluation).*

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

**Output**: `data/nav_train.jsonl`, `data/nav_eval.jsonl`

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

### 0.6 Encoder Selection Research ✅ COMPLETE

**What**: Evaluate encoder candidates before committing. The encoder is the system's perceptual bottleneck — a bad choice here caps everything downstream.

**Candidates** (discovered via ModelAtlas semantic search + domain knowledge, 5 tiers + extended math-domain set):

| Tier | Candidate | Params | Native Dim | Backend | Notes |
|------|-----------|--------|-----------|---------|-------|
| Baseline | `all-MiniLM-L6-v2` | 22M | 384 | SentenceTransformer | Current default |
| Small | `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | SentenceTransformer | ModernBERT architecture |
| Small | `nomic-ai/modernbert-embed-base` | 137M | 768 | SentenceTransformer | Matryoshka embeddings |
| Medium | `google/byt5-small` | 300M | 1472 | T5 (encoder-only) | Byte-level, tokenizer-free |
| Medium | `Snowflake/snowflake-arctic-embed-l-v2.0` | 335M | 1024 | SentenceTransformer | Matryoshka, code-aware |
| Medium | `Salesforce/SFR-Embedding-Code-400M_R` | 400M | 1024 | SentenceTransformer | Code-specialized |
| Large | `dunzhang/stella_en_1.5B_v5` | 1.5B | 8192 | SentenceTransformer | Qwen2-based, native load |
| XL | `intfloat/e5-mistral-7b-instruct` | 7B | 4096 | Decoder-only (last-token) | Instruct-tuned embedder |
| XL | `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7B | 3584 | SentenceTransformer | Qwen2-based, native load |
| XL | `nvidia/NV-Embed-v2` | 7.85B | 4096 | Decoder-only (last-token) | Unfixable (transformers 5.x) |
| Math | `kaiyuy/leandojo-lean4-retriever-byt5-small` | 218M | 1472 | T5 (encoder-only) | ReProver's Lean 4 premise retriever |
| Math | `math-similarity/Bert-MLM_arXiv-MP-class_zbMath` | 110M | 768 | SentenceTransformer | arXiv math similarity |
| Math | `tbs17/MathBERT` | 110M | 768 | SentenceTransformer | Math curriculum MLM (no embed fine-tune) |
| Novel | `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | SentenceTransformer | Causal→embedding, Matryoshka |
| Novel | `perplexity-ai/pplx-embed-v1-0.6b` | 596M | 1024 | SentenceTransformer | Bidirectional Qwen3 (novel arch) |
| Math+XL | `FrenzyMath/LeanSearch-PS` | 7B | 4096 | PEFT (LoRA on e5-mistral) | REAL-Prover premise selector |

**Evaluation script**: `scripts/eval_encoders.py` (standalone worker pattern, `--tier` and `--extended` flags for selective evaluation)

**Metrics per candidate**:
1. **Tokenizer math symbol coverage**: % of 32 critical Unicode symbols (∀, ∃, ⊢, ↔, ∣, ℕ, ℤ, ℝ, etc.) encoded without UNK
2. **Intra-theorem cosine similarity**: Mean cosine between goal states from the same theorem
3. **Inter-theorem cosine similarity**: Mean cosine between goal states from different theorems
4. **Separation score**: Intra minus inter (higher = better clustering)
5. **Encoding throughput**: Goals/second on target hardware
6. **Memory footprint**: Peak RSS during encoding

**How**:
```bash
# Default: small+medium tier (6 models)
python scripts/eval_encoders.py --eval-data data/nav_eval.jsonl --samples 500

# All original tiers including 7B+ models
python scripts/eval_encoders.py --tier all --samples 500 --device mps

# Extended math-domain + novel architecture models
python scripts/eval_encoders.py --tier extended --samples 500 --device mps

# Single model evaluation
python scripts/eval_encoders.py --model "kaiyuy/leandojo-lean4-retriever-byt5-small" --samples 500
```

**Results** (15/16 evaluated, sorted by separation — see `EXPERIMENT_RESULTS.md` EXP-0.3/0.3b for full tables):

| Rank | Candidate | Dim | Separation | Goals/sec | Notes |
|------|-----------|-----|-----------|-----------|-------|
| 1 | **leandojo-lean4-retriever-byt5-small** | 1472 | **0.6233** | 21.9 | Lean-specific fine-tune, 4.2x over vanilla byt5 |
| 2 | pplx-embed-v1-0.6b | 1024 | 0.6002 | 16.8 | Bidirectional Qwen3 |
| 3 | **all-MiniLM-L6-v2** | 384 | **0.5869** | **616.7** | **Selected — best throughput** |
| 4 | Qwen3-Embedding-0.6B | 1024 | 0.4659 | 14.6 | |
| 5 | MathBERT (raw MLM) | 768 | 0.4552 | 147.1 | No embedding fine-tuning |
| 6 | snowflake-arctic-embed-l-v2.0 | 1024 | 0.3709 | 43.6 | |
| 7 | Bert-MLM_arXiv-MP-class_zbMath | 768 | 0.3463 | 149.2 | |
| 8 | stella_en_1.5B_v5 | 1024 | 0.3377 | 9.9 | |
| 9-10 | gte-modernbert / modernbert-embed | 768 | ~0.30 | ~55 | |
| 11 | LeanSearch-PS (LoRA) | 4096 | 0.2080 | 2.0 | 2.3x over base e5-mistral |
| 12-15 | gte-Qwen2 / byt5 / SFR-Code / e5-mistral | — | 0.09-0.18 | 1-22 | Large models collapse space |
| — | NV-Embed-v2 | 4096 | — | — | Unfixable (transformers 5.x) |

**Key empirical findings:**
1. **Inverse size-separation relationship** confirmed across 15 models — larger models produce higher BOTH intra AND inter cosine similarity, collapsing discrimination.
2. **Domain-specific fine-tuning shifts the curve** — LeanDojo byt5 (sep=0.623) vs vanilla byt5 (sep=0.147) is a 4.2x improvement on the same architecture. LeanSearch-PS LoRA doubles base e5-mistral (0.089→0.208). But fine-tuning cannot overcome 7B model space collapse.
3. **Tokenizer coverage is a red herring** — MiniLM (71.9% coverage) beats all 100%-coverage models except LeanDojo and pplx-embed.
4. **Math-domain pretraining alone creates useful geometry** — MathBERT (raw MLM, no embedding fine-tuning) at sep=0.455 beats 7 purpose-built embedding models.

**Decision:** `all-MiniLM-L6-v2` (384d) selected as primary encoder — best throughput for iterative proof search (617 goals/sec), only 6% behind LeanDojo on separation. `leandojo-lean4-retriever-byt5-small` (1472d) reserved as ablation candidate for Phase 2.2 encoder comparison.

**Four encoding backends**: SentenceTransformer (standard `.encode()`), T5/ByT5 (encoder-only mean pooling), decoder-only (last-token pooling with instruction prefixes, fp16, batch_size=4), PEFT/LoRA (adapter on decoder-only base).

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

## Phase 1: Core Pipeline (Week 2-3) ✅ COMPLETE

*Goal: Full forward pass from goal state to navigational output, with proof network resolution.*
*Code status: All modules implemented. Integration validated by NAV-001/002 training runs (Phase 2.1). Full forward+backward passes confirmed on real data (321K examples, 5000 steps).*

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
*Status: Phase 2.1 training complete (NAV-001, NAV-002). Checkpoints saved at `models/NAV-001_step5000.pt` and `models/NAV-002_step5000.pt`. Peak nav accuracy 72.0% (NAV-002). PAB confirmed chaotic (stability_mean=0.341). Phase 2.2 (eval retrieval) and 2.3 (eval spreading) scripts ready, awaiting execution.*

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
- **Encoder ablation**: Repeat with `leandojo-lean4-retriever-byt5-small` (1472d, sep=0.623) instead of MiniLM (384d, sep=0.587). Does the +6% separation and Lean-domain fine-tuning translate to meaningful retrieval recall improvement, or does MiniLM's 28x throughput advantage make it the clear winner for iterative search? Also test `pplx-embed-v1-0.6b` (1024d, sep=0.600) as a domain-agnostic alternative with novel bidirectional architecture.

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
*Lane A status: Pantograph backend is **not yet implemented** — `lean_interface.py` raises `NotImplementedError`. Current benchmark uses `stub` (offline, no real verification) or `replay` (ground-truth matching). Real `raw_success` requires Pantograph integration (Phase 3.1). Lane B (Axle) is operational for cloud verification.*

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

**Statistical significance**: Use McNemar's test for paired comparison with published baselines where per-theorem results are available. For comparisons against published aggregate numbers only (ReProver, DeepSeek-Prover-V2), report 95% Clopper-Pearson CI on Wayfinder's prove rate and note whether baseline falls within CI.

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
| Encoder: LeanDojo-byt5 (1472d) | — | — | — | — | Domain encoder thesis (sep=0.623 vs MiniLM 0.587) |
| Encoder: pplx-embed bidir-Qwen3 (1024d) | — | — | — | — | Novel architecture thesis (sep=0.600) |

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

## Phase 6: Society of Mind Architecture (Week 10-15, v2)

*Goal: Decompose the monolithic navigator into a Society of Mind architecture guided by specification complexity theory and PAB stability measurement. See `WAYFINDER_RESEARCH.md` §2.9 and `WAYFINDER_DESIGN.md` §10 for theoretical and engineering foundations.*

*Prerequisites: Phase 1-2 (v1 navigator trained, PAB baselines established). Phase 3-5 not required — SoM development can proceed in parallel with v1 benchmark evaluation.*

### 6.0 Establish v1 Baselines (NAV-001/002 Analysis) ✅ COMPLETE

**What**: Confirm that the monolithic navigator produces chaotic PAB dynamics, validating the need for decomposition.

**Results** (from NAV-001/002):
- PAB stability_mean = 0.324 (NAV-001), 0.341 (NAV-002) — both "chaotic"
- Bank difficulty hierarchy confirmed: Tier 3 easy (DOMAIN 0.95+, CONTEXT 0.73-0.88, DECOMPOSITION 0.63-0.88) vs Tier 2 hard (STRUCTURE 0.43-0.66, AUTOMATION 0.25-0.60, DEPTH 0.24-0.60)
- Ternary crystallization instant (0.9985 from step 100) — decoder signs are σ ≈ 1
- Bridge representations freeze by step 3000 — shared bottleneck caps hard-bank learning
- Phase C accuracy crash on hard banks — bulk-to-tail transition (Theorem 3.4)

**Conclusion**: Chaotic PAB is genuine training dynamics, not measurement artifact. Decomposition is warranted.

### 6.0b Composition Gap Control Experiment

**What**: Distinguish "chaotic PAB = high γ from shared bridge" from alternative explanations (insufficient capacity, label noise, wrong LR). This is the key falsifiability test for the specification complexity motivation.

**How**: Train three configs with identical total parameter count (~400K):

1. **v1 monolithic** (NAV-002 reproduction) — 1 shared 128d bridge, all 6 banks
2. **Separate-bridge monolithic** — 2 independent 128d bridges (banks split by regime), but shared hidden layers downstream
3. **v2 two-specialist** — 2 fully independent specialists (own bridges, own hidden layers, own heads)

If (3) is stable but (1) is chaotic, the bridge-sharing γ explanation is supported.
If (2) is also stable, the bridge alone is the bottleneck (strongest evidence for γ).
If all three are chaotic, the issue is capacity or data, not composition gap — the SoM motivation weakens.

**Compute**: 3x NAV-002 (~30 min total on laptop). Run before investing in Phase 6.1+.

**Stop/go**: If (3) is NOT more stable than (1), do NOT proceed with SoM. Investigate capacity, LR, or data quality instead.

### 6.0c PAB Stability Threshold Calibration

**What**: The 0.30 threshold for "chaotic" was established on 6-bank monolithic training. A 2-bank specialist has inherently lower loss variance (fewer competing gradients). The threshold must be calibrated per-specialist to avoid false "stable" declarations.

**How**: For each specialist config, compute the expected stability_mean under a null model (random predictions, same data, same number of banks). The "chaotic" threshold is set at 2× the null model's stability_mean.

**Rationale**: A 2-bank specialist with stability_mean = 0.15 looks "stable" by the 0.30 threshold, but if the null model for 2 banks gives stability_mean = 0.12, then 0.15 is only 1.25× null — not meaningfully stable. The threshold scales with task complexity.

**Output**: Per-specialist stability thresholds recorded in `configs/wayfinder_v2.yaml`.

### 6.1 Template Extraction and Taxonomy

**What**: Extract proof strategy templates from the training corpus by clustering tactic sequences.

**How**:
```python
# scripts/extract_templates.py
#
# For each proof in nav_train.jsonl:
#   1. Extract tactic sequence (ordered list of tactic names)
#   2. Map each tactic to its 6-bank direction signature
#   3. Compute bank-signature centroid for the full sequence
#   4. Cluster proofs by centroid similarity (k-means or DBSCAN)
#   5. For each cluster: identify dominant tactic pattern → template label
#   6. Output: template_taxonomy.json (template_id → bank_signature, tactic_pattern, count)
#   7. Augment nav_train.jsonl with template_id labels
```

**Output**: `data/template_taxonomy.json` and augmented training data with template labels.

**Validation**:
- Template taxonomy should have 8-15 templates covering ≥ 90% of proofs
- **Quantitative coherence**: silhouette score ≥ 0.3 on bank-signature centroids (measures within-cluster similarity vs between-cluster separation)
- **NMI**: normalized mutual information between template labels and dominant tactic ≥ 0.5 (templates should correlate with tactic choice)
- Manual inspection of 20 proofs per template to verify qualitative coherence

**Stop/go**: If templates cover < 70% of proofs OR silhouette < 0.2, the taxonomy needs refinement. If > 30 templates needed, re-cluster with coarser bank-signature quantization.

**Stream 4 validation**: This phase tests whether proof strategies cluster into discrete templates at all. If tactic sequences are uniformly distributed (no clusters), the narrative regime conversion hypothesis (§2.9.5) is falsified.

### 6.2 RECOGNITION Slot: Template Classifier

**What**: Train a lightweight classifier to predict the proof template from goal state features.

**How**:
```python
# scripts/train_template_classifier.py
#
# Architecture: GoalAnalyzer features (256d) → Linear(256, 128) → ReLU
#               → Linear(128, k) → softmax
# Loss: Cross-entropy on template labels
# Training: On augmented nav_train.jsonl with template_id
# Evaluation: Top-1 and top-3 accuracy on nav_eval.jsonl
```

**Metrics**:
- Top-1 template accuracy (target: ≥ 60%)
- Top-3 template accuracy (target: ≥ 85%)
- Per-template precision/recall (identify templates that are hard to classify)
- PAB stability of the template classifier itself (should be "stable" — this is a Regime A task)

**Stop/go**: If top-3 accuracy < 50%, the templates may not be predictable from goal state features alone. Consider adding proof context (namespace, theorem statement) to the classifier input.

### 6.3 PLANNING Slot: Proof Sketch Predictor

**What**: For each template, produce a concrete proof sketch — ordered subgoal sequence with estimated difficulty and anchor targets.

**How (two-stage implementation)**:

**Stage 1: Deterministic template instantiation.** For simple templates (DECIDE, REWRITE_CHAIN, HAMMER_DELEGATE), the sketch IS the template — a fixed tactic sequence with placeholder arguments. No learning needed. Implementation: lookup table from template_id to sketch.

**Stage 2: Learned sketch predictor.** For complex templates (DECOMPOSE_AND_CONQUER, INDUCT_THEN_CLOSE, CASE_ANALYSIS), train a small model:
```python
# src/sketch_predictor.py
#
# Input: goal_embedding (384d) + template_features (64d) = 448d
# Output: ordered list of SubgoalSpec:
#   - subgoal_type: str (from template's expected sequence)
#   - anchor_targets: top-k anchors for this subgoal
#   - estimated_steps: int
#   - bank_hints: dict[str, int] (predicted direction per bank)
#
# Loss: Cross-entropy on subgoal types + BCE on anchor targets
#       + MSE on step estimates
```

**Stage 3 (deferred): Story-writing LLM integration.** For the hardest proofs where learned sketches fail, integrate a small LLM (DeepSeek-Math 1.3B or Qwen 3.5) fine-tuned to generate natural-language proof narratives. Parse narratives into structured sketches. This is the Relational-AI pattern: narrative model produces strategy, specialist models execute.

**Validation**: For 200 test proofs, compare sketch-predicted subgoal sequences to actual proof structure. Measure: subgoal count accuracy, anchor target overlap with ground truth, ordering correctness.

**Stop/go**: If sketch subgoal sequences match actual proof structure ≥ 40% of the time for complex templates, proceed. If < 20%, the sketch predictor needs more capacity or the templates need refinement.

### 6.3b Ternary Decoder Falsification Test

**What**: NAV-002 showed 99.85% ternary crystallization from step 100 — decoder signs never change after initialization. Two interpretations: (a) σ ≈ 1, signs are specified by a single observation at init (the signs encode meaningful navigational information); or (b) the decoder is dead, signs are random noise that never updates. This test distinguishes the two.

**How**: Take NAV-002 checkpoint at step 2000. Run three conditions:
1. **Control**: resume training for 500 steps, no modification
2. **50% sign flip**: randomly flip 50% of decoder ternary weight signs, resume training for 500 steps
3. **100% sign flip**: flip all signs, resume training for 500 steps

**Interpretation**:
- If accuracy drops immediately after flip and does NOT recover → signs encode information, σ ≈ 1 interpretation valid
- If accuracy drops but recovers within 500 steps → signs encode information but are re-learnable (σ > 1 but small)
- If accuracy does NOT drop after flip → decoder is dead; all learning happens in bridge/analyzer, not decoder weights

**Compute**: 3x 500 steps (~2 min total). Critical to do before investing in ternary-specific specialist architecture.

### 6.4 EXECUTION Slot: Specialist Decomposition

**What**: Decompose the v1 monolithic navigator into bank-cluster specialists, guided by PAB stability measurement.

**How**:
```python
# scripts/train_specialist.py
#
# Initial decomposition (from NAV-002 bank difficulty analysis):
#   Specialist-A: DOMAIN, CONTEXT (Regime A banks, PAB target: stable)
#   Specialist-B: STRUCTURE, AUTOMATION, DEPTH, DECOMPOSITION (Regime B banks)
#
# Each specialist has:
#   - Own bridge (128d) — eliminates shared-bridge γ
#   - Own hidden layers (TernaryLinear)
#   - Own direction heads (only for assigned banks)
#   - Shared: encoder (frozen), anchor logits, progress, critic (via fusion)
#
# Training: Same curriculum as v1, but each specialist sees only its bank losses
# PAB: Track stability_regime per specialist independently
#
# Decomposition protocol (DESIGN §10.8):
#   If Specialist-B PAB = "chaotic":
#     Split into B1 (STRUCTURE, DECOMPOSITION) + B2 (AUTOMATION, DEPTH)
#     Retrain, re-measure PAB
#     Iterate until all specialists reach "stable"
```

**Metrics per specialist**:
- PAB stability_regime (must reach "stable" or "transitional")
- Per-bank accuracy (should improve over v1 monolithic for assigned banks)
- Ternary crystallization rate
- Total training compute vs v1 (should be less due to additive σ)

**Comparison matrix**:

| Config | Banks | Bridges | PAB Target | Compute |
|--------|-------|---------|------------|---------|
| v1 monolithic (NAV-002 baseline) | 6 | 1 shared | Chaotic (0.34) | 1.0x |
| v2 two-specialist (A + B) | 2 + 4 | 2 independent | Both stable | ~1.3x |
| v2 three-specialist (A + B1 + B2) | 2 + 2 + 2 | 3 independent | All stable | ~1.5x |

**Stop/go**: If two-specialist PAB is stable for both specialists AND combined accuracy ≥ v1 monolithic, the decomposition thesis is validated. If Specialist-A is stable but Specialist-B is still chaotic, proceed to three-specialist. If all specialists are chaotic, the decomposition boundary is wrong — try different bank groupings.

### 6.5 Integration: Full SoM Pipeline

**What**: Wire all five slots together through the Arbiter and run end-to-end proof search.

**How**:
```python
# src/arbiter.py
#
# def som_search(theorem, slots, proof_network, lean_kernel, budget=600):
#   perception = slots.perceive(theorem.goal_state)
#   recognition = slots.recognize(perception)
#   sketch = slots.plan(perception, recognition)
#
#   for subgoal_spec in sketch.subgoals:
#     # Route to appropriate specialist based on bank hints
#     nav_output = slots.execute(perception, subgoal_spec)
#     tactics, premises = resolve(nav_output, proof_network, context)
#
#     for tactic, premise_set in candidates(tactics, premises):
#       result = slots.verify(subgoal_spec.goal_state, lower(tactic, premise_set))
#       if result.success:
#         advance to next subgoal
#         break
#
#   if sketch incomplete and budget remaining:
#     recognition = slots.recognize(perception, retry=True)  # try different template
#     # ... recurse
```

**Validation**: Run on same benchmark set as v1 (MiniF2F subset + Mathlib test split). Compare:
- Theorems proved (v2 SoM vs v1 monolithic)
- Search budget consumed per proof
- Template prediction accuracy during live search
- Specialist PAB stability during live inference

**Statistical significance**: MiniF2F has 488 problems — a 3% difference is ~15 theorems. Use McNemar's test (paired, per-theorem proved/not-proved) for v1 vs v2 comparison. Report p-value alongside point estimate. For differences < 5% (< 25 theorems), bootstrap 95% CI is more informative than p-value alone. Do NOT claim v2 > v1 without p < 0.05.

**Stop/go**:
- v2 ≥ v1 proved rate (p < 0.05) → SoM architecture validated
- v2 ≥ v1 proved rate but not significant → promising but inconclusive; run on larger Mathlib test split for power
- v2 < v1 proved rate but v2 proves different theorems → complementary; run regime analysis (are v2-only theorems predominantly Regime B?)
- v2 << v1 proved rate → SoM overhead exceeds decomposition benefit; run failure analysis (§6.5b)

### 6.5b Structured Failure Analysis

**What**: If v2 underperforms v1, diagnose WHICH slot failed. For every theorem that v1 proves but v2 does not, trace through the SoM pipeline to identify the failure point.

**Protocol**: For each v1-only theorem:
1. Was the **template prediction correct**? Compare RECOGNITION output to the actual tactic sequence pattern. If wrong → RECOGNITION failure.
2. Given the correct template, was the **sketch correct**? Compare PLANNING output to actual subgoal structure. If wrong → PLANNING failure.
3. Given the correct sketch, were the **specialist outputs correct**? Compare EXECUTION directions to ground-truth tactic directions. If wrong → EXECUTION failure (which specialist?).
4. Given correct specialist output, did **resolution/verification** fail? If wrong → VERIFICATION or resolution failure.
5. Was the **arbiter's routing/goal-selection** suboptimal? Compare search tree to v1's search tree.

**Output**: Per-slot failure counts and top failure reasons. This determines where to invest improvement effort — more templates, better sketch predictor, or more specialist capacity.

### 6.6 Ablation Matrix (Stream 4)

| Variant | What it tests |
|---------|--------------|
| v1 monolithic (NAV-002) | Baseline — shared bridge, all banks |
| v2 two-specialist, no templates | Decomposition benefit alone (no narrative conversion) |
| v2 two-specialist, with templates | Decomposition + narrative regime conversion |
| v2 three-specialist, with templates | Finer decomposition |
| v2 with sketch predictor | Full SoM pipeline |
| v2 with LLM sketch (Stage 3) | Maximum narrative capacity |
| Template classifier only (no specialists) | Is template prediction alone valuable? |
| Specialists only (no templates) | Is decomposition alone valuable? |

---

## Phase 7: Energy-Constrained Navigation (Week 16+, v3)

*Phase 7 introduces a new architectural center — negative-boundary learning plus constraint/energy-guided planning — as a parallel v3 runtime. It does NOT modify v1 or v2 code paths. v1 and v2 remain frozen baselines for A/B/C/D comparison.*

*Phase 7 is split by maturity into two tracks:*

- ***v3A (practical, committed)**: Negative data collection, standalone Censor, inference-time pruning, contrastive navigator training, active boundary learning, OTP-derived scoring reforms (bank-IDF, zero-sparsity curriculum). All validated by direct measurement against v2 baselines.*
- ***v3B (experimental, gated on v3A)**: Energy function formalization, continuous ternary relaxation (Gumbel-softmax), energy-based sketch refinement. Ships only after v3A demonstrates value on real proof outcomes.*

*Hard gate: Phase 7 benchmark evaluation does not begin until Phase 6.5 stop/go passes (v2 ≥ v1 on benchmark). v3A graduates from parallel runtime to default only when raw_success(v3A) ≥ raw_success(v2) AND lean_calls/theorem(v3A) < lean_calls/theorem(v2). Note: analytical work (7.1c ternary distribution analysis, 7.1a/b scoring reform design) can proceed in parallel with Phase 6 since they operate on existing training data, not benchmark verification.*

### Execution Waves

| Wave | What | Gate |
|------|------|------|
| **Wave 0** | Phase 6 complete. Shared interfaces defined. `--mode v1\|v2` benchmark. | Phase 6.5 stop/go passes |
| **Wave 1 (v3A)** | Negative data, asymmetric censor, contrastive training, pruning, active boundary learning, OTP scoring reforms. `--mode v3` added. | Censor AUROC ≥ 0.80, raw_success(v3A) ≥ raw_success(v2) |
| **Wave 2 (v3B)** | Energy function, continuous ternary relaxation, sketch refinement loop. | v3A demonstrates value first. Energy-refined ≥ discrete v3A. |

### Parallel Runtime Architecture

v3 is a parallel orchestration path, not a modification of v1/v2. Benchmark runs require explicit mode: `--mode v1|v2|v3`. All modes produce the same top-level metrics schema for comparability.

**Shared interfaces** (defined as dataclasses in `src/contracts.py` during Wave 0):

- **GoalContext**: theorem_id, goal_text, proof_history, accessible_premises, source_split_metadata.
- **ActionCandidate**: tactic, premises, provenance, navigational_scores, template_provenance (optional).
- **NegativeExample**: canonical `nav_negative.jsonl` schema with source, failure_category, paired_positive, split_metadata.
- **ConstraintReport**: bank scores, critic distance, censor score, anchor alignment, total score (or energy for v3B).
- **SketchProposal**: template_id, proposed_steps, latent_form (optional for v3B), total_constraint_score.
- **SearchTrace**: complete audit object for one theorem attempt, including pruning decisions and Lean calls.

### Theoretical Grounding

*This phase synthesizes three theoretical streams:*
- *OTP (Vinaik, 2025): The ternary alphabet {-1, 0, +1} where 0 is not absence but orthogonality — a third informational state. Wayfinder's 6-bank ternary decoder is an OTP projection; bank zeros are Informational Zeros (transparency, not ignorance). The Minority Channel Advantage predicts that sparse bank activations carry disproportionate information.*
- *COEC (Vinaik, 2025): Constraint-Oriented Emergent Computation — specify what a system CANNOT do; behavior emerges from constraint interactions. The bank scores + critic + censor form a constraint system; proof search is trajectory through constrained state space.*
- *Energy-Based Models (cf. Logical Intelligence/Kona, 2026): Define a scalar energy function over entire solutions, then minimize via gradient-based refinement in continuous latent space. Avoids the autoregressive failure mode where sequential commitment prevents revision of earlier decisions.*

*Learning-theoretic framing: Navigation (Slots 1-4) is PAC learning. Verification (Slot 5 + Lean kernel) is an exact oracle. Negative learning exploits the oracle to construct version-space boundaries from both sides. The energy function (v3B) unifies all constraint channels into a single differentiable objective.*

### 7.0 Prerequisites: Core Eval Validity [Wave 0 — gates both v3A and v3B]

**What**: Before investing in Phase 7 work, the positive-only pipeline must produce trustworthy metrics. Otherwise, improvements from negative data or energy refinement could mask underlying bugs.

**Checklist** (gate for Phase 7.1+):
- [ ] Checkpoint loading round-trips correctly (save → load → identical metrics)
- [ ] Real verification path (Lane A: Pantograph step-wise) produces consistent `raw_success`
- [ ] Accessible-premises filtering semantics match ReProver (import-based, not file-based)
- [ ] Train/eval split has zero theorem-ID overlap (verified by `assert set(train_ids) & set(eval_ids) == set()`)
- [ ] Baseline v1 PAB trajectory is reproducible within ±1% on same seed

**Stop/go**: Do NOT proceed to 7.1 until all five items are checked. This gate exists to prevent results from being confounded by pipeline issues.

### 7.1 OTP-Grounded Scoring Reforms (Fast Wins) [v3A — Wave 1]

*Goal: Apply OTP theory to the existing scoring system. These are small, testable changes to `proof_network.py` and `train_navigator.py` that can be validated against Phase 6 baselines immediately. No new modules required.*

#### 7.1a Bank-IDF Weighting (Minority Channel Advantage)

**What**: Weight bank scores inversely by their activation frequency across the training corpus. Banks that rarely fire non-zero for a given entity type carry more information per activation.

**OTP justification**: The Minority Channel Advantage — "When most of a ternary vector is zero, the few non-zero positions carry maximum information density. The rare signal is the valuable signal." Currently, `confidence_weighted` scoring treats all banks equally. MCA says a DECOMPOSITION signal that fires on 5% of proof steps is worth more per-activation than a DOMAIN signal that fires on 80%.

**How**:
```python
# In proof_network.py, compute bank activation frequencies from training data:
#
# For each bank b:
#   freq_b = count(training examples where ternary_target[b] != 0) / total_examples
#   bank_idf_b = log(1 / freq_b)  # rare banks get high IDF
#
# Modified scoring:
#   bank_alignment = Π (bank_score_i ** bank_idf_i)
#
# This is equivalent to the existing confidence_weighted mechanism but with
# IDF-derived exponents instead of softmax confidence. The two can be composed:
#   bank_alignment = Π (bank_score_i ** (confidence_i * bank_idf_i))
```

**Validation**: Compare `confidence_weighted` vs `confidence_weighted + bank_idf` on eval set:
- recall@16 for premise selection (expect improvement on hard theorems)
- nav accuracy per bank (expect hard banks STRUCTURE/AUTOMATION to benefit most)
- Overall raw_success on benchmark (must not degrade)

**Stop/go**: If recall@16 degrades > 2%, revert. MCA may not apply to this bank distribution.

#### 7.1b Zero-Sparsity as Training Signal (Informational Zero)

**What**: Count the number of zero-valued banks per training example. Use this as a difficulty/dimensionality signal for curriculum training.

**OTP justification**: The Informational Zero — "a position orthogonal to the observation axis." A proof step with ternary target `[+1, 0, 0, -1, 0, 0]` (2 active banks, 4 transparent) operates in a lower-dimensional subspace than `[+1, -1, +1, -1, +1, -1]` (6 active banks). Steps with more zeros are "simpler" in the OTP sense — they require fewer simultaneous navigational decisions. This should inform curriculum ordering.

**How**:
```python
# In build_nav_training_data.py, augment each example:
#   zero_count = sum(1 for b in bank_directions.values() if b == 0)
#   example["otp_dimensionality"] = 6 - zero_count  # 1-6, how many banks active
#
# In train_navigator.py, curriculum phases:
#   Phase 1 (warm-up): Train on examples with otp_dimensionality <= 3
#                       (≤3 active banks — simpler navigational decisions)
#   Phase 2 (expansion): Add examples with dimensionality 4-5
#   Phase 3 (full): All examples including dimensionality 6
#
# Hypothesis: this curriculum produces faster PAB convergence because the
# navigator learns low-dimensional projections first, then composes them.
# This parallels OTP's Progressive Revelation principle.
```

**Validation**: Compare standard curriculum (Phase 6 ordering) vs OTP-dimensionality curriculum:
- PAB convergence speed (steps to stability per bank)
- Final nav accuracy (should be equivalent or better)
- Bank-specific accuracy curves (expect hard banks to converge faster with curriculum)

#### 7.1c Ternary Target Distribution Analysis

**What**: Measure the empirical distribution of {-1, 0, +1} across all 6 banks in the training data. Validate OTP predictions about sparsity and minority channel structure.

**Expected findings**:
- Zero (transparent) should be the majority value for most banks — confirming Informational Zero as the dominant state
- Banks with sparser non-zero activation should have higher per-activation discrimination (measurable via nav accuracy conditioned on activation)
- The 729-bin direction space (3^6) should be very sparse — most bins empty, confirming that proof steps cluster in low-dimensional OTP subspaces

**Deliverable**: Distribution report added to `EXPERIMENT_RESULTS.md`. This grounds all subsequent Phase 7 decisions in empirical OTP statistics.

### 7.2 Negative Example Collection [v3A — Wave 1]

**What**: Define a structured negative example format and build three collection pipelines. This is the data foundation for censor training (7.3) and contrastive learning (7.4).

**Data format** — `data/nav_negative.jsonl`:
```json
{
  "goal_state": "⊢ ∀ n : ℕ, n + 0 = n",
  "theorem_id": "Nat.add_zero",
  "step_index": 2,
  "failed_tactic": "simp",
  "failure_reason": "semantic:no_rewrite_rules_match",
  "failure_category": "semantic",
  "source": "sorry_hole|perturbation|suggestion_trace|unchosen_weak",
  "proof_history": ["intro n", "induction n with d hd"],
  "paired_positive_tactic": "exact Nat.rec_aux ...",
  "paired_positive_premises": ["Nat.rec", "Nat.zero"],
  "bank_directions": {"STRUCTURE": 0, "DOMAIN": 1, "DEPTH": -1, "AUTOMATION": 0, "CONTEXT": 1, "DECOMPOSITION": 0},
  "otp_dimensionality": 2
}
```

**Critical label distinctions**:
- `failure_category: "semantic"` — tactic is genuinely wrong (wrong rewrite, type mismatch, inapplicable lemma). These are TRUE negatives. In OTP terms: the tactic's bank position is on the *wrong side* of the observation axis (-1 where +1 was needed).
- `failure_category: "infra"` — timeout, environment mismatch, API error, memory limit. These are NOT valid training signal and MUST be excluded from the loss.
- `failure_category: "weak_negative"` — tactic was unchosen in LeanDojo trace but may still be valid. NOT "invalid" — only "not chosen by the original proof author." Weighted down in training (0.1× loss weight vs 1.0× for semantic failures).

**Split hygiene**: Negative examples inherit the train/eval split from `nav_train.jsonl` / `nav_eval.jsonl` by `theorem_id`. Additionally, negative examples are split by `source` — a model trained on sorry-hole negatives must not be evaluated on sorry-hole-derived eval examples from the same Mathlib version.

**Three collectors** — `scripts/collect_negatives.py`:

```python
# Collector 1: Sorry Holes
# Input: Mathlib git history — commits/PRs with sorry'd intermediate states
# Process:
#   1. Find all sorry occurrences in Mathlib development history
#   2. For each sorry: extract goal state at the sorry site
#   3. Use Axle sorry2lemma to get the lemma statement
#   4. Try all 72 mapped tactics at this goal state via Axle verify_proof
#   5. Record failures (with failure reason from Lean error message)
#   6. If the sorry was later resolved (in a subsequent commit), record the
#      paired positive tactic
# Output: nav_negative.jsonl entries with source="sorry_hole"

# Collector 2: Perturbation + Repair Deltas
# Input: Working proofs from nav_train.jsonl
# Process:
#   1. For each proof: generate perturbations (wrong tactic, wrong argument,
#      swapped steps, omitted step)
#   2. Submit perturbed proof to Axle verify_proof → failure location + error
#   3. Submit to Axle repair_proofs → repair delta (if possible)
#   4. The (perturbation, repair) pair gives a (negative, positive) boundary example
#   5. Classify failure reason from Lean error message
# Output: nav_negative.jsonl entries with source="perturbation"
# Note: Most informative when perturbation is MINIMAL — single-tactic changes
#       produce cleaner boundary signal than wholesale proof corruption

# Collector 3: Suggestion Trace Failures
# Input: Lean 4 elaborator traces from exact?, apply?, simp? invocations
# Process:
#   1. Run suggestion tactics at goal states from nav_eval.jsonl
#   2. Capture the search trace: all candidates tried and their outcomes
#   3. The successful candidate is the positive; all others are negatives
#   4. Enrich with failure reason (type mismatch, universe error, etc.)
# Output: nav_negative.jsonl entries with source="suggestion_trace"
# Note: These are the STRONGEST negatives — the Lean elaborator itself
#       determined invalidity, not human choice
```

**Weak negatives from LeanDojo traces** (separate from the three collectors):
- For each training example in `nav_train.jsonl`, the 71 unchosen tactics (out of 72 total) are candidate weak negatives
- BUT: many unchosen tactics may also be valid at that goal state — `simp` might work even though the author chose `rw`
- Label these as `failure_category: "weak_negative"`, `source: "unchosen_weak"`
- These require NO Lean interaction — they're free from existing data
- Use 0.1× loss weight to prevent them from dominating training signal

**Volume estimates**:
- Sorry holes: ~5-10K examples (depends on Mathlib git history depth)
- Perturbation deltas: ~50K examples (bounded by Axle API quota)
- Suggestion trace failures: ~20K examples (bounded by Lean elaborator runtime)
- Weak negatives: ~23M examples (321K training examples × 71 unchosen, but subsample to ~500K for balance)

**Validation**: Verify label quality on 200 random negatives per source:
- Are "semantic" failures genuinely invalid? (Spot-check by re-running in Lean)
- Are "infra" failures correctly separated? (Should see timeouts, env errors, not semantic issues)
- Are weak negatives plausibly valid? (Run 100 "unchosen" tactics — if > 30% succeed, the 0.1× weighting is justified)

**Stop/go**: If semantic failure rate < 50% of collected negatives (i.e., most failures are infra), the collection pipeline needs better filtering before proceeding to training.

### 7.3 Asymmetric Censor Training (OTP -1 Channel) [v3A — Wave 1]

*Goal: Train the Censor (Slot 5) as a standalone failure predictor, with OTP-grounded asymmetric loss reflecting the Minority Channel Advantage.*

**OTP framing**: The Censor IS the -1 channel of OTP applied to action space. The ternary structure maps directly: +1 = "this tactic is valid" (support), -1 = "this tactic will fail" (contradiction), 0 = "uncertain/irrelevant" (transparent). The Negative Learning principle: "A censor's 'DON'T' is -1. Its silence is 0. The explicit permission is +1."

**Architecture** (lightweight, matches SoM Slot 5 design):
```python
# Input: goal_embedding (384d) + tactic_anchor_embedding (384d) = 768d
# Hidden: Linear(768, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 1) → sigmoid
# Training data: nav_negative.jsonl (failures) + nav_train.jsonl (successes)
# Balance: Undersample successes to 2:1 (neg:pos) ratio — failures are the
#          rare class in training data but the common case in search
```

**Asymmetric loss (MCA-motivated)**:
```python
# Standard BCE treats false positives and false negatives equally.
# MCA says the rare -1 signal carries more bits per symbol.
# Therefore: a missed suppression (FN: -1 classified as 0/+1) should be
# penalized MORE than a false suppression (FP: 0/+1 classified as -1).
#
# Weighted BCE:
#   L = -[w_neg * y * log(p) + w_pos * (1-y) * log(1-p)]
#   where w_neg = 2.0 (missed suppression penalty)
#         w_pos = 1.0 (false suppression penalty)
#
# Rationale: In proof search, a missed suppression wastes an expensive Lean call.
# A false suppression prunes a valid tactic (potentially fatal, but mitigated by
# the safety net: never prune ALL candidates). The 2:1 ratio is conservative;
# sweep {1.5, 2.0, 3.0, 5.0} to find optimal asymmetry.
```

**Metrics**:
- **AUROC** on held-out negative eval set (target: ≥ 0.80)
- **AUPRC** (more informative than AUROC when classes are imbalanced)
- **Calibration**: reliability diagram + expected calibration error (ECE)
- **False-prune rate**: fraction of valid tactics incorrectly rejected. Target: < 5% at operating threshold.
- **Per-source breakdown**: AUROC separately for sorry-hole, perturbation, suggestion-trace, and weak negatives
- **MCA validation**: Compare symmetric BCE vs asymmetric BCE. If asymmetric improves AUPRC without degrading false-prune rate, MCA applies.

**Stop/go**: If AUROC < 0.65, the failure signal is too noisy or the goal+tactic representation is insufficient. Investigate: is the encoder separating valid from invalid tactic applications? (Plot t-SNE of goal+tactic embeddings colored by success/failure.)

### 7.4 Contrastive Navigator Training [v3A — Wave 1]

**What**: Add an auxiliary contrastive loss term to the existing navigational training. For each goal state, push it toward correct tactic/anchor embeddings and away from negative tactic/anchor embeddings by a margin.

**How**:
```python
# Augmented training loss:
#   L_total = L_nav (existing MSE critic + direction losses)
#           + λ_contra * L_contrastive
#           + λ_censor * L_censor (optional, if training censor jointly)
#
# L_contrastive = triplet margin loss:
#   For each training example (goal, positive_tactic):
#     Sample k negative tactics from nav_negative.jsonl at same goal
#     (or at same theorem_id if exact-goal negatives unavailable)
#     L = max(0, d(goal, pos) - d(goal, neg) + margin)
#     where d() is cosine distance in the navigator's bridge embedding space
#
# Margin schedule:
#   Start: margin = 0.1 (gentle separation)
#   Anneal to: margin = 0.3 over 5000 steps (strong separation)
#
# Negative sampling strategy:
#   Hard negatives (most informative): negatives from same theorem_id
#   Semi-hard negatives: negatives from same template class
#   Easy negatives (least informative): random negatives from other theorems
#   Mix: 50% hard, 30% semi-hard, 20% easy (curriculum over training)
#
# Loss weighting by source:
#   semantic failures: 1.0×
#   suggestion_trace: 1.0×
#   perturbation: 0.8×
#   weak_negative (unchosen): 0.1×
```

**Key constraint**: The contrastive term must NOT degrade positive-only performance. Measure:
- v1 baseline metrics (nav accuracy, PAB stability) with λ_contra = 0
- v1 + contrastive metrics with λ_contra ∈ {0.01, 0.05, 0.1, 0.2}
- If any λ_contra degrades nav accuracy > 1%, that value is too high

**Per-bank impact analysis**: Report contrastive loss improvement broken down by bank. Hypothesis: STRUCTURE and AUTOMATION (hard banks, Tier 2) should benefit most from boundary information, since their decision boundaries are least well-defined by positive-only training.

**Stop/go**: If contrastive training degrades positive-only performance at all tested λ values, the negative data quality is suspect — return to 7.2 and audit labels.

### 7.5 Energy Function Formalization [v3B — Wave 2, gated on v3A]

*Goal: Define a composite, differentiable energy function that unifies all of Wayfinder's constraint channels into a single scalar. This is the theoretical foundation for energy-based sketch refinement (7.6) and provides a formal connection between OTP/COEC theory and the proof search objective.*

*Motivation (Logical Intelligence/Kona): Energy-based models evaluate entire solutions against all constraints simultaneously, producing a scalar energy where low = satisfied constraints. Gradient-based refinement in continuous latent space finds valid configurations without sequential commitment. Kona achieves 96.2% on Sudoku at 313ms vs 2% for frontier LLMs — because it evaluates holistically rather than committing token-by-token.*

**Energy function definition**:

```python
# E(sketch) : latent proof sketch → scalar energy
#
# Components (all differentiable):
#
# 1. Bank constraint energy — how well does the sketch satisfy bank alignment?
#    E_bank = Σ_i (1 - bank_score(sketch_position_i, target_direction_i))
#    Bank scores are cosine similarities → differentiable
#    Informational Zero: banks where sketch_position_i = 0 contribute ZERO energy
#    (transparent banks don't push the gradient — the optimization naturally
#    operates in a lower-dimensional subspace per proof step)
#
# 2. Critic distance energy — how far is the sketch from proof closure?
#    E_critic = critic_distance(sketch)
#    The critic predicts remaining steps via MSE — already differentiable
#
# 3. Censor violation energy — does the sketch contain forbidden actions?
#    E_censor = Σ_steps max(0, censor_score(step) - threshold)
#    The censor's sigmoid output is differentiable
#    This is the -1 channel of OTP: explicit constraint violation
#
# 4. Anchor misalignment energy — does the sketch match known proof patterns?
#    E_anchor = 1 - (Σ idf(shared_anchors) / Σ idf(union_anchors))
#    IDF-weighted Jaccard → differentiable approximation via soft matching
#
# Composite energy with learned weights:
#    E(sketch) = α·E_bank + β·E_critic + γ·E_censor + δ·E_anchor
#
# Initial weights: α=1.0, β=0.5, γ=2.0 (high penalty for forbidden actions,
#                  consistent with MCA — the rare censor signal is valuable),
#                  δ=0.3
# Weights are hyperparameters, swept in ablation (7.9).
```

**Continuous ternary relaxation** — the bridge from discrete OTP to differentiable optimization:

```python
# The ternary decoder outputs discrete {-1, 0, +1} per bank.
# For gradient-based refinement, we need a continuous relaxation.
#
# Gumbel-Softmax over 3 categories {-1, 0, +1}:
#   logits = decoder_output  # shape: (num_steps, 6, 3)
#   soft_ternary = gumbel_softmax(logits, tau=temperature)
#   continuous_position = soft_ternary @ [-1, 0, +1]  # weighted sum → continuous
#
# Temperature schedule during refinement:
#   Start: tau = 1.0 (soft, allows gradient flow through all options)
#   Anneal to: tau = 0.1 over N refinement steps (hardens toward discrete)
#   Final: snap to argmax → discrete {-1, 0, +1}
#
# Key property: at high temperature, the Informational Zero (0 category)
# naturally dominates because it's the entropy-maximizing choice.
# As temperature decreases, non-zero activations sharpen — this IS
# OTP's Progressive Revelation in the energy landscape.
```

**Deliverable**: `src/energy.py` — EnergyFunction class composing the four energy terms with configurable weights. Unit tests verifying differentiability (torch.autograd.gradcheck).

### 7.6 Energy-Based Sketch Refinement [v3B — Wave 2, gated on v3A]

*Goal: Add an energy-based refinement loop to the SoM inference pipeline. Instead of scoring discrete candidates sequentially, generate a full proof sketch in continuous latent space, then gradient-descend to minimize the energy function before decoding to tactics.*

*This is where COEC meets EBM: "Instead of specifying what a system should DO, specify what it CANNOT do. Behavior emerges from the interaction of constraints." The energy function defines violations; the gradient finds solutions.*

**Insertion point**: `src/sketch_predictor.py` (existing module from Phase 6.3).

**Refinement loop**:
```python
# In src/sketch_predictor.py, add energy_refine() method:
#
# def energy_refine(self, goal_state, template, energy_fn, config):
#     """Refine a proof sketch via energy minimization in continuous space."""
#
#     # 1. Initial sketch: template → continuous latent (existing sketch_predictor)
#     sketch_latent = self.predict_sketch(goal_state, template)
#     # shape: (max_steps, 6) — continuous bank positions per step
#
#     # 2. Enable gradient tracking
#     sketch_latent.requires_grad_(True)
#     optimizer = torch.optim.Adam([sketch_latent], lr=config.refine_lr)  # 0.01
#
#     # 3. Refinement loop
#     for step in range(config.refine_steps):  # default: 20
#         optimizer.zero_grad()
#
#         # Compute composite energy
#         energy = energy_fn(sketch_latent, goal_state)
#
#         # Gradient step
#         energy.backward()
#         optimizer.step()
#
#         # Anneal Gumbel temperature
#         tau = config.tau_start * (config.tau_end / config.tau_start) ** (
#             step / config.refine_steps)
#
#         # Early exit if energy below threshold
#         if energy.item() < config.energy_threshold:  # 0.1
#             break
#
#     # 4. Snap to discrete ternary
#     discrete_sketch = snap_to_ternary(sketch_latent)
#
#     # 5. Resolve through proof network (existing navigate() path)
#     tactics = [proof_network.navigate(step_direction) for step_direction in discrete_sketch]
#
#     return tactics, energy.item()
```

**Integration with SoM arbiter** — `src/arbiter.py`:
```python
# Modified som_search() with energy refinement:
#
# 1. Template classifier selects top-k templates (existing)
# 2. For each template:
#    a. sketch_predictor.energy_refine(goal, template, energy_fn, config)
#    b. Returns refined tactics + final energy
# 3. Select template with lowest final energy (not highest confidence)
# 4. Execute refined tactics via Lean (Lane A verification)
# 5. If verification fails: retry with next template (existing retry logic)
#
# Key difference from current arbiter: template selection uses energy
# (holistic constraint satisfaction) rather than critic score alone.
# The energy subsumes the critic — it includes critic distance as one term
# alongside bank alignment, censor violations, and anchor matching.
```

**Why this addresses the Kona insight**: Current Wayfinder generates candidate tactics one-at-a-time, scores them individually, and commits sequentially — the same failure mode Kona identifies in LLMs on Sudoku. Energy refinement generates a *complete sketch*, evaluates *all constraints simultaneously*, and revises *any part* of the sketch that violates constraints. If step 3 of a sketch creates a conflict with step 7, the gradient naturally flows back to adjust step 3 — something sequential search cannot do without expensive backtracking.

**Compute budget**: Energy refinement adds ~20 gradient steps per template attempt. Each gradient step is a forward pass through the energy function (bank scores + critic + censor + anchor matching) — all lightweight operations on the existing model's outputs. Estimated overhead: ~50ms per template on MPS, vs ~500ms for a Lean verification call. The refinement is amortized: if it produces better sketches, total Lean calls decrease.

**Config** — added to `configs/wayfinder_v2.yaml`:
```yaml
energy_refinement:
  enabled: true
  refine_steps: 20
  refine_lr: 0.01
  tau_start: 1.0      # Gumbel temperature start
  tau_end: 0.1         # Gumbel temperature end
  energy_threshold: 0.1  # early exit
  weights:
    bank: 1.0
    critic: 0.5
    censor: 2.0        # MCA: high penalty for forbidden actions
    anchor: 0.3
```

**Stop/go**: If energy-refined sketches produce lower raw_success than unrefined sketches (discrete scoring only), the energy function is miscalibrated. Diagnose: which energy component dominates? Is the continuous relaxation losing information during snap-to-ternary? Is the gradient vanishing through the Gumbel-softmax?

### 7.7 Active Boundary Learning (Axle Loop) [v3A — Wave 1]

**What**: Instead of passively collecting negatives, use the trained navigator to generate boundary-adjacent negative examples via Axle. This is Angluin-style active learning with an exact oracle.

**Energy-aware selection**: In addition to Censor uncertainty (confidence ∈ [0.4, 0.6]), use the energy function to identify high-energy sketches — proof attempts where the constraint system identifies violations but the navigator is uncertain which constraint is binding. These are the most informative oracle queries because they calibrate the energy function's component weights.

**How**:
```python
# Active learning loop:
#
# 1. Navigator proposes top-k candidates at each goal state
# 2. Score candidates via energy function AND censor
# 3. Identify UNCERTAIN candidates by two criteria:
#    a. Censor uncertainty: confidence ∈ [0.4, 0.6]
#    b. Energy disagreement: high bank energy but low censor energy (or vice versa)
#       — indicating component miscalibration
# 4. Submit uncertain candidates to Axle verify_proof
# 5. Axle returns exact pass/fail + error message
# 6. These boundary-adjacent examples are added to nav_negative.jsonl
#    (or nav_train.jsonl if they succeed!)
# 7. Retrain Censor + navigator + recalibrate energy weights on augmented data
# 8. Repeat until convergence
#
# Convergence criteria:
#   - Uncertainty fraction decreases by ≥ 10% per iteration
#   - Energy component disagreement decreases (components are calibrating)
#   - OR raw_success plateaus
#
# Budget: bounded by Axle API quota per iteration (N = 500)
# Plan for 5-10 iterations to see convergence signal
```

**Why boundary cases matter**: Random negatives are informative but redundant — most random tactics obviously fail at most goal states. The *interesting* cases are where the navigator is uncertain. Active learning theory (Cohn et al.) shows that querying at the decision boundary produces exponentially faster convergence than random sampling. The energy function adds a second dimension: cases where individual constraint channels disagree are maximally informative for calibrating the composite energy.

### 7.8 Inference-Time Integration [v3A pipeline + v3B extension]

**What**: The Phase 7 inference pipeline exists in two variants. v3A ships first; v3B extends it with energy refinement after v3A demonstrates value.

**v3A pipeline** (discrete scoring + censor pruning):
```python
# v3A inference pipeline (in src/proof_search.py, --mode v3):
#
# 1. PERCEPTION: Encode goal state (existing)
#    goal_embedding = encoder.encode(goal_state)
#
# 2. OTP SCORING: Navigate with bank-IDF weighting (7.1a)
#    candidates = proof_network.navigate(goal_embedding, bank_idf_weights)
#
# 3. CENSOR PRUNING: Asymmetric censor filters candidates (7.3)
#    scored = censor.score(goal_state, candidates)
#    pruned = [c for c in scored if c.censor_confidence >= threshold]
#    if len(pruned) == 0:
#        pruned = top_k(scored, k=3)  # safety net: never prune ALL
#
# 4. TEMPLATE SELECTION: Classify proof template (Phase 6.2)
#    templates = template_classifier.predict_top_k(goal_embedding, k=3)
#
# 5. DISCRETE SKETCH: Predict sketch, score via ConstraintReport (no energy)
#    for template in templates:
#        sketch = sketch_predictor.predict(goal_state, template)
#        report = constraint_report(sketch, bank_scores, critic, censor, anchors)
#        if report.total_score > best_score:
#            best_sketch, best_score = sketch, report.total_score
#
# 6. EXECUTION: Submit best tactics to Lean (Lane A)
#    result = lean_kernel.apply_sequence(best_sketch.tactics)
#
# 7. RETRY: If failed, try next template (existing arbiter retry logic)
```

**v3B pipeline** (extends v3A with energy refinement — gated on v3A success):
```python
# v3B inference pipeline (replaces steps 5-6 above):
#
# 5. ENERGY REFINEMENT: For each template, refine sketch in continuous space (7.6)
#    for template in templates:
#        tactics, energy = sketch_predictor.energy_refine(
#            goal_state, template, energy_fn, config)
#        if energy < best_energy:
#            best_tactics, best_energy = tactics, energy
#
# 6. EXECUTION: Submit best-energy tactics to Lean (Lane A)
#    result = lean_kernel.apply_sequence(best_tactics)
```

**Key difference**: v3A uses discrete `ConstraintReport.total_score` for template selection. v3B replaces this with continuous energy minimization + snap-to-ternary. Both share steps 1-4 (perception, OTP scoring, censor pruning, template selection).

### 7.9 Validation Plan: A/B/C/D Comparison [v3A: A→B, v3B: C→D]

**What**: Prove that each Phase 7 component produces genuine improvement. Conditions A through B are v3A validation; C and D are v3B validation (gated on v3A success).

**Experimental conditions**:

| Condition | Track | OTP Scoring | Censor | Energy Refinement | Active Learning |
|-----------|-------|------------|--------|-------------------|-----------------|
| **A (Phase 6 Baseline)** | — | No | No | No | No |
| **A+ (OTP Scoring Only)** | v3A | Bank-IDF + zero-curriculum | No | No | No |
| **B (+Passive Negatives)** | v3A | Yes | Asymmetric, trained | No | No |
| **B+ (+Active Boundary)** | v3A | Yes | Yes | No | Yes (5 iterations) |
| **C (+Energy Refinement)** | v3B | Yes | Yes | Yes (20 steps) | No |
| **D (+Energy + Active)** | v3B | Yes | Yes | Yes | Yes (5 iterations) |

**v3A stop/go**: B+ must show raw_success(B+) ≥ raw_success(A) AND lean_calls/theorem(B+) < lean_calls/theorem(A). If B+ fails this gate, v3B does not proceed.

**v3B stop/go**: D must show raw_success(D) ≥ raw_success(B+). If energy refinement degrades proof rate vs discrete v3A, it remains experimental and off by default.

**Primary metrics** (reported for all conditions):

| Metric | What it measures | Expected ordering |
|--------|-----------------|-------------------|
| `raw_success` | Theorems proved (Lane A only) | D ≥ C ≥ B ≥ A+ ≥ A |
| `attempts/theorem` | Search efficiency | D ≤ C ≤ B ≤ A+ ≤ A |
| `lean_calls/theorem` | Verification cost reduction | D ≤ C ≤ B ≤ A |
| `energy_at_submission` | Constraint satisfaction quality (C, D) | C, D << A, B |
| `false_prune_rate` | Censor safety (B, C, D only) | < 5% |

**OTP-specific metrics** (A+ and beyond):

| Metric | What it measures |
|--------|-----------------|
| Bank-IDF recall@16 | MCA validation: does inverse-frequency weighting help? |
| Zero-sparsity correlation | IZ validation: are sparser examples easier? |
| Per-bank nav accuracy gain | Which banks benefit from each Phase 7 component? |
| Ternary distribution entropy | Does energy refinement produce sparser (more OTP-aligned) outputs? |

**Energy-specific metrics** (C, D only):

| Metric | What it measures |
|--------|-----------------|
| Refinement convergence | Steps to energy < threshold (expect ~10-15 of 20) |
| Energy component breakdown | Which term dominates? Bank vs critic vs censor vs anchor |
| Snap-to-ternary fidelity | How much energy increases when continuous → discrete |
| Holistic vs sequential comparison | Energy-refined sketch vs best sequential candidate on same goals |

**Per-bank impact** (critical for thesis validation):
- Report all metrics broken down by bank, especially STRUCTURE, AUTOMATION, DEPTH (Tier 2 hard banks)
- Hypothesis: energy refinement disproportionately improves hard banks because the gradient can adjust multiple banks simultaneously — something sequential scoring cannot do

**Statistical protocol**: McNemar's test for proved/not-proved, bootstrap 95% CI for continuous metrics, p < 0.05 threshold for claims.

**Contamination controls**:
- Verify zero theorem-ID overlap between training negatives and eval set
- Verify zero source overlap (sorry-hole train negatives are not from same Mathlib commits as eval sorry-hole negatives)
- Report metrics on a held-out "clean" eval subset that shares zero Mathlib files with any negative source

### 7.10 Ablation Matrix (Energy-Constrained Navigation) [v3A ablations first, v3B ablations gated]

| Variant | Track | What it tests |
|---------|-------|--------------|
| **OTP Scoring (v3A)** | | |
| A: Phase 6 baseline (no OTP) | — | Current pipeline ceiling |
| A+idf: +bank-IDF weighting only | v3A | Does MCA improve scoring? |
| A+cur: +zero-sparsity curriculum only | v3A | Does IZ curriculum help convergence? |
| A+: +both OTP reforms | v3A | Combined OTP scoring value |
| **Negative Learning (v3A)** | | |
| B1: +sorry negatives only | v3A | Is incomplete-proof signal sufficient? |
| B2: +perturbation negatives only | v3A | Is synthetic boundary signal sufficient? |
| B3: +suggestion trace negatives only | v3A | Is elaborator-verified signal sufficient? |
| B4: +weak negatives only (unchosen) | v3A | Is free unchosen signal useful at all? |
| B: All passive negatives | v3A | Combined passive signal |
| B-sym: B with symmetric censor loss | v3A | Does MCA-asymmetric loss help? |
| B+censor, no contrastive | v3A | Censor-only pruning (no navigator change) |
| B+contrastive, no censor | v3A | Navigator change only (no pruning) |
| B+active-5: Active boundary (5 iterations) | v3A | Standard active learning |
| B+active-10: Active boundary (10 iterations) | v3A | Diminishing returns check |
| **Energy Refinement (v3B — gated on v3A)** | | |
| C-5: Energy refinement, 5 steps | v3B | Minimal refinement budget |
| C-20: Energy refinement, 20 steps | v3B | Default refinement budget |
| C-50: Energy refinement, 50 steps | v3B | Diminishing returns check |
| C-nobank: Energy w/o bank term | v3B | Is bank constraint energy necessary? |
| C-nocensor: Energy w/o censor term | v3B | Is censor violation energy necessary? |
| C-noanchor: Energy w/o anchor term | v3B | Is anchor matching energy necessary? |
| C-critic-only: Energy = critic only | v3B | Is the composite energy better than critic alone? |
| C-nosnap: Continuous output (no ternary snap) | v3B | Does discrete snapping lose information? |
| D-energy: Active selection by energy disagreement | v3B | Energy-aware vs censor-only uncertainty |

---

## Definition of Ready to Run

The primary Wayfinder config works end-to-end:

- [x] `proof_network.db` populated with ~98k entities (78,414 entities, 18,810 anchors)
- [x] Anchor gap analysis: top-16 recall >= 70% on perfect queries (100% achieved)
- [x] Encoder chosen and integrated (MiniLM-L6-v2, 384d, frozen)
- [ ] Full pipeline forward+backward works on real data
- [ ] Training runs 3000+ steps with decreasing loss, improving nav accuracy
- [ ] Proof search proves >= 5 trivial theorems end-to-end
- [ ] Benchmark runner produces metrics on MiniF2F subset
- [ ] Core tests pass (proof network, navigator, resolution, search, training)
- [ ] Reproducible within tolerance (same seed -> metrics within +/- 1%)

**Post-readiness hardening:**
- [ ] All 15 ablation configs load, train, and evaluate
- [ ] PAB trajectory comparison tooling
- [ ] Full test coverage for edge cases

**v2 (Society of Mind) readiness:**
- [x] v1 monolithic navigator trained (NAV-001/002), chaotic PAB confirmed
- [x] Bank difficulty hierarchy measured (Regime A vs Regime B)
- [x] Specification complexity theory documented (RESEARCH §2.9)
- [x] SoM architecture designed (DESIGN §10)
- [ ] Template taxonomy extracted from proof corpus (Phase 6.1)
- [ ] Template classifier trained, top-3 accuracy ≥ 85% (Phase 6.2)
- [ ] Two-specialist decomposition trained, both PAB stable (Phase 6.4)
- [ ] Full SoM pipeline end-to-end (Phase 6.5)
- [ ] v2 ≥ v1 on benchmark (Phase 6.5 stop/go)

**v3A (Boundary Learning) readiness:**
- [ ] Core eval validity gate passed (Phase 7.0 checklist)
- [ ] OTP ternary distribution analysis complete, MCA predictions validated (Phase 7.1c)
- [ ] Bank-IDF weighting validated: recall@16 ≥ baseline (Phase 7.1a)
- [ ] nav_negative.jsonl populated (≥ 10K semantic negatives across 3 sources) (Phase 7.2)
- [ ] Label quality verified (200 spot-checks per source, semantic vs infra separation confirmed)
- [ ] Asymmetric censor trained, AUROC ≥ 0.80, false-prune rate < 5% (Phase 7.3)
- [ ] Contrastive training does not degrade positive-only performance (Phase 7.4)
- [ ] Active boundary loop converges (uncertainty shrinks ≥ 10%/iteration) (Phase 7.7)
- [ ] raw_success(v3A) ≥ raw_success(v2) AND lean_calls/theorem(v3A) < lean_calls/theorem(v2)

**v3B (Energy Refinement) readiness — gated on v3A:**
- [ ] v3A demonstrates value (above gate passes)
- [ ] Energy function differentiable, gradcheck passes (Phase 7.5)
- [ ] Continuous ternary relaxation: snap-to-ternary fidelity loss < 10% energy increase (Phase 7.5)
- [ ] Energy-refined sketches ≥ discrete v3A scoring on raw_success (Phase 7.6)
- [ ] D ≥ B+ on raw_success in A/B/C/D comparison (Phase 7.9)
- [ ] If v3B fails gate: remains experimental, off by default. v3A is the shipping v3 runtime.

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
| Run benchmark | `python -m scripts.run_benchmark --config configs/wayfinder.yaml --checkpoint runs/<run-id>/checkpoint.pt` |
| Record results | `docs/EXPERIMENT_RESULTS.md` |
| Understand SoM architecture (v2) | `docs/WAYFINDER_DESIGN.md` Section 10 |
| Understand specification complexity theory | `docs/WAYFINDER_RESEARCH.md` Section 2.9 |
| Understand template taxonomy | `data/template_taxonomy.json` (Phase 6.1 output) |
| Train template classifier | `python -m scripts.train_template_classifier --config configs/wayfinder.yaml --run-id TC-001` |
| Train bank-cluster specialist | `python -m scripts.train_specialist --config configs/wayfinder.yaml --specialist A --run-id SP-A-001` |
| Run SoM proof search | `python -m scripts.run_benchmark --config configs/wayfinder_v2.yaml --checkpoint runs/<som-run>/checkpoint.pt` |
| Extract templates from corpus | `python -m scripts.extract_templates` |
| Collect negative examples | `python -m scripts.collect_negatives --source sorry_holes,perturbation,suggestion_trace` *(v3A — not yet implemented)* |
| Train Censor (failure predictor) | `python -m scripts.train_censor --negatives data/nav_negative.jsonl` *(v3A — not yet implemented)* |
| Run benchmark (mode-explicit) | `python -m scripts.run_benchmark --config configs/wayfinder.yaml --mode v1\|v2\|v3 --checkpoint runs/<run-id>/checkpoint.pt` |
| Understand v3 boundary learning | `docs/WAYFINDER_PLAN.md` Phase 7 (v3A: 7.0-7.4, 7.7) |
| Understand v3 energy refinement | `docs/WAYFINDER_PLAN.md` Phase 7 (v3B: 7.5-7.6, gated on v3A) |
| Understand v3 runtime architecture | `docs/WAYFINDER_DESIGN.md` Section 12 *(planned)* |
| Understand OTP/EBM theory | `docs/WAYFINDER_RESEARCH.md` Section 2.11 *(planned)* |

## Key Data Files

| File | Purpose |
|------|---------|
| `data/leandojo_mathlib.jsonl` | LeanDojo extracted Mathlib dataset (~98k theorems) |
| `data/proof_network.db` | SQLite semantic network of mathematical entities |
| `data/nav_train.jsonl` | Navigational training data (321K examples, 1.8GB) |
| `data/nav_eval.jsonl` | Navigational evaluation data (frozen) |
| `data/nav_negative.jsonl` | Negative training data (failed tactics, sorry holes, perturbation deltas) |
| `configs/wayfinder.yaml` | Primary experiment configuration |

## Invariants

1. **`data/nav_eval.jsonl` is frozen.** Never modify the evaluation set.
2. **The proof network is populated and gap-analyzed before training begins.** Bank positions, anchors, IDF, accessible-premises sets, and anchor gap analysis (Phase 0.5) must be complete before the first training step. Top-16 recall on perfect queries must exceed 70%.
3. **Neural inference happens once per proof state.** The resolution and search layers must not call neural forward passes. (Exception: hammer delegation calls external ATPs, which is allowed.)
4. **All retrieval scores are auditable.** Every premise retrieval must trace to specific shared anchors and bank alignments.
5. **PAB tracking is mandatory from step 0.** Retrofit trajectory analysis produces incomplete data.
6. **Scoring mechanism is configurable.** The scoring composition (multiplicative, confidence-weighted, etc.) is a config parameter, not hardcoded. Default: `confidence_weighted`.
7. **Encoder selection is complete.** Primary: `all-MiniLM-L6-v2` (384d, sep=0.587, 617 goals/sec). Ablation candidates: `leandojo-lean4-retriever-byt5-small` (1472d, sep=0.623) and `pplx-embed-v1-0.6b` (1024d, sep=0.600). See EXP-0.3/0.3b in EXPERIMENT_RESULTS.md for full 15-model evaluation.
8. **6-bank navigation is default.** All 6 banks are navigable. Graceful degradation to 3-bank is available via `navigable_banks` config but should not be the starting point.
9. **Critic uses soft targets.** MSE on normalized distance-to-completion, NOT binary BCE. This is a hard constraint per HTPS and AlphaProof findings. Binary critic is an ablation variant only.
10. **Accessible-premises filtering is always on.** Pre-filter to accessible premises before scoring. Free ~2% recall gain from ReProver. **Current implementation note:** `accessible_premises` in the proof network DB are derived from *used* premises (LeanDojo trace data), not full import-closure. This is a strict subset — valid but conservative. Full import-accessible computation requires Lean environment introspection (Phase 2+). The benchmark must pass `accessible_theorem_id` to `search()` for filtering to take effect.
11. **Hammer delegation for AUTOMATION = -1.** When the navigator predicts fully decidable, delegate to LeanHammer/Aesop before attempting navigational resolution.
12. **PAB stability guides decomposition (v2).** Specialist scope is determined by PAB stability_regime, not hand-tuned. Chaotic → decompose further. Stable → lock scope.
13. **Specialists share data, not weights (v2).** Slots communicate through typed contracts (navigational coordinates, template IDs, proof sketches), never through shared learned representations. This keeps γ ≈ 0.
14. **Template classifier is Regime A (v2).** The RECOGNITION slot must have PAB "stable" — if it's chaotic, the template taxonomy is wrong, not the classifier capacity.
15. **Negative labels distinguish semantic from infra failures (v2.1).** A tactic that times out or hits an environment mismatch is NOT a valid negative example. Only semantic failures (type mismatch, no rewrite rules, inapplicable lemma) enter the contrastive loss. Infrastructure failures are logged but excluded from training.
16. **Unchosen ≠ invalid (v2.1).** Tactics not chosen in LeanDojo traces are weak negatives only (0.1× loss weight). Many unchosen tactics are valid alternatives — treating them as hard negatives would corrupt the version-space boundaries.
17. **Negative examples inherit split hygiene (v2.1).** Train/eval split for negatives follows `theorem_id` from the positive split. Additionally, negatives are split by `source` to prevent leakage between collection methodology and evaluation.
18. **Censor false-prune rate < 5% at operating threshold (v2.1).** A Censor that prunes valid tactics is worse than no Censor. The safety net (never prune ALL candidates) is mandatory.
