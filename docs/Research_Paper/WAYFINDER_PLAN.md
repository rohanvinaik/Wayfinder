# Wayfinder: Operational Research Plan

**Version:** 2.4
**Date:** March 19, 2026
**Corresponding documents:** `WAYFINDER_RESEARCH.md` (theory), `WAYFINDER_DESIGN.md` (engineering), `EXPERIMENT_RESULTS.md` (results ledger)
**Status:** Canonical execution plan for `docs/Research_Paper/`. v1 implementation complete. Phase 0 data pipeline complete. NAV-001/002 trained (monolithic navigator, chaotic PAB). Pivot to Society of Mind architecture (v2) is complete at the design level; the current execution branch is to keep `cosine_rw` plus interleaved bootstrap as the frozen deployed local baseline, keep source-context compilation (`ContextIR`) as the parallel infrastructure track, and move the learned frontier above rewrite-family execution to executable action selection, controller-visible move typing, residual-conditioned orchestration, and specialist-specific training regimes once those specialist contracts are validated.

---

## Overview

This document specifies the sequence of work, tooling, compute decisions, and stop/go criteria for running the Wayfinder experiments. It is the authoritative plan for the `docs/Research_Paper/` folder. If `docs/PLAN_2.md` or scratch notes diverge from this file, this file wins.

This plan covers data pipeline execution and experimental evaluation. The current execution state is:

- canonical local-execution data is built and frozen
- the semantic `rw0` benchmark definition is frozen
- step-0 `rw0` is solved operationally by qualified-name cosine beam search
- step>0 replay is usable for semantic evaluation, and replayed states support cosine-ranked `rw0`
- `rw1` is operationally equivalent to `rw0` under tier-conditioned default direction
- theorem-faithful Tier B start states are validated for step-0 and usable for step>0 base states
- live Lean verification now validates executable action selection for `apply` (`selector_top1 = 35/91` vs `cosine_top1 = 15/91`)
- the next runtime step is not another always-on lane; it is expanding the provenance-aware executable dataset, deploying the `apply` selector into theorem search, and then extending the same regime to `refine_named`
- `ContextIR` is the parallel infrastructure track for replay coverage and future local families
- rewrite-family local execution is collapsed enough that learning now shifts above it: controller-visible move typing, residual-conditioned orchestration, and executable action selection over shortlists
- full SoM training remains downstream of multiple validated specialist regimes, not the next immediate milestone

This plan produces five categories of results:

- **Stream 1 (Navigation)**: Does structured semantic navigation — bank positions, IDF-weighted anchors, spreading activation — outperform dense embedding retrieval for premise selection and proof search?

- **Stream 2 (Architecture)**: Does a ternary navigational decoder that produces directional coordinates, resolved through a semantic network, outperform tactic-token classification?

- **Stream 3 (Process Evaluation)**: Does PAB's trajectory evaluation reveal information about the navigational learning process that endpoint metrics cannot?

- **Stream 4 (Decomposition)**: Does a Society of Mind architecture — multiple specialists with PAB-guided scope, narrative template classification, and typed temporal slots — outperform a monolithic navigator? Does specification complexity theory correctly predict when and how to decompose?

- **Stream 5 (Boundary Learning + Energy Refinement)**: Does incorporating structured negative examples — failed tactics, incomplete proofs, and actively-generated boundary cases — transform positive-only imitation into boundary-aware reasoning? Does an energy-based refinement loop, grounded in Orthogonal Ternary Projection theory (Vinaik, 2025) and energy-based constraint satisfaction (cf. Logical Intelligence/Kona, 2026), enable holistic proof sketch optimization that outperforms sequential candidate scoring?

Every phase states what it validates in each stream. The experimental pipeline is designed so that a single set of experiments tests all five claims. Results are recorded in `docs/Research_Paper/EXPERIMENT_RESULTS.md`.

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
#   6. Attach optional canonical move metadata:
#      - theorem_key = file_path::theorem_id (stable against duplicate names)
#      - local_family
#      - subtask_kind / subtask_summary / subtask_effect
#      - goal_target_head
#      - trigger_signature
#   7. Write to JSONL
```

**Output**: `data/nav_training.jsonl` with per-step move metadata, plus any frozen eval split

**Format**:
```json
{
  "goal_state": "⊢ ∀ x : ℕ, x + 0 = x",
  "theorem_id": "Nat.add_zero",
  "theorem_key": "Mathlib/Data/Nat/Basic.lean::Nat.add_zero",
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
*Lane A status: Pantograph backend is implemented and operational for local verification. Current gap is no longer kernel access; it is theorem-site source-context fidelity for step>0 local execution. Lane B (Axle) remains the cloud verification / repair lane.*

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

### 4.6 Multi-Pass Landmark Retrieval (Implemented)

*Goal: Replace single-pass anchor-based retrieval with a multi-pass landmark mesh that fuses heterogeneous scoring signals, classifies convergence, and supports incremental freeze/residual refinement. This work was done between Phase 2 and Phase 6, filling the gap between anchor gap analysis (Phase 0.5) and SoM architecture (Phase 6). In the current architecture this retrieval stack should be interpreted primarily as theorem-level orchestration and frontier collapse, not as a direct solver for post-structural premise-step targets.*

*Status: Implemented and validated. Multi-pass retrieval improves recall@16 from 26.7% to 43.3% (+17.6pp paired, seed=42, 164 samples, v3 DB). Lens guidance layer implemented but marginal (within +/- 2.7pp due to empty residuals from aggressive freezing).*

#### 4.6.1 Multi-Pass Landmark Mesh

**What**: Four selectors score each candidate landmark from orthogonal perspectives, then results are fused via signed Reciprocal Rank Fusion (RRF) with ternary voting.

**Selectors** (each returns a ranked list with scores):

| Selector | Signal | Rationale |
|----------|--------|-----------|
| `bridge_potential` | Semantic similarity between query embedding and candidate entity | Baseline dense retrieval signal |
| `self_match` | Entity self-reference and structural centrality | Finds landmarks that are well-connected in the proof network |
| `accessibility` | Import-graph reachability from the current theorem | Filters to premises that are actually accessible (extends Phase 0.5 accessible-premises) |
| `hub_suppressor` | Inverse document frequency / hub penalty | Down-weights overly generic lemmas (e.g., `rfl`, `trivial`) that appear in many proofs but carry low information |

**Fusion**: Signed RRF assigns each selector a ternary vote (+1 promote, 0 abstain, -1 suppress) per candidate. Votes are weighted by selector confidence and fused via reciprocal rank aggregation. This generalizes standard RRF to handle negative signals (hub suppression) natively.

**Convergence classification**: After fusion, each candidate is classified as:
- **Converged** — Multiple selectors agree (high inter-selector concordance)
- **Isolated** — Only one selector promotes, others abstain
- **Conflicted** — Selectors disagree (one promotes, another suppresses)

Converged candidates are high-confidence retrievals. Isolated candidates may indicate novel connections. Conflicted candidates are deferred to the residual pipeline.

#### 4.6.2 Freeze/Residual Pipeline

**What**: Converged landmarks are frozen (committed as high-confidence retrievals). Remaining candidates enter the residual pipeline for further refinement in subsequent passes.

**Components**:
- `FrozenLandmarkState` — Immutable record of frozen landmarks with provenance (which selectors agreed, at what confidence)
- `LandmarkResidualReport` — Describes unfrozen candidates: their scores, convergence status, and which selectors disagree. This report is the input to the lens guidance layer.

**Behavior**: The freeze threshold is configurable. Aggressive freezing (low threshold) commits more candidates early but leaves fewer residuals for the lens layer to refine — this is the current bottleneck (see results below).

#### 4.6.3 Lens Guidance Layer

**What**: Five specialist "lenses" provide additional scoring signals on residual (unfrozen) candidates. The coherence engine combines lens outputs as modulation (scaling existing scores), not replacement.

**Specialists** (each operates on the residual set):

| Specialist | Focus |
|------------|-------|
| Domain relevance | Mathematical domain matching (algebra vs. topology vs. analysis) |
| Structural similarity | Proof structure pattern matching |
| Depth alignment | Proof depth / complexity matching |
| Tactic affinity | Tactic-type compatibility |
| Decomposition potential | Subgoal decomposition applicability |

**Coherence engine**: Aggregates lens outputs, detects inter-lens conflicts, and produces a modulation factor (0.5x to 2.0x) applied to the residual candidates' fused scores. The design principle is modulation, not replacement — the lens layer cannot override the multi-pass mesh, only adjust confidence.

#### 4.6.4 Files

| File | Purpose |
|------|---------|
| `src/landmark_selectors.py` | Four selector implementations (bridge_potential, self_match, accessibility, hub_suppressor) |
| `src/retrieval_scoring.py` | Signed RRF fusion, ternary voting, convergence classification |
| `src/retrieval_stages.py` | Multi-pass orchestration, stage sequencing |
| `src/landmark_freeze.py` | FrozenLandmarkState, LandmarkResidualReport, freeze/residual pipeline |
| `src/lens_guidance.py` | Five specialist lenses, lens dispatch |
| `src/lens_models.py` | Lens model definitions and interfaces |
| `src/coherence_engine.py` | Coherence aggregation, conflict detection, modulation factor computation |

#### 4.6.5 Results

**Paired evaluation** (seed=42, 164 samples, v3 DB with 242K entities):

| Configuration | Recall@16 | Delta vs. bridge_potential |
|---------------|-----------|---------------------------|
| `bridge_potential` only (single-pass) | 26.7% | — |
| Multi-pass mesh (4 selectors + signed RRF) | 43.3% | +17.6pp |
| Multi-pass + lens guidance | ~40.6% | +13.9pp |

**Key findings**:
- Multi-pass retrieval delivers a substantial improvement (+17.6pp) over single-pass bridge_potential alone
- Lens guidance is currently marginal and slightly negative (-2.7pp vs. multi-pass alone) due to empty residuals from aggressive freezing — most candidates are frozen before the lens layer runs, leaving little for it to refine
- The freeze threshold is the primary tuning lever; reducing freeze aggressiveness should restore residual volume and allow the lens layer to contribute

**Next steps**: Reduce freeze overcommitment, wire retrieval config through runtime, re-evaluate with predicted (not perfect) queries from NAV-002.

#### 4.6.6 Architectural Correction: Theorem-Level Retrieval vs Residual Execution

Post-Mathlib experiments changed how this retrieval work should be used.

- The multi-pass landmark mesh is valuable as **temporal orchestration**: lane bias, theorem-level frontier collapse, guidance packets, and hammer premise context.
- It is **not** the right primary metric for local post-structural premise-step execution. Residual retrieval against step-level ground-truth premises is near-zero on the theorem-level v3 DB.
- Therefore, theorem-level retrieval should not block local executor training. The local executor should first be trained with oracle premises from the residual dataset, then evaluated with retrieved premises as an approximation to that oracle.

This produces a cleaner experimental decomposition:

1. **Task A** — theorem-level orchestration / frontier collapse
2. **Task B1** — residual tactic-family prediction
3. **Task B2** — family-conditioned premise grounding inside the collapsed frontier

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
#   6. Aggregate theorem/template-level move profiles from `SubtaskIR`
#   7. Output: template_taxonomy.json (template_id → bank_signature, tactic_pattern, count, move_profile)
#   8. Augment nav_training.jsonl with template_id labels + theorem-level `template_move_profile`
```

**Output**: `data/template_taxonomy.json` and `data/nav_train_templates.jsonl` with template labels and theorem-level move summaries.

**Operational pipeline**: `scripts/run_enhanced_controller_pipeline.sh`
- rebuild canonical local data if needed
- refresh `SubtaskIR` training data + validation
- rebuild `nav_training.jsonl` with stable theorem keys and canonical move metadata
- rebuild `template_taxonomy.json` / `nav_train_templates.jsonl`
- rebuild `move_inventory.json`

**Main experimental entrypoint**: `scripts/run_main_experiment.sh`
- Stage 1: run the enhanced controller data pipeline
- Stage 2: train the template classifier with theorem-level move supervision
- Stage 3: train the navigator with descriptive step-level move supervision
- Stage 4: optionally benchmark the unified rewrite executor (`--cosine-rw --cosine-rw-seq`)

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
#       + auxiliary move supervision on theorem-level move profiles:
#         - dominant_subtask_kind
#         - top_target_head
#         - top_trigger_signatures
# Training: On augmented nav_train_templates.jsonl with template_id + template_move_profile
# Evaluation: Top-1 and top-3 accuracy on nav_eval.jsonl
```

**Metrics**:
- Top-1 template accuracy (target: ≥ 60%)
- Top-3 template accuracy (target: ≥ 85%)
- Per-template precision/recall (identify templates that are hard to classify)
- PAB stability of the template classifier itself (should be "stable" — this is a Regime A task)

**Alignment note:** `subtask_kind` belongs here (RECOGNITION / PLANNING), not on the navigator.
The navigator may use only descriptive move metadata (`goal_target_head`, `trigger_signature`)
for regularization.

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

### 6.4 EXECUTION Slot: Theorem-Level Guidance + Residual Executor

**What**: Split the old monolithic execution problem into:

1. **Theorem-level execution guidance** — bank-cluster specialists over the raw theorem/proof-state stream
2. **Residual local executor** — a small post-structural classifier over the normalized local goal
3. **Family-conditioned premise grounding** — invoked only for premise-sensitive families

The post-Mathlib finding is that the theorem-level system was being asked to solve the wrong local problem. It should orchestrate over time, not directly predict the final post-structural step.

**How**:
```python
# Theorem-level guidance specialists (from NAV-002 bank difficulty analysis):
#   Guidance-A: DOMAIN, CONTEXT (Regime A banks, PAB target: stable)
#   Guidance-B: STRUCTURE, AUTOMATION, DEPTH, DECOMPOSITION (Regime B banks)
#
# Residual local executor:
#   - Input: post-structural residual goal
#   - Vocabulary: rw, simp, exact, refine, apply, other
#   - Output: top-k family gate for local search
#
# Family-conditioned premise grounding:
#   - Train first with oracle premises from residual dataset
#   - Then replace oracle with retrieved frontier premises
#   - Evaluate the oracle→retrieved gap directly
```

**Architectural clarification from current experiments**:
- theorem-level guidance is a temporal/frontier layer, not the final local-step oracle;
- the residual executor predicts a bounded tactic family on the normalized local goal;
- the next learned stage is a **family-specific constrained decoder** that emits structured local actions rather than raw Lean text.

Residual tactic analysis makes this explicit:
- `82%` of residual steps are parsable into family + structured arguments;
- only `32%` are covered by the first simple template subset;
- therefore the right next object is `ActionIR`, not raw tactic strings and not theorem-level step retrieval.

**Metrics per stage**:
- **Theorem-level guidance**: PAB stability_regime, per-bank accuracy, lane routing accuracy
- **Residual executor**: family top-1, top-3, macro-F1 on post-structural residual goals
- **Premise conditioning**: oracle-premise lift, retrieved-premise degradation from oracle, per-family gains (`rw`, `refine`, `apply`, `exact`)
- **Constrained decoder**: parsable coverage, exact-IR match, compiler-valid rate, Lean-valid tactic rate, theorem lift over top-k family gating
- **End-to-end**: theorem prove rate lift over bootstrap floor, learned-lane contribution, Lean calls / theorem

**Comparison matrix**:

| Config | Role | Target | Compute |
|--------|------|--------|---------|
| v1 monolithic (NAV-002 baseline) | Raw theorem → tactic/premise | Falsified on Mathlib learned lane | 1.0x |
| v2 guidance specialists (A + B) | Theorem-level orchestration | Stable PAB + good frontier shaping | ~1.3x |
| Residual executor (goal only) | Post-structural family prediction | Top-3 useful enough to gate local search | tiny (~166K params) |
| Residual executor (goal + premise) | Family prediction with lemma context | Beat goal-only, especially on `rw`/`refine` | still laptop-scale |

**Stop/go**:
- If guidance specialists are stable and beat the monolith on their own stage metrics, keep the decomposition.
- If the residual executor does not beat majority/top-k baselines on the residual dataset, revisit the vocabulary or normalization regime before touching retrieval.
- If oracle-premise conditioning does not improve residual family prediction, premise retrieval is not the bottleneck; revisit the family model.
- If oracle helps but retrieved premises do not, retrieval quality becomes the limiting factor for Task B2.

### 6.4b Constrained Output Layer: ActionIR + Deterministic Lowering

**What**: Replace raw tactic-string prediction with a family-specific intermediate representation over local symbols. This is the theorem-proving version of the Balanced Sashimi constrained decoder: the neural model chooses bounded structure; deterministic code lowers it to Lean; the verifier checks truth.

**Core contracts**:
```python
TermExpr      # local symbol, application, projection, hole, chain, ctor, lambda
RewriteAtom   # direction + expression for rw
ActionIR      # family-specific local action object
```

**Immediate execution ladder**:

1. **Canonicalize residual tactics**
   - Parse residual steps into `ActionIR`
   - Target: >= 80% parsable coverage

2. **Template-only subset**
   - Compile the simple subset directly from premise names / local symbols
   - Expected current coverage: ~32%

3. **Family-specific IR decoder**
   - Condition on residual goal + predicted family
   - Decode `ActionIR` using the local symbol table only

4. **Oracle-premise IR decoder**
   - Add ground-truth premise embeddings for premise-sensitive families
   - Measure the upper bound before touching retrieval

5. **Retrieved-premise IR decoder**
   - Replace oracle premises with theorem-level frontier premises or later step-level retrieval
   - Measure the oracle→retrieved degradation directly

**Family-specific scope (v1)**:
- `rw`: rewrite list with optional backward arrows
- `exact`: local term construction
- `apply`: local term construction
- `refine`: local term construction with holes
- `simp`: bare `simp`, `simp [lemmas]`, `simpa ... using ...`

**Out of scope for v1**:
- tactic combinator scripts (`<;>`, `all_goals`, long tactic chains)
- large lambda terms and complex structure constructors beyond the minimal local grammar

**Metrics**:
- parsable coverage
- exact `ActionIR` match
- family-conditioned term accuracy
- compiler-valid rate (`ActionIR -> Lean tactic`)
- Lean-valid tactic rate
- end-to-end theorem lift over family-gating-only baseline

**Stop/go**:
- If parsable coverage < 70%, the canonicalizer / family taxonomy is wrong.
- If `ActionIR` compiles but Lean-valid rate is low, the local symbol inventory or lowering is wrong.
- If the compiler-valid and Lean-valid rates are good on oracle premises but collapse on retrieved premises, retrieval becomes the immediate bottleneck for Task B2/B3.

### 6.4a Temporal Controller: Positive Progress Signal

**What**: Add a stateful temporal controller inside the Arbiter that predicts which goal, lane, family band, and escalation level should be attempted next given prior proof progress.

**Why**: The theorem-level system and residual executor are both conditional on proof history. Static goal rotation and fixed lane order discard this information. The controller should model:

`P(next_goal, next_lane, next_family_band, next_escalation | proof_state, prior_progress)`

**Current implementation status**:
- `src/temporal_controller.py` exists and is tested.
- It implements a rule-based v0 controller with four phases:
  - `structural_setup`
  - `local_close`
  - `automation_close`
  - `repair_or_replan`
- Shadow mode and active mode are both implemented and operative.
- Shadow traces collected from 50-theorem Mathlib benchmark (660 steps). All 12 proved theorems succeed at phase=structural_setup, escalation=0.
- The controller is correctly routing, but the executor has nothing to route TO in local_close phase. Executor quality is the current bottleneck, not orchestration.

**Interfaces**:
```python
TemporalState
OrchestrationDecision
TemporalController.decide(state) -> OrchestrationDecision
TemporalController.update(state, goal, lane, family, tactic, success) -> None
```

**Experiment ladder**:

| Variant | What it tests | Expected effect |
|--------|----------------|----------------|
| `TC0-log` | **DONE.** Shadow traces collected: 660 steps across 50 theorems. All 12 proved theorems succeed at phase=structural_setup, escalation=0. | Phase/lane signal validated but executor quality limits value |
| `TC1-goal` | Controller drives subgoal ordering only. Active mode implemented; value depends on executor quality. | Fewer wasted rotations, faster early progress |
| `TC2-lane` | Controller drives lane order + budget slices. Active mode implemented; value depends on executor quality. | Better use of search budget than fixed hammer→bootstrap→learned order |
| `TC3-escalate` | Controller drives escalation and replanning | Fewer repeated dead-end attempts |
| `TC4-learned` | Replace rule controller with learned temporal model | Higher prove rate / lower Lean calls if data signal is real |

**Metrics**:
- theorem prove rate
- attempts / theorem
- Lean calls / theorem
- time-to-first-progress
- lane hit rate: how often the controller’s top lane matches the lane that actually closes or advances the goal
- phase calibration: agreement between predicted phase and trace-derived successful phase
- replan utility: success after replan vs success without replan

**Stop/go**:
- Do not claim temporal-controller benefit until `TC1` or `TC2` beats the current static baseline on paired theorems.
- `TC0-log` must show non-trivial lane/phase signal before training a learned controller.
- If `TC2` increases Lean calls / theorem without improving prove rate, revert to shadow-only logging and revisit the state representation.

### 6.5 Integration: Full SoM Pipeline

**What**: Wire all six slots together through the Arbiter and run end-to-end proof search.

**How**:
```python
# src/arbiter.py
#
# def som_search(theorem, slots, proof_network, lean_kernel, budget=600):
#   perception = slots.perceive(theorem.goal_state)
#   recognition = slots.recognize(perception)
#   sketch = slots.plan(perception, recognition)
 #   temporal = TemporalState(theorem_id=theorem.id, open_goals=[theorem.goal_state])
#
#   while temporal.open_goals and budget_remaining:
#     decision = slots.temporal.decide(temporal)
#     current_goal = decision.next_goal_id
#     exec_guidance = slots.execute(perception, current_goal, decision)
#     residual_goal = structural_normalize(current_goal)
#     family_topk = slots.residual_execute(residual_goal, exec_guidance, decision.family_prior)
#     action_irs = decode_action_ir(family_topk, residual_goal, exec_guidance, context)
#     action_candidates = [compile(ir) for ir in action_irs]
#
#     for candidate in action_candidates:
#       result = slots.verify(current_goal, candidate.tactic_text)
#       slots.temporal.update(temporal, current_goal, candidate.provenance, candidate.family, candidate.tactic_text, result.success)
#       if result.success:
#         advance and update temporal state
#         break
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

*Phase 7 introduces a new architectural center — negative-boundary learning, guidance/distillation over collapsed frontiers, and constraint/energy-guided planning — as a parallel v3 runtime. It does NOT modify v1 or v2 code paths. v1 and v2 remain frozen baselines for A/B/C/D comparison.*

*Phase 7 is split by maturity into two tracks:*

- ***v3A (practical, committed)**: Negative data collection, standalone Censor, inference-time pruning, contrastive navigator training, active boundary learning, OTP-derived scoring reforms (bank-IDF, zero-sparsity curriculum), and guidance-layer distillation over collapsed retrieval frontiers. All validated by direct measurement against v2 baselines.*
- ***v3B (experimental, gated on v3A)**: Energy function formalization, continuous ternary relaxation (Gumbel-softmax), energy-based sketch refinement. Ships only after v3A demonstrates value on real proof outcomes.*

*Hard gate: Phase 7 benchmark evaluation does not begin until Phase 6.5 stop/go passes (v2 ≥ v1 on benchmark). v3A graduates from parallel runtime to default only when raw_success(v3A) ≥ raw_success(v2) AND lean_calls/theorem(v3A) < lean_calls/theorem(v2). Note: analytical work (7.1c ternary distribution analysis, 7.1a/b scoring reform design) can proceed in parallel with Phase 6 since they operate on existing training data, not benchmark verification.*

### Execution Waves

| Wave | What | Gate |
|------|------|------|
| **Wave 0** | Phase 6 complete. Shared interfaces defined. `--mode v1\|v2` benchmark. | Phase 6.5 stop/go passes |
| **Wave 1 (v3A)** | Negative data, asymmetric censor, contrastive training, pruning, active boundary learning, OTP scoring reforms. `--mode v3` added. | Censor AUROC ≥ 0.80, raw_success(v3A) ≥ raw_success(v2) |
| **Wave 1.5 (v3A.5)** | Guidance-layer distillation over deterministic collapse: symbolic lenses, modulation, learned specialist distillation, disagreement mining. | Modulation is non-destructive on paired eval and improves ambiguous-tail behavior or distillation yield |
| **Wave 2 (v3B)** | Energy function, continuous ternary relaxation, sketch refinement loop. | v3A demonstrates value first. Energy-refined ≥ discrete v3A. |

### Parallel Runtime Architecture

v3 is a parallel orchestration path, not a modification of v1/v2. Benchmark runs require explicit mode: `--mode v1|v2|v3`. All modes produce the same top-level metrics schema for comparability.

**Shared interfaces** (defined as dataclasses in `src/contracts.py` during Wave 0):

- **GoalContext**: theorem_id, goal_text, proof_history, accessible_premises, source_split_metadata.
- **ResidualGoal**: structurally normalized local goal, local hypotheses, proof-history summary, lane context.
- **ExecutionGuidance**: theorem-level directions, anchor logits, lane hints, frontier summaries, critic/progress estimates.
- **TemporalState**: stateful proof-progress record used by the temporal controller.
- **OrchestrationDecision**: next-goal, lane-order, family-prior, escalation, and replanning output from temporal control.
- **FamilyPrediction**: tactic-family logits, top-k families, optional premise-sensitivity score.
- **TermExpr**: typed local term node for constrained tactic synthesis.
- **RewriteAtom**: direction + expression for rewrite families.
- **ActionIR**: family-specific structured local action lowered deterministically to Lean.
- **ActionCandidate**: lowered tactic text plus optional `ActionIR`, premises, provenance, navigational_scores, template_provenance (optional).
- **NegativeExample**: canonical `nav_negative.jsonl` schema with source, failure_category, paired_positive, split_metadata.
- **GuidancePacket**: collapsed retrieval frontier, residual diagnostics, conflict clusters, and candidate summaries passed to lens specialists.
- **LensVote**: typed support / abstain / oppose output with confidence and provenance from one guidance lens.
- **ConstraintReport**: bank scores, critic distance, censor score, anchor alignment, total score (or energy for v3B).
- **SketchProposal**: template_id, proposed_steps, latent_form (optional for v3B), total_constraint_score.
- **SearchTrace**: complete audit object for one theorem attempt, including pruning decisions and Lean calls.

### Theoretical Grounding

*This phase synthesizes four theoretical streams:*
- *OTP (Vinaik, 2025): The ternary alphabet {-1, 0, +1} where 0 is not absence but orthogonality — a third informational state. Wayfinder's 6-bank ternary decoder is an OTP projection; bank zeros are Informational Zeros (transparency, not ignorance). The Minority Channel Advantage predicts that sparse bank activations carry disproportionate information.*
- *COEC (Vinaik, 2025): Constraint-Oriented Emergent Computation — specify what a system CANNOT do; behavior emerges from constraint interactions. The bank scores + critic + censor form a constraint system; proof search is trajectory through constrained state space.*
- *Guidance-layer distillation (Wayfinder / LintGate / data geometry convergence): deterministic collapse creates a bounded frontier, then lossy orthogonal lenses emit support / abstain / oppose over the ambiguous remainder. Agreement becomes pseudo-label confidence; disagreement becomes active-learning signal and distillation data.*
- *Energy-Based Models (cf. Logical Intelligence/Kona, 2026): Define a scalar energy function over entire solutions, then minimize via gradient-based refinement in continuous latent space. Avoids the autoregressive failure mode where sequential commitment prevents revision of earlier decisions.*

*Learning-theoretic framing: Navigation (Slots 1-4) is PAC learning. Verification (Slot 5 + Lean kernel) is an exact oracle. Negative learning exploits the oracle to construct version-space boundaries from both sides. Guidance-layer distillation turns deterministic collapse + oracle feedback into dense local supervision over ambiguous frontiers. The energy function (v3B) unifies all constraint channels into a single differentiable objective.*

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

### 7.4b Guidance-Layer Distillation [v3A — Wave 1.5]

**What**: Insert a guidance layer between deterministic retrieval collapse and downstream proof execution. The retrieval stack produces a bounded frontier plus provenance-rich diagnostics; symbolic or learned specialists operate only on that frontier and emit typed support / abstain / oppose decisions.

**Why**: Small models should learn `H(candidate | collapsed_frontier, residual_state)`, not full theorem proving. Deterministic collapse does the expensive search-space reduction. The guidance layer spends learned capacity only on the ambiguous remainder and turns committee agreement/disagreement into distillation and active-learning signal.

**Clarification from current experiments**: this guidance layer lives at the theorem/frontier level. It feeds the residual executor; it is not itself the final local-step predictor.

**Execution ladder**:

| Variant | What it tests | Shipping status |
|---------|---------------|-----------------|
| **G0-none** | No guidance layer after deterministic retrieval | Baseline for Wave 1.5 |
| **G0-replace** | Guidance replaces deterministic ranking | Ablation only; falsification condition |
| **G0-modulate** | Guidance modulates deterministic ranking | Default control regime |
| **G1-symbolic** | Candidate-grounded symbolic lenses only | First committed implementation |
| **G2-single** | One learned lens on ambiguous subset | Optional once G1 has stable targets |
| **G3-committee** | Multi-lens learned committee over collapsed frontier | Gated on G2 value |
| **G4-distill** | Distill committee outputs into lightweight reranker / specialist head | Preferred practical outcome |
| **G5-disagree** | Mine disagreement cases for active learning and retraining | Ongoing data engine |

**Metrics**:
- **Paired `recall@16`, `perfect`, `zero`** on identical perfect-query samples
- **Ambiguous-tail metrics**: gain on examples with non-converged landmark sets or non-zero residual entropy
- **Harmful demotion rate**: fraction of GT candidates pushed below cutoff by the committee
- **Abstention rate**: how often lenses emit informative zero rather than weak support/oppose
- **Committee informativeness**: distribution of no-op vs modulate vs branch-worthy cases
- **Distillation yield**: number of usable `GuidancePacket` / `LensVote` / verifier outcome triples

**Default rule**:
- `replace` remains ablation-only until it beats both `G0-none` and `G0-modulate` on paired evaluation.
- `modulate` is the only shipping guidance mode in Wave 1.5.
- If committee informativeness is low, modulation should converge toward a no-op rather than forcing rank inversions.

**Stop/go**:
- `G0-modulate` must remain within 3 percentage points of `G0-none` on paired recall, or exceed it outright.
- Guidance must also improve at least one of: ambiguous-tail recall, zero-recall reduction, or distillation yield.
- Candidate-grounded evidence is mandatory before symbolic guidance graduates from experimental to committed.
- Learned lenses (`G2+`) must beat `G1-symbolic` on the ambiguous subset before default promotion.

**Output artifacts**:
- `data/guidance_packet.jsonl` — collapsed-frontier packets with residual diagnostics
- `data/lens_votes.jsonl` — typed per-lens outputs over candidates / branches
- `data/ambiguous_guidance_train.jsonl` — distillation dataset for learned specialists

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

**v3A pipeline** (discrete scoring + guidance modulation + censor pruning):
```python
# v3A inference pipeline (in src/proof_search.py, --mode v3):
#
# 1. PERCEPTION: Encode goal state (existing)
#    goal_embedding = encoder.encode(goal_state)
#
# 2. OTP SCORING + DETERMINISTIC COLLAPSE: Navigate with bank-IDF weighting (7.1a)
#    candidates = proof_network.navigate(goal_embedding, bank_idf_weights)
#    collapsed = landmark_expand_retrieve(goal_state, candidates)
#
# 3. GUIDANCE MODULATION (7.4b, optional)
#    guided = guidance_layer.modulate(goal_state, collapsed)
#
# 4. CENSOR PRUNING: Asymmetric censor filters candidates (7.3)
#    scored = censor.score(goal_state, guided)
#    pruned = [c for c in scored if c.censor_confidence >= threshold]
#    if len(pruned) == 0:
#        pruned = top_k(scored, k=3)  # safety net: never prune ALL
#
# 5. TEMPLATE SELECTION: Classify proof template (Phase 6.2)
#    templates = template_classifier.predict_top_k(goal_embedding, k=3)
#
# 6. DISCRETE SKETCH: Predict sketch, score via ConstraintReport (no energy)
#    for template in templates:
#        sketch = sketch_predictor.predict(goal_state, template)
#        report = constraint_report(sketch, bank_scores, critic, censor, anchors)
#        if report.total_score > best_score:
#            best_sketch, best_score = sketch, report.total_score
#
# 7. EXECUTION: Submit best tactics to Lean (Lane A)
#    result = lean_kernel.apply_sequence(best_sketch.tactics)
#
# 8. RETRY: If failed, try next template (existing arbiter retry logic)
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

**What**: Prove that each Phase 7 component produces genuine improvement. Conditions A through B are v3A validation; C and D are v3B validation (gated on v3A success). Guidance-layer distillation (7.4b) is an orthogonal Wave 1.5 track: if `G0-modulate` passes its gate, A+/B/B+/C/D use guidance modulation as the default retrieval frontend. Otherwise guidance remains logging/distillation-only.

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
| **Guidance Distillation (v3A — Wave 1.5)** | | |
| G0-none: no guidance layer | v3A | Deterministic collapse only |
| G0-replace: committee replaces rerank | v3A | Falsification condition for guidance-as-driver |
| G0-modulate: committee modulates rerank | v3A | Default control regime |
| G1-symbolic: candidate-grounded symbolic lenses | v3A | Whether rule-based guidance adds value |
| G2-single: one learned ambiguous-case lens | v3A | Whether learned local resolution beats symbolic-only |
| G3-committee: multi-lens learned committee | v3A | Whether committee fusion adds value over single lens |
| G4-distill: distill committee into lightweight head | v3A | Practical deployment path |
| G5-disagree: train on disagreement cases | v3A | Active-learning value of committee conflict |
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
- [ ] Guidance modulation is non-destructive on paired eval and improves ambiguous-tail or distillation metrics (Phase 7.4b)
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
| Understand v3 guidance layer | `docs/WAYFINDER_PLAN.md` Section 7.4b, `src/lens_guidance.py`, `src/coherence_engine.py` |
| Understand v3 energy refinement | `docs/WAYFINDER_PLAN.md` Phase 7 (v3B: 7.5-7.6, gated on v3A) |
| Understand v3 runtime architecture | `docs/WAYFINDER_DESIGN.md` Section 12 |
| Understand OTP/guidance/EBM theory | `docs/WAYFINDER_RESEARCH.md` Section 2.11 |

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

---

## Critical Path Assessment (2026-03-19, post-EXP-048 executable-selector benchmark)

**Current state:** Pantograph is stable enough for benchmark iteration. The rewrite family is
operationally frozen around the additive cosine executor:
- single-step `cosine_rw`
- sequential fallback `cosine_rw_seq`

On the measured step-0 slices, `rw0` through `rw3` collapse to this cheap scoped cosine +
verification pattern; no standalone learned rewrite decoder is justified on the current evidence.
`NAV-004` is now the primary aligned checkpoint: it matches NAV-003 at `14/50` while removing the
misplaced `subtask_kind` navigator head. The theorem-search gain over NAV-002 is confirmed as a
routing/automation effect, not a learned local-tactic-lane effect. Declaration-faithful Tier B
start states produce `91%` GoalStart on step-0 and `94%` base-state coverage on the step>0 replay
sample. Step>0 replay is usable (`12/47 = 26%`) but remains a coverage track. The main execution
milestone after rewrite collapse is no longer another always-on local lane. It is now
**compile-aware executable action selection**:
- `apply` is validated at the action-selection level by live Lean verification (`35/91` accepted
  vs `15/91` for cosine top-1 on held-out step-0 apply goals)
- the next dataset milestone is to expand that selector beyond canonical step-0 into
  provenance-aware post-bootstrap and mid-search residuals
- `refine_named` remains the next structured-action family after `apply`

`ContextIR` remains the parallel compiler track that must unlock replay fidelity and future
families, but it is no longer the reason to delay the next executable-selector track.

**What is already validated:**
- PERCEPTION → RECOGNITION → PLANNING → TEMPORAL CONTROLLER → EXECUTION → VERIFICATION is the right architectural split.
- Theorem-level guidance is a temporal/frontier layer, not the final local-step oracle.
- `rw0` is the correct first formal lane for validating local execution, but learning is not needed on the easy step-0 slice because cosine nearly saturates the oracle ceiling.
- step>0 replayed states support the same scoped-cosine local execution regime as theorem-initial states once replay succeeds.
- `rw1` is not a new premise-selection problem. On the current split, tier identity already provides almost all of the direction signal.
- GoalStart / ReplaySuccess and LeanValid\|started are separate tracks and must remain separate in all reporting.
- Declaration-faithful Tier B is the right state abstraction: source declaration + `by sorry` matches LeanDojo's post-`by` regime far better than `env_inspect.type.pp`.
- Structured Lean compiler feedback is the right supervision object for the next learned stage:
  `unification_mismatch`, `typeclass_missing`, `unknown_identifier`, `accepted_with_goals`, and
  goal-start failures are meaningfully separable.
- `apply` executable selection now has a real live-execution win: a small selector can improve
  `LeanAccepted | started` materially over cosine on held-out goals.

**What is not ready yet:**
- The current wrapper path still reconstructs only a thin subset of Lean source context. It misses or conservatively drops many real Mathlib context effects: inline `... in`, local notation blocks, local attributes, scoped notation, and include/omit interactions.
- Step>0 replay is usable but still shallow: most remaining failures are at prefix index `0`, and higher replay coverage still depends on source-context fidelity rather than lexical name repair.
- `apply` is validated at the action-selection level, but theorem-level lift from the selector has
  not yet been measured in the live search loop.
- The executable-selector dataset is still too narrow. It must expand from canonical step-0 apply
  into canonical mid-step, post-bootstrap residual, and mid-search apply-shaped states with full
  provenance.
- `simp` remains a helper lane rather than a productive theorem-winning family on the current
  slice.
- `refine_named` remains the next structured-action family after `apply`; `refine_anon` is still
  a distinct harder regime.
- Full SoM training is premature until multiple specialist contracts are validated on live
  executable metrics rather than only offline ranking metrics.

**ContextIR compiler findings (new):**
- Whole-Mathlib census (`scripts/context_ir_census.py`) shows the relevant source-context constructs are common, not edge cases: `open scoped` 2370, `local notation` 496, `local attribute` 915, `scoped_notation` 402, `include` 1064, `omit` 422, and `inline_only` `... in` forms 6742.
- On `rw0_eval.jsonl`, the benchmark audit (`scripts/context_ir_benchmark_audit.py`) finds these constructs active at theorem sites frequently: `open` on 693 processed examples, `open_scoped` on 262, `local_notation` on 42, `local_attribute` on 42, `include` on 47. The most common unsupported patterns are exactly the next-declaration and scoped forms the current wrapper path does not compile (`open Classical in`, `variable (M) in`, `include Q in`, `open scoped Classical in`).

**Canonical next steps (in order):**

1. **Freeze the aligned baseline.**
   - Use `NAV-004` as the canonical checkpoint for all future theorem-search runs.
   - Use `cosine_rw` plus interleaved bootstrap as the canonical runtime baseline.
   - Treat standalone `cosine_rw_seq`, ungated `apply`, and global `simp` deployment as historical or diagnostic ablations, not the forward default.

2. **Expand the executable dataset with provenance before broadening the selector story.**
   - Collect `(goal, candidate)` probe rows from:
     - canonical step-0 `apply`
     - canonical mid-step `apply`
     - post-bootstrap apply-shaped residuals
     - mid-search apply-shaped states
   - Preserve provenance on every row:
     - `source_kind`
     - `search_stage`
     - `lane_provenance`
     - `goal_shape_ir`
     - `trigger_profile_ir`
     - `subtask_kind` when available
     - `feedback_category`
     - `accepted`
     - `closed`
   - Keep the first learning target binary:
     - `executable = accepted_with_goals ∪ closed`
     - `non_executable = all remaining failure categories`

3. **Train source-aware executable selectors, not another name reranker.**
   - `apply` is now the first validated execution-level selector regime.
   - Evaluate by source, not just pooled metrics:
     - canonical step-0
     - canonical mid-step
     - post-bootstrap residual
     - mid-search residual
   - Keep ranking metrics secondary. Primary metrics are:
     - `LeanAccepted | started`
     - executable candidate recall
     - failure-category composition
   - Treat static shape filters as features and instrumentation, not as standalone gates.

4. **Integrate the `apply` executable selector into theorem search.**
   - Run theorem-search ablations with:
     - baseline (`cosine_rw` + interleaved bootstrap)
     - baseline + selector-driven `apply`
     - baseline + selector-driven `apply` + residual-conditioned gating
   - The question is no longer premise-name recovery. It is whether compiler-validated
     `accepted_with_goals` converts into more solved theorems at acceptable call budget.

5. **Keep `ContextIR` as the enabling compiler track, not the primary local-family milestone.**
   - Parse theorem-site lexical context into an explicit IR: scope frames + directives + unsupported forms.
   - Treat Lean source notation as a DSL with scoped side effects, not as header text.
   - Prioritize the context effects that still matter for helper/specialist families:
     - `attribute [local simp]`
     - `open scoped`
     - local notation
     - local instances
     - `include` / `omit`
   - Validation scripts remain mandatory:
     - `python -m scripts.context_ir_census`
     - `python -m scripts.context_ir_benchmark_audit --dataset data/canonical/rw0_eval.jsonl`

6. **Extend the same executable-selection regime family-by-family.**
   - Immediate next family:
     - `refine_named` via structured action / skeleton selection
   - Immediate non-target:
     - `refine_anon`
   - Keep mixed-family lanes residual-conditioned, not globally enabled.
   - `simp` remains a helper lane, not a default theorem-winning finisher.
   - `apply` should remain residual-conditioned even after selector deployment unless theorem-level
     cost/yield justifies broader use.
   - Only promote a family to default-on theorem search after it shows nonzero theorem-level gain at acceptable call budget.

7. **Keep step>0 replay and semantic `rw` as a coverage track only.**
   - Keep state-guided replay and per-step alias rebuilding.
   - Prioritize post-step subgoal selection and alpha-equivalent goal matching.
   - Current replay gate: `ReplaySuccess | base_state >= 35%`.
   - Primary step>0 metric remains `LeanValid@k | replayed`, not oracle-match.

8. **Delay full SoM training until there are multiple validated specialist regimes.**
   - Full specialist routing/training should sit on top of validated executor contracts:
     - `rw` finisher
     - interleaved bootstrap scaffolder
     - `apply` executable selector
     - `refine_named` structured-action selector
   - Do not treat "Society of Mind" as a substitute for proving out these specialist contracts.

9. **Move the learned frontier above the rewrite executor.**
   - Do not spend more cycles trying to beat cosine on rewrite-family local execution with standalone rewrite decoders.
   - Do not build a standalone direction head for the current `rw0/rw1` split; the family gate already carries that signal.
   - Focus learning on the parts cheap geometry still does not solve:
     - executable action selection over scoped candidate sets (`apply`)
     - structured skeleton selection over scoped candidate sets (`refine_named`)
     - residual-conditioned orchestration (`rw` vs helper/specialist families)
     - controller-visible move selection over `SubtaskIR` / trigger profiles

**Do not run yet:**
- navigator aux-head tuning beyond the aligned `NAV-004` contract
- more rewrite-family decoder experiments
- theorem-level claims for new always-on local lanes that still fail the cost/yield gate
- standalone learned direction-head experiments on the current `rw0/rw1` split
- large step>0 learned `rw` benchmarks before replay coverage improves
- full SoM specialist training regimes before at least `apply` and one additional specialist
  family are validated at live executable metrics
- temporal-controller value claims that assume `apply` / `refine` are already productive global lanes
- theorem-search claims based only on offline gold recovery or MRR without live `LeanAccepted`

**Key insight:** The deterministic compiler layer now has two parts:
1. **Source-context compilation** (`ContextIR`) to reconstruct the executable theorem-site environment.
2. **Action compilation** (`ActionIR`) to lower family-specific local choices into Lean syntax.
3. **Executable feedback compilation** (`LeanFeedback` + probe datasets) to turn Lean's own
   elaboration/unification outcomes into training targets for executable selectors.

The remaining gap is no longer a missing rewrite-family scorer. It is the combination of:
- a missing compiler layer that links theorem source notation to executable local proof states, and
- family-specific executable-action selection over scoped candidates, and
- the higher-level orchestration / mixed-family selection problems that cheap geometry does not solve by itself.

**Planning-layer update (2026-03-18):** the project now also carries a controller-facing move
representation above `ActionIR`:

3. **Motivated move typing** (`SubtaskIR` + trigger profiles) to state:
   - why a local family is admissible in the current goal (`TriggerProfileIR`)
   - what local transformation the step is attempting (`SubtaskIR`)

This layer does not replace `ActionIR`. It is derived from canonical proof data and is meant for
controller training, auditability, and proof-schema mining. The relevant code path is now:
- `src/subtask_ir.py`
- `scripts/build_subtask_training_data.py`
- `scripts/validate_subtask_ir.py`
- `scripts/mine_move_schemas.py`

**Updated order after rewrite-family collapse:**
1. freeze `NAV-004` as the aligned theorem-search checkpoint
2. freeze rewrite-family execution around `cosine_rw` + interleaved bootstrap
3. expand the provenance-aware executable dataset for `apply`
4. train and deploy source-aware `apply` executable selectors into theorem search
5. extend the same executable-selection regime to `refine_named`
6. use `SubtaskIR` / trigger profiles to lift validated specialist behavior into a planner-facing move space
7. only then train the full SoM with explicit per-source / per-specialist training regimes
