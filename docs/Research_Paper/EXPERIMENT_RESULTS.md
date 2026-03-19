# Wayfinder: Experiment Results Ledger

**Project:** Wayfinder -- Navigational Proof Search Through Structured Semantic Networks
**Version:** 1.1
**Started:** March 2026
**Hardware:** Apple Silicon M-series (primary); Xeon macpro + i7 homebridge (CPU workers)
**Eval set:** Frozen `data/nav_eval.jsonl`; MiniF2F-test (488); Mathlib test split (~2k)
**Status:** Canonical results ledger for `docs/Research_Paper/`. Record infrastructure and model metrics separately; never fold infrastructure misses into model-negative counts.

Three-stream results ledger: navigation validation + architecture evaluation + PAB process evaluation.

### Result Entry Template

Each new result entry should include:

- `Date`
- `Script / command`
- `Data / split`
- `Protocol`
- one metric table
- `Key findings`
- `Decision / next action`

For local-execution experiments, always report both:

- `GoalStart` or equivalent infrastructure metric
- `LeanValid|started` or equivalent semantic/model metric

### Metric Categories (3-Lane Architecture)

All benchmark proof success metrics are reported with strict lane separation to prevent inflated claims:

| Category | Definition | Lane |
|----------|-----------|------|
| `raw_success` | Proved by Lane A (step-wise search) alone, no Axle involvement | A |
| `axle_assisted_success` | Lane A partial progress + Axle decompose/repair completed the proof | A+B |
| `axle_repair_only` | Axle `repair_proofs` alone closed remaining goals (no Lane A success) | B |
| `lane_c_verified` | Independently confirmed by lean4checker/Comparator/SafeVerify | C |

Primary comparison metric is always `raw_success`. Other categories are reported alongside but never summed into headline numbers.

---

## Baselines (Established Before Experiments)

### BL-1: Dense Retrieval Baseline (ReProver-style)

Cosine similarity between goal state embeddings and pre-computed premise embeddings.

| Metric | Value | Date | Notes |
|---|---|---|---|
| recall@1 | | | `scripts/eval_retrieval.py --samples 500` |
| recall@4 | | | |
| recall@8 | | | |
| recall@16 | | | |
| Avg retrieval time (ms) | | | Per-goal forward pass + dot product |
| Encoder | | | Selected in EXP-0.2 |
| Premise corpus size | | | Mathlib lemmas in proof_network.db |

### BL-2: Random Retrieval Baseline

| Metric | Value | Notes |
|---|---|---|
| recall@16 (random) | | Expected: ~16/N where N = corpus size |

### BL-3: Published Systems (from literature)

| System | MiniF2F-test | Mathlib | Params | FP/proof | Source |
|---|---|---|---|---|---|
| ReProver | ~26% | ~15% | ~299M | ~1000s | Yang et al. 2024 |
| LeanProgress | ~33% | — | ~299M | — | |
| DeepSeek-Prover-V2 | 88.9% | — | 671B | — | Ren et al. 2025 |
| BFS-Prover | — | — | — | — | |
| LeanHammer (hybrid) | 37.3% | — | — | — | |

---

## Experiment Cycle 0: Proof Network Validation

*Purpose: Validate that the proof network structure supports retrieval before any training.*

### EXP-0.1: Proof Network Coverage

**Date:** ____
**Script:** `scripts/extract_proof_network.py`

| Metric | Value | Target | Notes |
|---|---|---|---|
| Total entities | | ~98k | Mathlib lemmas |
| Bank position coverage (%) | | >= 95% | Entities with all 6 banks |
| Average anchors per entity | | >= 3 | |
| Total anchor types | | ~300+ | Bootstrap + gap analysis |
| IDF computed | | Yes | |
| Accessible premises extracted | | Yes | |

**Stop/go:** Coverage >= 95%, anchors/entity >= 3.

### EXP-0.2: Anchor Gap Analysis (Iterative)

**Date:** 2026-03-10 (initial, trace-bounded DB), **RE-RUN REQUIRED** after 2026-03-14 DB expansion
**Script:** `scripts/anchor_gap_analysis.py --db data/proof_network.db --samples 500`

| Iteration | recall@16 (perfect queries) | New anchors added | Top gap themes | Notes |
|---|---|---|---|---|
| 0 (bootstrap) | 100% | 0 | — | 2026-03-10, 78K entity DB (trace-bounded) |
| **RE-RUN** | pending | — | — | 242K entity DB (proof-bounded), domain labels fixed |

**IMPORTANT (2026-03-14):** The original 100% recall pass was on the trace-bounded DB (78K entities, 34.5% premise coverage). Re-run on expanded DB below.

**Re-run on expanded DB (2026-03-14, Colab T4):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Samples | 152 | 500 | Partial (script ran 152) |
| Mean recall@16 | **4.1%** | ≥70% | **FAIL** |
| Perfect recall | 2/152 (1.3%) | — | — |
| Zero recall | 140/152 (92.1%) | — | — |

**Root cause: dilution.** The 242K entity DB includes 163K premise-only entities with sparse metadata (namespace-derived positions only, no tactic anchors). These compete with traced entities in the ranking but can't be distinguished by the scoring function. Even "perfect queries" (ground-truth bank positions + anchors) can't discriminate correct premises from the flood of weakly-annotated competitors.

**Implication:** The proof network expansion (Stage 1) improved coverage from 34.5% → 85.9% but destroyed discrimination. The scoring function needs to incorporate provenance weighting — traced entities with full position/anchor data should rank higher than premise-only entities with sparse metadata. Alternatively, premise-only entities need richer annotations (type-level anchors, constant usage) before they can compete fairly.

**Top 20 gap anchors (final iteration):**

| Anchor | Miss count | Theme |
|---|---|---|
| | | |

**Stop/go:** recall@16 >= 70% on perfect queries. If <50% after 3 iterations, bank positioning needs rethinking.

### EXP-0.3: Encoder Selection

**Date:** 2026-03-10
**Script:** `scripts/eval_encoders.py --eval-data data/nav_eval.jsonl --samples 500 --tier all --device mps`
**Protocol:** Encode 500 Mathlib goal states (from 138 theorems) per candidate. Evaluate tokenizer math symbol coverage (32 symbols), intra/inter-theorem cosine similarity separation, encoding throughput, and memory.

| Tier | Candidate | Params | Dim | Tok% | Intra cos | Inter cos | Separation | Goals/sec | Status |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | all-MiniLM-L6-v2 | 22M | 384 | 71.9 | 0.9222 | 0.3353 | **0.5869** | 616.7 | OK |
| Small | gte-modernbert-base | 149M | 768 | 100.0 | 0.9375 | 0.6337 | 0.3038 | 51.8 | OK |
| Small | modernbert-embed-base | 137M | 768 | 100.0 | 0.9417 | 0.6464 | 0.2952 | 59.9 | OK |
| Medium | byt5-small | 300M | 1472 | 100.0 | 0.9583 | 0.8111 | 0.1472 | 22.1 | OK |
| Medium | snowflake-arctic-embed-l-v2.0 | 335M | 1024 | 100.0 | 0.9328 | 0.5619 | 0.3709 | 43.6 | OK |
| Medium | SFR-Embedding-Code-400M_R | 400M | 1024 | 71.9 | 0.9842 | 0.8531 | 0.1311 | 22.4 | Fixed (see notes) |
| Large | stella_en_1.5B_v5 | 1.5B | 1024 | 100.0 | — | — | 0.3377 | 9.9 | Fixed (native Qwen2) |
| XL | e5-mistral-7b-instruct | 7B | 4096 | 100.0 | 0.7205 | 0.6320 | 0.0885 | 2.2 | OK |
| XL | gte-Qwen2-7B-instruct | 7B | 3584 | 100.0 | 0.8089 | 0.6288 | 0.1801 | 1.0 | OK (native Qwen2) |
| XL | NV-Embed-v2 | 7.85B | 4096 | — | — | — | — | — | Unfixable (transformers 5.x) |

#### EXP-0.3b: Extended Encoder Evaluation (Math-Domain + Novel Architectures)

**Date:** 2026-03-10
**Script:** `scripts/eval_encoders.py --eval-data data/nav_eval.jsonl --samples 500 --tier extended`
**Protocol:** Same 500-sample eval set. Extended candidates include math-domain models, novel architectures, and Lean-specific fine-tunes.

| Tier | Candidate | Params | Dim | Tok% | Intra cos | Inter cos | Separation | Goals/sec | Status |
|---|---|---|---|---|---|---|---|---|---|
| Math | leandojo-lean4-retriever-byt5-small | 218M | 1472 | 100.0 | 0.6441 | 0.0208 | **0.6233** | 21.9 | **#1 overall** |
| Novel | pplx-embed-v1-0.6b (bidir. Qwen3) | 596M | 1024 | 100.0 | 0.8664 | 0.2662 | **0.6002** | 16.8 | OK |
| Novel | Qwen3-Embedding-0.6B | 596M | 1024 | 100.0 | 0.9350 | 0.4691 | 0.4659 | 14.6 | OK |
| Math | MathBERT (raw MLM, no embed fine-tune) | 110M | 768 | 71.9 | 0.9121 | 0.4569 | 0.4552 | 147.1 | OK |
| Math | Bert-MLM_arXiv-MP-class_zbMath | 110M | 768 | 71.9 | 0.9560 | 0.6097 | 0.3463 | 149.2 | OK |
| XL | LeanSearch-PS (LoRA on e5-mistral-7b) | 7B | 4096 | 100.0 | 0.5555 | 0.3475 | 0.2080 | 2.0 | OK |

#### Combined Rankings (all 15 evaluated models, sorted by separation)

| Rank | Candidate | Dim | Separation | Goals/sec | Source |
|---|---|---|---|---|---|
| 1 | **leandojo-lean4-retriever-byt5-small** | 1472 | **0.6233** | 21.9 | EXP-0.3b |
| 2 | pplx-embed-v1-0.6b | 1024 | 0.6002 | 16.8 | EXP-0.3b |
| 3 | all-MiniLM-L6-v2 | 384 | 0.5869 | 616.7 | EXP-0.3 |
| 4 | Qwen3-Embedding-0.6B | 1024 | 0.4659 | 14.6 | EXP-0.3b |
| 5 | MathBERT | 768 | 0.4552 | 147.1 | EXP-0.3b |
| 6 | snowflake-arctic-embed-l-v2.0 | 1024 | 0.3709 | 43.6 | EXP-0.3 |
| 7 | Bert-MLM_arXiv-MP-class_zbMath | 768 | 0.3463 | 149.2 | EXP-0.3b |
| 8 | stella_en_1.5B_v5 | 1024 | 0.3377 | 9.9 | EXP-0.3 |
| 9 | gte-modernbert-base | 768 | 0.3038 | 51.8 | EXP-0.3 |
| 10 | modernbert-embed-base | 768 | 0.2952 | 59.9 | EXP-0.3 |
| 11 | LeanSearch-PS (LoRA) | 4096 | 0.2080 | 2.0 | EXP-0.3b |
| 12 | gte-Qwen2-7B-instruct | 3584 | 0.1801 | 1.0 | EXP-0.3 |
| 13 | byt5-small (vanilla) | 1472 | 0.1472 | 22.1 | EXP-0.3 |
| 14 | SFR-Embedding-Code-400M_R | 1024 | 0.1311 | 22.4 | EXP-0.3 |
| 15 | e5-mistral-7b-instruct | 4096 | 0.0885 | 2.2 | EXP-0.3 |

**Key findings (updated with extended eval):**
1. **LeanDojo retriever is the new separation leader** (0.6233) — a ByT5-small fine-tuned specifically for Lean 4 premise retrieval. Compared to vanilla byt5-small (sep=0.147), Lean-specific fine-tuning yields a **4.2x improvement** on the same architecture. Inter-theorem similarity is remarkably low (0.021), meaning it spreads different theorems far apart in embedding space.
2. **pplx-embed bidirectional Qwen3 is #2** (0.6002) — Perplexity's novel bidirectional conversion of a causal decoder produces the best separation among general-purpose models. Architecturally significant: shows decoder→bidirectional conversion is viable for embeddings.
3. **MiniLM remains the throughput champion** (616.7 goals/sec) at separation 0.587 — only 6% behind LeanDojo on separation but 28x faster. Best efficiency trade-off for iterative proof search.
4. **Inverse size-separation relationship confirmed across 15 models**: Larger models produce higher intra AND inter cosine similarity, collapsing the gap. Domain fine-tuning shifts the curve (LeanSearch-PS LoRA: 0.089→0.208, 2.3x improvement over base e5-mistral) but cannot overcome the fundamental issue.
5. **MathBERT surprises** — a raw MLM (no embedding fine-tuning) pre-trained on math curriculum achieves sep=0.455, beating 7 purpose-built embedding models. Math-domain pretraining alone creates useful embedding geometry.
6. **Tokenizer coverage is a red herring**: MiniLM and MathBERT cover only 71.9% of math symbols but rank #3 and #5 on separation. Full Unicode coverage (100%) doesn't compensate for poor embedding geometry.
7. **Throughput scales inversely**: MiniLM at 617 goals/sec vs e5-mistral at 2.2 goals/sec (280x difference).

**Failure analysis:**
- **SFR-Embedding-Code-400M_R**: Custom code uses `persistent=False` position_id buffers that get corrupted on transformers 5.x. Fixed with `_fix_model_buffers()` (validates position_ids + rotary caches) and `_clear_accelerator_cache()` (prevents stale MPS memory between model load/unload cycles). NaN retry guard added to `encode_goals()`.
- **stella_en_1.5B_v5**: Custom Qwen2 code imports removed module paths in transformers 5.x (`tokenization_qwen2_fast`, `rope_theta`, `DynamicCache.from_legacy_cache`). Fixed by adding to `SKIP_TRUST_REMOTE_CODE` set — native SentenceTransformer Qwen2 support handles it correctly.
- **NV-Embed-v2**: Cascading transformers 5.x incompatibilities (`all_tied_weights_keys`, `DynamicCache.from_legacy_cache`, `DynamicCache.get_usable_length`). Requires `trust_remote_code=True` (fully custom architecture), so cannot bypass custom code. **Unfixable without downgrading transformers.**
- **gte-Qwen2-7B-instruct**: Same Qwen2 fix as stella (SKIP_TRUST_REMOTE_CODE). Loaded via native SentenceTransformer Qwen2 support. Separation 0.180 — better than e5-mistral (0.089) but still far below MiniLM (0.587). Confirms inverse size-separation trend: 7B model, 1.0 goals/sec, 3584d.

**MoE hypothesis (noted by user):** Different models may excel at different aspects — MiniLM for discrimination, ByT5 for Unicode fidelity, larger models for semantic depth. A mixture-of-experts approach combining specialized encoders could potentially outperform any single model. This is a Phase 5+ research direction.

**Decision:** Encoder selection requires weighing separation vs throughput. Top 3 candidates for Phase 1:

| Candidate | Separation | Goals/sec | Dim | Trade-off |
|---|---|---|---|---|
| leandojo-lean4-retriever-byt5-small | 0.6233 | 21.9 | 1472 | Best separation, domain-specific, 28x slower than MiniLM |
| pplx-embed-v1-0.6b | 0.6002 | 16.8 | 1024 | Novel arch, needs trust_remote_code, 37x slower |
| all-MiniLM-L6-v2 | 0.5869 | 616.7 | 384 | Best throughput, smallest, only 6% behind LeanDojo |

**[x] Primary: all-MiniLM-L6-v2** — best throughput for iterative proof search. 6% separation gap vs LeanDojo is acceptable given 28x speed advantage. 384d keeps memory footprint minimal for proof network.
**[ ] Ablation candidate: leandojo-lean4-retriever-byt5-small** — test in EXP-2.1 whether domain-specific embeddings improve retrieval recall enough to offset throughput cost.

---

## Experiment Cycle 1: Navigation Training

*Purpose: Train the navigational pipeline and track per-bank learning dynamics.*

### EXP-1.1: Curriculum Training Run

### NAV-001: Baseline Training (No LR Scheduler, Partial PAB)

**Date:** 2026-03-10
**Config:** `configs/wayfinder.yaml` (num_anchors=18729)
**Script:** `scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-001 --device mps`
**Checkpoint:** `models/NAV-001_step5000.pt`

**Training Configuration:**

| Parameter | Value |
|---|---|
| Encoder | all-MiniLM-L6-v2 (384d, frozen) |
| Navigable banks | 6 (structure, domain, depth, automation, context, decomposition) |
| Scoring mechanism | confidence_weighted |
| Batch size | 16 |
| Learning rate | 1e-4 (constant, no scheduler) |
| Max iterations | 5000 |
| Curriculum | Phase A: 0-500 (<=2 step), Phase B: 500-2000 (<=5 step), Phase C: 2000+ (all) |
| Device | mps (Apple Silicon) |
| Duration | 537s |

**Phase Gates:**

| Phase | Step range | Nav accuracy target | Actual (peak) | Status |
|---|---|---|---|---|
| A (warmup) | 0-500 | >= 60% on 1-2 step proofs | 64.8% (step 400) | Pass |
| B (growth) | 500-2000 | >= 50% on <=5 step proofs | 69.0% (step 950) | Pass |
| C (full) | 2000-5000 | >= 45% on all proofs | 70.3% (step 2800) | Pass |

**Training Dynamics (key checkpoints):**

| Step | L_total | L_nav | L_anchor | L_critic | Nav acc (mean) |
|---|---|---|---|---|---|
| 50 | 2596.70 | — | — | — | — |
| 200 | 293.71 | — | — | — | 0.607 |
| 500 | 108.31 | — | — | — | 0.648 |
| 1000 | 46.93 | — | — | — | 0.676 |
| 2000 | 23.42 | — | — | — | 0.614 |
| 3000 | 15.84 | — | — | — | 0.650 |
| 5000 | 10.50 | — | — | — | 0.548 |

**PAB Profile:** `runs/NAV-001/NAV-001_pab_profile.json`

| PAB Metric | Value | Assessment |
|---|---|---|
| Stability mean | 0.324 | > 0.30 = chaotic |
| Stability std | 0.317 | High variance |
| Predictability (final) | 49.6 | Not converged |
| Stability regime | **chaotic** | |
| Crystallization rate | 0.0 | Dead channel (no data fed) |
| Live PAB channels | 3/8 | stability, predictability, tier1_accuracy |
| Dead PAB channels | 5/8 | gen_gap, repr_evol, crystallization, tier2/3, loss_components |

**Observations:**
- Loss curve shows 247x reduction (2596 → 10.5) but high step-to-step variance
- Domain bank saturates at 0.95-1.0 (trivially easy); structure/automation hardest (0.3-0.55)
- No LR scheduler → constant 1e-4 throughout → no annealing
- Random eval sampling → noisy accuracy measurements
- 5/8 PAB channels dead → insufficient diagnostic signal
- UW-SO weights: nav=0.161, anchor=0.302, progress=0.211, critic=0.326

---

### NAV-002: Full PAB Instrumentation (Cosine LR, Deterministic Eval)

**Date:** 2026-03-10
**Config:** `configs/wayfinder.yaml` (identical model config to NAV-001)
**Script:** `scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-002 --device mps`
**Checkpoint:** `models/NAV-002_step5000.pt`
**Changes from NAV-001:**
1. Cosine LR scheduler (T_max=4800, warmup=200 steps linear)
2. Deterministic eval subset (64 examples from nav_eval.jsonl, seed=42)
3. All 8 PAB channels populated (val_loss, bridge embeddings, decoder weight signs, loss components, adaptive weights, tier2/tier3)

**Training Configuration:**

| Parameter | Value |
|---|---|
| Encoder | all-MiniLM-L6-v2 (384d, frozen) |
| Navigable banks | 6 |
| Batch size | 16 |
| Learning rate | 1e-4 peak, cosine decay with 200-step linear warmup |
| Max iterations | 5000 |
| Curriculum | Phase A: 0-500, Phase B: 500-2000, Phase C: 2000-5000 |
| Phase C start | Step 3700 (Phase B extended due to data volume) |
| Device | mps |
| Duration | 566s |

**Phase Gates:**

| Phase | Step range | Nav accuracy target | Actual (peak) | Status |
|---|---|---|---|---|
| A (warmup) | 0-500 | >= 60% | 64.8% (step 400) | Pass |
| B (growth) | 500-3700 | >= 50% | 68.3% (step 3400) | Pass |
| C (full) | 3700-5000 | >= 45% | 63.7% (step 5000) | Pass |

**Training Dynamics (key checkpoints):**

| Step | L_total | L_nav | L_anchor | L_critic | lr | Nav acc (mean) | t2(hard) | t3(easy) |
|---|---|---|---|---|---|---|---|---|
| 50 | 830.6 | 764.8 | 9.42 | 0.097 | 1.00e-4 | — | — | — |
| 200 | 264.5 | 252.5 | 0.085 | 0.038 | 1.00e-4 | 0.607 | 0.412 | 0.802 |
| 500 | 103.9 | 103.0 | 0.007 | 0.023 | 9.90e-5 | 0.648 | 0.467 | 0.828 |
| 1000 | 51.4 | 51.8 | 0.006 | 0.069 | 9.33e-5 | 0.676 | 0.487 | 0.865 |
| 2000 | 62.7 | 65.1 | 0.004 | 0.023 | 6.91e-5 | 0.614 | 0.437 | 0.792 |
| 2600 | 20.9 | 21.6 | 0.005 | 0.015 | 5.00e-5 | 0.679 | 0.507 | 0.852 |
| 3400 | 17.1 | 18.1 | 0.004 | 0.023 | 2.50e-5 | **0.683** | 0.485 | 0.882 |
| 3900 | 68.6 | 72.6 | 0.005 | 0.135 | 1.24e-5 | **0.720** | — | — |
| 5000 | 27.2 | 28.8 | 0.006 | 0.127 | ~0 | 0.637 | 0.530 | 0.743 |

**Per-Bank Navigation Accuracy (key checkpoints):**

| Bank | Step 200 | Step 1000 | Step 2600 | Step 3400 | Step 5000 | Difficulty |
|---|---|---|---|---|---|---|
| STRUCTURE | 0.480 | 0.465 | 0.580 | 0.435 | 0.650 | Hard |
| DOMAIN | 0.990 | 0.985 | 0.980 | 0.990 | 0.955 | Easy (saturated) |
| DEPTH | 0.480 | 0.535 | 0.560 | 0.600 | 0.340 | Hard |
| AUTOMATION | 0.275 | 0.460 | 0.380 | 0.420 | 0.600 | Hard |
| CONTEXT | 0.780 | 0.840 | 0.745 | 0.825 | 0.610 | Medium |
| DECOMPOSITION | 0.635 | 0.770 | 0.830 | 0.830 | 0.665 | Medium |

**PAB Profile:** `runs/NAV-002/NAV-002_pab_profile.json`

| PAB Metric | NAV-002 | NAV-001 | Assessment |
|---|---|---|---|
| Stability mean | 0.341 | 0.324 | Chaotic (>0.30) |
| Stability std | 0.289 | 0.317 | Slightly improved |
| Predictability (final) | 174.3 | 49.6 | Higher with val_loss tracking |
| Stability regime | **chaotic** | **chaotic** | Genuine training dynamics |
| Crystallization rate | 0.0006 | 0.0 | Now measured |
| Early stop epoch | 450 | — | Now detected |
| Live PAB channels | **8/8** | 3/8 | **All channels live** |
| Generalization gap | Active (range -24 to +69) | Dead | Overfitting visible late |
| Representation evolution | Active (0→0 convergence) | Dead | Bridge freezes by step 3000 |
| Ternary crystallization | 0.9998→1.0000 | Dead | Signs fixed from init |
| Tier2 (hard banks) | 0.30-0.56 | Dead | Structure/automation/depth |
| Tier3 (easy banks) | 0.57-0.88 | Dead | Domain/context/decomposition |
| Loss components | All tracked | Dead | L_nav dominates |
| Adaptive weights | Tracked | Dead | Critic gains most weight |

**Key Findings from NAV-002:**

1. **PAB instrumentation complete**: 0/8 → 8/8 live channels, providing comprehensive training diagnostics
2. **Cosine LR schedule works**: Learning rate properly decays from 1e-4 → ~0, visible in loss trajectory
3. **Phase C destabilizes**: After step 3700, accuracy drops ~3% as harder proofs are introduced; some banks (depth) drop catastrophically
4. **Ternary crystallization near-instant**: Weight signs are 99.85% stable from step 100, reaching 100% by end — the ternary structure crystallizes at initialization, not during training
5. **Representation evolution freezes early**: Bridge embeddings stop changing by step 3000 (cosine similarity → 1.0)
6. **Generalization gap insight**: Starts high (+65 val>train), converges to ~0, then goes negative (-20) in late training — eval subset becomes easier than training data after Phase C
7. **Best mean accuracy: 0.72 at step 3900** (just after Phase C transition), best pre-C: 0.683 at step 3400
8. **"Chaotic" is the true regime**: High loss variance (16-68 within 200 steps) is structural, not a measurement artifact

**UW-SO Weight Evolution:**

| Weight | Start (step 50) | Mid (step 2000) | End (step 5000) |
|---|---|---|---|
| w_nav | 0.250 | 0.228 | 0.215 |
| w_anchor | 0.250 | 0.254 | 0.262 |
| w_progress | 0.250 | 0.236 | 0.229 |
| w_critic | 0.250 | 0.281 | 0.294 |

**Observations:**

---

## Experiment Cycle 2: Retrieval Comparison

*Purpose: Stream 1 core question -- does navigational retrieval match or exceed dense retrieval?*

### EXP-2.1: Navigational vs Dense Retrieval ⚠️ BLOCKED BY DATA GAP

**Date:** 2026-03-14
**Script:** `scripts/eval_retrieval.py --config configs/wayfinder.yaml --checkpoint models/NAV-002_step5000.pt --samples 500`
**Checkpoint:** NAV-002 (step 5000, nav_acc=72.0%)

| k | Nav recall@k | Dense recall@k | Delta | Nav time (ms) | Dense time (ms) | Speedup |
|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0009 | -0.0009 | 6457.9 | 64.9 | 0.01x |
| 4 | 0.0000 | 0.0053 | -0.0053 | — | — | — |
| 8 | 0.0000 | 0.0064 | -0.0064 | — | — | — |
| 16 | 0.0000 | 0.0090 | -0.0090 | — | — | — |

**ROOT CAUSE: Data coverage gap.** Only **30% of ground-truth premises in nav_eval.jsonl exist in proof_network.db** (212 of 706 checked across 100 examples). Both nav and dense retrieval are at floor because the retrieval ceiling is ~30%, not 100%. This is a **data pipeline issue**, not a model failure.

**Diagnosis:** The proof network was built from `extract_proof_network.py` which processes theorems with tactic traces (78K entities). Many premises referenced in the training/eval data are in Mathlib modules that weren't extracted as entities — they exist as import-accessible lemmas but not as nodes in the proof network.

**Coverage analysis (2026-03-14):**
- Total lemma entities in DB: 78,414 (from theorems with tactic traces only)
- Tactic entities in DB: **0** (tactic entity creation not yet run)
- Unique premises in nav_eval.jsonl: 6,880 — found in DB: **2,377 (34.5%)**
- Unique premises in nav_train.jsonl (10K sample): 8,000 — found in DB: **2,650 (33.1%)**
- Cause: Extraction created entities only for theorems with tactic traces. Many premises are imported Mathlib lemmas that exist as references but not as proof network entities.
- accessible_premises table has 104K links across 47K theorems, but premise_id references may point to entities not in the DB.

**Fix required before re-running:**
1. **Expand proof network** to include all Mathlib lemmas as entities (not just those with tactic traces) — these need at minimum a name, namespace, and bank positions (can be inferred from type/namespace)
2. **Build tactic entities** — `build_proof_network_db.py` has tactic entity code but 0 tactic entities exist in DB (likely skipped during initial run)
3. Re-run `anchor_gap_analysis.py` after expansion to confirm recall@16 still passes

**Re-run on expanded DB (2026-03-14, Colab T4, 482 samples):**

| k | Nav recall | Dense recall | Nav cond_recall | Dense cond_recall |
|---|-----------|-------------|-----------------|-------------------|
| 1 | 0.0000 | 0.0012 | 0.0083 | 0.0097 |
| 4 | 0.0000 | 0.0066 | 0.0083 | 0.0154 |
| 8 | 0.0000 | 0.0092 | 0.0083 | 0.0185 |
| 16 | 0.0000 | 0.0159 | 0.0083 | 0.0255 |

Universe coverage: 85.9% mean (198 fully covered, 4 zero covered).
Timing: nav=1980ms, dense=19ms.

**Analysis:** Coverage improved (34.5% → 85.9%) but both methods are at floor. Nav raw recall is zero; conditional recall is 0.8% (non-zero but negligible). Dense conditional recall is 2.6% — also very low but consistently above nav. The 242K entity expansion diluted discrimination (see EXP-0.2 re-run).

**Target:** Nav recall@16 >= 80% of dense recall@16 → NOT MET.

**Stream 1 verdict:** [ ] Navigation matches/exceeds dense / [x] Dense superior / [ ] Inconclusive

### EXP-2.2: Spreading Activation Benefit ⚠️ BLOCKED BY DATA GAP

**Date:** 2026-03-14
**Script:** `scripts/eval_spreading.py --config configs/wayfinder.yaml --checkpoint models/NAV-002_step5000.pt --samples 500`

| k | No-spread recall@k | With-spread recall@k | Delta | Notes |
|---|---|---|---|---|
| 4 | 0.0000 | 0.0000 | +0.0000 | Both at floor |
| 8 | 0.0000 | 0.0000 | +0.0000 | Both at floor |
| 16 | 0.0000 | 0.0000 | +0.0000 | Both at floor |

| Timing | No-spread (ms) | With-spread (ms) |
|---|---|---|
| Average per-goal | 6205.3 | 6180.7 |

**Re-run on expanded DB (2026-03-14, Colab T4, 487 multi-step samples):**

| k | No-spread recall@k | With-spread recall@k | Delta |
|---|---|---|---|
| 4 | 0.0000 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 |
| 16 | 0.0000 | 0.0000 | 0.0000 |

Timing: no_spread=2124ms, with_spread=1794ms.

**Analysis:** Both methods at zero — spreading can't improve what doesn't work. The base retrieval is at floor due to the 242K entity dilution problem (see EXP-0.2 re-run). Spreading will become evaluable once discrimination is restored.

**Target:** Spreading adds >= 5% recall@16 on proof steps 3+ → NOT EVALUABLE (base retrieval at floor).

### EXP-2.3: Scoring Mechanism Comparison

**Date:** ____
**Protocol:** Same checkpoint, different scoring mechanisms in `navigate()`.

| Mechanism | recall@4 | recall@8 | recall@16 | Notes |
|---|---|---|---|---|
| multiplicative | | | | Pure product |
| confidence_weighted | | | | Default |
| soft_floor (eps=0.1) | | | | |
| geometric_mean | | | | |
| log_additive | | | | |

**Best mechanism:** ____
**Observations:**

---

## Experiment Cycle 3: Proof Search Evaluation

*Purpose: End-to-end theorem proving on standard benchmarks.*

### EXP-3.1: Smoke Test (trivial theorems)

**Date:** 2026-03-15
**Script:** `scripts/run_benchmark.py --backend pantograph --theorems data/init_logic_benchmark.jsonl`
**Milestone:** First end-to-end proof search with real Lean 4 verification via Pantograph.

**Benchmark evolution (Init-only logic, 30 theorems):**

| Version | Proved | Rate | Avg attempts | Key change |
|---------|--------|------|-------------|------------|
| v1 (no fallback) | 13/30 | 43.3% | 342 | Hammer only — all navigation-required theorems fail |
| v2 (+intro/assumption/constructor) | 19/30 | 63.3% | 222 | +6 from structural intro, automation 100% |
| **v3 (+intros/exact?/aesop)** | **27/30** | **90.0%** | **64** | +8 from exact? exhaustive search after intros |

**Two-lane separation:**

| Lane | v1 | v2 | v3 | Notes |
|------|-----|-----|-----|-------|
| Automation-solvable (14) | 12 (85.7%) | 14 (100%) | 14 (100%) | Hammer delegation: omega/decide/simp |
| Navigation-required (16) | 1 (6.2%) | 5 (31.2%) | 13 (81.2%) | Structural fallback: intros → exact? |

**Per-tactic contribution (v3 proved=27):**

| Tactic | Theorems closed | Source |
|--------|----------------|--------|
| exact? | 12 | Structural fallback (exhaustive search after intros) |
| omega | 7 | Hammer delegation |
| simp | 4 | Hammer delegation / structural |
| decide | 3 | Hammer delegation |
| trivial | 1 | Structural fallback |

**Remaining failures (3):**

| Theorem | Type | Needs | Why it fails |
|---------|------|-------|-------------|
| and_comm | p ∧ q → q ∧ p | ⟨h.2, h.1⟩ | Anonymous structure matching — exact? can't find it |
| and_assoc | (p ∧ q) ∧ r → p ∧ (q ∧ r) | ⟨h.1.1, h.1.2, h.2⟩ | Nested anonymous structure |
| iff_comm | (p ↔ q) → (q ↔ p) | h.symm | Method-style call on hypothesis |

**Key findings:**
1. **Pantograph integration works end-to-end** — full pipeline operational with real Lean 4 verification.
2. **Two-lane separation is clean** — automation (hammer) and navigation (structural fallback) are qualitatively different capabilities with different prove rates.
3. **Structural fallback is critical** — without it, navigation lane is 6.2%. With intros+exact?, it's 81.2%.
4. **exact? is the key tactic** — after intros introduces all binders, exact? exhaustively closes 12/27 proved theorems.
5. **NAV-002's trained navigator contributes zero proofs** on Init-only goals — all proofs come from hammer or structural fallback. Real Mathlib evaluation is needed to test navigational retrieval.

**Implementation:** Added `_STRUCTURAL_TACTICS` list and `_try_structural_fallback()` to `src/proof_search.py`. Tried before navigator candidates, costs only 10 Lean calls per search iteration.

**Stop/go:** ✅ PASS — 90.0% exceeds 80% target. Automation lane at 100%, navigation lane at 81.2%.

### EXP-3.2: Mathlib Benchmark — Three-Mode Comparison

**Date:** 2026-03-16
**Script:** `scripts/run_mathlib_benchmark.sh` (sequential 3-mode runner)
**DB:** proof_network.db (via proof_search resolution)
**Lean:** Pantograph 0.3.13 + Lean 4.27.0 + Mathlib v4.27.0 (data/lean_project/)
**Checkpoint:** NAV-002_step5000.pt
**Theorems:** 50 from nav_eval.jsonl (first step per unique theorem_id)

| Mode | Proved | Rate | Time (s) | Avg Attempts | Learned Lane |
|------|--------|------|----------|-------------|-------------|
| learned_only | 0/50 | 0% | 261 | 10.7 | 0 |
| learned_structural | 11/50 | 22% | 1137 | 38.7 | 0 |
| **full** | **12/50** | **24%** | **3835** | **48.0** | **0** |

**Close lane breakdown (full mode):**

| Lane | Count | Theorems |
|------|-------|----------|
| automation | 4 | Submodule.mem_dualAnnihilator, Module.End.eigenspace_zero, ne_one_of_map, Poly.sumsq_nonneg |
| solver_bootstrap | 7 | RingHom.ker_equiv_comp, Finsupp.neLocus_zero_right, dist_prod_same_right, ZSpan.fundamentalDomain_pi_basisFun, LinearMap.rTensor_comp, CategoryTheory.Limits.prod.triangle, List.destutter'_cons_neg |
| structural_core | 1 | Rat.cast_div_of_ne_zero |
| learned | 0 | — |
| skipped | 7 | (goal creation failed — universe/notation) |
| failed | 31 | — |

**Lane sequence distribution (12 proved):**

| Sequence | Count | Pattern |
|----------|-------|---------|
| structural_core→solver_bootstrap | 5 | intros → simp/aesop |
| automation | 4 | hammer with premise suggestions |
| solver_bootstrap | 2 | direct simp/aesop |
| structural_core | 1 | intros → constructor |

**Key findings:**
1. **Learned lane = 0 across ALL modes.** NAV-002's trained tactic prior contributes zero proofs on real Mathlib theorems. The bottleneck is conclusively localized to the learned local tactic prior.
2. **Hammer adds only 1 theorem** over bootstrap-only (12 vs 11). The 4 automation proofs use `aesop (add safe [premise_list])`. However, later paired ablation on the same 50-theorem slice showed no theorem-count delta with vs. without learned premises, so theorem-level premise value remains unvalidated on this benchmark.
3. **Dominant pattern: structural_core→solver_bootstrap** (intros→simp/aesop) — 5/12 proved theorems.
4. **24% prove rate from bootstrap + hammer** is the baseline that the learned lane must beat.
5. **7/50 skipped** due to universe/notation issues in goal creation — engineering fix, not fundamental.

**Stop/go:** [ ] NOT MET — learned lane must contribute >0 proofs before Phase 4+ proceeds. Retraining NAV with new domain labels and v3 DB is the next step.

### EXP-3.2a: Premise-Value Ablation on Mathlib Slice

**Date:** 2026-03-16
**Protocol:** Re-run `full` mode on the same 50-theorem Mathlib slice with `--no-learned-premises`.

| Configuration | Proved | Delta |
|---|---|---|
| full | 12/50 | — |
| full + `--no-learned-premises` | 12/50 | 0 |

**Interpretation:**
- Task A (theorem-level premise geometry) is **not yet validated by theorem-count uplift** on this slice.
- This does NOT imply theorem-level retrieval is useless. It implies the current benchmark is dominated by bootstrap/automation closures, so theorem-count is the wrong primary metric for local premise usefulness.
- Executor training should therefore proceed with oracle premises first, then measure the oracle→retrieved gap directly on residual-goal tasks.

### EXP-3.2b: Residual Executor — Goal Only vs Oracle Premise

**Date:** 2026-03-16
**Dataset:** `data/residual_train.jsonl` — 245K post-structural residual examples
**Model:** `src/residual_executor.py` — 166K parameter 6-family classifier
**Input:** MiniLM goal embeddings, with or without ground-truth premise embeddings

| Model | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| Goal only | 28.0% | 73.8% | 0.264 |
| Goal + oracle premise | **32.2%** | **77.4%** | **0.299** |
| Delta | +4.2pp | +3.6pp | +0.035 |

**Per-family recall at epoch 15:**

| Family | Goal only | Goal + premise | Delta |
|---|---|---|---|
| `rw` | 0.20 | **0.36** | **+0.16** |
| `simp` | 0.38 | 0.38 | 0.00 |
| `exact` | 0.39 | 0.35 | -0.04 |
| `refine` | 0.31 | **0.45** | **+0.14** |
| `apply` | 0.31 | 0.23 | -0.08 |
| `other` | 0.23 | 0.23 | 0.00 |

**Key findings:**
1. **The premise signal is real.** Oracle premises improve residual family prediction overall and especially on `rw` and `refine`, exactly the families where the identity of the lemma matters most.
2. **Theorem-level retrieval and residual execution are different problems.** Residual retrieval against theorem-level v3 anchors is near-zero for step targets; the local executor should be trained directly on residual goals, not blocked on theorem-level retrieval.
3. **The correct local architecture is two-stage.** First predict the tactic family on the normalized local goal; then ground specific premises only for premise-sensitive families.
4. **Premise conditioning must be family-sensitive.** `simp` is unchanged and `exact`/`apply` regress, so adding lemma context everywhere is not the right rule.

**Architecture update:** The theorem-level system is re-scoped as temporal orchestration and frontier collapse. The residual executor becomes the primary learned local policy. Retrieval quality is then measured by how closely retrieved-premise performance approaches this oracle-premise upper bound.

### EXP-3.2c: Temporal Controller Implementation Audit

**Date:** 2026-03-16
**Artifact:** `src/temporal_controller.py`, `tests/test_temporal_controller.py`

**What exists:**
- `TemporalState`
- `OrchestrationDecision`
- rule-based `TemporalController` v0 with four phases:
  - `structural_setup`
  - `local_close`
  - `automation_close`
  - `repair_or_replan`
- unit tests for phase detection, goal selection, escalation, lane order, and replan triggers

**Audit finding (superseded by later wiring):** the standalone controller was implemented before runtime integration. See EXP-3.2d for the operative benchmark path.

| Status | Finding |
|---|---|
| implemented | Standalone controller module and unit tests |
| implemented later | Wiring into `proof_search.search()` |
| still missing | Wiring into `arbiter.som_search()` |
| partial | Runtime use of `lane_order`; `budget_slice` and `replan` still not full control surfaces |
| implemented later | Search traces / benchmark outputs exposing temporal decisions |

**Interpretation:**
- The architecture should include temporal control as a first-class Arbiter component.
- The current codebase does **not** yet use it to drive theorem proving.
- Benchmarks therefore still reflect the older static scheduler: critic-based goal selection and fixed lane ordering.

**Decision:** Use this as the pre-wiring audit only. The authoritative runtime status is now: operative in `proof_search.search()` shadow/active modes; not yet fully wired into `arbiter.som_search()`; do not claim proof benefit until `TC1/TC2` paired benchmarks beat the static baseline.

### EXP-3.2d: Temporal Controller Three-Mode Mathlib Benchmark

**Date:** 2026-03-16
**Script:** `scripts/run_tc_benchmark.sh`
**Theorems:** 50 Mathlib (same set as EXP-3.2)

| Mode | Proved | Rate | Time (s) | Notes |
|------|--------|------|----------|-------|
| off (static) | 12/50 | 24% | 3090 | Baseline — hardcoded hammer→structural→learned |
| shadow (log only) | 12/50 | 24% | 3245 | +5% overhead from TC state maintenance |
| active (TC routing) | 12/50 | 24% | 4878 | Same proves, +58% slower — TC explores more paths but learned lane is empty |

**TC Shadow Trace Analysis (shadow mode, 50 theorems):**

| Metric | Value |
|--------|-------|
| Total TC steps | 660 |
| Replans triggered | 0 |
| Phase distribution: structural_setup | 263 (40%) |
| Phase distribution: local_close | 202 (31%) |
| Phase distribution: repair_or_replan | 195 (30%) |

**Key finding:** All 12 proved theorems succeed in `structural_setup` phase at escalation level 0. None needed phase transition or escalation. The 38 failures burn through all three phases without success. The TC correctly diagnoses stagnation but the learned lane has nothing useful to offer in `local_close` phase — confirming the residual executor is the bottleneck, not the routing.

### EXP-3.2e: Residual Tactic Parsability and Constrained Output Audit

**Date:** 2026-03-16
**Dataset:** Post-structural residual steps from `data/residual_train.jsonl`
**Question:** Is the local output space structured enough to justify a constrained decoder instead of raw tactic-string generation?

**Parsability by family:**

| Family | Parsable rate | Notes |
|---|---|---|
| `rw` | 100% | Rewrite list with optional backward arrows |
| `exact` | 100% | Single local term / chained expression |
| `apply` | 100% | Single local term |
| `refine` | 100% | Local term with holes |
| `simp` | 92% | Bare `simp`, `simp [lemmas]`, or `simpa ... using ...` |

**Aggregate finding:** `82%` of residual tactics are parsable into **family + structured arguments**.

**Template subset finding:** `32%` of residual tactics are already covered by the first simple template subset compiled from premise names / local symbols alone.

**Representative hard cases:**
- `rw [← mul_le_mul_left (norm_pos_iff.2 h), ← sq]`
- `exact (hfg x hx).trans (le_abs_self _)`
- `exact continuous_im.isOpen_preimage _ isOpen_Ioi`
- `refine isIso_of_yoneda_map_bijective _ (fun A => ⟨?_, ?_⟩)`

**Interpretation:**
1. The local problem is not free-form Lean generation.
2. The right representation is a **family-specific IR** over local symbols, applications, projections, holes, and rewrite directions.
3. `32%` exact-template coverage is already enough to start testing learned-lane closure on the easy structured subset.
4. The remaining `68%` are still constrained local synthesis, not open-ended theorem proving.

**Architecture consequence:** Add `ActionIR` / `TermExpr` / `RewriteAtom` and deterministic lowering. The executor stack becomes:
- family prediction
- family-specific constrained decoding
- deterministic lowering to Lean
- Lean verification

### EXP-RES-001: Residual Tactic Family Classifier

**Date:** 2026-03-16
**Script:** `scripts/train_residual_cached.py`
**Data:** 245K post-structural residual examples, pre-computed MiniLM embeddings
**Architecture:** 384d → 256d → 256d → 6-class (rw/simp/exact/refine/apply/other), class-weighted CE loss

| Metric | Goal Only | Goal + Oracle Premise | Delta |
|--------|-----------|----------------------|-------|
| Top-1 | 28.0% | **32.2%** | **+4.2pp** |
| Top-3 | 73.8% | **77.4%** | **+3.6pp** |
| Macro-F1 | 0.264 | **0.299** | **+0.035** |

**Per-family recall (goal+premise vs goal-only):**

| Family | Goal Only | Goal+Premise | Delta |
|--------|-----------|-------------|-------|
| rw | 0.20 | **0.36** | **+0.16** |
| simp | 0.38 | 0.38 | 0.00 |
| exact | 0.39 | 0.35 | -0.04 |
| refine | 0.31 | **0.45** | **+0.14** |
| apply | 0.31 | 0.23 | -0.08 |
| other | 0.23 | 0.23 | 0.00 |

**Key findings:**
1. Oracle premises improve family prediction by +4.2pp top-1, confirming the 2-stage architecture.
2. `rw` gets the biggest boost (+16pp) — it needs to know which lemma to rewrite with.
3. `refine` also improves substantially (+14pp).
4. `simp` is unchanged — doesn't depend on specific premises.
5. 75% top-3 is operationally useful as a search entropy gate NOW.

### EXP-RES-002: Premise Value Ablation (Mathlib)

**Date:** 2026-03-16
**Protocol:** Full Mathlib benchmark with vs without navigator premise suggestions in hammer calls.

| Condition | Proved | Rate |
|-----------|--------|------|
| With premises | 12/50 | 24% |
| Without premises | 12/50 | 24% |
| **Delta** | **0** | **0%** |

**Finding:** Navigator premise retrieval does not yet add value at the theorem-prove level. The hammer tactics (aesop, omega, simp, decide) work equally well without premise suggestions. Task A (premise routing) is unvalidated but not dead — the retrieval signal may not be exposed at the right granularity yet.

### EXP-RW-001: Rewrite Decoder (ActionIR, Unscoped Vocabulary)

**Date:** 2026-03-16
**Script:** `scripts/train_rw_decoder.py`
**Data:** 5K rw ActionIR examples from `data/rw_actionir_train.jsonl` (14K total available)
**Architecture:** RwDecoder (365K params) — two-stage: shape prediction (count + direction) + pointer attention over vocabulary

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| Count accuracy | 42.2% | ~40% (majority class = 1 rewrite) | Partial signal — need within-1 and majority baseline |
| Direction accuracy | 76.2% | 79% (always forward) | **Below baseline** — not working yet |
| Leaf top-1 | 17.8% | 0.02% (random on 5K vocab) | **Real signal** — 890x random baseline |
| Leaf top-5 | 23.0% | 0.1% (random) | Modest lift from top-1 |

**Key findings:**
1. **Leaf selection works.** 18% top-1 on 5000-way unscoped vocabulary from goal embedding alone is real signal — the model learns goal-to-lemma association.
2. **Direction head does not work.** 76% is below the 79% always-forward baseline. Should be made rule-based or treated as beam alternative.
3. **Candidate scoping is the highest-leverage next step.** Model currently searches 5K symbols when accessible premises per theorem are ~200. Scoping should dramatically improve leaf accuracy.
4. **Top-5 lift is modest** (18% → 23%) — suggests either sharp-but-wrong confidence or correct premise not near top. Rank histogram needed.

**Next steps:**
- Scope vocabulary to accessible premises + local hypotheses
- Split leaf prediction: source type (hyp vs premise) then symbol
- Default direction to forward, try backward as beam alternative
- Add MRR, recall@10, recall@50 metrics

### EXP-RW-002: Scoped Vocabulary Ablation (rw leaf selection)

**Date:** 2026-03-16
**Protocol:** Same pointer model architecture, 4 scoping conditions on same train/eval split (13K train, 1K eval). 15 epochs each. Measures search-space reduction benefit vs model quality.

| Condition | Top-1 | Top-5 | MRR | Gold Coverage | Mean Scope |
|-----------|-------|-------|-----|---------------|------------|
| full (15K vocab) | 30.5% | 55.6% | 0.427 | 100.0% | 15037 |
| hyps only | **82.7%** | **98.2%** | **0.895** | 9.6% | 10 |
| hyps + theorem* | **42.1%** | **83.5%** | **0.599** | 99.0% | 15 |
| hyps + thm + top100 | 36.8% | 74.7% | 0.536 | 100.0% | 113 |

*theorem scope is a proxy oracle built from other examples of the same theorem — not deployable, but validates the pointer architecture at small scope.

**Key findings:**
1. **Local hypothesis selection is near-solved** — 82.7% top-1 when gold is a hypothesis (9.6% of cases). The pointer model almost always finds the right hypothesis.
2. **Scope reduction is the primary lever** — 30.5% → 42.1% top-1 just from reducing 15K → 15 symbols. Same model, same embeddings.
3. **Top-5 at scope=15 is 83.5%** — the correct premise is almost always in the top 5. MRR 0.599 means it's typically rank 2.
4. **Adding top-100 global hurts** (42.1% → 36.8%) — dilution is worse than missing coverage. The model doesn't need a bigger scope, it needs a better-filtered one.
5. **The model is NOT the bottleneck. The scope is.** Next gain comes from family-specific redex-aware scoping, not more model capacity.

**Caution:** Theorem scope uses other training examples from the same theorem — it leaks information and is not a deployable scoping method. It is an upper bound for what redex-aware accessible-premise filtering should achieve.

**Next steps:**
- Source gate: predict hyp vs premise before leaf selection
- Redex-aware scoping: filter accessible premises by lhs/rhs head symbol match with goal subexpressions
- Direction from symbolic matching (lhs matches goal → forward, rhs → backward), not learned head

### EXP-RW-003: Grammar Tier Split

**Date:** 2026-03-16
**Data:** Canonical rw eval (3,508 examples) split by grammar complexity.

| Tier | Description | Count | % |
|------|-------------|-------|---|
| rw0 | Bare single forward rewrite: `rw [lemma]` | 848 | 24% |
| rw1 | Bare single backward rewrite: `rw [← lemma]` | 213 | 6% |
| rw2 | Applied single rewrite: `rw [lemma args]` | 361 | 10% |
| rw3 | Multi-rewrite: `rw [a, b, ...]` | 2086 | 59% |

**Decision:** rw0 is the focused benchmark for the first formal learned lane. It has the simplest grammar (no direction learning needed, no applied arguments, no multi-atom composition) and the cleanest curriculum for validating the decoder architecture.

### EXP-RW-004: rw0 Semantic Benchmark (Canonical Names)

**Date:** 2026-03-16
**Data:** 80 rw0 step-0 examples from canonical eval (bare single forward rewrite, no prefix replay needed)
**Protocol:** Three conditions on same examples. Module loading via `load_header`. Canonical names from `annotated_premise`.

| Condition | Lean-valid | Rate | Rate (of started) |
|-----------|-----------|------|-------------------|
| Oracle source names | 5/80 | 6% | 14% |
| **Oracle canonical names** | **11/80** | **14%** | **30%** |
| Learned cosine top-5 | 0/80 | 0% | — (server crash) |

**Infrastructure breakdown:**

| Category | Count | % |
|----------|-------|---|
| goal_creation_fail | 43 | 54% |
| goal_started | 37 | 46% |

**Of started goals (37):**

| Category | Count | % |
|----------|-------|---|
| **oracle canonical Lean-valid** | **11** | **30%** |
| tactic_fail (wrong premise / inapplicable) | 24 | 65% |
| no_premise annotation | 2 | 5% |

**Key findings:**
1. **GoalStart@rw0 = 46%** — infrastructure baseline. Goal creation is the dominant infrastructure bottleneck.
2. **LeanValid@rw0 | started = 30%** — semantic baseline. When the environment is valid, oracle canonical rewrites succeed 30% of the time.
3. **These are separate tracks.** GoalStart is infrastructure; LeanValid|started is decoder quality.
4. **Learned cosine = 0%** — invalidated by Pantograph server crash. Also used theorem-sibling proxy scope (not deployable). Must rerun with real accessible premises and stable runner.
5. **The 70% oracle failure on started goals** needs further splitting: wrong premise for rewrite site vs needs applied argument vs rewrite semantically inapplicable.

**Benchmark definition (frozen):**
- Canonical names only (source names are secondary/deployability metric)
- Bare single rewrite only (rw0 tier)
- Started-goal conditional metric is primary
- No theorem-sibling proxy scope in learned conditions
- Pantograph stability is a prerequisite (server restart/retry needed)

**Next milestone:** Learned rw0 > cosine baseline on LeanValid@rw0|started, with real accessible-premise scope, stable runner, and theorem-faithful goal creation.

### EXP-RW-005: rw0 Harness Readiness Audit

**Date:** 2026-03-16
**Code under audit:** `src/lean_interface.py`, `scripts/run_rw0_benchmark.py`
**Protocol:** Audit implementation against `WAYFINDER_PLAN.md` and `PLAN_2.md`, then run a targeted Pantograph probe on current `rw0` examples plus full regression tests.

**Verified positives:**
- env-keyed goal/tactic caching exists
- crash failures are separated from semantic failures via `ServerCrashError`
- Tier C now uses sequential `goal_tactic` replay instead of embedding prefixes into a `load_sorry` script
- benchmark runner aborts remaining conditions on an example after a crash
- regression suite passes: `1679 passed, 5 subtests passed`

**Current runtime probe (real Pantograph, targeted sample):**

| Slice | Started | Result |
|------|---------|--------|
| step-0 probe | 2/3 | non-zero start coverage, Tier A active |
| step>0 probe | 0/3 | all `goal_creation_fail` before replay can matter |

**Audit findings:**
1. **Theorem-faithful Tier B is still missing.** Current fallback still relies on `_goal_via_sorry` heuristics rather than the planned theorem-specific wrapper plus `check_compile` preflight.
2. **Step>0 started-goal coverage is still the gate.** Tier C replay mechanism exists, but it cannot help unless Tier A or Tier B produces a theorem-level base state first.
3. **Site-addressed replay must treat `goal_id` as part of tactic identity.** Multi-goal replay now needs cache purity at the `(env_key, goal_state, goal_id, tactic)` level; otherwise cached results can leak across subgoals.
4. **Headline learned rw0 comparison is not ready.** A full learned-vs-cosine run would still be dominated by goal-creation failures rather than decoder quality.

**Decision:** The project is **stable for infrastructure iteration but not ready for the main learned `rw0` experiment**. Run infrastructure experiments next:

- Tier A vs Tier B step-0 coverage lift
- Tier C replay success conditional on a started base state
- oracle semantic rw0 benchmark on the started-goal subset

Only after those improve should the learned `rw0` vs cosine experiment be treated as a paper-quality result.

### EXP-RW-006: Infrastructure Readiness — Tier Coverage + Replay + Oracle

**Date:** 2026-03-16
**Script:** Inline Pantograph probes via `src/lean_interface.goal_via_file_context()` + `try_tactic()`
**Data:** 100 stratified step-0 rw0 examples (distinct theorems, seed=42) + 28 step>0 examples with step-0 siblings + oracle canonical on started subset

#### Part A: Tier A/B Step-0 Coverage (n=100)

| Metric | Value |
|--------|-------|
| **GoalStart** | **81/100 (81%)** |
| Tier A (`env_inspect` → `goal_start`) | 81/100 |
| Tier B incremental lift (`_goal_via_sorry` fallback) | **+0** |
| goal_creation_fail | 19/100 |

**Finding:** Tier A alone meets the ≥80% fast-path gate from PLAN_2.md §2. Current `_goal_via_sorry` Tier B adds zero lift — all 19 failures that Tier A misses, Tier B also misses. These are polymorphic/typeclass-heavy types that need the planned theorem-specific wrapper.

#### Part B: Tier C Replay Success | Started (n=28 step>0)

| Metric | Value |
|--------|-------|
| Base state obtained (Tier A) | 25/28 (89%) |
| **Replay success \| base started** | **0/25 (0%)** |
| prefix_replay_fail | 23 |
| prefix_replay_fail at index 0 | 21/23 (91%) |
| prefix_replay_fail at index 1 | 2/23 |
| goal_match_fail | 2 |

**Finding:** Tier C replay mechanism is structurally correct but the theorem-level state from Tier A lacks namespace/variable context for prefix tactics. 91% of failures are at the very first prefix tactic. This is the same scoping gap that real Tier B (theorem-specific wrapper with `open`/`variable` directives) would address. Tier C is blocked on Tier B, not on its own logic.

#### Part C: Oracle Canonical rw0 on Started Step-0 (n=100, started=81)

| Metric | Value |
|--------|-------|
| GoalStart | 81/100 (81%) |
| **LeanValid\|started (tactic accepted)** | **11/81 (13.6%)** |
| LeanValid\|started (goals closed) | 0/81 (0%) |
| LeanValid\|started (remaining goals) | 11/81 (13.6%) |

**Of started goals (81), failure taxonomy:**

| Category | Count | % |
|----------|-------|---|
| **identifier_scope** | **48** | **59%** |
| rw_inapplicable | 21 | 26% |
| tactic_fail (other) | 1 | 1% |
| oracle accepted | 11 | 14% |

**Findings:**
1. **identifier_scope is the dominant failure at 59%.** The goal is created but the canonical lemma name can't be resolved. This is the same namespace/scoping gap that blocks Tier C.
2. **LeanValid|started = 13.6%** on this 100-theorem stratified sample vs 30% on the original 80-example EXP-RW-004. The difference is sample composition — EXP-RW-004 used a non-stratified sample; this uses 100 distinct theorems (harder distribution).
3. **All 11 accepted tactics have remaining goals** (`success_with_goals`), zero full closes. The rw tactic applies but doesn't close the goal — expected for bare single rewrites.
4. **rw_inapplicable (26%)** means the canonical rewrite target expression doesn't appear in the goal under the current Tier A scoping. Real Tier B's `open` directives would resolve some of these.

**Cross-experiment conclusion:**
- Tier A hits 81% step-0 start rate (meets gate)
- Tier B is confirmed as the single highest-leverage infrastructure target: it would address identifier_scope (59% of semantic failures), prefix replay (91% fail at index 0), and the 19% step-0 creation failures
- No learned rw0 comparison should be attempted until identifier_scope drops below 5%

**Next action:** Implement real Tier B (theorem-specific wrapper with file-derived `open`/`variable` directives + `check_compile` preflight) per PLAN_2.md §2.

### EXP-RW-007: Tier B Active-Context Wrapper + Contamination Audit

**Date:** 2026-03-16
**Script:** Inline Pantograph probes via `src/lean_interface.goal_via_file_context()`
**Code change:** Replaced `extract_file_header()` (line-grep) with `extract_active_context()` (scope-stack reconstruction). Added `_server_contaminated` flag to force server restart after any Tier B `load_sorry` call, preventing namespace pollution of subsequent Tier A goals.

#### Part A: Contamination Check — Tier A/B + Split Oracle (n=100, seed=42)

**Context:** Initial Tier B run (without decontamination) showed GoalStart 82 but oracle collapsed from 11→1. The `_server_contaminated` guard was added and this rerun confirms Tier A is unaffected.

| Metric | Value |
|--------|-------|
| **GoalStart** | **82/100 (82%)** |
| Tier A | 81 |
| Tier B (active-context wrapper) | 1 |
| Tier B incremental lift | **+1** |
| goal_creation_fail | 18 |

**Oracle split by tier:**

| Tier | Started | Oracle Accepted | Rate |
|------|---------|-----------------|------|
| A | 81 | 11 | **13.6%** |
| B | 1 | 0 | 0% |
| **Total** | **82** | **11** | **13.4%** |

**Failure taxonomy (of 82 started):**

| Category | Count | % |
|----------|-------|---|
| identifier_scope | 49 | 60% |
| rw_inapplicable | 21 | 26% |
| tactic_fail | 1 | 1% |

**Findings:**
1. **Contamination confirmed and fixed.** Without the decontamination guard, Tier B `load_sorry` polluted the shared Pantograph server, collapsing Tier A oracle from 11→1. With the guard, Tier A oracle is exactly 11/81 (13.6%) — matching EXP-RW-006.
2. **Tier B lift = +1.** The `extract_active_context()` scope reconstruction produces one additional started goal that Tier A missed. This is nonzero but minimal. The remaining 18 failures are polymorphic types that need deeper wrapper work.
3. **identifier_scope still dominant at 60%.** This is the same level as EXP-RW-006. The active-context wrapper helps goal *creation* but does not help *tactic resolution* on Tier A goals, because Tier A goals are created via `goal_start` without file context.
4. **Elapsed: 238s** (vs 44s without Tier B). Each decontamination restart costs ~38s Mathlib reload. This is acceptable for benchmark runs but not for interactive iteration.

#### Part B: Tier C Replay (n=10 step>0, decontamination active)

| Metric | Value |
|--------|-------|
| Base state obtained | 9/10 |
| **Replay success \| started** | **0/9 (0%)** |
| prefix_replay_fail | 8 (all at index 0) |
| goal_match_fail | 1 |

**Finding:** Identical to EXP-RW-006. Tier C replay mechanism is correct but blocked by missing scope in the Tier A base state. The first prefix tactic fails because the theorem-level goal from `goal_start` doesn't have the `open`/`variable` context that the original proof assumed.

#### Interpretation

The active-context reconstruction (`extract_active_context()`) is structurally correct — it properly handles nested `section`/`namespace`/`end` with stack discipline. But the current Tier B wrapper only helps the 19% of step-0 goals that Tier A can't create at all. It does **not** help:
- Tier A goals where the tactic fails due to identifier_scope (60% of started failures)
- Tier C replay where prefix[0] fails due to missing scope (100% of replay failures)

Both of those require the tactic to be *applied* inside the file-context scope, not just the goal to be *created* in it. That is a deeper change: either apply the tactic inside the wrapper via `load_sorry`, or make Tier C replay run on a Tier B-created state instead of a Tier A-created state.

**Next action:** Investigate whether Tier B wrapper states (created via `load_sorry` with active context) allow prefix tactics and canonical rw tactics to resolve. If yes, route step>0 through Tier B base state instead of Tier A.

### EXP-RW-008: Tier B Compile Audit + Prefix[0] Failure Audit

**Date:** 2026-03-16
**Script:** Inline Pantograph probes with per-example server restart isolation

#### Part A: Tier B Compile/Context Audit (n=18 Tier A failures)

For each of the 18 step-0 theorems where Tier A fails, tested the active-context wrapper via `load_sorry`.

| Outcome | Count | % |
|---------|-------|---|
| **load_sorry FAIL** | **14** | **78%** |
| theorem not found in source | 4 | 22% |
| load_sorry SUCCESS | 0 | 0% |

**load_sorry failure categories:**

| Category | Count | Example |
|----------|-------|---------|
| Type elaboration (⋯ tokens, implicit holes) | 4 | `invalid {...} notation, expected type is not known` |
| Typeclass resolution stuck | 5 | `synthInstanceFailed` |
| Name/scope errors | 2 | `Unknown identifier`, `end` name mismatch |
| Pretty-print elision (`⋯`) | 3 | `The '⋯' token is used by the pretty printer` |

**Root cause:** `env_inspect.type.pp` returns a pretty-printed type string with elision tokens (`⋯`) and implicit argument holes that cannot round-trip through `load_sorry`. The active-context reconstruction is correct for `open`/`namespace`/`variable`, but the type itself is the problem — not the wrapper context.

**Implication:** Tier B wrapper approach using `env_inspect.type.pp` cannot cover the 18-19% of theorems where Tier A also fails. These theorems have types that cannot be reconstructed from the pretty-printed representation. A different approach is needed: either use the source declaration directly (with rename), or use a different Pantograph API.

#### Part B: Prefix[0] Failure Audit (n=10 step>0, 9 with base state)

For each step>0 example where prefix replay fails at index 0, recorded the exact Lean error.

| Category | Count | Meaning |
|----------|-------|---------|
| `tactic_inapplicable` | 4 | rw pattern not in theorem-level goal (needs `intro` first) |
| `unknown_identifier` | 3 | Short name needs `open` or is a hypothesis |
| `base_state_failed` | 1 | Tier A/B couldn't start |
| `goal_match_fail` | 1 | Replay completed, wrong subgoal |
| hypothesis reference | 1 | Prefix references `h` which doesn't exist in theorem-level goal |

**Key finding:** The dominant prefix[0] failure is **goal shape mismatch, not missing scope**. The Tier A goal is the raw theorem type (`∀ {p x y : M}, Prime p → ...`). The prefix tactics expect an intro'd state where binders have been consumed. LeanDojo records proofs from after `by`, where Lean's tactic block auto-introduces binders. So `rw [or_iff_not_imp_right]` fails because the `∨` pattern only exists after the `∀` binders are introduced.

This means:
- Missing `open` directives explain ~30% of prefix[0] failures (the `unknown_identifier` cases)
- Goal shape mismatch explains ~50% (the `tactic_inapplicable` cases)
- The fix for step>0 is not just better scoping — it requires either auto-introducing binders before replay, or using a `load_sorry`-based initial state that starts inside the `by` block

**Next action:** For step>0, the initial state needs to be the post-`by` state, not the raw theorem type. Two options: (1) auto-`intro` all binders before prefix replay, or (2) use `load_sorry(theorem_decl_with_sorry)` from the source file to get the post-`by` initial state. Option (2) is exactly the original `load_sorry(header + decl)` approach from the implementation plan — but with proper active-context reconstruction and fresh-name renaming to avoid collisions.

### EXP-RW-009: Declaration-Faithful Tier B (B→A Cascade)

**Date:** 2026-03-16
**Script:** Inline Pantograph probes with `_tier_b_file_wrapper` using `env_inspect` module + `sourceStart`/`sourceEnd`
**Code change:** Rewrote `_tier_b_file_wrapper` to use declaration-faithful path: resolve source file via `env_inspect.module`, extract original declaration from `sourceStart`/`sourceEnd`, rename theorem, replace proof with `by sorry`, wrap in `extract_active_context`, load via `load_sorry`. Reversed cascade to B→A (Tier B primary, Tier A fallback).

#### Part A: Step-0 Coverage + Oracle (n=100, seed=42)

| Metric | EXP-RW-006 (A only) | EXP-RW-007 (A→B) | **EXP-RW-009 (B→A)** |
|--------|---------------------|-------------------|----------------------|
| **GoalStart** | 81/100 (81%) | 82/100 (82%) | **91/100 (91%)** |
| Tier B | 0 | 1 | **54** |
| Tier A | 81 | 81 | 37 |
| goal_creation_fail | 19 | 18 | **9** |
| **LeanValid\|started** | 11/81 (13.6%) | 11/82 (13.4%) | **20/91 (22.0%)** |

**Oracle split by tier:**

| Tier | Started | Oracle Accepted | Rate |
|------|---------|-----------------|------|
| B (decl-faithful) | 54 | 13 | **24.1%** |
| A (fallback) | 37 | 7 | 18.9% |
| **Total** | **91** | **20** | **22.0%** |

**Failure taxonomy (of 91 started):**

| Category | Count | % |
|----------|-------|---|
| identifier_scope | 49 | 54% |
| rw_inapplicable | 21 | 23% |
| tactic_fail | 1 | 1% |

#### Part B: Step>0 Replay (n=10, seed=42)

| Metric | EXP-RW-006 | EXP-RW-007 | **EXP-RW-009** |
|--------|-----------|-----------|---------------|
| Base state obtained | 9/10 | 9/10 | 9/10 |
| **Replay success** | **0/9** | **0/9** | **0/9** |
| prefix_fail at idx 0 | 8/8 | 8/8 | **4/8** |
| prefix_fail at idx >0 | 0 | 0 | **4/8** |

**Findings:**
1. **GoalStart: 81% → 91%.** Declaration-faithful Tier B adds +10pp over Tier A alone. This is the first material infrastructure improvement. 9 failures remain (down from 19).
2. **LeanValid|started: 13.6% → 22.0%.** The Tier B post-`by` state produces higher oracle acceptance (24.1%) than Tier A's raw theorem type (18.9%). Short names now resolve in the declaration context.
3. **Tier B is now the primary path.** 54/91 started goals use Tier B; 37/91 fall back to Tier A. The B→A cascade is correct.
4. **identifier_scope still dominant at 54%.** The absolute count (49) is nearly unchanged — the new Tier B successes aren't in the identifier_scope failure set. The remaining identifier_scope failures are names that need `open` directives that aren't in the declaration's active scope.
5. **Step>0 prefix failures shifted deeper.** With Tier A, 100% of replay failures were at prefix[0] (wrong state class). With Tier B, 50% now fail at prefix[2] or [3] — meaning prefix[0] and [1] succeed on the Tier B base state. The replay mechanism works when the base state is correct.
6. **Still 0/9 replay success.** The prefix chain fails at deeper indices, but no full replays complete yet. The step>0 sample is small (n=10); a larger sample would show nonzero replay success.
7. **Runtime: 1292s (21.5 min).** Each Tier B `load_sorry` contaminates the server, requiring a ~38s restart. This is acceptable for benchmark runs but not for interactive work.

**Interpretation:** The declaration-faithful Tier B validates the core thesis — starting from the post-`by` state is the right abstraction for matching LeanDojo traces. GoalStart clears the 80% gate. LeanValid|started improved from 13.6% to 22.0%. The remaining gap is identifier_scope (54%) and rw_inapplicable (23%), both of which are semantic failures, not infrastructure failures.

**Next action:** The infrastructure is now good enough for a started-goal-conditional learned rw0 experiment. identifier_scope (missing `open` directives) is the next Tier B improvement target, but 22% oracle acceptance is a viable baseline denominator.

### EXP-RW-010: Canonical vs Source Oracle (Same 91 Started)

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe (goal creation + tactic evaluation in same iteration to avoid GoalState lifetime bug)
**Data:** Same n=100 step-0 sample (seed=42), 91 started via B→A cascade

| Condition | Accepted | Rate |
|-----------|----------|------|
| **Canonical** | **20/91** | **22.0%** |
| **Source** | **21/91** | **23.1%** |
| Both | 20/91 | — |
| Canonical only | 0 | — |
| Source only | 1 | — |

**Failure taxonomy comparison:**

| Category | Canonical | Source |
|----------|-----------|--------|
| identifier_scope | 49 | 11 |
| rw_inapplicable | 21 | 2 |
| tactic_fail | 1 | 2 |

**Findings:**
1. **Name realization gap is negligible.** Only 1 example where source succeeds and canonical doesn't. Canonical `ActionIR` lowering is nearly lossless.
2. **49 canonical identifier_scope vs 11 source.** The 38-example difference is entirely `open`-dependent short names in canonical (`exp` vs `Complex.exp`, `sum_const` vs `Finset.sum_const`). Source names use the original proof text which often has fully qualified names or matches the active `open` scope.
3. **Source's 21 rw_inapplicable → 2.** Most source tactics that are "inapplicable" in canonical form actually work in source form — this is because source tactics may use different argument patterns or elaboration hints.
4. **The 22% canonical oracle rate is confirmed and stable** across EXP-RW-009 and EXP-RW-010.

**Implication for the decoder:** The learned decoder should target canonical names. The canonical→source gap is <2%, so there's no meaningful improvement from switching to source-name prediction. The dominant gap is identifier_scope (54%), which is an infrastructure problem (missing `open` directives), not a decoder problem.

**GoalState lifetime bug noted:** The initial experiment run (two-pass: collect goals, then evaluate) showed 1/91 oracle due to GoalState objects being invalidated by decontamination restarts between examples. Fixed by evaluating tactics in the same loop iteration as goal creation. The benchmark runner (`run_rw0_benchmark.py`) is already correct (single-pass per example).

### EXP-RW-011: Oracle vs Cosine rw0 Step-0 (Real Accessible-Premise Scope)

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe with MiniLM encoder + `scope_for_rw` + real accessible premises from `proof_network_v3.db`
**Data:** Same n=100 step-0 sample (seed=42), 91 started via B→A cascade
**Protocol:** Two conditions on same started examples. Cosine uses `scope_for_rw(goal, accessible_premises, max_premises=30)` then MiniLM cosine ranking. Top-1 symbol tried as `rw [sym]` and `rw [← sym]`.

| Condition | Accepted | Rate |
|-----------|----------|------|
| Oracle canonical | 20/91 | 22.0% |
| **Cosine top-1** | **26/91** | **28.6%** |

**Scope stats:**

| Metric | Value |
|--------|-------|
| Mean scope size | **10.9** |
| Gold in accessible | 78/83 (94%) |
| Gold in scope | 78/83 (94%) |
| No premise annotation | 8 |
| Gold cosine rank (when in scope) | mean=1.5, **median=1** |

**Oracle failure taxonomy (of 91 started):**

| Category | Count |
|----------|-------|
| identifier_scope | 49 |
| rw_inapplicable | 21 |
| tactic_fail | 1 |

**Findings:**
1. **Cosine top-1 beats oracle canonical: 28.6% vs 22.0% (+6.6pp).** This is because cosine uses fully-qualified premise names from the DB (e.g., `Finset.sum_const`), which resolve in the Tier B post-`by` context. Oracle canonical uses short names from the ActionIR (e.g., `sum_const`), which hit identifier_scope failures.
2. **Scope quality is excellent.** Gold in scope = 94%, mean scope = 10.9 (well under the ≤20 gate from PLAN_2.md §3). Gold median cosine rank = 1 — the correct premise is almost always the top-ranked symbol.
3. **The 22% oracle canonical rate is the real semantic baseline** for the canonical `ActionIR` decoder. The 28.6% cosine rate demonstrates that scope + fully-qualified names recover most of the identifier_scope gap.
4. **The 6.6pp cosine advantage is almost entirely from name qualification**, not from better premise selection. Oracle canonical knows the correct premise but can't resolve its short name. Cosine doesn't know the correct premise but uses qualified names that resolve.

**Implication for the learned decoder:**
- The learned decoder should emit **fully-qualified names** (or the scope should do the qualification), not short canonical names
- The cosine baseline is now a real target: learned rw0 should beat 28.6% on LeanValid|started
- Alternatively, fixing identifier_scope in the oracle (adding `open` directives to Tier B) would raise the oracle ceiling above cosine, restoring it as the upper bound

**Milestone reached:** This is the first valid cosine-vs-oracle comparison on rw0 with real accessible-premise scope and a stable, non-crashing runner. EXP-RW-004's cosine condition (0%, server crash + proxy scope) is now superseded.

### EXP-RW-012: Qualified-Name Oracle Ceiling + Three-Condition Comparison

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe with three conditions per started example
**Data:** Same n=100 step-0 sample (seed=42), 91 started, 83 with `annotated_premise`

| Condition | Accepted | Denominator | Rate |
|-----------|----------|-------------|------|
| Oracle canonical (short name) | 20 | 91 started | 22.0% |
| **Oracle qualified (full name)** | **50** | **83 with premise** | **60.2%** |
| Cosine top-1 (qualified) | 26 | 91 started | 28.6% |

**Key findings:**
1. **Oracle qualified ceiling = 60.2%.** The true upper bound for rw0 on this harness is 3x the short-name oracle. The 22%→60% gap is entirely identifier_scope from short names — not wrong premises, not wrong rewrite direction, not inapplicable rewrites.
2. **Cosine top-1 is at 47% of the qualified oracle ceiling** (28.6/60.2). This means cosine picks the right premise about half the time when it could be right. There is significant room for a learned decoder to improve.
3. **The correct decoder target is the scoped symbol identity**, not the short canonical name. The lowering layer should emit the qualified name from the scope entry. `ActionIR` should store `entity_id` or `full_name`, with `surface_name` as diagnostics only.
4. **8 examples have no `annotated_premise`** — these cannot be evaluated on the qualified oracle condition. The 60.2% rate is on the 83 with annotations.

**Design decision (frozen):**
- Primary eval mode: **qualified-name lowering** (use `full_name` from scope entry)
- `ActionIR` contract: decoder predicts semantic symbol identity (pointer into scoped vocabulary), lowering emits qualified name
- Short-name / canonical-name metrics demoted to secondary / diagnostics
- Cosine baseline: **28.6%** on qualified names (the target for the learned decoder)
- Oracle ceiling: **60.2%** on qualified names (the best possible with perfect premise selection)

**Frozen metrics for learned rw0 comparison:**

| Metric | Value | Source |
|--------|-------|--------|
| GoalStart@step0 | 91% | EXP-RW-009 |
| Oracle canonical LeanValid\|started | 22.0% | EXP-RW-009 |
| Oracle qualified LeanValid\|started | 60.2% | EXP-RW-012 |
| **Cosine top-1 LeanValid\|started** | **28.6%** | **EXP-RW-011** |
| Gold in scope | 94% | EXP-RW-011 |
| Mean scope size | 10.9 | EXP-RW-011 |
| Gold cosine rank median | 1 | EXP-RW-011 |

**Next milestone:** Learned rw0 decoder > 28.6% cosine baseline on LeanValid|started with qualified-name lowering.

### EXP-RW-013: Qualified rw0 Step-0 — Same Denominator

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe, qualified-name lowering for all conditions
**Data:** n=100 step-0 (seed=42) → 91 started → **78 overlap** (started + has `annotated_premise` + gold in accessible premises)
**Protocol:** All conditions evaluated on the same 78-example overlap set. Oracle uses `rw [full_name]` with both forward and backward. Cosine uses `scope_for_rw(max_premises=30)` + MiniLM cosine ranking, top-1 with both directions.

| Condition | Accepted | n | Rate |
|-----------|----------|---|------|
| **Oracle qualified** | **49** | **78** | **62.8%** |
| **Cosine top-1** | **26** | **78** | **33.3%** |
| Both | 25 | 78 | — |
| Oracle only | 24 | 78 | — |
| Cosine only | 1 | 78 | — |

**Scope stats (overlap set):**

| Metric | Value |
|--------|-------|
| Gold in scope | 78/78 (**100%**) |
| Mean scope size | 11.3 |
| Cosine gold top-1 | 35/78 (**45%**) |
| Cosine gold top-5 | 68/78 (**87%**) |

**Findings:**
1. **Oracle qualified ceiling = 62.8%.** On the clean overlap set, the fully-qualified gold premise succeeds as a rewrite 63% of the time. The other 37% are semantically inapplicable rewrites (wrong rewrite site, needs arguments, etc.).
2. **Cosine top-1 = 33.3%.** Cosine gets the right answer about half as often as the oracle (33.3/62.8 = 53%). The gap is entirely premise selection — both use the same qualified-name lowering.
3. **Cosine gold top-1 = 45%, top-5 = 87%.** The correct premise is cosine-top-1 in 45% of cases and top-5 in 87%. This confirms the finding from EXP-RW-002: the pointer model at scope ≤15 should beat cosine, because the correct premise is almost always in the top 5 but often not rank 1.
4. **Gold in scope = 100%.** The `scope_for_rw` scoper with real accessible premises captures the gold premise in every case on this overlap set. Scoping is solved for this tier.
5. **24 oracle-only examples** — these are cases where the gold premise is correct but cosine ranks a different symbol higher. A learned decoder that sees goal structure (not just embedding similarity) should recover some of these.
6. **1 cosine-only example** — cosine accidentally picks a different symbol that also works as a rewrite. Negligible.

**Frozen same-denominator baselines (n=78 overlap):**

| Metric | Value |
|--------|-------|
| Oracle qualified LeanValid | **62.8%** |
| **Cosine top-1 LeanValid** | **33.3%** |
| Gold in scope | 100% |
| Cosine gold top-1 | 45% |
| Cosine gold top-5 | 87% |

**The learned decoder must beat 33.3% on this overlap set to demonstrate value over cosine.**

The gap between cosine (33.3%) and oracle (62.8%) = 29.5pp. That is the learnable margin. The cosine gold top-5 at 87% suggests a learned reranker over the top-5 could recover much of this gap without needing better retrieval — just better selection within the already-good scope.

### EXP-RW-013b: Learned Decoder vs Cosine vs Oracle (Same Denominator)

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe with RW-001 decoder (epoch 4, leaf_top1=18.8% on 5K vocab)
**Data:** Same n=78 overlap set from EXP-RW-013
**Protocol:** RW-001 `predict()` on scoped vocabulary (~11 symbols), qualified-name lowering. Top-1: greedy argmax. Top-5: beam over top 5 pointer logits, try each with forward + backward.

| Condition | Accepted | n=78 | Rate |
|-----------|----------|------|------|
| Oracle qualified | 49 | 78 | 62.8% |
| **Learned top-5** | **30** | **78** | **38.5%** |
| Cosine top-1 | 26 | 78 | 33.3% |
| Learned top-1 | 3 | 78 | 3.8% |

**Findings:**
1. **Learned top-5 beats cosine top-1: 38.5% vs 33.3% (+5.2pp).** The RW-001 pointer decoder, trained on 5K unscoped vocab, produces a top-5 beam that outperforms cosine greedy selection when evaluated on the real ~11-symbol scoped vocabulary with qualified-name lowering. This is the first evidence that the learned decoder adds value over the cosine baseline.
2. **Learned top-1 is poor at 3.8%.** The pointer's greedy argmax is not well-calibrated for this scoped vocabulary — it was trained on a much larger vocabulary. The signal is in the ranking, not the peak.
3. **The beam is the right evaluation mode for the decoder.** At top-5, the decoder covers 38.5/62.8 = 61% of the oracle ceiling. At top-1, only 3.8/62.8 = 6%. A small beam (5 candidates × 2 directions = 10 Lean calls) is cheap and effective.
4. **The decoder was not trained for this scope.** RW-001 was trained on a 5K global vocab with proxy scope. Retraining on the real ~11-symbol scoped vocab with canonical data should substantially improve top-1 calibration.

**Frozen comparison (n=78, same denominator, qualified-name lowering):**

| Condition | LeanValid | % of Oracle |
|-----------|-----------|-------------|
| Oracle qualified | 49/78 (62.8%) | 100% |
| **Learned top-5** | **30/78 (38.5%)** | **61%** |
| Cosine top-1 | 26/78 (33.3%) | 53% |
| Learned top-1 | 3/78 (3.8%) | 6% |

**Next steps:**
1. Retrain decoder on real scoped vocab from canonical rw data (not proxy 5K scope)
2. Add cosine top-5 comparison for fair beam-vs-beam evaluation
3. Investigate learned+cosine ensemble (cosine reranks learned top-5, or vice versa)

### EXP-RW-014: Matched-Budget Beam + Rank Diagnostics

**Date:** 2026-03-17
**Script:** Single-pass Pantograph probe with cosine top-5 + learned top-5 + rank diagnostics
**Data:** Same n=78 overlap set
**Protocol:** All conditions use qualified-name lowering. Top-k beam tries each of the k highest-ranked symbols with `rw [sym]` and `rw [← sym]` (2k Lean calls per condition).

| Condition | Accepted | n=78 | Rate | Lean calls |
|-----------|----------|------|------|------------|
| Oracle qualified | 49 | 78 | 62.8% | 2 |
| **Cosine top-5** | **48** | **78** | **61.5%** | **10** |
| Learned top-5 | 30 | 78 | 38.5% | 10 |
| Cosine top-1 | 26 | 78 | 33.3% | 2 |
| Learned top-1 | 3 | 78 | 3.8% | 2 |

**Gold rank diagnostics (n=78, gold in scope):**

| Metric | Cosine | Learned |
|--------|--------|---------|
| MRR | **0.634** | 0.235 |
| R@1 | **45%** | 5% |
| R@3 | **79%** | 24% |
| R@5 | **87%** | 44% |

**Findings:**
1. **Cosine top-5 nearly matches oracle: 61.5% vs 62.8%.** At a 5-candidate beam (~10 Lean calls), cosine recovers 98% of the oracle ceiling. This means the scope is excellent and cosine ranking is strong enough that a small beam solves the rw0 local selection problem almost completely.
2. **Learned top-5 is well behind cosine top-5: 38.5% vs 61.5%.** At matched Lean-call budget, the learned decoder underperforms cosine by 23pp. The gold R@5 gap (87% cosine vs 44% learned) shows this is a ranking quality issue, not a scope issue.
3. **RW-001 was not trained for this setting.** The decoder was trained on 5K unscoped proxy vocab (EXP-RW-001). It has never seen the real ~11-symbol scoped vocabulary. MRR 0.235 vs cosine 0.634 reflects this distribution shift.
4. **Cosine top-5 is the practical deployment baseline for rw0.** At 10 Lean calls per goal, it achieves 61.5% — nearly oracle-level. A learned decoder only adds value if it beats this, which requires retraining on scoped data.
5. **The learned decoder's value proposition has shifted.** On rw0 with scope ~11, cosine top-5 is so strong that the learned decoder needs to either: (a) improve top-1 calibration to reduce Lean calls, or (b) handle harder grammar tiers (rw1-rw3) where cosine can't select direction/arguments.

**Interpretation for the rw lane architecture:**
- **rw0 is effectively solved by cosine top-5 + beam verify.** 61.5% LeanValid at 10 Lean calls is operationally useful.
- The learned decoder's value is not on rw0 selection — it's on **direction prediction** (rw1: backward rewrites), **argument construction** (rw2: applied rewrites), and **multi-atom composition** (rw3: `rw [a, b, c]`).
- For rw0 deployment, use cosine top-5 as the runtime policy. Reserve the learned decoder for the harder tiers.

**Frozen matched-budget comparison (n=78, qualified names, 10 Lean calls):**

| | Cosine top-5 | Learned top-5 |
|---|---|---|
| LeanValid | **61.5%** | 38.5% |
| Gold R@5 | **87%** | 44% |
| MRR | **0.634** | 0.235 |

### EXP-RW-015: Cosine rw Lane in Theorem Search (50 Mathlib)

**Date:** 2026-03-17
**Script:** `scripts/run_benchmark.py --cosine-rw` (same command as EXP-3.2 full mode + `--cosine-rw`)
**DB:** proof_network.db
**Lean:** Pantograph 0.3.13 + Lean 4.27.0 + Mathlib v4.27.0
**Checkpoint:** NAV-002_step5000.pt
**Theorems:** Same 50 from EXP-3.2

| Mode | Proved | Rate | Time (s) | cosine_rw provenance |
|------|--------|------|----------|---------------------|
| full (EXP-3.2 baseline) | 12/50 | 24% | 3835 | — |
| **full + cosine_rw** | **13/50** | **26%** | **3515** | **17 goals** |

**Gained theorem:** `WeierstrassCurve.Jacobian.addU_of_Z_eq_zero_right`
- Provenance: `automation → cosine_rw → automation`
- Cosine rw closed a subgoal that hammer could not, enabling the remaining goals to be closed by hammer

**Lane breakdown (full + cosine_rw):**

| Lane | Count |
|------|-------|
| automation | 4 |
| solver_bootstrap | 7 |
| **cosine_rw** | **1** |
| structural_core | 1 |
| failed | 30 |
| skipped | 7 |

**Findings:**
1. **+1 theorem, zero lost.** Cosine rw is purely additive on this benchmark. It never interferes with existing lanes.
2. **17 cosine_rw provenance entries** — the lane contributed goal progress on multiple theorems, not just the one it closed. It advances subgoals that other lanes then finish.
3. **Slightly faster** (3515s vs 3835s) — cosine rw closes some goals early, saving budget for other attempts.
4. **The 1-theorem lift is small but real.** On a 50-theorem benchmark where the baseline is 12, adding 1 theorem is an 8% relative improvement. The absolute number is limited by the 7 skipped theorems (goal creation failures) and the 30 that no lane can close.
5. **The cosine rw lane works as designed**: scope → encode → cosine rank → top-5 beam → Lean verify. It runs one-shot per goal, interleaves with other lanes, and uses ~10 Lean calls per goal.

**Comparison to EXP-3.2 modes:**

| Mode | Proved | Learned lane |
|------|--------|-------------|
| learned_only | 0/50 | 0 |
| learned_structural | 11/50 | 0 |
| full | 12/50 | 0 |
| **full + cosine_rw** | **13/50** | **1 (cosine_rw)** |

**Interpretation:** The cosine rw lane is the first learned-adjacent local lane to produce nonzero theorem lift on Mathlib. The learned navigator (NAV-002) produced 0 lane contributions across all modes. Cosine rw, using simple embedding similarity over scoped premises, produces both goal progress (17 events) and theorem closure (1 additional).

### EXP-RW-016: Beam Width Ablation (Cosine rw Lane)

**Date:** 2026-03-17
**Script:** 3 parallel `run_benchmark.py --cosine-rw --cosine-rw-beam N` runs (N=1,3,5)
**Data:** Same 50 theorems as EXP-3.2

| Beam | Proved | vs Baseline | cosine_rw progress | Theorems touched | Lean calls/thm | Time/thm |
|------|--------|-------------|-------------------|-----------------|----------------|----------|
| 0 (EXP-3.2) | 12/50 | — | — | — | 48 | 77s |
| **1** | **13/50** | **+1** | **8** | **5** | **52** | **75s** |
| 3 | 13/50 | +1 | 16 | 9 | 55 | 78s |
| 5 | 13/50 | +1 | 17 | 9 | 56 | 78s |

**All beam widths gain the same theorem:** `WeierstrassCurve.Jacobian.addU_of_Z_eq_zero_right`. Zero regressions at any width.

**Findings:**
1. **beam=1 is the optimal operating point.** Same +1 theorem gain, fewest Lean calls (52 vs 56/thm), fastest (75s vs 78s). Wider beams add progress events (8→16→17) and theorem touches (5→9→9) but don't convert additional proofs.
2. **The marginal value of wider beams is zero on this benchmark.** beam=3 and beam=5 are indistinguishable on theorem closes. The extra Lean calls probe more premises but none of them close new goals.
3. **beam=1 is ~2 Lean calls per rw attempt** (forward + backward). beam=5 is ~10. At 4 Lean calls/thm overhead (52 vs 48 baseline), beam=1 is extremely cheap.
4. **Progress events scale with beam width** (8→16→17) but asymptote quickly. beam=3 captures most of the beam=5 reach.

**Decision:** Set default `cosine_rw_beam=1` for production runs. The single-best cosine premise with forward/backward is sufficient for the current benchmark. Wider beams are available for harder theorem sets.

### EXP-RW-017: Multi-Family Cosine Lanes (rw + exact + apply + simp)

**Date:** 2026-03-17
**Script:** `run_benchmark.py --cosine-rw` with generalized `_try_cosine_family` for rw/exact/apply/simp
**Data:** Same 50 theorems

| Config | Proved | Gained | Attempts/thm | Time/thm |
|--------|--------|--------|-------------|----------|
| Baseline | 12/50 | — | 48 | 77s |
| cosine_rw only (beam=1) | 13/50 | +1 | 52 | 75s |
| **All 4 cosine lanes** | **13/50** | **+1** | **99** | **144s** |

**Per-family cosine progress events:**

| Lane | Progress events | Theorems touched |
|------|----------------|-----------------|
| cosine_rw | 18 | 10 |
| cosine_apply | 20 | 3 |
| cosine_simp | 15 | 6 |
| cosine_exact | 0 | 0 |

**Findings:**
1. **Same +1 theorem.** Adding exact/apply/simp does not convert additional proofs on this 50-theorem slice. The gained theorem is still only Weierstrass.
2. **apply and simp produce real progress events** (20 and 15 respectively), touching 3 and 6 theorems. These advance subgoals but don't close theorems on their own.
3. **exact produces 0 events.** Cosine ranking over accessible premises almost never finds an exact type match. The `exact` family likely needs a different scoping strategy (type-matching, not name similarity).
4. **2x cost overhead.** 99 vs 52 attempts/thm, 144s vs 75s. The extra lanes probe many premises that don't help.
5. **The compounding effect is subadditive.** More lanes = more progress events but not more proves. The bottleneck moves immediately past local cosine selection to deeper proof structure.

**Decision:** For production, use cosine_rw only (beam=1). The other families add cost without theorem lift on this benchmark. They remain available for larger/harder theorem sets where the additional progress events might compound into closes.

### EXP-RW-018: Step>0 Replay Audit (Stratified Sample)

**Date:** 2026-03-17
**Script:** Inline Pantograph probe with detailed prefix failure taxonomy
**Data:** 50 step>0 rw0 examples (30 pfx_len=1, 20 pfx_len=2-5, seed=42)

| Metric | Value |
|--------|-------|
| Base state obtained | 47/50 (94%) |
| **Replay success** | **5/47 (11%)** |
| goal_creation_fail | 3 |
| prefix_replay_fail | 36 |
| goal_match_fail | 6 |

**Replay success by prefix length:**

| pfx_len | Success | Total |
|---------|---------|-------|
| 1 | 4 | 30 |
| 2 | 0 | 4 |
| 3 | 0 | 4 |
| 4 | 0 | 9 |
| 5 | 1 | 3 |

**Prefix failure taxonomy (36 failures):**

| Category | Count | % |
|----------|-------|---|
| **unknown_identifier** | **28** | **78%** |
| other_tactic_fail | 5 | 14% |
| rw_inapplicable | 1 | 3% |
| crash_or_other | 1 | 3% |
| elaboration | 1 | 3% |

**Failing prefix index:** 32/36 at index 0 (89%).

**Findings:**
1. **First nonzero step>0 replay: 5/47 (11%).** Declaration-faithful Tier B enables replay on some theorems. This is up from 0% in EXP-RW-007.
2. **unknown_identifier is 78% of failures.** Prefix tactics reference short names (`hx`, `hf₀`, `one_mul`, `h`) that aren't resolved in the Tier B base state. These are either: hypothesis names (from binders not yet introduced), short lemma names (needing `open`), or local definitions.
3. **89% of failures are at prefix[0].** The first tactic in the prefix chain is the primary blocker. If prefix[0] succeeds, the rest usually follows.
4. **goal_match_fail = 6.** After successful replay, the expected goal isn't found in the resulting state. May be target-text differences from namespace qualification.
5. **pfx_len=1 is the easiest tier** (4/30 = 13% success). Longer prefixes compound failure probability.

**Interpretation:** The step>0 replay bottleneck is the same as the step-0 identifier_scope problem — short names in a context that doesn't have the right `open` directives. The 28 unknown_identifier failures split into:
- Hypothesis names (`h`, `hx`, `hf₀`) — these exist in the proof but not in the Tier B base state's goal. The goal from `load_sorry` starts at the theorem level, so hypotheses from the original proof's binders aren't available as prefix tactic arguments.
- Short lemma names (`one_mul`, `objs`, `map`) — same `open` directive gap as step-0.

**Next action:** The hypothesis-name failures are fundamental to the Tier A/B base state approach — hypotheses only exist after the proof's own intro/binder tactics run. These would resolve if the prefix included the necessary intros. The lemma-name failures would resolve with broader `open` directives in the Tier B wrapper.

### EXP-RW-019: Unknown Identifier Classification (Step>0)

**Date:** 2026-03-17
**Script:** Inline probe classifying each unknown_identifier by checking goal locals vs expected locals
**Data:** 31 unknown_identifier failures from the 50-example step>0 sample

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **global_name_missing** | **16** | **52%** | Short lemma/def names needing `open` or qualification (`objs`, `rnDeriv_tilted_right`, `coweightHom`) |
| **local_hyp_name_mismatch** | **8** | **26%** | Tier B state has Lean-generated names (`A✝`, `inst✝`) while prefix expects source names (`f`, `t`, `g`) |
| global_qualified_missing | 3 | 10% | Qualified names not resolving (`h.openSegment_subset`, `IntFractPair.stream`, `IsZero.of_iso`) |
| local_hyp_missing | 3 | 10% | Names that should be local defs/abbrevs (`log`, `nth`, `sub`) |
| name_exists_but_fails | 1 | 3% | Name in goal locals but tactic still fails (`𝕜` — Unicode/notation issue) |

**Findings:**
1. **Context engineering > state matching: 62% vs 26%.** The majority of failures (global_name_missing + global_qualified_missing = 19/31) would be fixed by better `open` directives or name qualification. Only 8/31 are local hypothesis name mismatches.
2. **local_hyp_name_mismatch is the state fidelity gap.** The Tier B `load_sorry` state uses Lean's auto-generated binder names (`A✝`, `R✝`, `inst✝`) rather than the source proof's names (`f`, `t`, `g`). The prefix tactics from LeanDojo use the source names.
3. **The 3 local_hyp_missing cases** (`log`, `nth`, `sub`) are likely section-local abbreviations or `noncomputable` definitions that aren't captured by `extract_active_context`. These are a subset of the context engineering problem.

**Interpretation:** The next week should focus on **context engineering** (broader `open` directives, name qualification in prefix tactics), not state matching. That addresses 62% of the gap. The 26% local name mismatch is a harder problem (requires either name-aware replay or a different base state construction) and should wait.

### EXP-RW-020: Prefix Qualification Effect on Step>0 Replay

**Date:** 2026-03-17
**Script:** Replay benchmark with `qualify_tactic()` resolving short global names to qualified forms via accessible-premise suffix index
**Data:** Same n=50 step>0 sample as EXP-RW-018

| Metric | Without qual (EXP-RW-018) | **With qual** | Delta |
|--------|--------------------------|---------------|-------|
| Replay success | 5/47 (11%) | **7/47 (15%)** | **+2** |
| prefix_replay_fail | 36 | 32 | -4 |
| goal_match_fail | 6 | 8 | +2 |
| fail at index 0 | 32/36 (89%) | 25/32 (78%) | -7pp |

**Findings:**
1. **Qualification helps modestly: 11% → 15%.** 4 prefix failures resolved, but 2 new goal_match_fail cases exposed (replay completes with wrong subgoal).
2. **Not yet at the 25% gate.** The remaining 32 prefix failures are likely dominated by local_hyp_name_mismatch (state fidelity), not global name resolution.
3. **Index-0 concentration dropped from 89% → 78%.** Some prefix[0] failures now pass, revealing failures at deeper indices.
4. **The qualification infrastructure works** — it correctly resolves unambiguous short names. But it addresses only the 52% global_name bucket from EXP-RW-019, which is ~16 of the original 36 failures. The remaining ~20 are local hyp / state fidelity issues.

**Next action:** The state fidelity problem (local_hyp_name_mismatch = 26% of failures) is now the primary blocker. The Tier B `load_sorry` state uses Lean-generated binder names (`A✝`, `inst✝`) instead of source proof names (`f`, `t`, `g`). Addressing this requires either name-aware replay or a different base state construction that preserves original binder names.

### EXP-RW-021: Local-Name Aliasing + Alpha Matching

**Date:** 2026-03-17
**Script:** Replay with `build_local_alias_map` (✝-stripping) + `rewrite_tactic_locals` + alpha-equivalent goal matching
**Data:** Same n=50 step>0 sample

| Metric | No qual (018) | Qual only (020) | **Qual + alias + alpha (021)** |
|--------|--------------|-----------------|-------------------------------|
| Replay success | 5/47 (11%) | 7/47 (15%) | **7/47 (15%)** |
| prefix_replay_fail | 36 | 32 | 32 |
| goal_match_fail | 6 | 8 | 8 |

**Finding: Local-name aliasing adds zero lift over qualification alone.** The ✝-renaming alias map builds correctly (e.g., `𝕜` → `𝕜✝`), but the failing prefix tactics don't reference ✝-renamed binders. They reference **hypothesis names that haven't been introduced yet** — names like `h`, `hx`, `hy` that would be created by `intro` tactics that aren't in the recorded prefix.

**Root cause diagnosis update:** The EXP-RW-019 classification of "local_hyp_name_mismatch" was partially wrong. The 8 cases classified as mismatch are actually a mix of:
- **Not-yet-introduced names** — the LeanDojo trace's prefix_tactics start at a point where some intros have already happened implicitly. The recorded prefix doesn't include those intros.
- **True ✝ mismatch** — binder variables like `f` vs `f✝`. These DO get aliased correctly, but they're a minority of the failing cases.

**Implication:** The step>0 replay problem is deeper than name aliasing. The LeanDojo prefix_tactics may not include all the tactics needed to reach the expected intermediate state. Some intros or elaboration steps happen implicitly in Lean and aren't recorded as discrete tactic steps. This is a data fidelity issue, not a harness issue.

### EXP-RW-022: Named Intro Prelude for Step>0

**Date:** 2026-03-17
**Script:** Replay with structural prelude: auto-`intro name` for each missing hypothesis from expected state before prefix replay. Skips instance/typeclass names and ✝-variants.

| Metric | No prelude (018) | Qual only (020) | **Intro prelude (022)** |
|--------|-----------------|-----------------|------------------------|
| Replay success | 5/47 (11%) | 7/47 (15%) | **7/47 (15%)** |
| pfx_len=1 success | 4/30 | 6/30 | **7/30** |
| prefix_replay_fail | 36 | 32 | 32 |
| goal_match_fail | 6 | 8 | 8 |
| fail at index 0 | 32 | 25 | 25 |

**Progression from no intervention to full pipeline:**

| Feature | Replay success |
|---------|---------------|
| Baseline (no qual, no alias, no prelude) | 5/47 (11%) |
| + global name qualification | 7/47 (15%) |
| + local-name aliasing | 7/47 (15%) |
| + alpha-equivalent goal matching | 7/47 (15%) |
| + named intro prelude | 7/47 (15%) |

**Finding:** The intro prelude recovered specific cases (StrictConvex verified working) but the overall rate plateaus at 15%. The remaining 25 index-0 failures are neither name mismatches nor missing introductions — they are structural failures where the prefix tactic references concepts not available in the current state.

**Assessment against gate:** Target was ≥25% replay success. Current best is 15%. The remaining gap is a replay-alignment problem (flat replay without per-step anchoring), not a base-state problem.

### EXP-RW-023: State-Guided Replay with Per-Step Alignment

**Date:** 2026-03-17
**Script:** Replay with `prefix_goal_states` for per-step intro prelude, alias map rebuild, and global qualification
**Data:** n=50 step>0 (same stratification, rebuilt data with `prefix_goal_states`)

| Metric | Flat (018) | Lexical (022) | **State-guided (023)** |
|--------|-----------|---------------|------------------------|
| Replay success | 5/47 (11%) | 7/47 (15%) | **9/47 (19%)** |
| prefix_replay_fail | 36 | 32 | **29** |
| goal_match_fail | 6 | 8 | 9 |
| fail at index 0 | 32 | 25 | **22** |
| pfx_len=1 | 4/30 | 7/30 | **8/30** |
| pfx_len=2 | 0 | 0 | **1** (first) |

**Findings:**
1. **State-guided replay moves the needle: 15% → 19%.** Per-step intro prelude and alias map rebuild each contribute. First pfx_len=2 success.
2. **Index-0 failures dropped from 32 → 22.** Per-step context alignment resolves 10 previously-failing prefix[0] cases.
3. **Still below 25% gate.** The remaining 22 index-0 failures are structural — tactics that reference concepts the replay state can't provide (file-local defs, scoped notation, complex elaboration).
4. **goal_match_fail rising (6 → 9).** More replays complete but land on wrong subgoal. Per-step subgoal selection would help but Pantograph doesn't support goal reordering.

**Progression (0% → 19%):**

| Improvement | Contribution |
|-------------|-------------|
| Declaration-faithful Tier B | 0% → 11% (+11pp) |
| Global name qualification | 11% → 15% (+4pp) |
| State-guided per-step replay | 15% → 19% (+4pp) |

### EXP-RW-024: Namespace-Fuzzy Goal Matching — Clears 25% Gate

**Date:** 2026-03-17
**Script:** State-guided replay + `_normalize_namespaces` for Tier 4 fuzzy goal matching
**Data:** Same n=50 step>0

| Metric | Flat (018) | Lexical (022) | State-guided (023) | **NS-fuzzy (024)** |
|--------|-----------|---------------|--------------------|--------------------|
| **Replay success** | 5/47 (11%) | 7/47 (15%) | 9/47 (19%) | **12/47 (26%)** |
| prefix_replay_fail | 36 | 32 | 29 | 29 |
| goal_match_fail | 6 | 8 | 9 | **6** |
| pfx_len=1 | 4/30 | 7/30 | 8/30 | **10/30** |
| pfx_len=2 | 0 | 0 | 1 | 1 |
| pfx_len=3 | 0 | 0 | 0 | **1** |

**Gate cleared: 26% ≥ 25%.** Step>0 replay is now usable for rw evaluation.

**Full progression (0% → 26%):**

| Improvement | Success rate |
|-------------|------------|
| No replay infrastructure | 0% |
| + Declaration-faithful Tier B | 11% |
| + Global name qualification | 15% |
| + State-guided per-step replay | 19% |
| + Namespace-fuzzy goal matching | **26%** |

**Remaining failures (35/47):**
- prefix_replay_fail: 29 (22 at index 0 — structural failures)
- goal_match_fail: 6 (structural divergence, not namespace)
- goal_creation_fail: 3

### EXP-RW-025: ContextIR Census + Benchmark Audit

**Date:** 2026-03-17
**Scripts:**
- `python -m scripts.context_ir_census`
- `python -m scripts.context_ir_benchmark_audit --dataset data/canonical/rw0_eval.jsonl`
- `python -m scripts.context_ir_benchmark_audit --dataset data/canonical/canonical_eval_replayable.jsonl --limit 500`

**Protocol:** Add a conservative source-context parser (`src/lean_context_ir.py`) and audit both the whole local Mathlib checkout and benchmark theorem sites to measure which theorem-site context constructs the current wrapper path must compile. This is a static audit: no Pantograph, no Lean execution, just theorem-site source analysis.

#### Part A: Whole-Mathlib Context Census

| Feature | Count |
|--------|------:|
| variable | 39,445 |
| namespace | 12,806 |
| section | 15,200 |
| open | 11,328 |
| open_scoped | 2,370 |
| universe | 2,595 |
| local_attribute | 915 |
| local_notation | 496 |
| scoped_notation | 402 |
| include | 1,064 |
| omit | 422 |
| inline_only (`... in`) | 6,742 |

**Finding:** The missing context features are not edge cases. Scoped source effects are common across Mathlib, especially `open scoped`, `local notation`, local attributes, and inline next-declaration forms like `open Classical in`.

#### Part B: Benchmark Audit — `rw0_eval.jsonl`

**Data:** 1,105 examples requested, 783 processed by the static theorem-line finder

| Active feature at theorem site | Examples |
|-------------------------------|---------:|
| variable | 720 |
| open | 693 |
| open_scoped | 262 |
| universe | 241 |
| include | 47 |
| local_notation | 42 |
| local_attribute | 42 |

**Top unsupported forms:**
- `open Classical in` (53)
- `variable (M) in` (33)
- `include Q in` (30)
- `open scoped Classical in` (29)

#### Part C: Benchmark Audit — `canonical_eval_replayable.jsonl` (500-sample)

**Data:** 500 examples requested, 342 processed by the static theorem-line finder

| Active feature at theorem site | Examples |
|-------------------------------|---------:|
| variable | 317 |
| open | 260 |
| open_scoped | 122 |
| universe | 118 |
| include | 29 |
| local_notation | 25 |
| local_attribute | 17 |

**Top unsupported forms:**
- `omit [SFinite ν] in` (60)
- `include 𝕜 in` (46)
- `open scoped Classical in` (22)
- multiline `local notation ... =>` fragments (11 each for the top two)

**Key findings:**
1. **The next bottleneck is a compiler gap, not just a replay heuristic gap.** The theorem wrapper path now reaches usable start states and step>0 replay clears 26%, but Mathlib theorem sites rely on context constructs the current wrapper path still drops or only partially models.
2. **Inline next-declaration forms are the most common unsupported pattern.** `... in` is frequent enough (`6,742` whole-corpus occurrences) that a serious Tier B wrapper/compiler must model it explicitly rather than treating it as noise.
3. **This directly affects future families.** `simp` depends on `[local simp]`, notation, and scoped openings; `apply` depends on cleaner theorem-site local/typeclass context. A source-context compiler is a prerequisite for making those lanes theorem-level productive.

**Decision:** Promote **ContextIR** to the next explicit execution phase.
- `src/lean_context_ir.py` becomes the canonical source-context parser
- `extract_active_context()` is now a compatibility wrapper over that parser
- future Tier B work should compile from `ContextIR`, not from ad hoc header extraction
- validation scripts (`context_ir_census.py`, `context_ir_benchmark_audit.py`) are now part of the formal experiment toolchain

### EXP-RW-025: ContextIR + Inline Rendering

**Date:** 2026-03-17
**Script:** Step>0 replay with ContextIR-backed wrappers including `inline_lines` for `... in` forms

| Metric | NS-fuzzy (024) | **ContextIR + inline (025)** |
|--------|---------------|------------------------------|
| Replay success | 12/47 (26%) | **12/47 (26%)** |
| pfx_len=3 | 1 | **2** |
| goal_match_fail | 6 | 6 |
| fail at index 0 | 22 | 23 |

**Finding:** Inline rendering holds at 26%, shifts some individual cases. The remaining 23 index-0 failures are structural — not addressable by more context forms on this sample. 26% is the current stable operating point for step>0 replay.

### EXP-RW-026: Step>0 Semantic rw0 on Replayed Examples

**Date:** 2026-03-17
**Script:** Oracle + cosine rw0 on step>0 examples that successfully replay + have gold in accessible scope
**Data:** n=9 from the 50-example step>0 sample (12 replayed, 9 with gold in scope)

| Condition | Step-0 (n=78, EXP-RW-013) | **Step>0 (n=9)** |
|-----------|--------------------------|------------------|
| Oracle qualified | 62.8% | 22.2% (2/9) |
| Cosine top-1 | 33.3% | **55.6%** (5/9) |
| Cosine top-5 | 61.5% | **66.7%** (6/9) |
| Gold in scope | 100% | 100% |
| Mean scope | 11.3 | 12.1 |

**Findings:**
1. **Step>0 looks like step-0 in miniature.** Cosine top-5 (66.7%) is comparable to step-0 (61.5%). Scope quality is identical. The replayed intermediate states are semantically usable.
2. **Oracle is lower but n is too small.** 2/9 = 22.2% vs 62.8% on step-0. On 9 examples, this is 2 cases — too noisy to conclude oracle is structurally worse on step>0.
3. **The step>0 substrate works for rw evaluation.** Cosine top-1 at 55.6% (5/9) is even better than step-0's 33.3% — likely because the replayed intermediate states are more specific (post-structural-setup), making cosine ranking more precise.
4. **The bottleneck is replay coverage (26%), not replay quality.** Once an example replays, it evaluates well. Expanding replay coverage is the path to a larger step>0 denominator.

**Implication:** Step>0 is ready for rw1 (backward rewrite) component benchmarks. The replay substrate produces usable intermediate states, and cosine top-5 is a strong baseline at those states.

### EXP-RW-027: Step>0 Semantic rw0 — Scaled (Parallelized)

**Date:** 2026-03-17
**Script:** 3 parallel batch workers (50 examples each), parallelized via separate Pantograph instances
**Data:** n=150 step>0 examples (pfx_len≤5, seed=42) → 21 replayed+scoped

| Condition | k/n | Rate | 95% Wilson CI |
|-----------|-----|------|---------------|
| Oracle qualified | 9/21 | 42.9% | [24%, 63%] |
| Cosine top-1 | 10/21 | 47.6% | [28%, 68%] |
| **Cosine top-5** | **15/21** | **71.4%** | **[50%, 86%]** |

| Metric | Value |
|--------|-------|
| Mean scope | 12.4 |
| Gold R@1 | 5/21 (24%) |
| Gold R@5 | 12/21 (57%) |
| Cosine-5 OK, oracle FAIL | 8 cases |

**Comparison to step-0:**

| Metric | Step-0 (n=78) | **Step>0 (n=21)** |
|--------|--------------|-------------------|
| Oracle qualified | 62.8% | 42.9% |
| Cosine top-1 | 33.3% | 47.6% |
| Cosine top-5 | 61.5% | **71.4%** |

**Findings:**
1. **Cosine top-5 at 71.4% on step>0** — comparable to step-0's 61.5%. The CIs overlap heavily ([50%,86%] vs step-0's point estimate). Step>0 is not a harder regime for cosine selection.
2. **Oracle lower at 42.9% vs 62.8%.** 8 cases where cosine-5 succeeds but oracle fails, suggesting annotated premises are sometimes stale/noisy on intermediate states, or alternative valid rewrites exist.
3. **n=21 is still small.** CIs are wide. The 150-example sample yielded only 21 through the full pipeline (replay + gold in scope). Expanding to the full 467 pfx≤5 examples would give ~50-60.
4. **Parallelization worked.** 3 batches × 50 examples completed in ~20 min instead of ~60 min. Fixed Pantograph cleanup (`_close()` instead of `del`) prevented process accumulation.

**Assessment:** Step>0 rw0 behaves like step-0. The substrate is usable for semantic evaluation. Ready to move to rw1.

### EXP-RW-028: Oracle/Cosine Disagreement Audit (Step>0)

**Date:** 2026-03-17
**Data:** 8 cases from EXP-RW-027 where cosine top-5 succeeds but oracle fails

| Category | Count | Meaning |
|----------|-------|---------|
| rw_pattern_not_found | 4 | **Replay state drift** — annotated premise is valid but rewrite pattern doesn't match replayed goal |
| not_an_equality | 2 | **Stale label** — annotated premise isn't an equality/iff (wrong family annotation) |
| equation_rewrite_fail | 2 | **Stale label** — equation compiler can't use premise at this state |

**Split: 50% replay drift, 50% stale labels.**

**Implications:**
1. **Step>0 oracle is not a clean ceiling.** Half the "oracle failures" are noisy labels, not semantic failures. The real oracle ceiling on step>0 is higher than 42.9%.
2. **Cosine top-k success is the more reliable metric.** It measures "can any premise in scope produce a valid rw" — independent of label quality.
3. **This strengthens the architecture.** LeanValid top-k via cosine + Lean verify is robust to label noise. Strict supervision against annotated premise is risky on step>0.
4. **No identifier_scope or qualification issues** in the disagreement set — the name engineering is working.

**Decision:** Use LeanValid top-k as the primary step>0 metric, not oracle-match. Move to rw1.

### EXP-RW-029: rw1 Step-0 Semantic Benchmark (Backward Rewrites)

**Date:** 2026-03-17
**Script:** 4 parallel batch workers on 110 rw1 step-0 examples
**Data:** 317 rw1 examples total, 110 step-0, 86 in overlap set (started + has premise + gold in scope)

| Condition | k/n | Rate | 95% Wilson CI |
|-----------|-----|------|---------------|
| Oracle (both dirs) | 50/86 | 58.1% | [48%, 68%] |
| Cosine top-1 (both dirs) | 24/86 | 27.9% | [20%, 38%] |
| **Cosine top-5 (both dirs)** | **52/86** | **60.5%** | **[50%, 70%]** |

**Direction breakdown (cosine top-1, n=24 successes):**

| Direction | Count |
|-----------|-------|
| Forward only | 9 |
| Backward only | 16 |
| Both work | 1 |
| **Direction matters** | **23/24 (96%)** |

**Scope stats:** Gold R@1=27%, R@5=76%, mean scope=10.6

**Comparison to rw0:**

| Metric | rw0 (n=78) | rw1 (n=86) |
|--------|-----------|-----------|
| Oracle | 62.8% | 58.1% |
| Cosine top-1 | 33.3% | 27.9% |
| Cosine top-5 | 61.5% | **60.5%** |

**Findings:**
1. **Cosine top-5 at 60.5% on rw1 — same as rw0.** Backward rewrites are not harder for premise selection. The scope + cosine pipeline works identically.
2. **Direction matters in 96% of cosine successes.** Only 1/24 cases where both forward and backward work. The raw rw1 slice therefore needs a direction policy, not just a premise scorer.
3. **This initially suggests direction prediction, but EXP-RW-030 narrows that conclusion.** On the current rw0/rw1 split, the family/tier identity already carries most of the signal.
4. **Oracle is comparable** (58.1% vs 62.8%) — rw1 isn't intrinsically harder, just requires backward application.

**Next milestone:** Extend the deployed rewrite lane from `rw0` to `rw0 + rw1`, then move learned work to `rw2/rw3`.

### EXP-RW-030: rw1 Direction Ablation

**Date:** 2026-03-17
**Script:** 4 parallel batches, oracle premise + symbolic direction vs both-dirs, cosine top-1 + symbolic vs both-dirs

| Config | Oracle LeanValid | Cosine-1 LeanValid | Lean calls/ex |
|--------|-----------------|-------------------|---------------|
| Both dirs | 58.1% (50/86) | 27.9% (24/86) | 2.0 |
| Symbolic dir | 45.3% (39/86) | 19.8% (17/86) | 1.0 |
| Always-backward | 54.7% (47/86) | — | 1.0 |

**Direction accuracy on oracle successes:**
- Symbolic heuristic: 39/50 (78%)
- Always-backward: 47/50 (**94%**) — trivially better because rw1 IS the backward tier
- Oracle correct direction: 94% backward, 6% forward

**Findings:**
1. **Symbolic heuristic loses 22% of oracle rewrites** for a 50% Lean call savings. Not a good trade on rw1 alone.
2. **Always-backward is the trivial baseline for rw1** — 94% correct. But it only works because rw1 is defined as backward rewrites.
3. **The real direction prediction value is across rw0+rw1 combined.** On rw0, always-forward is ~right. On rw1, always-backward is ~right. A direction predictor's value is predicting WHICH tier an example belongs to, not the direction within a tier.
4. **The clean deployment is: cosine top-1 + per-tier direction default + beam-both on ambiguous.** rw0 examples → try forward first. rw1 examples → try backward first. Unknown → beam both. This is essentially a family gate, not a direction head.

**Implication for learned direction:** A separate learned direction head is not justified as the next step on the current rw0/rw1 split — the family gate already provides the direction signal. Direction may reappear as a useful subproblem when merging tiers, handling mixed-direction multi-rewrite (rw3), or learning argumented rewrites (rw2). For now, reserve learning for argument construction and composition, not direction.

### EXP-RW-031: rw0+rw1 Directed Cosine Lane in Theorem Search

**Date:** 2026-03-18
**Script:** `run_benchmark.py --cosine-rw` with `infer_direction` per premise

| Config | Proved | cosine_rw events | Attempts/thm |
|--------|--------|-----------------|-------------|
| Baseline | 12/50 | — | 48 |
| cosine_rw (both dirs) | 13/50 | 17 | 52 |
| **cosine_rw (directed)** | **13/50** | **19** | **58** |

**Finding:** Direction heuristic increases progress events (19 vs 17) but no new theorem closes. The 50-theorem slice is saturated at +1 for cosine rw. Ready for rw2.

### EXP-RW-032: rw2 Step-0 Baseline (Argument Construction)

**Date:** 2026-03-17
**Script:** `scripts/run_rw2_benchmark.py --parallel 4 --output runs/rw2_step0_results.jsonl`
**Data:** `data/canonical/canonical_rw_eval.jsonl`, rw2_fwd_args + rw2_bwd_args, step_index==0
**Dataset size:** 67 examples (39 rw2_fwd_args, 28 rw2_bwd_args)
**Protocol:** Goal creation via `goal_via_file_context` (Tier B→A cascade). Real accessible-premise scope from `proof_network_v3.db`. Three conditions: oracle qualified, cosine top-1 + heuristic arg beam, cosine top-5 + heuristic arg beam. Direction mode: both (beam both fwd+bwd). Metric: tactic accepted by Lean (rw always leaves residual goals; "fully closed" is not the right metric for rw).

**GoalStart@rw2:** 54/67 = **80.6%** (13 goal_creation_fail)

| Condition | Accepted/started | Rate | Accepted/gold_in_scope |
|-----------|-----------------|------|------------------------|
| Oracle qualified | 20/54 | **37.0%** | 18/50 (36.0%) |
| Cosine top-1 + heuristic args | 18/54 | **33.3%** | 18/50 (36.0%) |
| Cosine top-5 + heuristic args | 33/54 | **61.1%** | 32/50 (64.0%) |

**Scope stats:**
- Gold in scope: 50/54 (92.6%)
- Mean scope size: 3.5 premises
- Mean arg-sequence beam: 4.8 sequences/example
- Mean heuristic calls (top-5): 23.1/example

**Oracle failure taxonomy (54 started):**

| Category | Count | Interpretation |
|----------|-------|----------------|
| `unknown_identifier` — dotted/premise name | 14 | Lowering gap: canonical name not in scope context |
| `unknown_identifier` — local arg | 11 | Arg is binder from source theorem, not in LeanDojo-traced goal |
| `oracle_accepted, remaining_goals` | 19 | rw fired, left side goals (expected) |
| `oracle_accepted, goals_closed` | 1 | Full close |
| `rewrite_pattern_not_found` | 9 | Lemma type doesn't unify with goal redex |

**Post-hoc ablations (from saved JSONL, no Lean re-runs):**

*Cosine-wins / oracle-fails audit (15 cases where cosine-5 succeeds, oracle fails):*

| Cause | Count |
|-------|-------|
| Same premise, cosine used bare tactic (oracle arg was source binder, heuristic dropped it) | 7 |
| Different premise (cosine found a simpler no-arg alternative) | 8 |
| Oracle failure due to `unknown_identifier` (name resolution, not arg difficulty) | 11 |

*Winning tactic arg structure for all 33 cosine-5 successes:*

| Arg pattern | Count |
|-------------|-------|
| Bare (no args) | **33** |
| Named locals or wildcards | 0 |

All 33 cosine-5 wins fired with bare rewrites, even though gold had 1–6 args. Cosine is succeeding by finding simpler arg-free alternatives, not by constructing args. The heuristic arg beam contributed zero wins.

*Tier-default direction estimate:* rw2_fwd_args winning tactics: 16 forward, 2 backward. rw2_bwd_args: 9 backward, 6 forward. Tier-default direction would reduce Lean calls by ~50% (1245 → ~620) with at most 2 losses from backward wins in fwd tier.

**Key findings (revised):**

1. **Cosine top-5 > oracle is explained by oracle identifier noise, not arg construction.** Of the 15 cosine-only wins, 11 had oracle fail due to `unknown_identifier` and 7 used the same premise with a bare tactic (source-binder arg simply dropped). Cosine is not solving argument construction — it is routing around oracle noise.

2. **The heuristic arg beam contributed zero wins.** All 33 cosine-5 successes used bare (no-arg) rewrites. The arg beam adds Lean call cost but no semantic gain on this step-0 sample.

3. **Two sources of oracle identifier failure are distinct:** (a) 14 cases are lowering/context gaps — premise or dotted name not in Tier B scope (same root cause as rw0/rw1 step>0). (b) 11 cases are source binders (`a`, `u`, `I`, `V`, `p`) that are quantified in the original theorem declaration but absent from the LeanDojo-traced goal state. These are structurally unfixable without the source theorem context — they require type inference or explicit `@`-application, not just name resolution.

4. **`remaining_goals` is not a failure** — 19/20 oracle acceptances leave side goals. rw accepted = correct primary metric.

5. **Arg profile (362 total rw2):** 1-arg=51%, 2-arg=22%, 3-arg=9%; 48% of arg instances are local hyps, 39% other names, 13% wildcards. Named args (`f :=`): 7%, inline `by`: 4%. Learnable core dominated by hypothesis names, but heuristic beam does not improve on bare-premise cosine at current scope sizes (mean 3.5 premises).

**Decision:** See EXP-RW-033 for the settled partition. Bare-premise cosine-5 is the canonical rw2 baseline. Heuristic arg beam and learned arg decoder are deferred — `args_necessary` = 4% does not justify either. Identifier resolution failures (the 35% unexecutable bucket) are tracked in the shared lowering backlog, not as a rw2-specific project.

---

### EXP-RW-033: rw2 Bare-Premise Ablation — args_redundant / args_necessary / unexecutable

**Date:** 2026-03-17
**Script:** `scripts/rw2_bare_ablation.py --parallel 4 --output runs/rw2_bare_ablation.jsonl`
**Data:** 54 started examples from EXP-RW-032 (same denominator)
**Protocol:** Second pass on started goals from EXP-RW-032. Three new conditions run against re-created goal states: oracle_bare (gold premise, gold direction, no args), cosine_top1_bare, cosine_top5_bare. Cross-tabulated against EXP-RW-032 heuristic results to produce the three-bucket partition.

| Condition | Accepted | Rate |
|-----------|----------|------|
| Oracle bare (gold premise, gold dir, no args) | 25/54 | **46.3%** |
| Cosine top-1 bare | 18/54 | **33.3%** |
| Cosine top-5 bare | 33/54 | **61.1%** |
| *(EXP-RW-032) Oracle qualified (with args)* | *20/54* | *37.0%* |
| *(EXP-RW-032) Cosine top-5 + heuristic args* | *33/54* | *61.1%* |

**rw2 partition (N=54 started):**

| Bucket | Count | % | Definition |
|--------|-------|---|------------|
| `args_redundant` | 33 | **61.1%** | Bare cosine-5 succeeds |
| `args_necessary` | 2 | **3.7%** | Bare fails, EXP-RW-032 heuristic/oracle succeeds |
| `unexecutable` | 19 | **35.2%** | All conditions fail |

**Cosine-5 cross-tab (heuristic vs bare):**

| | Bare succeeds | Bare fails |
|-|---|---|
| Heuristic succeeds | 33 | 0 |
| Heuristic fails | 0 | 21 |

Heuristic and bare are **identical** on this denominator. Zero cases where heuristic arg beam beats bare.

**Lean call budget:**

| Condition | Total calls | Per example |
|-----------|-------------|-------------|
| Heuristic+args (EXP-RW-032) | 1245 | 23.1 |
| Bare only (this run) | 194 | **3.6** |

Bare is 6.4× cheaper with identical output.

**Oracle bare (46.3%) > oracle qualified (37.0%)**. Dropping args from oracle *improves* acceptance. The source-binder args in the oracle tactic actively break name resolution in the Tier B goal context; bare oracle avoids this.

**Key findings:**

1. **`args_redundant` = 61%, `args_necessary` = 4%.** On step-0 rw2, argument construction is not a meaningful problem. The premise alone carries the rewrite in 61% of started cases; args add value in only 2/54 examples (3.7%) — too small to justify any learned arg decoder.

2. **Heuristic arg beam = bare on this sample.** The cross-tab is perfectly degenerate: every heuristic win is also a bare win, and every heuristic loss is also a bare loss. The arg beam contributes zero marginal successes.

3. **`unexecutable` = 35%** — these 19 examples fail under all conditions. They are not an argument problem; they are either identifier resolution failures (lowering gap, source binders) or genuine semantic mismatches. The 35% unexecutable ceiling is the real constraint.

4. **Bare cosine-5 is the correct rw2 executor.** Same semantic coverage as heuristic, 6.4× fewer Lean calls, no added complexity.

**Decision:** Deploy bare-premise cosine-5 as the rw2 lane (identical to rw0/rw1 architecture). Do not implement a learned arg decoder for rw2 until `args_necessary` grows substantially on a larger or harder sample. The 35% unexecutable ceiling is the shared lowering-gap issue, not an argument problem. Next: rw3 profiling.

---

### EXP-RW-034: rw3 First-Atom Benchmark (034a + 034b)

**Date:** 2026-03-17
**Script:** `scripts/run_rw3_benchmark.py --parallel 4 --output runs/rw3_first_atom_results.jsonl`
**Data:** `data/canonical/canonical_rw_eval.jsonl`, all-bare rw3 (every atom has no positional args), step_index==0
**Dataset size:** 487 examples (62% of 786 step-0 rw3)
**Protocol:** Goal creation via `goal_via_file_context` (Tier B→A cascade). Oracle = exact first RewriteAtom from canonical_action_ir with direction preserved. Cosine conditions try both directions. 034b: after each successful first-atom tactic, in-scope check for second gold atom against accessible premises (no extra Lean call).

**GoalStart@rw3_bare:** 451/487 = **92.6%** (36 goal_creation_fail)

**EXP-RW-034a: First-Atom Acceptance (N=451 started)**

| Condition | Accepted/started | Rate | Accepted/gold_in_scope |
|-----------|-----------------|------|------------------------|
| Oracle first atom (exact direction) | 319/451 | **70.7%** | 301/405 (74.3%) |
| Cosine top-1 (both dirs) | 229/451 | **50.8%** | 220/405 (54.3%) |
| Cosine top-5 (both dirs) | 323/451 | **71.6%** | 311/405 (76.8%) |

**Scope stats:**
- Gold first atom in scope: 405/451 (89.8%)
- Gold rank (when in scope): mean=0.8, median=0.0; rank-0: 220, rank≤2: 374
- Mean scope size: 3.3 premises
- Lean calls: cosine-1=1.6/ex, cosine-5=2.9/ex

**Oracle first-atom failure taxonomy (132 failures):**

| Category | Count | Interpretation |
|----------|-------|----------------|
| `rewrite_pattern_not_found` | 95 | LHS of lemma doesn't unify with any redex in goal |
| `identifier_scope` | 20 | Name not in Tier B context (lowering gap) |
| `other_tactic_fail` | 16 | Misc Lean errors |
| `type_mismatch` | 1 | Wrong type |

**EXP-RW-034b: Second-Atom Viability (composition metric)**

| Condition | Fired | Second gold atom in scope after | Rate |
|-----------|-------|--------------------------------|------|
| Oracle first-step | 319 | 273/319 | **85.6%** |
| Cosine-5 first-step | 323 | 275/323 | **85.1%** |

Composition breaks (oracle fires, second atom scope-lost): **46/319 (14.4%)**

**Atom count breakdown (oracle acceptance):**

| Atoms | N | Oracle accepted | Rate |
|-------|---|----------------|------|
| 2 | 205 | 150 | 73% |
| 3 | 125 | 88 | 70% |
| 4 | 73 | 47 | 64% |
| 5 | 28 | 20 | 71% |
| 6+ | 20 | 14 | 70% |

**Key findings:**

1. **Cosine top-5 (71.6%) matches oracle (70.7%) on the first atom.** The first-atom retrieval problem is essentially solved by bare-cosine-5. The gold premise is rank-0 in 220/451 cases; rank≤2 in 374/451. This is the same architecture as rw0/rw1/rw2 — no new component needed.

2. **Second-atom viability after first step: 85.6%.** When the first atom fires correctly, the second gold atom remains in scope 86% of the time. This is strong evidence that much of all-bare rw3 is reducible to repeated single-atom execution. The goal state after a successful first rewrite typically still exposes the next premise.

3. **Composition-break floor = 14%.** In 46/319 oracle-fires, the second gold atom is no longer in scope after step 1 — these cases are definitively broken for the gold path. This is a **lower bound** on composition difficulty, not an upper bound: when the second atom is in scope (85%), sequential execution can still fail later due to changed redex structure, wrong direction on a later atom, or divergence from the gold path at step k>1. Full sequential acceptance may be substantially lower than 85%.

4. **`rewrite_pattern_not_found` is the dominant oracle failure (95/132).** This is qualitatively different from rw0/rw1/rw2: the lemma is in scope but its LHS doesn't unify with the current goal's redex. This is a goal-state-specificity issue — the annotation is the right lemma for the full proof context, but after goal creation it may not apply to the presented goal. Identifier scope (lowering gap) is only 20 cases here.

5. **Atom count does not strongly predict first-atom failure.** Oracle acceptance is 64–73% across all atom counts 2–6, with no clear degradation. Composition failure (second atom lost) is the more relevant axis.

**Decision:** See EXP-RW-035. All-bare rw3 is operationally solved as repeated bare-cosine-5 execution. Not a separate learned tier.

---

### EXP-RW-035: rw3 Sequential Composition Benchmark

**Date:** 2026-03-17
**Script:** `scripts/run_rw3_sequential.py --parallel 4 --output runs/rw3_sequential_results.jsonl`
**Data:** `data/canonical/canonical_rw_eval.jsonl`, all-bare rw3, step_index==0
**Dataset size:** 487 examples (451 started, 36 goal_creation_fail)
**Protocol:** Three sequential-execution conditions on the same 451 started goals. Oracle seq: exact atoms in order with direction preserved. Cosine-5 seq: after each accepted step, refresh scope, re-rank cosine top-5, try both dirs. Oracle-1 + cosine-rest: oracle first atom, cosine-5 for remaining atoms. `FullSeqAccept` = all atoms fired without failure.

**GoalStart@rw3_bare:** 451/487 = **92.6%**

**FullSeqAccept (N=451 started):**

| Condition | Full seq accepted | Rate |
|-----------|------------------|------|
| Oracle step-by-step | 254/451 | **56.3%** |
| Cosine-5 sequential | 319/451 | **70.7%** |
| Oracle-1 + cosine-rest | 310/451 | **68.7%** |

**Accept@k (first k atoms all fired):**

| k | Oracle | Cosine-5 | Oracle-1+CRest |
|---|--------|----------|----------------|
| k=1 | 320/451 (71.0%) | 323/451 (71.6%) | 319/451 (70.7%) |
| k=2 | 278/451 (61.6%) | 321/451 (71.2%) | 313/451 (69.4%) |
| k=3 | 265/451 (58.8%) | 320/451 (71.0%) | 311/451 (69.0%) |
| k=4 | 257/451 (57.0%) | 319/451 (70.7%) | 310/451 (68.7%) |
| k=5+ | 254/451 (56.3%) | 319/451 (70.7%) | 310/451 (68.7%) |

**Mean divergence step (among failures):** Oracle=0.53, Cosine-5=0.05, Oracle-1+CRest=0.09

**Oracle failure taxonomy by step index:**

| Step | rewrite_pattern_not_found | identifier_scope | other |
|------|--------------------------|-----------------|-------|
| 0 | 78 | 38 | 15 |
| 1 | 19 | 18 | 5 |
| 2 | 7 | 4 | 2 |
| 3 | 3 | 3 | 2 |
| 4+ | 2 | 1 | 0 |

**FullSeqAccept by atom count:**

| Atoms | N | Oracle | Cosine-5 | Oracle-1+CRest |
|-------|---|--------|----------|----------------|
| 2 | 205 | 63% | **73%** | 71% |
| 3 | 125 | 57% | **73%** | 69% |
| 4 | 73 | 45% | **64%** | 64% |
| 5 | 28 | 54% | **71%** | 64% |
| 6 | 9 | 44% | **78%** | 100% |
| 7 | 5 | 0% | 40% | 40% |

**Lean call budget:** Oracle=2.3/ex, Cosine-5=5.8/ex, Oracle-1+CRest=3.9/ex

**Key findings:**

1. **Cosine-5 sequential (70.7%) substantially beats oracle step-by-step (56.3%).** This is the main result: the oracle path is not optimal. Oracle commits to a specific lemma+direction that often fails `rewrite_pattern_not_found` (78 step-0 failures) because the LHS doesn't unify with the current goal's redex. Cosine-5 finds an alternative first step that works and the remaining sequence follows. Oracle is a lower bound on task difficulty, not a ceiling.

2. **Oracle-1 + cosine-rest (68.7%) is close to but below cosine-5 (70.7%).** The oracle first step is slightly worse than cosine's best first step in aggregate. The cosine-5 beam finds better initial rewrites in ~2% of cases, confirming that even the first-atom selection benefits from cosine flexibility.

3. **Cosine Accept@k is nearly flat: 71.6% at k=1 dropping to 70.7% at k=5.** This is the key composition finding: once the first step fires, subsequent steps almost never fail. The composition problem for all-bare rw3 is almost entirely concentrated at step 0. After a successful first step, the sequence runs to completion in >99% of cases.

4. **Oracle divergence is dominated by step 0 (131/197 failures = 67%) but has a long tail.** Steps 1–4 account for the remaining 33% of oracle failures, distributed across `rewrite_pattern_not_found` and `identifier_scope`. This long tail is what separates oracle from cosine — cosine sidesteps these by finding a different valid path.

5. **FullSeqAccept degrades with atom count for oracle, not for cosine.** Oracle drops from 63% at 2-atoms to 44% at 4-atoms to 0% at 7-atoms. Cosine-5 is more stable: 73% at 2-atoms, 64% at 4-atoms, 40% at 7-atoms. The cosine flexibility is especially valuable at higher atom counts where oracle path-specificity becomes a liability.

6. **`rewrite_pattern_not_found` is the dominant oracle failure (78/131 step-0 failures = 60%).** This is qualitatively different from the identifier scope / lowering gap: the lemma is accessible and resolves, but its LHS doesn't unify with the current goal's redex in the Tier B goal state. This may reflect elaboration differences between the canonical data and the presented goal.

**Decision (frozen):** Sequential bare-cosine-5 is the deployed rw3_bare executor. Implemented as `cosine_rw_seq` lane in `src/proof_search.py` (capped loop: re-scope after each accepted step, `max_atoms=10`, `max_calls=50`). Oracle is diagnostic only; full-sequence Lean-valid is the primary metric. All-bare rw3 is NOT a separate learned tier — it is repeated bare-cosine execution. Learning is not justified here; the remaining gap is goal-state fidelity (elaboration mismatch, 60% of oracle failures) and identifier resolution, not composition control. Next: rw3-with-args profiling (EXP-RW-036).

---

### DATA: rw3-with-args Profile

**Date:** 2026-03-17
**Source:** `data/canonical/canonical_rw_eval.jsonl`

904 total rw3-with-args examples (at least one atom in rw[...] has positional args); 297 step-0 (33%).

**First-atom structure (step-0):**
- First atom bare: 161/297 (54%)
- First atom has args: 136/297 (46%)

**Args-atom count (step-0):**
- Exactly one atom with args: 209/297 (70%)
  - Position: last=44%, first=34%, middle=22%
  - First atom is bare in 137/209 (66%) of these — sequential bare executor can handle all preceding steps
- Multiple atoms with args: 88/297 (30%)

**Arg types across all args-atoms:**
- local_names: 49%
- complex (paren, dotted, chained): 43%
- wildcard: 4%
- named_arg (`f :=`): 3%
- by_tactic: 1%

**Atom count:** mean ≈ 3.3; 45% are 2-atom, 23% are 3-atom.

**Args-atom frequency by position:** Position 0: 46%, pos 1: 50%, pos 2: 37%, pos 3: 28% — args-atoms are roughly uniformly distributed across positions, slightly more common at pos 1 than pos 0.

**Implications for EXP-RW-036:**
- 70% of step-0 rw3-with-args have exactly one args-atom. Of those, 66% have a bare first atom — the bare sequential executor handles all preceding steps; only one step requires arg construction.
- The dominant arg type is local_names (49%), same as rw2 — same heuristic beam applies.
- The bare-first-atom cases (54%) can be handed to the rw3_bare executor until the args-atom is reached, then fall through to the rw2-style arg beam for that one step.
- The partition question (args_redundant / args_necessary / unexecutable) is the same as EXP-RW-033 but at the sequence level.

---

### EXP-RW-036: rw3-with-args Benchmark

**Date:** 2026-03-18
**Script:** `scripts/run_rw36_benchmark.py --parallel 4`
**Scope:** rw3-with-args (≥1 atom has positional args), step_index == 0, 297 examples

| Metric | Value |
|---|---|
| Total examples | 297 |
| GoalStart | 262/297 (88.2%) |
| Oracle first exact (dir+args) \| started | 151/262 (57.6%) |
| Cosine-5 bare \| started | 180/262 (68.7%) |
| Cosine-5 + heuristic hyp \| started | 180/262 (68.7%) |
| Oracle sequential (full) \| started | 99/262 (37.8%) |
| Cosine-5 sequential bare (full) \| started | 175/262 (66.8%) |

**Partition:**

| Bucket | Count | Rate |
|---|---|---|
| args_redundant | 180/297 | 60.6% |
| args_necessary | 9/297 | **3.0%** |
| unexecutable_oracle | 73/297 | 24.6% |
| not_started | 35/297 | 11.8% |

**Heuristic-only wins (local-hyp arg beam):** 0/262. The heuristic adds zero marginal lift over bare.

**Accept@k:**

| k | Oracle | Cosine bare |
|---|---|---|
| 1 | 157/262 (59.9%) | 180/262 (68.7%) |
| 2 | 117/262 (44.7%) | 176/262 (67.2%) |
| 3 | 106/262 (40.5%) | 175/262 (66.8%) |
| 4 | 104/262 (39.7%) | 175/262 (66.8%) |

**Oracle sequential drop** (59.9% → 44.7% at k=2): step-1 failures are dominated by `identifier_scope(args)` (29/40 step-1 failures). The canonical arg names from LeanDojo fail to resolve in the Tier-B elaborated context — the same lowering gap as rw0/rw2, but at higher frequency because args require fully-qualified local names that differ after file-faithful elaboration.

**Cosine bare is flat after k=1** (68.7% → 67.2% → 66.8%): same pattern as all-bare rw3. Composition is not the bottleneck; the difficulty is at step 0, and cosine bare handles it.

**Oracle step-0 failure taxonomy:** identifier_scope=54, rewrite_pattern_not_found=28, other_tactic_fail=20, type_mismatch=5, parse_error=4. Step-0 identifier_scope failures are predominantly args-bearing atoms (58 of 105 step-0 oracle failures).

**Key findings:**

1. **args_necessary = 3%**. Of 297 rw3-with-args examples, only 9 require args to succeed at step 0. The arg-construction problem is not material.
2. **Cosine bare outperforms oracle on this tier** (68.7% vs 57.6%), exactly as in rw2 and rw3_bare. Oracle is a noisy floor, not a ceiling.
3. **Heuristic adds zero lift.** Local-hyp arg beam (tried up to 5 hyps × 2 directions per premise) wins on 0 examples not already won by bare. The 9 `args_necessary` cases are not solved by this heuristic either — they are either in the `unexecutable_oracle` bucket or represent exotic arg patterns the heuristic doesn't cover.
4. **Sequential: cosine bare dominates oracle** (66.8% vs 37.8%). Oracle collapses at step 1 due to `identifier_scope(args)` at each step. Bare cosine re-scopes and avoids this entirely.
5. **Atom profile** (started): 183 examples have 1 args-atom, 69 have 2, 10 have 3-4. First atom is bare in 54% — matching the data profile prediction. The bare executor handles all preceding steps; only one step needed args.

**Decision (frozen):** The rewrite family is operationally solved **enough to deploy as a low-cost local lane** on the current step-0 started slices, using sequential bare cosine. `args_necessary = 3%` — no material argument-construction problem. The unified executor is cosine-5 bare sequential (`cosine_rw_seq` lane), applied uniformly across rw0/rw1/rw2/rw3_bare/rw3-with-args. Caveats: GoalStart is 88.2% (not 100%); step>0 coverage is usable but not saturated; theorem-level accumulation across all rewrite tiers is not yet measured. The 24.6% `unexecutable_oracle` gap is the lowering/elaboration gap (identifier_scope + rewrite_pattern_not_found) — a goal-state fidelity problem, not retrieval or arg-construction. **No standalone learned component is justified for the currently measured rewrite subtiers.** This leaves room for later shortlist reranking, mixed-family orchestration, or step>0 hard residue work if a theorem-level experiment reveals a real gap. Oracle demoted to diagnostic status. Next: theorem-search integration experiment (EXP-RW-037).

---

### EXP-RW-037: Theorem-Search Integration

**Date:** 2026-03-18
**Script:** `scripts/run_benchmark.py` (canonical), three sequential invocations with `--search-mode no_learned` and progressive `--cosine-rw` / `--cosine-rw-seq` flags
**Scope:** 50 Mathlib theorems (same split as EXP-3.2). Three conditions run on each theorem.

**Conditions:**

| Condition | Description |
|---|---|
| `baseline` | automation + structural_core only (`--search-mode no_learned`, no rw flags) |
| `+cosine_rw` | + single-step cosine_rw lane (top-5, both directions) |
| `+cosine_rw_seq` | + sequential bare cosine_rw_seq (max_atoms=10, max_calls=50) |

**Results:**

| Condition | Proved | Rate | Delta | Regressions | Avg time/thm |
|---|---|---|---|---|---|
| baseline | 12/50 | 24.0% | — | — | 99.7s |
| +cosine_rw | 13/50 | 26.0% | +1 | 0 | 114.9s |
| +cosine_rw_seq | 12/50 | 24.0% | 0 | 0 | 106.1s |

**Close-lane breakdown (baseline):** failed=31, skipped=7, solver_bootstrap=7, automation=4, structural_core=1

**Close-lane breakdown (+cosine_rw):** same as baseline + cosine_rw=1 (`WeierstrassCurve.Jacobian.addU_of_Z_eq_zero_right`, 13 attempts, path: automation→cosine_rw→automation). 9 theorems had `rw [...]` tactics fired; 19 subgoal closures attributed to cosine_rw lane.

**Close-lane breakdown (+cosine_rw_seq):** identical to baseline at theorem level — cosine_rw_seq did not win any theorems. However, it was active: 9 theorems had `rw_seq(...)` tactics fired, 16 subgoal closures attributed to cosine_rw_seq. The seq lane is not "dead"; it failed to convert subgoal progress into theorem closes on this split.

**Unique theorems won by cosine_rw:** 1 (`WeierstrassCurve.Jacobian.addU_of_Z_eq_zero_right`)

**Unique theorems won by cosine_rw_seq:** 0

**No regressions.** The one theorem cosine_rw won was lost under cosine_rw_seq (regression within rw conditions, not vs baseline): seq path was automation→cosine_rw_seq→structural_core with repeated `rw_seq(10)` attempts that failed to close; single-step path succeeded via automation→cosine_rw→automation. This is a runtime-policy issue: seq uses up the budget without handing back to automation promptly enough.

**Important semantic note (historical, discovered post-run):** at the time of this ablation,
`--cosine-rw-seq` *replaced* single-step `cosine_rw` in the static lane order — it was not
additive. This was not visible from the lane-summary reporting, which only showed theorem-closing
lanes, not subgoal touches. Per-theorem files (`runs/rw37_cosine_rw.jsonl`,
`runs/rw37_cosine_rw_seq.jsonl`) are the correct ground truth.

**Decision (frozen):** `+cosine_rw_seq` adds 0 theorem wins (delta < 2 threshold) and loses the
one theorem cosine_rw won, due to a policy issue in that historical run (seq exhausted budget
before returning to automation). `+cosine_rw` adds +1, also below the ≥2 freeze threshold.
**Current deployed policy:** `cosine_rw_seq` now runs as an additive fallback *after* single-step
`cosine_rw`; future benchmarks should use `--cosine-rw --cosine-rw-seq`. The proving bottleneck on
this split is structural decomposition (31/50 failed), not rewrite coverage.

---

### EXP-3.2f: NAV-003 + Move Supervision Benchmark

**Date:** 2026-03-18
**Checkpoint:** `models/NAV-003_step5000.pt`
**Training:** TC-001 template classifier (2000 steps, 222.6s, PAB stable) + NAV-003 navigator with
move supervision (5000 steps, 821.1s, PAB chaotic). Historical NAV-003 used move heads
`subtask=9`, `goal_heads=64`, `trigger_sigs=128`. Post-audit alignment correction removes
`subtask_kind` from navigator supervision and keeps it on the template/planning side.
`auxiliary_loss_weight=0.25`.
**Script:** `scripts/run_benchmark.py --cosine-rw` (full search mode, additive cosine_rw policy)
**Theorems:** Same 50 from EXP-3.2

| Condition | Proved | Rate | Δ baseline | Avg time/thm |
|---|---|---|---|---|
| NAV-002 baseline (no_learned) | 12/50 | 24.0% | — | 99.7s |
| NAV-002 +cosine_rw (no_learned) | 13/50 | 26.0% | +1 | 114.9s |
| **NAV-003 +cosine_rw (full)** | **14/50** | **28.0%** | **+2** | **22.7s** |

**Close-lane breakdown (NAV-003):** failed=29, skipped=7, automation=8, solver_bootstrap=5, cosine_rw=1

**Lane activity:** structural_core touches=28 closes=58; automation touches=24 closes=30; cosine_rw touches=10 closes=16; solver_bootstrap touches=8 closes=8. rw tactic rows=10.

**Key findings:**
- +2 theorems vs NAV-002 no_learned baseline (12→14) — but see search-mode ablation below
- **4.4× faster** (22.7s vs 99.7s/theorem)
- automation lane: 4→8; solver_bootstrap: 7→5; cosine_rw: 1 win (additive policy working)
- PAB still chaotic. goal_target_head_acc=71%, trigger_sig_micro_acc=96.5% healthy. subtask_kind_acc=7.7% — below majority baseline (21.9%) and below random (11.1%); head is collapsing, not undertrained. Root cause undiagnosed — do not adjust weights until confusion matrix and class-weighted loss are evaluated.

**Search-mode ablation (same 50 theorems, NAV-003 checkpoint):**

| Mode | Proved | Rate | Avg time/thm |
|---|---|---|---|
| full | 14/50 | 28.0% | 22.7s |
| no_learned | 14/50 | 28.0% | 22.0s |
| learned_structural | 14/50 | 28.0% | 19.8s |

Lane breakdown identical across all three modes. **The learned lane contributes zero theorem closes.** NAV-003's improvement over NAV-002 is a **routing/automation effect, not a learned local-tactic-lane effect.** The mechanism is: NAV-003 invokes automation (hammer) more frequently and earlier, producing more successes. This is consistent with the automation lane count rising 4→8 and the 4.4× speed gain — the learned component is shaping which goals reach automation and when, not closing goals directly via the candidate tactic prior.

**Hammer/routing audit (NAV-002 vs NAV-003, no_learned mode):**

| Metric | NAV-002 | NAV-003 |
|---|---|---|
| automation closes (theorems) | 4 | **8** |
| solver_bootstrap closes | 7 | 5 |
| budget hit (≥590 attempts) | **1** | 0 |
| mean attempts/theorem | 27.4 | **19.3** |
| max attempts | **600** | 58 |
| avg time/thm | 99.7s | 22.0s |

**Confirmed mechanism — routing/automation effect:** automation doubles (4→8), accounting for the full +2 net gain (+4 automation, −2 bootstrap). NAV-003 reaches the hammer-delegation decision faster and fires earlier; hammer either succeeds quickly or fails quickly — no wasted search. The 600-attempt outlier in NAV-002 explains most of the 4.4× wall-clock gap.

**Premise-value ablation (NAV-003 no_learned vs no_learned --no-learned-premises):**

| Metric | with premises | without premises |
|---|---|---|
| Proved | 14/50 | 14/50 |
| automation closes | 8 | 8 |
| auto subgoal closes | 30 | 25 |
| mean attempts | 19.3 | 20.7 |
| max attempts | 58 | 68 |

**Learned premises add zero theorem closes.** −5 automation subgoal closes without premises suggests marginal hammer efficiency loss, but not enough to affect theorem-level outcomes on this split. The routing gain is independent of the learned premise list.

**Decision (frozen):** NAV-003 is the new primary checkpoint. The confirmed mechanism is: better
hammer routing (automation 4→8), independent of both the learned candidate lane and learned
premises. The learned component is improving the controller's decision to delegate to automation,
not the content of what gets suggested.

**Alignment correction (post-audit):**
- the external motivated-move additions did **not** break runtime search, Lean lowering, or the
  rewrite executor
- the conceptual issue was placement: `subtask_kind` is a planning/controller label and should not
  be trained on the navigator trunk
- navigator auxiliary supervision is now limited to descriptive local-state heads
  (`goal_target_head`, `trigger_signature`)
- template recognition / planning remains the correct consumer of `subtask_kind`

---

### EXP-3.2g: NAV-004 Aligned Config Benchmark

**Date:** 2026-03-18
**Checkpoint:** `models/NAV-004_step5000.pt`
**Training:** Navigator with aligned move supervision — `goal_target_head + trigger_signature` only (subtask_kind removed: misplaced as navigator head, belongs on recognition/planning side). 5000 steps, 714s (13% faster than NAV-003). PAB chaotic. goal_target_head_acc=60%, trigger_sig_micro_acc=96.8%.
**Script:** `scripts/run_benchmark.py --cosine-rw` (full search mode)
**Theorems:** Same 50 from EXP-3.2

| Checkpoint | Mode | Proved | auto | bootstrap | cosine_rw | mean attempts | avg time/thm |
|---|---|---|---|---|---|---|---|
| NAV-002 | no_learned | 12/50 | 4 | 7 | 0 | 27.4 | 99.7s |
| NAV-003 | no_learned | 14/50 | 8 | 5 | 1 | 19.3 | 22.0s |
| NAV-003 | full | 14/50 | 8 | 5 | 1 | 39.7 | 22.7s |
| **NAV-004** | **full** | **14/50** | **7** | **5** | **1** | **39.5** | **21.3s** |

**Finding:** NAV-004 matches NAV-003 at 14/50. Removing the collapsed subtask head was operationally neutral — theorem count identical, routing behavior materially consistent with NAV-003, theorem-level outcome unchanged. The 12→14 gain over NAV-002 is stable and confirmed as a routing/automation effect independent of subtask supervision. The subtask head in NAV-003 was architecturally misplaced but did not corrupt theorem search.

**Decision (frozen):** NAV-004 is the new primary checkpoint — conceptually aligned (SubtaskIR on recognition/planning side only; descriptive regularization on navigator trunk) and operationally equivalent to NAV-003. All future benchmarks use NAV-004.

---

### EXP-META-001: Cross-Run Lane Meta-Analysis

**Date:** 2026-03-19
**Script:** `scripts/meta_analyze_local_lanes.py`
**Inputs:** `runs/rw37_results.json`, `runs/rw38_results.json`, `runs/rw39_results_v3.json`, `runs/rw41_results_v2.json`, `runs/rw42_results.json`, plus component runs `simp0_*`, `apply0`, `refine0`

**Purpose:** Aggregate theorem-search wrappers and local family component benchmarks into a single cross-run view of lane value, cost, and residual structure.

**Cross-run lane taxonomy:**
- **Finisher:** `cosine_rw` is the only currently productive theorem-winning local lane.
- **Scaffolder:** interleaved bootstrap is active and cheaper than the old structural fallback, but adds no theorem wins on its own.
- **Helper / transformer:** `simp` produces real local simplification and subgoal closures, but not theorem-level lift on the current 50-theorem slice.
- **Dormant specialist:** `apply` is locally real but theorem-dormant as a globally enabled lane.

**Cross-run synthesis:**
1. The rewrite family is the only currently productive theorem-winning family in search. This remains the deployed local finisher lane.
2. Interleaved bootstrap should be treated as a **systems baseline**, not a theorem-winning family. Its value is earlier structural cleanup and faster automation closure.
3. `simp` is supportive but not yet productive. It is safe and useful locally, but not yet justified as a default theorem-search lane on theorem-close yield alone.
4. `apply` is component-real but theorem-dormant on this slice. Ungated deployment spends calls on dead-end theorems; the gated version safely suppresses that spend.
5. The next research frontier is not “another always-on local lane.” It is **residual-conditioned selection and typed reranking** over small scoped candidate sets.

**Decision (frozen):** The local-family story is now explicit:
- deploy `cosine_rw` as the finisher lane
- deploy interleaved bootstrap as the structural runtime baseline
- keep `simp` as a helper lane and `apply` as a dormant/gated specialist
- shift the next learned target to typed reranking over cosine top-5, not more globally enabled lanes

---

### EXP-RANK-044: Typed Reranking Diagnostic for `apply` and `refine`

**Date:** 2026-03-19
**Script:** `scripts/run_rank044_reranker_diagnostic.py`
**Purpose:** Test whether cheap typed features can recover the right top-1 candidate from cosine top-5 on families where scope is often already adequate but cosine top-1 is weak.

#### apply (n=91, gold_in_top5=89%)

| Rule | top1/started | top1/eligible | MRR@5 |
|---|---|---|---|
| cosine_top1 | 37/91 (40.7%) | 37/81 | 0.643 |
| rule_head_shape | 46/91 (50.5%) | 46/81 | 0.728 |
| rule_combined | 45/91 (49.5%) | 45/81 | 0.726 |
| oracle_top5 | 81/91 | 81/81 | 1.000 |

#### refine_named (n=49, gold_in_top5=69%)

| Rule | top1/started | top1/eligible | MRR@5 |
|---|---|---|---|
| cosine_top1 | 9/49 (18.4%) | 9/34 | 0.491 |
| rule_combined | 14/49 (28.6%) | 14/34 | 0.591 |
| oracle_top5 | 34/49 | 34/34 | 1.000 |

#### refine_anon

Typed head/shape features are neutral-to-harmful relative to cosine. The anonymous subset is not
well captured by current surface typed features and should not be pooled with `refine_named` when
planning the next local improvement.

**Key findings:**
1. **`apply` is a reranking problem, not a scoping problem.** Gold is already in the cosine top-5 for 89% of started goals, and cheap head/shape rules recover +9 additional top-1 hits over cosine top-1.
2. **`refine_named` is also a reranking problem.** Gold is in the top-5 often enough to matter, and typed rule-combination yields a substantial relative gain over plain cosine top-1.
3. **`refine_anon` is a different computational regime.** Current typed features do not help; treat it as a separate residual family.
4. The remaining gap is still large: `apply` leaves 35 eligible top-5 cases misranked even after head+shape rules. This justifies a small learned or extended-feature reranker over top-5.

**Decision (frozen):**
- Do **not** deploy `refine` as a theorem-search lane yet.
- Do **not** invest in more globally enabled `apply` / `refine` lanes first.
- The next local learning target is a **typed reranker over cosine top-5** for `apply` and `refine_named`.
- `refine_anon` remains outside the current reranking scope.

---

### EXP-RERANK-045: Learned Reranker vs Typed Rules

**Date:** 2026-03-19
**Purpose:** Test whether a small learned reranker over cosine top-5 beats the typed head/shape heuristics on the families identified by `EXP-RANK-044`.

| Method | apply top1/eligible | apply MRR | refine_named top1/eligible | refine_named MRR |
|---|---|---|---|---|
| cosine_top1 | 37/81 (45.7%) | 0.643 | 9/34 (26.5%) | 0.491 |
| rule_head_shape | 46/81 (56.8%) | 0.728 | 13/34 (38.2%) | 0.577 |
| reranker (CV) | 40.1/81 (49.5%) | 0.675 | 20.9/34 (61.5%) | 0.789 |

**Key findings:**
1. **`refine_named` is now a validated learned-reranker target.** The learned reranker strongly beats both cosine and the typed rule, with a large MRR gain (`0.491 -> 0.789`).
2. **`apply` is not yet a learned-reranker target at current data scale.** The typed `head+shape` rule remains the best policy; the learned reranker is unstable and underperforms the rule.
3. The correct architecture is therefore **family-specific**:
   - `apply` -> typed heuristic reranker
   - `refine_named` -> learned reranker
   - `refine_anon` -> separate unresolved residual regime
4. The next immediate lever for `apply` is data scale or richer features, not theorem-search deployment of the current learned reranker.

**Decision (frozen):**
- Promote `refine_named` to the first **validated learned reranker** target.
- Keep `apply` on the typed heuristic reranker until a larger train-set reranker overtakes the rule.
- Do **not** collapse `apply` and `refine_named` into one uniform reranking policy.

---

### EXP-RERANK-045b: Scaled Reranker (apply + refine_named)

**Date:** 2026-03-19
**Purpose:** Re-run the reranker with substantially more training data to test whether the earlier `apply` failure was a small-data effect.

#### apply (`train=1501`, `eval=81`)

| Method | top1/eval | MRR@5 |
|---|---|---|
| cosine_top1 | 37/81 | 0.643 |
| rule_head_shape | 46/81 | 0.728 |
| reranker (held-out) | 50/81 | 0.758 |

#### refine_named (`train=724`, `eval=34`)

| Method | top1/eval | MRR@5 |
|---|---|---|
| cosine_top1 | 9/34 | 0.491 |
| rule_head_shape | 13/34 | 0.577 |
| reranker (held-out) | 20/34 | 0.765 |

**Key findings:**
1. **The earlier `apply` failure was a data-scale problem.** With 1501 training examples, the learned reranker now beats both cosine and the typed rule.
2. **`refine_named` remains a strong learned-reranker target.** The scaled reranker keeps a large lead over both cosine and the typed rule.
3. The correct current split is now:
   - `apply` -> learned reranker
   - `refine_named` -> learned reranker
   - `refine_anon` -> separate unresolved residual regime
4. This validates a **shared learned reranker policy at the component/ranking level** for `apply` and `refine_named`.

**Decision (current):**
- Supersede the narrower `EXP-RERANK-045` split conclusion.
- Use the learned reranker as the canonical component-level top-1 selector for both `apply` and `refine_named`.
- Keep the theorem-search claim precise: theorem-level deployment is the **next experiment**, not yet the present result.

---

### EXP-RERANK-046: Component-Level `LeanAccepted` Benchmark

**Date:** 2026-03-19
**Purpose:** Test whether the ranking lift from `EXP-RERANK-045b` transfers to actual executable tactic acceptance in Lean.

#### apply (`n=107`, `started=91`, `gold_in_scope=76/91`)

| Method | LeanAccepted / started |
|---|---|
| cosine_top1 | 15/91 (16.5%) |
| rule_top1 | 15/91 (16.5%) |
| reranker_top1 | 13/91 (14.3%) |

#### refine_named (`n=54`, `started=46`, `gold_in_scope=42/46`)

| Method | LeanAccepted / started |
|---|---|
| cosine_top1 | 1/46 (2.2%) |
| rule_top1 | 1/46 (2.2%) |
| reranker_top1 | 2/46 (4.3%) |

#### overall (`started=137`)

| Method | LeanAccepted / started |
|---|---|
| cosine_top1 | 16/137 (11.7%) |
| rule_top1 | 16/137 (11.7%) |
| reranker_top1 | 15/137 (10.9%) |

**Key findings:**
1. **Ranking lift does not transfer to Lean acceptance.** The learned reranker improves offline gold-premise recovery, but this does not produce better executable tactic acceptance.
2. **`apply` is not a theorem-name reranking problem at the executable layer.** The main bottleneck is whether a candidate actually unifies with the current goal under Lean elaboration, not whether the annotated premise is ranked first.
3. **`refine_named` remains a structural/planning problem at the executable layer.** Better premise ranking barely changes `LeanAccepted`; selecting the right name is not enough.
4. The real gap is therefore between:
   - ranking the annotated premise,
   - and selecting a **structured executable action** that Lean will accept.

**Decision (current):**
- `EXP-RERANK-045b` remains valid as a **ranking-level** result.
- It does **not** justify theorem-search deployment of the current reranker.
- The next learned target is no longer plain premise-name reranking. It is:
  - **executable action selection** / unification-aware filtering for `apply`
  - **structured skeleton/planning selection** for `refine`

---

### EXP-3.3: Mathlib Test Split

**Date:** ____

| Metric | Value | Notes |
|---|---|---|
| Total theorems | ~2000 | |
| Proved | | |
| Prove rate | | Target: >= 15% |
| By source: benchmark_theorems | / | |
| By source: mathlib_test_split | / | |

### EXP-3.4: Hammer Delegation Analysis

**Date:** ____

| Path | Theorems proved | % of total proved | Avg attempts | Notes |
|---|---|---|---|---|
| Navigator only | | | | AUTOMATION >= 0 |
| Hammer delegated | | | | AUTOMATION = -1 |
| Hammer fallback (failed first) | | | | |

**Observations:**

---

## Experiment Cycle 4: Ablation Matrix

*Purpose: Isolate the contribution of each architectural component.*

### Full Ablation Results

**Checkpoint:** Best from EXP-1.1
**Eval set:** MiniF2F-test (488) + Mathlib test split

| # | Variant | MiniF2F proved | MiniF2F rate | Mathlib proved | Mathlib rate | recall@16 | FP/proof | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Full Wayfinder | | | | | | | Primary |
| 2 | Dense retrieval (no proof network) | | | | | | | Stream 1 thesis |
| 3 | Tactic classification (no navigation) | | | | | | | Stream 2 thesis |
| 4 | No spreading activation | | | | | | | |
| 5 | No progress head | | | | | | | |
| 6 | Continuous decoder (no ternary) | | | | | | | |
| 7 | No IDF weighting | | | | | | | |
| 8 | No bank alignment (anchors only) | | | | | | | |
| 9 | Binary critic (BCE) | | | | | | | HTPS soft-target thesis |
| 10 | No proof history | | | | | | | LeanProgress thesis |
| 11 | No hammer delegation | | | | | | | |
| 12 | No accessible-premises filter | | | | | | | ReProver thesis |
| 13 | 3-bank navigation (original) | | | | | | | 6-bank expansion thesis |
| 14 | Encoder: LeanDojo-byt5 (1472d) | | | | | | | Domain encoder thesis (sep=0.623 vs 0.587) |
| 15 | Encoder: pplx-embed bidir-Qwen3 (1024d) | | | | | | | Novel architecture thesis (sep=0.600) |

### Per-Ablation PAB Comparison

| # | Variant | PAB stability | PAB regime | Crystallization rate | Nav acc (final) |
|---|---|---|---|---|---|
| 1 | Full Wayfinder | | | | |
| 6 | Continuous decoder | | | | |
| 13 | 3-bank navigation | | | | |

**Key findings:**
- Navigation thesis (1 vs 2): ____
- Architecture thesis (1 vs 3): ____
- Ternary thesis (1 vs 6): ____
- 6-bank thesis (1 vs 13): ____
- Spreading thesis (1 vs 4): ____

---

## Experiment Cycle 4.6: Multi-Pass Landmark Retrieval

*Purpose: Validate multi-pass landmark selection as a retrieval improvement over single-pass bridge_potential.*

### EXP-4.6a: Paired Retrieval Strategy Comparison

**Date:** 2026-03-15
**Script:** `scripts/anchor_gap_analysis.py --db data/proof_network_v3.db --strategy <strategy> --seed 42 --samples 200`
**DB:** proof_network_v3.db (242K entities, 339K depends_on links, typed anchor categories)
**Protocol:** Paired evaluation — identical 164 samples with premises, seed=42 for reproducibility.

| Strategy | recall@16 | Perfect (100%) | Zero (0%) | Δ vs bridge | Notes |
|----------|-----------|----------------|-----------|-------------|-------|
| bridge_potential | 26.7% | 13.4% | 54.9% | baseline | Single selector, IDF-weighted neighborhood overlap |
| multi_pass (no lens) | **43.3%** | **25.6%** | **34.1%** | **+17.6pp (+66%)** | 4-selector fusion, signed RRF, greedy diversity |
| multi_pass + lens (replace) | 11.3% | 4.3% | 73.8% | -15.4pp | BUG: committee replaced reranked order |
| multi_pass + lens (modulate) | 40.6% | 24.4% | 38.4% | +13.9pp | Committee modulates, doesn't replace |

**Architecture:**
- **4 selectors**: bridge_potential (neighborhood bridge scoring), self_match (entity-level cross-check), accessibility (import-accessible overlap), hub_suppressor (negative-only, penalizes hubs)
- **Ternary voting**: +1 (support), 0 (abstain), -1 (oppose). 0 is NOT "middle rank"
- **Signed RRF fusion**: `score = support_rrf - 0.5 * oppose_rrf` with confidence weighting
- **Convergence classification**: converged (≥2 families agree), isolated (1 strong supporter), conflicted (significant opposition)
- **Greedy diversity**: neighborhood overlap > 0.8 = skip (prevents redundant landmarks)

**Files created:** `src/landmark_selectors.py`, `src/retrieval_scoring.py`, `src/retrieval_stages.py`

**Key finding:** Multi-pass +17.6pp over bridge_potential is a 66% relative improvement. The gain comes from selector diversity (4 independent signals fused), not from any single selector being better than bridge_potential.

### EXP-4.6b: Freeze/Residual Pipeline

**Date:** 2026-03-15
**Purpose:** Category-weighted contextual freezing with iterative resolve.

**Architecture:**
- **FrozenLandmarkState**: Immutable committed knowledge (converged landmarks only; isolated requires confidence≥0.8 AND margin>0.005 AND novelty>0.3)
- **LandmarkResidualReport**: Uncovered anchor mass by category, phase signal detection
- **Category weighting** (Minority Channel Advantage): semantic(1.0) > structural(0.9) > constant(0.8) > proof(0.7) > locality(0.3)
- **Conflict bifurcation**: Productive conflicts (>10% novel residual) get 1.1x bonus, unproductive get 0.5x penalty
- **Iterative freeze-resolve**: Up to 3 passes, entropy stop when decrease < 0.05 or coverage gain < 0.01

**Files created:** `src/landmark_freeze.py`

**Observation:** Conservative freeze commits too many landmarks → 100% query anchor coverage → empty residual → 3/5 lenses have zero signal. This is the primary bottleneck for lens utility.

### EXP-4.6c: Lens Guidance Layer

**Date:** 2026-03-15
**Purpose:** Society-of-Mind style committee modulation over collapsed retrieval frontier.

**Architecture:**
- **GuidancePacket**: Deterministic collapse output with candidate anchor sets, residual diagnostics, negative evidence
- **5 rule-based specialists**: BridgeCoherenceLens, ResidualCoverageLens, ConstantMatchLens, LocalityGuardLens, HubPenaltyLens
- **Coherence states**: stable/isolated/coupled/conflicted/degraded → act/trust_lens/bifurcate/expand_more
- **Modulation**: `combined = rerank_score * (1 + 0.3 * informativeness * fused_score)` — committee adjusts, never replaces

**Files created:** `src/lens_guidance.py`, `src/lens_models.py`, `src/coherence_engine.py`

**Result:** Lens guidance provides marginal value (-2.7pp vs no-lens) because aggressive freezing leaves empty residuals. Only bridge_coherence (always votes) and hub_penalty (uses hub_in_degrees) are active. The other 3 lenses abstain on most candidates.

**Decision (2026-03-15):** Lens guidance committed as distillation bridge (Wave 1.5), not as primary retrieval driver. Modulation is the only shipping mode. Replace mode falsified.

**Hypothesis status updates:**
- H35 (guidance modulation preserves deterministic retrieval): **Partially supported** — 93.8% preservation, but -2.7pp means marginal negative impact
- H36 (replace underperforms modulation): **Supported** — replace 11.3% vs modulate 40.6% vs deterministic 43.3%

---

## Experiment Cycle 5: PAB Framework Validation

*Purpose: Stream 3 -- does PAB trajectory evaluation reveal information that endpoint metrics miss?*

### EXP-5.1: Similar Endpoints, Different Trajectories

**Date:** ____
**Configuration pair:** Two ablation variants with <= 2% endpoint difference.

| Metric | Config A | Config B |
|---|---|---|
| MiniF2F prove rate | | |
| PAB stability_mean | | |
| PAB regime | | |

**Stress test (distribution shift -- unseen theorem domains):**

| Test | Config A | Config B | Better |
|---|---|---|---|
| Novel domain theorems | | | |
| Longer proofs (10+ steps) | | | |
| Heavy type coercion | | | |

**PAB prediction (better trajectory = more robust):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

### EXP-5.2: Navigational Accuracy Trajectory vs Prove Rate

**Date:** ____

| Configuration | Nav accuracy trajectory slope | Final prove rate | Correlation |
|---|---|---|---|
| | | | |

**Does nav accuracy trajectory predict final proof success better than loss trajectory?**
[ ] Yes / [ ] No / [ ] Inconclusive

### EXP-5.3: Bank Crystallization Order

**Date:** ____

| Bank | Crystallization step | Predicted order | Actual order | Notes |
|---|---|---|---|---|
| DOMAIN | | 1 (Regime A, highest symmetry) | | NAV-002: 0.95+ from step 200 |
| CONTEXT | | 2 (Regime A) | | NAV-002: 0.73-0.88 |
| DECOMPOSITION | | 3 (Regime A/B boundary) | | NAV-002: 0.63-0.88 |
| STRUCTURE | | 4 (Regime B, low symmetry) | | NAV-002: 0.43-0.66, unstable |
| AUTOMATION | | 5 (Regime B) | | NAV-002: 0.25-0.60, unstable |
| DEPTH | | 6 (Regime B, worst fit to ternary) | | NAV-002: 0.24-0.60, crashes in Phase C |

**Prediction rationale:** Regime A banks (high |G_μ|) crystallize first because symmetry makes them easy to specify. Regime B banks (low |G_μ|) never fully crystallize in a monolithic navigator because the shared bridge creates composition gap γ with Regime A banks. See RESEARCH §2.9.3.

**Does crystallization order match specification complexity predictions?** [ ] Yes / [ ] Partially / [ ] No

---

## Experiment Cycle 6: Society of Mind Architecture (v2)

*Purpose: Stream 4 — does SoM decomposition outperform monolithic navigation? Does specification complexity theory predict optimal decomposition?*

### EXP-6.0: Composition Gap Control Experiment

**Date:** 2026-03-15
**Purpose:** Distinguish "chaotic PAB = high γ from shared bridge" from alternative explanations (insufficient capacity, label noise, wrong LR).

**Protocol:** Three configs with 5000 training steps each, seed=42, MPS device. Config 2 (separate-bridge monolithic) deferred — requires new DualBridgeNavigator module.

Note: PAB stability_regime was "chaotic" for all three, but the specialist training script uses a minimal PAB recording path that doesn't feed proper tier accuracies — only step numbers and training loss. The raw PAB stability_mean is therefore not directly comparable between monolithic (which feeds full nav accuracy into PAB) and specialists (which feed zeros). Loss trajectory variance is the more reliable stability measure.

| Config | Banks | Bridge | Final Loss (mean±std, last 200) | Late Stability (window std) | Overall Stability | PAB regime | Notes |
|--------|-------|--------|--------------------------------|---------------------------|-------------------|------------|-------|
| v1 monolithic | 6 (all) | 1 shared | 24.6 ± 5.8 | 5.9 | 26.7 | chaotic (0.346) | Reproduces NAV-002 |
| Separate-bridge mono | — | 2 independent | — | — | — | — | DEFERRED — needs DualBridgeNavigator |
| v2 Specialist A | 2 (domain, context) | independent | **8.6 ± 3.3** | **3.2** | **12.0** | chaotic* | *PAB under-instrumented |
| v2 Specialist B | 4 (struct, auto, depth, decomp) | independent | 24.9 ± 6.9 | 6.8 | 23.7 | chaotic* | *PAB under-instrumented |

**Convergence trajectories:**
- Monolithic: 108 → 22 (step 500 → final)
- Specialist A: 53 → 4 (converges cleanly, 2.9x lower final loss than monolithic)
- Specialist B: 61 → 22 (matches monolithic — hard banks are intrinsically chaotic)

**Interpretation:**
- **Specialist A** (easy banks: DOMAIN, CONTEXT) is significantly more stable — 1.8x lower late-stage variance, 2.9x lower final loss. Bridge sharing was suppressing easy-bank convergence.
- **Specialist B** (hard banks: STRUCTURE, AUTO, DEPTH, DECOMP) is indistinguishable from monolithic — same loss level, similar variance. Hard-bank chaos is intrinsic to the task, not caused by bridge sharing.
- **Partial support for composition gap thesis**: decomposition helps easy banks converge better, but doesn't resolve hard-bank instability. The shared bridge was hurting easy banks by forcing them through a bottleneck optimized for hard banks.

**Stream 4 verdict:** [x] Bridge sharing causes γ (for easy banks) / [ ] Hidden sharing causes γ / [ ] Capacity issue / [ ] Inconclusive

**Next step:** Config 2 (separate-bridge monolithic) would distinguish whether the bridge sharing alone accounts for Specialist A's improvement. If separate-bridge mono matches Specialist A stability on easy banks, the bridge is the culprit. If not, full specialist independence is needed.

### EXP-6.1: Template Extraction and Taxonomy ✅ COMPLETE

**Date:** 2026-03-14
**Script:** `scripts/extract_templates.py --data data/nav_train.jsonl --output-dir data/`

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Total templates discovered | 9 | 8-15 | Heuristic bank-signature matching (all 9 predefined templates used) |
| Coverage (% proofs assigned to template) | 100% | ≥ 90% | ✅ PASS — every proof maps to a template |
| Silhouette score | N/A | ≥ 0.3 | Not computed — heuristic assignment, not clustering |
| Largest template (% of corpus) | 36.7% (REWRITE_CHAIN) | ≤ 40% | ✅ PASS |
| Smallest template (% of corpus) | 0.3% (HAMMER_DELEGATE) | ≥ 2% | ⚠️ BELOW — 3 templates under 2%: HAMMER_DELEGATE (0.3%), CONTRAPOSITIVE (0.6%), EPSILON_DELTA (0.6%) |

**Per-template breakdown:**

| Template ID | Count | % | Notes |
|-------------|-------|---|-------|
| REWRITE_CHAIN | 28001 | 36.7% | Largest — sequence of rewrites to normal form |
| DECIDE | 25391 | 33.3% | Single automation closes goal (omega, simp, decide) |
| DECOMPOSE_AND_CONQUER | 10857 | 14.2% | Split into independent subgoals |
| APPLY_CHAIN | 5889 | 7.7% | Sequence of apply/exact targeting lemmas |
| CASE_ANALYSIS | 3314 | 4.3% | Split on data constructor or hypothesis |
| INDUCT_THEN_CLOSE | 1709 | 2.2% | Induction + base/step closed by automation |
| EPSILON_DELTA | 488 | 0.6% | Analysis-style witness introduction |
| CONTRAPOSITIVE | 445 | 0.6% | Negate goal, derive contradiction |
| HAMMER_DELEGATE | 241 | 0.3% | Fully delegated to ATP |

**Outputs:** `data/template_taxonomy.json`, `data/nav_train_templates.jsonl` (321,554 examples with template_id)

**Analysis:** The two dominant templates (REWRITE_CHAIN + DECIDE = 70%) represent the "easy" proof strategies — short, automation-heavy proofs. The long-tail templates (CONTRAPOSITIVE, EPSILON_DELTA, HAMMER_DELEGATE at <1% each) represent rare but structurally distinct strategies. The class imbalance will need handling in the template classifier (Phase 6.2) — consider class-weighted loss or oversampling.

**Stop/go:** ✅ PASS on coverage and template count. ⚠️ 3 templates below 2% minimum — acceptable because they're structurally meaningful (not noise clusters), but may be hard for the classifier to learn. Consider merging HAMMER_DELEGATE into DECIDE (both are automation-dominated) if classifier performance is poor on these classes.

### EXP-6.2: Template Classifier (RECOGNITION Slot)

**Date:** ____
**Script:** `scripts/train_template_classifier.py`

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Top-1 accuracy | | ≥ 60% | On nav_eval.jsonl |
| Top-3 accuracy | | ≥ 85% | |
| PAB stability_regime | | stable | Regime A task |
| Training steps to convergence | | | |

**Per-template precision/recall:**

| Template ID | Precision | Recall | F1 | Confusion with |
|-------------|-----------|--------|-----|---------------|
| | | | | |

**Stop/go:** Top-3 ≥ 50%. PAB must be "stable" (if chaotic, taxonomy is wrong, not classifier capacity).

### EXP-6.3: Specialist Decomposition (EXECUTION Slot)

**Date:** ____
**Script:** `scripts/train_specialist.py --specialist A|B|B1|B2`

**Per-specialist PAB comparison:**

| Specialist | Banks | PAB stability | PAB regime | Per-bank accuracy | vs v1 monolithic |
|-----------|-------|---------------|------------|-------------------|------------------|
| Navigator-A (easy) | DOMAIN, CONTEXT | | | | |
| Navigator-B (hard) | STRUCT, AUTO, DEPTH, DECOMP | | | | |
| Navigator-B1 (if needed) | STRUCT, DECOMP | | | | |
| Navigator-B2 (if needed) | AUTO, DEPTH | | | | |

**Combined specialist accuracy vs v1 monolithic:**

| Metric | v1 monolithic (NAV-002) | v2 two-specialist | v2 three-specialist | Notes |
|--------|------------------------|-------------------|--------------------|----|
| Mean nav accuracy | 0.637 (step 5000) | | | |
| PAB stability_mean | 0.341 | | | |
| PAB stability_regime | chaotic | | | Target: all stable |
| Total training compute | 566s | | | |
| Total params | ~400K | | | |

### EXP-6.4: Ternary Decoder Falsification Test

**Date:** 2026-03-15
**Purpose:** Test whether decoder sign crystallization means "specified at init" (σ≈1, the decoder matters) or "dead decoder" (signs are random noise).

**Protocol:** Resume NAV-002 from step 5000 (only step5000 checkpoint available, not step2000). Flip direction head weights (4608 total across 6 bank heads). Train 500 additional steps (to step 5500). Compare loss trajectory and nav accuracy.

| Condition | Weights flipped | Loss at resume (step 5000) | Loss at step 5100 | Loss at step 5500 | Nav acc (step 5400) | PAB stability | Recovery |
|-----------|----------------|---------------------------|--------------------|--------------------|---------------------|---------------|----------|
| No flip (control) | 0/4608 | 35.7 | 54.3 | 32.6 | 0.438 | 0.177 | N/A |
| 50% sign flip | 2324/4608 | **807.1** (22x) | 787.5 | 94.3 (3x) | 0.484 | 0.250 | Partial — 500 steps insufficient |
| 100% sign flip | 4608/4608 | **262.0** (7x) | 166.9 | 42.7 (1.3x) | 0.449 | 0.210 | Near-full recovery |

**Verdict:** [x] Decoder signs encode information (σ≈0.5-0.8 interpretation valid)

**Key findings:**
1. **50% flip causes immediate 22x loss spike** that only partially recovers in 500 steps (still 3x baseline). Signs carry learned navigational information.
2. **100% flip paradoxically recovers faster than 50%** — flipping ALL signs is equivalent to negating the decoder (learnable via sign inversion of output layer), while 50% corruption destroys internal consistency.
3. **σ ≈ 0.5-0.8**: Signs encode information but are partially re-learnable, consistent with "specified at init" interpretation from NAV-002 where crystallization was 99.85% from step 100.
4. **Nav accuracy similar across conditions** (0.438-0.484) because the 500 training steps bring all conditions toward the same attractor, but the loss gap shows the decoder is doing real work.

**Implication:** The ternary decoder design is validated — signs are not dead weights. Continue with ternary architecture for specialist decomposition (Phase 6.4).

### EXP-6.5: Full SoM Pipeline Benchmark

**Date:** ____
**Script:** `scripts/run_benchmark.py --mode som --config configs/wayfinder_v2.yaml`

| Metric | v1 monolithic | v2 SoM | Delta | p-value (McNemar) | Notes |
|--------|--------------|--------|-------|-------------------|-------|
| MiniF2F proved | | | | | |
| MiniF2F rate | | | | | |
| Mathlib proved | | | | | |
| Mathlib rate | | | | | |
| Avg attempts/proof | | | | | |
| Neural FP/proof | | | | | |

**Venn diagram of proved theorems:**

| Category | Count | Notes |
|----------|-------|-------|
| Proved by both v1 and v2 | | Core overlap |
| Proved by v1 only | | v1 advantage — which theorems? |
| Proved by v2 only | | v2 advantage — are these Regime B? |
| Proved by neither | | |

**Regime analysis of v2-only theorems** (key validation of the theory):

| v2-only theorem | Primary bank difficulty | Template used | Predicted regime | Notes |
|----------------|----------------------|--------------|-----------------|-------|
| | | | | |

**Stop/go:** v2 ≥ v1 OR v2-only theorems are predominately Regime B (validates narrative conversion).

### EXP-6.6: Stream 4 Ablation Matrix

| # | Variant | MiniF2F rate | vs v1 delta | PAB regime | Template acc | Notes |
|---|---------|-------------|-------------|------------|-------------|-------|
| 1 | v1 monolithic (NAV-002) | | baseline | chaotic | N/A | |
| 2 | v2 two-specialist, no templates | | | | N/A | Decomposition alone |
| 3 | v2 two-specialist, with templates | | | | | Decomp + narrative |
| 4 | v2 three-specialist, with templates | | | | | Finer decomposition |
| 5 | v2 full SoM (with sketch predictor) | | | | | Full pipeline |
| 6 | Template classifier only (no specialists) | | | | | Templates alone |
| 7 | Specialists only (no templates) | | | | N/A | Specialists alone |

**Key findings:**
- Decomposition benefit (#2 vs #1): ____
- Narrative conversion benefit (#3 vs #2): ____
- Finer decomposition benefit (#4 vs #3): ____
- Full SoM benefit (#5 vs #3): ____
- Templates alone (#6 vs #1): ____
- Specialists alone (#7 vs #1): ____

### EXP-6.7: Slot-by-Slot Failure Analysis

**Date:** ____
**Purpose:** For theorems that v2 fails on, diagnose WHICH slot failed.

| Failure Slot | Count | % | Top failure reason | Fix candidate |
|-------------|-------|---|-------------------|---------------|
| RECOGNITION (wrong template) | | | | |
| PLANNING (bad sketch) | | | | |
| EXECUTION (wrong tactic/premise) | | | | |
| VERIFICATION (correct tactic rejected) | | | | |
| ARBITER (bad goal selection/routing) | | | | |

---

## Experiment Cycle 7: Energy-Constrained Navigation (v3)

*All experiments in this cycle use `--mode v3` and produce SearchTrace output. v3A experiments run first; v3B experiments are gated on v3A success.*

**Runtime mode**: All metrics below are tagged by runtime mode (v1/v2/v3) for comparability with Cycles 1-6.

### EXP-7.0: Core Eval Validity Gate [v3A prerequisite]

**Date:** ____
**Purpose:** Validate that the positive-only pipeline produces trustworthy metrics before investing in negative data or energy refinement.

| Check | Status | Date | Notes |
|-------|--------|------|-------|
| Checkpoint round-trip parity | | | save → load → identical metrics |
| Lane A verification consistent | | | `raw_success` on same theorems ±0 |
| Accessible-premises: import-based | | | Not file-based |
| Zero train/eval theorem-ID overlap | | | `assert set(train) & set(eval) == set()` |
| v1 baseline reproducible (±1%) | | | Same seed, 3 runs |

**Gate:** All five checks must pass before proceeding to EXP-7.1.

### EXP-7.1: OTP Scoring Reforms [v3A]

**Date:** ____
**Purpose:** Validate OTP-derived scoring improvements against Phase 6 baseline.

#### EXP-7.1a: Bank-IDF Weighting (MCA Validation)

| Metric | Baseline (v2) | +Bank-IDF | Δ | Notes |
|--------|--------------|-----------|---|-------|
| recall@16 | | | | Premise selection |
| recall@16 (hard theorems) | | | | Expect largest gain here |
| nav accuracy (STRUCTURE) | | | | Sparse bank, MCA predicts gain |
| nav accuracy (AUTOMATION) | | | | Sparse bank |
| nav accuracy (DOMAIN) | | | | Dense bank, MCA predicts small Δ |
| raw_success | | | | Must not degrade |

#### EXP-7.1b: Zero-Sparsity Curriculum (IZ Validation)

| Metric | Standard curriculum | OTP curriculum | Δ | Notes |
|--------|-------------------|----------------|---|-------|
| PAB convergence (steps to stable) | | | | Per-bank |
| Final nav accuracy | | | | Should be ≥ standard |
| Bank-specific curves | | | | Expect hard banks converge faster |

#### EXP-7.1c: Ternary Distribution Analysis ✅ COMPLETE

**Date:** 2026-03-14
**Script:** `scripts/ternary_target_analysis.py`
**Sample:** 50,000 of 321,554 training examples

| Bank | % zero | % +1 | % -1 | Zero-Heavy? | Notes |
|------|--------|-------|-------|-------------|-------|
| STRUCTURE | 66.4% | 10.0% | 23.6% | YES | Symmetric neg/pos split |
| DOMAIN | 98.3% | 0.0% | 1.7% | YES | Near-degenerate — never positive, needs enrichment |
| DEPTH | 58.6% | 17.8% | 23.5% | YES | Symmetric neg/pos split |
| AUTOMATION | 24.7% | 51.7% | 23.5% | **NO** | Only non-zero-dominant bank; primary active signal |
| CONTEXT | 70.2% | 28.0% | 1.8% | YES | Positively skewed when active |
| DECOMPOSITION | 76.0% | 21.0% | 3.0% | YES | Positively skewed when active |

**729-bin direction space:**
- **17 of 729 bins occupied (2.3%)** — extreme sparsity confirmed
- Shannon entropy: 2.965 bits / 9.510 max (normalized = 0.312)
- Top 3 bins = 66.1% of examples:
  1. `(0,0,0,0,0,0)` — "maintain/do nothing" — 23.2%
  2. `(0,0,0,+1,0,0)` — "pure automation" — 22.6%
  3. `(-1,0,-1,-1,0,0)` — "simplify+reduce+de-automate" — 20.2%

**OTP dimensionality:**
- Mean active banks: 2.06, median: 2
- 79.2% of examples use 0-3 active banks
- 0.6% use all 6 banks — low-dimensional subspace confirmed
- Trimodal: peaks at dim-0 (23.2%), dim-1 (24.1%), dim-3 (23.4%)

**Bank co-activation (strongest pairs):**
- structure + depth: 32.7% (lift 2.35x) — encode related geometric info
- context + decomposition: 20.6% (lift 2.88x) — validates SoM specialist split
- depth + automation: 41.4% (lift 1.33x)

**OTP predictions validated:**
- [x] Zero is majority value for 5/6 banks (Informational Zero confirmed)
- [x] 729-bin direction space is very sparse (17/729 = 2.3% occupied)
- [ ] Sparser banks have higher per-activation discrimination (needs Phase 2.2 retrieval data)

**Design implications:**
1. **domain bank near-degenerate** (98.3% zero, never positive) — needs tactic mapping enrichment or merging
2. **17 effective bins, not 729** — OTP scoring can parameterize over archetypes
3. **automation is the backbone** (75.3% active) — correctly placed in Specialist B
4. **structure-depth coupling** (lift 2.35x) — potential OTP rotation target to decorrelate

### EXP-7.2: Negative Data Quality Audit [v3A]

**Date:** ____
**Purpose:** Validate negative example collection before training.

| Source | Count | % semantic | % infra | % weak | Spot-check valid? | Notes |
|--------|-------|-----------|---------|--------|-------------------|-------|
| sorry_hole | | | | — | | |
| perturbation | | | | — | | |
| suggestion_trace | | | | — | | |
| unchosen_weak | | — | — | 100% | | Re-run 100: ___ % succeed |

**Gate:** Semantic failure rate ≥ 50% of collected negatives. If not, improve filtering before training.

### EXP-7.3: Censor Training [v3A]

**Date:** ____
**Purpose:** Train and evaluate the asymmetric censor.

| Metric | Symmetric BCE | Asymmetric BCE (2:1) | Asymmetric (3:1) | Notes |
|--------|--------------|---------------------|------------------|-------|
| AUROC (overall) | | | | Target ≥ 0.80 |
| AUPRC | | | | More informative than AUROC |
| ECE (calibration) | | | | |
| False-prune rate | | | | Target < 5% |
| AUROC (sorry_hole) | | | | Per-source breakdown |
| AUROC (perturbation) | | | | |
| AUROC (suggestion_trace) | | | | |
| AUROC (unchosen_weak) | | | | |

**MCA validation:** If asymmetric improves AUPRC without degrading false-prune rate → MCA applies.

### EXP-7.4: Contrastive Navigator Training [v3A]

**Date:** ____
**Purpose:** Validate contrastive loss does not degrade positive-only performance.

| λ_contra | nav accuracy | PAB stability | raw_success | Δ vs λ=0 | Notes |
|----------|-------------|--------------|-------------|----------|-------|
| 0 (baseline) | | | | — | |
| 0.01 | | | | | |
| 0.05 | | | | | |
| 0.1 | | | | | |
| 0.2 | | | | | |

**Per-bank impact:**

| Bank | Best λ | Accuracy Δ | Notes |
|------|--------|-----------|-------|
| STRUCTURE | | | Expect largest contrastive benefit |
| AUTOMATION | | | |
| DOMAIN | | | |
| DEPTH | | | |
| CONTEXT | | | |
| DECOMPOSITION | | | |

### EXP-7.4b: Guidance-Layer Distillation [v3A — Wave 1.5]

**Date:** 2026-03-15
**Purpose:** Validate deterministic retrieval collapse as a teacher for multi-lens guidance and committee distillation.

**Engineering status:** 1,645 tests passing after guidance-layer integration.

| Strategy | recall@16 | Perfect | Zero | Notes |
|----------|-----------|---------|------|-------|
| bridge_potential | 26.7% | 13.4% | 54.9% | Paired baseline on seed=42 |
| multi_pass (no lens) | 43.3% | 25.6% | 34.1% | Deterministic collapse baseline |
| multi_pass + lens (v1, replace) | 11.3% | 4.3% | 73.8% | Replacement falsified as shipping mode |
| multi_pass + lens (v2, modulate) | 40.6% | 24.4% | 38.4% | Preserves 93.8% of deterministic recall |

**Outcome**:
- Deterministic collapse remains the primary retrieval driver. Guidance is a refinement layer, not a substitute.
- `replace` is now falsified as a default runtime mode and remains ablation-only.
- `modulate` preserves most deterministic retrieval performance while keeping the committee non-destructive.
- Current committee behavior is mostly corrective/negative (hub suppression) with weak positive signal. The next gain must come from candidate-grounded positive residual lenses or learned specialists.
- Guidance-layer distillation is now committed as a bridge track between symbolic retrieval and learned refinement.

**Next ladder**:
1. Candidate-grounded symbolic lenses over unsupported constants / residual structure.
2. One learned lens trained on ambiguous candidates only.
3. Multi-lens learned committee with typed abstention.
4. Distill committee outputs into a lightweight reranker / specialist head.
5. Mine committee disagreement as active-learning data.

### EXP-7.7: Active Boundary Learning [v3A]

**Date:** ____
**Purpose:** Active learning convergence.

| Iteration | Uncertainty fraction | Lean calls | New examples | raw_success | Δ | Notes |
|-----------|---------------------|-----------|--------------|-------------|---|-------|
| 0 (pre-active) | | — | — | | — | |
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

**Convergence:** Uncertainty fraction decreases ≥ 10%/iteration.

### EXP-7.9: A/B/C/D Comparison [v3A: A→B+, v3B: C→D]

**Date:** ____
**Purpose:** End-to-end validation of all Phase 7 components.

| Condition | Track | raw_success | attempts/thm | lean_calls/thm | false_prune_rate | Notes |
|-----------|-------|-------------|-------------|----------------|-----------------|-------|
| A (Phase 6 baseline) | — | | | | — | |
| A+ (OTP scoring) | v3A | | | | — | |
| B (+passive negatives) | v3A | | | | | |
| B+ (+active boundary) | v3A | | | | | |
| C (+energy refinement) | v3B | | | | | |
| D (+energy + active) | v3B | | | | | |

**Expected ordering:** D ≥ C ≥ B+ ≥ B ≥ A+ ≥ A (raw_success), reversed for lean_calls/theorem.

**v3A gate:** raw_success(B+) ≥ raw_success(A) AND lean_calls/thm(B+) < lean_calls/thm(A).
**v3B gate:** raw_success(D) ≥ raw_success(B+). Otherwise v3B stays experimental.

**Statistical protocol:** McNemar's test for proved/not-proved, bootstrap 95% CI, p < 0.05.

---

## Visualization Pipeline

All visualizations use `mlx-vis` (GPU-accelerated on Metal) unless noted.

### Figure Conventions

Every figure title is an assertion, not a topic:
- "Bank positions separate mathematical domains without training" (not "Bank position distribution")
- "AUTOMATION and STRUCTURE banks crystallize before DOMAIN and CONTEXT" (not "Training dynamics")
- "Navigational retrieval matches dense retrieval at 1/50th the neural compute" (not "Retrieval comparison")

### Planned Visualizations

| Phase | Visualization | Method | Data source | Purpose |
|---|---|---|---|---|
| EXP-0.2 | Entity positions colored by retrieval success/failure | UMAP | proof_network.db entity positions | Show where anchor gaps cluster |
| EXP-1.1 | Navigational embeddings across curriculum phases | PaCMAP (animated) | Bridge outputs at steps 500/2000/5000 | Progressive structure emergence |
| EXP-1.1 | Per-bank accuracy curves (6 panels) | Line plot | PAB checkpoints | Bank learning dynamics |
| EXP-2.1 | recall@k curves: nav vs dense | Line plot | eval_retrieval output | Core Stream 1 figure |
| EXP-3.2 | Proved/failed theorems in bank position space | t-SNE | proof_network.db positions | Where does navigation succeed/fail? |
| EXP-4 | Side-by-side: full Wayfinder vs no-bank ablation | UMAP | Bridge outputs | Visual fencing: what navigation adds |
| EXP-4 | Ablation matrix heatmap | Heatmap | Ablation results table | Component importance at a glance |
| EXP-5.3 | Ternary weight evolution per bank head | Histogram series | Navigator weight snapshots | Crystallization dynamics |
| EXP-6.0 | PAB stability: monolithic vs separate-bridge vs two-specialist | Bar chart (3 configs) | PAB profiles | Composition gap γ evidence |
| EXP-6.1 | Template clusters in bank-signature space | UMAP | Bank-signature centroids | Template taxonomy coherence |
| EXP-6.3 | Per-specialist PAB stability comparison | Line plot (overlaid) | Specialist PAB profiles | Decomposition validation |
| EXP-6.4 | Ternary sign flip accuracy impact | Line plot (3 conditions) | Flip experiment logs | Decoder falsification |
| EXP-6.5 | v1 vs v2 Venn diagram of proved theorems | Venn diagram | Benchmark results | Complementarity analysis |
| EXP-6.5 | v2-only theorems colored by regime (A/B) | Scatter plot | Bank positions + results | Regime conversion validation |
| EXP-6.7 | Slot-by-slot failure breakdown | Stacked bar chart | Failure analysis logs | Diagnostic |

### Visualization Toolkit

```bash
# Entity position UMAP (anchor gap analysis)
python -c "
import mlx_vis
# Load entity positions from proof_network.db
# Color by: domain, retrieval success, bank position
# Output: figures/entity_umap_*.png
"

# Training dynamics PaCMAP
python -c "
import mlx_vis
# Load bridge outputs from PAB checkpoints
# Animate across curriculum phases
# Output: figures/training_pacmap_*.png
"

# t-SNE of proved/failed theorems
python -c "
import mlx_vis
# Load theorem positions + benchmark results
# Color by proved/failed, size by attempt count
# Output: figures/benchmark_tsne_*.png
"
```

---

## Cumulative Findings

### Hypothesis Status Tracker

| # | Hypothesis | Status | Evidence | Cycle |
|---|---|---|---|---|
| H1 | Navigational retrieval matches or exceeds dense retrieval for premise selection | | | EXP-2.1 |
| H2 | Ternary navigational decoder outperforms tactic classification | | | EXP-4 (#1 vs #3) |
| H3 | 6-bank navigation disambiguates tactics better than 3-bank | | | EXP-4 (#1 vs #13) |
| H4 | IDF weighting is necessary for anchor relevance | | | EXP-4 (#1 vs #7) |
| H5 | Spreading activation improves later-step premise recall | | | EXP-2.2 |
| H6 | Soft critic targets outperform binary (per HTPS) | | | EXP-4 (#1 vs #9) |
| H7 | Proof history improves retrieval (per LeanProgress) | | | EXP-4 (#1 vs #10) |
| H8 | Hammer delegation for AUTOMATION=-1 goals improves prove rate | | | EXP-3.4 |
| H9 | Accessible-premises filtering gives free recall gain | | | EXP-4 (#1 vs #12) |
| H10 | Confidence-weighted scoring outperforms pure multiplicative | | | EXP-2.3 |
| H11 | AUTOMATION and STRUCTURE banks crystallize before DOMAIN and CONTEXT | | | EXP-5.3 |
| H12 | PAB trajectory predicts generalization better than endpoint metrics | | | EXP-5.1, 5.2 |
| H13 | Neural forward passes per proof are <= 50% of dense retrieval baselines | | | EXP-3.2 |
| H14 | Anchor gap analysis achieves >= 70% recall@16 on perfect queries | | | EXP-0.2 |
| H15 | The proof network separates mathematical domains without neural training | | | EXP-0.1 visualization |
| H16 | Domain-specific encoder (LeanDojo-byt5, sep=0.623) improves retrieval recall over MiniLM (sep=0.587) enough to offset 28x throughput cost | | | EXP-2.2, EXP-4 (#14) |
| H17 | Novel bidirectional decoder-to-encoder architecture (pplx-embed, sep=0.600) generalizes better than BERT-family encoders | | | EXP-4 (#15) |
| H18 | Chaotic PAB in monolithic navigator is caused by composition gap γ through shared bridge, not insufficient capacity | Partially supported | Specialist A (easy banks) converges 2.9x lower loss and 1.8x lower variance than monolithic. Specialist B (hard banks) matches monolithic. Bridge sharing hurts easy banks; hard-bank chaos is intrinsic. | EXP-6.0 (2026-03-15) |
| H19 | Proof tactic sequences cluster into discrete templates (silhouette ≥ 0.3, 8-15 clusters covering ≥ 90%) | | | EXP-6.1 |
| H20 | Template classification from goal state features is Regime A (PAB stable, top-3 ≥ 85%) | | | EXP-6.2 |
| H21 | Bank-cluster specialists each reach PAB "stable" when trained independently | | | EXP-6.3 |
| H22 | Combined specialist accuracy ≥ monolithic accuracy (additive σ, no γ penalty) | | | EXP-6.3 |
| H23 | Ternary decoder sign crystallization encodes information (not dead weights) | Supported | 50% flip: 22x loss spike, partial recovery in 500 steps. 100% flip: 7x spike, near-full recovery. Signs encode navigational info (σ≈0.5-0.8) | EXP-6.4 (2026-03-15) |
| H24 | Full SoM pipeline ≥ monolithic on benchmark, OR v2-only theorems are predominantly Regime B | | | EXP-6.5 |
| H25 | Narrative template conversion provides benefit beyond specialist decomposition alone (#3 vs #2 in ablation) | | | EXP-6.6 |
| H26 | Bank-IDF weighting (MCA) improves premise recall on hard theorems without degrading easy ones | | | EXP-7.1a |
| H27 | Zero-sparsity curriculum (IZ) accelerates PAB convergence vs standard curriculum | | | EXP-7.1b |
| H28 | Zero is the majority ternary value for most banks (Informational Zero as dominant state) | | | EXP-7.1c |
| H29 | Asymmetric censor loss (MCA-motivated) outperforms symmetric BCE on AUPRC | | | EXP-7.3 |
| H30 | Contrastive training with negative data improves STRUCTURE/AUTOMATION banks more than DOMAIN | | | EXP-7.4 |
| H31 | Active boundary learning reduces censor uncertainty ≥ 10%/iteration | | | EXP-7.7 |
| H32 | v3A (boundary learning) achieves raw_success ≥ v2 with fewer lean_calls/theorem | | | EXP-7.9 |
| H33 | [v3B] Energy-refined sketches ≥ discrete v3A scoring on raw_success | | | EXP-7.9 |
| H34 | [v3B] Energy refinement produces sparser (more OTP-aligned) ternary outputs | | | EXP-7.9 |
| H35 | Guidance modulation preserves most deterministic retrieval performance while reducing ambiguous-tail uncertainty | Partially supported | Modulation preserves 93.8% of deterministic recall (40.6% vs 43.3%) | EXP-7.4b, EXP-4.6a |
| H36 | Replace-mode guidance underperforms modulation, confirming committee is guidance not substitute | Supported | Paired seed=42 retrieval: replace 11.3%, modulate 40.6%, deterministic 43.3% | EXP-7.4b |
| H37 | Candidate-grounded or learned positive lenses outperform negative-only committee on ambiguous subset | | | EXP-7.4b |
| H38 | Committee disagreement is a high-value distillation and active-learning source | | | EXP-7.4b / EXP-7.7 |
| H39 | Multi-pass 4-selector landmark fusion outperforms single-selector bridge_potential for premise retrieval | Supported | Paired seed=42: bridge 26.7% → multi_pass 43.3% (+17.6pp, +66% relative) | EXP-4.6a |
| H40 | NAV-002's learned tactic prior contributes >0 proofs on real Mathlib theorems | Refuted | Three-mode Mathlib benchmark: learned_only=0/50, learned_structural=0/50 (learned lane=0 in both), full=0 learned-lane proofs out of 12 proved | EXP-3.2 (2026-03-16) |
| H41 | Theorem-level retrieval should be interpreted as temporal orchestration/frontier collapse, not direct post-structural step resolution | Supported | Residual retrieval on theorem-level v3 anchors is near-zero for step targets; theorem-level premise ablation yields 0 theorem-count delta on the 50-theorem slice | EXP-3.2a / EXP-3.2b |
| H42 | Oracle premise context improves residual tactic-family prediction, especially for premise-sensitive families | Supported | Goal+oracle premise beats goal-only by +4.2pp top-1, +3.6pp top-3, +0.035 macro-F1; `rw` +0.16, `refine` +0.14 recall | EXP-3.2b |
| H43 | Two-stage local execution (family → constrained local synthesis / premise grounding) is a better architectural fit than monolithic tactic prediction | Partially supported | Monolithic learned lane = 0 on Mathlib, but residual family predictor learns useful signal and improves with oracle premise context; parsability audit shows the next gap is structured argument construction, not free-form generation | EXP-3.2 / EXP-3.2b / EXP-3.2e |
| H44 | A stateful temporal controller improves proof search over static goal rotation and fixed lane order | | | EXP-6.4a / TC1-TC4 |
| H45 | Phase-conditioned lane ordering reduces Lean calls / theorem without harming prove rate | | | EXP-6.4a / TC2 |
| H46 | Temporal progress context improves next-family / next-lane prediction beyond raw goal-state features | | | EXP-6.4a / TC4 |
| H47 | Residual tactic family is predictable from goal embeddings alone (top-3 >= 70%) | Supported | RES-003: top-1=31.9%, top-3=75.0%, macF1=0.309 on 240K residual examples | EXP-RES-001 (2026-03-16) |
| H48 | Navigator premise suggestions add value to hammer delegation | Refuted (current) | Premise ablation delta=0 on 50 Mathlib theorems. Retrieval not yet exposed at right granularity | EXP-RES-002 (2026-03-16) |
| H49 | rw leaf selection from goal embedding alone exceeds random baseline on unscoped vocabulary | Supported | RW-001: 17.8% top-1 on 5K vocab (890x random). Real goal-to-lemma association learned. | EXP-RW-001 (2026-03-16) |
| H50 | Residual tactics are structured enough for family-specific constrained decoding rather than raw tactic-string generation | Supported | 82% parsable into family + structured arguments; rw/exact/apply/refine are 100% parsable, simp 92% | EXP-3.2e (2026-03-16) |
| H51 | The immediate local bottleneck is structured argument construction, not family prediction alone | Supported | Family top-3 is already 75%; oracle premises improve family prediction; template subset covers 32% and exposes the next gap as local term construction | EXP-3.2b / EXP-3.2e |
| H52 | Scoped vocabulary dramatically improves rw leaf selection over unscoped | Supported | Full 30.5% → hyps+theorem 42.1% top-1 (+38% relative). Top-5: 55.6% → 83.5%. MRR: 0.427 → 0.599. Scope=15 vs 15037. | EXP-RW-002 (2026-03-16) |
| H53 | Canonical full-name lowering dramatically improves Lean-valid rate over source names for rw | Supported | rw0: source 6% → canonical 14% overall, 14% → 30% conditional on started goals | EXP-RW-004 (2026-03-16) |
| H54 | rw0 learned decoder beats cosine baseline on LeanValid@rw0|started with real premise scope | Refuted (current) | Same-denominator step-0 benchmark: learned top-5 38.5% < cosine top-5 61.5%; learned top-1 3.8% < cosine top-1 33.3%. Cosine is the deployed rw0 baseline; learning shifts to harder tiers and reranking. | EXP-RW-013b / EXP-RW-014 (2026-03-17) |
| H55 | Step>0 replay can be made usable with theorem-faithful start states plus state-guided replay | Supported | Replay success improved 0% → 11% → 15% → 19% → 26%; first pfx_len=3 success achieved. Replay is now usable for evaluation, though still context-limited. | EXP-RW-018 / EXP-RW-020 / EXP-RW-023 / EXP-RW-024 (2026-03-17) |
| H56 | Replayed step>0 states support the same scoped-cosine `rw0` execution regime as step-0 | Supported (current) | Step>0 cosine top-5 reaches 71.4% on replayed+scoped examples, with 100% gold-in-scope and similar scope size. Step>0 is a coverage problem before it is a local-selection problem. | EXP-RW-026 / EXP-RW-027 / EXP-RW-028 (2026-03-17) |
| H57 | A standalone learned direction head is the right next learned target for the current rw0/rw1 split | Refuted (current) | rw1 direction is 94% backward among oracle successes, and always-backward nearly matches both-dir oracle success. The useful signal is largely tier identity, not a separate head. | EXP-RW-029 / EXP-RW-030 (2026-03-17) |
| H58 | The rewrite family still needs rewrite-specific learned decoders for arguments or composition on the current measured slices | Refuted (current) | rw2 and rw3-with-args both show `args_necessary` only 3-4%, and all-bare rw3 sequential cosine reaches 70.7% with almost all post-step failures concentrated at step 0. The remaining rewrite gap is state/lowering fidelity, not local argument or composition modeling. | EXP-RW-033 / EXP-RW-035 / EXP-RW-036 (2026-03-17 to 2026-03-18) |
| H59 | A controller-facing move layer above `ActionIR` is the right next abstraction target once rewrite execution collapses | Supported (design-current) | Once local rewrite execution reduces to cheap scoped cosine + verification, the unresolved problem becomes choosing the right local transformation type under explicit triggers. `SubtaskIR` and trigger profiles encode that controller-visible residue without polluting Lean lowering semantics. | Design integration (2026-03-18) |

### Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| 2026-03-06 | Architecture: navigational proof search via structured semantic networks | ModelAtlas paradigm applied to theorem proving | Dense retrieval (ReProver), token classification, MCTS |
| 2026-03-06 | 6-bank ternary navigation (729 direction bins) | Resolves many-to-one tactic mapping from 3-bank (27 bins) | 3-bank (fallback available), continuous directions |
| 2026-03-06 | UW-SO adaptive loss weighting | Handles different convergence rates across nav/anchor/progress/critic | Fixed coefficients, manual curriculum |
| 2026-03-06 | Soft critic targets (MSE, not BCE) | HTPS finding: binary critic worse than no critic | Binary critic (ablation only) |
| 2026-03-06 | Proof history via mean-pooled closed goals | LeanProgress: 61.8% -> 75.1% from history | No history (ablation only) |
| 2026-03-10 | Encoder selection: all-MiniLM-L6-v2 (384d) primary; LeanDojo-byt5 + pplx-embed as ablation candidates | 15 models evaluated (EXP-0.3/0.3b). MiniLM: best throughput (617 g/s), only 6% behind LeanDojo (sep=0.623) on separation. Inverse size-separation relationship confirmed across all 15 models. Domain fine-tuning shifts curve but can't overcome 7B space collapse. | LeanDojo-byt5 (#1 sep but 28x slower), pplx-embed (#2 sep, novel bidir arch), Qwen3-Embed, MathBERT, 7B models (all worse trade-offs) |
| 2026-03-10 | SoM pivot: decompose monolithic navigator into typed temporal slots | NAV-001/002 chaotic PAB (stability_mean 0.32-0.34), bank difficulty hierarchy confirms Regime A/B split, specification complexity composition gap theorem provides formal justification | Larger monolithic model (more capacity), curriculum tuning only, mixture-of-experts within single model |
| 2026-03-10 | Initial specialist split: Navigator-A (DOMAIN, CONTEXT) + Navigator-B (STRUCTURE, AUTO, DEPTH, DECOMP) | NAV-002 per-bank accuracy shows clean Regime A/B separation at DECOMPOSITION boundary | Per-bank specialists (6 total, too fine), 3-way split (hard to justify boundary), random groupings |
| 2026-03-10 | Narrative regime conversion via template classification | Specification complexity Theorem 4.1: |G_μ| determines regime. Story templates introduce symmetry. Cross-project evidence (ARC, Ralph, Relational-AI) | No templates (rely on specialist decomposition alone), continuous template embeddings (lose discrete regime conversion) |
| 2026-03-13 | v3 parallel runtime with v3A/v3B maturity split | Phase 7 integrates OTP theory (scoring reforms), negative learning (censor, contrastive), and energy-based refinement. v3A (practical: negatives, censor, OTP scoring) ships first; v3B (experimental: energy function, Gumbel-softmax) gated on v3A success. v1/v2 frozen as baselines. | Monolithic Phase 7 (no maturity split), modify v2 in place (contaminates baseline), energy-first (no proven censor to provide E_censor) |
| 2026-03-15 | Guidance committee defaults to modulation, not replacement | Paired perfect-query retrieval on identical samples: deterministic 43.3%, replace 11.3%, modulate 40.6%. Guidance is therefore committed as a distillation/refinement bridge; replacement remains ablation-only until it beats modulation and no-lens retrieval. | Replace-mode as shipping runtime, no guidance layer, energy-first without guidance distillation |
| 2026-03-15 | Multi-pass landmark retrieval: 4-selector fusion with signed RRF, convergence classification, greedy diversity | Paired eval: bridge_potential 26.7% → multi_pass 43.3% recall@16 (+17.6pp). Selector diversity beats single-selector optimization. | More selectors (diminishing returns predicted), single better selector (ceiling at ~30%), learned reranking (deferred to Phase 7 guidance distillation) |
| 2026-03-15 | Composition gap partially confirmed: specialist decomposition helps easy banks (DOMAIN, CONTEXT) but not hard banks (STRUCTURE, AUTO, DEPTH, DECOMP) | EXP-6.0b: Specialist A 2.9x lower loss, 1.8x lower variance vs monolithic. Specialist B ≈ monolithic. | Full independence (validated for easy), separate-bridge test (deferred), capacity increase for hard banks |
| 2026-03-15 | Pantograph integration: removed deliberate guard in run_benchmark.py, verified PyPantograph 0.3.13 works with Lean 4.28.0 on Init theorems | 6/6 Pantograph unit tests pass, 5/7 smoke test theorems proved (all decidable ones). Mathlib requires compatible Lean project. | Axle as fallback (slower, cloud), PyPantograph with Mathlib project, MathLinter LSP |
| 2026-03-16 | Three-mode Mathlib benchmark confirms learned lane = 0. Bootstrap (intros+simp+aesop) proves 22%, hammer adds 1 more. Navigator premise suggestions useful for hammer but tactic predictions non-functional. | EXP-3.2: 0/50 learned_only, 11/50 learned_structural (all bootstrap), 12/50 full (1 extra from hammer). Retraining needed. | Retrain NAV on v3 DB with new domain labels, fine-tune tactic vocabulary, add Mathlib-specific structural heuristics |
| 2026-03-16 | Re-scope theorem-level navigation as temporal orchestration/frontier collapse. Promote the residual executor to the primary learned local policy. | Premise ablation shows 0 theorem-count delta on the 50-theorem Mathlib slice, while oracle premises improve residual family prediction (`28.0% → 32.2%` top-1; `rw` +0.16, `refine` +0.14). Theorem-level retrieval and residual local execution are distinct stages. | Keep forcing theorem-level retrieval to solve step targets directly; retrain the monolithic navigator without changing the task decomposition |
| 2026-03-16 | Temporal controller is operative in `proof_search.search()` shadow/active modes, but still partial at the full-system level. | Shadow traces and active-mode runs now flow through the benchmarked search path. `arbiter.som_search()` still does not consume the controller, and `budget_slice` / `replan` are not yet full runtime controls. | Claim full SoM-level temporal integration already exists; leave the controller as offline-only |
| 2026-03-16 | Temporal controller is routing layer, not solver. All 12 proved theorems succeed at escalation=0, phase=structural_setup. TC value depends on residual executor quality in local_close phase. | EXP-3.2d shadow traces: 40% structural_setup, 31% local_close, 30% repair_or_replan. 0 replans. | Wait for learned executor to produce value in local_close before evaluating TC active mode |
| 2026-03-16 | 2-stage architecture confirmed: family prediction (Stage 1) learnable from embeddings, premise grounding (Stage 2) adds +4.2pp. rw/refine benefit most from premises. | Goal-only 28.0% vs goal+premise 32.2% top-1. Top-3=75% already useful as search gate. | Train premise-conditioned executor, wire as family gate, don't block on retrieval quality |
| 2026-03-16 | Freeze the semantic `rw0` benchmark and split infrastructure from decoder quality. | EXP-RW-004: source 6% → canonical 14% overall; 30% Lean-valid conditional on started goals. GoalStart=46% is an infrastructure number, not a decoder number. | Keep a single blended success metric; treat source-name failures as model errors |
| 2026-03-16 | The next gate is theorem-faithful goal creation, not more decoder work. | EXP-RW-005 harness audit: step-0 can start on a small probe, but step>0 remains blocked by `goal_creation_fail`. Tier C replay exists, but depends on Tier A/B producing a valid theorem-level base state first. | Run the full learned `rw0` benchmark now; optimize the decoder before the harness is trustworthy |
| 2026-03-16 | Site-addressed replay changes the tactic identity. | Once replay can target a specific subgoal, `(env_key, goal_state, tactic)` is no longer a pure cache key; `goal_id` must be part of the identity for multi-goal correctness. | Reuse old tactic cache shape and hope subgoal collisions are rare |
| 2026-03-16 | Move from raw tactic-string generation to family-specific constrained decoding with deterministic lowering. | 82% of residual tactics are parsable into family + structured arguments, but only 32% are covered by the first simple template subset. The gap is local term construction, not absence of structure. | Continue treating the local problem as unconstrained Lean generation; wait for theorem-level retrieval to solve step targets directly |
| 2026-03-16 | Canonical names are the semantic target for rw; source names demoted to secondary deployability metric. Benchmark split into GoalStart (infrastructure) and LeanValid|started (decoder quality). rw0 is the focused tier for first formal learned lane. | EXP-RW-004: source 6% → canonical 14% → 30% conditional. Infrastructure and decoder quality are separate tracks. | Pursue env fidelity and decoder quality in parallel tracks |
| 2026-03-17 | Deploy `cosine_rw` as the canonical rw0 runtime lane; demote standalone learned rw0 decoder work. | EXP-RW-014/015/016: cosine top-5 reaches 61.5% vs 62.8% oracle ceiling and yields +1/50 theorem lift with zero regressions. Learned top-5 is 38.5% and does not justify itself on rw0. | Keep optimizing the standalone rw0 decoder against a slice where cosine already nearly saturates the oracle |
| 2026-03-17 | Step>0 replay is now a usable benchmark track, but its next bottleneck is source-context compilation, not more lexical rewriting. | EXP-RW-024 clears the 25% replay gate (12/47). EXP-RW-025 census/audit shows that theorem sites commonly rely on `open scoped`, local notation, local attributes, include/omit, and inline `... in` forms. | Continue stacking ad hoc name heuristics on top of a thin active-context wrapper |
| 2026-03-17 | Treat theorem-site source context as a compiled DSL (`ContextIR`) and make it the next explicit execution phase. | The wrapper and replay path now fail mostly on scoped source effects, not theorem start-state creation. Validation scripts quantify the gap and make unsupported forms explicit. | Treat theorem source as plain header text and keep extending wrapper heuristics opportunistically |
| 2026-03-17 | Use `LeanValid@k | replayed` as the primary step>0 metric; demote oracle-match to diagnostic status. | EXP-RW-027/028: cosine top-5 succeeds on replayed step>0 states at 71.4%, while half of oracle/cosine disagreements come from stale labels and half from replay drift. Step>0 oracle is not a clean ceiling. | Treat annotated-premise oracle as the primary step>0 target despite replay drift and label noise |
| 2026-03-17 | Extend the runtime rewrite lane to `rw0 + rw1` with tier-conditioned default direction; do not build a standalone direction head on the current split. | EXP-RW-029/030: rw1 cosine top-5 (60.5%) matches rw0, and direction is 94% backward on rw1 oracle successes. The useful signal is the tier/family gate, not a separate direction predictor. | Build a standalone learned direction head for rw1 before deploying rw1 in search |
| 2026-03-17 | Keep `ContextIR` as a parallel coverage/compiler track, not the blocker for extending the runtime rewrite lane from `rw0` to `rw1`. | EXP-RW-027/029/030 together show that step>0 replayed states are already semantically usable and rw1 behaves like rw0 under tier-conditioned default direction. `ContextIR` still matters for replay denominator and future families, but not for delaying rw1 deployment. | Finish the full source-context compiler before extending the deployed rewrite lane beyond rw0 |
| 2026-03-18 | Freeze the rewrite family around the sequential bare cosine executor and stop pursuing rewrite-specific learned decoders on the current measured slices. | EXP-RW-033/035/036: rw2 `args_necessary` is 4%, rw3-with-args `args_necessary` is 3%, and all-bare rw3 composition collapses to repeated first-step retrieval. The remaining rewrite gap is lowering/state fidelity, not argument or composition modeling. | Continue building rewrite-specific decoders for rw2/rw3 before theorem-level integration |
| 2026-03-18 | Add a motivated move layer (`SubtaskIR` + trigger profiles) above `ActionIR`, and mine reusable move schemas from successful canonical traces. | The rewrite-family collapse changes the remaining problem from "emit a tactic string" to "choose the right local transformation under explicit trigger conditions." `SubtaskIR` gives the controller that contract, and schema mining turns solved traces into reusable classical infrastructure. | Keep the controller operating over raw tactic strings / lane IDs only; import the Human-Oriented ATP ideas only as prose |
| | Scoring mechanism: ____ | EXP-2.3 results | |

---

## Appendix A: Raw Eval Outputs

*Per-experiment run outputs stored at:*
`runs/<run_id>/eval_results.json` -- structured JSON from eval scripts
`runs/<run_id>/benchmark_results.json` -- benchmark runner output
`runs/<run_id>/benchmark_results.jsonl` -- per-theorem detail

## Appendix B: Training Curves

*Training logs stored at:*
`runs/<run_id>/<run_id>_training_log.jsonl`

*Schema per line:*
```json
{
  "step": 100,
  "L_total": 2.31,
  "L_nav": 1.45,
  "L_anchor": 0.52,
  "L_progress": 0.22,
  "L_critic": 0.12,
  "nav_accuracy_mean": 0.42,
  "nav_accuracy_STRUCTURE": 0.55,
  "nav_accuracy_DOMAIN": 0.35,
  "anchor_f1": 0.28,
  "progress_mae": 2.1,
  "ternary_crystallization": 0.45,
  "curriculum_phase": "B"
}
```

## Appendix C: Compute Log

| Date | Experiment | Machine | Duration | Notes |
|---|---|---|---|---|
| 2026-03-10 | EXP-0.3/0.3b Encoder eval | laptop (M-series) | ~10min | 15 models, MPS |
| 2026-03-10 | NAV-001 (baseline training) | laptop (M-series) | 537s | No scheduler, 3/8 PAB |
| 2026-03-10 | NAV-002 (full PAB) | laptop (M-series) | 566s | Cosine LR, 8/8 PAB |
| | Proof network extraction | macpro (Xeon) | | CPU-only |
| | Anchor gap analysis | macpro (Xeon) | | CPU-only, iterative |
| | Benchmark (MiniF2F) | laptop (M-series) | | Target: <24h |
| 2026-03-15 | EXP-4.6a/b/c Multi-pass retrieval | laptop (M-series) | ~20min | seed=42, 164 paired samples, v3 DB |
| 2026-03-15 | EXP-6.0b composition gap (3 configs) | laptop (M-series) | ~27min total (642s mono + 470s spec-A + 492s spec-B) | 5000 steps each, seed=42 |
| 2026-03-15 | EXP-6.3b ternary falsification (3 conditions) | laptop (M-series) | ~6min total (3x ~2min parallel) | 500 steps each from NAV-002_step5000 |
| 2026-03-15 | EXP-3.1 Init logic benchmark (3 iterations, 30 theorems each) | laptop (M-series) | v1: 1263s, v2: 575s, v3: 176s | Pantograph + structural fallback evolution |
| 2026-03-16 | EXP-3.2 Mathlib 3-mode benchmark (50 theorems × 3 modes) | laptop (M-series) | 5233s total (261+1137+3835) | Pantograph+Mathlib v4.27.0, NAV-002 |
| 2026-03-16 | RES-003 residual executor training (20 epochs, 240K) | macpro (Xeon) | 438s | Pre-computed embeddings, class-weighted CE |
| 2026-03-16 | Oracle-premise executor comparison (15 epochs, 50K) | laptop (M-series) | ~120s | Goal-only vs goal+premise side-by-side |
| 2026-03-16 | TC 3-mode Mathlib benchmark (off/shadow/active × 50) | laptop (M-series) | ~9000s total | Pantograph + Mathlib v4.27.0 |
| 2026-03-16 | RW-001 rw decoder training (20 epochs, 5K examples) | laptop (M-series) | 69s | Two-stage: shape + pointer over 5K vocab |
| 2026-03-16 | RW-002 scoped vocab ablation (4 conditions × 15 epochs) | laptop (M-series) | ~90s total | Pointer model, same split across conditions |
| 2026-03-16 | EXP-RW-003/004 rw0 tier split + semantic benchmark | laptop (M-series) | ~10min | 80 rw0 step-0, 3 conditions, Pantograph+Mathlib |
| | Ablation runs (13x) | laptop (M-series) | | |

## Appendix D: PAB Profile Schema

*Full PAB profiles saved per-run at: `runs/<run_id>/<run_id>_pab_profile.json`*

*Schema:*
```json
{
  "experiment_id": "NAV-001",
  "checkpoints": [50, 100, 150, "..."],
  "stability": [0.35, 0.22, 0.18, "..."],
  "predictability": [0.12, 0.08, 0.04, "..."],
  "domain_accuracies": {
    "STRUCTURE": [0.40, 0.55, 0.68, "..."],
    "DOMAIN": [0.25, 0.35, 0.45, "..."],
    "AUTOMATION": [0.42, 0.58, 0.72, "..."]
  },
  "ternary_crystallization": [0.33, 0.45, 0.62, "..."],
  "summary": {
    "stability_mean": 0.14,
    "stability_std": 0.08,
    "predictability_final": 0.03,
    "stability_regime": "stable",
    "crystallization_rate": 0.0015,
    "convergence_step": 2500
  }
}
```

## Appendix E: Anchor Gap Analysis Detail

*Gap analysis JSONL stored at: `data/gap_analysis.jsonl`*

*Schema per line:*
```json
{
  "theorem_id": "Nat.add_comm",
  "goal_state": "...",
  "ground_truth_premises": ["Nat.succ_add", "Nat.zero_add"],
  "retrieved_premises": ["Nat.add_zero", "Nat.succ_add", "..."],
  "recall_at_16": 0.5,
  "missed_premises": ["Nat.zero_add"],
  "gap_anchors": ["comm-pattern", "zero-identity"]
}
```

---

*Ledger version: 2.0. Updated 2026-03-10 with Cycle 6 (SoM) skeleton, H18-H25 hypotheses, corrected crystallization order predictions, v2 decision log entries, statistical significance framework, and v2 visualizations.*
