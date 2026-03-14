# Wayfinder: Experiment Results Ledger

**Project:** Wayfinder -- Navigational Proof Search Through Structured Semantic Networks
**Version:** 1.0
**Started:** March 2026
**Hardware:** Apple Silicon M-series (primary); Xeon macpro + i7 homebridge (CPU workers)
**Eval set:** Frozen `data/nav_eval.jsonl`; MiniF2F-test (488); Mathlib test split (~2k)

Three-stream results ledger: navigation validation + architecture evaluation + PAB process evaluation.

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

**IMPORTANT (2026-03-14):** The original 100% recall pass was on the trace-bounded DB (78K entities, 34.5% premise coverage). This result is partially invalid — it measured recall within a truncated candidate universe. The anchor gap analysis must be re-run on the expanded DB (242K entities, 90.4% coverage) to validate that retrieval still works when premise-only entities compete for ranking positions.

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

**Target:** Nav recall@16 >= 80% of dense recall@16 (NOT evaluable until data gap fixed).

**Stream 1 verdict:** [ ] Navigation matches/exceeds dense / [ ] Dense superior / [x] Inconclusive — blocked by data coverage

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

**Root cause:** Same 34.5% premise coverage gap as EXP-2.1. Cannot evaluate spreading activation when the premises being retrieved don't exist as entities. See EXP-2.1 for full coverage analysis and fix plan.

**Target:** Spreading adds >= 5% recall@16 on proof steps 3+ (NOT evaluable until data gap fixed).

**Observations:** Spreading adds negligible timing overhead (~0.4% faster with spread, within noise). Once the data gap is fixed, spreading should show benefit on multi-step proofs where closed goals provide seed entities for activation.

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

**Date:** ____
**Target:** >= 80% proved within budget=100 on 1-2 step proofs.

| Metric | Value |
|---|---|
| Theorems tested | 50 |
| Proved | |
| Prove rate | |
| Avg attempts per proof | |
| Avg time per proof (s) | |

**Stop/go:** >= 80% on trivials before proceeding to benchmarks.

### EXP-3.2: MiniF2F-test Benchmark

**Date:** ____
**Script:** `scripts/run_benchmark.py --config configs/wayfinder.yaml --checkpoint models/NAV-___`

| Metric | Value | vs BL-3 (ReProver) | Notes |
|---|---|---|---|
| Total theorems | 488 | — | |
| Proved | | Delta= | |
| Prove rate | | Delta= | |
| Failed | | | |
| Avg attempts/theorem | | | |
| Avg attempts (proved only) | | | |
| Avg time/theorem (s) | | | |
| Total time | | | |
| Neural FP per proof | | | Target: <= 50% of ReProver |
| 95% CI (Clopper-Pearson) | | | On prove rate |

**Statistical note:** Report 95% Clopper-Pearson CI on prove rate. For comparison with published baselines, note whether baseline falls within CI. Use McNemar's test for paired per-theorem comparison where available.

**Stop/go:** >= 20% proved (competitive at our parameter count).

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

**Date:** ____
**Purpose:** Distinguish "chaotic PAB = high γ from shared bridge" from alternative explanations (insufficient capacity, label noise, wrong LR).

**Protocol:** Train three configs with identical total parameter count:
1. **v1 monolithic** (NAV-002 reproduction) — 1 shared bridge, 6 banks
2. **Separate-bridge monolithic** — 2 independent bridges feeding into shared hidden layers; same total params
3. **v2 two-specialist** — 2 fully independent specialists (Navigator-A + Navigator-B)

If (2) is stable but (1) is chaotic → bridge sharing is the cause (γ from interface).
If (2) is also chaotic → shared hidden layers are the cause, not just bridge.
If (3) is stable but (2) is chaotic → full independence needed (supports composition gap theorem).
If all three are chaotic → capacity or data issue, not γ.

| Config | Bridges | Hidden | Banks per unit | Total params | PAB stability | PAB regime |
|--------|---------|--------|----------------|-------------|---------------|------------|
| v1 monolithic | 1 shared | shared | 6 | ~400K | | |
| Separate-bridge mono | 2 independent | shared | 3 + 3 | ~400K | | |
| v2 two-specialist | 2 independent | independent | 2 + 4 | ~400K | | |

**Stream 4 verdict:** [ ] Bridge sharing causes γ / [ ] Hidden sharing causes γ / [ ] Capacity issue / [ ] Inconclusive

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

**Date:** ____
**Purpose:** Test whether decoder sign crystallization means "specified at init" (σ≈1, the decoder matters) or "dead decoder" (signs are random noise).

**Protocol:** Take NAV-002 checkpoint at step 2000. Randomly flip 50% of decoder ternary signs. Resume training for 500 steps. Measure accuracy impact.

| Condition | Nav accuracy before | Nav accuracy after flip | Recovery after 500 steps | Interpretation |
|-----------|--------------------|-----------------------|-------------------------|----------------|
| No flip (control) | | | | |
| 50% sign flip | | | | If no drop → decoder is dead |
| 100% sign flip | | | | If significant drop → signs encode information |

**Verdict:** [ ] Decoder signs encode information (σ≈1 interpretation valid) / [ ] Decoder is dead (signs are noise)

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
| H18 | Chaotic PAB in monolithic navigator is caused by composition gap γ through shared bridge, not insufficient capacity | | | EXP-6.0 |
| H19 | Proof tactic sequences cluster into discrete templates (silhouette ≥ 0.3, 8-15 clusters covering ≥ 90%) | | | EXP-6.1 |
| H20 | Template classification from goal state features is Regime A (PAB stable, top-3 ≥ 85%) | | | EXP-6.2 |
| H21 | Bank-cluster specialists each reach PAB "stable" when trained independently | | | EXP-6.3 |
| H22 | Combined specialist accuracy ≥ monolithic accuracy (additive σ, no γ penalty) | | | EXP-6.3 |
| H23 | Ternary decoder sign crystallization encodes information (not dead weights) | | | EXP-6.4 |
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
