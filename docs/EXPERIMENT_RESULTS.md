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

**Date:** ____
**Script:** `scripts/anchor_gap_analysis.py --db data/proof_network.db --samples 500`

| Iteration | recall@16 (perfect queries) | New anchors added | Top gap themes | Notes |
|---|---|---|---|---|
| 0 (bootstrap) | | 0 | | Initial ~300 anchors |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Top 20 gap anchors (final iteration):**

| Anchor | Miss count | Theme |
|---|---|---|
| | | |

**Stop/go:** recall@16 >= 70% on perfect queries. If <50% after 3 iterations, bank positioning needs rethinking.

### EXP-0.3: Encoder Selection

**Date:** ____
**Protocol:** Encode 1000 Mathlib goal states per candidate. Evaluate clustering quality.

| Candidate | Params | Premise cluster overlap @10 | Alpha-equiv distance | Domain separability | Byte-level | Notes |
|---|---|---|---|---|---|---|
| ByT5-small (frozen) | 299M | | | | Yes | Baseline |
| ByT5-small + LoRA | 299M | | | | Yes | Fine-tuned |
| Math-native candidate | | | | | | |
| BitNet ternary | | | | | | |

**Decision:** [ ] ByT5-small / [ ] ByT5-small + LoRA / [ ] Math-native / [ ] BitNet / [ ] Other: ____

---

## Experiment Cycle 1: Navigation Training

*Purpose: Train the navigational pipeline and track per-bank learning dynamics.*

### EXP-1.1: Curriculum Training Run

**Date:** ____
**Config:** `configs/wayfinder.yaml`
**Script:** `scripts/train_navigator.py --config configs/wayfinder.yaml --run-id NAV-___`
**Checkpoint:** `models/NAV-____step____.pt`

**Training Configuration:**

| Parameter | Value |
|---|---|
| Encoder | |
| Navigable banks | 6 (STRUCTURE, DOMAIN, DEPTH, AUTOMATION, CONTEXT, DECOMPOSITION) |
| Scoring mechanism | confidence_weighted |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Max iterations | 5000 |
| Curriculum | Phase A: 0-500 (1-2 step), Phase B: 500-2000 (<=5 step), Phase C: 2000+ (all) |
| Device | mps |

**Phase Gates:**

| Phase | Step range | Nav accuracy target | Actual | Status |
|---|---|---|---|---|
| A (warmup) | 0-500 | >= 60% on 1-2 step proofs | | [ ] Pass / [ ] Fail |
| B (growth) | 500-2000 | >= 50% on <=5 step proofs | | [ ] Pass / [ ] Fail |
| C (full) | 2000-5000 | >= 45% on all proofs | | [ ] Pass / [ ] Fail |

**Training Dynamics (sampled at PAB checkpoints every 50 steps):**

| Step | L_total | L_nav | L_anchor | L_critic | Nav acc (mean) | Anchor F1 | Progress MAE |
|---|---|---|---|---|---|---|---|
| 50 | | | | | | | |
| 100 | | | | | | | |
| 250 | | | | | | | |
| 500 | | | | | | | |
| 1000 | | | | | | | |
| 2000 | | | | | | | |
| 3000 | | | | | | | |
| 5000 | | | | | | | |

**Per-Bank Navigation Accuracy (key checkpoints):**

| Bank | Step 500 | Step 1000 | Step 2000 | Step 3000 | Step 5000 | Crystallization order |
|---|---|---|---|---|---|---|
| STRUCTURE | | | | | | |
| DOMAIN | | | | | | |
| DEPTH | | | | | | |
| AUTOMATION | | | | | | |
| CONTEXT | | | | | | |
| DECOMPOSITION | | | | | | |

**Hypothesis:** AUTOMATION and STRUCTURE crystallize first (clearest signal from tactic analysis). DOMAIN and CONTEXT crystallize last (noisier extraction).

**PAB Trajectory Profile:**

*Full profile: `runs/NAV-___/NAV-____pab_profile.json`*

| PAB Metric | Value | Assessment |
|---|---|---|
| Stability mean | | <= 0.15 = stable |
| Stability std | | Low = consistent |
| Predictability (final) | | <= 0.05 = structured |
| Stability regime | | stable / chaotic / phase_transition |
| Ternary crystallization (final) | | % of stable weight signs |
| Crystallization rate | | Slope of crystallization curve |
| Convergence step | | Step where S < 0.10 for 5 consecutive |

**Observations:**

---

## Experiment Cycle 2: Retrieval Comparison

*Purpose: Stream 1 core question -- does navigational retrieval match or exceed dense retrieval?*

### EXP-2.1: Navigational vs Dense Retrieval

**Date:** ____
**Script:** `scripts/eval_retrieval.py --config configs/wayfinder.yaml --checkpoint models/NAV-___`
**Checkpoint:** Best from EXP-1.1

| k | Nav recall@k | Dense recall@k | Delta | Nav time (ms) | Dense time (ms) | Speedup |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |

**Target:** Nav recall@16 >= 80% of dense recall@16.

**Stream 1 verdict:** [ ] Navigation matches/exceeds dense / [ ] Dense superior / [ ] Inconclusive

### EXP-2.2: Spreading Activation Benefit

**Date:** ____
**Script:** `scripts/eval_spreading.py --config configs/wayfinder.yaml --checkpoint models/NAV-___`

| k | No-spread recall@k | With-spread recall@k | Delta | Notes |
|---|---|---|---|---|
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |

| Timing | No-spread (ms) | With-spread (ms) |
|---|---|---|
| Average per-goal | | |

**Target:** Spreading adds >= 5% recall@16 on proof steps 3+.

**Observations:**

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
| STRUCTURE | | 1-2 | | |
| AUTOMATION | | 1-2 | | |
| DEPTH | | 3-4 | | |
| DECOMPOSITION | | 3-4 | | |
| DOMAIN | | 5-6 | | |
| CONTEXT | | 5-6 | | |

**Does crystallization order match architectural expectations?** [ ] Yes / [ ] Partially / [ ] No

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

### Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| 2026-03-06 | Architecture: navigational proof search via structured semantic networks | ModelAtlas paradigm applied to theorem proving | Dense retrieval (ReProver), token classification, MCTS |
| 2026-03-06 | 6-bank ternary navigation (729 direction bins) | Resolves many-to-one tactic mapping from 3-bank (27 bins) | 3-bank (fallback available), continuous directions |
| 2026-03-06 | UW-SO adaptive loss weighting | Handles different convergence rates across nav/anchor/progress/critic | Fixed coefficients, manual curriculum |
| 2026-03-06 | Soft critic targets (MSE, not BCE) | HTPS finding: binary critic worse than no critic | Binary critic (ablation only) |
| 2026-03-06 | Proof history via mean-pooled closed goals | LeanProgress: 61.8% -> 75.1% from history | No history (ablation only) |
| | Encoder selection: ____ | EXP-0.3 results | |
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
| | Proof network extraction | macpro (Xeon) | | CPU-only |
| | Anchor gap analysis | macpro (Xeon) | | CPU-only, iterative |
| | Encoder evaluation | laptop (M-series) | | MLX/Metal |
| | Navigation training | laptop (M-series) | | Metal, ~5000 steps |
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

*Ledger version: 1.0. Template created 2026-03-06. Skeleton awaiting experimental data.*
