# Wayfinder: Gap Coverage & Experimental Hardening Plan

**Goal**: Get the project from current state (Balanced Sashimi codebase + Wayfinder design docs) to a ready-to-run experiment.

**Timeline**: ~16 working days across 7 sprints.

**Assumptions**:
- Lean tooling pipeline is functional (user-confirmed)
- Novel components are synergistic and prototyped — build together, not minimal-first

---

## Hardware & Distribution Strategy

Adapted from the proven ModelAtlas pipeline pattern: standalone workers, JSONL interchange, `scp`-deployable, `--resume` support.

| Machine | Role | Specs | Wayfinder Use |
|---------|------|-------|---------------|
| **MLX laptop** | Primary dev + training | Apple Silicon M-series, MLX/Metal | Encoder eval, training, proof search, benchmarks |
| **macpro** | Persistent CPU worker | Xeon 6C 3.5GHz, 12GB ECC, Ollama | LeanDojo extraction, proof network construction, anchor gap analysis, data conversion |
| **homebridge** | Persistent CPU worker | i7 4C 2.6GHz, 16GB | Parallel extraction shards, accessible-premises crawl |

**Distribution pattern** (same as ModelAtlas Phase C):
- CPU-heavy extraction/analysis scripts are standalone workers with zero Wayfinder imports
- All data exchange via JSONL (one JSON object per line, UTF-8)
- Workers support `--resume` (read output, build skip set, append)
- File sync via **parsync** (parallel SSH, resumable, chunked) instead of `scp` — significant speedup for proof_network.db and training data shards
- Merge results back into the main DB on the laptop

**Key parallelism**: While the laptop runs encoder evaluation (Sprint 2b, GPU-bound), macpro and homebridge run proof network extraction and anchor gap analysis (Sprints 1-2a, CPU-only). Training never blocks on data preparation.

---

## Current State

**What exists** (~3,600 lines, 22 src files):
- `ternary_decoder.py` — TernaryLinear, STE quantization (reuse for ProofNavigator)
- `bridge.py` — InformationBridge (reuse, extend for proof history)
- `encoder.py` — GoalEncoder wrapping MiniLM (replace with chosen encoder)
- `goal_analyzer.py` — Feature extraction (extend with bank/anchor heads)
- `losses.py` — UW-SO adaptive loss (reuse for navigational loss)
- `pab_tracker.py` — PAB tracking (extend with navigational metrics)
- `trainer.py` + trainer_*.py — Full training loop (reuse with new step logic)
- `lowering.py` — Deterministic tactic lowering (reuse)
- `contracts.py` — Data contracts (extend with NavigationalExample, StructuredQuery)
- `data.py`, `evaluate.py`, `verification.py`, `domain_gate.py`, `pab_profile.py`, `pab_metrics.py`, `behavioral_fingerprint.py` — All production quality, reusable

**What's missing** (the gap):
- `src/proof_network.py` — SQLite semantic network (the core data structure)
- `src/proof_navigator.py` — 6-bank ternary navigational decoder
- `src/resolution.py` — Nav output → tactics + premises via proof network
- `src/proof_search.py` — Outer search loop with Lean verification
- `src/lean_interface.py` — Pantograph-based Lean kernel interaction
- `scripts/extract_proof_network.py` — Populate proof network from LeanDojo
- `scripts/build_nav_training_data.py` — Convert proofs to navigational examples
- `scripts/anchor_gap_analysis.py` — Iterative recall validation
- `scripts/train_navigator.py` — Training script with curriculum
- `scripts/eval_retrieval.py` — Nav vs dense retrieval comparison
- `scripts/eval_spreading.py` — Spreading activation evaluation
- `scripts/run_benchmark.py` — MiniF2F + Mathlib benchmark runner
- `configs/wayfinder.yaml` — Full Wayfinder configuration
- Tests for all new modules

---

## Sprint 1: Foundation (Days 1-3)
### Proof Network + Data Pipeline

The proof network is the architectural foundation. Everything else queries it.

**Hardware distribution**:
- **Laptop**: Write `proof_network.py`, `contracts.py` extensions, schema design
- **macpro**: Run LeanDojo extraction (CPU-heavy, ~2hrs), populate proof network
- **homebridge**: Run accessible-premises crawl in parallel with macpro's main extraction
- Extraction scripts are standalone workers (ModelAtlas pattern): JSONL in/out, zero Wayfinder imports, `--resume` support

### 1a. `src/proof_network.py` — Schema + Query API

Adapt ModelAtlas's `db.py` for mathematical entities.

```
Tables:
  entities(id, name, type, namespace, file_path)
  entity_positions(entity_id, bank, sign, depth)  -- 6 rows per entity
  anchors(id, label, category)
  entity_anchors(entity_id, anchor_id, confidence)
  entity_links(source_id, target_id, relation, weight)
  anchor_idf(anchor_id, idf_value)
  accessible_premises(theorem_id, premise_id)

API:
  navigate(conn, query: StructuredQuery, limit: int) -> list[ScoredEntity]
  spread(conn, seed_ids: list[int], hops: int) -> dict[int, float]
  compose_bank_scores(positions, query_positions, mechanism: str) -> float
  get_accessible_premises(conn, theorem_id: int) -> set[int]
```

**Scoring mechanisms** (configurable via `mechanism` param):
- `confidence_weighted`: Π score_i^(confidence_i) — default
- `multiplicative`: Π score_i — pure, fragile
- `soft_floor`: Π max(score_i, 0.1) — prevents zeroing
- `log_additive`: Σ log(score_i) — additive in log space

**Validation**: Unit tests for insert, navigate, spread, all 4 scoring mechanisms.

### 1b. `scripts/extract_proof_network.py` — Standalone Worker

Follows ModelAtlas worker pattern: standalone script, reads JSONL (LeanDojo export), writes JSONL (entity records), zero Wayfinder imports, `--resume` support.

For each of ~98k Mathlib theorems:
1. Parse type → STRUCTURE position
2. Namespace → DOMAIN position
3. Proof length → DEPTH position
4. Tactic analysis → AUTOMATION position + automation anchors
5. Context (hypothesis count) → CONTEXT position
6. Proof structure (have/cases count) → DECOMPOSITION position
7. Assign anchors from type content, tactic usage, namespace
8. Extract accessible premises from import graph

**Distribution**: Shard the LeanDojo export into 2 JSONL files, run one on macpro and one on homebridge. Merge results on laptop.

**Stop/go**: ≥95% bank coverage, ≥3 anchors/entity avg, ground-truth premises ⊂ accessible set.

### 1c. `scripts/build_nav_training_data.py`

Convert each proof step to a NavigationalExample:
- Goal state text → input
- Tactic → 6-bank direction vector (using mapping from DESIGN.md §5.1)
- Goal + tactic + premises → ground-truth anchors
- Remaining steps → progress label (for soft critic target)
- Closed goals so far → proof history

**Runs on**: macpro (CPU-only, reads proof_network.db)

**Stop/go**: <10% unmapped tactics. Direction consistency for similar tactics.

### 1d. Extend `src/contracts.py`

Add:
```python
@dataclass
class NavigationalExample:
    goal_state: str
    theorem_id: str
    nav_directions: dict[str, int]    # bank_name -> {-1, 0, +1}
    ground_truth_anchors: list[str]
    ground_truth_tactic: str
    ground_truth_premises: list[str]
    remaining_steps: int
    proof_history: list[str]          # previously closed goal states

@dataclass
class StructuredQuery:
    bank_positions: dict[str, tuple[int, int]]  # bank -> (sign, depth)
    anchor_ids: list[int]
    anchor_weights: list[float]
    accessible_theorem_id: int | None  # for premise filtering

@dataclass
class NavOutput:
    directions: dict[str, int]        # bank -> {-1, 0, +1}
    direction_logits: dict[str, Tensor]
    anchor_logits: Tensor
    progress: float
    critic_score: float
```

---

## Sprint 2: Anchor Gap Analysis + Encoder (Days 3-5)
### Validate retrieval quality, choose encoder

These two tasks are parallelizable across hardware.

**Hardware distribution**:
- **macpro**: Run anchor gap analysis iterations (CPU-only, reads proof_network.db)
- **Laptop**: Run encoder evaluation (GPU/Metal, model loading + embedding)
- Gap analysis iterations on macpro don't block encoder eval on laptop

### 2a. `scripts/anchor_gap_analysis.py` — Runs on macpro

Standalone worker. For 500 random proof steps:
1. Build "perfect" query from ground-truth (correct bank positions + correct anchors)
2. `navigate(proof_network, perfect_query, limit=16)`
3. Is ground-truth premise in top-16?
4. For each miss: what anchors *would have* connected goal to premise?
5. Cluster gap anchors, add top clusters, re-populate, re-run

**Iterate until**: Top-16 recall ≥ 70%.
**Escape hatch**: If recall plateaus <50% after 3 iterations, the bank positioning scheme needs rethinking, not just anchors.

### 2b. Encoder Evaluation — Runs on laptop (MLX/Metal)

Two-stage evaluation: frozen embedding quality, then LoRA fine-tuned quality.

**Stage 1 — Frozen embeddings** (1000 Mathlib goal states):
1. Premise clustering: do goals with shared premises cluster?
2. Alpha-equivalence: do renamed-variable goals stay close?
3. Domain separation: algebra vs topology separable?

**Stage 2 — LoRA fine-tuning via mlx-tune** (top 1-2 candidates from Stage 1):
- Fine-tune with LoRA/QLoRA on Mathlib goal-state premise-clustering objective
- mlx-tune provides Unsloth-compatible API on Apple Silicon, no cloud GPU needed
- Unified memory handles models that would OOM on discrete VRAM
- Re-evaluate premise clustering after fine-tuning
- Export best model via GGUF for potential Ollama deployment on macpro/homebridge

**Priority order**:
1. ByT5-small (proven Lean standard, byte-level) — baseline, frozen + LoRA fine-tuned
2. One math-native candidate (Qwen 3.5 or DeepSeek-Math embedding mode) — frozen + LoRA
3. BitNet ternary (if architecturally interesting results from 1-2)
4. Aggressive pruning of 7B model + LoRA fine-tune (unified memory makes this practical)

**Decision**: Pick the encoder with best post-fine-tuning premise-clustering. Default to ByT5-small + LoRA if evaluation is inconclusive.

### 2c. Update `src/encoder.py`

Replace MiniLM wrapper with chosen encoder. Ensure:
- Byte-level handling for Lean Unicode (∀, ⊢, →, ℕ)
- Output dimension compatible with GoalAnalyzer
- `byte_level: true` config flag

---

## Sprint 3: Core Neural Pipeline (Days 5-8)
### Full forward+backward pass

### 3a. Extend `src/goal_analyzer.py`

Add to existing GoalAnalyzer:
```python
self.bank_heads: nn.ModuleDict  # 6 heads, each Linear(feat_dim, 3)
self.anchor_head: nn.Linear(feat_dim, num_anchors)  # sigmoid

# forward() returns:
#   features, bank_logits: dict[str, Tensor], anchor_logits: Tensor
```

### 3b. `src/proof_navigator.py` — 6-Bank Ternary Decoder

The core new module. Reuses TernaryLinear from `ternary_decoder.py`.

```python
class ProofNavigator(nn.Module):
    # Hidden: TernaryLinear layers (STE quantization)
    # Output heads:
    #   6 direction heads: TernaryLinear(hidden, 3) each
    #   anchor_head: Linear(hidden, num_anchors)
    #   progress_head: Linear(hidden, 1)
    #   critic_head: Linear(hidden, 1)
    #
    # navigable_banks configurable (default: all 6)
    # Graceful degradation to 3-bank via config
```

### 3c. `src/resolution.py` — Nav Output → Tactics + Premises

```python
def resolve(nav_output: NavOutput, conn, context: SearchContext) -> list[Candidate]:
    query = build_query(nav_output)
    if context.accessible_theorem_id:
        query.accessible_theorem_id = context.accessible_theorem_id
    tactics = navigate_tactics(conn, query)
    premises = navigate_premises(conn, query)
    spread_scores = spread(conn, context.seed_ids)
    return combine_and_rank(tactics, premises, spread_scores)
```

### 3d. Proof History Module

Mean-pool embeddings of previously closed goal states. Concatenate to bridge input.

### 3e. Update `src/bridge.py`

Accept optional proof history tensor. When present, concatenate to compressed representation before passing to navigator.

### 3f. Integration Smoke Test

End-to-end: goal_state → encoder → analyzer → bridge(+history) → navigator → resolution → tactic name.
- Forward pass completes for batch of 4
- Backward pass produces gradients on all learnable parameters
- Resolution returns real Mathlib tactic names
- Progress head outputs reasonable scalar
- Critic head outputs probability in [0, 1]

**Stop/go**: If gradients don't flow through all components, debug before Sprint 4.

---

## Sprint 4: Training Infrastructure (Days 8-11)
### Train the navigator on real data

### 4a. Navigational Loss Function

```python
L = L_nav + L_anchor + L_progress + L_critic  (UW-SO weighted)

L_nav: CrossEntropy on each bank direction head (6 terms)
L_anchor: BCE on multi-label anchor prediction
L_progress: MSE on normalized remaining steps (SOFT target)
L_critic: MSE on normalized distance-to-completion (SOFT target, NOT binary BCE)
```

The UW-SO weighting from `losses.py` is reusable — just add navigational loss terms.

### 4b. Extend `src/pab_tracker.py`

Add navigational metrics:
- Per-bank accuracy (6 curves)
- Anchor prediction F1
- Progress MAE (mean absolute error on remaining steps)
- Critic correlation (Spearman ρ)
- Ternary crystallization rate (% stable weight signs)
- Navigational consistency (similar goals → similar directions)

### 4c. `scripts/train_navigator.py`

Training loop with curriculum:
- Phase A (steps 0-500): 1-2 step proofs only
- Phase B (steps 500-1500): ≤5 step proofs
- Phase C (steps 1500+): all proofs, oversampling medium difficulty

Uses existing Trainer infrastructure with new step logic.

### 4d. NavigationalDataset in `src/data.py`

DataLoader for `nav_train.jsonl` with:
- Goal state encoding (batched)
- Direction labels (6 per example)
- Anchor multi-label targets
- Progress targets
- Proof history (variable length, padded)

### 4e. `configs/wayfinder.yaml`

Full config covering all Wayfinder-specific parameters:
```yaml
encoder:
  type: byt5-small  # or chosen alternative
  byte_level: true
  freeze: false  # fine-tune

navigation:
  navigable_banks: [structure, domain, depth, automation, context, decomposition]
  scoring_mechanism: confidence_weighted
  num_anchors: 300  # grows with gap analysis

training:
  soft_critic_targets: true
  proof_history: true
  curriculum: [phase_a, phase_b, phase_c]

search:
  hammer_delegation: true
  accessible_premises: true
  search_budget: 600
  hammer_timeout: 30
```

### Sprint 4 Gate

Run 500-step training on Phase A data. Requirements:
- Loss decreases
- Nav accuracy > 33% (above random) and improving
- No NaN/Inf in any metric
- PAB tracker logging all navigational metrics

If this fails, debug training before continuing.

---

## Sprint 5: Proof Search + Lean Integration (Days 11-14)
### Close the loop

### 5a. `src/lean_interface.py`

Adapt from user's existing Lean tooling pipeline.

```python
class LeanKernel:
    def try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Send tactic to Lean kernel via Pantograph."""

    def try_hammer(self, goal_state: str, premises: list[str], timeout: int) -> TacticResult:
        """Delegate to LeanHammer/Aesop with premise suggestions."""
```

### 5b. `src/proof_search.py`

```python
def search(theorem, navigator, proof_network, lean_kernel, budget=600):
    open_goals = [initial_goal]
    context = SearchContext()

    while open_goals and context.attempts < budget:
        goal = select_goal(open_goals, navigator)  # critic + progress

        # Hammer delegation for decidable goals
        if nav_output.automation_direction == -1:
            result = lean_kernel.try_hammer(goal, premises[:16])
            if result.success: ...

        # Navigational resolution for everything else
        candidates = predict_and_resolve(goal, navigator, proof_network, context)
        for tactic, premises in candidates:
            result = lean_kernel.try_tactic(goal, lower(tactic, premises))
            ...

    return len(open_goals) == 0
```

### 5c. Evaluation Scripts

- `scripts/eval_retrieval.py` — Nav vs dense retrieval, recall@k comparison
- `scripts/eval_spreading.py` — Spreading activation benefit measurement
- Both produce structured JSON reports

### Sprint 5 Gate

Prove ≥5 trivial theorems (1-step proofs) end-to-end: goal → neural → resolution → Lean verification → QED.

---

## Sprint 6: Evaluation Harness + Core Tests (Days 14-16)

### 6a. `scripts/run_benchmark.py`

Run proof search on:
- MiniF2F-test (488 problems)
- Mathlib test split (~2k theorems)

Metrics: theorems proved, avg budget consumed, neural forward passes, wall-clock time.

### 6b. Core Test Suite

- `tests/test_proof_network.py` — Schema, queries, scoring, accessible premises
- `tests/test_navigator.py` — Forward/backward, ternary quantization, 6-bank output
- `tests/test_resolution.py` — Query building, tactic/premise retrieval
- `tests/test_search.py` — Search loop, hammer delegation, goal selection
- `tests/test_training.py` — Loss computation, curriculum, PAB metrics

### 6c. Primary Config Validation

Full pipeline validation with `configs/wayfinder.yaml`:
- Train 3000 steps → search 50 theorems → produce benchmark report
- Verify all invariants enforced at startup
- This is the "ready to run" gate

---

## Post-Readiness: Ablation Hardening (Days 18+)

Once the primary config is validated end-to-end:

### Ablation Config Generator

Generate 13 variant configs (from PLAN.md §4.4):
1. Full Wayfinder (primary — already validated)
2. Dense retrieval (no proof network)
3. Tactic classification (no navigation)
4. No spreading activation
5. No progress head
6. Continuous decoder (no ternary)
7. No IDF weighting
8. No bank alignment (anchors only)
9. Binary critic (BCE)
10. No proof history
11. No hammer delegation
12. No accessible-premises filter
13. 3-bank navigation

Each must load, train 10 steps, and produce evaluation metrics without errors.

### PAB Comparison Tooling

Trajectory comparison plots across ablation variants. Visual inspection of training curves to identify which components contribute most.

---

## Sprint 7: Integration & Hardening (Days 16-18)

### 7a. End-to-End Integration Test

Full pipeline: download data → populate network → gap analysis → train 500 steps → search 50 theorems → produce benchmark report. No manual intervention.

### 7b. Invariant Enforcement

Startup checks for all 11 invariants from PLAN.md:
- Eval data frozen
- Proof network gap-analyzed before training
- Single neural inference per proof state
- Scoring auditable
- PAB from step 0
- Scoring configurable
- Encoder explicitly chosen
- 6-bank default
- Soft critic targets
- Accessible premises on
- Hammer delegation on

### 7c. Reproducibility

Same seed → metrics within statistical tolerance across 2 runs. M-series Metal/MLX and PyTorch nondeterminism make bit-identical results unrealistic. Bar: navigational accuracy within ±1%, theorem-proved count identical, no qualitative divergence in PAB trajectory shapes.

### 7d. Compute Profile

- Training fits in Apple Silicon memory (M-series)
- Proof search <30s/theorem average on laptop
- CPU-bound extraction runs on macpro/homebridge without blocking laptop
- Full benchmark (488 MiniF2F) completes in <24h on laptop

---

## Definition of "Ready to Run"

The primary Wayfinder config works end-to-end:

- [ ] `proof_network.db` populated with ~98k entities
- [ ] Anchor gap analysis: top-16 recall ≥ 70% on perfect queries
- [ ] Encoder chosen and integrated
- [ ] Full pipeline forward+backward works
- [ ] Training runs 3000+ steps with decreasing loss, improving nav accuracy
- [ ] Proof search proves ≥5 trivial theorems end-to-end
- [ ] Benchmark runner produces metrics on MiniF2F subset
- [ ] Core tests pass (proof network, navigator, resolution, search, training)
- [ ] Reproducible within tolerance (same seed → metrics within ±1%)

**Post-readiness hardening** (Sprint 6+):
- [ ] All 13 ablation configs load, train, and evaluate
- [ ] PAB trajectory comparison tooling
- [ ] Full test coverage for edge cases

---

## Module Creation Order (respects dependencies + hardware parallelism)

```
         LAPTOP (MLX/Metal)              macpro/homebridge (CPU)
         ─────────────────               ──────────────────────
Day 1:   contracts.py extensions         (setup: LeanDojo install, data download)
Day 1:   proof_network.py (schema+API)
Day 2:   (review extraction results)     extract_proof_network.py (sharded)
Day 2:                                   build_nav_training_data.py
Day 3:   encoder evaluation ─────────┐   anchor_gap_analysis.py iterations
Day 4:   encoder evaluation          │   gap analysis iterations
Day 5:   encoder.py replacement ─────┘   (gap analysis complete, merge results)
Day 5:   goal_analyzer.py extensions
Day 6:   proof_navigator.py
Day 6:   resolution.py
Day 7:   bridge.py extensions + proof history
Day 7:   pipeline smoke test
Day 8:   navigational loss
Day 9:   pab_tracker.py extensions + train_navigator.py
Day 10:  data.py extensions + wayfinder.yaml
Day 11:  training gate (500 steps)
Day 11:  lean_interface.py (adapt existing tooling)
Day 12:  proof_search.py
Day 13:  eval_retrieval.py + eval_spreading.py
Day 14:  search gate (prove 5 theorems)
Day 15:  run_benchmark.py + core tests
Day 16:  integration test + profiling
         ─── READY TO RUN ───
Day 17+: ablation configs (post-readiness hardening)
Day 18+: PAB comparison tooling
```

---

## Tools & Extensions Evaluated

| Tool | Verdict | Relevance |
|------|---------|-----------|
| **mlx-tune** (ARahim3) | **Adopted for Sprint 2b** | Encoder LoRA fine-tuning on Apple Silicon. Unsloth-compatible API, QLoRA, GGUF export. Eliminates need for cloud GPU in encoder evaluation. |
| **mlx-vis** (hanxiao) | **Adopted for Sprint 2a + Phase 4** | GPU-accelerated dimensionality reduction (UMAP/t-SNE/PaCMAP) on Metal. 70K points in ~3s. Visualization backbone for proof network semantic space, anchor gap analysis, training dynamics, and paper figures. |
| **parsync** (AlpinDale) | **Adopted for distribution** | Parallel SSH file sync with resumability and chunking. Replaces `scp` in laptop ↔ macpro ↔ homebridge workflow. Native Apple Silicon support. |
| **SymTorch** (arXiv 2602.21307) | Phase 4+ analysis | Symbolic distillation of trained bank heads → interpretable equations. Post-training interpretability, not design-relevant now. |
| **Text-to-LoRA** (SakanaAI) | Phase 5+ exploration | Hypernetwork generates LoRA from text descriptions. Possible "navigational expert routing": pre-generate domain LoRAs (algebra, topology, etc.) selected by DOMAIN bank position. Conflicts with single-inference principle if done per-state; viable as pre-generated library. |
| **LocalCowork** (Liquid4All) | Not adopted | Standard MCP agent architecture. Tool-selection benchmarks interesting but Wayfinder doesn't do LLM-based tool routing. |
| **mlx-lm-gui** (stevenatkin) | Optional convenience | GUI for MLX training. Useful for encoder LoRA step; Wayfinder's custom training loop (UW-SO, PAB, curriculum) needs its own monitoring. |

---

## Communication Architecture

Informed by personal semiotics framework (Winston's communication theory). The paper and experimental output are designed for mechanistic clarity.

### Winston's Star for Wayfinder

| Element | Content |
|---------|---------|
| **Symbol** | Navigation through mathematical space — the compass/map |
| **Slogan** | "Navigate, don't predict" — or — "One inference, then symbolic search" |
| **Surprise** | Zero neural inference at retrieval time; a knowledge graph replaces dense embedding |
| **Salience** | The 6-bank coordinate system: three ternary digits locate a tactic family |
| **Story** | "We built a system for navigating ML models through semantic coordinates. What if mathematical proofs worked the same way?" |

### Fencing

- **IS**: Navigational proof search through structured semantic networks
- **IS NOT**: An LLM predicting tactic tokens
- **Differs from nearest thing** (ReProver): Neural embeddings → symbolic navigation from single-inference coordinate. Neural network runs once; everything after is a database query.

### Three Laws Applied

1. **Alignment precedes argument**: Open with the problem (proof search is expensive because every candidate requires neural inference), not the solution. Let the reader arrive at "what if we could do this symbolically?"
2. **Minimize surprise except in content**: Schema, SQLite, bank positions should feel familiar. The surprise: this works for theorem proving.
3. **The cheese contains the medicine**: mlx-vis visualizations ARE the argument. 98K theorems clustered by domain = thesis understood before it's stated.

### Assertion-Evidence Figure Convention

Every figure title is a claim, not a topic:
- "Bank positions separate mathematical domains without training" (not "Bank position distribution")
- "AUTOMATION and STRUCTURE banks crystallize before DOMAIN and CONTEXT" (not "Training dynamics")
- "Navigational retrieval matches dense retrieval at 1/50th the neural compute" (not "Retrieval comparison")

### Visualization Pipeline (mlx-vis)

| Experiment Phase | Visualization | Purpose |
|-----------------|---------------|---------|
| Sprint 2a (anchor gap) | UMAP of entity positions colored by retrieval success/failure | Show where gaps cluster — the visual argument for new anchors |
| Sprint 4 (training) | Animated PaCMAP of navigational embeddings across curriculum phases | Progressive revelation of structure emerging during training |
| Phase 4 (evaluation) | t-SNE of bank positions colored by domain/tactic family | Core paper figure: bank positions create meaningful structure |
| Phase 4 (ablation) | Side-by-side embeddings: full Wayfinder vs no-bank-alignment ablation | The visual fencing: what navigation adds vs anchors alone |

### Three-Stream Cycling

The three experimental streams cycle through one thesis from different angles:
- **Stream 1 (Navigation)**: The proof network enables retrieval without neural inference
- **Stream 2 (Architecture)**: Ternary coordinates generalize better than token indices
- **Stream 3 (PAB Process)**: Training dynamics reveal structured learning invisible to endpoint metrics

A reader who misses one angle catches it in another. Each stream feels like progression, not repetition.
