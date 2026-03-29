# Wayfinder Model Selection Plan
## Organized by SoM Slot + Phase 0.6 Encoder Expansion

## Architecture Note (2026-03-25)

This file applies to the **SoM controller stack**: the tuned original / first-order SoM and the planned second-order SoM, plus adjacent runtime/teacher components.

It does **not** apply to Dr. Ducky's core execution path. Dr. Ducky is a deterministic symbolic/programmatic layer with typed proof skeleton synthesis and Lean-backed validation; it has no LLM stage and does not depend on the narrative/teacher model pool described here.

Current project order:
- preserve the original / first-order SoM controller assets (`EXP-SOM-010`, `models/som_torch_v1/best.pt`) as the current control substrate,
- finish the hard-run data-generation benchmark,
- use the resulting residual/gap corpus to train the second-order SoM,
- keep Dr. Ducky as the deterministic 1.5th-order executor beneath that controller.

### Storage: /Volumes/STORAGE/wayfinder_models/
### STORAGE free: 477 GB

---

## Selection Principles

1. **Select models by SoM slot, not by generic math benchmark score.**
   - Execution and verification require verifier-facing, compile-aware models.
   - Recognition, planning, and temporal orchestration require structure-sensitive models.

2. **Respect the compiler boundary.**
   - Below the boundary (`ContextIR`, goal startability, `ActionIR`, executable selectors), prefer:
     - symbolic logic
     - small classifiers
     - code/math-native encoders
   - Above the boundary (template recognition, proof sketching, temporal orchestration), story /
     reasoning models can be useful.

3. **Teacher vs runtime must be explicit.**
   - Large story/reasoning models are primarily **offline teacher models**.
   - Runtime controllers and planners should be distilled into compact typed models.

4. **Do not feed raw Lean to narrative models by default.**
   - Their input should be symbolicized theorem/trace packets:
     - theorem statement / namespace
     - proof-history summary
     - `SubtaskIR` / trigger-profile summaries
     - startability / repair status
     - lane history / temporal state

5. **Classical AI memory is part of model selection.**
   - `strategy_memory.json` / k-line-like retrieval is a first-class orchestration component, not
     just an analysis artifact.

---

## Phase 0.6: PERCEPTION Slot — Encoder Expansion (~9 GB new downloads)

### Already Evaluated (15 models, in HF cache)
| Model | Size | Sep | Throughput | Backend |
|-------|------|-----|-----------|---------|
| all-MiniLM-L6-v2 (CURRENT) | 22M | 0.587 | 617 g/s | SentenceTransformer |
| kaiyuy/leandojo-byt5-small (TOP) | 300M | 0.623 | 21.9 g/s | T5 encoder |
| pplx-embed bidir-Qwen3 (#2) | 600M | 0.600 | 16.8 g/s | Decoder→bidir |
| + 12 others | — | — | — | — |

### New Candidates to Download
| # | Model | Size | Why | Backend |
|---|-------|------|-----|---------|
| 1 | jinaai/jina-embeddings-v5-text-small | ~150MB | Latest gen small embedder, Matryoshka dims | SentenceTransformer |
| 2 | nomic-ai/nomic-embed-text-v1.5 | ~550MB | 137M, rotary, available in Ollama | SentenceTransformer |
| 3 | jhu-clsp/ettin-encoder-1b | ~4GB | 1B encoder, code+math domain anchors | SentenceTransformer |
| 4 | LSX-UniWue/ModernGBERT_1B | ~4GB | Modern BERT arch, 1B, long context | SentenceTransformer |
| 5 | nomic-ai/nomic-embed-code | ~550MB | Code-specific embedding variant | SentenceTransformer |

**Download dir:** `/Volumes/STORAGE/wayfinder_models/phase0_encoders/`

---

## SoM Slots 2-4 Above The Compiler Boundary: Narrative / Orchestration Teacher Pool (~38 GB new, ~40 GB local)

These models have skills that map to theorem-level narrative construction, proof sketching, and
temporal orchestration. The core insight is the same as the research/design docs: narrative
converts Regime B → Regime A, but that conversion belongs **above** Lean executability.

Use these models first as:
- taxonomy auditors
- soft-label teachers
- proof-sketch teachers
- temporal-control teachers
- strategy-memory miners

Do **not** treat them as default direct executors.

### Local Models (downloaded from HuggingFace to STORAGE)
| # | Model | Size | Quirky Skill | Recommended Role |
|---|-------|------|-------------|-------------|
| 6 | **Qwen3-8B-Drama-Thinking** | 15GB | Drama narrative + chain-of-thought | Offline proof-story / sketch teacher over symbolicized theorem packets |
| 7 | **MPT-7B-StoryWriter** | 12GB | Structured long-form creative writing (65k ctx) | Template audit, long proof-history summarization, strategy-memory induction |
| 8 | **Cerebrum-1.0-7b** | 13GB | Brain-inspired cognitive architecture model | Experimental meta-cognitive teacher for orchestration; defer unless it beats simpler teachers |

### HuggingFace Downloads
| # | Model | Size | Quirky Skill | Recommended Role |
|---|-------|------|-------------|-------------|
| 9 | deepseek-ai/DeepSeek-Prover-V1.5-SFT | ~14GB | Lean 4 proof generation (SFT on proofs) | Formal-planning teacher that can bridge theorem narratives and Lean structure |
| 10 | deepseek-ai/deepseek-math-7b-rl | ~14GB | RL-shaped math reasoning steps | Reward-shaped planning / controller teacher |
| 11 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ~3GB | Distilled reasoning (tiny) | Best candidate for compact planning/temporal teacher and later distillation target |
| 12 | stabilityai/stable-code-3b | ~6GB | Code + proof-level-math (ModelAtlas anchor) | Boundary case: more suitable near execution/compiler tasks than narrative planning |
| 13 | thelamapi/next-270m | ~1GB | Multi-domain in 270M params | Tiny teacher baseline for ablation on symbolicized packets |

**Download dir:** `/Volumes/STORAGE/wayfinder_models/som_planning/`

---

## SoM Quirky Architectures (~0 GB new, reference local)

These have fundamentally different computational mechanisms.

| # | Model | Size | Architecture | SoM Angle |
|---|-------|------|-------------|-----------|
| 14 | **RWKV-7 World** (local) | 2.8GB | Linear attention (sqReLU, O(1) per step) | Good candidate for compact temporal-control runtime or teacher due to recurrent-state bias |
| 15 | **nomos-1** (local) | 114GB | Qwen3 MoE (32 heads, sparse experts) | Conceptual MoE reference only; too large for first-line experiments |
| 16 | **polymathic-ai/aion-base** (local) | 1.7GB | Scientific foundation model | Different training distribution (physics, chemistry, biology data). Cross-domain transfer test. |
| 17 | **polymathic-ai/walrus** (local) | 9.6GB | Scientific foundation model (larger) | Same thesis as aion-base, at scale. |

**Copy dir:** `/Volumes/STORAGE/wayfinder_models/local_refs/`

---

## SoM Slot 5: VERIFICATION — Censor/Discriminator

| # | Model | Size | Skill | SoM Mapping |
|---|-------|------|-------|-------------|
| 18 | deepseek-ai/deepseek-math-7b-base | ~14GB | Math-pretrained base | Reward model initialization for censor network. Hidden representations already encode mathematical validity. |

---

## Runtime Use Policy

### Runtime models
- PERCEPTION encoders
- lightweight template classifier
- small sketch predictor
- compact temporal controller
- executable selectors
- censor / verification heads

### Offline teacher models
- story / narrative models
- reasoning models used for sketch labels
- temporal-policy teachers
- strategy-memory / k-line induction models

### Hard rule
- No story/reasoning model is evaluated as a primary runtime executor until it demonstrates value
  on symbolicized teacher tasks and survives distillation into a compact runtime model.

---

## Ollama Models (already available, no download needed)

| Model | Ollama Name | Skill |
|-------|------------|-------|
| deepseek-math 7B | ima/deepseek-math | Math reasoning (can use via ollama API) |
| qwen2.5-math | mightykatun/qwen2.5-math | Math-specialized Qwen variant |
| deepseek-math-7b-rl | t1c/deepseek-math-7b-rl | RL-trained math (GGUF quantized) |
| nomic-embed-text | nomic-embed-text | Embedding (already in Ollama) |

---

## Download Priority

### Immediate (Phase 0.6 encoder eval): ~9 GB
1. jina-embeddings-v5-text-small
2. nomic-embed-text-v1.5
3. ettin-encoder-1b
4. ModernGBERT_1B
5. nomic-embed-code

### Phase 6 (Teacher-model experiments above compiler boundary): ~38 GB
6. DeepSeek-Prover-V1.5-SFT
7. deepseek-math-7b-rl
8. DeepSeek-R1-Distill-Qwen-1.5B
9. stable-code-3b
10. next-270m

### Deferred (if needed): ~14 GB
11. deepseek-math-7b-base

**Total new downloads: ~61 GB** (477 GB available)

---

## Required Data Products For Model Selection

- `data/template_narrative_train.jsonl`
  - theorem statement / namespace
  - proof-history summary
  - theorem-level `SubtaskIR` / trigger-profile summaries
  - startability / repair status
- `data/temporal_train.jsonl`
  - typed `TemporalState` snapshots
  - lane history / family history
  - Lean-backed outcomes
- `data/strategy_memory.json`
  - k-line-like strategy entries mined from successful traces
- `data/apply_exec_dataset*.jsonl`
  - executable-selector supervision remains separate and verifier-facing

Teacher models should be evaluated on these symbolicized packets, not on raw Lean theorem scripts.

---

## Key SoM Skill Mapping

```
Slot 1 PERCEPTION:  Encoder eval models (Phase 0.6)
                    → Which embedding space best separates Lean goal states?

Slot 2 RECOGNITION: Runtime classifier over typed features
                    + optional teacher models for taxonomy audit / soft labels

Slot 3 PLANNING:    Offline narrative / reasoning teachers over symbolicized packets
                    Drama-Thinking → proof narratives (teacher)
                    StoryWriter → structured proof story frames (teacher)
                    DeepSeek-Prover / R1-Distill → formal sketch teachers
                    Runtime target: distilled small sketch predictor

Temporal Control:   Runtime compact controller over typed traces
                    + optional teacher models over `temporal_train.jsonl`
                    RWKV-7 → plausible compact runtime / teacher candidate
                    k-line strategy memory → symbolic prior, not a language model

Slot 4 EXECUTION:   Specialist executors / selectors
                    stable-code-3b → boundary case near compiler/execution
                    math/code-native models only where verifier-facing structure matters

Slot 5 VERIFICATION: Censor discriminator
                    deepseek-math-base → reward model initialization
```
