# Wayfinder Model Selection Plan
## Organized by SoM Slot + Phase 0.6 Encoder Expansion

### Storage: /Volumes/STORAGE/wayfinder_models/
### STORAGE free: 477 GB

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

## SoM Slot 3: PLANNING — Quirky Skills (~38 GB new, ~40 GB local)

These models have *skills* that map to proof narrative construction.
The core insight: "Narrative converts Regime B → Regime A" (spec_complexity_som.md §Narrative).

### Local Models (downloaded from HuggingFace to STORAGE)
| # | Model | Size | Quirky Skill | SoM Mapping |
|---|-------|------|-------------|-------------|
| 6 | **Qwen3-8B-Drama-Thinking** | 15GB | Drama narrative + chain-of-thought | Story template instantiation — generates proof narratives while reasoning. The "thinking" mode maps directly to proof sketch elaboration. |
| 7 | **MPT-7B-StoryWriter** | 12GB | Structured long-form creative writing (65k ctx) | Story frame composition — Winston's Strong Story Hypothesis applied to proofs. Trained on structured narrative, not math. |
| 8 | **Cerebrum-1.0-7b** | 13GB | Brain-inspired cognitive architecture model | Meta-cognitive proof strategy — cognitive model could reason about *which approach* to take, not just execute tactics. |

### HuggingFace Downloads
| # | Model | Size | Quirky Skill | SoM Mapping |
|---|-------|------|-------------|-------------|
| 9 | deepseek-ai/DeepSeek-Prover-V1.5-SFT | ~14GB | Lean 4 proof generation (SFT on proofs) | Direct proof sketch generation — actually knows tactic syntax, proof structure. |
| 10 | deepseek-ai/deepseek-math-7b-rl | ~14GB | RL-shaped math reasoning steps | Reward-shaped planning — RL training creates implicit value function for proof steps. |
| 11 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ~3GB | Distilled reasoning (tiny!) | Lightweight sketch predictor backbone — reasoning in 1.5B params. |
| 12 | stabilityai/stable-code-3b | ~6GB | Code + proof-level-math (ModelAtlas anchor) | Formal language understanding — code-native representations for Lean syntax. |
| 13 | thelamapi/next-270m | ~1GB | Multi-domain in 270M params | Tiny multi-skill baseline — can it do anything useful at 270M? SoM test of minimal component. |

**Download dir:** `/Volumes/STORAGE/wayfinder_models/som_planning/`

---

## SoM Quirky Architectures (~0 GB new, reference local)

These have fundamentally different computational mechanisms.

| # | Model | Size | Architecture | SoM Angle |
|---|-------|------|-------------|-----------|
| 14 | **RWKV-7 World** (local) | 2.8GB | Linear attention (sqReLU, O(1) per step) | Sequential proof search with constant memory. Different inductive bias from transformer — recurrent state accumulation vs attention. Could be specialist Slot 4 backbone. |
| 15 | **nomos-1** (local) | 114GB | Qwen3 MoE (32 heads, sparse experts) | Literal mixture of experts — MoE routing IS specialist selection. The gating network decides which expert handles each token. |
| 16 | **polymathic-ai/aion-base** (local) | 1.7GB | Scientific foundation model | Different training distribution (physics, chemistry, biology data). Cross-domain transfer test. |
| 17 | **polymathic-ai/walrus** (local) | 9.6GB | Scientific foundation model (larger) | Same thesis as aion-base, at scale. |

**Copy dir:** `/Volumes/STORAGE/wayfinder_models/local_refs/`

---

## SoM Slot 5: VERIFICATION — Censor/Discriminator

| # | Model | Size | Skill | SoM Mapping |
|---|-------|------|-------|-------------|
| 18 | deepseek-ai/deepseek-math-7b-base | ~14GB | Math-pretrained base | Reward model initialization for censor network. Hidden representations already encode mathematical validity. |

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

### Phase 6 (SoM planning): ~38 GB
6. DeepSeek-Prover-V1.5-SFT
7. deepseek-math-7b-rl
8. DeepSeek-R1-Distill-Qwen-1.5B
9. stable-code-3b
10. next-270m

### Deferred (if needed): ~14 GB
11. deepseek-math-7b-base

**Total new downloads: ~61 GB** (477 GB available)

---

## Key SoM Skill Mapping

```
Slot 1 PERCEPTION:  Encoder eval models (Phase 0.6)
                    → Which embedding space best separates Lean goal states?

Slot 2 RECOGNITION: Template classifier (MLP, no pre-trained model needed)
                    → Current 9-class template taxonomy

Slot 3 PLANNING:    ★ QUIRKY SKILLS ★
                    Drama-Thinking → proof narratives (Regime B→A conversion)
                    StoryWriter → structured proof story frames
                    Cerebrum → meta-cognitive strategy selection
                    DeepSeek-Prover → actual Lean proof generation
                    deepseek-math-rl → reward-shaped step planning
                    R1-Distill-1.5B → tiny reasoning for sketches

Slot 4 EXECUTION:   Specialist navigators (custom ternary arch)
                    RWKV-7 → linear attention specialist (alternative backbone)
                    nomos-1 → MoE routing as specialist selection
                    stable-code-3b → code-native representations

Slot 5 VERIFICATION: Censor discriminator
                    deepseek-math-base → reward model initialization
```
