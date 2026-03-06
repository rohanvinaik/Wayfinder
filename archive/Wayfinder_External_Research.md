# What existing theorem provers reveal for Wayfinder's design

**The navigational proof search concept behind Wayfinder — treating proof construction as movement through a structured mathematical space — finds both strong validation and critical gaps in the current landscape of neural theorem proving.** Eight major systems span the architecture space from 299M-parameter retrieval-augmented provers to 200B+ reasoning engines, yet none combines structured mathematical representations with proof search the way Wayfinder envisions. The field has converged on a few dominant patterns (autoregressive tactic generation, dense retrieval for premises, best-first or MCTS search) while leaving structured/navigational approaches almost entirely unexplored. This report maps how each system's technical choices inform Wayfinder's design across eight cross-cutting concerns.

---

## The architecture spectrum: from small retrievers to massive reasoning engines

The eight systems under investigation occupy dramatically different points in the design space, and understanding where each sits clarifies the tradeoffs Wayfinder must navigate.

**AlphaProof** (DeepMind, 2024) uses a **~3B parameter encoder-decoder transformer** — deliberately small — paired with AlphaZero-style AND/OR tree search. The encoder-decoder design exists for a specific reason: Lean proof states can be thousands of tokens long while individual tactics are short, so encoding the state once and sampling many tactics from the decoder via cross-attention maximizes throughput. The model serves dual duty as both policy (tactic generation) and value function (estimating remaining tactics to proof completion, not probability — because all proven states would have probability 1.0, destroying the training signal). AlphaProof's true power comes from **test-time reinforcement learning (TTRL)**: for each hard problem, it generates millions of variants via an LLM, then trains a specialist agent from scratch on this narrow curriculum, requiring **hundreds of TPU-days per IMO problem**.

**Aristotle** (Harmonic, 2025) takes the opposite scaling approach: a single **200B+ parameter transformer** serving as both policy and value function, distinguished only by a task token. Its search is Monte Carlo Graph Search (MCGS) — MCTS extended to hypergraphs with state equivalence classes that collapse the search space. What makes Aristotle architecturally distinctive is its **lemma-based informal reasoning pipeline**: it generates natural-language proof sketches, decomposes them into lemmas, formalizes each into Lean, and iterates with formal feedback. The model uses hidden chain-of-thought with a dynamically set thinking budget before each action. Test-time training (TTT) on search traces — lighter than AlphaProof's TTRL — provides problem-specific adaptation.

**DeepSeek-Prover-V2** introduces a two-model architecture: **DeepSeek-V3 (671B MoE, ~21B activated) for subgoal decomposition** and a **specialized 7B prover** for solving individual subgoals. The decomposition model produces sequences of `have` statements with `sorry` placeholders — effectively a proof blueprint — and the prover fills in each gap independently. This division of labor reflects a key insight: general-purpose LLMs can reason about proof structure but cannot reliably generate correct formal proofs, while small specialized models can handle individual proof steps but not complex multi-step reasoning.

**ReProver/LeanDojo** (NeurIPS 2023) is the most lightweight: a **ByT5-small (299M parameter) dual-encoder retriever** feeding into a ByT5 tactic generator, with simple best-first search. Total training: **120 GPU hours on a single A100**. Despite its simplicity, it established the standard pipeline that subsequent systems build on.

**Math.inc's Gauss** represents a fundamentally different paradigm: an **agentic autoformalization system** that interleaves natural language reasoning with tool use (Lean compiler, literature search, file management). It operates for hours autonomously, running thousands of concurrent agents. No technical paper exists, but its outputs — formalizing the strong Prime Number Theorem in 25,000 lines, sphere packing proofs in 200,000 lines — demonstrate a capability no other system approaches.

For Wayfinder, these architectures collectively suggest that the **encoder-decoder pattern with dual policy/value output** (AlphaProof's approach) offers the best efficiency for search-heavy systems, while **subgoal decomposition** (DeepSeek-Prover-V2) is the strongest approach for reducing proof complexity. The navigational metaphor could provide a third path: instead of decomposing into subgoals or searching exhaustively, Wayfinder could use structured mathematical representations to identify promising "directions" before committing to specific tactics.

---

## Premise selection: dense retrieval dominates but leaves clear gaps

How systems retrieve relevant lemmas from libraries of **~130,000 premises** (Lean's Mathlib) is perhaps the most directly relevant concern for Wayfinder's design.

**ReProver's dense retrieval** uses a ByT5-small encoder producing **1472-dimensional embeddings** via average pooling, with cosine similarity for ranking. ByT5 was chosen specifically because its byte-level operation handles Lean's Unicode symbols (ℕ, ⊢, →, ∀) without tokenization artifacts. The retriever trains with an MSE-based contrastive loss (regressing cosine similarity toward 0 or 1) rather than the standard InfoNCE loss, with **3 negatives per example** including 1 hard negative from the same source file. A critical architectural decision: retrieval is restricted to **accessible premises only** (~33,160 on average vs. ~130,262 total), using Lean's import structure to filter. This alone provides a **~2% recall improvement**.

The documented failure modes are instructive for Wayfinder. On premises never seen during training, ReProver's recall@10 drops from **38.4% to 27.6%** — a 30% degradation. Theorem proving success nearly halves from 51.2% to 26.3%. Dense retrieval inherently struggles with premises that are semantically important but syntactically distant from the goal state. The byte-level embedding space must encode all mathematical relationships, which is fundamentally limiting for distant analogies.

**Magnushammer** (ICLR 2024) adds a two-stage pipeline: SELECT (dense dual-encoder, top-1024 from 433K premises) followed by RERANK (cross-encoder scoring each state-premise pair). This is more expensive per query but produces substantially better rankings, achieving **59.5% proof rate on PISA** vs. Sledgehammer's 38.3%. The key insight: contrastive fine-tuning provides enormous gains over generic embeddings (OpenAI ada-002 produced "reasonable performance comparable to Sledgehammer" but far below Magnushammer).

**LeanHammer** (2025) demonstrates the power of hybrid approaches: combining neural premise selection (LeanPremise) with symbolic methods (MePo relevance filter) and routing to both proof search (Aesop) and ATP translation (Zipperposition). This hybrid achieves **37.3% cumulative proof rate** on Mathlib, outperforming any single method by 21%. The DeepMath result from 2016 already showed this pattern: the union of neural (def-CNN) and k-NN methods proved 74.25% of Mizar theorems vs. either method alone.

**A recent GNN-augmented approach** (arXiv 2510.23637) augments ReProver's ByT5 embeddings with a Relational Graph Convolutional Network that propagates structural information from Lean's dependency graph, and **Tree-Based Premise Selection** (NeurIPS 2025) proposes entirely training-free selection using Weisfeiler-Lehman kernels and tree edit distance on Lean expression trees.

Neither AlphaProof, Aristotle, nor DeepSeek-Prover-V2 use explicit premise retrieval — they rely on their large models' **implicit knowledge** of the mathematical library. This works at scale (200B+/671B parameters) but is infeasible for smaller models.

For Wayfinder's navigational approach, the critical finding is that **no system uses structured/symbolic retrieval based on a mathematical knowledge graph for premise selection**. Dense retrieval dominates, hybrid neural+symbolic is strongest, but graph-based retrieval — traversing dependency structures or activating related concepts through a proof network — remains unexplored. This is precisely the gap Wayfinder could fill: a premise selection mechanism that navigates structured mathematical space rather than searching an embedding space.

---

## Proof search: best-first search surprisingly competitive with MCTS

The proof search algorithm determines how systems explore the space of possible proofs, and recent results challenge assumptions about which approach is best.

**HTPS** (Lample et al., NeurIPS 2022) adapts MCTS for theorem proving's AND-OR hypergraph structure. When a tactic generates multiple subgoals, ALL must be proved (AND semantics); different tactics for the same goal are alternatives (OR semantics). The critic network shares weights with the policy (both are a 600M encoder-decoder transformer) and predicts provability by restricting decoder output to two tokens: `PROVABLE` and `UNPROVABLE`. Value backpropagation through AND nodes uses the **product of children's values**, interpreting them as independent solvability probabilities. PUCT selection with progressive widening handles the continuous action space. A crucial finding: **soft critic targets** (using search estimates for internal nodes) dramatically outperform hard targets (binary 0/1), achieving 78.1% vs. 63.1% on the Equations benchmark. The hard critic is too pessimistic — even worse than no critic at all.

**BFS-Prover** (ByteDance, Feb 2025) challenged MCTS dominance by achieving **72.95% on miniF2F-test** with plain best-first search plus three innovations: expert iteration with self-filtering (excluding already-solvable problems), DPO from Lean compiler feedback, and **length normalization** (score = Σ log p(aₜ|sₜ) / Lᵅ) to encourage deeper proofs. At fixed inference budget, BFS-Prover outperforms InternLM2.5-StepProver's value-guided search (65.9%) and DeepSeek-Prover-V1.5's MCTS (63.5%). This suggests that with a strong enough policy model, sophisticated search may be unnecessary.

**AlphaProof** uses AND/OR tree search with a specific innovation: backpropagating the **value of the hardest branch** rather than the product of branch values, which the authors found works better. Once a child of an AND node is proven, it is marked done and never revisited. Progressive widening explores diverse strategies on critical paths.

**Aristotle** extends this to full graph search with state equivalence classes — if two proof states have identical goal expressions, local contexts, and variable names, they are merged, transforming the hypertree into a hypergraph. It also introduces **disproof augmentation**: each single-goal state gets an additional transition corresponding to logical negation, allowing the search to allocate budget to pruning impossible branches.

For Wayfinder, the navigational metaphor suggests a fundamentally different search paradigm. Rather than tree/graph search with learned heuristics, proof search could work like **route planning**: identify the destination (theorem to prove), locate your current position (proof state in mathematical space), and plan a path through structured mathematical terrain. No existing system attempts this. The closest analogues are QEDCartographer's RL-learned value function (which learns proof-state values for A*/best-first search in Coq) and HTPS's critic, but both learn unstructured value estimates rather than structured spatial reasoning.

---

## Tactic prediction splits between generation and structured approaches

How systems predict the next proof step is where Wayfinder's "navigational directions" concept is most distinctive.

**Autoregressive generation dominates**: GPT-f, ReProver, HTPS, DeepSeek-Prover, Aristotle, and AlphaProof all generate tactics token-by-token from language models. The advantage is flexibility — the model can produce arbitrary tactic strings including novel compositions. The disadvantage is speed (sequential decoding), potential for hallucinated lemma names, and no structural guarantees about tactic validity.

**Classification-based approaches** from the earlier era are revealing. **TacticToe** (HOL4) uses the most explicitly "navigational" pattern: it abstracts tactics to templates (`apply <thm>`, `rewrite <thm>`, `induction <var>`) — roughly 100 base tactics — then separately predicts arguments. The paper found that **argument prediction for abstracted tactics has the highest impact** on success. **Graph2Tac** (Coq, ICML 2024) uses GNN-based classification over ~100 tactic categories plus separate argument prediction, solving 26.1% of held-out Coq theorems. **Proverbot9001** uses feed-forward classification for tactic names and RNN prediction for arguments.

**Structured prediction** has been explored by ASTactic, which generates tactics as programs using an RNN controlled by a context-free grammar ensuring syntactic validity. This approach produces only valid tactic syntax but sacrifices the flexibility of free generation.

The work most relevant to Wayfinder's direction-prediction concept is **activation steering for tactic prediction** (arXiv 2502.15507), which constructs steering vectors corresponding to different reasoning strategies and applies them to InternLM2 and Llemma. The key finding: "The LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately." Steering toward the right reasoning category improves tactic selection. **Proof transformation prediction** (Kaliszyk et al., FroCoS 2023) predicts which tactics achieve desired state transformations, guessing the correct tactic 74% of the time given before-and-after states.

For Wayfinder, this suggests a two-level architecture: first predict the **type of proof move** (simplify, case split, apply known result, unfold definition — the "direction"), then fill in specific arguments. No existing system cleanly separates these levels. TacticToe's abstraction scheme is the closest precedent, and its finding about argument prediction's importance validates the decomposition.

---

## Encoders: byte-level models dominate, math-native architectures absent

The encoder question is critical for Wayfinder because its proof network requires embedding mathematical entities in a structured space.

**ByT5-small** (byte-level T5, 299M parameters) is the de facto standard for Lean, used by ReProver, LeanAgent, and Lean Copilot. Its byte-level operation (no tokenizer) handles Unicode mathematical notation naturally, producing **1472-dimensional embeddings** via average pooling. Training requires only 120 GPU hours. However, ByT5 was designed as a generic model — it has no mathematical pretraining.

**No math-native encoder model exists** for formal theorem proving. The closest approaches are: a Lean-specific retrained tokenizer (arXiv 2501.13959) achieving 30.74% vs. ReProver's 28.28% on miniF2F; PACT's multi-task self-supervised training on Lean proofs; and Llemma's continued pretraining of Code Llama on math corpora (though Llemma is a decoder, not an encoder). Modern embedding models (E5, GTE, BGE) have **never been tested** for formal math premise selection.

**Graph Neural Networks** offer an alternative encoding paradigm. Graph2Tac (IBM, ICML 2024) uses GNNs on Coq proof-state graphs, and Paliwal et al. (AAAI 2020) demonstrated GNNs for HOL proof search. The GNN-augmented ReProver (arXiv 2510.23637) layers an RGCN over ByT5 embeddings, propagating structural information from Lean's dependency graph.

On aggressive compression: **quantization to 4-bit introduces up to 32% accuracy degradation** on math benchmarks, but fine-tuning on just **545 task-specific examples for 3 minutes** can recover full performance. A 4-bit quantized 14B model still outperforms its dense 7B counterpart. **BitNet b1.58** (ternary weights, 1.58 bits/parameter) at 2B parameters matches full-precision models on mathematical reasoning benchmarks, with 2.7× speedup and 3.5× memory reduction — but **no one has applied ternary architectures to formal theorem proving**. This represents a clear opportunity for Wayfinder: a ternary encoder could be extremely fast for real-time premise retrieval while retaining mathematical capability.

---

## Progress estimation and value networks remain underdeveloped

Predicting "how far" a proof state is from completion — the essence of Wayfinder's navigational distance — has surprisingly few implementations.

**LeanProgress** (2025) trains a fine-tuned **DeepSeek Coder 1.3B** to predict the exact number of remaining proof steps. Including proof history boosts accuracy from 61.8% to **75.1%**. Integration into best-first search uses a weighted combination: C(sᵢ) = α·N(sᵢ) + (1−α)·P(sᵢ), where N is normalized predicted remaining steps and P is cumulative log-probability, with α=0.2. This yields a modest **3.8% improvement** in proof rate on Mathlib4 — meaningful but not transformative.

**HTPS's critic** estimates binary provability rather than distance, using the elegant trick of restricting a decoder's vocabulary to two tokens (PROVABLE/UNPROVABLE). **QEDCartographer** (ICSE 2025) learns state values via reward-free RL, generalizing the Bellman equation to hyper-states. **GamePad** (2018) first proposed predicting remaining proof steps in Coq. **Crouse et al.** (2019) used GNNs for proof length prediction in first-order logic, noting it "provides a proxy for the complexity of a theorem."

**No system connects formal proof complexity theory to neural search heuristics.** This is a major research gap. Proof complexity measures (Frege proof length, resolution complexity) could theoretically provide principled distance metrics, but all existing approaches learn empirical estimates. For Wayfinder's navigational metaphor, this gap is both a challenge (no theoretical grounding for "distance" in proof space) and an opportunity (structured representations could enable more principled distance metrics than pure neural estimation).

---

## Mathematical knowledge graphs exist but nobody uses them for proving

The most striking gap for Wayfinder is that structured mathematical representations exist but are entirely disconnected from proof search.

**AutoMathKG** (2025) is a fully automated knowledge graph with **13,388 entities** (definitions, theorems, problems) and **29,459 directed edges** representing reference relationships. It uses SBERT embeddings and has been evaluated with KG embedding methods (TransE, DistMult, R-GCN). A separate **Lean 4 Mathlib knowledge graph** stored in Neo4j captures theorem-to-theorem dependencies with HAS_LINK_TO and EXTENDS_TO relationships. **MMLKG** builds a knowledge graph from the Mizar library using KOS/SKOS vocabularies.

Yet **not a single published system uses a knowledge graph to guide proof search or premise selection**. The integration stops at basic dependency filtering (ReProver's accessible premises, LeanHammer's module structure). No system implements **spreading activation** over mathematical knowledge structures — despite this being a natural fit for finding related lemmas by traversing dependency chains. No system attempts to **position mathematical entities in structured coordinate systems** beyond standard embedding spaces.

The LeanDojo infrastructure already extracts fine-grained dependency data from Mathlib. The GNN-augmented ReProver shows that layering structural information over embeddings helps. But the full potential of structured mathematical representations — where proof search becomes navigation through a typed, dependency-aware mathematical space — remains entirely unrealized. This is Wayfinder's most distinctive design opportunity.

---

## Training paradigms converge on expert iteration with curriculum shaping

How systems improve through training directly shapes what Wayfinder's learning loop should look like.

**Expert iteration** — alternating between proof search and training on discovered proofs — is now universal among competitive systems. HTPS pioneered online training (continuous model refresh every 5 minutes) over batch expert iteration, showing significantly faster convergence. AlphaProof scaled this to ~100 million formalized problems. Aristotle combines expert iteration with test-time training on individual problem search traces. DeepSeek-Prover-V2 uses GRPO (Group Relative Policy Optimization) with binary rewards from the Lean compiler, avoiding a separate critic model entirely.

**Curriculum learning** has two promising implementations. LeanAgent orders repositories by the count of easy theorems (using **eˢ** where S = proof steps as the complexity metric), training for exactly one epoch per repository. This achieves positive backward transfer — learning harder mathematics improves performance on easier topics. CuDIP (2025) partitions training data by difficulty levels with iterative DPO training. **STP (Self-play Theorem Prover)** generates conjectures at the "cusp of provability" where **47% of self-generated conjectures are successfully proved** vs. only 11.4% of real unproved statements, providing dramatically denser training signal.

A key observation across systems: **up to 98.5% of generated proofs during expert iteration are incorrect**, and remaining unproven theorems become exponentially harder. This is the saturation problem that self-play and curriculum approaches address. For Wayfinder, the navigational framework could provide a natural curriculum: order theorems by their distance in the proof network from already-proven results, creating a wavefront of learnability that expands outward through mathematical space.

---

## Conclusion: where Wayfinder fits in the design space

The landscape reveals several clear findings that should shape Wayfinder's architecture. **Encoder-decoder models with dual policy/value outputs** (AlphaProof pattern) offer the best efficiency-to-capability ratio for search-heavy systems. **Byte-level encoding** (ByT5) is critical for handling mathematical notation, but math-native encoders remain an open opportunity. **Hybrid neural+symbolic premise retrieval** consistently outperforms either approach alone, and **graph-augmented retrieval** (GNN over dependencies) is showing early promise. **Best-first search with a strong policy can match MCTS**, but value/critic networks provide consistent improvements when well-designed (soft targets, distance-based rather than binary prediction).

The most distinctive design opportunities for Wayfinder lie in three unexplored gaps. First, **no system navigates a structured mathematical knowledge graph during proof search** — all use either flat embedding retrieval or implicit LLM knowledge. Wayfinder's proof network is architecturally novel. Second, **no system predicts proof "directions" (tactic categories) as a separate step from specific tactic generation** — TacticToe's abstraction scheme is the closest precedent but was never developed into a navigation paradigm. Third, **ternary/1-bit encoder architectures** show competitive mathematical reasoning at dramatic efficiency gains but have never been applied to theorem proving, offering Wayfinder a potential path to real-time structured retrieval. The navigational metaphor — positioning proofs in structured space, predicting directions rather than specific steps, and using graph-based distance as a search heuristic — would be genuinely novel in a field that has otherwise converged on a narrow set of architectural patterns.