# Wayfinder

**Search for a Lean 4 proof by navigation, not next-token prediction — one neural pass per proof state, the rest deterministic arithmetic.**

<p align="center">
  <a href="https://github.com/rohanvinaik/Wayfinder/actions"><img src="https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3367d6.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3367d6.svg" alt="Python 3.11+"></a>
</p>

`1,277 / 2,000 Mathlib (raw) · one neural pass per state · deterministic search · Lean-verified · no embeddings at search`

A neural prover predicts the next tactic token by token, and it runs the network at every step of the search. Wayfinder does neither. It places lemmas, tactics, and proof states as coordinates in a structured semantic network, and it searches that network with deterministic arithmetic — IDF-weighted set intersection, Bellman-Ford spreading activation, multiplicative bank alignment. The neural network runs **once per proof state**, to place the state; everything after that is symbolic. The search is faster, it is auditable, and every step traces to a rule rather than a sampled guess.

On a frozen Mathlib test split of ~2,000 theorems, the published baseline (`EXP-058`) closes **1,277 — raw, no assist**, on Apple Silicon. A validated offline residual extension takes it to **~1,371**.

---

## The number is `raw_success`, and nothing is folded into it

A proof either type-checks or it does not, so a prover's headline is easy to inflate: fold in the ones a repair pass rescued, the ones a second system finished, the ones nothing independent re-checked. Wayfinder reports four lanes and never sums them:

| Lane | What closed the proof |
|------|-----------------------|
| **`raw_success`** | step-wise search alone — no assist. **The headline, always.** |
| `axle_assisted_success` | search made partial progress; a decompose/repair pass finished it |
| `axle_repair_only` | the repair pass alone closed the remaining goals |
| `lane_c_verified` | independently re-confirmed by `lean4checker` / a comparator / SafeVerify |

The `1,277 / 2,000` is `raw_success`. The other lanes are reported beside it and never added into it. An infrastructure miss — a worker that crashed, a timeout — is never counted as a model failure; the two are recorded on separate ledgers. The point of the discipline is that the number cannot drift upward by accident.

---

## One neural pass per proof state

A proof state goes in. A continuous encoder reads it, a learned information bottleneck compresses it, and a **ternary {−1, 0, +1} decoder** emits *directional coordinates in proof space* — not vocabulary indices. Those coordinates are orchestration signals: which lane to route to, which family of tactics to bias toward, which slice of the premise frontier to open. That is the whole job of the network, and it is done.

Everything downstream is arithmetic on structured data. Premise selection is set intersection over a shared anchor dictionary, weighted by rarity. Reachability is spreading activation over the network. The final local step is chosen by a deterministic residual executor, not a language model. The network does not run again until the next state — so search cost does not scale with proof depth the way a per-token prover's does, and every move it makes can be read back to the operation that produced it.

## The hard residual is compressed, not guessed

Most Mathlib goals fall to a cheap structural majority — the moves a deterministic pass can make without judgment. What is left is the residual: the goals where structure runs out. Wayfinder does not hand those to a bigger model and hope. It hands them to **Dr. Ducky**, a deterministic residual executor that emits typed capsules and proof skeletons and replays them Lean-side to validate progress before anything is claimed. What Dr. Ducky cannot close is handed onward as a *structured* residual — a smaller, cleaner problem — not as raw text.

That is the architecture's shape: a recursive proof-compiler. The first-order system removes the cheap structural majority; the next order learns over the residual the first one leaves; a higher order is justified only when it produces a strictly cleaner residual than the order below it. The residual training corpus is real and held out — 240,848 non-structural steps for training, 12,505 for evaluation.

---

## Why decomposition is safe — the composition gap

Splitting a monolithic prover into specialists is only sound if the pieces do not entangle. Specification-complexity theory gives the condition. Model each specialist by its specification complexity σ, and the composition gap theorem bounds the whole:

$$\sigma(A \circ B) \;\le\; \sigma(A) + \sigma(B) + \gamma(A, B).$$

When the specialists are independent, the cross term γ vanishes and complexity is **additive, not multiplicative** — so a system of small, stable specialists is provably no harder to specify than the sum of its parts. Wayfinder uses this to decompose the prover into typed temporal slots — PERCEPTION → RECOGNITION → PLANNING → ORCHESTRATION → EXECUTION → VERIFICATION — and it stops decomposing a slot once its process-stability proxy reads "stable." The move that makes this pay is narrative construction: it turns hard proof-structure prediction (low symmetry, exponential σ) into template classification (high symmetry, polynomial σ). Structure is not predicted. It is recognized.

---

## Run it

Wayfinder is an MCP server. Point it at a Lean goal and it navigates:

| Tool | What it does |
|------|--------------|
| `prove` | search for a proof of one Lean 4 goal, with the lane it closed in |
| `prove_corpus` | run the search over a whole eval set; `prove_corpus_status` reports progress |
| `send_to_aristotle` · `aristotle_result` | hand a goal to the external verifier lane and collect the checked result |

Everything the search does is arithmetic on a structured network plus one encoder pass per state — no dense retrieval index, no embedding store, no GPU required for the search itself.

## Lineage

Wayfinder integrates three earlier lines of work: the navigational semantic network of [ModelAtlas](https://github.com/rohanvinaik/ModelAtlas) (models as coordinates in a signed dimensional space), the hybrid continuous-ternary decoder of Balanced Sashimi ({−1, 0, +1} weights for categorical decisions), and the convergence guarantees of Mutation Theory, formalized in Lean 4. The target is Lean 4 proof generation over Mathlib, evaluated against ReProver (Lean-Dojo), LeanProgress, and DeepSeek-Prover-V2.

## Status

Publishable first-order theorem-search baseline (`EXP-058`, `1,277 / 2,000` raw on the Mathlib split, ~1,371 with the validated residual extension), a deterministic Dr. Ducky residual executor, and an integrated hard-tail bridge. The second-order Society-of-Mind program — learning over the first residual manifold — is the next measured lift, not the theoretical end of the family. Eval sets are frozen (`data/nav_eval.jsonl`, MiniF2F-test, the Mathlib test split); the full results ledger is in [`docs/Research_Paper/`](docs/Research_Paper/).

---

MIT — Rohan Vinaik. Proof search as navigation, verified in Lean 4.
