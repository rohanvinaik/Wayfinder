# Wayfinder

**Lean 4 proof search by navigation, not next-token prediction: deterministic search over a structured network of Mathlib, what it cannot close handed on as Lean with holes, and the holes closed by a small open-weights prover running in RAM.**

<p align="center">
  <a href="https://github.com/rohanvinaik/Wayfinder/actions"><img src="https://github.com/rohanvinaik/Wayfinder/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3367d6.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3367d6.svg" alt="Python 3.11+"></a>
</p>

`1,028 / 2,000 Mathlib at zero neural components · every close Lean-verified · the residual handed on as Lean · the last mile closed locally`

When a person proves something, they first work out what *kind* of statement it is, then how statements like that are proved. Novel theorems rarely need new mathematics; they need new combinations of tools that already exist. Wayfinder does the same thing mechanically. It reads a goal's structure — what it quantifies over, what domain it lives in, whether it splits, whether it is hammer-shaped — and searches a network of Mathlib's lemmas, tactics and proof states with deterministic arithmetic: IDF-weighted set intersection, spreading activation, bank alignment. Every step traces to a rule, and every close is checked by the Lean kernel.

A neural prover predicts the next tactic token by token and runs its network at every step. Wayfinder's deterministic search runs no network at all.

---

## The neural navigator measured zero, so it was taken out of the search

Wayfinder began with a learned navigator: a 22M-parameter encoder, an information bottleneck and a ternary {−1, 0, +1} decoder that placed each proof state on six signed banks. The 2026-08 re-founding ([`THE_REFOUNDING.md`](docs/Research_Paper/THE_REFOUNDING.md)) measured what it contributed to closures — **zero** (EXP-3.2, NAV-004) — and replaced it with the object it had been estimating: the banks are now *computed* structural reads of the goal, against norms mined from the corpus.

The deterministic path (`search_mode: deterministic`) never loads a checkpoint. On the frozen 2,000-theorem Mathlib split it closes **1,028 — raw = honest, 51.4%** — against **1,277** (63.8%, `EXP-058`) for the encoder stack it replaced: 80.5% of the solve count with no per-state inference, every close warranted and auditable. Counted as the union of every kernel-verified honest close across the loop's passes, it stands at **1,042 / 2,000** (`runs/loop_union.json`).

The encoder is demoted, not deleted: `search_mode: full` still loads NAV-004 as an ordering layer, and that mode — with the MiniLM cosine-rewrite lanes — is what the MCP `prove` tool runs today.

## The number is never inflated

A proof either type-checks or it does not, so a prover's headline is easy to inflate: fold in the ones a repair pass rescued, the ones a second system finished, the ones nothing re-checked. Wayfinder records `raw_success` and `honest_success` separately and never adds a lane into either. The loop's official count is the union of kernel-verified honest closes — a checked proof stays checked — and re-measurements of a standing bank wobble by up to ±3 under any ordering change, so a single-run delta inside that band is noise, not signal. An infrastructure miss (a crashed replay, a timeout) is ledgered apart from a model failure: the 1,028 run had 201 of them, none counted against it.

---

## What the search cannot close is handed on as Lean

Most Mathlib goals fall to a cheap structural majority — moves a deterministic pass can make without judgment. What is left is the residual, where structure runs out. Wayfinder does not hand that to a bigger model and hope. **Dr. Ducky**, a deterministic residual executor, emits typed capsules and proof skeletons and replays them Lean-side before anything is claimed.

When a search stops short, `prove` turns its stopping state into Lean: the declaration, Wayfinder's verified tactics in place, and one `sorry` per goal still open, each marked with the goal Lean prints there — compiled until it elaborates. A local finishing pass then tries each hole with deterministic closers (`simp_all`, `omega`, `linarith`, `aesop`, … with the file's own definitions unfolded) and tries the rest for a counterexample: a false hole gives back the step that produced it, and a refutation that reaches the root reports the statement itself as false. What remains is a smaller, cleaner problem, with a cheat sheet of what the search computed, eliminated and left open.

## The last mile runs on this machine

The holes go to an open-weights Lean prover served on the same machine — Goedel-Prover-V2, 8B (Q8_0, 8.7 GB) or 32B (Q4_K_M, 19.8 GB), through llama.cpp. Each hole is lifted to a standalone lemma (`extract_goal`), and Wayfinder's own reads of it decide which model gets it: a light hole goes to the 8B and on to the 32B if the 8B leaves it; a heavy hole goes straight to the 32B. The prompts and self-correction rounds are the model's own training protocol, Lean's errors shown back to it in the format it learned. The servers are sized to the memory free when they start and stand down when the machine needs its RAM back. Nothing leaves the machine.

The first theorem it closed, `kappa_additive_bound` from a specification-complexity paper, had been left open by the search; the hosted prover it would once have gone to no longer takes Lean. On an M1 Max:

| | 32B, Q4_K_M | 8B, Q8_0 (4 in parallel) |
|---|---|---|
| decode speed per sequence | 10.7 tok/s | 24.8 tok/s |
| attempts | 1, verified | 3: two named lemmas that do not exist, the third verified |
| time to a verified proof | ~165 s | 94 s |

That residual was the whole statement — the search had kept no step — so it shows the local closer works, not what decomposition buys it. The measurement that does is next: the same model on the whole statement and on Wayfinder's holes, across theorems where the search makes partial progress.

Aristotle 3.0 (2026-10-27) stopped accepting Lean-only projects. Nothing here depends on it.

## A proof is accepted as the proof that was asked for

A candidate that compiles can still prove a different statement or rest on an axiom. Every fill, from any prover, is checked in place before it is kept: a fill is refused before it is compiled if it holds a `sorry`, runs code at elaboration time (`#eval`, `run_tac`, parser extensions) or reaches past the kernel (`native_decide`, new axioms, `debug.` options); the file must elaborate at Lean v4.28; the theorem's statement must be unchanged, in its text and in its fully elaborated type; and `#print axioms` must name only `propext`, `Classical.choice` and `Quot.sound`. The fill that closes the last hole sends the whole file through the gate once more and saves it with every hole's source in its header.

Run end to end on that paper's eight open gap theorems: seven proved on this machine (six by the search and the finishing pass, one by the 32B on the residual) and one shown false as stated, with the counterexample.

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
| `prove` | search for a proof of one Lean 4 goal; if it stops short, build the residual and finish what can be finished locally |
| `prove_corpus` | run the search over a whole eval set; `prove_corpus_status` reports progress |
| `fill_hole` | take any prover's fill for one open hole of the residual — kept only if the file still compiles with the statement unchanged and no non-standard axiom; the last fill saves the proof |
| `send_to_local_prover` · `local_prover_result` | lift each open hole to a standalone lemma and route it, by how hard it reads, to Goedel-Prover-V2 8B or 32B running on this machine; verified fills go in through `fill_hole` |
| `send_to_aristotle` · `aristotle_result` | hand Aristotle the residual and judge what comes back through the same gate — Aristotle 3.0 refuses Lean-only projects, so this now declines locally unless forced |

The deterministic search is arithmetic on a structured network: no dense retrieval index, no embedding store, no GPU. The local prover lane is the one part that uses the GPU (Metal, through llama.cpp), and it only ever sees the residual.

## Lineage

Wayfinder integrates three earlier lines of work: the navigational semantic network of [ModelAtlas](https://github.com/rohanvinaik/ModelAtlas) (models as coordinates in a signed dimensional space), the hybrid continuous-ternary decoder of Balanced Sashimi ({−1, 0, +1} weights for categorical decisions — the navigator's original form, now an optional ordering layer), and the convergence guarantees of Mutation Theory and specification complexity, formalized in Lean 4. The censors and structural reads of the re-founding draw on its sibling projects, Detective and Peitho. The target is Lean 4 proof generation over Mathlib, evaluated against ReProver (Lean-Dojo), LeanProgress, and DeepSeek-Prover-V2.

The name is an homage to the Pacific wayfinders, who crossed open ocean by reading its structure — swells, stars, birds — rather than instruments.

## Status

A deterministic search at 1,028 / 2,000 on the frozen Mathlib split (1,042 as the cross-pass union), with the encoder stack's `EXP-058` (1,277) kept as the frozen baseline it is measured against; the residual hand-off, the local finishing pass, and the acceptance gate; and the local prover lane for the last mile. Next: the decomposition measurement above, and the second-order program — learning over the residual the first order leaves. Eval sets are frozen (`data/nav_eval.jsonl`, MiniF2F-test, the Mathlib test split); the full results ledger is in [`docs/Research_Paper/`](docs/Research_Paper/).

---

MIT — Rohan Vinaik. Proof search as navigation, verified in Lean 4.
