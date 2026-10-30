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

The encoder is demoted, not deleted: `search_mode: full` (`configs/wayfinder.yaml`) still loads NAV-004 as an ordering layer and reproduces the frozen `EXP-058` baseline. The MCP tools run the deterministic path only — no checkpoint, no sentence encoder, no cosine lane.

## The number is never inflated

A proof either type-checks or it does not, so a prover's headline is easy to inflate: fold in the ones a repair pass rescued, the ones a second system finished, the ones nothing re-checked. Wayfinder records `raw_success` and `honest_success` separately and never adds a lane into either. The loop's official count is the union of kernel-verified honest closes — a checked proof stays checked — and re-measurements of a standing bank wobble by up to ±3 under any ordering change, so a single-run delta inside that band is noise, not signal. An infrastructure miss (a crashed replay, a timeout) is ledgered apart from a model failure: the 1,028 run had 201 of them, none counted against it.

---

## What the search cannot close is handed on as Lean

Most goals fall to a cheap structural majority — moves a deterministic pass can make without judgment. What is left is the residual, where structure runs out. When a search stops short, `prove` turns its stopping state into Lean: the declaration, Wayfinder's verified tactics in place, and one `sorry` per goal still open, each marked with the goal Lean prints there — compiled until it elaborates. A local finishing pass tries each hole with deterministic closers (`simp_all`, `omega`, `linarith`, `aesop`, … with the file's own definitions unfolded) and tries the rest for a counterexample: a false hole gives back the step that produced it, and a refutation that reaches the root reports the statement itself as false.

**The residual holds validated progress only.** A step that commits to something — a disjunct (`left`), a witness (`use 1`), `congr`, `apply` — stands only if everything under it closed; otherwise it is withdrawn, its goal becomes the hole, and what was tried there travels with the hole as a note. Before this rule, 11 of MiniF2F-test's residuals were false goals no prover could close (`imo_1997_p5` had kept `left` and handed off `x = 1 ∧ y = 1`, false at (16, 2)).

Under each hole the residual carries the goal Lean has there — its hypotheses and its target — written as the comment the prover reads.

## The last mile: Wayfinder, Lean and a prover in conversation

The residual goes to an open-weights Lean prover served on the same machine — Goedel-Prover-V2, 8B (Q8_0) by default, through llama.cpp — in its own "complete the following Lean 4 code" form: the theorem, Wayfinder's validated partial proof, Lean's goal under each hole. Then the three parts work on what each other cannot:

- **Lean says which steps hold.** A complete answer goes to the acceptance path. Any other answer is taken apart: the steps Lean accepts stay, each failing step becomes a hole, and a `have` whose proof fails keeps its claim with a hole under it — the model's claims are what it is there for.
- **Wayfinder works the holes where they sit.** Its closers, seeded with the step the model wrote there, with `plausible` beside them (a refuted claim drops its skeleton); then its search on each remaining hole, started from the goal Lean has at that `sorry` — the file loaded with that hole the only `sorry`, the others closed by a placeholder — and what it builds spliced back, validated progress only.
- **The model sees what is left.** The best partial proof, if it holds validated progress, goes back to the model once more, with Lean's goals under its holes. The model never revises its own answers.

A hand-off is bounded: eight answers (six on the residual, two on the bare statement — a path that does not start from Wayfinder's prefix), at most one more round of four, 900 s. Checks inside the loop run through a persistent Lean REPL — Mathlib imported once, recycled and canaried because long-lived Lean processes drift — and every acceptance through a fresh compile. Search and checking run on one Lean, 4.28 (PyPantograph v0.3.14 with three fixes: `scripts/setup_pantograph.sh`). A version adaptation meets the model's training-era Lean: at v4.28, `norm_num at h ⊢` fails with "No goals" when `h` closes the goal, and a fifth of the proofs Lean had rejected were correct but for that.

On MiniF2F-test (2026-10-30): Wayfinder alone closes **102 / 244**, each proof re-checked by the v4.28 gate with zero tokens spent. The loop's deterministic half closed `aime_1983_p3` from a proof the 8B had failed — the model's plan, its two broken steps closed by Wayfinder — with no further model call. The full run on this architecture is under way in `runs/minif2f_v2`; `decomposition_test.py ledger` reports closures per stage, and answers and tokens per theorem against Goedel-Prover-V2's published 32 per theorem. The full design is §13 of [`WAYFINDER_DESIGN.md`](docs/Research_Paper/WAYFINDER_DESIGN.md).

Aristotle 3.0 (2026-10-27) stopped accepting Lean-only projects. Nothing here depends on it.

## A proof is accepted as the proof that was asked for

A candidate that compiles can still prove a different statement or rest on an axiom. Every fill, from any prover, is checked in place before it is kept: a fill is refused before it is compiled if it holds a `sorry`, runs code at elaboration time (`#eval`, `run_tac`, parser extensions) or reaches past the kernel (`native_decide`, new axioms, `debug.` options); the file must elaborate at Lean v4.28; the theorem's statement must be unchanged, in its text and in its fully elaborated type; and `#print axioms` must name only `propext`, `Classical.choice` and `Quot.sound`. The fill that closes the last hole sends the whole file through the gate once more and saves it with every hole's source in its header.

Run end to end on the eight gap lemmas that paper's Wayfinder manifests restate: seven proved on this machine (six by the deterministic search and the finishing pass, one by the 32B on the residual) and one shown false as stated, with the counterexample. They are restatements of the gaps; the paper's own Lean still has to be wired to them.

---

## Why decomposition is safe — the composition gap

Splitting a monolithic prover into specialists is only sound if the pieces do not entangle. Specification-complexity theory gives the condition. Model each specialist by its specification complexity σ, and the composition gap theorem bounds the whole:

$$\sigma(A \circ B) \;\le\; \sigma(A) + \sigma(B) + \gamma(A, B).$$

When the specialists are independent, the cross term γ vanishes and the bound is the sum of the parts — complexity adds rather than multiplies, so a system of small, stable specialists costs no more to specify than its parts do. (The Lean formalization proves this subadditive bound for κ, under injectivity and surjectivity conditions on the parts.) Wayfinder's Society-of-Mind runtime decomposes the prover into typed temporal slots — PERCEPTION → RECOGNITION → PLANNING → ORCHESTRATION → EXECUTION → VERIFICATION. The move that makes this pay is narrative construction: it turns hard proof-structure prediction (low symmetry, exponential σ) into template classification (high symmetry, polynomial σ). Structure is not predicted. It is recognized.

---

## Run it

Wayfinder is an MCP server. Point it at a Lean goal and it navigates:

| Tool | What it does |
|------|--------------|
| `prove` | search for a proof of one Lean 4 goal; if it stops short, build the residual and finish what can be finished locally |
| `prove_corpus` | run the search over a whole eval set; `prove_corpus_status` reports progress |
| `fill_hole` | take any prover's fill for one open hole of the residual — kept only if the file still compiles with the statement unchanged and no non-standard axiom; the last fill saves the proof |
| `send_to_local_prover` · `local_prover_result` | the hand-off, in the background: the residual to Goedel-Prover-V2 on this machine, its answers taken apart by Lean, Wayfinder's search on the holes where they sit, the improved residual back to the model once; a closed proof is gated at v4.28 and saved, otherwise the best partial proof is reported |
| `send_to_aristotle` · `aristotle_result` | hand Aristotle the residual and judge what comes back through the same gate — Aristotle 3.0 refuses Lean-only projects, so this now declines locally unless forced |

The deterministic search is arithmetic on a structured network: no dense retrieval index, no embedding store, no GPU. The local prover lane is the one part that uses the GPU (Metal, through llama.cpp), and it only ever sees the residual.

## Lineage

Wayfinder integrates three earlier lines of work: the navigational semantic network of [ModelAtlas](https://github.com/rohanvinaik/ModelAtlas) (models as coordinates in a signed dimensional space), the hybrid continuous-ternary decoder of Balanced Sashimi ({−1, 0, +1} weights for categorical decisions — the navigator's original form, now an optional ordering layer), and the convergence guarantees of Mutation Theory and specification complexity, formalized in Lean 4. The censors and structural reads of the re-founding draw on its sibling projects, Detective and Peitho. The target is Lean 4 proof generation over Mathlib, evaluated against ReProver (Lean-Dojo), LeanProgress, and DeepSeek-Prover-V2.

The name is an homage to the Pacific wayfinders, who crossed open ocean by reading its structure — swells, stars, birds — rather than instruments.

## Status

A deterministic search at 1,028 / 2,000 on the frozen Mathlib split (1,042 as the cross-pass union), with the encoder stack's `EXP-058` (1,277) kept as the frozen baseline it is measured against; the residual hand-off, the local finishing pass, and the acceptance gate; and the local proving loop for the last mile — the conversation among Wayfinder, Lean and Goedel-Prover-V2 above, with MiniF2F-test under way on it. Next: that run's numbers, the decomposition measurement (the same model on Wayfinder's holes and on whole statements, at equal budget), and the second-order program — learning over the residual the first order leaves. Eval sets are frozen (`data/nav_eval.jsonl`, MiniF2F-test, the Mathlib test split); the full results ledger is in [`docs/Research_Paper/`](docs/Research_Paper/).

---

MIT — Rohan Vinaik. Proof search as navigation, verified in Lean 4.
