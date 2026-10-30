# Wayfinder MCP — setup notes (gotchas found while wiring the SSL Fisher-bridge proofs, 2026-07-08)

Captured so a future MCP wrapper doesn't re-hit these. The MCP goal: future agents call
`wayfinder_triage(lean_file)` → context+premises, and `aristotle_prove(lean_file, hints)` → filled Lean,
cleanly, without rediscovering the invocation (which took ~an hour to reconstruct this session).

## Gotcha 1 — the premise DB path is cwd/repo-relative; it silently drops to `--no-retrieval`

`wayfinder_residuals.py`'s default `--db` resolved to `…/<cwd-repo>/Projects/Wayfinder/data/proof_network_v3.db`
(the base is not `$HOME`), so from `papers/COEC/proofs/` it looked in the wrong place, warned, and ran
**profile-only with ZERO premise retrieval** — the entire neural/DB premise mass off. The 6-bank profiler
still triaged correctly (structural layer is DB-independent), which masks the failure.
**MCP fix:** always pass the ABSOLUTE path:
`--db /Users/rohanvinaik/Projects/Wayfinder/data/proof_network_v3.db`. And surface the retrieval on/off
state in the tool result (don't let it fail silent).

## Gotcha 2 — Aristotle CLI drifted: `prove-from-file` → `submit`/`formalize`/`result`

`coec_proof_generator.py` calls `aristotle prove-from-file … --no-validate-lean-project …` — that
subcommand no longer exists in the installed CLI. Current surface:
`aristotle {submit, formalize, result, list, cancel}`.
- **Prove a Lean project:** `aristotle submit "<instructions>" --project-dir <dir> --wait --destination <out>`.
- **NL → Lean:** `aristotle formalize <file>`.
- Key via `ARISTOTLE_API_KEY` env (or `--api-key`). Aristotle wants Lean **v4.28.0** + Mathlib **v4.28.0**;
  give it a minimal project dir (toolchain + lakefile + the `.lean`), NOT a giant `.lake`.
**MCP fix:** target the current `submit`/`result` surface; don't hardcode `prove-from-file`.

## Gotcha 3 — the Wayfinder context `.md` is NOT ingested by `aristotle submit`

The pipeline's Wayfinder context (`context/wayfinder_proof_NN.md` — strategy + top-16 premises) is a
separate file the current `submit` CLI does not read. So "Wayfinder premises → Aristotle" only happens if
the MCP **injects** the premises/strategy into the `submit` prompt (or as comments in the `.lean`).
Otherwise Aristotle proves the bare statement unaided (weaker). **MCP fix:** the `aristotle_prove` tool
should fold the Wayfinder triage output into the submit prompt.

## Gotcha 4 — `wayfinder_residuals.py` is hardcoded to the COEC corpus

It scans `proof_{NN}.lean` (`--proofs 1-25` default) via `_choose_lean_file`. To triage an arbitrary file
you must name it `proof_{NN}.lean` in `--lean-dir` and pass `--proofs NN`. **MCP fix:** the tool should
accept any path and adapt (symlink/copy to a temp `proof_NN.lean`, or generalize `_choose_lean_file`).

## Environment facts
- Wayfinder venv: `/Users/rohanvinaik/Projects/Wayfinder/.venv/bin/python`.
- Premise DB: `/Users/rohanvinaik/Projects/Wayfinder/data/proof_network_v3.db` (Mathlib v4.27.0 entries).
- Aristotle key lives in `/Users/rohanvinaik/semantic_probing/.env` (the COEC pipeline reads it there).
- Reusable v4.28.0 Lean project exists at `papers/COEC/proofs/lean/` but is currently in an inconsistent
  build state (built oleans v4.27.0 vs toolchain v4.28.0 → local `lake build` needs `lake update mathlib`
  + `lake exe cache get`). Aristotle proves remotely so a local build isn't required for it.
- Register the MCP via `~/.config/mcp/servers.json` + `sync.py` (single source of truth). Name: `Wayfinder`.

## 2026-10-29 — how a theorem moves through the MCP now

**`prove(theorem_id)`** — search, then everything that can be done locally:
1. The search records each committed closure (`SearchResult.close_log`: the goal, its index in
   the open list, that list, the replayable Lean text, the goals it opened).
2. `src/residual_handoff.py` rebuilds the tree and renders it as Lean — the declaration,
   Wayfinder's tactics in place, a `sorry` per open goal — compiled at v4.28 until it elaborates
   (a failing step becomes a hole; its siblings stay). `faithful` / `matches_stopping_state` say
   whether the holes are exactly where the search stopped. `initial_goal_tier` decides whether
   the binders are reverted first (the search began from the closed type: "A"/"env_inspect"/"raw").
3. The local finishing pass (`_finish_locally`) tries each hole with deterministic closers —
   `simp_all`, `omega`, `decide`, `norm_num`, `linarith`, `nlinarith`, `positivity`, `tauto`,
   `aesop` (unfolding the file's own definitions the hole mentions), the Euclidean-division facts
   + `omega` for goals with `a / b`, then `grind` — and tries the rest for a counterexample
   (`plausible`). A refuted hole gives back the step that produced it; a refutation reaching the
   root marks the statement `refuted`. Each hole's close and refute probes are separate
   declarations in one compile, each with its own heartbeat budget.
4. A theorem with no holes left is gated and saved to its row's `proof_home` (warrant provenance
   says which steps came from the search and which from `local_finish`). Otherwise the skeleton
   is the residual, stored in the goal ledger.

**`send_to_aristotle`** submits only that residual (refuses without a fresh one, or when the
statement hash changed), asking Aristotle to fill exactly the marked holes. SDK 3.x: questions are
always `DISABLED` (the 3.0 default flipped to a 15-minute wait), and the project is named.

**`aristotle_result`** accepts a returned file only through `src/proof_gate.py`: nothing that runs
code or bypasses the kernel, every submitted declaration unchanged, the statement unchanged (text
and `pp.all` type), compiles at v4.28 with no `sorry`, axioms ⊆ {propext, Classical.choice,
Quot.sound} ∪ the row's `allowed_axioms`. Rejects stay in `runs/aristotle/<id>/` with `gate.json`.
A correct re-proof is accepted and reported as `reproved`. `get_files()` returns the INPUT when a
project has no result files, so `has_files` is checked first. `_verify_at_v428` uses the same rule.

**Aristotle 3.0 (2026-10-27) no longer takes Lean.** It verifies Verilog, C and Rust
(`aristotle formalize` removed; manifest property kinds `natural_language` / `sva` /
`c_function_no_ub`). A Lean-only project is refused at run time — "Lean-only projects aren't
supported right now as we focus on verified software and hardware" — reported as
`outcome: service_refused` and recorded in `runs/aristotle/service_status.json`;
`send_to_aristotle` then declines locally (`force=True` re-tries; an accepted result clears it).
The server's `/version` reports Lean v4.34.0; the account allows one project in progress.

**Search kernel fixes found by the production run:**
- A tactic now acts on its own goal: `_site_for` addresses the goal's position in its cached state
  with `auto_resume=False`. Before, every tactic hit the state's first goal (`exact hq` on the `q`
  of `p ∧ q` was applied to `p`), and the result carried the untouched siblings back as new goals
  (goals counted twice; a do-nothing `intros` read as progress).
- Tier B starts the declaration asked for: in a file of `sorry` stubs it took `targets[0]`, the
  file's first stub (`_target_state` now takes the last, and declines on a count mismatch).
- The hand-off file carries the source's `variable`/`universe` commands in scope at the theorem.

**Measured on this paper's 8 gap theorems (production, over MCP):** 6 proved locally and saved to
`Specification_Complexity_Paper/proofs/`; 1 refuted (T5.12 `greedy_step_decay`, r = 1, k = 2);
1 open (T7.17 `kappa_additive_bound`). Re-gating the 71 downloaded Aristotle results offline
(`scripts/regate_aristotle_results.py`): 59 accepted, 1 rejected (a `sorry` left in), 11
unverifiable (`lib/Pabkit.lean` changed after they were submitted; submissions now store their text).

## 2026-10-29 (evening) — closing the residual on this machine

**`fill_hole(theorem_id, k, proof, helpers, source)`** is the one acceptance path for a fill,
whoever proposes it (Claude, Codex, a person, the local lane). Nothing is compiled that could run
code or bypass the kernel: `sorry`/`admit`, `native_decide`, `run_tac`/`by_elab`, the `#eval`
family, parser extensions, axioms, `debug.` options, `exact?`/`apply?`. Command keywords are
refused even mid-line, because after a parse error Lean resumes at the next command token. The
fill replaces hole k's `sorry` (under a bullet, continuation lines sit at the bullet's column + 2),
helpers go directly above the theorem, and the file is compiled at v4.28 with the gate's probe: no
error, the theorem's `pp.all` type unchanged, axioms ⊆ the standard three ∪ the row's ∪ `sorryAx`
while other holes are open. A kept fill updates the ledger — `as_built` keeps the hand-off
`prove()` made, `holes` shrinks, `fills` records each source — so `send_to_aristotle` and the local
lane see the smaller residual. The fill that closes the last hole runs `gate.judge` against
`as_built` and saves the proof with every hole's source in its provenance header. A rejected fill
changes nothing; Lean's errors come back in the reply.

**`send_to_local_prover` / `local_prover_result`** run Goedel-Prover-V2 on this machine: 8B at
Q8_0 (8.7 GB) and 32B at Q4_K_M (19.8 GB), GGUFs in `models/gguf/` (gitignored, sha256-checked
against the Hugging Face LFS record), each served by its own llama-server on a free localhost port.
Port 8080 belongs to the squishos model router; leave it alone.
- **Routing** (`src/local_prover.assess`): votes from `src/structural_reads.py` plus an
  existential target, the file's own definitions, and a whole-statement residual. Under 3 votes a
  hole is light (8B, then 32B if the 8B leaves it); at 3 or more it is heavy (32B). `model=` forces one.
- **Lifting**: `extract_goal * using <target>_wf_hole_k` at each hole, all lifted statements
  re-elaborated in one compile before they are used; a whole-statement residual uses the
  declaration as written. In place, `rename_i` restores the names `extract_goal` gave inaccessible
  hypotheses; otherwise the proof goes in as a helper lemma applied to the explicit hypotheses.
- **Prompts** are Goedel-Prover-V2's own (Goedel-LM/Goedel-Prover-V2 `src/utils.py`, `dpskcot`):
  the round-0 request, then self-correction rounds that show Lean's errors between `<error>` tags.
  The rendered text matches the models' chat template byte for byte (`tests/fixtures`). Sampling
  is theirs too: T = 1, top-p 0.95, 32K new tokens, 8 attempts, 2 correction rounds × 2 samples
  for each of the 4 best failures.
- **Every answer**: deny-list → `modernize` (`∑ x in s` → `∑ x ∈ s`: the 2024 big-operator syntax
  these models write, which this Mathlib no longer parses — the 32B's first answer on kappa
  failed on exactly that) → the lifted lemma compiled with the model's proof (400000 heartbeats,
  not the unlimited `0` their header uses) → the fill compiled in place. Verified fills go to
  `fills.json`; `local_prover_result` applies them through `fill_hole`'s path, and only while the
  ledger still holds the hand-off the run was given.
- **Memory**: slots = full 40,960-token sequences (q8_0 KV cache) that fit beside the weights,
  keeping `reserve_gb` free; the server is stopped when its holes are done, when memory pressure
  holds at warn for 30 s (`kern.memorystatus_vm_pressure_level`), or on SIGTERM to the worker.
- **The log** (`runs/local_prover/<run_id>/samples.jsonl`) gives every sample an outcome:
  verified, wrong_proof, truncated, no_code, no_theorem, denied, in_place_failed, cancelled,
  superseded, memory_yield, server_error, context_full. Only wrong_proof counts against a model.

**Measured on `kappa_additive_bound` (M1 Max, 64 GB; the hole was the whole statement):**

| | 32B Q4_K_M, 1 slot | 8B Q8_0, 4 slots |
|---|---|---|
| decode speed per sequence | 10.7 tok/s | 24.8 tok/s |
| samples finished | 1 (verified) | 3 (2 hallucinated lemma names, 1 verified) |
| time to a verified proof | ~165 s | 94 s |
| load (warm page cache) | 4 s | 6 s |

The 32B's fill was applied: `Specification_Complexity_Paper/proofs/kappa_additive_bound.lean`,
standard axioms, provenance naming the model, quant, sample and run. That makes 7 of the paper's
8 gap theorems proved on this machine (`greedy_step_decay` is false as stated).

**Gate:** `set_option debug.*` (`debug.skipKernelTC` turns the kernel check off) is now a static
finding, in the Aristotle path as much as the local one.

**`prove` and `prove_corpus` run the deterministic path only** (2026-10-29, Rohan's call): the
resolved config is `configs/wayfinder_deterministic.yaml` — no navigator checkpoint, no sentence
encoder, no cosine lane (`--search-mode full`, `--cosine-rw`, `--cosine-rw-seq` are gone;
`--checkpoint` is still passed because the CLI requires it, and is never loaded). `configs/wayfinder.yaml`
reproduces the encoder stack's EXP-058 baseline through `run_benchmark` directly. Regression on the
paper's eight gap lemmas: the four T7.18 lemmas and `pigeonhole_cover` close as before;
`greedy_step_decay` is refuted as before; `sheaf_gluing_global_kill` regressed — a deterministic step
left a hole no closer took, where `grind` closes the statement whole — which is why the hand-off now
ends with a **whole-statement fallback** (`_close_whole`): when holes survive the finishing pass,
the closer battery gets the undecomposed statement once. With it, all six close again.
`kappa_additive_bound` is now a residual with two kept steps and one hole (it was bare).

**EXP-DEC-001** (pre-registered in `docs/Research_Paper/EXPERIMENT_RESULTS.md`):
`scripts/decomposition_test.py` — the same local prover on the whole statement and on Wayfinder's
residual, paired, on MiniF2F-test (not the Mathlib split, where a restatement can cite itself).

**MiniF2F-test, deterministic (EXP-DEC-001 phases 1–2, 2026-10-30):** the search closes 89/244, the
local finishing pass 19 more — **108/244 with zero neural components** — leaving 125 residuals and
11 that never start. The search runs in chunks of 20 per process: a Pantograph server that dies
after a run of REPL panics is not always restarted, and every theorem after it is then skipped
at once (57 of 61 on the first attempt).

**Process-shaped errors, cleared before they cost an attempt** (all local, Lean's own messages
decide): a hole is lifted twice, plain and with `pp.numericTypes`/`pp.funBinderTypes` (the plain
print can re-elaborate at another type); `∑ x in s` → `∑ x ∈ s`; and a model answer whose every
error is "No goals to be solved" / "made no progress" on its own proof lines has exactly those lines
deleted and is compiled again (the 8B's mathd_algebra_276 answer verifies after three). Next, the
typed version: replay a model's tactics step by step through Wayfinder's Pantograph kernel and keep
what acts on a goal; a rename table mined from Mathlib's `@[deprecated] alias`es; Wayfinder's IDF
premises in the prompt.

**Registration.** Wayfinder was withheld from Claude Code in `~/.config/mcp/servers.json`
(2026-10-20); restored and synced 2026-10-29 (still withheld from Claude Desktop). A running
server keeps the code it started with — restart it after changing `wayfinder_mcp.py` or `src/`.

## 2026-10-30 — the local proving loop behind `send_to_local_prover`

Full design: §13 of `docs/Research_Paper/WAYFINDER_DESIGN.md`. In the MCP:

- **`prove`** is unchanged in role. The residual it builds now holds validated progress only: a step that committed to something (`left`, `use 1`, `congr`, `apply`, a `have`, …) stands only if everything under it closed; otherwise its goal is the hole and the steps are noted (`TRIED_MARK`). The lift is fit-checked: the plain `extract_goal` print is used only if, on `sorry`, it closes its own hole; otherwise it is printed typed (numerals, binders, coercions).
- **`send_to_local_prover(theorem_id, manifest, …)`** runs, by default, the per-theorem ladder (`src/local_pipeline.py`), stopping at the first closure:
  1. the 8B at 8K tokens on the residual, with 2 revision rounds and the repair loop;
  2. the 8B at 30K, revising stage 1's rejected proofs, only if its answers hit the token cap;
  3. the whole statement;
  4. the 32B, revising the 8B's best proofs.

  A hole Wayfinder reads as hard puts the 32B first. `model=` runs only that model's stages. `start_at=` begins further down. The reply's `ladder` is the order that will run. `pipeline=False` runs one worker as before, with `samples`, `correction_rounds` and `max_new_tokens`; in the pipeline the ladder sets its own.
- **The repair loop**, inside every stage after each round, runs on the best rejected proofs and on drafts (answers whose only fault is `sorry`, or cut off by the cap):
  - Each becomes a skeleton: the plan is kept and the failing steps become holes.
  - Wayfinder's closers attack the holes, seeded with the model's own step and its premises, with `plausible` alongside them.
  - Wayfinder's search attacks the steps left over (`scripts/search_goals.py`, one Pantograph session).
  - Then all filled → acceptance; still open → the model completes the skeleton; refuted → a false claim, named in the revision.
- **`local_prover_result(run_id)`** reads a pipeline run as it read a single run (the run's directory mirrors the stage under way, then the winning stage). Hole fills go in through `fill_hole`'s path. A whole-statement closure (`fills[].whole`) is gated by a fresh v4.28 compile and saved there. Either is saved only if the row has a `proof_home`: MiniF2F rows have none, so a proof is accepted and reported but not written.
- **Validated with real models** on `mathd_algebra_484`:
  - stage 1: 66 s, applied, closed;
  - whole statement: 61 s, accepted;
  - 32B: 173 s, verified, its fill refused as the hole was already closed.

**Gotchas**

- **The MCP server keeps the code it started with.** After a change to `wayfinder_mcp.py`, reconnect it. The pipeline and the worker are subprocesses and pick up changes on their next run.
- **Checks in the loop run through a persistent REPL** (`data/repl_v428`). It is recycled every 30 commands (it keeps every environment it makes) and canaried every 15. `data/repl_v428/DISABLED` turns it off for new workers. Acceptance never uses it.
- **Two 8B servers do not fit beside each other with the REPLs on a 64 GB machine.** A worker sizes its server to free memory less `reserve_gb`, and gets zero slots if another run holds the GPU.
- **Pantograph sessions stay short.** A long-lived one drifts and crashes silently (Rohan). The search runs one session per batch and exits.
- **Wayfinder's own search closes count only once the v4.28 gate accepts their replay.** Of MiniF2F-test's 108, 102 did. `decomposition_test.py reconcile` puts the rest back into the pipeline.
