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

**Registration.** Wayfinder was withheld from Claude Code in `~/.config/mcp/servers.json`
(2026-10-20); restored and synced 2026-10-29 (still withheld from Claude Desktop). A running
server keeps the code it started with — restart it after changing `wayfinder_mcp.py` or `src/`.
