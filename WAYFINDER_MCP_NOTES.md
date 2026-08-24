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
