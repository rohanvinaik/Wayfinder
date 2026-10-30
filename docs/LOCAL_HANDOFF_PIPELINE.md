# The local hand-off pipeline — pass packet

**Status:** the owner confirmed the design on 2026-10-30 and gave the go-ahead the same day. Step 1 is
built (`e662f16`): paired check PRESERVED on all six inputs (`runs/goal_identity_check/`), the
obligation tests pass against Lean. Steps 2–6 follow in order. Written to the owner's process contract (`PROFESSIONAL_CODING_PERFORMANCE_ENGINEERING.md`):
what was observed is marked as such; what was only read in code is marked **HYPOTHESIS**.

## 1. Pass brief

| Field | Content |
|---|---|
| Objective | Recover what Aristotle did before it stopped taking Lean-only projects: close the owner's real theorems after Wayfinder stops short, with Goedel-Prover-V2-8B on this machine in Aristotle's place. |
| The thesis (unchanged) | Wayfinder makes the structure-focused attempt. At its dead end the residual — its Lean-checked partial proof, `sorry` on the goals still open — goes to Goedel. From Goedel's answers Wayfinder keeps only what Lean accepts; a `have` claim whose proof fails stays as a claim with a new gap under it. Wayfinder works the smaller gaps again with all its lanes, Dr. Ducky included. Goedel is asked a small fixed number of times, never revises its own work, and ranks below Dr. Ducky. |
| Protected behavior | Acceptance only through the fresh gate: the combined file compiles at the checking version, the statement's elaborated type is unchanged, no `sorry`/`sorryAx`, allowed axioms only. Nothing is kept on a model's word. |
| Permitted change | A deliberate change of the hand-off's algorithm (contract in §4), and the goal-identity fix in Wayfinder's Lean link, which changes Wayfinder's behavior on purpose (context-changing moves become available). No multi-hour runs; every check below is minutes. Runs start only on the owner's go-ahead. |
| Workload | The owner's real theorems. What was lost: the 21 rows of `data/rerun_v427_manifest.jsonl`, each handed off by Wayfinder and closed by Aristotle (gate-accepted). Also `runs/goal_ledger` (57 entries: `cka_*`, `cosineSim_*`, `ePAB_*`, `robDeg_*`, `sheaf_*`, `stability_*` …), with miniF2F (`data/minif2f_test.jsonl`, `runs/minif2f_v2`) for paired checks. |
| Entry point | MCP: `prove(theorem_id, manifest)` → `send_to_local_prover(theorem_id)` → `local_prover_result(run_id)`. |
| Gate | `ruff check src/ scripts/ tests/`; `.venv/bin/python -m pytest tests/ -q` (as CI runs them). |

## 2. Setup receipt (2026-10-30)

| Item | Observed | Code |
|---|---|---|
| Revision | `8fad0f4`; `tests/detective/certificates.json` untracked (Detective's state) | READY |
| `uv.lock` | the committed lock predates `b239581`'s `mcp<2` dev pin; the working lock adds `mcp` and its tree and passes `uv lock --check` | STALE → committed |
| `.venv` dev tools | `pytest`, `ruff`, `mcp` (declared in the `dev` extra) were absent; `uv sync --extra dev --inexact` added 25 packages, changed none | MISSING → fixed |
| PyPantograph | used by the search, declared nowhere in `pyproject.toml` | MISSING → declared in step 2 |
| Search runtime | `.venv` Python 3.11.14, PyPantograph 0.3.13 → Lean 4.27.0, `data/lean_project` (Mathlib 4.27) | READY |
| Checking | `data/lean_project_v428` (Mathlib 4.28), persistent REPL `data/repl_v428` | READY |
| Search vs checking version | 4.27 vs 4.28 | CONFLICT → step 2 |
| PyPantograph 0.3.14 | Its Pantograph submodule is Lean 4.28.0; 0.3.13→0.3.14 is 3 commits: version marker and toolchain only, no Python or protocol change (GitHub compare, both repositories) | READY to adopt |
| MCP server | `miniconda3` Python 3.14.6 (has `mcp`), launched through SquishOS `owned_resources.py run --label Wayfinder` | READY |
| Test suite | with the dev extra installed: 3,084 collected, no collection errors (the earlier "8 import errors" were the missing extra); 33 failures, all in `tests/detective`, identical on a clean `8fad0f4` checkout: generated goldens captured from pre-commit drafts (or, for `goal_tail_regime`, before `4f11a15`'s intended change) that the committed code never matched | STALE → regenerated through Detective |
| Goedel models | `models/gguf/Goedel-Prover-V2-8B.Q8_0.gguf`, `…-32B.Q4_K_M.gguf`; pool keys `goedel-v2-8b@local`, `goedel-v2-8b@squishhub` | UNVERIFIED (no server running; runs paused) |
| Goal ledger | 48 of 57 entries predate the hand-off (`status` absent): they need a fresh `prove` | STALE |
| Aristotle's record | `runs/aristotle`: 75 projects — 59 gate-accepted (all 21 rows of `data/rerun_v427_manifest.jsonl` among them), 11 unverifiable (submission drift), 3 not downloaded, 1 rejected | READY (the workload to recover) |
| Check inputs | `data/minif2f_test.jsonl` has `mathd_algebra_107`, `amc12a_2021_p9`, `aime_1983_p2`; `cka_nonneg` and `cosineSim_comm` have no manifest row anywhere under `data/` or `runs/` | READY / MISSING (replaced in §7) |
| `DEFAULT_MANIFEST` comment | says 9 theorems; the file has 21 | STALE |
| Design §13, `README.md`, `WAYFINDER_MCP_NOTES.md` | describe the Goedel-first ladder | STALE → step 6 |
| Agent note | in the owner's shell `ls` is `eza`, which hung twice in `runs/`; use `command ls` | — |

## 3. Measured this pass

Probes through `LeanKernel.try_tactic` against the live Pantograph state (Lean 4.27, `data/lean_project`):

| Move | Through Wayfinder's kernel | On Lean's live goal |
|---|---|---|
| `have hab : 0 < a * b` from `a b : ℝ, ha, hb ⊢ a * b + a > 0` | targets `['0 < a * b', 'a * b + a > 0']`; `linarith` on the continuation **fails** — it resolves to the original goal, without `hab` | goal 1 has `hab`; `linarith` **closes** it |
| `obtain ⟨x, hx⟩ := h` from `h : ∃ x, a = 2 * x ⊢ a % 2 = 0` | **`success=False`** — a valid move scored as a failed one | one goal with `x hx`; `omega` closes it |
| `rcases h with h \| h` from `h : a = 1 ∨ a = 2 ⊢ a * a ≤ 4` | `success=True`, `['a * a ≤ 4', 'a * a ≤ 4']`; `subst h; norm_num` **fails** — the branch resolves to the original goal | branch 0 has `h : a = 1`; `subst h; norm_num` **closes** it |

Cause (read in code, consistent with all three): the kernel names a goal by its target text alone
(`str(goal.target)`), keys its live states by that text, and keeps the first state stored under it;
a move that changes only the hypotheses therefore reads as no change, or hands back the old goal.

Also read in code, runtime effect not observed (**HYPOTHESIS**): Dr. Ducky's `execute_ducky_program`
follows only the first resulting goal and its finishers can report `closed` with a sibling open;
`_combined_program` serializes each program's attempted script rather than the tactics applied.
The design choices below are Codex's and mine, agreed over four turns; they are likewise HYPOTHESIS
until each step's check runs.

## 4. The pipeline as it will run

1. **`prove(theorem_id)`** — Wayfinder alone, unchanged in shape: the deterministic search, Dr. Ducky
   included. Closed → done. Otherwise the residual: the Lean-checked partial proof with a `sorry` on
   each open goal, unvalidated commitments withdrawn, closers and `plausible` already tried
   (`_build_handoff`, `_finish_locally`).
2. **`send_to_local_prover(theorem_id)`** — the hand-off loop, in the background:
   1. Goedel-Prover-V2-8B, one batch of 8 samples: 6 on the whole residual, 2 on the bare statement
      (an alternative path when Wayfinder's prefix is a dead end). Goedel's own "complete the following
      Lean 4 code" prompt, 8K-token answers, no revision conversations.
   2. A complete answer goes straight to the gate → `CLOSED`.
   3. Every other answer becomes a skeleton through Lean: accepted steps stay, failing steps become
      holes, a `have` whose proof fails keeps its claim with a hole under it (`skeleton_repair`, at the
      level of the whole theorem rather than a lifted lemma).
   4. Each hole: Wayfinder's closers seeded with Goedel's failed step, and `plausible` beside them; a
      refuted claim drops its skeleton.
   5. Wayfinder searches each remaining hole where it sits — from Lean's own goal at that `sorry`, not
      a lifted lemma — and its partial progress is spliced in, not only complete proofs.
   6. The merged file is recompiled. Every hole closed and the gate passed → `CLOSED`.
   7. Kept between rounds: the original residual (to fall back to) and at most two skeletons, ranked by
      obligations discharged and steps Lean accepted.
   8. One further batch of 4 on the best skeleton's smaller gaps, only if the first round kept
      progress; then steps 3–6 again.
   9. Outcome: `CLOSED`, `RESIDUAL` (the best validated partial proof), `BUDGET_EXHAUSTED` or
      `INFRA_FAILURE`.

   Ceiling: 900 s per hand-off by default (1,800 s on request; 300 s as a smoke mode), with fixed
   Goedel call limits and cancellation. Budgets never restart per skeleton; on exhaustion the best
   checked residual comes back. These limits are **HYPOTHESIS** until timed on the owner's theorems.
3. **`local_prover_result(run_id)`** applies the run's whole residual snapshot (refusing one whose
   parent residual has changed since); `CLOSED` only through the gate.

The 32B is not in the default path: an explicitly requested, separately capped run only.

## 5. Code shape of the changed areas

| Module | Owns | Change |
|---|---|---|
| `src/lean_interface.py` (`LeanKernel`) | the link between Wayfinder and Lean | goals identified by Lean's goal, not its target text (step 1) |
| `src/proof_search.py`, `src/dr_ducky_executor.py`, `src/residual_handoff.py` | the search, Dr. Ducky's execution, the proof tree | carry the goal handle; loop detection by a context-aware fingerprint; Dr. Ducky keeps siblings; the tree matches goal ids (step 1) |
| `scripts/search_goals.py` | Wayfinder's search on given goals | in-place: a skeleton file and its holes in, per hole a closed block or a partial block with its remaining goals out (step 3) |
| `src/local_prover.py` (`Worker`) | the hand-off loop | the whole residual as the unit of work; sampling transport, REPL checks, `modernize`, `skeleton_repair`, closers and refuter reused; revision rounds and lifting leave the runtime path (step 4) |
| `src/local_pipeline.py` | the 8B → 8B-long → whole → 32B ladder | leaves the runtime path (step 4) |
| `wayfinder_mcp.py` | `send_to_local_prover`, `local_prover_result` | whole-residual snapshot; closure means the gate passed on the combined file (step 5) |

## 6. The build — one reviewable change at a time

Each step lands as its own commit, with authored tests for its named obligations and its own short
check; the gate runs after every step.

1. **Goal identity in the Lean link** — one coherent correction across its consumers, not a new
   search policy. Each goal travels as a handle (session, state, goal) with its text and context as
   separate views; lanes keep reading the same text as today, and identity never comes from string
   equality. Where it lands:
   - `LeanKernel`: `goal_start`, file-context loading, `_get_or_create_goal`, `_site_for`,
     `_pantograph_try_tactic` resolve the handle and key tactic results by it; a move counts as
     progress when the context or metavariable assignments change, so an accepted `obtain` is progress.
   - `proof_search`: the `_expensive_tried` guards (Dr. Ducky, self-application, normalization, last
     resort), `_structural_tried`, `_interleaved_tried`, `_infer_cache`, `_post_apply_goals` keyed by
     handle; `_snapshot_frontier`/`_restore_frontier` keep handles.
   - Loop detection (`_goal_signature`, `_progress_pathology_tags`): a fingerprint of context, target
     and dependencies — not fresh state ids, which would hide genuine loops.
   - Dr. Ducky (`_cached_goal_variables`, `_cached_goal_targets`, `execute_ducky_program`,
     `run_ducky_on_goal`): keep the handle, act on the actual branch, keep every sibling; one child
     closing never means the program closed.
   - `_close_goal` records stable goal ids beside the text; `residual_handoff.build_tree`/`_resync`
     match ids; text-only logs stay readable as legacy.

   Obligations, as authored tests: CONTEXT_PROGRESS (`obtain` keeps `x hx`; `omega` closes);
   BRANCH_ISOLATION (each `rcases` branch gets its own fact); HAVE_DEPENDENCY (the continuation sees
   `hab`, and closing it leaves the claim's obligation open); a duplicate-target tree survives
   serialization and replay. MUST-NOT: act in a stale context, lose a sibling, accept `sorryAx`, read
   `obtain` as a loop, or let fresh ids hide a genuine loop. Check: §7's paired run.
2. **One Lean version: 4.28.** PyPantograph 0.3.14; search pointed at `data/lean_project_v428`.
   Obligations: the §3 probes pass at 4.28; known Wayfinder proofs still close, at least one of them
   a premise-lookup close (an `exact`/`rw` lane), since the premise network was built against
   Mathlib 4.27 and a renamed lemma would show there first.
3. **In-place hole search with partial progress.** Obligations: a two-hole fixture — each hole
   searched from its own goal, partial progress spliced back, the file recompiles.
4. **The hand-off loop.** Obligations, with Goedel's answers stubbed: a failing `have` proof keeps its
   claim and gains a hole; a refuted claim drops its skeleton; rounds and samples stay within their
   caps; Goedel never sees its own earlier answer.
5. **Result application.** Obligations: closing one of two holes stays `RESIDUAL` (the ladder's
   "any fill is closure" bug); closing both passes the gate; a stale parent is refused.
6. **Docs.** Design §13, `README.md`, `WAYFINDER_MCP_NOTES.md` rewritten to this pipeline.

## 7. Acceptance — minutes, not hours

**Step 1's paired check (≤ 20 minutes, no Goedel).** The incumbent runs from an unmodified `8fad0f4`
(a clean worktree once any edit has started); otherwise the pair does not share one environment and
the comparison is INVALID MEASUREMENT. Then the corrected code, on the same Lean 4.27 environment,
inputs, configuration, budgets and hardware, through the real benchmark entry:
- from `data/minif2f_test.jsonl`, all three closed by Wayfinder alone in `runs/minif2f_v2`:
  `mathd_algebra_107` (a solver-bootstrap close), `amc12a_2021_p9` (a structural close),
  `aime_1983_p2` (search plus the finishing pass's replay);
- the owner's theorems with a ledger `closed`: `sheaf_implies_local_objective`
  (`data/gap_T7_18_obstruction_intrinsic_manifest.jsonl`), `kappa_additive_bound`
  (`data/gap_T7_17_sheaf_gluing_manifest.jsonl`; its close was the local 32B on a bare hand-off, so
  Wayfinder alone leaves a residual), `pigeonhole_cover` (`data/gap_T5_12_greedy_harmonic_manifest.jsonl`).

30 s per miniF2F search, 75 s per real theorem; whether each closes within its cap is what the
incumbent run establishes, not an assumption. **PRESERVED** means the declared observing set agrees and the gate completes, not identical
tactics; a goal the fix newly closes is **CHANGED** on purpose; a missing proof basis is
**UNRESOLVED**; a wrong environment or an interrupted run is **INVALID MEASUREMENT**.

**The pipeline (after steps 2–5):**

- Live, through send → result: `minif2f_test_mathd_numbertheory_541` (a 162-second closure on record).
- Replay: `minif2f_test_aime_1983_p3`'s saved rejected answer, closed by Wayfinder's repair and the gate.
- The multi-hole fixture: one of two holes closed stays `RESIDUAL`; both closed passes the gate.
- What the owner lost: three of the 21 theorems in `data/rerun_v427_manifest.jsonl` — every one handed
  off by Wayfinder and closed by Aristotle (gate-accepted in `runs/aristotle/regate_report.json`) —
  chosen by fewest holes in a fresh residual: `prove`, then the hand-off.

These establish that the pipeline works; they do not establish a success rate.

## 8. Residual

- Dr. Ducky's sibling issue (§3) is fixed in step 1; whether `_combined_program`'s script
  serialization loses closes at replay remains HYPOTHESIS until step 1's paired check observes it.
- Lean 4.29.1 (PyPantograph 0.3.15) is deferred until this works: it needs a new Mathlib build and a
  REPL at 4.29.1, which has no tag.
- The success rate on the owner's theorems is unmeasured until the pipeline is used on them.
