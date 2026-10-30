# Structural resolution and the 4.28 rebuild — implementation specification

**Status:** approved for implementation by the owner (Rohan Vinaik), 2026-10-30. Nothing in it is
built yet. It is written to be implemented step by step, in order, by an agent that did not take
part in the design. Every step names its files, functions, behavior, tests, checks and commit. Where
this document and the code disagree, **stop and report** (§2.4); do not improvise.

Working papers of the design (read only if a step sends you there):
`runs/structural_resolution_consult_20261030/` — `01_design_reading.md` (the design before
consultation), `04_codex_prompt.md` and `05_codex_answer.out` (the Codex consultation),
`06_comparison.md`, and this pass's gate output. The session record that started the work:
`runs/2026-10-30_goedel_structural_resolution_record.md`.

---

## 1. Why — the owner's intent and decisions (verbatim; not to be re-litigated)

**The thesis this build serves.** "the point of Wayfinder *isn't* to solve proofs whole-cloth, but
to traverse the structural relationships between the mathematical concepts/llemas/etc to narrow the
problem space to the point where the correct solution emerges almost inevitably, where even some
dumb brute-force style hardware like "exact?" and "apply?" can do the work directly."

**Goedel's role.** "Lean's native tooling, as we already use in Wayfinder's symbolic proof-solving
process, with Goedel as the last-ditch neural oracle as a sort of shrug and a vibe. The point of the
neural component is to rescue the symbolic layers from the trap of over-literal blockers … helping
with that last-mile conceptual transport that lets Wayfinder/Dr.Ducky and Lean/Pantograph handle the
rest."

**Names.** "The names *are* structural evidence, but they are NOT a warrant to try token-matching
blindly … a name like "card_le_of_subset" is pretty unambiguous targeting for clean localization of
the structural approach to the proof that Wayfinder can process symbolically."

**No pre-re-founding machinery.** "I do not want to recover pre-"refounding" implementation" — no
encoder, no embeddings, no cosine scoring anywhere in this build.

**Decisions, 2026-10-30:**
- **A — agreed:** one repair turn per hand-off in Goedel's trained self-correction format (its own
  failed proof with Lean's errors), with Wayfinder's structural candidates added to the error text,
  taken from round 1's allocation. This amends §13.9 invariant 6 of the design reference (§S12).
- **B — agreed:** the residual goes to Goedel whole; a blocker record is rendered as comments at the
  hole (no lifted standalone lemmas).
- **C:** "That's correct for the order/implementation workflow, but the "scope" of the build is the
  full thing." Order: the isolated structural-resolution path first; changes to existing navigation
  only after their paired checks. Scope: everything in this document.
- **D:** answered by C and E — every known gap is built, including Dr. Ducky's compatibility rules
  (G3; listed as "D1" in `docs/Research_Paper/CLOSING_DECISIONS.md`).
- **E:** "No, full rebuild. The point is a complete/correct design of the project. Not implementing a
  known gap is...strange." The proof network is rebuilt at Lean 4.28 / Mathlib 4.28 (§S11).
- **F — yes:** remove the `Co-Authored-By` trailers from the pushed commits; prune the dangling
  worktrees (§S0).

---

## 2. Working rules for the implementing agent

### 2.1 House rules (the owner's, for every agent on this machine)
1. **Read whole units.** Before editing a function, read the whole function and its callers; before
   editing a module, read the whole module. Never pull lines out of text: no `grep`, `rg`, `ag`,
   `ack`, `git grep`, no `head`, `tail`, `sed`, `awk`, `cut`, `wc`, `jq`, no pipes into filters. Use
   the Read tool, `symbex <name> -f <file>` (whole function/class), `mdq '# "<heading>"' <file>`
   (whole Markdown section), `sqlite3` for databases, and short Python that *computes* a measurement.
2. **Who consumes it.** Before changing a symbol, find its callers (Serena `find_referencing_symbols`,
   or read the importing modules) and ask whether a second path does the same job. Change every
   path, or stop and report.
3. **Commits.** Work on `main`. No branches, no worktrees. One commit per step unless the step says
   otherwise. Messages in the repository's style (`area: plain sentence`). **No `Co-Authored-By`
   trailer, no session link, no AI attribution — ever.**
4. **The gate, before every commit:** `.venv/bin/ruff check src/ scripts/ tests/` → "All checks
   passed!"; `.venv/bin/python -m pytest tests/ -q` → 0 failures, 0 errors. Report the
   passed/skipped counts. Baseline (2026-10-30, before S0): 3,078 tests, 0 failures, 0 errors, 544
   skipped.
5. **Never modify:** `data/nav_eval.jsonl`, `data/nav_training.jsonl`, `data/mathlib_test.jsonl`,
   `data/minif2f_test.jsonl`, `data/rerun_v427_manifest.jsonl`, `runs/loop_union.json`,
   `data/proof_network_v3.db`, `configs/wayfinder_deterministic.yaml`'s existing keys, any model
   checkpoint, `src/hardtail_bridge.py` (legacy statistical layer — except the parameter rename in
   §S5), `tests/detective/certificates.json` (Detective's state; never commit it).
6. **Never introduce:** string similarity on identifiers (`difflib`, `SequenceMatcher`,
   `get_close_matches`, Levenshtein, `rapidfuzz`, edit distance, rename tables); any encoder,
   embedding, cosine score or learned model in the deterministic path; any model call inside Dr.
   Ducky's runtime (`src/dr_ducky*.py`).
7. **Named outcomes, never booleans,** for any decision with more than two meaningful outcomes
   (§4.1).
8. **Names:** Wayfinder, Dr. Ducky (always "Dr."), ShortcutForge, ModelAtlas, Detective, LintGate.

### 2.2 Long runs
Any single command expected to run longer than 30 minutes, and any change to a README or ledger
number, requires the owner's go-ahead. Measure first (each step says how), report the projection,
**stop and ask**.

### 2.3 Detective
For each new **pure** function named in a step as "Detective-pinned", follow
`~/Projects/SquishOS/ops/agents/reference/detective-protocol.md` (read it whole once) and record
Detective's verdict in the step report. Authored tests are required regardless of Detective.

### 2.4 Stop conditions — stop, write the step report, and ask the owner
- A function, file, field or command named here does not exist or has a different shape.
- A pinned expectation in a test fails and the cause is not a bug in your own new code (do not
  edit the expectation).
- A step's acceptance check fails after one fix attempt.
- A change would alter behavior that the step's contract does not list as changed.
- A paired check reports CHANGED where the step expects PRESERVED.

### 2.5 Step reports
Append one section per step to `runs/structural_resolution_build/STEP_REPORTS.md`: step id; commit
hash(es); gate counts; each acceptance check with its numbers; Detective verdicts; deviations (if
any, with the owner's answer); residual. Create the file in S0.

---

## 3. Glossary

| Term | Meaning in this build |
|---|---|
| **step** | One failing proof step of a Goedel answer, as `Worker.skeleton_repair` cut it out (`Skeleton.replaced[k]`): one or more tactic lines. |
| **identifier resolution** | Classifying each identifier of a step by **exact lookup**: local hypothesis, local declaration (the hand-off file's own), 4.28 catalog constant (also under each `open` namespace), or unresolved. Never by spelling similarity. |
| **slot** | A position in a step whose content Wayfinder must supply: an unresolved identifier, or a catalog constant Lean rejected at that position. Kinds: `const`, `rewrite`, `unfold`, `hint`, `arg` (§4.2). |
| **role** | What a slot must do, read from Lean: the signature features of the goal at the slot (lifted with `extract_goal`), restricted to the hypotheses the step passes. A retrieval query only — Lean's elaboration decides. |
| **signature** | Features of a declaration's elaborated type, computed in Lean from its `Expr` (§4.3). |
| **window** | The candidates for a slot, ranked; read in pages; never truncated (continuation). The **role window** comes from the role; the **name window** from the name decoder. They are unioned, never intersected. |
| **name decoder** | Reads an identifier through Mathlib's naming grammar into signature claims, from a closed vocabulary (§S3.3). Localizes; never filters, never decides. |
| **probe** | A Lean check of one candidate fill (REPL; Pantograph where a step says so). |
| **Jeeves / refine** | The deterministic question operator over a window (§S8), ported from ModelAtlas's `build_refinement_guidance`: picks the feature that best splits the window; answers order branches, never remove them. |
| **blocker record** | Comment lines at a hole saying what was tried there, what Lean rejected, and the goal's structural reading (§S9.2). |
| **repair turn** | The single Goedel call per hand-off in its trained self-correction format (decision A; §S9.1). |
| **environment fingerprint** | `sha256` of: Lean version string, Mathlib revision from `data/lean_project_v428/lake-manifest.json`, and the signature-dump program's source. Keys every cache and index. |
| **P1, P2, P3** | Evaluation sets (§S0.5–S0.6 and §5). |
| **PRESERVED / CHANGED / UNRESOLVED / INVALID MEASUREMENT** | Behavior verdicts of a paired check (owner's process contract). |

---

## 4. Shared data shapes

### 4.1 Outcome codes (`src/resolution_codes.py`, new; string constants)
`CLOSED`, `PROGRESS_ONLY`, `CANDIDATE_REJECTED`, `ROLE_UNDERCONSTRAINED`, `LOOKUP_UNRESOLVED`,
`PARSE_UNSUPPORTED`, `REPLAY_UNAVAILABLE`, `INFRA_FAILURE`, `BUDGET_EXHAUSTED`, `NO_SLOTS`.
`INFRA_FAILURE` and `BUDGET_EXHAUSTED` are never evidence against a candidate, a slot or a lane.

### 4.2 Step IR (`src/step_compiler.py`)
```
Ident      = {text, kind: "local_hyp"|"local_decl"|"env_const"|"unresolved", resolved_name|None}
Slot       = {slot_id, kind: "const"|"rewrite"|"unfold"|"hint"|"arg", tactic_index, ident: Ident,
              args: [TermExpr] (const), locations: [str] (rewrite/unfold; "⊢" = target),
              direction: "forward"|"backward" (rewrite), excluded: [name] (a rejected constant),
              decoded: [Claim] (filled in S3)}
TacticIR   = {index, text, family, action: ActionIR|None, outcome: str, slots: [Slot]}
StepIR     = {step_text, tactics: [TacticIR], outcome: str}
```
`ActionIR` gains `locations: list[str]` (for `… at h ⊢`) and `hint_terms: list[TermExpr]`
(`linarith`/`nlinarith`/`positivity` lists); `src/tactic_ir.lower()` renders them. Existing callers
of `ActionIR` must see no change when the new fields are empty.

### 4.3 Signature record (one JSON object per declaration; produced by the Lean dump, §S1)
```
{ "name", "kind": "theorem"|"def"|"axiom"|"inductive"|"ctor"|"recursor"|"opaque"|"quot",
  "module", "file", "range": [start_line, start_col, end_line, end_col] | null,
  "type_pp",                       # pretty-printed with default options
  "binders": [{"name","info":"explicit"|"implicit"|"inst"|"strict","is_prop":bool}],
  "concl_rel",                     # head constant of the conclusion (after the ∀/→ telescope)
  "concl_carrier",                 # head constant of the relation's type argument, or null
  "concl_lhs_head", "concl_rhs_head",   # heads of the last two explicit args (null if fewer)
  "concl_lhs_args": [head, …],     # heads of the lhs's explicit args (rewrite rules)
  "prem_rels": [head of each explicit Prop binder's type, in order],
  "prem_carriers": [carrier of each, or null],
  "norm": {"concl_rel": [...], "prem_rels": [[...], …]},  # normalized views, §S1.2
  "type_consts": [sorted constants in the type],
  "value_consts": [sorted constants in the value] | null   # theorems/defs only; S11
}
```
A head is `Name.toString` of a constant (`"Finset.card"`); `"_var"` for a bound or free variable;
`"_sort"`, `"_forall"`, `"_lam"`, `"_lit"`, `"_other"` otherwise.

### 4.4 Ledger row (`resolution_ledger.jsonl` in each hand-off run directory; S4)
`{run_id, sample_id, hole, step_text, step_ir (serialized), slots: [{slot_id, kind, ident,
role_features, decoded, windows: {role: n, name: n, union: n}, pages_read, questions: [...],
probes: [{candidate, fill, outcome, lean_error_kind}], chosen|null, outcome}], block_outcome,
elapsed_s, env_fingerprint}`. Written by the orchestration layer only; pure parts return records.

---

## 5. Current state (measured 2026-10-30; recheck in S0)

- Revision `3d3a384`; the local hand-off loop is built but **uncommitted** (`src/handoff_loop.py`,
  `src/hole_search.py`, `scripts/search_holes.py`, `tests/test_handoff_loop.py`,
  `tests/test_hole_search.py`, `tests/test_mcp_pid_alive.py` untracked; `src/local_pipeline.py` and
  `tests/test_local_pipeline.py` deleted; seven tracked files modified). Gate green on that tree.
- `.venv` Python 3.11.14; PyPantograph 0.3.14; Lean v4.28.0 in `data/lean_project_v428`; persistent
  REPL `data/repl_v428` (`src/lean_repl.py`: `LeanRepl.check(text)`, `ReplPool`).
- Network `data/proof_network_v3.db`, built from an older LeanDojo trace of Mathlib.

### 5.1 The gaps this build closes

| # | Gap (where) | Evidence |
|---|---|---|
| G1 | Lemma-side geometry regex-extracted from source text (`scripts/extract_proof_network.py::extract_premise_entity`: four banks zero by construction, STRUCTURE from an arrow count); all 163,343 `premise_only` lemmas have an empty `type_pp`; `Finset.card_le_card` has no type anchor | network queries |
| G2 | Network older than the checker's Mathlib: `Finset.card_le_of_subset`, `true_and_iff` absent; `Finset.card_le_card`, `true_and` present; 16,753 network lemmas absent from 4.28 (previous session's count) | network queries |
| G3 | Dr. Ducky's compatibility rules unbuilt: `ProofHoleSpec.compatibility` is never read by `_hole_candidates`; exact/apply holes take the lemma list unchecked | `src/dr_ducky.py`, `src/dr_ducky_executor.py` |
| G4 | Name-piece rankers where types should decide: `_lemma_token_score`; `scope_for_apply` tiers 2–3 | `src/dr_ducky_executor.py`, `src/apply_scoper.py` |
| G5 | Goedel's failed steps consumed as text: `seeded_closers` re-runs a one-line step verbatim and feeds bracket lists to `simp`/`norm_num`/`linarith` (the specimen's seeds carried the typo `survingDOF`); the in-place search receives no step intent; `tactic_canonicalizer` calls an identifier a hypothesis iff it is lowercase and ≤ 3 characters | `src/hole_fill.py`, `src/handoff_loop.py`, specimen run |
| G6 | No operator asks the disambiguating question that the AMBIGUOUS verdict calls for | `src/structural_reads.py::interference` |
| G7 | `exact?`/`apply?` closes recorded as the literal tactic (`_close_goal`: `lean_tactic = tactic`); fills and the gate deny those tokens | `src/proof_search.py` (read in code; S0 observes) |
| G8 | The file's own definitions opaque to the structural query (closers unfold them; the query does not) | specimen hole open after 67 attempts |
| G9 | The holographic scorer (encoder, first 20,000 typed lemmas, `LIMIT` without `ORDER BY`) is the only type-signature matcher; `search()`'s `holographic_premises` is never forwarded to Dr. Ducky by `_try_dr_ducky` | `src/hardtail_bridge.py`, `src/proof_search.py` |
| G10 | **The "accessible premises" table is each theorem's own proof premises.** All 339,530 rows of `accessible_premises` equal the `depends_on` links pair for pair (68,962 theorems, mean 4.9 premises); `get_accessible_premises` returns it as the **site pool** for a replayed Mathlib theorem. The deterministic Mathlib-split runs used site pools (12,209 times in the 1,028 run, per THE_REFOUNDING §9). MiniF2F and the owner's theorems have no site and are unaffected. | `sqlite3` join, `src/proof_network.py`, `src/proof_search.py::_symbolic_rank_premises` |

G9 is not revived (owner's ruling); its job passes to the signature index. G10 is fixed by the
rebuild (S11); until then, no new Mathlib-split number is reported as comparable to the old ones.

### 5.2 Evaluation sets
- **P1** — the resolution population, built in S0.6.
- **P2** — the adversarial fixtures F1–F10, written in S0.6.
- **P3-core** — the six search check theorems of the hand-off pass: `mathd_algebra_107`,
  `amc12a_2021_p9`, `aime_1983_p2` (rows of `data/minif2f_test.jsonl`); `sheaf_implies_local_objective`
  (`data/gap_T7_18_obstruction_intrinsic_manifest.jsonl`), `kappa_additive_bound`
  (`data/gap_T7_17_sheaf_gluing_manifest.jsonl`), `pigeonhole_cover`
  (`data/gap_T5_12_greedy_harmonic_manifest.jsonl`). 30 s per MiniF2F search, 75 s per owner theorem
  (as in that pass). Verdict: PRESERVED = the same closed set; CHANGED = list every difference.
- **P3-manifest** — `data/rerun_v427_manifest.jsonl` (21 theorems, 75 s each, ~26 min per arm), for
  steps that touch the search.
- **Deep checks** (owner's go-ahead only): MiniF2F-test in full; the Mathlib split.

Paired runs use `scripts/run_benchmark.py` exactly as `scripts/search_holes.py::search` invokes it
(read that function whole), with `--theorems <subset manifest>`; build the P3 subset manifests once in
S0 with `scripts/build_p3_manifests.py` (exact `theorem_id` match against the files named above) into
`data/eval/p3_core.jsonl`.

---

## 6. The build

Each step: **read first** (whole units), **change**, **contract** (what the step deliberately
changes), **tests** (authored; MUST-NOT fences included), **check**, **commit**. Lean-touching tests
follow the skip pattern of `tests/test_hole_search.py` (read it whole before writing the first one).

### S0 — Baseline, housekeeping, observations, provenance, evaluation sets

**S0.1** Create `runs/structural_resolution_build/STEP_REPORTS.md` (heading + §2.5's fields as a
template). `runs/` and `data/` are gitignored (`.gitignore` lines 11–12): step reports, consult papers,
evaluation populations and the 4.28 artifacts stay local; scripts, source, tests, fixtures and docs
are committed.

**S0.1b Commit this specification:** `git add docs/STRUCTURAL_RESOLUTION_PIPELINE.md`; commit
`docs: structural resolution and the 4.28 rebuild — implementation specification`.

**S0.2 Commit the existing hand-off work.** Run the gate. Stage exactly these eleven tracked paths —
modified: `README.md`, `WAYFINDER_MCP_NOTES.md`, `docs/Research_Paper/WAYFINDER_DESIGN.md`,
`scripts/run_benchmark.py`, `src/lean_interface.py`, `src/local_prover.py`,
`tests/test_mcp_proof_home.py`, `tests/test_mcp_proofs_lakefile.py`, `wayfinder_mcp.py`; deleted:
`src/local_pipeline.py`, `tests/test_local_pipeline.py` — and these six new files:
`src/handoff_loop.py`, `src/hole_search.py`, `scripts/search_holes.py`, `tests/test_handoff_loop.py`,
`tests/test_hole_search.py`, `tests/test_mcp_pid_alive.py`. Never `tests/detective/certificates.json`.
Commit: `Local hand-off loop: Lean takes Goedel's answers apart, Wayfinder searches each hole where it
sits`.

**S0.3 Remove the AI trailers (decision F).**
1. `git status --short` must list nothing but `tests/detective/certificates.json`.
2. `git rev-list 8fad0f4..HEAD` must contain `e74f51e`, `e662f16`, `bfced50`, `bf83239`, `1b17c69`,
   `2304562`, `3d3a384` and the S0.1b and S0.2 commits — nine commits. Otherwise stop.
3. `git update-ref refs/backup/pre-trailer-rewrite-20261030 HEAD`.
4. Write `/tmp/strip_coauthor.py`:
   ```python
   import sys
   lines = [l for l in sys.stdin.read().split("\n") if not l.startswith("Co-Authored-By:")]
   sys.stdout.write("\n".join(lines).rstrip("\n") + "\n")
   ```
5. `git filter-branch -f --msg-filter 'python3 /tmp/strip_coauthor.py' 8fad0f4..HEAD`.
6. Verify: `git diff refs/backup/pre-trailer-rewrite-20261030 HEAD` prints nothing (identical trees);
   `git log --format=%B 8fad0f4..HEAD` (read whole) contains no `Co-Authored-By`; still nine commits.
7. `git push --force-with-lease origin main`.
8. Report, without rewriting: `git log --format='%h %(trailers:key=Co-Authored-By,valueonly,separator=%x2C)' 8fad0f4`
   (read whole) — list older commits that carry the trailer; the owner decides about them.

**S0.4 Prune the worktrees (decision F).** `git worktree list` (read whole). For each worktree under
`/Users/rohanvinaik/.claude/jobs/cc620914/tmp/`: if its directory exists, `git -C <path> status
--short` must be empty (else stop), then `git worktree remove <path>`. Then `git worktree prune`;
`git worktree list` must show only the main tree. List any local branches the worktrees used; delete
none without the owner.

**S0.5 Observations** → `runs/structural_resolution_build/S0_observations.md`, one section each:
- **U1 (G7).** Fixture `tests/fixtures/resolution/exact_only/file.lean`:
  ```lean
  import Mathlib

  theorem wf_exact_only (s t : Finset ℕ) (h : s ⊆ t) : s.card ≤ t.card := by
    -- WAYFINDER RESIDUAL 1: close this goal
    sorry
  ```
  Run `scripts/search_holes.py` on a request shaped like
  `runs/local_prover/survivingDOF_antitone-20261030-153604/holes-wayfinder.json` (read it whole; same
  keys; `row` built the same way for this file). Record the returned block and whether it contains
  `exact?` or `apply?`.
- **U2 (capture).** (a) PyPantograph `Server(imports=["Mathlib"], project_path="data/lean_project_v428")`:
  `goal_start` of `∀ (s t : Finset ℕ), s ⊆ t → s.card ≤ t.card`, `goal_tactic` `intro s t h`, then
  `exact?`; record success, remaining goals, and `goal_root(state)` printed. (b) `LeanRepl.check` on the
  fixture with `exact?` in place of `sorry`; record every message (severity, data). Conclude which
  route yields a term that re-elaborates as `exact <term>`.
- **U3.** `env_inspect("Finset.card_le_card")` through Pantograph: the result's keys and the `type`
  object's keys.
- **U4.** `.venv/bin/python -m pytest tests/ -q -rs -p no:cacheprovider --junitxml
  runs/structural_resolution_build/S0_skips.xml`; tabulate skip reasons with counts (Python over the
  XML).
- **U5 (tracer).** `Server.tactic_invocations` on
  `data/lean_project_v428/.lake/packages/mathlib/Mathlib/Data/Finset/Card.lean`: wall time; number of
  units and invocations; the field names of one invocation; whether used constants per tactic and the
  enclosing declaration are present.
- **U6.** `env_parse("exact Finset.card_le_card h", "tactic")`: the output (informational).

**S0.6 Censor provenance (D4).** Read whole: `scripts/mine_censors.py`, `src/censor_store.py`, and the
`censor_store_path` loading site in `src/proof_search.py` (Serena references to `load_fences`; the
import is function-local). Change: (a) the miner writes top-level `"mined_at_commit"` (`git rev-parse
HEAD`) into the store JSON; (b) new pure `censor_store.staleness(mined_at_commit, changed_paths) ->
"fresh" | "stale" | "unknown"`; (c) at load: changed paths = `git diff --name-only
<mined_at_commit> HEAD -- src/` (none → fresh; field missing → unknown); log a warning when not
fresh; config `search.censor_store_stale_policy: warn | refuse` (default `warn`; `refuse` loads no
fences); the run report records `censor_store_staleness`. Tests `tests/test_censor_provenance.py`:
the three verdicts; `refuse` yields no fences on stale; the miner's writer includes the field.
Detective-pinned: `staleness`. Commit: `Censors: a store records the commit it was mined at, and the
loader says when the prover has changed since`.

**S0.7 Evaluation sets.**
- `scripts/build_resolution_population.py`: walk `runs/local_prover/` and `runs/minif2f_v2/` for every
  `samples.jsonl`; keep rows with `outcome == "wrong_proof"` and non-empty `errors`; class
  `first_unknown` if `errors[0]` contains "Unknown identifier" or "Unknown constant"; `any_unknown` if
  any error does; `type_mismatch_first` if `errors[0]` starts with "Type mismatch", "type mismatch",
  "Application type mismatch", or contains "Function expected". Answer text:
  `<run dir>/outputs/<id>.txt` if present, else `answer_missing: true`. Theorem: the run directory's
  `job.json` `theorem_id`. Split: `dev` if `int(sha256(theorem_id).hexdigest(), 16) % 2 == 0` else
  `held_out`. Write `data/eval/resolution_population_v1.jsonl` sorted by (`theorem_id`, `id`) and
  `data/eval/resolution_population_v1_summary.json` (rows and **distinct theorems** per class × split;
  the largest single-theorem share per class). Rerun must be byte-identical (test). `data/` is
  ignored, so the population is frozen by its committed script plus its sha256, recorded in the S0
  step report and in this document's status line.
- `scripts/build_p3_manifests.py` (§5.2) → `data/eval/p3_core.jsonl`.
- **P2 fixtures** in `tests/fixtures/resolution/<name>/` — `file.lean` (the hand-off file: imports,
  definitions, the theorem with the failing step in place), `step.txt` (the failing step), and
  `expect.json`:

  | Fixture | Content | Expected (S3 resolver) |
  |---|---|---|
  | F1a `specimen_subset` | the specimen's `file` (its `job.json`) with the theorem body = answer `t-residual-r0-s2`'s proof (last `lean4` block of `outputs/t-residual-r0-s2.txt`), the `h_main` proof replaced by `sorry`; the hole = the `h_subset` proof; `step.txt` = that proof's lines | CLOSED; the fill uses `true_and` |
  | F1b `specimen_main` | the same file with the `h_subset` proof replaced by `grind`; the hole = the `h_main` proof; `step.txt` = its lines (`rw [survingDOF]`, `rw [survingDOF]`, the comment, `exact Finset.card_le_of_subset h_subset`) | CLOSED; the fill uses `survivingDOF` and `Finset.card_le_card` |
  | F2a / F2b `herpderp` | F1a with `true_and_iff` → `blarg_17`; F1b with `Finset.card_le_of_subset` → `HerpDerp_theorem_47` and `survingDOF` → `xq_zz_9` | both CLOSED |
  | F3 `misleading` | F1b with `Finset.card_le_of_subset` → `Finset.card_lt_of_ssubset` | CLOSED |
  | F4 `renamed_candidate` | `def wfP (n : ℕ) : Prop := n % 7 = 3`; `theorem HerpDerp_theorem_47 (n : ℕ) (h : n % 7 = 3) : wfP n := h`; `theorem wf_f4 (n : ℕ) (h : n % 7 = 3) : wfP n := by exact wfP_of_mod h` | `HerpDerp_theorem_47` on the first page; its probe CLOSED |
  | F5 `dependent` | `theorem wf_f5 : ∃ n : ℕ, n = n := by exact missing_intro 3 rfl` | CLOSED, ROLE_UNDERCONSTRAINED or BUDGET_EXHAUSTED; never a wrong acceptance |
  | F6 `two_location_simp` | `theorem wf_f6 (p q : Prop) (h : True ∧ p) (hq : p → q) : True ∧ q := by simp only [missing_rule] at h ⊢; exact hq h` | CLOSED with `true_and` |
  | F7 `impossible_transport` | `theorem wf_f7 (a b c : ℕ) (h : a + b - c = 0) : a + b = c := by exact transport_lemma h` | never CLOSED |
  | F8 `continuation` | F2b run with `resolution_page_size = 1`, `resolution_max_pages = 64` | CLOSED, with `pages_read` equal to the merged-order rank of the first candidate that closes |
  | F9 `opaque_definition` | F1b with the step reduced to `exact Finset.card_le_of_subset h_subset` | S3: any outcome; after S6: CLOSED |
  | F10 `rejected_constant` | F1b with `Finset.card_le_of_subset` → `Finset.card_lt_card` (a real constant Lean rejects here) | CLOSED with `Finset.card_le_card`; slot `excluded` = [`Finset.card_lt_card`] |

Commit S0.7: `Evaluation: the resolution population and its adversarial fixtures, frozen`.

### S1 — Signatures computed in Lean; the isolated signature index

**Read first:** `src/lean_repl.py` whole; `data/lean_project_v428/lakefile*` and `lake-manifest.json`.

**S1.1 The dump program** `lean/signature_dump/dump_template.lean`: `import Mathlib`, `open Lean Meta`,
`set_option maxHeartbeats 0`, the definitions below, and a final `#eval` line with placeholders
`{{OUT}}`, `{{FILTER}}`, `{{VALUES}}`. Definitions:
- `wfSignature (ci : ConstantInfo) (withValue : Bool) : MetaM Json` — one record of §4.3.
- `wfDumpAll (out : System.FilePath) (moduleFilter : Option Name) (withValue : Bool) : MetaM Unit` —
  folds `(← getEnv).constants` (both maps), skips names for which `Name.isInternalDetail` holds (if
  the name differs at 4.28, use what `exact?`'s candidate filter in Lean's library search uses — read
  that source), skips modules not under `moduleFilter` when given, writes JSONL.
- `wfDumpNames (names : Array Name) : MetaM Unit` — `logInfo` one JSON line per name (for the REPL
  route: local declarations and lifted goals).
- `wfDumpImports (out : System.FilePath) : MetaM Unit` — one JSON line per module:
  `{"module", "imports": [...]}` from the environment header (S11).

**S1.2 Feature semantics** (must hold exactly):
- Telescope: `forallTelescope ci.type` (no reduction). Binder `info` from each local declaration;
  `is_prop` = `isProp (← inferType x)`.
- Heads: `e.getAppFn` — a constant gives its name; bvar/fvar `"_var"`; sort `"_sort"`; forall
  `"_forall"`; lambda `"_lam"`; literal `"_lit"`; else `"_other"`. Explicit arguments: the application
  arguments whose parameter is explicit per `getFunInfoNArgs`.
- `concl_rel` = head of the conclusion. `concl_lhs_head`/`concl_rhs_head` = heads of the last two
  explicit arguments (one explicit argument: lhs null, rhs its head; none: both null).
  `concl_lhs_args` = heads of the lhs's explicit arguments, positional. `concl_carrier` = head of
  the type of the conclusion's last explicit argument, for `concl_rel` in {`LE.le`, `LT.lt`, `GE.ge`,
  `GT.gt`, `Eq`, `Ne`, `HasSubset.Subset`, `HasSSubset.SSubset`, `Membership.mem`, `Dvd.dvd`}; null
  otherwise. `prem_rels`/`prem_carriers` = the same reads on each explicit Prop binder's type, in order.
- `norm`: lists that include the original value plus: (i) aliases — (`LE.le`, carrier `Finset` or
  `Set`) ↔ `HasSubset.Subset`; (`LT.lt`, carrier `Finset` or `Set`) ↔ `HasSSubset.SSubset`; `GE.ge` →
  `LE.le`; `GT.gt` → `LT.lt`; (ii) exposure — if the conclusion's head is one of `Monotone`,
  `Antitone`, `StrictMono`, `StrictAnti`, `Function.Injective`, `Function.Surjective`, unfold it once
  (`unfoldDefinition?`), telescope again, and add that form's conclusion relation, lhs/rhs heads and
  premise relations (with aliases). Keys: `norm.concl_rel`, `norm.concl_lhs_head`,
  `norm.concl_rhs_head`, `norm.prem_rels` (list per premise; exposed premises appended).
- `type_consts` = sorted unique constants of the type; `value_consts` (only with `withValue`) = sorted
  unique constants of the value for theorems and definitions, else null.
- `module` from `getModuleIdxFor?`; `file` = module name with `.` → `/` plus `.lean`; `range` from
  `findDeclarationRanges?`; `type_pp` = `ppExpr` with default options.

**S1.3** `scripts/build_signature_dump.py` — fills the template, writes it to a temporary directory,
runs `lake env lean <file>` with cwd `data/lean_project_v428`, validates the JSONL (parses; count),
writes `data/network428/MANIFEST.json` (fingerprint, Lean version, Mathlib revision, counts, wall
time). `--module-filter`, `--with-values`, `--imports`. Add `data/network428/` to `.gitignore` if it is
not ignored (check with `git check-ignore -v data/network428/x`).

**S1.4** `src/signature_index.py`:
- `build_index(dump_jsonl, db_path, fingerprint) -> dict` — SQLite tables
  `meta(key PRIMARY KEY, value)`; `decl(id INTEGER PRIMARY KEY, name UNIQUE, kind, module, file, line,
  type_pp, explicit_arity, record)`; `feature(decl_id, family, value, PRIMARY KEY(decl_id, family,
  value))` indexed on `(family, value)`; `feature_idf(family, value, df, idf, PRIMARY KEY(family,
  value))` with `idf = ln(N / df)`, N = declarations. Families: `concl_rel`, `concl_lhs_head`,
  `concl_rhs_head` (from the `norm` lists), `concl_lhs_arg` (value `"{i}:{head}"`), `concl_carrier`,
  `prem_rel` (all premises, from `norm`), `prem_carrier`, `type_const`, `type_const_ns` (each
  namespace prefix of each type constant, e.g. `Finset`), `head_last` (last name component of each
  `concl_lhs_head`/`concl_rhs_head` value, e.g. `card`).
- `exists(db, name) -> bool`; `record(db, name) -> dict | None`.
- `window(db, require, prefer, kinds, exclude, page, page_size=32) -> list[Candidate]` — `require`:
  list of `(family, set_of_values)`, each satisfied when the declaration has any of the values;
  `prefer`: list of `(family, value)`; score = sum of `idf` over the prefer features present; order by
  (−score, name); `page` is 0-based; `Candidate = (name, score, matched_features)`. Deterministic.
- `env_fingerprint(project_dir, template_path) -> str` (§3).
- Kinds for windows: `const` slots → `theorem`, `ctor`; `rewrite` → `theorem`; `unfold` → `def`.

**S1.5 Measure, then build.** Run the dump with `--module-filter Mathlib.Data.Finset.Card`; record
time per declaration; project the whole catalog. Over 30 minutes → stop and ask (§2.2). Then the full
dump (without values) and `build_index` → `data/network428/signatures.db`.

**Tests** `tests/test_signature_index.py`. Pure (Detective-pinned: `window`, the IDF computation): a
hand-written six-record fixture (Finset.card_le_card, Finset.card_mono, Finset.card_lt_card,
true_and, and_true, Exists.intro) built to §4.3:
- `window(require=[("concl_rel", {"LE.le"})], prefer=[("concl_lhs_head","Finset.card"),
  ("concl_rhs_head","Finset.card"), ("prem_rel","HasSubset.Subset")])` → `Finset.card_le_card`,
  `Finset.card_mono` (in that order; `Finset.card_lt_card` absent);
- `window(require=[("concl_rel", {"Eq","Iff"})], prefer=[("concl_lhs_head","And"),
  ("concl_lhs_arg","0:True")])` → `true_and` before `and_true`;
- building twice gives identical `feature` and `feature_idf` rows.
Lean integration (skip pattern): `wfDumpNames` through the REPL on `Finset.card_le_card`, `true_and`,
`Finset.card_mono` returns: card_le_card — `concl_rel` `LE.le`, `concl_lhs_head` and `concl_rhs_head`
`Finset.card`, `prem_rels` `["HasSubset.Subset"]`, `concl_carrier` `Nat`; true_and — `concl_rel`
`Eq`, `concl_lhs_head` `And`, `concl_lhs_args` `["True", "_var"]`; card_mono — `concl_rel`
`Monotone`, `"LE.le"` in `norm.concl_rel`, `"HasSubset.Subset"` in `norm.prem_rels[0]`. If a pinned
value differs, stop (§2.4).

**Commit:** `Signature index: lemma signatures computed in Lean from elaborated types at 4.28, kept
apart from the network`.

### S2 — The step compiler

**Read first:** `src/tactic_ir.py`, `src/tactic_canonicalizer.py`, `src/hole_fill.py` (all whole).

**Change** — new `src/step_compiler.py` and `src/resolution_codes.py`; `ActionIR` gains `locations`
and `hint_terms` (§4.2; `lower()` renders `at …` and hint lists; empty fields render exactly as
before):
- `split_tactics(step_text) -> list[Tactic]` — dedent; a line at the block's base indentation starts
  a tactic, deeper lines continue it; top-level `;` (outside brackets/parentheses) splits; `<;>` does
  not; comment lines are dropped; a `·` bullet with its body is one unit of family `bullet`; record
  each tactic's line/column span in the step.
- `parse_tactic(text) -> TacticIR` — family by first token: `exact`, `apply`, `refine`, `rw`/`rwa`/
  `rewrite`, `simp`, `simp only`, `simpa`, `simp_all`, `dsimp`, `unfold`, `linarith`, `nlinarith`,
  `positivity`, `norm_num`, `field_simp`, `have`, `intro`/`intros`/`rintro`, `obtain`/`rcases`/
  `cases`; others keep `action = None`. Terms via `tactic_canonicalizer._parse_term` and
  `_split_top_level`; record each identifier's span. `at` suffixes → `locations`. A one-line `have
  name : T := by <tac>` or `have name : T := <term>` is parsed as: the claim `T` (kept verbatim; its
  identifiers are resolved like any others), and the nested `<tac>` parsed recursively as a tactic or
  `<term>` as a term. `intro`/`intros`/`rintro` names and `obtain`/`rcases`/`cases … with` pattern
  names are binders, not identifiers to resolve.
- `resolve_identifiers(tactic_ir, ctx, bound)` — `ctx = ResolutionContext(local_hyps, local_decls,
  opens, catalog_exists)`; `bound` = the names bound by **earlier tactics of the same step**
  (`intro`/`intros`/`rintro` names and patterns, `obtain`/`rcases`/`cases` pattern names, `have`/`let`/
  `set` names), which count as hypotheses for this tactic. Identifiers: heads and leaves of terms,
  rewrite/simp atoms, hint terms, `have` claims. Skipped: numerals, `_`, `?_`, names bound inside the
  term (`fun x =>`, `∀ x,`, `∃ x,`), projection fields (`.1`, `.2`, `.mp`, `.mpr`, `.symm`, `.trans`,
  `.le`, `.lt`, `.left`, `.right`) — their base is resolved. Order: `local_hyps` ∪ `bound` →
  `local_decls` → `catalog_exists(name)` → `catalog_exists(f"{ns}.{name}")` for each `ns` in `opens`,
  first hit → else unresolved. The old lowercase-≤3-characters rule is not used here (leave
  `canonicalize()` itself unchanged for its other callers).
- `build_slots(tactic_ir, lean_errors)` — kind by position: application head or bare term under
  `exact`/`apply`/`refine` → `const` (with the application's args); atom of `rw`/`rwa` → `rewrite`
  (direction, locations); atom of `simp`/`simp only`/`simpa`/`dsimp`/`simp_all` → `rewrite`
  (locations); atom of `unfold` → `unfold`; term of a `linarith`/`nlinarith`/`positivity` list →
  `hint`; an argument → `arg`. A Lean error of this step whose message starts with "Application type
  mismatch", "Type mismatch" or "type mismatch", or contains "Function expected", and whose position
  falls on a resolved catalog constant in head or bare-term position under `exact`/`apply`/`refine`,
  turns that constant into a `const` slot with `excluded = [it]`.
- `compile_step(step_text, ctx, lean_errors=()) -> StepIR` — outcome `NO_SLOTS`; `PARSE_UNSUPPORTED`
  (a slot inside a `bullet`, a multi-line `have` body, `calc`, `match`, or a family with
  `action = None`);
  `ROLE_UNDERCONSTRAINED` (a tactic with more than two slots); else `LOOKUP_UNRESOLVED` (slots to
  resolve).

**Tests** `tests/test_step_compiler.py` (Detective-pinned: `split_tactics`, `resolve_identifiers`,
`build_slots`). With a fake catalog `{Finset.card_le_card, Finset.mem_filter, Finset.mem_univ,
Finset.card_lt_card, sq_nonneg}`:
- the specimen block `rw [survingDOF]` / `rw [survingDOF]` / `-- comment` / `exact
  Finset.card_le_of_subset h_subset`, ctx `local_hyps={D, lam, lam', t, h, h_subset}`,
  `local_decls={survivingDOF}`, `opens=[Finset]` → three tactics; slots: `rewrite(survingDOF)`,
  `rewrite(survingDOF)`, `const(Finset.card_le_of_subset, args=[h_subset as local_hyp])`;
- `simp only [Finset.mem_filter, Finset.mem_univ, true_and_iff] at hi ⊢` → one `rewrite` slot,
  `locations = ["hi", "⊢"]`;
- the specimen's `h_subset` body — `intro i hi` / `simp only [Finset.mem_filter, Finset.mem_univ,
  true_and_iff] at hi ⊢` / `-- comment` / `have h₁ : lam' i < t := by simpa using hi` / `have h₂ :
  lam i ≤ lam' i := h i` / `have h₃ : lam i < t := by linarith` / `exact h₃`, ctx
  `local_hyps={D, lam, lam', t, h}` → outcome `LOOKUP_UNRESOLVED` with exactly one slot,
  `rewrite(true_and_iff)` at `["hi", "⊢"]`; `i`, `hi`, `h₁`, `h₂`, `h₃` resolve as step-bound names;
- `h_subset` resolves as a hypothesis; `card_le_card` under `open Finset` resolves to
  `Finset.card_le_card`;
- `exact Finset.card_lt_card h_subset` with an "Application type mismatch" error on the constant →
  `const` slot, `excluded = ["Finset.card_lt_card"]`;
- `nlinarith [sq_nonneg (a - b), mystery_bound a]` → one `hint` slot (`mystery_bound`);
- MUST-NOT: replacing an unresolved identifier with any other unresolved string yields the same slots
  (kinds, positions, args) — spelling never matters.

**Commit:** `Step compiler: a failed step's names resolved by lookup; what resolves to nothing
becomes a typed slot`.

### S3 — Slot resolution

**Read first:** `src/local_prover.py` (`Worker.compile`, `_repl_pool`, `skeleton_repair`), the
lifting functions of `src/hole_fill.py`, `src/lean_repl.py`.

**S3.1 Context** (`src/slot_resolution.py`): `HoleContext(skeleton_decl, k, above, theorem_name,
compile, defs, helper_names, opens, fingerprint)`. `local_hyps` = names of the binders of hole k's
lifted statement (`hf.lifting_probe` compiled → `hf.lifted_variants(...)[k][0]` → `hf.binders`).
`local_decls` = `defs` ∪ `helper_names` ∪ `hf.declared_names(above)`.

**S3.2 Roles.** For a slot in tactic i: compile the skeleton with hole k's `sorry` replaced by tactics
0..i−1 (current assignment) followed by `extract_goal * using <name>_wf_role` and `sorry`; take the
lifted statement S; compile `above` + the dump definitions + `theorem __wf_role : <hf.closed_type(S)>
:= by sorry` + `#eval wfDumpNames #[``__wf_role]`; parse the record R. Queries:
- `const` (exact/apply/refine): require `("concl_rel", R.norm.concl_rel)`; prefer `concl_lhs_head`
  and `concl_rhs_head` values from R's norm lists, `concl_carrier`, and for each argument that is a
  hypothesis h, `("prem_rel", v)` for v in the norm relations of h's premise in R.
- `rewrite`: unfold candidates first — every constant of `R.type_consts` that is a local
  definition or a catalog `def` (order of first appearance in `R.type_pp`); then lemmas: require
  `("concl_rel", {"Eq","Iff"})`, prefer `("concl_lhs_head", c)` and `("concl_lhs_arg", f"{i}:{c}")`
  (i = 0..3) for every c in `R.type_consts`.
- `unfold`: the unfold candidates only.
- `hint`: the hypotheses first; then require `("concl_rel", {"LE.le","LT.lt","Eq","Ne"})`, prefer
  `concl_carrier` and `("type_const", c)` for c in `R.type_consts`.
- `arg`: the hypotheses, in context order.
Local declarations enter every const/rewrite window: dump them once per hand-off file through
`wfDumpNames` and score them with the same query.

**S3.3 Name decoder** `src/name_decoder.py` (pure; Detective-pinned): `decode(identifier) ->
list[(family, value)]`. Split on `.`; every proper prefix of the namespace path gives
`("type_const_ns", prefix)`; the last component, lowercased, split on `_` into words; words before the
first `of` are conclusion words, after it premise words. Relation words → `("concl_rel", X)` (or
`("prem_rel", X)` after `of`): `le`→`LE.le`, `lt`→`LT.lt`, `ge`→`LE.le`, `gt`→`LT.lt`, `eq`→`Eq`,
`ne`→`Ne`, `iff`→`Iff`, `subset`→`HasSubset.Subset`, `ssubset`→`HasSSubset.SSubset`,
`mem`→`Membership.mem`, `dvd`→`Dvd.dvd`, `nonneg`→`LE.le`, `nonpos`→`LE.le`, `pos`→`LT.lt`,
`neg`→`LT.lt`, `inj`/`injective`→`Function.Injective`, `surjective`→`Function.Surjective`,
`mono`/`monotone`→`Monotone`, `antitone`→`Antitone`. Connective words: `and`→`("concl_lhs_head",
"And")`, `or`→`("concl_lhs_head", "Or")`, `not`→`("concl_lhs_head", "Not")`, `true` directly before
`and`→`("concl_lhs_arg", "0:True")`, `true` directly after `and`→`("concl_lhs_arg", "1:True")`. Head
words → `("head_last", w)` for w in {card, sum, prod, filter, map, image, range, length, sup, inf, max,
min, abs, sqrt, exp, log, norm, dist, gcd, lcm, succ, pred, pow, mul, add, sub, div, mod, inv, union,
inter, sdiff, insert, erase, compl, univ, empty, zero, one, natabs, tofinset, floor, ceil}. Any other
word gives nothing. Order: namespace claims, then conclusion words, then premise words; duplicates
removed. Tests: `decode("Finset.card_le_of_subset") == [("type_const_ns","Finset"),
("head_last","card"), ("concl_rel","LE.le"), ("prem_rel","HasSubset.Subset")]`;
`decode("true_and_iff") == [("concl_lhs_arg","0:True"), ("concl_lhs_head","And"),
("concl_rel","Iff")]`; `decode("HerpDerp_theorem_47") == []`; `decode("survingDOF") == []`.

**S3.4 Windows and order.** Role window: `window(require, prefer, …)`; name window: `window(require=[],
prefer=decode(ident), …)`; both exclude the slot's `excluded`. Merge order (pure; Detective-pinned):
(1) unfold candidates; (2) names in both first pages, by role score; (3) role-only, by role score;
(4) name-only, by name score; ties by name. Read further pages as needed: page size
`resolution_page_size` (32), at most `resolution_max_pages` (8) per window, then `BUDGET_EXHAUSTED`.

**S3.5 Assignments and probes.** `resolve_block(step_ir, ctx, budget) -> BlockResult`: depth-first
over the slots in tactic order, each slot's candidates in merged order, the role of slot j computed
under the assignment of the slots before it (cache roles by `(fingerprint, sha256(probe text))`).
Each full assignment lowers to a fill: the step's tactics with slots replaced at their recorded spans.
`probe_fill(ctx, fill)`: compile the skeleton with hole k = fill and every other hole `exact sorry`
(`hf.only_hole` logic): `CLOSED` if no error in hole k's lines; `PROGRESS_ONLY` if the only errors
there are "unsolved goals"; else `CANDIDATE_REJECTED` with `lean_error_kind` = the first error message
up to its first colon. Cache probes by `(fingerprint, sha256(file text))`. Stop at the first
`CLOSED`. Budgets from the job (`resolution_max_probes` 200, `resolution_wall_s` 120,
`resolution_page_size` 32, `resolution_max_pages` 8). `resolve_candidates(step_ir, ctx, limit=8)` —
the same enumeration without probing, returning ordered fills (used by S4's batteries).

**Tests** `tests/test_slot_resolution.py`: pure — merge order, assignment order, budget accounting;
Lean integration — F1a–F10 against §S0.7's table (F9 exempt). Record probes-to-close for F1b and
F2b: the near-miss name must narrow — `probes(F1b) ≤ probes(F2b)`; if not, stop and report.

**Commit:** `Slot resolution: Lean reads each slot's role; the signature index and the name's reading
localize; Lean's elaboration decides`.

### S4 — Hand-off integration I: compiled steps in `skeleton_repair`

**Read first:** `src/local_prover.py` whole, `src/handoff_loop.py` whole, `src/hole_fill.py` whole,
`src/residual_handoff.py` (`closers`, `triage_block`, `triage_outcomes`, `render`).

**Change** in `Worker.skeleton_repair`, after the final skeleton is renumbered:
- Keep, for every `hf.skeleton` call, the map from each replaced step to the Lean errors that fell in
  its span. Pairing: if the skeleton came from exactly one `hf.skeleton` call, hole k ↔
  `sk.replaced[k-1]`; otherwise every hole is paired with every step.
- For each hole k and each paired step: `compile_step`; if `LOOKUP_UNRESOLVED`, `resolve_candidates`
  (≤ 8 fills per hole). A fill joins the hole's battery as `(t1; t2; …)` only when none of its lines
  contains `by`, `fun`, `=>`, `·`, `<;>` or `calc` (a nested `by` would absorb the tactics after it);
  every other fill is spliced into the hole with `hf.fill_holes` and checked with `probe_fill` before
  the battery runs.
- Per-hole battery: `[closers that are not heavy] + resolved fills of hole k + seeds of hole k +
  [heavy closers]`, where seeds come from `hf.seeded_closers(step)` **only for paired steps whose
  outcome is `NO_SLOTS`**. `probe(k)` uses its own battery.
- `src/repair_ledger.py` (new): `append(run_dir, row)`; one row per (hole, step) (§4.4), with the
  triage outcome of each resolved fill. `skeleton_repair`'s result gains `"resolution"` (each slot's
  outcome).
- `scripts/replay_answers.py <run_dir> [--sample ID] [--out PATH]` (new): rebuilds the run's Worker
  and Hole in-process the way `Worker.run` does (no model server), runs `skeleton_repair` on each
  saved answer (`hf.extract_code`, or `hf.cut_code` for cut answers), and prints one JSON line per
  answer: `{id, outcome, holes, closed_by, resolution}`.

**Contract:** a step with an unresolved name is never re-run or seeded as written; resolved fills are
tried at each hole before the heavy closers. Both repair paths (`Handoff.harvest`,
`Worker.repair_round`) inherit this through `skeleton_repair`.

**Tests** `tests/test_handoff_resolution.py` — stubbed Lean: no verbatim seed from a slot-bearing
step; resolved fills precede heavy closers; batteries differ per hole when steps differ; both callers
reach the new path. Lean integration: `replay_answers.py` on
`runs/local_prover/survivingDOF_antitone-20261030-153604`.

**Check:** the specimen run: at least 7 of 8 answers reach `verified`, and each accepted proof passes
`gate.judge` (report the exact count; fewer than 7 → stop). P1 dev, class `first_unknown`, through
`replay_answers.py` (project the time first; > 30 min → ask): identified / abstained / wrong, where
wrong = a battery close that `gate.judge` rejects (must be 0).

**Commit:** `Hand-off: Goedel's failed steps compiled and resolved; resolved fills join each hole's
battery, and a step with an unknown name is never re-run as written`.

### S5 — Hand-off integration II: the search and Dr. Ducky receive the compiled intent

**Read first:** `scripts/search_holes.py`, `scripts/run_benchmark.py` (`_run_search_loop`,
`_resolve_initial_goal_with_tier`), `src/proof_search.py` (`search`, `_SearchEnv`,
`_symbolic_rank_premises`, `_try_dr_ducky`), `src/dr_ducky_executor.py`
(`build_ducky_symbolic_frontier`, `build_ducky_programs`, `run_ducky_on_goal`,
`_build_binding_space`, `_hole_candidates`), the skeleton builders in `src/dr_ducky.py`.

**Change:**
1. Rename the parameter `holographic_premises` → `seed_premises` in every function and caller
   (Serena references; `src/hardtail_bridge.py` call sites included).
2. `Handoff.search_in_place` adds to its request `"hints": {str(k): {"seed_premises": [...],
   "unfold": [...]}}` from the partial's source answer's ledger rows: for hole k, the first 8
   candidates of its slots' merged windows and the unfold candidates. `hole_rows` sets
   `row["hole_hints"] = hints.get(str(k))`.
3. `_run_search_loop` passes `seed_premises` and `hole_unfold` from `thm.get("hole_hints")` to
   `search()`; `_SearchEnv` gains `seed_premises: list[str]` and `hole_unfold: list[str]`.
4. `_symbolic_rank_premises`: with seeds, the ranked list is the seeds (in order) followed by the
   existing ranking with duplicates removed; `pool_kind` gains the suffix `+seeds`. Without seeds:
   byte-identical output to before.
5. `_try_dr_ducky` passes `seed_premises=env.seed_premises` to `run_ducky_on_goal` (the old parameter
   was never forwarded here).
6. Dr. Ducky: `_BindingSpace.seed_premises`; `_build_binding_space` puts seeds first in
   `accessible_lemmas`; new hole kind `role_typed_lemma` in `_hole_candidates` = the seeds, in order;
   one skeleton family beside the existing `accessible_exact_lemma` skeletons, priority just above
   them: templates `exact {lemma} {fact}` and `exact {lemma}`, holes `lemma: role_typed_lemma`,
   `fact: any_local_fact`.

**Contract:** without hints, every search output is unchanged; Dr. Ducky receives structural data
only.

**Tests:** hints travel request → row → `search()` → `_symbolic_rank_premises` (seeds first) and →
`run_ducky_on_goal` (mocked); without hints, `_symbolic_rank_premises` output byte-identical on a
fixture database; MUST-NOT (invariant #8): an `ast` scan of `src/dr_ducky*.py` finds no import of
`local_prover`, `handoff_loop`, `llama`, `openai`, `requests`, or any HTTP client.

**Check:** P3-core paired — PRESERVED expected (benchmark rows carry no hints).

**Commit:** `Search and Dr. Ducky: a hole's resolved candidates arrive as seed premises, and the seed
seam reaches Dr. Ducky at last`.

### S6 — Lean's own search, captured; the file's definitions unfolded

**Read first:** `src/lean_interface.py` (`LeanKernel.try_tactic`, `_pantograph_try_tactic`,
`goal_start`), `src/proof_search.py::_try_structural_fallback`, `src/residual_handoff.py::render`,
S0's U1/U2 results.

**Change:**
1. `LeanKernel.capture_closing_term(goal, tactic) -> str | None` by the route U2 established: the
   proof term `exact?`/`apply?` produced, as text that re-elaborates; else None. If U2 found no route,
   implement it to return None (and record that in the step report).
2. Phase 4 of `_try_structural_fallback`: on success, `_close_goal(..., replay=term)` with the captured
   term (`exact <term>`, or the `refine …` form `apply?` reported); `replay=None` when nothing was
   captured.
3. `render` emits a hole, never text, for a close whose `lean_tactic` is None (implement if absent).
4. δ-view: Phase 1.5 tries `unfold <names>` once per goal when `env.hole_unfold` names a definition
   that occurs in the goal (identifiers by `structural_reads.target_identifiers`). In S3.2, when R's
   conclusion relation, lhs head or rhs head is a local definition, compute a second role after
   `unfold <it>` and union its window (rule of S3.4).

**Contract:** no fill, hole block or emitted proof contains `exact?`, `apply?` or `rw?`; a Lean-search
close without a captured term is a hole.

**Tests:** the U1 fixture closes through `search_holes.py` with a block that contains the captured
term and no `?`-tactic; a Phase-4 close without capture renders as a hole; F9 CLOSED.

**Check:** P3-core paired — CHANGED is allowed only where a closed theorem's proof text changes from
a `?`-tactic to its captured term (list them); a closed theorem that opens → stop.

**Commit:** `Search: what exact? and apply? find is captured and replayed; a hole's own definitions
unfold where it names them`.

### S7 — Dr. Ducky checks that pieces fit (G3)

**Read first:** `src/dr_ducky_executor.py` whole (all three consumers of `_hole_candidates`:
`_instantiate_via_proof_dsl`, `_instantiate_skeleton`, `_instantiate_skeleton_certificates`);
`docs/Research_Paper/CLOSING_DECISIONS.md` §D1 (its grounding names the single seam).

**Change:**
1. `_build_binding_space` computes, once per goal (cached by `sha256(goal_text)` and fingerprint),
   `target_record` (the role record R of the goal, via S3.2's extractor) and `fact_records` (each
   hypothesis's premise features in R, by name). Lean is the kernel, not a model: invariant #8 holds.
2. `_hole_candidates(hole, space, binding=None)` — all three consumers pass their current partial
   binding. Filter, inside this function only:
   - `accessible_exact_lemma`, `accessible_apply_lemma`, `role_typed_lemma`: keep a lemma whose
     `norm.concl_rel` meets `target_record.norm.concl_rel`;
   - `rewrite_lemma`: keep a lemma whose `concl_rel` is `Eq` or `Iff` and whose `concl_lhs_head` is
     among the target's or any fact's `type_consts`;
   - a fact hole whose `compatibility` holds `{"pairs_with": <lemma hole id>}` and whose binding
     already names that lemma: keep facts whose relation (normalized) meets the lemma's first
     explicit premise relation (`norm.prem_rels[0]`).
   A candidate with no signature record is **kept** (no signature → conservative keep). The skeleton
   families that pair a lemma with a fact gain `{"pairs_with": ...}` on the fact hole.
3. Pure `compatible(kind, cand_record, space, binding) -> bool` holds the rules (Detective-pinned).

**Tests:** type-blind pairs removed on fixture spaces; records-missing candidates kept; a required
hole whose candidates all lack records is never emptied; each of the three consumers routes through
the filter (one test per consumer).

**Check:** the first 20 rows of `runs/loop_pass3d_addressed.jsonl` that carry `ducky_telemetry`:
`run_ducky_on_goal` with the filter off and on — programs tried, Lean rejections by class, progress,
closes. Expected: fewer rejected programs, no fewer progresses or closes (fewer closes → stop).
P3-core paired: PRESERVED, or CHANGED with gains only.

**Also:** in `CLOSING_DECISIONS.md` §D1, tick Decided and Done and add: "Built as G3 of
`docs/STRUCTURAL_RESOLUTION_PIPELINE.md` under the owner's full-scope ruling of 2026-10-30 (commit
`<hash>`)."

**Commit:** `Dr. Ducky: a lemma joins a skeleton only if its signature can meet the hole — her
paper's §5.5 compatibility rules, built`.

### S8 — Refine: the Jeeves operator for proof slots (G6)

**Read first:** this document's §3 entry; ModelAtlas's `src/model_atlas/query_navigate.py`
`build_refinement_guidance`, `_anchor_hints`, `_axis_hints` (`/Users/rohanvinaik/tools/ModelAtlas`,
`symbex <name> -f <file>`), for the discipline being ported.

**Change** — `src/refine.py` (pure; Detective-pinned):
- `next_question(cands, asked) -> Question | None` — `cands`: each `{name, features: set[(family,
  value)]}` plus an `idf` map. For every (family, value) not in `asked` present on c of n candidates
  with 0 < c < n: weight = c·(n−c)·idf. Highest weight wins; ties by (family, value). `Question =
  {family, value, has: [names], lacks: [names]}`. None when nothing splits.
- `equivalence_classes(cands, records) -> list[list[name]]` — identical feature sets and identical
  `type_pp` after renaming each binder (from the record's `binders`, in order) to `_b{i}`.
- `order_branches(question, context_claims, role_prefer) -> ["has", "lacks"] | ["lacks", "has"]` —
  "has" first when (family, value) is in `context_claims` (the name decoder's claims) or `role_prefer`;
  otherwise the smaller side first.
- Integration in `slot_resolution`: when a slot's merged window exceeds one page and every probe of
  the first page is `CANDIDATE_REJECTED`, take the next ≤ 256 candidates; probe one representative per
  equivalence class (a class whose representative is rejected with a type mismatch is skipped — same
  statement, same verdict); ask `next_question`, probe the top candidate of each branch in
  `order_branches` order; continue through the branches in that order. No candidate is ever dropped
  except by its own probe or its class's.

**Tests:** the 50/50 split wins; a 0/n split is never asked; ties are deterministic; identical
statements collapse into one class; with page size 1 the correct candidate is reached through the
second branch.

**Commit:** `Refine: a wide slot window is split one question at a time; answers order the branches,
none is dropped`.

### S9 — Goedel's contract (decisions A and B)

**Read first:** `src/handoff_loop.py` whole (`ROUNDS`, `KEEP`, `run`, `sample`, `harvest`, `annotate`),
`src/local_prover.py` (`ROUND0`, `CORRECTION`, `_corrected`, `Worker.verify`, `Worker._check`),
`src/hole_fill.py::error_report`, `src/residual_handoff.py` (`TRIED_MARK`, `HOLE_MARK`).

**S9.1 The repair turn (A).**
1. `ROUNDS` must currently be `((6, 2), (4, 0))`; else stop. Round 1 becomes 3 completions on the best
   partial proof plus one repair ask. When round 0 made no validated progress, `Handoff.run` today
   ends the hand-off (`if self.best is start_of_round: break`); instead, round 1 then runs **the
   repair ask alone** (no completions on the unchanged residual). With no qualifying answer (item 3),
   the hand-off ends as before.
2. `Handoff.sample` records `self.asked[sid] = messages` for every ask (`self.asked = {}` in
   `__init__`).
3. After round 0 (harvest and in-place search done), the repair source is chosen among round-0
   answers with `outcome == "wrong_proof"` that are neither drafts nor cut: fewest holes left in
   their partial proofs (`min(p.holes)` over partials whose `source` is that sid; infinite when it
   has none), then fewest Lean errors, then sid order. None → no repair ask.
4. Messages: `_corrected(failure, 1)` with `failure = {"messages": self.asked[sid], "raw":
   gen["content"], "report": hf.error_report(code, lean_messages, notes=notes)}`. `code` =
   `hf.extract_code(gen["content"])`; `lean_messages` = the answer's standalone-check messages as
   `_check` produced them (if `res` does not keep them, compile the code once the way `_check` builds
   its standalone file).
5. `hf.error_report` gains `notes: dict[int, str] | None = None`; after error i's `Error Message: …`
   line it appends exactly `\nWayfinder note: {notes[i]}\n`. Error i gets a note when the source line
   at its position contains the identifier of a slot from that answer's ledger rows (an unresolved
   name, or an excluded constant). Note text: `the structure at this position fits: ` + `; `-joined
   `` `name : type_pp` `` for up to 3 candidates whose probes were CLOSED or PROGRESS_ONLY, else up
   to 3 from the top of the merged window; no candidates → no note.
6. The repair ask is sampled with round 1's asks (kind `repair`, sid `t-repair-r1-s<n>`); its answer
   takes the ordinary path: verify → harvest → search in place → acceptance.
7. Model calls per hand-off stay at most 12 (6 + 2, then 3 + 1 — or 1 alone after a round 0 without
   progress).

**S9.2 The blocker record (B).**
1. `src/residual_handoff.py`: `BLOCKER_MARK = "WAYFINDER BLOCKER"`; pure `blocker_lines(k, rows,
   tried, role_record) -> list[str]`, at most four lines, each starting `-- WAYFINDER BLOCKER k: `:
   `deterministic moves tried here: <up to 10 names>`; `the goal's shape: <concl_rel> on
   <concl_carrier> between <concl_lhs_head> and <concl_rhs_head>` (null parts omitted); `names that
   resolved to nothing: <ident> (<kind>), …` (up to 4); `Lean rejected: <candidate> (<error kind>), …`
   (up to 4). A line with nothing to say is omitted.
2. `Handoff.annotate` inserts them immediately above the hole's `TRIED_MARK` line when present, else
   immediately above its `-- WAYFINDER RESIDUAL k: close this goal` marker.

**Tests:** with blocker lines inserted, `hf.hole_sites`, `hf.tried_at` and `hf.renumber_holes` give the
same results as without; round 1 builds 3 + 1 asks with a qualifying answer and 3 without; a round 0
without validated progress and with a qualifying answer yields exactly one ask in round 1 — the
repair ask — and without a qualifying answer ends the hand-off; the repair ask contains
`Wayfinder note:` at the right errors and no note elsewhere; the round-1 residual prompt
carries the blocker lines; a stubbed full hand-off makes at most 12 model calls.

**Commit:** `Hand-off: one repair turn in Goedel's own correction format, Wayfinder's candidates in
Lean's error text; each blocker described at its hole`.

### S10 — The name-piece rankers (G4)

**Read first:** `src/dr_ducky_executor.py::_accessible_lemmas`, `_lemma_token_score`;
`src/apply_scoper.py` whole; every caller of both (Serena).

**Change:**
1. `_accessible_lemmas`: the sited pool is ranked by the S3.2 role query's prefer-score from the
   signature index (pool names without a record follow the scored ones in pool order).
   `_lemma_token_score` is deleted once it has no caller.
2. `scope_for_apply(…, goal_record=None)`: tier 2 = premises whose `norm.concl_rel` contains the goal
   record's relation and whose `norm.concl_lhs_head` meets the goal's lhs/rhs heads; tier 3 =
   premises whose `norm.concl_rel` contains the goal's relation. Callers pass `goal_record` when they
   have one; without it tiers 2–3 are empty (premises fall to tier 4 — never back to names).
   `_name_contains_head` and `_name_matches_shape` are deleted once they have no caller.

**Check (each change separately):** P3-core and P3-manifest paired. PRESERVED, or CHANGED with every
gained and lost close listed; any lost close → stop.

**Commits:** `Dr. Ducky: her sited lemma pool ranked by signature, not by pieces of names` and
`Apply scope: head and shape tiers read from signatures, not names`.

### S11 — The full rebuild at 4.28, with a true accessible-premise closure (E; G1, G2, G10)

**Read first:** `scripts/convert_leandojo.py`, `scripts/extract_proof_network.py`,
`scripts/build_proof_network_db.py` (each whole, including `main` and argument parsing);
`src/proof_network.py` whole; `scripts/run_benchmark.py` (`load_benchmark_theorems`,
`_resolve_initial_goal_with_tier`); `wayfinder_mcp.py::_resolved_config`; S0's U5 result.

**S11.1 Values and imports.** `build_signature_dump.py --with-values` over the whole catalog →
`data/network428/declarations.jsonl`; `--imports` → `data/network428/imports.jsonl`. (Long-run rule.)

**S11.2 Tracer** `scripts/trace_mathlib_428.py`: for each Mathlib source file under
`data/lean_project_v428/.lake/packages/mathlib/Mathlib/` (sorted), `Server.tactic_invocations`; map
each invocation to its enclosing declaration by source range (declarations.jsonl `range`, same
file); per theorem with at least one invocation, one row in `convert_leandojo`'s schema:
`full_name`; `theorem_statement` = text after the last `⊢` of the first invocation's goal-before;
`goal_states` = each invocation's goal-before; `tactics` = each invocation's tactic text; `premises` =
constants used across its invocations, in first-use order, excluding the theorem itself;
`file_path` = `Mathlib/…/X.lean`. Options `--shard i/n`, `--resume`; per-file timeout 600 s; failures
to `data/network428/trace_failures.jsonl` with `INFRA_FAILURE` or `PARSE_UNSUPPORTED`. If U5 showed no
per-invocation used constants, `premises` = the theorem's `value_consts` that are theorems, and the
step report says so. Project the full run from U5 and stop for the owner's go-ahead. Output
`data/network428/traced.jsonl`.

**S11.3 Premise corpus** `scripts/build_corpus_428.py`: from declarations.jsonl, grouped by file, the
LeanDojo corpus schema `{"path", "premises": [{"full_name", "code", "kind"}]}` with `code` = the
declaration's source text sliced from its file by `range` → `data/network428/corpus.jsonl`.

**S11.4 Entities.** `scripts/extract_proof_network.py` on traced.jsonl and corpus.jsonl (its own
flags) → `data/network428/entities.jsonl`. Then `scripts/add_signature_lens.py` (new): each entity
with a declaration record gains anchors `{"label": "sig:<family>:<value>", "category": "signature",
"confidence": 1.0}` for S1.4's families; each `premise_only` entity's STRUCTURE, DOMAIN, AUTOMATION and
DECOMPOSITION positions are replaced by `structural_reads.read_goal("⊢ " + type_pp)` (sign; depth =
`min(round(read.depth), 3)`), DEPTH and CONTEXT left at zero.

**S11.5 Database.** `scripts/build_proof_network_db.py` → `data/proof_network_v4_428.db`. Then
`scripts/build_accessibility_428.py` (new): tables `module(id, name)`, `module_import_closure(module_id,
imported_module_id)` (transitive, from imports.jsonl), `entity_site(entity_id, module_id,
decl_line)`; rename v4's `accessible_premises` (built from used premises) to `used_premises` — oracle
data, for oracle experiments only, never a site pool.

**S11.6 The site pool (G10).** `proof_network.site_pool(conn, theorem_id, goal_text, limit=200)` =
`[p for p in library_premise_pool(conn, goal_text, limit=1000) if accessible(conn, theorem_id, p)][:limit]`,
where p is accessible iff its module is in the closure of the theorem's module, or it is in the
theorem's module with a smaller `decl_line`. `_symbolic_rank_premises` and `_accessible_lemmas` use
`site_pool` when the connection has `entity_site`; on v3 their behavior is unchanged. MUST-NOT test:
on v4 a theorem's site pool never contains the theorem itself or a later declaration of its module.

**S11.7 Benchmarks at 4.28.** `scripts/derive_428_manifests.py`: each row of `data/mathlib_test.jsonl`
whose theorem exists in declarations.jsonl by exact name → the row with its 4.28 `file_path`, module
and line (the fields the benchmark reads) → `data/mathlib_test_428.jsonl`; others →
`data/mathlib_test_428_dropped.jsonl` with the reason. The original file is untouched.

**S11.8 Config** `configs/wayfinder_deterministic_v4.yaml`: the deterministic config with
`proof_network_db: data/proof_network_v4_428.db`, `mathlib_test_split: data/mathlib_test_428.jsonl`,
no censor store; nothing else changed.

**S11.9 Measurements:** (a) P3-core, v4 against v3; (b) P3-manifest, v4 against v3; (c) MiniF2F-test in
full on v4 — owner's go-ahead; (d) the Mathlib split at 4.28 on v4 with the true site pool — owner's
go-ahead. Report (d) beside the old numbers with this label: "the v3 Mathlib numbers were measured
with site pools equal to each theorem's own proof premises". No README or ledger number changes
without the owner's approval.

**S11.10 Switch.** When (a) and (b) are PRESERVED or CHANGED with gains only, point
`wayfinder_mcp._resolved_config()` (the MCP lane and the hand-off's in-place search) at the v4 config.

**S11.11 Censors on v4.** After (d)'s control run, re-mine with `scripts/mine_censors.py`
(`mined_at_commit` recorded); paired control against store on P3-core and P3-manifest; solve delta must
be ≥ 0; then add the store to the v4 config.

**Tests:** on a three-file fixture corpus — tracer mapping by range, corpus slicing, the import
closure, `site_pool`'s anti-circularity, byte-identical rebuild of the fixture tables.

**Commits** (one each): tracer; corpus; signature lens and computed positions; accessibility and site
pool; manifests and config; the MCP switch; censors on v4.

### S12 — Documentation

1. `docs/Research_Paper/WAYFINDER_DESIGN.md` §13: §13.4 (failed steps compiled and resolved; per-hole
   batteries; seeds only from steps without unresolved names); §13.5 (round 1 = 3 completions + 1
   repair turn); §13.9 invariant 6 replaced by: "The model is a source of proposals inside Wayfinder.
   It is asked a fixed small number of times; at most one repair turn per hand-off shows it one of
   its own earlier answers, in its trained correction format, with Lean's errors and Wayfinder's
   structural candidates; only what Lean accepts carries forward." New §14 "Structural resolution"
   describing S1–S8 and the v4 network as built.
2. `README.md` "The last mile" and "What the search cannot close": the mechanism. **No number changes
   and no G10 wording without the owner's approval.**
3. `WAYFINDER_MCP_NOTES.md`: the ledger file, hints, the repair turn, the v4 config.
4. `docs/Research_Paper/THE_REFOUNDING.md` §8 module diff and §11 status ledger: G1–G10 as built, with
   status tags; numbers only with approval.
5. This document's status line: "built", with each step's commit hashes.

---

## 7. Acceptance summary

| Step | Must hold before the commit |
|---|---|
| S0 | gate green; spec and hand-off work committed; trailers gone from the nine commits with identical trees; worktrees pruned; U1–U6 recorded; population rebuild byte-identical with its sha256 recorded; fixtures committed |
| S1 | pinned signatures for card_le_card, true_and, card_mono; deterministic index; dump time recorded |
| S2 | the compiler's pinned cases; the spelling-independence fence |
| S3 | F1a–F10 per §S0.7 (F9 exempt); `probes(F1b) ≤ probes(F2b)` |
| S4 | specimen ≥ 7/8 verified and gate-accepted; P1-dev wrong = 0; no slot-bearing step seeded verbatim |
| S5 | hints reach `_symbolic_rank_premises` and Dr. Ducky; byte-identical without hints; P3-core PRESERVED |
| S6 | no `?`-tactic in any fill or block; U1 fixture closes with a captured term; F9 CLOSED; P3-core per §S6 |
| S7 | three consumers filtered; telemetry check no fewer closes; P3-core PRESERVED or gains only |
| S8 | refine's pinned cases; continuation reaches the second branch |
| S9 | ≤ 12 model calls; 3 + 1 asks, or the repair ask alone after a round 0 without progress; notes placed; hole parsing unchanged by blocker lines |
| S10 | P3-core and P3-manifest PRESERVED or gains only, per ranker |
| S11 | site-pool anti-circularity; (a) and (b) before the switch; (c) and (d) only on the owner's word |
| S12 | the listed sections updated; no number changed without approval |

## 8. Frozen and out of scope

Untouched by this build: the statistical layer and its artifacts (`data/nav_training.jsonl`,
`data/nav_eval.jsonl`, checkpoints, second-order corpora), `src/hardtail_bridge.py` beyond S5's
rename, the v3 network and every config that names it, the union scoreboard `runs/loop_union.json`.
The v3 benchmark numbers stand as measured, under S11.9's label.

## 9. Owner checkpoints — stop and ask at each

1. S0.3 step 8: older commits that carry the trailer.
2. S1.5: if the full dump projects over 30 minutes.
3. S4: if the P1 replay projects over 30 minutes.
4. S11.1 and S11.2: the values dump and the full tracer run.
5. S11.9 (c) and (d): MiniF2F in full and the Mathlib split.
6. Any CHANGED verdict where a step expects PRESERVED or gains only.
7. Any README or ledger number, and the wording for G10.
