# The Closing Decisions

**Status:** open ledger — seven decisions to close out the multi-stage thesis, taken one at a
time, each marked when decided and again when done. The plain-language framing is the primary
text; the technical anchor is the footnote. (House rule: deep-theory decisions are thought
through conversationally; rigor lives in the artifacts that come out the other side.)

**How we got here:** the deterministic system proves 1,042 of 2,000 theorems with no neural
network anywhere, every proof checkable, every run reproducible to the byte. The neural
version scored 1,277 — so we kept ~80% of the performance and threw away the black box. What's
left is deciding what "finished" means.

---

## D1 — Teach Dr. Ducky to check that pieces fit before trying them

**The plain question.** When a person tries "apply this lemma to that fact," they glance first:
does this lemma even *take* that kind of thing? Dr. Ducky currently doesn't glance — she tries
every lemma against every fact and lets Lean reject the nonsense, which we measured at ~100%
rejection across ~3,000 attempts. Her own paper says she should check fit first; we never built
the checking.

**Recommendation:** build it. Ask the kernel once per lemma what it takes (cacheable), only
pair things whose shapes match. It's the last place the code does less than the theory says.

*Anchor: Dr. Ducky paper §5.5 "deterministic compatibility rules"; pass-3c/3d telemetry
(2,977 plan executions, 3 progresses); env_inspect signature fetch at binding-space build.*

**Grounding (Serena-verified 2026-08-31).**
- **The single seam**: `_hole_candidates` (src/dr_ducky_executor.py, ~line 2005) has THREE
  consumers — `_instantiate_via_proof_dsl` (:1090), `_instantiate_skeleton` (:2091),
  `_instantiate_skeleton_certificates` (:2139). The compatibility filter MUST land inside
  `_hole_candidates` itself (or in the hole-source functions it dispatches to), never at a
  caller — this session repeatedly measured the sibling-path trap (fix one path, the one that
  runs keeps the bug).
- **What's available to check against**: `space.local_facts` carries `(expr, type_text)`
  pairs — fact types are ALREADY in hand. Lemma types are NOT: `_accessible_lemmas` returns
  names only (the DB stores no signatures). The fetch point: `_build_binding_space` (:1660)
  receives `lean`; `lean._server.env_inspect(name)["type"]["pp"]` returns the printed
  signature (the same call EXP-RF-007 stage A used, proven reliable; cache per pool — 8
  lemmas/goal, and pools repeat heavily across goals).
- **The compatibility rule's first cut**: parse the lemma pp's first explicit argument type
  (text between the first `(` binder or after the last `→`… start simple: the final
  hypothesis segment before the conclusion) and require the fact's type text to match
  normalized (`sanitize_goal_text` both sides). Expect misses (implicit args, universe
  variables) — the telemetry re-run measures the filter's precision.
- **Evidence corpus for before/after**: `runs/loop_pass3d_addressed.jsonl` rows'
  `ducky_telemetry[].tried[]` (script + error per program) — failure classes: parse (fixed
  7469dcf), Application type mismatch / Function expected / instance synthesis (THE target),
  `accepted_without_goal_change`.
- **Gotcha**: the executor's engine CLASS path (`MathEngine.generate`) and module path both
  route through `_instantiate_skeleton_certificates` now, but `_instantiate_via_proof_dsl`
  (:1090) is a third instantiation route (rosette DSL bindings) — check whether its bindings
  bypass `_hole_candidates` filtering before assuming coverage.

**Decided:** ☐  **Done:** ☐

---

## D2 — What to do with the fourth layer we never built (the second-order SoM)

**The plain question.** The original plan had a trained model at the top of the stack, learning
from the residuals. Since then, the whole point sharpened: the thinking is the deterministic
part; anything learned is just a hunch about what to try first — allowed to suggest an order,
never allowed to decide. Do we build the hunch-machine now, or close the thesis without it?

**Recommendation:** close without it. "1,042 proofs with zero learned parameters" is the
demonstration; bolting on a trained orderer now blurs the line the whole project exists to
draw. It stays specified in FUTURE_DIRECTIONS — an escape hatch, on Rohan's schedule.

*Anchor: THE_REFOUNDING §4.3 (statistical = ordering over fenced spaces); the packet corpus
(rendered cheat sheets) as its training data if ever built.*

**Grounding (verified 2026-08-31).**
- **The layer's spec**: docs/Research_Paper/FUTURE_DIRECTIONS.md, the four-layer stack
  section ("4. Second-order SoM — post-benchmark orchestration layer trained on
  residual/gap/capsule/executor data — not yet the canonical theorem-search runtime").
- **The training corpus DOES NOT EXIST yet**: `runs/goal_ledger/` is EMPTY (0 files) —
  `wayfinder_mcp.prove()` persists ledgers per-call and hasn't run in bulk. The raw material
  exists as run artifacts (`runs/loop_pass*.jsonl` with dispositions, telemetry, traces) but
  zero rendered packets. Declaring D2 out-of-thesis costs nothing already built.
- **Where the decision lands in text**: THE_REFOUNDING §4.3 already states the law
  (statistical may order, never fence, never generate); the closing edit is a FUTURE_DIRECTIONS
  re-specification (orderer over slates/lanes, trained on rendered packets, consumed as a
  score-shaping prior through `program_score`/lane-order — the seams already exist:
  `bank_weight`/`program_score` in dr_ducky_executor, `deterministic_lane_order` in
  structural_reads).
- **Prior art to cite in the re-spec**: Wesker v3 (213K synthetic packets, 93%+ micro-accuracy
  on engine/backend routing) — the existing measured evidence that this layer CAN be trained
  when wanted.

**Decided:** ☐  **Done:** ☐

---

## D3 — Agree on how we count

**The plain question.** Some theorems flicker: change anything about the order of attempts and
three drop out while a different one drops in. But a checked proof is a checked proof forever —
once we've proven something, it's banked. So the honest score is "everything we've ever proven"
(the union), and single-run wobbles of ±3 are noise about ordering, not evidence about the
machinery.

**Recommendation:** write union-accounting into the canon as the loop's official measure, with
the flicker band documented so nobody (including us) mistakes wobble for signal.

*Anchor: §7.3 metric extension; the measured ±3 band across passes 2/3/3b/3d; current union
1,042/2,000.*

**Grounding (verified 2026-08-31).**
- **The union's constituent artifacts** (all in runs/): pass 1 = `loop_pass1_full2000.json`
  (1,028; byte-identical reproductions in `loop_pass1b_untraced_identity.json` and
  `loop_pass1c_traced.json`); residual closes = `loop_pass2_control.json` (13, the standing
  set) + `loop_pass3b_plans.json` (+1: `seminormFromConst_bddBelow`). Union = 1,028 + 14
  distinct = **1,042**. The flicker evidence: pass-3 (13, same set), pass-3b (11, −3+1),
  pass-3d (3 — the falsified-reserve run, excluded from any band estimate as a known-defect
  run), plus round-4's ±4 on the half-B slice.
- **No union script exists** — every union so far was computed ad hoc in session analyses.
  The implementation: `scripts/loop_union.py` reading a list of report JSONs, emitting the
  distinct honest-close set + per-run flicker deltas; its output file becomes the loop's
  official scoreboard (e.g. `runs/loop_union.json`).
- **Where the canon lands in text**: THE_REFOUNDING §7.3 (metric extension) gains the
  union-accounting law + the flicker-band caveat; the doc's own §12 discipline ("re-issued
  per wave with the ledger updated") covers the re-issue.
- **One decision inside the decision**: does the union admit closes from known-defect runs
  (pass-3d's 3 were a subset of the standing 13 — moot so far, but state the rule: a close
  counts iff its run's code passed the standing gate and the close is honest per the
  dispositions).

**Decision taken (2026-08-31).** Union accounting is canon, and the inner rule resolved MORE
permissively than sketched above: **the kernel owns the bank; run quality gates only the
evidence.** A row banks iff its disposition is `closed_value` — the Wave-5 calculus already
contained the whole policy (closed_value = value kill, banked forever; closed_suspect = run
kill, never banked). A known-defect run's kernel-valid honest closes still bank (a defect in
the run cannot un-check a proof), while the run is excluded from flicker-band and delta
claims — the two-ledger rule, value-kill ≠ run-kill applied to search. Flicker is defined
only against a standing bank (founding/disjoint-universe runs are machinery signal, not band
evidence); the measured band is ≤3 drops / ≤1 gain per evidence run. Shipped:
`scripts/loop_union.py` (banking consumes the `disposition` field, fails loudly on vocabulary
drift; 12 hand pins in `tests/test_loop_union.py`) → `runs/loop_union.json`, the scoreboard of
record: **1,042/2,000**. Canon text: THE_REFOUNDING §7.3, union-accounting paragraph.

**Decided:** ☑  **Done:** ☑

---

## D4 — Make the "stale fences" lesson a rule the code enforces

**The plain question.** We learned the hard way that a list of "things that never work" is only
true for the version of the system that generated it — change the prover, and yesterday's
never-works can become today's winner (that cost us 4 proofs once). The lesson is written down;
lessons written down get forgotten. Make the code check: the fence list remembers which version
mined it, and complains when the prover has changed underneath it.

**Recommendation:** implement — store records its mining commit; gate warns on mismatch; every
re-mine re-runs the safety check.

*Anchor: §10.8; censor store metadata `mined_from` + code commit; the paired delta ≥ 0 check.*

**Grounding (Serena-verified 2026-08-31).**
- **The load seam**: `src/proof_search.py:2375-2378` — `if cfg.censor_store_path:` →
  function-local `from src.censor_store import load_fences` → `env.censor_fences =
  load_fences(...)`. NOTE: the local import makes `load_fences` INVISIBLE to the reference
  graph (Serena shows zero references) — a known blind spot; the wiring is real.
- **Store metadata today** (runs/censor_store_v3.json): `version`, `mined_from` (corpus
  path), `mining_parameters`, `corpus` stats, rejection tallies. NO code-commit field — the
  invariant adds `mined_at_commit` (git rev at mining time, written by
  scripts/mine_censors.py) and the loader compares it against HEAD's positive-channel
  commits, warning (or refusing, config-gated) on mismatch.
- **What counts as "the positive channel changed"**: commits touching src/proof_search.py,
  src/tactic_compiler.py, src/dr_ducky*.py, src/structural_reads.py, src/resolution.py —
  the honest cheap check is any src/ commit since `mined_at_commit` (`git rev-list
  mined..HEAD -- src/` non-empty).
- **The safety re-check**: the paired battery pattern
  (scripts/run_loop_pass2_paired.sh is the template) — control vs store, delta ≥ 0 binding.
- **The stores' history for the invariant's docstring**: v1 (cosine-era corpus — fenced a
  `learned` lane that no longer exists), v2 (pass-1c regime — cost 4 solves when applied
  post-slate-fix... later re-attributed to the granularity transport, but the regime concern
  stands as hygiene), v3 (pass-2-control regime, current; 89 fences, Σκ 37,375). Un-mined
  spines waiting: rw_at lane (~3k failures), `positivity` (741/741), hyp-closer family
  (5,018 tries / 0 closes).

**Decided:** ☐  **Done:** ☐

---

## D5 — Pay the three measurement debts

**The plain question.** Three promised checks were never run: (1) do our emitted proofs
actually compile on the newer Lean, end to end? (2) the one theorem this system proved about
its own math — audit that its proof doesn't smuggle in extra axioms; (3) rerun the 21 novel
proofs from before the refit through everything we've built since — the cleanest before/after
we own. A thesis whose motto is "every claim measured" can't close with unmeasured claims.

**Recommendation:** run all three before declaring closure.

*Anchor: EXP-RF-006 `--verify`; `#print axioms` on `bridge_curvature_bound`; EXP-RF-021.*

**Grounding (verified 2026-08-31).**
- **EXP-RF-006 `--verify`**: the flag EXISTS — scripts/exp_rf006_emission_gate.py:86
  (`--verify … compile both artifacts at the v4.28 gate`); the dry mode already measured
  surface-identity-modulo-warrant. The verify project: `data/lean_project_v428`
  (wayfinder_mcp.VERIFY_PROJECT; per the MCP notes it may need `lake update mathlib` +
  `lake exe cache get` once — budget an hour of Mathlib cache the first time).
- **The axioms audit**: the Lean batch lives at
  `…/rohan-vinaik.github.io/papers/Core Documents/Semantic_Specification_Learning/proofs/`
  — `bridge_curvature_bound.lean` confirmed present alongside coverage_submodular,
  falsifiability_pivot, greedy_coverage_bound, et al. The audit = append
  `#print axioms <theorem>` per file (or one Audit.lean importing all), `lake env lean`,
  expect only `propext/Classical.choice/Quot.sound`. Until run, `bridge_curvature_bound`
  stays cited "proved-modulo-audit" — and it is LOAD-BEARING (THE_REFOUNDING §3.4).
- **EXP-RF-021**: the manifest is `data/rerun_v427_manifest.jsonl` — **21 theorems**
  (survivingDOF_antitone, resolve_iff_surprise, kl_eq_fisher_quadform, …). Correction to
  stale memory: not 9. These are the SSL/Ansatz novel theorems (no Mathlib replay concerns —
  the cleanest longitudinal instrument). The rerun: run_benchmark with
  `--theorems data/rerun_v427_manifest.jsonl` under configs/wayfinder_deterministic_traced
  at closing HEAD; compare against the pre-refit closure record (the 2026-08-24 batch,
  closed by the pre-refit prover per the theory-stack memory).

**Decided:** ☐  **Done:** ☐

---

## D6 — Don't move the goalposts (the norms question)

**The plain question.** All our "what's typical for a goal" baselines were measured on the full
corpus. The leftover hard theorems look different from typical. Should we re-baseline to the
leftovers? Instinct says careful: re-baselining silently re-keys every fence and coordinate at
once. Better to measure how different the leftovers are (a report), and only re-baseline if we
formally start over on a new corpus.

**Recommendation:** measure, don't re-base.

*Anchor: SurfaceNorms/ContextNorms mined 2026-08-24 from nav_eval; §10.7 population
re-estimation; re-basing = only with a formal re-founding.*

**Grounding (Serena-verified 2026-08-31).**
- **The blast radius, measured by references**: `DEFAULT_SURFACE` is the default parameter of
  FIVE read functions (read_size :337, read_casts :353, read_numerals :368, read_coupling
  :391, read_goal :444 — src/structural_reads.py). Through `read_goal` the norms key:
  every censor signature (`censor_store.slice_signature` → all 89 v3 fences), every
  `MINED_DUCKY_CELLS` cell (the 5 mined regime cells store signatures computed under
  CURRENT norms), and `goal_tail_regime`'s size read. Re-basing silently re-keys ALL of
  them simultaneously — the fences and cells would point at signatures no goal produces.
- **The report instead**: the re-mine functions exist (`mine_median`, `mine_top_tertile`,
  `mine_context_zero` — same file); the report = run them over the residual manifest's goal
  targets (`runs/loop_pass2_residual_manifest.jsonl`, 771 rows, `goal_state` field) and
  print old-vs-new zeros side by side. ~30 lines, no runtime change.
- **If a re-base ever happens** (formal re-founding only): norms, censor stores, AND mined
  cells re-mine together in one commit, followed by the paired delta check — they are one
  keyed system, never three.

**Decided:** ☐  **Done:** ☐

---

## D7 — Open the next universe (problems, not just proofs)

**The plain question.** The original hunch was never really about proofs. It was: people solve
hard problems by figuring out what *kind* of problem it is, then reaching for the tools that
kind needs — Einstein didn't reinvent calculus, he thought about trains and clocks and combined
what existed. Proofs were the perfect practice arena because Lean tells you instantly whether
you're right. Once D1–D6 close, the arena is complete — and the real move is pointing the same
machinery at open-ended math problems, where "what kind of thing is this and what tools does
that kind take" is the whole game.

**Recommendation:** this IS the closure. Declare the proof instrument finished, re-issue
THE_REFOUNDING with every experiment carrying its number, and begin the transport.

*Anchor: Rohan's March note (problem-solving above proofs); the corpus's transport record
(program identity → text meaning → proofs → …); same reads, same two signs, same loop.*

**Grounding (verified 2026-08-31).**
- **The declaration's preconditions**: D1 built · D2's re-spec written · D3-D4 in the canon ·
  D5's three numbers in the ledger · D6's report filed. Then THE_REFOUNDING's final re-issue:
  every §9 experiment carries its measurement (002 ✓, 003 ✓, 005a/b ✓, 006 needs --verify,
  007 ✓, 008 = the D1 re-measure, DET-001 ✓, 021 = D5).
- **What transports, concretely** (the machinery inventory a new-domain port would take):
  the read algebra (structural_reads — banks, mined norms, signatures, interference), the
  two-sign loop (censor_store + mine_censors; the promotion miner), the scheduler laws
  (close-before-unpack, tail-regime routing, slate reserve), warranted emission (lean_expr's
  value/source duality generalizes to any checkable output), the disposition calculus, the
  cheat-sheet hand-off, and the loop protocol itself (pass → residual → re-mine → pass).
  What does NOT transport: the anchors/IDF network (proof-network-specific — each domain
  mines its own), the tactic grammar (domain language), Pantograph (the referee).
- **The referee question is THE open design question for problems-space**: proofs had Lean
  (instant, incorruptible). Competition math has answer-checking (AIMO-style: the aimo3
  project at ~/Projects/Kaggle_Killer/competitions/aimo3 holds the cheat-sheet/ledger
  prototypes); open problems have no referee — the honest scope for the first transport is
  answer-checkable problems (the AIMO shape), where "what kind of problem is this and what
  tools does that kind take" has a measurable win condition.
- **First artifact of D7 when opened**: a THE_REFOUNDING-style founding doc for the new
  universe, written the same way (priors_do_not_rederive from THIS corpus, per-claim status
  tags, the experiments named before the code).

**Decided:** ☐  **Done:** ☐

---

*Order of attack: chosen one at a time, conversationally, starting wherever Rohan points.*
