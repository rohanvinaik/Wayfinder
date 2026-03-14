# LintGate Specification Coverage Upgrade Plan

**Date:** March 11, 2026
**Primary target:** LintGate
**Validation corpus:** Wayfinder and LintGate itself
**Intent:** Close the gap between LintGate's specification-complexity theory and the actual operator experience of using the tools in a real engineering session.

This is not a Wayfinder feature plan. Wayfinder is the proving ground that exposed where LintGate is useful, where it is vague, and where the feedback loop breaks.

---

## Goal

Turn LintGate's spec and mutation stack from a strong diagnostic concept into a practical engineering tool that reliably answers four questions:

1. What is under-specified?
2. Which concrete test should be written next?
3. Is this a test gap or a decomposition problem?
4. How much confidence should the operator place in the answer?

The end state is not "more theory surfaced." The end state is that `spec_file_analyze`, `mutation_run_full`, `mutation_prescribe`, and `convergence_analyze` produce guidance an engineer or coding agent can act on immediately.

---

## North star: the platonic ideal of a codebase

The deeper goal is not merely better test prescriptions. The goal is to provide a signal for moving a codebase toward its most coherent reasonable form: cleaner abstractions, more orthogonal responsibilities, less hidden coupling, stronger local reasoning, and faster repair when something breaks.

LintGate should approach that ideal by composing multiple orthogonal symbolic systems, each capturing a different aspect of software quality:

- specification and mutation signals
- structural and decomposition signals
- contracts and interface signals
- purity/testability signals
- dependency and coupling signals
- behavioral and drift signals

No single lens defines the ideal. The ideal is approached when several independent symbolic systems converge on the same explanation and the same next action.

This is where the SICP-like engineering tradition matters. The system should reward and guide codebases toward:

- explicit abstractions and data boundaries
- orthogonal decomposition rather than entangled cleverness
- generic operators over repeated special-case logic
- minimal hidden state and minimal surprise
- components that can be understood, tested, and repaired locally
- interfaces that compose cleanly instead of leaking internal complexity

"Reasonable" is part of the goal. The tool should not push decomposition past the point of clarity or operational usefulness.

---

## Product Thesis

### What a successful implementation looks like

After running LintGate on a function, the operator should get:

- a stable diagnosis grounded in actual mutation data when available
- a degraded but still useful diagnosis when mutation data is unavailable
- the top surviving behavioral gaps in concrete terms
- a suggested next test with enough witness information to write it quickly
- a decomposition recommendation only when evidence is stronger than "many things survived"

### Design principles

1. **Improve existing MCP tools instead of adding a parallel surface.**
   The right integration points already exist: `spec_file_analyze`, `mutation_run_sampling`, `mutation_run_full`, `mutation_prescribe`, `mutation_prescribe_tests`, `mutation_validate_tests`, and `convergence_analyze`.

2. **Operator value beats theorem completeness.**
   A precise-looking output that does not change what the engineer does next is not useful.

3. **Empirical evidence outranks symbolic prediction.**
   Static sigma and phase estimates remain valuable, but once mutation data exists they must be reconciled against it explicitly.

4. **Approximation must be labeled honestly.**
   Greedy set cover yields an upper bound on the minimum teaching set, not the minimum itself. The tool must say "upper bound" unless it has an exact witness.

5. **Fallbacks are mandatory.**
   Test discovery failures, import failures, and empty kill matrices must produce diagnostics, not silent "0.0 spec level" outputs.

6. **Orthogonal convergence is the real decomposition signal.**
   Mutation data alone should not drive architecture. Decomposition guidance should strengthen only when specification, structure, contracts, and coupling signals agree.

7. **Validation lives on real repos.**
   Wayfinder is the first validation corpus because it already revealed the operator pain points. LintGate itself is the second corpus because the tool should work on its own architecture.

---

## What Wayfinder revealed about LintGate

Using LintGate on Wayfinder surfaced a clear pattern:

1. **Mutation sampling was the most honest signal.**
   It immediately distinguished "untested" from "weakly tested."

2. **The feedback loop was fragile.**
   When test discovery failed, the entire spec loop degraded: mutation results became misleading, spec levels stayed flat, and progress was hard to verify.

3. **Prescriptions were too generic.**
   "Add exact-value assertions" is a taxonomy label, not a usable next step.

4. **Phase and hotspot summaries were useful, but not decision-oriented enough.**
   They helped prioritize work, but they did not consistently tell the operator what to do next.

5. **The system is already close to valuable.**
   The problem is not lack of theory or lack of tools. The problem is that the last mile from mutation evidence to concrete engineering action is incomplete.

---

## Current implementation gaps in LintGate

These are the gaps this plan addresses.

### Gap 1: Mutation outputs are not rich enough for grounded prescriptions

Current post-profiling analysis uses aggregate counts and a partial kill matrix. That supports category summaries, but not high-quality "write this exact test next" guidance.

### Gap 2: Discovery failure is a single point of failure

If tests are not discovered or loaded, mutation results can look like complete survival even when the project has tests. This poisons downstream spec metrics.

### Gap 2b: Mock-boundary opacity makes some survival rates misleading

Some orchestration functions sit behind heavy mocking or patching. In those cases, tests are discovered and linked, but mutations inside the function's internal call paths cannot be meaningfully exercised because the relevant behavior is replaced by mocks.

This is distinct from discovery failure and must be diagnosed separately. The correct output is not "100% survival => large specification gap." The correct output is:

- tests were found
- tests were linked
- survival is dominated by mocked call paths
- mutation score is not a trustworthy measure for this function under current test topology

### Gap 2c: Runtime and budget semantics can make the mutation tools unusable

The current mutation path can explode in wall-clock time because:

- `mutation_validate_tests` is effectively exhaustive profiling
- fallback discovery can load all tests from all discovered files
- the same `budget_ms` concept is used both as total-call budget and per-mutant timeout
- multi-function sampling can let early functions consume most of the budget

This is not a secondary performance issue. It is a product issue. A tool that hangs or expands to project-wide work when the operator expects validation cannot serve as a practical signal.

### Gap 3: Symbolic and empirical layers are not reconciled clearly

`spec_file_analyze` and `mutation_run_full` both speak about sigma/regime/phase, but they do not currently present a coherent static-vs-empirical story.

### Gap 4: Approximate quantities are framed too strongly

Greedy convergence and kill-matrix summaries are valuable, but they should not be presented as exact teaching-set or exact sigma results unless LintGate has exact evidence.

### Gap 5: Decomposition advice is under-evidenced

Surviving multiple mutation categories is not sufficient on its own to justify extraction or architectural decomposition.

### Gap 6: The reporting surface is organized around internals, not user decisions

The tool should foreground: next test, likely boundary, confidence, and whether to keep testing or refactor. Those are the operator decisions.

### Gap 7: The roadmap can drift toward local mutation cleverness instead of structural idealization

If LintGate does not explicitly encode abstraction quality, orthogonality, interface cleanliness, and local repairability as first-class goals, it will improve test guidance while missing the broader purpose.

---

## Target architecture inside LintGate

The implementation should live in LintGate, not Wayfinder.

### Existing MCP surfaces to extend

- `mutation_run_sampling`
- `mutation_run_full`
- `mutation_get_state`
- `mutation_prescribe`
- `mutation_prescribe_tests`
- `mutation_validate_tests`
- `spec_file_analyze`
- `spec_project_rollup`
- `convergence_analyze`

### Recommended internal modules

Add or extend modules along these lines:

- `lintgate/specification/empirical_kill_matrix.py`
  Canonical kill-matrix and survivor schema.

- `lintgate/specification/mutant_reporting.py`
  Human-usable mutant diffs, witness generation, and confidence scoring.

- `lintgate/specification/teaching_set.py`
  Greedy teaching-set upper bound and redundancy analysis.

- `lintgate/specification/trajectory_analysis.py`
  Empirical trajectory, tail detection, and phase-transition confidence.

- `lintgate/specification/static_empirical_reconciliation.py`
  Merge symbolic sigma/regime/phase with mutation-derived evidence.

- `lintgate/specification/decomposition_evidence.py`
  Interface-evidence layer for composition-gap and extraction guidance.

These names are illustrative. The important design choice is separation between:

- raw empirical data
- derived approximations
- user-facing prescriptions

And above that separation, the tool needs a convergence layer that asks:

- does this function violate local reasoning?
- is shared state or interface leakage driving complexity?
- do multiple symbolic systems agree that this should be decomposed?

---

## Output contract that the tool must satisfy

This is the most important section. If these outputs are good, the tool will be useful.

### 1. Mutation profile contract

`mutation_run_full` should return, per function:

- mutation coverage summary
- discovery diagnostics
- topology diagnostics
- runtime diagnostics
- explicit survivor records
- greedy teaching-set upper bound
- trajectory summary
- top actionable next tests

Each survivor record should include:

- `mutant_id`
- `category`
- `location`
- `operator`
- `diff_summary`
- `killed_by`
- `status`
- `witness_confidence`

If a witness or exact input cannot be derived, the record must still explain why the mutant matters.

All mutation MCP tools should also support bounded execution and partial return:

- `timeout`
- `partial_results`
- `budget_exhausted`
- `budget_scope`
- `per_mutant_timeout_ms`

### 2. Prescription contract

`mutation_prescribe` should no longer stop at category-level advice. It should output:

- `why_this_matters`
- `suggested_input` or `input_pattern`
- `expected_behavior`
- `assertion_shape`
- `confidence`
- `source_of_evidence`

The operator should be able to write the test without rereading the entire function unless the tool explicitly says source inspection is still required.

### 3. Spec-analysis contract

`spec_file_analyze` should return both:

- `static_estimate`
- `empirical_overlay`

The overlay should clearly state one of:

- `no empirical data yet`
- `empirical data agrees`
- `empirical data contradicts static estimate`
- `empirical data unavailable due to discovery failure`
- `empirical data limited by test topology / mocks`

This is what makes the tool trustworthy.

### 4. Decomposition contract

`convergence_analyze` and `mutation_decompose` should recommend extraction only when at least one of the following is true:

- persistent tail after multiple targeted tests
- interface-like survivors concentrated near a call boundary
- static interface complexity and empirical survival profile agree
- post-test validation shows the function is structurally resisting direct specification
- orthogonal lenses agree that the function violates abstraction or local-reasoning boundaries

If this evidence is absent, the tool should say "keep testing" rather than implying refactor-by-default.

The preferred output is not just "extract." It is "extract along this responsibility boundary because it reduces coupling, strengthens interface clarity, and makes the remaining parts easier to test and repair."

---

## Direct implementation sequence

The following sequence is the direct execution plan. It is organized as PR-sized work packages, names the concrete LintGate files to modify, defines the output contract changes, and states the tests required for each step.

### Implementation rules

These rules are non-negotiable and should be applied in every PR:

1. MCP tools remain the product surface.
2. Greedy analysis must use upper-bound language.
3. Discovery failure and mock-boundary opacity are both P0 truthfulness issues.
4. Runtime safety is part of correctness; mutation tools must fail bounded and return partial results rather than hang.
5. Survivor-specific guidance takes priority over new theory diagnostics.
6. Decomposition recommendations require cross-lens agreement.
7. Any MCP tool semantic change must update:
   - `README.md`
   - `docs/design.md`
   - `docs/agent/AGENTS.md`

### PR0: Runtime safety and budget semantics

**Goal:** Make mutation tools bounded, predictable, and safe to use on real repositories.

**Modify:**

- `lintgate/specification/mutation_engine.py`
- `mcp_tools/_mutation_impl.py`
- `mcp_tools/_mutation_tools_impl.py`
- `mcp_tools/mutation_tools.py`

**Required behavior changes:**

- `mutation_validate_tests` stops being implicit exhaustive profiling by default
- outer wall-clock budget is separated from per-mutant timeout
- fallback test loading is capped and relevance-ranked
- multi-function sampling splits budget per function
- all mutation MCP tools have a hard circuit-breaker and can return partial results

**Implementation details:**

1. In `mcp_tools/_mutation_tools_impl.py`, give `impl_refactor_loop` a `budget_ms` parameter and default validation to sampled mode.
   - Default: sampled validation, not exhaustive profiling
   - Suggested default budget: `300000` ms total
   - Reserve exhaustive mode for `mutation_run_full`

2. In `lintgate/specification/mutation_engine.py`, separate:
   - `budget_ms`: total call budget
   - `per_mutant_timeout_ms`: timeout for one mutant evaluation
   Suggested default: `500` ms per mutant for sampling

3. In `mcp_tools/_mutation_impl.py`, cap fallback-loaded tests.
   - When the impact map finds no direct refs, do not load every test from every discovered file
   - Load up to roughly `50` tests, ranked by:
     - same directory as source
     - filename/module-name match
     - everything else

4. In `mcp_tools/_mutation_tools_impl.py` and/or `mcp_tools/_mutation_impl.py`, split sampling budget across functions.
   - `per_func_budget = max(total_budget / num_functions, 200ms)`
   - Do not allow one slow function to consume the whole run

5. In `mcp_tools/mutation_tools.py` and the underlying impls, add a hard tool-level timeout/circuit breaker.
   - Suggested hard cap: 10 minutes
   - Return partial results with:
     - `timeout: true`
     - `partial_results: true`
     - `timed_out_functions`

**New output fields:**

- `timeout`
- `partial_results`
- `total_budget_ms`
- `per_func_budget_ms`
- `per_mutant_timeout_ms`
- `tests_considered`
- `tests_loaded`
- `fallback_test_cap_applied`
- `timed_out_functions`

**Tests to add/update:**

- `tests/test_mutation_engine.py`
- `tests/test_mutation_impl.py`
- `tests/test_mutation_tools_impl.py`
- `tests/test_mutation_tools.py`

**Specific test cases required:**

- `mutation_validate_tests` uses sampled mode by default
- sampling respects separate outer budget and per-mutant timeout
- fallback discovery caps loaded tests
- multi-function runs split budget instead of serially consuming the full budget
- tool-level timeout returns partial results instead of hanging

**Exit criteria:**

- No mutation MCP tool can run indefinitely without returning control
- `mutation_validate_tests` becomes safe enough to use routinely in the feedback loop
- Large repos degrade to partial but useful output instead of pathological runtime
- Runtime limits preserve analytical value rather than silently dropping all evidence

### PR1: Discovery truthfulness and test-topology diagnostics

**Goal:** Make mutation outputs honest before making them smarter.

**Modify:**

- `mcp_tools/_mutation_impl.py`
- `mcp_tools/_mutation_tools_impl.py`
- `lintgate/specification/test_impact.py`
- add `lintgate/specification/test_topology.py`

**Add output fields to per-function mutation results:**

- `discovery_state`
  - `NO_TEST_FILES`
  - `TEST_FILES_FOUND_NONE_LINKED`
  - `TESTS_LINKED_ZERO_KILLS`
  - `DISCOVERY_IMPORT_FAILED`
  - `DISCOVERY_OK`
- `topology_state`
  - `NORMAL`
  - `MOCK_BOUNDARY_DOMINANT`
  - `PATCHED_INTERNAL_CALLS`
  - `TOPOLOGY_UNKNOWN`
- `survival_interpretation`
  - `MEANINGFUL`
  - `DISCOVERY_ARTIFACT`
  - `MOCK_BOUNDARY_ARTIFACT`
  - `LOW_CONFIDENCE`
- `linked_test_count`
- `patched_symbol_count`
- `patched_symbols`
- `mocked_call_sites`
- `topology_confidence`

**Implementation details:**

- Extend `DiscoveryDiagnostics` in `mcp_tools/_mutation_impl.py` to emit an explicit `discovery_state`.
- In `lintgate/specification/test_topology.py`, analyze:
  - outbound calls in the target function AST
  - `unittest.mock.patch`, `patch.object`, and monkeypatch-style replacements in linked test files
  - overlap between patched symbols and outbound calls
- If most relevant internal calls are patched, mark `topology_state=MOCK_BOUNDARY_DOMINANT`.
- In `mcp_tools/_mutation_tools_impl.py`, surface a top-level warning when survival is likely a discovery or topology artifact.

**Tests to add/update:**

- `tests/test_mutation_impl.py`
- `tests/test_mutation_tools_impl.py`
- add `tests/test_mutation_topology.py`

**Validation targets:**

- Wayfinder `resolve()`
- at least one LintGate orchestration-style function with patched collaborators

**Exit criteria:**

- No mutation run reports ambiguous "100% survival" without a `survival_interpretation`.
- `resolve()`-style mock-heavy functions are flagged as topology-limited rather than under-specified by default.

### PR2: Survivor records as first-class data

**Goal:** Persist enough empirical detail to support grounded prescriptions.

**Modify:**

- `lintgate/specification/mutation_engine.py`
- `mcp_tools/_mutation_impl.py`
- `mcp_tools/_mutation_tools_impl.py`
- add `lintgate/specification/mutant_reporting.py`

**Add/extend data structures:**

- `Mutant` gains stable `mutant_id`, `lineno`, and `col_offset`
- `MutantResult` gains:
  - `status`
  - `test_name`
  - `killed_by`
  - `elapsed_ms`
- `ProfilingResult` gains:
  - `survivor_records`
  - `killed_records`
  - `sigma_upper_bound`

**Survivor record contract:**

- `mutant_id`
- `category`
- `location`
- `operator`
- `description`
- `diff_summary`
- `status`
- `killed_by`
- `killed_by_test`

**Implementation details:**

- In `mutation_engine.py`, retain the full per-mutant evaluation result instead of only aggregate category counts and killed-only `kill_matrix`.
- In `mutant_reporting.py`, compute a readable `diff_summary` from original vs mutated AST nodes.
- Cache survivor records so `mutation_prescribe` and `mutation_validate_tests` can reuse them.

**Tests to add/update:**

- `tests/test_mutation_engine.py`
- `tests/test_mutation_tools_impl.py`
- add `tests/test_mutant_reporting.py`

**Exit criteria:**

- `mutation_run_full` returns survivor records for every surviving mutant.
- Cached mutation state is sufficient to reconstruct survivor-aware prescriptions without rerunning profiling.

### PR3: Grounded prescriptions and witness generation

**Goal:** Make `mutation_prescribe` answer "what test should I write next?"

**Modify:**

- `mcp_tools/_mutation_tools_impl.py`
- `mcp_tools/_mutation_impl.py`
- add `lintgate/specification/witness_generation.py`
- optionally split `generate_test_skeleton` into a witness-aware helper

**New prescription contract:**

- `function`
- `mutant_id`
- `category`
- `why_this_matters`
- `suggested_input`
- `input_pattern`
- `expected_behavior`
- `assertion_shape`
- `confidence`
- `source_of_evidence`
- `needs_source_review`

**Implementation details:**

- Implement witness generation for easy/high-value cases first:
  - VALUE constant replacement
  - BOUNDARY comparator flips
  - SWAP first-two-arg transpositions
  - simple boolean inversions
- Fall back to branch summary + assertion shape when an exact witness cannot be synthesized.
- `mutation_prescribe_tests` should consume grounded prescriptions first and only fall back to category templates when evidence is insufficient.

**Tests to add/update:**

- `tests/test_mutation_tools.py`
- `tests/test_mutation_tools_impl.py`
- add `tests/test_witness_generation.py`

**Validation targets:**

- Wayfinder `bank_score`
- one boundary-heavy helper in LintGate
- one swap-sensitive helper in LintGate

**Exit criteria:**

- At least 80% of surviving VALUE, BOUNDARY, and SWAP mutants in the validation corpus produce mutant-specific prescriptions.
- The bank-score-style surviving mutant can be closed in one cycle from tool output alone.

### PR4: Greedy trajectory, redundancy, and teaching-set upper bounds

**Goal:** Add the empirical trajectory features that are actually useful to operators.

**Modify:**

- `lintgate/specification/greedy_convergence.py`
- add `lintgate/specification/trajectory_analysis.py`
- `mcp_tools/_mutation_impl.py`
- `mcp_tools/_mutation_tools_impl.py`

**New output fields:**

- `sigma_upper_bound`
- `teaching_set_upper_bound`
- `approx_phase_transition`
- `tail_test_candidates`
- `redundant_tests`
- `trajectory_summary`

**Implementation details:**

- Keep all naming explicitly approximate.
- Use survivor records and full kill data to compute:
  - greedy ordering
  - redundant tests
  - tail onset
  - high-value next tests
- Feed the summary into `mutation_run_full` output and `mutation_validate_tests`.

**Tests to add/update:**

- `tests/test_greedy_convergence.py`
- add `tests/test_trajectory_analysis.py`
- update `tests/test_redundancy_tools.py` if output contracts change

**Exit criteria:**

- The tool can identify tests that add no new mutant discrimination.
- Tail detection improves prescription prioritization versus flat category counts.

### PR5: Static/empirical reconciliation in `spec_file_analyze`

**Goal:** Make spec analysis trustworthy by showing how symbolic and empirical layers relate.

**Modify:**

- `lintgate/specification/types.py`
- `lintgate/specification/file_analyzer.py`
- `mcp_tools/_specification_helpers.py`
- possibly `lintgate/specification/ledger.py`

**Add an empirical overlay contract:**

- `status`
  - `NO_EMPIRICAL_DATA`
  - `AGREES`
  - `CONTRADICTS`
  - `DISCOVERY_FAILURE`
  - `TOPOLOGY_LIMITED`
- `mutation_runs_seen`
- `empirical_sigma_upper_bound`
- `empirical_tail`
- `overlay_confidence`
- `overlay_rationale`

**Implementation details:**

- Keep symbolic sigma/regime/phase as the predictive layer.
- Pull mutation cache into `spec_file_analyze` and compute an explicit overlay.
- Downgrade confidence when:
  - discovery failed
  - topology is mock-dominant
  - empirical tail contradicts symbolic tractability

**Tests to add/update:**

- `tests/test_file_analyzer.py`
- `tests/test_project_rollup.py`
- add `tests/test_static_empirical_overlay.py`

**Exit criteria:**

- `spec_file_analyze` no longer silently collapses discovery or topology problems into generic low spec coverage.
- The operator can see whether static and empirical layers agree in one tool call.

### PR6: Cross-lens decomposition guidance

**Goal:** Raise the bar for extraction recommendations so they track the codebase ideal rather than local mutation noise.

**Modify:**

- `mcp_tools/_mutation_tools_impl.py::impl_decompose`
- `mcp_tools/convergence_tools.py`
- `lintgate/convergence/integration.py`
- `lintgate/specification/composition.py`
- optionally add `lintgate/specification/decomposition_evidence.py`

**Required evidence inputs:**

- mutation tail or survivor diversity
- structural complexity or extraction evidence
- interface/call-boundary complexity
- coupling or shared-state indicators
- topology-adjusted confidence

**New decomposition output contract:**

- `recommendation`
  - `KEEP_TESTING`
  - `EXTRACT_BOUNDARY`
  - `INSUFFICIENT_EVIDENCE`
- `responsibility_boundary`
- `supporting_evidence`
- `cross_lens_score`
- `expected_benefits`
  - `reduced_coupling`
  - `cleaner_interface`
  - `better_local_testability`
  - `easier_repair`

**Implementation details:**

- Remove the current "two surviving categories => decompose" implication.
- Require at least two independent lenses before recommending extraction.
- If topology is mock-dominant, prefer `KEEP_TESTING` or `INSUFFICIENT_EVIDENCE`.

**Tests to add/update:**

- `tests/test_decompose_bridge.py`
- `tests/test_convergence_tools.py`
- add `tests/test_decomposition_evidence.py`

**Validation targets:**

- Wayfinder `_apply_transformers_compat_shims`
- one LintGate function where extraction is genuinely warranted
- one LintGate function where testing, not extraction, is the correct answer

**Exit criteria:**

- Extraction recommendations become less frequent but more defensible.
- Reviewers agree the recommendation boundary is materially better than mutation-only heuristics.

### PR7: Documentation and validation pass

**Goal:** Ship the semantics cleanly and validate them end-to-end.

**Modify:**

- `README.md`
- `docs/design.md`
- `docs/agent/AGENTS.md`
- optionally add a retrospective under `docs/retrospectives/`

**Required documentation updates:**

- mutation tool descriptions
- spec tool descriptions
- new output contract fields
- discovery vs topology diagnostics
- upper-bound language for greedy outputs
- cross-lens decomposition requirement

**Validation procedure:**

1. Run the full loop on Wayfinder:
   `spec_file_analyze` -> `mutation_run_full` -> `mutation_prescribe` -> write tests -> `mutation_validate_tests`
2. Run the same loop on a selected LintGate file.
3. Record:
   - whether the next test was obvious from tool output
   - whether survival interpretation was truthful
   - whether decomposition advice matched reviewer judgment

**Exit criteria:**

- The loop closes on both Wayfinder and LintGate.
- Tool descriptions and docs match actual behavior.

---

## Deferred work

These remain explicitly deferred until the execution plan above is complete and validated:

- symmetry refinement
- free-energy views
- prediction verification ledger

They may be added later as secondary analysis, but not before the direct operator loop is solid.

---

## Explicit de-prioritization

These ideas are interesting, but they should not lead the roadmap.

### Do not start with a new standalone script as the primary interface

If a script exists, it should be a developer utility. The product surface is the MCP tools.

### Do not lead with input-domain symmetry sampling

LintGate already has a mutation-based symmetry classifier. Improve it only after survivor reporting and static/empirical reconciliation are solid.

### Do not claim exact teaching sets from greedy algorithms

Exact teaching-set computation is a separate problem. Approximate results are still useful if labeled correctly.

### Do not let free-energy reporting displace basic prescription quality

If the operator still has to infer the next test manually, free-energy plots are premature.

---

## Validation plan

Wayfinder remains the first validation corpus because it exposed the real operator problems:

- weak/generic prescriptions
- discovery failure brittleness
- gap between mutation evidence and next action

LintGate itself is the second validation corpus because the tool should be able to improve its own test and decomposition guidance.

### Validation targets

Use a mix of:

- pure utility functions
- branch-heavy pure functions
- orchestration functions with mocks
- stateful methods
- previously hard-to-diagnose survivors

### Metrics that matter

1. **Prescription usability**
   How often can an engineer write the next test directly from the tool output?

2. **Feedback-loop integrity**
   How often does the system correctly distinguish discovery failure from genuine survival?

3. **Delta-to-value**
   How many prescribed tests actually reduce survival rate after `mutation_validate_tests`?

4. **Decomposition precision**
   How often are extraction recommendations later judged correct by a reviewer?

5. **Time-to-first-useful-action**
   Time from running the tool to writing the next high-value test.

6. **Movement toward cleaner structure**
   Whether accepted recommendations actually reduce coupling, clarify interfaces, or improve local testability in a reviewer-visible way.

7. **Runtime boundedness**
   Whether mutation tools return within their configured budgets and degrade to partial results rather than hanging or exploding.

---

## Success criteria

The plan succeeds when all of the following are true:

1. `mutation_run_full` returns survivor-level evidence rich enough for grounded prescriptions.

2. `mutation_prescribe` usually answers "what exact test should I write next?" rather than only "what category is weak?"

3. `spec_file_analyze` no longer collapses into misleading low-confidence outputs when empirical collection fails or when mutation results are topology-limited by mocks.

4. `mutation_prescribe_tests` generates templates that are anchored to actual mutant evidence for the easy categories.

5. Mutation MCP tools become operationally safe: bounded by explicit budgets, capable of partial return, and usable in the normal feedback loop without pathological runtime.

6. `convergence_analyze` and `mutation_decompose` become more trustworthy by recommending extraction less often, but with better evidence.

7. On Wayfinder and LintGate, operators can close the loop:
   `spec_file_analyze` -> `mutation_run_full` -> `mutation_prescribe` -> write tests -> `mutation_validate_tests`

8. Recommendations consistently move code toward cleaner abstractions, stronger orthogonality, and easier local repair rather than merely increasing test count.

9. The implementation makes LintGate feel more like an engineering partner for idealizing a codebase and less like a theory dashboard.

---

## Bottom line

The right plan is not "implement every theorem-shaped feature."

The right plan is:

1. make mutation evidence reliable
2. turn survivors into concrete test guidance
3. reconcile symbolic and empirical views with structural and interface lenses
4. use cross-lens convergence to push code toward a cleaner, more orthogonal, more repairable form
5. only then add deeper theory diagnostics if they improve real decisions

If those things are done well, LintGate becomes a genuinely powerful and practical engineering tool for approximating the platonic ideal of a codebase without losing pragmatism.
