# Dr_Ducky Validation Report

## Scope

This report now separates the six validation layers required by the canonical Dr. Ducky architecture:

1. routing validation
2. theorem-faithful replay validation
3. certificate generation
4. projector compilation
5. honest progress lift
6. honest closure lift

Run dir:
- `runs/exp_som012_hard_eval_r2`

## 1. Routing Validation

Population:
- input rows: `2921`
- validated local rows: `1984`
- local capsules materialized: `1975`

Key routing checks:
- recursive circuit-breaker routing: `47/52` (`90.38%`)
- symbolic sandbox routing: `787/889` (`88.53%`)
- domain numeric-solver suppression: `285/285` (`100.0%`)

Family alignment:
- `local_eq_close`: `35/35` aligned (`100.0%`)
- `local_ineq_close`: `10/10` aligned (`100.0%`)
- `membership_close`: `4/4` aligned (`100.0%`)
- `witness_construction_close`: `5/5` aligned (`100.0%`)
- `forward_context_close`: `6/6` aligned (`100.0%`)
- `forall_close`: `5/5` aligned (`100.0%`)
- `iff_close`: `4/4` aligned (`100.0%`)
- `subset_close`: `1/1` aligned (`100.0%`)
- `atomic_prop_close`: `3/3` aligned (`100.0%`)

## 2. Theorem-Faithful Replay Validation

- replay sample rows: `20`
- theorem-faithful starts: `20/20`
- replay tier mix:
  - `C`: `12`
  - `B`: `7`
  - `A`: `1`

## 3. Certificate Generation

- generated certificates: `1014`

## 4. Projector Compilation

- projector successes: `1014`
- projector rejections: `0`

## 5. Honest Progress Lift

- honest progress: `10`
- Lean compile proxy count: `248`

## 6. Honest Closure Lift

- honest closures: `0`

## 7. Production Runtime Smoke

The in-repo production backends now have a lightweight runtime smoke pass at [runtime_smoke_summary.json](/Users/rohanvinaik/Projects/Wayfinder/runs/exp_som012_hard_eval_r2/bundle/dr_ducky/runtime_smoke_summary.json):

- total cases: `5`
- progressed cases: `5`
- closed cases: `4`
- backend closures observed:
  - `egglog_eqsat`
  - `rosette_proof_dsl`
  - `kodkod_relational`
- remaining non-closure case:
  - `lean_arith` progressing `|n| < 1` to `n = 0`

This is not frozen-corpus closure evidence. It is a sanity check that the backend runtimes are now executing real symbolic programs in the repo rather than existing only as packet labels.

## Targeted Cases

### `Batteries.UnionFind.rootD_parent`

- Residual bucket: `multi_goal_large_progress`
- Gap family: `theorem_replanner`
- Goal bucket: `equality`
- Specialist targets: `recursive_circuit_breaker, human_calculator, side_condition_sweeper`
- Suppression hints: `suppress_repeat_rw, suppress_repeat_norm_num, suppress_full_recursive_unfold, suppress_blank_lane_retry, suppress_global_replanner`
- Top prescriptions: `recursive_loop_circuit_break, bounded_unfold, forward_local_context, sweep_side_conditions, select_local_fact`

### `ModularGroup.tendsto_normSq_coprime_pair`

- Residual bucket: `single_goal_stall`
- Gap family: `single_goal_stall`
- Goal bucket: `forall`
- Specialist targets: `symbolic_sandbox, binder_drilldown, filter_reasoner`
- Suppression hints: `prefer_symbolic_sandbox`
- Top prescriptions: `enter_symbolic_sandbox, normalize_coercions, normalize_arithmetic, binder_drilldown, normalize_eventual_filter, reduce_pointwise`

### `ModularGroup.smul_eq_lcRow0_add`

- Residual bucket: `single_goal_stall`
- Gap family: `single_goal_stall`
- Goal bucket: `forall`
- Specialist targets: `symbolic_sandbox, binder_drilldown`
- Suppression hints: `prefer_symbolic_sandbox`
- Top prescriptions: `enter_symbolic_sandbox, normalize_coercions, normalize_arithmetic, binder_drilldown`

### `ModularGroup.eq_zero_of_mem_fdo_of_T_zpow_mem_fdo`

- Residual bucket: `single_goal_near_miss`
- Gap family: `local_ineq_close`
- Goal bucket: `inequality`
- Specialist targets: `symbolic_sandbox, human_calculator, side_condition_sweeper`
- Suppression hints: `suppress_global_replanner, prefer_symbolic_sandbox`
- Top prescriptions: `enter_symbolic_sandbox, saturate_equality, forward_local_context, normalize_arithmetic, sweep_side_conditions, select_local_fact`
