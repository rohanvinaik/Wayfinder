from __future__ import annotations

import sqlite3
import unittest

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import (
    DuckyProgram,
    _build_closer_programs,
    build_ducky_programs,
    build_ducky_symbolic_frontier,
    execute_ducky_program,
    run_ducky_on_goal,
)
from src.nav_contracts import LeanFeedback, TacticResult
from src.proof_search import SearchConfig, _SearchEnv, _SearchState, _try_lane
from src.resolution import SearchContext


class _FakeLean:
    def __init__(self, transitions: dict[tuple[str, str], TacticResult]) -> None:
        self.transitions = transitions

    def try_tactic(self, goal: str, tactic: str) -> TacticResult:
        return self.transitions.get(
            (goal, tactic),
            TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message="miss",
                feedback=LeanFeedback(stage="tactic_exec", category="other", messages=[], raw_error="miss"),
            ),
        )


class _MockVar:
    def __init__(self, name: str, t: str) -> None:
        self.name = name
        self.t = t


class _MockGoal:
    def __init__(self, target: str, variables: list[_MockVar]) -> None:
        self.target = target
        self.variables = variables


class _MockGoalState:
    def __init__(self, goals: list[_MockGoal]) -> None:
        self.goals = goals


class _FakeLeanWithState(_FakeLean):
    def __init__(
        self,
        transitions: dict[tuple[str, str], TacticResult],
        goal_states: dict[tuple[str, str], _MockGoalState],
    ) -> None:
        super().__init__(transitions)
        self._current_env_key = "test-env"
        self._goal_states = goal_states


class TestDrDuckyExecutor(unittest.TestCase):
    def test_build_ducky_programs_includes_arithmetic_and_premise_bank(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT, namespace TEXT, file_path TEXT, provenance TEXT);
            CREATE TABLE accessible_premises (theorem_id INTEGER NOT NULL, premise_id INTEGER NOT NULL, PRIMARY KEY (theorem_id, premise_id));
            """
        )
        conn.execute(
            "INSERT INTO entities(id, name, entity_type, namespace, file_path, provenance) VALUES (1, ?, 'lemma', '', '', '')",
            ("Demo.theorem",),
        )
        conn.execute(
            "INSERT INTO entities(id, name, entity_type, namespace, file_path, provenance) VALUES (2, ?, 'lemma', '', '', '')",
            ("Int.abs_lt_one_iff",),
        )
        conn.execute(
            "INSERT INTO entities(id, name, entity_type, namespace, file_path, provenance) VALUES (3, ?, 'lemma', '', '', '')",
            ("Demo.helper",),
        )
        conn.execute("INSERT INTO accessible_premises(theorem_id, premise_id) VALUES (1, 2)")
        conn.execute("INSERT INTO accessible_premises(theorem_id, premise_id) VALUES (1, 3)")
        row = {
            "theorem_id": "Demo.theorem",
            "last_goal": "|n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 12,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.theorem",
            goal_text="|n| < 1",
            conn=conn,
            accessible_theorem_id=1,
            max_programs=64,
        )
        scripts = {program.script for program in programs}
        self.assertIn("norm_num", scripts)
        self.assertIn("rw [Int.abs_lt_one_iff]", scripts)
        self.assertIn("exact Int.abs_lt_one_iff", scripts)
        exact_program = next(program for program in programs if program.script == "exact Int.abs_lt_one_iff")
        self.assertEqual(exact_program.skeleton_id, "exact_accessible_lemma")
        self.assertEqual(exact_program.bindings["lemma"], "Int.abs_lt_one_iff")
        self.assertEqual(exact_program.engine_name, "ContextTransportEngine")
        self.assertEqual(exact_program.backend_family, "rosette_proof_dsl")
        self.assertEqual(exact_program.projector_status, "projected")
        self.assertEqual(exact_program.projector_backend, "proof_projector_v1")

    def test_build_ducky_programs_mines_local_hypothesis_closers(self) -> None:
        row = {
            "theorem_id": "Demo.local_fact",
            "last_goal": "h : n = 0\n⊢ n = 0",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "local_eq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 4,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.local_fact",
            goal_text=row["last_goal"],
            max_programs=40,
        )
        scripts = {program.script for program in programs}
        self.assertIn("exact h", scripts)
        self.assertIn("simpa using h", scripts)
        exact_program = next(program for program in programs if program.script == "exact h")
        self.assertEqual(exact_program.skeleton_id, "exact_local_fact")
        self.assertEqual(exact_program.bindings["fact"], "h")
        self.assertEqual(exact_program.certificate_shape, "term_transport")

    def test_closer_programs_include_symbolic_closers(self) -> None:
        programs = _build_closer_programs(
            "⊢ True",
            "atomic_prop",
            theorem_id="Demo.symbolic_close",
        )
        scripts = {program.script for program in programs}
        self.assertIn("solve_by_elim", scripts)
        self.assertIn("apply?", scripts)

    def test_build_ducky_programs_derives_mp_and_symm_facts(self) -> None:
        row = {
            "theorem_id": "Demo.fact_graph",
            "last_goal": "hiff : P ↔ Q\nhp : P\nheq : a = b\n⊢ Q",
            "last_goal_bucket": "other",
            "reasoning_gap_family": "forward_context_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 4,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.fact_graph",
            goal_text=row["last_goal"],
            max_programs=32,
        )
        scripts = {program.script for program in programs}
        self.assertIn("exact (hiff.mp hp)", scripts)

        row_symm = dict(row)
        row_symm["last_goal"] = "heq : a = b\n⊢ b = a"
        row_symm["goal_state"] = row_symm["last_goal"]
        capsule_symm = build_goal_capsule(row_symm)
        programs_symm = build_ducky_programs(
            capsule_symm,
            theorem_id="Demo.fact_graph",
            goal_text=row_symm["last_goal"],
            max_programs=32,
        )
        self.assertIn("exact heq.symm", {program.script for program in programs_symm})

    def test_eqsat_backend_emits_symbolic_rewrite_close(self) -> None:
        row = {
            "theorem_id": "Demo.eqsat",
            "last_goal": "h : a = b\n⊢ f b = f a",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "local_eq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 7,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.eqsat",
            goal_text=row["last_goal"],
            max_programs=32,
        )
        eqsat_programs = [program for program in programs if program.backend_family == "egglog_eqsat"]
        self.assertTrue(any(program.script.endswith("rfl") and program.script.startswith("rw [") for program in eqsat_programs))
        transitions = {
            ("h : a = b\n⊢ f b = f a", "rw [← h]"): TacticResult(
                success=True,
                tactic="rw [← h]",
                premises=[],
                new_goals=["f a = f a"],
                feedback=LeanFeedback.success(),
            ),
            ("h : a = b\n⊢ f b = f a", "rw [h]"): TacticResult(
                success=True,
                tactic="rw [h]",
                premises=[],
                new_goals=["f b = f b"],
                feedback=LeanFeedback.success(),
            ),
            ("f a = f a", "rfl"): TacticResult(
                success=True,
                tactic="rfl",
                premises=[],
                new_goals=[],
                feedback=LeanFeedback.success(),
            ),
            ("f b = f b", "rfl"): TacticResult(
                success=True,
                tactic="rfl",
                premises=[],
                new_goals=[],
                feedback=LeanFeedback.success(),
            ),
        }
        closed = [execute_ducky_program(_FakeLean(transitions), row["last_goal"], program) for program in eqsat_programs]
        self.assertTrue(any(run.closed for run in closed))

    def test_build_ducky_programs_uses_cached_goal_state_for_target_only_goal(self) -> None:
        row = {
            "theorem_id": "Demo.cached_goal",
            "last_goal": "n = 0",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "local_eq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 4,
        }
        capsule = build_goal_capsule(row)
        lean = _FakeLeanWithState(
            transitions={},
            goal_states={
                ("test-env", "n = 0"): _MockGoalState(
                    [_MockGoal("n = 0", [_MockVar("n", "Int"), _MockVar("h", "|n| < 1")])]
                )
            },
        )
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.cached_goal",
            goal_text="n = 0",
            lean=lean,
            max_programs=32,
        )
        scripts = {program.script for program in programs}
        self.assertIn("exact Int.abs_lt_one_iff.mp h", scripts)

    def test_witness_backend_builds_existential_witness(self) -> None:
        row = {
            "theorem_id": "Demo.witness",
            "last_goal": "hp : P a\n⊢ ∃ x, P x",
            "last_goal_bucket": "exists",
            "reasoning_gap_family": "witness_construction_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
            "attempts": 5,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.witness",
            goal_text=row["last_goal"],
            max_programs=32,
        )
        self.assertIn("exact ⟨a, hp⟩", {program.script for program in programs})

    def test_relational_backend_builds_membership_pair(self) -> None:
        row = {
            "theorem_id": "Demo.relational",
            "last_goal": "ha : x ∈ A\nhb : x ∈ B\n⊢ x ∈ A ∩ B",
            "last_goal_bucket": "membership",
            "reasoning_gap_family": "membership_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
            "attempts": 5,
        }
        capsule = build_goal_capsule(row)
        frontier = build_ducky_symbolic_frontier(
            capsule,
            theorem_id="Demo.relational",
            goal_text=row["last_goal"],
            max_programs=32,
        )
        programs = frontier["programs"]
        self.assertIn("exact ⟨ha, hb⟩", {program.script for program in programs})
        self.assertTrue(any(outcome.get("backend_family") == "kodkod_relational" for outcome in frontier["engine_outcomes"]))

    def test_arith_backend_emits_abs_normalization_program(self) -> None:
        row = {
            "theorem_id": "Demo.abs",
            "last_goal": "⊢ |n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 0,
            "goals_remaining": 1,
            "attempts": 3,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.abs",
            goal_text=row["last_goal"],
            max_programs=32,
        )
        self.assertIn("rw [Int.abs_lt_one_iff]; omega", {program.script for program in programs})

    def test_disabled_tactics_are_filtered_from_program_frontier(self) -> None:
        row = {
            "theorem_id": "Demo.abs_disabled",
            "last_goal": "⊢ |n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 0,
            "goals_remaining": 1,
            "attempts": 3,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.abs_disabled",
            goal_text=row["last_goal"],
            max_programs=32,
            disabled_tactics={"linarith", "nlinarith"},
        )
        scripts = {program.script for program in programs}
        self.assertIn("rw [Int.abs_lt_one_iff]; omega", scripts)
        self.assertTrue(all("linarith" not in script for script in scripts))

    def test_build_ducky_programs_can_filter_by_backend_family(self) -> None:
        row = {
            "theorem_id": "Demo.backend_filter",
            "last_goal": "h : a = b\n⊢ f b = f a",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "local_eq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 2,
            "goals_remaining": 1,
            "attempts": 7,
        }
        capsule = build_goal_capsule(row)
        programs = build_ducky_programs(
            capsule,
            theorem_id="Demo.backend_filter",
            goal_text=row["last_goal"],
            max_programs=32,
            allowed_backend_families={"egglog_eqsat"},
        )
        self.assertTrue(programs)
        self.assertTrue(all(program.backend_family == "egglog_eqsat" for program in programs))
        self.assertTrue(all(program.engine_name == "EqSatEngine" for program in programs))

    def test_execute_ducky_program_runs_multi_step_close(self) -> None:
        transitions = {
            ("∀ (n : Int), |n| < 1 -> n = 0", "intros"): TacticResult(
                success=True,
                tactic="intros",
                premises=[],
                new_goals=["n = 0"],
                feedback=LeanFeedback.success(),
            ),
            ("n = 0", "aesop"): TacticResult(
                success=True,
                tactic="aesop",
                premises=[],
                new_goals=[],
                feedback=LeanFeedback.success(),
            ),
        }
        program = DuckyProgram(
            program_id="binder:intros_aesop",
            bank="binder_instantiation",
            specialist="binder_drilldown",
            tactics=["intros", "aesop"],
            rationale="Expose binders then close locally.",
            score=1.0,
        )
        run = execute_ducky_program(_FakeLean(transitions), "∀ (n : Int), |n| < 1 -> n = 0", program)
        self.assertTrue(run.progressed)
        self.assertTrue(run.closed)
        self.assertEqual(run.tactics_applied, ["intros", "aesop"])

    def test_run_ducky_on_goal_prefers_verified_progress(self) -> None:
        transitions = {
            ("|n| < 1", "norm_num"): TacticResult(
                success=True,
                tactic="norm_num",
                premises=[],
                new_goals=["n = 0"],
                feedback=LeanFeedback.success(),
            ),
        }
        row = {
            "theorem_id": "Demo.abs_goal",
            "last_goal": "|n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
            "attempts": 9,
        }
        capsule = build_goal_capsule(row)
        result = run_ducky_on_goal(
            "|n| < 1",
            theorem_id="Demo.abs_goal",
            lean=_FakeLean(transitions),
            capsule=capsule,
            max_programs=12,
        )
        self.assertTrue(result.started)
        self.assertTrue(result.progressed)
        self.assertFalse(result.closed)
        self.assertEqual(result.final_goal, "n = 0")
        self.assertIsNotNone(result.ledger_snapshot)
        self.assertTrue(result.engine_outcomes)
        self.assertTrue(result.projector_outcomes)
        self.assertTrue(any(outcome.get("backend_family") for outcome in result.engine_outcomes))

    def test_run_ducky_on_goal_uses_post_progress_closer(self) -> None:
        transitions = {
            ("|n| < 1", "rw [Int.abs_lt_one_iff]"): TacticResult(
                success=True,
                tactic="rw [Int.abs_lt_one_iff]",
                premises=[],
                new_goals=["h : n = 0\n⊢ n = 0"],
                feedback=LeanFeedback.success(),
            ),
            ("h : n = 0\n⊢ n = 0", "exact h"): TacticResult(
                success=True,
                tactic="exact h",
                premises=[],
                new_goals=[],
                feedback=LeanFeedback.success(),
            ),
        }
        row = {
            "theorem_id": "Demo.abs_goal_close",
            "last_goal": "|n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
            "attempts": 9,
        }
        capsule = build_goal_capsule(row)
        result = run_ducky_on_goal(
            "|n| < 1",
            theorem_id="Demo.abs_goal_close",
            lean=_FakeLean(transitions),
            capsule=capsule,
            max_programs=16,
        )
        self.assertTrue(result.closed)
        self.assertIsNotNone(result.winning_program)
        self.assertIn("exact h", result.winning_program["tactics_applied"])

    def test_run_ducky_on_goal_uses_cached_state_for_target_only_followup_goal(self) -> None:
        transitions = {
            ("|n| < 1", "rw [Int.abs_lt_one_iff]"): TacticResult(
                success=True,
                tactic="rw [Int.abs_lt_one_iff]",
                premises=[],
                new_goals=["n = 0"],
                feedback=LeanFeedback.success(),
            ),
            ("n = 0", "exact Int.abs_lt_one_iff.mp h"): TacticResult(
                success=True,
                tactic="exact Int.abs_lt_one_iff.mp h",
                premises=[],
                new_goals=[],
                feedback=LeanFeedback.success(),
            ),
        }
        lean = _FakeLeanWithState(
            transitions=transitions,
            goal_states={
                ("test-env", "n = 0"): _MockGoalState(
                    [_MockGoal("n = 0", [_MockVar("n", "Int"), _MockVar("h", "|n| < 1")])]
                )
            },
        )
        row = {
            "theorem_id": "Demo.abs_goal_cached_close",
            "last_goal": "|n| < 1",
            "last_goal_bucket": "inequality",
            "reasoning_gap_family": "local_ineq_close",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
            "attempts": 9,
        }
        capsule = build_goal_capsule(row)
        result = run_ducky_on_goal(
            "|n| < 1",
            theorem_id="Demo.abs_goal_cached_close",
            lean=lean,
            capsule=capsule,
            max_programs=16,
        )
        self.assertTrue(result.closed)
        self.assertIsNotNone(result.winning_program)
        self.assertIn("exact Int.abs_lt_one_iff.mp h", result.winning_program["script"])

    def test_proof_search_dr_ducky_lane_consumes_progress(self) -> None:
        transitions = {
            ("|n| < 1", "norm_num"): TacticResult(
                success=True,
                tactic="norm_num",
                premises=[],
                new_goals=["n = 0"],
                feedback=LeanFeedback.success(),
            ),
        }
        state = _SearchState(open_goals=["|n| < 1"], theorem_id="Demo.abs_goal", closed_goals=["prelude"])
        env = _SearchEnv(
            conn=sqlite3.connect(":memory:"),
            lean=_FakeLean(transitions),  # type: ignore[arg-type]
            anchor_id_map=None,
            max_candidates=4,
        )
        context = SearchContext(accessible_theorem_id=None)
        cfg = SearchConfig(dr_ducky_enabled=True, dr_ducky_max_programs=12)
        progress = _try_lane(
            "dr_ducky",
            "|n| < 1",
            0,
            nav_output=None,  # type: ignore[arg-type]
            state=state,
            env=env,
            context=context,
            cfg=cfg,
        )
        self.assertTrue(progress)
        self.assertEqual(state.close_provenance[-1], "dr_ducky")
        self.assertIn("n = 0", state.open_goals)


if __name__ == "__main__":
    unittest.main()
