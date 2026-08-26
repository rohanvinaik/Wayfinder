from __future__ import annotations

import sqlite3
import unittest

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import (
    BANK_SUPPRESSED,
    DuckyProgram,
    _accessible_lemmas,
    _build_closer_programs,
    bank_weight,
    build_ducky_programs,
    build_ducky_symbolic_frontier,
    execute_ducky_program,
    program_score,
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


class BankGovernanceTest(unittest.TestCase):
    """Wave C: the capsule's filtering layer finally governs the firehose."""

    def _capsule(self, goal: str = "⊢ CategoryTheory.IsIso f"):
        return build_goal_capsule(
            {
                "theorem_id": "T",
                "last_goal": goal,
                "last_goal_bucket": "other",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 0,
                "goals_remaining": 1,
                "attempts": 0,
                "remaining_goals_snapshot": [goal],
            }
        )

    def test_absent_bank_is_neutral_not_zero(self):
        # Absence ≠ suppression: a bank the capsule holds no opinion on
        # must not silently starve its program families.
        capsule = self._capsule()
        known = {p.name for p in capsule.bank_priors}
        self.assertNotIn("no_such_bank", known)
        self.assertEqual(bank_weight(capsule, "no_such_bank"), 0.5)

    def test_suppressed_bank_returns_the_named_code(self):
        capsule = self._capsule()
        suppressed = [p.name for p in capsule.bank_priors if p.suppressed]
        if not suppressed:
            self.skipTest("capsule shape produced no suppressed banks")
        self.assertEqual(bank_weight(capsule, suppressed[0]), BANK_SUPPRESSED)

    def test_program_score_neutral_weight_preserves_base(self):
        self.assertEqual(program_score(0.8, 0.5), 0.8)

    def test_program_score_monotone_in_weight_and_base(self):
        self.assertLess(program_score(0.8, 0.0), program_score(0.8, 0.5))
        self.assertLess(program_score(0.8, 0.5), program_score(0.8, 1.0))
        self.assertLess(program_score(0.5, 0.5), program_score(0.8, 0.5))

    def test_suppressed_banks_generate_no_premise_programs(self):
        # An abstract structural goal suppresses the numeric banks; the
        # arith-flavored premise chains must not be generated at all.
        capsule = self._capsule()
        suppressed = {p.name for p in capsule.bank_priors if p.suppressed}
        if not suppressed:
            self.skipTest("capsule shape produced no suppressed banks")
        programs = build_ducky_programs(
            capsule,
            theorem_id="T",
            goal_text="⊢ CategoryTheory.IsIso f",
            lean=None,
            conn=None,
        )
        for prog in programs:
            self.assertNotIn(prog.bank, suppressed, prog.program_id)


class SitelessPremisePoolTest(unittest.TestCase):
    """Wave C: 'no site' is not 'no premises' inside the capsule either."""

    def _network(self):
        from src.proof_network import init_db

        conn = init_db(":memory:")
        conn.execute("INSERT INTO entities (id, name) VALUES (2, 'Nat.add_comm')")
        conn.execute("UPDATE entities SET entity_type='lemma' WHERE id=2")
        for aid, label in [(1, "ns:Nat"), (2, "name:add")]:
            conn.execute("INSERT INTO anchors (id, label) VALUES (?,?)", (aid, label))
        for aid in (1, 2):
            conn.execute(
                "INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (2,?)", (aid,)
            )
        conn.commit()
        return conn

    def test_siteless_goal_draws_the_library_pool(self):
        conn = self._network()
        pool = _accessible_lemmas(
            "External.no_site",
            "Nat.add a b = Nat.add b a",
            conn,
            accessible_theorem_id=None,
        )
        self.assertEqual(pool, ["Nat.add_comm"])

    def test_no_anchor_evidence_abstains(self):
        conn = self._network()
        self.assertEqual(
            _accessible_lemmas("T", "∀ ε > 0, True", conn, accessible_theorem_id=None),
            [],
        )

    def test_schemaless_connection_fails_closed(self):
        conn = sqlite3.connect(":memory:")
        self.assertEqual(
            _accessible_lemmas("T", "Nat.add a b", conn, accessible_theorem_id=None),
            [],
        )


class DuckyTelemetryTest(unittest.TestCase):
    """The diagnosis surface: collect_trace runs record every program she
    tried, and SearchResult carries it (both seams — the pass-1b lesson)."""

    def test_collect_trace_records_tried_programs_on_state(self):
        goal = "h : P q\n⊢ P q"
        lean = _FakeLean(
            {
                (goal, "exact h"): TacticResult(
                    success=True, tactic="exact h", premises=[], new_goals=[],
                    error_message="",
                    feedback=LeanFeedback(stage="tactic_exec", category="ok",
                                          messages=[], raw_error=""),
                )
            }
        )
        state = _SearchState(open_goals=[goal], theorem_id="T")
        env = _SearchEnv(conn=None, lean=lean, anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(dr_ducky_enabled=True, dr_ducky_max_programs=24,
                           collect_trace=True)
        progress = _try_lane(
            "dr_ducky", goal, 0, nav_output=None, state=state, env=env,
            context=SearchContext(accessible_theorem_id=None), cfg=cfg,
        )
        self.assertTrue(progress)
        self.assertEqual(len(state._ducky_telemetry), 1)
        entry = state._ducky_telemetry[0]
        self.assertTrue(entry["closed"])
        self.assertTrue(entry["tried"], "every tried program must be recorded")
        winner = [t for t in entry["tried"] if t["closed"]]
        self.assertTrue(winner)
        self.assertIn("exact h", winner[0]["script"])

    def test_untraced_runs_stay_lean(self):
        goal = "h : P q\n⊢ P q"
        lean = _FakeLean({})
        state = _SearchState(open_goals=[goal], theorem_id="T")
        env = _SearchEnv(conn=None, lean=lean, anchor_id_map=None, max_candidates=8)
        cfg = SearchConfig(dr_ducky_enabled=True, collect_trace=False)
        _try_lane("dr_ducky", goal, 0, nav_output=None, state=state, env=env,
                  context=SearchContext(accessible_theorem_id=None), cfg=cfg)
        self.assertEqual(state._ducky_telemetry, [])


class ComposedPlanSeamsTest(unittest.TestCase):
    """EXP-RF-008 slice 2's three repaired seams, pinned:
    (1) hole-bearing certificates are SELF-PROJECTING at the construction
    seam (every producer path inherits it); (2) the CLASS eligibility path
    delegates to _engine_banks (the always-on exemption — three layers
    must agree); (3) plan programs reach the slate end-to-end."""

    GOAL = "h : ∃ n : Nat, Foo.bar n\nhk : Foo.bar q\n⊢ ∃ m : Nat, Foo.bar m"

    def _fixture(self):
        from src.proof_network import init_db

        conn = init_db(":memory:")
        conn.execute("INSERT INTO entities (id, name) VALUES (1, 'Site.thm')")
        for i in range(2, 8):
            conn.execute(
                "INSERT INTO entities (id, name) VALUES (?, ?)", (i, f"Foo.bar_lemma_{i}")
            )
            conn.execute(
                "INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, ?)",
                (i,),
            )
        conn.commit()
        row = {
            "theorem_id": "Site.thm", "last_goal": self.GOAL,
            "last_goal_bucket": "exists", "residual_bucket": "single_goal_near_miss",
            "goals_closed": 0, "goals_remaining": 1, "attempts": 0,
            "remaining_goals_snapshot": [self.GOAL],
        }
        return conn, build_goal_capsule(row)

    def test_class_path_engines_include_always_on_hosts(self):
        # The exists bucket authors only WitnessEngine; the always-on
        # transport banks' host engine must still be eligible.
        from src.dr_ducky_executor import ContextTransportEngine

        _conn, capsule = self._fixture()
        self.assertNotIn("ContextTransportEngine", capsule.allowed_engines)
        banks = ContextTransportEngine().eligible_banks(capsule)
        self.assertIn("context_forward", banks)
        self.assertIn("local_fact_selector", banks)

    def test_hole_bearing_certificates_are_self_projecting(self):
        from src.dr_ducky_executor import (
            _build_binding_space,
            _build_shadow_ledger,
            _instantiate_skeleton_certificates,
            _make_engine_request,
        )

        conn, capsule = self._fixture()
        space = _build_binding_space(None, "Site.thm", self.GOAL, conn, 1)
        ledger = _build_shadow_ledger(space, theorem_id="Site.thm", goal_text=self.GOAL)
        sk = next(s for s in capsule.proof_skeletons if s.skeleton_id == "witness_transport")
        req = _make_engine_request(capsule=capsule, ledger=ledger,
                                   engine_name="WitnessEngine", bank=sk.bank,
                                   goal_text=self.GOAL)
        certs = _instantiate_skeleton_certificates(sk, space, engine_name="WitnessEngine",
                                                   request=req)
        self.assertTrue(certs)
        self.assertEqual(
            certs[0].projected_tactics,
            ["obtain ⟨w_ducky, hw_ducky⟩ := h", "refine ⟨w_ducky, ?_⟩",
             "simpa using hw_ducky"],
        )

    def test_plan_program_reaches_the_slate_end_to_end(self):
        conn, capsule = self._fixture()
        progs = build_ducky_programs(capsule, theorem_id="Site.thm", goal_text=self.GOAL,
                                     lean=None, conn=conn, accessible_theorem_id=1,
                                     max_programs=24)
        plans = [p for p in progs if len(p.tactics) >= 2 and "w_ducky" in " ".join(p.tactics)]
        self.assertTrue(plans, "the witness-transport plan must hold a slate slot")


class FenceGranularityTest(unittest.TestCase):
    """Loop pass-2's measured counterexample, pinned: a lane-family fence
    must NOT disable Dr. Ducky's internal programs. The fence's spine
    observed compile_exact over the IDF pool; her exact-headed local-fact
    programs are a different action class — negative evidence never
    transports across granularity (THE_REFOUNDING §10.8)."""

    def test_lane_exact_fence_does_not_mutilate_her_programs(self):
        from src.censor_store import censor_key, slice_signature
        from src.structural_reads import read_goal

        goal = "h : P q\n⊢ P q"
        reads = read_goal(goal)
        key = censor_key(reads.goal_bucket, slice_signature(reads))
        lean = _FakeLean(
            {
                (goal, "exact h"): TacticResult(
                    success=True, tactic="exact h", premises=[], new_goals=[],
                    error_message="",
                    feedback=LeanFeedback(stage="tactic_exec", category="ok",
                                          messages=[], raw_error=""),
                )
            }
        )
        state = _SearchState(open_goals=[goal], theorem_id="T")
        env = _SearchEnv(
            conn=None, lean=lean, anchor_id_map=None, max_candidates=8,
            censor_fences={key: {"exact"}},  # the lane-family fence
        )
        cfg = SearchConfig(dr_ducky_enabled=True, dr_ducky_max_programs=24)
        progress = _try_lane(
            "dr_ducky", goal, 0, nav_output=None, state=state, env=env,
            context=SearchContext(accessible_theorem_id=None), cfg=cfg,
        )
        self.assertTrue(progress)
        self.assertEqual(state.close_provenance[-1], "dr_ducky")
        self.assertEqual(state.open_goals, [])


class SlateDiversityTest(unittest.TestCase):
    """The skeleton frontier must survive the premise firehose.

    Measured on pass 1b: zero term-construction firings across 2,000
    theorems because the firehose's score lattice (0.80-1.03) crowded the
    skeleton priorities (~0.5) out of every top-24 slot. The reserve
    guarantees the designed source (paper §5.5) representation on the
    slate; both sources keep their internal order."""

    def _fed_network(self):
        from src.proof_network import init_db

        conn = init_db(":memory:")
        conn.execute("INSERT INTO entities (id, name) VALUES (1, 'Site.thm')")
        # 12 premises sharing goal tokens → nonzero token score → the
        # firehose generates its full per-lemma program family.
        for i in range(2, 14):
            conn.execute(
                "INSERT INTO entities (id, name) VALUES (?, ?)",
                (i, f"Foo.bar_lemma_{i}"),
            )
            conn.execute(
                "INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1, ?)",
                (i,),
            )
        conn.commit()
        return conn

    def test_skeleton_holed_programs_survive_a_fed_firehose(self):
        # atomic_prop target: the reserve guarantees the SOURCE a slate
        # share, not every family — on richer buckets (equality) higher-
        # extra skeletons legitimately outrank obtain within the reserve.
        conn = self._fed_network()
        goal = "h : ∃ n : Nat, Foo.bar n\nhk : Foo.bar q\n⊢ Foo.bar r"
        row = {
            "theorem_id": "Site.thm",
            "last_goal": goal,
            "last_goal_bucket": "atomic_prop",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 0,
            "goals_remaining": 1,
            "attempts": 0,
            "remaining_goals_snapshot": [goal],
        }
        progs = build_ducky_programs(
            build_goal_capsule(row),
            theorem_id="Site.thm",
            goal_text=goal,
            lean=None,
            conn=conn,
            accessible_theorem_id=1,
            max_programs=24,
        )
        self.assertEqual(len(progs), 24)
        # The firehose is genuinely fed (premise programs present)...
        self.assertTrue(any("Foo.bar_lemma" in p.tactics[0] for p in progs))
        # ...and the destructure skeleton still holds a slate slot.
        self.assertTrue(any(p.tactics[0].startswith("obtain ⟨") for p in progs))


if __name__ == "__main__":
    unittest.main()
