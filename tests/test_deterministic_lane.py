"""End-to-end pin of the deterministic path — the thesis artifact.

A full search step with pipeline=None and search_mode="deterministic":
goal-text tokens intersect the anchor dictionary, IDF weights rank the
accessible premises, the pure tactic grammar lowers them to Lean
candidates, and the kernel verifies. No encoder, no navigator, no
classifier anywhere on the path. If this test fails, the principled
layer is broken regardless of what the statistical layer does.
"""

import unittest
from unittest import mock

from src.proof_network import init_db
from src.proof_search import (
    SearchConfig,
    _search_step,
    _SearchEnv,
    _SearchState,
    _try_symbolic_family,
)
from src.resolution import SearchContext, build_symbolic_query, symbolic_rank_premises

GOAL = "⊢ Nat.add a b = Nat.add b a"


def _make_network():
    conn = init_db(":memory:")
    ents = [
        (1, "TestThm.goal_thm"),
        (2, "Nat.add_comm"),
        (3, "List.length_map"),
        (4, "Nat.mul_one"),
    ]
    for eid, name in ents:
        conn.execute("INSERT INTO entities (id, name) VALUES (?,?)", (eid, name))
    for aid, label in [(1, "Nat"), (2, "add"), (3, "List"), (4, "map"), (5, "mul")]:
        conn.execute("INSERT INTO anchors (id, label) VALUES (?,?)", (aid, label))
    for eid, aid in [(2, 1), (2, 2), (3, 3), (3, 4), (4, 1), (4, 5)]:
        conn.execute(
            "INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (?,?)", (eid, aid)
        )
    for aid, idf in [(1, 0.5), (2, 2.0), (3, 1.5), (4, 1.5), (5, 1.0)]:
        conn.execute("INSERT INTO anchor_idf (anchor_id, idf_value) VALUES (?,?)", (aid, idf))
    for pid in (2, 3, 4):
        conn.execute(
            "INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (1,?)", (pid,)
        )
    conn.commit()
    return conn


class _FakeLean:
    def __init__(self, succeed_on=None):
        self.submitted = []
        self._succeed_on = succeed_on or set()

    def try_tactic(self, goal, tactic):
        self.submitted.append(tactic)
        return mock.Mock(success=(tactic in self._succeed_on), new_goals=[])


def _make_env(conn, lean):
    return _SearchEnv(
        conn=conn,
        lean=lean,
        anchor_id_map={},
        max_candidates=8,
        id_to_name={2: "Nat.add_comm", 3: "List.length_map", 4: "Nat.mul_one"},
        name_to_id={},
    )


class SymbolicQueryTest(unittest.TestCase):
    def test_goal_tokens_intersect_anchor_dictionary(self):
        conn = _make_network()
        query = build_symbolic_query(conn, GOAL)
        # "Nat" (from Nat.add) and "add" match; List/map/mul do not.
        self.assertEqual(sorted(query.prefer_anchors), [1, 2])
        self.assertEqual(query.bank_directions, {})

    def test_no_matching_tokens_yields_empty_anchors(self):
        conn = _make_network()
        query = build_symbolic_query(conn, "⊢ ∀ ε > 0, ∃ δ > 0, True")
        self.assertEqual(query.prefer_anchors, [])


class SymbolicRankingTest(unittest.TestCase):
    def test_idf_weighted_intersection_ranks_shared_anchor_premise_first(self):
        conn = _make_network()
        ranked = symbolic_rank_premises(
            conn, GOAL, ["List.length_map", "Nat.mul_one", "Nat.add_comm"]
        )
        self.assertEqual(ranked[0], "Nat.add_comm")
        # Nat.mul_one shares only the low-IDF "Nat" anchor -> second.
        self.assertEqual(ranked[1], "Nat.mul_one")

    def test_no_anchor_match_preserves_pool_order(self):
        conn = _make_network()
        pool = ["List.length_map", "Nat.mul_one"]
        ranked = symbolic_rank_premises(conn, "⊢ True", pool)
        self.assertEqual(ranked, pool)

    def test_deterministic_across_calls(self):
        conn = _make_network()
        pool = ["Nat.mul_one", "Nat.add_comm", "List.length_map"]
        first = symbolic_rank_premises(conn, GOAL, pool)
        for _ in range(3):
            self.assertEqual(symbolic_rank_premises(conn, GOAL, pool), first)


class SymbolicLaneTest(unittest.TestCase):
    def test_exact_lane_closes_via_pure_grammar(self):
        conn = _make_network()
        lean = _FakeLean(succeed_on={"exact Nat.add_comm"})
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        closed = _try_symbolic_family(
            "exact", GOAL, 0, state, env, accessible_theorem_id=1, beam_width=2
        )
        self.assertTrue(closed)
        self.assertEqual(lean.submitted, ["exact Nat.add_comm"])

    def test_one_shot_cache_per_family_goal_pair(self):
        conn = _make_network()
        lean = _FakeLean()
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        _try_symbolic_family("exact", GOAL, 0, state, env, 1, beam_width=1)
        count = len(lean.submitted)
        self.assertFalse(_try_symbolic_family("exact", GOAL, 0, state, env, 1, beam_width=1))
        self.assertEqual(len(lean.submitted), count)


class DeterministicSearchStepTest(unittest.TestCase):
    def test_full_step_with_no_pipeline_closes_goal(self):
        conn = _make_network()
        lean = _FakeLean(succeed_on={"exact Nat.add_comm"})
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        cfg = SearchConfig(search_mode="deterministic", hammer_delegation=False)
        outcome = _search_step(None, state, env, SearchContext(accessible_theorem_id=1), cfg)
        self.assertTrue(outcome.progress)
        self.assertEqual(state.open_goals, [])
        self.assertEqual(state.close_provenance, ["symbolic_exact"])

    def test_deterministic_mode_never_schedules_statistical_lanes(self):
        conn = _make_network()
        lean = _FakeLean()
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        cfg = SearchConfig(search_mode="deterministic", hammer_delegation=False)
        _search_step(None, state, env, SearchContext(accessible_theorem_id=1), cfg)
        executed = state._last_lane_order_executed
        self.assertIn("symbolic_rw", executed)
        self.assertIn("symbolic_exact", executed)
        for lane in executed:
            self.assertFalse(lane.startswith("cosine"), lane)
            self.assertNotEqual(lane, "learned")


if __name__ == "__main__":
    unittest.main()
