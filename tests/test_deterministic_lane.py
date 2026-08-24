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

try:  # proof_search pulls torch — skip in torch-less envs, never poison collection
    from src.proof_network import init_db
    from src.proof_search import (
        SearchConfig,
        _search_step,
        _SearchEnv,
        _SearchState,
        _try_symbolic_family,
    )
    from src.resolution import SearchContext, build_symbolic_query, symbolic_rank_premises
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("src.proof_search requires torch") from None

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
    # TYPED labels, formed as extraction forms them (ns:/type:/name:) — the
    # fixture vocabulary must mirror the production dictionary; bare labels
    # here once let a query-side mismatch pass green while the live DB
    # matched zero anchors across a whole benchmark arm.
    for aid, label in [(1, "ns:Nat"), (2, "name:add"), (3, "ns:List"), (4, "name:map"), (5, "name:mul")]:
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
    def test_goal_labels_intersect_anchor_dictionary(self):
        conn = _make_network()
        query = build_symbolic_query(conn, GOAL)
        # ns:Nat and name:add match; List/map/mul labels do not.
        self.assertEqual(sorted(query.prefer_anchors), [1, 2])
        self.assertEqual(query.bank_directions, {})

    def test_query_labels_are_typed_like_extraction(self):
        from src.resolution import symbolic_query_labels

        labels = symbolic_query_labels(GOAL)
        self.assertIn("ns:Nat", labels)
        self.assertIn("type:Nat", labels)
        self.assertIn("name:add", labels)
        # Bare tokens never appear — they match nothing in the network.
        self.assertNotIn("Nat", labels)
        self.assertNotIn("add", labels)

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


class LibraryPoolTest(unittest.TestCase):
    """The site/library pool disposition: 'no site' is NOT 'no premises'.

    A siteless goal (external corpus — MiniF2F, a fresh conjecture) has no
    circularity hazard, so the pool is the library universe selected by
    anchor evidence; a goal with no anchor evidence gets NO pool — never an
    unranked dump of the library.
    """

    def test_library_pool_ranks_by_anchor_evidence(self):
        from src.resolution import library_premise_pool

        conn = _make_network()
        pool = library_premise_pool(conn, GOAL)
        self.assertEqual(pool[0], "Nat.add_comm")  # Nat+add anchors, highest IDF sum
        self.assertIn("Nat.mul_one", pool)  # shares only the Nat anchor
        self.assertNotIn("TestThm.goal_thm", pool)  # no anchors -> no evidence

    def test_no_anchor_evidence_means_no_pool(self):
        from src.resolution import library_premise_pool

        conn = _make_network()
        self.assertEqual(library_premise_pool(conn, "⊢ ∀ ε > 0, True"), [])

    def test_siteless_goal_closes_through_the_library_pool(self):
        conn = _make_network()
        lean = _FakeLean(succeed_on={"exact Nat.add_comm"})
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="External.no_site")
        closed = _try_symbolic_family(
            "exact", GOAL, 0, state, env, accessible_theorem_id=None, beam_width=2
        )
        self.assertTrue(closed)
        self.assertEqual(state._premise_pool_counts, {"library": 1})

    def test_site_pool_still_governs_when_a_site_exists(self):
        conn = _make_network()
        lean = _FakeLean()
        env = _make_env(conn, lean)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        _try_symbolic_family("exact", GOAL, 0, state, env, accessible_theorem_id=1, beam_width=1)
        self.assertEqual(state._premise_pool_counts, {"site": 1})

    def test_library_pool_fires_without_entity_maps(self):
        # THE round-3 regression: deterministic mode has no encoder, so the
        # entity-name maps are empty — the library pool queries names
        # directly and must not be gated on the map.
        conn = _make_network()
        lean = _FakeLean(succeed_on={"exact Nat.add_comm"})
        env = _SearchEnv(conn=conn, lean=lean, anchor_id_map={}, max_candidates=8)
        self.assertEqual(env.id_to_name, {})  # the exact runtime state
        state = _SearchState(open_goals=[GOAL], theorem_id="External.no_site")
        closed = _try_symbolic_family(
            "exact", GOAL, 0, state, env, accessible_theorem_id=None, beam_width=2
        )
        self.assertTrue(closed)
        self.assertEqual(state._premise_pool_counts, {"library": 1})

    def test_sited_goal_without_maps_abstains_never_substitutes_library(self):
        # A sited goal whose sound pool cannot be built must NOT fall back to
        # the library pool — that would break replay anti-circularity.
        conn = _make_network()
        lean = _FakeLean()
        env = _SearchEnv(conn=conn, lean=lean, anchor_id_map={}, max_candidates=8)
        state = _SearchState(open_goals=[GOAL], theorem_id="TestThm.goal_thm")
        closed = _try_symbolic_family(
            "exact", GOAL, 0, state, env, accessible_theorem_id=1, beam_width=1
        )
        self.assertFalse(closed)
        self.assertEqual(lean.submitted, [])
        self.assertEqual(state._premise_pool_counts, {"none": 1})

    def test_siteless_goal_without_evidence_abstains(self):
        conn = _make_network()
        lean = _FakeLean()
        env = _make_env(conn, lean)
        goal = "⊢ ∀ ε > 0, True"
        state = _SearchState(open_goals=[goal], theorem_id="External.no_site")
        closed = _try_symbolic_family(
            "exact", goal, 0, state, env, accessible_theorem_id=None, beam_width=1
        )
        self.assertFalse(closed)
        self.assertEqual(lean.submitted, [])  # abstained: no Lean call without evidence
        self.assertEqual(state._premise_pool_counts, {"none": 1})


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
