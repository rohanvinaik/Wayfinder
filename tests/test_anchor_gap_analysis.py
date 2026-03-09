"""Tests for anchor gap analysis pure functions and DB queries."""

import sqlite3
import unittest

from scripts.anchor_gap_analysis import (
    GapRecord,
    build_perfect_query,
    find_gap_anchors,
    load_proof_steps,
    navigate_with_query,
)


def _create_test_db(path=":memory:"):
    """Create an in-memory proof_network DB with test data."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, type TEXT, namespace TEXT)"
    )
    conn.execute(
        "CREATE TABLE entity_positions (entity_id INTEGER, bank TEXT, sign INTEGER, depth INTEGER)"
    )
    conn.execute("CREATE TABLE anchors (id INTEGER PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE entity_anchors (entity_id INTEGER, anchor_id INTEGER)")
    conn.execute("CREATE TABLE accessible_premises (theorem_id INTEGER, premise_id INTEGER)")

    # Insert entities: 2 theorems, 2 premises
    conn.execute("INSERT INTO entities VALUES (1, 'thm_A', 'theorem', 'Mathlib.Algebra')")
    conn.execute("INSERT INTO entities VALUES (2, 'thm_B', 'theorem', 'Mathlib.Topology')")
    conn.execute("INSERT INTO entities VALUES (3, 'lemma_X', 'lemma', 'Mathlib.Algebra')")
    conn.execute("INSERT INTO entities VALUES (4, 'lemma_Y', 'lemma', 'Mathlib.Topology')")

    # Positions for thm_A: STRUCTURE=+1, DOMAIN=-1
    conn.execute("INSERT INTO entity_positions VALUES (1, 'STRUCTURE', 1, 2)")
    conn.execute("INSERT INTO entity_positions VALUES (1, 'DOMAIN', -1, 1)")
    # Positions for thm_B: STRUCTURE=-1
    conn.execute("INSERT INTO entity_positions VALUES (2, 'STRUCTURE', -1, 3)")
    # Positions for lemma_X: STRUCTURE=+1 (matches thm_A)
    conn.execute("INSERT INTO entity_positions VALUES (3, 'STRUCTURE', 1, 1)")
    # Positions for lemma_Y: STRUCTURE=-1 (matches thm_B)
    conn.execute("INSERT INTO entity_positions VALUES (4, 'STRUCTURE', -1, 2)")

    # Anchors
    conn.execute("INSERT INTO anchors VALUES (1, 'ring')")
    conn.execute("INSERT INTO anchors VALUES (2, 'group')")
    conn.execute("INSERT INTO anchors VALUES (3, 'topology')")

    # Entity-anchor links
    conn.execute("INSERT INTO entity_anchors VALUES (1, 1)")  # thm_A -> ring
    conn.execute("INSERT INTO entity_anchors VALUES (1, 2)")  # thm_A -> group
    conn.execute("INSERT INTO entity_anchors VALUES (3, 1)")  # lemma_X -> ring
    conn.execute("INSERT INTO entity_anchors VALUES (4, 3)")  # lemma_Y -> topology

    # Accessible premises: thm_A can use lemma_X, thm_B can use lemma_Y
    conn.execute("INSERT INTO accessible_premises VALUES (1, 3)")
    conn.execute("INSERT INTO accessible_premises VALUES (2, 4)")

    conn.commit()
    return conn


class TestGapRecordToDict(unittest.TestCase):
    def test_round_trip_fields(self):
        record = GapRecord(
            theorem_id="thm_A",
            goal_state="⊢ P → P",
            ground_truth_premises=["lemma_X"],
            retrieved_premises=["lemma_X", "lemma_Y"],
            recall_at_16=1.0,
            missed_premises=[],
            gap_anchors=[],
        )
        d = record.to_dict()
        self.assertEqual(d["theorem_id"], "thm_A")
        self.assertEqual(d["recall_at_16"], 1.0)
        self.assertEqual(d["ground_truth_premises"], ["lemma_X"])
        self.assertEqual(d["retrieved_premises"], ["lemma_X", "lemma_Y"])
        self.assertEqual(d["missed_premises"], [])
        self.assertEqual(d["gap_anchors"], [])

    def test_goal_state_truncated_at_200(self):
        long_goal = "x" * 500
        record = GapRecord(
            theorem_id="t",
            goal_state=long_goal,
            ground_truth_premises=[],
            retrieved_premises=[],
            recall_at_16=0.0,
            missed_premises=[],
            gap_anchors=[],
        )
        d = record.to_dict()
        self.assertEqual(len(d["goal_state"]), 200)

    def test_all_keys_present(self):
        record = GapRecord(
            theorem_id="t",
            goal_state="g",
            ground_truth_premises=[],
            retrieved_premises=[],
            recall_at_16=0.0,
            missed_premises=[],
            gap_anchors=[],
        )
        d = record.to_dict()
        expected_keys = {
            "theorem_id",
            "goal_state",
            "ground_truth_premises",
            "retrieved_premises",
            "recall_at_16",
            "missed_premises",
            "gap_anchors",
        }
        self.assertEqual(set(d.keys()), expected_keys)


class TestBuildPerfectQuery(unittest.TestCase):
    def test_basic_query(self):
        step = {
            "positions": {"STRUCTURE": (1, 2), "DOMAIN": (-1, 1)},
            "anchors": ["ring", "group"],
        }
        query = build_perfect_query(step)
        self.assertEqual(query["bank_directions"], {"STRUCTURE": 1, "DOMAIN": -1})
        self.assertEqual(query["bank_confidences"], {"STRUCTURE": 1.0, "DOMAIN": 1.0})
        self.assertEqual(query["anchors"], ["ring", "group"])

    def test_empty_positions(self):
        step = {"positions": {}, "anchors": []}
        query = build_perfect_query(step)
        self.assertEqual(query["bank_directions"], {})
        self.assertEqual(query["bank_confidences"], {})
        self.assertEqual(query["anchors"], [])

    def test_single_bank(self):
        step = {"positions": {"DEPTH": (0, 5)}, "anchors": ["topo"]}
        query = build_perfect_query(step)
        self.assertEqual(query["bank_directions"], {"DEPTH": 0})
        self.assertEqual(query["bank_confidences"], {"DEPTH": 1.0})


class _DBTestCase(unittest.TestCase):
    """Base class that creates a temporary on-disk SQLite DB for tests."""

    def setUp(self):
        import tempfile

        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self._tmpfile.name
        self._tmpfile.close()
        conn = _create_test_db(self.db_path)
        conn.close()

    def tearDown(self):
        import os

        os.unlink(self.db_path)


class TestLoadProofSteps(_DBTestCase):
    def test_loads_expected_count(self):
        steps = load_proof_steps(self.db_path, sample_size=10)
        # Only 2 theorems in fixture
        self.assertEqual(len(steps), 2)

    def test_step_has_required_keys(self):
        steps = load_proof_steps(self.db_path, sample_size=10)
        for step in steps:
            self.assertIn("theorem_id", step)
            self.assertIn("name", step)
            self.assertIn("premises", step)
            self.assertIn("positions", step)
            self.assertIn("anchors", step)

    def test_premises_populated(self):
        steps = load_proof_steps(self.db_path, sample_size=10)
        by_name = {s["name"]: s for s in steps}
        self.assertEqual(by_name["thm_A"]["premises"], ["lemma_X"])
        self.assertEqual(by_name["thm_B"]["premises"], ["lemma_Y"])

    def test_positions_populated(self):
        steps = load_proof_steps(self.db_path, sample_size=10)
        by_name = {s["name"]: s for s in steps}
        self.assertIn("STRUCTURE", by_name["thm_A"]["positions"])
        self.assertEqual(by_name["thm_A"]["positions"]["STRUCTURE"], (1, 2))

    def test_anchors_populated(self):
        steps = load_proof_steps(self.db_path, sample_size=10)
        by_name = {s["name"]: s for s in steps}
        self.assertIn("ring", by_name["thm_A"]["anchors"])
        self.assertIn("group", by_name["thm_A"]["anchors"])


class TestNavigateWithQuery(_DBTestCase):
    def test_matching_direction_ranked_first(self):
        query = {
            "bank_directions": {"STRUCTURE": 1},
            "anchors": [],
        }
        results = navigate_with_query(self.db_path, query, limit=4)
        # Entities with STRUCTURE=+1 (thm_A, lemma_X) should rank higher
        self.assertIn("thm_A", results[:2])
        self.assertIn("lemma_X", results[:2])

    def test_empty_directions_returns_empty(self):
        query = {"bank_directions": {}, "anchors": []}
        results = navigate_with_query(self.db_path, query, limit=4)
        self.assertEqual(results, [])

    def test_anchor_boost(self):
        # Query with STRUCTURE=-1 matches thm_B and lemma_Y equally on bank score.
        # Adding anchor "topology" should boost lemma_Y.
        query = {
            "bank_directions": {"STRUCTURE": -1},
            "anchors": ["topology"],
        }
        results = navigate_with_query(self.db_path, query, limit=4)
        # lemma_Y has anchor "topology", so it should appear
        self.assertIn("lemma_Y", results)

    def test_limit_respected(self):
        query = {
            "bank_directions": {"STRUCTURE": 1},
            "anchors": [],
        }
        results = navigate_with_query(self.db_path, query, limit=1)
        self.assertLessEqual(len(results), 1)


class TestFindGapAnchors(_DBTestCase):
    def test_finds_missing_anchors(self):
        # lemma_Y has anchor "topology"; step_anchors has "ring"
        gaps = find_gap_anchors(self.db_path, "lemma_Y", ["ring"])
        self.assertEqual(gaps, ["topology"])

    def test_no_gap_when_anchors_overlap(self):
        # lemma_X has anchor "ring"; step_anchors also has "ring"
        gaps = find_gap_anchors(self.db_path, "lemma_X", ["ring"])
        self.assertEqual(gaps, [])

    def test_nonexistent_premise_returns_empty(self):
        gaps = find_gap_anchors(self.db_path, "nonexistent", ["ring"])
        self.assertEqual(gaps, [])


if __name__ == "__main__":
    unittest.main()
