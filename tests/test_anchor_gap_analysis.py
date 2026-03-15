"""Tests for anchor gap analysis pure functions and DB queries."""

import unittest

from scripts.anchor_gap_analysis import (
    GapRecord,
    build_perfect_query,
    find_gap_anchors_from_conn,
    load_proof_steps,
    navigate_with_query,
)
from src.proof_network import init_db, recompute_idf


def _create_test_db():
    """Create an in-memory proof_network DB with test data using real schema."""
    conn = init_db(":memory:")

    # Insert entities: 2 lemmas used as "theorems" and 2 premise lemmas
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, namespace, file_path) "
        "VALUES (1, 'thm_A', 'lemma', 'Mathlib.Algebra', '')"
    )
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, namespace, file_path) "
        "VALUES (2, 'thm_B', 'lemma', 'Mathlib.Topology', '')"
    )
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, namespace, file_path) "
        "VALUES (3, 'lemma_X', 'lemma', 'Mathlib.Algebra', '')"
    )
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, namespace, file_path) "
        "VALUES (4, 'lemma_Y', 'lemma', 'Mathlib.Topology', '')"
    )

    # Positions for thm_A: structure=+1, domain=-1
    conn.execute("INSERT INTO entity_positions VALUES (1, 'structure', 1, 2)")
    conn.execute("INSERT INTO entity_positions VALUES (1, 'domain', -1, 1)")
    # Positions for thm_B: structure=-1
    conn.execute("INSERT INTO entity_positions VALUES (2, 'structure', -1, 3)")
    # Positions for lemma_X: structure=+1 (matches thm_A)
    conn.execute("INSERT INTO entity_positions VALUES (3, 'structure', 1, 1)")
    # Positions for lemma_Y: structure=-1 (matches thm_B)
    conn.execute("INSERT INTO entity_positions VALUES (4, 'structure', -1, 2)")

    # Anchors
    conn.execute("INSERT INTO anchors VALUES (1, 'ring', 'general')")
    conn.execute("INSERT INTO anchors VALUES (2, 'group', 'general')")
    conn.execute("INSERT INTO anchors VALUES (3, 'topology', 'general')")

    # Entity-anchor links
    conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (1, 1)")
    conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (1, 2)")
    conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (3, 1)")
    conn.execute("INSERT INTO entity_anchors (entity_id, anchor_id) VALUES (4, 3)")

    # Accessible premises: thm_A can use lemma_X, thm_B can use lemma_Y
    conn.execute("INSERT INTO accessible_premises VALUES (1, 3)")
    conn.execute("INSERT INTO accessible_premises VALUES (2, 4)")

    recompute_idf(conn)
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
        conn = _create_test_db()
        step = {
            "theorem_id": 1,
            "positions": {"structure": (1, 2), "domain": (-1, 1)},
            "anchors": ["ring", "group"],
        }
        query = build_perfect_query(conn, step)
        self.assertEqual(query.bank_directions, {"structure": 1, "domain": -1})
        self.assertEqual(query.bank_confidences, {"structure": 1.0, "domain": 1.0})
        # Should have resolved anchor IDs
        self.assertEqual(len(query.prefer_anchors), 2)
        conn.close()

    def test_no_accessible_theorem_id(self):
        """After P0 #4 fix, query should NOT have accessible_theorem_id."""
        conn = _create_test_db()
        step = {
            "theorem_id": 1,
            "positions": {"structure": (1, 2)},
            "anchors": ["ring"],
        }
        query = build_perfect_query(conn, step)
        self.assertIsNone(query.accessible_theorem_id)
        conn.close()

    def test_empty_positions(self):
        conn = _create_test_db()
        step = {"theorem_id": 1, "positions": {}, "anchors": []}
        query = build_perfect_query(conn, step)
        self.assertEqual(query.bank_directions, {})
        self.assertEqual(query.bank_confidences, {})
        conn.close()


class TestNavigateWithQuery(unittest.TestCase):
    def test_returns_only_lemmas(self):
        """navigate_with_query should filter to entity_type='lemma'."""
        conn = _create_test_db()
        # Add a tactic entity to verify it's excluded
        conn.execute(
            "INSERT INTO entities (name, entity_type, namespace, file_path) "
            "VALUES ('simp', 'tactic', '', '')"
        )
        conn.commit()

        step = {
            "theorem_id": 1,
            "positions": {"structure": (1, 2)},
            "anchors": ["ring"],
        }
        query = build_perfect_query(conn, step)
        results = navigate_with_query(conn, query, limit=16)
        self.assertNotIn("simp", results)
        conn.close()

    def test_returns_results(self):
        conn = _create_test_db()
        step = {
            "theorem_id": 1,
            "positions": {"structure": (1, 2)},
            "anchors": ["ring"],
        }
        query = build_perfect_query(conn, step)
        results = navigate_with_query(conn, query, limit=16)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        conn.close()

    def test_limit_respected(self):
        conn = _create_test_db()
        step = {
            "theorem_id": 1,
            "positions": {"structure": (1, 2)},
            "anchors": [],
        }
        query = build_perfect_query(conn, step)
        results = navigate_with_query(conn, query, limit=1)
        self.assertLessEqual(len(results), 1)
        conn.close()


class TestFindGapAnchorsFromConn(unittest.TestCase):
    def test_finds_missing_anchors(self):
        conn = _create_test_db()
        # lemma_Y has anchor "topology"; step_anchors has "ring"
        flat, by_cat = find_gap_anchors_from_conn(conn, "lemma_Y", ["ring"])
        self.assertEqual(flat, ["topology"])
        self.assertIn("general", by_cat)
        conn.close()

    def test_no_gap_when_anchors_overlap(self):
        conn = _create_test_db()
        # lemma_X has anchor "ring"; step_anchors also has "ring"
        flat, by_cat = find_gap_anchors_from_conn(conn, "lemma_X", ["ring"])
        self.assertEqual(flat, [])
        self.assertEqual(by_cat, {})
        conn.close()

    def test_nonexistent_premise_returns_empty(self):
        conn = _create_test_db()
        flat, by_cat = find_gap_anchors_from_conn(conn, "nonexistent", ["ring"])
        self.assertEqual(flat, [])
        self.assertEqual(by_cat, {})
        conn.close()


class TestLoadProofSteps(unittest.TestCase):
    def test_loads_from_file_db(self):
        """load_proof_steps takes a file path, so test with a temp file DB."""
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            conn = init_db(tmp_path)
            conn.execute(
                "INSERT INTO entities (name, entity_type, namespace, file_path) "
                "VALUES ('t1', 'lemma', 'ns', '')"
            )
            eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("INSERT INTO entity_positions VALUES (?, 'structure', 1, 1)", (eid,))
            conn.commit()
            conn.close()

            steps = load_proof_steps(tmp_path, sample_size=10)
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0]["name"], "t1")
            self.assertIn("positions", steps[0])
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
