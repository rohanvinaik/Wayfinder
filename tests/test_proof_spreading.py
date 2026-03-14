"""Mutation-prescribed tests for proof_spreading (SWAP + VALUE + BOUNDARY)."""

import unittest

from src.proof_network import clear_caches, init_db
from src.proof_spreading import _get_link_neighbors, spread


def _make_db():
    conn = init_db(":memory:")
    return conn


class TestGetLinkNeighbors(unittest.TestCase):
    """VALUE + SWAP for _get_link_neighbors."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def _insert(self, eid, name):
        self.conn.execute(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, 'lemma')",
            (eid, name),
        )

    def _link(self, src, tgt, weight):
        self.conn.execute(
            "INSERT INTO entity_links (source_id, target_id, relation, weight) "
            "VALUES (?, ?, 'uses', ?)",
            (src, tgt, weight),
        )

    def test_exact_values(self):
        self._insert(1, "A")
        self._insert(2, "B")
        self._link(1, 2, 0.7)
        self.conn.commit()
        result = _get_link_neighbors(self.conn, 1, 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 2)
        self.assertAlmostEqual(result[0][1], 0.7)

    def test_bidirectional(self):
        """Reverse direction also discovered."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._link(1, 2, 0.5)
        self.conn.commit()
        result = _get_link_neighbors(self.conn, 2, 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)

    def test_limit_respected(self):
        self._insert(1, "A")
        for i in range(2, 12):
            self._insert(i, f"N{i}")
            self._link(1, i, 0.1 * i)
        self.conn.commit()
        result = _get_link_neighbors(self.conn, 1, 3)
        self.assertEqual(len(result), 3)

    def test_sorted_by_weight_descending(self):
        self._insert(1, "A")
        self._insert(2, "B")
        self._insert(3, "C")
        self._link(1, 2, 0.3)
        self._link(1, 3, 0.9)
        self.conn.commit()
        result = _get_link_neighbors(self.conn, 1, 10)
        self.assertGreaterEqual(result[0][1], result[1][1])

    def test_swap_entity_id(self):
        """SWAP: neighbors of entity 1 differ from neighbors of entity 2."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._insert(3, "C")
        self._link(1, 3, 0.9)
        self.conn.commit()
        r1 = _get_link_neighbors(self.conn, 1, 10)
        r2 = _get_link_neighbors(self.conn, 2, 10)
        self.assertNotEqual(r1, r2)

    def test_no_links(self):
        self._insert(1, "A")
        self.conn.commit()
        result = _get_link_neighbors(self.conn, 1, 10)
        self.assertEqual(result, [])


class TestSpreadSwap(unittest.TestCase):
    """SWAP + BOUNDARY prescriptions for spread — supplements TestSpread."""

    def setUp(self):
        self.conn = _make_db()
        clear_caches()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def _insert(self, eid, name):
        self.conn.execute(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, 'lemma')",
            (eid, name),
        )

    def _link(self, src, tgt, weight):
        self.conn.execute(
            "INSERT INTO entity_links (source_id, target_id, relation, weight) "
            "VALUES (?, ?, 'uses', ?)",
            (src, tgt, weight),
        )

    def test_swap_decay_changes_activation(self):
        """SWAP: different decay values produce different activation scores."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._link(1, 2, 1.0)
        self.conn.commit()
        r1 = spread(self.conn, [1], max_depth=3, decay=0.8)
        r2 = spread(self.conn, [1], max_depth=3, decay=0.5)
        self.assertNotEqual(r1[2], r2[2])

    def test_swap_seeds_changes_result(self):
        """SWAP: different seed sets produce different activations."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._insert(3, "C")
        self._link(1, 3, 1.0)
        self._link(2, 3, 0.5)
        self.conn.commit()
        r1 = spread(self.conn, [1], max_depth=3, decay=1.0)
        r2 = spread(self.conn, [2], max_depth=3, decay=1.0)
        self.assertNotEqual(r1.get(3), r2.get(3))

    def test_swap_max_depth_changes_reach(self):
        """SWAP: max_depth=1 vs max_depth=2 gives different reachability."""
        for i in range(1, 4):
            self._insert(i, chr(64 + i))
        self._link(1, 2, 1.0)
        self._link(2, 3, 1.0)
        self.conn.commit()
        r1 = spread(self.conn, [1], max_depth=1, decay=0.8)
        r2 = spread(self.conn, [1], max_depth=2, decay=0.8)
        self.assertNotIn(3, r1)
        self.assertIn(3, r2)

    def test_boundary_decay_one(self):
        """BOUNDARY: decay=1.0 means no attenuation."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._link(1, 2, 1.0)
        self.conn.commit()
        result = spread(self.conn, [1], max_depth=3, decay=1.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_boundary_decay_near_zero(self):
        """BOUNDARY: very low decay should produce tiny activations."""
        self._insert(1, "A")
        self._insert(2, "B")
        self._link(1, 2, 1.0)
        self.conn.commit()
        result = spread(self.conn, [1], max_depth=3, decay=0.01)
        self.assertAlmostEqual(result[2], 0.01, places=4)


if __name__ == "__main__":
    unittest.main()
