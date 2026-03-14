"""Tests for v3 OTP-grounded scoring."""

import sqlite3
import unittest

from src.nav_contracts import BANK_NAMES, NavOutput
from src.v3_scoring import (
    apply_bank_idf,
    build_constraint_report,
    compute_bank_idf,
    compute_otp_dimensionality,
    nav_output_to_candidates,
)


def _make_nav_output(
    directions: dict[str, int] | None = None,
    confidences: dict[str, float] | None = None,
) -> NavOutput:
    """Build a NavOutput for testing."""
    dirs = directions or {b: 0 for b in BANK_NAMES}
    confs = confidences or {b: 0.5 for b in BANK_NAMES}
    return NavOutput(
        directions=dirs,
        direction_confidences=confs,
        anchor_scores={},
        progress=0.5,
        critic_score=0.8,
    )


def _make_test_db() -> sqlite3.Connection:
    """Create an in-memory proof network DB for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, entity_id TEXT, kind TEXT)")
    conn.execute(
        "CREATE TABLE entity_positions "
        "(entity_id TEXT, bank TEXT, sign INTEGER, depth INTEGER, "
        "PRIMARY KEY (entity_id, bank))"
    )
    # Insert some entities with varying bank activations
    for i in range(100):
        eid = f"entity_{i}"
        conn.execute("INSERT INTO entities VALUES (?, ?, 'lemma')", (i, eid))
        # structure: sparse (only 10% nonzero)
        if i < 10:
            conn.execute("INSERT INTO entity_positions VALUES (?, 'structure', 1, 1)", (eid,))
        # domain: dense (80% nonzero)
        if i < 80:
            conn.execute("INSERT INTO entity_positions VALUES (?, 'domain', 1, 1)", (eid,))
        # automation: very sparse (5% nonzero)
        if i < 5:
            conn.execute("INSERT INTO entity_positions VALUES (?, 'automation', -1, 1)", (eid,))
    conn.commit()
    return conn


class TestComputeBankIdf(unittest.TestCase):
    def test_sparse_bank_has_higher_idf(self):
        conn = _make_test_db()
        idf = compute_bank_idf(conn)
        conn.close()
        # automation (5%) should have higher IDF than domain (80%)
        self.assertGreater(idf["automation"], idf["domain"])
        # structure (10%) should have higher IDF than domain (80%)
        self.assertGreater(idf["structure"], idf["domain"])

    def test_all_banks_present(self):
        conn = _make_test_db()
        idf = compute_bank_idf(conn)
        conn.close()
        for bank in BANK_NAMES:
            self.assertIn(bank, idf)
            self.assertGreater(idf[bank], 0)

    def test_empty_db(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE entity_positions "
            "(entity_id TEXT, bank TEXT, sign INTEGER, depth INTEGER, "
            "PRIMARY KEY (entity_id, bank))"
        )
        conn.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, entity_id TEXT)")
        conn.commit()
        idf = compute_bank_idf(conn)
        conn.close()
        for bank in BANK_NAMES:
            self.assertEqual(idf[bank], 1.0)


class TestApplyBankIdf(unittest.TestCase):
    def test_zero_direction_gets_zero_score(self):
        """Informational Zero: transparent banks contribute nothing."""
        nav = _make_nav_output(
            directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": -1,
                "context": 0,
                "decomposition": 0,
            },
            confidences={b: 0.9 for b in BANK_NAMES},
        )
        idf = {b: 2.0 for b in BANK_NAMES}
        scores = apply_bank_idf(nav, idf)

        self.assertEqual(scores["domain"], 0.0)
        self.assertEqual(scores["depth"], 0.0)
        self.assertEqual(scores["context"], 0.0)
        self.assertGreater(scores["structure"], 0.0)
        self.assertGreater(scores["automation"], 0.0)

    def test_idf_amplifies_score(self):
        nav = _make_nav_output(
            directions={
                "structure": 1,
                "domain": 1,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            confidences={
                "structure": 0.8,
                "domain": 0.8,
                "depth": 0.5,
                "automation": 0.5,
                "context": 0.5,
                "decomposition": 0.5,
            },
        )
        # structure has higher IDF
        idf = {b: 1.0 for b in BANK_NAMES}
        idf["structure"] = 3.0
        scores = apply_bank_idf(nav, idf)

        self.assertAlmostEqual(scores["structure"], 0.8 * 3.0)
        self.assertAlmostEqual(scores["domain"], 0.8 * 1.0)


class TestComputeOtpDimensionality(unittest.TestCase):
    def test_all_zero(self):
        self.assertEqual(compute_otp_dimensionality({b: 0 for b in BANK_NAMES}), 0)

    def test_all_active(self):
        dirs = {
            "structure": 1,
            "domain": -1,
            "depth": 1,
            "automation": -1,
            "context": 1,
            "decomposition": -1,
        }
        self.assertEqual(compute_otp_dimensionality(dirs), 6)

    def test_mixed(self):
        dirs = {
            "structure": 1,
            "domain": 0,
            "depth": -1,
            "automation": 0,
            "context": 0,
            "decomposition": 0,
        }
        self.assertEqual(compute_otp_dimensionality(dirs), 2)


class TestBuildConstraintReport(unittest.TestCase):
    def test_basic_composition(self):
        report = build_constraint_report(
            bank_scores={"structure": 1.5, "domain": 0.0, "automation": 2.0},
            critic_distance=3.0,
            censor_score=0.1,
            anchor_alignment=0.7,
        )
        # Verify all fields populated
        self.assertEqual(report.critic_distance, 3.0)
        self.assertEqual(report.censor_score, 0.1)
        self.assertEqual(report.anchor_alignment, 0.7)
        self.assertIsNone(report.energy)

    def test_high_censor_reduces_score(self):
        good = build_constraint_report(
            bank_scores={"s": 1.0},
            critic_distance=1.0,
            censor_score=0.0,
            anchor_alignment=0.5,
        )
        bad = build_constraint_report(
            bank_scores={"s": 1.0},
            critic_distance=1.0,
            censor_score=0.9,
            anchor_alignment=0.5,
        )
        self.assertGreater(good.total_score, bad.total_score)

    def test_custom_weights(self):
        report = build_constraint_report(
            bank_scores={"s": 1.0},
            critic_distance=0.0,
            censor_score=0.0,
            anchor_alignment=0.0,
            weights={"bank": 5.0, "critic": 0.0, "censor": 0.0, "anchor": 0.0},
        )
        self.assertAlmostEqual(report.total_score, 5.0)


class TestNavOutputToCandidates(unittest.TestCase):
    def test_produces_candidates(self):
        nav = _make_nav_output(
            directions={
                "structure": 1,
                "domain": 0,
                "depth": 0,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
        )
        idf = {b: 1.0 for b in BANK_NAMES}
        candidates = nav_output_to_candidates(
            nav,
            ["simp", "rw"],
            ["Nat.add_zero"],
            idf,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].tactic, "simp")
        self.assertEqual(candidates[0].premises, ["Nat.add_zero"])
        self.assertEqual(candidates[0].provenance, "navigate")
        self.assertIn("structure", candidates[0].navigational_scores)


if __name__ == "__main__":
    unittest.main()
