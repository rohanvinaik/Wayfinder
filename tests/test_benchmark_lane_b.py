"""Tests for benchmark_lane_b — _build_sorry_proof pure function."""

import unittest

from scripts.benchmark_lane_b import _build_sorry_proof


class TestBuildSorryProof(unittest.TestCase):
    """Exact-value tests for _build_sorry_proof (pure, SPEC006 target)."""

    def test_basic_proof_structure(self):
        entry = {"theorem_id": "Nat.add_comm", "tactics_used": ["simp", "ring"]}
        result = _build_sorry_proof(entry)
        self.assertIn("import Mathlib", result)
        self.assertIn("theorem Nat.add_comm := by", result)
        self.assertIn("simp", result)
        self.assertIn("ring", result)
        self.assertTrue(result.endswith("sorry\n"))

    def test_no_tactics_uses_sorry(self):
        entry = {"theorem_id": "trivial", "tactics_used": []}
        result = _build_sorry_proof(entry)
        self.assertIn("sorry", result)
        # Should have "sorry" as the tactic block AND the trailing sorry
        lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
        self.assertEqual(lines[-1], "sorry")

    def test_missing_theorem_id_defaults_to_unknown(self):
        entry = {"tactics_used": ["rfl"]}
        result = _build_sorry_proof(entry)
        self.assertIn("theorem unknown := by", result)

    def test_missing_tactics_key_uses_sorry(self):
        entry = {"theorem_id": "test_thm"}
        result = _build_sorry_proof(entry)
        self.assertIn("sorry", result)

    def test_single_tactic(self):
        entry = {"theorem_id": "my_thm", "tactics_used": ["omega"]}
        result = _build_sorry_proof(entry)
        self.assertIn("omega", result)
        self.assertTrue(result.strip().endswith("sorry"))

    def test_multiple_tactics_joined_with_newline(self):
        entry = {"theorem_id": "thm", "tactics_used": ["intro h", "exact h"]}
        result = _build_sorry_proof(entry)
        # Tactics should appear on separate lines
        self.assertIn("intro h\n", result)
        self.assertIn("exact h\n", result)

    def test_empty_entry(self):
        result = _build_sorry_proof({})
        self.assertIn("theorem unknown := by", result)
        self.assertIn("sorry", result)

    def test_return_type_is_string(self):
        result = _build_sorry_proof({"theorem_id": "t", "tactics_used": []})
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
