"""Tests for nav_contracts — dataclass serialization, defaults, roundtrips."""

import unittest

from src.nav_contracts import (
    BANK_NAMES,
    NavigationalExample,
    NavOutput,
    ScoredEntity,
    StructuredQuery,
    TacticResult,
)


class TestNavigationalExample(unittest.TestCase):
    def _make_example(self, **overrides):
        defaults = dict(
            goal_state="⊢ P → P",
            theorem_id="thm1",
            step_index=2,
            total_steps=5,
            nav_directions={"structure": 1, "domain": 0},
            anchor_labels=["Nat.add_comm"],
            ground_truth_tactic="intro",
            ground_truth_premises=["h"],
            remaining_steps=3,
        )
        defaults.update(overrides)
        return NavigationalExample(**defaults)

    def test_to_dict_required_fields(self):
        ex = self._make_example()
        d = ex.to_dict()
        self.assertEqual(d["goal_state"], "⊢ P → P")
        self.assertEqual(d["theorem_id"], "thm1")
        self.assertEqual(d["step_index"], 2)
        self.assertEqual(d["nav_directions"], {"structure": 1, "domain": 0})
        self.assertEqual(d["ground_truth_tactic"], "intro")

    def test_to_dict_excludes_bank_positions_when_none(self):
        ex = self._make_example(bank_positions=None)
        d = ex.to_dict()
        self.assertNotIn("bank_positions", d)

    def test_to_dict_includes_bank_positions_when_set(self):
        ex = self._make_example(bank_positions={"structure": [0, 1, 2]})
        d = ex.to_dict()
        self.assertEqual(d["bank_positions"], {"structure": [0, 1, 2]})

    def test_to_dict_excludes_metadata_when_empty(self):
        ex = self._make_example(metadata={})
        d = ex.to_dict()
        self.assertNotIn("metadata", d)

    def test_to_dict_includes_metadata_when_set(self):
        ex = self._make_example(metadata={"source": "test"})
        d = ex.to_dict()
        self.assertEqual(d["metadata"], {"source": "test"})

    def test_from_dict_roundtrip(self):
        ex = self._make_example(
            bank_positions={"depth": [1]},
            metadata={"k": "v"},
        )
        d = ex.to_dict()
        loaded = NavigationalExample.from_dict(d)
        self.assertEqual(loaded.goal_state, ex.goal_state)
        self.assertEqual(loaded.theorem_id, ex.theorem_id)
        self.assertEqual(loaded.step_index, ex.step_index)
        self.assertEqual(loaded.nav_directions, ex.nav_directions)
        self.assertEqual(loaded.bank_positions, ex.bank_positions)
        self.assertEqual(loaded.metadata, ex.metadata)

    def test_full_roundtrip_all_fields(self):
        """Every field survives a to_dict → from_dict cycle with exact values."""
        ex = self._make_example(
            goal_state="⊢ ∀ n, n + 0 = n",
            theorem_id="Nat.add_zero",
            step_index=1,
            total_steps=3,
            nav_directions={"structure": -1, "domain": 1, "automation": 0},
            anchor_labels=["Nat.add_zero", "Nat.succ"],
            ground_truth_tactic="induction",
            ground_truth_premises=["n", "ih"],
            remaining_steps=2,
            bank_positions={"structure": [0, 1, 2], "depth": [2]},
            proof_history=["intro n"],
            metadata={"source": "mathlib", "difficulty": 3},
        )
        loaded = NavigationalExample.from_dict(ex.to_dict())
        self.assertEqual(loaded.goal_state, "⊢ ∀ n, n + 0 = n")
        self.assertEqual(loaded.theorem_id, "Nat.add_zero")
        self.assertEqual(loaded.step_index, 1)
        self.assertEqual(loaded.total_steps, 3)
        self.assertEqual(loaded.nav_directions, {"structure": -1, "domain": 1, "automation": 0})
        self.assertEqual(loaded.anchor_labels, ["Nat.add_zero", "Nat.succ"])
        self.assertEqual(loaded.ground_truth_tactic, "induction")
        self.assertEqual(loaded.ground_truth_premises, ["n", "ih"])
        self.assertEqual(loaded.remaining_steps, 2)
        self.assertEqual(loaded.bank_positions, {"structure": [0, 1, 2], "depth": [2]})
        self.assertEqual(loaded.proof_history, ["intro n"])
        self.assertEqual(loaded.metadata, {"source": "mathlib", "difficulty": 3})

    def test_to_dict_key_completeness(self):
        """to_dict produces exactly the expected keys."""
        ex = self._make_example(
            bank_positions={"s": [1]},
            metadata={"k": "v"},
            proof_history=["intro"],
        )
        d = ex.to_dict()
        required_keys = {
            "goal_state",
            "theorem_id",
            "step_index",
            "total_steps",
            "nav_directions",
            "anchor_labels",
            "ground_truth_tactic",
            "ground_truth_premises",
            "remaining_steps",
            "solvable",
            "proof_history",
        }
        for key in required_keys:
            self.assertIn(key, d, f"Missing required key: {key}")

    def test_from_dict_defaults_for_optional_fields(self):
        minimal = {
            "goal_state": "g",
            "theorem_id": "t",
            "nav_directions": {"structure": 0},
        }
        loaded = NavigationalExample.from_dict(minimal)
        self.assertEqual(loaded.step_index, 0)
        self.assertEqual(loaded.total_steps, 1)
        self.assertEqual(loaded.anchor_labels, [])
        self.assertEqual(loaded.ground_truth_tactic, "")
        self.assertEqual(loaded.ground_truth_premises, [])
        self.assertEqual(loaded.remaining_steps, 0)
        self.assertEqual(loaded.solvable, True)
        self.assertEqual(loaded.proof_history, [])
        self.assertIsNone(loaded.bank_positions)
        self.assertEqual(loaded.metadata, {})

    def test_from_dict_preserves_solvable_false(self):
        d = {
            "goal_state": "g",
            "theorem_id": "t",
            "nav_directions": {},
            "solvable": False,
        }
        loaded = NavigationalExample.from_dict(d)
        self.assertFalse(loaded.solvable)
        # Verify solvable is exactly the bool False, not just falsy
        self.assertEqual(loaded.solvable, False)
        # Other fields should still use their defaults
        self.assertEqual(loaded.goal_state, "g")
        self.assertEqual(loaded.theorem_id, "t")
        self.assertEqual(loaded.nav_directions, {})

    def test_proof_history_preserved(self):
        ex = self._make_example(proof_history=["intro h", "exact h"])
        d = ex.to_dict()
        loaded = NavigationalExample.from_dict(d)
        self.assertEqual(loaded.proof_history, ["intro h", "exact h"])


class TestScoredEntityAndNavOutput(unittest.TestCase):
    def test_scored_entity_to_dict(self):
        se = ScoredEntity(
            entity_id=42,
            name="Nat.add_comm",
            final_score=0.95,
            bank_score=0.5,
            anchor_score=0.3,
            seed_score=0.15,
        )
        d = se.to_dict()
        self.assertEqual(d["entity_id"], 42)
        self.assertEqual(d["name"], "Nat.add_comm")
        self.assertAlmostEqual(d["final_score"], 0.95)
        self.assertAlmostEqual(d["bank_score"], 0.5)
        self.assertAlmostEqual(d["anchor_score"], 0.3)
        self.assertAlmostEqual(d["seed_score"], 0.15)

    def test_nav_output_to_dict(self):
        nav = NavOutput(
            directions={"structure": 1},
            direction_confidences={"structure": 0.9},
            anchor_scores={"anc": 0.7},
            progress=0.4,
            critic_score=0.8,
        )
        d = nav.to_dict()
        self.assertEqual(d["directions"], {"structure": 1})
        self.assertEqual(d["direction_confidences"], {"structure": 0.9})
        self.assertEqual(d["anchor_scores"], {"anc": 0.7})
        self.assertAlmostEqual(d["progress"], 0.4)
        self.assertAlmostEqual(d["critic_score"], 0.8)

    def test_tactic_result_to_dict(self):
        tr = TacticResult(
            success=True,
            tactic="rfl",
            premises=["h"],
            new_goals=["g2"],
            error_message="",
        )
        d = tr.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["tactic"], "rfl")
        self.assertEqual(d["premises"], ["h"])
        self.assertEqual(d["new_goals"], ["g2"])
        self.assertEqual(d["error_message"], "")

    def test_tactic_result_defaults(self):
        tr = TacticResult(success=False, tactic="omega", premises=[])
        self.assertEqual(tr.new_goals, [])
        self.assertEqual(tr.error_message, "")

    def test_scored_entity_to_dict_completeness(self):
        """to_dict contains exactly 6 keys with exact values."""
        se = ScoredEntity(
            entity_id=7,
            name="rfl",
            final_score=0.85,
            bank_score=0.4,
            anchor_score=0.25,
            seed_score=0.2,
        )
        d = se.to_dict()
        self.assertEqual(len(d), 6)
        self.assertEqual(d["entity_id"], 7)
        self.assertEqual(d["name"], "rfl")
        self.assertAlmostEqual(d["final_score"], 0.85, places=10)
        self.assertAlmostEqual(d["bank_score"], 0.4, places=10)
        self.assertAlmostEqual(d["anchor_score"], 0.25, places=10)
        self.assertAlmostEqual(d["seed_score"], 0.2, places=10)

    def test_nav_output_to_dict_completeness(self):
        """to_dict contains exactly 5 keys."""
        nav = NavOutput(
            directions={"structure": 1, "domain": -1},
            direction_confidences={"structure": 0.95, "domain": 0.6},
            anchor_scores={"simp": 0.8, "rfl": 0.3},
            progress=0.65,
            critic_score=0.72,
        )
        d = nav.to_dict()
        self.assertEqual(len(d), 5)
        self.assertEqual(d["directions"], {"structure": 1, "domain": -1})
        self.assertEqual(d["direction_confidences"], {"structure": 0.95, "domain": 0.6})
        self.assertEqual(d["anchor_scores"], {"simp": 0.8, "rfl": 0.3})
        self.assertAlmostEqual(d["progress"], 0.65, places=10)
        self.assertAlmostEqual(d["critic_score"], 0.72, places=10)

    def test_tactic_result_to_dict_completeness(self):
        """to_dict contains exactly 5 keys with exact values."""
        tr = TacticResult(
            success=False,
            tactic="apply h",
            premises=["h", "h2"],
            new_goals=["sub1"],
            error_message="unknown identifier",
        )
        d = tr.to_dict()
        self.assertEqual(len(d), 5)
        self.assertFalse(d["success"])
        self.assertEqual(d["tactic"], "apply h")
        self.assertEqual(d["premises"], ["h", "h2"])
        self.assertEqual(d["new_goals"], ["sub1"])
        self.assertEqual(d["error_message"], "unknown identifier")


class TestStructuredQuery(unittest.TestCase):
    def test_defaults(self):
        q = StructuredQuery(
            bank_directions={"structure": 1},
            bank_confidences={"structure": 0.9},
        )
        self.assertEqual(q.require_anchors, [])
        self.assertEqual(q.prefer_anchors, [])
        self.assertEqual(q.prefer_weights, [])
        self.assertEqual(q.avoid_anchors, [])
        self.assertEqual(q.seed_entity_ids, [])
        self.assertIsNone(q.accessible_theorem_id)

    def test_field_isolation(self):
        q1 = StructuredQuery(bank_directions={}, bank_confidences={})
        q2 = StructuredQuery(bank_directions={}, bank_confidences={})
        q1.prefer_anchors.append(1)
        self.assertEqual(q2.prefer_anchors, [])


class TestBankNames(unittest.TestCase):
    def test_bank_names_count(self):
        self.assertEqual(len(BANK_NAMES), 6)

    def test_bank_names_contents(self):
        expected = ["structure", "domain", "depth", "automation", "context", "decomposition"]
        self.assertEqual(BANK_NAMES, expected)

    def test_bank_names_are_strings(self):
        for name in BANK_NAMES:
            self.assertIsInstance(name, str)
        # Verify each bank name is a specific known value
        expected = {"structure", "domain", "depth", "automation", "context", "decomposition"}
        self.assertEqual(set(BANK_NAMES), expected)
        # Bank names should be lowercase identifiers (no spaces, no uppercase)
        for name in BANK_NAMES:
            self.assertEqual(name, name.lower())
            self.assertNotIn(" ", name)


if __name__ == "__main__":
    unittest.main()
