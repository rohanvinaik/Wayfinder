"""Tests for build_nav_training_data — pure helper functions."""

import unittest

from scripts.build_nav_training_data import _encode_bank_positions, _resolve_step_directions


class TestEncodeBankPositions(unittest.TestCase):
    """Exact-value tests for _encode_bank_positions (pure, SPEC006 target)."""

    def test_basic_encoding(self):
        positions = {
            "structure": {"sign": 1, "depth": 3},
            "domain": {"sign": -1, "depth": 2},
        }
        result = _encode_bank_positions(positions)
        self.assertEqual(result["structure"], [1, 3])
        self.assertEqual(result["domain"], [-1, 2])

    def test_missing_sign_defaults_to_zero(self):
        positions = {"structure": {"depth": 5}}
        result = _encode_bank_positions(positions)
        self.assertEqual(result["structure"], [0, 5])

    def test_missing_depth_defaults_to_zero(self):
        positions = {"domain": {"sign": 1}}
        result = _encode_bank_positions(positions)
        self.assertEqual(result["domain"], [1, 0])

    def test_empty_info_defaults_both(self):
        positions = {"automation": {}}
        result = _encode_bank_positions(positions)
        self.assertEqual(result["automation"], [0, 0])

    def test_empty_positions_returns_empty(self):
        result = _encode_bank_positions({})
        self.assertEqual(result, {})

    def test_all_six_banks(self):
        banks = ["structure", "domain", "depth", "automation", "context", "decomposition"]
        positions = {b: {"sign": i % 3 - 1, "depth": i} for i, b in enumerate(banks)}
        result = _encode_bank_positions(positions)
        self.assertEqual(len(result), 6)
        for b in banks:
            self.assertIsInstance(result[b], list)
            self.assertEqual(len(result[b]), 2)

    def test_preserves_bank_names(self):
        positions = {"custom_bank": {"sign": 1, "depth": 1}}
        result = _encode_bank_positions(positions)
        self.assertIn("custom_bank", result)

    def test_output_format_is_list_not_tuple(self):
        positions = {"structure": {"sign": 1, "depth": 2}}
        result = _encode_bank_positions(positions)
        self.assertIsInstance(result["structure"], list)


class TestResolveStepDirections(unittest.TestCase):
    """Mutation-prescribed tests for _resolve_step_directions.

    This function generates the training signal for 321K examples.
    It must correctly use entity-level domain positions (namespace-derived)
    while keeping tactic-level labels for other banks.
    """

    def _make_entity(self, domain_sign=0, tactic_dirs=None):
        """Build a minimal entity dict for testing."""
        return {
            "positions": {
                "domain": {"sign": domain_sign, "depth": 1},
                "structure": {"sign": 0, "depth": 0},
            },
            "tactic_directions": tactic_dirs or [],
        }

    def test_domain_overridden_by_entity_position(self):
        """DOMAIN bank should come from entity positions, not tactic directions."""
        entity = self._make_entity(
            domain_sign=-1,
            tactic_dirs=[{"directions": {"structure": 1, "domain": 0, "automation": -1}}],
        )
        result = _resolve_step_directions(entity, step_idx=0, tactic="simp")
        # Entity says domain=-1, tactic says domain=0 → entity wins
        self.assertEqual(result["domain"], -1)

    def test_non_domain_banks_from_tactic(self):
        """Non-DOMAIN banks should still come from tactic directions."""
        entity = self._make_entity(
            domain_sign=1,
            tactic_dirs=[{"directions": {"structure": -1, "domain": 0, "automation": -1}}],
        )
        result = _resolve_step_directions(entity, step_idx=0, tactic="omega")
        self.assertEqual(result["structure"], -1)  # from tactic
        self.assertEqual(result["automation"], -1)  # from tactic
        self.assertEqual(result["domain"], 1)  # from entity

    def test_fallback_to_tactic_directions_when_no_stored_dirs(self):
        """When entity has no tactic_directions, fall back to TACTIC_DIRECTIONS."""
        entity = self._make_entity(domain_sign=-1, tactic_dirs=[])
        result = _resolve_step_directions(entity, step_idx=0, tactic="omega")
        # omega is in TACTIC_DIRECTIONS → should use those for non-domain banks
        # Domain still comes from entity
        self.assertEqual(result["domain"], -1)

    def test_domain_zero_when_entity_has_no_domain_position(self):
        """Missing domain position in entity → default to 0."""
        entity = {"positions": {}, "tactic_directions": []}
        result = _resolve_step_directions(entity, step_idx=0, tactic="rfl")
        self.assertEqual(result["domain"], 0)

    def test_exact_values_concrete_domain(self):
        """Entity in Nat namespace (domain=-1) + simp tactic."""
        entity = self._make_entity(
            domain_sign=-1,
            tactic_dirs=[{"directions": {"structure": -1, "domain": 0, "automation": -1,
                                         "depth": -1, "context": 0, "decomposition": -1}}],
        )
        result = _resolve_step_directions(entity, step_idx=0, tactic="simp")
        self.assertEqual(result, {
            "structure": -1, "domain": -1, "automation": -1,
            "depth": -1, "context": 0, "decomposition": -1,
        })

    def test_exact_values_abstract_domain(self):
        """Entity in CategoryTheory namespace (domain=+1) + apply tactic."""
        entity = self._make_entity(
            domain_sign=1,
            tactic_dirs=[{"directions": {"structure": 0, "domain": 0, "automation": 1,
                                         "depth": 0, "context": 0, "decomposition": 0}}],
        )
        result = _resolve_step_directions(entity, step_idx=0, tactic="apply")
        self.assertEqual(result["domain"], 1)
        self.assertEqual(result["automation"], 1)

    def test_step_index_selects_correct_tactic_directions(self):
        """step_idx=1 should use tactic_directions[1], not [0]."""
        entity = self._make_entity(
            domain_sign=0,
            tactic_dirs=[
                {"directions": {"structure": -1, "domain": 0}},
                {"directions": {"structure": 1, "domain": 0}},
            ],
        )
        result = _resolve_step_directions(entity, step_idx=1, tactic="apply")
        self.assertEqual(result["structure"], 1)  # from index 1, not 0

    def test_returns_dict_not_reference(self):
        """Result should be a new dict, not a reference to the stored one."""
        dirs = {"structure": 0, "domain": 0}
        entity = self._make_entity(tactic_dirs=[{"directions": dirs}])
        result = _resolve_step_directions(entity, step_idx=0, tactic="rfl")
        result["structure"] = 999
        # Original should be unmodified
        self.assertEqual(dirs["structure"], 0)


if __name__ == "__main__":
    unittest.main()
