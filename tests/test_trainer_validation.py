"""Tests for trainer_validation — proof lowering + structural checking."""

import unittest

from src.contracts import ProofExample
from src.trainer_validation import get_default_verifier, validate_single_proof


def _make_example(tokens=None, statement="theorem t : True"):
    return ProofExample(
        theorem_id="test",
        goal_state="⊢ True",
        theorem_statement=statement,
        proof_text="",
        tier1_tokens=tokens or ["BOS", "trivial", "EOS"],
        tier2_blocks=[],
        tier3_slots=[],
    )


class TestValidateSingleProof(unittest.TestCase):
    """VALUE + SWAP for validate_single_proof."""

    def test_valid_proof_lowers(self):
        result = validate_single_proof(_make_example())
        self.assertTrue(result.lowered)
        self.assertTrue(result.structurally_valid)
        self.assertTrue(result.no_sorry)
        self.assertGreater(result.tactic_count, 0)

    def test_sorry_proof_not_valid(self):
        result = validate_single_proof(_make_example(["BOS", "sorry", "EOS"]))
        self.assertTrue(result.lowered)
        self.assertFalse(result.structurally_valid)
        self.assertFalse(result.no_sorry)

    def test_empty_proof_fails_lowering(self):
        result = validate_single_proof(_make_example(["BOS", "EOS"]))
        # Empty proof → lower_proof_to_lean returns "  sorry" → has_sorry=True
        self.assertTrue(result.lowered)
        self.assertFalse(result.no_sorry)

    def test_with_stub_verifier(self):
        verifier = get_default_verifier()
        result = validate_single_proof(_make_example(), verifier=verifier)
        self.assertTrue(result.lowered)
        # Stub verifier does structural checks — valid proof passes
        self.assertTrue(result.verified)

    def test_tactic_count_exact(self):
        result = validate_single_proof(_make_example(["BOS", "ring", "omega", "EOS"]))
        self.assertEqual(result.tactic_count, 2)

    def test_swap_verifier_none_vs_stub(self):
        """SWAP: without verifier → verified=False, with stub → verified=True."""
        r_none = validate_single_proof(_make_example())
        r_stub = validate_single_proof(_make_example(), verifier=get_default_verifier())
        self.assertEqual(r_none.lowered, r_stub.lowered)
        self.assertFalse(r_none.verified)
        self.assertTrue(r_stub.verified)


class TestGetDefaultVerifier(unittest.TestCase):
    def test_returns_verifier(self):
        v = get_default_verifier()
        self.assertIsNotNone(v)

    def test_stub_backend(self):
        v = get_default_verifier()
        self.assertEqual(v.config.backend, "stub")


if __name__ == "__main__":
    unittest.main()
