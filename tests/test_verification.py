"""Tests for the proof verification module."""

import unittest

from src.verification import ProofVerifier, VerificationConfig, check_proof_structural


class TestStubVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = ProofVerifier(VerificationConfig(backend="stub"))

    def test_empty_proof_fails(self):
        result = self.verifier.verify("theorem t : True", "")
        self.assertFalse(result.verified)
        self.assertIn("Empty", result.error_message)

    def test_sorry_proof_fails(self):
        result = self.verifier.verify("theorem t : True", "sorry")
        self.assertFalse(result.verified)
        self.assertIn("sorry", result.error_message)

    def test_valid_proof_passes(self):
        result = self.verifier.verify("theorem t : True", "trivial")
        self.assertTrue(result.verified)
        self.assertEqual(result.steps_used, 1)

    def test_multi_tactic_proof(self):
        result = self.verifier.verify("theorem t : P -> P", "intro h\nexact h")
        self.assertTrue(result.verified)
        self.assertEqual(result.steps_used, 2)

    def test_tactic_verify_requires_pantograph(self):
        result = self.verifier.verify_tactic("|- True", "trivial")
        self.assertFalse(result.verified)
        self.assertIn("pantograph", result.error_message)


class TestCheckProofStructural(unittest.TestCase):
    def test_detects_sorry(self):
        result = check_proof_structural("intro h\nsorry")
        self.assertTrue(result["has_sorry"])
        self.assertEqual(result["tactic_count"], 2)

    def test_detects_automation(self):
        result = check_proof_structural("simp\nomega")
        self.assertTrue(result["uses_automation"])

    def test_empty_proof(self):
        result = check_proof_structural("")
        self.assertEqual(result["tactic_count"], 0)
        self.assertFalse(result["has_sorry"])


if __name__ == "__main__":
    unittest.main()
