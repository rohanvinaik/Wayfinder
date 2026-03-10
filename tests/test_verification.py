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
        self.assertEqual(result["tactics"], ["intro h", "sorry"])

    def test_no_sorry(self):
        result = check_proof_structural("intro h\nexact h")
        self.assertFalse(result["has_sorry"])
        self.assertEqual(result["tactic_count"], 2)
        self.assertEqual(result["tactics"], ["intro h", "exact h"])

    def test_detects_each_automation_tactic(self):
        for tactic in ["simp", "omega", "linarith", "norm_num", "decide", "aesop", "tauto"]:
            result = check_proof_structural(tactic)
            self.assertTrue(
                result["uses_automation"],
                f"{tactic} should be detected as automation",
            )
            self.assertEqual(result["tactic_count"], 1)

    def test_non_automation_tactic(self):
        result = check_proof_structural("intro h\nexact h\napply foo")
        self.assertFalse(result["uses_automation"])
        self.assertEqual(result["tactic_count"], 3)
        self.assertEqual(result["tactics"], ["intro h", "exact h", "apply foo"])

    def test_empty_proof(self):
        result = check_proof_structural("")
        self.assertEqual(result["tactic_count"], 0)
        self.assertFalse(result["has_sorry"])
        self.assertFalse(result["uses_automation"])
        self.assertEqual(result["tactics"], [])

    def test_whitespace_only(self):
        result = check_proof_structural("   \n  \n  ")
        self.assertEqual(result["tactic_count"], 0)
        self.assertEqual(result["tactics"], [])

    def test_mixed_automation_and_manual(self):
        result = check_proof_structural("intro h\nsimp\nexact h")
        self.assertTrue(result["uses_automation"])
        self.assertEqual(result["tactic_count"], 3)
        self.assertFalse(result["has_sorry"])

    def test_sorry_not_substring(self):
        # "sorry_lemma" should NOT trigger has_sorry (exact match on "sorry")
        result = check_proof_structural("sorry_lemma")
        self.assertFalse(result["has_sorry"])
        self.assertEqual(result["tactic_count"], 1)
        self.assertEqual(result["tactics"], ["sorry_lemma"])


if __name__ == "__main__":
    unittest.main()
