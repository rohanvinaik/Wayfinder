"""Tests for proof lowering -- ProofExample to Lean tactic syntax."""

import unittest
from unittest.mock import patch

from src.contracts import ProofExample, Tier2Block, Tier3Slot
from src.lowering import lower_proof_to_lean, lower_to_theorem, roundtrip_validate


class TestLowerProofToLean(unittest.TestCase):
    def _make_example(self, tokens, blocks=None, slots=None):
        return ProofExample(
            theorem_id="test",
            goal_state="test goal",
            theorem_statement="theorem test : True",
            proof_text="",
            tier1_tokens=tokens,
            tier2_blocks=blocks or [],
            tier3_slots=slots or [],
        )

    def test_nullary_tactic(self):
        ex = self._make_example(["BOS", "omega", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertEqual(result.strip(), "omega")

    def test_intro_with_names(self):
        ex = self._make_example(
            ["BOS", "intro", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="intro", tokens=["h", "h2"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("intro h h2", result)

    def test_rw_with_lemmas(self):
        ex = self._make_example(
            ["BOS", "rw", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="rw", tokens=["Nat.add_comm"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("rw [Nat.add_comm]", result)

    def test_simp_with_lemmas(self):
        ex = self._make_example(
            ["BOS", "simp", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="simp", tokens=["List.map_id"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("simp [List.map_id]", result)

    def test_simp_bare(self):
        ex = self._make_example(["BOS", "simp", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertEqual(result.strip(), "simp")

    def test_have_with_type(self):
        ex = self._make_example(
            ["BOS", "have", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="have", tokens=["h"])],
            slots=[Tier3Slot(slot_id="have_1_type", value_kind="type", value="Nat")],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("have h : Nat := by", result)

    def test_cases_with_target(self):
        ex = self._make_example(
            ["BOS", "cases", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="cases", tokens=["n"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("cases n", result)

    def test_apply_with_premise(self):
        ex = self._make_example(
            ["BOS", "apply", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="apply", tokens=["Nat.succ_pos"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("apply Nat.succ_pos", result)

    def test_empty_proof_returns_sorry(self):
        ex = self._make_example(["BOS", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("sorry", result)

    def test_structural_tokens_filtered(self):
        ex = self._make_example(["BOS", "omega", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertNotIn("BOS", result)
        self.assertNotIn("EOS", result)

    def test_intro_bare(self):
        ex = self._make_example(["BOS", "intro", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertEqual(result.strip(), "intro")

    def test_cases_bare(self):
        ex = self._make_example(["BOS", "cases", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("cases _", result)

    def test_induction_with_target(self):
        ex = self._make_example(
            ["BOS", "induction", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="induction", tokens=["n"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("induction n", result)

    def test_induction_with_names(self):
        ex = self._make_example(
            ["BOS", "induction", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="induction", tokens=["n", "ih"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("induction n with ih", result)

    def test_induction_bare(self):
        ex = self._make_example(["BOS", "induction", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("induction _", result)

    def test_rw_bare(self):
        ex = self._make_example(["BOS", "rw", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("rw []", result)

    def test_have_without_type(self):
        ex = self._make_example(
            ["BOS", "have", "EOS"],
            blocks=[Tier2Block(tactic_index=1, tactic_name="have", tokens=["h"])],
        )
        result = lower_proof_to_lean(ex)
        self.assertIn("have h := by", result)

    def test_have_bare(self):
        ex = self._make_example(["BOS", "have", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("have h := by", result)

    def test_premise_tactic_bare(self):
        ex = self._make_example(["BOS", "apply", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("apply _", result)

    def test_unknown_tactic_passthrough(self):
        ex = self._make_example(["BOS", "custom_tactic", "EOS"])
        result = lower_proof_to_lean(ex)
        self.assertIn("custom_tactic", result)

    def test_multi_tactic_proof(self):
        ex = self._make_example(
            ["BOS", "intro", "exact", "EOS"],
            blocks=[
                Tier2Block(tactic_index=1, tactic_name="intro", tokens=["h"]),
                Tier2Block(tactic_index=2, tactic_name="exact", tokens=["h"]),
            ],
        )
        result = lower_proof_to_lean(ex)
        lines = [line.strip() for line in result.strip().splitlines()]
        self.assertEqual(lines[0], "intro h")
        self.assertEqual(lines[1], "exact h")


class TestLowerToTheorem(unittest.TestCase):
    def test_complete_theorem(self):
        ex = ProofExample(
            theorem_id="test",
            goal_state="",
            theorem_statement="theorem test : True",
            proof_text="",
            tier1_tokens=["BOS", "trivial", "EOS"],
            tier2_blocks=[],
            tier3_slots=[],
        )
        result = lower_to_theorem(ex)
        self.assertIn("theorem test : True := by", result)
        self.assertIn("trivial", result)


class TestRoundtripValidate(unittest.TestCase):
    def test_valid_proof(self):
        ex = ProofExample(
            theorem_id="test",
            goal_state="",
            theorem_statement="theorem test : True",
            proof_text="",
            tier1_tokens=["BOS", "trivial", "EOS"],
            tier2_blocks=[],
            tier3_slots=[],
        )
        ok, err = roundtrip_validate(ex)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_empty_proof_text_returns_false(self):
        ex = ProofExample(
            theorem_id="test",
            goal_state="",
            theorem_statement="theorem test : True",
            proof_text="",
            tier1_tokens=["BOS", "trivial", "EOS"],
            tier2_blocks=[],
            tier3_slots=[],
        )
        with patch("src.lowering.lower_proof_to_lean", return_value="   "):
            ok, err = roundtrip_validate(ex)
        self.assertFalse(ok)
        self.assertIn("Empty proof text", err)

    def test_lowering_error_returns_false(self):
        ex = ProofExample(
            theorem_id="test",
            goal_state="",
            theorem_statement="theorem test : True",
            proof_text="",
            tier1_tokens=["BOS", "trivial", "EOS"],
            tier2_blocks=[],
            tier3_slots=[],
        )
        # Corrupt tier1_tokens to be non-iterable to trigger exception
        ex.tier1_tokens = None
        ok, err = roundtrip_validate(ex)
        self.assertFalse(ok)
        self.assertIn("Lowering failed", err)


if __name__ == "__main__":
    unittest.main()
