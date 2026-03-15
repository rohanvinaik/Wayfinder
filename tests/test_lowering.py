"""Tests for proof lowering -- ProofExample to Lean tactic syntax."""

import unittest
from unittest.mock import patch

from src.contracts import ProofExample, Tier2Block, Tier3Slot
from src.lowering import (
    _lower_cases,
    _lower_have,
    _lower_induction,
    _lower_intro,
    _lower_rw,
    _lower_simp,
    _lower_tactic,
    lower_proof_to_lean,
    lower_to_theorem,
    roundtrip_validate,
)


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
        self.assertEqual(ok, True)
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


class TestLowerInnerFunctionsValue(unittest.TestCase):
    """Mutation-prescribed VALUE tests for inner _lower_* functions."""

    def _block(self, tokens):
        return Tier2Block(tactic_index=0, tactic_name="", tokens=tokens)

    def test_lower_intro_with_names_exact(self):
        self.assertEqual(_lower_intro(self._block(["h", "h2"]), {}, 0), "intro h h2")

    def test_lower_intro_bare_exact(self):
        self.assertEqual(_lower_intro(None, {}, 0), "intro")

    def test_lower_cases_with_target_exact(self):
        self.assertEqual(_lower_cases(self._block(["n"]), {}, 0), "cases n")

    def test_lower_cases_bare_exact(self):
        self.assertEqual(_lower_cases(None, {}, 0), "cases _")

    def test_lower_induction_single_exact(self):
        self.assertEqual(_lower_induction(self._block(["n"]), {}, 0), "induction n")

    def test_lower_induction_with_names_exact(self):
        self.assertEqual(
            _lower_induction(self._block(["n", "ih"]), {}, 0), "induction n with ih"
        )

    def test_lower_induction_bare_exact(self):
        self.assertEqual(_lower_induction(None, {}, 0), "induction _")

    def test_lower_rw_with_lemma_exact(self):
        self.assertEqual(_lower_rw(self._block(["add_comm"]), {}, 0), "rw [add_comm]")

    def test_lower_rw_multi_exact(self):
        self.assertEqual(
            _lower_rw(self._block(["add_comm", "mul_one"]), {}, 0), "rw [add_comm, mul_one]"
        )

    def test_lower_rw_bare_exact(self):
        self.assertEqual(_lower_rw(None, {}, 0), "rw []")

    def test_lower_simp_with_lemma_exact(self):
        self.assertEqual(_lower_simp(self._block(["map_id"]), {}, 0), "simp [map_id]")

    def test_lower_simp_bare_exact(self):
        self.assertEqual(_lower_simp(None, {}, 0), "simp")

    def test_lower_have_with_type_exact(self):
        term_map = {"have_5_type": "Nat"}
        self.assertEqual(
            _lower_have(self._block(["h"]), term_map, 5), "have h : Nat := by"
        )

    def test_lower_have_without_type_exact(self):
        self.assertEqual(_lower_have(self._block(["h"]), {}, 0), "have h := by")

    def test_lower_have_bare_exact(self):
        self.assertEqual(_lower_have(None, {}, 0), "have h := by")


class TestLowerTacticValue(unittest.TestCase):
    """Mutation-prescribed VALUE + SWAP tests for _lower_tactic."""

    def test_structural_returns_none(self):
        self.assertIsNone(_lower_tactic("BOS", 0, {}, {}))
        self.assertIsNone(_lower_tactic("EOS", 0, {}, {}))
        self.assertIsNone(_lower_tactic("PAD", 0, {}, {}))

    def test_nullary_returns_tactic_name(self):
        self.assertEqual(_lower_tactic("ring", 0, {}, {}), "ring")
        self.assertEqual(_lower_tactic("omega", 0, {}, {}), "omega")

    def test_premise_tactic_with_block(self):
        block = Tier2Block(tactic_index=1, tactic_name="apply", tokens=["Nat.succ_pos"])
        result = _lower_tactic("apply", 1, {1: block}, {})
        self.assertEqual(result, "apply Nat.succ_pos")

    def test_premise_tactic_bare(self):
        result = _lower_tactic("apply", 0, {}, {})
        self.assertEqual(result, "apply _")

    def test_unknown_tactic_passthrough(self):
        result = _lower_tactic("my_custom", 0, {}, {})
        self.assertEqual(result, "my_custom")

    def test_swap_handler_vs_premise(self):
        """SWAP: 'intro' uses handler, 'exact' uses premise path — different output."""
        block = Tier2Block(tactic_index=0, tactic_name="", tokens=["h"])
        intro_result = _lower_tactic("intro", 0, {0: block}, {})
        exact_result = _lower_tactic("exact", 0, {0: block}, {})
        self.assertEqual(intro_result, "intro h")
        self.assertEqual(exact_result, "exact h")
        self.assertNotEqual(intro_result, exact_result)


class TestLowerInductionBoundary(unittest.TestCase):
    """BOUNDARY: _lower_induction boundary at len(tokens) > 1."""

    def _block(self, tokens):
        return Tier2Block(tactic_index=0, tactic_name="", tokens=tokens)

    def test_boundary_one_token_no_with(self):
        result = _lower_induction(self._block(["n"]), {}, 0)
        self.assertNotIn("with", result)

    def test_boundary_two_tokens_has_with(self):
        result = _lower_induction(self._block(["n", "ih"]), {}, 0)
        self.assertIn("with", result)


if __name__ == "__main__":
    unittest.main()
