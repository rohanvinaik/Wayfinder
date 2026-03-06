"""Tests for data contracts — roundtrip serialization."""

import unittest

from src.contracts import (
    CoverageReport,
    NegativeBankEntry,
    OODPrompt,
    ProofExample,
    Tier2Block,
    Tier3Slot,
    VerificationResult,
)


class TestProofExample(unittest.TestCase):
    def _make_example(self) -> ProofExample:
        return ProofExample(
            theorem_id="test.add_comm",
            goal_state="a b : Nat |- a + b = b + a",
            theorem_statement="theorem add_comm (a b : Nat) : a + b = b + a",
            proof_text="  omega",
            tier1_tokens=["BOS", "omega", "EOS"],
            tier2_blocks=[],
            tier3_slots=[],
            metadata={"domain": "algebra"},
        )

    def test_roundtrip(self):
        ex = self._make_example()
        d = ex.to_dict()
        ex2 = ProofExample.from_dict(d)
        self.assertEqual(ex.theorem_id, ex2.theorem_id)
        self.assertEqual(ex.tier1_tokens, ex2.tier1_tokens)
        self.assertEqual(ex.goal_state, ex2.goal_state)

    def test_with_tier2_blocks(self):
        ex = self._make_example()
        ex.tier2_blocks = [
            Tier2Block(tactic_index=1, tactic_name="apply", tokens=["Nat.add_comm"])
        ]
        d = ex.to_dict()
        ex2 = ProofExample.from_dict(d)
        self.assertEqual(len(ex2.tier2_blocks), 1)
        self.assertEqual(ex2.tier2_blocks[0].tokens, ["Nat.add_comm"])

    def test_with_tier3_slots(self):
        ex = self._make_example()
        ex.tier3_slots = [
            Tier3Slot(slot_id="have_1_type", value_kind="type", value="Nat")
        ]
        d = ex.to_dict()
        ex2 = ProofExample.from_dict(d)
        self.assertEqual(len(ex2.tier3_slots), 1)
        self.assertEqual(ex2.tier3_slots[0].value, "Nat")


class TestVerificationResult(unittest.TestCase):
    def test_roundtrip(self):
        vr = VerificationResult(
            verified=True,
            goal_state="n : Nat |- n + 0 = n",
            tactic_trace=["omega"],
            steps_used=1,
        )
        d = vr.to_dict()
        vr2 = VerificationResult.from_dict(d)
        self.assertTrue(vr2.verified)
        self.assertEqual(vr2.tactic_trace, ["omega"])


class TestNegativeBankEntry(unittest.TestCase):
    def test_roundtrip(self):
        pos = ProofExample(
            theorem_id="t1", goal_state="g", theorem_statement="s",
            proof_text="p", tier1_tokens=["BOS", "simp", "EOS"],
            tier2_blocks=[], tier3_slots=[],
        )
        entry = NegativeBankEntry(
            goal_state="g", theorem_id="t1",
            positive=pos, negative=None,
            error_tags=["wrong_tactic"], source="synthetic_mutation",
        )
        d = entry.to_dict()
        entry2 = NegativeBankEntry.from_dict(d)
        self.assertEqual(entry2.error_tags, ["wrong_tactic"])
        self.assertIsNone(entry2.negative)


class TestOODPrompt(unittest.TestCase):
    def test_roundtrip(self):
        p = OODPrompt(prompt="Write a poem", label="ood", category="general_chat", source="synthetic")
        d = p.to_dict()
        p2 = OODPrompt.from_dict(d)
        self.assertEqual(p2.label, "ood")


class TestCoverageReport(unittest.TestCase):
    def test_roundtrip(self):
        cr = CoverageReport(
            scope="tier1", dataset="eval",
            total_tokens_in_eval=50, covered=48,
            uncovered=["tactic_a", "tactic_b"], coverage_pct=96.0,
        )
        d = cr.to_dict()
        cr2 = CoverageReport.from_dict(d)
        self.assertEqual(cr2.coverage_pct, 96.0)
        self.assertEqual(cr2.uncovered, ["tactic_a", "tactic_b"])


if __name__ == "__main__":
    unittest.main()
