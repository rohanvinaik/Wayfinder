"""Tests for v3 runtime data contracts."""

import unittest

from src.v3_contracts import (
    ActionCandidate,
    ConstraintReport,
    GoalContext,
    NegativeExample,
    SearchTrace,
    SketchProposal,
)


class TestGoalContext(unittest.TestCase):
    def test_defaults(self):
        gc = GoalContext(theorem_id="t1", goal_text="⊢ P")
        self.assertEqual(gc.theorem_id, "t1")
        self.assertEqual(gc.goal_text, "⊢ P")
        self.assertEqual(gc.proof_history, [])
        self.assertEqual(gc.accessible_premises, [])
        self.assertEqual(gc.source_split, "")

    def test_with_history(self):
        gc = GoalContext(
            theorem_id="t1",
            goal_text="⊢ P",
            proof_history=["intro n", "induction n"],
            source_split="eval",
        )
        self.assertEqual(len(gc.proof_history), 2)
        self.assertEqual(gc.source_split, "eval")


class TestActionCandidate(unittest.TestCase):
    def test_to_dict(self):
        ac = ActionCandidate(
            tactic="simp",
            premises=["Nat.add_zero"],
            provenance="navigate",
            navigational_scores={"structure": 0.8},
        )
        d = ac.to_dict()
        self.assertEqual(d["tactic"], "simp")
        self.assertEqual(d["premises"], ["Nat.add_zero"])
        self.assertNotIn("template_provenance", d)
        self.assertNotIn("censor_score", d)

    def test_optional_fields_in_dict(self):
        ac = ActionCandidate(
            tactic="apply",
            template_provenance="REWRITE_CHAIN",
            censor_score=0.3,
        )
        d = ac.to_dict()
        self.assertEqual(d["template_provenance"], "REWRITE_CHAIN")
        self.assertEqual(d["censor_score"], 0.3)


class TestNegativeExample(unittest.TestCase):
    def test_round_trip(self):
        neg = NegativeExample(
            goal_state="⊢ ∀ n, n + 0 = n",
            theorem_id="Nat.add_zero",
            step_index=2,
            failed_tactic="simp",
            failure_reason="no_rewrite_rules_match",
            failure_category="semantic",
            source="sorry_hole",
            bank_directions={
                "structure": 1,
                "domain": 0,
                "depth": -1,
                "automation": 0,
                "context": 0,
                "decomposition": 0,
            },
            otp_dimensionality=2,
        )
        d = neg.to_dict()
        restored = NegativeExample.from_dict(d)
        self.assertEqual(restored.goal_state, neg.goal_state)
        self.assertEqual(restored.theorem_id, neg.theorem_id)
        self.assertEqual(restored.failure_category, "semantic")
        self.assertEqual(restored.otp_dimensionality, 2)
        self.assertEqual(restored.bank_directions["depth"], -1)

    def test_infra_category(self):
        neg = NegativeExample(
            goal_state="⊢ P",
            theorem_id="t1",
            failure_category="infra",
            source="perturbation",
        )
        self.assertEqual(neg.failure_category, "infra")

    def test_weak_negative(self):
        neg = NegativeExample(
            goal_state="⊢ P",
            theorem_id="t1",
            failure_category="weak_negative",
            source="unchosen_weak",
            paired_positive_tactic="exact h",
        )
        d = neg.to_dict()
        self.assertEqual(d["paired_positive_tactic"], "exact h")


class TestConstraintReport(unittest.TestCase):
    def test_to_dict_without_energy(self):
        cr = ConstraintReport(
            bank_scores={"structure": 0.9, "domain": 0.5},
            critic_distance=3.0,
            censor_score=0.1,
            anchor_alignment=0.7,
            total_score=1.5,
        )
        d = cr.to_dict()
        self.assertNotIn("energy", d)
        self.assertEqual(d["total_score"], 1.5)

    def test_to_dict_with_energy(self):
        cr = ConstraintReport(
            bank_scores={},
            total_score=1.0,
            energy=0.42,
        )
        d = cr.to_dict()
        self.assertEqual(d["energy"], 0.42)


class TestSketchProposal(unittest.TestCase):
    def test_to_dict(self):
        sp = SketchProposal(
            template_id="REWRITE_CHAIN",
            proposed_steps=[
                ActionCandidate(tactic="rw", premises=["h1"]),
                ActionCandidate(tactic="simp"),
            ],
            total_constraint_score=2.1,
        )
        d = sp.to_dict()
        self.assertEqual(d["template_id"], "REWRITE_CHAIN")
        self.assertEqual(len(d["proposed_steps"]), 2)
        self.assertEqual(d["proposed_steps"][0]["tactic"], "rw")


class TestSearchTrace(unittest.TestCase):
    def test_defaults(self):
        st = SearchTrace(theorem_id="t1")
        self.assertEqual(st.mode, "v3")
        self.assertEqual(st.result, "failed")
        self.assertEqual(st.lean_calls, 0)

    def test_to_dict(self):
        st = SearchTrace(
            theorem_id="t1",
            lean_calls=5,
            result="proved",
            steps=[{"goal": "P", "tactic": "exact h"}],
        )
        d = st.to_dict()
        self.assertEqual(d["lean_calls"], 5)
        self.assertEqual(d["result"], "proved")
        self.assertEqual(len(d["steps"]), 1)


class TestConstraintReportAllFields(unittest.TestCase):
    """VALUE prescriptions: exact assertions for every to_dict field."""

    def test_all_fields_round_trip(self):
        cr = ConstraintReport(
            bank_scores={"structure": 0.9, "domain": 0.5},
            critic_distance=3.0,
            censor_score=0.1,
            anchor_alignment=0.7,
            total_score=1.5,
            energy=0.42,
        )
        d = cr.to_dict()
        self.assertEqual(d["bank_scores"], {"structure": 0.9, "domain": 0.5})
        self.assertAlmostEqual(d["critic_distance"], 3.0)
        self.assertAlmostEqual(d["censor_score"], 0.1)
        self.assertAlmostEqual(d["anchor_alignment"], 0.7)
        self.assertAlmostEqual(d["total_score"], 1.5)
        self.assertAlmostEqual(d["energy"], 0.42)

    def test_default_fields(self):
        cr = ConstraintReport(total_score=0.0)
        d = cr.to_dict()
        self.assertEqual(d["bank_scores"], {})
        self.assertAlmostEqual(d["critic_distance"], 0.0)
        self.assertAlmostEqual(d["censor_score"], 0.0)
        self.assertAlmostEqual(d["anchor_alignment"], 0.0)
        self.assertAlmostEqual(d["total_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
