from __future__ import annotations

import unittest

from scripts.run_exp_som012_oracle_gap import (
    closer_candidates,
    premise_tactic_candidates,
    summarize_oracle_rows,
)


class TestRunExpSom012OracleGap(unittest.TestCase):
    def test_closer_candidates_are_bucket_sensitive(self) -> None:
        eq = closer_candidates("equality")
        false = closer_candidates("false")
        ineq = closer_candidates("inequality", "local_ineq_close")
        cat = closer_candidates(
            "atomic_prop",
            "small_multigoal_side_conditions",
            "CategoryTheory.IsIso (h.counit.app X) ↔ L.essImage X",
            "CategoryTheory.Adjunction.isIso_counit_app_iff_mem_essImage",
        )
        self.assertIn("rfl", eq)
        self.assertIn("contradiction", false)
        self.assertIn("gcongr", ineq)
        self.assertIn("aesop", eq)
        self.assertIn("aesop", false)
        self.assertNotIn("norm_num", cat)

    def test_premise_tactic_candidates_prioritize_family_patterns(self) -> None:
        tactics = premise_tactic_candidates("Demo.eq_helper", "local_eq_close")
        witness = premise_tactic_candidates("Demo.witness", "witness_construction_close")
        self.assertIn("exact Demo.eq_helper", tactics)
        self.assertIn("apply Demo.eq_helper", tactics)
        self.assertIn("rw [Demo.eq_helper]", tactics)
        self.assertIn("apply Demo.witness", witness)

    def test_abstract_domain_closers_drop_numeric_tactics(self) -> None:
        alg_geo = closer_candidates(
            "other",
            "local_goal_close",
            "IsOpenMap (ConcreteCategory.hom (AffineSpace n S ↘ S).base)",
            "AlgebraicGeometry.AffineSpace.isOpenMap_over",
        )
        membership = closer_candidates(
            "membership",
            "membership_close",
            "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q",
            "AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier.smul_mem",
        )
        self.assertNotIn("norm_num", alg_geo)
        self.assertNotIn("norm_num", membership)

    def test_exists_close_adds_canonical_witness_closers(self) -> None:
        closers = closer_candidates(
            "exists",
            "exists_close",
            "∃ I, CategoryTheory.Injective I ∧ Nonempty { X₁ := I, X₂ := X, X₃ := Y, f := 0, g := f }.Splitting",
            "CategoryTheory.Abelian.epiWithInjectiveKernel_of_iso",
        )
        tactics = premise_tactic_candidates("CategoryTheory.Limits.isZero_zero", "exists_close")
        self.assertIn("use 0", closers)
        self.assertIn("refine ⟨0, ?_⟩", closers)
        self.assertIn("use 0", tactics)

    def test_summarize_oracle_rows(self) -> None:
        rows = [
            {
                "theorem_id": "Demo.one",
                "hard_track": "hard_proof_local",
                "reasoning_gap_family": "local_eq_close",
                "startable": True,
                "routing_label_available": True,
                "resolution_packet_available": True,
                "premise_progress": True,
                "premise_close": False,
                "closer_close": True,
                "combined_close": True,
            },
            {
                "theorem_id": "Demo.two",
                "hard_track": "hard_proof_planner",
                "reasoning_gap_family": "small_multigoal_planner",
                "startable": False,
                "routing_label_available": True,
                "resolution_packet_available": False,
                "premise_progress": False,
                "premise_close": False,
                "closer_close": False,
                "combined_close": False,
            },
        ]
        summary = summarize_oracle_rows(rows)
        self.assertEqual(summary["total_theorems"], 2)
        self.assertEqual(summary["startability_oracle"], 1)
        self.assertEqual(summary["final_closer_close"], 1)
        self.assertEqual(summary["combined_close"], 1)
        self.assertEqual(summary["by_reasoning_gap_family"]["local_eq_close"]["count"], 1)
        self.assertEqual(summary["by_hard_track"]["hard_proof_local"]["combined_close"], 1)


if __name__ == "__main__":
    unittest.main()
