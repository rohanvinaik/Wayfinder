from __future__ import annotations

import unittest

from src.dr_ducky import build_goal_capsule, build_goal_specification


class TestDrDucky(unittest.TestCase):
    def test_build_goal_specification_detects_membership_and_category_suppression(self) -> None:
        row = {
            "theorem_id": "CategoryTheory.Demo.mem",
            "last_goal": "x ∈ Submodule.carrier S",
            "last_goal_bucket": "membership",
            "reasoning_gap_family": "membership_close",
            "residual_bucket": "single_goal_near_miss",
            "difficulty_band": "hard",
            "goals_closed": 4,
            "goals_remaining": 1,
            "attempts": 20,
            "lane_sequence": "interleaved_bootstrap→last_resort_exact",
            "search_pathology_tags": [],
        }
        spec = build_goal_specification(row)
        self.assertEqual(spec.goal_bucket, "membership")
        self.assertTrue(spec.signals["has_membership_wall"])
        self.assertTrue(spec.signals["has_category_surface"])
        self.assertIn("domain_solver_mismatch_risk", spec.representation_pressures)
        self.assertIn("FiniteFilterEngine", spec.engine_eligibility)
        self.assertIn("structural_close", spec.projector_markers)

        capsule = build_goal_capsule(row)
        banks = {prior.name: prior for prior in capsule.bank_priors}
        self.assertIn("membership_exposure", banks)
        self.assertIn("structural_close", banks)
        self.assertIn("suppress_numeric_solvers", capsule.suppression_hints)
        self.assertIn("membership_surface_engine", capsule.specialist_targets)
        self.assertIn("FiniteFilterEngine", capsule.allowed_engines)
        self.assertIn("kodkod_relational", capsule.backend_preferences)
        self.assertTrue(capsule.relational_search_specs)
        self.assertGreaterEqual(len(capsule.ledger_seed.facts), 0)

    def test_build_goal_capsule_prioritizes_symbolic_sandbox_for_equality(self) -> None:
        row = {
            "theorem_id": "Demo.eq",
            "last_goal": "↑x + 0 = ↑x",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "local_eq_close",
            "residual_bucket": "single_goal_near_miss",
            "difficulty_band": "hard",
            "goals_closed": 5,
            "goals_remaining": 1,
            "attempts": 12,
            "lane_sequence": "interleaved_bootstrap",
            "search_pathology_tags": [],
        }
        capsule = build_goal_capsule(row)
        self.assertEqual(capsule.specification.goal_bucket, "equality")
        self.assertIn("human_calculator", capsule.specialist_targets)
        self.assertIn("symbolic_sandbox", capsule.specialist_targets)
        banks = {prior.name: prior for prior in capsule.bank_priors}
        self.assertIn("eq_sat", banks)
        self.assertFalse(banks["eq_sat"].suppressed)
        prescription_kinds = {item.prescription_kind for item in capsule.prescriptions}
        self.assertIn("enter_symbolic_sandbox", prescription_kinds)
        self.assertIn("normalize_coercions", prescription_kinds)
        self.assertIn("saturate_equality", prescription_kinds)
        skeleton_ids = {item.skeleton_id for item in capsule.proof_skeletons}
        self.assertIn("exact_local_fact", skeleton_ids)
        self.assertIn("rw_accessible_fwd", skeleton_ids)
        self.assertIn("ring_nf_norm_num", skeleton_ids)
        self.assertIn("EqSatEngine", capsule.allowed_engines)
        self.assertIn("egglog_eqsat", capsule.backend_preferences)
        self.assertIn("rewrite_chain", capsule.specification.projector_markers)
        self.assertTrue(capsule.projector_policy["require_projected_program"])
        self.assertIsNotNone(capsule.eqsat_plan)
        self.assertIn("algebraic_eq", capsule.eqsat_plan.rewrite_theories)
        self.assertTrue(capsule.proof_dsl_programs)
        self.assertTrue(capsule.solver_constraints)

    def test_unionfind_trace_routes_to_recursive_circuit_breaker(self) -> None:
        row = {
            "theorem_id": "Batteries.UnionFind.rootD_parent",
            "last_goal": "self.parent x = self.rootD x",
            "last_goal_bucket": "equality",
            "reasoning_gap_family": "theorem_replanner",
            "residual_bucket": "multi_goal_large_progress",
            "difficulty_band": "expert",
            "goals_closed": 21,
            "goals_remaining": 20,
            "attempts": 541,
            "lane_sequence": "automation→cosine_rw→interleaved_bootstrap",
            "search_pathology_tags": [
                "duplicate_goal_progress",
                "duplicate_goal_pseudo_progress",
                "goal_explosion",
                "no_progress_plateau",
                "blank_lane_plateau",
            ],
            "remaining_goals_snapshot": [
                "self.parent x = self.rootD x",
                "self.parent x = self.rootD x",
                "self.parent x = self.rootD x",
            ],
            "step_trace": [
                {
                    "goal_before": "self.rootD (self.parent x) = self.rootD x",
                    "lane": "cosine_rw",
                    "tactic": "rw [Batteries.UnionFind.rootD]",
                    "progress": True,
                },
                {
                    "goal_before": "self.parent x = self.rootD x",
                    "lane": "interleaved_bootstrap",
                    "tactic": "norm_num",
                    "progress": True,
                },
                {
                    "goal_before": "self.parent x = self.rootD x",
                    "lane": "",
                    "tactic": "",
                    "progress": False,
                },
                {
                    "goal_before": "self.parent x = self.rootD x",
                    "lane": "",
                    "tactic": "",
                    "progress": False,
                },
                {
                    "goal_before": "self.parent x = self.rootD x",
                    "lane": "",
                    "tactic": "",
                    "progress": False,
                },
                {
                    "goal_before": "self.parent x = self.rootD x",
                    "lane": "",
                    "tactic": "",
                    "progress": False,
                },
            ],
        }
        capsule = build_goal_capsule(row)
        self.assertTrue(capsule.specification.signals["recursive_loop_risk"])
        self.assertIn("recursive_circuit_breaker", capsule.specialist_targets)
        self.assertIn("suppress_repeat_rw", capsule.suppression_hints)
        self.assertIn("suppress_repeat_norm_num", capsule.suppression_hints)
        banks = {prior.name: prior for prior in capsule.bank_priors}
        self.assertIn("loop_breaker", banks)
        self.assertIn("recursive_unfold_one", banks)
        self.assertTrue(banks["eq_sat"].suppressed)
        prescription_kinds = {item.prescription_kind for item in capsule.prescriptions}
        self.assertIn("recursive_loop_circuit_break", prescription_kinds)
        self.assertIn("bounded_unfold", prescription_kinds)
        self.assertIn("RecursiveInvariantEngine", capsule.allowed_engines)
        self.assertIn("symbolic_rewrite_vm", capsule.backend_preferences)
        self.assertIn("recursive_bridge", capsule.specification.projector_markers)

    def test_modulargroup_symbolic_goal_prefers_sandbox(self) -> None:
        row = {
            "theorem_id": "ModularGroup.smul_eq_lcRow0_add",
            "last_goal": "∀ {g : Matrix.SpecialLinearGroup (Fin 2) ℤ} (z : UpperHalfPlane) {p : Fin 2 → ℤ},"
            " IsCoprime (p 0) (p 1) → ↑g 1 = p → ↑(g • z) ="
            " ↑((ModularGroup.lcRow0 p) ↑((Matrix.SpecialLinearGroup.map (Int.castRingHom ℝ)) g)) /"
            " (↑(p 0) ^ 2 + ↑(p 1) ^ 2) +"
            " (↑(p 1) * ↑z - ↑(p 0)) / ((↑(p 0) ^ 2 + ↑(p 1) ^ 2) * (↑(p 0) * ↑z + ↑(p 1)))",
            "last_goal_bucket": "forall",
            "reasoning_gap_family": "single_goal_stall",
            "residual_bucket": "single_goal_stall",
            "difficulty_band": "hard",
            "goals_closed": 0,
            "goals_remaining": 1,
            "attempts": 44,
            "lane_sequence": "",
            "search_pathology_tags": [],
        }
        capsule = build_goal_capsule(row)
        self.assertIn("symbolic_sandbox", capsule.specialist_targets)
        self.assertIn("binder_drilldown", capsule.specialist_targets)
        self.assertIn("prefer_symbolic_sandbox", capsule.suppression_hints)
        prescription_kinds = {item.prescription_kind for item in capsule.prescriptions}
        self.assertIn("enter_symbolic_sandbox", prescription_kinds)
        self.assertIn("binder_drilldown", prescription_kinds)
        bank_names = {prior.name for prior in capsule.bank_priors if not prior.suppressed}
        self.assertIn("transport_normalizer", bank_names)
        self.assertIn("arith_nf", bank_names)
        skeleton_ids = {item.skeleton_id for item in capsule.proof_skeletons}
        self.assertIn("intros_norm_cast", skeleton_ids)
        self.assertIn("intros_push_cast", skeleton_ids)
        self.assertIn("EqSatEngine", capsule.allowed_engines)
        self.assertIn("ContextTransportEngine", capsule.allowed_engines)
        self.assertIn("rosette_proof_dsl", capsule.backend_preferences)
        self.assertTrue(any(program.backend_family == "rosette_proof_dsl" for program in capsule.proof_dsl_programs))


if __name__ == "__main__":
    unittest.main()
