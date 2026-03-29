from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.hard_resolution_layer import (
    build_dependency_profile,
    build_proof_plan_geometry,
    build_residual_skeleton_geometry,
    build_search_control_geometry,
    closing_features,
    materialize_hard_resolution_layer,
)
from src.proof_network import init_db, recompute_idf


class TestHardResolutionLayer(unittest.TestCase):
    def test_materialize_hard_resolution_layer_emits_priors_and_exemplars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "proof_network.db"
            out_dir = Path(tmpdir) / "hard_resolution"
            conn = init_db(db_path)

            conn.execute(
                "INSERT INTO entities (id, name, entity_type, namespace, provenance) VALUES (1, ?, 'lemma', 'Demo', 'traced')",
                ("Demo.hard",),
            )
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, namespace, provenance) VALUES (2, ?, 'lemma', 'Demo', 'traced')",
                ("Demo.eq_helper",),
            )
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, namespace, provenance) VALUES (3, ?, 'lemma', 'Other', 'traced')",
                ("Other.unrelated",),
            )
            for entity_id in (1, 2, 3):
                conn.executemany(
                    "INSERT INTO entity_positions (entity_id, bank, sign, depth) VALUES (?, ?, ?, ?)",
                    [
                        (entity_id, "structure", 1, 2),
                        (entity_id, "domain", 1 if entity_id != 3 else -1, 2),
                        (entity_id, "depth", 1, 1),
                        (entity_id, "automation", -1, 1),
                        (entity_id, "context", 1, 1),
                        (entity_id, "decomposition", -1, 1),
                    ],
                )

            conn.execute("INSERT INTO anchors (id, label, category) VALUES (1, 'eq', 'general')")
            conn.execute("INSERT INTO anchors (id, label, category) VALUES (2, 'Demo', 'general')")
            conn.execute("INSERT INTO anchors (id, label, category) VALUES (3, 'group', 'general')")
            conn.executemany(
                "INSERT INTO entity_anchors (entity_id, anchor_id, confidence) VALUES (?, ?, ?)",
                [
                    (1, 1, 1.0),
                    (1, 2, 1.0),
                    (2, 1, 1.0),
                    (2, 2, 1.0),
                    (3, 3, 1.0),
                ],
            )
            conn.executemany(
                "INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (?, ?)",
                [
                    (1, 2),
                    (1, 3),
                ],
            )
            conn.execute(
                "INSERT INTO entity_links (source_id, target_id, relation, weight) VALUES (1, 2, 'depends_on', 1.0)"
            )
            recompute_idf(conn)

            rows = [
                {
                    "theorem_id": "Demo.hard",
                    "source": "hard_theorems",
                    "success": False,
                    "started": True,
                    "follow_on_stage": "hard_proof_solver",
                    "reasoning_gap_family": "local_eq_close",
                    "residual_bucket": "single_goal_near_miss",
                    "last_goal_bucket": "equality",
                    "last_goal": "f x = g x",
                    "theorem_statement": "theorem Demo.hard : f x = g x",
                    "initial_goal": "⊢ f x = g x",
                    "remaining_goals_snapshot": ["f x = g x"],
                    "template_id": "REWRITE_CHAIN",
                    "tactics_used": ["rw [Demo.eq_helper]", "aesop"],
                    "lane_sequence": "cosine_rw→automation",
                    "difficulty_band": "hard",
                    "namespace_prefix": "Demo",
                    "step_trace": [
                        {
                            "tactic": "rw [Demo.eq_helper]",
                            "lane": "cosine_rw",
                            "progress": True,
                            "goal_before": "f x = g x",
                            "open_goals_after": ["g x = g x"],
                        },
                        {
                            "tactic": "",
                            "lane": "",
                            "progress": False,
                            "goal_before": "g x = g x",
                            "open_goals_after": ["g x = g x"],
                        },
                    ],
                },
                {
                    "theorem_id": "Demo.loop_eq",
                    "source": "hard_theorems",
                    "success": False,
                    "started": True,
                    "follow_on_stage": "theorem_replanner",
                    "reasoning_gap_family": "theorem_replanner",
                    "residual_bucket": "single_goal_near_miss",
                    "last_goal_bucket": "equality",
                    "last_goal": "f x = g x",
                    "theorem_statement": "theorem Demo.loop_eq : f x = g x",
                    "initial_goal": "⊢ f x = g x",
                    "remaining_goals_snapshot": ["f x = g x"],
                    "template_id": "REWRITE_CHAIN",
                    "tactics_used": ["rw [Demo.eq_helper]", "rw [← Demo.eq_helper]"],
                    "lane_sequence": "cosine_rw→cosine_rw",
                    "difficulty_band": "hard",
                    "namespace_prefix": "Demo",
                    "search_pathology_tags": ["no_progress_plateau", "blank_lane_plateau", "state_loop"],
                },
                {
                    "theorem_id": "Demo.solved_eq",
                    "source": "hard_theorems",
                    "success": True,
                    "honest_success": True,
                    "started": True,
                    "follow_on_stage": "none",
                    "reasoning_gap_family": "none",
                    "residual_bucket": "proved",
                    "last_goal_bucket": "empty",
                    "last_goal": "",
                    "theorem_statement": "theorem Demo.solved_eq : h x = h x",
                    "initial_goal": "⊢ h x = h x",
                    "remaining_goals_snapshot": [],
                    "template_id": "REWRITE_CHAIN",
                    "tactics_used": ["rw [Demo.eq_helper]", "aesop"],
                    "lane_sequence": "cosine_rw→automation",
                    "close_lane": "automation",
                    "difficulty_band": "hard",
                    "namespace_prefix": "Demo",
                },
                {
                    "theorem_id": "Demo.self_app",
                    "source": "hard_theorems",
                    "success": True,
                    "started": True,
                    "follow_on_stage": "none",
                    "reasoning_gap_family": "none",
                    "residual_bucket": "proved",
                    "last_goal_bucket": "empty",
                    "last_goal": "",
                    "theorem_statement": "theorem Demo.self_app : f x = g x",
                    "initial_goal": "⊢ f x = g x",
                    "remaining_goals_snapshot": [],
                    "template_id": "REWRITE_CHAIN",
                    "tactics_used": ["exact Demo.self_app"],
                    "final_closer": "exact Demo.self_app",
                    "close_provenance": ["self_application"],
                    "lane_sequence": "self_application",
                    "close_lane": "self_application",
                    "difficulty_band": "hard",
                    "namespace_prefix": "Demo",
                },
                {
                    "theorem_id": "Demo.matrix_start",
                    "source": "hard_theorems",
                    "success": False,
                    "started": False,
                    "follow_on_stage": "compiler_specialist",
                    "goal_start_status": "failed",
                    "failure_category": "typeclass_missing",
                    "start_failure_family": "scoped_context_missing",
                    "start_failure_tags": ["start_failure_family:scoped_context_missing"],
                    "theorem_statement": "Matrix α n n",
                    "initial_goal": "Matrix α n n",
                    "context_features": {"open_scoped": 2},
                    "context_unsupported_kinds": ["open_scoped"],
                    "module": "Mathlib.Demo.Matrix",
                    "lean_path": "Mathlib/Demo/Matrix.lean",
                    "theorem_line": 12,
                    "split": "eval",
                },
            ]

            summary = materialize_hard_resolution_layer(
                rows=rows,
                output_dir=out_dir,
                conn_or_db=conn,
                candidate_limit=5,
                exemplar_limit=3,
            )

            self.assertEqual(summary["total_hard_packets"], 1)
            self.assertEqual(summary["packets_with_candidate_priors"], 1)
            self.assertEqual(summary["packets_with_kline_exemplars"], 1)

            packets = (out_dir / "resolution_packets.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(packets), 1)
            packet = json.loads(packets[0])
            self.assertEqual(packet["resolution_family"], "local_eq_close")
            self.assertIn("rewrite_chain", packet["closing_features"])
            self.assertIn("residual_skeleton_geometry", packet)
            self.assertIn("proof_plan_geometry", packet)
            self.assertIn("prior_graph_geometry", packet)
            self.assertIn("kline_geometry", packet)
            self.assertIn("search_control_geometry", packet)
            self.assertIn("negative_kline_geometry", packet)
            self.assertIn("representation_pressures", packet["residual_skeleton_geometry"])
            self.assertIn("dr_ducky_surface", packet)
            self.assertIn("engine_family", packet["dr_ducky_surface"])
            self.assertIn("backend_preferences", packet["dr_ducky_surface"])
            self.assertIn("projector_policy", packet["dr_ducky_surface"])
            self.assertIn("proof_dsl_program_count", packet["dr_ducky_surface"])
            self.assertIn("solver_constraint_profile", packet["dr_ducky_surface"])
            self.assertIn("candidate_methods", packet["proof_plan_geometry"])
            self.assertEqual(packet["candidate_priors"][0]["lemma"], "Demo.eq_helper")
            self.assertEqual(packet["kline_exemplars"][0]["theorem_id"], "Demo.solved_eq")
            self.assertGreaterEqual(packet["negative_kline_geometry"]["exemplar_count"], 1)
            self.assertIn("rw", packet["negative_kline_geometry"]["avoid_tactic_prefixes"])
            self.assertNotIn("Demo.self_app", [e["theorem_id"] for e in packet["kline_exemplars"]])

            self.assertTrue((out_dir / "by_resolution_family" / "local_eq_close.jsonl").exists())
            self.assertTrue((out_dir / "hard_som_packets.jsonl").exists())
            self.assertTrue((out_dir / "compiler_specialist_packets.jsonl").exists())
            self.assertTrue((out_dir / "surface_inventory.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            compiler_packet = json.loads(
                (out_dir / "compiler_specialist_packets.jsonl").read_text().strip().splitlines()[0]
            )
            actions = compiler_packet["startability_surface"]["reconstruction_actions"]
            self.assertIn("replay_open_scopes", actions)
            self.assertIn("prefer_file_context_replay", actions)
            hard_packet = json.loads(
                (out_dir / "hard_som_packets.jsonl").read_text().strip().splitlines()[0]
            )
            self.assertIn("dr_ducky_surface", hard_packet)
            conn.close()

    def test_category_theory_side_conditions_emit_domain_suppression_hints(self) -> None:
        row = {
            "theorem_id": "CategoryTheory.Adjunction.isIso_counit_app_iff_mem_essImage",
            "reasoning_gap_family": "small_multigoal_side_conditions",
            "last_goal_bucket": "atomic_prop",
            "last_goal": "L.essImage X",
            "initial_goal": "CategoryTheory.IsIso (h.counit.app X) ↔ L.essImage X",
            "theorem_statement": "CategoryTheory.IsIso (h.counit.app X) ↔ L.essImage X",
            "lane_sequence": "interleaved_bootstrap→cosine_rw",
            "tactics_used": ["aesop", "rw [CategoryTheory.asIso]"],
            "attempt_band": "40_79",
        }
        remaining_goals = ["L.essImage X", "CategoryTheory.IsIso (h.counit.app X)"]
        skeleton = build_residual_skeleton_geometry(row, remaining_goals)
        profile = build_dependency_profile(remaining_goals)
        plan = build_proof_plan_geometry(
            row,
            skeleton=skeleton,
            decomposition_profile=profile,
            closing_feature_list=closing_features("small_multigoal_side_conditions", row["last_goal"]),
        )

        self.assertIn("category_theory", skeleton["domain_hints"])
        self.assertIn("definition_unfolding", skeleton["representation_pressures"])
        self.assertEqual(profile["side_condition_profile"], "repeated_small_goals")
        self.assertIn("side_condition_sweep", plan["candidate_methods"])
        self.assertIn("domain_aware_lane_suppression", plan["candidate_methods"])
        self.assertIn("suppress_numeric_solver_lane", plan["lane_suppression_hints"])
        self.assertIn("side_condition_sweeper", plan["specialist_targets"])

    def test_forward_context_and_pathologies_surface_replanner_and_context_targets(self) -> None:
        row = {
            "theorem_id": "Algebra.FormallyUnramified.of_comp",
            "reasoning_gap_family": "forward_context_close",
            "last_goal_bucket": "other",
            "last_goal": "Function.Injective (Ideal.Quotient.mkₐ A I).comp",
            "initial_goal": "Algebra.FormallyUnramified A B",
            "theorem_statement": "Function.Injective (Ideal.Quotient.mkₐ A I).comp",
            "lane_sequence": "interleaved_bootstrap→cosine_rw",
            "tactics_used": ["aesop", "rw [Algebra.FormallyUnramified.iff_comp_injective]"],
            "attempt_band": "80_119",
            "search_pathology_tags": ["metavariable_corruption", "state_loop"],
        }
        remaining_goals = ["Function.Injective (Ideal.Quotient.mkₐ A I).comp"]
        skeleton = build_residual_skeleton_geometry(row, remaining_goals)
        profile = build_dependency_profile(remaining_goals)
        plan = build_proof_plan_geometry(
            row,
            skeleton=skeleton,
            decomposition_profile=profile,
            closing_feature_list=closing_features("forward_context_close", row["last_goal"]),
        )

        self.assertIn("forward_context_chase", skeleton["representation_pressures"])
        self.assertIn("metavariable_repair", skeleton["representation_pressures"])
        self.assertIn("loop_escape", skeleton["representation_pressures"])
        self.assertIn("forward_context_chase", plan["candidate_methods"])
        self.assertIn("metavariable_repair", plan["candidate_methods"])
        self.assertIn("forward_reasoner", plan["specialist_targets"])
        self.assertIn("suppress_fold_unfold_ping_pong", plan["lane_suppression_hints"])

    def test_membership_and_structural_eq_emit_new_specialists_and_hints(self) -> None:
        membership_row = {
            "theorem_id": "AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier.smul_mem",
            "reasoning_gap_family": "membership_close",
            "last_goal_bucket": "membership",
            "last_goal": "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q",
            "initial_goal": "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q",
            "theorem_statement": "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q",
            "lane_sequence": "interleaved_bootstrap→cosine_exact",
            "tactics_used": ["simp", "aesop"],
            "attempt_band": "80_119",
        }
        membership_skeleton = build_residual_skeleton_geometry(membership_row, [membership_row["last_goal"]])
        membership_plan = build_proof_plan_geometry(
            membership_row,
            skeleton=membership_skeleton,
            decomposition_profile=build_dependency_profile([membership_row["last_goal"]]),
            closing_feature_list=closing_features("membership_close", membership_row["last_goal"]),
        )
        self.assertIn("opaque_membership_unfolding", membership_skeleton["representation_pressures"])
        self.assertIn("membership_specialist", membership_plan["specialist_targets"])
        self.assertIn("preserve_smul_membership_operators", membership_plan["lane_suppression_hints"])

        eq_row = {
            "theorem_id": "Algebra.discr_eq_discr_of_toMatrix_coeff_isIntegral",
            "reasoning_gap_family": "local_eq_close",
            "last_goal_bucket": "equality",
            "last_goal": "(Algebra.traceMatrix ℚ ⇑b).det = Algebra.discr ℚ ⇑b'",
            "initial_goal": "Algebra.discr ℚ ⇑b = Algebra.discr ℚ ⇑b'",
            "theorem_statement": "IsIntegral ℤ (b.toMatrix (⇑b') i j) → Algebra.discr ℚ ⇑b = Algebra.discr ℚ ⇑b'",
            "lane_sequence": "interleaved_bootstrap→cosine_rw",
            "tactics_used": ["aesop", "rw [Algebra.discr]", "rw [Matrix.det]", "rw [← Matrix.det]"],
            "attempt_band": "80_119",
            "search_pathology_tags": ["state_loop", "definition_tug_of_war"],
        }
        eq_skeleton = build_residual_skeleton_geometry(eq_row, [eq_row["last_goal"]])
        eq_plan = build_proof_plan_geometry(
            eq_row,
            skeleton=eq_skeleton,
            decomposition_profile=build_dependency_profile([eq_row["last_goal"]]),
            closing_feature_list=closing_features("local_eq_close", eq_row["last_goal"]),
        )
        self.assertIn("hypothesis_injection", eq_skeleton["representation_pressures"])
        self.assertIn("symmetric_unfolding", eq_skeleton["representation_pressures"])
        self.assertIn("close_before_unpack", eq_plan["candidate_methods"])
        self.assertIn("prefer_structural_exact_before_unfold", eq_plan["lane_suppression_hints"])

    def test_category_theory_exists_close_surfaces_canonical_witness_methods(self) -> None:
        row = {
            "theorem_id": "CategoryTheory.Abelian.epiWithInjectiveKernel_of_iso",
            "reasoning_gap_family": "exists_close",
            "last_goal_bucket": "exists",
            "last_goal": "∃ I, CategoryTheory.Injective I ∧ Nonempty { X₁ := I, X₂ := X, X₃ := Y, f := 0, g := f }.Splitting",
            "initial_goal": "epiWithInjectiveKernel f",
            "theorem_statement": "CategoryTheory.IsIso f → epiWithInjectiveKernel f",
            "lane_sequence": "interleaved_bootstrap→cosine_rw→automation",
            "tactics_used": ["aesop", "rw [epiWithInjectiveKernel]", "aesop"],
            "attempt_band": "80_119",
        }
        skeleton = build_residual_skeleton_geometry(row, [row["last_goal"]])
        plan = build_proof_plan_geometry(
            row,
            skeleton=skeleton,
            decomposition_profile=build_dependency_profile([row["last_goal"]]),
            closing_feature_list=closing_features("exists_close", row["last_goal"]),
        )

        self.assertIn("canonical_object_witness", skeleton["representation_pressures"])
        self.assertIn("zero_object_instantiation", skeleton["representation_pressures"])
        self.assertIn("zero_object_instantiation", plan["candidate_methods"])
        self.assertIn("witness_instantiation_specialist", plan["specialist_targets"])
        self.assertIn("favor_canonical_witnesses_before_search", plan["lane_suppression_hints"])

    def test_plateau_and_eventual_filter_surface_negative_memory_methods(self) -> None:
        row = {
            "theorem_id": "AkraBazziRecurrence.GrowsPolynomially.abs",
            "reasoning_gap_family": "theorem_replanner",
            "last_goal_bucket": "other",
            "last_goal": "AkraBazziRecurrence.GrowsPolynomially ?m.6",
            "initial_goal": "AkraBazziRecurrence.GrowsPolynomially fun x => |f x|",
            "theorem_statement": "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
            "lane_sequence": "cosine_rw→interleaved_bootstrap",
            "tactics_used": ["rw [← AkraBazziRecurrence.GrowsPolynomially.iff_eventuallyEq]", "simp"],
            "attempt_band": "80_119",
            "search_pathology_tags": ["no_progress_plateau", "blank_lane_plateau", "state_loop"],
            "step_trace": [
                {
                    "tactic": "",
                    "lane": "",
                    "progress": False,
                    "goal_before": "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "open_goals_after": ["AkraBazziRecurrence.GrowsPolynomially ?m.6"],
                },
                {
                    "tactic": "",
                    "lane": "",
                    "progress": False,
                    "goal_before": "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "open_goals_after": ["AkraBazziRecurrence.GrowsPolynomially ?m.6"],
                },
                {
                    "tactic": "",
                    "lane": "",
                    "progress": False,
                    "goal_before": "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "open_goals_after": ["AkraBazziRecurrence.GrowsPolynomially ?m.6"],
                },
                {
                    "tactic": "",
                    "lane": "",
                    "progress": False,
                    "goal_before": "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "open_goals_after": ["AkraBazziRecurrence.GrowsPolynomially ?m.6"],
                },
            ],
        }
        remaining_goals = [
            "AkraBazziRecurrence.GrowsPolynomially ?m.6",
            "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
            "ℝ → ℝ",
        ]
        skeleton = build_residual_skeleton_geometry(row, remaining_goals)
        search_control = build_search_control_geometry(row)
        plan = build_proof_plan_geometry(
            row,
            skeleton=skeleton,
            decomposition_profile=build_dependency_profile(remaining_goals),
            closing_feature_list=closing_features("theorem_replanner", row["last_goal"]),
            search_control_geometry=search_control,
        )

        self.assertIn("eventual_filter_reasoning", skeleton["representation_pressures"])
        self.assertEqual(search_control["plateau_detected"], 1)
        self.assertIn("negative_kline_retrieval", plan["candidate_methods"])
        self.assertIn("plateau_bailout", plan["candidate_methods"])
        self.assertIn("eventual_filter_normalization", plan["candidate_methods"])
        self.assertIn("eventual_filter_specialist", plan["specialist_targets"])
        self.assertIn("plateau_escape_replanner", plan["specialist_targets"])
        self.assertIn("bail_out_identical_blank_lane_plateaus", plan["lane_suppression_hints"])


if __name__ == "__main__":
    unittest.main()
