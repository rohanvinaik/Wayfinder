from __future__ import annotations

import unittest

from src.hard_data_tags import (
    canonicalize_theorem_id,
    classify_goal_bucket,
    classify_reasoning_gap_family,
    classify_start_failure_family,
    goal_bucket_tags,
    sanitize_goal_text,
    start_failure_tags,
    trace_pathology_tags,
)


class TestHardDataTags(unittest.TestCase):
    def test_goal_bucket_and_reasoning_gap_family(self) -> None:
        self.assertEqual(classify_goal_bucket("False"), "false")
        self.assertEqual(classify_goal_bucket("x = y"), "equality")
        self.assertEqual(
            classify_goal_bucket("n ≤ sSup {N | ∃ s, s.card = N ∧ True}"),
            "inequality",
        )
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=True,
                residual_bucket="single_goal_near_miss",
                last_goal_bucket="equality",
            ),
            "local_eq_close",
        )
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=False,
                residual_bucket="skipped_start",
                last_goal_bucket="empty",
            ),
            "compiler_specialist",
        )
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=True,
                residual_bucket="single_goal_near_miss",
                last_goal_bucket="inequality",
                goal_text="n ≤ sSup {N | ∃ s, s.card = N ∧ True}",
            ),
            "witness_construction_close",
        )
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=True,
                residual_bucket="single_goal_near_miss",
                last_goal_bucket="other",
                goal_text="Function.Injective (Ideal.Quotient.mkₐ A I).comp",
            ),
            "forward_context_close",
        )
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=True,
                residual_bucket="multi_goal_small_progress",
                last_goal_bucket="atomic_prop",
                remaining_goals=[
                    "a ≤ #α",
                    "a ≤ #α",
                    "#α ≤ a",
                    "a ≤ #α",
                ],
            ),
            "small_multigoal_side_conditions",
        )
        tags = goal_bucket_tags("n ≤ sSup {N | ∃ s, s.card = N ∧ True}")
        self.assertIn("contains_existential_substructure", tags)
        self.assertIn("witness_construction_pressure", tags)
        self.assertEqual(
            classify_goal_bucket("autoParam (∀ (X Y : C), f = f) Demo.auto"),
            "forall",
        )

    def test_sanitize_goal_text_strips_meta_wrappers(self) -> None:
        cleaned = sanitize_goal_text("autoParam (∀ (X Y : C), f = f) CategoryTheory.Functor.ext._auto_1")
        self.assertEqual(cleaned, "∀ (X Y : C), f = f")

    def test_canonicalize_theorem_id_collapses_duplicate_prefix(self) -> None:
        self.assertEqual(
            canonicalize_theorem_id(
                "AlgebraicGeometry.FormallyUnramified.AlgebraicGeometry.FormallyUnramified.isOpenImmersion_diagonal"
            ),
            "AlgebraicGeometry.FormallyUnramified.isOpenImmersion_diagonal",
        )

    def test_goal_bucket_tags_surface_structural_and_membership_walls(self) -> None:
        membership_tags = goal_bucket_tags(
            "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q"
        )
        self.assertIn("opaque_membership_wall", membership_tags)
        structural_tags = goal_bucket_tags(
            "IsOpenMap (ConcreteCategory.hom (AffineSpace n S ↘ S).base)"
        )
        self.assertIn("structural_property_goal", structural_tags)
        bridge_tags = goal_bucket_tags(
            "(Algebra.traceMatrix ℚ ⇑b).det = Algebra.discr ℚ ⇑b'"
        )
        self.assertIn("local_hypothesis_bridge_goal", bridge_tags)
        exists_tags = goal_bucket_tags(
            "∃ I, CategoryTheory.Injective I ∧ Nonempty { X₁ := I, X₂ := X, X₃ := Y, f := 0, g := f }.Splitting"
        )
        self.assertIn("canonical_exists_witness", exists_tags)

    def test_trace_pathologies_drive_replanner(self) -> None:
        step_trace = [
            {
                "tactic": "rw [← Demo.helper]",
                "progress": True,
                "open_goals_after": [
                    "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
                    "ℝ → ℝ",
                ],
            },
            {
                "tactic": "simp",
                "progress": True,
                "open_goals_after": [
                    "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
                    "ℝ → ℝ",
                ],
            },
        ]
        tags = trace_pathology_tags(
            step_trace,
            remaining_goals=[
                "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
                "ℝ → ℝ",
            ],
        )
        self.assertIn("metavariable_corruption", tags)
        self.assertIn("backward_rewrite_metavariable", tags)
        self.assertIn("state_loop", tags)
        self.assertIn("definition_tug_of_war", tags)
        self.assertEqual(
            classify_reasoning_gap_family(
                success=False,
                started=True,
                residual_bucket="multi_goal_small_progress",
                last_goal_bucket="other",
                goal_text="AkraBazziRecurrence.GrowsPolynomially ?m.6",
                remaining_goals=[
                    "AkraBazziRecurrence.GrowsPolynomially ?m.6",
                    "?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
                    "ℝ → ℝ",
                ],
                pathology_tags=tags,
            ),
            "theorem_replanner",
        )

    def test_blank_lane_plateau_is_tagged(self) -> None:
        step_trace = [
            {
                "tactic": "",
                "lane": "",
                "progress": False,
                "goal_before": "IsOpenMap f",
                "open_goals_after": ["IsOpenMap f"],
            },
            {
                "tactic": "",
                "lane": "",
                "progress": False,
                "goal_before": "IsOpenMap f",
                "open_goals_after": ["IsOpenMap f"],
            },
            {
                "tactic": "",
                "lane": "",
                "progress": False,
                "goal_before": "IsOpenMap f",
                "open_goals_after": ["IsOpenMap f"],
            },
            {
                "tactic": "",
                "lane": "",
                "progress": False,
                "goal_before": "IsOpenMap f",
                "open_goals_after": ["IsOpenMap f"],
            },
        ]
        tags = trace_pathology_tags(step_trace, remaining_goals=["IsOpenMap f"])
        self.assertIn("no_progress_plateau", tags)
        self.assertIn("blank_lane_plateau", tags)

    def test_start_failure_family_and_tags(self) -> None:
        family = classify_start_failure_family(
            failure_category="universe_compilation_fail",
            goal_text="∀ {α : Type u_1}, Foo α → Bar α",
            module="Mathlib.Algebra.Ring.Ext",
            theorem_line=100,
            context_features={"variable": 1},
            context_unsupported_kinds=[],
        )
        self.assertEqual(family, "universe_binder_pressure")
        tags = start_failure_tags(
            failure_category="goal_creation_fail",
            goal_text="∀ {α : Type u_1}, Foo α ⋯",
            module="",
            theorem_line=0,
            context_features={"variable": 12, "open": 1},
            context_unsupported_kinds=["variable"],
        )
        self.assertIn("module_metadata_missing", tags)
        self.assertIn("unsafe_pretty_print", tags)
        self.assertIn("variable_heavy_context", tags)

    def test_replay_namespace_leakage_is_start_failure_family(self) -> None:
        family = classify_start_failure_family(
            failure_category="goal_creation_fail",
            goal_text="_wayfinder_replay__wayfinder_decl_0.CategoryTheory.Foo.bar",
            module="Mathlib.CategoryTheory.Foo",
            theorem_line=12,
            context_features={},
            context_unsupported_kinds=[],
        )
        self.assertEqual(family, "replay_namespace_leakage")
        tags = start_failure_tags(
            failure_category="goal_creation_fail",
            goal_text="_wayfinder_replay__wayfinder_decl_0.CategoryTheory.Foo.bar",
            module="Mathlib.CategoryTheory.Foo",
            theorem_line=12,
            context_features={},
            context_unsupported_kinds=[],
        )
        self.assertIn("replay_namespace_leakage", tags)

    def test_open_scoped_missing_becomes_scoped_context_failure(self) -> None:
        family = classify_start_failure_family(
            failure_category="typeclass_missing",
            goal_text="Matrix α n n",
            module="Mathlib.CategoryTheory.Foo",
            theorem_line=12,
            context_features={"open_scoped": 2},
            context_unsupported_kinds=["open_scoped"],
        )
        self.assertEqual(family, "scoped_context_missing")


if __name__ == "__main__":
    unittest.main()
