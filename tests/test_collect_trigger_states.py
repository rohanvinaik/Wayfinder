"""Tests for trigger-state collection helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.collect_trigger_states import (
    _build_goal_start_failure_row,
    _goal_shape_features,
)
from src.nav_contracts import LeanFeedback


class _FakeLean:
    def __init__(self, feedback: LeanFeedback | None) -> None:
        self._last_goal_feedback = feedback


class TestGoalShapeFeatures(unittest.TestCase):
    def test_counts_basic_connectives_and_binders(self) -> None:
        goal = "∀ (α : Type) (x : α), ∃ y, x = y ∧ x ≠ y → False"
        feat = _goal_shape_features(goal)
        self.assertEqual(feat["forall_count"], 1)
        self.assertEqual(feat["exists_count"], 1)
        self.assertGreaterEqual(feat["binder_count"], 2)
        self.assertEqual(feat["eq_count"], 1)
        self.assertEqual(feat["neq_count"], 1)
        self.assertEqual(feat["and_count"], 1)
        self.assertEqual(feat["arrow_count"], 1)
        self.assertEqual(feat["type_count"], 1)


class TestGoalStartFailureRow(unittest.TestCase):
    @patch("scripts.collect_trigger_states._resolve_theorem_site_metadata")
    def test_build_goal_start_failure_row_with_repair(self, mock_site) -> None:
        mock_site.return_value = {
            "module": "Mathlib.MeasureTheory.Basic",
            "lean_path": "/tmp/MeasureTheory/Basic.lean",
            "theorem_line": 123,
            "theorem_type": "∀ (μ : Measure α), μ = μ",
            "context_features": {
                "open": 1,
                "open_scoped": 2,
                "variable": 1,
                "universe": 0,
                "local_notation": 0,
                "scoped_notation": 0,
                "notation": 0,
                "local_attribute": 1,
                "include": 0,
                "omit": 0,
            },
            "context_unsupported_kinds": ["open_scoped"],
        }
        lean = _FakeLean(
            LeanFeedback(
                stage="goal_creation",
                category="typeclass_missing",
                messages=[{"data": "failed to synthesize instance"}],
                raw_error="failed to synthesize instance",
            )
        )

        class _Repair:
            success = True
            goal_state = "μ = μ"
            tier_used = "B"
            failure_category = ""
            feedback = LeanFeedback.success()

        row = _build_goal_start_failure_row(
            {
                "theorem_id": "MeasureTheory.foo",
                "goal_state": "raw goal",
                "source": "benchmark_theorems",
            },
            lean,
            repaired_goal="μ = μ",
            repair_result=_Repair(),
        )

        self.assertEqual(row["row_type"], "goal_start_failure")
        self.assertEqual(row["namespace_prefix"], "MeasureTheory")
        self.assertEqual(row["failure_category"], "typeclass_missing")
        self.assertTrue(row["repair_success"])
        self.assertEqual(row["repair_tier_used"], "B")
        self.assertEqual(row["context_features"]["open_scoped"], 2)
        self.assertIn("open_scoped", row["context_unsupported_kinds"])
        self.assertEqual(row["goal_start_status"], "recovered_via_file_context")
        self.assertIsNotNone(row["feedback"])
        self.assertEqual(row["start_failure_family"], "scoped_context_missing")


if __name__ == "__main__":
    unittest.main()
