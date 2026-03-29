from __future__ import annotations

import unittest

from src.benchmark_residuals import (
    augment_result_entry,
    detect_self_application,
    summarize_residual_structure,
)


class TestBenchmarkResiduals(unittest.TestCase):
    def test_augment_result_entry_classifies_skipped_start(self) -> None:
        entry = augment_result_entry(
            {
                "theorem_id": "Demo.skipped",
                "success": False,
                "attempts": 0,
                "goals_closed": 0,
                "goals_remaining": 1,
                "close_lane": "skipped",
            }
        )
        self.assertEqual(entry["residual_bucket"], "skipped_start")
        self.assertEqual(entry["follow_on_stage"], "compiler_specialist")
        self.assertEqual(entry["progress_band"], "unstarted")
        self.assertEqual(entry["started"], False)

    def test_augment_result_entry_classifies_near_miss(self) -> None:
        entry = augment_result_entry(
            {
                "theorem_id": "Demo.near_miss",
                "success": False,
                "attempts": 37,
                "goals_closed": 4,
                "goals_remaining": 1,
                "close_lane": "failed",
            }
        )
        self.assertEqual(entry["residual_bucket"], "single_goal_near_miss")
        self.assertEqual(entry["follow_on_stage"], "hard_proof_solver")
        self.assertEqual(entry["progress_band"], "near_miss")

    def test_summarize_residual_structure_reports_bucket_counts(self) -> None:
        results = [
            {
                "theorem_id": "Demo.proved",
                "success": True,
                "attempts": 3,
                "goals_closed": 1,
                "goals_remaining": 0,
                "close_lane": "automation",
            },
            {
                "theorem_id": "Demo.skipped",
                "success": False,
                "attempts": 0,
                "goals_closed": 0,
                "goals_remaining": 1,
                "close_lane": "skipped",
            },
            {
                "theorem_id": "Demo.one_goal_left",
                "success": False,
                "attempts": 19,
                "goals_closed": 2,
                "goals_remaining": 1,
                "close_lane": "failed",
            },
            {
                "theorem_id": "Demo.large_residual",
                "success": False,
                "attempts": 25,
                "goals_closed": 1,
                "goals_remaining": 6,
                "close_lane": "failed",
            },
        ]
        summary = summarize_residual_structure(results)

        self.assertEqual(summary["started_theorems"], 3)
        self.assertEqual(summary["skipped_start"], 1)
        self.assertEqual(summary["progressed_but_unsolved"], 2)
        self.assertEqual(summary["one_goal_left_failures"], 2)
        self.assertEqual(summary["by_residual_bucket"]["proved"], 1)
        self.assertEqual(summary["by_residual_bucket"]["skipped_start"], 1)
        self.assertEqual(summary["by_residual_bucket"]["single_goal_near_miss"], 1)
        self.assertEqual(summary["by_follow_on_stage"]["compiler_specialist"], 1)
        self.assertEqual(summary["by_follow_on_stage"]["hard_proof_solver"], 1)
        self.assertEqual(summary["by_follow_on_stage"]["theorem_replanner"], 1)

    def test_self_application_is_detected_and_excluded_from_honest_success(self) -> None:
        entry = augment_result_entry(
            {
                "theorem_id": "ArithmeticFunction.sum_moebius_mul_log_eq",
                "success": True,
                "attempts": 44,
                "goals_closed": 5,
                "goals_remaining": 0,
                "close_lane": "cosine_rw",
                "close_provenance": ["cosine_rw", "self_application"],
                "final_closer": "exact ArithmeticFunction.sum_moebius_mul_log_eq",
                "tactics_used": ["rw [← foo]", "exact ArithmeticFunction.sum_moebius_mul_log_eq"],
            }
        )
        self.assertTrue(detect_self_application(entry))
        self.assertTrue(entry["self_application_detected"])
        self.assertFalse(entry["honest_success"])
        self.assertEqual(entry["success_category"], "self_application")

    def test_self_application_detection_uses_token_boundaries(self) -> None:
        entry = {
            "theorem_id": "Demo.foo",
            "success": True,
            "final_closer": "exact Demo.foobar x",
        }
        self.assertFalse(detect_self_application(entry))

    def test_self_application_detection_catches_theorem_projections(self) -> None:
        entry = augment_result_entry(
            {
                "theorem_id": "WithBot.eq_top_iff_forall_ge",
                "success": True,
                "attempts": 18,
                "goals_closed": 2,
                "goals_remaining": 0,
                "close_lane": "last_resort_exact",
                "close_provenance": ["automation", "last_resort_exact"],
                "final_closer": "exact WithBot.eq_top_iff_forall_ge.mpr a",
                "tactics_used": [
                    "aesop (add safe [WithTop.eq_top_iff_forall_ge])",
                    "exact WithBot.eq_top_iff_forall_ge.mpr a",
                ],
            }
        )
        self.assertTrue(detect_self_application(entry))
        self.assertTrue(entry["self_application_detected"])
        self.assertFalse(entry["honest_success"])
        self.assertEqual(entry["success_category"], "self_application")


if __name__ == "__main__":
    unittest.main()
