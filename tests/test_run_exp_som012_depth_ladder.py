from __future__ import annotations

import unittest

from scripts.run_exp_som012_depth_ladder import (
    _condition_name,
    _parse_csv_ints,
    summarize_depth_condition,
)


class TestRunExpSom012DepthLadder(unittest.TestCase):
    def test_parse_csv_ints(self) -> None:
        self.assertEqual(_parse_csv_ints("128, 256,512"), [128, 256, 512])

    def test_condition_name(self) -> None:
        self.assertEqual(_condition_name(256, 8), "b256_d8")

    def test_summarize_depth_condition_tracks_input_buckets(self) -> None:
        rows = [
            {
                "success": True,
                "honest_success": True,
                "self_application_detected": False,
                "started": True,
                "attempts": 12,
                "progress_steps": 2,
                "depth_capped": False,
                "residual_bucket": "proved",
                "input_hard_track": "hard_proof_local",
                "input_reasoning_gap_family": "local_eq_close",
            },
            {
                "success": False,
                "honest_success": False,
                "self_application_detected": False,
                "started": True,
                "attempts": 40,
                "progress_steps": 4,
                "depth_capped": True,
                "residual_bucket": "single_goal_near_miss",
                "input_hard_track": "hard_proof_planner",
                "input_reasoning_gap_family": "small_multigoal_planner",
                "goals_remaining": 1,
                "goals_closed": 2,
            },
        ]
        summary = summarize_depth_condition(rows, budget=256, depth=4)
        self.assertEqual(summary["budget"], 256)
        self.assertEqual(summary["depth"], 4)
        self.assertEqual(summary["honest_success"], 1)
        self.assertEqual(summary["depth_capped_rows"], 1)
        self.assertEqual(summary["solved_by_input_hard_track"]["hard_proof_local"], 1)
        self.assertEqual(summary["by_input_reasoning_gap_family"]["small_multigoal_planner"], 1)


if __name__ == "__main__":
    unittest.main()
