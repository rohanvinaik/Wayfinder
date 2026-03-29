from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_hard_proof_dataset import build_hard_proof_dataset


class TestBuildHardProofDataset(unittest.TestCase):
    def test_build_hard_proof_dataset_splits_and_reattaches_last_goal(self) -> None:
        hard_rows = [
            {
                "theorem_id": "Demo.local",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 3,
                "goals_remaining": 1,
            },
            {
                "theorem_id": "Demo.planner",
                "residual_bucket": "multi_goal_small_progress",
                "goals_closed": 2,
                "goals_remaining": 3,
            },
        ]
        residual_rows = [
            {
                "theorem_id": "Demo.local",
                "residual_type": "single_goal_stall",
                "remaining_goals": 1,
                "attempts": 19,
                "last_goal": "⊢ x = x",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            hard_path = tmp / "hard.jsonl"
            residual_path = tmp / "residual.jsonl"
            out_dir = tmp / "out"
            hard_path.write_text("".join(json.dumps(row) + "\n" for row in hard_rows))
            residual_path.write_text("".join(json.dumps(row) + "\n" for row in residual_rows))

            summary = build_hard_proof_dataset(
                hard_path,
                out_dir,
                residual_source_paths=[residual_path],
                condition="norm_then_close",
            )

            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["last_goal_available"], 1)
            self.assertEqual(summary["by_hard_track"]["hard_proof_local"], 1)
            self.assertEqual(summary["by_hard_track"]["hard_proof_planner"], 1)

            local_rows = [json.loads(line) for line in (out_dir / "hard_proof_local.jsonl").open()]
            planner_rows = [json.loads(line) for line in (out_dir / "hard_proof_planner.jsonl").open()]
            last_goal_rows = [json.loads(line) for line in (out_dir / "last_goal_residuals.jsonl").open()]

        self.assertEqual(local_rows[0]["hard_track"], "hard_proof_local")
        self.assertEqual(local_rows[0]["last_goal"], "⊢ x = x")
        self.assertEqual(planner_rows[0]["hard_track"], "hard_proof_planner")
        self.assertEqual(last_goal_rows[0]["theorem_id"], "Demo.local")


if __name__ == "__main__":
    unittest.main()
