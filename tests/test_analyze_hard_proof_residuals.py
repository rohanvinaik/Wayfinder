from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_hard_proof_residuals import analyze_hard_proof_residuals


class TestAnalyzeHardProofResiduals(unittest.TestCase):
    def test_analyze_hard_proof_residuals_counts_geometry_and_templates(self) -> None:
        hard_rows = [
            {
                "theorem_id": "Demo.eq",
                "hard_track": "hard_proof_local",
                "residual_bucket": "single_goal_near_miss",
                "last_goal": "x = y",
                "last_goal_available": True,
                "lane_sequence": "automation→cosine_rw",
                "tactics_used": ["aesop", "rw [foo]"],
            },
            {
                "theorem_id": "Demo.iff",
                "hard_track": "hard_proof_planner",
                "residual_bucket": "multi_goal_small_progress",
                "last_goal": "P ↔ Q",
                "last_goal_available": True,
                "lane_sequence": "interleaved_bootstrap",
                "tactics_used": ["norm_num"],
            },
        ]
        narrative_rows = [
            {
                "theorem_id": "Demo.eq",
                "template_id": "REWRITE_CHAIN",
                "proof_history_summary": {"total_steps": 5, "unique_premises": 2},
            },
            {
                "theorem_id": "Demo.iff",
                "template_id": "DECOMPOSE_AND_CONQUER",
                "proof_history_summary": {"total_steps": 7, "unique_premises": 3},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            hard_path = tmp / "hard.jsonl"
            narrative_path = tmp / "narratives.jsonl"
            hard_path.write_text("".join(json.dumps(row) + "\n" for row in hard_rows))
            narrative_path.write_text("".join(json.dumps(row) + "\n" for row in narrative_rows))

            summary = analyze_hard_proof_residuals([hard_path], narrative_path=narrative_path)

        self.assertEqual(summary["total_theorems"], 2)
        self.assertEqual(summary["last_goal_available"], 2)
        self.assertEqual(summary["by_primary_goal_geometry"]["equality"], 1)
        self.assertEqual(summary["by_primary_goal_geometry"]["iff"], 1)
        self.assertEqual(summary["by_template_id"]["REWRITE_CHAIN"], 1)
        self.assertEqual(summary["by_tactic_prefix"]["aesop"], 1)


if __name__ == "__main__":
    unittest.main()
