from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hard_run import build_hard_run_audit


class TestAuditHardRun(unittest.TestCase):
    def test_build_hard_run_audit_summarizes_run_artifacts(self) -> None:
        details = [
            {
                "theorem_id": "Demo.proved",
                "success": True,
                "attempts": 3,
                "goals_closed": 1,
                "goals_remaining": 0,
                "close_lane": "automation",
                "lane_sequence": "automation",
                "proof_steps": 5,
                "final_closer": "aesop",
            },
            {
                "theorem_id": "Demo.self",
                "success": True,
                "attempts": 12,
                "goals_closed": 2,
                "goals_remaining": 0,
                "close_lane": "last_resort_exact",
                "lane_sequence": "automation→last_resort_exact",
                "proof_steps": 6,
                "final_closer": "exact Demo.self.mpr h",
            },
            {
                "theorem_id": "Demo.partial",
                "success": False,
                "attempts": 44,
                "goals_closed": 5,
                "goals_remaining": 2,
                "progress_band": "partial_progress",
                "residual_bucket": "multi_goal_small_progress",
                "reasoning_gap_family": "theorem_replanner",
                "last_goal_bucket": "equality",
                "last_goal": "x = y",
                "search_pathology_tags": ["duplicate_goal_progress", "blank_lane_plateau"],
            },
        ]
        start_failures = [
            {
                "theorem_id": "Demo.start",
                "failure_category": "goal_creation_fail",
                "start_failure_family": "metadata_missing",
                "context_unsupported_kinds": ["open", "variable"],
                "module": "Mathlib.Demo",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "details.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
            (run_dir / "goal_start_failures.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in start_failures)
            )

            report = build_hard_run_audit(run_dir)

        self.assertEqual(report["processed"], 3)
        self.assertEqual(report["raw_success"], 2)
        self.assertEqual(report["honest_success"], 1)
        self.assertEqual(report["self_application_successes"], 1)
        self.assertEqual(report["partial_progress"], 1)
        self.assertEqual(report["counts"]["reasoning_gap_families"]["theorem_replanner"], 1)
        self.assertEqual(report["start_failures"]["families"]["metadata_missing"], 1)
        self.assertEqual(report["start_failures"]["unsupported_context_kinds"]["open"], 1)
        self.assertEqual(report["likely_self_replay_rows"][0]["theorem_id"], "Demo.self")
        self.assertEqual(report["best_partials"][0]["theorem_id"], "Demo.partial")


if __name__ == "__main__":
    unittest.main()
