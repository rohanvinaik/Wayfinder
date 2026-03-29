from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_benchmark_residuals import build_residual_report


class TestAnalyzeBenchmarkResiduals(unittest.TestCase):
    def test_build_residual_report_reattaches_goal_metadata(self) -> None:
        benchmark_report = {
            "benchmark": {
                "total_theorems": 3,
                "raw_success": 1,
            },
            "details": [
                {
                    "theorem_id": "Demo.proved",
                    "source": "benchmark_theorems",
                    "success": True,
                    "attempts": 2,
                    "goals_closed": 1,
                    "goals_remaining": 0,
                    "close_lane": "automation",
                },
                {
                    "theorem_id": "Demo.skip",
                    "source": "benchmark_theorems",
                    "success": False,
                    "attempts": 0,
                    "goals_closed": 0,
                    "goals_remaining": 1,
                    "close_lane": "skipped",
                },
                {
                    "theorem_id": "Demo.near",
                    "source": "benchmark_theorems",
                    "success": False,
                    "attempts": 11,
                    "goals_closed": 2,
                    "goals_remaining": 1,
                    "close_lane": "failed",
                },
            ],
        }
        theorem_rows = [
            {"theorem_id": "Demo.proved", "goal_state": "⊢ True", "file_path": "Mathlib/Demo.lean"},
            {"theorem_id": "Demo.skip", "goal_state": "⊢ False", "file_path": "Mathlib/Demo.lean"},
            {"theorem_id": "Demo.near", "goal_state": "⊢ x = x", "file_path": "Mathlib/Demo.lean"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "benchmark.json"
            theorem_path = tmp / "theorems.jsonl"
            report_path.write_text(json.dumps(benchmark_report))
            theorem_path.write_text("".join(json.dumps(row) + "\n" for row in theorem_rows))

            report = build_residual_report(report_path, [theorem_path])

        self.assertEqual(report["raw_success"], 1)
        self.assertEqual(report["residual_structure"]["skipped_start"], 1)
        self.assertEqual(report["residual_structure"]["one_goal_left_failures"], 2)
        self.assertEqual(report["example_theorems"]["by_follow_on_stage"]["compiler_specialist"], ["Demo.skip"])

        by_id = {entry["theorem_id"]: entry for entry in report["details"]}
        self.assertEqual(by_id["Demo.near"]["goal_state"], "⊢ x = x")
        self.assertEqual(by_id["Demo.near"]["follow_on_stage"], "hard_proof_solver")
        self.assertEqual(by_id["Demo.skip"]["follow_on_stage"], "compiler_specialist")

    def test_build_residual_report_counts_self_application_separately(self) -> None:
        benchmark_report = {
            "benchmark": {
                "total_theorems": 1,
                "raw_success": 1,
            },
            "details": [
                {
                    "theorem_id": "Demo.self",
                    "source": "benchmark_theorems",
                    "success": True,
                    "attempts": 12,
                    "goals_closed": 2,
                    "goals_remaining": 0,
                    "close_lane": "last_resort_exact",
                    "final_closer": "exact Demo.self x",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "benchmark.json"
            report_path.write_text(json.dumps(benchmark_report))
            report = build_residual_report(report_path, [])

        self.assertEqual(report["raw_success"], 1)
        self.assertEqual(report["honest_success"], 0)
        self.assertEqual(report["self_application_successes"], 1)
        self.assertEqual(report["details"][0]["success_category"], "self_application")


if __name__ == "__main__":
    unittest.main()
