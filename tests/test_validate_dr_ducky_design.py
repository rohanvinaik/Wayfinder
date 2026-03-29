from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_dr_ducky_design import build_validation_summary


class TestValidateDrDuckyDesign(unittest.TestCase):
    def test_build_validation_summary_detects_recursive_and_symbolic_modes(self) -> None:
        rows = [
            {
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
                ],
                "step_trace": [
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
            },
            {
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
            },
            {
                "theorem_id": "ModularGroup.eq_zero_of_mem_fdo_of_T_zpow_mem_fdo",
                "last_goal": "|n| < 1",
                "last_goal_bucket": "inequality",
                "reasoning_gap_family": "local_ineq_close",
                "residual_bucket": "single_goal_near_miss",
                "difficulty_band": "hard",
                "goals_closed": 2,
                "goals_remaining": 1,
                "attempts": 40,
                "lane_sequence": "interleaved_bootstrap→cosine_rw",
                "search_pathology_tags": [],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            details_path = run_dir / "details.jsonl"
            with details_path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = build_validation_summary(run_dir)

        self.assertEqual(summary["input_rows"], 3)
        self.assertEqual(summary["validated_rows"], 3)
        self.assertEqual(summary["recursive_circuit_breaker"]["count"], 1)
        self.assertEqual(summary["recursive_circuit_breaker"]["routed"], 1)
        self.assertGreaterEqual(summary["symbolic_sandbox"]["count"], 2)
        self.assertEqual(summary["family_alignment"]["local_ineq_close"]["count"], 1)
        self.assertEqual(summary["family_alignment"]["local_ineq_close"]["aligned"], 1)
        self.assertIn("Batteries.UnionFind.rootD_parent", summary["targeted_cases"])
        self.assertIn("ModularGroup.smul_eq_lcRow0_add", summary["targeted_cases"])


if __name__ == "__main__":
    unittest.main()
