from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_dr_ducky_worklist import build_dr_ducky_worklist


class TestBuildDrDuckyWorklist(unittest.TestCase):
    def test_build_dr_ducky_worklist_filters_near_miss_rows(self) -> None:
        details = [
            {
                "theorem_id": "Demo.near",
                "success": False,
                "attempts": 18,
                "goals_closed": 4,
                "goals_remaining": 1,
                "residual_bucket": "single_goal_near_miss",
                "last_goal": "x = y",
                "last_goal_bucket": "equality",
                "reasoning_gap_family": "local_eq_close",
                "difficulty_band": "hard",
            },
            {
                "theorem_id": "Demo.other",
                "success": False,
                "attempts": 44,
                "goals_closed": 2,
                "goals_remaining": 3,
                "residual_bucket": "multi_goal_small_progress",
                "last_goal": "∃ z, z = z",
                "last_goal_bucket": "exists",
                "reasoning_gap_family": "small_multigoal_planner",
                "difficulty_band": "expert",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output_dir = Path(tmpdir) / "out"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "details.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))

            summary = build_dr_ducky_worklist(
                run_dir=run_dir,
                output_dir=output_dir,
                residual_buckets={"single_goal_near_miss"},
            )

            capsules = (output_dir / "dr_ducky_capsules.jsonl").read_text().strip().splitlines()
            ledgers = (output_dir / "dr_ducky_ledger_packets.jsonl").read_text().strip().splitlines()

        self.assertEqual(summary["selected_rows"], 1)
        self.assertEqual(summary["total_capsules"], 1)
        self.assertEqual(summary["ledger_packets"], 1)
        self.assertEqual(len(capsules), 1)
        self.assertEqual(len(ledgers), 1)
        payload = json.loads(capsules[0])
        self.assertEqual(payload["specification"]["theorem_id"], "Demo.near")
        ledger_payload = json.loads(ledgers[0])
        self.assertIn("allowed_engines", ledger_payload)
        self.assertIn("projector_policy", ledger_payload)


if __name__ == "__main__":
    unittest.main()
