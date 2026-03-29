from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_hard_collection_bundle import build_hard_collection_bundle


class TestBuildHardCollectionBundle(unittest.TestCase):
    def test_bundle_builds_residual_and_temporal_artifacts(self) -> None:
        rows = [
            {
                "theorem_id": "Demo.solved",
                "source": "hard_theorems",
                "success": True,
                "close_lane": "automation",
                "attempts": 4,
                "goals_closed": 1,
                "goals_remaining": 0,
                "template_id": "REWRITE_CHAIN",
                "step_trace": [
                    {
                        "step": 0,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "automation",
                        "closing_family": "exact",
                        "progress": True,
                        "closed_goals_count": 1,
                        "open_goals_after": [],
                    }
                ],
            },
            {
                "theorem_id": "Demo.near",
                "source": "hard_theorems",
                "success": False,
                "close_lane": "failed",
                "attempts": 17,
                "goals_closed": 3,
                "goals_remaining": 1,
                "template_id": "DECOMPOSE_AND_CONQUER",
                "step_trace": [
                    {
                        "step": 0,
                        "goal_before": "⊢ P",
                        "open_goals_before": ["⊢ P"],
                        "lane": "automation",
                        "closing_family": "rw",
                        "progress": True,
                        "closed_goals_count": 3,
                        "open_goals_after": ["⊢ x = x"],
                    },
                    {
                        "step": 1,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "",
                        "closing_family": "",
                        "progress": False,
                        "closed_goals_count": 3,
                        "open_goals_after": ["⊢ x = x"],
                    },
                ],
            },
            {
                "theorem_id": "Demo.skipped",
                "source": "hard_theorems",
                "success": False,
                "close_lane": "skipped",
                "attempts": 0,
                "goals_closed": 0,
                "goals_remaining": 1,
                "step_trace": [],
            },
            {
                "theorem_id": "Demo.loop",
                "source": "hard_theorems",
                "success": False,
                "close_lane": "failed",
                "attempts": 44,
                "goals_closed": 1,
                "goals_remaining": 1,
                "template_id": "REWRITE_CHAIN",
                "follow_on_stage": "hard_proof_solver",
                "step_trace": [
                    {
                        "step": 0,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "cosine_rw",
                        "tactic": "rw [Demo.eq_helper]",
                        "closing_family": "rw",
                        "progress": True,
                        "closed_goals_count": 1,
                        "open_goals_after": ["⊢ x = x"],
                    },
                    {
                        "step": 1,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "",
                        "tactic": "",
                        "closing_family": "",
                        "progress": False,
                        "closed_goals_count": 1,
                        "open_goals_after": ["⊢ x = x"],
                    },
                    {
                        "step": 2,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "",
                        "tactic": "",
                        "closing_family": "",
                        "progress": False,
                        "closed_goals_count": 1,
                        "open_goals_after": ["⊢ x = x"],
                    },
                    {
                        "step": 3,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "",
                        "tactic": "",
                        "closing_family": "",
                        "progress": False,
                        "closed_goals_count": 1,
                        "open_goals_after": ["⊢ x = x"],
                    },
                    {
                        "step": 4,
                        "goal_before": "⊢ x = x",
                        "open_goals_before": ["⊢ x = x"],
                        "lane": "",
                        "tactic": "",
                        "closing_family": "",
                        "progress": False,
                        "closed_goals_count": 1,
                        "open_goals_after": ["⊢ x = x"],
                    },
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "details.jsonl"
            out_dir = tmp / "bundle"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            summary = build_hard_collection_bundle(
                inputs=[input_path],
                output_dir=out_dir,
                min_trace_length=1,
                min_strategy_support=1,
            )

            self.assertEqual(summary["total_theorems"], 4)
            self.assertEqual(summary["goal_start_failures"], 1)
            self.assertEqual(summary["hard_residuals"], 1)
            self.assertEqual(summary["last_goal_available"], 1)
            self.assertEqual(summary["dr_ducky_capsules"], 1)
            self.assertEqual(summary["dr_ducky_ledger_packets"], 1)
            self.assertGreaterEqual(summary["temporal_rows"], 2)
            self.assertGreaterEqual(summary["strategy_entries"], 1)

            hard_rows = [json.loads(line) for line in (out_dir / "hard_residuals.jsonl").open()]
            last_goal_rows = [
                json.loads(line) for line in (out_dir / "last_goal_residuals.jsonl").open()
            ]
            ducky_capsules = [
                json.loads(line) for line in (out_dir / "dr_ducky_capsules.jsonl").open()
            ]
            ducky_ledgers = [
                json.loads(line) for line in (out_dir / "dr_ducky_ledger_packets.jsonl").open()
            ]
            temporal_rows = [
                json.loads(line) for line in (out_dir / "temporal_dataset.jsonl").open()
            ]
            all_rows = [json.loads(line) for line in (out_dir / "collection_all.jsonl").open()]
            loop_row = next(row for row in all_rows if row["theorem_id"] == "Demo.loop")

        self.assertEqual(hard_rows[0]["hard_track"], "hard_proof_local")
        self.assertEqual(hard_rows[0]["last_goal"], "⊢ x = x")
        self.assertEqual(hard_rows[0]["last_goal_bucket"], "equality")
        self.assertEqual(hard_rows[0]["reasoning_gap_family"], "local_eq_close")
        self.assertEqual(last_goal_rows[0]["theorem_id"], "Demo.near")
        self.assertEqual(ducky_capsules[0]["specification"]["goal_bucket"], "equality")
        self.assertIn("allowed_engines", ducky_ledgers[0])
        self.assertIn("projector_policy", ducky_ledgers[0])
        self.assertEqual(temporal_rows[0]["theorem_id"], "Demo.solved")
        self.assertEqual(loop_row["reasoning_gap_family"], "theorem_replanner")
        self.assertEqual(loop_row["follow_on_stage"], "theorem_replanner")
        self.assertEqual(loop_row["follow_on_stage_previous"], "hard_proof_solver")


if __name__ == "__main__":
    unittest.main()
