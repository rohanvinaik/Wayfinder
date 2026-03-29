from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dr_ducky_executor_validation import (
    _failure_payload,
    _load_rows,
    _load_rows_with_filters,
    _write_progress_reports,
)


class TestRunDrDuckyExecutorValidation(unittest.TestCase):
    def test_failure_payload_preserves_ducky_surface(self) -> None:
        row = {
            "theorem_id": "Demo.timeout",
            "last_goal": "n = 0",
            "last_goal_bucket": "equality",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
        }
        payload = _failure_payload(row, failure_category="row_timeout", error_message="row exceeded timeout")
        self.assertEqual(payload["theorem_id"], "Demo.timeout")
        self.assertFalse(payload["started"])
        self.assertEqual(payload["replay_failure_category"], "row_timeout")
        self.assertIn("bank_priors", payload)
        self.assertIsNotNone(payload["ledger_snapshot"])

    def test_progress_reports_write_incremental_summary(self) -> None:
        row = {
            "theorem_id": "Demo.progress",
            "last_goal": "n = 0",
            "last_goal_bucket": "equality",
            "residual_bucket": "single_goal_near_miss",
            "goals_closed": 1,
            "goals_remaining": 1,
        }
        result = {
            "theorem_id": "Demo.progress",
            "started": True,
            "theorem_faithful": True,
            "start_goal_kind": "direct_goal",
            "file_path": "",
            "replay_tier": "live",
            "replay_failure_category": "",
            "replay_failing_prefix_idx": -1,
            "residual_bucket": "single_goal_near_miss",
            "goal_bucket": "equality",
            "specialist_targets": ["symbolic_sandbox"],
            "bank_priors": ["eq_sat"],
            "programs_considered": 2,
            "closed": False,
            "progressed": True,
            "winning_program": None,
            "final_goal": "n = 0",
            "final_goal_bucket": "equality",
            "goals_after": ["n = 0"],
            "ledger_snapshot": {},
            "engine_outcomes": [{"engine_name": "EqSatEngine", "certificate_count": 1, "backend_family": "egglog_eqsat"}],
            "projector_outcomes": [{"projector_status": "projected"}],
            "tried_programs": [{"tactics_applied": ["norm_num"], "certificate_shape": "normal_form"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "summary.json"
            closure_json = Path(tmpdir) / "closure.json"
            _write_progress_reports(
                results=[result],
                rows=[row],
                residual_buckets={"single_goal_near_miss"},
                goal_buckets={"equality"},
                allowed_backend_families={"egglog_eqsat"},
                allowed_engine_names={"EqSatEngine"},
                output_json=output_json,
                closure_report_json=closure_json,
                projector_rows=[{"projector_status": "projected"}],
            )
            summary = json.loads(output_json.read_text())
            closure = json.loads(closure_json.read_text())
        self.assertEqual(summary["input_rows"], 1)
        self.assertEqual(summary["completed_rows"], 1)
        self.assertEqual(summary["goal_buckets"], ["equality"])
        self.assertEqual(summary["allowed_backend_families"], ["egglog_eqsat"])
        self.assertEqual(closure["honest_progress_count"], 1)
        self.assertEqual(closure["projector_success_count"], 1)

    def test_load_rows_round_robins_across_buckets(self) -> None:
        rows = [
            {
                "theorem_id": "A.eq1",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 2,
                "goals_remaining": 1,
                "attempts": 10,
            },
            {
                "theorem_id": "A.eq2",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 11,
            },
            {
                "theorem_id": "B.multi1",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 4,
                "goals_remaining": 2,
                "attempts": 20,
            },
            {
                "theorem_id": "B.multi2",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 3,
                "goals_remaining": 2,
                "attempts": 21,
            },
            {
                "theorem_id": "C.stall1",
                "residual_bucket": "single_goal_stall",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 12,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            loaded = _load_rows(
                path,
                {"single_goal_near_miss", "multi_goal_large_progress", "single_goal_stall"},
                5,
            )
        self.assertEqual(
            [row["residual_bucket"] for row in loaded],
            [
                "multi_goal_large_progress",
                "single_goal_near_miss",
                "single_goal_stall",
                "multi_goal_large_progress",
                "single_goal_near_miss",
            ],
        )

    def test_load_rows_respects_limit_even_with_targeted_rows(self) -> None:
        rows = [
            {
                "theorem_id": "Batteries.UnionFind.rootD_parent",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 3,
                "goals_remaining": 1,
                "attempts": 10,
            },
            {
                "theorem_id": "ModularGroup.tendsto_normSq_coprime_pair",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 3,
                "goals_remaining": 1,
                "attempts": 11,
            },
            {
                "theorem_id": "ModularGroup.smul_eq_lcRow0_add",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 3,
                "goals_remaining": 1,
                "attempts": 12,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            loaded = _load_rows(path, {"multi_goal_large_progress"}, 2)
        self.assertEqual(len(loaded), 2)

    def test_load_rows_filters_goal_bucket(self) -> None:
        rows = [
            {
                "theorem_id": "Demo.eq",
                "residual_bucket": "single_goal_near_miss",
                "last_goal_bucket": "equality",
                "goals_closed": 2,
                "goals_remaining": 1,
                "attempts": 3,
            },
            {
                "theorem_id": "Demo.mem",
                "residual_bucket": "single_goal_near_miss",
                "last_goal_bucket": "membership",
                "goals_closed": 2,
                "goals_remaining": 1,
                "attempts": 4,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            loaded = _load_rows_with_filters(
                path,
                residual_buckets={"single_goal_near_miss"},
                goal_buckets={"membership"},
                limit=10,
            )
        self.assertEqual([row["theorem_id"] for row in loaded], ["Demo.mem"])


if __name__ == "__main__":
    unittest.main()
