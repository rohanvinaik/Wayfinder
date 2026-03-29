from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_second_order_packet_freeze import build_second_order_packet_freeze


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestBuildSecondOrderPacketFreeze(unittest.TestCase):
    def test_builds_merged_packet_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            ducky_dir = run_dir / "bundle" / "dr_ducky"
            res_dir = run_dir / "bundle" / "hard_resolution_layer"

            _write_jsonl(
                res_dir / "hard_som_packets.jsonl",
                [
                    {
                        "theorem_id": "Demo.hard",
                        "split": "eval",
                        "difficulty_band": "hard",
                        "residual_bucket": "single_goal_near_miss",
                        "goal_bucket": "equality",
                        "resolution_family": "local_eq_close",
                    }
                ],
            )
            _write_jsonl(
                res_dir / "compiler_specialist_packets.jsonl",
                [
                    {
                        "theorem_id": "Demo.compiler",
                        "split": "eval",
                        "startability_surface": {"start_failure_family": "metadata_missing"},
                    }
                ],
            )
            _write_jsonl(
                ducky_dir / "dr_ducky_ledger_packets.jsonl",
                [
                    {
                        "theorem_id": "Demo.hard",
                        "residual_bucket": "single_goal_near_miss",
                        "goal_bucket": "equality",
                        "allowed_engines": ["EqSatEngine", "ContextTransportEngine"],
                    }
                ],
            )
            _write_jsonl(
                ducky_dir / "executor_validation_stratified120_rows.jsonl",
                [
                    {
                        "theorem_id": "Demo.hard",
                        "residual_bucket": "single_goal_near_miss",
                        "goal_bucket": "equality",
                        "started": True,
                        "theorem_faithful": True,
                        "progressed": True,
                        "closed": False,
                        "tried_programs": [{"tactics_applied": ["norm_num"]}],
                    }
                ],
            )
            _write_jsonl(
                ducky_dir / "executor_validation_stratified120_engine_outcomes.jsonl",
                [
                    {
                        "theorem_id": "Demo.hard",
                        "engine_name": "EqSatEngine",
                        "backend_family": "egglog_eqsat",
                        "certificate_shape": "rewrite_chain",
                    }
                ],
            )
            _write_jsonl(
                ducky_dir / "executor_validation_stratified120_projector_outcomes.jsonl",
                [
                    {
                        "theorem_id": "Demo.hard",
                        "projector_status": "projected",
                    }
                ],
            )
            (ducky_dir / "executor_validation_stratified120_closure_report.json").write_text(
                json.dumps({"honest_progress_count": 1, "honest_closure_count": 0})
            )

            summary = build_second_order_packet_freeze(run_dir, run_dir / "bundle" / "second_order_som")

            self.assertEqual(summary["hard_residual_packets"], 1)
            self.assertEqual(summary["compiler_packets"], 1)
            self.assertEqual(summary["ducky_observed_packets"], 1)
            self.assertEqual(summary["ducky_progress_packets"], 1)

            packet_rows = [json.loads(line) for line in (run_dir / "bundle" / "second_order_som" / "second_order_packets.jsonl").read_text().splitlines()]
            self.assertEqual(len(packet_rows), 2)
            hard_packet = next(row for row in packet_rows if row["packet_kind"] == "hard_residual")
            self.assertTrue(hard_packet["second_order_labels"]["invoke_ducky"])
            self.assertTrue(hard_packet["second_order_labels"]["observed_progress"])
            self.assertIn("EqSatEngine", hard_packet["ducky_outcome_surface"]["engine_counts"])


if __name__ == "__main__":
    unittest.main()
