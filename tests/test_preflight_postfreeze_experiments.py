from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.preflight_postfreeze_experiments import _run_preflight


class TestPreflightPostfreezeExperiments(unittest.TestCase):
    def test_dd014_preflight_passes_when_required_bundle_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "exp_som012_hard_eval_r2"
            (run_dir / "bundle" / "dr_ducky").mkdir(parents=True)
            (run_dir / "bundle" / "hard_resolution_layer").mkdir(parents=True)
            for rel in (
                "details.jsonl",
                "bundle/dr_ducky/summary.json",
                "bundle/dr_ducky/dr_ducky_capsules.jsonl",
                "bundle/dr_ducky/dr_ducky_ledger_packets.jsonl",
                "bundle/hard_resolution_layer/resolution_packets.jsonl",
            ):
                (run_dir / rel).write_text("{}\n")
            payload = _run_preflight(run_dir, "dd014a")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["missing"])

    def test_som013a_preflight_reports_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "exp_som012_hard_eval_r2"
            (run_dir / "bundle").mkdir(parents=True)
            payload = _run_preflight(run_dir, "som013a")
        self.assertFalse(payload["ok"])
        self.assertGreaterEqual(len(payload["missing"]), 1)

    def test_dd015_preflight_passes_when_bridge_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "exp_som012_hard_eval_r2"
            (run_dir / "bundle" / "dr_ducky").mkdir(parents=True)
            (run_dir / "bundle" / "hard_resolution_layer").mkdir(parents=True)
            (run_dir / "bundle" / "second_order_som").mkdir(parents=True)
            for rel in (
                "details.jsonl",
                "bundle/dr_ducky/summary.json",
                "bundle/dr_ducky/dr_ducky_ledger_packets.jsonl",
                "bundle/hard_resolution_layer/hard_som_packets.jsonl",
                "bundle/second_order_som/second_order_packets.jsonl",
            ):
                (run_dir / rel).write_text("{}\n")
            payload = _run_preflight(run_dir, "dd015")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["missing"])

    def test_som013d_preflight_passes_when_feature_bundle_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "exp_som012_hard_eval_r2"
            (run_dir / "bundle" / "second_order_som" / "features").mkdir(parents=True)
            for rel in (
                "bundle/second_order_som/features/train.npz",
                "bundle/second_order_som/features/eval.npz",
                "bundle/second_order_som/features/metadata.json",
            ):
                (run_dir / rel).write_text("{}\n")
            payload = _run_preflight(run_dir, "som013d")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["missing"])


if __name__ == "__main__":
    unittest.main()
