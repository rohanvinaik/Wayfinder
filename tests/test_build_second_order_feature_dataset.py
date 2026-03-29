from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_second_order_feature_dataset import build_second_order_feature_dataset


class BuildSecondOrderFeatureDatasetTests(unittest.TestCase):
    def test_builds_npz_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packets = tmp_path / "packets.jsonl"
            packets.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "packet_kind": "hard_residual",
                                "theorem_id": "T1",
                                "split": "train",
                                "difficulty_band": "hard",
                                "residual_bucket": "single_goal_near_miss",
                                "goal_bucket": "equality",
                                "resolution_family": "local_eq_close",
                                "hard_som_surface": {
                                    "residual_skeleton_geometry": {
                                        "domain_hints": ["algebra"],
                                        "representation_pressures": ["transport_alignment"],
                                        "top_symbols": ["foo"],
                                        "goal_shape_features": {"char_len": 10, "token_len": 3, "eq_count": 1},
                                        "theorem_shape_features": {"char_len": 20, "token_len": 6},
                                        "symbol_count": 1,
                                    },
                                    "proof_plan_geometry": {
                                        "candidate_methods": ["equality_solver_chain"],
                                        "specialist_targets": ["human_calculator"],
                                        "lane_suppression_hints": ["avoid_bidirectional_rw_cycles"],
                                        "lane_history": ["cosine_rw"],
                                        "lane_count": 1,
                                        "search_control_geometry": {"step_count": 5, "no_progress_ratio": 0.2},
                                    },
                                    "prior_graph_geometry": {
                                        "candidate_count": 2,
                                        "same_namespace_candidates": 1,
                                        "theorem_surface": {
                                            "accessible_premise_count": 4,
                                            "anchor_labels": ["rewriting"],
                                        },
                                    },
                                },
                                "ducky_outcome_surface": {
                                    "observed": True,
                                    "started_count": 1,
                                    "theorem_faithful_count": 1,
                                    "progressed_count": 1,
                                    "closed_count": 0,
                                    "compile_proxy_count": 2,
                                    "certificate_generation_count": 3,
                                    "projector_event_count": 3,
                                    "engine_counts": {"EqSatEngine": 2},
                                    "backend_family_counts": {"egglog_eqsat": 2},
                                },
                                "second_order_labels": {
                                    "invoke_ducky": True,
                                    "observed_progress": True,
                                    "observed_close": False,
                                    "engine_family_budget_targets": ["EqSatEngine"],
                                    "backend_budget_targets": ["egglog_eqsat"],
                                    "projector_rejection_seen": False,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "packet_kind": "compiler_specialist",
                                "theorem_id": "T2",
                                "split": "eval",
                                "difficulty_band": "",
                                "residual_bucket": "skipped_start",
                                "goal_bucket": "",
                                "resolution_family": "compiler_specialist",
                                "compiler_surface": {"failure_family": "metadata_missing"},
                                "ducky_outcome_surface": {},
                                "second_order_labels": {
                                    "invoke_ducky": False,
                                    "observed_progress": False,
                                    "observed_close": False,
                                    "engine_family_budget_targets": [],
                                    "backend_budget_targets": [],
                                    "projector_rejection_seen": False,
                                },
                            }
                        ),
                    ]
                )
                + "\n"
            )
            out_dir = tmp_path / "features"
            summary = build_second_order_feature_dataset(packets, out_dir)
            self.assertEqual(summary["train_packets"], 1)
            self.assertEqual(summary["eval_packets"], 1)
            self.assertTrue((out_dir / "train.npz").exists())
            self.assertTrue((out_dir / "eval.npz").exists())
            self.assertTrue((out_dir / "metadata.json").exists())
            train = np.load(out_dir / "train.npz", allow_pickle=True)
            self.assertEqual(train["features"].shape[0], 1)
            self.assertEqual(int(train["invoke_ducky"][0]), 1)
            self.assertEqual(int(train["observed_progress"][0]), 1)
            metadata = json.loads((out_dir / "metadata.json").read_text())
            self.assertIn("EqSatEngine", metadata["engine_vocab"])
            self.assertIn("egglog_eqsat", metadata["backend_vocab"])

    def test_reshards_when_source_split_is_degenerate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packets = tmp_path / "packets.jsonl"
            rows = []
            for i in range(8):
                rows.append(
                    {
                        "packet_kind": "hard_residual",
                        "theorem_id": f"T{i}",
                        "split": "eval",
                        "difficulty_band": "hard",
                        "residual_bucket": "single_goal_stall",
                        "goal_bucket": "forall",
                        "resolution_family": "single_goal_stall",
                        "hard_som_surface": {},
                        "ducky_outcome_surface": {},
                        "second_order_labels": {
                            "invoke_ducky": True,
                            "observed_progress": False,
                            "observed_close": False,
                            "engine_family_budget_targets": [],
                            "backend_budget_targets": [],
                            "projector_rejection_seen": False,
                        },
                    }
                )
            packets.write_text("".join(json.dumps(row) + "\n" for row in rows))
            out_dir = tmp_path / "features"
            summary = build_second_order_feature_dataset(packets, out_dir)
            self.assertEqual(summary["split_strategy"], "deterministic_theorem_reshard")
            self.assertGreater(summary["train_packets"], 0)
            self.assertGreater(summary["eval_packets"], 0)


if __name__ == "__main__":
    unittest.main()
