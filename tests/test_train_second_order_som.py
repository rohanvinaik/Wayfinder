from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train_second_order_som import train_second_order_som


def test_train_second_order_som_smoke(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)
    np.savez(
        feature_dir / "train.npz",
        features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        invoke_ducky=np.array([1, 0], dtype=np.int64),
        observed_progress=np.array([1, 0], dtype=np.int64),
        observed_close=np.array([0, 0], dtype=np.int64),
        projector_rejection_seen=np.array([0, 1], dtype=np.int64),
        packet_kind=np.array([1, 0], dtype=np.int64),
        engine_targets=np.array([[1, 0], [0, 1]], dtype=np.int64),
        backend_targets=np.array([[1, 0], [0, 1]], dtype=np.int64),
        theorem_ids=np.array(["T1", "T2"], dtype=object),
    )
    np.savez(
        feature_dir / "eval.npz",
        features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        invoke_ducky=np.array([1, 0], dtype=np.int64),
        observed_progress=np.array([1, 0], dtype=np.int64),
        observed_close=np.array([0, 0], dtype=np.int64),
        projector_rejection_seen=np.array([0, 1], dtype=np.int64),
        packet_kind=np.array([1, 0], dtype=np.int64),
        engine_targets=np.array([[1, 0], [0, 1]], dtype=np.int64),
        backend_targets=np.array([[1, 0], [0, 1]], dtype=np.int64),
        theorem_ids=np.array(["T1", "T2"], dtype=object),
    )
    metadata = {
        "token_vocab": ["packet_kind:hard_residual"],
        "numeric_feature_names": ["goal_char_len"],
        "packet_kind_vocab": ["compiler_specialist", "hard_residual"],
        "engine_vocab": ["EqSatEngine", "ContextTransportEngine"],
        "backend_vocab": ["egglog_eqsat", "rosette_proof_dsl"],
    }
    (feature_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
    output_dir = tmp_path / "trained"
    summary = train_second_order_som(
        feature_dir,
        output_dir,
        epochs=2,
        batch_size=2,
        hidden_dim=8,
        device="cpu",
        stage1_epochs=1,
        stage2_epochs=1,
        stage3_epochs=1,
    )
    assert summary["train_rows"] == 2
    assert summary["eval_rows"] == 2
    assert summary["stage_epochs"] == {"stage1": 1, "stage2": 1, "stage3": 1}
    assert "stage1" in summary["best_metrics"]
    assert "final_eval" in summary
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "summary.json").exists()
