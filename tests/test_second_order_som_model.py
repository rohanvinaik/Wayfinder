from __future__ import annotations

import json
from pathlib import Path

import torch

from src.second_order_som_model import (
    LearnedSecondOrderRuntime,
    SecondOrderSoMConfig,
    SecondOrderSoMNet,
    load_learned_second_order_runtime,
)


def test_load_learned_second_order_runtime_predicts_named_scores(tmp_path: Path) -> None:
    metadata = {
        "token_vocab": ["packet_kind:hard_residual", "goal_bucket:equality"],
        "numeric_feature_names": ["goal_char_len", "goal_eq_count"],
        "packet_kind_vocab": ["compiler_specialist", "hard_residual"],
        "engine_vocab": ["EqSatEngine", "ContextTransportEngine"],
        "backend_vocab": ["egglog_eqsat", "rosette_proof_dsl"],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata) + "\n")
    cfg = SecondOrderSoMConfig(input_dim=4, packet_kind_dim=2, engine_dim=2, backend_dim=2, hidden_dim=8)
    model = SecondOrderSoMNet(cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "feature_mean": [0.0, 0.0, 0.0, 0.0],
            "feature_std": [1.0, 1.0, 1.0, 1.0],
        },
        checkpoint,
    )
    runtime = load_learned_second_order_runtime(checkpoint, metadata_path, device="cpu")
    assert isinstance(runtime, LearnedSecondOrderRuntime)
    packet = {
        "packet_kind": "hard_residual",
        "goal_bucket": "equality",
        "hard_som_surface": {"residual_skeleton_geometry": {"goal_shape_features": {"char_len": 10, "eq_count": 1}}},
    }
    prediction = runtime.predict_packet(packet)
    assert "invoke_prob" in prediction
    assert set(prediction["engine_probs"]) == {"ContextTransportEngine", "EqSatEngine"}
    assert set(prediction["backend_probs"]) == {"egglog_eqsat", "rosette_proof_dsl"}
