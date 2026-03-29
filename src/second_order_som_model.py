from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.second_order_feature_codec import encode_packet_features


@dataclass
class SecondOrderSoMConfig:
    input_dim: int
    packet_kind_dim: int
    engine_dim: int
    backend_dim: int
    hidden_dim: int = 384
    controller_hidden_dim: int = 128
    dropout: float = 0.1
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-3
    lr_stage3: float = 1e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.05


class SecondOrderSoMNet(nn.Module):
    def __init__(self, cfg: SecondOrderSoMConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.invoke_head = nn.Linear(cfg.hidden_dim, 1)
        self.progress_head = nn.Linear(cfg.hidden_dim, 1)
        self.engine_head = nn.Linear(cfg.hidden_dim, cfg.engine_dim)
        self.backend_head = nn.Linear(cfg.hidden_dim, cfg.backend_dim)
        controller_in_dim = cfg.hidden_dim + 2 + cfg.engine_dim + cfg.backend_dim
        self.controller_mixer = nn.Sequential(
            nn.Linear(controller_in_dim, cfg.controller_hidden_dim),
            nn.LayerNorm(cfg.controller_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.projector_head = nn.Linear(cfg.controller_hidden_dim, 1)
        self.packet_kind_head = nn.Linear(cfg.controller_hidden_dim, cfg.packet_kind_dim)

    def forward_local(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "hidden": hidden,
            "invoke_logit": self.invoke_head(hidden).squeeze(-1),
            "progress_logit": self.progress_head(hidden).squeeze(-1),
            "engine_logits": self.engine_head(hidden),
            "backend_logits": self.backend_head(hidden),
        }

    def _controller_input(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        invoke = outputs["invoke_logit"].unsqueeze(-1)
        progress = outputs["progress_logit"].unsqueeze(-1)
        return torch.cat(
            [
                outputs["hidden"],
                invoke,
                progress,
                outputs["engine_logits"],
                outputs["backend_logits"],
            ],
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward_local(features)
        controller_hidden = self.controller_mixer(self._controller_input(outputs))
        outputs["projector_logit"] = self.projector_head(controller_hidden).squeeze(-1)
        outputs["packet_kind_logits"] = self.packet_kind_head(controller_hidden)
        return outputs

    def local_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.trunk, self.invoke_head, self.progress_head, self.engine_head, self.backend_head):
            params.extend(list(module.parameters()))
        return params

    def controller_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.controller_mixer, self.projector_head, self.packet_kind_head):
            params.extend(list(module.parameters()))
        return params


class LearnedSecondOrderRuntime:
    def __init__(
        self,
        *,
        model: SecondOrderSoMNet,
        metadata: dict[str, Any],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.metadata = metadata
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.device = device

    def predict_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        vector = np.array(encode_packet_features(packet, self.metadata), dtype=np.float32)
        norm = (vector - self.feature_mean) / np.clip(self.feature_std, 1e-6, None)
        tensor = torch.from_numpy(norm).to(self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
        invoke_prob = float(torch.sigmoid(outputs["invoke_logit"])[0].item())
        progress_prob = float(torch.sigmoid(outputs["progress_logit"])[0].item())
        projector_prob = float(torch.sigmoid(outputs["projector_logit"])[0].item())
        packet_kind_probs = torch.softmax(outputs["packet_kind_logits"], dim=-1)[0].cpu().numpy()
        engine_probs = torch.sigmoid(outputs["engine_logits"])[0].cpu().numpy()
        backend_probs = torch.sigmoid(outputs["backend_logits"])[0].cpu().numpy()
        packet_kind_vocab = list(self.metadata.get("packet_kind_vocab", []) or [])
        engine_vocab = list(self.metadata.get("engine_vocab", []) or [])
        backend_vocab = list(self.metadata.get("backend_vocab", []) or [])
        return {
            "invoke_prob": invoke_prob,
            "progress_prob": progress_prob,
            "projector_rejection_prob": projector_prob,
            "packet_kind_probs": {
                name: float(packet_kind_probs[idx]) for idx, name in enumerate(packet_kind_vocab)
            },
            "engine_probs": {name: float(engine_probs[idx]) for idx, name in enumerate(engine_vocab)},
            "backend_probs": {name: float(backend_probs[idx]) for idx, name in enumerate(backend_vocab)},
        }


def load_learned_second_order_runtime(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    device: str = "cpu",
) -> LearnedSecondOrderRuntime:
    metadata = json.loads(metadata_path.read_text())
    payload = torch.load(checkpoint_path, map_location=device)
    cfg_payload = dict(payload["config"])
    cfg = SecondOrderSoMConfig(**cfg_payload)
    model = SecondOrderSoMNet(cfg)
    model.load_state_dict(payload["model_state"])
    feature_mean = np.array(payload["feature_mean"], dtype=np.float32)
    feature_std = np.array(payload["feature_std"], dtype=np.float32)
    return LearnedSecondOrderRuntime(
        model=model,
        metadata=metadata,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )
