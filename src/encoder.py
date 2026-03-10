"""
Goal-state encoder — produces semantic embeddings from Lean goal state text.

Supports two backends:
  - SentenceTransformer models (e.g. all-MiniLM-L6-v2, 384-dim)
  - HuggingFace T5/ByT5 encoder models (e.g. google/byt5-small, 1472-dim)

Backend is auto-detected from model_name. T5/ByT5 models use encoder-only
with mean pooling over non-pad tokens.

If output_dim is specified and differs from the model's native dimension,
a learned linear projection is added to normalize the output.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _is_t5_model(model_name: str) -> bool:
    """Check if model_name refers to a T5-family model."""
    lower = model_name.lower()
    return any(k in lower for k in ("t5", "byt5", "mt5", "flan-t5"))


class GoalEncoder(nn.Module):
    """Encoder for goal state embeddings.

    Supports SentenceTransformer and HuggingFace T5/ByT5 backends.
    Optionally projects to a fixed output_dim for downstream compatibility.

    Args:
        model_name: HuggingFace model identifier or SentenceTransformer name.
        output_dim: Desired output dimension. If None, uses model's native dim.
            If specified and differs from native dim, adds a learned projection.
        frozen: If True, freeze encoder weights (no gradient flow).
        device: Target device ("mps", "cpu").
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        output_dim: int | None = None,
        frozen: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self._output_dim = output_dim
        self.frozen = frozen
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_t5 = _is_t5_model(model_name)
        self._native_dim: int | None = None
        self._projection: nn.Linear | None = None

    @property
    def output_dim(self) -> int:
        """Return the output embedding dimension."""
        if self._output_dim is not None:
            return self._output_dim
        if self._native_dim is not None:
            return self._native_dim
        # Fallback before model is loaded
        return 384

    def _load_model(self) -> None:
        if self._is_t5:
            self._load_t5()
        else:
            self._load_sentence_transformer()

        # Add projection if output_dim differs from native dim
        if (
            self._output_dim is not None
            and self._native_dim is not None
            and self._output_dim != self._native_dim
        ):
            self._projection = nn.Linear(self._native_dim, self._output_dim).to(self.device)

    def _load_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        # Detect native dimension from the model
        self._native_dim = self._model.get_sentence_embedding_dimension()
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _load_t5(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # nosec B615 — model name from config, not user input
        self._model = AutoModel.from_pretrained(self.model_name).encoder.to(self.device)  # nosec B615
        self._native_dim = self._model.config.d_model
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _encode_t5(self, goal_states: list[str]) -> torch.Tensor:
        """Encode using T5/ByT5 encoder with mean pooling."""
        if self._tokenizer is None or self._model is None:
            msg = "Encoder not initialized — call _init_t5() first"
            raise RuntimeError(msg)

        tokens = self._tokenizer(
            goal_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        if self.frozen:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
        else:
            outputs = self._model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        # Mean pool over non-pad tokens
        hidden = outputs.last_hidden_state  # [batch, seq, dim]
        mask = tokens["attention_mask"].unsqueeze(-1).float()  # [batch, seq, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def _encode_st(self, goal_states: list[str]) -> torch.Tensor:
        """Encode using SentenceTransformer."""
        if self._model is None:
            msg = "Encoder not initialized — call _init_st() first"
            raise RuntimeError(msg)
        with torch.no_grad():
            embeddings = self._model.encode(goal_states, convert_to_tensor=True)
        return embeddings.to(self.device)

    def encode(self, goal_states: list[str]) -> torch.Tensor:
        """Encode goal states to [batch, output_dim] embeddings.

        Returns:
            Float32 tensor on self.device, shape [len(goal_states), output_dim].
        """
        if self._model is None:
            self._load_model()

        if self._is_t5:
            embeddings = self._encode_t5(goal_states)
        else:
            embeddings = self._encode_st(goal_states)

        if self._projection is not None:
            embeddings = self._projection(embeddings)

        return embeddings

    def forward(self, goal_states: list[str]) -> torch.Tensor:
        return self.encode(goal_states)
