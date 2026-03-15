"""
Goal-state encoder — produces semantic embeddings from Lean goal state text.

Supports four backends, auto-detected from local config files when available:
  - SentenceTransformer models
  - HuggingFace T5/ByT5 encoder models
  - Decoder-only models with last-token pooling
  - PEFT/LoRA adapters on decoder bases

The encoder owns backend-specific handling so training/eval scripts only need
to specify an encoder name/path in config. Device movement, instruction-aware
prefixes, chunking, tokenizer padding, dtype choice, and adapter loading all
live here instead of being re-implemented per script.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Encoder config utilities — extracted to src/encoder_config.py
from src.encoder_config import (  # noqa: F401
    _DEFAULT_MODEL_NAME,
    _apply_instruction_prefix,
    _detect_backend,
    _ensure_pad_token,
    _preferred_encode_batch_size,
    _preferred_torch_dtype,
    _resolve_base_model_name,
    _should_trust_remote_code,
    get_encoder_model_name,
)


class GoalEncoder(nn.Module):
    """Encoder for goal state embeddings.

    Supports SentenceTransformer, T5/ByT5, decoder-only, and PEFT backends.
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
        model_name: str = _DEFAULT_MODEL_NAME,
        output_dim: int | None = None,
        frozen: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_model_name = _resolve_base_model_name(model_name)
        self._output_dim = output_dim
        self.frozen = frozen
        self.device = str(torch.device(device))
        self._model: Any = None
        self._tokenizer: Any = None
        self._backend = _detect_backend(model_name)
        self._is_t5 = self._backend == "t5"
        self._native_dim: int | None = None
        self._projection: nn.Linear | None = None
        self._encode_batch_size = _preferred_encode_batch_size(self._backend, self.device)

    @classmethod
    def from_config(cls, config: dict[str, Any], device: str = "cpu") -> GoalEncoder:
        """Build an encoder directly from a config section."""
        return cls(
            model_name=get_encoder_model_name(config),
            output_dim=config.get("output_dim"),
            frozen=config.get("frozen", True),
            device=device,
        )

    @property
    def output_dim(self) -> int:
        """Return the output embedding dimension."""
        if self._output_dim is not None:
            return self._output_dim
        if self._native_dim is not None:
            return self._native_dim
        # Fallback before model is loaded
        return 384

    @property
    def backend(self) -> str:
        """Return the detected backend family."""
        return self._backend

    @property
    def native_dim(self) -> int | None:
        """Return the native embedding dimension once the model is loaded."""
        return self._native_dim

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary for logs/debugging."""
        return {
            "model_name": self.model_name,
            "base_model_name": self.base_model_name,
            "backend": self._backend,
            "native_dim": self._native_dim,
            "output_dim": self.output_dim,
            "device": self.device,
            "frozen": self.frozen,
        }

    def _load_model(self) -> None:
        if self._backend == "peft":
            self._load_peft()
        elif self._backend == "t5":
            self._load_t5()
        elif self._backend == "decoder":
            self._load_decoder()
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

        trust_remote_code = _should_trust_remote_code(self.model_name)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=trust_remote_code,
            )
        except TypeError:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        # Detect native dimension from the model
        self._native_dim = self._model.get_sentence_embedding_dimension()
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _load_t5(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        trust_remote_code = _should_trust_remote_code(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — model name from config, not user input
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        raw_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)  # nosec B615
        # T5Model has .encoder; T5EncoderModel is already the encoder
        self._model = (raw_model.encoder if hasattr(raw_model, "encoder") else raw_model).to(
            self.device
        )
        self._native_dim = self._model.config.d_model
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _load_decoder(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        trust_remote_code = _should_trust_remote_code(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        _ensure_pad_token(self._tokenizer)
        self._model = AutoModel.from_pretrained(  # nosec B615
            self.model_name,
            torch_dtype=_preferred_torch_dtype(self.device),
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self._native_dim = self._model.config.hidden_size
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _load_peft(self) -> None:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        trust_remote_code = _should_trust_remote_code(self.base_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            self.base_model_name,
            trust_remote_code=trust_remote_code,
        )
        _ensure_pad_token(self._tokenizer)
        base_model = AutoModel.from_pretrained(  # nosec B615
            self.base_model_name,
            torch_dtype=_preferred_torch_dtype(self.device),
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self._model = PeftModel.from_pretrained(base_model, self.model_name).to(self.device)  # nosec B614 — trusted local/hub model from config
        self._native_dim = base_model.config.hidden_size
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad = False

    def _encode_text_chunks(
        self,
        goal_states: list[str],
        encode_fn,
    ) -> torch.Tensor:
        """Chunk encoding to keep large decoder models within memory limits."""
        batches: list[torch.Tensor] = []
        for i in range(0, len(goal_states), self._encode_batch_size):
            batches.append(encode_fn(goal_states[i : i + self._encode_batch_size]))
        if batches:
            return torch.cat(batches, dim=0)
        return torch.zeros((0, self.output_dim), device=self.device, dtype=torch.float32)

    def _encode_t5(self, goal_states: list[str]) -> torch.Tensor:
        """Encode using T5/ByT5 encoder with mean pooling."""
        if self._tokenizer is None or self._model is None:
            msg = "Encoder not initialized — call _init_t5() first"
            raise RuntimeError(msg)

        def _encode_batch(batch: list[str]) -> torch.Tensor:
            tokens = self._tokenizer(
                batch,
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

            hidden = outputs.last_hidden_state  # [batch, seq, dim]
            mask = tokens["attention_mask"].unsqueeze(-1).float()  # [batch, seq, 1]
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        pooled = self._encode_text_chunks(goal_states, _encode_batch)
        return pooled.clone() if self.frozen else pooled

    def _encode_decoder(self, goal_states: list[str]) -> torch.Tensor:
        """Encode using decoder-only model with last-token pooling."""
        if self._tokenizer is None or self._model is None:
            msg = "Encoder not initialized — call _load_decoder() first"
            raise RuntimeError(msg)

        goal_states = _apply_instruction_prefix(self.model_name, goal_states)

        def _encode_batch(batch: list[str]) -> torch.Tensor:
            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            if self.frozen:
                with torch.no_grad():
                    outputs = self._model(**tokens)
            else:
                outputs = self._model(**tokens)

            hidden = outputs.last_hidden_state
            seq_lengths = tokens["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, seq_lengths]
            return torch.nn.functional.normalize(pooled, p=2, dim=1).float()

        pooled = self._encode_text_chunks(goal_states, _encode_batch)
        return pooled.clone() if self.frozen else pooled

    def _encode_st(self, goal_states: list[str]) -> torch.Tensor:
        """Encode using SentenceTransformer."""
        if self._model is None:
            msg = "Encoder not initialized — call _init_st() first"
            raise RuntimeError(msg)
        with torch.no_grad():
            embeddings = self._model.encode(
                goal_states,
                batch_size=self._encode_batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        # Clone to exit inference mode so downstream trainable modules can use autograd
        return embeddings.to(self.device).float().clone()

    def ensure_loaded(self) -> None:
        """Force-initialize the model if not already loaded.

        Must be called before load_state_dict() when restoring from a
        checkpoint, since the lazy _load_model() won't have run yet.
        """
        if self._model is None:
            self._load_model()

    def to(self, *args: Any, **kwargs: Any) -> GoalEncoder:  # type: ignore[override]
        """Keep the lazy-loaded backend aligned with module device moves."""
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = first

        result = super().to(*args, **kwargs)
        if device is not None:
            self.device = str(torch.device(device))
            self._encode_batch_size = _preferred_encode_batch_size(self._backend, self.device)
            if self._projection is not None:
                self._projection = self._projection.to(self.device)
            if self._model is not None and hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
        return result  # type: ignore[return-value]

    def encode(self, goal_states: list[str]) -> torch.Tensor:
        """Encode goal states to [batch, output_dim] embeddings.

        Returns:
            Float32 tensor on self.device, shape [len(goal_states), output_dim].
        """
        self.ensure_loaded()

        if self._backend == "t5":
            embeddings = self._encode_t5(goal_states)
        elif self._backend == "decoder":
            embeddings = self._encode_decoder(goal_states)
        else:
            embeddings = self._encode_st(goal_states)

        if self._projection is not None:
            embeddings = self._projection(embeddings)

        return embeddings

    def forward(self, goal_states: list[str]) -> torch.Tensor:
        return self.encode(goal_states)
