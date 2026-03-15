"""Encoder configuration utilities — pure functions for model detection and config.

Extracted from encoder.py per LintGate extraction_plan + convergence analysis.
These are stateless utility functions used by GoalEncoder but independent of it.
All have 80%+ convergence confidence on the purity lens.

Post-extraction opportunities (from extraction_plan):
- _detect_backend: cacheable (deterministic_io)
- get_encoder_model_name: cacheable (projected_pure)
- _preferred_encode_batch_size: cacheable (projected_pure)
- _preferred_torch_dtype: cacheable (projected_pure)
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

import torch

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_T5_MODEL_TYPES = {"t5", "byt5", "mt5", "umt5", "longt5"}
_DECODER_MODEL_TYPES = {
    "mistral", "qwen2", "qwen3", "qwen3_moe", "llama", "gpt2", "gpt_neox",
    "mpt", "falcon", "phi", "phi3", "gemma", "gemma2", "gemma3_text",
    "starcoder2", "codegen", "rwkv", "rwkv7", "stablelm", "deepseek_v3",
}
_DECODER_ARCHITECTURES = {
    "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3MoeForCausalLM",
    "LlamaForCausalLM", "GPT2LMHeadModel", "GPTNeoXForCausalLM",
    "MPTForCausalLM", "FalconForCausalLM", "PhiForCausalLM", "Phi3ForCausalLM",
    "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "RWKV7ForCausalLM", "RWKVForCausalLM", "StableLmForCausalLM",
    "Starcoder2ForCausalLM", "QWenLMHeadModel", "Qwen2ForSequenceClassification",
}
_PEFT_BASE_MODELS = {
    "FrenzyMath/LeanSearch-PS": "intfloat/e5-mistral-7b-instruct",
}
_SKIP_TRUST_REMOTE_CODE = {
    "dunzhang/stella_en_1.5B_v5",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
}


def get_encoder_model_name(config: dict[str, Any]) -> str:
    """Return the configured encoder model name/path."""
    return str(
        config.get("type") or config.get("model_name") or _DEFAULT_MODEL_NAME
    )


def _read_model_config(model_name: str) -> dict[str, Any] | None:
    """Read config.json from a local model directory, if present."""
    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        return _json.load(f)


def _read_adapter_config(model_name: str) -> dict[str, Any] | None:
    """Read adapter_config.json from a local PEFT adapter directory."""
    config_path = Path(model_name) / "adapter_config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        return _json.load(f)


def _resolve_base_model_name(model_name: str) -> str:
    """Resolve the underlying base model for PEFT adapters."""
    if model_name in _PEFT_BASE_MODELS:
        return _PEFT_BASE_MODELS[model_name]
    adapter_cfg = _read_adapter_config(model_name)
    if adapter_cfg is not None:
        return str(adapter_cfg.get("base_model_name_or_path", model_name))
    return model_name


def _is_peft_model(model_name: str) -> bool:
    """Return True when the model is a known or local PEFT adapter."""
    return (
        model_name in _PEFT_BASE_MODELS
        or _read_adapter_config(model_name) is not None
    )


def _should_trust_remote_code(model_name: str) -> bool:
    """Some models break under transformers 5.x; avoid remote code there."""
    base_name = _resolve_base_model_name(model_name)
    return (
        model_name not in _SKIP_TRUST_REMOTE_CODE
        and base_name not in _SKIP_TRUST_REMOTE_CODE
    )


def _preferred_torch_dtype(device: str) -> torch.dtype:
    """Choose a safe default dtype for the target device."""
    return torch.float32 if str(device).startswith("cpu") else torch.float16


def _preferred_encode_batch_size(backend: str, device: str) -> int:
    """Choose a conservative per-backend encode chunk size."""
    on_cpu = str(device).startswith("cpu")
    if backend in {"decoder", "peft"}:
        return 1 if on_cpu else 4
    if backend == "t5":
        return 4 if on_cpu else 16
    return 16 if on_cpu else 64


def _get_instruction_prefix(model_name: str) -> str:
    """Add retrieval instructions for instruction-aware embedding models."""
    config = _read_model_config(model_name)
    if config is not None and (
        config.get("is_instruction_model") or config.get("instruction_template")
    ):
        return (
            "Instruct: Retrieve semantically similar mathematical"
            " proof states\nQuery: "
        )
    lower = _resolve_base_model_name(model_name).lower()
    if any(
        token in lower
        for token in ("e5-mistral", "gte-qwen", "pplx-embed", "qwen3-embedding")
    ):
        return (
            "Instruct: Retrieve semantically similar mathematical"
            " proof states\nQuery: "
        )
    return ""


def _apply_instruction_prefix(
    model_name: str, goal_states: list[str]
) -> list[str]:
    """Prefix inputs for instruction-aware query format."""
    prefix = _get_instruction_prefix(model_name)
    return [prefix + goal for goal in goal_states] if prefix else goal_states


def _ensure_pad_token(tokenizer: Any) -> None:
    """Decoder models often ship without a pad token; reuse EOS."""
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None)
    ):
        tokenizer.pad_token = tokenizer.eos_token
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None)
    ):
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _detect_backend(model_name: str) -> str:
    """Detect encoder backend from config.json, with fallback.

    Returns one of: "peft", "t5", "decoder", "sentence_transformer"
    """
    if _is_peft_model(model_name):
        return "peft"

    config = _read_model_config(model_name)
    if config is not None:
        model_type = config.get("model_type", "").lower()
        architectures = config.get("architectures", [])
        if model_type in _T5_MODEL_TYPES:
            return "t5"
        if model_type in _DECODER_MODEL_TYPES:
            return "decoder"
        if any(arch in _DECODER_ARCHITECTURES for arch in architectures):
            return "decoder"
        if any(
            "CausalLM" in arch or "LMHead" in arch for arch in architectures
        ):
            return "decoder"

    lower = _resolve_base_model_name(model_name).lower()
    if any(k in lower for k in ("t5", "byt5", "mt5", "flan-t5")):
        return "t5"
    if any(
        k in lower
        for k in (
            "mistral", "qwen2", "qwen3", "llama", "mpt", "rwkv",
            "falcon", "gpt", "deepseek", "pplx-embed", "gte-qwen",
            "qwen3-embedding",
        )
    ):
        return "decoder"

    return "sentence_transformer"
