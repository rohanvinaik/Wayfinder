"""Encoder backend helpers — transformers compat shims, model loading, and encoding.

Split from eval_encoders.py for cohesion: these functions handle transformers 5.x
compatibility, buffer repair, memory management, tokenizer loading, and the
multi-backend encode_goals dispatcher.

Backend selection uses config.json auto-detection when available, with string-matching
fallback for models without local config files (HF hub downloads, etc.).
"""

from __future__ import annotations

import gc
import json as _json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# Models whose custom code is broken on transformers 5.x — load via native support
# (SentenceTransformer handles pooling/projection via the model's config)
SKIP_TRUST_REMOTE_CODE = {
    "dunzhang/stella_en_1.5B_v5",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
}


# --- Config.json-based auto-detection ---

# model_type values from config.json that indicate T5-family encoders
_T5_MODEL_TYPES = {"t5", "byt5", "mt5", "umt5", "longt5"}

# model_type values that indicate decoder-only architectures (need last-token pooling)
_DECODER_MODEL_TYPES = {
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "llama",
    "gpt2",
    "gpt_neox",
    "mpt",
    "falcon",
    "phi",
    "phi3",
    "gemma",
    "gemma2",
    "gemma3_text",
    "starcoder2",
    "codegen",
    "rwkv",
    "rwkv7",
    "stablelm",
    "deepseek_v3",
}

# architectures field values that indicate decoder-only
_DECODER_ARCHITECTURES = {
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3MoeForCausalLM",
    "LlamaForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoXForCausalLM",
    "MPTForCausalLM",
    "FalconForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "RWKV7ForCausalLM",
    "RWKVForCausalLM",
    "StableLmForCausalLM",
    "Starcoder2ForCausalLM",
    "QWenLMHeadModel",
    "Qwen2ForSequenceClassification",
}


def _read_model_config(model_name_or_path: str) -> dict | None:
    """Read config.json from a local model directory, if available."""
    path = Path(model_name_or_path)
    config_path = path / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            return _json.load(f)
    return None


def _has_adapter_config(model_name_or_path: str) -> bool:
    """Check if model directory contains adapter_config.json (PEFT/LoRA)."""
    path = Path(model_name_or_path)
    return (path / "adapter_config.json").is_file()


def _detect_from_config(config: dict) -> str | None:
    """Detect model type from a parsed config.json. Returns None if unrecognized."""
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    if model_type in _T5_MODEL_TYPES:
        return "t5"
    if model_type in _DECODER_MODEL_TYPES:
        return "decoder"
    if any(arch in _DECODER_ARCHITECTURES for arch in architectures):
        return "decoder"
    # Catch-all: *ForCausalLM or *LMHead patterns in architectures
    if any("CausalLM" in arch or "LMHead" in arch for arch in architectures):
        return "decoder"
    return None


def _detect_from_name(model_name_or_path: str) -> str:
    """Detect model type via string-matching on the model name/path."""
    lower = model_name_or_path.lower()
    if any(k in lower for k in ("t5", "byt5", "mt5", "flan-t5")):
        return "t5"
    if model_name_or_path in LAST_TOKEN_POOL_MODELS:
        return "decoder"
    if any(k in lower for k in ("mistral", "qwen2", "llama", "mpt", "rwkv", "falcon", "gpt")):
        return "decoder"
    return "sentence_transformer"


def detect_model_type(model_name_or_path: str) -> str:
    """Auto-detect model backend type from config.json, with string-matching fallback.

    Returns one of: "peft", "t5", "decoder", "sentence_transformer"
    """
    # 1. Check for PEFT adapter (local dir or known registry)
    if model_name_or_path in PEFT_MODELS or _has_adapter_config(model_name_or_path):
        return "peft"

    # 2. Try config.json auto-detection (works for local model directories)
    config = _read_model_config(model_name_or_path)
    if config is not None:
        result = _detect_from_config(config)
        if result is not None:
            return result

    # 3. String-matching fallback (HF hub names or unrecognized config)
    return _detect_from_name(model_name_or_path)


def _load_tokenizer(model_name: str):  # type: ignore[no-untyped-def]
    """Load the appropriate tokenizer for a model, with auto-detection.

    For PEFT models, loads the base model's tokenizer. For all others,
    loads directly from the model name/path.
    """
    from transformers import AutoTokenizer

    # PEFT adapters use the base model's tokenizer
    if model_name in PEFT_MODELS:
        model_name = PEFT_MODELS[model_name]
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # nosec B615


def _shim_qwen2_tokenizer_path() -> None:
    """Shim Qwen2 tokenizer module path changed in transformers 5.x."""
    target = "transformers.models.qwen2.tokenization_qwen2_fast"
    if target not in sys.modules:
        try:
            import transformers.models.qwen2 as qwen2_mod

            shim = types.ModuleType(target)
            if hasattr(qwen2_mod, "Qwen2TokenizerFast"):
                shim.Qwen2TokenizerFast = qwen2_mod.Qwen2TokenizerFast  # type: ignore[attr-defined]
            sys.modules[target] = shim
        except ImportError:
            pass


def _shim_qwen2_rope_theta() -> None:
    """Redirect config.rope_theta to rope_parameters dict in transformers 5.x."""
    try:
        from transformers import Qwen2Config

        _orig_qwen2_getattr = Qwen2Config.__getattribute__

        def _qwen2_getattr(self, name: str):  # type: ignore[no-untyped-def]
            try:
                return _orig_qwen2_getattr(self, name)
            except AttributeError:
                if name == "rope_theta":
                    rp = _orig_qwen2_getattr(self, "rope_parameters")
                    if isinstance(rp, dict) and "rope_theta" in rp:
                        return rp["rope_theta"]
                raise

        Qwen2Config.__getattribute__ = _qwen2_getattr  # type: ignore[assignment]
    except ImportError:
        pass


def _shim_dynamic_cache_legacy() -> None:
    """Restore DynamicCache.from_legacy_cache removed in transformers 5.x."""
    try:
        from transformers import DynamicCache

        if not hasattr(DynamicCache, "from_legacy_cache"):

            @classmethod  # type: ignore[misc]
            def _from_legacy_cache(cls, past_key_values=None):  # type: ignore[no-untyped-def]
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, (key, value) in enumerate(past_key_values):
                        cache.update(key, value, layer_idx)
                return cache

            DynamicCache.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]
    except ImportError:
        pass


def _shim_nv_embed_tied_weights() -> None:
    """Guard mark_tied_weights_as_initialized for NV-Embed custom code."""
    try:
        from transformers import PreTrainedModel

        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized

        def _safe_mark(self):  # type: ignore[no-untyped-def]
            if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
                self.all_tied_weights_keys = {}
            _orig_mark(self)

        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass


def _apply_transformers_compat_shims() -> None:
    """Patch module paths for models whose custom code targets older transformers versions.

    transformers 5.x restructured internal modules — e.g.
    `transformers.models.qwen2.tokenization_qwen2_fast` no longer exists as a
    standalone submodule.  Models like stella_en_1.5B_v5 and gte-Qwen2-7B do a
    direct import of that path, so we shim it at the sys.modules level.
    """
    _shim_qwen2_tokenizer_path()
    _shim_qwen2_rope_theta()
    _shim_dynamic_cache_legacy()
    _shim_nv_embed_tied_weights()


_apply_transformers_compat_shims()


def _fix_position_ids(embeddings) -> None:  # type: ignore[no-untyped-def]
    """Validate and re-initialize position_ids if corrupted."""
    if not hasattr(embeddings, "position_ids"):
        return
    buf = embeddings.position_ids
    expected = torch.arange(buf.shape[-1], device=buf.device, dtype=buf.dtype)
    if buf.dim() == 2:
        expected = expected.unsqueeze(0)
    if not torch.equal(buf, expected):
        embeddings.register_buffer("position_ids", expected, persistent=False)


def _fix_rotary_caches(embeddings) -> None:  # type: ignore[no-untyped-def]
    """Recompute rotary embedding caches if NaN/Inf detected."""
    rotary = getattr(embeddings, "rotary_emb", None)
    if rotary is None:
        return
    for cache_name in ("cos_cached", "sin_cached"):
        cache = getattr(rotary, cache_name, None)
        if cache is None or not (torch.isnan(cache).any() or torch.isinf(cache).any()):
            continue
        delattr(rotary, cache_name)
        if hasattr(rotary, "_set_cos_sin_cache"):
            inv_freq = getattr(rotary, "inv_freq", None)
            if inv_freq is not None:
                rotary._set_cos_sin_cache(cache.shape[0], device=inv_freq.device, dtype=cache.dtype)


def _fix_model_buffers(st_model) -> None:  # type: ignore[no-untyped-def]
    """Re-initialize buffers corrupted by transformers 5.x persistent=False bug."""
    try:
        auto_model = st_model[0].auto_model
    except (IndexError, AttributeError):
        return
    embeddings = getattr(auto_model, "embeddings", None)
    if embeddings is None:
        return
    _fix_position_ids(embeddings)
    _fix_rotary_caches(embeddings)


def _clear_accelerator_cache() -> None:
    """Clear GPU/MPS memory caches to prevent stale buffer corruption on reload."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool using the last non-pad token (for decoder-only embedding models)."""
    # Find the index of the last non-pad token per sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, seq_lengths]


def _get_instruction_prefix(model_name: str) -> str:
    """Get the query instruction prefix for models that require it.

    Checks config.json for instruction template fields, falls back to name matching.
    """
    # Check config.json for instruction-aware models
    config = _read_model_config(model_name)
    if config is not None:
        # Some models declare instruction awareness in config
        if config.get("is_instruction_model") or config.get("instruction_template"):
            return "Instruct: Retrieve semantically similar mathematical proof states\nQuery: "

    lower = model_name.lower()
    if "e5-mistral" in lower:
        return "Instruct: Retrieve semantically similar mathematical proof states\nQuery: "
    if "gte-qwen" in lower:
        return "Instruct: Retrieve semantically similar mathematical proof states\nQuery: "
    if "pplx-embed" in lower:
        return "Instruct: Retrieve semantically similar mathematical proof states\nQuery: "
    if "qwen3-embedding" in lower:
        return "Instruct: Retrieve semantically similar mathematical proof states\nQuery: "
    return ""


def _apply_instruction_prefix(model_name: str, goals: list[str]) -> list[str]:
    prefix = _get_instruction_prefix(model_name)
    return [prefix + g for g in goals] if prefix else goals


# Decoder-only models that need last-token pooling instead of mean pooling
LAST_TOKEN_POOL_MODELS = {
    "intfloat/e5-mistral-7b-instruct",
    "nvidia/NV-Embed-v2",
}

# LoRA adapter models — maps adapter repo to base model
PEFT_MODELS: dict[str, str] = {
    "FrenzyMath/LeanSearch-PS": "intfloat/e5-mistral-7b-instruct",
}


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states using attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class _ExplicitKeysWrapper(torch.nn.Module):
    """Wraps a model to forward only input_ids and attention_mask (for T5 encoders)."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.inner(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"])


def _pool_and_collect(
    pool_fn,  # type: ignore[no-untyped-def]
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize: bool,
) -> np.ndarray:
    pooled = pool_fn(hidden, attention_mask)
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.float().cpu().numpy()


def _batch_encode_with_model(  # internal helper: 6 args justified by backend dispatch
    model: torch.nn.Module,
    tokenizer,  # type: ignore[no-untyped-def]
    texts: list[str],
    batch_size: int,
    pool_fn,  # type: ignore[no-untyped-def]
    normalize: bool,
) -> tuple[list[np.ndarray], float]:
    all_emb: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(next(model.parameters()).device)
            all_emb.append(
                _pool_and_collect(
                    pool_fn,
                    model(**tokens).last_hidden_state,
                    tokens["attention_mask"],
                    normalize,
                )
            )
    return all_emb, time.perf_counter() - start


def _encode_peft(
    model_name: str,
    goals: list[str],
    device: str,
) -> tuple[list[np.ndarray], float, int]:
    """Encode via PEFT/LoRA adapter on a base model."""
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    base_name = PEFT_MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(base_name)  # nosec B615
    base_model = AutoModel.from_pretrained(  # nosec B615
        base_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, model_name).to(device)  # nosec B614
    model.eval()
    native_dim = base_model.config.hidden_size

    all_emb, elapsed = _batch_encode_with_model(
        model,
        tokenizer,
        _apply_instruction_prefix(base_name, goals),
        4,
        _last_token_pool,
        normalize=True,
    )
    del model, base_model, tokenizer
    _clear_accelerator_cache()
    return all_emb, elapsed, native_dim


def _encode_t5(
    model_name: str,
    goals: list[str],
    device: str,
) -> tuple[list[np.ndarray], float, int]:
    """Encode via T5 encoder with mean pooling."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # nosec B615
    raw_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)  # nosec B615
    encoder = (raw_model.encoder if hasattr(raw_model, "encoder") else raw_model).to(device)
    encoder.eval()
    native_dim = encoder.config.d_model

    all_emb, elapsed = _batch_encode_with_model(
        _ExplicitKeysWrapper(encoder),
        tokenizer,
        goals,
        32,
        _mean_pool,
        normalize=False,
    )
    del encoder, raw_model, tokenizer
    _clear_accelerator_cache()
    return all_emb, elapsed, native_dim


def _encode_decoder(
    model_name: str,
    goals: list[str],
    device: str,
) -> tuple[list[np.ndarray], float, int]:
    """Encode via decoder-only model with last-token pooling."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # nosec B615
    model = AutoModel.from_pretrained(  # nosec B615
        model_name,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    native_dim = model.config.hidden_size

    all_emb, elapsed = _batch_encode_with_model(
        model,
        tokenizer,
        _apply_instruction_prefix(model_name, goals),
        4,
        _last_token_pool,
        normalize=True,
    )
    del model, tokenizer
    _clear_accelerator_cache()
    return all_emb, elapsed, native_dim


def _encode_sentence_transformer(
    model_name: str,
    goals: list[str],
    device: str,
) -> tuple[list[np.ndarray], float, int]:
    """Encode via SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    trust = model_name not in SKIP_TRUST_REMOTE_CODE
    model = SentenceTransformer(model_name, device=device, trust_remote_code=trust)
    _fix_model_buffers(model)
    native_dim = model.get_sentence_embedding_dimension() or 0

    start = time.perf_counter()
    emb = model.encode(goals, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    elapsed = time.perf_counter() - start
    del model
    _clear_accelerator_cache()
    return [emb], elapsed, native_dim


def encode_goals(
    model_name: str,
    goals: list[str],
    device: str,
    _retry: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Encode goal states, return (embeddings, throughput, native_dim).

    Backend is auto-detected from config.json when available (local model dirs),
    with string-matching fallback for HF hub model names.
    """
    backend = detect_model_type(model_name)
    if backend == "peft":
        all_emb, elapsed, native_dim = _encode_peft(model_name, goals, device)
    elif backend == "t5":
        all_emb, elapsed, native_dim = _encode_t5(model_name, goals, device)
    elif backend == "decoder":
        all_emb, elapsed, native_dim = _encode_decoder(model_name, goals, device)
    else:
        all_emb, elapsed, native_dim = _encode_sentence_transformer(model_name, goals, device)

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)

    # Guard against intermittent NaN from MPS buffer corruption — retry once
    if np.any(np.isnan(embeddings)) and not _retry:
        print(f"    WARNING: NaN detected in embeddings for {model_name}, retrying...")
        _clear_accelerator_cache()
        return encode_goals(model_name, goals, device, _retry=True)

    throughput = len(goals) / elapsed
    return embeddings, throughput, native_dim or embeddings.shape[1]


# Re-export for backward compatibility — canonical location is scripts.similarity_metrics
from scripts.similarity_metrics import compute_similarity_metrics  # noqa: F401, E402
