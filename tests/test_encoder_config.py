"""Tests for encoder_config — pure config detection functions.

Unlocks mutation profiling (DISCOVERY_FAILURE → testable).
All functions are pure with cacheable optimization hints.
"""

import unittest
from types import SimpleNamespace

import torch

from src.encoder_config import (
    _apply_instruction_prefix,
    _detect_backend,
    _ensure_pad_token,
    _get_instruction_prefix,
    _is_peft_model,
    _preferred_encode_batch_size,
    _preferred_torch_dtype,
    _resolve_base_model_name,
    _should_trust_remote_code,
    get_encoder_model_name,
)


# ---------------------------------------------------------------------------
# get_encoder_model_name (σ=2, pure)
# ---------------------------------------------------------------------------


class TestGetEncoderModelName(unittest.TestCase):
    def test_exact_from_type(self):
        self.assertEqual(get_encoder_model_name({"type": "bert-base"}), "bert-base")

    def test_exact_from_model_name(self):
        self.assertEqual(
            get_encoder_model_name({"model_name": "all-MiniLM-L6-v2"}),
            "all-MiniLM-L6-v2",
        )

    def test_default_when_empty(self):
        self.assertEqual(get_encoder_model_name({}), "all-MiniLM-L6-v2")

    def test_type_takes_priority(self):
        self.assertEqual(
            get_encoder_model_name({"type": "a", "model_name": "b"}), "a"
        )


# ---------------------------------------------------------------------------
# _resolve_base_model_name (σ=3, pure)
# ---------------------------------------------------------------------------


class TestResolveBaseModelName(unittest.TestCase):
    def test_known_peft(self):
        self.assertEqual(
            _resolve_base_model_name("FrenzyMath/LeanSearch-PS"),
            "intfloat/e5-mistral-7b-instruct",
        )

    def test_unknown_returns_self(self):
        self.assertEqual(
            _resolve_base_model_name("some-random-model"), "some-random-model"
        )


# ---------------------------------------------------------------------------
# _is_peft_model (σ=2, pure)
# ---------------------------------------------------------------------------


class TestIsPeftModel(unittest.TestCase):
    def test_known_peft_true(self):
        self.assertTrue(_is_peft_model("FrenzyMath/LeanSearch-PS"))

    def test_unknown_false(self):
        self.assertFalse(_is_peft_model("all-MiniLM-L6-v2"))


# ---------------------------------------------------------------------------
# _should_trust_remote_code (σ=3, pure)
# ---------------------------------------------------------------------------


class TestShouldTrustRemoteCode(unittest.TestCase):
    def test_skip_stella(self):
        self.assertFalse(
            _should_trust_remote_code("dunzhang/stella_en_1.5B_v5")
        )

    def test_skip_gte_qwen(self):
        self.assertFalse(
            _should_trust_remote_code("Alibaba-NLP/gte-Qwen2-7B-instruct")
        )

    def test_normal_model_trusted(self):
        self.assertTrue(_should_trust_remote_code("all-MiniLM-L6-v2"))


# ---------------------------------------------------------------------------
# _preferred_torch_dtype (σ=2, pure)
# ---------------------------------------------------------------------------


class TestPreferredTorchDtype(unittest.TestCase):
    def test_cpu_returns_float32(self):
        self.assertEqual(_preferred_torch_dtype("cpu"), torch.float32)

    def test_cuda_returns_float16(self):
        self.assertEqual(_preferred_torch_dtype("cuda"), torch.float16)

    def test_mps_returns_float16(self):
        self.assertEqual(_preferred_torch_dtype("mps"), torch.float16)

    def test_cpu_prefix_returns_float32(self):
        self.assertEqual(_preferred_torch_dtype("cpu:0"), torch.float32)


# ---------------------------------------------------------------------------
# _preferred_encode_batch_size (σ=6, pure)
# ---------------------------------------------------------------------------


class TestPreferredEncodeBatchSize(unittest.TestCase):
    def test_decoder_cpu(self):
        self.assertEqual(_preferred_encode_batch_size("decoder", "cpu"), 1)

    def test_decoder_cuda(self):
        self.assertEqual(_preferred_encode_batch_size("decoder", "cuda"), 4)

    def test_peft_cpu(self):
        self.assertEqual(_preferred_encode_batch_size("peft", "cpu"), 1)

    def test_peft_cuda(self):
        self.assertEqual(_preferred_encode_batch_size("peft", "cuda"), 4)

    def test_t5_cpu(self):
        self.assertEqual(_preferred_encode_batch_size("t5", "cpu"), 4)

    def test_t5_cuda(self):
        self.assertEqual(_preferred_encode_batch_size("t5", "cuda"), 16)

    def test_sentence_transformer_cpu(self):
        self.assertEqual(
            _preferred_encode_batch_size("sentence_transformer", "cpu"), 16
        )

    def test_sentence_transformer_cuda(self):
        self.assertEqual(
            _preferred_encode_batch_size("sentence_transformer", "cuda"), 64
        )

    def test_swap_backend_changes_result(self):
        """SWAP: different backends on same device produce different sizes."""
        self.assertNotEqual(
            _preferred_encode_batch_size("decoder", "cuda"),
            _preferred_encode_batch_size("t5", "cuda"),
        )


# ---------------------------------------------------------------------------
# _get_instruction_prefix (σ=4, pure)
# ---------------------------------------------------------------------------


class TestGetInstructionPrefix(unittest.TestCase):
    def test_plain_model_empty(self):
        self.assertEqual(_get_instruction_prefix("all-MiniLM-L6-v2"), "")

    def test_e5_mistral_has_prefix(self):
        prefix = _get_instruction_prefix("intfloat/e5-mistral-7b-instruct")
        self.assertIn("Instruct:", prefix)
        self.assertIn("Query:", prefix)

    def test_pplx_embed_has_prefix(self):
        prefix = _get_instruction_prefix("pplx-embed-v1")
        self.assertIn("Instruct:", prefix)

    def test_qwen3_embedding_has_prefix(self):
        prefix = _get_instruction_prefix("Qwen/qwen3-embedding-0.6B")
        self.assertIn("Instruct:", prefix)

    def test_peft_resolves_base(self):
        """PEFT adapter resolves to base model for instruction detection."""
        prefix = _get_instruction_prefix("FrenzyMath/LeanSearch-PS")
        self.assertIn("Instruct:", prefix)  # base is e5-mistral


# ---------------------------------------------------------------------------
# _apply_instruction_prefix (σ=2, pure)
# ---------------------------------------------------------------------------


class TestApplyInstructionPrefix(unittest.TestCase):
    def test_no_prefix_passthrough(self):
        goals = ["⊢ P", "⊢ Q"]
        result = _apply_instruction_prefix("all-MiniLM-L6-v2", goals)
        self.assertEqual(result, goals)

    def test_with_prefix_prepended(self):
        goals = ["⊢ P"]
        result = _apply_instruction_prefix("intfloat/e5-mistral-7b-instruct", goals)
        self.assertTrue(result[0].startswith("Instruct:"))
        self.assertTrue(result[0].endswith("⊢ P"))


# ---------------------------------------------------------------------------
# _ensure_pad_token (σ=4, side-effectful but testable)
# ---------------------------------------------------------------------------


class TestEnsurePadToken(unittest.TestCase):
    def test_sets_pad_from_eos(self):
        tok = SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=None, eos_token_id=2)
        _ensure_pad_token(tok)
        self.assertEqual(tok.pad_token, "</s>")
        self.assertEqual(tok.pad_token_id, 2)

    def test_leaves_existing_pad(self):
        tok = SimpleNamespace(pad_token="[PAD]", eos_token="</s>", pad_token_id=0, eos_token_id=2)
        _ensure_pad_token(tok)
        self.assertEqual(tok.pad_token, "[PAD]")
        self.assertEqual(tok.pad_token_id, 0)

    def test_no_eos_no_change(self):
        tok = SimpleNamespace(pad_token=None, eos_token=None, pad_token_id=None, eos_token_id=None)
        _ensure_pad_token(tok)
        self.assertIsNone(tok.pad_token)


# ---------------------------------------------------------------------------
# _detect_backend (σ=14, pure)
# ---------------------------------------------------------------------------


class TestDetectBackend(unittest.TestCase):
    def test_peft_model_detected(self):
        self.assertEqual(_detect_backend("FrenzyMath/LeanSearch-PS"), "peft")

    def test_sentence_transformer_default(self):
        self.assertEqual(_detect_backend("all-MiniLM-L6-v2"), "sentence_transformer")

    def test_name_fallback_t5(self):
        self.assertEqual(_detect_backend("google/flan-t5-base"), "t5")

    def test_name_fallback_decoder(self):
        self.assertEqual(_detect_backend("mistralai/Mistral-7B-v0.1"), "decoder")

    def test_name_fallback_qwen3(self):
        self.assertEqual(_detect_backend("Qwen/qwen3-embedding-0.6B"), "decoder")


if __name__ == "__main__":
    unittest.main()
