"""Tests for the ProofAuditor (Lane B) — cache, guards, metric separation."""

import asyncio
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

from src.proof_auditor import AuditorConfig, AuditResult, ProofAuditor, SuccessCategory


class TestSuccessCategory(unittest.TestCase):
    def test_values(self):
        self.assertEqual(SuccessCategory.RAW.value, "raw_success")
        self.assertEqual(SuccessCategory.AXLE_ASSISTED.value, "axle_assisted_success")
        self.assertEqual(SuccessCategory.AXLE_REPAIR_ONLY.value, "axle_repair_only")


class TestAuditResult(unittest.TestCase):
    def test_to_dict(self):
        result = AuditResult(
            success=True,
            category=SuccessCategory.RAW,
            verified=True,
        )
        d = result.to_dict()
        self.assertEqual(d["success"], True)
        self.assertEqual(d["category"], "raw_success")
        self.assertEqual(d["verified"], True)
        self.assertEqual(d["cached"], False)

    def test_defaults(self):
        result = AuditResult(success=False, category=SuccessCategory.AXLE_REPAIR_ONLY)
        self.assertEqual(result.subgoals, [])
        self.assertEqual(result.lean_errors, [])
        self.assertEqual(result.api_error, "")


class TestAuditorConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AuditorConfig()
        self.assertEqual(cfg.environment, "lean-4.28.0")
        self.assertFalse(cfg.ignore_imports)
        self.assertTrue(cfg.cache_enabled)
        self.assertEqual(cfg.max_concurrency, 10)
        self.assertIn("grind", cfg.repair_terminal_tactics)


class TestProofAuditorGuards(unittest.TestCase):
    """Test that operations fail clearly without connect()."""

    def test_require_client_before_verify(self):
        auditor = ProofAuditor()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(auditor.verify("stmt", "proof"))
        self.assertIn("connect()", str(ctx.exception))

    def test_require_client_before_repair(self):
        auditor = ProofAuditor()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(auditor.repair("content"))
        self.assertIn("connect()", str(ctx.exception))

    def test_require_client_before_decompose(self):
        auditor = ProofAuditor()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(auditor.decompose("content"))
        self.assertIn("connect()", str(ctx.exception))

    def test_require_client_before_check(self):
        auditor = ProofAuditor()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(auditor.check("content"))
        self.assertIn("connect()", str(ctx.exception))

    def test_require_client_before_extract(self):
        auditor = ProofAuditor()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(auditor.extract_theorems("content"))
        self.assertIn("connect()", str(ctx.exception))


class TestCacheBehavior(unittest.TestCase):
    """Test cache key correctness and immutability."""

    def test_cache_key_includes_all_params(self):
        auditor = ProofAuditor()
        key1 = auditor._cache_key("verify", "stmt", "proof", "")
        key2 = auditor._cache_key("verify", "stmt", "proof", '["helper"]')
        self.assertNotEqual(key1, key2, "Different permitted_sorries must produce different keys")

    def test_cache_key_differs_by_operation(self):
        auditor = ProofAuditor()
        key1 = auditor._cache_key("verify", "content")
        key2 = auditor._cache_key("repair", "content")
        self.assertNotEqual(key1, key2)

    def test_cache_key_differs_by_environment(self):
        a1 = ProofAuditor(AuditorConfig(environment="lean-4.27.0"))
        a2 = ProofAuditor(AuditorConfig(environment="lean-4.28.0"))
        key1 = a1._cache_key("check", "content")
        key2 = a2._cache_key("check", "content")
        self.assertNotEqual(key1, key2)

    def test_cache_returns_copy_not_shared_reference(self):
        auditor = ProofAuditor()
        original = AuditResult(success=True, category=SuccessCategory.RAW)
        auditor._cache_put("test_key", original)

        fetched1 = auditor._cache_get("test_key")
        fetched2 = auditor._cache_get("test_key")

        # Fetched results should be independent copies
        self.assertIsNot(fetched1, fetched2)
        self.assertIsNot(fetched1, original)

        # Mutating fetched should not affect stored or other fetches
        fetched1.success = False
        fetched_again = auditor._cache_get("test_key")
        self.assertTrue(fetched_again.success, "Cache mutation leaked across callers")

    def test_cache_get_sets_cached_flag(self):
        auditor = ProofAuditor()
        original = AuditResult(success=True, category=SuccessCategory.RAW, cached=False)
        auditor._cache_put("key", original)

        fetched = auditor._cache_get("key")
        self.assertTrue(fetched.cached)
        # Original should NOT have cached=True
        self.assertFalse(original.cached)

    def test_cache_disabled(self):
        auditor = ProofAuditor(AuditorConfig(cache_enabled=False))
        auditor._cache_put("key", AuditResult(success=True, category=SuccessCategory.RAW))
        self.assertIsNone(auditor._cache_get("key"))

    def test_cache_clear(self):
        auditor = ProofAuditor()
        auditor._cache_put("key", AuditResult(success=True, category=SuccessCategory.RAW))
        self.assertEqual(auditor.cache_stats()["entries"], 1)
        auditor.cache_clear()
        self.assertEqual(auditor.cache_stats()["entries"], 0)


class TestApiErrorResult(unittest.TestCase):
    def test_error_result_structure(self):
        result = ProofAuditor._api_error_result(ValueError("test error"))
        self.assertFalse(result.success)
        self.assertEqual(result.category, SuccessCategory.RAW)
        self.assertIn("ValueError", result.api_error)
        self.assertIn("test error", result.api_error)


@dataclass
class _FakeMessages:
    errors: list[str] = field(default_factory=list)


@dataclass
class _FakeVerifyResult:
    okay: bool = True
    lean_messages: _FakeMessages = field(default_factory=_FakeMessages)
    tool_messages: _FakeMessages = field(default_factory=_FakeMessages)
    timings: dict = field(default_factory=dict)


@dataclass
class _FakeRepairResult:
    lean_messages: _FakeMessages = field(default_factory=_FakeMessages)
    tool_messages: _FakeMessages = field(default_factory=_FakeMessages)
    timings: dict = field(default_factory=dict)


@dataclass
class _FakeSorry2LemmaResult:
    lemma_names: list[str] = field(default_factory=lambda: ["sub1"])
    lean_messages: _FakeMessages = field(default_factory=_FakeMessages)
    tool_messages: _FakeMessages = field(default_factory=_FakeMessages)
    timings: dict = field(default_factory=dict)


@dataclass
class _FakeDoc:
    name: str = "test_thm"
    signature: str = "theorem test_thm : True"
    type: str = "theorem"
    type_hash: str = "abc123"
    proof_length: int = 1
    tactic_counts: dict = field(default_factory=dict)
    is_sorry: bool = False
    local_type_dependencies: list = field(default_factory=list)
    local_value_dependencies: list = field(default_factory=list)
    external_type_dependencies: list = field(default_factory=list)
    external_value_dependencies: list = field(default_factory=list)
    local_syntactic_dependencies: list = field(default_factory=list)
    external_syntactic_dependencies: list = field(default_factory=list)


@dataclass
class _FakeExtractResult:
    documents: dict = field(default_factory=lambda: {"test_thm": _FakeDoc()})
    lean_messages: _FakeMessages = field(default_factory=_FakeMessages)
    timings: dict = field(default_factory=dict)


def _make_auditor_with_mock_client():
    """Create a ProofAuditor with a mock Axle client injected."""
    auditor = ProofAuditor()
    client = MagicMock()
    client.verify_proof = AsyncMock(return_value=_FakeVerifyResult())
    client.repair_proofs = AsyncMock(return_value=_FakeRepairResult())
    client.sorry2lemma = AsyncMock(return_value=_FakeSorry2LemmaResult())
    client.check = AsyncMock(return_value=_FakeVerifyResult())
    client.extract_theorems = AsyncMock(return_value=_FakeExtractResult())
    client.close = AsyncMock()
    auditor._client = client
    return auditor, client


class TestVerify(unittest.TestCase):
    def test_verify_success(self):
        auditor, client = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.verify("stmt", "proof"))
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.category, SuccessCategory.RAW)
        client.verify_proof.assert_called_once()

    def test_verify_with_permitted_sorries(self):
        auditor, _ = _make_auditor_with_mock_client()
        result = asyncio.run(
            auditor.verify("stmt", "proof", permitted_sorries=["helper"])
        )
        self.assertTrue(result.success)

    def test_verify_uses_cache(self):
        auditor, client = _make_auditor_with_mock_client()
        asyncio.run(auditor.verify("stmt", "proof"))
        r2 = asyncio.run(auditor.verify("stmt", "proof"))
        self.assertTrue(r2.cached)
        self.assertEqual(client.verify_proof.call_count, 1)

    def test_verify_api_error(self):
        auditor, client = _make_auditor_with_mock_client()
        client.verify_proof = AsyncMock(side_effect=TimeoutError("timeout"))
        result = asyncio.run(auditor.verify("stmt", "proof"))
        self.assertFalse(result.success)
        self.assertIn("TimeoutError", result.api_error)


class TestRepair(unittest.TestCase):
    def test_repair_success(self):
        auditor, client = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.repair("content"))
        self.assertTrue(result.success)
        self.assertTrue(result.repaired)
        self.assertEqual(result.category, SuccessCategory.AXLE_REPAIR_ONLY)
        client.repair_proofs.assert_called_once()

    def test_repair_with_errors(self):
        auditor, client = _make_auditor_with_mock_client()
        client.repair_proofs = AsyncMock(
            return_value=_FakeRepairResult(
                lean_messages=_FakeMessages(errors=["error 1"])
            )
        )
        result = asyncio.run(auditor.repair("bad content"))
        self.assertFalse(result.success)

    def test_repair_api_error(self):
        auditor, client = _make_auditor_with_mock_client()
        client.repair_proofs = AsyncMock(side_effect=ConnectionError("refused"))
        result = asyncio.run(auditor.repair("content"))
        self.assertFalse(result.success)
        self.assertIn("ConnectionError", result.api_error)


class TestDecompose(unittest.TestCase):
    def test_decompose_success(self):
        auditor, client = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.decompose("content"))
        self.assertTrue(result.success)
        self.assertEqual(result.category, SuccessCategory.AXLE_ASSISTED)
        self.assertEqual(result.subgoals, ["sub1"])
        client.sorry2lemma.assert_called_once()

    def test_decompose_api_error(self):
        auditor, client = _make_auditor_with_mock_client()
        client.sorry2lemma = AsyncMock(side_effect=RuntimeError("boom"))
        result = asyncio.run(auditor.decompose("content"))
        self.assertFalse(result.success)
        self.assertIn("RuntimeError", result.api_error)


class TestCheck(unittest.TestCase):
    def test_check_success(self):
        auditor, client = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.check("content"))
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.category, SuccessCategory.RAW)

    def test_check_api_error(self):
        auditor, client = _make_auditor_with_mock_client()
        client.check = AsyncMock(side_effect=OSError("network"))
        result = asyncio.run(auditor.check("content"))
        self.assertFalse(result.success)
        self.assertIn("OSError", result.api_error)


class TestExtractTheorems(unittest.TestCase):
    def test_extract_success(self):
        auditor, _ = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.extract_theorems("content"))
        self.assertIn("test_thm", result["documents"])
        doc = result["documents"]["test_thm"]
        self.assertEqual(doc["name"], "test_thm")
        self.assertEqual(doc["type"], "theorem")

    def test_extract_api_error(self):
        auditor, client = _make_auditor_with_mock_client()
        client.extract_theorems = AsyncMock(side_effect=ValueError("bad"))
        result = asyncio.run(auditor.extract_theorems("content"))
        self.assertEqual(result["documents"], {})
        self.assertIn("bad", result["errors"][0])


class TestClose(unittest.TestCase):
    def test_close_clears_client(self):
        auditor, client = _make_auditor_with_mock_client()
        asyncio.run(auditor.close())
        self.assertIsNone(auditor._client)
        client.close.assert_called_once()

    def test_close_noop_without_client(self):
        auditor = ProofAuditor()
        asyncio.run(auditor.close())  # should not raise


if __name__ == "__main__":
    unittest.main()
