"""Tests for the ProofAuditor (Lane B) — cache, guards, metric separation."""

import asyncio
import types
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
        self.assertEqual(fetched.cached, True)
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


class TestCacheKeyExactValues(unittest.TestCase):
    """Exact-value assertions for _cache_key determinism and structure."""

    def test_cache_key_deterministic_same_input(self):
        """Same inputs must produce identical hash on repeated calls."""
        auditor = ProofAuditor()
        key_a = auditor._cache_key("verify", "theorem foo : True", "exact trivial")
        key_b = auditor._cache_key("verify", "theorem foo : True", "exact trivial")
        self.assertEqual(key_a, key_b)

    def test_cache_key_is_64_char_hex(self):
        """SHA-256 hexdigest is exactly 64 lowercase hex characters."""
        auditor = ProofAuditor()
        key = auditor._cache_key("check", "some content")
        self.assertEqual(len(key), 64)
        self.assertRegex(key, r"^[0-9a-f]{64}$")

    def test_cache_key_differs_by_content(self):
        """Different proof content must produce different keys."""
        auditor = ProofAuditor()
        key1 = auditor._cache_key("verify", "stmt", "proof_a")
        key2 = auditor._cache_key("verify", "stmt", "proof_b")
        self.assertNotEqual(key1, key2)

    def test_cache_key_differs_by_statement(self):
        """Different formal statements must produce different keys."""
        auditor = ProofAuditor()
        key1 = auditor._cache_key("verify", "theorem a : True", "exact trivial")
        key2 = auditor._cache_key("verify", "theorem b : False", "exact trivial")
        self.assertNotEqual(key1, key2)

    def test_cache_key_sensitive_to_content_order(self):
        """Multi-content args are order-sensitive (not commutative)."""
        auditor = ProofAuditor()
        key1 = auditor._cache_key("verify", "alpha", "beta")
        key2 = auditor._cache_key("verify", "beta", "alpha")
        self.assertNotEqual(key1, key2)

    def test_cache_key_empty_content(self):
        """Empty content strings produce a valid deterministic key."""
        auditor = ProofAuditor()
        key1 = auditor._cache_key("check", "")
        key2 = auditor._cache_key("check", "")
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)

    def test_cache_key_matches_manual_sha256(self):
        """Verify _cache_key matches hand-computed SHA-256 for known input."""
        import hashlib

        auditor = ProofAuditor(AuditorConfig(environment="lean-4.28.0"))
        h = hashlib.sha256()
        h.update(b"verify")
        h.update(b"lean-4.28.0")
        h.update(b"stmt")
        h.update(b"proof")
        expected = h.hexdigest()
        actual = auditor._cache_key("verify", "stmt", "proof")
        self.assertEqual(actual, expected)


class TestCachePutExactValues(unittest.TestCase):
    """Exact-value assertions for _cache_put storage behavior."""

    def test_cache_put_stores_exact_result(self):
        """_cache_put stores the result retrievable by _cache_get with all fields intact."""
        auditor = ProofAuditor()
        result = AuditResult(
            success=True,
            category=SuccessCategory.AXLE_ASSISTED,
            verified=True,
            repaired=False,
            subgoals=["lemma1", "lemma2"],
            lean_errors=[],
            tool_errors=["warning: slow"],
            timings={"verify_ms": 450},
            api_error="",
            cached=False,
        )
        auditor._cache_put("exact_key", result)
        fetched = auditor._cache_get("exact_key")

        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.success, True)
        self.assertEqual(fetched.category, SuccessCategory.AXLE_ASSISTED)
        self.assertEqual(fetched.verified, True)
        self.assertFalse(fetched.repaired)
        self.assertEqual(fetched.subgoals, ["lemma1", "lemma2"])
        self.assertEqual(fetched.lean_errors, [])
        self.assertEqual(fetched.tool_errors, ["warning: slow"])
        self.assertEqual(fetched.timings, {"verify_ms": 450})
        self.assertEqual(fetched.api_error, "")
        # cached flag is set by _cache_get, not by the stored value
        self.assertEqual(fetched.cached, True)

    def test_cache_put_overwrites_existing_entry(self):
        """Putting a new result under the same key replaces the old one."""
        auditor = ProofAuditor()
        result_v1 = AuditResult(success=True, category=SuccessCategory.RAW)
        result_v2 = AuditResult(
            success=False, category=SuccessCategory.AXLE_REPAIR_ONLY, api_error="replaced"
        )
        auditor._cache_put("same_key", result_v1)
        auditor._cache_put("same_key", result_v2)

        fetched = auditor._cache_get("same_key")
        self.assertFalse(fetched.success)
        self.assertEqual(fetched.category, SuccessCategory.AXLE_REPAIR_ONLY)
        self.assertEqual(fetched.api_error, "replaced")

    def test_cache_put_noop_when_disabled(self):
        """_cache_put does not store when cache_enabled=False."""
        auditor = ProofAuditor(AuditorConfig(cache_enabled=False))
        result = AuditResult(success=True, category=SuccessCategory.RAW)
        auditor._cache_put("key", result)
        # Internal dict should remain empty
        self.assertEqual(len(auditor._cache), 0)

    def test_cache_put_multiple_keys(self):
        """Storing under different keys maintains independent entries."""
        auditor = ProofAuditor()
        r1 = AuditResult(success=True, category=SuccessCategory.RAW, verified=True)
        r2 = AuditResult(success=False, category=SuccessCategory.AXLE_REPAIR_ONLY, repaired=True)
        auditor._cache_put("key_a", r1)
        auditor._cache_put("key_b", r2)

        f1 = auditor._cache_get("key_a")
        f2 = auditor._cache_get("key_b")
        self.assertEqual(f1.success, True)
        self.assertEqual(f1.verified, True)
        self.assertFalse(f2.success)
        self.assertEqual(f2.repaired, True)


class TestCacheGetExactValues(unittest.TestCase):
    """Exact-value assertions for _cache_get retrieval behavior."""

    def test_cache_get_miss_returns_none(self):
        """_cache_get returns None for a key that was never stored."""
        auditor = ProofAuditor()
        self.assertIsNone(auditor._cache_get("nonexistent_key"))

    def test_cache_get_preserves_category_enum(self):
        """Retrieved result has the same SuccessCategory enum, not a string."""
        auditor = ProofAuditor()
        for cat in SuccessCategory:
            result = AuditResult(success=True, category=cat)
            key = f"cat_{cat.value}"
            auditor._cache_put(key, result)
            fetched = auditor._cache_get(key)
            self.assertIsInstance(fetched.category, SuccessCategory)
            self.assertEqual(fetched.category, cat)

    def test_cache_get_preserves_list_contents(self):
        """Mutable list fields survive put/get cycle with exact values."""
        auditor = ProofAuditor()
        result = AuditResult(
            success=False,
            category=SuccessCategory.RAW,
            lean_errors=["error: unknown identifier 'foo'", "error: type mismatch"],
            subgoals=["sub_a", "sub_b", "sub_c"],
        )
        auditor._cache_put("list_key", result)
        fetched = auditor._cache_get("list_key")
        self.assertEqual(
            fetched.lean_errors,
            ["error: unknown identifier 'foo'", "error: type mismatch"],
        )
        self.assertEqual(fetched.subgoals, ["sub_a", "sub_b", "sub_c"])

    def test_cache_get_disabled_always_returns_none(self):
        """With cache disabled, _cache_get returns None even if _cache dict has data."""
        auditor = ProofAuditor(AuditorConfig(cache_enabled=False))
        # Bypass _cache_put guard by writing directly
        result = AuditResult(success=True, category=SuccessCategory.RAW)
        auditor._cache["forced_key"] = result
        self.assertIsNone(auditor._cache_get("forced_key"))


class TestApiErrorResultExactValues(unittest.TestCase):
    """Exact-value assertions for _api_error_result fields."""

    def test_error_result_exact_fields(self):
        """All fields of the returned AuditResult match expected defaults."""
        err = ValueError("proof content was empty")
        result = ProofAuditor._api_error_result(err)

        self.assertFalse(result.success)
        self.assertEqual(result.category, SuccessCategory.RAW)
        self.assertFalse(result.verified)
        self.assertFalse(result.repaired)
        self.assertEqual(result.subgoals, [])
        self.assertEqual(result.lean_errors, [])
        self.assertEqual(result.tool_errors, [])
        self.assertEqual(result.timings, {})
        self.assertEqual(result.api_error, "ValueError: proof content was empty")
        self.assertFalse(result.cached)

    def test_error_result_timeout_error(self):
        """TimeoutError produces correct error type prefix."""
        result = ProofAuditor._api_error_result(TimeoutError("120s exceeded"))
        self.assertEqual(result.api_error, "TimeoutError: 120s exceeded")
        self.assertFalse(result.success)
        self.assertEqual(result.category, SuccessCategory.RAW)

    def test_error_result_connection_error(self):
        """ConnectionError produces correct error type prefix."""
        result = ProofAuditor._api_error_result(ConnectionError("refused"))
        self.assertEqual(result.api_error, "ConnectionError: refused")

    def test_error_result_empty_message(self):
        """Exception with empty message still formats correctly."""
        result = ProofAuditor._api_error_result(RuntimeError(""))
        self.assertEqual(result.api_error, "RuntimeError: ")
        self.assertFalse(result.success)

    def test_error_result_to_dict_round_trip(self):
        """_api_error_result().to_dict() produces expected keys and values."""
        result = ProofAuditor._api_error_result(OSError("network down"))
        d = result.to_dict()
        self.assertEqual(d["success"], False)
        self.assertEqual(d["category"], "raw_success")
        self.assertEqual(d["verified"], False)
        self.assertEqual(d["repaired"], False)
        self.assertEqual(d["subgoals"], [])
        self.assertEqual(d["lean_errors"], [])
        self.assertEqual(d["tool_errors"], [])
        self.assertEqual(d["timings"], {})
        self.assertEqual(d["api_error"], "OSError: network down")
        self.assertEqual(d["cached"], False)


class TestCacheStatsExactValues(unittest.TestCase):
    """Exact-value assertions for cache_stats after known operation sequences."""

    def test_cache_stats_empty(self):
        """Fresh auditor has zero entries."""
        auditor = ProofAuditor()
        stats = auditor.cache_stats()
        self.assertEqual(stats, {"entries": 0})

    def test_cache_stats_after_puts(self):
        """cache_stats reflects exact number of stored entries."""
        auditor = ProofAuditor()
        auditor._cache_put("k1", AuditResult(success=True, category=SuccessCategory.RAW))
        self.assertEqual(auditor.cache_stats(), {"entries": 1})

        auditor._cache_put("k2", AuditResult(success=False, category=SuccessCategory.AXLE_ASSISTED))
        self.assertEqual(auditor.cache_stats(), {"entries": 2})

        auditor._cache_put(
            "k3",
            AuditResult(success=True, category=SuccessCategory.AXLE_REPAIR_ONLY),
        )
        self.assertEqual(auditor.cache_stats(), {"entries": 3})

    def test_cache_stats_overwrite_does_not_increase(self):
        """Overwriting an existing key does not change entry count."""
        auditor = ProofAuditor()
        auditor._cache_put("k1", AuditResult(success=True, category=SuccessCategory.RAW))
        auditor._cache_put("k1", AuditResult(success=False, category=SuccessCategory.RAW))
        self.assertEqual(auditor.cache_stats(), {"entries": 1})

    def test_cache_stats_after_clear(self):
        """cache_stats returns zero after cache_clear."""
        auditor = ProofAuditor()
        for i in range(5):
            auditor._cache_put(f"k{i}", AuditResult(success=True, category=SuccessCategory.RAW))
        self.assertEqual(auditor.cache_stats(), {"entries": 5})
        auditor.cache_clear()
        self.assertEqual(auditor.cache_stats(), {"entries": 0})

    def test_cache_stats_gets_do_not_change_count(self):
        """_cache_get (hit or miss) does not alter entry count."""
        auditor = ProofAuditor()
        auditor._cache_put("k1", AuditResult(success=True, category=SuccessCategory.RAW))
        auditor._cache_get("k1")  # hit
        auditor._cache_get("k1")  # hit again
        auditor._cache_get("nonexistent")  # miss
        self.assertEqual(auditor.cache_stats(), {"entries": 1})

    def test_cache_stats_disabled_cache_stays_zero(self):
        """With cache disabled, puts are no-ops so stats remain zero."""
        auditor = ProofAuditor(AuditorConfig(cache_enabled=False))
        auditor._cache_put("k1", AuditResult(success=True, category=SuccessCategory.RAW))
        auditor._cache_put("k2", AuditResult(success=False, category=SuccessCategory.RAW))
        self.assertEqual(auditor.cache_stats(), {"entries": 0})


class TestAuditResultToDictRoundTrip(unittest.TestCase):
    """Exact-value assertions for AuditResult.to_dict key set and values."""

    def test_to_dict_has_exactly_expected_keys(self):
        """to_dict returns exactly the 10 expected keys, no more, no less."""
        result = AuditResult(success=True, category=SuccessCategory.RAW)
        d = result.to_dict()
        expected_keys = {
            "success",
            "category",
            "verified",
            "repaired",
            "subgoals",
            "lean_errors",
            "tool_errors",
            "timings",
            "api_error",
            "cached",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_category_is_string_not_enum(self):
        """to_dict serializes SuccessCategory to its string value."""
        for cat in SuccessCategory:
            result = AuditResult(success=True, category=cat)
            d = result.to_dict()
            self.assertIsInstance(d["category"], str)
            self.assertEqual(d["category"], cat.value)

    def test_to_dict_full_populated_result(self):
        """to_dict with all fields populated returns exact expected values."""
        result = AuditResult(
            success=True,
            category=SuccessCategory.AXLE_ASSISTED,
            verified=True,
            repaired=True,
            subgoals=["lem_a", "lem_b"],
            lean_errors=["warning: unused variable"],
            tool_errors=["timeout on step 3"],
            timings={"verify_ms": 100, "repair_ms": 200},
            api_error="",
            cached=True,
        )
        d = result.to_dict()
        self.assertEqual(d["success"], True)
        self.assertEqual(d["category"], "axle_assisted_success")
        self.assertEqual(d["verified"], True)
        self.assertEqual(d["repaired"], True)
        self.assertEqual(d["subgoals"], ["lem_a", "lem_b"])
        self.assertEqual(d["lean_errors"], ["warning: unused variable"])
        self.assertEqual(d["tool_errors"], ["timeout on step 3"])
        self.assertEqual(d["timings"], {"verify_ms": 100, "repair_ms": 200})
        self.assertEqual(d["api_error"], "")
        self.assertEqual(d["cached"], True)

    def test_to_dict_default_result(self):
        """to_dict for a minimal AuditResult returns correct defaults."""
        result = AuditResult(success=False, category=SuccessCategory.AXLE_REPAIR_ONLY)
        d = result.to_dict()
        self.assertEqual(d["success"], False)
        self.assertEqual(d["category"], "axle_repair_only")
        self.assertEqual(d["verified"], False)
        self.assertEqual(d["repaired"], False)
        self.assertEqual(d["subgoals"], [])
        self.assertEqual(d["lean_errors"], [])
        self.assertEqual(d["tool_errors"], [])
        self.assertEqual(d["timings"], {})
        self.assertEqual(d["api_error"], "")
        self.assertEqual(d["cached"], False)


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


def _make_fake_doc(**kwargs: object) -> types.SimpleNamespace:
    """Create a fake Axle Document with default fields."""
    defaults = {
        "name": "test_thm",
        "signature": "theorem test_thm : True",
        "type": "theorem",
        "type_hash": "abc123",
        "proof_length": 1,
        "tactic_counts": {},
        "is_sorry": False,
        "local_type_dependencies": [],
        "local_value_dependencies": [],
        "external_type_dependencies": [],
        "external_value_dependencies": [],
        "local_syntactic_dependencies": [],
        "external_syntactic_dependencies": [],
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


@dataclass
class _FakeExtractResult:
    documents: dict = field(default_factory=lambda: {"test_thm": _make_fake_doc()})
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
        self.assertEqual(result.success, True)
        self.assertEqual(result.verified, True)
        self.assertEqual(result.category, SuccessCategory.RAW)
        client.verify_proof.assert_called_once()

    def test_verify_with_permitted_sorries(self):
        auditor, _ = _make_auditor_with_mock_client()
        result = asyncio.run(auditor.verify("stmt", "proof", permitted_sorries=["helper"]))
        self.assertEqual(result.success, True)

    def test_verify_uses_cache(self):
        auditor, client = _make_auditor_with_mock_client()
        asyncio.run(auditor.verify("stmt", "proof"))
        r2 = asyncio.run(auditor.verify("stmt", "proof"))
        self.assertEqual(r2.cached, True)
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
        self.assertEqual(result.success, True)
        self.assertEqual(result.repaired, True)
        self.assertEqual(result.category, SuccessCategory.AXLE_REPAIR_ONLY)
        client.repair_proofs.assert_called_once()

    def test_repair_with_errors(self):
        auditor, client = _make_auditor_with_mock_client()
        client.repair_proofs = AsyncMock(
            return_value=_FakeRepairResult(lean_messages=_FakeMessages(errors=["error 1"]))
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
        self.assertEqual(result.success, True)
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
        self.assertEqual(result.success, True)
        self.assertEqual(result.verified, True)
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
