"""Tests for the ProofAuditor (Lane B) — cache, guards, metric separation."""

import asyncio
import unittest

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


if __name__ == "__main__":
    unittest.main()
