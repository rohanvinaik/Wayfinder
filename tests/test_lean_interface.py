"""Tests for lean_interface — LeanKernel stub backend and config."""

import unittest

from src.lean_interface import LeanConfig, LeanKernel
from src.nav_contracts import TacticResult


class TestLeanConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = LeanConfig()
        self.assertEqual(cfg.backend, "stub")
        self.assertEqual(cfg.timeout, 30)
        self.assertEqual(cfg.hammer_timeout, 60)
        self.assertEqual(cfg.project_root, "")

    def test_custom_values(self):
        cfg = LeanConfig(backend="pantograph", timeout=10, hammer_timeout=120)
        self.assertEqual(cfg.backend, "pantograph")
        self.assertEqual(cfg.timeout, 10)
        self.assertEqual(cfg.hammer_timeout, 120)


class TestLeanKernelInit(unittest.TestCase):

    def test_default_config(self):
        kernel = LeanKernel()
        self.assertEqual(kernel.config.backend, "stub")

    def test_custom_config(self):
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)
        self.assertEqual(kernel._backend, "pantograph")


class TestTryTactic(unittest.TestCase):

    def test_stub_always_fails(self):
        kernel = LeanKernel()
        result = kernel.try_tactic("⊢ True", "trivial")
        self.assertIsInstance(result, TacticResult)
        self.assertFalse(result.success)
        self.assertEqual(result.tactic, "trivial")
        self.assertEqual(result.premises, [])
        self.assertIn("stub", result.error_message)

    def test_stub_returns_empty_new_goals(self):
        kernel = LeanKernel()
        result = kernel.try_tactic("⊢ P", "intro h")
        self.assertEqual(result.new_goals, [])

    def test_unknown_backend_raises(self):
        cfg = LeanConfig(backend="nonexistent")
        kernel = LeanKernel(config=cfg)
        with self.assertRaises(ValueError) as ctx:
            kernel.try_tactic("⊢ P", "rfl")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_pantograph_backend_raises_not_implemented(self):
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)
        with self.assertRaises(NotImplementedError):
            kernel.try_tactic("⊢ P", "rfl")


class TestTryHammer(unittest.TestCase):

    def test_stub_always_fails(self):
        kernel = LeanKernel()
        result = kernel.try_hammer("⊢ 1 + 1 = 2", ["Nat.add_comm"])
        self.assertIsInstance(result, TacticResult)
        self.assertFalse(result.success)
        self.assertIn("stub", result.error_message)

    def test_stub_includes_premise_count_in_tactic(self):
        kernel = LeanKernel()
        result = kernel.try_hammer("⊢ P", ["a", "b", "c"])
        self.assertIn("3", result.tactic)

    def test_stub_preserves_premises(self):
        kernel = LeanKernel()
        result = kernel.try_hammer("⊢ P", ["h1", "h2"])
        self.assertEqual(result.premises, ["h1", "h2"])

    def test_timeout_override(self):
        kernel = LeanKernel()
        # Stub doesn't use timeout, but verify it doesn't crash
        result = kernel.try_hammer("⊢ P", [], timeout=5)
        self.assertFalse(result.success)

    def test_unknown_backend_raises(self):
        cfg = LeanConfig(backend="nonexistent")
        kernel = LeanKernel(config=cfg)
        with self.assertRaises(ValueError):
            kernel.try_hammer("⊢ P", [])

    def test_pantograph_backend_raises_not_implemented(self):
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)
        with self.assertRaises(NotImplementedError):
            kernel.try_hammer("⊢ P", [])


if __name__ == "__main__":
    unittest.main()
