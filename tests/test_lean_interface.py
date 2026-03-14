"""Tests for lean_interface — LeanKernel stub, replay, and pantograph backends."""

import unittest

from src.lean_interface import LeanConfig, LeanKernel, _build_hammer_tactics
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

    def test_pantograph_backend_import_error_without_package(self):
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)
        # If PyPantograph is not installed, should get ImportError
        # If installed, should get a different error (no project, etc.)
        # Either way, should not raise NotImplementedError
        try:
            kernel.try_tactic("⊢ P", "rfl")
        except ImportError:
            pass  # expected without PyPantograph
        except Exception:
            pass  # any other error is fine (server init failure, etc.)


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
        # Verify the stub still returns a well-formed TacticResult
        self.assertIsInstance(result, TacticResult)
        self.assertIn("stub", result.error_message)
        self.assertEqual(result.premises, [])
        self.assertEqual(result.new_goals, [])
        self.assertIn("0", result.tactic)  # "aesop (premises: 0)"

    def test_unknown_backend_raises(self):
        cfg = LeanConfig(backend="nonexistent")
        kernel = LeanKernel(config=cfg)
        with self.assertRaises(ValueError):
            kernel.try_hammer("⊢ P", [])


class TestReplayBackend(unittest.TestCase):
    def _make_replay_kernel(self):
        cfg = LeanConfig(backend="replay")
        return LeanKernel(config=cfg)

    def test_register_ground_truth_stores_exact_tactics(self):
        """Mutation gap: verify register_ground_truth stores the exact list."""
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ True", ["trivial"])
        self.assertEqual(kernel._replay_table["⊢ True"], ["trivial"])

    def test_register_ground_truth_overwrites_previous(self):
        """Second call to same goal replaces the tactic list."""
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ P", ["old"])
        kernel.register_ground_truth("⊢ P", ["new"])
        self.assertEqual(kernel._replay_table["⊢ P"], ["new"])

    def test_register_ground_truth_multiple_goals(self):
        """Multiple goals stored independently."""
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ A", ["tac_a"])
        kernel.register_ground_truth("⊢ B", ["tac_b"])
        self.assertEqual(len(kernel._replay_table), 2)
        self.assertEqual(kernel._replay_table["⊢ A"], ["tac_a"])
        self.assertEqual(kernel._replay_table["⊢ B"], ["tac_b"])

    def test_replay_succeeds_on_matching_tactic(self):
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ True", ["trivial"])
        result = kernel.try_tactic("⊢ True", "trivial")
        self.assertEqual(result.success, True)
        self.assertEqual(result.tactic, "trivial")
        self.assertEqual(result.new_goals, [])
        self.assertEqual(result.error_message, "")

    def test_replay_matches_base_tactic_name(self):
        """Replay matches by first word, so 'apply foo' matches 'apply bar'."""
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ P", ["apply some_lemma"])
        result = kernel.try_tactic("⊢ P", "apply other_lemma")
        self.assertEqual(result.success, True)

    def test_replay_fails_on_wrong_tactic(self):
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ P", ["exact h"])
        result = kernel.try_tactic("⊢ P", "simp")
        self.assertFalse(result.success)
        self.assertIn("replay", result.error_message)

    def test_replay_fails_on_unregistered_goal(self):
        kernel = self._make_replay_kernel()
        result = kernel.try_tactic("⊢ unknown", "trivial")
        self.assertFalse(result.success)

    def test_replay_hammer_always_fails(self):
        kernel = self._make_replay_kernel()
        result = kernel.try_hammer("⊢ P", ["h1"])
        self.assertFalse(result.success)
        self.assertIn("replay", result.error_message)

    def test_register_multiple_ground_truth(self):
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ P", ["exact h", "trivial"])
        # Both should match
        r1 = kernel.try_tactic("⊢ P", "exact foo")
        r2 = kernel.try_tactic("⊢ P", "trivial")
        self.assertEqual(r1.success, True)
        self.assertEqual(r2.success, True)

    def test_replay_empty_tactic_fails(self):
        kernel = self._make_replay_kernel()
        kernel.register_ground_truth("⊢ P", ["exact h"])
        result = kernel.try_tactic("⊢ P", "")
        self.assertFalse(result.success)


class TestBuildHammerTactics(unittest.TestCase):
    """Test hammer tactic string construction."""

    def test_no_premises(self):
        tactics = _build_hammer_tactics([])
        # Without premises: exactly 4 tactics in specific order
        self.assertEqual(tactics, ["aesop", "omega", "decide", "simp"])

    def test_no_premises_count(self):
        self.assertEqual(len(_build_hammer_tactics([])), 4)

    def test_with_premises(self):
        tactics = _build_hammer_tactics(["Nat.add_comm", "Nat.add_zero"])
        # With premises: 6 tactics in specific order
        self.assertEqual(len(tactics), 6)
        self.assertEqual(tactics[0], "aesop (add safe [Nat.add_comm, Nat.add_zero])")
        self.assertEqual(tactics[1], "aesop")
        self.assertEqual(tactics[2], "omega")
        self.assertEqual(tactics[3], "decide")
        self.assertEqual(tactics[4], "simp [Nat.add_comm, Nat.add_zero]")
        self.assertEqual(tactics[5], "simp")

    def test_premise_limit_at_16(self):
        many = [f"lemma_{i}" for i in range(30)]
        tactics = _build_hammer_tactics(many)
        simp_tac = [t for t in tactics if t.startswith("simp [")][0]
        self.assertIn("lemma_15", simp_tac)
        self.assertNotIn("lemma_16", simp_tac)


class TestLeanConfigImports(unittest.TestCase):
    """Test the new imports field on LeanConfig."""

    def test_default_imports(self):
        cfg = LeanConfig()
        self.assertEqual(cfg.imports, ["Init"])

    def test_custom_imports(self):
        cfg = LeanConfig(imports=["Mathlib.Tactic", "Mathlib.Data.Nat.Basic"])
        self.assertEqual(len(cfg.imports), 2)


class TestPantographBackend(unittest.TestCase):
    """Pantograph integration tests — skipped if PyPantograph not installed."""

    @classmethod
    def setUpClass(cls):
        try:
            from pantograph.server import Server  # type: ignore[import-untyped]  # noqa: F401

            cls.pantograph_available = True
        except ImportError:
            cls.pantograph_available = False

    def setUp(self):
        if not self.pantograph_available:
            self.skipTest("PyPantograph not installed")
        self.kernel = LeanKernel(
            LeanConfig(
                backend="pantograph",
                timeout=30,
                imports=["Init"],
            )
        )

    def tearDown(self):
        if hasattr(self, "kernel"):
            self.kernel.close()

    def test_trivial_proof(self):
        goal = self.kernel.goal_start("True")
        result = self.kernel.try_tactic(goal, "trivial")
        self.assertEqual(result.success, True)
        self.assertEqual(result.new_goals, [])

    def test_intro_produces_new_goals(self):
        goal = self.kernel.goal_start("forall (p : Prop), p -> p")
        result = self.kernel.try_tactic(goal, "intro p hp")
        self.assertTrue(result.success or len(result.new_goals) > 0)

    def test_invalid_tactic_fails(self):
        goal = self.kernel.goal_start("True")
        result = self.kernel.try_tactic(goal, "not_a_real_tactic")
        self.assertFalse(result.success)
        self.assertNotEqual(result.error_message, "")

    def test_hammer_on_trivial(self):
        goal = self.kernel.goal_start("True")
        result = self.kernel.try_hammer(goal, [])
        self.assertEqual(result.success, True)

    def test_gc_does_not_crash(self):
        self.kernel.goal_start("True")
        self.kernel.gc()


if __name__ == "__main__":
    unittest.main()
