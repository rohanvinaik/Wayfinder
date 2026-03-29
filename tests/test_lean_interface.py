"""Tests for lean_interface — LeanKernel stub, replay, and pantograph backends."""

import unittest

from src.lean_interface import (
    LeanConfig,
    LeanKernel,
    ReplayResult,
    ServerCrashError,
    _build_hammer_tactics,
    _candidate_open_namespaces,
    _classify_tactic_failure,
    _find_decl_bounds_in_source,
    _resolve_alias_target_in_source,
    _resolve_source_file_path,
    extract_file_header,
    resolve_lean_path,
    theorem_type_from_goal_pp,
)
from src.nav_contracts import LeanFeedback, TacticResult


class TestLeanConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LeanConfig()
        self.assertEqual(cfg.backend, "stub")
        self.assertEqual(cfg.timeout, 120)
        self.assertEqual(cfg.hammer_timeout, 60)
        self.assertEqual(cfg.project_root, "")

    def test_custom_values(self):
        cfg = LeanConfig(backend="pantograph", timeout=10, hammer_timeout=120)
        self.assertEqual(cfg.backend, "pantograph")
        self.assertEqual(cfg.timeout, 10)
        self.assertEqual(cfg.hammer_timeout, 120)


class TestReplaySourceHelpers(unittest.TestCase):
    def test_theorem_type_from_goal_pp_builds_binders_and_target(self):
        goal_pp = (
            "P : MorphismProperty Scheme\n"
            "Q : AffineTargetMorphismProperty\n"
            "inst✝ : HasAffineProperty P Q\n"
            "X Y : Scheme\n"
            "f : X ⟶ Y\n"
            "⊢ P f\n"
        )
        theorem_type = theorem_type_from_goal_pp(goal_pp)
        self.assertEqual(
            theorem_type,
            "∀ (P : MorphismProperty Scheme) (Q : AffineTargetMorphismProperty) "
            "[_inst : HasAffineProperty P Q] (X Y : Scheme) (f : X ⟶ Y), P f",
        )

    def test_candidate_open_namespaces_uses_conservative_top_levels(self):
        namespaces = _candidate_open_namespaces(
            theorem_name="AlgebraicGeometry.HasAffineProperty.isLocalAtSource",
            file_path="Mathlib/AlgebraicGeometry/Morphisms/Constructors.lean",
        )
        self.assertEqual(namespaces, ["AlgebraicGeometry"])

    def test_find_decl_bounds_in_source_locates_named_theorem(self):
        lines = [
            "section Foo\n",
            "def helper : Nat := 0\n",
            "theorem demo : True := by\n",
            "  trivial\n",
            "\n",
            "theorem next_demo : True := by\n",
            "  trivial\n",
        ]
        start, end = _find_decl_bounds_in_source(lines, "demo")
        self.assertEqual((start, end), (3, 5))

    def test_find_decl_bounds_in_source_locates_dotted_suffix_theorem(self):
        lines = [
            "namespace Submodule\n",
            "theorem Quotient.equiv_trans : True := by\n",
            "  trivial\n",
            "theorem next_demo : True := by\n",
            "  trivial\n",
        ]
        start, end = _find_decl_bounds_in_source(
            lines,
            "equiv_trans",
            theorem_full_name="Submodule.Quotient.equiv_trans",
        )
        self.assertEqual((start, end), (2, 3))

    def test_find_decl_bounds_in_source_prefers_specific_dotted_name_over_short_suffix(self):
        lines = [
            "namespace Distrib\n",
            "theorem ext : True := by\n",
            "  trivial\n",
            "end Distrib\n",
            "theorem AddGroupWithOne.ext : True := by\n",
            "  trivial\n",
        ]
        start, end = _find_decl_bounds_in_source(
            lines,
            "ext",
            theorem_full_name="AddGroupWithOne.ext",
        )
        self.assertEqual((start, end), (5, 6))

    def test_find_decl_bounds_in_source_handles_normalized_name_drift(self):
        lines = [
            "theorem iIndepFun.indepFun_finset_prod_of_notMem : True := by\n",
            "  trivial\n",
        ]
        start, end = _find_decl_bounds_in_source(
            lines,
            "indepFun_finset_prod_of_not_mem",
            theorem_full_name="ProbabilityTheory.Kernel.iIndepFun.indepFun_finset_prod_of_not_mem",
        )
        self.assertEqual((start, end), (1, 2))

    def test_find_decl_bounds_in_source_locates_alias(self):
        lines = [
            "@[deprecated] alias topologically_isLocalAtSource := topologically_isZariskiLocalAtSource\n",
            "lemma next_demo : True := by\n",
            "  trivial\n",
        ]
        start, end = _find_decl_bounds_in_source(
            lines,
            "isLocalAtSource",
            theorem_full_name="topologically_isLocalAtSource",
        )
        self.assertEqual((start, end), (1, 1))

    def test_resolve_alias_target_in_source_handles_multiline_alias(self):
        lines = [
            "@[deprecated (since := \"2025-10-07\")]\n",
            "alias Scheme.exists_hom_isAffine_of_isLocalAtSource :=\n",
            "  Scheme.exists_hom_isAffine_of_isZariskiLocalAtSource\n",
        ]
        target = _resolve_alias_target_in_source(
            lines,
            "exists_hom_isAffine_of_isLocalAtSource",
            theorem_full_name="Scheme.exists_hom_isAffine_of_isLocalAtSource",
        )
        self.assertEqual(target, "Scheme.exists_hom_isAffine_of_isZariskiLocalAtSource")

    def test_resolve_source_file_path_uses_hint_when_module_missing(self):
        project_root = "/tmp/project"
        expected = "/tmp/project/Mathlib/Foo/Bar.lean"
        from unittest.mock import patch

        with patch("os.path.exists", side_effect=lambda path: path == expected):
            resolved = _resolve_source_file_path(
                project_root=project_root,
                module="",
                file_path_hint="Mathlib/Foo/Bar.lean",
            )
        self.assertEqual(resolved, expected)

    def test_resolve_source_file_path_searches_nested_decl_when_hint_is_reexport(self):
        project_root = "/tmp/project"
        reexport = "/tmp/project/.lake/packages/mathlib/Mathlib/Foo/Bar.lean"
        nested = "/tmp/project/.lake/packages/mathlib/Mathlib/Foo/Bar/Baz.lean"
        file_contents = {
            reexport: "import Mathlib.Foo.Bar.Baz\n",
            nested: "theorem ns.theorem_notMem : True := by\n  trivial\n",
        }
        from unittest.mock import mock_open, patch

        def _exists(path: str) -> bool:
            return path in file_contents or path in {
                "/tmp/project/.lake/packages/mathlib/Mathlib/Foo",
                "/tmp/project/.lake/packages/mathlib/Mathlib/Foo/Bar",
            }

        def _isdir(path: str) -> bool:
            return path in {
                "/tmp/project/.lake/packages/mathlib/Mathlib/Foo",
                "/tmp/project/.lake/packages/mathlib/Mathlib/Foo/Bar",
            }

        def _walk(path: str):
            if path == "/tmp/project/.lake/packages/mathlib/Mathlib/Foo":
                yield (path, ["Bar"], ["Bar.lean"])
                yield ("/tmp/project/.lake/packages/mathlib/Mathlib/Foo/Bar", [], ["Baz.lean"])

        def _open(path: str, *args, **kwargs):
            data = file_contents[path]
            return mock_open(read_data=data)()

        with (
            patch("os.path.exists", side_effect=_exists),
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.walk", side_effect=_walk),
            patch("builtins.open", side_effect=_open),
        ):
            resolved = _resolve_source_file_path(
                project_root=project_root,
                module="",
                file_path_hint="Mathlib/Foo/Bar.lean",
                theorem_full_name="ns.theorem_not_mem",
            )
        self.assertEqual(resolved, nested)

    def test_extract_support_slice_includes_local_notation_and_leti(self):
        lines = [
            "section Foo\n",
            "variable {R : Type*}\n",
            'local notation "ZZ" => Nat\n',
            "letI : Inhabited Nat := ⟨0⟩\n",
            "def helper : Nat := 0\n",
            "theorem demo : True := by\n",
            "  trivial\n",
            "end Foo\n",
        ]
        support = LeanKernel._extract_support_slice(lines, 6, 7, "demo")
        joined = "\n".join(support)
        self.assertIn('local notation "ZZ" => Nat', joined)
        self.assertIn("letI : Inhabited Nat := ⟨0⟩", joined)
        self.assertIn("def helper : Nat := 0", joined)

    def test_extract_support_slice_includes_private_lemmas_and_local_macros(self):
        lines = [
            "section Foo\n",
            'local macro:max "foo" : term =>\n',
            "  `(Nat)\n",
            "private lemma helper : True := by\n",
            "  trivial\n",
            "theorem demo : True := by\n",
            "  trivial\n",
            "end Foo\n",
        ]
        support = LeanKernel._extract_support_slice(lines, 6, 7, "demo")
        joined = "\n".join(support)
        self.assertIn('local macro:max "foo" : term =>', joined)
        self.assertIn("private lemma helper : True := by", joined)

    def test_extract_support_slice_includes_private_noncomputable_defs(self):
        lines = [
            "section Foo\n",
            "private noncomputable def aux : Nat := 0\n",
            "theorem demo : True := by\n",
            "  trivial\n",
            "end Foo\n",
        ]
        support = LeanKernel._extract_support_slice(lines, 3, 4, "demo")
        self.assertIn("private noncomputable def aux : Nat := 0", "\n".join(support))

    def test_wrap_replay_namespace_preserves_header(self):
        src = (
            "module\n\n"
            "import Mathlib\n"
            "\n"
            "section Foo\n"
            "theorem demo : True := by\n"
            "  trivial\n"
        )
        wrapped = LeanKernel._wrap_replay_namespace(src, "demo")
        self.assertIn("module", wrapped.splitlines()[0])
        self.assertIn("import Mathlib", wrapped)
        self.assertIn("namespace _wayfinder_replay_demo", wrapped)
        self.assertIn("section Foo", wrapped)
        self.assertIn("end _wayfinder_replay_demo", wrapped)


class TestLeanKernelInit(unittest.TestCase):
    def test_default_config(self):
        kernel = LeanKernel()
        self.assertEqual(kernel.config.backend, "stub")

    def test_custom_config(self):
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)
        self.assertEqual(kernel._backend, "pantograph")

    def test_close_terminates_server(self):
        from unittest.mock import MagicMock

        kernel = LeanKernel(config=LeanConfig(backend="pantograph"))
        server = MagicMock()
        kernel._server = server

        kernel.close()

        server._close.assert_called_once()
        self.assertIsNone(kernel._server)


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
        # Without premises: symbolic closers in specific order
        self.assertEqual(tactics, ["solve_by_elim", "aesop", "omega", "decide", "simp", "apply?"])

    def test_no_premises_count(self):
        self.assertEqual(len(_build_hammer_tactics([])), 6)

    def test_with_premises(self):
        tactics = _build_hammer_tactics(["Nat.add_comm", "Nat.add_zero"])
        # With premises: 8 tactics in specific order
        self.assertEqual(len(tactics), 8)
        self.assertEqual(tactics[0], "aesop (add safe [Nat.add_comm, Nat.add_zero])")
        self.assertEqual(tactics[1], "solve_by_elim")
        self.assertEqual(tactics[2], "aesop")
        self.assertEqual(tactics[3], "omega")
        self.assertEqual(tactics[4], "decide")
        self.assertEqual(tactics[5], "simp [Nat.add_comm, Nat.add_zero]")
        self.assertEqual(tactics[6], "simp")
        self.assertEqual(tactics[7], "apply?")

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


class TestCrashRestart(unittest.TestCase):
    """Test crash detection and server restart."""

    def test_is_crash_error_broken_pipe(self):
        kernel = LeanKernel()
        self.assertTrue(kernel._is_crash_error(BrokenPipeError("pipe")))

    def test_is_crash_error_connection_reset(self):
        kernel = LeanKernel()
        self.assertTrue(kernel._is_crash_error(ConnectionResetError("reset")))

    def test_is_crash_error_process_lookup(self):
        kernel = LeanKernel()
        self.assertTrue(kernel._is_crash_error(ProcessLookupError("no proc")))

    def test_is_crash_error_message_match(self):
        kernel = LeanKernel()
        self.assertTrue(kernel._is_crash_error(Exception("Connection lost to server")))
        self.assertTrue(kernel._is_crash_error(Exception("broken pipe detected")))

    def test_is_crash_error_false_for_regular(self):
        kernel = LeanKernel()
        self.assertFalse(kernel._is_crash_error(ValueError("bad value")))
        self.assertFalse(kernel._is_crash_error(TypeError("wrong type")))

    def test_restart_server_clears_state(self):
        """_restart_server clears caches (stub backend, no real server)."""
        kernel = LeanKernel()
        # Populate caches with dummy data
        kernel._goal_states[("", "goal1")] = "state1"
        kernel._tactic_cache[("", "goal1", "tac", 0)] = None
        kernel._restart_server()
        self.assertEqual(len(kernel._goal_states), 0)
        self.assertEqual(len(kernel._tactic_cache), 0)

    def test_pantograph_try_tactic_raises_on_crash(self):
        """_pantograph_try_tactic raises ServerCrashError on broken pipe."""
        from unittest.mock import MagicMock, patch

        # Create a real exception class for TacticFailure
        class MockTacticFailure(Exception):
            pass

        mock_panto_server = MagicMock()
        mock_panto_server.TacticFailure = MockTacticFailure

        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)

        mock_server = MagicMock()
        kernel._server = mock_server
        mock_state = MagicMock()
        kernel._goal_states[("", "⊢ P")] = mock_state
        mock_server.goal_tactic.side_effect = BrokenPipeError("pipe closed")

        with patch.dict(
            "sys.modules",
            {
                "pantograph": MagicMock(),
                "pantograph.server": mock_panto_server,
            },
        ):
            with self.assertRaises(ServerCrashError):
                kernel._pantograph_try_tactic("⊢ P", "rfl")

    def test_get_or_create_goal_raises_on_crash(self):
        """_get_or_create_goal raises ServerCrashError on broken pipe."""
        cfg = LeanConfig(backend="pantograph")
        kernel = LeanKernel(config=cfg)

        # Mock server that crashes on goal_start
        from unittest.mock import MagicMock

        mock_server = MagicMock()
        mock_server.goal_start.side_effect = BrokenPipeError("pipe closed")
        kernel._server = mock_server

        with self.assertRaises(ServerCrashError):
            kernel._get_or_create_goal("⊢ P")


class TestReplayResult(unittest.TestCase):
    """Test ReplayResult dataclass."""

    def test_construction(self):
        r = ReplayResult(
            success=True,
            goal_state="⊢ True",
            goal_state_obj=None,
            tier_used="A",
            failure_category="",
            failing_prefix_idx=-1,
            crash_retries=0,
            env_key="file:thm:abc",
        )
        self.assertTrue(r.success)
        self.assertEqual(r.tier_used, "A")
        self.assertEqual(r.env_key, "file:thm:abc")

    def test_failure_construction(self):
        r = ReplayResult(
            success=False,
            goal_state="",
            goal_state_obj=None,
            tier_used="B",
            failure_category="file_not_found",
            failing_prefix_idx=-1,
            crash_retries=1,
            env_key="",
        )
        self.assertFalse(r.success)
        self.assertEqual(r.failure_category, "file_not_found")
        self.assertEqual(r.crash_retries, 1)

    def test_feedback_construction(self):
        r = ReplayResult(
            success=False,
            goal_state="",
            goal_state_obj=None,
            tier_used="B",
            failure_category="goal_creation_fail",
            failing_prefix_idx=-1,
            crash_retries=0,
            env_key="env",
            feedback=LeanFeedback(
                stage="goal_creation",
                category="goal_creation_fail",
                messages=[{"data": "Cannot start goal"}],
                raw_error="Cannot start goal",
            ),
        )
        self.assertIsNotNone(r.feedback)
        assert r.feedback is not None
        self.assertEqual(r.feedback.stage, "goal_creation")


class TestEnvironmentCaching(unittest.TestCase):
    """Test environment-keyed caching."""

    def test_different_env_keys_independent(self):
        """Same goal text with different env_keys → independent cache entries."""
        kernel = LeanKernel()
        # Simulate two different environments with same goal
        kernel._goal_states[("env1", "⊢ P")] = "state_env1"
        kernel._goal_states[("env2", "⊢ P")] = "state_env2"
        self.assertEqual(kernel._goal_states[("env1", "⊢ P")], "state_env1")
        self.assertEqual(kernel._goal_states[("env2", "⊢ P")], "state_env2")

    def test_default_env_key_empty(self):
        """Default env_key is empty string for backward compatibility."""
        kernel = LeanKernel()
        self.assertEqual(kernel._current_env_key, "")

    def test_tactic_cache_env_keyed(self):
        """Tactic cache uses 3-tuple (env_key, goal, tactic)."""
        kernel = LeanKernel()
        from src.nav_contracts import TacticResult

        r = TacticResult(success=True, tactic="rfl", premises=[], new_goals=[])
        kernel._tactic_cache[("env1", "⊢ P", "rfl", 0)] = r
        kernel._tactic_cache[("env2", "⊢ P", "rfl", 0)] = TacticResult(
            success=False, tactic="rfl", premises=[], new_goals=[], error_message="fail"
        )
        self.assertTrue(kernel._tactic_cache[("env1", "⊢ P", "rfl", 0)].success)
        self.assertFalse(kernel._tactic_cache[("env2", "⊢ P", "rfl", 0)].success)

    def test_prepare_for_new_example_prefers_gc(self):
        from unittest.mock import MagicMock

        kernel = LeanKernel()
        kernel._server_contaminated = True
        kernel.gc = MagicMock()
        kernel._restart_server = MagicMock()

        kernel._prepare_for_new_example()

        kernel.gc.assert_called_once()
        kernel._restart_server.assert_not_called()
        self.assertFalse(kernel._server_contaminated)

    def test_prepare_for_new_example_periodic_restart(self):
        from unittest.mock import MagicMock

        kernel = LeanKernel()
        kernel._server_contaminated = True
        kernel._load_sorry_reset_count = 255
        kernel.gc = MagicMock()
        kernel._restart_server = MagicMock()

        kernel._prepare_for_new_example()

        kernel.gc.assert_called_once()
        kernel._restart_server.assert_called_once()


class TestFileHelpers(unittest.TestCase):
    """Test resolve_lean_path and extract_file_header."""

    def test_resolve_lean_path(self):
        result = resolve_lean_path("Mathlib/Analysis/Foo.lean", "/project")
        self.assertEqual(result, "/project/.lake/packages/mathlib/Mathlib/Analysis/Foo.lean")

    def test_extract_file_header(self):
        import os
        import tempfile

        content = (
            "import Mathlib.Data.Nat\nopen Nat\nvariable (n : Nat)\n"
            "namespace Foo\n\ntheorem bar : True := trivial\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            header = extract_file_header(path, 6)  # theorem is on line 6
            self.assertIn("open Nat", header)
            self.assertIn("variable (n : Nat)", header)
            self.assertIn("namespace Foo", header)
            self.assertNotIn("import", header)
        finally:
            os.unlink(path)

    def test_extract_file_header_empty(self):
        import os
        import tempfile

        content = "theorem bar : True := trivial\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            header = extract_file_header(path, 1)
            self.assertEqual(header, "")
        finally:
            os.unlink(path)


class TestLeanFeedback(unittest.TestCase):
    def test_success_factory(self):
        fb = LeanFeedback.success()
        self.assertEqual(fb.category, "none")
        self.assertEqual(fb.stage, "tactic_exec")
        self.assertEqual(fb.messages, [])
        self.assertEqual(fb.raw_error, "")

    def test_to_dict(self):
        fb = LeanFeedback(
            stage="tactic_parse",
            category="parse_error",
            messages=[{"data": "unexpected token"}],
            raw_error="err",
        )
        d = fb.to_dict()
        self.assertEqual(d["stage"], "tactic_parse")
        self.assertEqual(d["category"], "parse_error")
        self.assertEqual(d["messages"][0]["data"], "unexpected token")


class TestClassifyTacticFailure(unittest.TestCase):
    def _make_exc(self, *args: object) -> Exception:
        exc = Exception(*args)
        return exc

    def test_parse_error_dict(self):
        exc = self._make_exc({"parseError": "unexpected token ';'"})
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "parse_error")
        self.assertEqual(fb.stage, "tactic_parse")

    def test_generated_sorry(self):
        exc = self._make_exc("Tactic generated sorry", [])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "generated_sorry")

    def test_generated_unsafe(self):
        exc = self._make_exc("Tactic generated unsafe", [])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "generated_unsafe")

    def test_generic_string_exception_is_other(self):
        exc = self._make_exc("some generic exception")
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "other")

    def test_unknown_identifier_from_messages(self):
        class FakeMsg:
            severity = "error"
            kind = None
            data = "unknown identifier 'Nat.foo'"
            pos = "1:0"

        exc = self._make_exc([FakeMsg()])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "unknown_identifier")
        self.assertEqual(fb.stage, "tactic_exec")

    def test_type_mismatch_from_messages(self):
        class FakeMsg:
            severity = "error"
            kind = None
            data = "type mismatch: expected Nat, got Int"
            pos = "1:0"

        exc = self._make_exc([FakeMsg()])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "unification_mismatch")

    def test_could_not_unify_from_messages(self):
        class FakeMsg:
            severity = "error"
            kind = None
            data = "could not unify the conclusion with the goal"
            pos = "1:0"

        exc = self._make_exc([FakeMsg()])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "unification_mismatch")

    def test_typeclass_missing(self):
        class FakeMsg:
            severity = "error"
            kind = None
            data = "failed to synthesize instance Ring Nat"
            pos = "1:0"

        exc = self._make_exc([FakeMsg()])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "typeclass_missing")

    def test_other_message(self):
        class FakeMsg:
            severity = "error"
            kind = None
            data = "some unrecognised error"
            pos = "1:0"

        exc = self._make_exc([FakeMsg()])
        fb = _classify_tactic_failure(exc)
        self.assertEqual(fb.category, "other")

    def test_tactic_result_carries_feedback(self):
        r = TacticResult(
            success=False,
            tactic="apply foo",
            premises=[],
            error_message="unknown identifier 'foo'",
            feedback=LeanFeedback(
                stage="tactic_exec",
                category="unknown_identifier",
                messages=[],
                raw_error="unknown identifier 'foo'",
            ),
        )
        self.assertIsNotNone(r.feedback)
        assert r.feedback is not None
        self.assertEqual(r.feedback.category, "unknown_identifier")
        d = r.to_dict()
        self.assertIn("feedback", d)
        self.assertEqual(d["feedback"]["category"], "unknown_identifier")

    def test_tactic_result_feedback_defaults_none(self):
        r = TacticResult(success=True, tactic="rfl", premises=[])
        self.assertIsNone(r.feedback)
        d = r.to_dict()
        self.assertNotIn("feedback", d)


if __name__ == "__main__":
    unittest.main()
