"""Tests for proof_search — pure functions and dataclasses."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.nav_contracts import NavOutput
from src.proof_search import (
    _META_WRAPPER_TACTICS,
    SearchConfig,
    _apply_domain_lane_policy,
    _build_tactic_text,
    _classify_goal_family,
    _goal_signature,
    _is_self_application_tactic,
    _progress_pathology_tags,
    _SearchEnv,
    _SearchState,
    _should_hammer,
)
from src.resolution import Candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nav_output(automation=0, critic=0.5):
    """Build a NavOutput with a specified automation direction."""
    return NavOutput(
        directions={"automation": automation} if automation is not None else {},
        direction_confidences={"automation": 0.9},
        anchor_scores={},
        progress=0.5,
        critic_score=critic,
    )


def _make_candidate(tactic="apply", premises=None, score=0.9):
    """Build a Candidate with optional premises."""
    return Candidate(
        tactic_name=tactic,
        premises=premises or [],
        score=score,
    )


class _FakeSentenceEncoder:
    def encode(self, texts, normalize_embeddings=True):
        del texts, normalize_embeddings
        return np.full((1, 384), 0.25, dtype=np.float32)


class _FakeTorchFamilyClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, goal_emb, goal_shape, step_context):
        self.calls.append((goal_emb, goal_shape, step_context))
        return "solver"


# ---------------------------------------------------------------------------
# _should_hammer
# ---------------------------------------------------------------------------


class TestShouldHammer(unittest.TestCase):
    def test_automation_neg1_returns_true(self):
        nav = _make_nav_output(automation=-1)
        self.assertEqual(_should_hammer(nav), True)

    def test_automation_zero_returns_false(self):
        nav = _make_nav_output(automation=0)
        self.assertEqual(_should_hammer(nav), False)

    def test_automation_pos1_returns_false(self):
        nav = _make_nav_output(automation=1)
        self.assertEqual(_should_hammer(nav), False)

    def test_automation_key_missing_returns_false(self):
        nav = _make_nav_output(automation=None)  # directions dict is empty
        self.assertEqual(_should_hammer(nav), False)


# ---------------------------------------------------------------------------
# _build_tactic_text
# ---------------------------------------------------------------------------


class TestBuildTacticText(unittest.TestCase):
    def test_no_premises_returns_tactic_name(self):
        c = _make_candidate(tactic="rfl", premises=[])
        self.assertEqual(_build_tactic_text(c), "rfl")

    def test_one_premise(self):
        c = _make_candidate(tactic="exact", premises=["Nat.add_comm"])
        self.assertEqual(_build_tactic_text(c), "exact Nat.add_comm")

    def test_four_premises_all_included(self):
        prems = ["a", "b", "c", "d"]
        c = _make_candidate(tactic="apply", premises=prems)
        self.assertEqual(_build_tactic_text(c), "apply a b c d")

    def test_six_premises_only_first_four(self):
        prems = ["a", "b", "c", "d", "e", "f"]
        c = _make_candidate(tactic="simp", premises=prems)
        self.assertEqual(_build_tactic_text(c), "simp a b c d")


# ---------------------------------------------------------------------------
# _SearchState dataclass
# ---------------------------------------------------------------------------


class TestSearchState(unittest.TestCase):
    def test_open_goals_required(self):
        state = _SearchState(open_goals=["goal1"])
        self.assertEqual(state.open_goals, ["goal1"])

    def test_defaults_empty(self):
        state = _SearchState(open_goals=["g"])
        self.assertEqual(state.closed_goals, [])
        self.assertEqual(state.tactics_used, [])
        self.assertEqual(state.attempts, 0)

    def test_default_factory_isolation(self):
        """Two instances must have independent lists."""
        s1 = _SearchState(open_goals=["a"])
        s2 = _SearchState(open_goals=["b"])
        s1.closed_goals.append("x")
        s1.tactics_used.append("t")
        self.assertEqual(s2.closed_goals, [])
        self.assertEqual(s2.tactics_used, [])


# ---------------------------------------------------------------------------
# SearchConfig dataclass
# ---------------------------------------------------------------------------


class TestSearchConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.budget, 600)
        self.assertEqual(cfg.hammer_delegation, True)
        self.assertEqual(cfg.accessible_premises, True)
        self.assertEqual(cfg.max_candidates_per_step, 8)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.family_classifier_path, "")
        self.assertEqual(cfg.family_classifier_torch_path, "")
        self.assertEqual(cfg.norm_then_close_enabled, True)
        self.assertEqual(cfg.max_progress_steps, 0)
        self.assertEqual(cfg.allow_self_application, False)
        self.assertEqual(cfg.metavariable_penalty_enabled, True)
        self.assertEqual(cfg.state_loop_penalty_enabled, True)
        self.assertEqual(cfg.state_loop_window, 3)

    def test_self_application_tactic_helper(self):
        self.assertTrue(
            _is_self_application_tactic(
                "exact ArithmeticFunction.sum_moebius_mul_log_eq",
                "ArithmeticFunction.sum_moebius_mul_log_eq",
            )
        )
        self.assertFalse(
            _is_self_application_tactic(
                "exact ArithmeticFunction.log_mul_moebius_eq_vonMangoldt",
                "ArithmeticFunction.sum_moebius_mul_log_eq",
            )
        )
        self.assertFalse(
            _is_self_application_tactic(
                "exact Foo.barbaz x",
                "Foo.bar",
            )
        )
        self.assertTrue(
            _is_self_application_tactic(
                "exact WithBot.eq_top_iff_forall_ge.mpr a",
                "WithBot.eq_top_iff_forall_ge",
            )
        )


class TestFamilyClassifier(unittest.TestCase):
    def _make_env(self, **overrides):
        env = _SearchEnv(
            conn=MagicMock(),
            lean=MagicMock(),
            anchor_id_map=None,
            max_candidates=8,
            sentence_encoder=_FakeSentenceEncoder(),
        )
        for key, value in overrides.items():
            setattr(env, key, value)
        return env

    def test_numpy_family_classifier_fallback(self):
        env = self._make_env(
            family_classifier={
                "W1": np.zeros((400, 1), dtype=np.float32),
                "b1": np.zeros((1,), dtype=np.float32),
                "W2": np.zeros((1, 5), dtype=np.float32),
                "b2": np.array([0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32),
            }
        )

        predicted = _classify_goal_family("⊢ x ≤ y", env, SearchConfig())
        self.assertEqual(predicted, "solver")

    def test_torch_family_classifier_takes_precedence(self):
        torch_clf = _FakeTorchFamilyClassifier()
        env = self._make_env(
            family_classifier={
                "W1": np.zeros((400, 1), dtype=np.float32),
                "b1": np.zeros((1,), dtype=np.float32),
                "W2": np.zeros((1, 5), dtype=np.float32),
                "b2": np.array([3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            family_classifier_torch=torch_clf,
        )

        predicted = _classify_goal_family("⊢ x ≤ y", env, SearchConfig())
        self.assertEqual(predicted, "solver")
        self.assertEqual(len(torch_clf.calls), 1)


class TestDomainLanePolicy(unittest.TestCase):
    def test_structural_domains_prefer_exact_before_rw(self):
        lanes = _apply_domain_lane_policy(
            "AlgebraicGeometry.AffineSpace.isOpenMap_over",
            "IsOpenMap (ConcreteCategory.hom (AffineSpace n S ↘ S).base)",
            ["structural_core", "cosine_rw", "cosine_exact", "learned"],
        )
        self.assertEqual(lanes[0], "cosine_exact")
        self.assertLess(lanes.index("cosine_exact"), lanes.index("cosine_rw"))

    def test_membership_wall_promotes_exact(self):
        lanes = _apply_domain_lane_policy(
            "AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier.smul_mem",
            "c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q",
            ["automation", "structural_core", "cosine_rw", "cosine_exact"],
        )
        self.assertEqual(lanes[0], "cosine_exact")

    def test_replay_prefixed_goal_still_gets_structural_policy(self):
        lanes = _apply_domain_lane_policy(
            "CategoryTheory.Arrow.AugmentedCechNerve.ExtraDegeneracy.s_comp_π_succ",
            "_wayfinder_replay__wayfinder_decl_0.CategoryTheory.Arrow.AugmentedCechNerve.ExtraDegeneracy.s f S n ≫ g = 0",
            ["automation", "structural_core", "cosine_rw", "cosine_exact"],
        )
        self.assertEqual(lanes[0], "cosine_exact")


class TestSearchPathologies(unittest.TestCase):
    def test_progress_pathology_detects_metavar_and_loop(self):
        goals_before = ["⊢ AkraBazziRecurrence.GrowsPolynomially fun x => |f x|"]
        goals_after = [
            "⊢ AkraBazziRecurrence.GrowsPolynomially ?m.6",
            "⊢ ?m.6 =ᶠ[Filter.atTop] fun x => |f x|",
            "⊢ ℝ → ℝ",
        ]
        tags = _progress_pathology_tags(
            goals_before=goals_before,
            goals_after=goals_after,
            tactic="rw [← AkraBazziRecurrence.GrowsPolynomially.iff_eventuallyEq]",
            recent_signatures=[_goal_signature(goals_before), _goal_signature(goals_after)],
            loop_window=3,
        )
        self.assertIn("metavariable_corruption", tags)
        self.assertIn("backward_rewrite_metavariable", tags)
        self.assertIn("state_loop", tags)

    def test_progress_pathology_detects_duplicate_goal_pseudoprogress(self):
        tags = _progress_pathology_tags(
            goals_before=["⊢ F''.IsLeftKanExtension τ", "⊢ F'.IsLeftKanExtension α"],
            goals_after=[
                "⊢ F''.IsLeftKanExtension τ",
                "⊢ F''.IsLeftKanExtension τ",
                "⊢ F'.IsLeftKanExtension α",
            ],
            tactic="norm_num",
            recent_signatures=[],
            loop_window=3,
        )
        self.assertIn("duplicate_goal_progress", tags)
        self.assertIn("duplicate_goal_pseudo_progress", tags)


class TestMetaWrapperTactics(unittest.TestCase):
    def test_meta_wrapper_tactic_bank_present(self):
        self.assertIn("dsimp only [autoParam, optParam]", _META_WRAPPER_TACTICS)


if __name__ == "__main__":
    unittest.main()
