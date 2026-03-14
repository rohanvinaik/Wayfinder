"""Tests for SoM data contracts — construction, defaults, isolation, roundtrips."""

import unittest

import torch

from src.som_contracts import (
    CensorPrediction,
    ExecutionOutput,
    PerceptionOutput,
    PlanningOutput,
    RecognitionOutput,
    SubgoalSpec,
    TemplateInfo,
    VerificationOutput,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_perception(**overrides):
    defaults = dict(embedding=torch.zeros(384), in_domain=True)
    defaults.update(overrides)
    return PerceptionOutput(**defaults)


def _make_template_info(**overrides):
    defaults = dict(
        template_id="tpl_001",
        pattern="apply_then_simp",
        bank_signature={"structure": 1, "domain": 0},
    )
    defaults.update(overrides)
    return TemplateInfo(**defaults)


def _make_recognition(**overrides):
    defaults = dict(
        template_id=3,
        template_name="induction",
        template_confidence=0.92,
        template_features=torch.randn(64),
    )
    defaults.update(overrides)
    return RecognitionOutput(**defaults)


def _make_subgoal(**overrides):
    defaults = dict(subgoal_type="simplify")
    defaults.update(overrides)
    return SubgoalSpec(**defaults)


def _make_planning(**overrides):
    defaults = dict(
        sketch=[_make_subgoal(), _make_subgoal(subgoal_type="apply")],
        total_estimated_depth=4,
        template_id=1,
    )
    defaults.update(overrides)
    return PlanningOutput(**defaults)


def _make_execution(**overrides):
    defaults = dict(
        directions={"structure": 1, "domain": -1},
        direction_confidences={"structure": 0.95, "domain": 0.8},
        anchor_logits=torch.randn(100),
        progress=0.6,
        critic=0.75,
    )
    defaults.update(overrides)
    return ExecutionOutput(**defaults)


def _make_verification(**overrides):
    defaults = dict(success=True)
    defaults.update(overrides)
    return VerificationOutput(**defaults)


def _make_censor(**overrides):
    defaults = dict(should_prune=False, failure_probability=0.1)
    defaults.update(overrides)
    return CensorPrediction(**defaults)


# ---------------------------------------------------------------------------
# PerceptionOutput
# ---------------------------------------------------------------------------


class TestPerceptionOutput(unittest.TestCase):
    def test_construction_with_defaults(self):
        p = _make_perception()
        self.assertEqual(p.embedding.shape, (384,))
        self.assertTrue(p.in_domain)

    def test_construction_out_of_domain(self):
        p = _make_perception(in_domain=False)
        self.assertFalse(p.in_domain)

    def test_embedding_tensor_dtype(self):
        p = _make_perception(embedding=torch.ones(384, dtype=torch.float32))
        self.assertEqual(p.embedding.dtype, torch.float32)
        self.assertAlmostEqual(p.embedding.sum().item(), 384.0, places=2)


# ---------------------------------------------------------------------------
# TemplateInfo
# ---------------------------------------------------------------------------


class TestTemplateInfo(unittest.TestCase):
    def test_construction_with_defaults(self):
        t = _make_template_info()
        self.assertEqual(t.template_id, "tpl_001")
        self.assertEqual(t.pattern, "apply_then_simp")
        self.assertEqual(t.tactic_patterns, [])
        self.assertFalse(t.is_simple)

    def test_construction_explicit(self):
        t = _make_template_info(
            tactic_patterns=["apply", "simp"],
            is_simple=True,
        )
        self.assertEqual(t.tactic_patterns, ["apply", "simp"])
        self.assertTrue(t.is_simple)

    def test_to_dict_roundtrip(self):
        t = _make_template_info(
            tactic_patterns=["rw", "ring"],
            is_simple=True,
        )
        d = t.to_dict()
        # Verify exact dict values before round-trip
        self.assertEqual(d["template_id"], "tpl_001")
        self.assertEqual(d["pattern"], "apply_then_simp")
        self.assertEqual(d["bank_signature"], {"structure": 1, "domain": 0})
        self.assertEqual(d["tactic_patterns"], ["rw", "ring"])
        self.assertTrue(d["is_simple"])
        # Verify round-trip
        t2 = TemplateInfo.from_dict(d)
        self.assertEqual(t2.template_id, "tpl_001")
        self.assertEqual(t2.pattern, "apply_then_simp")
        self.assertEqual(t2.bank_signature, {"structure": 1, "domain": 0})
        self.assertEqual(t2.tactic_patterns, ["rw", "ring"])
        self.assertTrue(t2.is_simple)

    def test_from_dict_missing_optional_keys(self):
        d = {
            "template_id": "tpl_min",
            "pattern": "exact",
            "bank_signature": {"depth": -1},
        }
        t = TemplateInfo.from_dict(d)
        self.assertEqual(t.template_id, "tpl_min")
        self.assertEqual(t.tactic_patterns, [])
        self.assertFalse(t.is_simple)

    def test_field_isolation_tactic_patterns(self):
        t1 = _make_template_info()
        t2 = _make_template_info()
        t1.tactic_patterns.append("omega")
        self.assertEqual(t2.tactic_patterns, [])


# ---------------------------------------------------------------------------
# RecognitionOutput
# ---------------------------------------------------------------------------


class TestRecognitionOutput(unittest.TestCase):
    def test_construction_with_defaults(self):
        r = _make_recognition()
        self.assertEqual(r.template_id, 3)
        self.assertEqual(r.template_name, "induction")
        self.assertAlmostEqual(r.template_confidence, 0.92, places=2)
        self.assertEqual(r.template_features.shape, (64,))
        self.assertEqual(r.top_k_templates, [])

    def test_construction_with_top_k(self):
        top_k = [(3, "induction", 0.92), (7, "apply_chain", 0.85)]
        r = _make_recognition(top_k_templates=top_k)
        self.assertEqual(len(r.top_k_templates), 2)
        self.assertEqual(r.top_k_templates[0][1], "induction")
        self.assertAlmostEqual(r.top_k_templates[1][2], 0.85, places=2)

    def test_field_isolation_top_k(self):
        r1 = _make_recognition()
        r2 = _make_recognition()
        r1.top_k_templates.append((0, "test", 0.5))
        self.assertEqual(r2.top_k_templates, [])

    def test_template_features_shape(self):
        feats = torch.randn(128)
        r = _make_recognition(template_features=feats)
        self.assertEqual(r.template_features.shape, (128,))


# ---------------------------------------------------------------------------
# SubgoalSpec
# ---------------------------------------------------------------------------


class TestSubgoalSpec(unittest.TestCase):
    def test_construction_with_defaults(self):
        s = _make_subgoal()
        self.assertEqual(s.subgoal_type, "simplify")
        self.assertEqual(s.anchor_targets, [])
        self.assertEqual(s.estimated_steps, 1)
        self.assertEqual(s.bank_hints, {})

    def test_construction_explicit(self):
        s = _make_subgoal(
            anchor_targets=["Nat.add_comm", "Nat.succ_pred"],
            estimated_steps=3,
            bank_hints={"structure": 1, "automation": -1},
        )
        self.assertEqual(len(s.anchor_targets), 2)
        self.assertEqual(s.estimated_steps, 3)
        self.assertEqual(s.bank_hints["automation"], -1)

    def test_to_dict_roundtrip(self):
        s = _make_subgoal(
            anchor_targets=["List.map"],
            estimated_steps=2,
            bank_hints={"depth": 0},
        )
        d = s.to_dict()
        # Verify exact dict values
        self.assertEqual(d["subgoal_type"], "simplify")
        self.assertEqual(d["anchor_targets"], ["List.map"])
        self.assertEqual(d["estimated_steps"], 2)
        self.assertEqual(d["bank_hints"], {"depth": 0})
        # Verify round-trip
        s2 = SubgoalSpec.from_dict(d)
        self.assertEqual(s2.subgoal_type, "simplify")
        self.assertEqual(s2.anchor_targets, ["List.map"])
        self.assertEqual(s2.estimated_steps, 2)
        self.assertEqual(s2.bank_hints, {"depth": 0})

    def test_from_dict_missing_optional_keys(self):
        d = {"subgoal_type": "rewrite"}
        s = SubgoalSpec.from_dict(d)
        self.assertEqual(s.subgoal_type, "rewrite")
        self.assertEqual(s.anchor_targets, [])
        self.assertEqual(s.estimated_steps, 1)
        self.assertEqual(s.bank_hints, {})

    def test_field_isolation_anchor_targets(self):
        s1 = _make_subgoal()
        s2 = _make_subgoal()
        s1.anchor_targets.append("Nat.zero_add")
        self.assertEqual(s2.anchor_targets, [])

    def test_field_isolation_bank_hints(self):
        s1 = _make_subgoal()
        s2 = _make_subgoal()
        s1.bank_hints["domain"] = 1
        self.assertEqual(s2.bank_hints, {})


# ---------------------------------------------------------------------------
# PlanningOutput
# ---------------------------------------------------------------------------


class TestPlanningOutput(unittest.TestCase):
    def test_construction_with_defaults(self):
        p = PlanningOutput(sketch=[])
        self.assertEqual(p.sketch, [])
        self.assertEqual(p.total_estimated_depth, 0)
        self.assertEqual(p.template_id, 0)

    def test_construction_explicit(self):
        p = _make_planning()
        self.assertEqual(len(p.sketch), 2)
        self.assertEqual(p.sketch[0].subgoal_type, "simplify")
        self.assertEqual(p.sketch[1].subgoal_type, "apply")
        self.assertEqual(p.total_estimated_depth, 4)
        self.assertEqual(p.template_id, 1)

    def test_sketch_subgoals_are_independent(self):
        p = _make_planning()
        p.sketch[0].anchor_targets.append("modified")
        self.assertEqual(p.sketch[1].anchor_targets, [])


# ---------------------------------------------------------------------------
# ExecutionOutput
# ---------------------------------------------------------------------------


class TestExecutionOutput(unittest.TestCase):
    def test_construction(self):
        e = _make_execution()
        self.assertEqual(e.directions["structure"], 1)
        self.assertEqual(e.directions["domain"], -1)
        self.assertAlmostEqual(e.direction_confidences["structure"], 0.95, places=2)
        self.assertEqual(e.anchor_logits.shape, (100,))
        self.assertAlmostEqual(e.progress, 0.6, places=2)
        self.assertAlmostEqual(e.critic, 0.75, places=2)

    def test_anchor_logits_shape_varies(self):
        e = _make_execution(anchor_logits=torch.randn(500))
        self.assertEqual(e.anchor_logits.shape, (500,))

    def test_direction_keys_match(self):
        e = _make_execution()
        self.assertEqual(set(e.directions.keys()), set(e.direction_confidences.keys()))


# ---------------------------------------------------------------------------
# VerificationOutput
# ---------------------------------------------------------------------------


class TestVerificationOutput(unittest.TestCase):
    def test_construction_success(self):
        v = _make_verification()
        self.assertEqual(v.success, True)
        self.assertEqual(v.new_goals, [])
        self.assertIsNone(v.failure_reason)

    def test_construction_failure(self):
        v = _make_verification(
            success=False,
            new_goals=[],
            failure_reason="type mismatch",
        )
        self.assertFalse(v.success)
        self.assertEqual(v.failure_reason, "type mismatch")

    def test_new_goals_populated(self):
        v = _make_verification(new_goals=["goal_a", "goal_b"])
        self.assertEqual(len(v.new_goals), 2)
        self.assertIn("goal_a", v.new_goals)

    def test_field_isolation_new_goals(self):
        v1 = _make_verification()
        v2 = _make_verification()
        v1.new_goals.append("leaked")
        self.assertEqual(v2.new_goals, [])


# ---------------------------------------------------------------------------
# CensorPrediction
# ---------------------------------------------------------------------------


class TestCensorPrediction(unittest.TestCase):
    def test_construction_no_prune(self):
        c = _make_censor()
        self.assertFalse(c.should_prune)
        self.assertAlmostEqual(c.failure_probability, 0.1, places=2)

    def test_construction_prune(self):
        c = _make_censor(should_prune=True, failure_probability=0.95)
        self.assertTrue(c.should_prune)
        self.assertAlmostEqual(c.failure_probability, 0.95, places=2)

    def test_failure_probability_bounds(self):
        low = _make_censor(failure_probability=0.0)
        high = _make_censor(failure_probability=1.0)
        self.assertAlmostEqual(low.failure_probability, 0.0, places=5)
        self.assertAlmostEqual(high.failure_probability, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
