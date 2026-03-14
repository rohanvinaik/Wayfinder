"""Tests for SketchPredictor — PLANNING slot proof sketch prediction."""

import unittest

import torch

from src.sketch_predictor import _SIMPLE_SKETCHES, SUBGOAL_TYPES, SketchPredictor
from src.som_contracts import PlanningOutput, RecognitionOutput, SubgoalSpec


def _make_recognition(template_id=0, template_name="DECIDE", confidence=0.9, feature_dim=64):
    return RecognitionOutput(
        template_id=template_id,
        template_name=template_name,
        template_confidence=confidence,
        template_features=torch.randn(feature_dim),
        top_k_templates=[(template_id, template_name, confidence)],
    )


class TestSketchPredictorConstruction(unittest.TestCase):
    """Test construction with default and custom parameters."""

    def test_default_construction(self):
        model = SketchPredictor()
        self.assertEqual(model.embedding_dim, 384)
        self.assertEqual(model.template_feature_dim, 64)
        self.assertEqual(model.max_subgoals, 6)
        self.assertEqual(model.num_subgoal_types, len(SUBGOAL_TYPES))

    def test_custom_construction(self):
        model = SketchPredictor(
            embedding_dim=512,
            template_feature_dim=128,
            hidden_dim=512,
            max_subgoals=4,
            num_subgoal_types=10,
        )
        self.assertEqual(model.embedding_dim, 512)
        self.assertEqual(model.template_feature_dim, 128)
        self.assertEqual(model.max_subgoals, 4)
        self.assertEqual(model.num_subgoal_types, 10)

    def test_num_subgoal_types_defaults_to_vocabulary(self):
        model = SketchPredictor()
        self.assertEqual(model.num_subgoal_types, 15)


class TestConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_subgoal_types_has_15_entries(self):
        self.assertEqual(len(SUBGOAL_TYPES), 15)

    def test_simple_sketches_has_4_entries(self):
        self.assertEqual(len(_SIMPLE_SKETCHES), 4)

    def test_simple_sketches_template_names(self):
        expected = {"DECIDE", "REWRITE_CHAIN", "APPLY_CHAIN", "HAMMER_DELEGATE"}
        self.assertEqual(set(_SIMPLE_SKETCHES.keys()), expected)


class TestSketchPredictorForward(unittest.TestCase):
    """Test forward pass output shapes and values."""

    def setUp(self):
        self.model = SketchPredictor()

    def test_forward_output_shapes(self):
        goal = torch.randn(4, 384)
        feats = torch.randn(4, 64)
        length_logits, type_logits, step_preds = self.model(goal, feats)
        self.assertEqual(length_logits.shape, (4, 6))
        self.assertIsInstance(type_logits, list)
        self.assertEqual(len(type_logits), 6)
        for tl in type_logits:
            self.assertEqual(tl.shape, (4, 15))
        self.assertIsInstance(step_preds, list)
        self.assertEqual(len(step_preds), 6)
        for sp in step_preds:
            self.assertEqual(sp.shape, (4, 1))

    def test_forward_batch_size_1(self):
        goal = torch.randn(1, 384)
        feats = torch.randn(1, 64)
        length_logits, type_logits, step_preds = self.model(goal, feats)
        self.assertEqual(length_logits.shape, (1, 6))
        for tl in type_logits:
            self.assertEqual(tl.shape, (1, 15))
        for sp in step_preds:
            self.assertEqual(sp.shape, (1, 1))

    def test_forward_batch_size_4(self):
        goal = torch.randn(4, 384)
        feats = torch.randn(4, 64)
        length_logits, type_logits, step_preds = self.model(goal, feats)
        self.assertEqual(length_logits.shape, (4, 6))
        self.assertEqual(len(type_logits), 6)
        self.assertEqual(len(step_preds), 6)

    def test_forward_step_preds_non_negative(self):
        goal = torch.randn(8, 384)
        feats = torch.randn(8, 64)
        _, _, step_preds = self.model(goal, feats)
        for sp in step_preds:
            self.assertTrue((sp >= 0).all(), "step_preds should be non-negative (softplus)")

    def test_forward_gradient_flow(self):
        goal = torch.randn(2, 384, requires_grad=True)
        feats = torch.randn(2, 64, requires_grad=True)
        length_logits, type_logits, step_preds = self.model(goal, feats)
        loss = length_logits.sum() + type_logits[0].sum() + step_preds[0].sum()
        loss.backward()
        self.assertIsNotNone(goal.grad)
        self.assertIsNotNone(feats.grad)
        assert goal.grad is not None  # noqa: S101
        assert feats.grad is not None  # noqa: S101
        self.assertTrue((goal.grad.abs() > 0).any())
        self.assertTrue((feats.grad.abs() > 0).any())


class TestSketchPredictorPredictSimple(unittest.TestCase):
    """Test predict() with simple (deterministic) templates."""

    def setUp(self):
        self.model = SketchPredictor()
        self.goal = torch.randn(384)

    def test_predict_returns_planning_output(self):
        rec = _make_recognition(template_name="DECIDE")
        result = self.model.predict(self.goal, rec)
        self.assertIsInstance(result, PlanningOutput)

    def test_predict_decide_returns_1_subgoal(self):
        rec = _make_recognition(template_name="DECIDE")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(len(result.sketch), 1)

    def test_predict_decide_subgoal_type(self):
        rec = _make_recognition(template_name="DECIDE")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(result.sketch[0].subgoal_type, "automation_close")

    def test_predict_rewrite_chain_returns_2_subgoals(self):
        rec = _make_recognition(template_name="REWRITE_CHAIN")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(len(result.sketch), 2)

    def test_predict_apply_chain_returns_2_subgoals(self):
        rec = _make_recognition(template_name="APPLY_CHAIN")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(len(result.sketch), 2)
        self.assertEqual(result.sketch[0].subgoal_type, "apply_lemma")
        self.assertEqual(result.sketch[1].subgoal_type, "close_subgoal")

    def test_predict_hammer_delegate_returns_1_subgoal(self):
        rec = _make_recognition(template_name="HAMMER_DELEGATE")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(len(result.sketch), 1)
        self.assertEqual(result.sketch[0].subgoal_type, "hammer")

    def test_predict_total_estimated_depth_matches_sum(self):
        rec = _make_recognition(template_name="REWRITE_CHAIN")
        result = self.model.predict(self.goal, rec)
        expected_depth = sum(s.estimated_steps for s in result.sketch)
        self.assertEqual(result.total_estimated_depth, expected_depth)

    def test_predict_template_id_preserved(self):
        rec = _make_recognition(template_id=7, template_name="DECIDE")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(result.template_id, 7)


class TestSketchPredictorPredictComplex(unittest.TestCase):
    """Test predict() with complex (learned) templates."""

    def setUp(self):
        self.model = SketchPredictor()
        self.goal = torch.randn(384)

    def test_predict_complex_returns_planning_output(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        self.assertIsInstance(result, PlanningOutput)

    def test_predict_complex_sketch_length_in_range(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        self.assertGreaterEqual(len(result.sketch), 1)
        self.assertLessEqual(len(result.sketch), 6)

    def test_predict_complex_subgoal_types_from_vocabulary(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        for subgoal in result.sketch:
            self.assertIn(subgoal.subgoal_type, SUBGOAL_TYPES)

    def test_predict_complex_estimated_steps_at_least_1(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        for subgoal in result.sketch:
            self.assertGreaterEqual(subgoal.estimated_steps, 1)

    def test_predict_complex_total_depth_matches_sum(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        expected_depth = sum(s.estimated_steps for s in result.sketch)
        self.assertEqual(result.total_estimated_depth, expected_depth)

    def test_predict_complex_template_id_preserved(self):
        rec = _make_recognition(template_id=5, template_name="CASE_ANALYSIS")
        result = self.model.predict(self.goal, rec)
        self.assertEqual(result.template_id, 5)

    def test_predict_complex_subgoals_are_subgoal_spec(self):
        rec = _make_recognition(template_id=2, template_name="INDUCT_THEN_CLOSE")
        result = self.model.predict(self.goal, rec)
        for subgoal in result.sketch:
            self.assertIsInstance(subgoal, SubgoalSpec)


if __name__ == "__main__":
    unittest.main()
