"""Tests for TemplateClassifier — RECOGNITION slot neural classifier."""

import unittest

import torch

from src.som_contracts import RecognitionOutput
from src.story_templates import TEMPLATE_NAMES, get_num_templates
from src.template_classifier import TemplateClassifier


class TestTemplateClassifierConstruction(unittest.TestCase):
    """Test construction with default and custom parameters."""

    def test_default_construction(self):
        model = TemplateClassifier()
        self.assertEqual(model.input_dim, 256)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.feature_dim, 64)
        self.assertEqual(model.num_templates, get_num_templates())

    def test_custom_construction(self):
        model = TemplateClassifier(input_dim=512, hidden_dim=256, feature_dim=32, num_templates=5)
        self.assertEqual(model.input_dim, 512)
        self.assertEqual(model.hidden_dim, 256)
        self.assertEqual(model.feature_dim, 32)
        self.assertEqual(model.num_templates, 5)

    def test_num_templates_defaults_to_taxonomy(self):
        model = TemplateClassifier()
        self.assertEqual(model.num_templates, 9)
        self.assertEqual(model.num_templates, len(TEMPLATE_NAMES))


class TestTemplateClassifierForward(unittest.TestCase):
    """Test forward pass output shapes and values."""

    def setUp(self):
        self.model = TemplateClassifier()

    def test_forward_output_shapes(self):
        features = torch.randn(4, 256)
        logits, template_features = self.model(features)
        self.assertEqual(logits.shape, (4, 9))
        self.assertEqual(template_features.shape, (4, 64))

    def test_forward_batch_size_1(self):
        features = torch.randn(1, 256)
        logits, template_features = self.model(features)
        self.assertEqual(logits.shape, (1, 9))
        self.assertEqual(template_features.shape, (1, 64))

    def test_forward_batch_size_4(self):
        features = torch.randn(4, 256)
        logits, template_features = self.model(features)
        self.assertEqual(logits.shape, (4, 9))
        self.assertEqual(template_features.shape, (4, 64))

    def test_forward_batch_size_16(self):
        features = torch.randn(16, 256)
        logits, template_features = self.model(features)
        self.assertEqual(logits.shape, (16, 9))
        self.assertEqual(template_features.shape, (16, 64))

    def test_forward_outputs_are_finite(self):
        features = torch.randn(8, 256)
        logits, template_features = self.model(features)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(template_features).all())

    def test_forward_custom_dims(self):
        model = TemplateClassifier(input_dim=128, hidden_dim=64, feature_dim=32, num_templates=5)
        features = torch.randn(3, 128)
        logits, template_features = model(features)
        self.assertEqual(logits.shape, (3, 5))
        self.assertEqual(template_features.shape, (3, 32))


class TestTemplateClassifierPredict(unittest.TestCase):
    """Test predict() returns correct RecognitionOutput."""

    def setUp(self):
        self.model = TemplateClassifier()

    def test_predict_returns_recognition_output(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertIsInstance(result, RecognitionOutput)

    def test_predict_template_id_in_range(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertGreaterEqual(result.template_id, 0)
        self.assertLess(result.template_id, self.model.num_templates)

    def test_predict_template_name_in_taxonomy(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertIn(result.template_name, TEMPLATE_NAMES)

    def test_predict_template_confidence_in_unit_interval(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertGreaterEqual(result.template_confidence, 0.0)
        self.assertLessEqual(result.template_confidence, 1.0)

    def test_predict_template_features_shape(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertEqual(result.template_features.shape, (64,))

    def test_predict_top_k_at_most_3(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        self.assertLessEqual(len(result.top_k_templates), 3)
        self.assertGreater(len(result.top_k_templates), 0)

    def test_predict_top_k_entries_are_tuples(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        for entry in result.top_k_templates:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 3)
            idx, name, conf = entry
            self.assertIsInstance(idx, int)
            self.assertIsInstance(name, str)
            self.assertIsInstance(conf, float)

    def test_predict_top_k_confidences_sum_leq_1(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        total = sum(conf for _, _, conf in result.top_k_templates)
        self.assertLessEqual(total, 1.0 + 1e-6)

    def test_predict_top_k_first_matches_best(self):
        features = torch.randn(1, 256)
        result = self.model.predict(features)
        first_idx, first_name, first_conf = result.top_k_templates[0]
        self.assertEqual(first_idx, result.template_id)
        self.assertEqual(first_name, result.template_name)
        self.assertAlmostEqual(first_conf, result.template_confidence, places=6)


class TestTemplateClassifierGradient(unittest.TestCase):
    """Test gradient flow through the classifier."""

    def test_backward_pass(self):
        model = TemplateClassifier()
        features = torch.randn(4, 256)
        logits, template_features = model(features)
        loss = logits.sum() + template_features.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"Non-finite gradient for {name}",
            )


if __name__ == "__main__":
    unittest.main()
