"""Init tests for GoalAnalyzer (mutation-prescribed)."""
import unittest
from src.goal_analyzer import GoalAnalyzer

class TestGoalAnalyzerInit(unittest.TestCase):
    def test_stored_attributes(self):
        ga = GoalAnalyzer(input_dim=384, feature_dim=256)
        self.assertEqual(ga.input_dim, 384)
        self.assertEqual(ga.feature_dim, 256)
    def test_no_bank_heads_default(self):
        self.assertIsNone(GoalAnalyzer().bank_heads)
    def test_bank_heads_created(self):
        ga = GoalAnalyzer(navigable_banks=["structure", "domain"])
        self.assertIsNotNone(ga.bank_heads)
        self.assertEqual(set(ga.bank_heads.keys()), {"structure", "domain"})
    def test_no_anchor_head_zero(self):
        self.assertIsNone(GoalAnalyzer(num_anchors=0).anchor_head)
    def test_anchor_head_positive(self):
        self.assertIsNotNone(GoalAnalyzer(num_anchors=100).anchor_head)
    def test_swap(self):
        a = GoalAnalyzer(input_dim=384, feature_dim=256)
        b = GoalAnalyzer(input_dim=256, feature_dim=384)
        self.assertNotEqual(a.projection.in_features, b.projection.in_features)
    def test_projection_shape(self):
        ga = GoalAnalyzer(input_dim=384, feature_dim=128)
        self.assertEqual(ga.projection.in_features, 384)
        self.assertEqual(ga.projection.out_features, 128)

if __name__ == "__main__":
    unittest.main()
