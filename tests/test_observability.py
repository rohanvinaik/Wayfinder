"""Tests for compute_observability_score (confidence-aware scoring)."""

import unittest

from src.proof_scoring import compute_observability_score


class TestComputeObservabilityScore(unittest.TestCase):
    """VALUE + BOUNDARY tests for observability scoring."""

    def test_traced_full_positions_high_score(self):
        """Fully observed traced entity should score near 1.0."""
        positions = {"structure": 1, "domain": -1, "depth": 1,
                     "automation": -1, "context": 0, "decomposition": 1}
        anchors = set(range(15))  # 15 anchors
        score = compute_observability_score(positions, anchors, "traced")
        self.assertGreater(score, 0.8)

    def test_premise_only_sparse_low_score(self):
        """Premise-only entity with sparse data should score lower."""
        positions = {"domain": -1}  # only 1 non-zero bank
        anchors = {1, 2, 3}  # 3 anchors
        score = compute_observability_score(positions, anchors, "premise_only")
        self.assertLess(score, 0.5)

    def test_traced_beats_premise_only_same_data(self):
        """Traced entity scores higher than premise_only with identical data."""
        positions = {"structure": 1, "domain": -1}
        anchors = {1, 2, 3, 4, 5}
        traced = compute_observability_score(positions, anchors, "traced")
        premise = compute_observability_score(positions, anchors, "premise_only")
        self.assertGreater(traced, premise)

    def test_more_banks_higher_score(self):
        """More populated banks → higher observability."""
        anchors = set(range(10))
        sparse = compute_observability_score({"domain": -1}, anchors, "traced")
        rich = compute_observability_score(
            {"structure": 1, "domain": -1, "depth": 1, "automation": -1},
            anchors, "traced"
        )
        self.assertGreater(rich, sparse)

    def test_more_anchors_higher_score(self):
        """More anchors → higher observability (up to saturation)."""
        positions = {"structure": 1}
        few = compute_observability_score(positions, {1, 2}, "traced")
        many = compute_observability_score(positions, set(range(10)), "traced")
        self.assertGreater(many, few)

    def test_empty_entity_positive(self):
        """Even empty entities should get a positive (nonzero) score."""
        score = compute_observability_score({}, set(), "premise_only")
        self.assertGreater(score, 0)

    def test_return_range(self):
        """Score should always be in (0, 1]."""
        for prov in ("traced", "premise_only", "tactic"):
            for n_banks in (0, 3, 6):
                pos = {f"bank{i}": 1 for i in range(n_banks)}
                for n_anchors in (0, 5, 20):
                    anchors = set(range(n_anchors))
                    score = compute_observability_score(pos, anchors, prov)
                    self.assertGreater(score, 0)
                    self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
