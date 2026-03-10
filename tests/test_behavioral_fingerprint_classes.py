"""Tests for behavioral_fingerprint — drift, stability, dataclass methods, save/load."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.behavioral_fingerprint import (
    BehavioralFingerprint,
    _distribution_drift,
    fingerprint_stability,
)


class TestDistributionDrift(unittest.TestCase):
    """Test Jensen-Shannon divergence between action distributions."""

    def test_empty_distributions_return_zero(self):
        self.assertEqual(_distribution_drift({}, {}), 0.0)

    def test_identical_distributions_near_zero(self):
        dist = {"a": 0.5, "b": 0.5}
        result = _distribution_drift(dist, dist)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_disjoint_distributions_positive(self):
        dist_a = {"a": 1.0}
        dist_b = {"b": 1.0}
        result = _distribution_drift(dist_a, dist_b)
        self.assertGreater(result, 0.0)
        # JSD is bounded by log2(2) = 1.0
        self.assertLessEqual(result, 1.0 + 1e-6)

    def test_one_empty_one_full(self):
        result = _distribution_drift({"a": 1.0}, {})
        # One distribution has mass, other is zero everywhere
        self.assertGreater(result, 0.0)

    def test_symmetric(self):
        dist_a = {"a": 0.7, "b": 0.3}
        dist_b = {"a": 0.3, "b": 0.7}
        self.assertAlmostEqual(
            _distribution_drift(dist_a, dist_b),
            _distribution_drift(dist_b, dist_a),
            places=10,
        )


class TestFingerprintStability(unittest.TestCase):
    """Test stability metrics across training checkpoints."""

    def test_empty_list_returns_zeros(self):
        result = fingerprint_stability([])
        self.assertEqual(result["entropy_variance"], 0.0)
        self.assertEqual(result["distribution_drift"], 0.0)
        self.assertEqual(result["discreteness_variance"], 0.0)

    def test_single_fingerprint_returns_zeros(self):
        fp = BehavioralFingerprint(action_entropy=1.5, discreteness_score=0.8)
        result = fingerprint_stability([fp])
        self.assertEqual(result["entropy_variance"], 0.0)
        self.assertEqual(result["distribution_drift"], 0.0)
        self.assertEqual(result["discreteness_variance"], 0.0)

    def test_identical_fingerprints_zero_variance(self):
        fp = BehavioralFingerprint(
            action_entropy=1.0,
            discreteness_score=0.5,
            action_distribution={"a": 0.5, "b": 0.5},
        )
        result = fingerprint_stability([fp, fp, fp])
        self.assertAlmostEqual(result["entropy_variance"], 0.0, places=5)
        self.assertAlmostEqual(result["distribution_drift"], 0.0, places=5)
        self.assertAlmostEqual(result["discreteness_variance"], 0.0, places=5)

    def test_varying_fingerprints_positive_metrics(self):
        fp1 = BehavioralFingerprint(
            action_entropy=0.5,
            discreteness_score=0.2,
            action_distribution={"a": 0.9, "b": 0.1},
        )
        fp2 = BehavioralFingerprint(
            action_entropy=1.5,
            discreteness_score=0.8,
            action_distribution={"a": 0.3, "b": 0.7},
        )
        result = fingerprint_stability([fp1, fp2])
        self.assertGreater(result["entropy_variance"], 0.0)
        self.assertGreater(result["distribution_drift"], 0.0)
        self.assertGreater(result["discreteness_variance"], 0.0)


class TestBehavioralFingerprintDataclass(unittest.TestCase):
    """Test BehavioralFingerprint methods: to_dict, from_dict, save, load, factories."""

    def _make_fingerprint(self):
        return BehavioralFingerprint(
            experiment_id="EXP1",
            step=500,
            action_entropy=1.234,
            action_distribution={"simp": 0.6, "ring": 0.4},
            variance_eigenvalues=[3.2, 1.1, 0.5],
            discreteness_score=0.78,
            probe_responses={"probe_1": "simp", "probe_2": "ring"},
        )

    def test_to_dict_all_fields(self):
        fp = self._make_fingerprint()
        d = fp._to_dict()
        self.assertEqual(d["experiment_id"], "EXP1")
        self.assertEqual(d["step"], 500)
        self.assertAlmostEqual(d["action_entropy"], 1.234)
        self.assertEqual(d["action_distribution"], {"simp": 0.6, "ring": 0.4})
        self.assertEqual(d["variance_eigenvalues"], [3.2, 1.1, 0.5])
        self.assertAlmostEqual(d["discreteness_score"], 0.78)
        self.assertEqual(d["probe_responses"], {"probe_1": "simp", "probe_2": "ring"})

    def test_from_dict_roundtrip(self):
        fp = self._make_fingerprint()
        d = fp._to_dict()
        loaded = BehavioralFingerprint._from_dict(d)
        self.assertEqual(loaded.experiment_id, fp.experiment_id)
        self.assertEqual(loaded.step, fp.step)
        self.assertAlmostEqual(loaded.action_entropy, fp.action_entropy)
        self.assertEqual(loaded.action_distribution, fp.action_distribution)
        self.assertEqual(loaded.variance_eigenvalues, fp.variance_eigenvalues)
        self.assertAlmostEqual(loaded.discreteness_score, fp.discreteness_score)
        self.assertEqual(loaded.probe_responses, fp.probe_responses)

    def test_from_dict_missing_keys_use_defaults(self):
        loaded = BehavioralFingerprint._from_dict({})
        self.assertEqual(loaded.experiment_id, "")
        self.assertEqual(loaded.step, 0)
        self.assertEqual(loaded.action_entropy, 0.0)
        self.assertEqual(loaded.action_distribution, {})
        self.assertEqual(loaded.variance_eigenvalues, [])
        self.assertEqual(loaded.discreteness_score, 0.0)
        self.assertEqual(loaded.probe_responses, {})

    def test_save_load_roundtrip(self):
        fp = self._make_fingerprint()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            fp.save(tmp.name)
            loaded = BehavioralFingerprint.load(tmp.name)
        self.assertEqual(loaded.experiment_id, fp.experiment_id)
        self.assertEqual(loaded.step, fp.step)
        self.assertAlmostEqual(loaded.action_entropy, fp.action_entropy)
        self.assertEqual(loaded.action_distribution, fp.action_distribution)
        self.assertEqual(loaded.variance_eigenvalues, fp.variance_eigenvalues)

    def test_save_creates_parent_dirs(self):
        fp = BehavioralFingerprint(experiment_id="X")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "fp.json"
            fp.save(path)
            self.assertTrue(path.exists())

    def test_save_writes_valid_json(self):
        fp = self._make_fingerprint()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="r") as tmp:
            fp.save(tmp.name)
            with open(tmp.name) as f:
                data = json.load(f)
            self.assertIn("experiment_id", data)
            self.assertIn("action_entropy", data)

    def test_from_outputs(self):
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        predictions = ["simp", "ring", "simp"]
        fp = BehavioralFingerprint.from_outputs(
            experiment_id="EXP",
            step=100,
            output_logits=logits,
            action_predictions=predictions,
        )
        self.assertEqual(fp.experiment_id, "EXP")
        self.assertEqual(fp.step, 100)
        self.assertGreater(fp.action_entropy, 0.0)
        self.assertIn("simp", fp.action_distribution)
        self.assertGreater(len(fp.variance_eigenvalues), 0)
        self.assertGreater(fp.discreteness_score, 0.0)
        self.assertEqual(fp.probe_responses, {})

    def test_from_outputs_with_probe_labels(self):
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        predictions = ["a", "b"]
        labels = ["probe_x", "probe_y"]
        fp = BehavioralFingerprint.from_outputs(
            "E", 50, logits, predictions, probe_labels=labels,
        )
        self.assertEqual(fp.probe_responses, {"probe_x": "a", "probe_y": "b"})

    def test_from_text_only(self):
        predictions = ["simp", "ring", "simp"]
        fp = BehavioralFingerprint.from_text_only(
            experiment_id="TXT",
            step=200,
            action_predictions=predictions,
        )
        self.assertEqual(fp.experiment_id, "TXT")
        self.assertEqual(fp.step, 200)
        self.assertGreater(fp.action_entropy, 0.0)
        self.assertIn("simp", fp.action_distribution)
        self.assertEqual(fp.variance_eigenvalues, [])
        self.assertEqual(fp.discreteness_score, 0.0)
        self.assertEqual(fp.probe_responses, {})

    def test_from_text_only_with_probe_labels(self):
        predictions = ["a", "b"]
        labels = ["p1", "p2"]
        fp = BehavioralFingerprint.from_text_only(
            "E", 50, predictions, probe_labels=labels,
        )
        self.assertEqual(fp.probe_responses, {"p1": "a", "p2": "b"})

    def test_defaults(self):
        fp = BehavioralFingerprint()
        self.assertEqual(fp.experiment_id, "")
        self.assertEqual(fp.step, 0)
        self.assertEqual(fp.action_entropy, 0.0)
        self.assertEqual(fp.action_distribution, {})
        self.assertEqual(fp.variance_eigenvalues, [])
        self.assertEqual(fp.discreteness_score, 0.0)
        self.assertEqual(fp.probe_responses, {})


if __name__ == "__main__":
    unittest.main()
