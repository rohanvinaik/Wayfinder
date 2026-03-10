"""Tests for pab_profile — serialization, deserialization, save/load."""

import json
import tempfile
import unittest
from pathlib import Path

from src.pab_profile import (
    PABCoreSeries,
    PABDomainData,
    PABLossSeries,
    PABProfile,
    PABSummary,
    PABTierSeries,
    _deserialize_profile,
    _serialize_profile,
)


class TestSerializeProfile(unittest.TestCase):
    """Test _serialize_profile produces correct flat JSON structure."""

    def _make_profile(self):
        return PABProfile(
            experiment_id="SER",
            config_hash="hash123",
            checkpoints=[50, 100, 150],
            core=PABCoreSeries(
                stability=[0.0, 0.5, 0.3],
                predictability=[0.0, 0.0, 0.01],
                generalization_gap=[0.2, 0.3, 0.25],
                representation_evolution=[1.0, 0.1, 0.05],
            ),
            tiers=PABTierSeries(
                tier1_accuracy=[0.4, 0.6, 0.75],
                tier2_accuracy=[0.3, 0.5, 0.65],
                tier3_accuracy=[0.2, 0.35, 0.5],
                ternary_crystallization=[0.0, 0.8, 0.9],
            ),
            domains=PABDomainData(
                domain_progression={"algebra": [0.7, 0.8]},
                domain_classification={"algebra": "early"},
                tactic_progression={"simp": [0.9, 0.95]},
            ),
            losses=PABLossSeries(
                loss_ce=[1.5, 0.7, 0.3],
                loss_margin=[0.3, 0.2, 0.1],
                loss_repair=[0.2, 0.1, 0.1],
                loss_adaptive_weights=[{"ce": 0.5}, {"ce": 0.6}, {"ce": 0.7}],
            ),
            summary=PABSummary(
                stability_mean=0.267,
                stability_std=0.205,
                predictability_final=0.01,
                early_stop_epoch=None,
                convergence_epoch=100,
                stability_regime="moderate",
                tier1_convergence_step=150,
                tier2_convergence_step=100,
                crystallization_rate=0.45,
                feature_importance_L=0.12,
            ),
        )

    def test_top_level_fields(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        self.assertEqual(d["experiment_id"], "SER")
        self.assertEqual(d["config_hash"], "hash123")
        self.assertEqual(d["checkpoints"], [50, 100, 150])

    def test_core_series_flattened(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        self.assertEqual(d["stability"], [0.0, 0.5, 0.3])
        self.assertEqual(d["predictability"], [0.0, 0.0, 0.01])
        self.assertEqual(d["generalization_gap"], [0.2, 0.3, 0.25])
        self.assertEqual(d["representation_evolution"], [1.0, 0.1, 0.05])

    def test_tier_series_flattened(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        self.assertEqual(d["tier1_accuracy"], [0.4, 0.6, 0.75])
        self.assertEqual(d["tier2_accuracy"], [0.3, 0.5, 0.65])
        self.assertEqual(d["ternary_crystallization"], [0.0, 0.8, 0.9])

    def test_domain_data_flattened(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        self.assertEqual(d["domain_progression"], {"algebra": [0.7, 0.8]})
        self.assertEqual(d["domain_classification"], {"algebra": "early"})
        self.assertEqual(d["tactic_progression"], {"simp": [0.9, 0.95]})

    def test_losses_flattened(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        self.assertEqual(d["loss_ce"], [1.5, 0.7, 0.3])
        self.assertEqual(d["loss_margin"], [0.3, 0.2, 0.1])
        self.assertEqual(d["loss_adaptive_weights"][0], {"ce": 0.5})

    def test_summary_nested(self):
        p = self._make_profile()
        d = _serialize_profile(p)
        s = d["summary"]
        self.assertAlmostEqual(s["stability_mean"], 0.267)
        self.assertAlmostEqual(s["stability_std"], 0.205)
        self.assertEqual(s["convergence_epoch"], 100)
        self.assertIsNone(s["early_stop_epoch"])
        self.assertEqual(s["stability_regime"], "moderate")
        self.assertEqual(s["tier1_convergence_step"], 150)
        self.assertAlmostEqual(s["crystallization_rate"], 0.45)

    def test_empty_profile_serializes(self):
        p = PABProfile()
        d = _serialize_profile(p)
        self.assertEqual(d["experiment_id"], "")
        self.assertEqual(d["checkpoints"], [])
        self.assertEqual(d["stability"], [])
        self.assertEqual(d["summary"]["stability_regime"], "unknown")


class TestDeserializeProfile(unittest.TestCase):
    """Test _deserialize_profile reconstructs PABProfile correctly."""

    def test_roundtrip(self):
        original = PABProfile(
            experiment_id="RT",
            config_hash="abc",
            checkpoints=[10, 20],
            core=PABCoreSeries(stability=[0.1, 0.2]),
            tiers=PABTierSeries(tier1_accuracy=[0.8, 0.9]),
            summary=PABSummary(stability_mean=0.15, stability_regime="stable"),
        )
        d = _serialize_profile(original)
        loaded = _deserialize_profile(d)
        self.assertEqual(loaded.experiment_id, "RT")
        self.assertEqual(loaded.config_hash, "abc")
        self.assertEqual(loaded.checkpoints, [10, 20])
        self.assertEqual(loaded.core.stability, [0.1, 0.2])
        self.assertEqual(loaded.tiers.tier1_accuracy, [0.8, 0.9])
        self.assertAlmostEqual(loaded.summary.stability_mean, 0.15)
        self.assertEqual(loaded.summary.stability_regime, "stable")

    def test_missing_keys_use_defaults(self):
        loaded = _deserialize_profile({})
        self.assertEqual(loaded.experiment_id, "")
        self.assertEqual(loaded.config_hash, "")
        self.assertEqual(loaded.checkpoints, [])
        self.assertEqual(loaded.core.stability, [])
        self.assertEqual(loaded.tiers.tier1_accuracy, [])
        self.assertEqual(loaded.domains.domain_progression, {})
        self.assertEqual(loaded.losses.loss_ce, [])
        self.assertEqual(loaded.summary.stability_mean, 0.0)
        self.assertEqual(loaded.summary.stability_regime, "unknown")
        self.assertIsNone(loaded.summary.early_stop_epoch)
        self.assertIsNone(loaded.summary.convergence_epoch)

    def test_summary_none_values_preserved(self):
        data = {
            "summary": {
                "stability_mean": 0.5,
                "early_stop_epoch": None,
                "convergence_epoch": None,
                "tier1_convergence_step": None,
                "tier2_convergence_step": None,
            }
        }
        loaded = _deserialize_profile(data)
        self.assertIsNone(loaded.summary.early_stop_epoch)
        self.assertIsNone(loaded.summary.convergence_epoch)
        self.assertIsNone(loaded.summary.tier1_convergence_step)
        self.assertAlmostEqual(loaded.summary.stability_mean, 0.5)

    def test_summary_with_values_preserved(self):
        data = {
            "summary": {
                "early_stop_epoch": 200,
                "convergence_epoch": 150,
                "tier1_convergence_step": 100,
                "tier2_convergence_step": 75,
                "crystallization_rate": 0.88,
                "feature_importance_L": 0.33,
            }
        }
        loaded = _deserialize_profile(data)
        self.assertEqual(loaded.summary.early_stop_epoch, 200)
        self.assertEqual(loaded.summary.convergence_epoch, 150)
        self.assertEqual(loaded.summary.tier1_convergence_step, 100)
        self.assertEqual(loaded.summary.tier2_convergence_step, 75)
        self.assertAlmostEqual(loaded.summary.crystallization_rate, 0.88)
        self.assertAlmostEqual(loaded.summary.feature_importance_L, 0.33)


class TestPABProfileSaveLoad(unittest.TestCase):
    """Test PABProfile.save and PABProfile.load file I/O."""

    def _make_full_profile(self):
        return PABProfile(
            experiment_id="FILE",
            config_hash="xyz",
            checkpoints=[50, 100],
            core=PABCoreSeries(
                stability=[0.0, 0.5],
                predictability=[0.0, 0.01],
                generalization_gap=[0.2, 0.3],
                representation_evolution=[1.0, 0.1],
            ),
            tiers=PABTierSeries(
                tier1_accuracy=[0.6, 0.8],
                tier2_accuracy=[0.5, 0.7],
                tier3_accuracy=[0.3, 0.5],
                ternary_crystallization=[0.0, 0.9],
            ),
            domains=PABDomainData(
                domain_progression={"alg": [0.7]},
                domain_classification={"alg": "early"},
                tactic_progression={"simp": [0.8]},
            ),
            losses=PABLossSeries(
                loss_ce=[1.0, 0.5],
                loss_margin=[0.2, 0.1],
                loss_repair=[0.1, 0.05],
                loss_adaptive_weights=[{"ce": 0.5}, {"ce": 0.6}],
            ),
            summary=PABSummary(
                stability_mean=0.25,
                stability_std=0.25,
                predictability_final=0.01,
                convergence_epoch=100,
                stability_regime="moderate",
                tier1_convergence_step=100,
            ),
        )

    def test_save_load_roundtrip(self):
        profile = self._make_full_profile()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            profile.save(tmp.name)
            loaded = PABProfile.load(tmp.name)
        self.assertEqual(loaded.experiment_id, "FILE")
        self.assertEqual(loaded.config_hash, "xyz")
        self.assertEqual(loaded.checkpoints, [50, 100])
        self.assertEqual(loaded.core.stability, [0.0, 0.5])
        self.assertEqual(loaded.tiers.tier1_accuracy, [0.6, 0.8])
        self.assertEqual(loaded.domains.domain_progression, {"alg": [0.7]})
        self.assertEqual(loaded.losses.loss_ce, [1.0, 0.5])
        self.assertAlmostEqual(loaded.summary.stability_mean, 0.25)
        self.assertEqual(loaded.summary.convergence_epoch, 100)
        self.assertEqual(loaded.summary.stability_regime, "moderate")

    def test_save_creates_parent_directories(self):
        profile = PABProfile(experiment_id="DIR")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "profile.json"
            profile.save(path)
            self.assertTrue(path.exists())

    def test_save_writes_valid_json(self):
        profile = self._make_full_profile()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            profile.save(tmp.name)
            with open(tmp.name) as f:
                data = json.load(f)
        self.assertIn("experiment_id", data)
        self.assertIn("stability", data)
        self.assertIn("summary", data)
        self.assertIsInstance(data["summary"], dict)

    def test_empty_profile_roundtrip(self):
        profile = PABProfile()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            profile.save(tmp.name)
            loaded = PABProfile.load(tmp.name)
        self.assertEqual(loaded.experiment_id, "")
        self.assertEqual(loaded.checkpoints, [])
        self.assertEqual(loaded.core.stability, [])
        self.assertEqual(loaded.summary.stability_regime, "unknown")
        self.assertIsNone(loaded.summary.early_stop_epoch)


if __name__ == "__main__":
    unittest.main()
