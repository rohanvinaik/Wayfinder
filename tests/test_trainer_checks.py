"""Tests for trainer_checks -- stateless helper functions for trainer step checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from src.pab_tracker import CheckpointData
from src.ternary_decoder import TernaryLinear
from src.trainer_checks import (
    PABMetricsSnapshot,
    build_checkpoint_data,
    check_gradient_abort,
    check_gradient_health,
    collect_decoder_weight_signs,
    log_ternary_distribution,
    record_pab_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------


def _make_decoder_with_known_weights():
    """Build a minimal decoder with two TernaryLinear layers with known weights.

    Layer 0 (2x4): [[ 1.0,  0.0, -1.0,  0.5],
                     [-0.5,  0.0,  0.0,  1.0]]
    Layer 1 (2x2): [[ 1.0, -1.0],
                     [ 0.0,  0.0]]
    """
    decoder = nn.Module()
    layer0 = TernaryLinear(4, 2, bias=False)
    layer0.weight = nn.Parameter(torch.tensor([[1.0, 0.0, -1.0, 0.5], [-0.5, 0.0, 0.0, 1.0]]))
    layer1 = TernaryLinear(2, 2, bias=False)
    layer1.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [0.0, 0.0]]))
    decoder.add_module("tl0", layer0)
    decoder.add_module("tl1", layer1)
    return decoder


def _make_pipeline(nan_param=False, inf_param=False):
    """Build a mock pipeline with four named sub-modules.

    Each sub-module has a single parameter with a gradient.
    If nan_param or inf_param is True, the decoder's gradient is corrupted.
    """

    def _module(grad_value=1.0):
        m = nn.Linear(2, 2)
        # Simulate a backward pass so .grad is populated
        for p in m.parameters():
            p.grad = torch.full_like(p, grad_value)
        return m

    gate = _module()
    analyzer = _module()
    bridge = _module()
    decoder = _module()

    if nan_param:
        for p in decoder.parameters():
            p.grad = torch.full_like(p, float("nan"))
    if inf_param:
        for p in decoder.parameters():
            p.grad = torch.full_like(p, float("inf"))

    return SimpleNamespace(
        domain_gate=gate,
        goal_analyzer=analyzer,
        bridge=bridge,
        decoder=decoder,
    )


# ---------------------------------------------------------------------------
# PABMetricsSnapshot
# ---------------------------------------------------------------------------


class TestPABMetricsSnapshot(unittest.TestCase):
    def test_construction_with_all_fields(self):
        snap = PABMetricsSnapshot(
            val_loss=0.5,
            tier_accuracies={"tier1": 0.8},
            domain_accuracies={"algebra": 0.9},
            tactic_accuracies={"simp": 0.7},
        )
        self.assertEqual(snap.val_loss, 0.5)
        self.assertEqual(snap.tier_accuracies, {"tier1": 0.8})
        self.assertEqual(snap.domain_accuracies, {"algebra": 0.9})
        self.assertEqual(snap.tactic_accuracies, {"simp": 0.7})

    def test_defaults_are_none(self):
        snap = PABMetricsSnapshot()
        self.assertIsNone(snap.val_loss)
        self.assertIsNone(snap.tier_accuracies)
        self.assertIsNone(snap.domain_accuracies)
        self.assertIsNone(snap.tactic_accuracies)

    def test_partial_defaults(self):
        snap = PABMetricsSnapshot(val_loss=1.0)
        self.assertEqual(snap.val_loss, 1.0)
        self.assertIsNone(snap.tier_accuracies)
        self.assertIsNone(snap.domain_accuracies)
        self.assertIsNone(snap.tactic_accuracies)

    def test_is_namedtuple(self):
        snap = PABMetricsSnapshot(val_loss=0.1)
        expected = (
            "val_loss",
            "tier_accuracies",
            "domain_accuracies",
            "tactic_accuracies",
        )
        self.assertEqual(snap._fields, expected)


# ---------------------------------------------------------------------------
# log_ternary_distribution
# ---------------------------------------------------------------------------


class TestLogTernaryDistribution(unittest.TestCase):
    def test_known_weights(self):
        """Manually computed percentages from known weight tensors.

        Layer tl0 (2x4 = 8 elements):
          Values: [1.0, 0.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0]
          neg (< -0.01):  -1.0, -0.5  => 2/8 = 25.0%
          zero (>= -0.01 and <= 0.01): 0.0, 0.0, 0.0 => 3/8 = 37.5%
          pos (> 0.01): 1.0, 0.5, 1.0 => 3/8 = 37.5%

        Layer tl1 (2x2 = 4 elements):
          Values: [1.0, -1.0, 0.0, 0.0]
          neg: -1.0 => 1/4 = 25.0%
          zero: 0.0, 0.0 => 2/4 = 50.0%
          pos: 1.0 => 1/4 = 25.0%
        """
        decoder = _make_decoder_with_known_weights()
        dist = log_ternary_distribution(decoder)

        self.assertEqual(set(dist.keys()), {"tl0", "tl1"})

        self.assertAlmostEqual(dist["tl0"]["neg_pct"], 25.0)
        self.assertAlmostEqual(dist["tl0"]["zero_pct"], 37.5)
        self.assertAlmostEqual(dist["tl0"]["pos_pct"], 37.5)

        self.assertAlmostEqual(dist["tl1"]["neg_pct"], 25.0)
        self.assertAlmostEqual(dist["tl1"]["zero_pct"], 50.0)
        self.assertAlmostEqual(dist["tl1"]["pos_pct"], 25.0)

    def test_all_positive_weights(self):
        decoder = nn.Module()
        tl = TernaryLinear(3, 2, bias=False)
        tl.weight = nn.Parameter(torch.ones(2, 3))
        decoder.add_module("layer", tl)
        dist = log_ternary_distribution(decoder)
        self.assertAlmostEqual(dist["layer"]["neg_pct"], 0.0)
        self.assertAlmostEqual(dist["layer"]["zero_pct"], 0.0)
        self.assertAlmostEqual(dist["layer"]["pos_pct"], 100.0)

    def test_all_zero_weights(self):
        decoder = nn.Module()
        tl = TernaryLinear(3, 2, bias=False)
        tl.weight = nn.Parameter(torch.zeros(2, 3))
        decoder.add_module("layer", tl)
        dist = log_ternary_distribution(decoder)
        self.assertAlmostEqual(dist["layer"]["neg_pct"], 0.0)
        self.assertAlmostEqual(dist["layer"]["zero_pct"], 100.0)
        self.assertAlmostEqual(dist["layer"]["pos_pct"], 0.0)

    def test_no_ternary_layers_returns_empty(self):
        decoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        dist = log_ternary_distribution(decoder)
        self.assertEqual(dist, {})

    def test_percentages_sum_to_100(self):
        decoder = _make_decoder_with_known_weights()
        dist = log_ternary_distribution(decoder)
        for layer_name, pcts in dist.items():
            total = pcts["neg_pct"] + pcts["zero_pct"] + pcts["pos_pct"]
            self.assertAlmostEqual(total, 100.0, msg=f"{layer_name} pcts sum to {total}")


    def test_boundary_at_threshold(self):
        """BOUNDARY: weights exactly at ±0.01 boundary."""
        decoder = nn.Module()
        tl = TernaryLinear(4, 1, bias=False)
        # Values: -0.02 (neg), -0.01 (zero), 0.01 (zero), 0.02 (pos)
        tl.weight = nn.Parameter(torch.tensor([[-0.02, -0.01, 0.01, 0.02]]))
        decoder.add_module("layer", tl)
        dist = log_ternary_distribution(decoder)
        self.assertAlmostEqual(dist["layer"]["neg_pct"], 25.0)   # -0.02
        self.assertAlmostEqual(dist["layer"]["zero_pct"], 50.0)  # -0.01, 0.01
        self.assertAlmostEqual(dist["layer"]["pos_pct"], 25.0)   # 0.02

    def test_boundary_just_inside_zero_band(self):
        """BOUNDARY: values at ±0.005 are within zero band."""
        decoder = nn.Module()
        tl = TernaryLinear(2, 1, bias=False)
        tl.weight = nn.Parameter(torch.tensor([[-0.005, 0.005]]))
        decoder.add_module("layer", tl)
        dist = log_ternary_distribution(decoder)
        self.assertAlmostEqual(dist["layer"]["zero_pct"], 100.0)


# ---------------------------------------------------------------------------
# collect_decoder_weight_signs
# ---------------------------------------------------------------------------


class TestCollectDecoderWeightSigns(unittest.TestCase):
    def test_known_weights(self):
        """Signs of the known weights.

        Layer tl0: [1.0, 0.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0]
          signs:   [1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0]
        Layer tl1: [1.0, -1.0, 0.0, 0.0]
          signs:   [1.0, -1.0, 0.0, 0.0]
        Concatenated: 12 elements total.
        """
        decoder = _make_decoder_with_known_weights()
        signs = collect_decoder_weight_signs(decoder)
        self.assertIsNotNone(signs)
        expected = np.array([1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0])
        np.testing.assert_array_equal(signs, expected)

    def test_no_ternary_layers_returns_none(self):
        decoder = nn.Sequential(nn.Linear(4, 4))
        result = collect_decoder_weight_signs(decoder)
        self.assertIsNone(result)

    def test_shape_matches_total_weights(self):
        decoder = _make_decoder_with_known_weights()
        signs = collect_decoder_weight_signs(decoder)
        # tl0: 2*4=8, tl1: 2*2=4, total=12
        self.assertEqual(signs.shape, (12,))


# ---------------------------------------------------------------------------
# check_gradient_health
# ---------------------------------------------------------------------------


class TestCheckGradientHealth(unittest.TestCase):
    def test_healthy_gradients(self):
        pipeline = _make_pipeline()
        healthy, msg = check_gradient_health(pipeline)
        self.assertTrue(healthy)
        self.assertIsNone(msg)

    def test_nan_gradient_detected(self):
        pipeline = _make_pipeline(nan_param=True)
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("NaN/Inf", msg)
        self.assertIn("decoder", msg)

    def test_inf_gradient_detected(self):
        pipeline = _make_pipeline(inf_param=True)
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("NaN/Inf", msg)
        self.assertIn("decoder", msg)

    def test_no_gradient_is_healthy(self):
        """Parameters with grad=None should not trigger unhealthy."""
        pipeline = _make_pipeline()
        for p in pipeline.decoder.parameters():
            p.grad = None
        healthy, msg = check_gradient_health(pipeline)
        self.assertTrue(healthy)
        self.assertIsNone(msg)

    def test_nan_in_gate_detected(self):
        """NaN in domain_gate should be caught and named."""
        pipeline = _make_pipeline()
        for p in pipeline.domain_gate.parameters():
            p.grad = torch.full_like(p, float("nan"))
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("gate", msg)

    def test_nan_in_analyzer_detected(self):
        """NaN in goal_analyzer should be caught and named."""
        pipeline = _make_pipeline()
        for p in pipeline.goal_analyzer.parameters():
            p.grad = torch.full_like(p, float("nan"))
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("analyzer", msg)

    def test_nan_in_bridge_detected(self):
        """NaN in bridge should be caught and named."""
        pipeline = _make_pipeline()
        for p in pipeline.bridge.parameters():
            p.grad = torch.full_like(p, float("nan"))
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("bridge", msg)

    def test_inf_in_gate_detected(self):
        """Inf in domain_gate should be caught and named."""
        pipeline = _make_pipeline()
        for p in pipeline.domain_gate.parameters():
            p.grad = torch.full_like(p, float("inf"))
        healthy, msg = check_gradient_health(pipeline)
        self.assertFalse(healthy)
        self.assertIn("gate", msg)

    def test_all_none_grads_healthy(self):
        """Pipeline where every parameter has grad=None is healthy."""
        pipeline = _make_pipeline()
        for mod in [
            pipeline.domain_gate,
            pipeline.goal_analyzer,
            pipeline.bridge,
            pipeline.decoder,
        ]:
            for p in mod.parameters():
                p.grad = None
        healthy, msg = check_gradient_health(pipeline)
        self.assertTrue(healthy)
        self.assertIsNone(msg)

    def test_return_types(self):
        """Return type is always (bool, str|None)."""
        pipeline = _make_pipeline()
        result = check_gradient_health(pipeline)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)

        pipeline_bad = _make_pipeline(nan_param=True)
        _, msg = check_gradient_health(pipeline_bad)
        self.assertIsInstance(msg, str)


# ---------------------------------------------------------------------------
# check_gradient_abort
# ---------------------------------------------------------------------------


class TestCheckGradientAbort(unittest.TestCase):
    def test_step_not_divisible_returns_none(self):
        """Step not on check interval -> no check, return None."""
        safety = {"gradient_check_every_n_steps": 10, "nan_abort": True}
        pipeline = _make_pipeline(nan_param=True)
        result = check_gradient_abort(7, safety, pipeline, MagicMock())
        self.assertIsNone(result)

    def test_healthy_gradients_returns_none(self):
        safety = {"gradient_check_every_n_steps": 5, "nan_abort": True}
        pipeline = _make_pipeline()
        result = check_gradient_abort(10, safety, pipeline, MagicMock())
        self.assertIsNone(result)

    def test_nan_grad_with_nan_abort_true(self):
        safety = {
            "gradient_check_every_n_steps": 5,
            "nan_abort": True,
            "nan_checkpoint_before_abort": True,
        }
        pipeline = _make_pipeline(nan_param=True)
        save_fn = MagicMock()
        result = check_gradient_abort(10, safety, pipeline, save_fn)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "nan_abort")
        self.assertEqual(result["step"], 10)

    def test_nan_abort_false_returns_none(self):
        """Even with bad gradients, nan_abort=False means no abort."""
        safety = {"gradient_check_every_n_steps": 5, "nan_abort": False}
        pipeline = _make_pipeline(nan_param=True)
        result = check_gradient_abort(10, safety, pipeline, MagicMock())
        self.assertIsNone(result)

    def test_nan_checkpoint_before_abort_triggers_save(self):
        safety = {
            "gradient_check_every_n_steps": 5,
            "nan_abort": True,
            "nan_checkpoint_before_abort": True,
        }
        pipeline = _make_pipeline(nan_param=True)
        save_fn = MagicMock()
        check_gradient_abort(15, safety, pipeline, save_fn)
        save_fn.assert_called_once_with(15)

    def test_nan_checkpoint_before_abort_false_skips_save(self):
        safety = {
            "gradient_check_every_n_steps": 5,
            "nan_abort": True,
            "nan_checkpoint_before_abort": False,
        }
        pipeline = _make_pipeline(nan_param=True)
        save_fn = MagicMock()
        check_gradient_abort(15, safety, pipeline, save_fn)
        save_fn.assert_not_called()

    def test_nan_checkpoint_default_true(self):
        """When nan_checkpoint_before_abort is absent, defaults to True."""
        safety = {
            "gradient_check_every_n_steps": 5,
            "nan_abort": True,
        }
        pipeline = _make_pipeline(nan_param=True)
        save_fn = MagicMock()
        check_gradient_abort(10, safety, pipeline, save_fn)
        save_fn.assert_called_once_with(10)

    def test_step_zero_is_divisible(self):
        """Step 0 is divisible by any interval, so check runs."""
        safety = {"gradient_check_every_n_steps": 10, "nan_abort": True}
        pipeline = _make_pipeline()
        result = check_gradient_abort(0, safety, pipeline, MagicMock())
        self.assertIsNone(result)  # healthy grads -> None

    def test_inf_grad_also_aborts(self):
        safety = {
            "gradient_check_every_n_steps": 1,
            "nan_abort": True,
            "nan_checkpoint_before_abort": False,
        }
        pipeline = _make_pipeline(inf_param=True)
        result = check_gradient_abort(1, safety, pipeline, MagicMock())
        self.assertEqual(result["status"], "nan_abort")
        self.assertEqual(result["step"], 1)


# ---------------------------------------------------------------------------
# build_checkpoint_data
# ---------------------------------------------------------------------------


class TestBuildCheckpointData(unittest.TestCase):
    def test_exact_field_values(self):
        decoder = _make_decoder_with_known_weights()
        loss_dict = {
            "L_total": 2.5,
            "L_ce": 1.0,
            "L_margin": 0.8,
            "L_repair": 0.7,
            "w_ce": 0.4,
            "w_margin": 0.3,
        }
        metrics = PABMetricsSnapshot(
            val_loss=1.2,
            tier_accuracies={"tier1": 0.9},
            domain_accuracies={"algebra": 0.85},
            tactic_accuracies={"simp": 0.75},
        )
        bridge_features = np.array([[1.0, 2.0], [3.0, 4.0]])

        data = build_checkpoint_data(100, loss_dict, metrics, bridge_features, decoder)

        self.assertIsInstance(data, CheckpointData)
        self.assertEqual(data.step, 100)
        self.assertEqual(data.train_loss, 2.5)
        self.assertEqual(data.val_loss, 1.2)
        self.assertEqual(data.loss_components, {"ce": 1.0, "margin": 0.8, "repair": 0.7})
        self.assertEqual(data.adaptive_weights, {"w_ce": 0.4, "w_margin": 0.3})
        self.assertEqual(data.tier_accuracies, {"tier1": 0.9})
        np.testing.assert_array_equal(data.bottleneck_embeddings, bridge_features)
        self.assertEqual(data.domain_accuracies, {"algebra": 0.85})
        self.assertEqual(data.tactic_accuracies, {"simp": 0.75})
        # decoder_weight_signs from known decoder
        expected_signs = np.array([1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0])
        np.testing.assert_array_equal(data.decoder_weight_signs, expected_signs)

    def test_none_metrics_uses_defaults(self):
        decoder = _make_decoder_with_known_weights()
        loss_dict = {"L_total": 1.0}
        data = build_checkpoint_data(50, loss_dict, None, None, decoder)
        self.assertIsNone(data.val_loss)
        self.assertEqual(data.tier_accuracies, {})
        self.assertEqual(data.domain_accuracies, {})
        self.assertEqual(data.tactic_accuracies, {})
        self.assertIsNone(data.bottleneck_embeddings)

    def test_missing_loss_components_default_zero(self):
        decoder = _make_decoder_with_known_weights()
        loss_dict = {"L_total": 3.0}
        data = build_checkpoint_data(10, loss_dict, None, None, decoder)
        self.assertEqual(data.loss_components, {"ce": 0.0, "margin": 0.0, "repair": 0.0})

    def test_adaptive_weights_filters_non_w_prefix(self):
        decoder = _make_decoder_with_known_weights()
        loss_dict = {
            "L_total": 1.0,
            "w_alpha": 0.5,
            "w_beta": 0.3,
            "something_else": 0.9,
        }
        data = build_checkpoint_data(10, loss_dict, None, None, decoder)
        self.assertEqual(data.adaptive_weights, {"w_alpha": 0.5, "w_beta": 0.3})

    def test_adaptive_weights_skips_non_float(self):
        decoder = _make_decoder_with_known_weights()
        loss_dict = {
            "L_total": 1.0,
            "w_ok": 0.5,
            "w_bad": "not_a_float",
        }
        data = build_checkpoint_data(10, loss_dict, None, None, decoder)
        self.assertEqual(data.adaptive_weights, {"w_ok": 0.5})

    def test_decoder_without_ternary_layers(self):
        decoder = nn.Sequential(nn.Linear(4, 4))
        loss_dict = {"L_total": 1.0}
        data = build_checkpoint_data(1, loss_dict, None, None, decoder)
        self.assertIsNone(data.decoder_weight_signs)


# ---------------------------------------------------------------------------
# record_pab_checkpoint
# ---------------------------------------------------------------------------


class TestRecordPABCheckpoint(unittest.TestCase):
    def test_no_tracker_returns_none(self):
        infra = SimpleNamespace(pab_tracker=None)
        result = record_pab_checkpoint(
            step=100,
            loss_dict={"L_total": 1.0},
            config={},
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)

    def test_step_not_on_interval_returns_none(self):
        tracker = MagicMock()
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {"pab": {"checkpoint_interval": 50}}
        result = record_pab_checkpoint(
            step=23,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)
        tracker.record.assert_not_called()

    def test_step_on_interval_records(self):
        tracker = MagicMock()
        tracker.should_early_exit.return_value = False
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {"pab": {"checkpoint_interval": 50, "early_exit_enabled": False}}
        result = record_pab_checkpoint(
            step=100,
            loss_dict={"L_total": 2.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)
        tracker.record.assert_called_once()
        recorded_data = tracker.record.call_args[0][0]
        self.assertIsInstance(recorded_data, CheckpointData)
        self.assertEqual(recorded_data.step, 100)
        self.assertEqual(recorded_data.train_loss, 2.0)

    def test_early_exit_triggered(self):
        tracker = MagicMock()
        tracker.should_early_exit.return_value = True
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {"pab": {"checkpoint_interval": 50, "early_exit_enabled": True}}
        result = record_pab_checkpoint(
            step=200,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "pab_early_exit")
        self.assertEqual(result["step"], 200)

    def test_early_exit_disabled_returns_none(self):
        tracker = MagicMock()
        tracker.should_early_exit.return_value = True
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {"pab": {"checkpoint_interval": 50, "early_exit_enabled": False}}
        result = record_pab_checkpoint(
            step=100,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)

    def test_default_interval_50(self):
        """When checkpoint_interval is absent, default is 50."""
        tracker = MagicMock()
        tracker.should_early_exit.return_value = False
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {"pab": {}}

        # Step 25 is not divisible by 50
        result = record_pab_checkpoint(
            step=25,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)
        tracker.record.assert_not_called()

        # Step 50 is divisible by 50
        result = record_pab_checkpoint(
            step=50,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)
        tracker.record.assert_called_once()

    def test_no_pab_config_key_uses_defaults(self):
        """When 'pab' key is missing from config, defaults apply."""
        tracker = MagicMock()
        tracker.should_early_exit.return_value = False
        infra = SimpleNamespace(pab_tracker=tracker)
        config = {}  # no "pab" key

        result = record_pab_checkpoint(
            step=50,
            loss_dict={"L_total": 1.0},
            config=config,
            infra=infra,
            last_bridge_features=None,
            decoder=_make_decoder_with_known_weights(),
        )
        self.assertIsNone(result)
        tracker.record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
