"""Tests for specialist navigator — bank-cluster decomposition, fusion, ExecutionSlot."""

import unittest

import torch

from src.nav_contracts import BANK_NAMES, NavOutput
from src.som_contracts import ExecutionOutput
from src.specialist_navigator import (
    SPECIALIST_A_BANKS,
    SPECIALIST_B_BANKS,
    ExecutionSlot,
    SpecialistNavigator,
    fuse_specialist_outputs,
    fuse_to_nav_output,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

NUM_ANCHORS = 100
FEATURE_DIM = 256
BRIDGE_DIM = 128
HIDDEN_DIM = 256
HISTORY_DIM = 64
BATCH_SIZE = 2


def _make_nav_output(
    directions=None,
    confidences=None,
    anchor_scores=None,
    progress=0.5,
    critic=0.8,
):
    return NavOutput(
        directions=directions or {"domain": 1, "context": 0},
        direction_confidences=confidences or {"domain": 0.9, "context": 0.7},
        anchor_scores=anchor_scores or {"simp": 0.8, "apply": 0.6},
        progress=progress,
        critic_score=critic,
    )


def _make_specialist(name, banks, **kwargs):
    defaults = dict(
        feature_dim=FEATURE_DIM,
        bridge_dim=BRIDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_anchors=NUM_ANCHORS,
        num_layers=2,
        ternary_enabled=True,
        history_dim=HISTORY_DIM,
    )
    defaults.update(kwargs)
    return SpecialistNavigator(name=name, banks=banks, **defaults)


def _make_features(batch=BATCH_SIZE):
    return torch.randn(batch, FEATURE_DIM)


def _make_history(batch=BATCH_SIZE):
    return torch.randn(batch, HISTORY_DIM)


def _make_default_config():
    return {
        "specialists": {
            "A": {"banks": ["domain", "context"], "bridge_dim": 128, "hidden_dim": 256},
            "B": {
                "banks": ["structure", "automation", "depth", "decomposition"],
                "bridge_dim": 128,
                "hidden_dim": 256,
            },
        },
        "model": {
            "goal_analyzer": {"feature_dim": 256},
            "navigator": {"num_anchors": NUM_ANCHORS, "ternary_enabled": True},
            "bridge": {"history_dim": 64},
        },
    }


# ---------------------------------------------------------------------------
# Bank constants
# ---------------------------------------------------------------------------


class TestBankConstants(unittest.TestCase):
    def test_specialist_a_banks(self):
        self.assertEqual(SPECIALIST_A_BANKS, ["domain", "context"])

    def test_specialist_b_banks(self):
        self.assertEqual(
            SPECIALIST_B_BANKS,
            ["structure", "automation", "depth", "decomposition"],
        )

    def test_no_overlap(self):
        overlap = set(SPECIALIST_A_BANKS) & set(SPECIALIST_B_BANKS)
        self.assertEqual(overlap, set())

    def test_cover_all_six_banks(self):
        combined = set(SPECIALIST_A_BANKS) | set(SPECIALIST_B_BANKS)
        self.assertEqual(combined, set(BANK_NAMES))


# ---------------------------------------------------------------------------
# SpecialistNavigator construction
# ---------------------------------------------------------------------------


class TestSpecialistNavigatorConstruction(unittest.TestCase):
    def test_name_and_banks_stored(self):
        spec = _make_specialist("A", ["domain", "context"])
        self.assertEqual(spec.name, "A")
        self.assertEqual(spec.banks, ["domain", "context"])

    def test_bridge_created(self):
        spec = _make_specialist("A", ["domain", "context"])
        self.assertIsNotNone(spec.bridge)

    def test_navigator_created(self):
        spec = _make_specialist("B", SPECIALIST_B_BANKS)
        self.assertIsNotNone(spec.navigator)
        self.assertEqual(spec.navigator.navigable_banks, SPECIALIST_B_BANKS)

    def test_is_nn_module(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        self.assertIsInstance(spec, torch.nn.Module)


# ---------------------------------------------------------------------------
# SpecialistNavigator forward
# ---------------------------------------------------------------------------


class TestSpecialistNavigatorForward(unittest.TestCase):
    def test_direction_logits_keys_match_assigned_banks(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        direction_logits, _, _, _ = spec.forward(features)
        self.assertEqual(set(direction_logits.keys()), set(SPECIALIST_A_BANKS))

    def test_direction_logits_keys_specialist_b(self):
        spec = _make_specialist("B", SPECIALIST_B_BANKS)
        features = _make_features()
        direction_logits, _, _, _ = spec.forward(features)
        self.assertEqual(set(direction_logits.keys()), set(SPECIALIST_B_BANKS))

    def test_direction_logits_shape(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        direction_logits, _, _, _ = spec.forward(features)
        for bank in SPECIALIST_A_BANKS:
            self.assertEqual(direction_logits[bank].shape, (BATCH_SIZE, 3))

    def test_anchor_logits_shape(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        _, anchor_logits, _, _ = spec.forward(features)
        self.assertEqual(anchor_logits.shape, (BATCH_SIZE, NUM_ANCHORS))

    def test_progress_shape(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        _, _, progress, _ = spec.forward(features)
        self.assertEqual(progress.shape, (BATCH_SIZE, 1))

    def test_critic_shape(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        _, _, _, critic = spec.forward(features)
        self.assertEqual(critic.shape, (BATCH_SIZE, 1))

    def test_forward_with_proof_history(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        history = _make_history()
        direction_logits, anchor_logits, _progress, _critic = spec.forward(
            features, proof_history=history
        )
        self.assertEqual(set(direction_logits.keys()), set(SPECIALIST_A_BANKS))
        self.assertEqual(anchor_logits.shape, (BATCH_SIZE, NUM_ANCHORS))

    def test_forward_without_proof_history(self):
        """Forward with None proof_history should still work (bridge pads zeros)."""
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features()
        _direction_logits, anchor_logits, _progress, _critic = spec.forward(
            features, proof_history=None
        )
        self.assertEqual(anchor_logits.shape, (BATCH_SIZE, NUM_ANCHORS))


# ---------------------------------------------------------------------------
# SpecialistNavigator predict
# ---------------------------------------------------------------------------


class TestSpecialistNavigatorPredict(unittest.TestCase):
    def test_returns_nav_output(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features(batch=1)
        result = spec.predict(features)
        self.assertIsInstance(result, NavOutput)

    def test_directions_keys_match_banks(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features(batch=1)
        result = spec.predict(features)
        self.assertEqual(set(result.directions.keys()), set(SPECIALIST_A_BANKS))

    def test_direction_values_are_ternary(self):
        spec = _make_specialist("B", SPECIALIST_B_BANKS)
        features = _make_features(batch=1)
        result = spec.predict(features)
        for bank, val in result.directions.items():
            self.assertIn(val, {-1, 0, 1}, f"Bank {bank} has non-ternary direction {val}")

    def test_predict_with_proof_history(self):
        spec = _make_specialist("A", SPECIALIST_A_BANKS)
        features = _make_features(batch=1)
        history = _make_history(batch=1)
        result = spec.predict(features, proof_history=history)
        self.assertIsInstance(result, NavOutput)
        self.assertEqual(set(result.directions.keys()), set(SPECIALIST_A_BANKS))


# ---------------------------------------------------------------------------
# fuse_specialist_outputs
# ---------------------------------------------------------------------------


class TestFuseSpecialistOutputs(unittest.TestCase):
    def test_directions_merged_from_both_specialists(self):
        out_a = _make_nav_output(
            directions={"domain": 1, "context": 0},
            confidences={"domain": 0.9, "context": 0.7},
        )
        out_b = _make_nav_output(
            directions={"structure": -1, "automation": 1, "depth": 0, "decomposition": 1},
            confidences={"structure": 0.8, "automation": 0.6, "depth": 0.5, "decomposition": 0.7},
        )
        fused = fuse_specialist_outputs({"A": out_a, "B": out_b})
        self.assertIsInstance(fused, ExecutionOutput)
        self.assertEqual(set(fused.directions.keys()), set(BANK_NAMES))
        self.assertEqual(fused.directions["domain"], 1)
        self.assertEqual(fused.directions["structure"], -1)

    def test_critic_is_min(self):
        out_a = _make_nav_output(critic=0.9)
        out_b = _make_nav_output(critic=0.3)
        fused = fuse_specialist_outputs({"A": out_a, "B": out_b})
        self.assertAlmostEqual(fused.critic, 0.3, places=5)

    def test_critic_is_min_reversed(self):
        out_a = _make_nav_output(critic=0.2)
        out_b = _make_nav_output(critic=0.8)
        fused = fuse_specialist_outputs({"A": out_a, "B": out_b})
        self.assertAlmostEqual(fused.critic, 0.2, places=5)

    def test_progress_is_confidence_weighted(self):
        out_a = _make_nav_output(
            progress=0.4,
            confidences={"domain": 0.8, "context": 0.8},
        )
        out_b = _make_nav_output(
            progress=0.6,
            confidences={"domain": 0.6, "context": 0.6},
        )
        fused = fuse_specialist_outputs({"A": out_a, "B": out_b})
        # A mean_conf = 0.8, B mean_conf = 0.6
        # weighted progress = (0.4*0.8 + 0.6*0.6) / (0.8 + 0.6) = (0.32+0.36)/1.4
        expected = (0.4 * 0.8 + 0.6 * 0.6) / (0.8 + 0.6)
        self.assertAlmostEqual(fused.progress, expected, places=4)

    def test_single_specialist_passthrough(self):
        out = _make_nav_output(progress=0.7, critic=0.5)
        fused = fuse_specialist_outputs({"solo": out})
        self.assertAlmostEqual(fused.progress, 0.7, places=4)
        self.assertAlmostEqual(fused.critic, 0.5, places=5)
        self.assertEqual(fused.directions, out.directions)


# ---------------------------------------------------------------------------
# fuse_to_nav_output
# ---------------------------------------------------------------------------


class TestFuseToNavOutput(unittest.TestCase):
    def test_returns_nav_output_type(self):
        out_a = _make_nav_output()
        fused = fuse_to_nav_output({"A": out_a})
        self.assertIsInstance(fused, NavOutput)

    def test_anchor_scores_max_pooled(self):
        out_a = _make_nav_output(anchor_scores={"simp": 0.9, "apply": 0.3})
        out_b = _make_nav_output(anchor_scores={"simp": 0.4, "apply": 0.7, "rw": 0.5})
        fused = fuse_to_nav_output({"A": out_a, "B": out_b})
        self.assertAlmostEqual(fused.anchor_scores["simp"], 0.9, places=5)
        self.assertAlmostEqual(fused.anchor_scores["apply"], 0.7, places=5)
        self.assertAlmostEqual(fused.anchor_scores["rw"], 0.5, places=5)

    def test_directions_merged(self):
        out_a = _make_nav_output(
            directions={"domain": 1, "context": -1},
            confidences={"domain": 0.9, "context": 0.8},
        )
        out_b = _make_nav_output(
            directions={"structure": 0, "depth": 1},
            confidences={"structure": 0.7, "depth": 0.6},
        )
        fused = fuse_to_nav_output({"A": out_a, "B": out_b})
        self.assertIn("domain", fused.directions)
        self.assertIn("structure", fused.directions)
        self.assertEqual(fused.directions["domain"], 1)
        self.assertEqual(fused.directions["structure"], 0)


# ---------------------------------------------------------------------------
# ExecutionSlot construction
# ---------------------------------------------------------------------------


class TestExecutionSlotConstruction(unittest.TestCase):
    def test_construction_with_two_specialists(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        self.assertIn("A", slot.specialists)
        self.assertIn("B", slot.specialists)

    def test_is_nn_module(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        slot = ExecutionSlot({"A": spec_a})
        self.assertIsInstance(slot, torch.nn.Module)


# ---------------------------------------------------------------------------
# ExecutionSlot._get_specialist
# ---------------------------------------------------------------------------


class TestExecutionSlotGetSpecialist(unittest.TestCase):
    def test_returns_specialist_navigator(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        retrieved = slot._get_specialist("A")
        self.assertIsInstance(retrieved, SpecialistNavigator)
        self.assertEqual(retrieved.name, "A")

    def test_get_specialist_b(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        retrieved = slot._get_specialist("B")
        self.assertEqual(retrieved.banks, SPECIALIST_B_BANKS)


# ---------------------------------------------------------------------------
# ExecutionSlot.forward
# ---------------------------------------------------------------------------


class TestExecutionSlotForward(unittest.TestCase):
    def test_returns_dict_with_specialist_names(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        features = _make_features()
        results = slot.forward(features)
        self.assertEqual(set(results.keys()), {"A", "B"})

    def test_each_specialist_returns_four_tuple(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        features = _make_features()
        results = slot.forward(features)
        for name in ("A", "B"):
            direction_logits, anchor_logits, progress, critic = results[name]
            self.assertIsInstance(direction_logits, dict)
            self.assertIsInstance(anchor_logits, torch.Tensor)
            self.assertIsInstance(progress, torch.Tensor)
            self.assertIsInstance(critic, torch.Tensor)


# ---------------------------------------------------------------------------
# ExecutionSlot.predict
# ---------------------------------------------------------------------------


class TestExecutionSlotPredict(unittest.TestCase):
    def test_returns_fused_nav_output(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        features = _make_features(batch=1)
        result = slot.predict(features)
        self.assertIsInstance(result, NavOutput)

    def test_fused_output_has_all_six_bank_directions(self):
        spec_a = _make_specialist("A", SPECIALIST_A_BANKS)
        spec_b = _make_specialist("B", SPECIALIST_B_BANKS)
        slot = ExecutionSlot({"A": spec_a, "B": spec_b})
        features = _make_features(batch=1)
        result = slot.predict(features)
        self.assertEqual(set(result.directions.keys()), set(BANK_NAMES))


# ---------------------------------------------------------------------------
# ExecutionSlot.from_config
# ---------------------------------------------------------------------------


class TestExecutionSlotFromConfig(unittest.TestCase):
    def test_from_config_creates_slot(self):
        config = _make_default_config()
        slot = ExecutionSlot.from_config(config)
        self.assertIsInstance(slot, ExecutionSlot)
        self.assertIn("A", slot.specialists)
        self.assertIn("B", slot.specialists)

    def test_from_config_specialist_banks(self):
        config = _make_default_config()
        slot = ExecutionSlot.from_config(config)
        spec_a = slot._get_specialist("A")
        spec_b = slot._get_specialist("B")
        self.assertEqual(spec_a.banks, ["domain", "context"])
        self.assertEqual(spec_b.banks, ["structure", "automation", "depth", "decomposition"])

    def test_from_config_forward_runs(self):
        config = _make_default_config()
        slot = ExecutionSlot.from_config(config)
        features = _make_features()
        results = slot.forward(features)
        self.assertEqual(set(results.keys()), {"A", "B"})

    def test_from_config_predict_returns_all_banks(self):
        config = _make_default_config()
        slot = ExecutionSlot.from_config(config)
        features = _make_features(batch=1)
        result = slot.predict(features)
        self.assertEqual(set(result.directions.keys()), set(BANK_NAMES))

    def test_from_config_respects_num_anchors(self):
        config = _make_default_config()
        slot = ExecutionSlot.from_config(config)
        spec_a = slot._get_specialist("A")
        self.assertEqual(spec_a.navigator.num_anchors, NUM_ANCHORS)


if __name__ == "__main__":
    unittest.main()
