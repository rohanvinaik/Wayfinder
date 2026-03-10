"""PAB tracker tests — core, tier, and aux metric recording."""

import unittest

import numpy as np

from src.pab_tracker import CheckpointData, PABTracker


class TestRecordCoreMetrics(unittest.TestCase):
    """Test _record_core_metrics state changes through finalize()."""

    def test_first_checkpoint_stability_zero(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        profile = tracker.finalize()
        self.assertEqual(profile.core.stability[0], 0.0)

    def test_second_checkpoint_stability_computed(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.0))
        profile = tracker.finalize()
        # stability = |2.0 - 1.0| / (|2.0| + 1e-8) = 1.0 / 2.0 = 0.5
        self.assertAlmostEqual(profile.core.stability[1], 0.5, places=5)

    def test_val_loss_records_gen_gap(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=1.0, val_loss=1.5))
        profile = tracker.finalize()
        # gen_gap = val_loss - train_loss = 0.5
        self.assertAlmostEqual(profile.core.generalization_gap[0], 0.5, places=5)

    def test_no_val_loss_gen_gap_carries_forward(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=1.0, val_loss=1.5))
        tracker.record(CheckpointData(step=100, train_loss=0.8))  # no val_loss
        profile = tracker.finalize()
        # Second gen_gap should carry forward from first (0.5)
        self.assertAlmostEqual(profile.core.generalization_gap[1], 0.5, places=5)

    def test_no_val_loss_ever_gen_gap_zero(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.core.generalization_gap[0], 0.0)

    def test_bottleneck_embeddings_repr_evolution(self):
        tracker = PABTracker(experiment_id="CORE")
        emb1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        tracker.record(CheckpointData(step=50, train_loss=2.0, bottleneck_embeddings=emb1))
        profile = tracker.finalize()
        # First repr_evolution with no prev_mean -> r_t = 1.0
        self.assertAlmostEqual(profile.core.representation_evolution[0], 1.0, places=5)

    def test_bottleneck_embeddings_second_checkpoint(self):
        tracker = PABTracker(experiment_id="CORE")
        emb1 = np.array([[1.0, 0.0], [1.0, 0.0]])
        emb2 = np.array([[1.0, 0.0], [1.0, 0.0]])  # identical
        tracker.record(CheckpointData(step=50, train_loss=2.0, bottleneck_embeddings=emb1))
        tracker.record(CheckpointData(step=100, train_loss=1.5, bottleneck_embeddings=emb2))
        profile = tracker.finalize()
        # Same embeddings -> cos_sim=1.0, r_t = 1.0 - 1.0 = 0.0
        self.assertAlmostEqual(profile.core.representation_evolution[1], 0.0, places=4)

    def test_no_bottleneck_repr_evolution_carries_forward(self):
        tracker = PABTracker(experiment_id="CORE")
        emb1 = np.array([[1.0, 0.0]])
        tracker.record(CheckpointData(step=50, train_loss=2.0, bottleneck_embeddings=emb1))
        tracker.record(CheckpointData(step=100, train_loss=1.5))
        profile = tracker.finalize()
        # Second checkpoint has no embeddings, should carry forward 1.0
        self.assertAlmostEqual(profile.core.representation_evolution[1], 1.0, places=5)

    def test_no_bottleneck_ever_repr_evolution_zero(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        profile = tracker.finalize()
        self.assertEqual(profile.core.representation_evolution[0], 0.0)

    def test_predictability_single_loss_is_zero(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        profile = tracker.finalize()
        self.assertEqual(profile.core.predictability[0], 0.0)

    def test_predictability_two_losses_is_zero(self):
        # Two losses yield one delta, variance of one element = 0.0
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.5))
        profile = tracker.finalize()
        self.assertEqual(profile.core.predictability[1], 0.0)

    def test_predictability_three_losses_computed(self):
        tracker = PABTracker(experiment_id="CORE")
        tracker.record(CheckpointData(step=50, train_loss=2.0))
        tracker.record(CheckpointData(step=100, train_loss=1.5))
        tracker.record(CheckpointData(step=150, train_loss=1.2))
        profile = tracker.finalize()
        # Losses: [2.0, 1.5, 1.2]
        # At step 3: deltas from index 1..2 = [-0.5, -0.3]
        # var([-0.5, -0.3]) = ((0.1)^2 + (0.1)^2)/2 = 0.01
        expected = float(np.var([-0.5, -0.3]))
        self.assertAlmostEqual(profile.core.predictability[2], expected, places=7)


class TestRecordTierMetrics(unittest.TestCase):
    """Test _record_tier_metrics through finalize()."""

    def test_no_tier_accuracies_defaults_to_zero(self):
        tracker = PABTracker(experiment_id="TIER")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.tiers.tier1_accuracy[0], 0.0)
        self.assertEqual(profile.tiers.tier2_accuracy[0], 0.0)
        self.assertEqual(profile.tiers.tier3_accuracy[0], 0.0)

    def test_tier_accuracies_recorded(self):
        tracker = PABTracker(experiment_id="TIER")
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                tier_accuracies={"tier1": 0.85, "tier2": 0.72, "tier3": 0.40},
            )
        )
        profile = tracker.finalize()
        self.assertEqual(profile.tiers.tier1_accuracy[0], 0.85)
        self.assertEqual(profile.tiers.tier2_accuracy[0], 0.72)
        self.assertEqual(profile.tiers.tier3_accuracy[0], 0.40)

    def test_no_decoder_signs_crystallization_zero(self):
        tracker = PABTracker(experiment_id="TIER")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.tiers.ternary_crystallization[0], 0.0)

    def test_crystallization_carries_forward(self):
        tracker = PABTracker(experiment_id="TIER")
        signs = np.array([1.0, -1.0, 0.0])
        tracker.record(CheckpointData(step=50, train_loss=1.0, decoder_weight_signs=signs))
        tracker.record(CheckpointData(step=100, train_loss=0.9, decoder_weight_signs=signs))
        # Third checkpoint has no decoder_weight_signs
        tracker.record(CheckpointData(step=150, train_loss=0.8))
        profile = tracker.finalize()
        # Third crystallization carries forward from second
        self.assertEqual(
            profile.tiers.ternary_crystallization[2],
            profile.tiers.ternary_crystallization[1],
        )


class TestRecordAuxMetrics(unittest.TestCase):
    """Test _record_aux_metrics accumulation through finalize()."""

    def test_loss_components_recorded(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                loss_components={"ce": 0.8, "margin": 0.15, "repair": 0.05},
            )
        )
        profile = tracker.finalize()
        self.assertEqual(profile.losses.loss_ce[0], 0.8)
        self.assertEqual(profile.losses.loss_margin[0], 0.15)
        self.assertEqual(profile.losses.loss_repair[0], 0.05)

    def test_no_loss_components_defaults_to_zero(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.losses.loss_ce[0], 0.0)
        self.assertEqual(profile.losses.loss_margin[0], 0.0)
        self.assertEqual(profile.losses.loss_repair[0], 0.0)

    def test_adaptive_weights_recorded(self):
        tracker = PABTracker(experiment_id="AUX")
        weights = {"ce": 0.5, "margin": 0.3, "repair": 0.2}
        tracker.record(CheckpointData(step=50, train_loss=1.0, adaptive_weights=weights))
        profile = tracker.finalize()
        self.assertEqual(profile.losses.loss_adaptive_weights[0], weights)

    def test_no_adaptive_weights_empty_dict(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.losses.loss_adaptive_weights[0], {})

    def test_domain_accuracies_accumulated(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                domain_accuracies={"algebra": 0.70, "topology": 0.55},
            )
        )
        tracker.record(
            CheckpointData(
                step=100,
                train_loss=0.8,
                domain_accuracies={"algebra": 0.75, "topology": 0.60},
            )
        )
        profile = tracker.finalize()
        self.assertEqual(profile.domains.domain_progression["algebra"], [0.70, 0.75])
        self.assertEqual(profile.domains.domain_progression["topology"], [0.55, 0.60])

    def test_tactic_accuracies_accumulated(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                domain_accuracies={"algebra": 0.70},
                tactic_accuracies={"simp": 0.80, "ring": 0.65},
            )
        )
        tracker.record(
            CheckpointData(
                step=100,
                train_loss=0.8,
                domain_accuracies={"algebra": 0.75},
                tactic_accuracies={"simp": 0.85, "ring": 0.70},
            )
        )
        profile = tracker.finalize()
        self.assertEqual(profile.domains.tactic_progression["simp"], [0.80, 0.85])
        self.assertEqual(profile.domains.tactic_progression["ring"], [0.65, 0.70])

    def test_no_domain_accuracies_empty(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.domains.domain_progression, {})

    def test_no_tactic_accuracies_empty(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(CheckpointData(step=50, train_loss=1.0))
        profile = tracker.finalize()
        self.assertEqual(profile.domains.tactic_progression, {})

    def test_domain_new_key_added_later(self):
        tracker = PABTracker(experiment_id="AUX")
        tracker.record(
            CheckpointData(
                step=50,
                train_loss=1.0,
                domain_accuracies={"algebra": 0.70},
            )
        )
        tracker.record(
            CheckpointData(
                step=100,
                train_loss=0.8,
                domain_accuracies={"algebra": 0.75, "analysis": 0.50},
            )
        )
        profile = tracker.finalize()
        self.assertEqual(profile.domains.domain_progression["algebra"], [0.70, 0.75])
        self.assertEqual(profile.domains.domain_progression["analysis"], [0.50])


if __name__ == "__main__":
    unittest.main()
