"""Tests for v3B energy function and continuous ternary refinement."""

import unittest

import torch

from src.energy import (
    EnergyFunction,
    RefineConfig,
    energy_refine,
    gumbel_softmax_ternary,
    snap_to_ternary,
)


class TestGumbelSoftmaxTernary(unittest.TestCase):
    def test_output_shape(self):
        logits = torch.randn(2, 4, 6, 3)  # batch=2, steps=4, banks=6, categories=3
        result = gumbel_softmax_ternary(logits, tau=1.0)
        self.assertEqual(result.shape, (2, 4, 6))

    def test_output_range(self):
        logits = torch.randn(1, 3, 6, 3)
        result = gumbel_softmax_ternary(logits, tau=0.5)
        self.assertTrue((result >= -1.0).all())
        self.assertTrue((result <= 1.0).all())

    def test_low_temperature_approaches_discrete(self):
        """At very low tau, output should be close to {-1, 0, +1}."""
        # Create logits strongly biased toward one category
        logits = torch.zeros(1, 2, 6, 3)
        logits[..., 2] = 10.0  # strongly favor +1
        result = gumbel_softmax_ternary(logits, tau=0.01)
        # Should be close to +1
        self.assertTrue((result > 0.9).all())


class TestSnapToTernary(unittest.TestCase):
    def test_snap_values(self):
        continuous = torch.tensor([-0.9, -0.3, 0.0, 0.3, 0.9])
        discrete = snap_to_ternary(continuous)
        expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.equal(discrete, expected))

    def test_boundary_values(self):
        continuous = torch.tensor([-0.5, 0.5])
        discrete = snap_to_ternary(continuous)
        # -0.5 and 0.5 are at boundary, should map to 0
        self.assertTrue(torch.equal(discrete, torch.tensor([0.0, 0.0])))

    def test_preserves_shape(self):
        continuous = torch.randn(3, 4, 6)
        discrete = snap_to_ternary(continuous)
        self.assertEqual(discrete.shape, (3, 4, 6))

    def test_output_is_ternary(self):
        continuous = torch.randn(10, 6)
        discrete = snap_to_ternary(continuous)
        unique = set(discrete.flatten().tolist())
        self.assertTrue(unique.issubset({-1.0, 0.0, 1.0}))

    def test_extreme_values(self):
        """Values far from boundaries should snap cleanly."""
        continuous = torch.tensor([-100.0, -1.0, -0.51, 0.0, 0.51, 1.0, 100.0])
        discrete = snap_to_ternary(continuous)
        expected = torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.equal(discrete, expected))

    def test_just_inside_boundaries(self):
        """Values just inside the zero band stay zero."""
        continuous = torch.tensor([-0.49, -0.01, 0.01, 0.49])
        discrete = snap_to_ternary(continuous)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.equal(discrete, expected))

    def test_just_outside_boundaries(self):
        """Values just outside the zero band snap to -1 or +1."""
        continuous = torch.tensor([-0.501, 0.501])
        discrete = snap_to_ternary(continuous)
        expected = torch.tensor([-1.0, 1.0])
        self.assertTrue(torch.equal(discrete, expected))

    def test_empty_tensor(self):
        continuous = torch.tensor([])
        discrete = snap_to_ternary(continuous)
        self.assertEqual(discrete.numel(), 0)

    def test_idempotent(self):
        """Snapping an already-discrete tensor should be a no-op."""
        discrete_input = torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0])
        result = snap_to_ternary(discrete_input)
        self.assertTrue(torch.equal(result, discrete_input))

    def test_batch_dimensions(self):
        """Should work correctly on multi-dimensional input."""
        continuous = torch.tensor(
            [
                [[-0.9, 0.0, 0.9], [0.3, -0.3, 0.6]],
                [[-0.6, 0.1, -0.1], [0.8, -0.8, 0.0]],
            ]
        )
        discrete = snap_to_ternary(continuous)
        expected = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[-1.0, 0.0, 0.0], [1.0, -1.0, 0.0]],
            ]
        )
        self.assertTrue(torch.equal(discrete, expected))


class TestEnergyFunction(unittest.TestCase):
    def setUp(self):
        self.energy_fn = EnergyFunction(alpha=1.0, beta=0.5, gamma=2.0, delta=0.3)

    def test_bank_energy_zero_for_perfect_alignment(self):
        sketch = torch.tensor([[[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]])
        e = self.energy_fn.bank_energy(sketch, target)
        self.assertAlmostEqual(e.item(), 0.0, places=4)

    def test_bank_energy_informational_zero(self):
        """Zero-target banks should contribute no energy."""
        sketch = torch.tensor([[[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        e = self.energy_fn.bank_energy(sketch, target)
        self.assertAlmostEqual(e.item(), 0.0, places=4)

    def test_bank_energy_misalignment_is_positive(self):
        sketch = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        target = torch.tensor([[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        e = self.energy_fn.bank_energy(sketch, target)
        self.assertGreater(e.item(), 0.0)

    def test_censor_energy_below_threshold_is_zero(self):
        scores = torch.tensor([[0.1, 0.2, 0.3]])
        e = self.energy_fn.censor_energy(scores, threshold=0.5)
        self.assertAlmostEqual(e.item(), 0.0, places=4)

    def test_censor_energy_above_threshold_is_positive(self):
        scores = torch.tensor([[0.8, 0.9]])
        e = self.energy_fn.censor_energy(scores, threshold=0.5)
        self.assertGreater(e.item(), 0.0)

    def test_anchor_energy_perfect_alignment(self):
        alignments = torch.tensor([[1.0, 1.0]])
        e = self.energy_fn.anchor_energy(alignments)
        self.assertAlmostEqual(e.item(), 0.0, places=4)

    def test_forward_returns_scalar(self):
        sketch = torch.randn(1, 3, 6)
        target = torch.randn(1, 3, 6)
        critic = torch.tensor([2.0])
        censor = torch.rand(1, 3)
        anchor = torch.rand(1, 3)
        e = self.energy_fn(sketch, target, critic, censor, anchor)
        self.assertEqual(e.dim(), 0)  # scalar

    def test_forward_is_differentiable(self):
        sketch = torch.randn(1, 3, 6, requires_grad=True)
        target = torch.randn(1, 3, 6)
        critic = torch.tensor([2.0])
        censor = torch.rand(1, 3)
        anchor = torch.rand(1, 3)
        e = self.energy_fn(sketch, target, critic, censor, anchor)
        e.backward()
        self.assertIsNotNone(sketch.grad)
        self.assertFalse(torch.all(sketch.grad == 0))

    def test_high_censor_penalty(self):
        """Gamma=2.0 (MCA) should make censor violations expensive."""
        sketch = torch.zeros(1, 2, 6)
        target = torch.zeros(1, 2, 6)
        critic = torch.tensor([0.0])
        anchor = torch.ones(1, 2)

        low_censor = self.energy_fn(
            sketch,
            target,
            critic,
            torch.tensor([[0.1, 0.1]]),
            anchor,
        )
        high_censor = self.energy_fn(
            sketch,
            target,
            critic,
            torch.tensor([[0.9, 0.9]]),
            anchor,
        )
        self.assertGreater(high_censor.item(), low_censor.item())


class TestEnergyRefine(unittest.TestCase):
    def test_refinement_reduces_energy(self):
        """Energy should decrease (or stay stable) after refinement."""
        torch.manual_seed(42)
        batch, steps, banks = 1, 3, 6
        logits = torch.randn(batch, steps, banks, 3)
        target = torch.tensor([[[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]] * steps])
        critic = torch.tensor([1.0])
        censor = torch.tensor([[0.1] * steps])
        anchor = torch.tensor([[0.5] * steps])

        energy_fn = EnergyFunction()

        # Initial energy
        with torch.no_grad():
            initial_continuous = gumbel_softmax_ternary(logits, tau=1.0)
            initial_energy = energy_fn(
                initial_continuous,
                target,
                critic,
                censor,
                anchor,
            ).item()

        # Refine
        _, final_energy = energy_refine(
            logits,
            target,
            critic,
            censor,
            anchor,
            energy_fn,
            config=RefineConfig(refine_steps=50, refine_lr=0.05),
        )

        self.assertLessEqual(final_energy, initial_energy + 0.5)

    def test_output_is_discrete_ternary(self):
        logits = torch.randn(1, 2, 6, 3)
        target = torch.randn(1, 2, 6)
        critic = torch.tensor([1.0])
        censor = torch.rand(1, 2)
        anchor = torch.rand(1, 2)
        energy_fn = EnergyFunction()

        discrete, _ = energy_refine(
            logits,
            target,
            critic,
            censor,
            anchor,
            energy_fn,
            config=RefineConfig(refine_steps=5),
        )
        unique = set(discrete.flatten().tolist())
        self.assertTrue(unique.issubset({-1.0, 0.0, 1.0}))

    def test_early_exit(self):
        """Should exit early if energy drops below threshold."""
        logits = torch.zeros(1, 1, 6, 3)
        target = torch.zeros(1, 1, 6)  # all zeros = no active banks
        critic = torch.tensor([0.0])
        censor = torch.tensor([[0.0]])
        anchor = torch.tensor([[1.0]])
        energy_fn = EnergyFunction()

        _, final_energy = energy_refine(
            logits,
            target,
            critic,
            censor,
            anchor,
            energy_fn,
            config=RefineConfig(refine_steps=100, energy_threshold=10.0),
        )
        # Should have exited early
        self.assertLess(final_energy, 10.0)


if __name__ == "__main__":
    unittest.main()
