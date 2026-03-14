"""
Energy function and continuous ternary refinement for Wayfinder v3B.

Defines a composite differentiable energy function over proof sketches:
  E(sketch) = alpha * E_bank + beta * E_critic + gamma * E_censor + delta * E_anchor

Includes Gumbel-softmax continuous ternary relaxation for gradient-based
sketch refinement, and snap-to-discrete for final output.

Gated behind energy_refinement.enabled config flag. Does not ship until
v3A demonstrates value on real proof outcomes.

See PLAN §7.5-7.6 and DESIGN §12.5 for specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefineConfig:
    """Hyperparameters for energy-based sketch refinement."""

    refine_steps: int = 20
    refine_lr: float = 0.01
    tau_start: float = 1.0
    tau_end: float = 0.1
    energy_threshold: float = 0.1
    censor_threshold: float = 0.5


def gumbel_softmax_ternary(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """Gumbel-softmax over 3 ternary categories {-1, 0, +1}.

    Args:
        logits: Raw logits [batch, num_steps, 6, 3] for 6 banks x 3 options.
        tau: Temperature. High (1.0) = soft/exploratory, low (0.1) = hard/discrete.
        hard: If True, use straight-through estimator (argmax forward, soft backward).

    Returns:
        Continuous ternary positions [batch, num_steps, 6] in range (-1, 1).
        At high tau, biased toward 0 (Informational Zero dominates as
        entropy-maximizing choice). At low tau, hardens toward {-1, 0, +1}.
    """
    # Gumbel-softmax over 3 categories
    soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [batch, steps, 6, 3]

    # Convert softmax weights over {-1, 0, +1} to continuous position
    ternary_values = torch.tensor([-1.0, 0.0, 1.0], device=logits.device)
    continuous = (soft * ternary_values).sum(dim=-1)  # [batch, steps, 6]

    return continuous


def snap_to_ternary(continuous: torch.Tensor) -> torch.Tensor:
    """Snap continuous positions to discrete {-1, 0, +1}.

    Uses nearest-value rounding:
      x < -0.5 → -1
      -0.5 <= x <= 0.5 → 0
      x > 0.5 → +1

    Args:
        continuous: Continuous positions [...] in range (-1, 1).

    Returns:
        Discrete ternary values [...] in {-1, 0, +1}.
    """
    result = torch.zeros_like(continuous)
    result[continuous < -0.5] = -1.0
    result[continuous > 0.5] = 1.0
    return result


class EnergyFunction(nn.Module):
    """Composite energy function over proof sketches.

    E(sketch) = alpha * E_bank + beta * E_critic + gamma * E_censor + delta * E_anchor

    All components are differentiable. Low energy = well-satisfied constraints.

    Args:
        alpha: Bank alignment weight (default 1.0).
        beta: Critic distance weight (default 0.5).
        gamma: Censor violation weight (default 2.0, MCA-motivated).
        delta: Anchor alignment weight (default 0.3).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 2.0,
        delta: float = 0.3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def bank_energy(
        self,
        sketch_positions: torch.Tensor,
        target_directions: torch.Tensor,
    ) -> torch.Tensor:
        """Bank constraint energy: how well does the sketch satisfy alignment?

        E_bank = mean(1 - cosine_similarity(sketch_pos, target_dir))

        Informational Zero: banks where target = 0 contribute zero energy
        (transparent banks don't push the gradient).

        Args:
            sketch_positions: Continuous positions [batch, steps, 6].
            target_directions: Target directions [batch, steps, 6].

        Returns:
            Scalar energy.
        """
        # Mask out zero-target banks (Informational Zero)
        active_mask = target_directions.abs() > 0.1
        if not active_mask.any():
            return torch.tensor(0.0, device=sketch_positions.device)

        # Cosine similarity on active banks only
        sketch_active = sketch_positions * active_mask.float()
        target_active = target_directions * active_mask.float()

        cos_sim = F.cosine_similarity(
            sketch_active.reshape(-1, sketch_active.shape[-1]),
            target_active.reshape(-1, target_active.shape[-1]),
            dim=-1,
        )
        return (1.0 - cos_sim).mean()

    def critic_energy(self, critic_distances: torch.Tensor) -> torch.Tensor:
        """Critic distance energy: how far from proof closure?

        Args:
            critic_distances: Estimated remaining steps [batch] or scalar.

        Returns:
            Scalar energy (mean critic distance).
        """
        return critic_distances.mean()

    def censor_energy(
        self,
        censor_scores: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Censor violation energy: does the sketch contain forbidden actions?

        E_censor = mean(max(0, censor_score - threshold))

        Only penalizes actions that exceed the censor threshold.

        Args:
            censor_scores: P(failure) per step [batch, steps].
            threshold: Censor operating threshold.

        Returns:
            Scalar energy.
        """
        violations = F.relu(censor_scores - threshold)
        return violations.mean()

    def anchor_energy(self, anchor_alignments: torch.Tensor) -> torch.Tensor:
        """Anchor misalignment energy.

        E_anchor = 1 - mean(alignment)

        Args:
            anchor_alignments: IDF-weighted Jaccard per step [batch, steps].

        Returns:
            Scalar energy.
        """
        return 1.0 - anchor_alignments.mean()

    def forward(
        self,
        sketch_positions: torch.Tensor,
        target_directions: torch.Tensor,
        critic_distances: torch.Tensor,
        censor_scores: torch.Tensor,
        anchor_alignments: torch.Tensor,
        censor_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Compute composite energy.

        Args:
            sketch_positions: [batch, steps, 6] continuous bank positions.
            target_directions: [batch, steps, 6] target directions.
            critic_distances: [batch] estimated remaining steps.
            censor_scores: [batch, steps] P(failure) per step.
            anchor_alignments: [batch, steps] IDF-weighted alignment.
            censor_threshold: Censor operating threshold.

        Returns:
            Scalar composite energy.
        """
        e_bank = self.bank_energy(sketch_positions, target_directions)
        e_critic = self.critic_energy(critic_distances)
        e_censor = self.censor_energy(censor_scores, censor_threshold)
        e_anchor = self.anchor_energy(anchor_alignments)

        return (
            self.alpha * e_bank
            + self.beta * e_critic
            + self.gamma * e_censor
            + self.delta * e_anchor
        )


def energy_refine(
    sketch_logits: torch.Tensor,
    target_directions: torch.Tensor,
    critic_distances: torch.Tensor,
    censor_scores: torch.Tensor,
    anchor_alignments: torch.Tensor,
    energy_fn: EnergyFunction,
    config: RefineConfig | None = None,
) -> tuple[torch.Tensor, float]:
    """Refine a proof sketch via energy minimization in continuous space.

    Gradient-descends on the energy function with annealing Gumbel temperature.
    Returns discrete ternary sketch after snap.

    Args:
        sketch_logits: Initial logits [batch, steps, 6, 3].
        target_directions: Target directions [batch, steps, 6].
        critic_distances: [batch].
        censor_scores: [batch, steps].
        anchor_alignments: [batch, steps].
        energy_fn: Composite energy function.
        config: Refinement hyperparameters (defaults to RefineConfig()).

    Returns:
        (discrete_sketch [batch, steps, 6], final_energy float).
    """
    if config is None:
        config = RefineConfig()

    logits = sketch_logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=config.refine_lr)

    final_energy = float("inf")

    for step in range(config.refine_steps):
        optimizer.zero_grad()

        # Anneal temperature
        tau = config.tau_start * (config.tau_end / config.tau_start) ** (
            step / max(config.refine_steps - 1, 1)
        )

        # Continuous relaxation
        continuous = gumbel_softmax_ternary(logits, tau=tau)

        # Compute energy
        energy = energy_fn(
            continuous,
            target_directions,
            critic_distances,
            censor_scores,
            anchor_alignments,
            config.censor_threshold,
        )

        final_energy = energy.item()

        # Early exit
        if final_energy < config.energy_threshold:
            break

        energy.backward()
        optimizer.step()

    # Final forward pass at lowest temperature, then snap
    with torch.no_grad():
        continuous = gumbel_softmax_ternary(logits, tau=config.tau_end)
        discrete = snap_to_ternary(continuous)

    return discrete, final_energy
