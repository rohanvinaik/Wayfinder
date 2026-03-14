"""
Society of Mind data contracts for Wayfinder v2.

Defines typed data contracts for communication between SoM slots:
PERCEPTION → RECOGNITION → PLANNING → EXECUTION → VERIFICATION.
Slots communicate through these contracts, never through shared weights
(Invariant #13: γ ≈ 0 by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PerceptionOutput:
    """Output of PERCEPTION slot (Slot 1).

    Deterministic encoding of goal state into a fixed embedding.
    σ ≈ O(1) — frozen encoder, no learning.
    """

    embedding: torch.Tensor  # [384]
    in_domain: bool = True


@dataclass
class TemplateInfo:
    """Metadata for a proof strategy template."""

    template_id: str
    pattern: str
    bank_signature: dict[str, int]  # bank_name -> dominant direction
    tactic_patterns: list[str] = field(default_factory=list)
    is_simple: bool = False  # True = deterministic sketch (no learning needed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "pattern": self.pattern,
            "bank_signature": self.bank_signature,
            "tactic_patterns": self.tactic_patterns,
            "is_simple": self.is_simple,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemplateInfo:
        return cls(
            template_id=d["template_id"],
            pattern=d["pattern"],
            bank_signature=d["bank_signature"],
            tactic_patterns=d.get("tactic_patterns", []),
            is_simple=d.get("is_simple", False),
        )


@dataclass
class RecognitionOutput:
    """Output of RECOGNITION slot (Slot 2).

    Template classification result with features for downstream planning.
    σ ≈ O(log k) — Regime A task (high symmetry).
    """

    template_id: int
    template_name: str
    template_confidence: float
    template_features: torch.Tensor  # [64]
    top_k_templates: list[tuple[int, str, float]] = field(default_factory=list)


@dataclass
class SubgoalSpec:
    """Specification for a single subgoal in a proof sketch."""

    subgoal_type: str
    anchor_targets: list[str] = field(default_factory=list)
    estimated_steps: int = 1
    bank_hints: dict[str, int] = field(default_factory=dict)  # bank_name -> {-1, 0, +1}

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_type": self.subgoal_type,
            "anchor_targets": self.anchor_targets,
            "estimated_steps": self.estimated_steps,
            "bank_hints": self.bank_hints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubgoalSpec:
        return cls(
            subgoal_type=d["subgoal_type"],
            anchor_targets=d.get("anchor_targets", []),
            estimated_steps=d.get("estimated_steps", 1),
            bank_hints=d.get("bank_hints", {}),
        )


@dataclass
class PlanningOutput:
    """Output of PLANNING slot (Slot 3).

    Proof sketch: ordered sequence of subgoals with estimated difficulty.
    σ ≈ O(poly(n)) — complex but bounded.
    """

    sketch: list[SubgoalSpec]
    total_estimated_depth: int = 0
    template_id: int = 0


@dataclass
class ExecutionOutput:
    """Output of EXECUTION slot (Slot 4).

    Fused navigational output from specialist navigators.
    σ varies by specialist (each should be Regime A after decomposition).
    """

    directions: dict[str, int]  # bank_name -> {-1, 0, +1}
    direction_confidences: dict[str, float]
    anchor_logits: torch.Tensor  # [num_anchors]
    progress: float
    critic: float


@dataclass
class VerificationOutput:
    """Output of VERIFICATION slot (Slot 5).

    Result of tactic application via the Lean kernel.
    σ ≈ O(1) — deterministic kernel check.
    """

    success: bool
    new_goals: list[str] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass
class CensorPrediction:
    """Output of the Censor network (VERIFICATION enhancement)."""

    should_prune: bool
    failure_probability: float
