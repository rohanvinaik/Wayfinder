"""
v3 runtime data contracts for Wayfinder.

Defines typed interfaces for the boundary learning + energy refinement pipeline:
GoalContext, ActionCandidate, NegativeExample, ConstraintReport, SketchProposal,
SearchTrace. All v3 data flows through these contracts.

See DESIGN §12.2 for specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoalContext:
    """Full context for a proof goal under search."""

    theorem_id: str
    goal_text: str
    proof_history: list[str] = field(default_factory=list)
    accessible_premises: list[str] = field(default_factory=list)
    source_split: str = ""  # "train" | "eval"


@dataclass
class ActionCandidate:
    """A tactic-premise candidate with provenance and scoring."""

    tactic: str
    premises: list[str] = field(default_factory=list)
    provenance: str = "navigate"  # "navigate" | "spread" | "hammer"
    navigational_scores: dict[str, float] = field(default_factory=dict)
    template_provenance: str | None = None
    censor_score: float | None = None  # P(failure), set after censor scoring

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tactic": self.tactic,
            "premises": self.premises,
            "provenance": self.provenance,
            "navigational_scores": self.navigational_scores,
        }
        if self.template_provenance is not None:
            d["template_provenance"] = self.template_provenance
        if self.censor_score is not None:
            d["censor_score"] = self.censor_score
        return d


@dataclass
class NegativeExample:
    """A structured negative training example (failed tactic application).

    Schema for data/nav_negative.jsonl. Split hygiene: inherits
    train/eval split from nav_train/nav_eval by theorem_id.
    """

    goal_state: str
    theorem_id: str
    step_index: int = 0
    failed_tactic: str = ""
    failure_reason: str = ""
    failure_category: str = ""  # "semantic" | "infra" | "weak_negative"
    source: str = ""  # "sorry_hole" | "perturbation" | "suggestion_trace" | "unchosen_weak"
    proof_history: list[str] = field(default_factory=list)
    paired_positive_tactic: str | None = None
    paired_positive_premises: list[str] = field(default_factory=list)
    bank_directions: dict[str, int] = field(default_factory=dict)
    otp_dimensionality: int = 0  # 6 - count(zeros in bank_directions)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "goal_state": self.goal_state,
            "theorem_id": self.theorem_id,
            "step_index": self.step_index,
            "failed_tactic": self.failed_tactic,
            "failure_reason": self.failure_reason,
            "failure_category": self.failure_category,
            "source": self.source,
            "proof_history": self.proof_history,
            "bank_directions": self.bank_directions,
            "otp_dimensionality": self.otp_dimensionality,
        }
        if self.paired_positive_tactic is not None:
            d["paired_positive_tactic"] = self.paired_positive_tactic
        if self.paired_positive_premises:
            d["paired_positive_premises"] = self.paired_positive_premises
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NegativeExample:
        return cls(
            goal_state=d["goal_state"],
            theorem_id=d["theorem_id"],
            step_index=d.get("step_index", 0),
            failed_tactic=d.get("failed_tactic", ""),
            failure_reason=d.get("failure_reason", ""),
            failure_category=d.get("failure_category", ""),
            source=d.get("source", ""),
            proof_history=d.get("proof_history", []),
            paired_positive_tactic=d.get("paired_positive_tactic"),
            paired_positive_premises=d.get("paired_positive_premises", []),
            bank_directions=d.get("bank_directions", {}),
            otp_dimensionality=d.get("otp_dimensionality", 0),
        )


@dataclass
class ConstraintReport:
    """Composite constraint evaluation for a candidate or sketch.

    Unifies bank alignment, critic distance, censor score, and anchor
    matching into a single report. v3A uses total_score; v3B adds energy.
    """

    bank_scores: dict[str, float] = field(default_factory=dict)
    critic_distance: float = 0.0
    censor_score: float = 0.0  # P(failure)
    anchor_alignment: float = 0.0  # IDF-weighted Jaccard
    total_score: float = 0.0  # v3A: weighted composite
    energy: float | None = None  # v3B: differentiable energy scalar

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bank_scores": self.bank_scores,
            "critic_distance": self.critic_distance,
            "censor_score": self.censor_score,
            "anchor_alignment": self.anchor_alignment,
            "total_score": self.total_score,
        }
        if self.energy is not None:
            d["energy"] = self.energy
        return d


@dataclass
class SketchProposal:
    """A proof sketch with constraint scoring."""

    template_id: str
    proposed_steps: list[ActionCandidate] = field(default_factory=list)
    total_constraint_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "proposed_steps": [s.to_dict() for s in self.proposed_steps],
            "total_constraint_score": self.total_constraint_score,
        }


@dataclass
class SearchTrace:
    """Complete audit trail for one theorem search attempt.

    Captures every decision the v3 runtime makes, including pruning
    decisions and constraint reports, for post-hoc analysis.
    """

    theorem_id: str
    mode: str = "v3"  # "v1" | "v2" | "v3"
    steps: list[dict[str, Any]] = field(default_factory=list)
    pruning_decisions: list[dict[str, Any]] = field(default_factory=list)
    lean_calls: int = 0
    result: str = "failed"  # "proved" | "failed" | "timeout"
    constraint_reports: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_id": self.theorem_id,
            "mode": self.mode,
            "steps": self.steps,
            "pruning_decisions": self.pruning_decisions,
            "lean_calls": self.lean_calls,
            "result": self.result,
            "constraint_reports": self.constraint_reports,
        }
