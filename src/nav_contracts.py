"""
Wayfinder navigational data contracts.

Defines the data types for the navigational proof search pipeline:
training examples, structured queries, scored entities, navigator output,
and tactic results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The 6 navigable banks and their canonical order.
BANK_NAMES: list[str] = [
    "structure",
    "domain",
    "depth",
    "automation",
    "context",
    "decomposition",
]


@dataclass
class _NavExampleCore:
    """Core proof-step identification and position."""

    goal_state: str = ""
    theorem_id: str = ""
    theorem_key: str = ""
    step_index: int = 0
    total_steps: int = 1
    nav_directions: dict[str, int] = field(default_factory=dict)
    remaining_steps: int = 0
    solvable: bool = True


@dataclass
class NavigationalExample(_NavExampleCore):
    """A single navigational training/eval example (data contract — flat for JSONL interchange).

    Maps a proof step to 6-bank direction labels, anchor targets,
    and progress/solvability information for the critic heads.
    """

    anchor_labels: list[str] = field(default_factory=list)
    ground_truth_tactic: str = ""
    ground_truth_premises: list[str] = field(default_factory=list)
    proof_history: list[str] = field(default_factory=list)
    bank_positions: dict[str, list[int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "goal_state": self.goal_state,
            "theorem_id": self.theorem_id,
            "theorem_key": self.theorem_key or self.theorem_id,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "nav_directions": self.nav_directions,
            "anchor_labels": self.anchor_labels,
            "ground_truth_tactic": self.ground_truth_tactic,
            "ground_truth_premises": self.ground_truth_premises,
            "remaining_steps": self.remaining_steps,
            "solvable": self.solvable,
            "proof_history": self.proof_history,
        }
        if self.bank_positions is not None:
            d["bank_positions"] = self.bank_positions
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NavigationalExample:
        return cls(
            goal_state=d["goal_state"],
            theorem_id=d["theorem_id"],
            theorem_key=d.get("theorem_key", d["theorem_id"]),
            step_index=d.get("step_index", 0),
            total_steps=d.get("total_steps", 1),
            nav_directions=d.get("nav_directions", {}),
            anchor_labels=d.get("anchor_labels", []),
            ground_truth_tactic=d.get("ground_truth_tactic", ""),
            ground_truth_premises=d.get("ground_truth_premises", []),
            remaining_steps=d.get("remaining_steps", 0),
            solvable=d.get("solvable", True),
            proof_history=d.get("proof_history", []),
            bank_positions=d.get("bank_positions"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class StructuredQuery:
    """A navigational query for the proof network.

    Built from ProofNavigator output — bank directions + anchor preferences
    are used by navigate() to find matching entities.
    """

    bank_directions: dict[str, int]  # bank_name -> {-1, 0, +1}
    bank_confidences: dict[str, float]  # bank_name -> [0, 1]
    require_anchors: list[int] = field(default_factory=list)
    prefer_anchors: list[int] = field(default_factory=list)
    prefer_weights: list[float] = field(default_factory=list)
    avoid_anchors: list[int] = field(default_factory=list)
    seed_entity_ids: list[int] = field(default_factory=list)
    accessible_theorem_id: int | None = None


@dataclass
class ScoredEntity:
    """A proof network entity with its retrieval score breakdown."""

    entity_id: int
    name: str
    final_score: float
    bank_score: float
    anchor_score: float
    seed_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "final_score": self.final_score,
            "bank_score": self.bank_score,
            "anchor_score": self.anchor_score,
            "seed_score": self.seed_score,
        }


@dataclass
class NavOutput:
    """Output of the ProofNavigator neural network.

    Contains navigational directions, anchor predictions, and critic/progress
    estimates. Used to build a StructuredQuery for proof network resolution.
    """

    directions: dict[str, int]  # bank_name -> {-1, 0, +1} (after argmax)
    direction_confidences: dict[str, float]  # bank_name -> softmax max prob
    anchor_scores: dict[str, float]  # anchor_label -> sigmoid score
    progress: float  # estimated remaining steps
    critic_score: float  # estimated solvability (soft, not binary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directions": self.directions,
            "direction_confidences": self.direction_confidences,
            "anchor_scores": self.anchor_scores,
            "progress": self.progress,
            "critic_score": self.critic_score,
        }


@dataclass
class LeanFeedback:
    """Structured compiler/elaborator feedback from a Lean tactic attempt.

    Preserves the Pantograph message objects so callers can route repair
    actions based on the category of failure rather than raw error strings.

    Categories
    ----------
    parse_error        -- TacticFailure raised with ``parseError`` key.
                          Bad tactic template / lowering syntax.
    unknown_identifier -- Message data contains "unknown identifier" or
                          "unknown constant".  Name not in scope.
    unification_mismatch -- "type mismatch", "failed to synthesize",
                          "application type mismatch".
                          Candidate theorem does not fit the goal.
    typeclass_missing  -- "failed to synthesize" alone (no type mismatch).
                          Elaboration support missing; different from wrong lemma.
    generated_sorry    -- hasSorry flag: tactic leaves a sorry hole.
    generated_unsafe   -- hasUnsafe flag.
    goal_creation_fail -- Lean could not create the initial goal state.
    prefix_replay_fail -- A prefix tactic failed during Tier C replay.
    goal_match_fail    -- Replay reached goals, but none matched the expected goal.
    accepted_with_goals -- tactic succeeded but left open subgoals
                          (not a failure, stored for completeness).
    other              -- Unclassified error.
    none               -- No feedback (success with all goals closed).
    """

    stage: str       # goal_creation | tactic_parse | elaboration | tactic_exec
    category: str    # see docstring
    messages: list[dict]   # serialised Message objects (severity, kind, data, pos)
    raw_error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "category": self.category,
            "messages": self.messages,
            "raw_error": self.raw_error,
        }

    @staticmethod
    def success() -> "LeanFeedback":
        return LeanFeedback(
            stage="tactic_exec", category="none", messages=[], raw_error=""
        )


@dataclass
class TacticResult:
    """Result of applying a tactic to a goal state via the Lean kernel."""

    success: bool
    tactic: str
    premises: list[str]
    new_goals: list[str] = field(default_factory=list)
    error_message: str = ""
    feedback: LeanFeedback | None = None  # structured elaboration feedback

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "success": self.success,
            "tactic": self.tactic,
            "premises": self.premises,
            "new_goals": self.new_goals,
            "error_message": self.error_message,
        }
        if self.feedback is not None:
            d["feedback"] = self.feedback.to_dict()
        return d
