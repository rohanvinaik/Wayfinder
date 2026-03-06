"""
Wayfinder data contracts.

All persisted artifacts (JSONL, JSON) serialize/deserialize through these
dataclasses. Field names are stable across the pipeline.

Includes both the legacy three-tier decomposition (Balanced Sashimi) and
the navigational contracts (Wayfinder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Three-tier output types
# ---------------------------------------------------------------------------


@dataclass
class Tier2Block:
    """A single tactic's premise/argument block (Tier 2 decoding output).

    Attributes:
        tactic_index: Position of this tactic in the proof.
        tactic_name: The tier-1 tactic (e.g., "apply", "rw").
        tokens: Premise/argument token strings (persisted form).
        token_ids: Integer vocab indices for training (optional, not persisted).
    """

    tactic_index: int
    tactic_name: str
    tokens: list[str]
    token_ids: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tactic_index": self.tactic_index,
            "tactic_name": self.tactic_name,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tier2Block:
        return cls(
            tactic_index=d["tactic_index"],
            tactic_name=d["tactic_name"],
            tokens=d["tokens"],
        )


@dataclass
class Tier3Slot:
    """A free-form term slot filled by the TermFiller (Tier 3 output).

    Attributes:
        slot_id: Unique identifier for this slot.
        value_kind: Type of value ("expr", "name", "type", "numeral").
        value: The actual term string.
        source_tactic: Which tactic this term belongs to.
    """

    slot_id: str
    value_kind: str
    value: str | int | float
    source_tactic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "slot_id": self.slot_id,
            "value_kind": self.value_kind,
            "value": self.value,
        }
        if self.source_tactic is not None:
            d["source_tactic"] = self.source_tactic
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tier3Slot:
        return cls(
            slot_id=d["slot_id"],
            value_kind=d["value_kind"],
            value=d["value"],
            source_tactic=d.get("source_tactic"),
        )


# ---------------------------------------------------------------------------
# Training data types
# ---------------------------------------------------------------------------


@dataclass
class ProofExample:
    """A single training/eval example with three-tier decomposition.

    This is the canonical persisted format for proof training data.

    Attributes:
        theorem_id: Unique identifier (e.g., "Mathlib.Algebra.Group.Basic.mul_comm").
        goal_state: The Lean goal state text (what needs to be proved).
        theorem_statement: Full theorem statement in Lean syntax.
        proof_text: The complete proof text in Lean tactic mode.
        tier1_tokens: Tactic sequence tokens.
        tier2_blocks: Per-tactic premise/argument blocks.
        tier3_slots: Free-form term slots.
        metadata: Additional info (source file, difficulty, domain, etc.)
    """

    theorem_id: str
    goal_state: str
    theorem_statement: str
    proof_text: str
    tier1_tokens: list[str]
    tier2_blocks: list[Tier2Block]
    tier3_slots: list[Tier3Slot]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_id": self.theorem_id,
            "goal_state": self.goal_state,
            "theorem_statement": self.theorem_statement,
            "proof_text": self.proof_text,
            "tier1_tokens": self.tier1_tokens,
            "tier2_blocks": [b.to_dict() for b in self.tier2_blocks],
            "tier3_slots": [s.to_dict() for s in self.tier3_slots],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProofExample:
        return cls(
            theorem_id=d["theorem_id"],
            goal_state=d.get("goal_state", ""),
            theorem_statement=d.get("theorem_statement", ""),
            proof_text=d.get("proof_text", ""),
            tier1_tokens=d["tier1_tokens"],
            tier2_blocks=[Tier2Block.from_dict(b) for b in d.get("tier2_blocks", [])],
            tier3_slots=[Tier3Slot.from_dict(s) for s in d.get("tier3_slots", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass
class NegativeBankEntry:
    """A contrastive pair for margin loss training.

    Negatives come from: failed proof attempts, wrong tactic selections,
    incorrect premise choices, and synthetic perturbations.
    """

    goal_state: str
    theorem_id: str
    positive: ProofExample
    negative: ProofExample | None
    error_tags: list[str]
    source: str  # "failed_proof", "wrong_tactic", "synthetic_mutation"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "goal_state": self.goal_state,
            "theorem_id": self.theorem_id,
            "positive": self.positive.to_dict(),
            "error_tags": self.error_tags,
            "source": self.source,
        }
        if self.negative is not None:
            d["negative"] = self.negative.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NegativeBankEntry:
        return cls(
            goal_state=d["goal_state"],
            theorem_id=d["theorem_id"],
            positive=ProofExample.from_dict(d["positive"]),
            negative=ProofExample.from_dict(d["negative"]) if d.get("negative") else None,
            error_tags=d.get("error_tags", []),
            source=d.get("source", "unknown"),
        )


# ---------------------------------------------------------------------------
# Domain gate types
# ---------------------------------------------------------------------------


@dataclass
class OODPrompt:
    """An out-of-domain prompt for domain gate training/evaluation."""

    prompt: str
    label: str  # "in_domain" or "ood"
    category: str  # "lean_goal", "general_chat", "code_generation", etc.
    source: str  # "leandojo", "synthetic", "manual"

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "label": self.label,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> OODPrompt:
        return cls(
            prompt=d["prompt"],
            label=d["label"],
            category=d.get("category", "unknown"),
            source=d.get("source", "unknown"),
        )


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of verifying a generated proof against the Lean kernel."""

    verified: bool
    goal_state: str
    tactic_trace: list[str]
    error_message: str = ""
    remaining_goals: list[str] = field(default_factory=list)
    steps_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "goal_state": self.goal_state,
            "tactic_trace": self.tactic_trace,
            "error_message": self.error_message,
            "remaining_goals": self.remaining_goals,
            "steps_used": self.steps_used,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerificationResult:
        return cls(
            verified=d["verified"],
            goal_state=d.get("goal_state", ""),
            tactic_trace=d.get("tactic_trace", []),
            error_message=d.get("error_message", ""),
            remaining_goals=d.get("remaining_goals", []),
            steps_used=d.get("steps_used", 0),
        )


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Vocab coverage metric for acceptance tests."""

    scope: str  # "tier1" or "tier2"
    dataset: str  # "eval"
    total_tokens_in_eval: int
    covered: int
    uncovered: list[str]
    coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dataset": self.dataset,
            "total_tokens_in_eval": self.total_tokens_in_eval,
            "covered": self.covered,
            "uncovered": self.uncovered,
            "coverage_pct": self.coverage_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CoverageReport:
        return cls(
            scope=d["scope"],
            dataset=d["dataset"],
            total_tokens_in_eval=d["total_tokens_in_eval"],
            covered=d["covered"],
            uncovered=sorted(d.get("uncovered", [])),
            coverage_pct=d["coverage_pct"],
        )