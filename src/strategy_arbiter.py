"""Dominance-aware rule Arbiter for SoM orchestration.

Uses strategy_memory.json to produce lane orderings that respect
empirical lane dominance patterns. Implements the SoM Arbiter role
as a typed rule system over symbolic packets.

Key principle: route to the cheapest adequate closer.
- If strategy memory says exact? closes 93% of equality goals after IB,
  skip intermediate lanes and go straight to the finisher.
- If a lane has never succeeded on this goal shape, suppress it.

EXP-SOM-002: Rule Arbiter vs static EXP-058.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.hard_data_tags import sanitize_goal_text
from src.temporal_controller import OrchestrationDecision, TemporalState


@dataclass
class StrategyEntry:
    """One entry from the strategy memory."""

    template_id: str
    namespace_prefix: str
    goal_shape_bucket: str
    recent_lanes: str
    preferred_lane_order: list[str]
    family_prior: list[str]
    support: int
    top_lane_rate: float


def _goal_shape_bucket(goal: str) -> str:
    """Cheap goal shape classifier matching mine_strategy_memory.py."""
    text = sanitize_goal_text((goal or "").strip())
    if "⊢" in text:
        text = text.split("⊢", 1)[1].strip()
    if not text:
        return "empty"
    if text.startswith("∀") and "→" in text:
        return "forall_implication"
    if text.startswith("∀"):
        return "forall"
    if text.startswith("∃") or text.startswith("Exists"):
        return "exists"
    if "↔" in text:
        return "iff"
    if "=" in text and "≤" not in text and "≥" not in text:
        return "equality"
    if "≤" in text or "≥" in text or "<" in text or ">" in text:
        return "inequality"
    if "∈" in text:
        return "membership"
    if "∧" in text or "∨" in text:
        return "connective"
    return "other"


def _domain_hints(theorem_id: str, goal: str) -> set[str]:
    source = " ".join([sanitize_goal_text(theorem_id or ""), sanitize_goal_text(goal or "")])
    hints: set[str] = set()
    checks = {
        "category_theory": ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"],
        "algebraic_geometry": ["AlgebraicGeometry", "Scheme", "LocallyRingedSpace", "PrimeSpectrum", "HomogeneousLocalization"],
        "abstract_algebra": ["IsIntegral", "traceMatrix", "Matrix.det", "discr", "FormallyUnramified", "HasRingHomProperty"],
        "cardinal": ["Cardinal", "#", "countable", "mk_", "aleph"],
        "geometric_analysis": ["Besicovitch", "dist", "Metric", "δ", "norm", "ball"],
        "witness_goal": ["sSup", "sInf", "iSup", "iInf", "range", "image", "∃", "Exists"],
        "structural_property": ["IsOpenMap", "HasRingHomProperty", "FormallyUnramified", "IsIso", "essImage"],
        "membership_wall": [".carrier", "Submodule", "Ideal", "PrimeSpectrum", "HomogeneousLocalization"],
    }
    for label, markers in checks.items():
        if label == "membership_wall":
            if "∈" in source and any(marker in source for marker in markers):
                hints.add(label)
            continue
        if any(marker in source for marker in markers):
            hints.add(label)
    return hints


def _recent_lanes_key(recent: list[str]) -> str:
    """Normalize recent lanes into a hashable key."""
    if not recent:
        return "(none)"
    deduped: list[str] = []
    for lane in recent[-3:]:
        if not deduped or deduped[-1] != lane:
            deduped.append(lane)
    return "+".join(deduped)


class StrategyArbiter:
    """Dominance-aware rule-based Arbiter.

    Modes:
        "full": goal ordering + lane-order control from strategy memory
        "goal_only": goal ordering only, static lane order
        "lane_only": static goal ordering, lane-order control from memory
        "off": disabled, returns None (fall back to static EXP-058 schedule)
    """

    # Default lane order for the current integrated first-order stack.
    # Keep this aligned with proof_search._DEFAULT_LANE_ORDER, but favor
    # interleaved_bootstrap over structural_core in arbiter mode because the
    # guarded benchmark enables the interleaved bootstrap path by default.
    _DEFAULT_LANES = [
        "automation",
        "interleaved_bootstrap",
        "structural_core",
        "cosine_rw",
        "cosine_exact",
        "cosine_apply",
        "learned",
        "dr_ducky",
    ]

    def __init__(
        self,
        strategy_memory_path: str = "",
        mode: str = "full",
    ) -> None:
        self.mode = mode
        self.memory: list[StrategyEntry] = []

        if strategy_memory_path and Path(strategy_memory_path).exists():
            with open(strategy_memory_path) as f:
                raw = json.load(f)
            for entry in raw:
                self.memory.append(
                    StrategyEntry(
                        template_id=entry["key"].get("template_id", ""),
                        namespace_prefix=entry["key"].get("namespace_prefix", ""),
                        goal_shape_bucket=entry["key"].get("goal_shape_bucket", ""),
                        recent_lanes=entry["key"].get("recent_lanes", ""),
                        preferred_lane_order=entry["value"].get("preferred_lane_order", []),
                        family_prior=entry["value"].get("family_prior", []),
                        support=entry["value"].get("support", 0),
                        top_lane_rate=entry["value"].get("top_lane_rate", 0),
                    )
                )

    def decide(self, state: TemporalState) -> OrchestrationDecision | None:
        """Produce a routing decision, or None to fall back to static schedule."""
        if self.mode == "off":
            return None

        goal = self._select_goal(state)
        lane_order = self._lane_order(state, goal)

        return OrchestrationDecision(
            next_goal_id=goal,
            phase=state.phase,
            lane_order=lane_order,
            family_prior=[],
            escalation_level=state.escalation_level,
            budget_slice=max(state.budget_remaining // max(len(state.open_goals), 1), 10),
            replan=False,
        )

    def _select_goal(self, state: TemporalState) -> str:
        """Goal selection: prefer goals with simpler shapes (more likely closeable)."""
        if self.mode == "lane_only":
            return state.open_goals[0] if state.open_goals else ""

        if not state.open_goals:
            return ""

        # Score each goal by estimated closability
        scored = []
        for goal in state.open_goals:
            shape = _goal_shape_bucket(goal)
            # Equality and forall goals are most closeable by finisher stack
            shape_score = {
                "equality": 5,
                "membership": 4,
                "forall": 4,
                "forall_implication": 3,
                "inequality": 3,
                "iff": 2,
                "connective": 2,
                "exists": 1,
                "other": 0,
                "empty": 0,
            }.get(shape, 0)

            # Shorter goals are often simpler
            length_score = max(0, 5 - len(goal) // 100)

            # Penalize goals that have been tried many times
            attempt_penalty = state.goal_attempt_counts.get(goal, 0)

            scored.append((goal, shape_score + length_score - attempt_penalty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _lane_order(self, state: TemporalState, goal: str) -> list[str]:
        """Lane ordering from strategy memory or default."""
        if self.mode == "goal_only":
            return self._complete_lane_order(self._DEFAULT_LANES)

        shape = _goal_shape_bucket(goal)
        recent_key = _recent_lanes_key(state.prior_lanes[-3:])
        domain_hints = _domain_hints(state.theorem_id, goal)

        # Look up strategy memory
        best_entry = None
        best_score = -1
        for entry in self.memory:
            score = 0
            if entry.goal_shape_bucket == shape:
                score += 3
            if entry.recent_lanes == recent_key:
                score += 2
            if score > best_score and entry.support >= 5:
                best_score = score
                best_entry = entry

        if best_entry and best_entry.top_lane_rate > 0.5:
            # Strategy memory has a strong preference — use it
            # Put the preferred lane first, then fill with defaults
            preferred = best_entry.preferred_lane_order
            lanes = self._complete_lane_order(preferred)
            return self._apply_domain_overrides(lanes, shape, domain_hints)

        # No strong memory match — use shape-based heuristics
        if shape == "equality":
            # Equalities often close with cosine_rw or exact?
            return self._apply_domain_overrides(
                [
                    "cosine_rw",
                    "automation",
                    "interleaved_bootstrap",
                    "cosine_exact",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                ],
                shape,
                domain_hints,
            )
        if shape in ("forall", "forall_implication"):
            # Need intros first
            return self._apply_domain_overrides(
                [
                    "interleaved_bootstrap",
                    "automation",
                    "cosine_rw",
                    "cosine_exact",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                ],
                shape,
                domain_hints,
            )
        if shape in ("exists", "connective"):
            # Structural tactics first
            return self._apply_domain_overrides(
                [
                    "interleaved_bootstrap",
                    "automation",
                    "cosine_exact",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                ],
                shape,
                domain_hints,
            )
        if shape == "inequality":
            return self._apply_domain_overrides(
                [
                    "automation",
                    "interleaved_bootstrap",
                    "cosine_rw",
                    "cosine_exact",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                ],
                shape,
                domain_hints,
            )
        if shape == "membership":
            return self._apply_domain_overrides(
                [
                    "cosine_exact",
                    "interleaved_bootstrap",
                    "cosine_rw",
                    "automation",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                ],
                shape,
                domain_hints,
            )
        return self._apply_domain_overrides(self._DEFAULT_LANES, shape, domain_hints)

    def _complete_lane_order(self, lanes: list[str]) -> list[str]:
        ordered = list(dict.fromkeys(lanes))
        for lane in self._DEFAULT_LANES:
            if lane not in ordered:
                ordered.append(lane)
        return ordered

    def _apply_domain_overrides(
        self,
        lanes: list[str],
        shape: str,
        domain_hints: set[str],
    ) -> list[str]:
        ordered = self._complete_lane_order(lanes)
        if {"category_theory", "algebraic_geometry", "abstract_algebra", "structural_property"} & domain_hints:
            if shape in {"forall", "forall_implication", "exists", "connective", "iff"}:
                promoted = [
                    "interleaved_bootstrap",
                    "cosine_exact",
                    "cosine_rw",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                    "automation",
                ]
            else:
                promoted = [
                    "cosine_exact",
                    "interleaved_bootstrap",
                    "cosine_rw",
                    "cosine_apply",
                    "learned",
                    "dr_ducky",
                    "automation",
                ]
            ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]
            if "automation" in ordered:
                ordered = [lane for lane in ordered if lane != "automation"] + ["automation"]
        if "witness_goal" in domain_hints and shape in {"inequality", "exists", "other"}:
            promoted = [
                "interleaved_bootstrap",
                "cosine_rw",
                "cosine_apply",
                "learned",
                "dr_ducky",
                "automation",
                "cosine_exact",
            ]
            ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]
        if "cardinal" in domain_hints and shape in {"connective", "other", "forall"}:
            promoted = [
                "interleaved_bootstrap",
                "cosine_exact",
                "cosine_rw",
                "cosine_apply",
                "learned",
                "dr_ducky",
                "automation",
            ]
            ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]
        if "membership_wall" in domain_hints:
            promoted = [
                "cosine_exact",
                "interleaved_bootstrap",
                "cosine_rw",
                "cosine_apply",
                "learned",
                "dr_ducky",
                "automation",
            ]
            ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]
        return ordered
