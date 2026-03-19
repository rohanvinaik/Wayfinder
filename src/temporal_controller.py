"""Temporal controller — online phase tracking and routing for proof search.

Sits between PLANNING and EXECUTION in the SoM pipeline. Predicts:
- next_goal_id: which subgoal to attempt
- phase: structural_setup / local_close / automation_close / repair_or_replan
- lane_order: priority ordering of search lanes
- family_prior: expected tactic family distribution
- escalation_level: solver sophistication
- replan: whether to abandon current template

The controller is STATEFUL — it conditions on prior progress, not just
the current goal in isolation. This is the Robot Chef / HTN insight:
proof search has temporal dependencies and phase transitions.

The controller does NOT decide truth. It routes and prioritizes.
Lean decides truth. The residual executor chooses local families.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Proof search phases (finite state, not continuous)
PHASES = [
    "structural_setup",  # intros, constructors, trivial shape normalization
    "local_close",  # exact / apply / refine / rw / simp family execution
    "automation_close",  # hammer / aesop / decide / omega
    "repair_or_replan",  # lane exhausted, residual stagnant, failures clustered
]


@dataclass
class TemporalState:
    """Full proof search state visible to the temporal controller."""

    theorem_id: str
    active_template: str | None = None

    # Goal tracking
    open_goals: list[str] = field(default_factory=list)
    closed_goals: list[str] = field(default_factory=list)
    current_goal_id: str | None = None

    # Prior progress
    prior_lanes: list[str] = field(default_factory=list)
    prior_families: list[str] = field(default_factory=list)
    successful_tactics: list[str] = field(default_factory=list)
    failed_tactics: list[str] = field(default_factory=list)

    # Search state
    phase: str = "structural_setup"
    escalation_level: int = 0
    budget_remaining: int = 600
    total_attempts: int = 0

    # Per-goal failure counts (for escalation)
    goal_attempt_counts: dict[str, int] = field(default_factory=dict)
    goal_lane_failures: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class OrchestrationDecision:
    """Output of the temporal controller — routing decision for one step."""

    next_goal_id: str
    phase: str
    lane_order: list[str]
    family_prior: list[str]
    escalation_level: int
    budget_slice: int
    replan: bool


class TemporalController:
    """Rule-based temporal controller (v0).

    Implements the 4-phase finite state machine with escalation.
    Will be replaced by a learned controller once training data
    is collected from real proof traces.
    """

    def __init__(
        self,
        escalation_thresholds: tuple[int, ...] = (3, 6, 10),
        structural_budget_frac: float = 0.1,
        local_budget_frac: float = 0.5,
        automation_budget_frac: float = 0.3,
    ) -> None:
        self.escalation_thresholds = escalation_thresholds
        self.structural_frac = structural_budget_frac
        self.local_frac = local_budget_frac
        self.auto_frac = automation_budget_frac

    def decide(self, state: TemporalState) -> OrchestrationDecision:
        """Produce a routing decision given the current proof state."""
        # Select goal (complexity curriculum: prefer goals with fewer prior failures)
        goal = self._select_goal(state)

        # Determine phase
        phase = self._determine_phase(state, goal)

        # Determine lane order based on phase
        lane_order = self._lane_order_for_phase(phase)

        # Family prior based on phase and prior progress
        family_prior = self._family_prior(state, phase)

        # Escalation
        escalation = self._escalation_level(state, goal)

        # Budget allocation
        budget_slice = self._budget_for_phase(state, phase)

        # Replan check
        replan = self._should_replan(state, goal)

        return OrchestrationDecision(
            next_goal_id=goal,
            phase=phase,
            lane_order=lane_order,
            family_prior=family_prior,
            escalation_level=escalation,
            budget_slice=budget_slice,
            replan=replan,
        )

    def update(
        self,
        state: TemporalState,
        goal: str,
        lane: str,
        family: str,
        tactic: str,
        success: bool,
    ) -> None:
        """Update temporal state after a tactic attempt."""
        state.total_attempts += 1
        state.budget_remaining -= 1
        state.goal_attempt_counts[goal] = state.goal_attempt_counts.get(goal, 0) + 1

        if success:
            state.successful_tactics.append(tactic)
            state.prior_lanes.append(lane)
            state.prior_families.append(family)
        else:
            state.failed_tactics.append(tactic)
            state.goal_lane_failures.setdefault(goal, set()).add(lane)

    def _select_goal(self, state: TemporalState) -> str:
        """Select next goal — prefer goals with fewer prior failures."""
        if not state.open_goals:
            return state.current_goal_id or ""

        # Sort by attempt count (ascending = easier/fresher first)
        scored = [(state.goal_attempt_counts.get(g, 0), g) for g in state.open_goals]
        scored.sort()
        return scored[0][1]

    def _determine_phase(self, state: TemporalState, goal: str) -> str:
        """Determine current phase based on goal state and history."""
        attempts = state.goal_attempt_counts.get(goal, 0)
        failed_lanes = state.goal_lane_failures.get(goal, set())

        # Phase 1: structural_setup (first few attempts)
        if attempts == 0:
            return "structural_setup"

        # Phase 4: repair_or_replan (too many failures)
        if attempts >= self.escalation_thresholds[-1]:
            return "repair_or_replan"

        # Phase 3: automation_close (structural + local exhausted)
        if "structural_core" in failed_lanes and "learned" in failed_lanes:
            return "automation_close"

        # Phase 2: local_close (structural done, try learned executor)
        if "structural_core" in failed_lanes or attempts >= self.escalation_thresholds[0]:
            return "local_close"

        return "structural_setup"

    def _lane_order_for_phase(self, phase: str) -> list[str]:
        """Return lane priority order for the current phase."""
        if phase == "structural_setup":
            return ["structural_core", "solver_bootstrap", "learned"]
        if phase == "local_close":
            return ["learned", "solver_bootstrap", "structural_core"]
        if phase == "automation_close":
            return ["automation", "solver_bootstrap", "learned"]
        # repair_or_replan
        return ["automation", "solver_bootstrap"]

    def _family_prior(self, state: TemporalState, phase: str) -> list[str]:
        """Return expected family distribution for the current phase."""
        if phase == "structural_setup":
            return ["intros", "constructor", "assumption"]
        if phase == "local_close":
            return ["rw", "exact", "simp", "apply", "refine"]
        if phase == "automation_close":
            return ["omega", "simp", "aesop", "decide"]
        return ["aesop", "simp"]

    def _escalation_level(self, state: TemporalState, goal: str) -> int:
        """Compute escalation level from attempt count."""
        attempts = state.goal_attempt_counts.get(goal, 0)
        level = 0
        for threshold in self.escalation_thresholds:
            if attempts >= threshold:
                level += 1
        return level

    def _budget_for_phase(self, state: TemporalState, phase: str) -> int:
        """Allocate budget slice for the current phase."""
        remaining = state.budget_remaining
        if phase == "structural_setup":
            return max(int(remaining * self.structural_frac), 5)
        if phase == "local_close":
            return max(int(remaining * self.local_frac), 10)
        if phase == "automation_close":
            return max(int(remaining * self.auto_frac), 5)
        return max(remaining // 4, 3)  # repair gets modest budget

    def _should_replan(self, state: TemporalState, goal: str) -> bool:
        """Check if replanning is warranted."""
        attempts = state.goal_attempt_counts.get(goal, 0)
        failed_lanes = state.goal_lane_failures.get(goal, set())

        # Replan if all lanes exhausted for this goal
        if len(failed_lanes) >= 3:
            return True

        # Replan if too many failures with no progress
        if attempts >= self.escalation_thresholds[-1] and not state.successful_tactics:
            return True

        return False
