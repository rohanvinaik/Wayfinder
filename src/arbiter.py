"""
Society of Mind arbiter — orchestrator for Wayfinder v2.

Manages the five-slot SoM pipeline:
  PERCEPTION → RECOGNITION → PLANNING → EXECUTION → VERIFICATION

The Arbiter runs proof search at the SoM level:
1. Receives initial goal from PERCEPTION
2. Routes to RECOGNITION for template classification
3. Routes to PLANNING for proof sketch generation
4. For each subgoal: routes to EXECUTION specialist
5. Routes tactic candidates to VERIFICATION
6. On failure: re-routes with retry flag for alternative template

See DESIGN §10.6 for specification.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import torch

from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.lean_interface import LeanKernel
from src.nav_contracts import NavOutput, TacticResult
from src.proof_search import SearchConfig, SearchResult
from src.resolution import Candidate, SearchContext, resolve
from src.sketch_predictor import SketchPredictor
from src.som_contracts import PlanningOutput, RecognitionOutput, SubgoalSpec
from src.specialist_navigator import ExecutionSlot
from src.template_classifier import TemplateClassifier


@dataclass
class SoMSlots:
    """Bundle of all SoM slot components."""

    encoder: GoalEncoder  # Slot 1: PERCEPTION
    analyzer: GoalAnalyzer  # Slot 1: PERCEPTION (feature extraction)
    classifier: TemplateClassifier  # Slot 2: RECOGNITION
    sketch_predictor: SketchPredictor  # Slot 3: PLANNING
    execution: ExecutionSlot  # Slot 4: EXECUTION
    lean: LeanKernel  # Slot 5: VERIFICATION


@dataclass
class _SoMSearchState:
    """Mutable state for a SoM proof search run."""

    open_goals: list[str]
    closed_goals: list[str] = field(default_factory=list)
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    templates_tried: dict[str, list[int]] = field(
        default_factory=dict
    )  # goal -> tried template IDs
    # Cache: goal text → analyzer features (cleared on goal set mutation)
    _feature_cache: dict[str, torch.Tensor] = field(default_factory=dict)


def _som_search_step(
    state: _SoMSearchState,
    env: _SoMSearchEnv,
    max_template_retries: int,
) -> None:
    """Execute one iteration of the SoM search loop.

    Selects a goal, runs perception → recognition → planning → execution.
    On template exhaustion, rotates the goal. On sketch failure, increments
    attempts to trigger retry with a different template next iteration.
    """
    slots = env.slots
    goal_idx = _select_goal(state.open_goals, slots, state)
    goal = state.open_goals[goal_idx]

    # Slot 1: PERCEPTION
    features = _cached_encode_goal(goal, slots, state)

    # Slot 2: RECOGNITION (with retry support)
    tried = state.templates_tried.get(goal, [])
    if len(tried) >= max_template_retries:
        recognition = None
    else:
        recognition = _recognize_with_retry(features, slots.classifier, tried)

    if recognition is None:
        # All templates exhausted — rotate goal
        if len(state.open_goals) > 1:
            state.open_goals.append(state.open_goals.pop(goal_idx))
        state.attempts += 1
        return

    state.templates_tried.setdefault(goal, []).append(recognition.template_id)

    # Slot 3: PLANNING
    embedding = slots.encoder.encode([goal])
    plan = slots.sketch_predictor.predict(embedding, recognition)

    # Slot 4 + 5: EXECUTION + VERIFICATION
    if not _execute_sketch(goal, goal_idx, plan, state, env):
        state.attempts += 1


@dataclass
class SoMSearchParams:
    """Optional parameters for som_search, grouped to reduce argument count."""

    config: SearchConfig | None = None
    anchor_id_map: dict[str, int] | None = None
    accessible_theorem_id: int | None = None
    max_template_retries: int = 3


def som_search(
    theorem_id: str,
    initial_goal: str,
    slots: SoMSlots,
    conn: sqlite3.Connection,
    params: SoMSearchParams | None = None,
) -> SearchResult:
    """Run SoM proof search on a single theorem.

    The Arbiter manages the five-slot pipeline:
      PERCEPTION → RECOGNITION → PLANNING → EXECUTION → VERIFICATION
    with template retry on sketch failure (up to max_template_retries).

    Args:
        theorem_id: Identifier for the theorem being proved.
        initial_goal: Initial goal state text.
        slots: SoM slot components.
        conn: SQLite connection to proof network.
        params: Optional search parameters.

    Returns:
        SearchResult with proof attempt details.
    """
    p = params or SoMSearchParams()
    cfg = p.config or SearchConfig()
    context = SearchContext(
        accessible_theorem_id=p.accessible_theorem_id if cfg.accessible_premises else None,
    )
    state = _SoMSearchState(open_goals=[initial_goal])
    env = _SoMSearchEnv(
        slots=slots,
        conn=conn,
        context=context,
        anchor_id_map=p.anchor_id_map,
        config=cfg,
    )

    while state.open_goals and state.attempts < cfg.budget:
        _som_search_step(state, env, p.max_template_retries)

    return SearchResult(
        success=len(state.open_goals) == 0,
        theorem_id=theorem_id,
        tactics_used=state.tactics_used,
        attempts=state.attempts,
        goals_closed=len(state.closed_goals),
        goals_remaining=len(state.open_goals),
    )


def _select_goal(
    open_goals: list[str],
    slots: SoMSlots,
    state: _SoMSearchState | None = None,
) -> int:
    """Select the most promising open goal using critic scores.

    When state is provided, populates the feature cache so subsequent
    _cached_encode_goal calls are free for the selected goal.
    """
    if len(open_goals) <= 1:
        return 0

    best_idx = 0
    best_score = -1.0
    for i, goal in enumerate(open_goals):
        if state is not None:
            features = _cached_encode_goal(goal, slots, state)
        else:
            features = _encode_goal(goal, slots)
        nav_output = slots.execution.predict(features)
        if nav_output.critic_score > best_score:
            best_score = nav_output.critic_score
            best_idx = i

    return best_idx


def _recognize_with_retry(
    features: torch.Tensor,
    classifier: TemplateClassifier,
    tried_template_ids: list[int],
) -> RecognitionOutput | None:
    """Get template classification, skipping already-tried templates.

    Returns None if all templates in top-k have been tried.
    """
    recognition = classifier.predict(features)

    # Find best untried template from top-k
    for template_id, template_name, confidence in recognition.top_k_templates:
        if template_id not in tried_template_ids:
            return RecognitionOutput(
                template_id=template_id,
                template_name=template_name,
                template_confidence=confidence,
                template_features=recognition.template_features,
                top_k_templates=recognition.top_k_templates,
            )

    return None


@dataclass
class _SoMSearchEnv:
    """Immutable environment for SoM sketch execution."""

    slots: SoMSlots
    conn: sqlite3.Connection
    context: SearchContext
    anchor_id_map: dict[str, int] | None
    config: SearchConfig


def _execute_sketch(
    goal: str,
    goal_idx: int,
    plan: PlanningOutput,
    state: _SoMSearchState,
    env: _SoMSearchEnv,
) -> bool:
    """Execute a proof sketch by trying each subgoal.

    Returns True if the current goal was closed.
    """
    for subgoal_spec in plan.sketch:
        if state.attempts >= env.config.budget:
            return False
        result = _execute_subgoal(goal, goal_idx, subgoal_spec, state, env)
        if result:
            return True
    return False


def _execute_subgoal(
    goal: str,
    goal_idx: int,
    subgoal_spec: SubgoalSpec,
    state: _SoMSearchState,
    env: _SoMSearchEnv,
) -> bool:
    """Execute one subgoal step: encode → predict → hammer or candidates.

    Returns True if the goal was closed.
    """
    current_features = _cached_encode_goal(goal, env.slots, state)
    nav_output = env.slots.execution.predict(current_features)

    if env.config.hammer_delegation and _should_hammer(nav_output, subgoal_spec):
        return _try_hammer(goal, goal_idx, nav_output, state, env)

    return _try_candidates(goal, goal_idx, nav_output, state, env)


def _encode_goal(goal: str, slots: SoMSlots) -> torch.Tensor:
    """Encode a goal state to analyzer features."""
    embedding = slots.encoder.encode([goal])
    features, _, _ = slots.analyzer(embedding)
    return features


def _cached_encode_goal(
    goal: str,
    slots: SoMSlots,
    state: _SoMSearchState,
) -> torch.Tensor:
    """Cached goal encoding — avoids redundant encoder + analyzer forward passes."""
    cached = state._feature_cache.get(goal)
    if cached is not None:
        return cached
    features = _encode_goal(goal, slots)
    state._feature_cache[goal] = features
    return features


def _close_goal(
    goal: str,
    goal_idx: int,
    result: TacticResult,
    state: _SoMSearchState,
) -> None:
    """Close a goal and update search state. Invalidates feature cache."""
    state.open_goals.pop(goal_idx)
    state.closed_goals.append(goal)
    state.tactics_used.append(result.tactic)
    state.open_goals.extend(result.new_goals)
    state._feature_cache.clear()


def _try_hammer(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SoMSearchState,
    env: _SoMSearchEnv,
) -> bool:
    """Attempt hammer delegation. Returns True if goal was closed."""
    candidates = resolve(nav_output, env.conn, env.context, env.anchor_id_map, premise_limit=16)
    premise_names = candidates[0].premises[:16] if candidates else []
    result = env.slots.lean.try_hammer(goal, premise_names)
    state.attempts += 1
    if result.success:
        _close_goal(goal, goal_idx, result, state)
        return True
    return False


def _try_candidates(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SoMSearchState,
    env: _SoMSearchEnv,
) -> bool:
    """Try navigational candidates. Returns True if goal was closed."""
    candidates = resolve(
        nav_output,
        env.conn,
        env.context,
        env.anchor_id_map,
        tactic_limit=env.config.max_candidates_per_step,
    )
    for candidate in candidates:
        if state.attempts >= env.config.budget:
            return False
        tactic_text = _build_tactic_text(candidate)
        result = env.slots.lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, result, state)
            return True
    return False


def _should_hammer(nav_output: NavOutput, subgoal_spec: SubgoalSpec) -> bool:
    """Check if hammer delegation is appropriate for this subgoal."""
    auto_dir = nav_output.directions.get("automation", 0)
    hint_auto = subgoal_spec.bank_hints.get("automation", 0)
    return auto_dir == -1 or hint_auto == -1


def _build_tactic_text(candidate: Candidate) -> str:
    """Build a tactic application string from a candidate."""
    if not candidate.premises:
        return candidate.tactic_name
    premises_str = " ".join(candidate.premises[:4])
    return f"{candidate.tactic_name} {premises_str}"
