"""
Proof search — outer loop managing goal selection, neural inference, and verification.

Coordinates the full pipeline: goal state → encoder → analyzer → bridge →
navigator → resolution → Lean kernel verification. Manages open goals,
proof context, search budget, and hammer delegation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.bridge import InformationBridge
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.lean_interface import LeanKernel
from src.nav_contracts import NavOutput
from src.proof_navigator import ProofNavigator
from src.resolution import Candidate, SearchContext, resolve


@dataclass
class SearchConfig:
    """Configuration for proof search."""

    budget: int = 600
    hammer_delegation: bool = True
    accessible_premises: bool = True
    max_candidates_per_step: int = 8
    device: str = "cpu"


@dataclass
class SearchResult:
    """Result of a proof search attempt."""

    success: bool
    theorem_id: str
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    goals_closed: int = 0
    goals_remaining: int = 0


@dataclass
class Pipeline:
    """Bundles the neural pipeline components for proof search."""

    encoder: GoalEncoder
    analyzer: GoalAnalyzer
    bridge: InformationBridge
    navigator: ProofNavigator


@dataclass
class _SearchEnv:
    """Infrastructure shared across search steps (immutable during a search run)."""

    conn: sqlite3.Connection
    lean: LeanKernel
    anchor_id_map: dict[str, int] | None
    max_candidates: int


@dataclass
class _SearchState:
    """Mutable state for a single proof search run."""

    open_goals: list[str]
    closed_goals: list[str] = field(default_factory=list)
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    # Cache: goal_state text → NavOutput (cleared on goal set mutation)
    _infer_cache: dict[str, NavOutput] = field(default_factory=dict)


def _close_goal(
    goal: str,
    goal_idx: int,
    tactic: str,
    new_goals: list[str],
    state: _SearchState,
) -> None:
    """Close a goal and update search state. Invalidates inference cache."""
    state.open_goals.pop(goal_idx)
    state.closed_goals.append(goal)
    state.tactics_used.append(tactic)
    state.open_goals.extend(new_goals)
    state._infer_cache.clear()


def _cached_infer(
    goal_state: str,
    pipeline: Pipeline,
    state: _SearchState,
) -> NavOutput:
    """Cached neural inference — avoids redundant forward passes for the same goal."""
    cached = state._infer_cache.get(goal_state)
    if cached is not None:
        return cached
    result = _infer(goal_state, pipeline)
    state._infer_cache[goal_state] = result
    return result


def _try_hammer(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
) -> bool:
    """Attempt hammer delegation. Returns True if goal was closed."""
    candidates = resolve(nav_output, env.conn, context, env.anchor_id_map, premise_limit=16)
    premise_names = candidates[0].premises[:16] if candidates else []
    result = env.lean.try_hammer(goal, premise_names)
    state.attempts += 1
    if result.success:
        _close_goal(goal, goal_idx, result.tactic, result.new_goals, state)
        context.seed_entity_ids.clear()
        return True
    return False


def _try_candidates(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
) -> bool:
    """Try navigational candidates. Returns True if goal was closed."""
    candidates = resolve(nav_output, env.conn, context, env.anchor_id_map)
    for candidate in candidates[: env.max_candidates]:
        tactic_text = _build_tactic_text(candidate)
        result = env.lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic_text, result.new_goals, state)
            return True
    return False


def _search_step(
    pipeline: Pipeline,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    cfg: SearchConfig,
) -> None:
    """Execute one iteration of the search loop.

    Selects a goal, runs inference, tries hammer then candidates.
    On failure, rotates the goal to the back of the queue.
    """
    goal, goal_idx = _select_goal(state.open_goals, pipeline, cfg.device, state)
    nav_output = _cached_infer(goal, pipeline, state)

    if cfg.hammer_delegation and _should_hammer(nav_output):
        if _try_hammer(goal, goal_idx, nav_output, state, env, context):
            return

    if _try_candidates(goal, goal_idx, nav_output, state, env, context):
        return

    # All candidates failed — rotate goal to back of queue
    if len(state.open_goals) > 1:
        state.open_goals.append(state.open_goals.pop(goal_idx))
    state.attempts += 1


def search(
    theorem_id: str,
    initial_goal: str,
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    config: SearchConfig | None = None,
    anchor_id_map: dict[str, int] | None = None,
    accessible_theorem_id: int | None = None,
) -> SearchResult:
    """Run proof search on a single theorem."""
    cfg = config or SearchConfig()
    context = SearchContext(
        accessible_theorem_id=accessible_theorem_id if cfg.accessible_premises else None,
    )
    state = _SearchState(open_goals=[initial_goal])
    env = _SearchEnv(
        conn=conn,
        lean=lean,
        anchor_id_map=anchor_id_map,
        max_candidates=cfg.max_candidates_per_step,
    )

    while state.open_goals and state.attempts < cfg.budget:
        _search_step(pipeline, state, env, context, cfg)

    return SearchResult(
        success=len(state.open_goals) == 0,
        theorem_id=theorem_id,
        tactics_used=state.tactics_used,
        attempts=state.attempts,
        goals_closed=len(state.closed_goals),
        goals_remaining=len(state.open_goals),
    )


def _infer(
    goal_state: str,
    pipeline: Pipeline,
) -> NavOutput:
    """Single neural inference: goal → NavOutput."""
    embeddings = pipeline.encoder.encode([goal_state])
    features, _, _ = pipeline.analyzer(embeddings)
    bridge_out = pipeline.bridge(features)
    return pipeline.navigator.predict(bridge_out)


def _select_goal(
    open_goals: list[str],
    pipeline: Pipeline,
    _device: str,
    state: _SearchState | None = None,
) -> tuple[str, int]:
    """Select the most promising open goal using critic scores.

    When state is provided, uses the inference cache to avoid redundant
    forward passes — the selected goal's NavOutput is already warm for
    the subsequent _cached_infer call in _search_step.
    """
    if len(open_goals) == 1:
        return open_goals[0], 0

    best_idx = 0
    best_score = -1.0
    for i, goal in enumerate(open_goals):
        if state is not None:
            nav = _cached_infer(goal, pipeline, state)
        else:
            nav = _infer(goal, pipeline)
        if nav.critic_score > best_score:
            best_score = nav.critic_score
            best_idx = i

    return open_goals[best_idx], best_idx


def _should_hammer(nav_output: NavOutput) -> bool:
    """Check if the navigator suggests automation (hammer delegation)."""
    auto_dir = nav_output.directions.get("automation", 0)
    return auto_dir == -1


def _build_tactic_text(candidate: Candidate) -> str:
    """Build a tactic application string from a candidate."""
    if not candidate.premises:
        return candidate.tactic_name
    premises_str = " ".join(candidate.premises[:4])
    return f"{candidate.tactic_name} {premises_str}"
