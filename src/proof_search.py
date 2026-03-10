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
class _SearchState:
    """Mutable state for a single proof search run."""

    open_goals: list[str]
    closed_goals: list[str] = field(default_factory=list)
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0


def _try_hammer(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    context: SearchContext,
    anchor_id_map: dict[str, int] | None,
) -> bool:
    """Attempt hammer delegation. Returns True if goal was closed."""
    candidates = resolve(nav_output, conn, context, anchor_id_map, premise_limit=16)
    premise_names = candidates[0].premises[:16] if candidates else []
    result = lean.try_hammer(goal, premise_names)
    state.attempts += 1
    if result.success:
        state.open_goals.pop(goal_idx)
        state.closed_goals.append(goal)
        state.tactics_used.append(result.tactic)
        state.open_goals.extend(result.new_goals)
        context.seed_entity_ids.clear()
        return True
    return False


def _try_candidates(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    context: SearchContext,
    anchor_id_map: dict[str, int] | None,
    max_candidates: int,
) -> bool:
    """Try navigational candidates. Returns True if goal was closed."""
    candidates = resolve(nav_output, conn, context, anchor_id_map)
    for candidate in candidates[:max_candidates]:
        tactic_text = _build_tactic_text(candidate)
        result = lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        if result.success:
            state.open_goals.pop(goal_idx)
            state.closed_goals.append(goal)
            state.tactics_used.append(tactic_text)
            state.open_goals.extend(result.new_goals)
            return True
    return False


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

    while state.open_goals and state.attempts < cfg.budget:
        goal, goal_idx = _select_goal(state.open_goals, pipeline, cfg.device)
        nav_output = _infer(goal, pipeline)

        if cfg.hammer_delegation and _should_hammer(nav_output):
            if _try_hammer(goal, goal_idx, nav_output, state, conn, lean, context, anchor_id_map):
                continue

        if not _try_candidates(
            goal,
            goal_idx,
            nav_output,
            state,
            conn,
            lean,
            context,
            anchor_id_map,
            cfg.max_candidates_per_step,
        ):
            if len(state.open_goals) > 1:
                state.open_goals.append(state.open_goals.pop(goal_idx))
            state.attempts += 1

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
) -> tuple[str, int]:
    """Select the most promising open goal using critic scores."""
    if len(open_goals) == 1:
        return open_goals[0], 0

    best_idx = 0
    best_score = -1.0
    for i, goal in enumerate(open_goals):
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
