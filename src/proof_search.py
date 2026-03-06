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

    open_goals = [initial_goal]
    closed_goals: list[str] = []
    tactics_used: list[str] = []
    attempts = 0

    while open_goals and attempts < cfg.budget:
        goal, goal_idx = _select_goal(open_goals, pipeline, cfg.device)

        nav_output = _infer(goal, pipeline, cfg.device)

        # Hammer delegation for automated goals
        if cfg.hammer_delegation and _should_hammer(nav_output):
            candidates = resolve(nav_output, conn, context, anchor_id_map, premise_limit=16)
            premise_names = candidates[0].premises[:16] if candidates else []
            result = lean.try_hammer(goal, premise_names)
            attempts += 1
            if result.success:
                open_goals.pop(goal_idx)
                closed_goals.append(goal)
                tactics_used.append(result.tactic)
                open_goals.extend(result.new_goals)
                context.seed_entity_ids.clear()
                continue

        # Navigational resolution
        candidates = resolve(nav_output, conn, context, anchor_id_map)

        found = False
        for candidate in candidates[: cfg.max_candidates_per_step]:
            tactic_text = _build_tactic_text(candidate)
            result = lean.try_tactic(goal, tactic_text)
            attempts += 1

            if result.success:
                open_goals.pop(goal_idx)
                closed_goals.append(goal)
                tactics_used.append(tactic_text)
                open_goals.extend(result.new_goals)
                found = True
                break

        if not found:
            # Move this goal to the back and try another
            if len(open_goals) > 1:
                open_goals.append(open_goals.pop(goal_idx))
            attempts += 1

    return SearchResult(
        success=len(open_goals) == 0,
        theorem_id=theorem_id,
        tactics_used=tactics_used,
        attempts=attempts,
        goals_closed=len(closed_goals),
        goals_remaining=len(open_goals),
    )


def _infer(
    goal_state: str,
    pipeline: Pipeline,
    device: str,
) -> NavOutput:
    """Single neural inference: goal → NavOutput."""
    embeddings = pipeline.encoder.encode([goal_state])
    features, _, _ = pipeline.analyzer(embeddings)
    bridge_out = pipeline.bridge(features)
    return pipeline.navigator.predict(bridge_out)


def _select_goal(
    open_goals: list[str],
    pipeline: Pipeline,
    device: str,
) -> tuple[str, int]:
    """Select the most promising open goal using critic scores."""
    if len(open_goals) == 1:
        return open_goals[0], 0

    best_idx = 0
    best_score = -1.0
    for i, goal in enumerate(open_goals):
        nav = _infer(goal, pipeline, device)
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
