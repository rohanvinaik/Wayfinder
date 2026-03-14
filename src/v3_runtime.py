"""
v3 runtime — boundary learning + energy refinement orchestrator.

Parallel orchestration path that does NOT modify v1 (proof_search.py)
or v2 (arbiter.py). Adds OTP-scored navigation, censor pruning, and
constraint-report-based template selection. Energy refinement (v3B) is
gated behind config flag.

See DESIGN §12 and PLAN §Phase 7 for specification.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import torch

from src.censor import CensorNetwork
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.lean_interface import LeanKernel
from src.nav_contracts import NavOutput
from src.proof_search import SearchConfig, SearchResult
from src.resolution import Candidate, SearchContext, resolve
from src.sketch_predictor import SketchPredictor
from src.specialist_navigator import ExecutionSlot
from src.template_classifier import TemplateClassifier
from src.v3_contracts import SearchTrace
from src.v3_scoring import apply_bank_idf, build_constraint_report


@dataclass
class V3Slots:
    """Bundle of all v3 pipeline components."""

    encoder: GoalEncoder
    analyzer: GoalAnalyzer
    classifier: TemplateClassifier
    sketch_predictor: SketchPredictor
    execution: ExecutionSlot
    lean: LeanKernel
    censor: CensorNetwork | None = None


@dataclass
class V3Config:
    """v3-specific configuration layered on top of SearchConfig."""

    bank_idf: dict[str, float] = field(default_factory=dict)
    censor_threshold: float = 0.5
    safety_net_k: int = 3
    constraint_weights: dict[str, float] = field(
        default_factory=lambda: {"bank": 1.0, "critic": 0.5, "censor": 2.0, "anchor": 0.3}
    )
    energy_enabled: bool = False


@dataclass
class _V3SearchState:
    """Mutable state for a v3 proof search run."""

    open_goals: list[str]
    closed_goals: list[str] = field(default_factory=list)
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    lean_calls: int = 0
    templates_tried: dict[str, list[int]] = field(default_factory=dict)
    trace: SearchTrace = field(default_factory=lambda: SearchTrace(theorem_id=""))
    _feature_cache: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class _V3SearchEnv:
    """Immutable environment for v3 search."""

    slots: V3Slots
    conn: sqlite3.Connection
    context: SearchContext
    anchor_id_map: dict[str, int] | None
    config: SearchConfig
    v3_config: V3Config


@dataclass
class V3SearchParams:
    """Optional parameters for v3_search, grouped to reduce argument count."""

    config: SearchConfig | None = None
    v3_config: V3Config | None = None
    anchor_id_map: dict[str, int] | None = None
    accessible_theorem_id: int | None = None
    max_template_retries: int = 3


def v3_search(
    theorem_id: str,
    initial_goal: str,
    slots: V3Slots,
    conn: sqlite3.Connection,
    params: V3SearchParams | None = None,
) -> SearchResult:
    """Run v3 proof search on a single theorem.

    Pipeline: encode → OTP-scored navigate → censor prune →
    template classify → constraint-report score → Lean verify → retry.

    Args:
        theorem_id: Identifier for the theorem being proved.
        initial_goal: Initial goal state text.
        slots: v3 pipeline components.
        conn: SQLite connection to proof network.
        params: Optional search parameters (config, v3_config, anchor_id_map, etc.).
    Returns:
        SearchResult with proof attempt details.
    """
    p = params or V3SearchParams()
    cfg = p.config or SearchConfig()
    v3_cfg = p.v3_config or V3Config()
    context = SearchContext(
        accessible_theorem_id=p.accessible_theorem_id if cfg.accessible_premises else None,
    )
    state = _V3SearchState(
        open_goals=[initial_goal],
        trace=SearchTrace(theorem_id=theorem_id, mode="v3"),
    )
    env = _V3SearchEnv(
        slots=slots,
        conn=conn,
        context=context,
        anchor_id_map=p.anchor_id_map,
        config=cfg,
        v3_config=v3_cfg,
    )

    while state.open_goals and state.attempts < cfg.budget:
        _v3_search_step(state, env, p.max_template_retries)

    success = len(state.open_goals) == 0
    state.trace.result = "proved" if success else "failed"
    state.trace.lean_calls = state.lean_calls

    return SearchResult(
        success=success,
        theorem_id=theorem_id,
        tactics_used=state.tactics_used,
        attempts=state.attempts,
        goals_closed=len(state.closed_goals),
        goals_remaining=len(state.open_goals),
    )


def _v3_search_step(
    state: _V3SearchState,
    env: _V3SearchEnv,
    max_template_retries: int,
) -> None:
    """One iteration: goal select → infer → censor prune → try candidates."""
    slots = env.slots
    goal_idx = _select_goal(state.open_goals, slots, state)
    goal = state.open_goals[goal_idx]

    # Check template retry budget for this goal
    tried = state.templates_tried.get(goal, [])
    if len(tried) >= max_template_retries:
        _rotate_goal(state, goal_idx)
        state.attempts += 1
        return

    # Slot 1: PERCEPTION
    features = _cached_encode_goal(goal, slots, state)

    # Slot 4: EXECUTION (navigator inference)
    nav_output = slots.execution.predict(features)

    # OTP scoring: apply bank-IDF weights
    bank_scores = apply_bank_idf(nav_output, env.v3_config.bank_idf)

    # Resolve candidates from proof network
    candidates = resolve(
        nav_output,
        env.conn,
        env.context,
        env.anchor_id_map,
        tactic_limit=env.config.max_candidates_per_step,
    )

    if not candidates:
        _rotate_goal(state, goal_idx)
        state.attempts += 1
        return

    # Record this attempt for template retry tracking
    state.templates_tried.setdefault(goal, []).append(state.attempts)

    # Censor pruning
    pruned = _censor_prune(
        goal,
        features,
        candidates,
        slots,
        env.v3_config,
        state,
    )

    # Hammer delegation check
    if env.config.hammer_delegation and _should_hammer(nav_output):
        if _try_hammer(goal, goal_idx, candidates, state, env):
            return

    # Try pruned candidates
    if _try_candidates(goal, goal_idx, pruned, bank_scores, nav_output, state, env):
        return

    # All failed — rotate goal
    _rotate_goal(state, goal_idx)
    state.attempts += 1


def _censor_prune(
    goal: str,
    goal_features: torch.Tensor,
    candidates: list[Candidate],
    slots: V3Slots,
    v3_cfg: V3Config,
    state: _V3SearchState,
) -> list[Candidate]:
    """Apply censor to prune candidates predicted to fail.

    Safety net: never prune ALL candidates — keep top-k by censor
    confidence if everything would be pruned.
    """
    if slots.censor is None:
        return candidates

    scored: list[tuple[Candidate, float]] = []
    for candidate in candidates:
        # Build tactic feature placeholder — full pipeline uses tactic anchor embeddings
        censor_input_dim = int(getattr(slots.censor.network[0], "in_features", 384))
        tactic_dim = censor_input_dim - int(goal_features.shape[-1])
        tactic_feat = torch.zeros(1, tactic_dim)
        pred = slots.censor.predict(goal_features, tactic_feat)

        scored.append((candidate, pred.failure_probability))

        state.trace.pruning_decisions.append(
            {
                "goal": goal[:80],
                "tactic": candidate.tactic_name,
                "failure_prob": pred.failure_probability,
                "pruned": pred.should_prune,
            }
        )

    # Filter by threshold
    kept = [c for c, prob in scored if prob < v3_cfg.censor_threshold]

    # Safety net: never prune ALL
    if not kept:
        scored.sort(key=lambda x: x[1])  # lowest failure prob first
        kept = [c for c, _ in scored[: v3_cfg.safety_net_k]]

    return kept


def _try_candidates(
    goal: str,
    goal_idx: int,
    candidates: list[Candidate],
    bank_scores: dict[str, float],
    nav_output: NavOutput,
    state: _V3SearchState,
    env: _V3SearchEnv,
) -> bool:
    """Try candidates, building ConstraintReport for each. Returns True if goal closed."""
    for candidate in candidates:
        if state.attempts >= env.config.budget:
            return False

        # Build constraint report for audit trail
        report = build_constraint_report(
            bank_scores=bank_scores,
            critic_distance=nav_output.critic_score,
            censor_score=0.0,  # already pruned by censor
            anchor_alignment=candidate.score,
            weights=env.v3_config.constraint_weights,
        )
        state.trace.constraint_reports.append(report.to_dict())

        tactic_text = _build_tactic_text(candidate)
        result = env.slots.lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        state.lean_calls += 1

        state.trace.steps.append(
            {
                "goal": goal[:80],
                "tactic": tactic_text,
                "success": result.success,
                "constraint_score": report.total_score,
            }
        )

        if result.success:
            _close_goal(goal, goal_idx, result.tactic, result.new_goals, state)
            return True

    return False


def _try_hammer(
    goal: str,
    goal_idx: int,
    candidates: list[Candidate],
    state: _V3SearchState,
    env: _V3SearchEnv,
) -> bool:
    """Attempt hammer delegation. Returns True if goal was closed."""
    premise_names = candidates[0].premises[:16] if candidates else []
    result = env.slots.lean.try_hammer(goal, premise_names)
    state.attempts += 1
    state.lean_calls += 1
    if result.success:
        _close_goal(goal, goal_idx, result.tactic, result.new_goals, state)
        return True
    return False


def _select_goal(
    open_goals: list[str],
    slots: V3Slots,
    state: _V3SearchState,
) -> int:
    """Select the most promising open goal using critic scores."""
    if len(open_goals) <= 1:
        return 0

    best_idx = 0
    best_score = -1.0
    for i, goal in enumerate(open_goals):
        features = _cached_encode_goal(goal, slots, state)
        nav_output = slots.execution.predict(features)
        if nav_output.critic_score > best_score:
            best_score = nav_output.critic_score
            best_idx = i
    return best_idx


def _cached_encode_goal(
    goal: str,
    slots: V3Slots,
    state: _V3SearchState,
) -> torch.Tensor:
    """Cached goal encoding — avoids redundant encoder + analyzer forward passes."""
    cached = state._feature_cache.get(goal)
    if cached is not None:
        return cached
    embedding = slots.encoder.encode([goal])
    features, _, _ = slots.analyzer(embedding)
    state._feature_cache[goal] = features
    return features


def _close_goal(
    goal: str,
    goal_idx: int,
    tactic: str,
    new_goals: list[str],
    state: _V3SearchState,
) -> None:
    """Close a goal and update state. Invalidates feature cache."""
    state.open_goals.pop(goal_idx)
    state.closed_goals.append(goal)
    state.tactics_used.append(tactic)
    state.open_goals.extend(new_goals)
    state._feature_cache.clear()


def _rotate_goal(state: _V3SearchState, goal_idx: int) -> None:
    """Rotate failed goal to back of queue."""
    if len(state.open_goals) > 1:
        state.open_goals.append(state.open_goals.pop(goal_idx))


def _should_hammer(nav_output: NavOutput) -> bool:
    """Check if navigator suggests automation (hammer delegation)."""
    return nav_output.directions.get("automation", 0) == -1


def _build_tactic_text(candidate: Candidate) -> str:
    """Build a tactic application string from a candidate."""
    if not candidate.premises:
        return candidate.tactic_name
    premises_str = " ".join(candidate.premises[:4])
    return f"{candidate.tactic_name} {premises_str}"
