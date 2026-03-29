"""
Proof search — outer loop managing goal selection, neural inference, and verification.

Coordinates the full pipeline: goal state → encoder → analyzer → bridge →
navigator → resolution → Lean kernel verification. Manages open goals,
proof context, search budget, and hammer delegation.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.bridge import InformationBridge
from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import run_ducky_on_goal
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.hard_data_tags import classify_goal_bucket
from src.hard_data_tags import sanitize_goal_text as _sanitize_goal_text
from src.lean_interface import LeanKernel
from src.nav_contracts import NavOutput, TacticResult
from src.proof_navigator import ProofNavigator
from src.resolution import Candidate, SearchContext, resolve

_LOGGER = logging.getLogger(__name__)
_EXEC_SELECTOR_CACHE: dict[str, tuple[Any, Any] | None] = {}
_FAMILY_TORCH_CACHE: dict[str, Any | None] = {}
_FAMILY_NUMPY_CACHE: dict[str, dict[str, Any] | None] = {}


@dataclass
class SearchConfig:
    """Configuration for proof search.

    search_mode controls which lanes are active:
        "full"               — hammer + structural + solver + learned (default)
        "learned_only"       — learned candidates only (no bootstrap, no hammer)
        "learned_structural" — learned + structural_core (no solver_bootstrap)
        "no_learned"         — hammer + structural + solver (no learned candidates)
    """

    budget: int = 600
    hammer_delegation: bool = True
    accessible_premises: bool = True
    max_candidates_per_step: int = 8
    device: str = "cpu"
    search_mode: str = "full"
    # Strip learned premises from hammer calls (for premise-value ablation)
    no_learned_premises: bool = False
    # Cosine rw beam width (top-k symbols to try per goal)
    cosine_rw_beam: int = 1
    # Sequential bare-rewrite chain (rw3_bare lane): max atoms and Lean calls per activation
    cosine_rw_seq_max_atoms: int = 10
    cosine_rw_seq_max_calls: int = 50
    # Temporal controller: "off" (default), "shadow" (log only), "active" (controls routing)
    temporal_mode: str = "off"
    # Strategy Arbiter: path to strategy_memory.json. When set + temporal_mode="arbiter",
    # uses dominance-aware routing from strategy memory instead of static lane order.
    # Modes: "arbiter_full", "arbiter_goal_only", "arbiter_lane_only"
    strategy_memory_path: str = ""
    # Enable cosine_rw_seq in static lane order (requires sentence_encoder)
    cosine_rw_seq_enabled: bool = False
    # Enable cosine_simp lane: bare simp then simp [top1] (requires sentence_encoder)
    cosine_simp_enabled: bool = False
    # Enable interleaved structural bootstrap lane (intros → simp → aesop loop)
    interleaved_bootstrap_enabled: bool = False
    # Max depth (intro rounds) for interleaved bootstrap
    interleaved_bootstrap_max_depth: int = 4
    # Max total Lean calls for interleaved bootstrap per activation
    interleaved_bootstrap_max_calls: int = 20
    # Enable cosine_apply lane: top-k apply attempts (requires sentence_encoder)
    cosine_apply_enabled: bool = False
    # Gated cosine_apply: only fire after IB has failed on a single apply-shaped goal.
    # Recommended over cosine_apply_enabled (always-on). Requires cosine_apply_enabled=True.
    cosine_apply_gated: bool = False
    # Beam width for cosine_apply (top-k symbols to try per goal)
    cosine_apply_beam: int = 5
    # ExecSelector checkpoint path (optional). When set, replaces cosine ranking in
    # the cosine_apply lane with selector-ranked candidates. Requires cosine_apply_enabled=True.
    exec_apply_selector_path: str = ""
    # Candidate pool fed to the selector (top-k by cosine before selector rescoring)
    exec_apply_selector_pool: int = 20
    # Apply trigger classifier checkpoint (optional). When set + cosine_apply_gated=True,
    # replaces the deterministic _apply_gate with a learned trigger.
    apply_trigger_path: str = ""
    apply_trigger_threshold: float = 0.47
    # Collect per-step trace in SearchResult.step_trace (off by default; for dataset harvesting)
    collect_trace: bool = False
    # Family classifier: path to trained model (npz). When set, reorders lanes
    # based on predicted specialist family for the current goal.
    family_classifier_path: str = ""
    # PyTorch SoM family classifier checkpoint path (best.pt). When set, uses the
    # three-stage SoM router for lane reordering and falls back to family_classifier_path
    # if loading fails.
    family_classifier_torch_path: str = ""
    # Enable the residual normalization fallback (`norm_cast`/`ring_nf`/`push_neg`; exact?).
    # Used for paired benchmarking of the EXP-SOM-009 deployment.
    norm_then_close_enabled: bool = True
    # Optional cap on the number of progress-making search steps. Used by the
    # hard-proof depth ladder to compare shallow vs deeper post-main search.
    # `0` means no explicit depth cap beyond the normal search budget.
    max_progress_steps: int = 0
    # Permit the target theorem itself to be used as a closer.
    # This is useful for diagnostics only; benchmark-style runs should keep it off.
    allow_self_application: bool = False
    # Reject progress that introduces obvious metavariable-corrupted branches
    # (`?m.*`, placeholder type goals) and keep the theorem for a replanner stage.
    metavariable_penalty_enabled: bool = True
    # Reject progress that simply recreates a recent goal-state signature, which
    # typically indicates a fold/unfold or rewrite loop.
    state_loop_penalty_enabled: bool = True
    # Holographic premises: entity names identified by multi-projection coherence
    # scoring on the structured residual.  When set, these are injected into the
    # premise pool for cosine ranking, giving the first-order search access to
    # premises that the proof network's accessible_premises may not include.
    holographic_premises: list[str] | None = None
    # Number of recent goal-state signatures used when checking loop repeats.
    state_loop_window: int = 3
    # Dr. Ducky: bounded local symbolic repair lane for human-style cleanup.
    dr_ducky_enabled: bool = False
    dr_ducky_max_programs: int = 24
    dr_ducky_max_rounds: int = 3
    dr_ducky_goal_limit: int = 3


@dataclass
class SearchResult:
    """Result of a proof search attempt."""

    success: bool
    theorem_id: str
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    goals_closed: int = 0
    goals_remaining: int = 0
    progress_steps: int = 0
    final_goals: list[str] = field(default_factory=list)
    # Per-goal provenance: which lane closed each goal
    close_provenance: list[str] = field(default_factory=list)
    # Temporal controller trace (shadow or active mode)
    temporal_trace: list[dict] = field(default_factory=list)
    # Per-step trace for dataset harvesting (populated when collect_trace=True)
    # Each entry: {step, goal_before, lane, tactic, progress, new_goals}
    step_trace: list[dict] = field(default_factory=list)
    # Apply trigger/lane instrumentation
    trigger_fire_count: int = 0
    trigger_reject_count: int = 0
    apply_attempt_count: int = 0
    apply_accept_count: int = 0
    apply_goal_close_count: int = 0
    metavariable_penalty_count: int = 0
    state_loop_penalty_count: int = 0
    replanner_trigger_count: int = 0


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
    # Cosine rw lane: encoder + entity maps (optional, loaded on demand)
    sentence_encoder: Any | None = None
    id_to_name: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)
    # ExecSelector for apply lane (optional, loaded when exec_apply_selector_path is set)
    exec_apply_selector: Any | None = None  # ExecSelector nn.Module
    exec_apply_encoder: Any | None = None  # SentenceTransformer for selector features
    # Candidate type lookup: name -> type_pp (populated from DB entities.type_pp)
    premise_type_cache: dict[str, str] = field(default_factory=dict)
    # Family classifier weights (optional, loaded from npz)
    family_classifier: dict[str, Any] | None = None
    # PyTorch SoM family classifier (optional, loaded from best.pt)
    family_classifier_torch: Any | None = None
    # Goal embedding cache: avoids re-encoding the same goal text across family attempts
    _goal_emb_cache: dict[str, Any] = field(default_factory=dict)
    # Holographic premises: injected by the bridge from multi-projection coherence
    # scoring.  These are entity names that structurally match the residual and should
    # be included in every cosine ranking pass.
    holographic_premises: list[str] = field(default_factory=list)
    # Apply trigger classifier (optional, loaded when apply_trigger_path is set)
    apply_trigger_trunk: Any | None = None
    apply_trigger_head: Any | None = None
    apply_trigger_threshold: float = 0.47
    allow_self_application: bool = False


@dataclass
class _SearchState:
    """Mutable state for a single proof search run."""

    open_goals: list[str]
    theorem_id: str = ""
    closed_goals: list[str] = field(default_factory=list)
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    # Per-goal provenance: which lane closed each goal
    close_provenance: list[str] = field(default_factory=list)
    # One-shot cache for family/goal attempts keyed by explicit cache keys
    # (for example cosine-family lanes). Structural lanes keep separate
    # guards so they do not starve each other.
    _expensive_tried: set[str] = field(default_factory=set)
    # Goals where the one-shot structural fallback has already been tried.
    _structural_tried: set[str] = field(default_factory=set)
    # Goals where the interleaved bootstrap lane has already been tried.
    _interleaved_tried: set[str] = field(default_factory=set)
    # Cache: goal_state text → NavOutput (cleared on goal set mutation)
    _infer_cache: dict[str, NavOutput] = field(default_factory=dict)
    # Trigger instrumentation: last trigger evaluation result
    _last_trigger_prob: float = -1.0
    _last_trigger_fired: bool = False
    _trigger_fire_count: int = 0
    _trigger_reject_count: int = 0
    _apply_attempt_count: int = 0
    _apply_accept_count: int = 0
    _apply_goal_close_count: int = 0
    _last_apply_candidates: list[dict] = field(default_factory=list)
    _last_apply_diag: dict = field(default_factory=dict)
    _post_apply_goals: set[str] = field(default_factory=set)
    _last_lanes_tried: list[str] = field(default_factory=list)
    _last_lane_order_executed: list[str] = field(default_factory=list)
    _goal_signature_history: list[str] = field(default_factory=list)
    _metavariable_penalty_count: int = 0
    _state_loop_penalty_count: int = 0
    _replanner_trigger_count: int = 0
    _last_pathology_tags: list[str] = field(default_factory=list)


_METAVAR_RE = re.compile(r"\?[A-Za-z][A-Za-z0-9_.]*")


def _goal_target_text(goal: str) -> str:
    text = _sanitize_goal_text((goal or "").strip())
    if "⊢" in text:
        return text.split("⊢", 1)[1].strip()
    return text


def _goal_signature(goals: list[str]) -> str:
    normalized = [" ".join(_goal_target_text(goal).split()) for goal in goals if str(goal).strip()]
    return " || ".join(sorted(normalized))


def _goal_has_metavariable(goal: str) -> bool:
    return bool(_METAVAR_RE.search(goal or ""))


def _goal_is_bare_type(goal: str) -> bool:
    target = _goal_target_text(goal)
    if not target or _goal_has_metavariable(target):
        return False
    compact = " ".join(target.replace("\n", " ").split())
    if compact in {"P", "Type", "Prop"}:
        return True
    tokens = compact.split()
    if len(tokens) <= 3 and all(
        token in {"→", "Type", "Prop", "ℝ", "ℕ", "ℤ"} or token[:1].isupper()
        for token in tokens
    ):
        return True
    return False


def _snapshot_frontier(state: _SearchState) -> dict[str, Any]:
    return {
        "open_goals": list(state.open_goals),
        "closed_goals": list(state.closed_goals),
        "tactics_used": list(state.tactics_used),
        "close_provenance": list(state.close_provenance),
        "post_apply_goals": set(state._post_apply_goals),
    }


def _restore_frontier(state: _SearchState, snapshot: dict[str, Any]) -> None:
    state.open_goals = list(snapshot["open_goals"])
    state.closed_goals = list(snapshot["closed_goals"])
    state.tactics_used = list(snapshot["tactics_used"])
    state.close_provenance = list(snapshot["close_provenance"])
    state._post_apply_goals = set(snapshot["post_apply_goals"])
    state._infer_cache.clear()


def _progress_pathology_tags(
    *,
    goals_before: list[str],
    goals_after: list[str],
    tactic: str,
    recent_signatures: list[str],
    loop_window: int,
) -> list[str]:
    tags: list[str] = []
    signature_before = _goal_signature(goals_before)
    signature_after = _goal_signature(goals_after)
    normalized_after = [" ".join(_goal_target_text(goal).split()) for goal in goals_after if str(goal).strip()]
    if any(_goal_has_metavariable(goal) for goal in goals_after):
        tags.append("metavariable_corruption")
    if any(_goal_is_bare_type(goal) for goal in goals_after):
        tags.append("bare_type_side_goal")
    if len(goals_after) >= 5 or len(goals_after) >= len(goals_before) + 3:
        tags.append("goal_explosion")
    if normalized_after and len(set(normalized_after)) < len(normalized_after):
        tags.append("duplicate_goal_progress")
        if tactic.startswith("simp") or tactic in _NUMERIC_SOLVER_TACTICS:
            tags.append("duplicate_goal_pseudo_progress")
    if signature_after and signature_after in recent_signatures[-max(loop_window, 1):]:
        tags.append("state_loop")
    if signature_after and signature_before and signature_after == signature_before and tactic:
        tags.append("state_loop")
    if ("state_loop" in tags) and (tactic.startswith("simp") or tactic.startswith("rw ")):
        tags.append("definition_tug_of_war")
    if ("metavariable_corruption" in tags) and (tactic.startswith("rw [←") or tactic.startswith("rw[←")):
        tags.append("backward_rewrite_metavariable")
    return list(dict.fromkeys(tags))


def _close_goal(
    goal: str,
    goal_idx: int,
    tactic: str,
    new_goals: list[str],
    state: _SearchState,
    provenance: str = "learned",
) -> None:
    """Close a goal and update search state. Invalidates inference cache.

    Args:
        provenance: Which lane closed this goal — "automation", "bootstrap", or "learned".
    """
    state.open_goals.pop(goal_idx)
    state.closed_goals.append(goal)
    state.tactics_used.append(tactic)
    state.close_provenance.append(provenance)
    # After apply/exec_selector_apply, prepend new subgoals so they're tried first
    # (they're often simple and closeable by automation). Other lanes append.
    # Also mark them as post-apply for subgoal-scoped retrieval.
    if provenance in ("cosine_apply", "exec_selector_apply") and new_goals:
        state.open_goals = list(new_goals) + state.open_goals
        state._post_apply_goals.update(new_goals)
    else:
        state.open_goals.extend(new_goals)
    state._infer_cache.clear()
    # When a goal closes completely (no new subgoals), remaining goals may
    # now be closeable (e.g. after apply, sibling goals get simpler).
    # Reset structural cache for remaining goals so they get fresh tries.
    if not new_goals and len(state.open_goals) <= 3:
        state._structural_tried.clear()



def _cached_infer(
    goal_state: str,
    pipeline: Pipeline,
    state: _SearchState,
) -> NavOutput:
    """Cached neural inference — avoids redundant forward passes for the same goal."""
    cached = state._infer_cache.get(goal_state)
    if cached is not None:
        return cached
    result = _infer(_sanitize_goal_text(goal_state), pipeline)
    state._infer_cache[goal_state] = result
    return result


def _try_hammer(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    no_learned_premises: bool = False,
) -> bool:
    """Attempt hammer delegation. Returns True if goal was closed."""
    if no_learned_premises:
        premise_names: list[str] = []
    else:
        candidates = resolve(nav_output, env.conn, context, env.anchor_id_map, premise_limit=16)
        premise_names = candidates[0].premises[:16] if candidates else []
    result = env.lean.try_hammer(goal, premise_names)
    state.attempts += 1
    if result.success:
        _close_goal(goal, goal_idx, result.tactic, result.new_goals, state, "automation")
        context.seed_entity_ids.clear()
        return True
    return False


def _filter_candidates_by_family(
    candidates: list[Candidate],
    allowed_families: set[str] | None,
) -> list[Candidate]:
    """Filter candidates to only those matching allowed tactic families.

    If allowed_families is None, returns all candidates (no filtering).
    Family matching is by first word of the tactic name.
    """
    if allowed_families is None:
        return candidates
    return [c for c in candidates if c.tactic_name.split()[0] in allowed_families]


def _is_self_application_tactic(tactic: str, theorem_id: str) -> bool:
    text = (tactic or "").strip()
    theorem = (theorem_id or "").strip()
    if not text or not theorem:
        return False
    theorem_pat = re.escape(theorem)
    return bool(
        re.match(
            rf"^(?:exact|apply|refine)\s+\(?@?{theorem_pat}(?:\.[A-Za-z0-9_']+)?(?=$|[\s\)\]\}};,])",
            text,
        )
    )


def _try_candidates(
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    allowed_families: set[str] | None = None,
) -> bool:
    """Try navigational candidates. Returns True if goal was closed.

    If allowed_families is set, only candidates from those tactic families
    are tried (residual executor top-k gate).
    """
    candidates = resolve(nav_output, env.conn, context, env.anchor_id_map)
    candidates = _filter_candidates_by_family(candidates, allowed_families)
    for candidate in candidates[: env.max_candidates]:
        tactic_text = _build_tactic_text(candidate)
        if not env.allow_self_application and _is_self_application_tactic(tactic_text, state.theorem_id):
            continue
        result = env.lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic_text, result.new_goals, state, "learned")
            return True
    return False


def _live_dr_ducky_row(goal: str, state: _SearchState) -> dict[str, Any]:
    goals_remaining = len(state.open_goals)
    goals_closed = len(state.closed_goals)
    if goals_remaining <= 1:
        residual_bucket = "single_goal_near_miss" if goals_closed > 0 else "single_goal_stall"
    elif goals_remaining <= 5:
        residual_bucket = "multi_goal_small_progress" if goals_closed > 0 else "multi_goal_small_stall"
    else:
        residual_bucket = "multi_goal_large_progress" if goals_closed > 0 else "multi_goal_large_stall"
    return {
        "theorem_id": state.theorem_id,
        "last_goal": goal,
        "goal_state": goal,
        "last_goal_bucket": classify_goal_bucket(goal),
        "residual_bucket": residual_bucket,
        "goals_closed": goals_closed,
        "goals_remaining": goals_remaining,
        "attempts": state.attempts,
        "lane_sequence": "→".join(dict.fromkeys(state.close_provenance)),
        "search_pathology_tags": list(state._last_pathology_tags),
        "remaining_goals_snapshot": list(state.open_goals),
    }


def _try_dr_ducky(
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    cfg: SearchConfig,
) -> bool:
    if len(state.open_goals) > cfg.dr_ducky_goal_limit:
        return False
    cache_key = f"dr_ducky:{goal}"
    if cache_key in state._expensive_tried:
        return False
    state._expensive_tried.add(cache_key)

    row = _live_dr_ducky_row(goal, state)
    capsule = build_goal_capsule(row)
    result = run_ducky_on_goal(
        goal,
        theorem_id=state.theorem_id,
        lean=env.lean,
        conn=env.conn,
        accessible_theorem_id=context.accessible_theorem_id,
        capsule=capsule,
        row_overrides=row,
        max_programs=cfg.dr_ducky_max_programs,
        max_rounds=cfg.dr_ducky_max_rounds,
    )
    winning = result.winning_program or {}
    if not (result.closed or result.progressed):
        return False
    script = str(winning.get("script", "") or "")
    program_id = str(winning.get("program_id", "") or "")
    tactic_desc = f"dr_ducky[{program_id}] {script}".strip()
    new_goals = [] if result.closed else list(winning.get("goals_after") or [])
    _close_goal(goal, goal_idx, tactic_desc or "dr_ducky", new_goals, state, "dr_ducky")
    return True


# ---------------------------------------------------------------------------
# Cosine local lanes: scope → encode → cosine rank → tactic beam + Lean verify
#
# Each family generates tactics from cosine-ranked accessible premises:
#   rw:    rw [sym], rw [← sym]
#   exact: exact sym
#   apply: apply sym
#   simp:  simp [top1, top2, top3]
# ---------------------------------------------------------------------------


def _cosine_rank_premises(
    goal: str,
    env: _SearchEnv,
    accessible_theorem_id: int | None,
    max_premises: int = 30,
    family: str = "rw",
    subgoal_scoped: bool = False,
) -> list[str] | None:
    """Get cosine-ranked premises for a goal. Returns None on failure.

    For rw/simp families, uses rw_scoper filtering (rewrite-compatible premises).
    For apply/exact, uses direct cosine ranking over accessible premises.

    When subgoal_scoped=True and the accessible pool is small (<5), falls back
    to global cosine search over all DB entities — needed for post-apply subgoals
    where the theorem's accessible premises may not contain the right closer.
    """
    if env.sentence_encoder is None or not env.id_to_name:
        return None
    clean_goal = _sanitize_goal_text(goal)

    from src.proof_network import get_accessible_premises

    premise_names: list[str] = []
    premise_ids: set[int] = set()
    if accessible_theorem_id is not None:
        premise_ids = get_accessible_premises(env.conn, accessible_theorem_id)
        premise_names = [env.id_to_name[pid] for pid in premise_ids if pid in env.id_to_name]

    # Inject holographic premises from multi-projection coherence scoring.
    # These are high-confidence matches identified by the second-order system
    # and should participate in cosine ranking alongside DB premises.
    if env.holographic_premises:
        seen = set(premise_names)
        for hp in env.holographic_premises:
            if hp not in seen:
                premise_names.append(hp)
                seen.add(hp)

    # For subgoal-scoped retrieval (post-apply subgoals), if the accessible
    # pool is small, supplement with global cosine search over all entities
    # that have type_pp (theorem-like entities only, not anchors/tactics)
    if subgoal_scoped and family in ("exact", "apply") and len(premise_names) < 5:
        typed_names = [
            name for name, tpp in env.premise_type_cache.items() if tpp
        ]
        if typed_names:
            # Merge: accessible premises first, then typed entities
            seen = set(premise_names)
            for n in typed_names:
                if n not in seen:
                    premise_names.append(n)
                    seen.add(n)
                if len(premise_names) >= 500:  # cap for performance
                    break

    if not premise_names:
        return None

    import numpy as np

    if family in ("rw", "simp"):
        # rw/simp: scope-filtered, then cosine-ranked within scope
        from src.rw_scoper import scope_for_rw

        scope = scope_for_rw(clean_goal, premise_names, max_premises=max_premises)
        if not scope.all_symbols:
            return None
        candidates = scope.all_symbols
    else:
        # apply/exact: direct cosine ranking over all accessible premises
        # Pool expansion: if pool is small, add one-hop neighbors (premises of premises).
        # NOTE: this relaxes strict theorem-site accessible-premises — disclose in benchmarks.
        if len(premise_ids) < 10 and family in ("apply", "exact"):
            neighbor_ids: set[int] = set()
            for pid in premise_ids:
                hop_rows = env.conn.execute(
                    "SELECT premise_id FROM accessible_premises WHERE theorem_id = ?", (pid,)
                ).fetchall()
                for (nid,) in hop_rows:
                    if nid not in premise_ids and nid in env.id_to_name:
                        neighbor_ids.add(nid)
            if neighbor_ids:
                extra = [env.id_to_name[nid] for nid in list(neighbor_ids)[:20]]
                premise_names = premise_names + extra
        candidates = premise_names

    cached_goal_emb = env._goal_emb_cache.get(clean_goal)
    if cached_goal_emb is not None:
        goal_emb = cached_goal_emb
    else:
        goal_emb = env.sentence_encoder.encode([clean_goal], normalize_embeddings=True)
        env._goal_emb_cache[clean_goal] = goal_emb
    cand_embs = env.sentence_encoder.encode(candidates, normalize_embeddings=True)
    scores = (goal_emb @ cand_embs.T).flatten()
    ranked_indices = np.argsort(-scores).tolist()
    return [candidates[i] for i in ranked_indices[:max_premises]]


def _exec_selector_topk(
    goal: str,
    cosine_ranked: list[str],
    env: _SearchEnv,
    k: int = 1,
) -> list[str]:
    """Rescore cosine_ranked candidates with ExecSelector; return top-k by selector score.

    Supports both v1 (770d) and v2 (796d, with compat features) based on
    the model's input dimension. Falls back to cosine top-k on any failure.
    """
    try:
        import re

        import numpy as np
        import torch

        model = env.exec_apply_selector
        encoder = env.exec_apply_encoder
        if model is None or encoder is None or not cosine_ranked:
            return cosine_ranked[:k]

        emb_dim = 384  # all-MiniLM-L6-v2
        clean_goal = _sanitize_goal_text(goal)
        texts = [clean_goal] + cosine_ranked
        embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        goal_emb = embs[0]
        cand_embs = embs[1:]
        cosine_scores = (cand_embs @ goal_emb).tolist()

        n = len(cosine_ranked)

        # Detect model version by input dim
        first_param = next(model.parameters())
        model_in_dim = first_param.shape[1] if first_param.dim() >= 2 else emb_dim * 2 + 2
        has_compat = model_in_dim > emb_dim * 2 + 2

        X = np.zeros((n, model_in_dim), dtype=np.float32)
        for i in range(n):
            X[i, :emb_dim] = goal_emb
            X[i, emb_dim:2 * emb_dim] = cand_embs[i]
            X[i, 2 * emb_dim] = float(cosine_scores[i])
            X[i, 2 * emb_dim + 1] = 1.0  # filter_passed

        if has_compat:
            # v2: add compatibility features using real candidate types from DB
            goal_target = ""
            for line in clean_goal.split("\n"):
                if "⊢" in line:
                    goal_target = line.split("⊢", 1)[1].strip()
                    break
            if not goal_target:
                goal_target = goal

            _TOP_HEADS = ["eq", "le", "lt", "iff", "not", "and", "exists", "mem"]
            _BINDER_RE = re.compile(r"[({⦃][^:(){}⦃⦄]*:\s*[^(){}⦃⦄]+[)}⦄]")

            def _head(text: str) -> str:
                text = text.strip().lstrip("(")
                m = re.match(r"(\w[\w.]*)", text)
                return m.group(1).lower() if m else ""

            def _head_oh(head: str) -> list[float]:
                return [1.0 if h == head else 0.0 for h in _TOP_HEADS]

            goal_head = _head(goal_target)
            goal_binders = len(_BINDER_RE.findall(goal_target)) / 10.0
            has_forall = float("∀" in goal)
            has_eq = float("=" in goal_target)
            has_impl = float("→" in goal_target)
            has_exists = float("∃" in goal)
            has_iff = float("↔" in goal_target)
            goal_head_oh = _head_oh(goal_head)

            # Extract goal local names from hypothesis lines
            goal_locals = []
            for line in clean_goal.split("\n"):
                if ":" in line and "⊢" not in line:
                    name = line.split(":")[0].strip()
                    if name and name[0].isalpha():
                        goal_locals.append(name)

            for i in range(n):
                cand_name = cosine_ranked[i]
                cand_type = env.premise_type_cache.get(cand_name, "")

                # Candidate conclusion head (after last →)
                cand_parts_split = re.split(r"→|->", cand_type)
                cand_conclusion = cand_parts_split[-1].strip() if cand_parts_split else ""
                cand_head = _head(cand_conclusion)

                head_match = 1.0 if goal_head and cand_head and goal_head == cand_head else 0.0
                cand_head_oh = _head_oh(cand_head)
                cand_binders = len(_BINDER_RE.findall(cand_type)) / 10.0
                cand_arity = (cand_type.count("→") + cand_type.count("->")) / 5.0
                name_overlap = sum(1 for nm in goal_locals if nm in cand_type) / 5.0

                compat = [
                    head_match,
                    *goal_head_oh,
                    *cand_head_oh,
                    goal_binders,
                    cand_binders,
                    name_overlap,
                    has_forall, has_eq, has_impl, has_exists, has_iff,
                    cand_arity,
                ]
                X[i, 2 * emb_dim + 2:] = compat

        with torch.no_grad():
            scores = model.score(torch.from_numpy(X)).cpu().numpy()

        ranked_idx = np.argsort(-scores).tolist()
        return [cosine_ranked[i] for i in ranked_idx[:k]]
    except Exception:
        return cosine_ranked[:k]


def _try_cosine_family(
    family: str,
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
    accessible_theorem_id: int | None = None,
    beam_width: int = 1,
) -> bool:
    """Try tactics from a cosine-ranked premise list for a given family.

    One-shot per (family, goal) pair. Returns True if any tactic succeeds.
    """
    cache_key = f"cosine_{family}:{goal}"
    if cache_key in state._expensive_tried:
        return False
    state._expensive_tried.add(cache_key)

    ranked = _cosine_rank_premises(goal, env, accessible_theorem_id, family=family)
    if not ranked:
        return False
    if not env.allow_self_application and state.theorem_id:
        ranked = [sym for sym in ranked if sym != state.theorem_id]
        if not ranked:
            return False

    provenance = f"cosine_{family}"
    top = ranked[:beam_width]

    # ExecSelector reranking for the apply family.
    # Mixed beam: union of selector top-k and cosine top-1 (deduplicated).
    # The selector finds structurally compatible candidates; cosine finds
    # semantically nearest. The union covers both failure modes.
    if (
        family == "apply"
        and env.exec_apply_selector is not None
        and env.exec_apply_encoder is not None
    ):
        selector_top = _exec_selector_topk(goal, ranked, env, k=beam_width)
        cosine_top1 = ranked[:1]
        # Merge: selector picks first, then cosine top-1 if not already included
        seen = set(selector_top)
        top = list(selector_top)
        for c in cosine_top1:
            if c not in seen:
                top.append(c)
        provenance = "exec_selector_apply"

    if family == "rw":
        from src.rw_scoper import extract_goal_target, infer_direction

        goal_target = extract_goal_target(goal)
        for sym in top:
            # Per-tier direction: try the more likely direction first
            d = infer_direction(sym, goal_target)
            if d == "backward":
                dirs = [f"rw [← {sym}]", f"rw [{sym}]"]
            else:
                dirs = [f"rw [{sym}]", f"rw [← {sym}]"]
            for tac in dirs:
                result = env.lean.try_tactic(goal, tac)
                state.attempts += 1
                if result.success:
                    _close_goal(goal, goal_idx, tac, result.new_goals, state, provenance)
                    return True

    elif family == "exact":
        for sym in top:
            tac = f"exact {sym}"
            result = env.lean.try_tactic(goal, tac)
            state.attempts += 1
            if result.success:
                _close_goal(goal, goal_idx, tac, result.new_goals, state, provenance)
                return True

    elif family == "apply":
        for sym in top:
            # Try multiple tactic forms per candidate:
            # 1. exact (full unification, closes completely)
            # 2. refine with wildcards (lets elaborator fill implicits)
            # 3. bare apply (fallback, may leave metavar holes)
            tactics_to_try = [
                f"exact {sym}",
                f"refine {sym} ?_ ?_",
                f"refine {sym} ?_ ?_ ?_",
                f"apply {sym}",
            ]
            for tac in tactics_to_try:
                result = env.lean.try_tactic(goal, tac)
                state.attempts += 1
                state._last_apply_candidates.append({
                    "candidate": sym,
                    "tactic": tac,
                    "accepted": result.success,
                    "new_goals": len(result.new_goals) if result.success else 0,
                })
                if result.success:
                    _close_goal(goal, goal_idx, tac, result.new_goals, state, provenance)
                    return True

    elif family == "simp":
        # Policy: bare simp first (most robust), then simp [top1] as fallback.
        # top-5 hinted simp underperforms top-1 in benchmarks (more hints hurt acceptance).
        simp_tactics: list[str] = ["simp"]
        if top:
            simp_tactics.append(f"simp [{top[0]}]")
        for tac in simp_tactics:
            result = env.lean.try_tactic(goal, tac)
            state.attempts += 1
            if result.success:
                _close_goal(goal, goal_idx, tac, result.new_goals, state, provenance)
                return True

    return False


def _try_cosine_rw_seq(
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
    accessible_theorem_id: int | None,
    beam_width: int,
    max_atoms: int,
    max_calls: int,
) -> bool:
    """Sequential bare-rewrite chain (rw3_bare lane).

    Executes a capped loop of single bare rewrites against the current goal:
      1. Rank accessible premises by cosine similarity to current goal.
      2. Try top-k bare rw tactics (both directions).
      3. On acceptance: replace current goal with the new residual goal,
         refresh scope, repeat.
      4. Stop when: goal closes, max_atoms reached, max_calls reached,
         or no premise fires.

    Each accepted step is recorded as progress via _close_goal on the
    *original* goal_idx using the chain label "cosine_rw_seq". Only the
    final closed-goal state is propagated; intermediate residuals are
    internal to this function.

    Returns True if at least one atom fired (progress made).
    """
    cache_key = f"cosine_rw_seq:{goal}"
    if cache_key in state._expensive_tried:
        return False
    state._expensive_tried.add(cache_key)

    from src.rw_scoper import extract_goal_target, infer_direction

    current_goal = goal
    total_calls = 0
    atoms_fired = 0
    provenance = "cosine_rw_seq"

    for _ in range(max_atoms):
        if total_calls >= max_calls:
            break

        ranked = _cosine_rank_premises(current_goal, env, accessible_theorem_id)
        if not ranked:
            break
        if not env.allow_self_application and state.theorem_id:
            ranked = [sym for sym in ranked if sym != state.theorem_id]
            if not ranked:
                break

        goal_target = extract_goal_target(current_goal)
        top = ranked[:beam_width]
        step_fired = False

        for sym in top:
            d = infer_direction(sym, goal_target)
            dirs = (
                [f"rw [← {sym}]", f"rw [{sym}]"]
                if d == "backward"
                else [f"rw [{sym}]", f"rw [← {sym}]"]
            )
            for tac in dirs:
                if total_calls >= max_calls:
                    break
                result = env.lean.try_tactic(current_goal, tac)
                state.attempts += 1
                total_calls += 1
                if result.success:
                    atoms_fired += 1
                    if not result.new_goals:
                        # Goal fully closed by this rewrite
                        _close_goal(goal, goal_idx, tac, [], state, provenance)
                        return True
                    # Progress: advance to residual goal
                    new_g = result.new_goals[0]
                    current_goal = new_g if isinstance(new_g, str) else str(new_g)
                    step_fired = True
                    break
            if step_fired:
                break

        if not step_fired:
            break

    # Report progress even if goal not fully closed
    if atoms_fired > 0:
        _close_goal(goal, goal_idx, f"rw_seq({atoms_fired})", [current_goal], state, provenance)
        return True
    return False


# Backward compat wrapper
def _try_cosine_rw(
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
    accessible_theorem_id: int | None = None,
    beam_width: int = 1,
) -> bool:
    """Try rw via cosine-ranked premises. Delegates to _try_cosine_family."""
    return _try_cosine_family("rw", goal, goal_idx, state, env, accessible_theorem_id, beam_width)


# Structural fallback tactics split into two sub-lanes:
#
# structural_core: premise-free logical scaffolding (intro, assumption, etc.)
#   These produce new goals but don't close proofs alone.
# solver_bootstrap: automated closers (solve_by_elim, simp, decide, aesop)
#   These can close goals outright via exhaustive search.
#
# The split matters for attribution: a proof that needs intro→exact? is
# "structural_core setup + solver_bootstrap close", not "learned navigation."

_SOLVER_BOOTSTRAP = [
    "simp",  # simplification (fast)
    "simp_all",  # simp using all hypotheses (moderate)
    "decide",  # decidable propositions (fast)
    "omega",  # arithmetic (fast)
    "linarith",  # linear arithmetic over ordered fields (fast)
    "positivity",  # positivity/nonnegativity goals (fast)
    "field_simp",  # clear denominators before ring/norm_num (moderate)
    "push_cast",  # normalize casts (fast)
    "ext",  # extensionality (structural)
    "funext",  # function extensionality (structural)
    "congr",  # congruence (structural)
    "solve_by_elim",  # symbolic local proof search (moderate)
    "aesop",  # general automation (moderate)
]

# Expensive tactics tried only once per goal (not every search iteration).
# apply?/exact? are powerful but expensive teacher/oracle calls.
_EXPENSIVE_CLOSERS: list[str] = [
    "apply?",  # Lean's symbolic theorem-application search
    "exact?",  # Lean's own search — teacher/oracle for argument-aware closure
]

_NUMERIC_SOLVER_TACTICS = {
    "norm_num",
    "ring",
    "omega",
    "linarith",
    "nlinarith",
    "positivity",
    "field_simp",
    "push_cast",
}

_META_WRAPPER_TACTICS = [
    "dsimp only [autoParam, optParam]",
    "simpa only [autoParam, optParam]",
]


def _goal_domain_hints(theorem_id: str, goal: str) -> set[str]:
    source = " ".join([_sanitize_goal_text(theorem_id or ""), _sanitize_goal_text(goal or "")])
    hints: set[str] = set()
    checks = {
        "category_theory": ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"],
        "algebraic_geometry": ["AlgebraicGeometry", "Scheme", "LocallyRingedSpace", "PrimeSpectrum", "HomogeneousLocalization"],
        "abstract_algebra": ["IsIntegral", "traceMatrix", "Matrix.det", "discr", "FormallyUnramified", "HasRingHomProperty"],
        "cardinal": ["Cardinal", "#", "countable", "mk_", "aleph"],
        "geometric_analysis": ["Besicovitch", "dist", "Metric", "δ", "norm", "ball"],
        "membership_wall": [".carrier", "Submodule", "Ideal", "PrimeSpectrum", "HomogeneousLocalization"],
        "structural_property": ["IsOpenMap", "HasRingHomProperty", "FormallyUnramified", "IsIso", "essImage"],
    }
    for label, markers in checks.items():
        if label == "membership_wall":
            if "∈" in source and any(marker in source for marker in markers):
                hints.add(label)
            continue
        if any(marker in source for marker in markers):
            hints.add(label)
    return hints


def _suppress_numeric_solver_tactics(theorem_id: str, goal: str) -> bool:
    hints = _goal_domain_hints(theorem_id, goal)
    return bool({"category_theory", "algebraic_geometry", "abstract_algebra", "membership_wall"} & hints)


def _apply_domain_lane_policy(theorem_id: str, goal: str, lane_order: list[str]) -> list[str]:
    ordered = list(dict.fromkeys(lane_order))
    hints = _goal_domain_hints(theorem_id, goal)
    target = _goal_target_text(goal)

    if {"category_theory", "algebraic_geometry", "abstract_algebra", "structural_property"} & hints:
        if target.startswith("∀") or target.startswith("∃") or "↔" in target:
            promoted = ["interleaved_bootstrap", "structural_core", "cosine_exact", "cosine_rw", "learned"]
        else:
            promoted = ["cosine_exact", "interleaved_bootstrap", "structural_core", "cosine_rw", "learned"]
        ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]

    if "membership_wall" in hints:
        promoted = ["cosine_exact", "interleaved_bootstrap", "structural_core", "cosine_rw"]
        ordered = [lane for lane in promoted if lane in ordered] + [lane for lane in ordered if lane not in promoted]

    if "automation" in ordered and ({"category_theory", "algebraic_geometry", "abstract_algebra"} & hints):
        ordered = [lane for lane in ordered if lane != "automation"] + ["automation"]

    return ordered


def _try_structural_fallback(
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
) -> bool:
    """Try structural core tactics, then solver bootstrap as fallback.

    All structural/solver tactics are tried ONCE per goal (tracked via
    _expensive_tried set). This prevents the search from retrying the
    same failing tactics on the same goal across iterations.

    Provenance: structural_core (intros/assumption/etc) vs solver_bootstrap
    (simp/omega/decide/solve_by_elim/aesop/apply?/exact?).
    """
    # One-shot guard: only try structural fallback once per goal
    if goal in state._structural_tried:
        return False
    state._structural_tried.add(goal)
    suppress_numeric = _suppress_numeric_solver_tactics(state.theorem_id, goal)
    if "autoParam" in goal or "optParam" in goal:
        for tactic in _META_WRAPPER_TACTICS:
            result = env.lean.try_tactic(goal, tactic)
            state.attempts += 1
            if result.success:
                _close_goal(goal, goal_idx, tactic, result.new_goals, state, "structural_core")
                return True

    # Phase 1: intros (greedy — introduces ALL binders at once)
    result = env.lean.try_tactic(goal, "intros")
    state.attempts += 1
    if result.success and result.new_goals:
        _close_goal(goal, goal_idx, "intros", result.new_goals, state, "structural_core")
        return True

    # Phase 1.5: Goal-shape-conditioned tactics
    # These address specific structural patterns that the generic closers miss.
    shape_tactics: list[str] = []
    target = goal.split("⊢")[-1].strip() if "⊢" in goal else goal.strip()
    if target.startswith("∃") or target.startswith("Exists"):
        shape_tactics.extend(["use ?_", "exact ⟨_, _⟩"])
    if "∨" in target:
        shape_tactics.extend(["left", "right"])
    # Induction: try on specific variables from the goal context
    _INDUCTION_TYPES = {"Nat", "List", "Finset", "Multiset", "Bool", "Option"}
    for line in goal.split("\n"):
        if ":" in line and "⊢" not in line:
            parts = line.split(":")
            var_name = parts[0].strip()
            var_type = parts[1].strip() if len(parts) > 1 else ""
            if var_name and any(t in var_type for t in _INDUCTION_TYPES):
                shape_tactics.append(f"induction {var_name}")
                break  # Only try one induction variable
    if "↔" in target:
        shape_tactics.append("constructor")
    for tactic in shape_tactics:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "structural_core")
            return True

    # Phase 2: Non-intro structural closers
    structural_closers = [
        "assumption", "constructor", "rfl", "trivial",
        "exact le_top", "exact bot_le", "exact le_refl _",
        "exact Subsingleton.elim _ _", "infer_instance",
        "norm_num", "ring",
    ]
    if suppress_numeric:
        structural_closers = [tactic for tactic in structural_closers if tactic not in _NUMERIC_SOLVER_TACTICS]
    for tactic in structural_closers:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "structural_core")
            return True

    # Phase 3: Solver bootstrap (fast automated closers)
    solver_bootstrap = [
        tactic for tactic in _SOLVER_BOOTSTRAP
        if not (suppress_numeric and tactic in _NUMERIC_SOLVER_TACTICS)
    ]
    for tactic in solver_bootstrap:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "solver_bootstrap")
            return True

    # Phase 4: Expensive closers (apply?/exact? — Lean symbolic search)
    for tactic in _EXPENSIVE_CLOSERS:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "solver_bootstrap")
            return True

    return False


def _try_interleaved_bootstrap(
    goal: str,
    goal_idx: int,
    state: _SearchState,
    env: _SearchEnv,
    max_depth: int = 4,
    max_calls: int = 20,
) -> bool:
    """Interleaved structural bootstrap: intros → simp → aesop, depth-bounded.

    Unlike the one-shot structural fallback, this lane runs a tight loop:

      For each intro round (up to max_depth):
        1. Try `intros` — if it opens subgoals, recurse into each
        2. On each resulting subgoal, try: simp, then immediately aesop
        3. Also try omega / decide for arithmetic/decidable residuals
        4. Stop on no progress or budget exhausted

    Provenance: "interleaved_bootstrap" for all closes from this lane.
    Simp-then-aesop closes are tagged "interleaved_bootstrap/simp_aesop".
    Budget: max_calls across all tactics fired in this activation.

    Returns True if any progress was made (even partial goal close).
    """
    if goal in state._interleaved_tried:
        return False
    state._interleaved_tried.add(goal)
    suppress_numeric = _suppress_numeric_solver_tactics(state.theorem_id, goal)

    calls = [0]  # mutable counter via list

    def _try(tac: str, g: str) -> TacticResult | None:
        if calls[0] >= max_calls:
            return None
        calls[0] += 1
        state.attempts += 1
        try:
            return env.lean.try_tactic(g, tac)
        except Exception:
            return None

    def _close(g: str, g_idx: int, tac: str, new_goals: list, prov: str) -> None:
        _close_goal(g, g_idx, tac, new_goals, state, prov)

    def _bootstrap_one(g: str, g_idx: int, depth: int) -> bool:
        """Try to close/advance one goal. Returns True on any progress."""
        if calls[0] >= max_calls or depth > max_depth:
            return False

        if "autoParam" in g or "optParam" in g:
            for tac in _META_WRAPPER_TACTICS:
                r_meta = _try(tac, g)
                if r_meta is not None and r_meta.success:
                    _close(g, g_idx, tac, r_meta.new_goals, "interleaved_bootstrap")
                    return True

        # --- Tier 1: cheap direct closers ---
        tier1 = ["rfl", "trivial", "assumption", "decide", "omega", "norm_num"]
        if suppress_numeric:
            tier1 = [tac for tac in tier1 if tac not in _NUMERIC_SOLVER_TACTICS]
        for tac in tier1:
            r = _try(tac, g)
            if r is not None and r.success:
                _close(g, g_idx, tac, r.new_goals, "interleaved_bootstrap")
                return True

        # --- Tier 2: simp, then immediately symbolic search if simp accepted ---
        r_simp = _try("simp", g)
        if r_simp is not None and r_simp.success:
            if not r_simp.new_goals:
                # simp closed the goal outright
                _close(g, g_idx, "simp", [], "interleaved_bootstrap/simp")
                return True
            # simp made progress — try symbolic closers on each residual
            simp_residuals = list(r_simp.new_goals)
            all_closed = True
            for res_goal in simp_residuals:
                res_str = res_goal if isinstance(res_goal, str) else str(res_goal)
                for symbolic_tac in ("solve_by_elim", "aesop"):
                    r_symbolic = _try(symbolic_tac, res_str)
                    if r_symbolic is not None and r_symbolic.success and not r_symbolic.new_goals:
                        break
                else:
                    r_symbolic = None
                if r_symbolic is not None and r_symbolic.success and not r_symbolic.new_goals:
                    continue  # symbolic search closed this residual
                # Try omega/decide as symbolic-search alternatives
                closed_residual = False
                fallback_tactics = ["omega", "decide", "norm_num"]
                if suppress_numeric:
                    fallback_tactics = [tac for tac in fallback_tactics if tac not in _NUMERIC_SOLVER_TACTICS]
                for fallback in fallback_tactics:
                    r_fb = _try(fallback, res_str)
                    if r_fb is not None and r_fb.success and not r_fb.new_goals:
                        closed_residual = True
                        break
                if not closed_residual:
                    all_closed = False
                    break
            if all_closed:
                # Record as simp→symbolic-search chain close on the original goal
                _close(g, g_idx, "simp_then_symbolic", [], "interleaved_bootstrap/simp_aesop")
                return True
            # simp made progress but didn't fully close — record partial advance
            _close(g, g_idx, "simp", simp_residuals, "interleaved_bootstrap/simp")
            return True

        # --- Tier 3: symbolic search alone ---
        for symbolic_tac in ("solve_by_elim", "aesop"):
            r_symbolic = _try(symbolic_tac, g)
            if r_symbolic is not None and r_symbolic.success:
                _close(g, g_idx, symbolic_tac, r_symbolic.new_goals, "interleaved_bootstrap")
                return True

        # --- Tier 4: intros, then recurse on subgoals ---
        r_intros = _try("intros", g)
        if r_intros is not None and r_intros.success and r_intros.new_goals:
            _close(g, g_idx, "intros", r_intros.new_goals, "interleaved_bootstrap")
            # Eagerly try to close each freshly introduced subgoal
            made_progress = True
            # New subgoals were appended to state.open_goals by _close_goal.
            # Recurse on them by their new indices (they are at the end).
            n_new = len(r_intros.new_goals)
            base_idx = len(state.open_goals) - n_new
            for sub_i in range(n_new):
                idx = base_idx + sub_i
                if idx < len(state.open_goals):
                    sub_goal = state.open_goals[idx]
                    _bootstrap_one(sub_goal, idx, depth + 1)
            return made_progress

        return False

    return _bootstrap_one(goal, goal_idx, depth=0)


@dataclass
class StepOutcome:
    """Result of a single search step — clean attribution for temporal controller."""

    attempted_goal: str
    progress: bool
    closing_lane: str = ""
    closing_family: str = ""
    closing_tactic: str = ""


def _success_outcome(goal: str, state: _SearchState) -> StepOutcome:
    """Build a success StepOutcome from the last closed goal's state."""
    return StepOutcome(
        attempted_goal=goal,
        progress=True,
        closing_lane=state.close_provenance[-1] if state.close_provenance else "",
        closing_family=state.tactics_used[-1].split()[0] if state.tactics_used else "",
        closing_tactic=state.tactics_used[-1] if state.tactics_used else "",
    )


def _trigger_gate(goal: str, state: _SearchState, env: _SearchEnv, _cfg: SearchConfig) -> bool:
    """Learned trigger gate for the cosine_apply lane.

    Uses the trained trigger classifier (EXP-051) to decide whether the apply
    specialist should fire at this search state. Falls back to _apply_gate
    if no trigger model is loaded.
    """
    if env.apply_trigger_trunk is None or env.apply_trigger_head is None:
        return _apply_gate(goal, state)
    if env.sentence_encoder is None:
        return _apply_gate(goal, state)

    try:
        import re

        import numpy as np
        import torch

        # Stage classification (same logic as collect_trigger_states._classify_search_stage)
        recent_lanes = state.close_provenance[-5:] if state.close_provenance else []

        stage = "mid_search"  # default
        if recent_lanes:
            last = recent_lanes[-1]
            if last in ("interleaved_bootstrap", "interleaved_bootstrap/simp",
                        "interleaved_bootstrap/simp_aesop"):
                stage = "post_ib_fail"
            elif last in ("cosine_rw", "cosine_rw_seq"):
                stage = "post_rw"
            elif last in ("automation",):
                stage = "post_auto"

        # Build feature vector (must match train_trigger_classifier.py exactly)
        _STAGES = ["post_ib_fail", "post_rw", "post_auto", "mid_search"]
        stage_vec = [1.0 if s == stage else 0.0 for s in _STAGES]

        # Goal embedding (cached)
        _trigger_goal_key = _sanitize_goal_text(goal)
        _cached_trigger = env._goal_emb_cache.get(_trigger_goal_key)
        if _cached_trigger is not None:
            goal_emb = _cached_trigger[0] if _cached_trigger.ndim > 1 else _cached_trigger
        else:
            _raw = env.sentence_encoder.encode(
                [_trigger_goal_key], normalize_embeddings=True, show_progress_bar=False
            )
            env._goal_emb_cache[_trigger_goal_key] = _raw
            goal_emb = _raw[0]

        # Goal shape features
        _BINDER_RE = re.compile(r"[({⦃][^:(){}⦃⦄]*:\s*[^(){}⦃⦄]+[)}⦄]")
        _TC_RE = re.compile(r"\[[^\]]+]")
        shape_norms = {
            "char_len": 500.0, "token_len": 100.0, "binder_count": 10.0,
            "forall_count": 5.0, "exists_count": 3.0, "arrow_count": 5.0,
            "iff_count": 3.0, "eq_count": 5.0, "neq_count": 3.0,
            "and_count": 3.0, "or_count": 3.0, "not_count": 3.0,
            "typeclass_count": 10.0, "type_count": 3.0,
        }
        shape_raw = {
            "char_len": len(goal), "token_len": len(goal.split()),
            "binder_count": len(_BINDER_RE.findall(goal)),
            "forall_count": goal.count("∀"), "exists_count": goal.count("∃"),
            "arrow_count": goal.count("→") + goal.count("->"),
            "iff_count": goal.count("↔"), "eq_count": goal.count("="),
            "neq_count": goal.count("≠"), "and_count": goal.count("∧"),
            "or_count": goal.count("∨"), "not_count": goal.count("¬"),
            "typeclass_count": len(_TC_RE.findall(goal)),
            "type_count": goal.count("Type") + goal.count("Sort"),
        }
        shape_keys = list(shape_norms.keys())
        shape_feat = [float(shape_raw.get(k, 0)) / shape_norms[k] for k in shape_keys]

        # Namespace hash (8 buckets) — theorem_id not available at runtime, zero-fill
        ns_vec = [0.0] * 8

        feat = np.concatenate([
            goal_emb,                                                        # 384d
            stage_vec,                                                       # 4d
            [float(len(state.open_goals)) / 10.0],                           # 1d
            [min(len(recent_lanes), 5) / 5.0],                               # 1d
            shape_feat,                                                      # 14d
            ns_vec,                                                          # 8d
            [float(state.attempts) / 50.0],                                  # 1d
            [1.0],                                                           # 1d lane_provenance
        ]).astype(np.float32)

        x = torch.tensor(feat).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(
                env.apply_trigger_head(env.apply_trigger_trunk(x))
            ).item()

        fired = prob > env.apply_trigger_threshold
        state._last_trigger_prob = prob
        state._last_trigger_fired = fired
        if fired:
            state._trigger_fire_count += 1
        else:
            state._trigger_reject_count += 1
        return fired

    except Exception:
        state._last_trigger_prob = -1.0
        state._last_trigger_fired = False
        return _apply_gate(goal, state)


def _apply_gate(goal: str, state: _SearchState) -> bool:
    """Cheap deterministic gate for the cosine_apply lane.

    Passes (returns True) when all of:
      1. Interleaved bootstrap has already run and failed on this goal.
      2. Exactly one open goal remains (avoid cross-goal contamination).
      3. Goal is not automation-shaped (pure arithmetic/decidable).
      4. Goal is not rw/simp-shaped (symmetric equation or simp-normal predicate).
      5. Goal head suggests a named-predicate conclusion (apply target).
    """
    import re  # local — avoid module-level cost

    # Gate 1: IB must have already been tried (and failed)
    if goal not in state._interleaved_tried:
        return False
    # Gate 2: single open goal
    if len(state.open_goals) != 1:
        return False
    # Extract the ⊢ target from goal text
    target = ""
    for line in goal.split("\n"):
        if "⊢" in line:
            target = line.split("⊢", 1)[1].strip()
            break
    if not target:
        return False
    t = target.lower()
    # Gate 3: not automation-shaped (omega/norm_num/decide territory)
    _AUTO_HEADS = {"true", "false", "decide"}
    if any(t.startswith(h) for h in _AUTO_HEADS):
        return False
    # Pure arithmetic: both sides look numeric or variable + arithmetic op
    if re.search(r"\b\d+\b", t) and re.search(r"[+\-\*/%]", t):
        return False
    # Gate 4: not obviously rw/simp-shaped
    # Symmetric equality between two similar-length expressions suggests rw
    if " = " in t and "tendsto" not in t and "continuous" not in t:
        lhs, _, rhs = t.partition(" = ")
        # Both sides are simple identifiers → simp/rfl territory
        if re.match(r"^[\w.' ]+$", lhs.strip()) and re.match(r"^[\w.' ]+$", rhs.strip()):
            return False
    # Gate 5: goal head is a named predicate (apply-shaped)
    # Exclude pure-logic connectives — those are handled by intro/constructor
    _LOGIC_HEADS = {"true", "false", "and", "or", "iff", "exists", "not"}
    m = re.match(r"\(?(\w[\w.]*)", target)
    if m:
        head = m.group(1).lower()
        if head in _LOGIC_HEADS:
            return False
    return True


def _try_lane(
    lane: str,
    goal: str,
    goal_idx: int,
    nav_output: NavOutput,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    cfg: SearchConfig,
) -> bool:
    """Try a single lane. Returns True if the goal was closed/progressed."""
    if lane == "automation":
        if cfg.hammer_delegation and _should_hammer(nav_output):
            return _try_hammer(
                goal, goal_idx, nav_output, state, env, context, cfg.no_learned_premises
            )
    elif lane == "structural_core":
        return _try_structural_fallback(goal, goal_idx, state, env)
    elif lane in ("solver_bootstrap", "structural"):
        return _try_structural_fallback(goal, goal_idx, state, env)
    elif lane == "cosine_rw":
        return _try_cosine_family(
            "rw", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_rw_beam
        )
    elif lane == "cosine_rw_seq":
        return _try_cosine_rw_seq(
            goal,
            goal_idx,
            state,
            env,
            context.accessible_theorem_id,
            cfg.cosine_rw_beam,
            cfg.cosine_rw_seq_max_atoms,
            cfg.cosine_rw_seq_max_calls,
        )
    elif lane == "cosine_exact":
        is_post_apply = goal in state._post_apply_goals
        ranked = _cosine_rank_premises(
            goal, env, context.accessible_theorem_id,
            family="exact", subgoal_scoped=is_post_apply,
        )
        if not ranked:
            return False
        cache_key = f"cosine_exact:{goal}"
        if cache_key in state._expensive_tried:
            return False
        state._expensive_tried.add(cache_key)
        for sym in ranked[:cfg.cosine_rw_beam]:
            tac = f"exact {sym}"
            result = env.lean.try_tactic(goal, tac)
            state.attempts += 1
            if result.success:
                _close_goal(goal, goal_idx, tac, result.new_goals, state, "cosine_exact")
                return True
        return False
    elif lane == "cosine_apply":
        # Check one-shot cache BEFORE running the trigger MLP
        cache_key = f"cosine_apply:{goal}"
        if cache_key in state._expensive_tried:
            return False
        if cfg.cosine_apply_gated and not _trigger_gate(goal, state, env, cfg):
            return False
        state._apply_attempt_count += 1

        from src.proof_network import get_accessible_premises

        accessible_count = 0
        if context.accessible_theorem_id is not None:
            premise_ids = get_accessible_premises(env.conn, context.accessible_theorem_id)
            accessible_count = len(premise_ids)
        state._last_apply_diag = {
            "accessible_count": accessible_count,
            "accessible_theorem_id": context.accessible_theorem_id,
        }

        goals_before = len(state.open_goals)
        result = _try_cosine_family(
            "apply",
            goal,
            goal_idx,
            state,
            env,
            context.accessible_theorem_id,
            cfg.cosine_apply_beam,
        )
        if result:
            state._apply_accept_count += 1
            if len(state.open_goals) < goals_before:
                state._apply_goal_close_count += 1
        return result
    elif lane == "cosine_simp":
        return _try_cosine_family(
            "simp", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_rw_beam
        )
    elif lane == "interleaved_bootstrap":
        return _try_interleaved_bootstrap(
            goal,
            goal_idx,
            state,
            env,
            max_depth=cfg.interleaved_bootstrap_max_depth,
            max_calls=cfg.interleaved_bootstrap_max_calls,
        )
    elif lane == "learned":
        return _try_candidates(goal, goal_idx, nav_output, state, env, context)
    elif lane == "dr_ducky":
        if not cfg.dr_ducky_enabled:
            return False
        return _try_dr_ducky(goal, goal_idx, state, env, context, cfg)
    return False


# Default lane order when temporal controller is off
# ---------------------------------------------------------------------------
# Family classifier → lane reordering
# ---------------------------------------------------------------------------

_SPECIALIST_NAMES = ["rewrite", "structural", "solver", "apply", "closer"]

# Map specialist family → which lanes to promote to the front
_FAMILY_LANE_PRIORITY: dict[str, list[str]] = {
    "rewrite": ["cosine_rw", "cosine_rw_seq"],
    "structural": ["interleaved_bootstrap", "structural_core"],
    "solver": ["interleaved_bootstrap"],  # solver tactics are in structural fallback
    "apply": ["cosine_apply", "cosine_exact"],
    "closer": ["cosine_exact", "interleaved_bootstrap"],
}


class _TorchFamilyClassifier:
    """Thin runtime wrapper around the PyTorch SoM family router."""

    def __init__(self, model: Any, specialist_names: list[str]) -> None:
        self.model = model
        self.specialist_names = specialist_names

    def predict(
        self,
        goal_emb: np.ndarray,
        goal_shape: np.ndarray,
        step_context: np.ndarray,
    ) -> str:
        import torch

        with torch.inference_mode():
            trust_weights, _ = self.model(
                torch.from_numpy(goal_emb),
                torch.from_numpy(goal_shape),
                torch.from_numpy(step_context),
            )
        best_idx = int(trust_weights[0].argmax().item())
        return self.specialist_names[best_idx]


def _load_exec_apply_selector_cached(selector_path: str) -> tuple[Any, Any] | None:
    if not selector_path:
        return None
    cached = _EXEC_SELECTOR_CACHE.get(selector_path)
    if cached is not None or selector_path in _EXEC_SELECTOR_CACHE:
        return cached
    try:
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer

        class _ExecSelector(nn.Module):
            def __init__(self, in_dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, 1),
                )

            def score(self, x: "torch.Tensor") -> "torch.Tensor":
                return torch.sigmoid(self.net(x).squeeze(-1))

        ckpt = torch.load(selector_path, map_location="cpu", weights_only=False)
        emb_dim = ckpt.get("emb_dim", 384)
        compat_dim = ckpt.get("compat_dim", 0)
        in_dim = emb_dim * 2 + 2 + compat_dim
        selector = _ExecSelector(in_dim, ckpt.get("hidden", 256))
        selector.load_state_dict(ckpt["model_state_dict"])
        selector.eval()
        encoder = SentenceTransformer(ckpt.get("encoder", "all-MiniLM-L6-v2"))
        loaded = (selector, encoder)
        _EXEC_SELECTOR_CACHE[selector_path] = loaded
        return loaded
    except Exception as exc:
        _LOGGER.warning("ExecSelector load failed: %s", exc)
        _EXEC_SELECTOR_CACHE[selector_path] = None
        return None


def _load_family_classifier_torch_cached(path: str) -> Any | None:
    if not path:
        return None
    if path in _FAMILY_TORCH_CACHE:
        return _FAMILY_TORCH_CACHE[path]
    try:
        import torch

        from src.som_torch import SPECIALIST_NAMES as _SOM_SPECIALISTS
        from src.som_torch import SoMConfig as _SoMConfig
        from src.som_torch import SoMModel as _SoMModel

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        som_cfg = _SoMConfig(**ckpt.get("config", {}))
        som_model = _SoMModel(som_cfg)
        som_model.load_state_dict(ckpt["model_state_dict"])
        som_model.eval()
        loaded = _TorchFamilyClassifier(som_model, list(_SOM_SPECIALISTS))
        _LOGGER.info("Loaded PyTorch family classifier from %s", path)
        _FAMILY_TORCH_CACHE[path] = loaded
        return loaded
    except Exception as exc:
        _LOGGER.warning("PyTorch family classifier load failed: %s", exc)
        _FAMILY_TORCH_CACHE[path] = None
        return None


def _load_family_classifier_numpy_cached(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    if path in _FAMILY_NUMPY_CACHE:
        return _FAMILY_NUMPY_CACHE[path]
    try:
        fc_data = np.load(path)
        loaded = {k: fc_data[k] for k in fc_data.files}
        _LOGGER.info("Loaded family classifier from %s", path)
        _FAMILY_NUMPY_CACHE[path] = loaded
        return loaded
    except Exception as exc:
        _LOGGER.warning("Family classifier load failed: %s", exc)
        _FAMILY_NUMPY_CACHE[path] = None
        return None


def _family_classifier_features(
    goal: str,
    env: "_SearchEnv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract shared feature tensors for family routing."""
    if env.sentence_encoder is None:
        return None

    clean_goal = _sanitize_goal_text(goal)
    target = clean_goal.split("⊢")[-1].strip() if "⊢" in clean_goal else clean_goal.strip()
    target = target[:300]

    goal_emb = env.sentence_encoder.encode([target], normalize_embeddings=True)
    shape = np.array([
        float("∀" in target), float("∃" in target),
        float("=" in target and "≤" not in target and "≥" not in target),
        float(any(s in target for s in ["≤", "≥", "<", ">"])),
        float("↔" in target), float("∧" in target),
        float("∨" in target), float("¬" in target),
        float("fun " in target or "λ" in target),
        min(clean_goal.count("\n") / 20.0, 1.0),
        min(len(target) / 500.0, 1.0),
        float(target.split()[0] in ("Eq", "eq", "HEq", "Iff") if target.split() else False),
    ], dtype=np.float32).reshape(1, -1)
    step_ctx = np.array([[0.1, 0.5, min(clean_goal.count("\n") / 20.0, 1.0),
                          min(len(target) / 500.0, 1.0)]], dtype=np.float32)
    return goal_emb.astype(np.float32), shape, step_ctx


def _classify_goal_family(
    goal: str,
    env: "_SearchEnv",
    cfg: "SearchConfig",
) -> str | None:
    """Predict the specialist family for a goal using the family classifier.

    Returns the predicted specialist name, or None if no classifier loaded.
    """
    clf = env.family_classifier
    torch_clf = env.family_classifier_torch
    if clf is None and torch_clf is None:
        return None

    try:
        features = _family_classifier_features(goal, env)
        if features is None:
            return None
        goal_emb, shape, step_ctx = features

        if torch_clf is not None:
            return torch_clf.predict(goal_emb, shape, step_ctx)

        # Forward: X @ W1 + b1 → ReLU → @ W2 + b2 → argmax
        X = np.concatenate([goal_emb, shape, step_ctx], axis=1).astype(np.float32)
        h = np.maximum(0, X @ clf["W1"] + clf["b1"])
        logits = h @ clf["W2"] + clf["b2"]
        pred_idx = int(np.argmax(logits[0]))
        return _SPECIALIST_NAMES[pred_idx]
    except Exception:
        return None


def _reorder_lanes_by_family(
    lane_order: list[str],
    predicted_family: str,
) -> list[str]:
    """Reorder lanes to prioritize the predicted specialist family's lanes."""
    priority_lanes = _FAMILY_LANE_PRIORITY.get(predicted_family, [])
    if not priority_lanes:
        return lane_order

    # Move priority lanes to the front, preserving relative order of the rest
    promoted = [l for l in lane_order if l in priority_lanes]
    rest = [l for l in lane_order if l not in priority_lanes]
    return promoted + rest


_DEFAULT_LANE_ORDER = [
    "automation",
    "structural_core",
    "cosine_rw",
    "cosine_rw_seq",
    "cosine_exact",
    "cosine_apply",
    "cosine_simp",
    "learned",
    "dr_ducky",
]


def _search_step(
    pipeline: Pipeline,
    state: _SearchState,
    env: _SearchEnv,
    context: SearchContext,
    cfg: SearchConfig,
    tc_decision: object | None = None,
) -> StepOutcome:
    """Execute one iteration of the search loop.

    In active temporal mode, uses tc_decision.next_goal_id and
    tc_decision.lane_order. Otherwise uses static policy.
    """
    # Goal selection: active TC or Arbiter overrides _select_goal
    if (cfg.temporal_mode == "active" or cfg.temporal_mode.startswith("arbiter")) and tc_decision is not None:
        tc_goal = getattr(tc_decision, "next_goal_id", None)
        if tc_goal and tc_goal in state.open_goals:
            goal_idx = state.open_goals.index(tc_goal)
            goal = tc_goal
        else:
            goal, goal_idx = _select_goal(state.open_goals, pipeline, cfg.device, state)
    else:
        goal, goal_idx = _select_goal(state.open_goals, pipeline, cfg.device, state)

    nav_output = _cached_infer(goal, pipeline, state)

    # Hammer is always a pre-check when applicable — not subject to lane ordering.
    # The TC controls non-hammer lane priority, not whether hammer fires.
    if cfg.hammer_delegation and _should_hammer(nav_output):
        if _try_lane("automation", goal, goal_idx, nav_output, state, env, context, cfg):
            return _success_outcome(goal, state)

    # Lane ordering: active TC or Arbiter overrides static order (for non-hammer lanes)
    if (cfg.temporal_mode == "active" or cfg.temporal_mode.startswith("arbiter")) and tc_decision is not None:
        lane_order = [
            l for l in getattr(tc_decision, "lane_order", _DEFAULT_LANE_ORDER) if l != "automation"
        ]  # automation already tried above
    else:
        mode = cfg.search_mode
        lane_order = []
        if mode != "learned_only":
            # Interleaved bootstrap is a structural policy replacement, not an
            # extra post-structural lane. If appended after structural_core, the
            # one-shot structural setup consumes the only useful opportunity and
            # the interleaved loop never gets control on the fresh post-intro
            # subgoal states. When enabled, it should therefore occupy the
            # structural slot directly.
            if cfg.interleaved_bootstrap_enabled:
                lane_order.append("interleaved_bootstrap")
            else:
                lane_order.append("structural_core")
        # Cosine family lanes run after structural setup, before learned navigation.
        # cosine_rw: always active when encoder present.
        # cosine_simp/cosine_apply: opt-in via config flags (benchmarked separately).
        if env.sentence_encoder is not None:
            lane_order.append("cosine_rw")
            if cfg.cosine_rw_seq_enabled:
                lane_order.append("cosine_rw_seq")
            lane_order.append("cosine_exact")
            if cfg.cosine_simp_enabled:
                lane_order.append("cosine_simp")
            if cfg.cosine_apply_enabled:
                lane_order.append("cosine_apply")
        if mode != "no_learned":
            lane_order.append("learned")
        if cfg.dr_ducky_enabled:
            lane_order.append("dr_ducky")

    # Family classifier reordering: predict specialist and promote relevant lanes
    predicted = _classify_goal_family(goal, env, cfg)
    if predicted is not None:
        lane_order = _reorder_lanes_by_family(lane_order, predicted)
    lane_order = _apply_domain_lane_policy(state.theorem_id, goal, lane_order)
    state._last_lane_order_executed = list(lane_order)

    # Try non-hammer lanes in order, tracking which were attempted
    lanes_tried: list[str] = []
    for lane in lane_order:
        lanes_tried.append(lane)
        if _try_lane(lane, goal, goal_idx, nav_output, state, env, context, cfg):
            # Record tried lanes on state for trace capture
            state._last_lanes_tried = lanes_tried
            return _success_outcome(goal, state)
    state._last_lanes_tried = lanes_tried

    # Self-application closer: try `exact <theorem_id>` on any single
    # remaining goal. EXP-056 showed 17/17 exact? solutions are
    # self-applications. Lean's unifier handles implicit args.
    if cfg.allow_self_application and len(state.open_goals) == 1 and state.theorem_id:
        self_app_key = f"self_app:{goal}"
        if self_app_key not in state._expensive_tried:
            state._expensive_tried.add(self_app_key)
            for tac in [
                f"exact {state.theorem_id}",
                f"exact @{state.theorem_id}",
            ]:
                result = env.lean.try_tactic(goal, tac)
                state.attempts += 1
                if result.success and not result.new_goals:
                    _close_goal(goal, goal_idx, tac, [], state, "self_application")
                    return _success_outcome(goal, state)

    # Pre-close normalization: try norm_cast/ring_nf to simplify the goal
    # before apply?/exact?. EXP-SOM-009 showed 28/46 2-step closures use
    # norm_cast as setup. These tactics transform the goal without closing it,
    # making it recognizable to the symbolic closers.
    _NORM_SETUPS = ["norm_cast", "ring_nf", "push_neg"]
    if cfg.norm_then_close_enabled and len(state.open_goals) == 1:
        norm_key = f"norm_setup:{goal}"
        if norm_key not in state._expensive_tried:
            state._expensive_tried.add(norm_key)
            for norm_tac in _NORM_SETUPS:
                result = env.lean.try_tactic(goal, norm_tac)
                state.attempts += 1
                if result.success and result.new_goals and len(result.new_goals) == 1:
                    new_goal = result.new_goals[0]
                    if new_goal != goal:
                        # Goal changed — try symbolic closers on the normalized form
                        for closer in _EXPENSIVE_CLOSERS:
                            closer_result = env.lean.try_tactic(new_goal, closer)
                            state.attempts += 1
                            if closer_result.success and not closer_result.new_goals:
                                tactic_desc = f"{norm_tac}; {closer}"
                                _close_goal(goal, goal_idx, tactic_desc, [], state, "norm_then_close")
                                return _success_outcome(goal, state)

    # Last-resort closer: Lean's symbolic search teachers/oracles.
    if len(state.open_goals) == 1 and _EXPENSIVE_CLOSERS:
        cache_key = f"last_resort:{goal}"
        if cache_key not in state._expensive_tried:
            state._expensive_tried.add(cache_key)
            for tac in _EXPENSIVE_CLOSERS:
                result = env.lean.try_tactic(goal, tac)
                state.attempts += 1
                if result.success and not result.new_goals:
                    # Try to capture the suggestion from Pantograph messages
                    suggestion = tac
                    try:
                        gs = env.lean._goal_states.get(
                            (env.lean._current_env_key, goal)
                        )
                        if gs is not None:
                            from pantograph.expr import Site

                            new_gs = env.lean._server.goal_tactic(
                                gs, tac, site=Site(goal_id=0)
                            )
                            for msg in getattr(new_gs, "messages", []):
                                if "Try this" in getattr(msg, "data", ""):
                                    # Extract: "Try this:\n  exact foo bar"
                                    lines = msg.data.split("\n")
                                    for line in lines:
                                        s = line.strip()
                                        if (
                                            s.startswith("exact ")
                                            or s.startswith("apply ")
                                            or s.startswith("refine ")
                                            or s.startswith("simpa")
                                        ):
                                            suggestion = s
                                            break
                                        # Also handle "[apply] exact ..." format
                                        if "] exact " in s or "] apply " in s or "] refine " in s:
                                            suggestion = s.split("] ", 1)[-1]
                                            break
                    except Exception:
                        pass  # Suggestion capture failed; use raw tactic name
                    if (
                        not cfg.allow_self_application
                        and _is_self_application_tactic(suggestion, state.theorem_id)
                    ):
                        continue
                    _close_goal(goal, goal_idx, suggestion, [], state, "last_resort_exact")
                    return _success_outcome(goal, state)

    # All lanes failed — rotate goal to back of queue
    if len(state.open_goals) > 1:
        state.open_goals.append(state.open_goals.pop(goal_idx))
    state.attempts += 1
    return StepOutcome(attempted_goal=goal, progress=False)


def search(
    theorem_id: str,
    initial_goal: str | list[str],
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    config: SearchConfig | None = None,
    anchor_id_map: dict[str, int] | None = None,
    accessible_theorem_id: int | None = None,
    sentence_encoder: Any | None = None,
    exec_apply_selector: Any | None = None,
    exec_apply_encoder: Any | None = None,
    holographic_premises: list[str] | None = None,
) -> SearchResult:
    """Run proof search on a single theorem.

    Args:
        sentence_encoder: Optional SentenceTransformer for cosine_rw lane.
            When provided, enables the cosine rw beam lane (scope → encode →
            cosine rank → top-5 beam + Lean verify).
        exec_apply_selector: Pre-loaded ExecSelector nn.Module. When provided,
            skips per-call checkpoint loading even if cfg.exec_apply_selector_path
            is set. Load once per process and pass through here.
        exec_apply_encoder: Pre-loaded SentenceTransformer paired with the
            exec_apply_selector. Must be provided together with exec_apply_selector.
    """
    cfg = config or SearchConfig()
    initial_goals = list(initial_goal) if isinstance(initial_goal, list) else [initial_goal]
    context = SearchContext(
        accessible_theorem_id=accessible_theorem_id if cfg.accessible_premises else None,
    )
    state = _SearchState(theorem_id=theorem_id, open_goals=initial_goals)

    # Build entity maps for cosine_rw lane (once per search call)
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    if sentence_encoder is not None:
        rows = conn.execute("SELECT id, name FROM entities").fetchall()
        id_to_name = {eid: name for eid, name in rows}
        name_to_id = {name: eid for eid, name in rows}

    # Load ExecSelector for apply lane (optional).
    # Pre-loaded objects (passed as arguments) take priority — no per-call cost.
    # Only fall back to checkpoint loading when path is set but nothing was pre-loaded.
    if exec_apply_selector is None and exec_apply_encoder is None and cfg.exec_apply_selector_path:
        loaded = _load_exec_apply_selector_cached(cfg.exec_apply_selector_path)
        if loaded is not None:
            exec_apply_selector, exec_apply_encoder = loaded

    # Load apply trigger classifier (optional)
    apply_trigger_trunk = None
    apply_trigger_head = None
    if cfg.apply_trigger_path and Path(cfg.apply_trigger_path).exists():
        try:
            import torch
            import torch.nn as nn

            ckpt = torch.load(cfg.apply_trigger_path, map_location="cpu", weights_only=False)
            _in_dim = ckpt["in_dim"]
            _hidden = ckpt["hidden"]

            trunk = nn.Sequential(
                nn.Linear(_in_dim, _hidden), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(_hidden, _hidden // 2), nn.ReLU(),
            )
            head = nn.Linear(_hidden // 2, 1)
            trunk.load_state_dict(ckpt["trunk_state_dict"])
            head.load_state_dict(ckpt["head_primary_state_dict"])
            trunk.eval()
            head.eval()
            apply_trigger_trunk = trunk
            apply_trigger_head = head
        except Exception as _e:
            import logging as _log

            _log.getLogger(__name__).warning("Apply trigger load failed: %s", _e)

    # Load family classifier (optional, numpy-based MLP)
    family_clf: dict[str, Any] | None = None
    family_clf_torch = None
    if cfg.family_classifier_torch_path and Path(cfg.family_classifier_torch_path).exists():
        family_clf_torch = _load_family_classifier_torch_cached(cfg.family_classifier_torch_path)
    if cfg.family_classifier_path and Path(cfg.family_classifier_path).exists():
        family_clf = _load_family_classifier_numpy_cached(cfg.family_classifier_path)

    env = _SearchEnv(
        conn=conn,
        lean=lean,
        anchor_id_map=anchor_id_map,
        max_candidates=cfg.max_candidates_per_step,
        sentence_encoder=sentence_encoder,
        id_to_name=id_to_name,
        name_to_id=name_to_id,
        exec_apply_selector=exec_apply_selector,
        exec_apply_encoder=exec_apply_encoder,
        apply_trigger_trunk=apply_trigger_trunk,
        apply_trigger_head=apply_trigger_head,
        apply_trigger_threshold=cfg.apply_trigger_threshold,
        allow_self_application=cfg.allow_self_application,
        family_classifier=family_clf,
        family_classifier_torch=family_clf_torch,
        holographic_premises=list(holographic_premises or []),
    )

    # Populate premise type cache from DB (for v2 selector compat features)
    if exec_apply_selector is not None:
        try:
            type_rows = conn.execute(
                'SELECT name, type_pp FROM entities WHERE type_pp != ""'
            ).fetchall()
            env.premise_type_cache = {name: tpp for name, tpp in type_rows}
        except Exception:
            pass  # Column may not exist in older DBs

    # Temporal controller (shadow or active mode)
    tc_trace: list[dict] = []
    tc_state = None
    tc = None
    if cfg.temporal_mode in ("shadow", "active"):
        from src.temporal_controller import TemporalController, TemporalState

        tc = TemporalController()
        tc_state = TemporalState(
            theorem_id=theorem_id,
            open_goals=list(state.open_goals),
            budget_remaining=cfg.budget,
        )

    # Strategy Arbiter (dominance-aware routing from strategy memory)
    arbiter = None
    if cfg.temporal_mode.startswith("arbiter"):
        from src.strategy_arbiter import StrategyArbiter
        from src.temporal_controller import TemporalState as _TS

        arbiter_mode = cfg.temporal_mode.replace("arbiter_", "") if "_" in cfg.temporal_mode else "full"
        arbiter = StrategyArbiter(
            strategy_memory_path=cfg.strategy_memory_path,
            mode=arbiter_mode,
        )
        tc_state = _TS(
            theorem_id=theorem_id,
            open_goals=list(state.open_goals),
            budget_remaining=cfg.budget,
        )

    stall_count = 0
    max_stall = max(len(state.open_goals) * 2, 3)
    step_trace: list[dict] = []
    progress_steps = 0
    state._goal_signature_history = [_goal_signature(state.open_goals)]

    # NOTE: Pre-search self-application (`exact theorem_id`) was tested and
    # achieves 82.4% but is NOT honest proof search — it just looks up the
    # theorem from Mathlib. Disabled. The last-resort self-application
    # (after search has reduced to 1 goal) is kept as it fires on
    # genuinely reduced goals, not the raw theorem type.

    decision = None
    while state.open_goals and state.attempts < cfg.budget:
        # Strategy Arbiter: produce decision (dominance-aware routing)
        if arbiter is not None and tc_state is not None:
            tc_state.open_goals = list(state.open_goals)
            tc_state.closed_goals = list(state.closed_goals)
            tc_state.budget_remaining = cfg.budget - state.attempts
            tc_state.total_attempts = state.attempts
            decision = arbiter.decide(tc_state)
            if decision is not None:
                tc_trace.append(
                    {
                        "step": state.attempts,
                        "phase": decision.phase,
                        "next_goal": decision.next_goal_id,
                        "lane_order": decision.lane_order,
                        "escalation": decision.escalation_level,
                        "replan": decision.replan,
                        "source": "arbiter",
                    }
                )

        # Temporal controller: produce decision (legacy path)
        elif tc is not None and tc_state is not None:
            tc_state.open_goals = list(state.open_goals)
            tc_state.closed_goals = list(state.closed_goals)
            tc_state.budget_remaining = cfg.budget - state.attempts
            decision = tc.decide(tc_state)
            tc_trace.append(
                {
                    "step": state.attempts,
                    "phase": decision.phase,
                    "next_goal": decision.next_goal_id,
                    "lane_order": decision.lane_order,
                    "escalation": decision.escalation_level,
                    "replan": decision.replan,
                }
            )

        goals_before = list(state.open_goals)
        frontier_snapshot = _snapshot_frontier(state)
        state._last_apply_candidates = []  # reset per step
        state._last_pathology_tags = []
        outcome = _search_step(pipeline, state, env, context, cfg, tc_decision=decision)
        progress_pathologies: list[str] = []
        if outcome.progress:
            progress_pathologies = _progress_pathology_tags(
                goals_before=goals_before,
                goals_after=list(state.open_goals),
                tactic=outcome.closing_tactic,
                recent_signatures=state._goal_signature_history,
                loop_window=cfg.state_loop_window,
            )
            abstract_domain = bool(
                {"category_theory", "algebraic_geometry", "abstract_algebra", "structural_property"}
                & _goal_domain_hints(state.theorem_id, outcome.attempted_goal or (goals_before[0] if goals_before else ""))
            )
            severe_metavar = cfg.metavariable_penalty_enabled and any(
                tag in progress_pathologies
                for tag in {"metavariable_corruption", "backward_rewrite_metavariable", "bare_type_side_goal"}
            )
            severe_loop = cfg.state_loop_penalty_enabled and "state_loop" in progress_pathologies
            severe_duplicate = abstract_domain and "duplicate_goal_pseudo_progress" in progress_pathologies
            if severe_metavar or severe_loop or severe_duplicate:
                attempts_after = state.attempts
                last_lanes_tried = list(state._last_lanes_tried)
                last_apply_diag = dict(state._last_apply_diag)
                last_apply_candidates = list(state._last_apply_candidates)
                _restore_frontier(state, frontier_snapshot)
                state.attempts = attempts_after
                state._last_lanes_tried = last_lanes_tried
                state._last_apply_diag = last_apply_diag
                state._last_apply_candidates = last_apply_candidates
                state._last_pathology_tags = progress_pathologies
                if severe_metavar:
                    state._metavariable_penalty_count += 1
                if severe_loop:
                    state._state_loop_penalty_count += 1
                if severe_duplicate:
                    state._state_loop_penalty_count += 1
                state._replanner_trigger_count += 1
                outcome = StepOutcome(
                    attempted_goal=outcome.attempted_goal,
                    progress=False,
                    closing_lane="replanner_penalty",
                    closing_family="replanner_penalty",
                    closing_tactic=outcome.closing_tactic,
                )

        if cfg.collect_trace:
            trace_entry: dict = {
                "step": state.attempts,
                "goal_before": goals_before[0] if goals_before else "",
                "open_goals_before": goals_before,
                "lane": outcome.closing_lane,
                "tactic": outcome.closing_tactic,
                "progress": outcome.progress,
                "open_goals_after": list(state.open_goals),
                "attempted_goal": outcome.attempted_goal,
                "closing_family": outcome.closing_family,
                "closed_goals_count": len(state.closed_goals),
                "pathology_tags": list(progress_pathologies or state._last_pathology_tags),
                "progress_accepted": bool(outcome.progress),
            }
            if decision is not None:
                trace_entry["phase"] = getattr(decision, "phase", "")
                trace_entry["decision_lane_order"] = getattr(decision, "lane_order", [])
                trace_entry["family_prior"] = getattr(decision, "family_prior", [])
                trace_entry["escalation_level"] = getattr(decision, "escalation_level", 0)
                trace_entry["replan"] = getattr(decision, "replan", False)
            if state._last_lane_order_executed:
                trace_entry["lane_order"] = list(state._last_lane_order_executed)
            # Trigger instrumentation: record last trigger evaluation
            if state._last_trigger_prob >= 0:
                trace_entry["trigger_prob"] = round(state._last_trigger_prob, 4)
                trace_entry["trigger_fired"] = state._last_trigger_fired
                state._last_trigger_prob = -1.0  # reset for next step
            # Apply lane diagnostics
            if state._last_apply_diag:
                trace_entry["apply_diag"] = state._last_apply_diag
                state._last_apply_diag = {}
            if state._last_apply_candidates:
                trace_entry["apply_candidates"] = state._last_apply_candidates
            # Record which lanes were tried (including failures) for censor training
            if state._last_lanes_tried:
                trace_entry["lanes_tried"] = state._last_lanes_tried
                # The successful lane is in outcome.closing_lane; all others failed
                trace_entry["lanes_failed"] = [
                    l for l in state._last_lanes_tried if l != outcome.closing_lane
                ]
            step_trace.append(trace_entry)

        # Update temporal state from StepOutcome (clean attribution)
        if tc is not None and tc_state is not None:
            tc.update(
                tc_state,
                goal=outcome.attempted_goal,
                lane=outcome.closing_lane,
                family=outcome.closing_family,
                tactic=outcome.closing_tactic,
                success=outcome.progress,
            )
            # Maintain explicit state fields
            tc_state.current_goal_id = outcome.attempted_goal
            if decision is not None:
                tc_state.phase = decision.phase
                tc_state.escalation_level = decision.escalation_level

        if outcome.progress:
            progress_steps += 1
            stall_count = 0
            # After apply opens subgoals, give extra budget to work on them
            if outcome.closing_lane in ("cosine_apply", "exec_selector_apply"):
                max_stall = max(len(state.open_goals) * 4, 20)
            else:
                max_stall = max(len(state.open_goals) * 2, 10)
            if cfg.max_progress_steps > 0 and progress_steps >= cfg.max_progress_steps and state.open_goals:
                break
            signature = _goal_signature(state.open_goals)
            if signature:
                state._goal_signature_history.append(signature)
        else:
            stall_count += 1
            if stall_count >= max_stall:
                break

    return SearchResult(
        success=len(state.open_goals) == 0,
        theorem_id=theorem_id,
        tactics_used=state.tactics_used,
        attempts=state.attempts,
        goals_closed=len(state.closed_goals),
        goals_remaining=len(state.open_goals),
        progress_steps=progress_steps,
        final_goals=list(state.open_goals),
        close_provenance=state.close_provenance,
        temporal_trace=tc_trace,
        step_trace=step_trace,
        trigger_fire_count=state._trigger_fire_count,
        trigger_reject_count=state._trigger_reject_count,
        apply_attempt_count=state._apply_attempt_count,
        apply_accept_count=state._apply_accept_count,
        apply_goal_close_count=state._apply_goal_close_count,
        metavariable_penalty_count=state._metavariable_penalty_count,
        state_loop_penalty_count=state._state_loop_penalty_count,
        replanner_trigger_count=state._replanner_trigger_count,
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
