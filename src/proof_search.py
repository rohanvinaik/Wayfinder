"""
Proof search — outer loop managing goal selection, neural inference, and verification.

Coordinates the full pipeline: goal state → encoder → analyzer → bridge →
navigator → resolution → Lean kernel verification. Manages open goals,
proof context, search budget, and hammer delegation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src.bridge import InformationBridge
from src.encoder import GoalEncoder
from src.goal_analyzer import GoalAnalyzer
from src.lean_interface import LeanKernel
from src.nav_contracts import NavOutput, TacticResult
from src.proof_navigator import ProofNavigator
from src.resolution import Candidate, SearchContext, resolve


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


@dataclass
class SearchResult:
    """Result of a proof search attempt."""

    success: bool
    theorem_id: str
    tactics_used: list[str] = field(default_factory=list)
    attempts: int = 0
    goals_closed: int = 0
    goals_remaining: int = 0
    # Per-goal provenance: which lane closed each goal
    close_provenance: list[str] = field(default_factory=list)
    # Temporal controller trace (shadow or active mode)
    temporal_trace: list[dict] = field(default_factory=list)


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
    exec_apply_encoder: Any | None = None   # SentenceTransformer for selector features


@dataclass
class _SearchState:
    """Mutable state for a single proof search run."""

    open_goals: list[str]
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
        result = env.lean.try_tactic(goal, tactic_text)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic_text, result.new_goals, state, "learned")
            return True
    return False


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
) -> list[str] | None:
    """Get cosine-ranked accessible premises for a goal. Returns None on failure."""
    if env.sentence_encoder is None or not env.id_to_name or accessible_theorem_id is None:
        return None

    from src.proof_network import get_accessible_premises

    premise_ids = get_accessible_premises(env.conn, accessible_theorem_id)
    premise_names = [env.id_to_name[pid] for pid in premise_ids if pid in env.id_to_name]
    if not premise_names:
        return None

    # Use rw_scoper for filtering (works for all families — filters by
    # rewrite-pattern names which overlap with useful exact/apply premises)
    from src.rw_scoper import scope_for_rw
    scope = scope_for_rw(goal, premise_names, max_premises=max_premises)
    if not scope.all_symbols:
        return None

    import numpy as np
    goal_emb = env.sentence_encoder.encode([goal], normalize_embeddings=True)
    sym_embs = env.sentence_encoder.encode(scope.all_symbols, normalize_embeddings=True)
    scores = (goal_emb @ sym_embs.T).flatten()
    ranked_indices = np.argsort(-scores).tolist()
    return [scope.all_symbols[i] for i in ranked_indices]


def _exec_selector_top1(
    goal: str,
    cosine_ranked: list[str],
    env: _SearchEnv,
) -> list[str]:
    """Rescore cosine_ranked candidates with ExecSelector; return [top1] by selector score.

    Falls back to cosine top-1 on any failure (import error, shape mismatch, etc.)
    so the apply lane always gets at least one candidate to try.
    """
    try:
        import numpy as np
        import torch

        model = env.exec_apply_selector
        encoder = env.exec_apply_encoder
        if model is None or encoder is None or not cosine_ranked:
            return cosine_ranked[:1]

        emb_dim = 384  # all-MiniLM-L6-v2
        texts = [goal] + cosine_ranked
        embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        goal_emb = embs[0]
        cand_embs = embs[1:]
        cosine_scores = (cand_embs @ goal_emb).tolist()
        # All candidates are from the filtered pool — mark all as passed
        filter_passed = [1.0] * len(cosine_ranked)

        n = len(cosine_ranked)
        X = np.zeros((n, emb_dim * 2 + 2), dtype=np.float32)
        for i in range(n):
            X[i, :emb_dim] = goal_emb
            X[i, emb_dim : 2 * emb_dim] = cand_embs[i]
            X[i, -2] = float(cosine_scores[i])
            X[i, -1] = filter_passed[i]

        with torch.no_grad():
            scores = model.score(torch.from_numpy(X)).cpu().numpy()

        best = int(np.argmax(scores))
        return [cosine_ranked[best]]
    except Exception:
        return cosine_ranked[:1]


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

    ranked = _cosine_rank_premises(goal, env, accessible_theorem_id)
    if not ranked:
        return False

    provenance = f"cosine_{family}"
    top = ranked[:beam_width]

    # ExecSelector reranking for the apply family.
    # When selector is loaded, score the wider candidate pool (exec_apply_selector_pool)
    # and pick the highest-scoring candidate as the single top-1 attempt.
    if family == "apply" and env.exec_apply_selector is not None and env.exec_apply_encoder is not None:
        top = _exec_selector_top1(goal, ranked, env)
        provenance = "exec_selector_apply"

    if family == "rw":
        from src.rw_scoper import infer_direction, extract_goal_target
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
            tac = f"apply {sym}"
            result = env.lean.try_tactic(goal, tac)
            state.attempts += 1
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

        goal_target = extract_goal_target(current_goal)
        top = ranked[:beam_width]
        step_fired = False

        for sym in top:
            d = infer_direction(sym, goal_target)
            dirs = ([f"rw [← {sym}]", f"rw [{sym}]"] if d == "backward"
                    else [f"rw [{sym}]", f"rw [← {sym}]"])
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
# solver_bootstrap: automated closers (exact?, simp, decide, aesop)
#   These can close goals outright via exhaustive search.
#
# The split matters for attribution: a proof that needs intro→exact? is
# "structural_core setup + solver_bootstrap close", not "learned navigation."

_SOLVER_BOOTSTRAP = [
    "simp",  # simplification (fast)
    "decide",  # decidable propositions (fast)
    "omega",  # arithmetic (fast)
    "aesop",  # general automation (moderate)
]

# Expensive tactics tried only once per goal (not every search iteration).
# exact? is powerful but takes 5-10s per call — disable for fast benchmarks,
# enable for production runs via SearchConfig.
_EXPENSIVE_CLOSERS: list[str] = [
    # "exact?",  # disabled by default — too slow for iterative benchmarks
]


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
    (simp/omega/decide/aesop/exact?).
    """
    # One-shot guard: only try structural fallback once per goal
    if goal in state._structural_tried:
        return False
    state._structural_tried.add(goal)

    # Phase 1: intros (greedy — introduces ALL binders at once)
    result = env.lean.try_tactic(goal, "intros")
    state.attempts += 1
    if result.success and result.new_goals:
        _close_goal(goal, goal_idx, "intros", result.new_goals, state, "structural_core")
        return True

    # Phase 2: Non-intro structural closers
    for tactic in ["assumption", "constructor", "rfl", "trivial"]:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "structural_core")
            return True

    # Phase 3: Solver bootstrap (fast automated closers)
    for tactic in _SOLVER_BOOTSTRAP:
        result = env.lean.try_tactic(goal, tactic)
        state.attempts += 1
        if result.success:
            _close_goal(goal, goal_idx, tactic, result.new_goals, state, "solver_bootstrap")
            return True

    # Phase 4: Expensive closers (exact? — full Lean search)
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

        # --- Tier 1: cheap direct closers ---
        for tac in ["rfl", "trivial", "assumption", "decide", "omega", "norm_num"]:
            r = _try(tac, g)
            if r is not None and r.success:
                _close(g, g_idx, tac, r.new_goals, "interleaved_bootstrap")
                return True

        # --- Tier 2: simp, then immediately aesop if simp accepted ---
        r_simp = _try("simp", g)
        if r_simp is not None and r_simp.success:
            if not r_simp.new_goals:
                # simp closed the goal outright
                _close(g, g_idx, "simp", [], "interleaved_bootstrap/simp")
                return True
            # simp made progress — try aesop on each residual
            simp_residuals = list(r_simp.new_goals)
            all_closed = True
            for res_goal in simp_residuals:
                res_str = res_goal if isinstance(res_goal, str) else str(res_goal)
                r_aesop = _try("aesop", res_str)
                if r_aesop is not None and r_aesop.success and not r_aesop.new_goals:
                    continue  # aesop closed this residual
                # Try omega/decide as aesop alternative
                closed_residual = False
                for fallback in ["omega", "decide", "norm_num"]:
                    r_fb = _try(fallback, res_str)
                    if r_fb is not None and r_fb.success and not r_fb.new_goals:
                        closed_residual = True
                        break
                if not closed_residual:
                    all_closed = False
                    break
            if all_closed:
                # Record as simp→aesop chain close on the original goal
                _close(g, g_idx, "simp_then_aesop", [], "interleaved_bootstrap/simp_aesop")
                return True
            # simp made progress but didn't fully close — record partial advance
            _close(g, g_idx, "simp", simp_residuals, "interleaved_bootstrap/simp")
            return True

        # --- Tier 3: aesop alone ---
        r_aesop = _try("aesop", g)
        if r_aesop is not None and r_aesop.success:
            _close(g, g_idx, "aesop", r_aesop.new_goals, "interleaved_bootstrap")
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
        return _try_cosine_family("rw", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_rw_beam)
    elif lane == "cosine_rw_seq":
        return _try_cosine_rw_seq(
            goal, goal_idx, state, env, context.accessible_theorem_id,
            cfg.cosine_rw_beam, cfg.cosine_rw_seq_max_atoms, cfg.cosine_rw_seq_max_calls,
        )
    elif lane == "cosine_exact":
        return _try_cosine_family("exact", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_rw_beam)
    elif lane == "cosine_apply":
        if cfg.cosine_apply_gated and not _apply_gate(goal, state):
            return False
        return _try_cosine_family("apply", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_apply_beam)
    elif lane == "cosine_simp":
        return _try_cosine_family("simp", goal, goal_idx, state, env, context.accessible_theorem_id, cfg.cosine_rw_beam)
    elif lane == "interleaved_bootstrap":
        return _try_interleaved_bootstrap(
            goal, goal_idx, state, env,
            max_depth=cfg.interleaved_bootstrap_max_depth,
            max_calls=cfg.interleaved_bootstrap_max_calls,
        )
    elif lane == "learned":
        return _try_candidates(goal, goal_idx, nav_output, state, env, context)
    return False


# Default lane order when temporal controller is off
_DEFAULT_LANE_ORDER = ["automation", "structural_core", "cosine_rw", "cosine_rw_seq", "cosine_exact", "cosine_apply", "cosine_simp", "learned"]


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
    # Goal selection: active TC overrides _select_goal
    if cfg.temporal_mode == "active" and tc_decision is not None:
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

    # Lane ordering: active TC overrides static order (for non-hammer lanes)
    if cfg.temporal_mode == "active" and tc_decision is not None:
        lane_order = [l for l in getattr(tc_decision, "lane_order", _DEFAULT_LANE_ORDER)
                      if l != "automation"]  # automation already tried above
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
            if cfg.cosine_simp_enabled:
                lane_order.append("cosine_simp")
            if cfg.cosine_apply_enabled:
                lane_order.append("cosine_apply")
        if mode != "no_learned":
            lane_order.append("learned")

    # Try non-hammer lanes in order
    for lane in lane_order:
        if _try_lane(lane, goal, goal_idx, nav_output, state, env, context, cfg):
            return _success_outcome(goal, state)

    # All lanes failed — rotate goal to back of queue
    if len(state.open_goals) > 1:
        state.open_goals.append(state.open_goals.pop(goal_idx))
    state.attempts += 1
    return StepOutcome(attempted_goal=goal, progress=False)


def search(
    theorem_id: str,
    initial_goal: str,
    pipeline: Pipeline,
    conn: sqlite3.Connection,
    lean: LeanKernel,
    config: SearchConfig | None = None,
    anchor_id_map: dict[str, int] | None = None,
    accessible_theorem_id: int | None = None,
    sentence_encoder: Any | None = None,
) -> SearchResult:
    """Run proof search on a single theorem.

    Args:
        sentence_encoder: Optional SentenceTransformer for cosine_rw lane.
            When provided, enables the cosine rw beam lane (scope → encode →
            cosine rank → top-5 beam + Lean verify).
    """
    cfg = config or SearchConfig()
    context = SearchContext(
        accessible_theorem_id=accessible_theorem_id if cfg.accessible_premises else None,
    )
    state = _SearchState(open_goals=[initial_goal])

    # Build entity maps for cosine_rw lane (once per search call)
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    if sentence_encoder is not None:
        rows = conn.execute("SELECT id, name FROM entities").fetchall()
        id_to_name = {eid: name for eid, name in rows}
        name_to_id = {name: eid for eid, name in rows}

    # Load ExecSelector for apply lane (optional)
    exec_apply_selector = None
    exec_apply_encoder = None
    if cfg.exec_apply_selector_path:
        try:
            import torch
            import torch.nn as nn
            from sentence_transformers import SentenceTransformer

            class _ExecSelector(nn.Module):
                def __init__(self, emb_dim: int, hidden: int) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(emb_dim * 2 + 2, hidden), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden, hidden // 2), nn.ReLU(),
                        nn.Linear(hidden // 2, 1),
                    )

                def score(self, x: "torch.Tensor") -> "torch.Tensor":
                    return torch.sigmoid(self.net(x).squeeze(-1))

            ckpt = torch.load(cfg.exec_apply_selector_path, map_location="cpu", weights_only=False)
            sel = _ExecSelector(ckpt.get("emb_dim", 384), ckpt.get("hidden", 256))
            sel.load_state_dict(ckpt["model_state_dict"])
            sel.eval()
            exec_apply_selector = sel
            exec_apply_encoder = SentenceTransformer(ckpt.get("encoder", "all-MiniLM-L6-v2"))
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning("ExecSelector load failed: %s", _e)

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
    )

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

    stall_count = 0
    max_stall = max(len(state.open_goals) * 2, 3)

    decision = None
    while state.open_goals and state.attempts < cfg.budget:
        # Temporal controller: produce decision
        if tc is not None and tc_state is not None:
            tc_state.open_goals = list(state.open_goals)
            tc_state.closed_goals = list(state.closed_goals)
            tc_state.budget_remaining = cfg.budget - state.attempts
            decision = tc.decide(tc_state)
            tc_trace.append({
                "step": state.attempts,
                "phase": decision.phase,
                "next_goal": decision.next_goal_id,
                "lane_order": decision.lane_order,
                "escalation": decision.escalation_level,
                "replan": decision.replan,
            })

        outcome = _search_step(pipeline, state, env, context, cfg, tc_decision=decision)

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
            stall_count = 0
            max_stall = max(len(state.open_goals) * 2, 10)
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
        close_provenance=state.close_provenance,
        temporal_trace=tc_trace,
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
