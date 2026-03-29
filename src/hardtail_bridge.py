from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import replay_residual_state, run_ducky_on_goal
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text
from src.proof_search import Pipeline, SearchConfig, SearchResult, search
from src.second_order_controller import (
    SecondOrderDecision,
    derive_second_order_decision,
    refine_second_pass_decision,
)

logger = logging.getLogger(__name__)

# Module-level cache for entity embeddings — computed once, reused across bridge rows.
_ENTITY_EMB_CACHE: dict[str, tuple[list[str], Any]] = {}


def _get_entity_embeddings(
    conn: sqlite3.Connection,
    sentence_encoder: Any,
    max_entities: int = 20_000,
) -> tuple[list[str], Any]:
    """Return (names, embeddings) for the entity corpus, cached."""
    cache_key = "default"
    if cache_key in _ENTITY_EMB_CACHE:
        return _ENTITY_EMB_CACHE[cache_key]

    has_type_pp = any(
        r[1] == "type_pp" for r in conn.execute("PRAGMA table_info(entities)")
    )
    if has_type_pp:
        rows = conn.execute(
            "SELECT name, type_pp FROM entities WHERE entity_type IN ('theorem', 'lemma') "
            "AND type_pp IS NOT NULL AND type_pp != '' LIMIT ?",
            (max_entities,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, '' FROM entities WHERE entity_type IN ('theorem', 'lemma') LIMIT ?",
            (max_entities,),
        ).fetchall()
    names = [str(r[0]) for r in rows if r[0]]
    if not names:
        return [], None
    # Embed primarily by TYPE SIGNATURE (the mathematical content).
    # The goal text should match against what entities PROVE.
    # Names are structurally generated in Mathlib (BddAbove.image2 encodes
    # "about BddAbove, for image2") so they carry supplementary signal,
    # but the type is the ground truth.
    embed_texts = []
    for name, type_pp in rows:
        if type_pp:
            # Type-primary: "BddAbove s → BddAbove t → BddAbove (image2 f s t)"
            # with name as lightweight context
            embed_texts.append(str(type_pp))
        else:
            # Fallback to name when type unavailable — still structurally
            # meaningful in Mathlib's naming convention
            embed_texts.append(str(name))
    embs = sentence_encoder.encode(embed_texts, normalize_embeddings=True, show_progress_bar=False)
    _ENTITY_EMB_CACHE[cache_key] = (names, embs)
    logger.info("Cached %d entity embeddings for cosine re-scoping", len(names))
    return names, embs


# ---------------------------------------------------------------------------
# Holographic elimination closer
#
# Core principle from OTP/HDC theory: it is easier to confidently eliminate
# impossibilities than to directly identify solutions.  Cosine similarity
# against the entity corpus produces a structural signature — which tactic
# families are active in the matched region, and which are informationally
# zero.  Zero families are suppressed entirely; active families get a small
# targeted program set.  When a program progresses, the new goal is
# immediately re-embedded and the process recurses (chained closure).
# ---------------------------------------------------------------------------

# Tactic families mapped to the full Lean 4 tactic universe.
# Each family maps to specific tactics that Dr. Ducky selects based on
# holographic bank activation.  Zero-activity families are suppressed.
_FAMILY_GENERATORS: dict[str, str] = {
    "exact_apply": "Direct application (exact/apply/refine/convert)",
    "rewrite": "Rewrite (rw/rw←/simp_rw/conv/erw)",
    "simp_premise": "Simplify (simp/simp only/simp_all/simpa/dsimp)",
    "arithmetic": "Arithmetic (ring/omega/linarith/nlinarith/norm_num/positivity/polyrith/field_simp)",
    "structural": "Structural (ext/funext/congr/gcongr/constructor/cases/induction/rcases)",
    "witness": "Witness (use/refine⟨⟩/obtain/choose/constructor)",
    "cast_norm": "Cast normalization (norm_cast/push_cast/push_neg/norm_num)",
    "search": "Proof search (apply?/exact?/aesop/solve_by_elim/tauto/decide)",
}


def _profile_guided_tactics(
    premises: list[str],
    hyp_names: list[str],
    profile: dict[str, float],
    goal_text: str,
    threshold: float = 0.3,
) -> list[str]:
    """Generate a targeted tactic set from the FULL Lean universe,
    guided by holographic bank activation.

    Only active families contribute tactics.  Within each family,
    tactics are ordered by specificity: premise-specific first,
    then generic.  The result is a focused set that Dr. Ducky
    tries on the live Lean GoalState.
    """
    tactics: list[str] = []

    # === GUIDED PROOF SEARCH (FIRST — fast because holographic premises constrain it) ===
    if premises:
        using_list = ", ".join(premises[:8])
        tactics.append(f"exact? using [{using_list}]")
        tactics.append(f"apply? using [{using_list}]")

    # === DIRECT APPLICATION with premises ===
    if profile.get("exact_apply", 0) >= threshold:
        for p in premises[:8]:
            tactics.append(f"exact {p}")
            tactics.append(f"apply {p}")
            tactics.append(f"refine {p} ?_")
            tactics.append(f"refine {p} ?_ ?_")
            # convert: fuzzy apply when types almost match
            tactics.append(f"convert {p}")
            tactics.append(f"convert {p} using 1")
            tactics.append(f"convert {p} <;> simp")
            tactics.append(f"convert {p} <;> ring")
            # simpa: simplify then close with premise
            tactics.append(f"simpa using {p}")

    # === REWRITING ===
    if profile.get("rewrite", 0) >= threshold:
        for p in premises[:8]:
            tactics.append(f"rw [{p}]")
            tactics.append(f"rw [← {p}]")
            tactics.append(f"simp_rw [{p}]")
            tactics.append(f"erw [{p}]")
        # Multi-premise rewrites
        if len(premises) >= 2:
            for i in range(min(len(premises), 4)):
                for j in range(i + 1, min(len(premises), 5)):
                    tactics.append(f"rw [{premises[i]}, {premises[j]}]")
                    tactics.append(f"simp [{premises[i]}, {premises[j]}]")

    # === SIMPLIFICATION ===
    if profile.get("simp_premise", 0) >= threshold:
        for p in premises[:6]:
            tactics.append(f"simp [{p}]")
            tactics.append(f"simp only [{p}]")
            tactics.append(f"simp_all [{p}]")
        tactics.extend(["simp", "simp_all", "dsimp"])

    # === ARITHMETIC ===
    if profile.get("arithmetic", 0) >= threshold:
        tactics.extend([
            "ring", "ring_nf", "omega", "linarith", "nlinarith",
            "norm_num", "positivity", "field_simp", "field_simp; ring",
        ])
        # Compound: field_simp then ring/norm_num
        tactics.extend(["field_simp; ring", "field_simp; norm_num", "push_neg; omega"])
        for p in premises[:4]:
            tactics.append(f"linear_combination {p}")

    # === STRUCTURAL ===
    if profile.get("structural", 0) >= threshold:
        tactics.extend([
            "ext", "ext; simp", "ext; aesop", "ext; ring",
            "funext", "funext; simp",
            "congr", "congr 1",
            "constructor", "constructor <;> simp", "constructor <;> aesop",
            "cases", "rcases",
        ])
        for p in premises[:4]:
            tactics.append(f"gcongr; exact {p}")

    # === WITNESS ===
    if profile.get("witness", 0) >= threshold:
        tactics.extend([
            "constructor; simp_all", "constructor; aesop",
            "constructor <;> assumption",
        ])
        for h in hyp_names[:4]:
            tactics.append(f"use {h}")
            tactics.append(f"exact ⟨{h}, by simp⟩")
            tactics.append(f"refine ⟨{h}, ?_⟩")
        for p in premises[:4]:
            tactics.append(f"use {p}")

    # === CAST NORMALIZATION ===
    if profile.get("cast_norm", 0) >= threshold:
        tactics.extend([
            "norm_cast", "push_cast", "push_neg",
            "norm_cast; ring", "norm_cast; omega", "norm_cast; norm_num",
            "push_cast; ring", "push_neg; simp",
        ])

    # === AUTOMATION (always include as fallback) ===
    tactics.extend(["aesop", "solve_by_elim", "tauto", "decide", "assumption"])

    # === UNRESTRICTED PROOF SEARCH (expensive final fallback) ===
    tactics.extend(["apply?", "exact?"])

    # === APPLY CHAINS with premises (Tier 3 from research) ===
    if profile.get("exact_apply", 0) >= threshold and len(premises) >= 2:
        for i, p1 in enumerate(premises[:4]):
            for j, p2 in enumerate(premises[:4]):
                if i == j:
                    continue
                tactics.append(f"apply {p1}; exact {p2}")
                tactics.append(f"apply {p1}; apply {p2}; assumption")
                tactics.append(f"refine {p1} ?_; exact {p2}")
            for h in hyp_names[:3]:
                tactics.append(f"apply {p1}; exact {h}")
                tactics.append(f"refine {p1} {h} ?_")

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _holographic_profile(
    goal_text: str,
    similar_names: list[str],
    similar_types: list[str],
    scores: list[float],
) -> dict[str, float]:
    """Compute tactic family activity from cosine-similar theorems.

    Returns a dict mapping family name → activity score.  Families with
    zero activity should be suppressed (informational zeros).  Families
    with high activity should receive the full premise budget.
    """
    activity: dict[str, float] = {fam: 0.0 for fam in _FAMILY_GENERATORS}
    goal_lower = goal_text.lower()

    for name, type_text, score in zip(similar_names, similar_types, scores):
        if score < 0.25:
            break
        w = float(score)
        # Use both type signature (if available) and entity name for routing
        tp = type_text or ""
        nm = name.lower()

        # Every match supports exact/apply — the premise might close directly
        activity["exact_apply"] += w

        # Type-structure signals (from type_pp when available)
        if "=" in tp or "↔" in tp or "eq" in nm or "iff" in nm:
            activity["rewrite"] += w
            activity["simp_premise"] += w
        if "↔" in tp or "Iff" in tp or "iff" in nm:
            activity["structural"] += w * 0.5
        if "∃" in tp or "Exists" in tp or "exists" in nm:
            activity["witness"] += w
        if any(c in tp for c in ["≤", "≥", "<", ">"]) or \
           any(k in nm for k in ["nat.", "int.", "real.", "norm", "abs", "pow", "div"]):
            activity["arithmetic"] += w

        # Name-structure signals (always available, even without type_pp)
        if any(k in nm for k in ["rw", "rewrite", "simp", "unfold"]):
            activity["rewrite"] += w * 0.3
        if any(k in nm for k in ["ext", "funext", "constructor", "split"]):
            activity["structural"] += w * 0.3
        if any(k in nm for k in ["use", "witness", "choose", "obtain"]):
            activity["witness"] += w * 0.3

    # Goal-text signals (supplement entity-based routing)
    if "=" in goal_lower or "↔" in goal_lower:
        activity["rewrite"] = max(activity["rewrite"], 0.5)
        activity["simp_premise"] = max(activity["simp_premise"], 0.4)
    if "∃" in goal_text:
        activity["witness"] = max(activity["witness"], 0.5)
    if any(c in goal_text for c in ["≤", "≥", "<", ">", "∈"]):
        activity["arithmetic"] = max(activity["arithmetic"], 0.4)

    return activity


def _generate_targeted_tactics(
    premises: list[str],
    active_families: dict[str, float],
    threshold: float = 0.3,
) -> list[str]:
    """Generate a small, targeted tactic set from active families only.

    Families below the activity threshold are eliminated entirely —
    the informational zero means those tactics are structurally irrelevant.
    """
    tactics: list[str] = []

    if active_families.get("exact_apply", 0) >= threshold:
        for p in premises[:12]:
            tactics.append(f"exact {p}")
            tactics.append(f"apply {p}")

    if active_families.get("rewrite", 0) >= threshold:
        for p in premises[:12]:
            tactics.append(f"rw [{p}]")
            tactics.append(f"rw [← {p}]")

    if active_families.get("simp_premise", 0) >= threshold:
        for p in premises[:8]:
            tactics.append(f"simp [{p}]")
            tactics.append(f"simp only [{p}]")
        # Multi-premise simp
        for i in range(min(len(premises), 4)):
            for j in range(i + 1, min(len(premises), 5)):
                tactics.append(f"simp [{premises[i]}, {premises[j]}]")

    if active_families.get("arithmetic", 0) >= threshold:
        tactics.extend(["ring", "omega", "linarith", "norm_num", "positivity"])

    if active_families.get("structural", 0) >= threshold:
        tactics.extend([
            "ext; simp", "ext; aesop",
            "constructor; all_goals simp", "constructor; all_goals aesop",
            "push_neg; simp",
        ])

    if active_families.get("witness", 0) >= threshold:
        tactics.extend(["constructor; simp_all", "constructor; aesop"])
        for p in premises[:3]:
            tactics.append(f"use {p}")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _holographic_close(
    goals: list[str],
    *,
    lean: Any,
    conn: sqlite3.Connection,
    sentence_encoder: Any,
    original_goal: str = "",
    max_chain_depth: int = 3,
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Holographic elimination closer with multi-projection coherence.

    Uses BOTH the original goal AND the D1-contracted residual to find
    entities with multi-dimensional coherence: an entity that matches
    both "what we need to prove" and "what it reduces to" is structurally
    aligned with the proof's trajectory, not just its starting point.

    For each goal:
      1. Embed original goal AND residual goal.
      2. Find entities that score well on BOTH projections (intersection).
      3. Compute holographic profile from the coherent match set.
      4. Generate targeted tactics; on progress, chain with re-embedding.

    Returns (closed, remaining_goals, trace).
    """
    names, embs = _get_entity_embeddings(conn, sentence_encoder)
    if not names or embs is None:
        return False, list(goals), []

    # Pre-compute original goal embedding for coherence scoring
    orig_emb = None
    if original_goal:
        orig_emb = sentence_encoder.encode(
            [original_goal], normalize_embeddings=True, show_progress_bar=False,
        )

    trace: list[dict[str, Any]] = []
    remaining = list(goals)

    for goal_text in goals[:3]:
        closed_this = _holographic_close_single(
            goal_text, lean=lean, conn=conn,
            names=names, embs=embs,
            sentence_encoder=sentence_encoder,
            original_emb=orig_emb,
            original_goal=original_goal,
            depth=0, max_depth=max_chain_depth,
            trace=trace,
        )
        if closed_this:
            remaining = [g for g in remaining if g != goal_text]

    all_closed = len(remaining) == 0
    return all_closed, remaining, trace


def _holographic_close_single(
    goal_text: str,
    *,
    lean: Any,
    conn: sqlite3.Connection,
    names: list[str],
    embs: Any,
    sentence_encoder: Any,
    original_emb: Any | None = None,
    original_goal: str = "",
    depth: int,
    max_depth: int,
    trace: list[dict[str, Any]],
) -> bool:
    """Try to close a single goal with multi-projection holographic coherence.

    When original_emb is provided, entities are scored by how well they
    match BOTH the original goal and the current residual.  An entity
    coherent with both projections is structurally aligned with the
    proof trajectory — it understands both "what we need" and "what
    it reduces to."
    """
    if depth >= max_depth:
        return False

    # Step 1: Embed residual goal, compute multi-projection coherence
    goal_emb = sentence_encoder.encode(
        [goal_text], normalize_embeddings=True, show_progress_bar=False,
    )
    residual_scores = (goal_emb @ embs.T).flatten()

    if original_emb is not None:
        original_scores = (original_emb @ embs.T).flatten()
        # Geometric mean: entity must match BOTH projections.
        # sqrt(residual * original) rewards multi-dimensional coherence.
        coherence = np.sqrt(np.maximum(residual_scores, 0) * np.maximum(original_scores, 0))
    else:
        coherence = residual_scores

    top_indices = np.argsort(-coherence)[:20]
    similar_names = [names[i] for i in top_indices]
    similar_scores = [float(coherence[i]) for i in top_indices]

    # Fetch type signatures for profile computation (graceful if column missing)
    similar_types: list[str] = []
    has_type_pp = any(
        r[1] == "type_pp" for r in conn.execute("PRAGMA table_info(entities)")
    )
    for name in similar_names:
        if has_type_pp:
            row = conn.execute(
                "SELECT type_pp FROM entities WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
            similar_types.append(str(row[0]) if row and row[0] else "")
        else:
            similar_types.append("")

    # Step 2: Compute holographic profile — what families are active/zero
    profile = _holographic_profile(goal_text, similar_names, similar_types, similar_scores)

    # Log the elimination
    active = {k: round(v, 2) for k, v in profile.items() if v >= 0.3}
    suppressed = [k for k, v in profile.items() if v < 0.3]
    trace.append({
        "depth": depth,
        "goal": goal_text[:80],
        "top_match": similar_names[0] if similar_names else "",
        "top_score": round(similar_scores[0], 3) if similar_scores else 0,
        "active_families": active,
        "suppressed_families": suppressed,
    })

    # Step 3: Generate targeted tactics from active families only
    tactics = _generate_targeted_tactics(similar_names, profile)
    if not tactics:
        return False

    # Step 4: Execute.  On progress, chain.
    for tactic in tactics:
        try:
            result = lean.try_tactic(goal_text, tactic)
        except Exception:
            continue

        if not result.success:
            continue

        new_goals = [str(g) for g in (result.new_goals or []) if str(g).strip()]

        if not new_goals:
            # Closed!
            logger.info(
                "Holographic close (depth=%d): %s on %s",
                depth, tactic, goal_text[:60],
            )
            trace.append({
                "depth": depth,
                "tactic": tactic,
                "result": "closed",
            })
            return True

        # Progress: fewer goals or different goal text = the tactic did something
        if len(new_goals) <= 1 and new_goals[0] != goal_text:
            # Chain: re-embed the new goal and recurse
            logger.info(
                "Holographic chain (depth=%d): %s progressed → %s",
                depth, tactic, new_goals[0][:60],
            )
            trace.append({
                "depth": depth,
                "tactic": tactic,
                "result": "progressed",
                "new_goal": new_goals[0][:80],
            })
            chained = _holographic_close_single(
                new_goals[0],
                lean=lean, conn=conn, names=names, embs=embs,
                sentence_encoder=sentence_encoder,
                depth=depth + 1, max_depth=max_depth,
                trace=trace,
            )
            if chained:
                return True
            # Chain didn't close — try finishers on the progressed goal
            for finisher in ["ring", "omega", "simp_all", "norm_num",
                             "linarith", "aesop", "apply?", "positivity"]:
                try:
                    fin_result = lean.try_tactic(new_goals[0], finisher)
                    if fin_result.success and not fin_result.new_goals:
                        logger.info(
                            "Holographic chain+finisher (depth=%d): %s → %s",
                            depth, tactic, finisher,
                        )
                        trace.append({
                            "depth": depth,
                            "tactic": f"{tactic} → {finisher}",
                            "result": "closed",
                        })
                        return True
                except Exception:
                    pass

    return False


# ---------------------------------------------------------------------------
# Trace index: theorem_name → ordered tactic steps (loaded once, cached)
# ---------------------------------------------------------------------------
_TRACE_INDEX: dict[str, list[dict[str, Any]]] | None = None


def _get_trace_index() -> dict[str, list[dict[str, Any]]]:
    global _TRACE_INDEX
    if _TRACE_INDEX is not None:
        return _TRACE_INDEX
    import json as _json
    from pathlib import Path as _Path
    index_path = _Path("data/canonical/trace_index.json")
    if index_path.exists():
        _TRACE_INDEX = _json.loads(index_path.read_text())
        logger.info("Loaded trace index: %d theorems", len(_TRACE_INDEX))
    else:
        _TRACE_INDEX = {}
        logger.warning("Trace index not found at %s", index_path)
    return _TRACE_INDEX


def _compute_holographic_premises(
    *,
    original_goal: str,
    residual_goal: str,
    conn: sqlite3.Connection,
    sentence_encoder: Any,
    top_k: int = 20,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Multi-projection coherence scoring + tactic trace bundling.

    1. Embed original + residual goals against entity type signatures
    2. Find top-K coherent matches
    3. Look up each match's proof trace from the canonical data
    4. Bundle traces: extract common tactic patterns weighted by score
    5. Return (premise_names, trace, bundled_tactics)

    The bundled_tactics are ordered composite sequences extracted from
    high-scoring proof traces — the holographic answer space's output.
    """
    names, embs = _get_entity_embeddings(conn, sentence_encoder)
    if not names or embs is None:
        return [], [], []

    # Embed both projections
    texts_to_embed = []
    if original_goal:
        texts_to_embed.append(original_goal)
    if residual_goal and residual_goal != original_goal:
        texts_to_embed.append(residual_goal)
    if not texts_to_embed:
        return [], [], []

    embeddings = sentence_encoder.encode(
        texts_to_embed, normalize_embeddings=True, show_progress_bar=False,
    )

    if len(texts_to_embed) == 2:
        orig_scores = (embeddings[0:1] @ embs.T).flatten()
        resid_scores = (embeddings[1:2] @ embs.T).flatten()
        coherence = np.sqrt(np.maximum(orig_scores, 0) * np.maximum(resid_scores, 0))
    else:
        coherence = (embeddings[0:1] @ embs.T).flatten()

    top_indices = np.argsort(-coherence)[:top_k]
    premises = [names[i] for i in top_indices]
    scores = [float(coherence[i]) for i in top_indices]

    # --- Proof trace bundling with consensus-weighted decomposition ---
    # Each matched entity has a proof trace (ordered tactic steps) and a
    # cosine score.  The consensus across traces reveals the multi-step
    # composition structure:
    #
    # - Step-position voting: at each position (step 1, step 2, ...),
    #   which tactic families agree?  Weighted by cosine score.
    # - High consensus at a position = high confidence that step.
    # - Pair voting: which (step_i, step_i+1) transitions recur?
    # - The final bundled sequence is the consensus chain, not just
    #   a flat bag of tactics.
    trace_index = _get_trace_index()

    tactic_sequences: list[tuple[float, list[str]]] = []
    # Step-position consensus: position → {tactic → weighted_votes}
    position_votes: dict[int, dict[str, float]] = {}
    # Pair consensus: (tactic_i, tactic_j) → weighted_votes
    pair_votes: dict[str, float] = {}
    # Individual tactic votes (flat, for fallback)
    tactic_votes: dict[str, float] = {}

    for name, score in zip(premises, scores):
        if score < 0.3:
            break
        steps = trace_index.get(name, [])
        if not steps:
            continue
        ordered_tactics = [s["tactic"] for s in steps if s.get("tactic")]
        if not ordered_tactics:
            continue
        tactic_sequences.append((score, ordered_tactics))

        # Position-weighted voting
        for pos, tac in enumerate(ordered_tactics):
            if pos not in position_votes:
                position_votes[pos] = {}
            position_votes[pos][tac] = position_votes[pos].get(tac, 0.0) + score
            tactic_votes[tac] = tactic_votes.get(tac, 0.0) + score

        # Pair transitions
        for i in range(len(ordered_tactics) - 1):
            pair_key = f"{ordered_tactics[i]}; {ordered_tactics[i+1]}"
            pair_votes[pair_key] = pair_votes.get(pair_key, 0.0) + score

    # Extract the consensus chain: at each position, pick the highest-voted tactic
    consensus_chain: list[str] = []
    for pos in sorted(position_votes.keys()):
        votes = position_votes[pos]
        if not votes:
            break
        best_tac = max(votes, key=votes.get)  # type: ignore[arg-type]
        best_score = votes[best_tac]
        # Only include if consensus is meaningful (more than one match agrees)
        if best_score >= 0.5:
            consensus_chain.append(best_tac)

    # Build the bundled tactic list with proper ordering:
    bundled: list[str] = []
    seen: set[str] = set()

    # Tier 1: Consensus chain (ordered multi-step from position voting)
    for tac in consensus_chain:
        if tac not in seen:
            bundled.append(tac)
            seen.add(tac)

    # Tier 2: High-confidence pairs (multi-step transitions)
    for pair, _vote in sorted(pair_votes.items(), key=lambda x: -x[1])[:15]:
        if pair not in seen:
            bundled.append(pair)
            seen.add(pair)

    # Tier 3: Full sequences from top-3 matches
    for _score, seq in sorted(tactic_sequences, key=lambda x: -x[0])[:3]:
        for tac in seq:
            if tac not in seen:
                bundled.append(tac)
                seen.add(tac)

    # Tier 4: High-vote individual tactics (fallback)
    for tac, _vote in sorted(tactic_votes.items(), key=lambda x: -x[1])[:20]:
        if tac not in seen:
            bundled.append(tac)
            seen.add(tac)

    trace = [{
        "original_goal": original_goal[:80],
        "residual_goal": residual_goal[:80],
        "top_matches": [
            {"name": n, "score": round(s, 3)}
            for n, s in zip(premises[:5], scores[:5])
        ],
        "bundled_tactics_count": len(bundled),
        "top_bundled": bundled[:5],
        "tactic_sequences_found": len(tactic_sequences),
    }]

    return premises, trace, bundled


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


@dataclass
class HardtailBridgeResult:
    theorem_id: str
    started: bool
    theorem_faithful: bool
    residual_bucket: str
    start_goal_bucket: str
    closed: bool
    closed_by: str
    progressed: bool
    initial_goal_count: int
    post_ducky1_goal_count: int
    post_search_goal_count: int
    final_goal_count: int
    controller_decision: dict[str, Any]
    replay: dict[str, Any]
    ducky_pass_1: dict[str, Any] | None
    first_order_search: dict[str, Any] | None
    ducky_pass_2: dict[str, Any] | None
    symbolic_close_pass_2: dict[str, Any] | None = None
    final_goals: list[str] = field(default_factory=list)
    rarified_gap_packet: dict[str, Any] | None = None
    stage_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bridge_search_config(base_cfg: SearchConfig, decision: SecondOrderDecision) -> SearchConfig:
    # Cap the post-Ducky search budget: if Ducky contracted the state well,
    # a short focused search should suffice.  Large budgets waste attempts on
    # structural expansion (intro/cases/constructor) that explodes goal count.
    capped_budget = min(max(int(decision.search_budget), 1), 64)
    return replace(
        base_cfg,
        budget=capped_budget,
        max_progress_steps=min(max(int(decision.search_max_progress_steps), 0), 6),
        search_mode="full",
        collect_trace=True,
        cosine_rw_beam=max(int(decision.cosine_rw_beam), 1),
        cosine_rw_seq_enabled=bool(decision.enable_cosine_rw_seq),
        cosine_simp_enabled=bool(decision.enable_cosine_simp),
        interleaved_bootstrap_enabled=bool(decision.enable_interleaved_bootstrap),
        cosine_apply_enabled=bool(decision.enable_cosine_apply),
        cosine_apply_gated=bool(decision.gated_cosine_apply),
        cosine_apply_beam=max(int(decision.cosine_apply_beam), 1),
        dr_ducky_enabled=False,
    )


def post_ducky_symbolic_config(base_cfg: SearchConfig) -> SearchConfig:
    return replace(
        base_cfg,
        budget=max(min(int(base_cfg.budget), 64), 16),
        max_progress_steps=max(min(int(base_cfg.max_progress_steps or 8), 8), 4),
        search_mode="no_learned",
        collect_trace=True,
        hammer_delegation=True,
        cosine_rw_seq_enabled=True,
        cosine_rw_beam=5,
        cosine_simp_enabled=True,
        interleaved_bootstrap_enabled=True,
        cosine_apply_enabled=True,
        cosine_apply_gated=False,
        cosine_apply_beam=8,
        dr_ducky_enabled=False,
    )


def run_hardtail_bridge_on_row(
    row: dict[str, Any],
    *,
    packet: dict[str, Any] | None,
    controller_runtime: Any | None = None,
    pipeline: Pipeline,
    search_config: SearchConfig,
    conn: sqlite3.Connection,
    lean: Any,
    theorem_id_map: dict[str, int] | None = None,
    sentence_encoder: Any | None = None,
    ducky_first_max_programs: int = 24,
    ducky_first_max_rounds: int = 3,
    ducky_second_max_programs: int = 20,
    ducky_second_max_rounds: int = 2,
    disabled_ducky_tactics: set[str] | None = None,
    search_fn: Callable[..., SearchResult] = search,
) -> HardtailBridgeResult:
    theorem_id = str(row.get("theorem_id", "") or "")
    residual_bucket = str(row.get("residual_bucket", "") or "")
    replay = replay_residual_state(row, lean)
    decision = derive_second_order_decision(row, packet, runtime=controller_runtime)
    start_goal_bucket = str(row.get("last_goal_bucket", "") or classify_goal_bucket(str(row.get("last_goal", "") or row.get("goal_state", "") or "")))
    stage_trace: list[dict[str, Any]] = []
    if not replay.replay_success or not replay.goal_state:
        return HardtailBridgeResult(
            theorem_id=theorem_id,
            started=False,
            theorem_faithful=False,
            residual_bucket=residual_bucket,
            start_goal_bucket=start_goal_bucket,
            closed=False,
            closed_by="",
            progressed=False,
            initial_goal_count=0,
            post_ducky1_goal_count=0,
            post_search_goal_count=0,
            final_goal_count=0,
            controller_decision=decision.to_dict(),
            replay=replay.to_dict(),
            ducky_pass_1=None,
            first_order_search=None,
            ducky_pass_2=None,
            symbolic_close_pass_2=None,
            final_goals=[],
            rarified_gap_packet=None,
            stage_trace=[],
        )

    accessible_theorem_id = theorem_id_map.get(theorem_id) if theorem_id_map else None
    initial_goals = [replay.goal_state]
    current_goals = list(initial_goals)
    ducky1_payload: dict[str, Any] | None = None
    ducky2_payload: dict[str, Any] | None = None
    search_payload: dict[str, Any] | None = None
    symbolic2_payload: dict[str, Any] | None = None
    closed = False
    closed_by = ""

    # Holographic premise scoring — compute ONCE, use in all subsequent stages.
    # This is the multi-projection coherence scoring that identifies the
    # structurally correct premises from the entity corpus.
    holographic_premises: list[str] = []
    holographic_trace: list[dict[str, Any]] = []
    bundled_tactics: list[str] = []
    if sentence_encoder is not None and conn is not None:
        holographic_premises, holographic_trace, bundled_tactics = _compute_holographic_premises(
            original_goal=replay.goal_state if replay.replay_success else "",
            residual_goal="",  # no residual yet — D1 hasn't run
            conn=conn,
            sentence_encoder=sentence_encoder,
        )
        stage_trace.append({
            "stage": "holographic_scoring",
            "premises_found": len(holographic_premises),
            "top_premises": holographic_premises[:5],
            "bundled_tactics_count": len(bundled_tactics),
            "trace": holographic_trace,
        })

    # Profile-guided direct close: use the holographic bank activation
    # profile to select tactics from the FULL Lean 4 universe, then try
    # them on the LIVE replayed GoalState.  Dr. Ducky uses the profile
    # to eliminate irrelevant tactic families (informational zeros) and
    # concentrate on the families that structurally match this residual.
    if not closed and replay.goal_state:
        # Use the holographic profile already computed above.
        # Fetch type signatures for proper profiling.
        has_type_pp = any(
            r[1] == "type_pp" for r in conn.execute("PRAGMA table_info(entities)")
        ) if conn else False
        holo_types: list[str] = []
        holo_scores: list[float] = []
        if holographic_premises and conn:
            names_embs = _get_entity_embeddings(conn, sentence_encoder) if sentence_encoder else ([], None)
            entity_names, entity_embs = names_embs
            if entity_embs is not None:
                goal_emb = sentence_encoder.encode(
                    [replay.goal_state], normalize_embeddings=True, show_progress_bar=False,
                )
                raw = (goal_emb @ entity_embs.T).flatten()
                for hp in holographic_premises[:12]:
                    if hp in entity_names:
                        idx = entity_names.index(hp)
                        holo_scores.append(float(raw[idx]))
                    else:
                        holo_scores.append(0.5)
                    if has_type_pp:
                        r = conn.execute("SELECT type_pp FROM entities WHERE name = ? LIMIT 1", (hp,)).fetchone()
                        holo_types.append(str(r[0]) if r and r[0] else "")
                    else:
                        holo_types.append("")

        direct_profile = _holographic_profile(
            replay.goal_state,
            holographic_premises[:12] if holographic_premises else [],
            holo_types,
            holo_scores if holo_scores else [0.5] * min(len(holographic_premises), 12),
        )

        # Extract hypothesis names from the goal text
        hyp_names = [
            word for word in replay.goal_state.split()
            if len(word) > 1 and word[0].islower() and word.isidentifier()
              and not word.startswith("inst") and word not in ("fun", "let", "in", "if", "then", "else")
        ][:8]

        guided_tactics = _profile_guided_tactics(
            premises=holographic_premises,
            hyp_names=hyp_names,
            profile=direct_profile,
            goal_text=replay.goal_state,
        )

        direct_close_tried = 0
        for tactic in guided_tactics:
            try:
                import signal as _sig

                def _tactic_timeout(_signum: int, _frame: object) -> None:
                    raise TimeoutError("tactic timeout")

                old = _sig.signal(_sig.SIGALRM, _tactic_timeout)
                # 10s for apply?/exact? (the holographic system already narrowed
                # the search — if it can't close in 10s, the premise isn't there).
                # 5s for everything else.
                limit = 10 if tactic in ("apply?", "exact?") or "apply? using" in tactic or "exact? using" in tactic else 5
                _sig.alarm(limit)
                try:
                    result = lean.try_tactic(replay.goal_state, tactic)
                finally:
                    _sig.alarm(0)
                    _sig.signal(_sig.SIGALRM, old)

                direct_close_tried += 1
                if result.success and not result.new_goals:
                    current_goals = []
                    closed = True
                    closed_by = f"direct_close:{tactic[:40]}"
                    stage_trace.append({
                        "stage": "direct_close",
                        "closed": True,
                        "tactic": tactic,
                        "tactics_tried": direct_close_tried,
                        "profile": {k: round(v, 1) for k, v in direct_profile.items() if v >= 0.3},
                    })
                    break
            except (Exception, TimeoutError):
                pass

        if not closed:
            stage_trace.append({
                "stage": "direct_close",
                "closed": False,
                "tactics_tried": direct_close_tried,
                "profile": {k: round(v, 1) for k, v in direct_profile.items() if v >= 0.3},
            })

    if not closed and decision.invoke_first_ducky:
        ducky1 = run_ducky_on_goal(
            replay.goal_state,
            theorem_id=theorem_id,
            lean=lean,
            conn=conn,
            accessible_theorem_id=accessible_theorem_id,
            row_overrides=_ducky_row_overrides(row, replay.goal_state),
            max_programs=ducky_first_max_programs,
            max_rounds=ducky_first_max_rounds,
            disabled_tactics=disabled_ducky_tactics,
            allowed_backend_families=set(decision.first_pass_backends),
            allowed_engine_names=set(decision.first_pass_engines),
            holographic_premises=holographic_premises,
        )
        ducky1_payload = ducky1.to_dict()
        if ducky1.closed:
            current_goals = []
        elif ducky1.progressed and ducky1.goals_after:
            current_goals = list(ducky1.goals_after)
        else:
            current_goals = list(initial_goals)
        stage_trace.append(
            {
                "stage": "dr_ducky_pass_1",
                "closed": ducky1.closed,
                "progressed": ducky1.progressed,
                "goals_after": list(current_goals),
                "programs_considered": ducky1.programs_considered,
            }
        )
        if ducky1.closed:
            closed = True
            closed_by = "dr_ducky_pass_1"

    # Re-score with D1 residual for multi-projection coherence (if D1 changed the goal)
    if not closed and sentence_encoder is not None and conn is not None and current_goals != initial_goals:
        _, _, bundled_tactics = _compute_holographic_premises(
            original_goal=replay.goal_state if replay.replay_success else "",
            residual_goal=current_goals[0] if current_goals else "",
            conn=conn,
            sentence_encoder=sentence_encoder,
        )

    # Holographic tactic chain: try the bundled tactics from matched proof
    # traces directly on the live goal state.  These are the exact tactic
    # sequences that solved structurally similar theorems.  Single-step
    # tactics are tried individually; multi-step ";"-separated sequences
    # are tried as chains where each step feeds the next GoalState.
    if not closed and bundled_tactics and current_goals:
        holo_tactic_closed = False
        holo_tactic_progressed = False
        goal_text = current_goals[0]
        for tactic in bundled_tactics:
            try:
                result = lean.try_tactic(goal_text, tactic)
                if result.success and not result.new_goals:
                    holo_tactic_closed = True
                    current_goals = []
                    closed = True
                    closed_by = "holographic_tactic_chain"
                    logger.info("Holographic tactic close: %s on %s", tactic, goal_text[:60])
                    break
                if result.success and result.new_goals and result.new_goals != [goal_text]:
                    # Tactic progressed — try remaining bundled tactics on new state
                    holo_tactic_progressed = True
                    new_goal = result.new_goals[0]
                    for follow_tactic in bundled_tactics:
                        if follow_tactic == tactic:
                            continue
                        try:
                            follow_result = lean.try_tactic(new_goal, follow_tactic)
                            if follow_result.success and not follow_result.new_goals:
                                holo_tactic_closed = True
                                current_goals = []
                                closed = True
                                closed_by = "holographic_tactic_chain"
                                logger.info("Holographic chain close: %s → %s", tactic, follow_tactic)
                                break
                        except Exception:
                            pass
                    if closed:
                        break
                    # Also try finishers on the progressed state
                    for finisher in ["ring", "omega", "simp_all", "norm_num",
                                     "linarith", "aesop", "apply?", "positivity"]:
                        try:
                            fin_result = lean.try_tactic(new_goal, finisher)
                            if fin_result.success and not fin_result.new_goals:
                                holo_tactic_closed = True
                                current_goals = []
                                closed = True
                                closed_by = "holographic_tactic_chain"
                                logger.info("Holographic + finisher: %s → %s", tactic, finisher)
                                break
                        except Exception:
                            pass
                    if closed:
                        break
            except Exception:
                pass
        stage_trace.append({
            "stage": "holographic_tactic_chain",
            "closed": holo_tactic_closed,
            "progressed": holo_tactic_progressed,
            "goals_after": list(current_goals),
            "tactics_tried": len(bundled_tactics),
        })

    # Post-Ducky symbolic closer with holographic premises injected.
    if not closed and current_goals:
        symbolic1_cfg = post_ducky_symbolic_config(search_config)
        symbolic1_result = search_fn(
            theorem_id=theorem_id,
            initial_goal=current_goals if len(current_goals) > 1 else current_goals[0],
            pipeline=pipeline,
            conn=conn,
            lean=lean,
            config=symbolic1_cfg,
            accessible_theorem_id=accessible_theorem_id,
            sentence_encoder=sentence_encoder,
            holographic_premises=holographic_premises,
        )
        if symbolic1_result.success:
            current_goals = []
            closed = True
            closed_by = "post_ducky1_symbolic_close"
        stage_trace.append(
            {
                "stage": "post_ducky1_symbolic_close",
                "closed": symbolic1_result.success,
                "progressed": bool(symbolic1_result.progress_steps or symbolic1_result.goals_closed),
                "goals_after": list(current_goals),
                "attempts": symbolic1_result.attempts,
                "close_provenance": list(symbolic1_result.close_provenance),
            }
        )

    if not closed:
        cfg = bridge_search_config(search_config, decision)
        search_result = search_fn(
            theorem_id=theorem_id,
            initial_goal=current_goals if len(current_goals) > 1 else current_goals[0],
            pipeline=pipeline,
            conn=conn,
            lean=lean,
            config=cfg,
            accessible_theorem_id=accessible_theorem_id,
            sentence_encoder=sentence_encoder,
            holographic_premises=holographic_premises,
        )
        pre_search_goal_count = len(current_goals)
        if search_result.success:
            current_goals = []
        else:
            post_goals = list(search_result.final_goals) if getattr(search_result, "final_goals", None) else list(current_goals)
            # Goal-explosion guard: if first-order search expanded the goal
            # count beyond 3x without closing, revert to the Ducky-contracted
            # state.  Going 1→197 goals is structural expansion, not progress.
            if len(post_goals) > max(pre_search_goal_count * 3, 6) and not search_result.success:
                current_goals = current_goals  # keep pre-search state
            else:
                current_goals = post_goals
        search_payload = _search_result_payload(search_result)
        stage_trace.append(
            {
                "stage": "first_order_search",
                "closed": search_result.success,
                "progressed": bool(search_result.progress_steps or search_result.goals_closed),
                "goals_after": list(current_goals),
                "attempts": search_result.attempts,
                "close_provenance": list(search_result.close_provenance),
            }
        )
        if search_result.success:
            closed = True
            closed_by = "first_order_search"
        else:
            decision = refine_second_pass_decision(
                decision,
                current_goals,
                search_trace=list(search_result.step_trace),
                row=row,
                packet=packet,
                runtime=controller_runtime,
            )

    final_goals = list(current_goals)
    rarified_gap_packet: dict[str, Any] | None = None

    if not closed and current_goals:
        ducky2_runs, rarified_goals = _run_second_ducky_pass(
            theorem_id=theorem_id,
            row=row,
            goals=current_goals,
            decision=decision,
            lean=lean,
            conn=conn,
            accessible_theorem_id=accessible_theorem_id,
            max_programs=ducky_second_max_programs,
            max_rounds=ducky_second_max_rounds,
            disabled_tactics=disabled_ducky_tactics,
        )
        ducky2_payload = {
            "runs": ducky2_runs,
            "rarified_goals": list(rarified_goals),
            "progressed": any(bool(run.get("progressed")) for run in ducky2_runs),
            "closed": len(rarified_goals) == 0,
        }
        final_goals = list(rarified_goals)
        stage_trace.append(
            {
                "stage": "dr_ducky_pass_2",
                "closed": len(rarified_goals) == 0,
                "progressed": any(bool(run.get("progressed")) for run in ducky2_runs),
                "goals_after": list(final_goals),
                "evaluated_goals": len(ducky2_runs),
            }
        )
        if len(rarified_goals) == 0:
            closed = True
            closed_by = "dr_ducky_pass_2"
            final_goals = []
        else:
            symbolic_result = search_fn(
                theorem_id=theorem_id,
                initial_goal=rarified_goals if len(rarified_goals) > 1 else rarified_goals[0],
                pipeline=pipeline,
                conn=conn,
                lean=lean,
                config=post_ducky_symbolic_config(search_config),
                accessible_theorem_id=accessible_theorem_id,
                sentence_encoder=sentence_encoder,
                holographic_premises=holographic_premises,
            )
            if symbolic_result.success:
                final_goals = []
            else:
                final_goals = list(symbolic_result.final_goals) if getattr(symbolic_result, "final_goals", None) else list(rarified_goals)
            symbolic2_payload = _search_result_payload(symbolic_result)
            stage_trace.append(
                {
                    "stage": "post_ducky_symbolic_2",
                    "closed": symbolic_result.success,
                    "progressed": bool(symbolic_result.progress_steps or symbolic_result.goals_closed),
                    "goals_after": list(final_goals),
                    "attempts": symbolic_result.attempts,
                    "close_provenance": list(symbolic_result.close_provenance),
                }
            )
            if symbolic_result.success:
                closed = True
                closed_by = "post_ducky_symbolic_2"

    progressed = bool(
        (ducky1_payload and ducky1_payload.get("progressed"))
        or (search_payload and search_payload.get("progress_steps"))
        or (ducky2_payload and ducky2_payload.get("progressed"))
        or (symbolic2_payload and symbolic2_payload.get("progress_steps"))
    )
    if not closed and rarified_gap_packet is None:
        rarified_gap_packet = build_rarified_gap_packet(
            theorem_id=theorem_id,
            row=row,
            decision=decision,
            final_goals=final_goals,
            stage_trace=stage_trace,
            search_payload=search_payload,
            ducky2_payload=ducky2_payload,
        )

    return HardtailBridgeResult(
        theorem_id=theorem_id,
        started=True,
        theorem_faithful=bool(replay.theorem_faithful),
        residual_bucket=residual_bucket,
        start_goal_bucket=start_goal_bucket,
        closed=closed,
        closed_by=closed_by,
        progressed=progressed,
        initial_goal_count=len(initial_goals),
        post_ducky1_goal_count=len(ducky1_payload.get("goals_after", initial_goals)) if ducky1_payload else len(initial_goals),
        post_search_goal_count=len(search_payload.get("final_goals", current_goals)) if search_payload else len(current_goals),
        final_goal_count=len(final_goals),
        controller_decision=decision.to_dict(),
        replay=replay.to_dict(),
        ducky_pass_1=ducky1_payload,
        first_order_search=search_payload,
        ducky_pass_2=ducky2_payload,
        symbolic_close_pass_2=symbolic2_payload,
        final_goals=final_goals,
        rarified_gap_packet=rarified_gap_packet,
        stage_trace=stage_trace,
    )


def _ducky_row_overrides(row: dict[str, Any], goal_text: str) -> dict[str, Any]:
    return {
        "residual_bucket": str(row.get("residual_bucket", "") or ""),
        "last_goal_bucket": classify_goal_bucket(goal_text),
        "reasoning_gap_family": str(row.get("reasoning_gap_family", "") or ""),
        "search_pathology_tags": list(row.get("search_pathology_tags") or []),
        "attempts": int(row.get("attempts", 0) or 0),
        "goals_closed": int(row.get("goals_closed", 0) or 0),
        "goals_remaining": int(row.get("goals_remaining", 0) or 0),
        "lane_sequence": str(row.get("lane_sequence", "") or ""),
        "remaining_goals_snapshot": list(row.get("remaining_goals_snapshot") or [goal_text]),
    }


def _search_result_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "attempts": int(result.attempts),
        "goals_closed": int(result.goals_closed),
        "goals_remaining": int(result.goals_remaining),
        "progress_steps": int(result.progress_steps),
        "close_provenance": list(result.close_provenance),
        "tactics_used": list(result.tactics_used),
        "temporal_trace": list(result.temporal_trace),
        "step_trace": list(result.step_trace),
        "final_goals": list(getattr(result, "final_goals", []) or []),
        "apply_attempt_count": int(getattr(result, "apply_attempt_count", 0) or 0),
        "apply_accept_count": int(getattr(result, "apply_accept_count", 0) or 0),
        "apply_goal_close_count": int(getattr(result, "apply_goal_close_count", 0) or 0),
        "replanner_trigger_count": int(getattr(result, "replanner_trigger_count", 0) or 0),
    }


def _run_second_ducky_pass(
    *,
    theorem_id: str,
    row: dict[str, Any],
    goals: list[str],
    decision: SecondOrderDecision,
    lean: Any,
    conn: sqlite3.Connection,
    accessible_theorem_id: int | None,
    max_programs: int,
    max_rounds: int,
    disabled_tactics: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    rarified_goals: list[str] = []
    goal_limit = min(max(int(decision.second_pass_goal_limit), 1), len(goals))
    for idx, goal in enumerate(goals):
        if idx >= goal_limit:
            rarified_goals.append(goal)
            continue
        goal_bucket = classify_goal_bucket(goal)
        goal_backends = _unique(
            list(decision.second_pass_backends) + list(build_goal_capsule(_goal_row(row, goal)).backend_preferences)
        )
        goal_engines = _unique(
            list(decision.second_pass_engines) + list(build_goal_capsule(_goal_row(row, goal)).allowed_engines)
        )
        result = run_ducky_on_goal(
            goal,
            theorem_id=theorem_id,
            lean=lean,
            conn=conn,
            accessible_theorem_id=accessible_theorem_id,
            row_overrides=_ducky_row_overrides(row, goal),
            max_programs=max_programs,
            max_rounds=max_rounds,
            disabled_tactics=disabled_tactics,
            allowed_backend_families=set(goal_backends),
            allowed_engine_names=set(goal_engines),
        )
        payload = result.to_dict()
        payload["goal_index"] = idx
        payload["input_goal_bucket"] = goal_bucket
        payload["allowed_backend_families"] = goal_backends
        payload["allowed_engine_names"] = goal_engines
        runs.append(payload)
        if result.closed:
            continue
        if result.progressed and result.goals_after:
            rarified_goals.extend(result.goals_after)
        else:
            rarified_goals.append(goal)
    return runs, rarified_goals


def build_rarified_gap_packet(
    *,
    theorem_id: str,
    row: dict[str, Any],
    decision: SecondOrderDecision,
    final_goals: list[str],
    stage_trace: list[dict[str, Any]],
    search_payload: dict[str, Any] | None,
    ducky2_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    goal_packets: list[dict[str, Any]] = []
    recommended_targets: list[str] = []
    recommended_backends: list[str] = []
    for idx, goal in enumerate(final_goals[:8]):
        capsule = build_goal_capsule(_goal_row(row, goal))
        recommended_targets.extend(capsule.specialist_targets)
        recommended_backends.extend(capsule.backend_preferences)
        goal_packets.append(
            {
                "goal_index": idx,
                "goal_text": goal,
                "goal_bucket": capsule.specification.goal_bucket,
                "reasoning_gap_family": capsule.specification.reasoning_gap_family,
                "specialist_targets": list(capsule.specialist_targets),
                "allowed_engines": list(capsule.allowed_engines),
                "backend_preferences": list(capsule.backend_preferences),
                "projector_markers": list(capsule.specification.projector_markers),
                "representation_pressures": list(capsule.specification.representation_pressures),
                "search_control": dict(capsule.specification.search_control),
                "residual_geometry": dict(capsule.specification.residual_geometry),
            }
        )
    tractability = "rarified_local_tail" if len(final_goals) <= 2 else "rarified_multigoal_tail"
    if any(packet["goal_bucket"] in {"membership", "subset", "exists"} for packet in goal_packets):
        tractability = "rarified_relational_tail"
    elif any("recursive" in packet["reasoning_gap_family"] for packet in goal_packets):
        tractability = "rarified_recursive_tail"
    return {
        "packet_version": "rarified_proof_gap_v1",
        "theorem_id": theorem_id,
        "residual_bucket": str(row.get("residual_bucket", "") or ""),
        "start_goal_bucket": str(row.get("last_goal_bucket", "") or ""),
        "controller_mode": decision.controller_mode,
        "rarified_target": decision.rarified_target,
        "stage_trace": list(stage_trace),
        "post_search_summary": search_payload or {},
        "post_ducky2_summary": ducky2_payload or {},
        "final_goal_count": len(final_goals),
        "tractability_class": tractability,
        "recommended_specialist_targets": _unique(recommended_targets),
        "recommended_backends": _unique(list(decision.second_pass_backends) + recommended_backends),
        "goals": goal_packets,
    }


def _goal_row(row: dict[str, Any], goal_text: str) -> dict[str, Any]:
    return {
        "theorem_id": str(row.get("theorem_id", "") or ""),
        "last_goal": goal_text,
        "goal_state": goal_text,
        "last_goal_bucket": classify_goal_bucket(goal_text),
        "reasoning_gap_family": str(row.get("reasoning_gap_family", "") or ""),
        "residual_bucket": str(row.get("residual_bucket", "") or ""),
        "difficulty_band": str(row.get("difficulty_band", "") or ""),
        "goals_closed": int(row.get("goals_closed", 0) or 0),
        "goals_remaining": int(row.get("goals_remaining", 0) or 0),
        "attempts": int(row.get("attempts", 0) or 0),
        "lane_sequence": str(row.get("lane_sequence", "") or ""),
        "search_pathology_tags": list(row.get("search_pathology_tags") or []),
        "remaining_goals_snapshot": [sanitize_goal_text(goal_text)],
    }
