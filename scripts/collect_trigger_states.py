"""EXP-050: Apply Trigger Dataset collector.

Runs proof search on the Mathlib benchmark with collect_trace=True and
extracts candidate trigger states — goal states where the apply lane should
or should not have been invoked.

A trigger state is recorded at four checkpoints in the search trace:
  post_ib_fail    — after interleaved_bootstrap failed on a goal
  post_rw         — after cosine_rw made progress (residual goal)
  post_auto       — after automation closed subgoals leaving residue
  mid_search      — any open goal reached after step >= mid_search_threshold

For each state the ExecSelector is probed: does it have any LeanAccepted
candidate in the top-k pool? That probe result is the trigger label:
  trigger=1 — apply selector has ≥1 LeanAccepted candidate
  trigger=0 — no LeanAccepted candidate in pool

Output schema (one row per candidate trigger state):
  theorem_id, step, source_kind, search_stage, lane_provenance,
  goal_state, open_goal_count, recent_lanes,
  cosine_top1_accepted, selector_top1_accepted,
  trigger_label,
  theorem_success (did this theorem eventually get proved?)

Usage:
    python -m scripts.collect_trigger_states \\
        --config configs/wayfinder.yaml \\
        --checkpoint models/NAV-004_step5000.pt \\
        --selector models/apply_exec_selector_v1.pt \\
        --theorems data/mathlib_benchmark_50.jsonl \\
        --lean-project data/lean_project/ \\
        --db data/proof_network_v3.db \\
        --output data/apply_trigger_train.jsonl \\
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scripts.run_benchmark import (  # type: ignore[attr-defined]
    _resolve_initial_goal,
    load_benchmark_theorems,
)
from src.hard_data_tags import (
    canonicalize_theorem_id,
    classify_start_failure_family,
    start_failure_tags,
)
from src.lean_context_ir import extract_context_ir
from src.lean_interface import LeanConfig, LeanKernel
from src.nav_model_factory import load_navigational_checkpoint
from src.proof_network import get_accessible_premises
from src.proof_search import Pipeline, SearchConfig, search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MID_SEARCH_THRESHOLD = 5  # steps before a goal becomes a mid_search state
_BINDER_RE = re.compile(r"[({⦃][^:(){}⦃⦄]*:\s*[^(){}⦃⦄]+[)}⦄]")
_TYPECLASS_RE = re.compile(r"\[[^\]]+]")
_CONTEXT_FEATURE_KEYS = (
    "open",
    "open_scoped",
    "variable",
    "universe",
    "local_notation",
    "scoped_notation",
    "notation",
    "local_attribute",
    "include",
    "omit",
)


# ---------------------------------------------------------------------------
# ExecSelector probe — same architecture as train_apply_exec_selector.py
# ---------------------------------------------------------------------------


def _load_exec_selector(selector_path: str) -> tuple[Any, Any] | None:
    """Load ExecSelector + encoder. Returns (selector, encoder) or None."""
    if not selector_path or not Path(selector_path).exists():
        return None
    try:
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer

        class _ExecSelector(nn.Module):
            def __init__(self, emb_dim: int, hidden: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(emb_dim * 2 + 2, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, 1),
                )

            def score(self, x: "torch.Tensor") -> "torch.Tensor":
                return torch.sigmoid(self.net(x).squeeze(-1))

        ckpt = torch.load(selector_path, map_location="cpu", weights_only=False)
        sel = _ExecSelector(ckpt.get("emb_dim", 384), ckpt.get("hidden", 256))
        sel.load_state_dict(ckpt["model_state_dict"])
        sel.eval()
        enc = SentenceTransformer(ckpt.get("encoder", "all-MiniLM-L6-v2"))
        logger.info("Loaded ExecSelector: %s", selector_path)
        return sel, enc
    except Exception as e:
        logger.warning("ExecSelector load failed: %s", e)
        return None


@dataclass
class SelectorProbeResult:
    """Rich probe result from ExecSelector over a candidate pool."""

    cosine_top1_score_positive: bool = False
    selector_top1_score_positive: bool = False
    selector_top1_candidate: str = ""
    selector_top1_score: float = 0.0
    cosine_top1_candidate: str = ""
    num_candidates: int = 0
    # Per-candidate selector scores, sorted by selector score descending
    ranked_candidates: list[tuple[str, float]] = field(default_factory=list)


def _probe_selector(
    goal: str,
    candidates: list[str],
    selector: Any,
    encoder: Any,
) -> SelectorProbeResult:
    """Score all candidates via ExecSelector. Returns rich probe result.

    Candidates are assumed to be cosine-ranked (index 0 = cosine top-1).
    The selector re-ranks by P(executable|goal, candidate).
    """
    empty = SelectorProbeResult(num_candidates=len(candidates))
    if not candidates:
        return empty
    try:
        import torch

        texts = [goal] + candidates
        embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        goal_emb = embs[0]
        cand_embs = embs[1:]
        cosine_scores = (cand_embs @ goal_emb).tolist()

        n = len(candidates)
        goal_rep = torch.tensor(goal_emb, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        cand_rep = torch.tensor(cand_embs, dtype=torch.float32)
        cos_t = torch.tensor(cosine_scores, dtype=torch.float32).unsqueeze(1)
        filter_t = torch.ones(n, 1, dtype=torch.float32)
        feat = torch.cat([goal_rep, cand_rep, cos_t, filter_t], dim=1)

        with torch.no_grad():
            scores = selector.score(feat).tolist()

        # Rank by selector score descending
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        sel_top1_name, sel_top1_score = ranked[0]

        return SelectorProbeResult(
            cosine_top1_score_positive=float(scores[0]) > 0.5,
            selector_top1_score_positive=sel_top1_score > 0.5,
            selector_top1_candidate=sel_top1_name,
            selector_top1_score=round(sel_top1_score, 4),
            cosine_top1_candidate=candidates[0],
            num_candidates=n,
            ranked_candidates=[(name, round(s, 4)) for name, s in ranked],
        )
    except Exception:
        return empty


def _namespace_prefix(theorem_id: str) -> str:
    return theorem_id.split(".")[0] if "." in theorem_id else "(root)"


def _goal_shape_features(goal_text: str) -> dict[str, int]:
    """Cheap structural summary of a theorem type / goal string."""
    text = goal_text or ""
    return {
        "char_len": len(text),
        "token_len": len(text.split()),
        "binder_count": len(_BINDER_RE.findall(text)),
        "forall_count": text.count("∀"),
        "exists_count": text.count("∃"),
        "arrow_count": text.count("→") + text.count("->"),
        "iff_count": text.count("↔"),
        "eq_count": text.count("="),
        "neq_count": text.count("≠"),
        "and_count": text.count("∧"),
        "or_count": text.count("∨"),
        "not_count": text.count("¬"),
        "typeclass_count": len(_TYPECLASS_RE.findall(text)),
        "type_count": text.count("Type") + text.count("Sort"),
        "set_count": text.count("Set."),
    }


def _resolve_theorem_site_metadata(lean: LeanKernel, theorem_id: str) -> dict[str, Any]:
    """Recover declaring module/source line and ContextIR features when possible."""
    metadata: dict[str, Any] = {
        "module": "",
        "lean_path": "",
        "theorem_line": 0,
        "theorem_type": "",
        "context_features": {k: 0 for k in _CONTEXT_FEATURE_KEYS},
        "context_unsupported_kinds": [],
    }
    if lean._backend != "pantograph" or lean._server is None:
        return metadata

    canonical_id = canonicalize_theorem_id(theorem_id)
    try:
        info = lean._server.env_inspect(name=canonical_id or theorem_id)
    except Exception:
        return metadata
    if not isinstance(info, dict):
        return metadata

    module = info.get("module", "")
    src_start = info.get("sourceStart")
    thm_type_obj = info.get("type")
    if isinstance(thm_type_obj, dict):
        metadata["theorem_type"] = thm_type_obj.get("pp", "")
    elif thm_type_obj:
        metadata["theorem_type"] = str(thm_type_obj)

    if not module or not isinstance(src_start, dict):
        return metadata

    theorem_line = int(src_start.get("line", 0) or 0)
    metadata["module"] = module
    metadata["theorem_line"] = theorem_line

    project_root = Path(lean.config.project_root)
    module_path = Path(*module.split(".")).with_suffix(".lean")
    candidate_paths = [
        project_root / ".lake" / "packages" / "mathlib" / module_path,
        project_root / module_path,
    ]
    lean_path = next((p for p in candidate_paths if p.exists()), None)
    if lean_path is None or theorem_line <= 0:
        return metadata

    metadata["lean_path"] = str(lean_path)

    try:
        ctx = extract_context_ir(lean_path, theorem_line)
        counts = ctx.feature_counts()
        metadata["context_features"] = {
            key: int(counts.get(key, 0)) for key in _CONTEXT_FEATURE_KEYS
        }
        metadata["context_unsupported_kinds"] = sorted(
            {directive.kind for directive in ctx.unsupported}
        )
    except Exception:
        pass

    return metadata


def _build_goal_start_failure_row(
    thm: dict[str, Any],
    lean: LeanKernel,
    repaired_goal: str | None = None,
    repair_result: Any | None = None,
) -> dict[str, Any]:
    """Structured row for a theorem whose initial goal could not be created directly."""
    theorem_id = canonicalize_theorem_id(str(thm.get("theorem_id", "")))
    site = _resolve_theorem_site_metadata(lean, theorem_id)
    feedback = getattr(lean, "_last_goal_feedback", None)
    theorem_type = site.get("theorem_type") or thm.get("goal_state", "")
    raw_goal_state = thm.get("goal_state", "")
    failure_category = feedback.category if feedback is not None else "goal_creation_fail"
    tags = start_failure_tags(
        failure_category=failure_category,
        goal_text=str(theorem_type or raw_goal_state),
        module=str(site["module"]),
        theorem_line=int(site["theorem_line"] or 0),
        context_features=site["context_features"],
        context_unsupported_kinds=site["context_unsupported_kinds"],
    )

    row = {
        "row_type": "goal_start_failure",
        "theorem_id": theorem_id,
        "source": thm.get("source", ""),
        "source_kind": "goal_start_failure",
        "search_stage": "goal_start",
        "lane_provenance": "goal_creation",
        "goal_state": theorem_type,
        "raw_goal_state": raw_goal_state,
        "theorem_type": theorem_type,
        "namespace_prefix": _namespace_prefix(theorem_id),
        "goal_shape_features": _goal_shape_features(theorem_type or raw_goal_state),
        "context_features": site["context_features"],
        "context_unsupported_kinds": site["context_unsupported_kinds"],
        "module": site["module"],
        "lean_path": site["lean_path"],
        "theorem_line": site["theorem_line"],
        "goal_start_status": (
            "recovered_via_file_context" if repaired_goal is not None else "failed"
        ),
        "failure_category": failure_category,
        "start_failure_family": classify_start_failure_family(
            failure_category=failure_category,
            goal_text=str(theorem_type or raw_goal_state),
            module=str(site["module"]),
            theorem_line=int(site["theorem_line"] or 0),
            context_features=site["context_features"],
            context_unsupported_kinds=site["context_unsupported_kinds"],
        ),
        "start_failure_tags": tags,
        "feedback": feedback.to_dict() if feedback is not None else None,
        "repair_attempted": repair_result is not None,
        "repair_success": repaired_goal is not None,
        "repair_goal_state": repaired_goal,
        "repair_tier_used": getattr(repair_result, "tier_used", ""),
        "repair_failure_category": getattr(repair_result, "failure_category", ""),
        "repair_feedback": (
            repair_result.feedback.to_dict()
            if repair_result is not None and getattr(repair_result, "feedback", None) is not None
            else None
        ),
        "label_source": "goal_start_feedback",
        "trigger_label": None,
        "theorem_success": False,
    }
    return row


# ---------------------------------------------------------------------------
# Trigger state extraction from step trace
# ---------------------------------------------------------------------------


def _classify_search_stage(
    step: int,
    lane: str,
    recent_lanes: list[str],
) -> str | None:
    """Classify a step in the trace as a trigger checkpoint, or None."""
    # All trigger checkpoints fire on no-progress steps (lane == "") — we want
    # to record the goal state the search is stuck on, not the goal that was
    # just closed by a successful lane.
    if lane != "":
        return None

    # post_ib_fail: IB was the most recent progress lane and we're now stuck
    if recent_lanes and recent_lanes[-1] in (
        "interleaved_bootstrap",
        "interleaved_bootstrap/simp",
        "interleaved_bootstrap/simp_aesop",
    ):
        return "post_ib_fail"
    # post_rw: cosine_rw was the most recent progress lane — residual goal after rw
    if recent_lanes and recent_lanes[-1] in ("cosine_rw", "cosine_rw_seq"):
        return "post_rw"
    # post_auto: automation was the most recent progress lane — residual after partial close
    if recent_lanes and recent_lanes[-1] in ("automation",):
        return "post_auto"
    # mid_search: no recent progress context, but we've spent enough steps
    if step >= _MID_SEARCH_THRESHOLD:
        return "mid_search"
    return None


def extract_trigger_states(
    theorem_id: str,
    step_trace: list[dict],
    theorem_success: bool,
    accessible_premises: list[str],
    selector: Any,
    selector_enc: Any,
    sentence_encoder: Any,
    lean: Any | None = None,
    probe_lean: bool = False,
    probe_k: int = 5,
) -> list[dict[str, Any]]:
    """Extract trigger state rows from a single theorem's step trace."""
    rows: list[dict[str, Any]] = []
    if not step_trace or not accessible_premises:
        return rows

    # Rolling window of recent lanes (last 5 progress steps)
    recent_lanes: deque[str] = deque(maxlen=5)

    seen_stage_goals: set[tuple[str, str]] = set()  # (stage, goal_text)

    for entry in step_trace:
        step = entry.get("step", 0)
        goal = entry.get("goal_before", "")
        lane = entry.get("lane", "")
        progress = entry.get("progress", False)
        open_before = len(entry.get("open_goals_before", [goal] if goal else []))
        stage = _classify_search_stage(step, lane, list(recent_lanes))

        if stage is not None and goal and (stage, goal) not in seen_stage_goals:
            seen_stage_goals.add((stage, goal))

            # Cosine rank candidates for this goal
            candidates: list[str] = []
            if sentence_encoder is not None and accessible_premises:
                try:
                    texts = [goal] + accessible_premises
                    embs = sentence_encoder.encode(
                        texts, show_progress_bar=False, normalize_embeddings=True
                    )
                    scores = (embs[1:] @ embs[0]).tolist()
                    ranked = sorted(zip(scores, accessible_premises), reverse=True)[:20]
                    candidates = [name for _, name in ranked]
                except Exception:
                    pass

            probe = (
                _probe_selector(goal, candidates, selector, selector_enc)
                if selector is not None and candidates
                else SelectorProbeResult()
            )

            # Live Lean probes: test top-k candidates to compute can_apply
            # and selector_top1_accepted. Probe order: selector top-1 first,
            # then cosine top-1 (if different), then next-best by selector score.
            can_apply: int | None = None
            selector_top1_accepted: int | None = None
            accepted_candidates: list[str] = []
            best_feedback_category: str = ""
            num_probed = 0
            label_source = "selector_proxy"

            if probe_lean and lean is not None and candidates:
                label_source = "lean_probe"
                # Build probe order: selector top-1, cosine top-1, then rest by selector score
                probe_order: list[str] = []
                if probe.selector_top1_candidate:
                    probe_order.append(probe.selector_top1_candidate)
                if probe.cosine_top1_candidate and probe.cosine_top1_candidate not in probe_order:
                    probe_order.append(probe.cosine_top1_candidate)
                for name, _score in probe.ranked_candidates:
                    if name not in probe_order:
                        probe_order.append(name)
                    if len(probe_order) >= probe_k:
                        break

                for cand_name in probe_order:
                    try:
                        tac = f"apply {cand_name}"
                        result = lean.try_tactic(goal, tac)
                        num_probed += 1
                        if result.success:
                            accepted_candidates.append(cand_name)
                            if not best_feedback_category:
                                best_feedback_category = "accepted"
                        elif not best_feedback_category:
                            best_feedback_category = getattr(result, "category", "tactic_fail")
                    except Exception:
                        num_probed += 1

                can_apply = 1 if len(accepted_candidates) > 0 else 0
                selector_top1_accepted = (
                    1 if probe.selector_top1_candidate in accepted_candidates else 0
                )
            else:
                # Offline proxy — not ground truth
                can_apply = 1 if probe.selector_top1_score_positive else 0
                selector_top1_accepted = None

            rows.append(
                {
                    "row_type": "trigger_state",
                    "theorem_id": theorem_id,
                    "step": step,
                    "source_kind": "search_residual",
                    "search_stage": stage,
                    "lane_provenance": lane or "none",
                    "goal_state": goal,
                    "open_goal_count": open_before,
                    "recent_lanes": list(recent_lanes),
                    "namespace_prefix": _namespace_prefix(theorem_id),
                    "goal_shape_features": _goal_shape_features(goal),
                    "cosine_top1_score_positive": probe.cosine_top1_score_positive,
                    "selector_top1_score_positive": probe.selector_top1_score_positive,
                    "selector_top1_candidate": probe.selector_top1_candidate,
                    "selector_top1_score": probe.selector_top1_score,
                    "num_candidates_considered": probe.num_candidates,
                    "num_candidates_probed": num_probed,
                    "num_accepted_in_pool": len(accepted_candidates),
                    "accepted_candidates": accepted_candidates,
                    "best_feedback_category": best_feedback_category,
                    "label_source": label_source,
                    "can_apply": can_apply,
                    "selector_top1_accepted": selector_top1_accepted,
                    "theorem_success": theorem_success,
                }
            )

        if progress and lane:
            recent_lanes.append(lane)

    return rows


# ---------------------------------------------------------------------------
# Build infra
# ---------------------------------------------------------------------------


def _build_pipeline(config: dict, checkpoint_path: str, device: str) -> Pipeline:
    _, modules = load_navigational_checkpoint(Path(checkpoint_path), config, device)
    return Pipeline(
        encoder=modules["encoder"],  # type: ignore[arg-type]
        analyzer=modules["analyzer"],  # type: ignore[arg-type]
        bridge=modules["bridge"],  # type: ignore[arg-type]
        navigator=modules["navigator"],  # type: ignore[arg-type]
    )


def _build_lean(config: dict) -> LeanKernel:
    lean_section = config.get("lean", {})
    search_cfg = config.get("search", {})
    lean_cfg = LeanConfig(
        backend=lean_section.get("backend", "stub"),
        hammer_timeout=search_cfg.get("hammer_timeout", 60),
        project_root=lean_section.get("project_root", ""),
        imports=lean_section.get("imports", ["Init"]),
    )
    return LeanKernel(lean_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--selector", default="models/apply_exec_selector_v1.pt")
    parser.add_argument("--theorems", default="data/mathlib_benchmark_50.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-imports", nargs="+", default=["Mathlib"])
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--output", default="data/apply_trigger_train.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N theorems (for parallel sharding)"
    )
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--probe-lean",
        action="store_true",
        help="Probe top-k candidates via live Lean for ground-truth trigger labels "
        "(slower but avoids circular selector-proxy labeling)",
    )
    parser.add_argument(
        "--probe-k",
        type=int,
        default=5,
        help="Number of candidates to probe via Lean per trigger state",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("evaluation", {})["benchmark_theorems"] = args.theorems
    config["evaluation"].pop("mathlib_test_split", None)
    config.setdefault("lean", {})["project_root"] = args.lean_project
    config.setdefault("lean", {})["backend"] = args.backend
    config.setdefault("lean", {})["imports"] = args.lean_imports
    config.setdefault("data", {})["proof_network_db"] = args.db

    pipeline = _build_pipeline(config, args.checkpoint, args.device)
    lean = _build_lean(config)
    conn = sqlite3.connect(args.db)

    id_to_name = {eid: n for eid, n in conn.execute("SELECT id, name FROM entities")}
    name_to_id = {v: k for k, v in id_to_name.items()}

    # Pre-load encoder and selector
    from sentence_transformers import SentenceTransformer

    sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    sel_result = _load_exec_selector(args.selector)
    selector, selector_enc = sel_result if sel_result else (None, None)

    theorems = load_benchmark_theorems(config, None)  # load all, slice below
    if args.offset:
        theorems = theorems[args.offset :]
    if args.limit:
        theorems = theorems[: args.limit]
    logger.info("Loaded %d theorems (offset=%d)", len(theorems), args.offset)

    cfg = SearchConfig(
        budget=args.budget,
        hammer_delegation=True,
        accessible_premises=True,
        search_mode="no_learned",
        temporal_mode="off",
        interleaved_bootstrap_enabled=True,
        cosine_apply_enabled=False,  # off — we harvest trigger states, not execute them
        collect_trace=True,
    )

    if lean._backend == "pantograph":
        lean._ensure_server()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows_total = 0
    trigger_rows_total = 0
    trigger_positive = 0
    goal_start_failures = 0
    goal_start_repairs = 0

    with open(args.output, "w") as fout:
        for i, thm in enumerate(theorems):
            theorem_id = thm.get("theorem_id", "")
            logger.info("[%d/%d] %s", i + 1, len(theorems), theorem_id)

            accessible_id = name_to_id.get(theorem_id)
            initial_goal = _resolve_initial_goal(thm, lean)
            if initial_goal is None:
                repair_result = None
                repaired_goal = None
                try:
                    repair_result = lean.goal_via_file_context(
                        theorem_id,
                        "",
                        [],
                        "",
                        project_root=args.lean_project,
                    )
                    if repair_result.success:
                        repaired_goal = repair_result.goal_state
                except Exception:
                    repair_result = None

                start_row = _build_goal_start_failure_row(
                    thm,
                    lean,
                    repaired_goal=repaired_goal,
                    repair_result=repair_result,
                )
                fout.write(json.dumps(start_row) + "\n")
                fout.flush()
                rows_total += 1
                goal_start_failures += 1
                if repaired_goal is None:
                    logger.warning("Could not create goal for %s — skipping", theorem_id)
                    continue
                goal_start_repairs += 1
                logger.info("Recovered goal for %s via file_context replay", theorem_id)
                initial_goal = repaired_goal

            try:
                sr = search(
                    theorem_id=theorem_id,
                    initial_goal=initial_goal,
                    pipeline=pipeline,
                    conn=conn,
                    lean=lean,
                    config=cfg,
                    accessible_theorem_id=accessible_id,
                    sentence_encoder=sentence_encoder,
                    exec_apply_selector=selector,
                    exec_apply_encoder=selector_enc,
                )
            except Exception as exc:
                logger.warning("Search error on %s: %s", theorem_id, exc)
                continue

            # Get accessible premise names for probe
            premise_names: list[str] = []
            if accessible_id is not None:
                premise_ids = get_accessible_premises(conn, accessible_id)
                premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]

            trigger_rows = extract_trigger_states(
                theorem_id=theorem_id,
                step_trace=sr.step_trace,
                theorem_success=sr.success,
                accessible_premises=premise_names,
                selector=selector,
                selector_enc=selector_enc,
                sentence_encoder=sentence_encoder,
                lean=lean if args.probe_lean else None,
                probe_lean=args.probe_lean,
                probe_k=args.probe_k,
            )

            for row in trigger_rows:
                fout.write(json.dumps(row) + "\n")
                rows_total += 1
                trigger_rows_total += 1
                trigger_positive += row.get("can_apply") or 0

            if trigger_rows:
                fout.flush()

    pos_pct = 100 * trigger_positive / max(trigger_rows_total, 1)
    print("\n" + "=" * 60)
    print("EXP-050: Apply Trigger Dataset")
    print("=" * 60)
    print(f"  theorems processed : {len(theorems)}")
    print(f"  total rows         : {rows_total}")
    print(f"  trigger rows       : {trigger_rows_total}")
    print(f"  trigger=1 (pos)    : {trigger_positive}  ({pos_pct:.1f}%)")
    print(f"  trigger=0 (neg)    : {trigger_rows_total - trigger_positive}")
    print(f"  goal-start fails   : {goal_start_failures}")
    print(f"  repaired starts    : {goal_start_repairs}")
    print(f"  output             : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
