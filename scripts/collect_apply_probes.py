"""
Scaled probe collector for the apply executable-validity dataset.

Samples from canonical_residual_train.jsonl (family=="apply"), creates
goal states via Pantograph (Tier A for step=0, Tier B for step>0), then
probes all scope-filtered candidates against Lean.

Output schema (one row per probed candidate):
  theorem_full_name, file_path, step_index, source_split, source_kind,
  canonical_apply, goal_state, goal_target_head, goal_shape,
  candidate, candidate_type_pp,
  cosine_score, cosine_rank, passed_static_filter,
  feedback_category, feedback_stage,
  accepted, closed, executable  (binary: accepted_with_goals|closed -> 1)

Usage:
    python -m scripts.collect_apply_probes \\
        --train data/canonical/canonical_residual_train.jsonl \\
        --db    data/proof_network_v3.db \\
        --lean-project data/lean_project/ \\
        --output data/apply_exec_train.jsonl \\
        --limit 2000 \\
        --step0-only          # fast mode: skip prefix replay
        --restart-every 64

Drop --step0-only for the full distribution run (prefix replay required).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.lean_interface import LeanConfig, LeanKernel, ServerCrashError
from src.proof_network import get_accessible_premises

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Static compatibility filter
# ---------------------------------------------------------------------------

_SHAPE_COMPAT: dict[str, set[str]] = {
    "eq":     {"eq", "iff", "le", "ge", "dvd", "prop"},
    "iff":    {"iff", "eq", "prop"},
    "le":     {"le", "lt", "eq", "prop"},
    "lt":     {"le", "lt", "prop"},
    "ge":     {"ge", "gt", "eq", "prop"},
    "gt":     {"ge", "gt", "prop"},
    "mem":    {"mem", "subset", "prop"},
    "subset": {"subset", "mem", "prop"},
    "dvd":    {"dvd", "eq", "prop"},
    "and":    {"and", "prop"},
    "or":     {"or", "prop"},
    "not":    {"not", "prop"},
    "prop":   set(),
    "other":  set(),
}

_AUTO_HEADS = {"True", "False", "trivial"}
_BINDER_RE  = re.compile(r"^(∀\s*[\{\[\(].*?[\}\]\)],?\s*)+")
_HEAD_RE    = re.compile(r"⊢\s*(\S+)")


def _strip_binders(pp: str) -> str:
    return _BINDER_RE.sub("", pp.strip()).strip()


def _conclusion_of(pp: str) -> str:
    stripped = _strip_binders(pp)
    if "→" in stripped:
        return stripped.rsplit("→", 1)[-1].strip()
    return stripped


def _prop_shape(conclusion: str) -> str:
    for sym, shape in [
        ("=", "eq"), ("↔", "iff"), ("≤", "le"), ("≥", "ge"),
        ("<", "lt"), (">", "gt"), ("∈", "mem"), ("⊆", "subset"),
        ("∣", "dvd"), ("∧", "and"), ("∨", "or"), ("¬", "not"),
    ]:
        if sym in conclusion[:60]:
            return shape
    return "other"


def _goal_head(goal_str: str) -> str:
    m = _HEAD_RE.search(goal_str)
    if not m:
        return ""
    return m.group(1).rstrip(".,;:")


def _goal_shape(goal_str: str) -> str:
    m = _HEAD_RE.search(goal_str)
    if not m:
        return "other"
    tail = goal_str[m.end():].strip()
    return _prop_shape(m.group(1) + " " + tail)


def _is_compatible(g_head: str, g_shape: str, cand_pp: str) -> bool:
    if g_head in _AUTO_HEADS:
        return False
    concl = _conclusion_of(cand_pp)
    cand_shape = _prop_shape(concl)
    allowed = _SHAPE_COMPAT.get(g_shape, set())
    if not allowed:
        return True
    if not _SHAPE_COMPAT.get(cand_shape, set()):
        return True
    return cand_shape in allowed


# ---------------------------------------------------------------------------
# Candidate type cache
# ---------------------------------------------------------------------------

_type_cache: dict[str, str | None] = {}


def _get_decl_type(kernel: LeanKernel, name: str) -> str | None:
    if name in _type_cache:
        return _type_cache[name]
    if kernel._server is None:
        _type_cache[name] = None
        return None
    try:
        info = kernel._server.env_inspect(
            name=name, print_value=False, print_dependency=False
        )
        pp = getattr(info, "type", None)
        if pp is None and isinstance(info, dict):
            pp = info.get("type")
        result = str(pp) if pp is not None else None
    except Exception:
        result = None
    _type_cache[name] = result
    return result


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

def _outcome_class(accepted: bool, closed: bool, fb_cat: str) -> str:
    if closed:
        return "closed"
    if accepted:
        return "accepted_with_goals"
    if fb_cat in {
        "unification_mismatch", "typeclass_missing",
        "unknown_identifier", "parse_error", "other",
    }:
        return fb_cat
    return "other"


# ---------------------------------------------------------------------------
# Cosine scoring
# ---------------------------------------------------------------------------

def _cosine_scores(
    encoder: SentenceTransformer,
    goal_str: str,
    candidates: list[str],
) -> list[float]:
    if not candidates:
        return []
    texts = [goal_str] + candidates
    embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return (embs[1:] @ embs[0]).tolist()


# ---------------------------------------------------------------------------
# Per-example processing
# ---------------------------------------------------------------------------

@dataclass
class ProbeRow:
    # Identity
    theorem_full_name: str
    file_path: str
    step_index: int
    # Provenance — locked schema for v2 selector
    source_split: str         # "train" | "eval"
    source_kind: str          # "canonical_step0" | "canonical_midstep" |
                              # "post_bootstrap_residual" | "search_residual"
    search_stage: str         # "n/a" | "initial" | "post_intro" | "post_simp" |
                              # "post_aesop" | "mid_search"
    lane_provenance: str      # "canonical" | "bootstrap" | "hammer" | "learned"
    # Semantic IR (from canonical data)
    goal_shape_ir: dict       # raw goal_shape_ir dict from training data
    trigger_profile_ir: dict  # raw trigger_profile_ir dict from training data
    # Goal
    canonical_apply: str
    goal_state: str
    goal_target_head: str
    goal_shape: str
    # Candidate
    candidate: str
    candidate_type_pp: str
    cosine_score: float
    cosine_rank: int
    passed_static_filter: bool
    # Outcome
    feedback_category: str
    feedback_stage: str
    accepted: bool
    closed: bool
    executable: int           # 1 if accepted or closed

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_full_name": self.theorem_full_name,
            "file_path": self.file_path,
            "step_index": self.step_index,
            "source_split": self.source_split,
            "source_kind": self.source_kind,
            "search_stage": self.search_stage,
            "lane_provenance": self.lane_provenance,
            "goal_shape_ir": self.goal_shape_ir,
            "trigger_profile_ir": self.trigger_profile_ir,
            "canonical_apply": self.canonical_apply,
            "goal_state": self.goal_state,
            "goal_target_head": self.goal_target_head,
            "goal_shape": self.goal_shape,
            "candidate": self.candidate,
            "candidate_type_pp": self.candidate_type_pp,
            "cosine_score": round(self.cosine_score, 5),
            "cosine_rank": self.cosine_rank,
            "passed_static_filter": self.passed_static_filter,
            "feedback_category": self.feedback_category,
            "feedback_stage": self.feedback_stage,
            "accepted": self.accepted,
            "closed": self.closed,
            "executable": self.executable,
        }


def process_example(
    ex: dict[str, Any],
    kernel: LeanKernel,
    encoder: SentenceTransformer,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    project_root: str,
    top_n: int = 20,
) -> list[ProbeRow]:
    name = ex["theorem_full_name"]
    step = ex.get("step_index", 0)
    goal_str_raw = ex.get("goal_state_before", "")
    canonical_apply = ex.get("annotated_premise", "")
    g_head = _goal_head(goal_str_raw)
    g_shape = _goal_shape(goal_str_raw)
    source_kind = "canonical_step0" if step == 0 else "canonical_midstep"
    goal_shape_ir = ex.get("goal_shape_ir") or {}
    trigger_profile_ir = ex.get("trigger_profile_ir") or {}

    # ---- 1. Create goal state via file-context replay ----
    # goal_state_before is the full Pantograph display (hypotheses + conclusion)
    # and cannot be fed directly to goal_start(). Use goal_via_file_context for
    # all steps so the server sees the real theorem context.
    replay = kernel.goal_via_file_context(
        theorem_full_name=name,
        file_path=ex.get("file_path", ""),
        prefix_tactics=ex.get("prefix_tactics", []) if step > 0 else [],
        expected_goal=goal_str_raw,
        project_root=project_root,
    )
    if not replay.success:
        return []
    goal_str = replay.goal_state
    goal_id = replay.goal_id

    # ---- 2. Accessible premises ----
    theorem_id = name_to_id.get(name)
    if theorem_id is None:
        return []
    premise_ids = get_accessible_premises(conn, theorem_id)
    premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]
    if not premise_names:
        return []

    # ---- 3. Cosine rank ----
    scores = _cosine_scores(encoder, goal_str, premise_names)
    ranked = sorted(zip(scores, premise_names), reverse=True)[:top_n]

    # ---- 4. Probe each candidate ----
    rows: list[ProbeRow] = []
    for cosine_rank, (score, cand) in enumerate(ranked):
        cand_pp = _get_decl_type(kernel, cand) or ""
        passed = _is_compatible(g_head, g_shape, cand_pp) if cand_pp else True

        tac = f"apply {cand}"
        try:
            result = kernel.try_tactic(goal_str, tac, goal_id=goal_id)
            accepted = result.success
            closed = result.success and not result.new_goals
            fb = result.feedback
            fb_cat = fb.category if fb else "other"
            fb_stage = fb.stage if fb else "tactic_exec"
        except ServerCrashError:
            raise
        except Exception:
            accepted, closed = False, False
            fb_cat, fb_stage = "other", "tactic_exec"

        executable = 1 if (accepted or closed) else 0

        rows.append(ProbeRow(
            theorem_full_name=name,
            file_path=ex.get("file_path", ""),
            step_index=step,
            source_split="train",
            source_kind=source_kind,
            search_stage="n/a",
            lane_provenance="canonical",
            goal_shape_ir=goal_shape_ir,
            trigger_profile_ir=trigger_profile_ir,
            canonical_apply=canonical_apply,
            goal_state=goal_str,
            goal_target_head=g_head,
            goal_shape=g_shape,
            candidate=cand,
            candidate_type_pp=cand_pp[:256],
            cosine_score=score,
            cosine_rank=cosine_rank,
            passed_static_filter=passed,
            feedback_category=_outcome_class(accepted, closed, fb_cat),
            feedback_stage=fb_stage,
            accepted=accepted,
            closed=closed,
            executable=executable,
        ))

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",
        default="data/canonical/canonical_residual_train.jsonl")
    parser.add_argument("--db",
        default="data/proof_network_v3.db")
    parser.add_argument("--lean-project",
        default="data/lean_project/")
    parser.add_argument("--output",
        default="data/apply_exec_train.jsonl")
    parser.add_argument("--limit", type=int, default=0,
        help="Max examples to process (0 = no limit)")
    parser.add_argument("--step0-only", action="store_true",
        help="Restrict to step_index==0 (fast: no prefix replay)")
    parser.add_argument("--top-n", type=int, default=20,
        help="Number of cosine-ranked candidates to probe per goal")
    parser.add_argument("--restart-every", type=int, default=64,
        help="Periodic server restart every N examples")
    parser.add_argument("--resume", action="store_true",
        help="Skip already-written theorem_full_name+step_index pairs")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done: set[tuple[str, int]] = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["theorem_full_name"], r["step_index"]))
        logger.info("Resume: %d theorem+step pairs already done", len(done))

    # DB
    conn = sqlite3.connect(args.db)
    id_to_name = {
        eid: n for eid, n in conn.execute("SELECT id, name FROM entities")
    }
    name_to_id = {v: k for k, v in id_to_name.items()}
    logger.info("DB: %d entities", len(id_to_name))

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    config = LeanConfig(
        backend="pantograph",
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    kernel = LeanKernel(config=config)

    # Load examples
    examples: list[dict[str, Any]] = []
    with open(args.train) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("family") != "apply":
                continue
            if args.step0_only and ex.get("step_index", 0) != 0:
                continue
            examples.append(ex)

    if args.limit:
        examples = examples[:args.limit]

    logger.info("Processing %d apply examples", len(examples))

    mode = "a" if args.resume else "w"
    rows_total = 0
    exec_total = 0
    t_start = time.time()

    with open(args.output, mode) as fout:
        for i, ex in enumerate(examples):
            name = ex["theorem_full_name"]
            step = ex.get("step_index", 0)

            if args.resume and (name, step) in done:
                continue

            if args.restart_every and i > 0 and i % args.restart_every == 0:
                logger.info("Periodic restart at example %d", i)
                kernel._restart_server()

            logger.info("[%d/%d] %s (step %d)", i + 1, len(examples), name, step)

            try:
                probe_rows = process_example(
                    ex, kernel, encoder, conn, id_to_name, name_to_id,
                    args.lean_project, top_n=args.top_n,
                )
            except ServerCrashError as exc:
                logger.warning("Server crash on %s step %d: %s", name, step, exc)
                probe_rows = []
            except Exception as exc:
                logger.warning("Error on %s step %d: %s", name, step, exc)
                probe_rows = []

            for row in probe_rows:
                fout.write(json.dumps(row.to_dict()) + "\n")
                rows_total += 1
                exec_total += row.executable

            if probe_rows:
                fout.flush()

    elapsed = time.time() - t_start
    pos_pct = 100 * exec_total / max(rows_total, 1)
    logger.info(
        "Done: %d rows, %d executable (%.1f%%), %.1fs",
        rows_total, exec_total, pos_pct, elapsed,
    )

    print("\n" + "=" * 60)
    print("apply_exec_train dataset summary")
    print("=" * 60)
    print(f"  examples processed : {len(examples)}")
    print(f"  probe rows written : {rows_total}")
    print(f"  executable (pos)   : {exec_total}  ({pos_pct:.1f}%)")
    print(f"  non-executable     : {rows_total - exec_total}")
    print(f"  elapsed            : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
