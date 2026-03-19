"""EXP-048: Live Pantograph benchmark for ExecSelector v1.

Conditions (shared started-goal subset):
  cosine_top1          — raw cosine top-1, no filter
  filter_cosine_top1   — cosine top-1 after compatibility filter
  selector_top1        — ExecSelector v1 top-1 over filtered candidates
  best_filtered_top5   — oracle: best of filtered top-5 via Lean verify

Primary metric: LeanAccepted|started
Verifies the offline 37.8% vs 19.5% lift holds against live Lean.

Usage:
    python -m scripts.run_exp048_exec_selector \\
        --db    data/proof_network_v3.db \\
        --eval  data/canonical/canonical_residual_eval.jsonl \\
        --lean-project data/lean_project/ \\
        --selector models/apply_exec_selector_v1.pt \\
        --output runs/exp048_results.jsonl
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

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.lean_interface import LeanConfig, LeanKernel
from src.proof_network import get_accessible_premises

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

try:
    from src.lean_interface import ServerCrashError  # type: ignore[attr-defined]
except ImportError:
    class ServerCrashError(Exception):  # type: ignore[no-redef]
        pass

# ---------------------------------------------------------------------------
# ExecSelector (must match train_apply_exec_selector.py architecture)
# ---------------------------------------------------------------------------

class ExecSelector(nn.Module):
    def __init__(self, emb_dim: int = 384, hidden: int = 256) -> None:
        super().__init__()
        in_dim = emb_dim * 2 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


def load_selector(
    ckpt_path: str, device: torch.device
) -> tuple[ExecSelector, SentenceTransformer, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    emb_dim = ckpt.get("emb_dim", 384)
    hidden  = ckpt.get("hidden", 256)
    encoder_name = ckpt.get("encoder", "all-MiniLM-L6-v2")
    model = ExecSelector(emb_dim=emb_dim, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    encoder = SentenceTransformer(encoder_name)
    logger.info(
        "Loaded ExecSelector (emb=%d, hidden=%d, PR-AUC=%.4f) from %s",
        emb_dim, hidden, ckpt.get("best_val_pr_auc", 0.0), ckpt_path,
    )
    return model, encoder, emb_dim


# ---------------------------------------------------------------------------
# Static compatibility filter (from apply047)
# ---------------------------------------------------------------------------

_SHAPE_COMPAT: dict[str, set[str]] = {
    "eq":     {"eq", "iff", "le", "ge", "dvd", "imp", "prop"},
    "iff":    {"iff", "eq", "imp", "prop"},
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
    "imp":    set(),  # permissive
    "prop":   set(),
    "other":  set(),
}

_AUTO_HEADS = {"True", "False", "trivial"}
_BINDER_RE  = re.compile(r"^(∀\s*[\{\[\(].*?[\}\]\)],?\s*)+")
_HEAD_RE    = re.compile(r"⊢\s*(\S+)")


def _strip_binders(pp: str) -> str:
    return _BINDER_RE.sub("", pp.strip()).strip()


def _conclusion_of(pp: str) -> str:
    s = _strip_binders(pp)
    if " → " in s:
        return s.rsplit(" → ", 1)[-1].strip()
    return s


def _prop_shape(conclusion: str) -> str:
    c = conclusion[:80]
    if " → " in c:
        return "imp"
    for sym, shape in [
        ("=", "eq"), ("↔", "iff"), ("≤", "le"), ("≥", "ge"),
        ("<", "lt"), (">", "gt"), ("∈", "mem"), ("⊆", "subset"),
        ("∣", "dvd"), ("∧", "and"), ("∨", "or"), ("¬", "not"),
    ]:
        if sym in c:
            return shape
    return "other"


def _goal_head(goal_str: str) -> str:
    m = _HEAD_RE.search(goal_str)
    return m.group(1).rstrip(".,;:") if m else ""


def _goal_shape(goal_str: str) -> str:
    m = _HEAD_RE.search(goal_str)
    if not m:
        return "other"
    return _prop_shape(goal_str[m.end():].strip())


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
# Declaration type cache
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
# Tactic helpers (from apply047)
# ---------------------------------------------------------------------------

def try_apply(
    kernel: LeanKernel,
    goal_str: str,
    lemma: str,
    goal_id: int = 0,
) -> tuple[bool, bool, bool, Any]:
    """(accepted, closed_all, was_crash, feedback)"""
    tac = f"apply {lemma}"
    result = kernel.try_tactic(goal_str, tac, goal_id=goal_id)
    fb = result.feedback
    accepted = result.success
    closed = result.success and not result.new_goals
    return accepted, closed, False, fb


# ---------------------------------------------------------------------------
# Per-example result
# ---------------------------------------------------------------------------

@dataclass
class ExampleResult:
    theorem_full_name: str
    file_path: str

    goal_started: bool = False
    failure_category: str = ""
    crash_retries: int = 0

    gold_in_scope: bool = False
    n_accessible: int = 0

    # LeanAccepted per condition
    cosine_accepted: bool = False
    filter_cosine_accepted: bool = False
    selector_accepted: bool = False
    best_filtered_accepted: bool = False

    # Feedback categories
    cosine_feedback_category: str = ""
    filter_cosine_feedback_category: str = ""
    selector_feedback_category: str = ""

    # Chosen candidates
    cosine_candidate: str = ""
    filter_cosine_candidate: str = ""
    selector_candidate: str = ""

    # Selector score for chosen candidate
    selector_score: float = 0.0

    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_full_name": self.theorem_full_name,
            "file_path": self.file_path,
            "goal_started": self.goal_started,
            "failure_category": self.failure_category,
            "crash_retries": self.crash_retries,
            "gold_in_scope": self.gold_in_scope,
            "n_accessible": self.n_accessible,
            "cosine_accepted": self.cosine_accepted,
            "filter_cosine_accepted": self.filter_cosine_accepted,
            "selector_accepted": self.selector_accepted,
            "best_filtered_accepted": self.best_filtered_accepted,
            "cosine_feedback_category": self.cosine_feedback_category,
            "filter_cosine_feedback_category": self.filter_cosine_feedback_category,
            "selector_feedback_category": self.selector_feedback_category,
            "cosine_candidate": self.cosine_candidate,
            "filter_cosine_candidate": self.filter_cosine_candidate,
            "selector_candidate": self.selector_candidate,
            "selector_score": round(self.selector_score, 5),
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Selector scoring
# ---------------------------------------------------------------------------

def selector_rank(
    model: ExecSelector,
    encoder: SentenceTransformer,
    emb_dim: int,
    goal_str: str,
    candidates: list[str],
    cosine_scores: list[float],
    filter_passed: list[bool],
    device: torch.device,
) -> list[float]:
    """Return selector scores for each candidate (same order as input)."""
    if not candidates:
        return []
    texts = [goal_str] + candidates
    embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    goal_emb = embs[0]
    cand_embs = embs[1:]

    n = len(candidates)
    X = np.zeros((n, emb_dim * 2 + 2), dtype=np.float32)
    for i in range(n):
        X[i, :emb_dim] = goal_emb
        X[i, emb_dim : 2 * emb_dim] = cand_embs[i]
        X[i, -2] = float(cosine_scores[i])
        X[i, -1] = float(filter_passed[i])

    with torch.no_grad():
        scores = model.score(torch.from_numpy(X).to(device)).cpu().numpy()
    return scores.tolist()


# ---------------------------------------------------------------------------
# Main per-example processing
# ---------------------------------------------------------------------------

def run_one(
    ex: dict[str, Any],
    kernel: LeanKernel,
    encoder_retrieval: SentenceTransformer,
    selector_model: ExecSelector,
    selector_encoder: SentenceTransformer,
    selector_emb_dim: int,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    gold: str,
    project_root: str,
    device: torch.device,
    top_k: int = 20,
) -> ExampleResult:
    t0 = time.time()
    name = ex["theorem_full_name"]
    res = ExampleResult(theorem_full_name=name, file_path=ex.get("file_path", ""))

    # 1. Create goal state
    replay = kernel.goal_via_file_context(
        theorem_full_name=name,
        file_path=ex.get("file_path", ""),
        prefix_tactics=[],
        expected_goal=ex.get("goal_state_before", ""),
        project_root=project_root,
    )
    if not replay.success:
        res.failure_category = replay.failure_category or "goal_creation_fail"
        res.elapsed_s = time.time() - t0
        return res

    res.goal_started = True
    goal_str = replay.goal_state
    goal_id  = replay.goal_id
    g_head   = _goal_head(goal_str)
    g_shape  = _goal_shape(goal_str)

    # 2. Accessible premises
    theorem_id = name_to_id.get(name)
    if theorem_id is None:
        res.failure_category = "no_db_entry"
        res.elapsed_s = time.time() - t0
        return res
    premise_ids = get_accessible_premises(conn, theorem_id)
    premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]
    res.n_accessible = len(premise_names)
    res.gold_in_scope = gold in premise_names

    if not premise_names:
        res.failure_category = "no_premises"
        res.elapsed_s = time.time() - t0
        return res

    # 3. Cosine ranking
    goal_emb = encoder_retrieval.encode(
        [goal_str], show_progress_bar=False, normalize_embeddings=True
    )[0]
    cand_embs = encoder_retrieval.encode(
        premise_names, show_progress_bar=False,
        batch_size=256, normalize_embeddings=True,
    )
    cosine_raw = (cand_embs @ goal_emb).tolist()
    ranked_idx = sorted(range(len(premise_names)), key=lambda i: -cosine_raw[i])
    top_names  = [premise_names[i] for i in ranked_idx[:top_k]]
    top_scores = [cosine_raw[i]    for i in ranked_idx[:top_k]]

    # 4. Static filter
    top_pp     = [_get_decl_type(kernel, c) or "" for c in top_names]
    top_passed = [_is_compatible(g_head, g_shape, pp) for pp in top_pp]
    filtered   = [(c, s, pp) for c, s, pp, p in zip(top_names, top_scores, top_pp, top_passed) if p]
    if not filtered:
        filtered = list(zip(top_names, top_scores, top_pp))  # fallback

    # 5. Selector scoring over filtered candidates
    f_names  = [c for c, _, _ in filtered]
    f_scores = [s for _, s, _ in filtered]
    f_passed = [True] * len(f_names)

    sel_scores = selector_rank(
        selector_model, selector_encoder, selector_emb_dim,
        goal_str, f_names, f_scores, f_passed, device,
    )

    # 6. Pick top-1 per condition
    # cosine_top1: best cosine across all top_k (no filter)
    res.cosine_candidate = top_names[0]
    # filter_cosine_top1: best cosine among filtered
    res.filter_cosine_candidate = f_names[0]  # already cosine-sorted
    # selector_top1: best selector score among filtered
    best_sel_idx = int(np.argmax(sel_scores))
    res.selector_candidate = f_names[best_sel_idx]
    res.selector_score = sel_scores[best_sel_idx]

    # 7. Lean verify each condition
    _conditions = [
        (res.cosine_candidate,        "cosine_feedback_category",        "cosine_accepted"),
        (res.filter_cosine_candidate, "filter_cosine_feedback_category", "filter_cosine_accepted"),
        (res.selector_candidate,      "selector_feedback_category",      "selector_accepted"),
    ]
    for cand, fb_attr, acc_attr in _conditions:
        try:
            acc, _, crash, fb = try_apply(kernel, goal_str, cand, goal_id=goal_id)
        except ServerCrashError:
            res.crash_retries += 1
            break
        if crash:
            res.crash_retries += 1
            continue
        setattr(res, acc_attr, acc)
        setattr(res, fb_attr, fb.category if fb else "")

    # 8. Oracle: probe all filtered top-5
    for cand in f_names[:5]:
        try:
            acc, _, crash, fb = try_apply(kernel, goal_str, cand, goal_id=goal_id)
        except ServerCrashError:
            res.crash_retries += 1
            break
        if crash:
            res.crash_retries += 1
            continue
        if acc:
            res.best_filtered_accepted = True
            break

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_apply_step0(eval_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("family") != "apply":
                continue
            if ex.get("step_index", 0) != 0:
                continue
            rows.append(ex)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[ExampleResult]) -> None:
    n = len(results)
    started = [r for r in results if r.goal_started]
    ns = len(started)
    in_scope = [r for r in started if r.gold_in_scope]
    ni = len(in_scope)

    def _acc(attr: str, pool: list[ExampleResult]) -> str:
        k = sum(getattr(r, attr) for r in pool)
        return f"{k}/{len(pool)}  ({100*k/max(len(pool),1):.1f}%)"

    fb_counts: dict[str, dict[str, int]] = {
        "cosine": {}, "filter_cosine": {}, "selector": {},
    }
    for r in started:
        for cond in ("cosine", "filter_cosine", "selector"):
            cat = getattr(r, f"{cond}_feedback_category") or "no_feedback"
            fb_counts[cond][cat] = fb_counts[cond].get(cat, 0) + 1

    print()
    print("=" * 72)
    print("EXP-048: ExecSelector v1 — Live Pantograph benchmark")
    print("=" * 72)
    print(f"  n={n}, started={ns}/{n} ({100*ns/max(n,1):.1f}%)")
    print(f"  gold_in_scope={ni}/{ns} ({100*ni/max(ns,1):.1f}%)")
    print()
    print(f"  LeanAccepted|started ({ns} started):")
    print()
    print(f"    {'Condition':<28} {'Acc|started':<22} {'Acc|gold_in_scope'}")
    print(f"    {'-'*28} {'-'*22} {'-'*20}")
    for cond, label in [
        ("cosine_accepted",        "cosine_top1"),
        ("filter_cosine_accepted", "filter_cosine_top1"),
        ("selector_accepted",      "selector_top1"),
        ("best_filtered_accepted", "best_filtered_top5"),
    ]:
        a_s  = _acc(cond, started)
        a_i  = _acc(cond, in_scope)
        print(f"    {label:<28} {a_s:<22} {a_i}")
    print()
    print("  Selector vs cosine lift:")
    sel_k  = sum(r.selector_accepted for r in started)
    cos_k  = sum(r.cosine_accepted   for r in started)
    fcos_k = sum(r.filter_cosine_accepted for r in started)
    print(f"    cosine_top1        : {cos_k}/{ns}")
    print(f"    filter_cosine_top1 : {fcos_k}/{ns}")
    print(
        f"    selector_top1      : {sel_k}/{ns}"
        f"  (Δ={sel_k-cos_k:+d} vs cosine,"
        f" Δ={sel_k-fcos_k:+d} vs filter+cosine)"
    )
    print()
    print("  Feedback breakdown (selector):")
    for cat, cnt in sorted(fb_counts["selector"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<30} {cnt}")
    print(f"  Server crash retries: {sum(r.crash_retries for r in results)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",    default="data/proof_network_v3.db")
    parser.add_argument("--eval",  default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--selector", default="models/apply_exec_selector_v1.pt")
    parser.add_argument("--output",   default="runs/exp048_results.jsonl")
    parser.add_argument("--limit",    type=int, default=0)
    parser.add_argument("--restart-every", type=int, default=64)
    parser.add_argument("--top-k",    type=int, default=20)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load selector
    selector_model, selector_encoder, selector_emb_dim = load_selector(
        args.selector, device
    )

    # Retrieval encoder (same as training)
    retrieval_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # DB
    conn = sqlite3.connect(args.db)
    id_to_name = {eid: n for eid, n in conn.execute("SELECT id, name FROM entities")}
    name_to_id = {v: k for k, v in id_to_name.items()}
    logger.info("DB: %d entities", len(id_to_name))

    # Eval examples
    examples = load_apply_step0(args.eval)
    if args.limit:
        examples = examples[:args.limit]
    logger.info("Eval examples: %d step-0 apply", len(examples))

    # Kernel
    config = LeanConfig(
        backend="pantograph",
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    kernel = LeanKernel(config=config)

    results: list[ExampleResult] = []

    with open(args.output, "w") as fout:
        for i, ex in enumerate(examples):
            if args.restart_every and i > 0 and i % args.restart_every == 0:
                logger.info("Periodic restart at example %d", i)
                kernel._restart_server()

            gold = ex.get("annotated_premise", "")
            name = ex["theorem_full_name"]
            logger.info("[%d/%d] %s", i + 1, len(examples), name)

            try:
                res = run_one(
                    ex, kernel, retrieval_encoder,
                    selector_model, selector_encoder, selector_emb_dim,
                    conn, id_to_name, name_to_id,
                    gold=gold,
                    project_root=args.lean_project,
                    device=device,
                    top_k=args.top_k,
                )
            except Exception as exc:
                logger.warning("Error on %s: %s", name, exc)
                res = ExampleResult(theorem_full_name=name, file_path=ex.get("file_path", ""))

            results.append(res)
            fout.write(json.dumps(res.to_dict()) + "\n")
            fout.flush()

    print_report(results)


if __name__ == "__main__":
    main()
