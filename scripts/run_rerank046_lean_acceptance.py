"""EXP-RERANK-046: Component-level LeanAccepted benchmark for apply + refine_named.

Takes the same eval set used in EXP-RERANK-045b and measures the fraction of
started goals where the top-1 selector's candidate is accepted by Lean.

Three conditions (shared started-goal subset):
  cosine_top1     — top-1 by cosine score
  rule_top1       — top-1 by rule_head_shape heuristic
  reranker_top1   — top-1 by learned MLP reranker (models/reranker045_apply.pt etc.)

For apply:   try `apply {candidate}`
For refine_named: try 6 refine suffix variants (same as EXP-REFINE-043)

Metric: LeanAccepted|started  (not ranking — Lean kernel verdict)

This is NOT theorem-search integration. It validates whether the reranker's
ranking improvement translates to tactic acceptance. Theorem-search integration
is a subsequent experiment.

Usage:
    python -m scripts.run_rerank046_lean_acceptance \\
        --db data/proof_network_v3.db \\
        --eval data/canonical/canonical_residual_eval.jsonl \\
        --apply-model models/reranker045_apply.pt \\
        --refine-model models/reranker045_refine.pt \\
        --lean-project data/lean_project/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.apply_scoper import (  # type: ignore[import]
    _SHAPE_SUFFIXES,
    classify_goal_shape,
    extract_goal_head,
)
from src.lean_interface import LeanConfig, LeanKernel, ReplayResult
from src.proof_network import get_accessible_premises

logger = logging.getLogger(__name__)

try:
    from src.lean_interface import ServerCrashError  # type: ignore[attr-defined]
except ImportError:
    class ServerCrashError(Exception):  # type: ignore[no-redef]
        pass

# ---------------------------------------------------------------------------
# Feature extraction (mirrors train_reranker045)
# ---------------------------------------------------------------------------

N_FEATURES = 7


def _tokens(name: str) -> set[str]:
    parts = re.split(r"[._]", name)
    tokens: set[str] = set()
    for p in parts:
        for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p):
            tokens.add(sub.lower())
    return tokens


def _hypothesis_names(goal_state: str) -> set[str]:
    names: set[str] = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w'✝]*)(?:\s+\w[\w'✝]*)*\s*:", line)
        if m and m.group(1) not in ("case", "⊢"):
            names.add(m.group(1))
    return names


def feature_vector(
    candidate: str,
    cosine_score: float,
    cosine_rank: int,
    goal_head: str,
    goal_shape: str,
    theorem_namespace: str,
    hyp_names: set[str],
) -> list[float]:
    name_lower = candidate.lower()

    head_compat = 0.0
    if goal_head and len(goal_head) >= 3:
        h = goal_head.lower()
        if h in name_lower or h in candidate.split(".")[-1].lower():
            head_compat = 1.0

    shape_compat = 0.0
    suffixes = _SHAPE_SUFFIXES.get(goal_shape, [])
    if any(suf in name_lower for suf in suffixes):
        shape_compat = 1.0

    _CONCL_SFXS = ["_iff", "_eq", "_le", "_lt", "_mem", "_surjective", "_injective",
                   "_continuous", "_tendsto", "_measurable", "_mono", "_nonneg"]
    conclusion_sfx = 1.0 if any(name_lower.endswith(s) or s + "_" in name_lower
                                  for s in _CONCL_SFXS) else 0.0

    cand_ns = ".".join(candidate.split(".")[:2]) if "." in candidate else ""
    thm_ns = ".".join(theorem_namespace.split(".")[:2]) if "." in theorem_namespace else ""
    namespace_match = 1.0 if cand_ns and cand_ns == thm_ns else 0.0

    cand_tokens = _tokens(candidate)
    hyp_tokens: set[str] = set()
    for h in hyp_names:
        hyp_tokens |= _tokens(h)
    local_overlap = len(cand_tokens & hyp_tokens) / max(len(cand_tokens), 1)

    rank_norm = (5 - cosine_rank) / 5.0

    return [head_compat, shape_compat, conclusion_sfx, namespace_match,
            local_overlap, float(cosine_score), rank_norm]


# ---------------------------------------------------------------------------
# Reranker model (same architecture as train_reranker045)
# ---------------------------------------------------------------------------

class CandidateReranker(nn.Module):
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (k, F) → (k,)
        return self.net(x).squeeze(-1)


def load_reranker(path: str) -> CandidateReranker | None:
    p = Path(path)
    if not p.exists():
        logger.warning("Reranker model not found: %s", path)
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = CandidateReranker()
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Selection functions
# ---------------------------------------------------------------------------

def _rule_head_shape_score(cand: str, goal_head: str, goal_shape: str, cosine: float) -> float:
    name_lower = cand.lower()
    head_compat = 0.0
    if goal_head and len(goal_head) >= 3:
        h = goal_head.lower()
        if h in name_lower or h in cand.split(".")[-1].lower():
            head_compat = 1.0
    shape_compat = 0.0
    suffixes = _SHAPE_SUFFIXES.get(goal_shape, [])
    if any(suf in name_lower for suf in suffixes):
        shape_compat = 1.0
    return head_compat * 20 + shape_compat * 10 + cosine


def select_top1(
    method: str,
    ranked: list[tuple[float, str]],
    goal_state: str,
    theorem_full_name: str,
    reranker: CandidateReranker | None,
) -> str | None:
    """Return the top-1 candidate name under a given selection method."""
    if not ranked:
        return None

    top_k = ranked[:5]

    if method == "cosine":
        return top_k[0][1]

    goal_head = extract_goal_head(goal_state)
    goal_shape = classify_goal_shape(goal_state)

    if method == "rule":
        best_score = -1e9
        best_name = top_k[0][1]
        for cscore, name in top_k:
            s = _rule_head_shape_score(name, goal_head, goal_shape, cscore)
            if s > best_score:
                best_score = s
                best_name = name
        return best_name

    if method == "reranker" and reranker is not None:
        hyp_names = _hypothesis_names(goal_state)
        thm_ns = theorem_full_name
        feats = []
        for rank, (cscore, name) in enumerate(top_k):
            fv = feature_vector(name, cscore, rank, goal_head, goal_shape, thm_ns, hyp_names)
            feats.append(fv)
        x = torch.tensor(feats, dtype=torch.float32)  # (k, F)
        with torch.no_grad():
            scores = reranker(x)  # (k,)
        best_idx = int(scores.argmax().item())
        return top_k[best_idx][1]

    return top_k[0][1]  # fallback


# ---------------------------------------------------------------------------
# Lean helpers (mirrors run_apply0_benchmark / run_refine0_benchmark)
# ---------------------------------------------------------------------------

_REFINE_VARIANTS = [
    "{lemma} ?_",
    "{lemma}.1 ?_",
    "{lemma}.2 ?_",
    "{lemma}.mp ?_",
    "{lemma}.mpr ?_",
    "{lemma} _ ?_",
]


from src.nav_contracts import LeanFeedback  # noqa: E402


def try_tactic_safe(
    kernel: LeanKernel,
    goal_state: str,
    tactic: str,
    goal_id: int = 0,
) -> tuple[bool, bool, str, bool, LeanFeedback | None]:
    """(accepted, closed_all, error, was_crash, feedback)"""
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        closed = result.success and not result.new_goals
        return result.success, closed, result.error_message, False, result.feedback
    except ServerCrashError as e:
        return False, False, f"server_crash: {e}", True, None
    except Exception as e:
        return False, False, str(e), False, None


def try_apply(
    kernel: LeanKernel,
    goal_state: str,
    candidate: str,
    goal_id: int = 0,
) -> tuple[bool, bool, bool, LeanFeedback | None]:
    """(accepted, closed_all, was_crash, feedback)"""
    tac = f"apply {candidate}"
    acc, closed, _, crash, fb = try_tactic_safe(kernel, goal_state, tac, goal_id=goal_id)
    return acc, closed, crash, fb


def try_refine_variants(
    kernel: LeanKernel,
    goal_state: str,
    lemma: str,
    goal_id: int = 0,
) -> tuple[bool, bool, bool, LeanFeedback | None]:
    """(accepted, closed_all, was_crash, feedback)"""
    last_fb: LeanFeedback | None = None
    for tmpl in _REFINE_VARIANTS:
        tac = f"refine {tmpl.format(lemma=lemma)}"
        acc, closed, _, crash, fb = try_tactic_safe(kernel, goal_state, tac, goal_id=goal_id)
        if crash:
            return False, False, True, None
        last_fb = fb
        if acc or closed:
            return acc, closed, False, fb
    return False, False, False, last_fb


# ---------------------------------------------------------------------------
# Per-example result
# ---------------------------------------------------------------------------

@dataclass
class ExampleResult:
    theorem_full_name: str
    file_path: str
    subset: str   # "apply" | "refine_named"

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    gold_in_scope: bool = False
    n_accessible_premises: int = 0
    gold_rank_cosine: int = -1

    # LeanAccepted per method
    cosine_accepted: bool = False
    rule_accepted: bool = False
    reranker_accepted: bool = False

    # Which candidate was chosen per method
    cosine_candidate: str = ""
    rule_candidate: str = ""
    reranker_candidate: str = ""

    # LeanFeedback category per method
    cosine_feedback_category: str = ""
    rule_feedback_category: str = ""
    reranker_feedback_category: str = ""

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_examples(eval_path: str) -> tuple[list[dict], list[dict]]:
    """Load step-0 apply and refine_named examples."""
    apply_exs: list[dict] = []
    refine_named: list[dict] = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("step_index", 0) != 0:
                continue
            family = ex.get("family", "")
            if family == "apply":
                apply_exs.append(ex)
            elif family == "refine":
                prem = ex.get("annotated_premise", "")
                ir = ex.get("canonical_action_ir", "")
                tac = ex.get("tactic_text", "")
                body = re.sub(r"^refine\s*", "", (ir or tac).strip())
                if prem and (body.startswith(prem) or body.startswith(f"({prem}")):
                    refine_named.append(ex)
    return apply_exs, refine_named


# ---------------------------------------------------------------------------
# Per-example runner
# ---------------------------------------------------------------------------

def run_one(
    example: dict,
    subset: str,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder,
    reranker: CandidateReranker | None,
    project_root: str,
) -> ExampleResult:
    t0 = time.time()
    res = ExampleResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example.get("file_path", ""),
        subset=subset,
    )

    # Goal creation via file context
    replay: ReplayResult = kernel.goal_via_file_context(
        theorem_full_name=res.theorem_full_name,
        file_path=res.file_path,
        prefix_tactics=example.get("prefix_tactics", []),
        expected_goal=example.get("goal_state_before", ""),
        project_root=project_root,
        prefix_goal_states=example.get("prefix_goal_states", []),
    )
    res.goal_started = replay.success
    res.tier_used = replay.tier_used
    res.failure_category = replay.failure_category
    res.crash_retries = replay.crash_retries

    if not replay.success:
        res.elapsed_s = time.time() - t0
        return res

    goal_str = replay.goal_state
    goal_id = replay.goal_id

    # Build accessible premise list
    tid = name_to_id.get(res.theorem_full_name)
    if tid is None:
        res.elapsed_s = time.time() - t0
        return res

    pids = get_accessible_premises(conn, tid)
    accessible = [id_to_name[pid] for pid in pids if pid in id_to_name]
    res.n_accessible_premises = len(accessible)

    gold = example.get("annotated_premise", "")
    if gold and gold in accessible:
        res.gold_in_scope = True

    if not accessible:
        res.elapsed_s = time.time() - t0
        return res

    # Cosine rank
    goal_emb = encoder.encode([goal_str], normalize_embeddings=True)
    sym_embs = encoder.encode(accessible, normalize_embeddings=True)
    scores = (goal_emb @ sym_embs.T).flatten()
    ranked = sorted(zip(scores.tolist(), accessible), reverse=True)

    if gold:
        res.gold_rank_cosine = next(
            (i for i, (_, p) in enumerate(ranked) if p == gold), -1
        )

    # Select top-1 per method
    for method in ("cosine", "rule", "reranker"):
        cand = select_top1(method, ranked, goal_str, res.theorem_full_name,
                           reranker if method == "reranker" else None)
        if cand is None:
            continue

        if subset == "apply":
            acc, _, crash, fb = try_apply(kernel, goal_str, cand, goal_id=goal_id)
        else:  # refine_named
            acc, _, crash, fb = try_refine_variants(kernel, goal_str, cand, goal_id=goal_id)

        if crash:
            res.crash_retries += 1
            continue

        fb_cat = fb.category if fb else ""
        if method == "cosine":
            res.cosine_accepted = acc
            res.cosine_candidate = cand
            res.cosine_feedback_category = fb_cat
        elif method == "rule":
            res.rule_accepted = acc
            res.rule_candidate = cand
            res.rule_feedback_category = fb_cat
        else:
            res.reranker_accepted = acc
            res.reranker_candidate = cand
            res.reranker_feedback_category = fb_cat

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    apply_res: list[ExampleResult],
    refine_res: list[ExampleResult],
) -> None:
    def _section(results: list[ExampleResult], label: str) -> None:
        n = len(results)
        started = [r for r in results if r.goal_started]
        ns = len(started)
        in_scope = [r for r in started if r.gold_in_scope]
        ni = len(in_scope)

        print(f"\n  {label}  (n={n}, started={ns}/{n} {100*ns/max(n,1):.1f}%,"
              f" gold_in_scope={ni}/{ns} {100*ni/max(ns,1):.1f}%)")

        if ns == 0:
            return

        def _acc(lst: list, field: str) -> int:
            return sum(getattr(r, field) for r in lst)

        methods = [
            ("cosine_top1",   "cosine_accepted"),
            ("rule_top1",     "rule_accepted"),
            ("reranker_top1", "reranker_accepted"),
        ]

        print(f"\n    {'Method':<20} {'Acc|started':>12}  {'Acc|gold_in_scope':>18}")
        print(f"    {'-'*55}")
        for name, field in methods:
            acc_s = _acc(started, field)
            acc_i = _acc(in_scope, field)
            s = f"{acc_s}/{ns}  ({100*acc_s/max(ns,1):.1f}%)"
            i = f"{acc_i}/{ni}  ({100*acc_i/max(ni,1):.1f}%)" if ni > 0 else "—"
            print(f"    {name:<20} {s:>12}  {i:>18}")

    print("\n" + "=" * 68)
    print("EXP-RERANK-046: Component-level LeanAccepted benchmark")
    print("=" * 68)
    _section(apply_res, "apply")
    _section(refine_res, "refine_named")

    all_started = [r for r in apply_res + refine_res if r.goal_started]
    ns = len(all_started)
    print(f"\n  OVERALL  started={ns}")
    for method, field in [("cosine_top1", "cosine_accepted"),
                           ("rule_top1", "rule_accepted"),
                           ("reranker_top1", "reranker_accepted")]:
        acc = sum(getattr(r, field) for r in all_started)
        print(f"    {method:<20} {acc}/{ns}  ({100*acc/max(ns,1):.1f}%)")

    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-RERANK-046: LeanAccepted component benchmark")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--eval", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--apply-model", default="models/reranker045_apply.pt")
    parser.add_argument("--refine-model", default="models/reranker045_refine.pt")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="runs/rerank046_results.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    apply_exs, refine_exs = load_eval_examples(args.eval)
    logger.info("apply=%d, refine_named=%d", len(apply_exs), len(refine_exs))

    if args.limit > 0:
        apply_exs = apply_exs[:args.limit]
        refine_exs = refine_exs[:args.limit]

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Encoder loaded")

    apply_reranker = load_reranker(args.apply_model)
    refine_reranker = load_reranker(args.refine_model)

    kernel = LeanKernel(LeanConfig(
        backend=args.backend,
        project_root=args.lean_project,
        imports=["Mathlib"],
    ))
    kernel._ensure_server()
    logger.info("Lean server started")

    conn = sqlite3.connect(args.db)
    rows = conn.execute("SELECT id, name FROM entities").fetchall()
    id_to_name = {eid: name for eid, name in rows}
    name_to_id = {name: eid for eid, name in rows}

    apply_results: list[ExampleResult] = []
    for i, ex in enumerate(apply_exs):
        logger.info("[apply %d/%d] %s", i + 1, len(apply_exs), ex["theorem_full_name"])
        r = run_one(ex, "apply", kernel, conn, id_to_name, name_to_id,
                    encoder, apply_reranker, args.lean_project)
        apply_results.append(r)

    refine_results: list[ExampleResult] = []
    for i, ex in enumerate(refine_exs):
        logger.info("[refine_named %d/%d] %s", i + 1, len(refine_exs), ex["theorem_full_name"])
        r = run_one(ex, "refine_named", kernel, conn, id_to_name, name_to_id,
                    encoder, refine_reranker, args.lean_project)
        refine_results.append(r)

    kernel.close()
    conn.close()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in apply_results + refine_results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Written to %s", args.output)

    print_report(apply_results, refine_results)


if __name__ == "__main__":
    main()
