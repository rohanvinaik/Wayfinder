"""EXP-APPLY-047: Executable compatibility filter for apply candidate shortlists.

Pipeline under test:
  scope candidates (accessible premises, cosine top-k)
  filter by static compatibility (goal shape vs candidate conclusion shape)
  rerank survivors (cosine or learned)
  Lean verify

Four conditions on the same apply step-0 eval set (shared started-goal subset):
  cosine_top1         — raw cosine top-1, no filter
  compat_cosine_top1  — cosine top-1 after compatibility filter
  compat_reranker_top1 — reranker top-1 after compatibility filter
  best_filtered_top5  — oracle upper bound: best of filtered top-5 via Lean verify

Compatibility filter uses declaration type from env_inspect (not name heuristics):
  - strip leading ∀/implicit binders from candidate type → conclusion
  - extract conclusion head symbol and proposition shape
  - pass if conclusion shape is compatible with goal target shape
  - pass if conclusion head matches goal target head (when goal head is named)

Key metrics:
  GoalStart
  candidates before/after filter (filter yield)
  executable-candidate recall: among goals with >=1 accepted in raw top-5,
    how often does the filter keep at least one accepted candidate
  LeanAccepted|started per condition
  Lean probes / goal (cost)

This is NOT theorem-search integration. It validates whether a static type-shape
filter raises the LeanAccepted ceiling before any learned module is added.

Usage:
    python -m scripts.run_apply047_compat_filter \\
        --db data/proof_network_v3.db \\
        --eval data/canonical/canonical_residual_eval.jsonl \\
        --lean-project data/lean_project/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
# Declaration type cache (env_inspect per candidate)
# ---------------------------------------------------------------------------

_type_cache: dict[str, dict] = {}


def get_decl_type(kernel: LeanKernel, name: str) -> dict | None:
    """Return env_inspect result for a declaration name. Cached."""
    if name in _type_cache:
        return _type_cache[name]
    try:
        if kernel._server is None:
            return None
        result = kernel._server.env_inspect(name, print_value=False, print_dependency=False)
        _type_cache[name] = result
        return result
    except Exception:
        _type_cache[name] = {}
        return None


# ---------------------------------------------------------------------------
# Conclusion shape extraction
# ---------------------------------------------------------------------------

def _strip_binders(s: str) -> str:
    """Strip leading ∀ binders (implicit, explicit, instance)."""
    s = s.strip()
    while True:
        m = re.match(
            r'^∀\s*((?:(?:\{[^{}]*\}|\([^()]*\)|\[[^\[\]]*\])\s*)+),\s*', s
        )
        if m:
            s = s[m.end():].strip()
            continue
        break
    return s


def conclusion_of(pp_type: str) -> str:
    """Get the final consequent of a type string (strip binders + implications)."""
    s = _strip_binders(pp_type)
    # Split on → at top level, take last part
    parts = re.split(r' → ', s)
    return parts[-1].strip() if parts else s


def prop_shape(concl: str) -> str:
    c = concl[:100]
    if ' → ' in c:
        return 'imp'
    if ' = ' in c:
        return 'eq'
    if ' ↔ ' in c or concl.startswith('Iff '):
        return 'iff'
    if ' ≤ ' in c:
        return 'le'
    if ' < ' in c:
        return 'lt'
    if concl.startswith('∃ ') or '∃ ' in c[:20]:
        return 'exists'
    if ' ∧ ' in c:
        return 'and'
    if ' ∨ ' in c:
        return 'or'
    if concl.startswith('¬ '):
        return 'not'
    return 'prop'


def head_symbol(concl: str) -> str:
    m = re.match(r'(\w[\w.]*)', concl.strip())
    return m.group(1) if m else ''


# Shape compatibility table: goal shape → compatible candidate shapes
_SHAPE_COMPAT: dict[str, set[str]] = {
    'eq':     {'eq', 'iff', 'le', 'prop'},   # le_antisymm proves eq
    'iff':    {'iff', 'eq', 'prop'},
    'imp':    {'imp', 'iff', 'prop'},
    'le':     {'le', 'lt', 'eq', 'prop'},
    'lt':     {'lt', 'le', 'prop'},
    'exists': {'exists', 'prop'},
    'and':    {'and', 'prop'},
    'or':     {'or', 'prop'},
    'not':    {'not', 'iff', 'prop'},
    'prop':   {'eq', 'iff', 'le', 'lt', 'exists', 'and', 'or', 'not', 'prop'},
}


def _goal_shape_from_ir(goal_shape_ir: dict) -> str:
    """Extract proposition shape from pre-computed goal_shape_ir."""
    if goal_shape_ir.get('has_equality'):
        return 'eq'
    if goal_shape_ir.get('has_iff'):
        return 'iff'
    if goal_shape_ir.get('has_exists'):
        return 'exists'
    if goal_shape_ir.get('has_implication'):
        return 'imp'
    return 'prop'


def is_compatible(
    goal_target_head: str,
    goal_shape: str,
    cand_pp_type: str,
) -> bool:
    """
    Return True if candidate conclusion shape is compatible with goal shape,
    or if the candidate conclusion head matches the goal target head.
    """
    if not cand_pp_type:
        return True  # unknown type: let through

    stripped = _strip_binders(cand_pp_type)
    concl = conclusion_of(cand_pp_type)
    cand_shape = prop_shape(stripped if goal_shape == 'imp' else concl)
    cand_head = head_symbol(concl)

    # Shape compatibility
    compat_shapes = _SHAPE_COMPAT.get(goal_shape, {'prop'})
    if cand_shape in compat_shapes:
        return True

    # Head match: goal_target_head must be a named predicate (not eq/variable)
    if (goal_target_head
            and goal_target_head not in ('eq', 'iff', 'le', 'lt', 'and', 'or', 'not', 'a', 'b')
            and len(goal_target_head) >= 4
            and cand_head.lower() == goal_target_head.lower()):
        return True

    return False


# ---------------------------------------------------------------------------
# Reranker (same architecture as train_reranker045)
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


class CandidateReranker(nn.Module):
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# Lean helpers
# ---------------------------------------------------------------------------

from src.nav_contracts import LeanFeedback  # noqa: E402  (after logger)


def _feedback_to_dict(feedback: LeanFeedback | None) -> dict[str, Any] | None:
    return feedback.to_dict() if feedback is not None else None


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
        return False, False, str(e), False, LeanFeedback(
            stage="tactic_exec",
            category="other",
            messages=[],
            raw_error=str(e),
        )


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


# ---------------------------------------------------------------------------
# Per-example result
# ---------------------------------------------------------------------------

@dataclass
class ExampleResult:
    theorem_full_name: str
    file_path: str

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0
    goal_feedback: dict[str, Any] | None = None

    # Scope stats
    n_accessible: int = 0
    gold_in_scope: bool = False
    gold_rank_cosine: int = -1

    # Filter stats
    n_filtered: int = 0    # candidates passing filter (out of top-k)
    filter_kept_gold: bool = False  # gold premise survived filter

    # executable-candidate recall denominator:
    # True if at least one of top-5 raw is Lean-accepted
    raw_top5_has_accepted: bool = False

    # Does at least one filtered candidate get accepted by Lean?
    filtered_has_accepted: bool = False

    # LeanAccepted per condition
    cosine_accepted: bool = False
    compat_cosine_accepted: bool = False
    compat_reranker_accepted: bool = False
    best_filtered_accepted: bool = False   # upper bound

    # Structured failure categories for cosine_top1 and compat_cosine_top1
    # Values: none | accepted_with_goals | parse_error | unknown_identifier |
    #         unification_mismatch | typeclass_missing | generated_sorry |
    #         generated_unsafe | other | (empty = not attempted)
    cosine_feedback_category: str = ""
    compat_cosine_feedback_category: str = ""
    compat_reranker_feedback_category: str = ""
    cosine_feedback: dict[str, Any] | None = None
    compat_cosine_feedback: dict[str, Any] | None = None
    compat_reranker_feedback: dict[str, Any] | None = None

    raw_top5_probes: list[dict[str, Any]] = field(default_factory=list)
    filtered_top5_probes: list[dict[str, Any]] = field(default_factory=list)

    # Lean probe count (best_filtered_top5 only)
    lean_probes: int = 0

    # Chosen candidates
    cosine_candidate: str = ""
    compat_cosine_candidate: str = ""
    compat_reranker_candidate: str = ""

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Per-example runner
# ---------------------------------------------------------------------------

TOP_K = 10  # cosine shortlist before filtering


def run_one(
    example: dict,
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
    )

    # Goal creation
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
    res.goal_feedback = _feedback_to_dict(replay.feedback)

    if not replay.success:
        res.elapsed_s = time.time() - t0
        return res

    goal_str = replay.goal_state
    goal_id = replay.goal_id

    # Accessible premises
    tid = name_to_id.get(res.theorem_full_name)
    if tid is None:
        res.elapsed_s = time.time() - t0
        return res

    pids = get_accessible_premises(conn, tid)
    accessible = [id_to_name[pid] for pid in pids if pid in id_to_name]
    res.n_accessible = len(accessible)

    gold = example.get("annotated_premise", "")
    if gold and gold in accessible:
        res.gold_in_scope = True

    if not accessible:
        res.elapsed_s = time.time() - t0
        return res

    # Cosine ranking (top-k)
    goal_emb = encoder.encode([goal_str], normalize_embeddings=True)
    sym_embs = encoder.encode(accessible, normalize_embeddings=True)
    scores = (goal_emb @ sym_embs.T).flatten()
    ranked = sorted(zip(scores.tolist(), accessible), reverse=True)[:TOP_K]

    if gold:
        res.gold_rank_cosine = next(
            (i for i, (_, p) in enumerate(ranked) if p == gold), -1
        )

    # Cosine top-1 (baseline, no filter)
    if ranked:
        res.cosine_candidate = ranked[0][1]
        acc, _, crash, fb = try_apply(kernel, goal_str, ranked[0][1], goal_id=goal_id)
        if crash:
            res.crash_retries += 1
        else:
            res.cosine_accepted = acc
            res.cosine_feedback_category = fb.category if fb else ""
            res.cosine_feedback = _feedback_to_dict(fb)

    # Check raw top-5 for executable-candidate recall baseline
    for _, cand in ranked[:5]:
        acc, closed, crash, fb = try_apply(kernel, goal_str, cand, goal_id=goal_id)
        res.raw_top5_probes.append({
            "candidate": cand,
            "accepted": acc,
            "closed": closed,
            "crashed": crash,
            "feedback": _feedback_to_dict(fb),
        })
        if crash:
            res.crash_retries += 1
            break
        if acc:
            res.raw_top5_has_accepted = True
            break

    # Goal shape from pre-computed IR (no extra Lean call)
    gsi = example.get("goal_shape_ir", {})
    goal_target_head = gsi.get("target_head", "")
    goal_shape = _goal_shape_from_ir(gsi)

    # Fetch declaration types and apply compatibility filter
    filtered: list[tuple[float, str]] = []
    for cscore, cand in ranked:
        decl = get_decl_type(kernel, cand)
        if decl is None:
            continue
        pp_type = (decl.get("type") or {}).get("pp", "")
        if is_compatible(goal_target_head, goal_shape, pp_type):
            filtered.append((cscore, cand))

    res.n_filtered = len(filtered)
    if gold:
        res.filter_kept_gold = any(p == gold for _, p in filtered)

    if not filtered:
        # Nothing survived filter — fall back to cosine_top1 result already recorded
        res.elapsed_s = time.time() - t0
        return res

    # compat_cosine_top1: best cosine among filtered
    res.compat_cosine_candidate = filtered[0][1]
    acc, _, crash, fb = try_apply(kernel, goal_str, filtered[0][1], goal_id=goal_id)
    if crash:
        res.crash_retries += 1
    else:
        res.compat_cosine_accepted = acc
        res.compat_cosine_feedback_category = fb.category if fb else ""
        res.compat_cosine_feedback = _feedback_to_dict(fb)

    # compat_reranker_top1
    if reranker is not None:
        goal_head_str = extract_goal_head(goal_str)
        goal_shape_str = classify_goal_shape(goal_str)
        hyp_names = _hypothesis_names(goal_str)
        feats = []
        for rank, (cscore, cand) in enumerate(filtered[:5]):
            fv = feature_vector(cand, cscore, rank, goal_head_str, goal_shape_str,
                                res.theorem_full_name, hyp_names)
            feats.append(fv)
        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            rr_scores = reranker(x)
        best_idx = int(rr_scores.argmax().item())
        best_cand = filtered[min(best_idx, len(filtered) - 1)][1]
        res.compat_reranker_candidate = best_cand

        acc, _, crash, fb = try_apply(kernel, goal_str, best_cand, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
        else:
            res.compat_reranker_accepted = acc
            res.compat_reranker_feedback_category = fb.category if fb else ""
            res.compat_reranker_feedback = _feedback_to_dict(fb)

    # best_filtered_top5: oracle upper bound — try all filtered top-5 in Lean
    accepted_any = False
    for _, cand in filtered[:5]:
        acc, closed, crash, fb = try_apply(kernel, goal_str, cand, goal_id=goal_id)
        res.lean_probes += 1
        res.filtered_top5_probes.append({
            "candidate": cand,
            "accepted": acc,
            "closed": closed,
            "crashed": crash,
            "feedback": _feedback_to_dict(fb),
        })
        if crash:
            res.crash_retries += 1
            break
        if acc:
            accepted_any = True
            res.best_filtered_accepted = True
            break
    res.filtered_has_accepted = accepted_any

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[ExampleResult]) -> None:
    n = len(results)
    started = [r for r in results if r.goal_started]
    ns = len(started)
    in_scope = [r for r in started if r.gold_in_scope]
    ni = len(in_scope)

    print("\n" + "=" * 72)
    print("EXP-APPLY-047: Executable compatibility filter")
    print("=" * 72)
    print(f"\n  n={n}, started={ns}/{n} ({100*ns/max(n,1):.1f}%)")
    print(f"  gold_in_scope={ni}/{ns} ({100*ni/max(ns,1):.1f}%)")

    # Filter yield
    mean_raw = TOP_K  # always k candidates before filter
    filtered_sizes = [r.n_filtered for r in started]
    mean_filt = sum(filtered_sizes) / max(len(filtered_sizes), 1)
    gold_kept = sum(r.filter_kept_gold for r in started if r.gold_in_scope)
    print("\n  Filter yield:")
    print(f"    mean candidates before filter: {mean_raw}")
    print(f"    mean candidates after filter:  {mean_filt:.1f}")
    print(f"    filter reduction:              {100*(1-mean_filt/mean_raw):.1f}%")
    print(f"    gold_in_filter|gold_in_scope:  {gold_kept}/{ni}  ({100*gold_kept/max(ni,1):.1f}%)")

    # Executable-candidate recall
    raw_has = sum(r.raw_top5_has_accepted for r in started)
    filt_has = sum(r.filtered_has_accepted for r in started)
    print("\n  Executable-candidate recall:")
    def _pct(a: int, b: int) -> str:
        return f"{a}/{b}  ({100*a/max(b,1):.1f}%)"

    print(f"    goals w/ >=1 accepted in raw top-5:      {_pct(raw_has, ns)}")
    print(f"    goals w/ >=1 accepted in filtered top-5: {_pct(filt_has, ns)}")
    if raw_has > 0:
        kept_recall = sum(
            1 for r in started if r.raw_top5_has_accepted and r.filtered_has_accepted
        )
        print(f"    executable-candidate recall:            {kept_recall}/{raw_has}  "
              f"({100*kept_recall/max(raw_has,1):.1f}%)")

    # LeanAccepted per condition
    print(f"\n  LeanAccepted|started ({ns} started):")
    conditions = [
        ("cosine_top1",          "cosine_accepted"),
        ("compat_cosine_top1",   "compat_cosine_accepted"),
        ("compat_reranker_top1", "compat_reranker_accepted"),
        ("best_filtered_top5",   "best_filtered_accepted"),
    ]
    print(f"\n    {'Condition':<28} {'Acc|started':>12}  {'Acc|gold_in_scope':>18}")
    print(f"    {'-'*63}")
    for label, attr in conditions:
        acc_s = sum(getattr(r, attr) for r in started)
        acc_i = sum(getattr(r, attr) for r in in_scope)
        s = f"{acc_s}/{ns}  ({100*acc_s/max(ns,1):.1f}%)"
        i = f"{acc_i}/{ni}  ({100*acc_i/max(ni,1):.1f}%)" if ni > 0 else "—"
        print(f"    {label:<28} {s:>12}  {i:>18}")

    # Cost
    total_probes = sum(r.lean_probes for r in started)
    print(f"\n  Lean probes (best_filtered_top5 only): {total_probes} total"
          f", {total_probes/max(ns,1):.1f}/started")
    crashes = sum(r.crash_retries for r in results)
    print(f"  Server crash retries: {crashes}")

    goal_failures = [r for r in results if not r.goal_started]
    if goal_failures:
        by_goal_feedback: dict[str, int] = {}
        for r in goal_failures:
            cat = (r.goal_feedback or {}).get("category", r.failure_category or "other")
            by_goal_feedback[cat] = by_goal_feedback.get(cat, 0) + 1
        print("\n  Goal-start feedback:")
        for label, count in sorted(by_goal_feedback.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {label:<28} {count}")

    def _probe_feedback_counts(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in started:
            for probe in getattr(r, field):
                cat = ((probe.get("feedback") or {}).get("category")) or "none"
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    raw_probe_counts = _probe_feedback_counts("raw_top5_probes")
    if raw_probe_counts:
        print("\n  Raw top-5 probe feedback:")
        for label, count in sorted(raw_probe_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {label:<28} {count}")

    filtered_probe_counts = _probe_feedback_counts("filtered_top5_probes")
    if filtered_probe_counts:
        print("\n  Filtered top-5 probe feedback:")
        for label, count in sorted(filtered_probe_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {label:<28} {count}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-APPLY-047: apply compatibility filter benchmark"
    )
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--eval", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--apply-model", default="models/reranker045_apply.pt")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="runs/apply047_results.jsonl")
    parser.add_argument("--restart-every", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load apply step-0 examples
    examples: list[dict] = []
    with open(args.eval) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("step_index", 0) == 0 and ex.get("family") == "apply":
                examples.append(ex)
    logger.info("apply step-0: %d examples", len(examples))

    if args.limit > 0:
        examples = examples[:args.limit]

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Encoder loaded")

    reranker = load_reranker(args.apply_model)
    if reranker:
        logger.info("Apply reranker loaded from %s", args.apply_model)

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

    results: list[ExampleResult] = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        for i, ex in enumerate(examples):
            if i > 0:
                kernel.gc()
            if args.restart_every > 0 and i > 0 and i % args.restart_every == 0:
                logger.info("Periodic server restart")
                kernel._restart_server()
            logger.info("[apply %d/%d] %s", i + 1, len(examples), ex["theorem_full_name"])
            r = run_one(
                ex,
                kernel,
                conn,
                id_to_name,
                name_to_id,
                encoder,
                reranker,
                args.lean_project,
            )
            results.append(r)
            out_f.write(json.dumps(asdict(r)) + "\n")
            out_f.flush()

    kernel.close()
    conn.close()
    logger.info("Written to %s", args.output)

    print_report(results)


if __name__ == "__main__":
    main()
