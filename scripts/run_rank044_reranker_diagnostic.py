"""EXP-RANK-044: Typed reranker diagnostic on apply + refine.

Offline analysis — no Lean execution.

For each step-0 apply and refine example where gold premise is in accessible scope:
  1. Cosine-rank accessible premises → top-5 candidates
  2. Compute cheap typed features for each candidate
  3. Apply reranking rules (feature combinations)
  4. Measure: does gold move to rank 1?

Features:
  head_compat      — candidate name contains goal head symbol
  shape_compat     — candidate name suffix matches goal conclusion shape
  namespace_match  — candidate shares top-level namespace with theorem
  local_overlap    — candidate name tokens overlap with hypothesis names in goal
  conclusion_sfx   — candidate ends in conclusion-type suffix (_iff, _eq, _le, ...)
  cosine_score     — raw cosine similarity (float)

Reranking rules (over cosine top-5):
  cosine_top1      — baseline: cosine rank 1 (no reranking)
  rule_head        — head_compat candidates first, else cosine
  rule_shape       — shape_compat candidates first, else cosine
  rule_head_shape  — head_compat > shape_compat > cosine
  rule_local       — local_overlap (descending) then cosine
  rule_combined    — weighted: head*3 + shape*2 + local + conclusion_sfx*1 + cosine_score
  oracle_top5      — ceiling: gold anywhere in top-5

Metrics per family (apply, refine_named):
  gold_top1 @ each rule
  MRR@5 @ each rule
  improvement vs cosine_top1 baseline

Usage:
    python -m scripts.run_rank044_reranker_diagnostic
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from src.apply_scoper import _SHAPE_SUFFIXES, classify_goal_shape, extract_goal_head
from src.proof_network import get_accessible_premises


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_examples(eval_path: str, families: list[str]) -> list[dict]:
    examples = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("step_index", 0) == 0 and ex.get("family") in families:
                examples.append(ex)
    return examples


def label_subset(ex: dict) -> str:
    """Label example as 'apply', 'refine_named', or 'refine_anon'."""
    if ex.get("family") == "apply":
        return "apply"
    prem = ex.get("annotated_premise", "")
    ir = ex.get("canonical_action_ir", "")
    tac = ex.get("tactic_text", "")
    body = re.sub(r"^refine\s*", "", (ir or tac).strip())
    if prem and (body.startswith(prem) or body.startswith(f"({prem}")):
        return "refine_named"
    return "refine_anon"


# ---------------------------------------------------------------------------
# Scope + cosine ranking
# ---------------------------------------------------------------------------


def get_accessible_names(
    conn: sqlite3.Connection,
    theorem_full_name: str,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
) -> list[str]:
    tid = name_to_id.get(theorem_full_name)
    if tid is None:
        return []
    pids = get_accessible_premises(conn, tid)
    return [id_to_name[pid] for pid in pids if pid in id_to_name]


def cosine_rank(
    goal_text: str,
    candidates: list[str],
    encoder,
    top_k: int = 5,
) -> list[tuple[float, str]]:
    if not candidates or encoder is None:
        return [(0.0, c) for c in candidates[:top_k]]
    model = encoder
    goal_emb = model.encode([goal_text], normalize_embeddings=True)
    cand_embs = model.encode(candidates, normalize_embeddings=True)
    scores = (goal_emb @ cand_embs.T).flatten()
    ranked = sorted(zip(scores.tolist(), candidates), reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _tokens(name: str) -> set[str]:
    """Split camelCase/snake_case/namespace name into tokens."""
    # Split on dots, underscores, and camelCase boundaries
    parts = re.split(r"[._]", name)
    tokens: set[str] = set()
    for p in parts:
        # camelCase split
        for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", p):
            tokens.add(sub.lower())
    return tokens


def _hypothesis_names(goal_state: str) -> set[str]:
    """Extract hypothesis names from the goal state."""
    names: set[str] = set()
    for line in goal_state.split("\n"):
        line = line.strip()
        m = re.match(r"^(\w[\w'✝]*)(?:\s+\w[\w'✝]*)*\s*:", line)
        if m and m.group(1) not in ("case", "⊢"):
            names.add(m.group(1))
    return names


def compute_features(
    candidate: str,
    cosine_score: float,
    goal_head: str,
    goal_shape: str,
    theorem_namespace: str,
    hyp_names: set[str],
) -> dict:
    name_lower = candidate.lower()

    # head_compat: name contains goal head symbol
    head_compat = 0
    if goal_head and len(goal_head) >= 3:
        h = goal_head.lower()
        if h in name_lower or h in candidate.split(".")[-1].lower():
            head_compat = 1

    # shape_compat: name suffix encodes conclusion shape
    shape_compat = 0
    suffixes = _SHAPE_SUFFIXES.get(goal_shape, [])
    if any(suf in name_lower for suf in suffixes):
        shape_compat = 1

    # conclusion_sfx: ends in a recognizable conclusion suffix
    _CONCL_SFXS = ["_iff", "_eq", "_le", "_lt", "_mem", "_surjective", "_injective",
                   "_continuous", "_tendsto", "_measurable", "_mono", "_nonneg"]
    conclusion_sfx = 1 if any(name_lower.endswith(s) or s + "_" in name_lower
                               for s in _CONCL_SFXS) else 0

    # namespace_match: shared top-level namespace
    cand_ns = ".".join(candidate.split(".")[:2]) if "." in candidate else ""
    thm_ns = ".".join(theorem_namespace.split(".")[:2]) if "." in theorem_namespace else ""
    namespace_match = 1 if cand_ns and cand_ns == thm_ns else 0

    # local_overlap: candidate tokens ∩ hypothesis name tokens
    cand_tokens = _tokens(candidate)
    hyp_tokens: set[str] = set()
    for h in hyp_names:
        hyp_tokens |= _tokens(h)
    local_overlap = len(cand_tokens & hyp_tokens) / max(len(cand_tokens), 1)

    return {
        "head_compat": head_compat,
        "shape_compat": shape_compat,
        "conclusion_sfx": conclusion_sfx,
        "namespace_match": namespace_match,
        "local_overlap": local_overlap,
        "cosine_score": cosine_score,
    }


# ---------------------------------------------------------------------------
# Reranking rules
# ---------------------------------------------------------------------------

RULES = [
    "cosine_top1",
    "rule_head",
    "rule_shape",
    "rule_head_shape",
    "rule_local",
    "rule_combined",
    "oracle_top5",
]


def rerank(
    candidates_with_features: list[tuple[str, dict]],
    rule: str,
) -> list[str]:
    """Return candidates reordered by rule."""
    if rule == "cosine_top1":
        return [c for c, _ in candidates_with_features]

    if rule == "oracle_top5":
        return [c for c, _ in candidates_with_features]  # ceiling = gold anywhere in list

    def _score(feats: dict, rule: str) -> float:
        if rule == "rule_head":
            return feats["head_compat"] * 10 + feats["cosine_score"]
        if rule == "rule_shape":
            return feats["shape_compat"] * 10 + feats["cosine_score"]
        if rule == "rule_head_shape":
            return feats["head_compat"] * 20 + feats["shape_compat"] * 10 + feats["cosine_score"]
        if rule == "rule_local":
            return feats["local_overlap"] * 5 + feats["cosine_score"]
        if rule == "rule_combined":
            return (feats["head_compat"] * 3 + feats["shape_compat"] * 2
                    + feats["local_overlap"] + feats["conclusion_sfx"]
                    + feats["namespace_match"] * 0.5 + feats["cosine_score"])
        return feats["cosine_score"]

    ranked = sorted(candidates_with_features, key=lambda x: _score(x[1], rule), reverse=True)
    return [c for c, _ in ranked]


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------


@dataclass
class ExResult:
    theorem_id: str
    subset: str
    gold: str
    n_accessible: int
    gold_in_top5: bool
    gold_cosine_rank: int      # rank among all accessible (0-based, -1 if not ranked)
    # Per-rule: rank of gold in reranked top-5 (0-based, -1 if not in top-5)
    ranks: dict


def eval_one(
    ex: dict,
    subset: str,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder,
) -> ExResult | None:
    gold = ex.get("annotated_premise", "")
    if not gold:
        return None

    theorem_id = ex["theorem_full_name"]
    goal_state = ex.get("goal_state_before", "")

    accessible = get_accessible_names(conn, theorem_id, id_to_name, name_to_id)
    if not accessible or gold not in accessible:
        return None  # gold not in accessible scope — skip

    # Cosine rank all accessible
    ranked = cosine_rank(goal_state, accessible, encoder, top_k=len(accessible))
    ranked_names = [c for _, c in ranked]
    gold_full_rank = next((i for i, c in enumerate(ranked_names) if c == gold), -1)
    top5 = ranked[:5]
    top5_names = [c for _, c in top5]
    gold_in_top5 = gold in top5_names

    # Compute features for top-5
    goal_head = extract_goal_head(goal_state)
    goal_shape = classify_goal_shape(goal_state)
    thm_ns = theorem_id
    hyp_names = _hypothesis_names(goal_state)

    cands_with_feats: list[tuple[str, dict]] = []
    for score, name in top5:
        feats = compute_features(name, score, goal_head, goal_shape, thm_ns, hyp_names)
        cands_with_feats.append((name, feats))

    # Evaluate each rule
    rule_ranks: dict[str, int] = {}
    for rule in RULES:
        if rule == "oracle_top5":
            rule_ranks[rule] = 0 if gold_in_top5 else -1
            continue
        reranked = rerank(cands_with_feats, rule)
        rule_ranks[rule] = next((i for i, c in enumerate(reranked) if c == gold), -1)

    return ExResult(
        theorem_id=theorem_id,
        subset=subset,
        gold=gold,
        n_accessible=len(accessible),
        gold_in_top5=gold_in_top5,
        gold_cosine_rank=gold_full_rank,
        ranks=rule_ranks,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[ExResult], subset_filter: str | None = None) -> None:
    if subset_filter:
        results = [r for r in results if r.subset == subset_filter]
    n = len(results)
    if n == 0:
        print(f"  No results for subset={subset_filter}")
        return

    top5_eligible = [r for r in results if r.gold_in_top5]
    ne = len(top5_eligible)

    print(f"  n={n}, gold_in_top5={ne}/{n} ({100*ne/n:.1f}%)")
    print(f"  {'Rule':<22} {'top1/started':>14} {'top1/eligible':>14} {'MRR@5/elig':>12}")
    print("  " + "-" * 64)

    for rule in RULES:
        # top1 rate among all started (gold may not be in top5)
        top1_all = sum(1 for r in results if r.ranks.get(rule, -1) == 0)
        # top1 rate among top5-eligible
        top1_elig = sum(1 for r in top5_eligible if r.ranks.get(rule, -1) == 0)
        # MRR@5 among eligible
        mrr = 0.0
        for r in top5_eligible:
            rank = r.ranks.get(rule, -1)
            if 0 <= rank < 5:
                mrr += 1.0 / (rank + 1)
        mrr /= max(ne, 1)
        print(f"  {rule:<22} {top1_all:>6}/{n:<6}  {top1_elig:>6}/{ne:<6}  {mrr:>10.3f}")


def print_full_report(results: list[ExResult]) -> None:
    print("\n" + "=" * 72)
    print("EXP-RANK-044: Typed reranker diagnostic (apply + refine)")
    print("=" * 72)

    for subset in ["apply", "refine_named", "refine_anon"]:
        sub = [r for r in results if r.subset == subset]
        if not sub:
            continue
        print(f"\n  [{subset}]")
        print_report(results, subset)

    print(f"\n  [all subsets combined]")
    print_report(results, None)

    # Feature importance: for top5-eligible, how often does each feature = 1 when gold is top5?
    eligible = [r for r in results if r.gold_in_top5]
    print(f"\n  Top-5-eligible examples: {len(eligible)}")

    # Improvement summary
    apply_elig = [r for r in eligible if r.subset == "apply"]
    refine_elig = [r for r in eligible if r.subset == "refine_named"]
    for label, sub in [("apply", apply_elig), ("refine_named", refine_elig)]:
        if not sub:
            continue
        cos_top1 = sum(1 for r in sub if r.ranks.get("cosine_top1", -1) == 0)
        best_top1 = max(
            sum(1 for r in sub if r.ranks.get(rule, -1) == 0)
            for rule in RULES if rule not in ("cosine_top1", "oracle_top5")
        )
        oracle = sum(1 for r in sub if r.ranks.get("oracle_top5", -1) == 0)
        print(f"\n  {label}: cosine_top1={cos_top1}/{len(sub)}  "
              f"best_rule={best_top1}/{len(sub)}  oracle_ceiling={oracle}/{len(sub)}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-RANK-044: typed reranker diagnostic")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--eval", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--output", default="runs/rank044_results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = load_examples(args.eval, ["apply", "refine"])
    logger.info("Loaded %d step-0 apply+refine examples", len(examples))

    if args.limit > 0:
        examples = examples[: args.limit]

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded MiniLM encoder")
    except ImportError:
        logger.warning("sentence-transformers not available")

    conn = sqlite3.connect(args.db)
    rows = conn.execute("SELECT id, name FROM entities").fetchall()
    id_to_name = {eid: name for eid, name in rows}
    name_to_id = {name: eid for eid, name in rows}

    results: list[ExResult] = []
    skipped = 0
    for i, ex in enumerate(examples):
        if i % 50 == 0:
            logger.info("[%d/%d] subset=%s gold_in_top5=%d/%d",
                        i, len(examples), ex.get("family"), sum(r.gold_in_top5 for r in results), len(results))
        subset = label_subset(ex)
        r = eval_one(ex, subset, conn, id_to_name, name_to_id, encoder)
        if r is None:
            skipped += 1
        else:
            results.append(r)

    conn.close()
    logger.info("Done: %d results, %d skipped (no gold or gold not in scope)", len(results), skipped)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            row = {
                "theorem_id": r.theorem_id,
                "subset": r.subset,
                "gold": r.gold,
                "n_accessible": r.n_accessible,
                "gold_in_top5": r.gold_in_top5,
                "gold_cosine_rank": r.gold_cosine_rank,
                "ranks": r.ranks,
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Written to %s", args.output)

    print_full_report(results)


if __name__ == "__main__":
    main()
