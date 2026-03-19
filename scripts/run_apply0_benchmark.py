"""apply0 benchmark — evaluate apply tactic prediction on step-0 apply examples.

Subsets:
  apply_pure:  tactic_base=="apply", no sub-goals argument
  refine_pure: tactic_base=="refine"
  exact_pure:  tactic_base=="exact"
  (run separately, reported per subset)

Four conditions per started goal (shared start set):
  1. oracle:      GT canonical_action_ir tactic
  2. cosine_top1: apply [top-1 cosine symbol]
  3. cosine_top5: try each of top-5 cosine symbols as apply target
  4. head_top1:   apply top-1 from head_match tier (if non-empty), else cosine_top1

Primary metric: LeanAccepted|started (apply can leave subgoals)
Secondary: GoalClosed|started, gold_in_scope by tier, scope size

Usage:
    python -m scripts.run_apply0_benchmark \\
        --db data/proof_network_v3.db \\
        --apply0 data/canonical/canonical_residual_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from src.apply_scoper import ApplyScope, gold_in_scope, scope_for_apply
from src.lean_interface import LeanConfig, LeanKernel, ReplayResult, ServerCrashError
from src.proof_network import get_accessible_premises

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ApplyExampleResult:
    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    goal_state_before: str = ""
    canonical_action_ir: str = ""
    annotated_premise: str = ""
    subset: str = ""   # "apply_pure" | "refine_pure" | "exact_pure"

    # Goal creation
    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    # Scope
    n_accessible_premises: int = 0
    n_scope_symbols: int = 0
    goal_head: str = ""
    goal_shape: str = ""
    gold_in_scope: bool = False
    gold_scope_tier: str = ""        # "local_hyp" | "head_match" | "shape_match" | "premise" | ""
    n_head_matches: int = 0
    n_shape_matches: int = 0

    # Condition 1: oracle
    oracle_accepted: bool = False
    oracle_closed: bool = False
    oracle_error: str = ""

    # Condition 2: cosine top-1
    cosine_top1_accepted: bool = False
    cosine_top1_closed: bool = False
    cosine_top1_tactic: str = ""
    cosine_top1_error: str = ""

    # Condition 3: cosine top-5 (best of 5)
    cosine_top5_accepted: bool = False
    cosine_top5_closed: bool = False
    cosine_top5_tactic: str = ""
    cosine_top5_rank: int = -1

    # Condition 4: head_top1 (top-1 from head_match tier)
    head_top1_accepted: bool = False
    head_top1_closed: bool = False
    head_top1_tactic: str = ""
    head_top1_error: str = ""

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_subset(example: dict) -> str:
    base = example.get("tactic_base", "")
    if base == "apply":
        return "apply_pure"
    if base == "refine":
        return "refine_pure"
    if base == "exact":
        return "exact_pure"
    return "other"


def extract_apply_target(canonical_ir: str) -> str:
    """Extract the lemma name from 'apply LemmaName ...' canonical IR."""
    m = re.match(r"(?:apply|refine|exact)\s+(.+)", canonical_ir.strip())
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Premise lookup
# ---------------------------------------------------------------------------

def load_entity_maps(conn: sqlite3.Connection) -> tuple[dict[int, str], dict[str, int]]:
    rows = conn.execute("SELECT id, name FROM entities").fetchall()
    id_to_name = {eid: name for eid, name in rows}
    name_to_id = {name: eid for eid, name in rows}
    return id_to_name, name_to_id


def get_premise_names(
    conn: sqlite3.Connection,
    theorem_full_name: str,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
) -> list[str]:
    tid = name_to_id.get(theorem_full_name)
    if tid is None:
        return []
    premise_ids = get_accessible_premises(conn, tid)
    return [id_to_name[pid] for pid in premise_ids if pid in id_to_name]


# ---------------------------------------------------------------------------
# Cosine ranking
# ---------------------------------------------------------------------------

def cosine_rank_symbols(
    goal_text: str,
    symbols: list[str],
    encoder: object | None,
) -> list[tuple[float, str]]:
    if not symbols or encoder is None:
        return [(0.0, s) for s in symbols]
    try:
        from sentence_transformers import SentenceTransformer
        model: SentenceTransformer = encoder  # type: ignore[assignment]
        goal_emb = model.encode([goal_text], normalize_embeddings=True)
        sym_embs = model.encode(symbols, normalize_embeddings=True)
        scores = (goal_emb @ sym_embs.T).flatten()
        return sorted(zip(scores.tolist(), symbols), reverse=True)
    except Exception as e:
        logger.warning("Cosine encoding failed: %s", e)
        return [(0.0, s) for s in symbols]


# ---------------------------------------------------------------------------
# Tactic execution
# ---------------------------------------------------------------------------

def try_tactic_safe(
    kernel: LeanKernel,
    goal_state: str,
    tactic: str,
    goal_id: int = 0,
) -> tuple[bool, bool, str, bool]:
    """(accepted, closed_all, error, was_crash)"""
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        closed = result.success and not result.new_goals
        return result.success, closed, result.error_message, False
    except ServerCrashError as e:
        return False, False, f"server_crash: {e}", True
    except Exception as e:
        return False, False, str(e), False


# ---------------------------------------------------------------------------
# Per-example runner
# ---------------------------------------------------------------------------

def run_one_example(
    example: dict,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder: object | None = None,
    skip_cosine: bool = False,
    project_root: str = "",
) -> ApplyExampleResult:
    t0 = time.time()
    res = ApplyExampleResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example.get("file_path", ""),
        step_index=example.get("step_index", 0),
        goal_state_before=example.get("goal_state_before", ""),
        canonical_action_ir=example.get("canonical_action_ir", ""),
        annotated_premise=example.get("annotated_premise", ""),
        subset=classify_subset(example),
    )

    # --- Goal creation ---
    prefix_tactics = example.get("prefix_tactics", [])
    replay: ReplayResult = kernel.goal_via_file_context(
        theorem_full_name=res.theorem_full_name,
        file_path=res.file_path,
        prefix_tactics=prefix_tactics,
        expected_goal=res.goal_state_before,
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

    # --- Scope ---
    premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
    res.n_accessible_premises = len(premise_names)
    scope = scope_for_apply(
        res.goal_state_before or goal_str,
        premise_names,
        max_head_matches=10,
        max_shape_matches=10,
        max_premises=20,
    )
    res.n_scope_symbols = len(scope.all_symbols)
    res.goal_head = scope.goal_head
    res.goal_shape = scope.goal_shape
    res.n_head_matches = len(scope.head_matches)
    res.n_shape_matches = len(scope.shape_matches)

    # Gold recall
    gold_prem = res.annotated_premise
    if gold_prem:
        in_s, tier = gold_in_scope(gold_prem, scope)
        res.gold_in_scope = in_s
        res.gold_scope_tier = tier

    crashed = False

    # --- Condition 1: oracle ---
    oracle_tactic = res.canonical_action_ir or example.get("tactic_text", "")
    if oracle_tactic:
        acc, closed, err, crash = try_tactic_safe(kernel, goal_str, oracle_tactic, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
            res.oracle_error = err
            crashed = True
        else:
            res.oracle_accepted = acc
            res.oracle_closed = closed
            res.oracle_error = err

    # --- Cosine ranking (shared for conditions 2-4) ---
    ranked: list[tuple[float, str]] = []
    if not crashed and not skip_cosine and scope.all_symbols and encoder is not None:
        ranked = cosine_rank_symbols(goal_str, scope.all_symbols, encoder)

    # --- Condition 2: cosine top-1 ---
    if not crashed and ranked:
        top1_sym = ranked[0][1]
        tactic1 = f"apply {top1_sym}"
        res.cosine_top1_tactic = tactic1
        acc, closed, err, crash = try_tactic_safe(kernel, goal_str, tactic1, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
            crashed = True
        else:
            res.cosine_top1_accepted = acc
            res.cosine_top1_closed = closed
            res.cosine_top1_error = err

    # --- Condition 3: cosine top-5 (best accepted) ---
    if not crashed and ranked:
        for rank, (_, sym) in enumerate(ranked[:5]):
            tactic5 = f"apply {sym}"
            acc, closed, err, crash = try_tactic_safe(kernel, goal_str, tactic5, goal_id=goal_id)
            if crash:
                res.crash_retries += 1
                crashed = True
                break
            if acc:
                res.cosine_top5_accepted = True
                res.cosine_top5_closed = closed
                res.cosine_top5_tactic = tactic5
                res.cosine_top5_rank = rank
                break

    # --- Condition 4: head_top1 ---
    if not crashed and scope.head_matches:
        # Rank head_match symbols by cosine if encoder available, else use order
        head_syms = scope.head_matches
        if ranked:
            ranked_heads = [(s, sc) for sc, s in ranked if s in set(head_syms)]
            if ranked_heads:
                head_syms = [s for s, _ in ranked_heads]
        top_head = head_syms[0]
        tactic_h = f"apply {top_head}"
        res.head_top1_tactic = tactic_h
        acc, closed, err, crash = try_tactic_safe(kernel, goal_str, tactic_h, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
        else:
            res.head_top1_accepted = acc
            res.head_top1_closed = closed
            res.head_top1_error = err

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100*num/den:.1f}%" if den > 0 else "N/A"


def print_report(results: list[ApplyExampleResult], total: int) -> None:
    started = [r for r in results if r.goal_started]
    n = len(started)

    # Subsets
    def sub(tag: str) -> list[ApplyExampleResult]:
        return [r for r in started if r.subset == tag]

    apply_s = sub("apply_pure")
    refine_s = sub("refine_pure")
    exact_s = sub("exact_pure")

    crashes = sum(r.crash_retries for r in results)
    fail_cats: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.goal_started:
            fail_cats[r.failure_category or "unknown"] += 1

    print("\n" + "=" * 64)
    print("apply0 Benchmark Report")
    print("=" * 64)

    print(f"\nGoalStart@apply0: {n}/{total} ({_pct(n, total)})")
    print(f"  apply_pure:  {len(apply_s)}/{sum(1 for r in results if r.subset=='apply_pure')}")
    print(f"  refine_pure: {len(refine_s)}/{sum(1 for r in results if r.subset=='refine_pure')}")
    print(f"  exact_pure:  {len(exact_s)}/{sum(1 for r in results if r.subset=='exact_pure')}")
    print(f"  Server crashes: {crashes}")
    if fail_cats:
        print("\n  Failure breakdown:")
        for cat, cnt in sorted(fail_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")

    if n == 0:
        print("\nNo goals started.")
        return

    def _row(label: str, subset: list[ApplyExampleResult], field: str) -> None:
        k = sum(1 for r in subset if getattr(r, field))
        ns = len(subset)
        print(f"  {label:<28} {k:>4}/{ns:<4} ({_pct(k, ns)})")

    print(f"\nLeanAccepted|started (N={n}) — PRIMARY:")
    _row("oracle", started, "oracle_accepted")
    _row("cosine_top1", started, "cosine_top1_accepted")
    _row("cosine_top5", started, "cosine_top5_accepted")
    _row("head_top1", started, "head_top1_accepted")

    print(f"\nGoalClosed|started (N={n}) — secondary:")
    _row("oracle", started, "oracle_closed")
    _row("cosine_top1", started, "cosine_top1_closed")
    _row("cosine_top5", started, "cosine_top5_closed")
    _row("head_top1", started, "head_top1_closed")

    # Per-subset
    for label, subset in [("apply_pure", apply_s), ("refine_pure", refine_s)]:
        ns = len(subset)
        if ns == 0:
            continue
        print(f"\n  {label} (N={ns}):")
        for cond, field in [
            ("oracle", "oracle_accepted"),
            ("cosine_top1", "cosine_top1_accepted"),
            ("cosine_top5", "cosine_top5_accepted"),
            ("head_top1", "head_top1_accepted"),
        ]:
            k = sum(1 for r in subset if getattr(r, field))
            print(f"    {cond:<20} {k:>4}/{ns:<4} ({_pct(k, ns)})")

    # Scope stats
    gold_s = [r for r in started if r.annotated_premise]
    in_scope = [r for r in gold_s if r.gold_in_scope]
    tier_counts: dict[str, int] = defaultdict(int)
    for r in in_scope:
        tier_counts[r.gold_scope_tier] += 1

    scope_sizes = [r.n_scope_symbols for r in started if r.n_scope_symbols > 0]
    head_sizes = [r.n_head_matches for r in started]

    print(f"\nScope stats:")
    if scope_sizes:
        print(f"  Mean scope size: {np.mean(scope_sizes):.1f}")
        print(f"  Mean head_matches: {np.mean(head_sizes):.1f}")
    print(f"  Gold in scope: {len(in_scope)}/{len(gold_s)} ({_pct(len(in_scope), len(gold_s))})")
    if in_scope:
        print(f"  Gold tier breakdown:")
        for tier in ["local_hyp", "head_match", "shape_match", "premise"]:
            cnt = tier_counts.get(tier, 0)
            print(f"    {tier:<15} {cnt:>4} ({_pct(cnt, len(in_scope))})")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="apply0 benchmark runner")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--apply0", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/apply0_results.jsonl")
    parser.add_argument("--subset", choices=["apply_pure", "refine_pure", "exact_pure", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-cosine", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-every", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load step-0 apply-family examples
    logger.info("Loading apply0 data from %s", args.apply0)
    examples = []
    with open(args.apply0) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e.get("step_index", 0) == 0 and e.get("tactic_base", "") in ("apply", "refine", "exact"):
                    examples.append(e)
    logger.info("Loaded %d apply-family step-0 examples", len(examples))

    if args.subset != "all":
        examples = [e for e in examples if classify_subset(e) == args.subset]
        logger.info("Filtered to %d examples (subset=%s)", len(examples), args.subset)

    if args.limit > 0:
        examples = examples[:args.limit]

    examples.sort(key=lambda x: (x.get("file_path", ""), x.get("theorem_full_name", ""), x.get("step_index", 0)))
    total = len(examples)

    # Resume
    done_keys: set[tuple[str, int]] = set()
    prior_results: list[ApplyExampleResult] = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    key = (d["theorem_full_name"], d["step_index"])
                    done_keys.add(key)
                    prior_results.append(ApplyExampleResult(**{
                        k: v for k, v in d.items()
                        if k in ApplyExampleResult.__dataclass_fields__
                    }))
        logger.info("Resuming: %d examples already done", len(done_keys))

    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)
    logger.info("Loaded %d entities from DB", len(id_to_name))

    encoder = None
    if not args.skip_cosine:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded MiniLM encoder")
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping cosine")
            args.skip_cosine = True

    kernel = LeanKernel(LeanConfig(
        backend="pantograph",
        timeout=120,
        project_root=args.lean_project,
        imports=["Mathlib"],
    ))
    logger.info("Initializing Pantograph with Mathlib (~40s)...")

    results = list(prior_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume else "w"

    n_since_restart = 0

    with open(args.output, mode) as out_f:
        for i, ex in enumerate(examples):
            key = (ex["theorem_full_name"], ex["step_index"])
            if key in done_keys:
                continue

            n_since_restart += 1
            if n_since_restart >= args.restart_every:
                logger.info("Periodic server restart")
                kernel.gc()
                kernel._restart_server()
                n_since_restart = 0

            if (i + 1) % 10 == 0 or i == 0:
                n_done = len(done_keys) + len(results) - len(prior_results)
                started_so_far = sum(1 for r in results if r.goal_started)
                oracle_so_far = sum(1 for r in results if r.oracle_accepted)
                logger.info(
                    "Progress: %d/%d (%.0f%%) — started: %d, oracle_accepted: %d",
                    n_done, total, 100 * n_done / max(total, 1),
                    started_so_far, oracle_so_far,
                )

            try:
                res = run_one_example(
                    example=ex,
                    kernel=kernel,
                    conn=conn,
                    id_to_name=id_to_name,
                    name_to_id=name_to_id,
                    encoder=encoder,
                    skip_cosine=args.skip_cosine,
                    project_root=args.lean_project,
                )
            except Exception as e:
                logger.error("Unhandled error on %s: %s", ex["theorem_full_name"], e)
                res = ApplyExampleResult(
                    theorem_full_name=ex["theorem_full_name"],
                    file_path=ex.get("file_path", ""),
                    step_index=ex.get("step_index", 0),
                    subset=classify_subset(ex),
                    failure_category="unhandled_error",
                )

            results.append(res)
            done_keys.add(key)
            out_f.write(json.dumps(asdict(res)) + "\n")
            out_f.flush()

    print_report(results, total)
    kernel.close()
    conn.close()


if __name__ == "__main__":
    main()
