"""simp0 benchmark — evaluate simp hint prediction on step-0 simp examples.

Subsets:
  simp_bare:  tactic_base=="simp", no args (bare simp)
  simp_hints: tactic_base=="simp", has_args (simp with lemmas)
  diagnostic: simpa / simp_all  (run but reported separately)

Four conditions per started goal (shared start set):
  1. oracle_bare:  bare `simp` with no hints
  2. oracle_hints: GT canonical_action_ir tactic (full lemma list)
  3. cosine_top1:  bare simp + simp [top-1 cosine hint]
  4. cosine_top5:  bare simp + simp [top-5 cosine hints]

Primary metric: LeanAccepted|started  (simp can succeed without closing all goals)
Secondary: GoalClosed|started, gold_hint_in_scope, scope size, failure taxonomy

Usage:
    python -m scripts.run_simp0_benchmark \\
        --db data/proof_network_v3.db \\
        --simp0 data/canonical/canonical_residual_eval.jsonl \\
        --lean-project data/lean_project/
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

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult, ServerCrashError
from src.proof_network import get_accessible_premises
from src.simp_scoper import gold_hints_in_scope, scope_for_simp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimpExampleResult:
    """Result for one simp0 example across all conditions."""

    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    goal_state_before: str = ""
    canonical_action_ir: str = ""
    annotated_premise: str = ""
    subset: str = ""           # "simp_bare" | "simp_hints" | "diagnostic"

    # Goal creation
    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    # Scope
    n_accessible_premises: int = 0
    n_scope_symbols: int = 0
    gold_hints_total: int = 0
    gold_hints_in_scope: int = 0

    # Condition 1: oracle bare
    oracle_bare_accepted: bool = False
    oracle_bare_closed: bool = False
    oracle_bare_error: str = ""

    # Condition 2: oracle hints
    oracle_hints_accepted: bool = False
    oracle_hints_closed: bool = False
    oracle_hints_error: str = ""

    # Condition 3: cosine top-1
    cosine_top1_accepted: bool = False
    cosine_top1_closed: bool = False
    cosine_top1_tactic: str = ""
    cosine_top1_error: str = ""

    # Condition 4: cosine top-5
    cosine_top5_accepted: bool = False
    cosine_top5_closed: bool = False
    cosine_top5_tactic: str = ""
    cosine_top5_error: str = ""

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_simp_hints(canonical_action_ir: str) -> list[str]:
    """Extract hint names from a canonical simp tactic string.

    Examples:
        'simp only [mul_comm, le_refl]'  -> ['mul_comm', 'le_refl']
        'simp [h, Finset.sum_empty]'      -> ['h', 'Finset.sum_empty']
        'simp'                            -> []
    """
    m = re.search(r"\[(.+)\]", canonical_action_ir, re.DOTALL)
    if not m:
        return []
    raw = m.group(1)
    # Split on commas, strip whitespace and ← arrows
    hints = []
    for part in raw.split(","):
        part = part.strip().lstrip("←").strip()
        if part:
            hints.append(part)
    return hints


def build_simp_tactic(hints: list[str]) -> str:
    """Build a `simp [h1, h2, ...]` tactic string."""
    if not hints:
        return "simp"
    return "simp [" + ", ".join(hints) + "]"


def classify_subset(example: dict) -> str:
    """Return 'simp_bare', 'simp_hints', or 'diagnostic'."""
    base = example.get("tactic_base", "")
    if base in ("simpa", "simp_all"):
        return "diagnostic"
    has_args = example.get("has_args", False)
    if has_args:
        return "simp_hints"
    return "simp_bare"


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
    """Rank symbols by cosine similarity to goal text, descending."""
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
    """Try a tactic. Returns (accepted, closed_all_goals, error_msg, was_crash).

    accepted: tactic was accepted (no syntax/type error), goals may remain
    closed_all_goals: accepted AND no remaining goals
    """
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        closed = result.success and not result.new_goals
        accepted = result.success  # success = no error, goals may remain
        return accepted, closed, result.error_message, False
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
) -> SimpExampleResult:
    t0 = time.time()
    res = SimpExampleResult(
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
    scope = scope_for_simp(res.goal_state_before or goal_str, premise_names, max_premises=30)
    res.n_scope_symbols = len(scope.all_symbols)

    # Gold hint recall
    gold_hints = parse_simp_hints(res.canonical_action_ir)
    in_scope, total = gold_hints_in_scope(gold_hints, scope)
    res.gold_hints_total = total
    res.gold_hints_in_scope = in_scope

    crashed = False

    # --- Condition 1: oracle bare ---
    acc, closed, err, crash = try_tactic_safe(kernel, goal_str, "simp", goal_id=goal_id)
    if crash:
        res.crash_retries += 1
        res.oracle_bare_error = err
        crashed = True
    else:
        res.oracle_bare_accepted = acc
        res.oracle_bare_closed = closed
        res.oracle_bare_error = err

    # --- Condition 2: oracle hints ---
    # For simp_bare examples canonical_action_ir is empty — use tactic_text as fallback.
    oracle_hints_tactic = res.canonical_action_ir or example.get("tactic_text", "")
    if not crashed and oracle_hints_tactic:
        oracle_tactic = oracle_hints_tactic
        acc, closed, err, crash = try_tactic_safe(kernel, goal_str, oracle_tactic, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
            res.oracle_hints_error = err
            crashed = True
        else:
            res.oracle_hints_accepted = acc
            res.oracle_hints_closed = closed
            res.oracle_hints_error = err

    # --- Conditions 3 & 4: cosine top-1 and top-5 ---
    if not crashed and not skip_cosine and scope.all_symbols and encoder is not None:
        ranked = cosine_rank_symbols(goal_str, scope.all_symbols, encoder)

        # top-1
        if ranked:
            top1 = [ranked[0][1]]
            tactic1 = build_simp_tactic(top1)
            res.cosine_top1_tactic = tactic1
            acc, closed, err, crash = try_tactic_safe(kernel, goal_str, tactic1, goal_id=goal_id)
            if crash:
                res.crash_retries += 1
                res.cosine_top1_error = err
                crashed = True
            else:
                res.cosine_top1_accepted = acc
                res.cosine_top1_closed = closed
                res.cosine_top1_error = err
                # Fallback: try bare simp if top-1 hint didn't work
                if not acc and not crash:
                    acc2, closed2, _, crash2 = try_tactic_safe(kernel, goal_str, "simp", goal_id=goal_id)
                    if crash2:
                        res.crash_retries += 1
                        crashed = True
                    elif acc2:
                        res.cosine_top1_accepted = True
                        res.cosine_top1_closed = closed2
                        res.cosine_top1_tactic = "simp"

        # top-5
        if not crashed and len(ranked) >= 1:
            top5 = [sym for _, sym in ranked[:5]]
            tactic5 = build_simp_tactic(top5)
            res.cosine_top5_tactic = tactic5
            acc, closed, err, crash = try_tactic_safe(kernel, goal_str, tactic5, goal_id=goal_id)
            if crash:
                res.crash_retries += 1
                res.cosine_top5_error = err
                crashed = True
            else:
                res.cosine_top5_accepted = acc
                res.cosine_top5_closed = closed
                res.cosine_top5_error = err
                if not acc and not crash:
                    acc2, closed2, _, crash2 = try_tactic_safe(kernel, goal_str, "simp", goal_id=goal_id)
                    if crash2:
                        res.crash_retries += 1
                        crashed = True
                    elif acc2:
                        res.cosine_top5_accepted = True
                        res.cosine_top5_closed = closed2
                        res.cosine_top5_tactic = "simp"

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{100*num/den:.1f}%"


def print_report(results: list[SimpExampleResult], total: int) -> None:
    started = [r for r in results if r.goal_started]
    n = len(started)

    bare_all = [r for r in results if r.subset == "simp_bare"]
    hints_all = [r for r in results if r.subset == "simp_hints"]
    diag_all = [r for r in results if r.subset == "diagnostic"]

    bare_s = [r for r in started if r.subset == "simp_bare"]
    hints_s = [r for r in started if r.subset == "simp_hints"]
    diag_s = [r for r in started if r.subset == "diagnostic"]

    crashes = sum(r.crash_retries for r in results)
    fail_cats: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.goal_started:
            fail_cats[r.failure_category or "unknown"] += 1

    print("\n" + "=" * 64)
    print("simp0 Benchmark Report")
    print("=" * 64)

    print(f"\nGoalStart@simp0: {n}/{total} ({_pct(n, total)})")
    print(f"  simp_bare:  {len(bare_s)}/{len(bare_all)}")
    print(f"  simp_hints: {len(hints_s)}/{len(hints_all)}")
    print(f"  diagnostic: {len(diag_s)}/{len(diag_all)}")
    print(f"  Server crashes: {crashes}")

    if fail_cats:
        print("\n  Failure breakdown:")
        for cat, cnt in sorted(fail_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")

    if n == 0:
        print("\nNo goals started — cannot report acceptance metrics.")
        return

    # --- Primary: LeanAccepted|started ---
    def _row(label: str, subset: list[SimpExampleResult], field: str) -> None:
        k = sum(1 for r in subset if getattr(r, field))
        ns = len(subset)
        print(f"  {label:<22} {k:>4}/{ns:<4} ({_pct(k, ns)})")

    print(f"\nLeanAccepted|started (N={n}) — PRIMARY:")
    _row("oracle_bare", started, "oracle_bare_accepted")
    _row("oracle_hints", started, "oracle_hints_accepted")
    _row("cosine_top1", started, "cosine_top1_accepted")
    _row("cosine_top5", started, "cosine_top5_accepted")

    print(f"\nGoalClosed|started (N={n}) — secondary:")
    _row("oracle_bare", started, "oracle_bare_closed")
    _row("oracle_hints", started, "oracle_hints_closed")
    _row("cosine_top1", started, "cosine_top1_closed")
    _row("cosine_top5", started, "cosine_top5_closed")

    # Subset breakdown
    for label, subset in [("simp_bare", bare_s), ("simp_hints", hints_s)]:
        ns = len(subset)
        if ns == 0:
            continue
        print(f"\n  {label} (N={ns}):")
        for cond, field in [
            ("oracle_bare", "oracle_bare_accepted"),
            ("oracle_hints", "oracle_hints_accepted"),
            ("cosine_top1", "cosine_top1_accepted"),
            ("cosine_top5", "cosine_top5_accepted"),
        ]:
            k = sum(1 for r in subset if getattr(r, field))
            print(f"    {cond:<20} {k:>4}/{ns:<4} ({_pct(k, ns)})")

    # Scope stats
    hints_started = [r for r in started if r.gold_hints_total > 0]
    scope_sizes = [r.n_scope_symbols for r in started if r.n_scope_symbols > 0]
    total_gold = sum(r.gold_hints_total for r in hints_started)
    in_scope_gold = sum(r.gold_hints_in_scope for r in hints_started)

    print(f"\nScope stats:")
    print(f"  Mean scope size: {np.mean(scope_sizes):.1f}" if scope_sizes else "  No scope data")
    if hints_started:
        print(f"  Gold hints in scope: {in_scope_gold}/{total_gold} ({_pct(in_scope_gold, total_gold)})")
        print(f"  Examples with ≥1 gold hint: {len(hints_started)}")
        full_recall = sum(1 for r in hints_started if r.gold_hints_in_scope == r.gold_hints_total)
        print(f"  Full gold recall (all hints in scope): {full_recall}/{len(hints_started)} ({_pct(full_recall, len(hints_started))})")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="simp0 benchmark runner")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--simp0", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/simp0_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0=all)")
    parser.add_argument("--subset", choices=["simp_bare", "simp_hints", "diagnostic", "all"],
                        default="all", help="Which subset to run")
    parser.add_argument("--skip-cosine", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-every", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data — step-0 simp only
    logger.info("Loading simp0 data from %s", args.simp0)
    examples = []
    with open(args.simp0) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e.get("step_index", 0) == 0 and e.get("tactic_base", "") in ("simp", "simpa", "simp_all"):
                    examples.append(e)

    logger.info("Loaded %d simp step-0 examples", len(examples))

    # Subset filter
    if args.subset != "all":
        examples = [e for e in examples if classify_subset(e) == args.subset]
        logger.info("Filtered to %d examples (subset=%s)", len(examples), args.subset)

    if args.limit > 0:
        examples = examples[:args.limit]

    examples.sort(key=lambda x: (x.get("file_path", ""), x.get("theorem_full_name", ""), x.get("step_index", 0)))
    total = len(examples)

    # Resume
    done_keys: set[tuple[str, int]] = set()
    prior_results: list[SimpExampleResult] = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    key = (d["theorem_full_name"], d["step_index"])
                    done_keys.add(key)
                    prior_results.append(SimpExampleResult(**{
                        k: v for k, v in d.items()
                        if k in SimpExampleResult.__dataclass_fields__
                    }))
        logger.info("Resuming: %d examples already done", len(done_keys))

    # DB
    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)
    logger.info("Loaded %d entities from DB", len(id_to_name))

    # Encoder
    encoder = None
    if not args.skip_cosine:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded MiniLM encoder")
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping cosine")
            args.skip_cosine = True

    # Lean kernel
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
                logger.info("Periodic server restart (every %d)", args.restart_every)
                kernel.gc()
                kernel._restart_server()
                n_since_restart = 0

            if (i + 1) % 10 == 0 or i == 0:
                n_done = len(done_keys) + len(results) - len(prior_results)
                started_so_far = sum(1 for r in results if r.goal_started)
                accepted_so_far = sum(1 for r in results if r.oracle_hints_accepted)
                logger.info(
                    "Progress: %d/%d (%.0f%%) — started: %d, oracle_hints_accepted: %d",
                    n_done, total, 100 * n_done / max(total, 1),
                    started_so_far, accepted_so_far,
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
                res = SimpExampleResult(
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
