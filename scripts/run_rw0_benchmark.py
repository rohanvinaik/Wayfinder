"""rw0 benchmark — evaluate rewrite tactic prediction on bare single-rewrite examples.

Three conditions (shared start set):
  1. Oracle canonical: GT canonical_action_ir tactic
  2. Cosine top-1: cosine-ranked from real accessible scope
  3. Learned top-1: pointer decoder from real scope (if checkpoint exists)

Usage:
    python -m scripts.run_rw0_benchmark --db data/proof_network_v3.db --rw0 data/canonical/rw0_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult, ServerCrashError
from src.proof_network import get_accessible_premises
from src.rw_scoper import scope_for_rw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExampleResult:
    """Result for one rw0 example across all conditions."""

    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    goal_state_before: str = ""
    canonical_action_ir: str = ""
    annotated_premise: str = ""

    # Goal creation
    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    # Scope
    n_accessible_premises: int = 0
    n_scope_symbols: int = 0
    gold_in_accessible: bool = False
    gold_in_scope: bool = False

    # Conditions
    oracle_success: bool = False
    oracle_error: str = ""
    oracle_has_remaining_goals: bool = False
    cosine_success: bool = False
    cosine_error: str = ""
    cosine_rank: int = -1
    cosine_tactic: str = ""
    learned_success: bool = False
    learned_error: str = ""
    learned_rank: int = -1
    learned_tactic: str = ""

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Premise lookup
# ---------------------------------------------------------------------------

def load_entity_maps(conn: sqlite3.Connection) -> tuple[dict[int, str], dict[str, int]]:
    """Load entity ID ↔ name maps from DB."""
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
    """Get accessible premise names for a theorem."""
    tid = name_to_id.get(theorem_full_name)
    if tid is None:
        return []
    premise_ids = get_accessible_premises(conn, tid)
    return [id_to_name[pid] for pid in premise_ids if pid in id_to_name]


# ---------------------------------------------------------------------------
# Cosine baseline
# ---------------------------------------------------------------------------

def cosine_rank_symbols(
    goal_text: str,
    symbols: list[str],
    encoder: object | None = None,
) -> list[tuple[float, str]]:
    """Rank symbols by cosine similarity to goal text.

    Returns list of (score, symbol_name) sorted descending.
    """
    if not symbols or encoder is None:
        return [(0.0, s) for s in symbols]

    try:
        from sentence_transformers import SentenceTransformer
        model: SentenceTransformer = encoder  # type: ignore[assignment]
        goal_emb = model.encode([goal_text], normalize_embeddings=True)
        sym_embs = model.encode(symbols, normalize_embeddings=True)
        scores = (goal_emb @ sym_embs.T).flatten()
        ranked = sorted(zip(scores.tolist(), symbols), reverse=True)
        return ranked
    except Exception as e:
        logger.warning("Cosine encoding failed: %s", e)
        return [(0.0, s) for s in symbols]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def try_tactic_safe(
    kernel: LeanKernel,
    goal_state: str,
    tactic: str,
    goal_id: int = 0,
) -> tuple[bool, str, bool, bool]:
    """Try a tactic, return (success, error_message, has_remaining_goals, was_crash).

    The was_crash flag distinguishes infrastructure failures (server died)
    from semantic failures (tactic wrong). Callers should not count crashes
    as model/tactic misses.

    goal_id: index into GoalState.goals for Site addressing after
    branching replay. Default 0 targets the first/only goal.
    """
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        has_remaining = bool(result.new_goals) and result.success
        return result.success, result.error_message, has_remaining, False
    except ServerCrashError as e:
        return False, f"server_crash: {e}", False, True
    except Exception as e:
        return False, str(e), False, False


def run_one_example(
    example: dict,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder: object | None = None,
    decoder: object | None = None,
    skip_cosine: bool = False,
    skip_learned: bool = False,
    project_root: str = "",
) -> ExampleResult:
    """Run all conditions on one rw0 example."""
    t0 = time.time()
    res = ExampleResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example["file_path"],
        step_index=example["step_index"],
        goal_state_before=example.get("goal_state_before", ""),
        canonical_action_ir=example.get("canonical_action_ir", ""),
        annotated_premise=example.get("annotated_premise", ""),
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

    scope = scope_for_rw(res.goal_state_before or goal_str, premise_names, max_premises=30)
    res.n_scope_symbols = len(scope.all_symbols)

    # Check gold in scope
    gold_premise = res.annotated_premise
    if gold_premise:
        res.gold_in_accessible = gold_premise in premise_names
        res.gold_in_scope = gold_premise in scope.all_symbols

    # --- Condition 1: Oracle canonical ---
    # After any crash, abort remaining conditions for this example.
    # The server restart invalidates cached goal states, so later
    # conditions would evaluate against a different (or absent) state.
    example_crashed = False

    oracle_tactic = res.canonical_action_ir
    if oracle_tactic:
        ok, err, remaining, crashed = try_tactic_safe(kernel, goal_str, oracle_tactic, goal_id=goal_id)
        if crashed:
            res.crash_retries += 1
            res.oracle_error = err
            example_crashed = True
        else:
            res.oracle_success = ok
            res.oracle_error = err
            res.oracle_has_remaining_goals = remaining

    # --- Condition 2: Cosine top-1 ---
    if not example_crashed and not skip_cosine and scope.all_symbols and encoder is not None:
        ranked = cosine_rank_symbols(goal_str, scope.all_symbols, encoder)
        if ranked:
            # Try top-1 forward and backward
            top_sym = ranked[0][1]
            res.cosine_rank = 0

            for tactic in [f"rw [{top_sym}]", f"rw [← {top_sym}]"]:
                ok, err, _, crashed = try_tactic_safe(kernel, goal_str, tactic, goal_id=goal_id)
                if crashed:
                    res.crash_retries += 1
                    res.cosine_error = err
                    example_crashed = True
                    break
                if ok:
                    res.cosine_success = True
                    res.cosine_tactic = tactic
                    res.cosine_error = ""
                    break
                res.cosine_error = err
                res.cosine_tactic = tactic

            # Find rank of gold premise in cosine ranking
            if not example_crashed and gold_premise:
                for i, (_, sym) in enumerate(ranked):
                    if sym == gold_premise:
                        res.cosine_rank = i
                        break

    # --- Condition 3: Learned decoder ---
    if not example_crashed and not skip_learned and decoder is not None and scope.all_symbols:
        try:
            import torch
            from src.rw_decoder import RwDecoder

            model: RwDecoder = decoder  # type: ignore[assignment]
            # Encode goal + scope
            goal_emb = encoder.encode([goal_str], normalize_embeddings=True)  # type: ignore[union-attr]
            vocab_embs = encoder.encode(scope.all_symbols, normalize_embeddings=True)  # type: ignore[union-attr]
            goal_t = torch.tensor(goal_emb[0], dtype=torch.float32)
            vocab_t = torch.tensor(vocab_embs, dtype=torch.float32)

            pred = model.predict(goal_t, vocab_t, scope.all_symbols)
            tactic_str = pred.to_action_ir().lower()
            res.learned_tactic = tactic_str

            ok, err, _, crashed = try_tactic_safe(kernel, goal_str, tactic_str, goal_id=goal_id)
            if crashed:
                res.crash_retries += 1
                res.learned_error = err
            else:
                res.learned_success = ok
                res.learned_error = err

            # Find rank of predicted symbol
            if pred.symbol_indices:
                res.learned_rank = pred.symbol_indices[0]
        except ServerCrashError as e:
            res.crash_retries += 1
            res.learned_error = f"server_crash: {e}"
        except Exception as e:
            res.learned_error = str(e)

    res.elapsed_s = time.time() - t0
    return res


def _extract_target(goal_state: str) -> str:
    """Extract the ⊢ target from a goal state string."""
    for line in goal_state.split("\n"):
        if "⊢" in line:
            return line.split("⊢", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------

def print_report(results: list[ExampleResult], total: int) -> None:
    """Print the aggregate benchmark report."""
    started = [r for r in results if r.goal_started]
    step0 = [r for r in results if r.step_index == 0]
    step_gt0 = [r for r in results if r.step_index > 0]
    step0_started = [r for r in started if r.step_index == 0]
    step_gt0_started = [r for r in started if r.step_index > 0]

    gold_in_scope = [r for r in started if r.gold_in_scope]
    gold_in_accessible = [r for r in started if r.gold_in_accessible]

    crashes = sum(r.crash_retries for r in results)

    # Failure breakdown
    fail_cats: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.goal_started:
            fail_cats[r.failure_category or "unknown"] += 1

    print("\n" + "=" * 60)
    print("rw0 Benchmark Report")
    print("=" * 60)

    print(f"\nGoalStart@rw0: {len(started)}/{total} ({100*len(started)/max(total,1):.1f}%)")
    print(f"  Tier A (step-0): {len(step0_started)}/{len(step0)}")
    print(f"  Tier C (replay):  {len(step_gt0_started)}/{len(step_gt0)}")
    print(f"  Server crashes: {crashes}")

    if fail_cats:
        print("\n  Failure breakdown:")
        for cat, cnt in sorted(fail_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")

    n = len(started)
    if n == 0:
        print("\nNo goals started — cannot report validity metrics.")
        return

    oracle_ok = sum(1 for r in started if r.oracle_success)
    cosine_ok = sum(1 for r in started if r.cosine_success)
    learned_ok = sum(1 for r in started if r.learned_success)

    print(f"\nLeanValid@rw0 (primary, semantic mode):")
    print(f"  Oracle canonical: {oracle_ok}/{total} ({100*oracle_ok/max(total,1):.1f}%)")

    print(f"\nLeanValid@rw0|started (N={n}):")
    print(f"  Oracle canonical: {oracle_ok}/{n} ({100*oracle_ok/max(n,1):.1f}%)")
    print(f"  Cosine top-1:    {cosine_ok}/{n} ({100*cosine_ok/max(n,1):.1f}%)")
    print(f"  Learned top-1:   {learned_ok}/{n} ({100*learned_ok/max(n,1):.1f}%)")

    m = len(gold_in_scope)
    if m > 0:
        oracle_scope = sum(1 for r in gold_in_scope if r.oracle_success)
        cosine_scope = sum(1 for r in gold_in_scope if r.cosine_success)
        learned_scope = sum(1 for r in gold_in_scope if r.learned_success)
        print(f"\nLeanValid@rw0|started,gold_in_scope (M={m}):")
        print(f"  Oracle canonical: {oracle_scope}/{m} ({100*oracle_scope/max(m,1):.1f}%)")
        print(f"  Cosine top-1:    {cosine_scope}/{m} ({100*cosine_scope/max(m,1):.1f}%)")
        print(f"  Learned top-1:   {learned_scope}/{m} ({100*learned_scope/max(m,1):.1f}%)")

    print(f"\nStep breakdown:")
    s0_ok = sum(1 for r in step0_started if r.oracle_success)
    sg_ok = sum(1 for r in step_gt0_started if r.oracle_success)
    print(f"  Step-0:  {len(step0_started)}/{len(step0)} started, {s0_ok}/{len(step0_started)} valid (oracle)")
    print(f"  Step>0:  {len(step_gt0_started)}/{len(step_gt0)} started, {sg_ok}/{len(step_gt0_started)} valid (oracle)")

    scope_miss = sum(1 for r in started if r.annotated_premise and not r.gold_in_scope)
    acc_miss = sum(1 for r in started if r.annotated_premise and not r.gold_in_accessible)
    scope_sizes = [r.n_scope_symbols for r in started if r.n_scope_symbols > 0]
    print(f"\nScope stats:")
    print(f"  Gold in accessible: {len(gold_in_accessible)}/{n} ({100*len(gold_in_accessible)/max(n,1):.1f}%)")
    print(f"  Gold in scope: {m}/{n} ({100*m/max(n,1):.1f}%)")
    if scope_sizes:
        print(f"  Mean scope size: {np.mean(scope_sizes):.1f}")
    print(f"  data_scope_miss: {scope_miss}")
    print(f"  data_accessible_miss: {acc_miss}")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="rw0 benchmark runner")
    parser.add_argument("--db", default="data/proof_network_v3.db", help="Proof network DB path")
    parser.add_argument("--rw0", default="data/canonical/rw0_eval.jsonl", help="rw0 eval JSONL")
    parser.add_argument("--lean-project", default="data/lean_project/", help="Lean project root")
    parser.add_argument("--output", default="runs/rw0_results.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="", help="RW decoder checkpoint path")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    parser.add_argument("--skip-cosine", action="store_true", help="Skip cosine condition")
    parser.add_argument("--skip-learned", action="store_true", help="Skip learned condition")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--restart-every", type=int, default=100, help="Restart server every N examples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data
    logger.info("Loading rw0 data from %s", args.rw0)
    examples = []
    with open(args.rw0) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    logger.info("Loaded %d rw0 examples", len(examples))

    if args.limit > 0:
        examples = examples[:args.limit]

    # Sort by file_path for environment coherence
    examples.sort(key=lambda x: (x.get("file_path", ""), x.get("theorem_full_name", ""), x.get("step_index", 0)))
    total = len(examples)

    # Load existing results for resume
    done_keys: set[tuple[str, int]] = set()
    prior_results: list[ExampleResult] = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    key = (d["theorem_full_name"], d["step_index"])
                    done_keys.add(key)
                    prior_results.append(ExampleResult(**{
                        k: v for k, v in d.items() if k in ExampleResult.__dataclass_fields__
                    }))
        logger.info("Resuming: %d examples already done", len(done_keys))

    # Connect to DB
    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)
    logger.info("Loaded %d entities from DB", len(id_to_name))

    # Load encoder (for cosine baseline)
    encoder = None
    if not args.skip_cosine or not args.skip_learned:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded MiniLM encoder")
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping cosine/learned")
            args.skip_cosine = True
            args.skip_learned = True

    # Load decoder
    decoder = None
    if not args.skip_learned and args.model and os.path.exists(args.model):
        try:
            import torch
            from src.rw_decoder import RwDecoder
            checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
            decoder = RwDecoder()
            decoder.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
            decoder.eval()
            logger.info("Loaded RW decoder from %s", args.model)
        except Exception as e:
            logger.warning("Failed to load decoder: %s", e)

    # Initialize Lean kernel
    kernel = LeanKernel(LeanConfig(
        backend="pantograph",
        timeout=120,
        project_root=args.lean_project,
        imports=["Mathlib"],
    ))
    logger.info("Initializing Pantograph with Mathlib (this takes ~40s)...")

    # Run benchmark
    results = list(prior_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume else "w"

    n_since_restart = 0
    n_crashes_total = 0

    with open(args.output, mode) as out_f:
        for i, ex in enumerate(examples):
            key = (ex["theorem_full_name"], ex["step_index"])
            if key in done_keys:
                continue

            # Periodic server restart
            n_since_restart += 1
            if n_since_restart >= args.restart_every:
                logger.info("Periodic server restart (every %d examples)", args.restart_every)
                kernel.gc()
                kernel._restart_server()
                n_since_restart = 0

            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                n_done = len(done_keys) + len(results) - len(prior_results)
                logger.info(
                    "Progress: %d/%d (%.0f%%) — started: %d, oracle: %d, crashes: %d",
                    n_done, total, 100 * n_done / max(total, 1),
                    sum(1 for r in results if r.goal_started),
                    sum(1 for r in results if r.oracle_success),
                    n_crashes_total,
                )

            try:
                res = run_one_example(
                    example=ex,
                    kernel=kernel,
                    conn=conn,
                    id_to_name=id_to_name,
                    name_to_id=name_to_id,
                    encoder=encoder,
                    decoder=decoder,
                    skip_cosine=args.skip_cosine,
                    skip_learned=args.skip_learned,
                    project_root=args.lean_project,
                )
            except Exception as e:
                logger.error("Unhandled error on %s step %d: %s", ex["theorem_full_name"], ex["step_index"], e)
                res = ExampleResult(
                    theorem_full_name=ex["theorem_full_name"],
                    file_path=ex.get("file_path", ""),
                    step_index=ex["step_index"],
                    failure_category="unhandled_error",
                )

            n_crashes_total += res.crash_retries
            results.append(res)
            done_keys.add(key)

            # Flush result
            out_f.write(json.dumps(asdict(res)) + "\n")
            out_f.flush()

    # Report
    print_report(results, total)

    # Cleanup
    kernel.close()
    conn.close()


if __name__ == "__main__":
    main()
