"""rw2 bare-premise ablation (second pass).

Reads the existing rw2_step0_results.jsonl, re-creates goals only for started
examples, then runs three new conditions side-by-side:
  oracle_bare   — gold premise, no args, gold direction
  cosine1_bare  — cosine top-1, no args, both directions
  cosine5_bare  — cosine top-5, no args, both directions

Cross-tabulates against the original heuristic results to produce the
args-redundant / args-necessary / unexecutable-oracle partition.
"""
from __future__ import annotations

import argparse
import json
import os
import logging
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult
from src.rw_scoper import scope_for_rw
from scripts.run_rw2_benchmark import (
    cosine_rank_symbols,
    get_premise_names,
    load_entity_maps,
    resolve_gold_premise_name,
    try_tactic_safe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tactic builders
# ---------------------------------------------------------------------------

def _build_bare(premise: str, direction: str) -> str:
    arrow = "← " if direction == "backward" else ""
    return f"rw [{arrow}{premise}]"


def _gold_direction(grammar_tier: str) -> str:
    return "backward" if grammar_tier == "rw2_bwd_args" else "forward"


# ---------------------------------------------------------------------------
# Per-example second pass
# ---------------------------------------------------------------------------

@dataclass
class BareResult:
    theorem_full_name: str = ""
    step_index: int = 0
    grammar_tier: str = ""

    oracle_bare_success: bool = False
    oracle_bare_error: str = ""
    oracle_bare_tactic: str = ""

    cosine1_bare_success: bool = False
    cosine1_bare_tactic: str = ""
    cosine1_bare_rank: int = -1

    cosine5_bare_success: bool = False
    cosine5_bare_tactic: str = ""
    cosine5_bare_rank: int = -1

    cosine5_bare_calls: int = 0
    elapsed_s: float = 0.0


def run_bare_example(
    orig: dict,
    example: dict,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict,
    name_to_id: dict,
    encoder,
    project_root: str,
    cosine_topk: int = 5,
    max_scope_premises: int = 30,
) -> BareResult:
    t0 = time.time()
    res = BareResult(
        theorem_full_name=orig["theorem_full_name"],
        step_index=orig["step_index"],
        grammar_tier=orig["grammar_tier"],
    )

    # Re-create goal
    replay: ReplayResult = kernel.goal_via_file_context(
        theorem_full_name=orig["theorem_full_name"],
        file_path=orig["file_path"],
        prefix_tactics=example.get("prefix_tactics", []),
        expected_goal=orig.get("goal_state_before", ""),
        project_root=project_root,
        prefix_goal_states=example.get("prefix_goal_states", []),
    )
    if not replay.success:
        res.elapsed_s = time.time() - t0
        return res

    goal_str = replay.goal_state
    goal_id = replay.goal_id

    # Scope
    premise_names = get_premise_names(conn, orig["theorem_full_name"], id_to_name, name_to_id)
    scope = scope_for_rw(goal_str, premise_names, max_premises=max_scope_premises)
    ranked = cosine_rank_symbols(goal_str, scope.premises, encoder)
    gold_name = resolve_gold_premise_name(example, scope.premises)
    tier = orig["grammar_tier"]
    gold_dir = _gold_direction(tier)

    # --- oracle bare ---
    if gold_name:
        tac = _build_bare(gold_name, gold_dir)
        res.oracle_bare_tactic = tac
        ok, err, _, crashed = try_tactic_safe(kernel, goal_str, tac, goal_id=goal_id)
        if not crashed:
            res.oracle_bare_success = ok
            res.oracle_bare_error = err

    # --- cosine bare (top-1 and top-5) ---
    top_k = min(cosine_topk, len(ranked))
    top1_done = False
    top5_done = False
    for rank, (_, premise) in enumerate(ranked[:top_k]):
        for direction in ("forward", "backward"):
            tac = _build_bare(premise, direction)
            ok, err, _, crashed = try_tactic_safe(kernel, goal_str, tac, goal_id=goal_id)
            res.cosine5_bare_calls += 1
            if crashed:
                break
            if ok:
                if rank == 0 and not top1_done:
                    res.cosine1_bare_success = True
                    res.cosine1_bare_tactic = tac
                    res.cosine1_bare_rank = rank
                    top1_done = True
                if not top5_done:
                    res.cosine5_bare_success = True
                    res.cosine5_bare_tactic = tac
                    res.cosine5_bare_rank = rank
                    top5_done = True
                break
        if top5_done:
            break

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Partition logic
# ---------------------------------------------------------------------------

def partition_bucket(orig_row: dict, bare: BareResult) -> str:
    """
    args_redundant   — bare premise succeeds (heuristic arg beam not needed)
    args_necessary   — bare fails, original oracle/heuristic succeeds
    unexecutable     — oracle, heuristic, and bare all fail
    """
    orig_c5 = orig_row.get("cosine_top5_success", False)
    orig_oracle = orig_row.get("oracle_success", False)
    bare_ok = bare.cosine5_bare_success

    if bare_ok:
        return "args_redundant"
    if orig_c5 or orig_oracle:
        return "args_necessary"
    return "unexecutable"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(orig_rows: list[dict], bare_results: list[BareResult]) -> None:
    n = len(orig_rows)
    assert len(bare_results) == n

    ob_ok = sum(r.oracle_bare_success for r in bare_results)
    c1b_ok = sum(r.cosine1_bare_success for r in bare_results)
    c5b_ok = sum(r.cosine5_bare_success for r in bare_results)

    # Original heuristic numbers for comparison
    orig_oracle = sum(r.get("oracle_success", False) for r in orig_rows)
    orig_c1 = sum(r.get("cosine_top1_success", False) for r in orig_rows)
    orig_c5 = sum(r.get("cosine_top5_success", False) for r in orig_rows)

    print("\n" + "=" * 60)
    print("rw2 Bare-Premise Ablation")
    print("=" * 60)
    print(f"\nStarted examples: {n}")
    print(f"\n{'Condition':<35} {'Accepted':>10} {'Rate':>8}")
    print("-" * 55)
    print(f"{'Oracle bare (gold, gold-dir, no args)':<35} {ob_ok:>10} {100*ob_ok/n:>7.1f}%")
    print(f"{'Cosine top-1 bare':<35} {c1b_ok:>10} {100*c1b_ok/n:>7.1f}%")
    print(f"{'Cosine top-5 bare':<35} {c5b_ok:>10} {100*c5b_ok/n:>7.1f}%")
    print()
    print(f"{'--- Original EXP-RW-032 (for comparison) ---'}")
    print(f"{'Oracle qualified (with args)':<35} {orig_oracle:>10} {100*orig_oracle/n:>7.1f}%")
    print(f"{'Cosine top-1 + heuristic args':<35} {orig_c1:>10} {100*orig_c1/n:>7.1f}%")
    print(f"{'Cosine top-5 + heuristic args':<35} {orig_c5:>10} {100*orig_c5/n:>7.1f}%")

    # Partition
    buckets = Counter(partition_bucket(o, b) for o, b in zip(orig_rows, bare_results))
    print(f"\n{'=== rw2 Partition ==='}")
    total = sum(buckets.values())
    for bkt in ("args_redundant", "args_necessary", "unexecutable"):
        v = buckets[bkt]
        print(f"  {bkt:<25}: {v:>3}/{total} ({100*v/total:.1f}%)")

    # Cross-tab: bare vs heuristic
    both_ok = sum(1 for o, b in zip(orig_rows, bare_results)
                  if o.get("cosine_top5_success") and b.cosine5_bare_success)
    bare_only = sum(1 for o, b in zip(orig_rows, bare_results)
                    if not o.get("cosine_top5_success") and b.cosine5_bare_success)
    heur_only = sum(1 for o, b in zip(orig_rows, bare_results)
                    if o.get("cosine_top5_success") and not b.cosine5_bare_success)
    neither = sum(1 for o, b in zip(orig_rows, bare_results)
                  if not o.get("cosine_top5_success") and not b.cosine5_bare_success)

    print(f"\n{'=== Cosine-5 cross-tab: heuristic vs bare ==='}")
    print(f"  Both succeed:     {both_ok}")
    print(f"  Bare only:        {bare_only}")
    print(f"  Heuristic only:   {heur_only}  ← args_necessary confirmed cases")
    print(f"  Neither:          {neither}")

    calls_bare = sum(r.cosine5_bare_calls for r in bare_results)
    calls_heur = sum(o.get("cosine_top5_calls", 0) for o in orig_rows)
    print(f"\nLean call budget:")
    print(f"  Heuristic+args (EXP-RW-032): {calls_heur} total, {calls_heur/n:.1f}/example")
    print(f"  Bare only (this run):        {calls_bare} total, {calls_bare/n:.1f}/example")

    elapsed = [r.elapsed_s for r in bare_results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Worker / parallel
# ---------------------------------------------------------------------------

def _worker_cmd(args: argparse.Namespace, shard_index: int, num_shards: int, shard_out: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.rw2_bare_ablation",
        "--worker",
        "--shard-index", str(shard_index),
        "--num-shards", str(num_shards),
        "--output", shard_out,
        "--orig", args.orig,
        "--data", args.data,
        "--db", args.db,
        "--lean-project", args.lean_project,
        "--cosine-topk", str(args.cosine_topk),
        "--max-scope-premises", str(args.max_scope_premises),
        "--restart-every", str(args.restart_every),
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    return cmd


def run_parallel(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    shard_dir = out_path.parent / f"{out_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    procs, shard_paths = [], []
    for i in range(args.parallel):
        sp = shard_dir / f"shard_{i}.jsonl"
        shard_paths.append(sp)
        cmd = _worker_cmd(args, i, args.parallel, str(sp))
        logger.info("Starting shard %d/%d", i + 1, args.parallel)
        procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))

    rc = 0
    for i, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            logger.error("Shard %d failed: exit %d", i, code)
            rc = code
    if rc != 0:
        raise SystemExit(rc)

    merged: list[dict] = []
    for sp in shard_paths:
        if sp.exists():
            with open(sp) as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))
    merged.sort(key=lambda d: (d.get("theorem_full_name", ""), d.get("step_index", 0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    # Load originals for report
    orig_map = {}
    with open(args.orig) as f:
        for line in f:
            row = json.loads(line)
            if row.get("goal_started"):
                orig_map[(row["theorem_full_name"], row["step_index"])] = row

    bare_results = [BareResult(**{k: v for k, v in row.items() if k in BareResult.__dataclass_fields__})
                    for row in merged]
    orig_rows = [orig_map[(b.theorem_full_name, b.step_index)] for b in bare_results
                 if (b.theorem_full_name, b.step_index) in orig_map]
    assert len(orig_rows) == len(bare_results), f"Mismatch: {len(orig_rows)} orig vs {len(bare_results)} bare"
    print_report(orig_rows, bare_results)


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load original results — only started examples
    orig_map: dict[tuple, dict] = {}
    with open(args.orig) as f:
        for line in f:
            row = json.loads(line)
            if row.get("goal_started"):
                key = (row["theorem_full_name"], row["step_index"])
                orig_map[key] = row
    orig_keys = sorted(orig_map.keys())

    # Load source examples for prefix_tactics / prefix_goal_states
    source_map: dict[tuple, dict] = {}
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line)
            key = (ex.get("theorem_full_name", ""), ex.get("step_index", 0))
            source_map[key] = ex

    # Shard
    if args.num_shards > 1:
        orig_keys = [k for i, k in enumerate(orig_keys) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        orig_keys = orig_keys[:args.limit]
    logger.info("Worker shard %d/%d: %d started examples", args.shard_index + 1, args.num_shards, len(orig_keys))

    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not available")

    kernel = LeanKernel(LeanConfig(
        backend="pantograph", timeout=120,
        project_root=args.lean_project, imports=["Mathlib"],
    ))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_since_restart = 0
    bare_list: list[BareResult] = []
    with open(out_path, "w") as out_f:
        for i, key in enumerate(orig_keys):
            n_since_restart += 1
            if n_since_restart >= args.restart_every:
                logger.info("Periodic server restart")
                kernel.gc()
                kernel._restart_server()
                n_since_restart = 0
            if i == 0 or (i + 1) % 5 == 0:
                logger.info("Shard %d: %d/%d", args.shard_index + 1, i + 1, len(orig_keys))

            orig = orig_map[key]
            example = source_map.get(key, orig)
            try:
                bare = run_bare_example(
                    orig=orig,
                    example=example,
                    kernel=kernel,
                    conn=conn,
                    id_to_name=id_to_name,
                    name_to_id=name_to_id,
                    encoder=encoder,
                    project_root=args.lean_project,
                    cosine_topk=args.cosine_topk,
                    max_scope_premises=args.max_scope_premises,
                )
            except Exception as e:
                logger.error("Error on %s: %s", key, e)
                bare = BareResult(
                    theorem_full_name=orig["theorem_full_name"],
                    step_index=orig["step_index"],
                    grammar_tier=orig.get("grammar_tier", ""),
                )
            bare_list.append(bare)
            out_f.write(json.dumps(asdict(bare)) + "\n")
            out_f.flush()

    kernel.close()
    conn.close()

    # Local shard report
    orig_rows = [orig_map[k] for k in orig_keys if k in orig_map]
    assert len(orig_rows) == len(bare_list)
    print_report(orig_rows, bare_list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="rw2 bare-premise ablation (second pass)")
    parser.add_argument("--orig", default="runs/rw2_step0_results.jsonl", help="Original EXP-RW-032 results")
    parser.add_argument("--data", default="data/canonical/canonical_rw_eval.jsonl", help="Source JSONL for prefix_tactics")
    parser.add_argument("--db", default="data/proof_network.db", help="Proof network DB")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/rw2_bare_ablation.jsonl")
    parser.add_argument("--cosine-topk", type=int, default=5)
    parser.add_argument("--max-scope-premises", type=int, default=30)
    parser.add_argument("--restart-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.parallel > 1 and not args.worker:
        run_parallel(args)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
