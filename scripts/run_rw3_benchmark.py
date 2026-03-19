"""rw3 first-atom benchmark (EXP-RW-034a + 034b).

Scope: all-bare rw3, step_index == 0 (487 examples).
"All-bare" = every atom in rw [a, b, c, ...] has no positional args.

Conditions (EXP-RW-034a):
  oracle_first    — exact first RewriteAtom from canonical_action_ir (direction preserved)
  cosine_top1     — cosine-ranked first premise, both directions tried
  cosine_top5     — top-5 cosine, both directions

Post-first-step metric (EXP-RW-034b):
  After each successful first-atom tactic, check whether the second gold atom's
  premise is still accessible in the resulting goal state (in-scope check only,
  no Lean call). This is the first composition metric: does a correct first step
  preserve second-step viability?

Parallel mode: --parallel N shards across N worker subprocesses.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult
from src.proof_network import get_accessible_premises
from src.rw_scoper import scope_for_rw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom splitting / classification
# ---------------------------------------------------------------------------


def split_atoms(ir: str) -> list[str]:
    """Split rw[a, b, c] into individual atom strings (preserves ← prefix)."""
    m = re.search(r"\[(.+)\]", ir, re.DOTALL)
    if not m:
        return []
    content = m.group(1)
    depth, atoms, current = 0, [], []
    for ch in content:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            atoms.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        atoms.append("".join(current).strip())
    return atoms


def atom_is_bare(atom: str) -> bool:
    """True if atom has no positional args (bare name, possibly with ←)."""
    clean = atom.lstrip("← ").strip()
    return not re.search(r"\s", clean)


def atom_premise_name(atom: str) -> str:
    """Extract the premise name from a bare atom string."""
    return atom.lstrip("← ").strip()


def atom_direction(atom: str) -> str:
    """'backward' if atom starts with ←, else 'forward'."""
    return "backward" if atom.lstrip().startswith("←") else "forward"


def build_rw_tactic(premise: str, direction: str) -> str:
    arrow = "← " if direction == "backward" else ""
    return f"rw [{arrow}{premise}]"


def is_all_bare_rw3(ir: str) -> bool:
    """True if ir is a multi-atom rw with every atom bare."""
    atoms = split_atoms(ir)
    if len(atoms) < 2:
        return False
    return all(atom_is_bare(a) for a in atoms)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rw3_bare_examples(path: str, step0_only: bool = True, limit: int = 0) -> list[dict]:
    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ir = ex.get("canonical_action_ir", "")
            if not is_all_bare_rw3(ir):
                continue
            if step0_only and ex.get("step_index", 0) != 0:
                continue
            examples.append(ex)
    examples.sort(key=lambda x: (x.get("file_path", ""), x.get("theorem_full_name", ""), x.get("step_index", 0)))
    if limit > 0:
        examples = examples[:limit]
    return examples


# ---------------------------------------------------------------------------
# DB / cosine utilities
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


def cosine_rank(goal_text: str, symbols: list[str], encoder) -> list[tuple[float, str]]:
    if not symbols or encoder is None:
        return [(0.0, s) for s in symbols]
    try:
        g_emb = encoder.encode([goal_text], normalize_embeddings=True)
        s_embs = encoder.encode(symbols, normalize_embeddings=True)
        scores = (g_emb @ s_embs.T).flatten()
        return sorted(zip(scores.tolist(), symbols), reverse=True)
    except Exception as e:
        logger.warning("Cosine encoding failed: %s", e)
        return [(0.0, s) for s in symbols]


def resolve_first_atom_premise(ir: str, scope_premises: list[str]) -> str:
    """Resolve the first atom's premise name against the accessible scope."""
    atoms = split_atoms(ir)
    if not atoms:
        return ""
    short = atom_premise_name(atoms[0])
    if short in scope_premises:
        return short
    # Try suffix match
    suffix_matches = [p for p in scope_premises if p.rsplit(".", 1)[-1] == short]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    # Annotated_premise fallback is handled by caller
    return short


def resolve_atom_premise(atom_short: str, scope_premises: list[str]) -> str:
    """Resolve a bare atom name against the accessible scope."""
    if atom_short in scope_premises:
        return atom_short
    suffix_matches = [p for p in scope_premises if p.rsplit(".", 1)[-1] == atom_short]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return atom_short


# ---------------------------------------------------------------------------
# Tactic execution
# ---------------------------------------------------------------------------


def try_tactic_safe(
    kernel: LeanKernel, goal_state: str, tactic: str, goal_id: int = 0
) -> tuple[bool, str, str, bool]:
    """Returns (success, new_goal_state, error_msg, was_crash)."""
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        new_goal = ""
        if result.success and result.new_goals:
            new_goal = result.new_goals[0] if isinstance(result.new_goals[0], str) else str(result.new_goals[0])
        return result.success, new_goal, result.error_message, False
    except Exception as e:
        msg = str(e)
        is_crash = any(kw in msg.lower() for kw in ("broken pipe", "connection reset", "process", "crash"))
        return False, "", msg, is_crash


# ---------------------------------------------------------------------------
# Second-atom viability check (034b, no Lean call)
# ---------------------------------------------------------------------------


def second_atom_in_scope(
    post_first_goal: str,
    second_atom_short: str,
    premise_names: list[str],
    max_scope_premises: int,
) -> bool:
    """After first-atom success, check if second gold atom's premise is in scope."""
    if not post_first_goal or not second_atom_short:
        return False
    scope = scope_for_rw(post_first_goal, premise_names, max_premises=max_scope_premises)
    resolved = resolve_atom_premise(second_atom_short, scope.all_symbols)
    return resolved in scope.all_symbols


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Rw3FirstAtomResult:
    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    n_atoms: int = 0
    canonical_action_ir: str = ""
    annotated_premise: str = ""

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    n_accessible_premises: int = 0
    n_scope_premises: int = 0
    gold_first_premise: str = ""
    gold_first_direction: str = ""
    gold_first_in_scope: bool = False
    gold_first_rank: int = -1

    oracle_first_success: bool = False
    oracle_first_tactic: str = ""
    oracle_first_error: str = ""
    oracle_first_new_goal: str = ""

    cosine_top1_success: bool = False
    cosine_top1_tactic: str = ""
    cosine_top1_rank: int = -1
    cosine_top1_calls: int = 0

    cosine_top5_success: bool = False
    cosine_top5_tactic: str = ""
    cosine_top5_rank: int = -1
    cosine_top5_calls: int = 0

    # 034b: second-atom viability after oracle first step
    has_second_atom: bool = False
    gold_second_premise: str = ""
    second_in_scope_after_oracle: bool = False
    second_in_scope_after_cosine5: bool = False

    oracle_first_error_category: str = ""
    elapsed_s: float = 0.0


def _categorize_error(err: str) -> str:
    if not err:
        return ""
    if "Unknown identifier" in err:
        return "identifier_scope"
    if "Did not find an occurrence" in err or "did not find" in err.lower():
        return "rewrite_pattern_not_found"
    if "type mismatch" in err:
        return "type_mismatch"
    if "parse" in err.lower():
        return "parse_error"
    return "other_tactic_fail"


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------


def run_one_example(
    example: dict,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder,
    project_root: str,
    max_scope_premises: int,
    cosine_topk: int,
) -> Rw3FirstAtomResult:
    t0 = time.time()
    ir = example.get("canonical_action_ir", "")
    atoms = split_atoms(ir)

    res = Rw3FirstAtomResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example["file_path"],
        step_index=example["step_index"],
        n_atoms=len(atoms),
        canonical_action_ir=ir,
        annotated_premise=example.get("annotated_premise", ""),
    )

    # Parse first and second atom gold info
    first_atom = atoms[0] if atoms else ""
    gold_first_short = atom_premise_name(first_atom)
    gold_first_dir = atom_direction(first_atom)
    second_atom = atoms[1] if len(atoms) > 1 else ""
    gold_second_short = atom_premise_name(second_atom) if second_atom else ""
    res.has_second_atom = bool(second_atom)
    res.gold_second_premise = gold_second_short
    res.gold_first_direction = gold_first_dir

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
    if not replay.success:
        res.elapsed_s = time.time() - t0
        return res

    goal_str = replay.goal_state
    goal_id = replay.goal_id

    # Scope
    premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
    res.n_accessible_premises = len(premise_names)
    scope = scope_for_rw(goal_str, premise_names, max_premises=max_scope_premises)
    res.n_scope_premises = len(scope.premises)
    ranked = cosine_rank(goal_str, scope.premises, encoder)

    # Resolve first atom premise against scope
    gold_first_full = resolve_atom_premise(gold_first_short, scope.premises)
    # Fall back to annotated_premise if available and in scope
    ann = example.get("annotated_premise", "")
    if ann and ann in scope.premises:
        gold_first_full = ann
    res.gold_first_premise = gold_first_full
    res.gold_first_in_scope = gold_first_full in scope.premises

    # Gold rank
    for i, (_, prem) in enumerate(ranked):
        if prem == gold_first_full:
            res.gold_first_rank = i
            break

    # --- Oracle first atom (exact: direction preserved) ---
    oracle_tac = build_rw_tactic(gold_first_full, gold_first_dir)
    res.oracle_first_tactic = oracle_tac
    ok, new_goal, err, crashed = try_tactic_safe(kernel, goal_str, oracle_tac, goal_id=goal_id)
    if crashed:
        res.crash_retries += 1
        res.oracle_first_error = err
        res.elapsed_s = time.time() - t0
        return res
    res.oracle_first_success = ok
    res.oracle_first_error = err
    res.oracle_first_new_goal = new_goal
    res.oracle_first_error_category = _categorize_error(err) if not ok else ""

    # 034b: second-atom viability after oracle first step
    if ok and second_atom and new_goal:
        res.second_in_scope_after_oracle = second_atom_in_scope(
            new_goal, gold_second_short, premise_names, max_scope_premises
        )

    # --- Cosine top-1 and top-5 ---
    top_k = min(cosine_topk, len(ranked))
    top1_done = False
    top5_done = False
    cosine5_new_goal = ""

    for rank, (_, premise) in enumerate(ranked[:top_k]):
        for direction in ("forward", "backward"):
            tac = build_rw_tactic(premise, direction)
            ok_c, new_g_c, _, crashed_c = try_tactic_safe(kernel, goal_str, tac, goal_id=goal_id)
            res.cosine_top5_calls += 1
            if rank == 0:
                res.cosine_top1_calls += 1
            if crashed_c:
                res.crash_retries += 1
                res.elapsed_s = time.time() - t0
                return res
            if ok_c:
                if rank == 0 and not top1_done:
                    res.cosine_top1_success = True
                    res.cosine_top1_tactic = tac
                    res.cosine_top1_rank = rank
                    top1_done = True
                if not top5_done:
                    res.cosine_top5_success = True
                    res.cosine_top5_tactic = tac
                    res.cosine_top5_rank = rank
                    cosine5_new_goal = new_g_c
                    top5_done = True
                break
        if top5_done:
            break

    # 034b: second-atom viability after cosine-5 first step
    if top5_done and second_atom and cosine5_new_goal:
        res.second_in_scope_after_cosine5 = second_atom_in_scope(
            cosine5_new_goal, gold_second_short, premise_names, max_scope_premises
        )

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[Rw3FirstAtomResult], total: int) -> None:
    started = [r for r in results if r.goal_started]
    n = len(started)
    fail_cats = Counter(r.failure_category or "unknown" for r in results if not r.goal_started)
    gold_scope = [r for r in started if r.gold_first_in_scope]
    m = len(gold_scope)

    oracle_ok = sum(r.oracle_first_success for r in started)
    c1_ok = sum(r.cosine_top1_success for r in started)
    c5_ok = sum(r.cosine_top5_success for r in started)
    oracle_gs = sum(r.oracle_first_success for r in gold_scope)
    c5_gs = sum(r.cosine_top5_success for r in gold_scope)

    print("\n" + "=" * 60)
    print("rw3 First-Atom Benchmark (EXP-RW-034a + 034b)")
    print("=" * 60)
    print(f"\nExamples (all-bare rw3, step-0): {total}")
    print(f"GoalStart@rw3_bare: {n}/{total} ({100*n/max(total,1):.1f}%)")
    if fail_cats:
        for cat, cnt in fail_cats.most_common():
            print(f"  {cat}: {cnt}")
    if n == 0:
        print("\nNo started goals.")
        return

    print(f"\n{'--- EXP-RW-034a: First-Atom Acceptance ---'}")
    print(f"\n{'Condition':<40} {'|started':>10} {'Rate':>7} {'|gold_in_scope':>16} {'Rate':>7}")
    print("-" * 82)
    def row(label, ok_s, ok_gs):
        r1 = f"{ok_s}/{n}"
        r2 = f"{100*ok_s/max(n,1):.1f}%"
        r3 = f"{ok_gs}/{m}"
        r4 = f"{100*ok_gs/max(m,1):.1f}%"
        print(f"{label:<40} {r1:>10} {r2:>7} {r3:>16} {r4:>7}")

    row("Oracle first atom (exact dir)", oracle_ok, oracle_gs)
    row("Cosine top-1 (both dirs)", c1_ok, sum(r.cosine_top1_success for r in gold_scope))
    row("Cosine top-5 (both dirs)", c5_ok, c5_gs)

    print(f"\nGold first atom in scope: {len(gold_scope)}/{n} ({100*len(gold_scope)/max(n,1):.1f}%)")
    ranks = [r.gold_first_rank for r in started if r.gold_first_rank >= 0]
    if ranks:
        print(f"Gold rank (when in scope): mean={np.mean(ranks):.1f}, median={np.median(ranks):.1f}")
        print(f"  rank-0: {sum(1 for r in ranks if r == 0)}, rank<=2: {sum(1 for r in ranks if r <= 2)}")
    scope_sizes = [r.n_scope_premises for r in started if r.n_scope_premises > 0]
    if scope_sizes:
        print(f"Mean scope size: {np.mean(scope_sizes):.1f}")

    # Error taxonomy for oracle failures
    err_cats = Counter(r.oracle_first_error_category for r in started if not r.oracle_first_success and r.oracle_first_error_category)
    if err_cats:
        print(f"\nOracle first-atom failure taxonomy:")
        for cat, cnt in err_cats.most_common():
            print(f"  {cat}: {cnt}")

    # EXP-RW-034b: second-atom viability
    print(f"\n{'--- EXP-RW-034b: Second-Atom Viability (composition metric) ---'}")
    has_second = [r for r in started if r.has_second_atom]
    if has_second:
        oracle_fired_second = [r for r in has_second if r.oracle_first_success]
        c5_fired_second = [r for r in has_second if r.cosine_top5_success]

        if oracle_fired_second:
            viab_oracle = sum(r.second_in_scope_after_oracle for r in oracle_fired_second)
            print(f"After oracle first-step ({len(oracle_fired_second)} fired):")
            print(f"  Second gold atom in scope: {viab_oracle}/{len(oracle_fired_second)} ({100*viab_oracle/max(len(oracle_fired_second),1):.1f}%)")

        if c5_fired_second:
            viab_c5 = sum(r.second_in_scope_after_cosine5 for r in c5_fired_second)
            print(f"After cosine-5 first-step ({len(c5_fired_second)} fired):")
            print(f"  Second gold atom in scope: {viab_c5}/{len(c5_fired_second)} ({100*viab_c5/max(len(c5_fired_second),1):.1f}%)")

        # Composition gap proxy
        # Cases where oracle fires but second atom NOT in scope = composition break
        comp_breaks = sum(1 for r in oracle_fired_second if not r.second_in_scope_after_oracle)
        print(f"\nComposition breaks (oracle fires, second atom lost): {comp_breaks}/{len(oracle_fired_second)}")
    else:
        print("  (No second-atom data)")

    # Atom count breakdown
    atom_counts = Counter(r.n_atoms for r in started)
    print(f"\nAtom count (started):")
    for k in sorted(atom_counts):
        v = atom_counts[k]
        oracle_k = sum(r.oracle_first_success for r in started if r.n_atoms == k)
        print(f"  {k} atoms: {v} examples, oracle={oracle_k}/{v} ({100*oracle_k/max(v,1):.0f}%)")

    # Lean call budget
    c1_calls = sum(r.cosine_top1_calls for r in started)
    c5_calls = sum(r.cosine_top5_calls for r in started)
    print(f"\nLean call budget: cosine-1={c1_calls/max(n,1):.1f}/ex, cosine-5={c5_calls/max(n,1):.1f}/ex")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"Timing: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Worker / parallel
# ---------------------------------------------------------------------------


def _worker_cmd(args: argparse.Namespace, shard_index: int, num_shards: int, shard_out: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.run_rw3_benchmark",
        "--worker", "--shard-index", str(shard_index), "--num-shards", str(num_shards),
        "--output", shard_out,
        "--data", args.data, "--db", args.db,
        "--lean-project", args.lean_project,
        "--cosine-topk", str(args.cosine_topk),
        "--max-scope-premises", str(args.max_scope_premises),
        "--restart-every", str(args.restart_every),
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.include_step_gt0:
        cmd.append("--include-step-gt0")
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
    merged.sort(key=lambda d: (d.get("file_path", ""), d.get("theorem_full_name", ""), d.get("step_index", 0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    results = [Rw3FirstAtomResult(**{k: v for k, v in row.items() if k in Rw3FirstAtomResult.__dataclass_fields__})
               for row in merged]
    print_report(results, len(results))


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = load_rw3_bare_examples(args.data, step0_only=not args.include_step_gt0, limit=args.limit)
    if args.num_shards > 1:
        examples = [ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_index]
    logger.info("Worker shard %d/%d: %d all-bare rw3 examples", args.shard_index + 1, args.num_shards, len(examples))

    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded MiniLM encoder")
    except ImportError:
        logger.warning("sentence-transformers not available")

    kernel = LeanKernel(LeanConfig(
        backend="pantograph", timeout=120,
        project_root=args.lean_project, imports=["Mathlib"],
    ))
    logger.info("Initializing Pantograph with Mathlib")

    results: list[Rw3FirstAtomResult] = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_since_restart = 0
    with open(out_path, "w") as out_f:
        for i, ex in enumerate(examples):
            n_since_restart += 1
            if n_since_restart >= args.restart_every:
                logger.info("Periodic server restart")
                kernel.gc()
                kernel._restart_server()
                n_since_restart = 0
            if i == 0 or (i + 1) % 10 == 0:
                logger.info("Shard %d: %d/%d", args.shard_index + 1, i + 1, len(examples))

            try:
                res = run_one_example(
                    example=ex, kernel=kernel, conn=conn,
                    id_to_name=id_to_name, name_to_id=name_to_id,
                    encoder=encoder, project_root=args.lean_project,
                    max_scope_premises=args.max_scope_premises,
                    cosine_topk=args.cosine_topk,
                )
            except Exception as e:
                logger.error("Error on %s: %s", ex.get("theorem_full_name"), e)
                res = Rw3FirstAtomResult(
                    theorem_full_name=ex.get("theorem_full_name", ""),
                    file_path=ex.get("file_path", ""),
                    step_index=ex.get("step_index", 0),
                    failure_category="unhandled_error",
                )
            results.append(res)
            out_f.write(json.dumps(asdict(res)) + "\n")
            out_f.flush()

    kernel.close()
    conn.close()
    print_report(results, len(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="rw3 first-atom benchmark (EXP-RW-034a+b)")
    parser.add_argument("--data", default="data/canonical/canonical_rw_eval.jsonl")
    parser.add_argument("--db", default="data/proof_network.db")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/rw3_first_atom_results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-step-gt0", action="store_true")
    parser.add_argument("--restart-every", type=int, default=50)
    parser.add_argument("--max-scope-premises", type=int, default=30)
    parser.add_argument("--cosine-topk", type=int, default=5)
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
