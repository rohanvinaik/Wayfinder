"""rw3 sequential composition benchmark (EXP-RW-035).

Scope: all-bare rw3, step_index == 0 (487 examples).

Three conditions:
  oracle_seq     — replay exact atoms in order (direction preserved)
  cosine_seq     — after each accepted step: refresh goal state, refresh scope,
                   rerank, try cosine top-5 (both dirs); stop on first failure
  oracle1_cosine_rest — oracle first atom, then cosine_seq for remaining atoms

Metrics reported:
  FullSeqAccept       — all atoms fired
  Accept@k (k=1,2,3) — first k atoms all fired
  mean divergence step — mean step index at which execution first fails
  Lean calls/example
  Failure taxonomy by step index (rewrite_pattern_not_found, identifier_scope, etc.)
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult
from src.proof_network import get_accessible_premises
from src.rw_scoper import scope_for_rw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atom utilities (shared with run_rw3_benchmark)
# ---------------------------------------------------------------------------


def split_atoms(ir: str) -> list[str]:
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
    return not re.search(r"\s", atom.lstrip("← ").strip())


def atom_premise_name(atom: str) -> str:
    return atom.lstrip("← ").strip()


def atom_direction(atom: str) -> str:
    return "backward" if atom.lstrip().startswith("←") else "forward"


def build_rw_tactic(premise: str, direction: str) -> str:
    return f"rw [{'← ' if direction == 'backward' else ''}{premise}]"


def is_all_bare_rw3(ir: str) -> bool:
    atoms = split_atoms(ir)
    return len(atoms) >= 2 and all(atom_is_bare(a) for a in atoms)


def _categorize_error(err: str) -> str:
    if not err:
        return ""
    if "Unknown identifier" in err:
        return "identifier_scope"
    if "Did not find" in err or "did not find" in err.lower():
        return "rewrite_pattern_not_found"
    if "type mismatch" in err:
        return "type_mismatch"
    if "parse" in err.lower():
        return "parse_error"
    return "other_tactic_fail"


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
            if not is_all_bare_rw3(ex.get("canonical_action_ir", "")):
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
    return [id_to_name[pid] for pid in get_accessible_premises(conn, tid) if pid in id_to_name]


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


def resolve_atom_premise(short: str, scope_premises: list[str]) -> str:
    if short in scope_premises:
        return short
    matches = [p for p in scope_premises if p.rsplit(".", 1)[-1] == short]
    return matches[0] if len(matches) == 1 else short


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
            ng = result.new_goals[0]
            new_goal = ng if isinstance(ng, str) else str(ng)
        return result.success, new_goal, result.error_message, False
    except Exception as e:
        msg = str(e)
        is_crash = any(kw in msg.lower() for kw in ("broken pipe", "connection reset", "process"))
        return False, "", msg, is_crash


# ---------------------------------------------------------------------------
# Sequential step execution
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step_idx: int = 0
    tactic: str = ""
    success: bool = False
    error: str = ""
    error_category: str = ""
    lean_calls: int = 0


@dataclass
class SeqResult:
    """Per-condition sequential execution result."""
    condition: str = ""           # oracle_seq | cosine_seq | oracle1_cosine_rest
    n_atoms: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    full_seq_accept: bool = False
    divergence_step: int = -1     # step index of first failure (-1 if fully succeeded)
    total_lean_calls: int = 0


def _run_oracle_seq(
    atoms: list[str],
    start_goal: str,
    start_goal_id: int,
    kernel: LeanKernel,
    scope_premises: list[str],
) -> SeqResult:
    """Replay exact gold atoms in order."""
    res = SeqResult(condition="oracle_seq", n_atoms=len(atoms))
    goal = start_goal
    goal_id = start_goal_id

    for i, atom in enumerate(atoms):
        short = atom_premise_name(atom)
        direction = atom_direction(atom)
        full_name = resolve_atom_premise(short, scope_premises)
        tac = build_rw_tactic(full_name, direction)

        ok, new_goal, err, crashed = try_tactic_safe(kernel, goal, tac, goal_id=goal_id)
        sr = StepRecord(step_idx=i, tactic=tac, success=ok, error=err,
                        error_category=_categorize_error(err) if not ok else "", lean_calls=1)
        res.steps.append(sr)
        res.total_lean_calls += 1

        if crashed:
            res.divergence_step = i
            break
        if not ok:
            res.divergence_step = i
            break
        # Advance: the next step operates on the new goal state
        # Pantograph returns new goal state; use it as the next start
        if new_goal:
            goal = new_goal
            goal_id = 0  # new goal state has id 0
        else:
            # No remaining goals — sequence is complete (rare)
            break

    if res.divergence_step == -1 and all(s.success for s in res.steps):
        res.full_seq_accept = True

    return res


def _run_cosine_seq(
    atoms: list[str],
    start_goal: str,
    start_goal_id: int,
    kernel: LeanKernel,
    premise_names: list[str],
    encoder,
    max_scope_premises: int,
    cosine_topk: int,
    first_atom_tactic: str = "",   # if set, use this for step 0 instead of cosine
) -> SeqResult:
    """At each step: refresh scope, rerank, try cosine top-k (both dirs)."""
    condition = "oracle1_cosine_rest" if first_atom_tactic else "cosine_seq"
    res = SeqResult(condition=condition, n_atoms=len(atoms))
    goal = start_goal
    goal_id = start_goal_id

    for i, _atom in enumerate(atoms):
        # Refresh scope for current goal state
        scope = scope_for_rw(goal, premise_names, max_premises=max_scope_premises)
        ranked = cosine_rank(goal, scope.premises, encoder)

        lean_calls = 0
        step_ok = False
        step_tac = ""
        step_err = ""   # set on failure; pre-initialized so it's always bound
        new_goal = ""
        err = ""

        if i == 0 and first_atom_tactic:
            # Use provided first-atom tactic (oracle1_cosine_rest)
            ok, ng, err, crashed = try_tactic_safe(kernel, goal, first_atom_tactic, goal_id=goal_id)
            lean_calls += 1
            if crashed or not ok:
                sr = StepRecord(step_idx=i, tactic=first_atom_tactic, success=False,
                                error=err, error_category=_categorize_error(err), lean_calls=lean_calls)
                res.steps.append(sr)
                res.total_lean_calls += lean_calls
                res.divergence_step = i
                break
            step_ok = True
            step_tac = first_atom_tactic
            new_goal = ng
        else:
            # Cosine beam
            top_k = min(cosine_topk, len(ranked))
            for _, premise in ranked[:top_k]:
                for direction in ("forward", "backward"):
                    tac = build_rw_tactic(premise, direction)
                    ok, ng, err, crashed = try_tactic_safe(kernel, goal, tac, goal_id=goal_id)
                    lean_calls += 1
                    if crashed:
                        sr = StepRecord(step_idx=i, tactic=tac, success=False,
                                        error=err, error_category="server_crash", lean_calls=lean_calls)
                        res.steps.append(sr)
                        res.total_lean_calls += lean_calls
                        res.divergence_step = i
                        return res
                    if ok:
                        step_ok = True
                        step_tac = tac
                        step_err = ""
                        new_goal = ng
                        break
                if step_ok:
                    break
            if not step_ok:
                step_err = err

        sr = StepRecord(step_idx=i, tactic=step_tac, success=step_ok,
                        error=step_err, error_category=_categorize_error(step_err) if not step_ok else "",
                        lean_calls=lean_calls)
        res.steps.append(sr)
        res.total_lean_calls += lean_calls

        if not step_ok:
            res.divergence_step = i
            break

        if new_goal:
            goal = new_goal
            goal_id = 0
        else:
            break

    if res.divergence_step == -1 and all(s.success for s in res.steps):
        res.full_seq_accept = True

    return res


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Rw3SeqResult:
    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    n_atoms: int = 0
    canonical_action_ir: str = ""

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    n_accessible_premises: int = 0
    n_scope_premises_initial: int = 0

    # oracle_seq
    oracle_full: bool = False
    oracle_divergence: int = -1
    oracle_lean_calls: int = 0
    oracle_step_errors: list = field(default_factory=list)  # list of (step_idx, category)

    # cosine_seq
    cosine_full: bool = False
    cosine_divergence: int = -1
    cosine_lean_calls: int = 0
    cosine_step_errors: list = field(default_factory=list)

    # oracle1_cosine_rest
    oc_rest_full: bool = False
    oc_rest_divergence: int = -1
    oc_rest_lean_calls: int = 0
    oc_rest_step_errors: list = field(default_factory=list)

    elapsed_s: float = 0.0


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
) -> Rw3SeqResult:
    t0 = time.time()
    ir = example.get("canonical_action_ir", "")
    atoms = split_atoms(ir)

    res = Rw3SeqResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example["file_path"],
        step_index=example["step_index"],
        n_atoms=len(atoms),
        canonical_action_ir=ir,
    )

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

    premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
    res.n_accessible_premises = len(premise_names)

    # Initial scope for oracle first-atom tactic (oracle1_cosine_rest)
    init_scope = scope_for_rw(goal_str, premise_names, max_premises=max_scope_premises)
    res.n_scope_premises_initial = len(init_scope.premises)

    # Build oracle first-atom tactic for oracle1_cosine_rest condition
    first_atom = atoms[0] if atoms else ""
    first_short = atom_premise_name(first_atom)
    first_dir = atom_direction(first_atom)
    first_full = resolve_atom_premise(first_short, init_scope.premises)
    ann = example.get("annotated_premise", "")
    if ann and ann in init_scope.premises:
        first_full = ann
    oracle_first_tac = build_rw_tactic(first_full, first_dir)

    # --- oracle_seq ---
    oracle_r = _run_oracle_seq(atoms, goal_str, goal_id, kernel, init_scope.premises)
    res.oracle_full = oracle_r.full_seq_accept
    res.oracle_divergence = oracle_r.divergence_step
    res.oracle_lean_calls = oracle_r.total_lean_calls
    res.oracle_step_errors = [(s.step_idx, s.error_category) for s in oracle_r.steps if not s.success]

    # --- cosine_seq ---
    cosine_r = _run_cosine_seq(
        atoms, goal_str, goal_id, kernel, premise_names, encoder,
        max_scope_premises, cosine_topk,
    )
    res.cosine_full = cosine_r.full_seq_accept
    res.cosine_divergence = cosine_r.divergence_step
    res.cosine_lean_calls = cosine_r.total_lean_calls
    res.cosine_step_errors = [(s.step_idx, s.error_category) for s in cosine_r.steps if not s.success]

    # --- oracle1_cosine_rest ---
    oc_r = _run_cosine_seq(
        atoms, goal_str, goal_id, kernel, premise_names, encoder,
        max_scope_premises, cosine_topk,
        first_atom_tactic=oracle_first_tac,
    )
    res.oc_rest_full = oc_r.full_seq_accept
    res.oc_rest_divergence = oc_r.divergence_step
    res.oc_rest_lean_calls = oc_r.total_lean_calls
    res.oc_rest_step_errors = [(s.step_idx, s.error_category) for s in oc_r.steps if not s.success]

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[Rw3SeqResult], total: int) -> None:
    started = [r for r in results if r.goal_started]
    n = len(started)
    fail_cats = Counter(r.failure_category or "unknown" for r in results if not r.goal_started)

    oracle_full = sum(r.oracle_full for r in started)
    cosine_full = sum(r.cosine_full for r in started)
    oc_full = sum(r.oc_rest_full for r in started)

    print("\n" + "=" * 65)
    print("rw3 Sequential Composition Benchmark (EXP-RW-035)")
    print("=" * 65)
    print(f"\nExamples (all-bare rw3, step-0): {total}")
    print(f"GoalStart@rw3_bare: {n}/{total} ({100*n/max(total,1):.1f}%)")
    if fail_cats:
        for cat, cnt in fail_cats.most_common():
            print(f"  {cat}: {cnt}")
    if n == 0:
        return

    print(f"\n{'Condition':<40} {'FullSeq':>10} {'Rate':>7}")
    print("-" * 60)
    print(f"{'Oracle step-by-step':<40} {oracle_full:>10} {100*oracle_full/n:>6.1f}%")
    print(f"{'Cosine-5 sequential':<40} {cosine_full:>10} {100*cosine_full/n:>6.1f}%")
    print(f"{'Oracle-1 + cosine-rest':<40} {oc_full:>10} {100*oc_full/n:>6.1f}%")

    print(f"\nAccept@k (fraction where first k steps all succeeded):")
    print(f"{'k':<6} {'Oracle':>10} {'Cosine-5':>10} {'Oracle1+CRest':>14}")
    for k in range(1, 6):
        o_k = sum(1 for r in started if r.oracle_divergence == -1 or r.oracle_divergence >= k)
        c_k = sum(1 for r in started if r.cosine_divergence == -1 or r.cosine_divergence >= k)
        oc_k = sum(1 for r in started if r.oc_rest_divergence == -1 or r.oc_rest_divergence >= k)
        print(f"  k={k}  {o_k:>6}/{n} {100*o_k/n:>5.1f}%   {c_k:>6}/{n} {100*c_k/n:>5.1f}%   {oc_k:>6}/{n} {100*oc_k/n:>5.1f}%")

    # Mean divergence step
    oracle_divs = [r.oracle_divergence for r in started if r.oracle_divergence >= 0]
    cosine_divs = [r.cosine_divergence for r in started if r.cosine_divergence >= 0]
    oc_divs = [r.oc_rest_divergence for r in started if r.oc_rest_divergence >= 0]
    print(f"\nMean divergence step (among failures):")
    if oracle_divs:
        print(f"  Oracle:           {np.mean(oracle_divs):.2f} (median {np.median(oracle_divs):.0f})")
    if cosine_divs:
        print(f"  Cosine-5:         {np.mean(cosine_divs):.2f} (median {np.median(cosine_divs):.0f})")
    if oc_divs:
        print(f"  Oracle1+CRest:    {np.mean(oc_divs):.2f} (median {np.median(oc_divs):.0f})")

    # Failure taxonomy by step index (oracle)
    oracle_err_by_step: dict[int, Counter] = defaultdict(Counter)
    for r in started:
        for step_idx, cat in r.oracle_step_errors:
            oracle_err_by_step[step_idx][cat] += 1
    if oracle_err_by_step:
        print(f"\nOracle failure taxonomy by step index:")
        for step in sorted(oracle_err_by_step)[:6]:
            cats = oracle_err_by_step[step].most_common()
            cat_str = ", ".join(f"{c}:{v}" for c, v in cats)
            print(f"  step {step}: {cat_str}")

    # Lean call budget
    o_calls = sum(r.oracle_lean_calls for r in started)
    c_calls = sum(r.cosine_lean_calls for r in started)
    oc_calls = sum(r.oc_rest_lean_calls for r in started)
    print(f"\nLean call budget (per started example):")
    print(f"  Oracle seq:        {o_calls/n:.1f}")
    print(f"  Cosine-5 seq:      {c_calls/n:.1f}")
    print(f"  Oracle1+CRest:     {oc_calls/n:.1f}")

    # Per atom-count breakdown
    atom_groups = sorted(set(r.n_atoms for r in started))
    print(f"\nFullSeqAccept by atom count:")
    print(f"  {'atoms':<8} {'N':>5} {'Oracle':>10} {'Cosine5':>10} {'OC-rest':>10}")
    for k in atom_groups:
        grp = [r for r in started if r.n_atoms == k]
        o_k = sum(r.oracle_full for r in grp)
        c_k = sum(r.cosine_full for r in grp)
        oc_k = sum(r.oc_rest_full for r in grp)
        print(f"  {k:<8} {len(grp):>5} {o_k:>6}/{len(grp)} {100*o_k/len(grp):>4.0f}%  "
              f"{c_k:>6}/{len(grp)} {100*c_k/len(grp):>4.0f}%  "
              f"{oc_k:>6}/{len(grp)} {100*oc_k/len(grp):>4.0f}%")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Worker / parallel
# ---------------------------------------------------------------------------


def _worker_cmd(args: argparse.Namespace, shard_index: int, num_shards: int, shard_out: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.run_rw3_sequential",
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

    results = [Rw3SeqResult(**{k: v for k, v in row.items() if k in Rw3SeqResult.__dataclass_fields__})
               for row in merged]
    print_report(results, len(results))


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = load_rw3_bare_examples(args.data, step0_only=not args.include_step_gt0, limit=args.limit)
    if args.num_shards > 1:
        examples = [ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_index]
    logger.info("Worker shard %d/%d: %d examples", args.shard_index + 1, args.num_shards, len(examples))

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

    results: list[Rw3SeqResult] = []
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
                res = Rw3SeqResult(
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
    parser = argparse.ArgumentParser(description="rw3 sequential composition benchmark (EXP-RW-035)")
    parser.add_argument("--data", default="data/canonical/canonical_rw_eval.jsonl")
    parser.add_argument("--db", default="data/proof_network.db")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/rw3_sequential_results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-step-gt0", action="store_true")
    parser.add_argument("--restart-every", type=int, default=30)
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
