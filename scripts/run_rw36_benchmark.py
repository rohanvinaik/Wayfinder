"""rw3-with-args benchmark (EXP-RW-036).

Scope: rw3-with-args = multi-atom rw where ≥1 atom has positional args.
       step_index == 0 only (297 examples from data profile).

Conditions:
  oracle_first_exact   — exact first atom from canonical_action_ir (direction + args preserved)
  oracle_seq_exact     — full sequential, exact atoms in order (direction + args preserved)
  cosine5_bare_first   — cosine top-5, bare only (no args), both directions
  cosine5_heur_first   — cosine top-5 + heuristic local-hyp args for first atom
  cosine_seq_bare      — sequential cosine bare (re-rank each step, both dirs, no args)

Partition (per started example, keyed on first-atom results):
  args_redundant       — cosine5_bare_first succeeds (args not needed)
  args_necessary       — oracle_first_exact succeeds but cosine5_bare_first fails
  unexecutable_oracle  — oracle_first_exact fails (lowering / elaboration gap)
  not_started          — goal creation failed

Primary metrics:
  FirstAtomAccept|started  per condition
  FullSeqAccept|started    for oracle_seq_exact and cosine_seq_bare
  Partition counts / fractions
  Accept@k (k=1,2,3) for sequential conditions
  Failure taxonomy by step index (oracle)

Parallel: --parallel 4 for 4 shards.
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
# Atom utilities
# ---------------------------------------------------------------------------


def split_atoms(ir: str) -> list[str]:
    """Split rw[a, b, c] into individual atom strings (preserves ← prefix and args)."""
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
    """Extract the premise name (first whitespace-delimited token after stripping ←)."""
    clean = atom.lstrip("← ").strip()
    return clean.split()[0] if clean else ""


def atom_direction(atom: str) -> str:
    return "backward" if atom.lstrip().startswith("←") else "forward"


def atom_args(atom: str) -> list[str]:
    """Extract positional args from an atom (everything after the premise name)."""
    clean = atom.lstrip("← ").strip()
    parts = clean.split()
    return parts[1:] if len(parts) > 1 else []


def build_rw_tactic(premise: str, direction: str, args: list[str] | None = None) -> str:
    arrow = "← " if direction == "backward" else ""
    if args:
        return f"rw [{arrow}{premise} {' '.join(args)}]"
    return f"rw [{arrow}{premise}]"


def atom_to_tactic(atom: str, resolved_premise: str | None = None) -> str:
    """Convert a canonical atom string to a Lean tactic, optionally swapping the premise."""
    direction = atom_direction(atom)
    args = atom_args(atom)
    premise = resolved_premise if resolved_premise is not None else atom_premise_name(atom)
    return build_rw_tactic(premise, direction, args if args else None)


def is_rw3_with_args(ir: str) -> bool:
    """True if ir is a multi-atom rw with at least one atom having positional args."""
    atoms = split_atoms(ir)
    if len(atoms) < 2:
        return False
    return any(not atom_is_bare(a) for a in atoms)


def extract_local_hyps(goal_str: str) -> list[str]:
    """Extract local hypothesis names from a Pantograph goal string.

    Pantograph formats goals as lines like:
      h : SomeType
      ⊢ conclusion
    Extract identifier names before ' : ' that appear before '⊢'.
    """
    hyps = []
    for line in goal_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("⊢"):
            break
        m = re.match(r"^(\w+)\s*:", stripped)
        if m:
            hyps.append(m.group(1))
    return hyps


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rw3_args_examples(path: str, step0_only: bool = True, limit: int = 0) -> list[dict]:
    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if not is_rw3_with_args(ex.get("canonical_action_ir", "")):
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
# Sequential step execution (for oracle_seq and cosine_seq_bare)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step_idx: int = 0
    tactic: str = ""
    success: bool = False
    error: str = ""
    error_category: str = ""
    lean_calls: int = 0
    atom_had_args: bool = False


@dataclass
class SeqResult:
    condition: str = ""
    n_atoms: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    full_seq_accept: bool = False
    divergence_step: int = -1
    total_lean_calls: int = 0


def _run_oracle_seq(
    atoms: list[str],
    start_goal: str,
    start_goal_id: int,
    kernel: LeanKernel,
    scope_premises: list[str],
) -> SeqResult:
    """Replay exact gold atoms in order (direction + args preserved)."""
    res = SeqResult(condition="oracle_seq_exact", n_atoms=len(atoms))
    goal = start_goal
    goal_id = start_goal_id

    for i, atom in enumerate(atoms):
        short = atom_premise_name(atom)
        full_name = resolve_atom_premise(short, scope_premises)
        tac = atom_to_tactic(atom, full_name)

        ok, new_goal, err, crashed = try_tactic_safe(kernel, goal, tac, goal_id=goal_id)
        sr = StepRecord(
            step_idx=i, tactic=tac, success=ok, error=err,
            error_category=_categorize_error(err) if not ok else "",
            lean_calls=1, atom_had_args=not atom_is_bare(atom),
        )
        res.steps.append(sr)
        res.total_lean_calls += 1

        if crashed or not ok:
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


def _run_cosine_seq_bare(
    atoms: list[str],
    start_goal: str,
    start_goal_id: int,
    kernel: LeanKernel,
    premise_names: list[str],
    encoder,
    max_scope_premises: int,
    cosine_topk: int,
) -> SeqResult:
    """Sequential cosine bare: at each step refresh scope, rerank, try top-k bare."""
    res = SeqResult(condition="cosine_seq_bare", n_atoms=len(atoms))
    goal = start_goal
    goal_id = start_goal_id

    for i, atom in enumerate(atoms):
        scope = scope_for_rw(goal, premise_names, max_premises=max_scope_premises)
        ranked = cosine_rank(goal, scope.premises, encoder)

        lean_calls = 0
        step_ok = False
        step_tac = ""
        step_err = ""
        new_goal = ""
        err = ""

        top_k = min(cosine_topk, len(ranked))
        for _, premise in ranked[:top_k]:
            for direction in ("forward", "backward"):
                tac = build_rw_tactic(premise, direction)
                ok, ng, err, crashed = try_tactic_safe(kernel, goal, tac, goal_id=goal_id)
                lean_calls += 1
                if crashed:
                    sr = StepRecord(step_idx=i, tactic=tac, success=False,
                                    error=err, error_category="server_crash", lean_calls=lean_calls,
                                    atom_had_args=not atom_is_bare(atom))
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
                        lean_calls=lean_calls, atom_had_args=not atom_is_bare(atom))
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
class Rw36Result:
    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    n_atoms: int = 0
    n_args_atoms: int = 0           # atoms with positional args
    first_atom_bare: bool = False   # first atom is bare
    canonical_action_ir: str = ""

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    n_accessible_premises: int = 0
    n_scope_premises: int = 0
    gold_first_in_scope: bool = False
    gold_first_rank: int = -1

    # oracle_first_exact
    oracle_first_success: bool = False
    oracle_first_tactic: str = ""
    oracle_first_error: str = ""
    oracle_first_error_category: str = ""
    oracle_first_new_goal: str = ""

    # cosine5_bare_first
    cosine5_bare_success: bool = False
    cosine5_bare_tactic: str = ""
    cosine5_bare_rank: int = -1
    cosine5_bare_calls: int = 0
    cosine5_bare_new_goal: str = ""

    # cosine5_heuristic_first (bare + local-hyp args)
    cosine5_heur_success: bool = False
    cosine5_heur_tactic: str = ""
    cosine5_heur_rank: int = -1
    cosine5_heur_calls: int = 0

    # oracle_seq_exact
    oracle_seq_full: bool = False
    oracle_seq_divergence: int = -1
    oracle_seq_lean_calls: int = 0
    oracle_seq_step_errors: list = field(default_factory=list)

    # cosine_seq_bare
    cosine_seq_full: bool = False
    cosine_seq_divergence: int = -1
    cosine_seq_lean_calls: int = 0
    cosine_seq_step_errors: list = field(default_factory=list)

    # partition
    partition: str = ""   # args_redundant | args_necessary | unexecutable_oracle | not_started

    elapsed_s: float = 0.0


def _assign_partition(res: Rw36Result) -> str:
    if not res.goal_started:
        return "not_started"
    if res.cosine5_bare_success:
        return "args_redundant"
    if res.oracle_first_success:
        return "args_necessary"
    return "unexecutable_oracle"


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
) -> Rw36Result:
    t0 = time.time()
    ir = example.get("canonical_action_ir", "")
    atoms = split_atoms(ir)

    res = Rw36Result(
        theorem_full_name=example["theorem_full_name"],
        file_path=example["file_path"],
        step_index=example.get("step_index", 0),
        n_atoms=len(atoms),
        n_args_atoms=sum(1 for a in atoms if not atom_is_bare(a)),
        first_atom_bare=atom_is_bare(atoms[0]) if atoms else False,
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
        res.partition = "not_started"
        res.elapsed_s = time.time() - t0
        return res

    goal_str = replay.goal_state
    goal_id = replay.goal_id

    premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
    res.n_accessible_premises = len(premise_names)
    scope = scope_for_rw(goal_str, premise_names, max_premises=max_scope_premises)
    res.n_scope_premises = len(scope.premises)

    # Gold first atom info
    first_atom = atoms[0] if atoms else ""
    gold_first_short = atom_premise_name(first_atom)
    gold_first_full = resolve_atom_premise(gold_first_short, scope.premises)
    ann = example.get("annotated_premise", "")
    if ann and ann in scope.premises:
        gold_first_full = ann
    res.gold_first_in_scope = gold_first_full in scope.premises

    ranked = cosine_rank(goal_str, scope.premises, encoder)
    for i, (_, prem) in enumerate(ranked):
        if prem == gold_first_full:
            res.gold_first_rank = i
            break

    # --- oracle_first_exact ---
    oracle_tac = atom_to_tactic(first_atom, gold_first_full)
    res.oracle_first_tactic = oracle_tac
    ok, new_goal, err, crashed = try_tactic_safe(kernel, goal_str, oracle_tac, goal_id=goal_id)
    if crashed:
        res.crash_retries += 1
        res.oracle_first_error = err
        res.partition = _assign_partition(res)
        res.elapsed_s = time.time() - t0
        return res
    res.oracle_first_success = ok
    res.oracle_first_error = err
    res.oracle_first_error_category = _categorize_error(err) if not ok else ""
    res.oracle_first_new_goal = new_goal

    # --- cosine5_bare_first ---
    top_k = min(cosine_topk, len(ranked))
    for rank, (_, premise) in enumerate(ranked[:top_k]):
        for direction in ("forward", "backward"):
            tac = build_rw_tactic(premise, direction)
            ok_c, ng_c, _, crashed_c = try_tactic_safe(kernel, goal_str, tac, goal_id=goal_id)
            res.cosine5_bare_calls += 1
            if crashed_c:
                res.crash_retries += 1
                res.partition = _assign_partition(res)
                res.elapsed_s = time.time() - t0
                return res
            if ok_c:
                res.cosine5_bare_success = True
                res.cosine5_bare_tactic = tac
                res.cosine5_bare_rank = rank
                res.cosine5_bare_new_goal = ng_c
                break
        if res.cosine5_bare_success:
            break

    # --- cosine5_heuristic_first (bare + local-hyp args) ---
    # Only run if bare already failed; adds local hypothesis names as positional args.
    if not res.cosine5_bare_success:
        local_hyps = extract_local_hyps(goal_str)
        # Try: for each top-5 premise, with each local hyp appended as positional arg
        crashed_h = False
        for rank, (_, premise) in enumerate(ranked[:top_k]):
            for hyp in local_hyps[:5]:
                for direction in ("forward", "backward"):
                    tac = build_rw_tactic(premise, direction, [hyp])
                    ok_h, _, _, crashed_h = try_tactic_safe(kernel, goal_str, tac, goal_id=goal_id)
                    res.cosine5_heur_calls += 1
                    if crashed_h:
                        res.crash_retries += 1
                        break
                    if ok_h:
                        res.cosine5_heur_success = True
                        res.cosine5_heur_tactic = tac
                        res.cosine5_heur_rank = rank
                        break
                if res.cosine5_heur_success or crashed_h:
                    break
            if res.cosine5_heur_success or crashed_h:
                break
    else:
        # Bare succeeded; heuristic is strictly dominated
        res.cosine5_heur_success = True
        res.cosine5_heur_tactic = res.cosine5_bare_tactic
        res.cosine5_heur_rank = res.cosine5_bare_rank

    # Assign partition (before sequential, which doesn't affect partition)
    res.partition = _assign_partition(res)

    # --- oracle_seq_exact ---
    oracle_seq_r = _run_oracle_seq(atoms, goal_str, goal_id, kernel, scope.premises)
    res.oracle_seq_full = oracle_seq_r.full_seq_accept
    res.oracle_seq_divergence = oracle_seq_r.divergence_step
    res.oracle_seq_lean_calls = oracle_seq_r.total_lean_calls
    res.oracle_seq_step_errors = [(s.step_idx, s.error_category, s.atom_had_args)
                                   for s in oracle_seq_r.steps if not s.success]

    # --- cosine_seq_bare ---
    cosine_seq_r = _run_cosine_seq_bare(
        atoms, goal_str, goal_id, kernel, premise_names, encoder,
        max_scope_premises, cosine_topk,
    )
    res.cosine_seq_full = cosine_seq_r.full_seq_accept
    res.cosine_seq_divergence = cosine_seq_r.divergence_step
    res.cosine_seq_lean_calls = cosine_seq_r.total_lean_calls
    res.cosine_seq_step_errors = [(s.step_idx, s.error_category, s.atom_had_args)
                                   for s in cosine_seq_r.steps if not s.success]

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[Rw36Result], total: int) -> None:
    started = [r for r in results if r.goal_started]
    n = len(started)
    fail_cats = Counter(r.failure_category or "unknown" for r in results if not r.goal_started)

    if total == 0:
        print("No examples.")
        return

    print("\n" + "=" * 70)
    print("rw3-with-args Benchmark (EXP-RW-036)")
    print("=" * 70)
    print(f"\nExamples (rw3-with-args, step-0): {total}")
    print(f"GoalStart: {n}/{total} ({100*n/max(total,1):.1f}%)")
    if fail_cats:
        for cat, cnt in fail_cats.most_common():
            print(f"  {cat}: {cnt}")

    if n == 0:
        return

    # Atom profile
    n_args_atoms_hist = Counter(r.n_args_atoms for r in started)
    first_bare_count = sum(r.first_atom_bare for r in started)
    print(f"\nAtom profile (started):")
    print(f"  First atom is bare: {first_bare_count}/{n} ({100*first_bare_count/n:.1f}%)")
    print(f"  Args-atoms per example: " +
          ", ".join(f"{k}={v}" for k, v in sorted(n_args_atoms_hist.items())))

    # --- First-atom conditions ---
    oracle_ok = sum(r.oracle_first_success for r in started)
    c5b_ok = sum(r.cosine5_bare_success for r in started)
    c5h_ok = sum(r.cosine5_heur_success for r in started)

    gold_scope = [r for r in started if r.gold_first_in_scope]
    m = len(gold_scope)
    oracle_gs = sum(r.oracle_first_success for r in gold_scope)
    c5b_gs = sum(r.cosine5_bare_success for r in gold_scope)

    print(f"\n{'--- EXP-RW-036a: First-Atom Acceptance ---'}")
    print(f"\n{'Condition':<45} {'|started':>10} {'Rate':>7} {'|gold_in_scope':>16} {'Rate':>7}")
    print("-" * 85)

    def row(label: str, ok_s: int, ok_gs: int) -> None:
        r1, r2 = f"{ok_s}/{n}", f"{100*ok_s/max(n,1):.1f}%"
        r3, r4 = f"{ok_gs}/{m}", f"{100*ok_gs/max(m,1):.1f}%"
        print(f"{label:<45} {r1:>10} {r2:>7} {r3:>16} {r4:>7}")

    row("Oracle first atom (exact: dir+args)", oracle_ok, oracle_gs)
    row("Cosine top-5 bare (no args)", c5b_ok, c5b_gs)
    row("Cosine top-5 + heuristic hyp args", c5h_ok, sum(r.cosine5_heur_success for r in gold_scope))

    print(f"\nGold first atom in scope: {m}/{n} ({100*m/max(n,1):.1f}%)")
    ranks = [r.gold_first_rank for r in started if r.gold_first_rank >= 0]
    if ranks:
        print(f"Gold first-atom rank: mean={np.mean(ranks):.1f}, median={np.median(ranks):.1f}")

    # Oracle failure taxonomy
    err_cats = Counter(r.oracle_first_error_category for r in started
                       if not r.oracle_first_success and r.oracle_first_error_category)
    if err_cats:
        print(f"\nOracle first-atom failure taxonomy:")
        for cat, cnt in err_cats.most_common():
            print(f"  {cat}: {cnt}")

    # --- Partition ---
    part_counts = Counter(r.partition for r in results)
    print(f"\n{'=== EXP-RW-036 Partition ==='}")
    total_all = len(results)
    for bkt in ("args_redundant", "args_necessary", "unexecutable_oracle", "not_started"):
        v = part_counts[bkt]
        print(f"  {bkt:<25}: {v:>4}/{total_all} ({100*v/total_all:.1f}%)")

    # Cross-tab: bare vs heuristic (among started)
    heur_only = sum(1 for r in started if not r.cosine5_bare_success and r.cosine5_heur_success)
    both = sum(1 for r in started if r.cosine5_bare_success and r.cosine5_heur_success)
    neither = sum(1 for r in started if not r.cosine5_bare_success and not r.cosine5_heur_success)
    print(f"\n  Bare succeeds:        {c5b_ok}/{n} ({100*c5b_ok/n:.1f}%)")
    print(f"  Heuristic-only wins:  {heur_only}/{n}  ← args confirmed necessary cases")
    print(f"  Both succeed:         {both}/{n}")
    print(f"  Neither:              {neither}/{n}")

    # --- Sequential conditions ---
    oracle_seq_full = sum(r.oracle_seq_full for r in started)
    cosine_seq_full = sum(r.cosine_seq_full for r in started)

    print(f"\n{'--- EXP-RW-036b: Sequential Acceptance ---'}")
    print(f"\n{'Condition':<45} {'FullSeq':>8} {'Rate':>7}")
    print("-" * 62)
    print(f"{'Oracle sequential (exact: dir+args)':<45} {oracle_seq_full:>8} {100*oracle_seq_full/n:>6.1f}%")
    print(f"{'Cosine-5 sequential bare':<45} {cosine_seq_full:>8} {100*cosine_seq_full/n:>6.1f}%")

    # Accept@k for oracle_seq
    print(f"\nAccept@k (oracle_seq_exact):")
    for k in range(1, 5):
        ok_k = sum(1 for r in started if r.oracle_seq_divergence == -1 or r.oracle_seq_divergence >= k)
        print(f"  k={k}: {ok_k}/{n} ({100*ok_k/n:.1f}%)")

    print(f"\nAccept@k (cosine_seq_bare):")
    for k in range(1, 5):
        ok_k = sum(1 for r in started if r.cosine_seq_divergence == -1 or r.cosine_seq_divergence >= k)
        print(f"  k={k}: {ok_k}/{n} ({100*ok_k/n:.1f}%)")

    # Failure taxonomy by step index (oracle_seq), annotated with atom_had_args
    oracle_err_by_step: dict[int, Counter] = defaultdict(Counter)
    for r in started:
        for step_idx, cat, had_args in r.oracle_seq_step_errors:
            key = f"{cat}({'args' if had_args else 'bare'})"
            oracle_err_by_step[step_idx][key] += 1
    if oracle_err_by_step:
        print(f"\nOracle sequential failure taxonomy by step index (bare|args annotation):")
        for step in sorted(oracle_err_by_step)[:6]:
            cats = oracle_err_by_step[step].most_common()
            cat_str = ", ".join(f"{c}:{v}" for c, v in cats)
            print(f"  step {step}: {cat_str}")

    # Lean call budget
    o_calls = sum(r.oracle_seq_lean_calls for r in started)
    c_calls = sum(r.cosine_seq_lean_calls for r in started)
    c5_calls = sum(r.cosine5_bare_calls for r in started)
    c5h_calls = sum(r.cosine5_heur_calls for r in started)
    print(f"\nLean call budget (per started example):")
    print(f"  cosine5_bare_first:    {c5_calls/n:.1f}")
    print(f"  cosine5_heur_first:    {c5h_calls/n:.1f} (additional heuristic calls)")
    print(f"  oracle_seq_exact:      {o_calls/n:.1f}")
    print(f"  cosine_seq_bare:       {c_calls/n:.1f}")

    # Per atom-count breakdown
    atom_groups = sorted(set(r.n_atoms for r in started))
    if len(atom_groups) > 1:
        print(f"\nPartition by total atom count:")
        for k in atom_groups:
            grp = [r for r in started if r.n_atoms == k]
            c_arg = Counter(r.partition for r in grp)
            print(f"  {k} atoms ({len(grp)}): " +
                  ", ".join(f"{bkt}={c_arg[bkt]}" for bkt in ("args_redundant", "args_necessary", "unexecutable_oracle")))

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Worker / parallel
# ---------------------------------------------------------------------------


def _worker_cmd(args: argparse.Namespace, shard_index: int, num_shards: int, shard_out: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.run_rw36_benchmark",
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

    results = [Rw36Result(**{k: v for k, v in row.items() if k in Rw36Result.__dataclass_fields__})
               for row in merged]
    print_report(results, len(results))


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = load_rw3_args_examples(args.data, step0_only=not args.include_step_gt0, limit=args.limit)
    if args.num_shards > 1:
        examples = [ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_index]
    logger.info("Worker shard %d/%d: %d rw3-with-args examples", args.shard_index + 1, args.num_shards, len(examples))

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

    results: list[Rw36Result] = []
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
                res = Rw36Result(
                    theorem_full_name=ex.get("theorem_full_name", ""),
                    file_path=ex.get("file_path", ""),
                    step_index=ex.get("step_index", 0),
                    failure_category="unhandled_error",
                    partition="not_started",
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
    parser = argparse.ArgumentParser(description="rw3-with-args benchmark (EXP-RW-036)")
    parser.add_argument("--data", default="data/canonical/canonical_rw_eval.jsonl")
    parser.add_argument("--db", default="data/proof_network.db")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--output", default="runs/rw36_results.jsonl")
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
