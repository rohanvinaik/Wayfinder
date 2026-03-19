"""rw2 benchmark — evaluate rewrite-with-args prediction on step-0 examples.

Default dataset: data/canonical/canonical_rw_eval.jsonl filtered to rw2
(single rewrite with applied args). The runner evaluates:
  1. Oracle qualified: canonical ActionIR, with the rewrite head qualified from scope when possible
  2. Cosine top-1: top cosine premise + heuristic local-arg beam
  3. Cosine top-5: top-5 cosine premises + heuristic local-arg beam

The arg baseline is intentionally simple: extract local names from the started goal state,
rank the most promising locals, then try ordered argument combinations. This is the first
component benchmark for rw2; it is not intended to solve named args / inline `by` / dotted
lemma-as-arg cases.

Parallel mode (`--parallel N`) shards the filtered example list across N worker subprocesses,
so each shard gets its own Pantograph instance.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult, ServerCrashError
from src.proof_network import get_accessible_premises
from src.rw_scoper import infer_direction, scope_for_rw
from src.tactic_canonicalizer import canonicalize
from src.tactic_ir import ActionIR, Direction, TermExpr, TermKind

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    theorem_full_name: str = ""
    file_path: str = ""
    step_index: int = 0
    grammar_tier: str = ""
    goal_state_before: str = ""
    canonical_action_ir: str = ""
    annotated_premise: str = ""
    gold_premise_name: str = ""
    gold_direction: str = ""
    gold_arg_count: int = 0

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    n_accessible_premises: int = 0
    n_scope_symbols: int = 0
    n_scope_premises: int = 0
    n_local_terms: int = 0
    n_arg_sequences: int = 0
    gold_in_accessible: bool = False
    gold_in_scope: bool = False

    oracle_success: bool = False
    oracle_error: str = ""
    oracle_has_remaining_goals: bool = False
    oracle_tactic: str = ""

    cosine_top1_success: bool = False
    cosine_top1_error: str = ""
    cosine_top1_rank: int = -1
    cosine_top1_tactic: str = ""
    cosine_top1_calls: int = 0

    cosine_top5_success: bool = False
    cosine_top5_error: str = ""
    cosine_top5_rank: int = -1
    cosine_top5_tactic: str = ""
    cosine_top5_calls: int = 0

    heuristic_failure: str = ""
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class LocalTerm:
    text: str
    position: int
    score: float


# ---------------------------------------------------------------------------
# Data loading / classification
# ---------------------------------------------------------------------------


def _count_term_args(term: TermExpr) -> int:
    if term.kind == TermKind.APP:
        return len(term.args)
    if term.kind == TermKind.CHAIN:
        return max(0, len(term.args) - 1)
    if term.kind == TermKind.CTOR:
        return len(term.args)
    return 0


def classify_rw_tier(canonical_action_ir: str) -> tuple[str, ActionIR | None]:
    ir = canonicalize(canonical_action_ir, "rw")
    if ir is None:
        return "unknown", None
    if len(ir.rewrites) != 1:
        return "rw3_multi", ir

    atom = ir.rewrites[0]
    nargs = _count_term_args(atom.expr)
    if nargs > 0:
        return ("rw2_bwd_args" if atom.direction == Direction.BACKWARD else "rw2_fwd_args"), ir
    if atom.direction == Direction.BACKWARD:
        return "rw1_bwd", ir
    return "rw0_fwd", ir


def load_rw2_examples(path: str, step0_only: bool = True, limit: int = 0) -> list[dict]:
    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            tier, _ = classify_rw_tier(ex.get("canonical_action_ir", ""))
            if tier not in {"rw2_fwd_args", "rw2_bwd_args"}:
                continue
            ex["grammar_tier"] = tier
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


def cosine_rank_symbols(goal_text: str, symbols: list[str], encoder: object | None = None) -> list[tuple[float, str]]:
    if not symbols or encoder is None:
        return [(0.0, s) for s in symbols]
    try:
        goal_emb = encoder.encode([goal_text], normalize_embeddings=True)
        sym_embs = encoder.encode(symbols, normalize_embeddings=True)
        scores = (goal_emb @ sym_embs.T).flatten()
        return sorted(zip(scores.tolist(), symbols), reverse=True)
    except Exception as e:
        logger.warning("Cosine encoding failed: %s", e)
        return [(0.0, s) for s in symbols]


# ---------------------------------------------------------------------------
# Goal-state parsing / heuristic arg beam
# ---------------------------------------------------------------------------


def _parse_goal_locals(goal_state: str) -> list[tuple[str, str, int]]:
    locals_out: list[tuple[str, str, int]] = []
    for i, raw_line in enumerate(goal_state.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("⊢") or "⊢" in line:
            break
        if line.startswith("case "):
            continue
        if " : " not in line:
            continue
        lhs, rhs = line.split(" : ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or lhs == "case":
            continue
        names = [tok for tok in lhs.split() if tok and tok != "case"]
        for name in names:
            locals_out.append((name, rhs, i))
    return locals_out


def _is_type_line(type_text: str) -> bool:
    return type_text.startswith("Type") or type_text.startswith("Sort")


def _local_score(name: str, type_text: str) -> float:
    score = 0.0
    if name.startswith("inst"):
        score -= 4.0
    if _is_type_line(type_text):
        score -= 4.0
    if name[:1].islower():
        score += 1.0
    if len(name) <= 4:
        score += 1.5
    if name.startswith(("h", "this", "eq", "ne", "hs", "hm", "hp", "hf", "hg")):
        score += 2.5
    if any(tok in type_text for tok in (" = ", " ↔ ", " ≤ ", " < ", " ∈ ", " ≠ ", "¬", "∃", "∀")):
        score += 1.0
    return score


def extract_local_terms(goal_state: str, max_locals: int = 8) -> list[LocalTerm]:
    locals_raw = _parse_goal_locals(goal_state)
    candidates: list[tuple[float, int, LocalTerm]] = []
    for name, type_text, position in locals_raw:
        if name.startswith("inst") or _is_type_line(type_text):
            continue
        score = _local_score(name, type_text)
        if score <= -1.0:
            continue
        candidates.append((score, position, LocalTerm(text=name, position=position, score=score)))
        if any(tok in type_text for tok in (" = ", " ↔ ", " ≠ ")):
            candidates.append((score - 0.2, position, LocalTerm(text=f"{name}.symm", position=position, score=score - 0.2)))
        if any(tok in type_text for tok in (" ∧ ", " × ", " Exists", "∃")):
            candidates.append((score - 0.5, position, LocalTerm(text=f"{name}.1", position=position, score=score - 0.5)))
            candidates.append((score - 0.5, position, LocalTerm(text=f"{name}.2", position=position, score=score - 0.5)))

    # Select highest-scoring terms, then restore source order for combination generation.
    chosen = sorted(candidates, key=lambda x: (-x[0], x[1], x[2].text))[:max_locals]
    dedup: dict[str, LocalTerm] = {}
    for _, _, term in chosen:
        dedup.setdefault(term.text, term)
    return sorted(dedup.values(), key=lambda t: (t.position, -t.score, t.text))


def build_arg_sequences(
    local_terms: list[LocalTerm],
    max_args: int = 4,
    max_sequences: int = 24,
) -> list[tuple[str, ...]]:
    if not local_terms:
        return [tuple()]

    scored: list[tuple[float, tuple[str, ...]]] = []
    terms = list(local_terms)
    for n in range(1, min(max_args, len(terms)) + 1):
        for combo in itertools.combinations(terms, n):
            texts = tuple(t.text for t in combo)
            combo_score = sum(t.score for t in combo) - 0.15 * n
            scored.append((combo_score, texts))

    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    sequences = [tuple()]
    seen = {tuple()}
    for _, seq in scored:
        if seq in seen:
            continue
        sequences.append(seq)
        seen.add(seq)
        if len(sequences) >= max_sequences:
            break

    # Add a few wildcard fallbacks for the hard tail.
    for n in range(1, min(2, max_args) + 1):
        seq = tuple("_" for _ in range(n))
        if seq not in seen and len(sequences) < max_sequences:
            sequences.append(seq)
            seen.add(seq)
    return sequences


# ---------------------------------------------------------------------------
# Tactic building / oracle qualification
# ---------------------------------------------------------------------------


def _extract_head_name(term: TermExpr) -> str:
    if term.kind in {TermKind.CONST, TermKind.VAR}:
        return term.head
    if term.kind == TermKind.APP:
        return term.head
    if term.kind == TermKind.CHAIN and term.args:
        return _extract_head_name(term.args[0])
    if term.kind == TermKind.PROJ and term.args:
        return _extract_head_name(term.args[0])
    return ""


def _qualify_term_head(term: TermExpr, full_name: str) -> bool:
    short = full_name.rsplit(".", 1)[-1]
    if term.kind == TermKind.CONST:
        if term.head == short or term.head == full_name:
            term.head = full_name
            return True
        return False
    if term.kind == TermKind.APP:
        if term.head == short or term.head == full_name:
            term.head = full_name
            return True
        return False
    if term.kind in {TermKind.CHAIN, TermKind.PROJ} and term.args:
        return _qualify_term_head(term.args[0], full_name)
    return False


def resolve_gold_premise_name(example: dict, scope_premises: list[str]) -> str:
    annotated = example.get("annotated_premise", "") or ""
    if annotated and annotated in scope_premises:
        return annotated

    _, ir = classify_rw_tier(example.get("canonical_action_ir", ""))
    if ir is None or len(ir.rewrites) != 1:
        return annotated
    short_head = _extract_head_name(ir.rewrites[0].expr)
    if not short_head:
        return annotated

    exact = [p for p in scope_premises if p == short_head]
    if exact:
        return exact[0]

    suffix = [p for p in scope_premises if p.rsplit(".", 1)[-1] == short_head]
    if len(suffix) == 1:
        return suffix[0]
    if annotated:
        suffix = [p for p in scope_premises if p.endswith(f".{short_head}") and p.endswith(annotated.split(".")[-1])]
        if suffix:
            return suffix[0]
    return annotated


def qualify_oracle_tactic(canonical_action_ir: str, gold_premise_name: str) -> str:
    if not gold_premise_name:
        return canonical_action_ir
    ir = canonicalize(canonical_action_ir, "rw")
    if ir is None or len(ir.rewrites) != 1:
        return canonical_action_ir
    ir = copy.deepcopy(ir)
    if _qualify_term_head(ir.rewrites[0].expr, gold_premise_name):
        return ir.lower()
    return canonical_action_ir


def build_rw_tactic(premise: str, args: Iterable[str], direction: str) -> str:
    prefix = "← " if direction == "backward" else ""
    atom = prefix + premise
    arg_list = list(args)
    if arg_list:
        atom = f"{atom} {' '.join(arg_list)}"
    return f"rw [{atom}]"


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def try_tactic_safe(kernel: LeanKernel, goal_state: str, tactic: str, goal_id: int = 0) -> tuple[bool, str, bool, bool]:
    try:
        result = kernel.try_tactic(goal_state, tactic, goal_id=goal_id)
        has_remaining = bool(result.new_goals) and result.success
        return result.success, result.error_message, has_remaining, False
    except ServerCrashError as e:
        return False, f"server_crash: {e}", False, True
    except Exception as e:
        return False, str(e), False, False


# ---------------------------------------------------------------------------
# Example evaluation
# ---------------------------------------------------------------------------


def _direction_candidates(mode: str, premise: str, goal_text: str, grammar_tier: str) -> list[str]:
    if mode == "tier-default":
        return ["backward"] if grammar_tier == "rw2_bwd_args" else ["forward"]
    if mode == "infer":
        inferred = infer_direction(premise, goal_text)
        if inferred == "ambiguous":
            return ["forward", "backward"]
        return [inferred]
    return ["forward", "backward"]


def _heuristic_failure_bucket(res: ExampleResult) -> str:
    if res.cosine_top5_success:
        return ""
    if not res.goal_started:
        return res.failure_category or "goal_creation_fail"
    if not res.gold_in_scope:
        return "premise_wrong"
    return "args_or_lowering"


def run_one_example(
    example: dict,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder: object | None,
    project_root: str,
    max_scope_premises: int,
    cosine_topk: int,
    max_locals: int,
    max_args: int,
    max_arg_sequences: int,
    direction_mode: str,
) -> ExampleResult:
    t0 = time.time()
    grammar_tier, ir = classify_rw_tier(example.get("canonical_action_ir", ""))
    gold_arg_count = 0
    gold_direction = ""
    if ir is not None and len(ir.rewrites) == 1:
        gold_arg_count = _count_term_args(ir.rewrites[0].expr)
        gold_direction = ir.rewrites[0].direction.value

    res = ExampleResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example["file_path"],
        step_index=example["step_index"],
        grammar_tier=grammar_tier,
        goal_state_before=example.get("goal_state_before", ""),
        canonical_action_ir=example.get("canonical_action_ir", ""),
        annotated_premise=example.get("annotated_premise", ""),
        gold_direction=gold_direction,
        gold_arg_count=gold_arg_count,
    )

    replay: ReplayResult = kernel.goal_via_file_context(
        theorem_full_name=res.theorem_full_name,
        file_path=res.file_path,
        prefix_tactics=example.get("prefix_tactics", []),
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

    premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
    res.n_accessible_premises = len(premise_names)
    scope = scope_for_rw(goal_str, premise_names, max_premises=max_scope_premises)
    ranked = cosine_rank_symbols(goal_str, scope.premises, encoder)
    res.n_scope_symbols = len(scope.all_symbols)
    res.n_scope_premises = len(scope.premises)

    res.gold_premise_name = resolve_gold_premise_name(example, scope.premises)
    if res.gold_premise_name:
        res.gold_in_accessible = res.gold_premise_name in premise_names
        res.gold_in_scope = res.gold_premise_name in scope.premises

    local_terms = extract_local_terms(goal_str, max_locals=max_locals)
    arg_sequences = build_arg_sequences(local_terms, max_args=max_args, max_sequences=max_arg_sequences)
    res.n_local_terms = len(local_terms)
    res.n_arg_sequences = len(arg_sequences)

    # Oracle
    oracle_tactic = qualify_oracle_tactic(res.canonical_action_ir, res.gold_premise_name)
    res.oracle_tactic = oracle_tactic
    if oracle_tactic:
        ok, err, remaining, crashed = try_tactic_safe(kernel, goal_str, oracle_tactic, goal_id=goal_id)
        if crashed:
            res.crash_retries += 1
            res.oracle_error = err
            res.elapsed_s = time.time() - t0
            return res
        res.oracle_success = ok
        res.oracle_error = err
        res.oracle_has_remaining_goals = remaining

    # Cosine top-k with arg heuristic
    goal_target = ""
    for line in goal_str.splitlines():
        if "⊢" in line:
            goal_target = line.split("⊢", 1)[1].strip()
            break

    gold_rank = -1
    if res.gold_premise_name:
        for i, (_, prem) in enumerate(ranked):
            if prem == res.gold_premise_name:
                gold_rank = i
                break

    top_k = min(cosine_topk, len(ranked))
    top1_done = False
    top5_done = False

    for prem_rank in range(top_k):
        _, premise = ranked[prem_rank]
        directions = _direction_candidates(direction_mode, premise, goal_target, res.grammar_tier)
        for direction in directions:
            for arg_seq in arg_sequences:
                tactic = build_rw_tactic(premise, arg_seq, direction)
                ok, err, _, crashed = try_tactic_safe(kernel, goal_str, tactic, goal_id=goal_id)
                res.cosine_top5_calls += 1
                if prem_rank == 0:
                    res.cosine_top1_calls += 1
                if crashed:
                    res.crash_retries += 1
                    if not top1_done:
                        res.cosine_top1_error = err
                    if not top5_done:
                        res.cosine_top5_error = err
                    res.elapsed_s = time.time() - t0
                    return res
                if ok:
                    if prem_rank == 0 and not top1_done:
                        res.cosine_top1_success = True
                        res.cosine_top1_rank = prem_rank
                        res.cosine_top1_tactic = tactic
                        res.cosine_top1_error = ""
                        top1_done = True
                    if not top5_done:
                        res.cosine_top5_success = True
                        res.cosine_top5_rank = prem_rank
                        res.cosine_top5_tactic = tactic
                        res.cosine_top5_error = ""
                        top5_done = True
                    break
                if prem_rank == 0 and not top1_done:
                    res.cosine_top1_error = err
                    res.cosine_top1_tactic = tactic
                if not top5_done:
                    res.cosine_top5_error = err
                    res.cosine_top5_tactic = tactic
            if top5_done or (prem_rank == 0 and top1_done):
                # Stop extra directions for top-1 once successful, but keep top-5 premise loop unless already done.
                if top5_done:
                    break
                if prem_rank == 0 and top1_done:
                    continue
        if top5_done:
            break

    if top1_done and res.cosine_top1_rank < 0:
        res.cosine_top1_rank = 0
    if top5_done and res.cosine_top5_rank < 0:
        res.cosine_top5_rank = 0

    res.heuristic_failure = _heuristic_failure_bucket(res)
    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[ExampleResult], total: int) -> None:
    started = [r for r in results if r.goal_started]
    started_n = len(started)
    tier_counts = Counter(r.grammar_tier for r in results)
    fail_cats = Counter(r.failure_category or "unknown" for r in results if not r.goal_started)
    heuristic_fail = Counter(r.heuristic_failure for r in started if r.heuristic_failure)
    gold_scope = [r for r in started if r.gold_in_scope]
    scope_sizes = [r.n_scope_premises for r in started if r.n_scope_premises > 0]
    local_sizes = [r.n_local_terms for r in started if r.n_local_terms > 0]
    seq_sizes = [r.n_arg_sequences for r in started if r.n_arg_sequences > 0]

    oracle_ok = sum(r.oracle_success for r in started)
    c1_ok = sum(r.cosine_top1_success for r in started)
    c5_ok = sum(r.cosine_top5_success for r in started)
    total_calls_1 = sum(r.cosine_top1_calls for r in started)
    total_calls_5 = sum(r.cosine_top5_calls for r in started)

    print("\n" + "=" * 60)
    print("rw2 Benchmark Report")
    print("=" * 60)
    print(f"\nExamples: {total}")
    for tier, cnt in sorted(tier_counts.items()):
        print(f"  {tier}: {cnt}")

    print(f"\nGoalStart@rw2: {started_n}/{total} ({100*started_n/max(total,1):.1f}%)")
    if fail_cats:
        print("  Failure breakdown:")
        for cat, cnt in fail_cats.most_common():
            print(f"    {cat}: {cnt}")
    if started_n == 0:
        print("\nNo started goals.")
        return

    print(f"\nLeanValid@rw2|started (N={started_n}):")
    print(f"  Oracle qualified: {oracle_ok}/{started_n} ({100*oracle_ok/max(started_n,1):.1f}%)")
    print(f"  Cosine top-1:     {c1_ok}/{started_n} ({100*c1_ok/max(started_n,1):.1f}%)")
    print(f"  Cosine top-5:     {c5_ok}/{started_n} ({100*c5_ok/max(started_n,1):.1f}%)")

    if gold_scope:
        print(f"\nLeanValid@rw2|started,gold_in_scope (M={len(gold_scope)}):")
        print(f"  Oracle qualified: {sum(r.oracle_success for r in gold_scope)}/{len(gold_scope)} ({100*sum(r.oracle_success for r in gold_scope)/max(len(gold_scope),1):.1f}%)")
        print(f"  Cosine top-1:     {sum(r.cosine_top1_success for r in gold_scope)}/{len(gold_scope)} ({100*sum(r.cosine_top1_success for r in gold_scope)/max(len(gold_scope),1):.1f}%)")
        print(f"  Cosine top-5:     {sum(r.cosine_top5_success for r in gold_scope)}/{len(gold_scope)} ({100*sum(r.cosine_top5_success for r in gold_scope)/max(len(gold_scope),1):.1f}%)")

    print("\nScope stats:")
    print(f"  Gold in scope: {len(gold_scope)}/{started_n} ({100*len(gold_scope)/max(started_n,1):.1f}%)")
    if scope_sizes:
        print(f"  Mean scope size: {np.mean(scope_sizes):.1f}")
    if local_sizes:
        print(f"  Mean local-term beam: {np.mean(local_sizes):.1f}")
    if seq_sizes:
        print(f"  Mean arg-sequence beam: {np.mean(seq_sizes):.1f}")

    print("\nHeuristic call budget:")
    print(f"  Cosine top-1 calls/example: {total_calls_1/max(started_n,1):.1f}")
    print(f"  Cosine top-5 calls/example: {total_calls_5/max(started_n,1):.1f}")

    if heuristic_fail:
        print("\nCosine failure taxonomy:")
        for cat, cnt in heuristic_fail.most_common():
            print(f"  {cat}: {cnt}")

    elapsed = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed:
        print(f"\nTiming: {np.mean(elapsed):.1f}s/example, total {sum(elapsed):.0f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Worker / parallel orchestration
# ---------------------------------------------------------------------------


def _base_worker_cmd(args: argparse.Namespace, shard_index: int, num_shards: int, shard_output: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_rw2_benchmark",
        "--worker",
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        "--output",
        shard_output,
        "--db",
        args.db,
        "--data",
        args.data,
        "--lean-project",
        args.lean_project,
        "--restart-every",
        str(args.restart_every),
        "--max-scope-premises",
        str(args.max_scope_premises),
        "--cosine-topk",
        str(args.cosine_topk),
        "--max-locals",
        str(args.max_locals),
        "--max-args",
        str(args.max_args),
        "--max-arg-sequences",
        str(args.max_arg_sequences),
        "--direction-mode",
        args.direction_mode,
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.include_step_gt0:
        cmd.append("--include-step-gt0")
    return cmd


def run_parallel(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    shard_dir = output_path.parent / f"{output_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen[str]] = []
    shard_paths: list[Path] = []
    for shard_index in range(args.parallel):
        shard_path = shard_dir / f"shard_{shard_index}.jsonl"
        shard_paths.append(shard_path)
        cmd = _base_worker_cmd(args, shard_index, args.parallel, str(shard_path))
        logger.info("Starting shard %d/%d", shard_index + 1, args.parallel)
        procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))

    rc = 0
    for i, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            logger.error("Shard %d failed with exit code %d", i, code)
            rc = code
    if rc != 0:
        raise SystemExit(rc)

    merged: list[dict] = []
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        with open(shard_path) as f:
            for line in f:
                if line.strip():
                    merged.append(json.loads(line))
    merged.sort(key=lambda d: (d.get("file_path", ""), d.get("theorem_full_name", ""), d.get("step_index", 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out_f:
        for row in merged:
            out_f.write(json.dumps(row) + "\n")

    results = [ExampleResult(**{k: v for k, v in row.items() if k in ExampleResult.__dataclass_fields__}) for row in merged]
    if not args.worker or args.num_shards == 1:
        print_report(results, len(results))


def run_worker(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    examples = load_rw2_examples(args.data, step0_only=not args.include_step_gt0, limit=args.limit)
    if args.num_shards > 1:
        examples = [ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_index]
    logger.info("Worker shard %d/%d loaded %d rw2 examples", args.shard_index + 1, args.num_shards, len(examples))

    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded MiniLM encoder")
    except ImportError:
        logger.warning("sentence-transformers not installed; cosine ranking disabled")

    kernel = LeanKernel(LeanConfig(
        backend="pantograph",
        timeout=120,
        project_root=args.lean_project,
        imports=["Mathlib"],
    ))
    logger.info("Initializing Pantograph with Mathlib")

    results: list[ExampleResult] = []
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
                logger.info("Shard %d progress: %d/%d", args.shard_index + 1, i + 1, len(examples))

            try:
                res = run_one_example(
                    example=ex,
                    kernel=kernel,
                    conn=conn,
                    id_to_name=id_to_name,
                    name_to_id=name_to_id,
                    encoder=encoder,
                    project_root=args.lean_project,
                    max_scope_premises=args.max_scope_premises,
                    cosine_topk=args.cosine_topk,
                    max_locals=args.max_locals,
                    max_args=args.max_args,
                    max_arg_sequences=args.max_arg_sequences,
                    direction_mode=args.direction_mode,
                )
            except Exception as e:
                logger.error("Unhandled error on %s step %d: %s", ex.get("theorem_full_name"), ex.get("step_index"), e)
                res = ExampleResult(
                    theorem_full_name=ex.get("theorem_full_name", ""),
                    file_path=ex.get("file_path", ""),
                    step_index=ex.get("step_index", 0),
                    grammar_tier=ex.get("grammar_tier", ""),
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
    parser = argparse.ArgumentParser(description="rw2 benchmark runner")
    parser.add_argument("--db", default="data/proof_network.db", help="Proof network DB path")
    parser.add_argument("--data", default="data/canonical/canonical_rw_eval.jsonl", help="Canonical rw eval JSONL")
    parser.add_argument("--lean-project", default="data/lean_project/", help="Lean project root")
    parser.add_argument("--output", default="runs/rw2_results.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Max rw2 examples after filtering (0 = all)")
    parser.add_argument("--include-step-gt0", action="store_true", help="Include step>0 examples (default step-0 only)")
    parser.add_argument("--restart-every", type=int, default=50, help="Restart server every N examples")
    parser.add_argument("--max-scope-premises", type=int, default=30, help="Max premises in rw scope")
    parser.add_argument("--cosine-topk", type=int, default=5, help="Premise beam for cosine baseline")
    parser.add_argument("--max-locals", type=int, default=8, help="Max local terms kept for arg beam")
    parser.add_argument("--max-args", type=int, default=4, help="Max positional args in heuristic beam")
    parser.add_argument("--max-arg-sequences", type=int, default=24, help="Max arg sequences per premise")
    parser.add_argument("--direction-mode", choices=["both", "infer", "tier-default"], default="both", help="Direction policy for heuristic tactics")
    parser.add_argument("--parallel", type=int, default=1, help="Shard into N worker subprocesses")
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
