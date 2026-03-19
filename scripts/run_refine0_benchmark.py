"""EXP-REFINE-043: refine step-0 component benchmark.

Two subsets from canonical_residual_eval.jsonl (step_index==0, family=='refine'):

  refine_named (54):  annotated_premise is the direct head of the refine call.
                      These are cosine-predictable (like apply).
  refine_anon  (146): ⟨⟩ constructor or premise internal to the tactic.
                      Oracle-only diagnostic tail.

Conditions (refine_named):
  oracle      — use canonical_action_ir verbatim
  cosine_top1 — top-1 premise × 6 suffix variants
  cosine_top5 — top-5 premises × 6 variants each (first accepted)

Conditions (refine_anon):
  oracle      — use canonical_action_ir verbatim (diagnostic only)

Uses goal_via_file_context() for Lean goal creation (same as apply0).

Usage:
    python -m scripts.run_refine0_benchmark \\
        --db data/proof_network_v3.db \\
        --eval data/canonical/canonical_residual_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

from src.lean_interface import LeanConfig, LeanKernel, ReplayResult
from src.proof_network import get_accessible_premises

try:
    from src.lean_interface import ServerCrashError  # type: ignore[attr-defined]
except ImportError:
    class ServerCrashError(Exception):  # type: ignore[no-redef]
        pass

# Suffix variants to try when applying a lemma via refine
_REFINE_VARIANTS = [
    "{lemma} ?_",
    "{lemma}.1 ?_",
    "{lemma}.2 ?_",
    "{lemma}.mp ?_",
    "{lemma}.mpr ?_",
    "{lemma} _ ?_",
]


# ---------------------------------------------------------------------------
# Data loading + classification
# ---------------------------------------------------------------------------


def load_refine_step0(eval_path: str) -> tuple[list[dict], list[dict]]:
    """Split step-0 refine examples into named/anon subsets."""
    named: list[dict] = []
    anon: list[dict] = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("step_index", 0) != 0 or ex.get("family") != "refine":
                continue
            prem = ex.get("annotated_premise", "")
            ir = ex.get("canonical_action_ir", "")
            tac = ex.get("tactic_text", "")
            body = re.sub(r"^refine\s*", "", (ir or tac).strip())
            if prem and (body.startswith(prem) or body.startswith(f"({prem}")):
                named.append(ex)
            else:
                anon.append(ex)
    return named, anon


# ---------------------------------------------------------------------------
# Helpers (mirrors run_apply0_benchmark)
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


def try_refine_variants(
    kernel: LeanKernel,
    goal_state: str,
    lemma: str,
    goal_id: int = 0,
) -> tuple[bool, bool, str, bool]:
    """Try all refine suffix variants for a lemma. Return first accepted result."""
    for tmpl in _REFINE_VARIANTS:
        tac = f"refine {tmpl.format(lemma=lemma)}"
        acc, closed, err, crash = try_tactic_safe(kernel, goal_state, tac, goal_id=goal_id)
        if crash:
            return False, False, err, True
        if acc or closed:
            return acc, closed, err, False
    return False, False, "", False


# ---------------------------------------------------------------------------
# Per-example result
# ---------------------------------------------------------------------------


@dataclass
class RefineExampleResult:
    theorem_full_name: str
    file_path: str
    subset: str          # "named" | "anon"
    annotated_premise: str
    canonical_action_ir: str

    goal_started: bool = False
    tier_used: str = ""
    failure_category: str = ""
    crash_retries: int = 0

    oracle_accepted: bool = False
    oracle_closed: bool = False
    oracle_error: str = ""

    # Named subset only
    cosine_top1_accepted: bool = False
    cosine_top1_closed: bool = False
    cosine_top1_rank: int = -1

    cosine_top5_accepted: bool = False
    cosine_top5_closed: bool = False
    cosine_top5_rank: int = -1

    gold_in_scope: bool = False
    gold_scope_tier: str = ""
    n_accessible_premises: int = 0
    n_scope_symbols: int = 0

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Per-example runner
# ---------------------------------------------------------------------------


def run_one_example(
    example: dict,
    subset: str,
    kernel: LeanKernel,
    conn: sqlite3.Connection,
    id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    encoder: object | None = None,
    project_root: str = "",
) -> RefineExampleResult:
    t0 = time.time()
    res = RefineExampleResult(
        theorem_full_name=example["theorem_full_name"],
        file_path=example.get("file_path", ""),
        subset=subset,
        annotated_premise=example.get("annotated_premise", ""),
        canonical_action_ir=example.get("canonical_action_ir", ""),
    )

    # Goal creation via file context
    prefix_tactics = example.get("prefix_tactics", [])
    replay: ReplayResult = kernel.goal_via_file_context(
        theorem_full_name=res.theorem_full_name,
        file_path=res.file_path,
        prefix_tactics=prefix_tactics,
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

    # Oracle condition
    oracle_tactic = res.canonical_action_ir or example.get("tactic_text", "")
    if oracle_tactic:
        acc, closed, err, crash = try_tactic_safe(kernel, goal_str, oracle_tactic, goal_id=goal_id)
        if crash:
            res.crash_retries += 1
            res.oracle_error = err
        else:
            res.oracle_accepted = acc
            res.oracle_closed = closed
            res.oracle_error = err

    # Cosine conditions — named subset only
    if subset == "named" and encoder is not None:
        premise_names = get_premise_names(conn, res.theorem_full_name, id_to_name, name_to_id)
        res.n_accessible_premises = len(premise_names)
        res.n_scope_symbols = len(premise_names)

        gold_prem = res.annotated_premise
        if gold_prem and gold_prem in premise_names:
            res.gold_in_scope = True
            res.gold_scope_tier = "in_scope"

        ranked = cosine_rank_symbols(goal_str, premise_names, encoder)

        # Cosine top-1
        if ranked:
            _, top1 = ranked[0]
            acc, closed, _, crash = try_refine_variants(kernel, goal_str, top1, goal_id=goal_id)
            if crash:
                res.crash_retries += 1
            else:
                res.cosine_top1_accepted = acc
                res.cosine_top1_closed = closed
                res.cosine_top1_rank = next(
                    (i for i, (_, p) in enumerate(ranked) if p == gold_prem), -1
                )

        # Cosine top-5
        for rank, (_, prem) in enumerate(ranked[:5]):
            acc, closed, _, crash = try_refine_variants(kernel, goal_str, prem, goal_id=goal_id)
            if crash:
                res.crash_retries += 1
                break
            if acc or closed:
                res.cosine_top5_accepted = acc
                res.cosine_top5_closed = closed
                res.cosine_top5_rank = rank
                break

    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(named: list[RefineExampleResult], anon: list[RefineExampleResult]) -> None:
    def _section(results: list[RefineExampleResult], label: str) -> None:
        n = len(results)
        started = [r for r in results if r.goal_started]
        ns = len(started)
        if n == 0:
            print(f"\n  {label}: 0 examples")
            return

        oracle_acc = sum(r.oracle_accepted for r in started)
        oracle_cls = sum(r.oracle_closed for r in started)
        print(f"\n  {label}  (n={n}, started={ns}/{n}  {100*ns/n:.1f}%)")
        print(f"    oracle  accepted|started: {oracle_acc}/{ns}  ({100*oracle_acc/max(ns,1):.1f}%)")
        print(f"    oracle  closed|started:   {oracle_cls}/{ns}  ({100*oracle_cls/max(ns,1):.1f}%)")

        if label == "refine_named":
            cos1 = sum(r.cosine_top1_accepted for r in started)
            cos5 = sum(r.cosine_top5_accepted for r in started)
            in_scope = sum(r.gold_in_scope for r in started)
            print(f"    cosine_1  accepted|started: {cos1}/{ns}  ({100*cos1/max(ns,1):.1f}%)")
            print(f"    cosine_5  accepted|started: {cos5}/{ns}  ({100*cos5/max(ns,1):.1f}%)")
            print(f"    gold_in_scope:              {in_scope}/{ns}  ({100*in_scope/max(ns,1):.1f}%)")
            sizes = [r.n_accessible_premises for r in started if r.n_accessible_premises > 0]
            if sizes:
                print(f"    mean accessible premises:  {sum(sizes)/len(sizes):.0f}")

        fail_cats = Counter(r.failure_category for r in results if r.failure_category)
        if fail_cats:
            print(f"    failures: {dict(fail_cats.most_common(5))}")

    print("\n" + "=" * 68)
    print("EXP-REFINE-043: refine step-0 component benchmark")
    print("=" * 68)
    _section(named, "refine_named")
    _section(anon, "refine_anon")

    all_started = [r for r in named + anon if r.goal_started]
    ns = len(all_started)
    oracle_acc = sum(r.oracle_accepted for r in all_started)
    print(f"\n  OVERALL  (started={ns})")
    print(f"    oracle  accepted|started: {oracle_acc}/{ns}  ({100*oracle_acc/max(ns,1):.1f}%)")

    named_started = [r for r in named if r.goal_started]
    nns = len(named_started)
    named_oracle = sum(r.oracle_accepted for r in named_started)
    pct = 100 * named_oracle / max(nns, 1)
    gate = "PASS" if pct >= 50.0 else "FAIL"
    print(f"\n  Deployment gate (oracle named ≥ 50%): {pct:.1f}%  [{gate}]")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-REFINE-043: refine step-0 benchmark")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--eval", default="data/canonical/canonical_residual_eval.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--named-only", action="store_true")
    parser.add_argument("--output", default="runs/refine0_results.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    named_raw, anon_raw = load_refine_step0(args.eval)
    logger.info("refine_named: %d, refine_anon: %d", len(named_raw), len(anon_raw))

    if args.limit > 0:
        named_raw = named_raw[: args.limit]
        anon_raw = anon_raw[: max(args.limit // 3, 5)]

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded MiniLM encoder")
    except ImportError:
        logger.warning("sentence-transformers not available — cosine conditions skipped")

    kernel = LeanKernel(LeanConfig(
        backend=args.backend,
        project_root=args.lean_project,
        imports=["Mathlib"],
    ))
    kernel._ensure_server()
    logger.info("Lean server started")

    conn = sqlite3.connect(args.db)
    id_to_name, name_to_id = load_entity_maps(conn)

    named_results: list[RefineExampleResult] = []
    for i, ex in enumerate(named_raw):
        logger.info("[named %d/%d] %s", i + 1, len(named_raw), ex["theorem_full_name"])
        r = run_one_example(
            ex, "named", kernel, conn, id_to_name, name_to_id, encoder, args.lean_project
        )
        named_results.append(r)

    anon_results: list[RefineExampleResult] = []
    if not args.named_only:
        for i, ex in enumerate(anon_raw):
            logger.info("[anon %d/%d] %s", i + 1, len(anon_raw), ex["theorem_full_name"])
            r = run_one_example(
                ex, "anon", kernel, conn, id_to_name, name_to_id, encoder, args.lean_project
            )
            anon_results.append(r)

    kernel.close()
    conn.close()

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in named_results + anon_results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Written to %s", args.output)

    print_report(named_results, anon_results)


if __name__ == "__main__":
    main()
