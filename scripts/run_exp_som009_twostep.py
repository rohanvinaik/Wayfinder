"""EXP-SOM-009: 2-Step Search on Single-Goal Residual.

For each of the 256 single-goal stalls from EXP-058, tries setup tactics
(cosine-ranked rw premises, normalization, structural) then closers (exact?,
simp_all, linarith, omega, ring) on the modified goal.

This is the "2-ply look-ahead" for proofs: (setup; closer) pairs.

Usage:
    python -m scripts.run_exp_som009_twostep \
        --config configs/wayfinder.yaml \
        --residual data/residual_exp058_started.jsonl \
        --db data/proof_network_v3.db \
        --output runs/exp_som009/ \
        --limit 256
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------------
# Setup tactic generators
# ---------------------------------------------------------------------------

# Normalization tactics that might transform a goal into closable form
_NORM_SETUPS = [
    "push_cast",
    "norm_cast",
    "simp only []",
    "ring_nf",
    "norm_num",
]

# Structural tactics that change goal shape
_STRUCTURAL_SETUPS = [
    "push_neg",
    "contrapose",
    "by_contra",
]

# Closers to try on modified goals
_CLOSERS = [
    "exact?",
    "simp_all",
    "linarith",
    "omega",
    "ring",
    "norm_num",
    "decide",
]

# Per-theorem timeout (seconds)
_THEOREM_TIMEOUT = 180


@dataclass
class TwoStepResult:
    """Result for one residual theorem."""
    theorem_id: str
    last_goal: str
    goal_started: bool = False
    start_failure: str = ""

    # Setup results
    setups_tried: int = 0
    setups_changed_goal: int = 0
    setup_tactic: str = ""  # winning setup (if any)
    modified_goal: str = ""  # goal after setup (if changed)

    # Closer results
    closers_tried: int = 0
    closer_tactic: str = ""  # winning closer (if any)
    closed: bool = False
    remaining_goals: int = -1

    # Cost
    lean_calls: int = 0
    elapsed_s: float = 0.0

    # Category
    setup_category: str = ""  # "rw", "norm", "structural"

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_id": self.theorem_id,
            "last_goal": self.last_goal,
            "goal_started": self.goal_started,
            "start_failure": self.start_failure,
            "setups_tried": self.setups_tried,
            "setups_changed_goal": self.setups_changed_goal,
            "setup_tactic": self.setup_tactic,
            "modified_goal": self.modified_goal,
            "closers_tried": self.closers_tried,
            "closer_tactic": self.closer_tactic,
            "closed": self.closed,
            "remaining_goals": self.remaining_goals,
            "lean_calls": self.lean_calls,
            "elapsed_s": self.elapsed_s,
            "setup_category": self.setup_category,
        }


def _cosine_rw_candidates(
    goal: str,
    theorem_id: str,
    conn: sqlite3.Connection,
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    sentence_encoder: Any,
    max_premises: int = 10,
) -> list[str]:
    """Get cosine-ranked rw candidates from accessible premises."""
    from src.proof_network import get_accessible_premises

    tid = name_to_id.get(theorem_id)
    if tid is None:
        return []

    premise_ids = get_accessible_premises(conn, tid)
    premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]
    if not premise_names:
        return []

    try:
        goal_emb = sentence_encoder.encode([goal], normalize_embeddings=True)
        premise_embs = sentence_encoder.encode(premise_names, normalize_embeddings=True)
        scores = (goal_emb @ premise_embs.T).flatten()
        top_idx = np.argsort(scores)[::-1][:max_premises]
        return [premise_names[i] for i in top_idx]
    except Exception as e:
        logger.warning("Cosine ranking failed for %s: %s", theorem_id, e)
        return []


def _try_setup_then_close(
    goal: str,
    setup_tactic: str,
    setup_category: str,
    lean: Any,
    result: TwoStepResult,
) -> bool:
    """Try a setup tactic, then if it changes the goal, try all closers.

    Returns True if the theorem is closed.
    """
    result.setups_tried += 1
    result.lean_calls += 1

    setup_result = lean.try_tactic(goal, setup_tactic)
    if not setup_result.success:
        return False

    # Setup must change the goal (progress) but not close it outright
    new_goals = setup_result.new_goals
    if not new_goals:
        # Setup closed the goal entirely — count as success
        result.closed = True
        result.setup_tactic = setup_tactic
        result.setup_category = setup_category
        result.closer_tactic = "(setup closed)"
        result.remaining_goals = 0
        result.setups_changed_goal += 1
        return True

    if len(new_goals) == 1 and new_goals[0] == goal:
        # Setup didn't change anything
        return False

    # Goal changed — try closers on the new goal(s)
    result.setups_changed_goal += 1

    # Only try closers if we still have exactly 1 goal
    if len(new_goals) != 1:
        return False

    modified_goal = new_goals[0]

    for closer in _CLOSERS:
        result.closers_tried += 1
        result.lean_calls += 1

        closer_result = lean.try_tactic(modified_goal, closer)
        if closer_result.success and not closer_result.new_goals:
            result.closed = True
            result.setup_tactic = setup_tactic
            result.setup_category = setup_category
            result.closer_tactic = closer
            result.modified_goal = modified_goal
            result.remaining_goals = 0
            return True

    return False


class _TimeoutError(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _TimeoutError("theorem timeout")


def process_theorem(
    theorem: dict[str, Any],
    lean: Any,
    conn: sqlite3.Connection,
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    sentence_encoder: Any | None,
    rw_premises: int = 10,
) -> TwoStepResult:
    """Run 2-step search on one residual theorem."""
    tid = theorem["theorem_id"]
    last_goal = theorem["last_goal"]

    result = TwoStepResult(theorem_id=tid, last_goal=last_goal)
    t0 = time.time()

    # --- Goal creation ---
    # Strategy: try to create goals from BOTH the last_goal (stalled state,
    # simpler) AND the initial theorem type. Try 2-step on each.
    # The last_goal is the actual target — it's the reduced form after partial
    # proof where exact? already failed. The initial goal is harder but may
    # also be closable via a different setup path.
    goals_to_try: list[str] = []

    # Priority 1: last_goal directly (the stalled goal — usually simpler)
    try:
        g = lean.goal_start(last_goal, theorem_name=tid)
        goals_to_try.append(g)
        result.goal_started = True
    except Exception:
        pass

    # Priority 2: initial theorem type via env_inspect
    try:
        info = lean._server.env_inspect(tid)
        theorem_type = info["type"]["pp"]
        g = lean.goal_start(theorem_type, theorem_name=tid)
        if g not in goals_to_try:
            goals_to_try.append(g)
        result.goal_started = True
    except Exception:
        pass

    if not goals_to_try:
        result.start_failure = "both goal_start paths failed"
        result.elapsed_s = time.time() - t0
        return result

    # --- Try 2-step on each goal ---
    for goal in goals_to_try:
        # Phase 1: Cosine-ranked rw setup
        if sentence_encoder is not None:
            premises = _cosine_rw_candidates(
                goal, tid, conn, name_to_id, id_to_name, sentence_encoder, rw_premises,
            )
            for premise in premises:
                for direction in [f"rw [{premise}]", f"rw [← {premise}]"]:
                    if _try_setup_then_close(goal, direction, "rw", lean, result):
                        result.elapsed_s = time.time() - t0
                        return result

        # Phase 2: Normalization setup
        for tactic in _NORM_SETUPS:
            if _try_setup_then_close(goal, tactic, "norm", lean, result):
                result.elapsed_s = time.time() - t0
                return result

        # Phase 3: Structural setup
        for tactic in _STRUCTURAL_SETUPS:
            if _try_setup_then_close(goal, tactic, "structural", lean, result):
                result.elapsed_s = time.time() - t0
                return result

    result.elapsed_s = time.time() - t0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", default="data/residual_exp058_started.jsonl")
    parser.add_argument("--residual-type", default="single_goal_stall",
                        choices=["single_goal_stall", "multi_goal_small", "all"],
                        help="Which residual type to target")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--output", default="runs/exp_som009")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rw-premises", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lean-project", default="data/lean_project/")
    args = parser.parse_args()

    # --- Load residual ---
    residual: list[dict[str, Any]] = []
    with open(args.residual) as f:
        for line in f:
            d = json.loads(line)
            rtype = d.get("residual_type", "")
            if args.residual_type == "all" or rtype == args.residual_type:
                residual.append(d)

    logger.info("Loaded %d residual theorems (type=%s)", len(residual), args.residual_type)
    if args.limit:
        residual = residual[: args.limit]

    # --- Resume support ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "twostep_results.jsonl"
    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add(d["theorem_id"])
        logger.info("Resuming: %d already done", len(done_ids))

    # --- DB setup ---
    conn = sqlite3.connect(args.db)
    id_to_name: dict[int, str] = {
        eid: name for eid, name in conn.execute("SELECT id, name FROM entities")
    }
    name_to_id: dict[str, int] = {v: k for k, v in id_to_name.items()}

    # --- Lean setup ---
    from src.lean_interface import LeanConfig, LeanKernel

    lean_cfg = LeanConfig(
        backend="pantograph",
        hammer_timeout=60,
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    lean = LeanKernel(lean_cfg)
    lean._ensure_server()
    logger.info("Lean server started")

    # --- Sentence encoder ---
    sentence_encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence encoder")
    except Exception as e:
        logger.warning("No sentence encoder: %s", e)

    # --- Run ---
    total = len(residual)
    closed_count = 0
    started_count = 0
    total_lean_calls = 0
    t_start = time.time()

    # Set up SIGALRM for per-theorem timeout
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    try:
        with open(results_path, "a") as out_f:
            for i, thm in enumerate(residual):
                tid = thm["theorem_id"]
                if tid in done_ids:
                    continue

                signal.alarm(_THEOREM_TIMEOUT)
                try:
                    r = process_theorem(
                        thm, lean, conn, name_to_id, id_to_name,
                        sentence_encoder, args.rw_premises,
                    )
                except _TimeoutError:
                    r = TwoStepResult(
                        theorem_id=tid,
                        last_goal=thm["last_goal"],
                        start_failure="timeout",
                    )
                except Exception as e:
                    r = TwoStepResult(
                        theorem_id=tid,
                        last_goal=thm["last_goal"],
                        start_failure=f"crash: {e!s:.200}",
                    )
                    # Restart server on crash
                    try:
                        lean._restart_server()
                    except Exception:
                        pass
                finally:
                    signal.alarm(0)

                out_f.write(json.dumps(r.to_dict()) + "\n")
                out_f.flush()

                if r.goal_started:
                    started_count += 1
                if r.closed:
                    closed_count += 1
                total_lean_calls += r.lean_calls

                if (i + 1) % 10 == 0 or r.closed:
                    elapsed = time.time() - t_start
                    logger.info(
                        "[%d/%d] closed=%d started=%d lean_calls=%d elapsed=%.0fs%s",
                        i + 1, total, closed_count, started_count,
                        total_lean_calls, elapsed,
                        f" ** CLOSED {tid} via {r.setup_tactic} + {r.closer_tactic}" if r.closed else "",
                    )
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    # --- Summary ---
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("EXP-SOM-009 2-Step Search Results")
    logger.info("=" * 60)
    logger.info("Single-goal stalls: %d", total)
    logger.info("Goal started:       %d (%.1f%%)", started_count, 100 * started_count / max(total, 1))
    logger.info("Closed (2-step):    %d (%.1f%% of stalls, %.1f%% of started)",
                closed_count,
                100 * closed_count / max(total, 1),
                100 * closed_count / max(started_count, 1))
    logger.info("Total Lean calls:   %d", total_lean_calls)
    logger.info("Elapsed:            %.0fs", elapsed)

    # Breakdown by setup category
    by_cat: dict[str, int] = {}
    by_closer: dict[str, int] = {}
    if results_path.exists():
        results = []
        with open(results_path) as f:
            for line in f:
                results.append(json.loads(line))

        for r in results:
            if r.get("closed"):
                cat = r.get("setup_category", "unknown")
                by_cat[cat] = by_cat.get(cat, 0) + 1
                cl = r.get("closer_tactic", "unknown")
                by_closer[cl] = by_closer.get(cl, 0) + 1

        logger.info("\nBy setup category:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", cat, n)
        logger.info("\nBy closer tactic:")
        for cl, n in sorted(by_closer.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", cl, n)

    summary = {
        "experiment": "EXP-SOM-009",
        "date": time.strftime("%Y-%m-%d"),
        "total_stalls": total,
        "started": started_count,
        "closed": closed_count,
        "closed_rate_of_stalls": closed_count / max(total, 1),
        "closed_rate_of_started": closed_count / max(started_count, 1),
        "total_lean_calls": total_lean_calls,
        "elapsed_s": elapsed,
        "by_setup_category": by_cat,
        "by_closer": by_closer,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
