"""Rerun tractable MiniF2F gaps through the fixed pipeline.

Usage:
    python -m scripts.rerun_tractable_gaps \
        --run-dir runs/minif2f_test \
        --manifest data/minif2f_test.jsonl \
        --output runs/minif2f_rerun.jsonl

Reads the details.jsonl from the original run, identifies all non-proved
theorems with goals_closed >= 1 OR skipped_start, and reruns them through
the fixed pipeline with:
  - Track 1: Server resilience (load_sorry with full_declaration)
  - Track 2: Subgoal addressing (all_goals combinators)
  - Track 3: Witness injection (CAS-computed have-chains)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cas_engine import cas_solve_goal
from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import (
    COMPOSITIONAL_REWRITES,
    run_ducky_on_goal,
)
from src.lean_interface import LeanConfig, LeanKernel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_manifest(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            out[row["theorem_id"]] = row
    return out


def _load_gaps(run_dir: Path) -> list[dict]:
    """Load all non-proved theorems from the original run."""
    gaps = []
    details = run_dir / "details.jsonl"
    with open(details) as f:
        for line in f:
            row = json.loads(line)
            if row.get("success"):
                continue
            gaps.append(row)
    return gaps


def _try_tactic_battery(
    lean: LeanKernel, goal_text: str, tactics: list[str], timeout_s: int = 10
) -> str | None:
    """Try each tactic on goal_text, return the first that closes."""
    for tac in tactics:
        try:
            old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(timeout_s)
            r = lean.try_tactic(goal_text, tac)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            if r.success and not r.new_goals:
                return tac
        except Exception:
            signal.alarm(0)
    return None


def _build_tier0(goal_text: str) -> list[str]:
    """Tier 0: cheap closers that need no CAS."""
    return [
        "norm_num", "positivity", "decide", "omega", "ring", "simp",
        "linarith", "nlinarith",
        "constructor <;> norm_num",
        "constructor <;> omega",
        "constructor <;> linarith",
        "constructor <;> simp",
        "refine ⟨by norm_num, by norm_num⟩",
        "refine ⟨by omega, by omega⟩",
        "refine ⟨by linarith, by linarith⟩",
        "refine ⟨by norm_num, by norm_num, by norm_num⟩",
        "refine ⟨by omega, by omega, by omega⟩",
        "all_goals (first | norm_num | positivity | omega | decide | rfl)",
        "all_goals norm_num",
        "all_goals omega",
        "simp_all; norm_num",
        "simp_all; linarith",
        "simp_all; nlinarith",
        "simp_all; ring",
        "simp only [*] at *; norm_num",
        "simp only [*] at *; linarith",
        "simp only [*] at *; nlinarith",
        "simp only [*] at *; omega",
        "simp only [*] at *; ring",
        "native_decide",
    ]


def _build_cas_tactics(goal_text: str) -> list[str]:
    """Tier 1+2+3: CAS analysis + compositional rewrites + witness injection."""
    tactics = cas_solve_goal(goal_text)

    # Compositional rewrites
    for marker, rewrites in COMPOSITIONAL_REWRITES.items():
        if marker in goal_text:
            for _name, rtactics, _score in rewrites:
                tactics.append("; ".join(rtactics))

    # Hypothesis-unfolded variants
    for tac in list(tactics)[:6]:
        tactics.append(f"simp only [*] at *; {tac}")

    return tactics


def run_one_gap(
    lean: LeanKernel,
    thm: dict,
    gap: dict,
    *,
    per_theorem_timeout: int = 120,
) -> dict:
    """Try to close a single tractable gap."""
    tid = thm["theorem_id"]
    result = {
        "theorem_id": tid,
        "original_goals_closed": gap.get("goals_closed", 0),
        "original_goals_remaining": gap.get("goals_remaining", 0),
        "original_residual": gap.get("last_goal", "")[:100],
    }

    # Step 1: Create the goal via load_sorry (Track 1)
    full_decl = thm.get("full_declaration", "")
    opens = thm.get("opens") or []
    if not full_decl:
        result["status"] = "no_declaration"
        return result

    # Ensure server is alive
    if lean._server is None:
        try:
            lean._restart_server()
        except Exception:
            pass
    if lean._server is None:
        result["status"] = "server_dead"
        return result

    open_prefix = "\n".join(str(o) for o in opens) + "\n" if opens else ""
    try:
        targets = lean._server.load_sorry(f"{open_prefix}{full_decl}")
        if not targets or not targets[0].goal_state.goals:
            result["status"] = "load_sorry_no_goals"
            return result
        gs = targets[0].goal_state
        goal = lean.adopt(gs)  # the live goal the batteries act on
        goal_text = goal.rendering  # what the builders read: its hypotheses and target
        lean._server_contaminated = True
    except Exception as e:
        if hasattr(lean, "_is_crash_error") and lean._is_crash_error(e):
            lean._restart_server()
        result["status"] = f"load_sorry_error:{str(e)[:60]}"
        return result

    result["goal_started"] = True
    result["goal_text_len"] = len(goal_text)

    # Step 2: Try Tier 0 closers directly on the goal
    tier0 = _build_tier0(goal_text)
    closer = _try_tactic_battery(lean, goal, tier0)
    if closer:
        result["status"] = "closed"
        result["closed_by"] = f"tier0:{closer}"
        return result

    # Step 3: Try CAS tactics on the goal
    cas_tactics = _build_cas_tactics(goal_text)
    closer = _try_tactic_battery(lean, goal, cas_tactics, timeout_s=15)
    if closer:
        result["status"] = "closed"
        result["closed_by"] = f"cas:{closer}"
        return result

    # Step 4: Try intros then Tier 0 + CAS
    if "∀" in goal_text:
        try:
            r = lean.try_tactic(goal, "intros")
            if r.success and r.new_goals:
                intro_goal = r.new_goals[0]
                closer = _try_tactic_battery(lean, intro_goal, tier0)
                if closer:
                    result["status"] = "closed"
                    result["closed_by"] = f"intros;tier0:{closer}"
                    return result
                closer = _try_tactic_battery(lean, intro_goal, cas_tactics, timeout_s=15)
                if closer:
                    result["status"] = "closed"
                    result["closed_by"] = f"intros;cas:{closer}"
                    return result
        except Exception:
            pass

    # Step 5: Run Ducky on the goal
    try:
        row_for_ducky = {
            "theorem_id": tid,
            "last_goal": goal_text,
            "goal_state": goal_text,
            "full_declaration": full_decl,
            "opens": opens,
            "file_path": thm.get("file_path", ""),
            "lean_path": thm.get("lean_path", ""),
            "module": thm.get("module", ""),
        }
        capsule = build_goal_capsule(row_for_ducky)
        ducky_result = run_ducky_on_goal(
            goal,  # the live goal; the capsule row above reads its text
            theorem_id=tid,
            lean=lean,
            conn=None,
            accessible_theorem_id=None,
            capsule=capsule,
            row_overrides=row_for_ducky,
            max_programs=128,
            max_rounds=3,
            disabled_tactics=set(),
        )
        if ducky_result.closed:
            result["status"] = "closed"
            result["closed_by"] = f"ducky:{ducky_result.winning_program}"
            return result
        result["ducky_progressed"] = ducky_result.progressed
        result["ducky_programs"] = ducky_result.programs_considered

        # Step 6: CAS on post-Ducky residual
        if ducky_result.goals_after:
            for ducky_goal in ducky_result.goals_after[:3]:
                closer = _try_tactic_battery(lean, ducky_goal, tier0)
                if closer:
                    result["status"] = "closed_post_ducky"
                    result["closed_by"] = f"ducky+tier0:{closer}"
                    return result
    except Exception as e:
        result["ducky_error"] = str(e)[:60]

    result["status"] = "not_closed"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/minif2f_test")
    parser.add_argument("--manifest", default="data/minif2f_test.jsonl")
    parser.add_argument("--output", default="runs/minif2f_rerun.jsonl")
    parser.add_argument("--lean-project", default="data/lean_project")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = _load_manifest(Path(args.manifest))
    gaps = _load_gaps(run_dir)

    logger.info("Loaded %d gaps from %s", len(gaps), run_dir)

    cfg = LeanConfig(
        backend="pantograph",
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    lean = LeanKernel(cfg)

    output_path = Path(args.output)
    closed = 0
    started = 0
    total = 0

    with open(output_path, "w") as out:
        for gap in gaps:
            tid = gap.get("theorem_id", "")
            thm = manifest.get(tid)
            if not thm:
                continue
            total += 1
            name = tid.replace("minif2f_test_", "")[:35]

            try:
                lean._prepare_for_new_example()
            except Exception:
                pass

            result = run_one_gap(lean, thm, gap, per_theorem_timeout=args.timeout)
            out.write(json.dumps(result) + "\n")
            out.flush()

            status = result.get("status", "?")
            if result.get("goal_started"):
                started += 1
            if status in ("closed", "closed_post_ducky"):
                closed += 1
                logger.info("✓ %s: %s", name, result.get("closed_by", ""))
            else:
                logger.info("✗ %s: %s", name, status)

            # Periodic restart to prevent server accumulation
            if total % 20 == 0:
                try:
                    lean._restart_server()
                except Exception:
                    pass

    logger.info("=== RERUN COMPLETE ===")
    logger.info("Total gaps: %d", total)
    logger.info("Started: %d", started)
    logger.info("Closed: %d", closed)
    logger.info("Rate: %d/%d = %.0f%%", closed, total, 100 * closed / max(total, 1))

    lean.close()


if __name__ == "__main__":
    main()
