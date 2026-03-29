from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import build_ducky_programs, execute_ducky_program
from src.nav_contracts import LeanFeedback, TacticResult


class _SmokeLean:
    def __init__(self, transitions: dict[tuple[str, str], TacticResult]) -> None:
        self.transitions = transitions

    def try_tactic(self, goal: str, tactic: str) -> TacticResult:
        return self.transitions.get(
            (goal, tactic),
            TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message="miss",
                feedback=LeanFeedback(stage="tactic_exec", category="other", messages=[], raw_error="miss"),
            ),
        )


def _success(new_goals: list[str]) -> TacticResult:
    return TacticResult(
        success=True,
        tactic="",
        premises=[],
        new_goals=new_goals,
        feedback=LeanFeedback.success(),
    )


def _case_rows() -> list[tuple[str, dict[str, object], dict[tuple[str, str], TacticResult]]]:
    return [
        (
            "eqsat_close",
            {
                "theorem_id": "Smoke.eqsat",
                "last_goal": "h : a = b\n⊢ f b = f a",
                "last_goal_bucket": "equality",
                "reasoning_gap_family": "local_eq_close",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 5,
            },
            {
                ("h : a = b\n⊢ f b = f a", "rw [← h]"): _success(["f a = f a"]),
                ("h : a = b\n⊢ f b = f a", "rw [h]"): _success(["f b = f b"]),
                ("f a = f a", "rfl"): _success([]),
                ("f b = f b", "rfl"): _success([]),
            },
        ),
        (
            "transport_close",
            {
                "theorem_id": "Smoke.transport",
                "last_goal": "hiff : P ↔ Q\nhp : P\n⊢ Q",
                "last_goal_bucket": "other",
                "reasoning_gap_family": "forward_context_close",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 5,
            },
            {
                ("hiff : P ↔ Q\nhp : P\n⊢ Q", "exact (hiff.mp hp)"): _success([]),
            },
        ),
        (
            "witness_close",
            {
                "theorem_id": "Smoke.witness",
                "last_goal": "hp : P a\n⊢ ∃ x, P x",
                "last_goal_bucket": "exists",
                "reasoning_gap_family": "witness_construction_close",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 5,
            },
            {
                ("hp : P a\n⊢ ∃ x, P x", "exact ⟨a, hp⟩"): _success([]),
            },
        ),
        (
            "relational_close",
            {
                "theorem_id": "Smoke.relational",
                "last_goal": "ha : x ∈ A\nhb : x ∈ B\n⊢ x ∈ A ∩ B",
                "last_goal_bucket": "membership",
                "reasoning_gap_family": "membership_close",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 1,
                "goals_remaining": 1,
                "attempts": 5,
            },
            {
                ("ha : x ∈ A\nhb : x ∈ B\n⊢ x ∈ A ∩ B", "exact ⟨ha, hb⟩"): _success([]),
            },
        ),
        (
            "arith_progress",
            {
                "theorem_id": "Smoke.arith",
                "last_goal": "⊢ |n| < 1",
                "last_goal_bucket": "inequality",
                "reasoning_gap_family": "local_ineq_close",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 0,
                "goals_remaining": 1,
                "attempts": 2,
            },
            {
                ("⊢ |n| < 1", "rw [Int.abs_lt_one_iff]"): _success(["n = 0"]),
                ("⊢ |n| < 1", "rw [Int.abs_lt_one_iff]; omega"): _success([]),
            },
        ),
    ]


def run_smoke() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    total_closed = 0
    total_progressed = 0
    for case_id, row, transitions in _case_rows():
        capsule = build_goal_capsule(row)
        goal = str(row["last_goal"])
        programs = build_ducky_programs(
            capsule,
            theorem_id=str(row["theorem_id"]),
            goal_text=goal,
            max_programs=24,
        )
        lean = _SmokeLean(transitions)
        runs = [execute_ducky_program(lean, goal, program) for program in programs]
        closed = next((run for run in runs if run.closed), None)
        progressed = next((run for run in runs if run.progressed), None)
        if closed is not None:
            total_closed += 1
        if progressed is not None:
            total_progressed += 1
        cases.append(
            {
                "case_id": case_id,
                "backend_preferences": list(capsule.backend_preferences),
                "program_count": len(programs),
                "closed": closed is not None,
                "progressed": progressed is not None,
                "winning_program": closed.to_dict() if closed is not None else (progressed.to_dict() if progressed is not None else None),
            }
        )
    return {
        "total_cases": len(cases),
        "closed_cases": total_closed,
        "progressed_cases": total_progressed,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a light smoke test over Dr. Ducky backend runtimes.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_smoke()
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
