from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_self_application_tactic(tactic: str, theorem_id: str) -> bool:
    text = (tactic or "").strip()
    theorem = (theorem_id or "").strip()
    if not text or not theorem:
        return False
    theorem_pat = re.escape(theorem)
    return bool(
        re.match(
            rf"^(?:exact|apply|refine)\s+\(?@?{theorem_pat}(?:\.[A-Za-z0-9_']+)?(?=$|[\s\)\]\}};,])",
            text,
        )
    )


def _mentions_self_application_tactic(tactic: str, theorem_id: str) -> bool:
    return is_self_application_tactic(tactic, theorem_id)


def detect_self_application(entry: dict[str, Any]) -> bool:
    theorem_id = str(entry.get("theorem_id", "") or "")
    if not theorem_id:
        return False
    provenance = [str(p) for p in entry.get("close_provenance", [])]
    if "self_application" in provenance:
        return True
    final_closer = str(entry.get("final_closer", "") or "")
    if _mentions_self_application_tactic(final_closer, theorem_id):
        return True
    for tactic in entry.get("tactics_used", []) or []:
        if isinstance(tactic, str) and _mentions_self_application_tactic(tactic, theorem_id):
            return True
    return False


def is_skipped_start(entry: dict[str, Any]) -> bool:
    """Return True when the theorem never entered the main proof-search loop."""
    if entry.get("close_lane") == "skipped":
        return True
    return (
        not bool(entry.get("success"))
        and _as_int(entry.get("attempts")) == 0
        and _as_int(entry.get("goals_closed")) == 0
    )


def classify_progress_band(entry: dict[str, Any]) -> str:
    if bool(entry.get("success")):
        return "solved"
    if is_skipped_start(entry):
        return "unstarted"

    goals_closed = _as_int(entry.get("goals_closed"))
    goals_remaining = _as_int(entry.get("goals_remaining"))
    if goals_closed == 0:
        return "no_progress"
    if goals_remaining == 1:
        return "near_miss"
    return "partial_progress"


def classify_residual_bucket(entry: dict[str, Any]) -> str:
    if bool(entry.get("success")):
        return "proved"
    if is_skipped_start(entry):
        return "skipped_start"

    goals_closed = _as_int(entry.get("goals_closed"))
    goals_remaining = _as_int(entry.get("goals_remaining"))

    if goals_remaining <= 1:
        return "single_goal_near_miss" if goals_closed > 0 else "single_goal_stall"
    if goals_remaining <= 5:
        return "multi_goal_small_progress" if goals_closed > 0 else "multi_goal_small_stall"
    return "multi_goal_large_progress" if goals_closed > 0 else "multi_goal_large_stall"


def classify_follow_on_stage(entry: dict[str, Any]) -> str:
    if bool(entry.get("success")):
        return "none"

    bucket = classify_residual_bucket(entry)
    if bucket == "skipped_start":
        return "compiler_specialist"
    if bucket in {
        "single_goal_near_miss",
        "single_goal_stall",
        "multi_goal_small_progress",
        "multi_goal_small_stall",
    }:
        return "hard_proof_solver"
    return "theorem_replanner"


def augment_result_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach theory-aligned residual decomposition fields to a benchmark entry."""
    out = dict(entry)
    self_application = detect_self_application(out)
    residual_bucket = classify_residual_bucket(out)
    progress_band = classify_progress_band(out)
    out["started"] = residual_bucket != "skipped_start"
    out["progress_band"] = progress_band
    out["residual_bucket"] = residual_bucket
    out["follow_on_stage"] = classify_follow_on_stage(out)
    out["self_application_detected"] = self_application
    out["honest_success"] = bool(out.get("success")) and not self_application
    if bool(out.get("success")) and self_application:
        out["success_category"] = "self_application"
    elif bool(out.get("success")):
        out["success_category"] = "raw_success"
    else:
        out["success_category"] = out.get("success_category", "failed") or "failed"
    return out


def summarize_residual_structure(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize benchmark results as typed residual buckets."""
    augmented = [augment_result_entry(entry) for entry in results]
    n = len(augmented)
    started = sum(1 for entry in augmented if entry["started"])
    failed = [entry for entry in augmented if not bool(entry.get("success"))]
    progressed = [
        entry
        for entry in failed
        if entry["progress_band"] in {"near_miss", "partial_progress"}
    ]

    by_residual_bucket = Counter(entry["residual_bucket"] for entry in augmented)
    by_progress_band = Counter(entry["progress_band"] for entry in augmented)
    by_follow_on_stage = Counter(entry["follow_on_stage"] for entry in augmented)

    return {
        "started_theorems": started,
        "skipped_start": n - started,
        "raw_success": sum(1 for entry in augmented if bool(entry.get("success"))),
        "honest_success": sum(1 for entry in augmented if bool(entry.get("honest_success"))),
        "self_application_successes": sum(
            1 for entry in augmented if bool(entry.get("self_application_detected"))
        ),
        "started_success_rate": round(
            sum(1 for entry in augmented if bool(entry.get("success"))) / max(started, 1),
            4,
        ),
        "started_honest_success_rate": round(
            sum(1 for entry in augmented if bool(entry.get("honest_success"))) / max(started, 1),
            4,
        ),
        "progressed_but_unsolved": len(progressed),
        "one_goal_left_failures": sum(1 for entry in failed if _as_int(entry.get("goals_remaining")) == 1),
        "by_residual_bucket": dict(by_residual_bucket),
        "by_progress_band": dict(by_progress_band),
        "by_follow_on_stage": dict(by_follow_on_stage),
        "failed_goal_metrics": {
            "mean_attempts": round(
                sum(_as_int(entry.get("attempts")) for entry in failed) / max(len(failed), 1),
                1,
            ),
            "mean_goals_closed": round(
                sum(_as_int(entry.get("goals_closed")) for entry in failed) / max(len(failed), 1),
                2,
            ),
            "mean_goals_remaining": round(
                sum(_as_int(entry.get("goals_remaining")) for entry in failed) / max(len(failed), 1),
                2,
            ),
        },
    }
