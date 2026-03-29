from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sqlite3
import sys
from pathlib import Path
from typing import Any

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import run_ducky_on_row, summarize_ducky_execution
from src.lean_interface import LeanConfig, LeanKernel

LOCAL_BUCKETS = {
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
}
TARGETED_THEOREMS = {
    "Batteries.UnionFind.rootD_parent",
    "ModularGroup.tendsto_normSq_coprime_pair",
    "ModularGroup.smul_eq_lcRow0_add",
    "ModularGroup.eq_zero_of_mem_fdo_of_T_zpow_mem_fdo",
}


class RowTimeoutError(TimeoutError):
    pass


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row.get("residual_bucket", "")),
        -int(row.get("goals_closed", 0) or 0),
        int(row.get("goals_remaining", 0) or 0),
        int(row.get("attempts", 0) or 0),
        str(row.get("theorem_id", "") or ""),
    )


def _round_robin_rows(by_bucket: dict[str, list[dict]], limit: int) -> list[dict]:
    selected: list[dict] = []
    bucket_order = [bucket for bucket in sorted(by_bucket) if by_bucket[bucket]]
    while bucket_order and (limit <= 0 or len(selected) < limit):
        next_order: list[str] = []
        for bucket in bucket_order:
            bucket_rows = by_bucket[bucket]
            if not bucket_rows:
                continue
            selected.append(bucket_rows.pop(0))
            if limit > 0 and len(selected) >= limit:
                return selected
            if bucket_rows:
                next_order.append(bucket)
        bucket_order = next_order
    return selected


def _load_rows(path: Path, residual_buckets: set[str], limit: int) -> list[dict]:
    return _load_rows_with_filters(path, residual_buckets=residual_buckets, goal_buckets=set(), limit=limit)


def _load_rows_with_filters(
    path: Path,
    *,
    residual_buckets: set[str],
    goal_buckets: set[str],
    limit: int,
) -> list[dict]:
    rows: list[dict] = []
    targeted: list[dict] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            bucket = str(row.get("residual_bucket", "") or "")
            if bucket not in residual_buckets:
                continue
            goal_bucket = str(row.get("last_goal_bucket", "") or "")
            if goal_buckets and goal_bucket not in goal_buckets:
                continue
            if str(row.get("theorem_id", "") or "") in TARGETED_THEOREMS:
                targeted.append(row)
            else:
                rows.append(row)
    rows.sort(key=_row_sort_key)
    targeted.sort(key=_row_sort_key)
    selected = list(targeted[:limit]) if limit > 0 else list(targeted)
    if limit > 0:
        remaining = max(limit - len(selected), 0)
        if remaining <= 0:
            return selected
        if len(residual_buckets) <= 1:
            selected.extend(rows[:remaining])
            return selected
        by_bucket: dict[str, list[dict]] = {}
        for row in rows:
            by_bucket.setdefault(str(row.get("residual_bucket", "") or ""), []).append(row)
        selected.extend(_round_robin_rows(by_bucket, remaining))
    else:
        if len(residual_buckets) <= 1:
            selected.extend(rows)
        else:
            by_bucket: dict[str, list[dict]] = {}
            for row in rows:
                by_bucket.setdefault(str(row.get("residual_bucket", "") or ""), []).append(row)
            selected.extend(_round_robin_rows(by_bucket, 0))
    return selected


def _theorem_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        str(name): int(eid)
        for eid, name in conn.execute("SELECT id, name FROM entities")
    }


@contextlib.contextmanager
def _row_timeout(seconds: int) -> Any:
    if seconds <= 0:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: Any) -> None:
        raise RowTimeoutError(f"row exceeded timeout of {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _restart_lean(lean: LeanKernel) -> None:
    try:
        lean.close()
    except Exception:
        pass
    if getattr(lean, "_backend", "") == "pantograph":
        lean._ensure_server()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _progress_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _failure_payload(
    row: dict[str, Any],
    *,
    failure_category: str,
    error_message: str = "",
) -> dict[str, Any]:
    capsule = build_goal_capsule(row)
    target = str(row.get("last_goal", "") or row.get("goal_state", "") or row.get("theorem_statement", "") or "")
    payload = {
        "theorem_id": str(row.get("theorem_id", "") or ""),
        "started": False,
        "theorem_faithful": False,
        "start_goal_kind": "validation_guard",
        "file_path": str(row.get("file_path", "") or ""),
        "replay_tier": "guarded",
        "replay_failure_category": failure_category,
        "replay_failing_prefix_idx": -1,
        "residual_bucket": str(row.get("residual_bucket", "") or ""),
        "goal_bucket": str(row.get("last_goal_bucket", "") or ""),
        "specialist_targets": list(capsule.specialist_targets),
        "bank_priors": [prior.name for prior in capsule.bank_priors if not prior.suppressed],
        "programs_considered": 0,
        "closed": False,
        "progressed": False,
        "winning_program": None,
        "final_goal": "",
        "final_goal_bucket": "",
        "goals_after": [],
        "ledger_snapshot": capsule.ledger_seed.to_dict(),
        "engine_outcomes": [],
        "projector_outcomes": [],
        "tried_programs": (
            [{"program_id": "validation_guard", "first_failure_error": error_message, "progressed": False, "closed": False}]
            if error_message
            else []
        ),
        "validation_target": target,
    }
    return payload


def _write_progress_reports(
    *,
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    residual_buckets: set[str],
    goal_buckets: set[str],
    allowed_backend_families: set[str],
    allowed_engine_names: set[str],
    output_json: Path,
    closure_report_json: Path,
    projector_rows: list[dict[str, Any]],
) -> None:
    summary = summarize_ducky_execution([type("ResultProxy", (), payload)() for payload in results])
    summary["input_rows"] = len(rows)
    summary["completed_rows"] = len(results)
    summary["residual_buckets"] = sorted(residual_buckets)
    summary["goal_buckets"] = sorted(goal_buckets)
    summary["allowed_backend_families"] = sorted(allowed_backend_families)
    summary["allowed_engine_names"] = sorted(allowed_engine_names)
    summary["targeted_rows"] = [
        payload["theorem_id"]
        for payload in results
        if payload["theorem_id"] in TARGETED_THEOREMS
    ]
    _atomic_write_json(output_json, summary)
    closure_report = {
        "input_rows": len(rows),
        "completed_rows": len(results),
        "residual_buckets": sorted(residual_buckets),
        "goal_buckets": sorted(goal_buckets),
        "allowed_backend_families": sorted(allowed_backend_families),
        "allowed_engine_names": sorted(allowed_engine_names),
        "theorem_faithful_start_rate": round(summary["theorem_faithful_starts"] / max(summary["started"], 1), 4),
        "certificate_generation_count": summary.get("certificate_generation_count", 0),
        "projector_success_count": sum(1 for row in projector_rows if str(row.get("projector_status", "")) == "projected"),
        "projector_rejection_count": sum(1 for row in projector_rows if str(row.get("projector_status", "")) != "projected"),
        "lean_compile_proxy_count": sum(1 for payload in results for tried in payload.get("tried_programs", []) if tried.get("tactics_applied")),
        "honest_progress_count": summary["progressed"],
        "honest_closure_count": summary["closed"],
        "by_engine": summary.get("by_engine", {}),
        "by_certificate_shape": summary.get("by_certificate_shape", {}),
        "by_projector_status": summary.get("by_projector_status", {}),
        "by_goal_bucket": summary.get("by_goal_bucket", {}),
        "by_residual_bucket": summary.get("by_residual_bucket", {}),
    }
    _atomic_write_json(closure_report_json, closure_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/exp_som012_hard_eval_r2")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--engine-outcomes-jsonl", default="")
    parser.add_argument("--projector-outcomes-jsonl", default="")
    parser.add_argument("--closure-report-json", default="")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument(
        "--residual-bucket",
        action="append",
        default=[],
        help="Residual bucket(s) to include. Defaults to Dr. Ducky local buckets.",
    )
    parser.add_argument(
        "--goal-bucket",
        action="append",
        default=[],
        help="Goal bucket(s) to include. Defaults to all goal buckets inside the selected residual set.",
    )
    parser.add_argument(
        "--backend-family",
        action="append",
        default=[],
        help="Restrict validation to specific Dr. Ducky backend family/families.",
    )
    parser.add_argument(
        "--engine-name",
        action="append",
        default=[],
        help="Restrict validation to specific Dr. Ducky engine name(s).",
    )
    parser.add_argument("--max-programs", type=int, default=24)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--row-timeout-seconds", type=int, default=180)
    parser.add_argument("--restart-every", type=int, default=12)
    parser.add_argument(
        "--disable-tactic",
        action="append",
        default=["linarith", "nlinarith"],
        help="Disable specific tactics during validation replay. Defaults suppress unstable arithmetic solvers.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    details_path = run_dir / "details.jsonl"
    output_json = Path(args.output_json) if args.output_json else run_dir / "bundle/dr_ducky/executor_validation.json"
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else run_dir / "bundle/dr_ducky/executor_validation_rows.jsonl"
    engine_outcomes_jsonl = Path(args.engine_outcomes_jsonl) if args.engine_outcomes_jsonl else run_dir / "bundle/dr_ducky/dr_ducky_engine_outcomes.jsonl"
    projector_outcomes_jsonl = Path(args.projector_outcomes_jsonl) if args.projector_outcomes_jsonl else run_dir / "bundle/dr_ducky/dr_ducky_projector_outcomes.jsonl"
    closure_report_json = Path(args.closure_report_json) if args.closure_report_json else run_dir / "bundle/dr_ducky/dr_ducky_closure_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    engine_outcomes_jsonl.parent.mkdir(parents=True, exist_ok=True)
    projector_outcomes_jsonl.parent.mkdir(parents=True, exist_ok=True)
    closure_report_json.parent.mkdir(parents=True, exist_ok=True)

    residual_buckets = set(args.residual_bucket or LOCAL_BUCKETS)
    goal_buckets = {str(bucket).strip() for bucket in (args.goal_bucket or []) if str(bucket).strip()}
    allowed_backend_families = {str(name).strip() for name in (args.backend_family or []) if str(name).strip()}
    allowed_engine_names = {str(name).strip() for name in (args.engine_name or []) if str(name).strip()}
    rows = _load_rows_with_filters(
        details_path,
        residual_buckets=residual_buckets,
        goal_buckets=goal_buckets,
        limit=args.limit,
    )

    conn = sqlite3.connect(args.db)
    theorem_id_map = _theorem_id_map(conn)
    lean_cfg = LeanConfig(
        backend="pantograph",
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    lean = LeanKernel(lean_cfg)
    if lean._backend == "pantograph":
        lean._ensure_server()
    disabled_tactics = {str(tactic).strip().lower() for tactic in (args.disable_tactic or []) if str(tactic).strip()}

    results = []
    engine_rows = []
    projector_rows = []
    with (
        output_jsonl.open("w") as handle,
        engine_outcomes_jsonl.open("w") as engine_handle,
        projector_outcomes_jsonl.open("w") as projector_handle,
    ):
        for idx, row in enumerate(rows):
            theorem_id = str(row.get("theorem_id", "") or "")
            if args.restart_every > 0 and idx > 0 and idx % args.restart_every == 0:
                _progress_log(f"[{idx}/{len(rows)}] restarting Pantograph before next theorem")
                _restart_lean(lean)
            try:
                _progress_log(f"[{idx + 1}/{len(rows)}] start {theorem_id}")
                with _row_timeout(args.row_timeout_seconds):
                    result = run_ducky_on_row(
                        row,
                        lean=lean,
                        conn=conn,
                        theorem_id_map=theorem_id_map,
                        max_programs=args.max_programs,
                        max_rounds=args.max_rounds,
                        disabled_tactics=disabled_tactics,
                        allowed_backend_families=allowed_backend_families,
                        allowed_engine_names=allowed_engine_names,
                    )
                payload = result.to_dict()
            except RowTimeoutError as exc:
                _progress_log(f"[{idx + 1}/{len(rows)}] timeout {theorem_id}: {exc}")
                _restart_lean(lean)
                payload = _failure_payload(row, failure_category="row_timeout", error_message=str(exc))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                _progress_log(f"[{idx + 1}/{len(rows)}] exception {theorem_id}: {type(exc).__name__}: {exc}")
                _restart_lean(lean)
                payload = _failure_payload(
                    row,
                    failure_category=f"executor_exception:{type(exc).__name__}",
                    error_message=str(exc),
                )
            results.append(payload)
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            for outcome in payload.get("engine_outcomes", []):
                engine_row = {
                    "theorem_id": payload["theorem_id"],
                    "residual_bucket": payload["residual_bucket"],
                    **outcome,
                }
                engine_rows.append(engine_row)
                engine_handle.write(json.dumps(engine_row) + "\n")
            engine_handle.flush()
            for outcome in payload.get("projector_outcomes", []):
                projector_row = {
                    "theorem_id": payload["theorem_id"],
                    "residual_bucket": payload["residual_bucket"],
                    "goal_bucket": payload["goal_bucket"],
                    **outcome,
                }
                projector_rows.append(projector_row)
                projector_handle.write(json.dumps(projector_row) + "\n")
            projector_handle.flush()
            _write_progress_reports(
                results=results,
                rows=rows,
                residual_buckets=residual_buckets,
                goal_buckets=goal_buckets,
                allowed_backend_families=allowed_backend_families,
                allowed_engine_names=allowed_engine_names,
                output_json=output_json,
                closure_report_json=closure_report_json,
                projector_rows=projector_rows,
            )
            _progress_log(
                f"[{idx + 1}/{len(rows)}] done {theorem_id} "
                f"started={payload['started']} progressed={payload['progressed']} closed={payload['closed']}"
            )

    try:
        lean.close()
    except Exception:
        pass
    conn.close()


if __name__ == "__main__":
    main()
