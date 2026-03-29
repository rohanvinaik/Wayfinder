from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark import _build_search_components, _build_theorem_id_map
from scripts.run_exp_som012_hard_collect import _load_sentence_encoder
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text
from src.hardtail_bridge import HardtailBridgeResult, run_hardtail_bridge_on_row
from src.second_order_controller import load_second_order_packet_index
from src.second_order_som_model import load_learned_second_order_runtime

logger = logging.getLogger(__name__)

LOCAL_BUCKETS = {
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
}

SYMBOLIC_CLOSER_TOKENS = {
    "solve_by_elim",
    "apply?",
    "exact?",
    "omega",
    "aesop",
    "decide",
    "simp",
    "simpa",
    "assumption",
    "trivial",
    "constructor",
    "tauto",
}

TRACTABLE_BUCKET_ORDER = [
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
]


class RowTimeoutError(TimeoutError):
    pass


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row.get("residual_bucket", "")),
        -int(row.get("goals_closed", 0) or 0),
        int(row.get("goals_remaining", 0) or 0),
        int(row.get("attempts", 0) or 0),
        str(row.get("theorem_id", "") or ""),
    )


def _round_robin_rows(
    by_bucket: dict[str, list[dict[str, Any]]],
    limit: int,
    *,
    bucket_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if bucket_order is None:
        active_order = [bucket for bucket in sorted(by_bucket) if by_bucket[bucket]]
    else:
        seen = set()
        active_order = []
        for bucket in bucket_order:
            if bucket in seen:
                continue
            seen.add(bucket)
            if by_bucket.get(bucket):
                active_order.append(bucket)
        for bucket in sorted(by_bucket):
            if bucket in seen:
                continue
            if by_bucket[bucket]:
                active_order.append(bucket)
    while active_order and (limit <= 0 or len(selected) < limit):
        next_order: list[str] = []
        for bucket in active_order:
            rows = by_bucket[bucket]
            if not rows:
                continue
            selected.append(rows.pop(0))
            if limit > 0 and len(selected) >= limit:
                return selected
            if rows:
                next_order.append(bucket)
        active_order = next_order
    return selected


def _row_index(path: Path, residual_buckets: set[str]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = str(row.get("theorem_id", "") or "")
            if not theorem_id or theorem_id in indexed:
                continue
            bucket = str(row.get("residual_bucket", "") or "")
            if bucket in residual_buckets:
                indexed[theorem_id] = row
    return indexed


def _validated_seed_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    bucket = str(row.get("residual_bucket", "") or "")
    goals_after = list(row.get("goals_after") or [])
    final_goal = sanitize_goal_text(str(row.get("final_goal", "") or ""))
    final_goal_bucket = str(row.get("final_goal_bucket", "") or "")
    bucket_rank = TRACTABLE_BUCKET_ORDER.index(bucket) if bucket in TRACTABLE_BUCKET_ORDER else len(TRACTABLE_BUCKET_ORDER)
    close_rank = 0 if final_goal_bucket in {"equality", "inequality", "false", "atomic_prop"} else 1
    progress_rank = 0 if row.get("progressed") else 1
    return (
        bucket_rank,
        progress_rank,
        len(goals_after),
        close_rank * 1000 + len(final_goal),
        str(row.get("theorem_id", "") or ""),
    )


def _load_validated_seed_theorem_ids(
    path: Path,
    residual_buckets: set[str],
    limit: int,
    theorem_ids: list[str] | None = None,
) -> list[str]:
    requested = {str(name or "").strip() for name in (theorem_ids or []) if str(name or "").strip()}
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in TRACTABLE_BUCKET_ORDER}
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = str(row.get("theorem_id", "") or "")
            if requested and theorem_id not in requested:
                continue
            bucket = str(row.get("residual_bucket", "") or "")
            if bucket not in residual_buckets:
                continue
            if not row.get("started"):
                continue
            if not row.get("progressed"):
                continue
            if row.get("closed"):
                continue
            by_bucket.setdefault(bucket, []).append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=_validated_seed_sort_key)
    selected_rows = _round_robin_rows(by_bucket, limit, bucket_order=TRACTABLE_BUCKET_ORDER)
    return [str(row.get("theorem_id", "") or "") for row in selected_rows if str(row.get("theorem_id", "") or "")]


def _load_rows(
    path: Path,
    residual_buckets: set[str],
    limit: int,
    theorem_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    theorem_ids = {str(name or "").strip() for name in (theorem_ids or set()) if str(name or "").strip()}
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = str(row.get("theorem_id", "") or "")
            if theorem_ids and theorem_id not in theorem_ids:
                continue
            bucket = str(row.get("residual_bucket", "") or "")
            if bucket in residual_buckets:
                rows.append(row)
    rows.sort(key=_row_sort_key)
    if limit <= 0:
        by_bucket: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_bucket.setdefault(str(row.get("residual_bucket", "") or ""), []).append(row)
        return _round_robin_rows(by_bucket, 0, bucket_order=TRACTABLE_BUCKET_ORDER)
    by_bucket = {}
    for row in rows:
        by_bucket.setdefault(str(row.get("residual_bucket", "") or ""), []).append(row)
    return _round_robin_rows(by_bucket, limit, bucket_order=TRACTABLE_BUCKET_ORDER)


def _select_rows(
    *,
    rows_path: Path,
    residual_buckets: set[str],
    limit: int,
    theorem_ids: list[str] | None = None,
    selection_source: str = "validated_progress",
    validated_seed_path: Path | None = None,
    allow_unvalidated_backfill: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    explicit_theorem_ids = [str(name or "").strip() for name in (theorem_ids or []) if str(name or "").strip()]
    if explicit_theorem_ids:
        details_index = _row_index(rows_path, residual_buckets)
        selected = [details_index[name] for name in explicit_theorem_ids if name in details_index]
        return selected[: limit or None], "explicit_theorem_ids"

    if selection_source == "validated_progress" and validated_seed_path and validated_seed_path.exists():
        details_index = _row_index(rows_path, residual_buckets)
        selected_ids = _load_validated_seed_theorem_ids(
            validated_seed_path,
            residual_buckets,
            limit,
        )
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for theorem_id in selected_ids:
            row = details_index.get(theorem_id)
            if row is None or theorem_id in seen:
                continue
            selected.append(row)
            seen.add(theorem_id)
            if limit > 0 and len(selected) >= limit:
                return selected, "validated_progress"
        if allow_unvalidated_backfill and (limit <= 0 or len(selected) < limit):
            remaining = _load_rows(rows_path, residual_buckets, 0)
            for row in remaining:
                theorem_id = str(row.get("theorem_id", "") or "")
                if not theorem_id or theorem_id in seen:
                    continue
                selected.append(row)
                seen.add(theorem_id)
                if limit > 0 and len(selected) >= limit:
                    break
        return selected[: limit or None], "validated_progress_backfill" if allow_unvalidated_backfill else "validated_progress"

    return _load_rows(rows_path, residual_buckets, limit, theorem_ids=None), "residual_round_robin"


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


def _restart_lean(lean: Any) -> None:
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


def _normalize_tactic_name(tactic: str) -> str:
    token = str(tactic or "").strip()
    if not token:
        return ""
    first = token.split()[0]
    if first in {"exact", "apply", "refine"}:
        return first
    return first


def _symbolic_tactic_counts(payload: dict[str, Any] | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not payload:
        return counts
    for tactic in payload.get("tactics_used") or []:
        name = _normalize_tactic_name(str(tactic or ""))
        if name in SYMBOLIC_CLOSER_TOKENS or name in {"exact", "apply", "refine"}:
            counts[name] += 1
    return counts


def _search_payload_symbolic_close(payload: dict[str, Any] | None) -> str:
    if not payload or not payload.get("success"):
        return ""
    symbolic_counts = _symbolic_tactic_counts(payload)
    if not symbolic_counts:
        return ""
    return symbolic_counts.most_common(1)[0][0]


def _bridge_summary(results: list[dict[str, Any]], rarified_packets: list[dict[str, Any]]) -> dict[str, Any]:
    closed_by = Counter(str(row.get("closed_by", "") or "") for row in results if row.get("closed"))
    residuals = Counter(str(row.get("residual_bucket", "") or "") for row in results)
    targets = Counter(str(row.get("controller_decision", {}).get("rarified_target", "") or "") for row in results)
    controller_modes = Counter(str(row.get("controller_decision", {}).get("controller_mode", "") or "") for row in results)
    tractability = Counter(str(packet.get("tractability_class", "") or "") for packet in rarified_packets)
    first_order_symbolic_close_tactics = Counter()
    first_order_symbolic_progress_tactics = Counter()
    post_ducky_symbolic_close_tactics = Counter()
    post_ducky_symbolic_progress_tactics = Counter()
    first_ducky_progress = sum(1 for row in results if (row.get("ducky_pass_1") or {}).get("progressed"))
    first_ducky_close = sum(1 for row in results if (row.get("ducky_pass_1") or {}).get("closed"))
    search_progress = sum(
        1
        for row in results
        if (row.get("first_order_search") or {}).get("success")
        or int((row.get("first_order_search") or {}).get("progress_steps", 0) or 0) > 0
    )
    search_close = sum(1 for row in results if (row.get("first_order_search") or {}).get("success"))
    for row in results:
        search_payload = row.get("first_order_search") or {}
        search_symbolic = _symbolic_tactic_counts(search_payload)
        if search_symbolic:
            first_order_symbolic_progress_tactics.update(search_symbolic)
            if search_payload.get("success"):
                first_order_symbolic_close_tactics.update(search_symbolic)
    second_ducky_progress = sum(1 for row in results if (row.get("ducky_pass_2") or {}).get("progressed"))
    second_ducky_close = sum(1 for row in results if (row.get("ducky_pass_2") or {}).get("closed"))
    symbolic2_progress = sum(
        1
        for row in results
        if (row.get("symbolic_close_pass_2") or {}).get("success")
        or int((row.get("symbolic_close_pass_2") or {}).get("progress_steps", 0) or 0) > 0
    )
    symbolic2_close = sum(1 for row in results if (row.get("symbolic_close_pass_2") or {}).get("success"))
    for row in results:
        symbolic_payload = row.get("symbolic_close_pass_2") or {}
        symbolic_counts = _symbolic_tactic_counts(symbolic_payload)
        if symbolic_counts:
            post_ducky_symbolic_progress_tactics.update(symbolic_counts)
            if symbolic_payload.get("success"):
                post_ducky_symbolic_close_tactics.update(symbolic_counts)
    final_goal_counts = [int(row.get("final_goal_count", 0) or 0) for row in results]
    return {
        "total_rows": len(results),
        "started": sum(1 for row in results if row.get("started")),
        "theorem_faithful_starts": sum(1 for row in results if row.get("theorem_faithful")),
        "progressed": sum(1 for row in results if row.get("progressed")),
        "closed": sum(1 for row in results if row.get("closed")),
        "closed_by": dict(sorted(closed_by.items(), key=lambda item: (-item[1], item[0]))),
        "by_controller_mode": dict(sorted(controller_modes.items(), key=lambda item: (-item[1], item[0]))),
        "ducky_pass_1_progress": first_ducky_progress,
        "ducky_pass_1_close": first_ducky_close,
        "first_order_search_progress": search_progress,
        "first_order_search_close": search_close,
        "first_order_symbolic_progress": sum(1 for row in results if _symbolic_tactic_counts(row.get("first_order_search") or {})),
        "first_order_symbolic_close": sum(1 for row in results if _search_payload_symbolic_close(row.get("first_order_search") or {})),
        "first_order_symbolic_progress_by_tactic": dict(sorted(first_order_symbolic_progress_tactics.items(), key=lambda item: (-item[1], item[0]))),
        "first_order_symbolic_close_by_tactic": dict(sorted(first_order_symbolic_close_tactics.items(), key=lambda item: (-item[1], item[0]))),
        "ducky_pass_2_progress": second_ducky_progress,
        "ducky_pass_2_close": second_ducky_close,
        "symbolic_close_pass_2_progress": symbolic2_progress,
        "symbolic_close_pass_2_close": symbolic2_close,
        "symbolic_close_pass_2_progress_by_tactic": dict(sorted(post_ducky_symbolic_progress_tactics.items(), key=lambda item: (-item[1], item[0]))),
        "symbolic_close_pass_2_theorem_progress": sum(1 for row in results if _symbolic_tactic_counts(row.get("symbolic_close_pass_2") or {})),
        "symbolic_close_pass_2_theorem_close": sum(1 for row in results if (row.get("symbolic_close_pass_2") or {}).get("success")),
        "symbolic_close_pass_2_close_by_tactic": dict(sorted(post_ducky_symbolic_close_tactics.items(), key=lambda item: (-item[1], item[0]))),
        "rarified_gap_packets": len(rarified_packets),
        "by_residual_bucket": dict(sorted(residuals.items(), key=lambda item: (-item[1], item[0]))),
        "by_rarified_target": dict(sorted(targets.items(), key=lambda item: (-item[1], item[0]))),
        "by_tractability_class": dict(sorted(tractability.items(), key=lambda item: (-item[1], item[0]))),
        "avg_final_goal_count": round(sum(final_goal_counts) / max(len(final_goal_counts), 1), 3),
        "goal_bucket_counts": dict(
            sorted(
                Counter(
                    classify_goal_bucket(goal)
                    for row in results
                    for goal in (row.get("final_goals") or [])
                ).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def _failure_result(row: dict[str, Any], *, error_category: str, error_message: str) -> HardtailBridgeResult:
    theorem_id = str(row.get("theorem_id", "") or "")
    residual_bucket = str(row.get("residual_bucket", "") or "")
    start_goal_bucket = str(row.get("last_goal_bucket", "") or "")
    return HardtailBridgeResult(
        theorem_id=theorem_id,
        started=False,
        theorem_faithful=False,
        residual_bucket=residual_bucket,
        start_goal_bucket=start_goal_bucket,
        closed=False,
        closed_by="",
        progressed=False,
        initial_goal_count=0,
        post_ducky1_goal_count=0,
        post_search_goal_count=0,
        final_goal_count=0,
        controller_decision={"controller_mode": "deterministic_packet_policy_v1"},
        replay={"success": False, "failure_category": error_category, "raw_error": error_message},
        ducky_pass_1=None,
        first_order_search=None,
        ducky_pass_2=None,
        symbolic_close_pass_2=None,
        final_goals=[],
        rarified_gap_packet={
            "packet_version": "rarified_proof_gap_v1",
            "theorem_id": theorem_id,
            "tractability_class": "bridge_execution_failure",
            "error_category": error_category,
            "error_message": error_message,
            "goals": [],
        },
        stage_trace=[{"stage": "bridge_failure", "error_category": error_category, "error_message": error_message}],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/exp_som012_hard_eval_r2")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--config", default="configs/wayfinder_specialist_apply.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--per-theorem-timeout", type=int, default=420)
    parser.add_argument("--restart-every", type=int, default=8)
    parser.add_argument("--sentence-encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--ducky-first-max-programs", type=int, default=24)
    parser.add_argument("--ducky-first-max-rounds", type=int, default=3)
    parser.add_argument("--ducky-second-max-programs", type=int, default=20)
    parser.add_argument("--ducky-second-max-rounds", type=int, default=2)
    parser.add_argument("--disable-ducky-tactic", action="append", default=["linarith", "nlinarith"])
    parser.add_argument("--residual-bucket", action="append", default=[])
    parser.add_argument("--theorem-id", action="append", default=[])
    parser.add_argument("--second-order-model", default="")
    parser.add_argument("--second-order-metadata", default="")
    parser.add_argument("--controller-device", default="cpu")
    parser.add_argument("--selection-source", choices=("validated_progress", "residual_round_robin"), default="validated_progress")
    parser.add_argument("--validated-seed-path", default="")
    parser.add_argument("--allow-unvalidated-backfill", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for logger_name in ("sentence_transformers", "transformers", "huggingface_hub", "httpx"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir.parent / f"exp_dd015_integrated_bridge_from_{run_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = run_dir / "details.jsonl"
    packet_path = run_dir / "bundle" / "second_order_som" / "second_order_packets.jsonl"
    residual_buckets = set(args.residual_bucket or LOCAL_BUCKETS)
    validated_seed_path = (
        Path(args.validated_seed_path).resolve()
        if args.validated_seed_path
        else run_dir / "bundle" / "dr_ducky" / "executor_validation_stratified120_rows.jsonl"
    )
    rows, selection_source_used = _select_rows(
        rows_path=rows_path,
        residual_buckets=residual_buckets,
        limit=args.limit,
        theorem_ids=list(args.theorem_id or []),
        selection_source=args.selection_source,
        validated_seed_path=validated_seed_path,
        allow_unvalidated_backfill=bool(args.allow_unvalidated_backfill),
    )
    output_jsonl = output_dir / "rows.jsonl"
    controller_jsonl = output_dir / "controller_decisions.jsonl"
    rarified_jsonl = output_dir / "rarified_gap_packets.jsonl"
    summary_json = output_dir / "summary.json"
    selected_theorems_json = output_dir / "selected_theorems.json"

    selected_theorems = [
        {
            "theorem_id": str(row.get("theorem_id", "") or ""),
            "residual_bucket": str(row.get("residual_bucket", "") or ""),
            "goals_closed": int(row.get("goals_closed", 0) or 0),
            "goals_remaining": int(row.get("goals_remaining", 0) or 0),
            "attempts": int(row.get("attempts", 0) or 0),
        }
        for row in rows
    ]
    _atomic_write_json(
        selected_theorems_json,
        {
            "selection_source_requested": args.selection_source,
            "selection_source_used": selection_source_used,
            "validated_seed_path": str(validated_seed_path),
            "allow_unvalidated_backfill": bool(args.allow_unvalidated_backfill),
            "selected_count": len(selected_theorems),
            "selected_theorems": selected_theorems,
        },
    )
    _atomic_write_json(
        summary_json,
        {
            "status": "startup",
            "selection_source_requested": args.selection_source,
            "selection_source_used": selection_source_used,
            "validated_seed_path": str(validated_seed_path),
            "selected_count": len(selected_theorems),
            "planned_residual_buckets": dict(
                sorted(
                    Counter(str(row.get("residual_bucket", "") or "") for row in rows).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
    )
    _progress_log(
        f"selected {len(rows)} rows via {selection_source_used}"
        + (f" using {validated_seed_path}" if selection_source_used.startswith("validated_progress") else "")
    )

    packet_index = load_second_order_packet_index(packet_path)
    controller_runtime = None
    if args.second_order_model:
        metadata_path = Path(args.second_order_metadata).resolve() if args.second_order_metadata else run_dir / "bundle" / "second_order_som" / "features" / "metadata.json"
        controller_runtime = load_learned_second_order_runtime(
            Path(args.second_order_model).resolve(),
            metadata_path,
            device=args.controller_device,
        )

    config = _load_config(Path(args.config))
    config.setdefault("lean", {})
    config["lean"]["backend"] = args.backend
    config["lean"]["project_root"] = args.lean_project
    config["lean"]["imports"] = ["Mathlib"]
    search_cfg = config.setdefault("search", {})
    search_cfg["cosine_rw_seq_enabled"] = True
    search_cfg["cosine_simp_enabled"] = True
    search_cfg["interleaved_bootstrap_enabled"] = True
    search_cfg["cosine_apply_enabled"] = True
    search_cfg["cosine_apply_gated"] = True
    search_cfg["exec_apply_selector_path"] = "models/apply_exec_selector_v2.pt"
    search_cfg["apply_trigger_path"] = "models/apply_trigger_v3.pt"
    search_cfg["family_classifier_torch_path"] = "models/som_torch_v1/best.pt"
    search_cfg["dr_ducky_enabled"] = False
    search_cfg["collect_trace"] = True

    pipeline, cfg, lean, _lean_cfg, conn = _build_search_components(config, Path(args.checkpoint), args.device)
    theorem_id_map = _build_theorem_id_map(conn)
    sentence_encoder = _load_sentence_encoder(args.sentence_encoder)
    if getattr(lean, "_backend", "") == "pantograph":
        lean._ensure_server()

    results: list[dict[str, Any]] = []
    rarified_packets: list[dict[str, Any]] = []
    disabled_tactics = set(args.disable_ducky_tactic or [])

    with output_jsonl.open("w") as rows_handle, controller_jsonl.open("w") as controller_handle, rarified_jsonl.open("w") as rarified_handle:
        for idx, row in enumerate(rows, start=1):
            theorem_id = str(row.get("theorem_id", "") or "")
            _progress_log(f"[{idx}/{len(rows)}] start {theorem_id}")
            if idx > 1 and args.restart_every > 0 and (idx - 1) % args.restart_every == 0:
                _restart_lean(lean)
            try:
                with _row_timeout(args.per_theorem_timeout):
                    result = run_hardtail_bridge_on_row(
                        row,
                        packet=packet_index.get(theorem_id),
                        controller_runtime=controller_runtime,
                        pipeline=pipeline,
                        search_config=cfg,
                        conn=conn,
                        lean=lean,
                        theorem_id_map=theorem_id_map,
                        sentence_encoder=sentence_encoder,
                        ducky_first_max_programs=args.ducky_first_max_programs,
                        ducky_first_max_rounds=args.ducky_first_max_rounds,
                        ducky_second_max_programs=args.ducky_second_max_programs,
                        ducky_second_max_rounds=args.ducky_second_max_rounds,
                        disabled_ducky_tactics=disabled_tactics,
                    )
            except RowTimeoutError as exc:
                logger.warning("Bridge timeout on %s: %s", theorem_id, exc)
                result = _failure_result(row, error_category="timeout", error_message=str(exc))
            except Exception as exc:
                logger.warning("Bridge error on %s: %s", theorem_id, exc)
                result = _failure_result(row, error_category="bridge_error", error_message=str(exc))

            payload = result.to_dict()
            results.append(payload)
            rows_handle.write(json.dumps(payload) + "\n")
            rows_handle.flush()
            controller_handle.write(json.dumps({"theorem_id": theorem_id, **payload.get("controller_decision", {})}) + "\n")
            controller_handle.flush()
            gap_packet = payload.get("rarified_gap_packet")
            if gap_packet:
                rarified_packets.append(gap_packet)
                rarified_handle.write(json.dumps(gap_packet) + "\n")
                rarified_handle.flush()
            _atomic_write_json(summary_json, _bridge_summary(results, rarified_packets))

    conn.close()
    try:
        lean.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
