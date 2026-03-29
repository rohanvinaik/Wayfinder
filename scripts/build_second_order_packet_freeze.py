"""Build the frozen second-order SoM packet surface from a completed hard run.

This consolidates three post-freeze surfaces into a single training/eval bundle:

- hard residual symbolic packets from `hard_resolution_layer`
- compiler/startability packets
- Dr. Ducky ledger + observed executor outcomes

Outputs:
    second_order_packets.jsonl
    ducky_outcome_packets.jsonl
    summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stable_split(theorem_id: str) -> str:
    digest = hashlib.sha256(theorem_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "eval" if bucket == 0 else "train"


def _row_compile_proxy_count(row: dict[str, Any]) -> int:
    return sum(1 for tried in row.get("tried_programs", []) if tried.get("tactics_applied"))


def _load_bridge_rows(
    run_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    """Convert inline hardtail bridge rows into eval/engine/projector surfaces.

    EXP-SOM-016 r6+ emits Ducky data inline in ``hardtail_bridge_rows.jsonl``
    rather than as separate executor-validation files.  This function flattens
    the bridge rows into the same per-theorem index structures that the
    canonical-source path produces, so the rest of the packet freeze works
    unchanged.

    Returns (eval_by_theorem, engine_by_theorem, projector_by_theorem).
    """
    bridge_path = run_dir / "hardtail_bridge_rows.jsonl"
    if not bridge_path.exists():
        return {}, {}, {}

    eval_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    engine_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    projector_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for raw in bridge_path.open():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        theorem_id = str(row.get("theorem_id", "") or "")
        if not theorem_id:
            continue

        # Build an eval-style row from the bridge top level
        eval_row: dict[str, Any] = {
            "source": "bridge_inline",
            "theorem_id": theorem_id,
            "residual_bucket": str(row.get("residual_bucket", "") or ""),
            "goal_bucket": str(row.get("start_goal_bucket", "") or ""),
            "started": bool(row.get("started")),
            "theorem_faithful": bool(row.get("theorem_faithful")),
            "progressed": bool(row.get("progressed")),
            "closed": bool(row.get("closed")),
            "closed_by": str(row.get("closed_by", "") or ""),
            "winning_program": None,
        }

        # Collect tried_programs from both passes for compile proxy
        all_tried: list[dict[str, Any]] = []
        ducky1 = row.get("ducky_pass_1") or {}
        all_tried.extend(ducky1.get("tried_programs") or [])
        ducky2 = row.get("ducky_pass_2") or {}
        for run in ducky2.get("runs") or []:
            all_tried.extend(run.get("tried_programs") or [])
        eval_row["tried_programs"] = all_tried

        # Find winning program if any tried program closed
        for tp in all_tried:
            if tp.get("closed"):
                eval_row["winning_program"] = tp.get("program_id", "")
                break

        eval_by_theorem[theorem_id].append(eval_row)

        # Flatten engine outcomes from both passes
        for eo in ducky1.get("engine_outcomes") or []:
            engine_by_theorem[theorem_id].append(
                {"source": "bridge_inline_pass1", "theorem_id": theorem_id, **eo}
            )
        for run in ducky2.get("runs") or []:
            for eo in run.get("engine_outcomes") or []:
                engine_by_theorem[theorem_id].append(
                    {"source": "bridge_inline_pass2", "theorem_id": theorem_id, **eo}
                )

        # Flatten projector outcomes from both passes
        for po in ducky1.get("projector_outcomes") or []:
            projector_by_theorem[theorem_id].append(
                {"source": "bridge_inline_pass1", "theorem_id": theorem_id, **po}
            )
        for run in ducky2.get("runs") or []:
            for po in run.get("projector_outcomes") or []:
                projector_by_theorem[theorem_id].append(
                    {"source": "bridge_inline_pass2", "theorem_id": theorem_id, **po}
                )

    return eval_by_theorem, engine_by_theorem, projector_by_theorem


def _canonical_ducky_sources(ducky_dir: Path) -> list[tuple[str, Path]]:
    names = [
        "executor_validation_local20_vnext_rows.jsonl",
        "executor_validation_stratified120_rows.jsonl",
        "ablation_single_goal_near_miss_rows.jsonl",
        "ablation_single_goal_stall_rows.jsonl",
        "ablation_multi_goal_small_progress_rows.jsonl",
        "ablation_multi_goal_large_progress_rows.jsonl",
    ]
    return [(name.replace("_rows.jsonl", ""), ducky_dir / name) for name in names]


def _canonical_engine_sources(ducky_dir: Path) -> list[tuple[str, Path]]:
    names = [
        "executor_validation_local20_vnext_engine_outcomes.jsonl",
        "executor_validation_stratified120_engine_outcomes.jsonl",
        "ablation_single_goal_near_miss_engine_outcomes.jsonl",
        "ablation_single_goal_stall_engine_outcomes.jsonl",
        "ablation_multi_goal_small_progress_engine_outcomes.jsonl",
        "ablation_multi_goal_large_progress_engine_outcomes.jsonl",
    ]
    return [(name.replace("_engine_outcomes.jsonl", ""), ducky_dir / name) for name in names]


def _canonical_projector_sources(ducky_dir: Path) -> list[tuple[str, Path]]:
    names = [
        "executor_validation_local20_vnext_projector_outcomes.jsonl",
        "executor_validation_stratified120_projector_outcomes.jsonl",
        "ablation_single_goal_near_miss_projector_outcomes.jsonl",
        "ablation_single_goal_stall_projector_outcomes.jsonl",
        "ablation_multi_goal_small_progress_projector_outcomes.jsonl",
        "ablation_multi_goal_large_progress_projector_outcomes.jsonl",
    ]
    return [(name.replace("_projector_outcomes.jsonl", ""), ducky_dir / name) for name in names]


def _canonical_closure_reports(ducky_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "executor_validation_local20_vnext_closure_report.json",
        "executor_validation_stratified120_closure_report.json",
        "ablation_single_goal_near_miss_closure_report.json",
        "ablation_single_goal_stall_closure_report.json",
        "ablation_multi_goal_small_progress_closure_report.json",
        "ablation_multi_goal_large_progress_closure_report.json",
    ]
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        path = ducky_dir / name
        if path.exists():
            out[name.replace("_closure_report.json", "")] = json.loads(path.read_text())
    return out


def build_second_order_packet_freeze(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    bundle_dir = run_dir / "bundle"
    ducky_dir = bundle_dir / "dr_ducky"
    resolution_dir = bundle_dir / "hard_resolution_layer"

    hard_packets = _load_jsonl(resolution_dir / "hard_som_packets.jsonl")
    compiler_packets = _load_jsonl(resolution_dir / "compiler_specialist_packets.jsonl")
    ducky_ledgers = {
        str(row.get("theorem_id", "") or ""): row
        for row in _load_jsonl(ducky_dir / "dr_ducky_ledger_packets.jsonl")
    }

    row_sources = _canonical_ducky_sources(ducky_dir)
    engine_sources = _canonical_engine_sources(ducky_dir)
    projector_sources = _canonical_projector_sources(ducky_dir)
    closure_reports = _canonical_closure_reports(ducky_dir)

    eval_rows_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    engine_rows_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    projector_rows_by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_name, path in row_sources:
        for row in _load_jsonl(path):
            theorem_id = str(row.get("theorem_id", "") or "")
            if theorem_id:
                eval_rows_by_theorem[theorem_id].append({"source": source_name, **row})

    for source_name, path in engine_sources:
        for row in _load_jsonl(path):
            theorem_id = str(row.get("theorem_id", "") or "")
            if theorem_id:
                engine_rows_by_theorem[theorem_id].append({"source": source_name, **row})

    for source_name, path in projector_sources:
        for row in _load_jsonl(path):
            theorem_id = str(row.get("theorem_id", "") or "")
            if theorem_id:
                projector_rows_by_theorem[theorem_id].append({"source": source_name, **row})

    # Merge inline bridge rows (EXP-SOM-016 r6+ format)
    bridge_eval, bridge_engine, bridge_projector = _load_bridge_rows(run_dir)
    for theorem_id, rows in bridge_eval.items():
        eval_rows_by_theorem[theorem_id].extend(rows)
    for theorem_id, rows in bridge_engine.items():
        engine_rows_by_theorem[theorem_id].extend(rows)
    for theorem_id, rows in bridge_projector.items():
        projector_rows_by_theorem[theorem_id].extend(rows)

    all_engine_rows = [
        row
        for theorem_id in sorted(engine_rows_by_theorem)
        for row in engine_rows_by_theorem[theorem_id]
    ]
    all_projector_rows = [
        row
        for theorem_id in sorted(projector_rows_by_theorem)
        for row in projector_rows_by_theorem[theorem_id]
    ]

    ducky_outcome_packets: list[dict[str, Any]] = []
    for theorem_id in sorted(set(eval_rows_by_theorem) | set(ducky_ledgers)):
        eval_rows = eval_rows_by_theorem.get(theorem_id, [])
        engine_rows = engine_rows_by_theorem.get(theorem_id, [])
        projector_rows = projector_rows_by_theorem.get(theorem_id, [])
        ledger = ducky_ledgers.get(theorem_id)

        residual_bucket = ""
        goal_bucket = ""
        if eval_rows:
            residual_bucket = str(eval_rows[0].get("residual_bucket", "") or "")
            goal_bucket = str(eval_rows[0].get("goal_bucket", "") or "")
        elif ledger:
            residual_bucket = str(ledger.get("residual_bucket", "") or "")
            goal_bucket = str(ledger.get("goal_bucket", "") or "")

        eval_sources = sorted({row["source"] for row in eval_rows})
        engine_counter = Counter(str(row.get("engine_name", "") or "") for row in engine_rows)
        backend_counter = Counter(str(row.get("backend_family", "") or "") for row in engine_rows)
        cert_counter = Counter(str(row.get("certificate_shape", "") or "") for row in engine_rows)
        projector_counter = Counter(str(row.get("projector_status", "") or "") for row in projector_rows)
        winning_counter = Counter(str(row.get("winning_program", "") or "") for row in eval_rows if row.get("winning_program"))

        observed_progress = any(bool(row.get("progressed")) for row in eval_rows)
        observed_close = any(bool(row.get("closed")) for row in eval_rows)
        observed_start = any(bool(row.get("started")) for row in eval_rows)
        theorem_faithful = any(bool(row.get("theorem_faithful")) for row in eval_rows)

        packet = {
            "packet_version": "second_order_ducky_outcomes_v1",
            "theorem_id": theorem_id,
            "split": _stable_split(theorem_id),
            "residual_bucket": residual_bucket,
            "goal_bucket": goal_bucket,
            "observed": bool(eval_rows),
            "observation_count": len(eval_rows),
            "evaluation_sources": eval_sources,
            "started_count": sum(1 for row in eval_rows if row.get("started")),
            "theorem_faithful_count": sum(1 for row in eval_rows if row.get("theorem_faithful")),
            "progressed_count": sum(1 for row in eval_rows if row.get("progressed")),
            "closed_count": sum(1 for row in eval_rows if row.get("closed")),
            "compile_proxy_count": sum(_row_compile_proxy_count(row) for row in eval_rows),
            "certificate_generation_count": len(engine_rows),
            "projector_event_count": len(projector_rows),
            "observed_progress": observed_progress,
            "observed_close": observed_close,
            "observed_start": observed_start,
            "theorem_faithful_observed": theorem_faithful,
            "best_outcome": "closed" if observed_close else "progressed" if observed_progress else "started" if observed_start else "unobserved",
            "engine_counts": dict(engine_counter),
            "backend_family_counts": dict(backend_counter),
            "certificate_shape_counts": dict(cert_counter),
            "projector_status_counts": dict(projector_counter),
            "winning_program_counts": dict(winning_counter),
            "ducky_ledger_surface": ledger,
        }
        ducky_outcome_packets.append(packet)

    ducky_outcome_by_theorem = {packet["theorem_id"]: packet for packet in ducky_outcome_packets}

    second_order_packets: list[dict[str, Any]] = []
    for packet in hard_packets:
        theorem_id = str(packet.get("theorem_id", "") or "")
        ducky_outcome = ducky_outcome_by_theorem.get(theorem_id, {})
        second_order_packets.append(
            {
                "packet_version": "second_order_controller_surface_v1",
                "packet_kind": "hard_residual",
                "theorem_id": theorem_id,
                "split": packet.get("split", _stable_split(theorem_id)),
                "difficulty_band": packet.get("difficulty_band", ""),
                "residual_bucket": packet.get("residual_bucket", ""),
                "goal_bucket": packet.get("goal_bucket", ""),
                "resolution_family": packet.get("resolution_family", ""),
                "hard_som_surface": packet,
                "ducky_outcome_surface": ducky_outcome,
                "second_order_labels": {
                    "invoke_ducky": True,
                    "observed_progress": bool(ducky_outcome.get("observed_progress")),
                    "observed_close": bool(ducky_outcome.get("observed_close")),
                    "engine_family_budget_targets": sorted((ducky_outcome.get("engine_counts") or {}).keys()),
                    "backend_budget_targets": sorted((ducky_outcome.get("backend_family_counts") or {}).keys()),
                    "projector_rejection_seen": bool((ducky_outcome.get("projector_status_counts") or {}).get("rejected", 0)),
                },
            }
        )

    for packet in compiler_packets:
        theorem_id = str(packet.get("theorem_id", "") or "")
        second_order_packets.append(
            {
                "packet_version": "second_order_controller_surface_v1",
                "packet_kind": "compiler_specialist",
                "theorem_id": theorem_id,
                "split": packet.get("split", _stable_split(theorem_id)),
                "difficulty_band": "",
                "residual_bucket": "skipped_start",
                "goal_bucket": "",
                "resolution_family": "compiler_specialist",
                "compiler_surface": packet,
                "ducky_outcome_surface": ducky_outcome_by_theorem.get(theorem_id, {}),
                "second_order_labels": {
                    "invoke_ducky": False,
                    "observed_progress": False,
                    "observed_close": False,
                    "engine_family_budget_targets": [],
                    "backend_budget_targets": [],
                    "projector_rejection_seen": False,
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(ducky_dir / "dr_ducky_engine_outcomes.jsonl", all_engine_rows)
    _write_jsonl(ducky_dir / "dr_ducky_projector_outcomes.jsonl", all_projector_rows)
    _write_jsonl(output_dir / "ducky_outcome_packets.jsonl", ducky_outcome_packets)
    _write_jsonl(output_dir / "second_order_packets.jsonl", second_order_packets)

    summary = {
        "packet_version": "second_order_controller_surface_v1",
        "run_dir": str(run_dir),
        "hard_residual_packets": len(hard_packets),
        "compiler_packets": len(compiler_packets),
        "ducky_ledger_packets": len(ducky_ledgers),
        "ducky_outcome_packets": len(ducky_outcome_packets),
        "ducky_observed_packets": sum(1 for packet in ducky_outcome_packets if packet.get("observed")),
        "ducky_progress_packets": sum(1 for packet in ducky_outcome_packets if packet.get("observed_progress")),
        "ducky_closed_packets": sum(1 for packet in ducky_outcome_packets if packet.get("observed_close")),
        "second_order_packets": len(second_order_packets),
        "by_packet_kind": dict(Counter(str(packet.get("packet_kind", "")) for packet in second_order_packets)),
        "observed_ducky_by_residual_bucket": dict(
            Counter(
                str(packet.get("residual_bucket", "") or "")
                for packet in ducky_outcome_packets
                if packet.get("observed")
            )
        ),
        "progress_ducky_by_residual_bucket": dict(
            Counter(
                str(packet.get("residual_bucket", "") or "")
                for packet in ducky_outcome_packets
                if packet.get("observed_progress")
            )
        ),
        "closure_reports": closure_reports,
        "top_engine_counts": dict(
            Counter(
                engine
                for packet in ducky_outcome_packets
                for engine, count in (packet.get("engine_counts") or {}).items()
                for _ in range(int(count))
            ).most_common(12)
        ),
        "top_backend_family_counts": dict(
            Counter(
                backend
                for packet in ducky_outcome_packets
                for backend, count in (packet.get("backend_family_counts") or {}).items()
                if backend
                for _ in range(int(count))
            ).most_common(12)
        ),
    }
    _write_json(ducky_dir / "dr_ducky_closure_report.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/exp_som012_hard_eval_r2")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to <run-dir>/bundle/second_order_som",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "bundle" / "second_order_som"
    summary = build_second_order_packet_freeze(run_dir=run_dir, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
