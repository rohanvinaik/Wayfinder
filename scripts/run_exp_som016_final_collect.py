"""EXP-SOM-016: final random-2,000 Mathlib benchmark with full data collection.

This is the canonical first-order benchmark collector for the current integrated
Wayfinder theorem-search stack. It freezes a random sample from the full Mathlib
corpus, runs theorem-faithful proof search with rich telemetry, and materializes
the postrun bundle needed for later first-order and second-order analysis.

Key properties:
- random sample frozen to a manifest for reproducibility
- per-theorem crash-stable JSONL writes
- live `MONITOR.txt` + `summary.json`
- full `step_trace` / `temporal_trace` capture
- goal-start failure and repair diagnostics
- trigger-state probing for apply/selector training
- postrun materialization of the hard-resolution bundle and second-order packet surfaces
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_hard_collection_bundle import build_hard_collection_bundle
from scripts.build_second_order_feature_dataset import build_second_order_feature_dataset
from scripts.build_second_order_packet_freeze import build_second_order_packet_freeze
from scripts.collect_trigger_states import (
    _build_goal_start_failure_row,
    _goal_shape_features,
    _load_exec_selector,
    extract_trigger_states,
)
from scripts.run_benchmark import (
    _build_search_components,
    _build_theorem_id_map,
    _group_by_source,
    _resolve_initial_goal,
)
from scripts.run_exp_som012_hard_collect import (
    _attach_file_logger,
    _load_sentence_encoder,
    _skipped_row,
    _theorem_row,
)
from src.benchmark_residuals import summarize_residual_structure
from src.hard_data_tags import canonicalize_theorem_id, classify_goal_bucket, goal_bucket_tags
from src.hard_resolution_layer import load_jsonl as load_resolution_rows
from src.hard_resolution_layer import materialize_hard_resolution_layer
from src.hardtail_bridge import run_hardtail_bridge_on_row
from src.proof_network import get_accessible_premises
from src.proof_search import SearchResult, search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_BRIDGE_RESIDUAL_BUCKETS = {
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
}
_BRIDGE_DISABLED_DUCKY_TACTICS = {"linarith", "nlinarith"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_mathlib_corpus(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            normalized = _normalize_mathlib_row(json.loads(raw))
            theorem_id = str(normalized.get("theorem_id", "") or "")
            if not theorem_id or theorem_id in seen:
                continue
            seen.add(theorem_id)
            rows.append(normalized)
    return rows


def _normalize_mathlib_row(row: dict[str, Any]) -> dict[str, Any]:
    theorem_id = canonicalize_theorem_id(
        str(
            row.get("theorem_id")
            or row.get("full_name")
            or row.get("name")
            or ""
        )
    )
    theorem_statement = str(
        row.get("theorem_statement")
        or row.get("goal_state")
        or row.get("statement")
        or ""
    )
    file_path = str(row.get("file_path", "") or "")
    module = str(row.get("module", "") or "")
    if not module and file_path.startswith("Mathlib/") and file_path.endswith(".lean"):
        module = file_path[:-5].replace("/", ".")
    proofs = row.get("tactics")
    if not isinstance(proofs, list):
        proofs = row.get("proof")
    proof_steps = len(proofs) if isinstance(proofs, list) else 0
    premises = row.get("premises")
    unique_premises = len(set(str(p) for p in premises)) if isinstance(premises, list) else 0
    return {
        "theorem_id": theorem_id,
        "raw_theorem_id": str(
            row.get("raw_theorem_id")
            or row.get("theorem_id")
            or row.get("full_name")
            or row.get("name")
            or theorem_id
        ),
        "goal_state": theorem_statement,
        "ground_truth_tactic": str(row.get("ground_truth_tactic", "") or ""),
        "source": str(row.get("source", "") or "mathlib_random_2000"),
        "namespace_prefix": theorem_id.rsplit(".", 1)[0] if "." in theorem_id else "",
        "theorem_statement": theorem_statement,
        "template_id": str(row.get("template_id", "") or ""),
        "proof_steps": int(row.get("proof_steps", proof_steps) or 0),
        "unique_premises": int(row.get("unique_premises", unique_premises) or 0),
        "difficulty_band": str(row.get("difficulty_band", "") or ""),
        "hard_half": bool(row.get("hard_half", False)),
        "split": str(row.get("split", "") or ""),
        "file_path": file_path,
        "module": module,
        "lean_path": str(row.get("lean_path", "") or file_path),
        "theorem_line": int(row.get("theorem_line", row.get("line", 0)) or 0),
    }


def _freeze_sample(
    *,
    corpus_rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    if manifest_path is not None and manifest_path.exists():
        sampled: list[dict[str, Any]] = []
        with manifest_path.open() as handle:
            for raw in handle:
                raw = raw.strip()
                if raw:
                    sampled.append(_normalize_mathlib_row(json.loads(raw)))
        metadata = {
            "mode": "reused_manifest",
            "manifest_path": str(manifest_path),
            "sample_size": len(sampled),
            "sample_seed": seed,
            "corpus_size": len(corpus_rows),
        }
        return sampled, manifest_path, metadata

    if sample_size <= 0:
        raise SystemExit("sample-size must be positive")
    if sample_size > len(corpus_rows):
        raise SystemExit(
            f"sample-size {sample_size} exceeds corpus size {len(corpus_rows)}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(corpus_rows, sample_size)
    sample_manifest_path = output_dir / "sample_manifest.jsonl"
    _write_jsonl(sample_manifest_path, sampled)
    metadata = {
        "mode": "fresh_sample",
        "manifest_path": str(sample_manifest_path),
        "sample_size": len(sampled),
        "sample_seed": seed,
        "corpus_size": len(corpus_rows),
        "corpus_source": "",
    }
    return sampled, sample_manifest_path, metadata


def _difficulty_bucket(row: dict[str, Any]) -> str:
    if bool(row.get("honest_success")) or bool(row.get("success")):
        return "resolved"
    follow_on_stage = str(row.get("follow_on_stage", "") or "")
    residual_bucket = str(row.get("residual_bucket", "") or "")
    if follow_on_stage == "compiler_specialist":
        return "compiler_or_startability"
    if residual_bucket.startswith("single_goal_"):
        return "hard_local"
    if residual_bucket.startswith("multi_goal_small_"):
        return "hard_planner_small"
    if residual_bucket.startswith("multi_goal_large_"):
        return "hard_planner_large"
    if follow_on_stage == "theorem_replanner":
        return "theorem_replanner"
    if follow_on_stage == "hard_proof_solver":
        return "hard_other"
    return "other"


def _summary_rows(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    residual_structure = summarize_residual_structure(summary_rows)
    raw_success = sum(1 for row in summary_rows if row.get("success"))
    honest_success = sum(1 for row in summary_rows if row.get("honest_success"))
    difficulty_counts: dict[str, int] = {}
    for row in summary_rows:
        difficulty_counts[_difficulty_bucket(row)] = difficulty_counts.get(_difficulty_bucket(row), 0) + 1
    return {
        "total_theorems": len(summary_rows),
        "raw_success": raw_success,
        "raw_success_rate": round(raw_success / max(len(summary_rows), 1), 4),
        "honest_success": honest_success,
        "honest_success_rate": round(honest_success / max(len(summary_rows), 1), 4),
        "started_theorems": residual_structure["started_theorems"],
        "skipped_start": residual_structure["skipped_start"],
        "started_success_rate": residual_structure["started_success_rate"],
        "residual_structure": residual_structure,
        "difficulty_bucket_counts": difficulty_counts,
    }


def _bridge_summary_rows(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    invoked = [row for row in summary_rows if bool(row.get("bridge_invoked"))]
    closed_by = Counter(
        str(row.get("bridge_closed_by", "") or "")
        for row in invoked
        if bool(row.get("bridge_closed"))
    )
    return {
        "invoked": len(invoked),
        "progressed": sum(1 for row in invoked if bool(row.get("bridge_progressed"))),
        "closed": sum(1 for row in invoked if bool(row.get("bridge_closed"))),
        "by_closed_stage": dict(sorted(closed_by.items(), key=lambda item: (-item[1], item[0]))),
        "rarified_gap_packets": sum(1 for row in invoked if row.get("bridge_rarified_gap_packet")),
    }


def _write_monitor(
    path: Path,
    *,
    total: int,
    processed: int,
    raw_success: int,
    honest_success: int,
    started: int,
    skipped_start: int,
    goal_start_failures: int,
    goal_start_repairs: int,
    hard_residuals: int,
    trigger_rows: int,
    total_attempts: int,
    started_at: float,
    current_theorem: str,
    sample_manifest_path: Path,
    details_path: Path,
    goal_start_path: Path,
    trigger_path: Path,
    log_path: Path,
    summary_path: Path,
    report_path: Path,
    bridge_invoked: int = 0,
    bridge_progressed: int = 0,
    bridge_closed: int = 0,
    bridge_rows_path: Path | None = None,
    controller_path: Path | None = None,
    rarified_path: Path | None = None,
) -> None:
    elapsed = time.time() - started_at
    lines = [
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Current theorem: {current_theorem or '(idle)'}",
        "",
        f"Processed: {processed}/{total}",
        f"Raw success: {raw_success}",
        f"Honest success: {honest_success}",
        f"Started: {started}",
        f"Skipped start: {skipped_start}",
        f"Goal-start failures seen: {goal_start_failures}",
        f"Goal-start repairs: {goal_start_repairs}",
        f"Hard residuals: {hard_residuals}",
        f"Bridge invoked: {bridge_invoked}",
        f"Bridge progressed: {bridge_progressed}",
        f"Bridge closed: {bridge_closed}",
        f"Trigger rows: {trigger_rows}",
        f"Avg attempts/theorem: {round(total_attempts / max(processed, 1), 1)}",
        f"Elapsed seconds: {round(elapsed, 1)}",
        "",
        "Artifacts",
        f"sample_manifest.jsonl: {sample_manifest_path}",
        f"details.jsonl: {details_path}",
        f"goal_start_failures.jsonl: {goal_start_path}",
        f"trigger_states.jsonl: {trigger_path}",
        f"hardtail_bridge_rows.jsonl: {str(bridge_rows_path) if bridge_rows_path is not None else '(none)'}",
        f"controller_decisions.jsonl: {str(controller_path) if controller_path is not None else '(none)'}",
        f"rarified_gap_packets.jsonl: {str(rarified_path) if rarified_path is not None else '(none)'}",
        f"run.log: {log_path}",
        f"summary.json: {summary_path}",
        f"report.json: {report_path}",
        f"summary exists: {summary_path.exists()}",
        f"report exists: {report_path.exists()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_live_summary(
    path: Path,
    *,
    run_id: str,
    status: str,
    sample_metadata: dict[str, Any],
    processed: int,
    summary_rows: list[dict[str, Any]],
    total_attempts: int,
    started_at: float,
    config_summary: dict[str, Any],
    artifact_summary: dict[str, Any],
) -> None:
    bench = _summary_rows(summary_rows)
    payload = {
        "experiment": run_id,
        "status": status,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample": sample_metadata,
        "benchmark": bench,
        "bridge": _bridge_summary_rows(summary_rows),
        "efficiency": {
            "processed": processed,
            "total_attempts": total_attempts,
            "avg_attempts_per_theorem": round(total_attempts / max(processed, 1), 1),
            "avg_time_per_theorem_s": round(
                sum(float(row.get("time_s", 0.0)) for row in summary_rows) / max(processed, 1),
                2,
            ),
            "elapsed_s": round(time.time() - started_at, 1),
        },
        "by_source": _group_by_source(summary_rows),
        "config": config_summary,
        "artifacts": artifact_summary,
    }
    _write_json(path, payload)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _bridge_eligible(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("started"))
        and not bool(row.get("success"))
        and str(row.get("residual_bucket", "") or "") in _BRIDGE_RESIDUAL_BUCKETS
    )


def _bridge_attempts(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0
    if "programs_considered" in payload:
        return int(payload.get("programs_considered", 0) or 0)
    runs = payload.get("runs") or []
    return sum(int(run.get("programs_considered", 0) or 0) for run in runs if isinstance(run, dict))


def _bridge_search_attempts(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0
    return int(payload.get("attempts", 0) or 0)


def _bridge_total_attempts(bridge: dict[str, Any]) -> int:
    return (
        _bridge_attempts(bridge.get("ducky_pass_1"))
        + _bridge_search_attempts(bridge.get("first_order_search"))
        + _bridge_attempts(bridge.get("ducky_pass_2"))
        + _bridge_search_attempts(bridge.get("symbolic_close_pass_2"))
    )


def _bridge_final_closer(bridge: dict[str, Any]) -> str:
    closed_by = str(bridge.get("closed_by", "") or "")
    if closed_by == "first_order_search":
        payload = bridge.get("first_order_search") or {}
        tactics = payload.get("tactics_used") or []
        if tactics:
            return str(tactics[-1])
        provenance = payload.get("close_provenance") or []
        if provenance:
            return str(provenance[-1])
    if closed_by == "post_ducky_symbolic_2":
        payload = bridge.get("symbolic_close_pass_2") or {}
        tactics = payload.get("tactics_used") or []
        if tactics:
            return str(tactics[-1])
        provenance = payload.get("close_provenance") or []
        if provenance:
            return str(provenance[-1])
    if closed_by == "dr_ducky_pass_1":
        payload = bridge.get("ducky_pass_1") or {}
        winning = payload.get("winning_program") or {}
        return str(winning.get("script") or winning.get("program_id") or closed_by)
    if closed_by == "dr_ducky_pass_2":
        payload = bridge.get("ducky_pass_2") or {}
        for run in payload.get("runs") or []:
            if run.get("closed"):
                return str(run.get("script") or run.get("program_id") or closed_by)
    return closed_by


def _apply_bridge_result(theorem_row: dict[str, Any], bridge: dict[str, Any], *, bridge_time_s: float) -> dict[str, Any]:
    out = dict(theorem_row)
    out = _with_first_order_snapshot(out)

    out["bridge_invoked"] = True
    out["bridge_started"] = bool(bridge.get("started"))
    out["bridge_theorem_faithful"] = bool(bridge.get("theorem_faithful"))
    out["bridge_progressed"] = bool(bridge.get("progressed"))
    out["bridge_closed"] = bool(bridge.get("closed"))
    out["bridge_closed_by"] = str(bridge.get("closed_by", "") or "")
    out["bridge_time_s"] = round(float(bridge_time_s), 3)
    out["bridge_attempts"] = int(_bridge_total_attempts(bridge))
    out["bridge_controller_decision"] = bridge.get("controller_decision") or {}
    out["bridge_stage_trace"] = list(bridge.get("stage_trace") or [])
    out["bridge_ducky_pass_1"] = bridge.get("ducky_pass_1")
    out["bridge_first_order_search"] = bridge.get("first_order_search")
    out["bridge_ducky_pass_2"] = bridge.get("ducky_pass_2")
    out["bridge_symbolic_close_pass_2"] = bridge.get("symbolic_close_pass_2")
    out["bridge_rarified_gap_packet"] = bridge.get("rarified_gap_packet")
    out["bridge_final_goals"] = list(bridge.get("final_goals") or [])

    out["attempts"] = int(out["first_order_attempts"]) + int(out["bridge_attempts"])
    out["time_s"] = round(float(out["first_order_time_s"]) + float(bridge_time_s), 3)

    final_goals = list(bridge.get("final_goals") or [])
    if bridge.get("closed"):
        out["success"] = True
        out["honest_success"] = bool(bridge.get("theorem_faithful"))
        out["success_category"] = "bridge_success"
        out["residual_bucket"] = "proved"
        out["raw_success"] = True
        out["close_lane"] = f"hardtail_bridge:{str(bridge.get('closed_by', '') or '')}"
        out["final_closer"] = _bridge_final_closer(bridge)
        provenance = list(out["first_order_close_provenance"])
        provenance.append(out["close_lane"])
        out["close_provenance"] = provenance
        lane_sequence = [token for token in str(out["first_order_lane_sequence"]).split("→") if token]
        lane_sequence.append(out["close_lane"])
        out["lane_sequence"] = "→".join(dict.fromkeys(lane_sequence))
        out["remaining_goals_snapshot"] = []
        out["last_goal"] = ""
        out["last_goal_available"] = False
        out["last_goal_bucket"] = "empty"
        out["last_goal_tags"] = []
        out["last_goal_shape_features"] = {}
        out["goals_remaining"] = 0
        total_goal_count = max(
            int(out["first_order_goals_closed"]) + int(out["first_order_goals_remaining"]),
            int(bridge.get("initial_goal_count", 0) or 0),
            1,
        )
        out["goals_closed"] = total_goal_count
    else:
        out["success"] = False
        out["honest_success"] = False
        out["success_category"] = "failed"
        out["remaining_goals_snapshot"] = final_goals
        out["last_goal"] = str(final_goals[0] if final_goals else "")
        out["last_goal_available"] = bool(final_goals)
        out["last_goal_bucket"] = classify_goal_bucket(out["last_goal"]) if out["last_goal"] else "empty"
        out["last_goal_tags"] = goal_bucket_tags(out["last_goal"]) if out["last_goal"] else []
        out["last_goal_shape_features"] = _goal_shape_features(out["last_goal"]) if out["last_goal"] else {}
        out["goals_remaining"] = len(final_goals)
        total_goal_count = max(
            int(out["first_order_goals_closed"]) + int(out["first_order_goals_remaining"]),
            int(bridge.get("initial_goal_count", 0) or 0),
            len(final_goals),
        )
        out["goals_closed"] = max(total_goal_count - len(final_goals), 0)

    return out


def _with_first_order_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("first_order_success", bool(row.get("success")))
    out.setdefault("first_order_honest_success", bool(row.get("honest_success")))
    out.setdefault("first_order_success_category", str(row.get("success_category", "") or ""))
    out.setdefault("first_order_close_lane", str(row.get("close_lane", "") or ""))
    out.setdefault("first_order_final_closer", str(row.get("final_closer", "") or ""))
    out.setdefault("first_order_lane_sequence", str(row.get("lane_sequence", "") or ""))
    out.setdefault("first_order_close_provenance", list(row.get("close_provenance") or []))
    out.setdefault("first_order_attempts", int(row.get("attempts", 0) or 0))
    out.setdefault("first_order_time_s", float(row.get("time_s", 0.0) or 0.0))
    out.setdefault("first_order_goals_closed", int(row.get("goals_closed", 0) or 0))
    out.setdefault("first_order_goals_remaining", int(row.get("goals_remaining", 0) or 0))
    out.setdefault("first_order_remaining_goals_snapshot", list(row.get("remaining_goals_snapshot") or []))
    out.setdefault("first_order_last_goal", str(row.get("last_goal", "") or ""))
    out.setdefault("first_order_last_goal_bucket", str(row.get("last_goal_bucket", "") or ""))
    out.setdefault("first_order_residual_bucket", str(row.get("residual_bucket", "") or ""))
    out.setdefault("first_order_follow_on_stage", str(row.get("follow_on_stage", "") or ""))
    out.setdefault("first_order_reasoning_gap_family", str(row.get("reasoning_gap_family", "") or ""))
    out.setdefault("first_order_difficulty_bucket", str(row.get("difficulty_bucket", "") or ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--mathlib-corpus", default="data/leandojo_mathlib.jsonl")
    parser.add_argument("--output-dir", default="runs/exp_som016_final_random2000_r1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-project", default="data/lean_project")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--per-theorem-timeout", type=int, default=300)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--sentence-encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--search-mode", default="full")
    parser.add_argument("--temporal", default="arbiter_full")
    parser.add_argument("--strategy-memory-path", default="data/strategy_memory_som.json")
    parser.add_argument("--cosine-rw", action="store_true", default=True)
    parser.add_argument("--cosine-rw-seq", action="store_true")
    parser.add_argument("--cosine-rw-beam", type=int, default=5)
    parser.add_argument("--interleaved-bootstrap", action="store_true", default=True)
    parser.add_argument("--interleaved-bootstrap-max-depth", type=int, default=4)
    parser.add_argument("--interleaved-bootstrap-max-calls", type=int, default=20)
    parser.add_argument("--family-classifier-path", default="")
    parser.add_argument("--family-classifier-torch-path", default="models/som_torch_v1/best.pt")
    parser.add_argument("--exec-apply-selector-path", default="models/apply_exec_selector_v2.pt")
    parser.add_argument("--exec-apply-selector-pool", type=int, default=20)
    parser.add_argument("--apply-trigger-path", default="models/apply_trigger_v3.pt")
    parser.add_argument("--apply-trigger-threshold", type=float, default=0.47)
    parser.add_argument("--cosine-apply-beam", type=int, default=5)
    parser.add_argument("--disable-dr-ducky", action="store_true")
    parser.add_argument("--dr-ducky-max-programs", type=int, default=24)
    parser.add_argument("--dr-ducky-max-rounds", type=int, default=3)
    parser.add_argument("--dr-ducky-goal-limit", type=int, default=3)
    parser.add_argument("--selector", default="models/apply_exec_selector_v1.pt")
    parser.add_argument("--probe-lean", action="store_true")
    parser.add_argument("--probe-k", type=int, default=3)
    parser.add_argument("--min-trace-length", type=int, default=2)
    parser.add_argument("--min-strategy-support", type=int, default=3)
    parser.add_argument("--hard-resolution-candidate-limit", type=int, default=12)
    parser.add_argument("--hard-resolution-exemplar-limit", type=int, default=5)
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--second-order-model", default="", help="Path to trained second-order SoM checkpoint for learned bridge controller")
    parser.add_argument("--second-order-metadata", default="", help="Path to feature metadata.json for the second-order SoM")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]
    search_cfg = config.setdefault("search", {})
    search_cfg["search_mode"] = args.search_mode
    search_cfg["temporal_mode"] = args.temporal
    search_cfg["strategy_memory_path"] = args.strategy_memory_path
    search_cfg["budget"] = args.budget
    search_cfg["cosine_rw_seq_enabled"] = bool(args.cosine_rw_seq)
    search_cfg["interleaved_bootstrap_enabled"] = bool(args.interleaved_bootstrap)
    search_cfg["interleaved_bootstrap_max_depth"] = args.interleaved_bootstrap_max_depth
    search_cfg["interleaved_bootstrap_max_calls"] = args.interleaved_bootstrap_max_calls
    search_cfg["family_classifier_path"] = args.family_classifier_path
    search_cfg["family_classifier_torch_path"] = args.family_classifier_torch_path
    search_cfg["cosine_apply_enabled"] = True
    search_cfg["cosine_apply_gated"] = True
    search_cfg["cosine_apply_beam"] = args.cosine_apply_beam
    search_cfg["exec_apply_selector_path"] = args.exec_apply_selector_path
    search_cfg["exec_apply_selector_pool"] = args.exec_apply_selector_pool
    search_cfg["apply_trigger_path"] = args.apply_trigger_path
    search_cfg["apply_trigger_threshold"] = args.apply_trigger_threshold
    search_cfg["dr_ducky_enabled"] = not args.disable_dr_ducky
    search_cfg["dr_ducky_max_programs"] = args.dr_ducky_max_programs
    search_cfg["dr_ducky_max_rounds"] = args.dr_ducky_max_rounds
    search_cfg["dr_ducky_goal_limit"] = args.dr_ducky_goal_limit

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    goal_start_path = output_dir / "goal_start_failures.jsonl"
    trigger_path = output_dir / "trigger_states.jsonl"
    bridge_path = output_dir / "hardtail_bridge_rows.jsonl"
    controller_path = output_dir / "controller_decisions.jsonl"
    rarified_path = output_dir / "rarified_gap_packets.jsonl"
    monitor_path = output_dir / "MONITOR.txt"
    log_path = output_dir / "run.log"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.json"
    sample_summary_path = output_dir / "sample_summary.json"
    _attach_file_logger(log_path)

    corpus_rows = _load_mathlib_corpus(Path(args.mathlib_corpus))
    manifest_override = Path(args.sample_manifest) if args.sample_manifest else None
    if args.resume and manifest_override is None:
        existing_manifest = output_dir / "sample_manifest.jsonl"
        if existing_manifest.exists():
            manifest_override = existing_manifest
    theorems, sample_manifest_path, sample_metadata = _freeze_sample(
        corpus_rows=corpus_rows,
        sample_size=args.sample_size,
        seed=args.sample_seed,
        output_dir=output_dir,
        manifest_path=manifest_override,
    )
    sample_metadata["corpus_source"] = str(Path(args.mathlib_corpus).resolve())
    _write_json(sample_summary_path, sample_metadata)
    logger.info("Loaded Mathlib corpus: %d theorems", len(corpus_rows))
    logger.info("Frozen benchmark sample: %d theorems -> %s", len(theorems), sample_manifest_path)

    pipeline, cfg, lean, lean_cfg, conn = _build_search_components(
        config,
        Path(args.checkpoint),
        args.device,
    )
    cfg.collect_trace = True
    cfg.budget = args.budget
    cfg.cosine_rw_beam = args.cosine_rw_beam
    cfg.interleaved_bootstrap_enabled = bool(args.interleaved_bootstrap)
    cfg.interleaved_bootstrap_max_depth = args.interleaved_bootstrap_max_depth
    cfg.interleaved_bootstrap_max_calls = args.interleaved_bootstrap_max_calls
    cfg.family_classifier_path = args.family_classifier_path
    cfg.family_classifier_torch_path = args.family_classifier_torch_path
    cfg.strategy_memory_path = args.strategy_memory_path
    cfg.temporal_mode = args.temporal
    cfg.cosine_apply_enabled = True
    cfg.cosine_apply_gated = True
    cfg.cosine_apply_beam = args.cosine_apply_beam
    cfg.exec_apply_selector_path = args.exec_apply_selector_path
    cfg.exec_apply_selector_pool = args.exec_apply_selector_pool
    cfg.apply_trigger_path = args.apply_trigger_path
    cfg.apply_trigger_threshold = args.apply_trigger_threshold
    cfg.dr_ducky_enabled = not args.disable_dr_ducky
    cfg.dr_ducky_max_programs = args.dr_ducky_max_programs
    cfg.dr_ducky_max_rounds = args.dr_ducky_max_rounds
    cfg.dr_ducky_goal_limit = args.dr_ducky_goal_limit

    sentence_encoder = _load_sentence_encoder(args.sentence_encoder)
    selector = None
    selector_enc = None
    if args.selector:
        selector_result = _load_exec_selector(args.selector)
        if selector_result is not None:
            selector, selector_enc = selector_result

    # Load learned second-order controller if provided
    controller_runtime = None
    if args.second_order_model and args.second_order_metadata:
        from src.second_order_som_model import load_learned_second_order_runtime
        controller_runtime = load_learned_second_order_runtime(
            checkpoint_path=Path(args.second_order_model),
            metadata_path=Path(args.second_order_metadata),
            device="cpu",
        )
        logger.info("Loaded second-order controller from %s", args.second_order_model)

    resumed_rows = _load_jsonl_rows(details_path) if args.resume else []
    resumed_goal_start_rows = _load_jsonl_rows(goal_start_path) if args.resume else []
    resumed_trigger_rows = _load_jsonl_rows(trigger_path) if args.resume else []
    processed_theorem_ids = {
        str(row.get("theorem_id", "") or "")
        for row in resumed_rows
        if str(row.get("theorem_id", "") or "")
    }

    started_at = time.time()
    processed = len(resumed_rows)
    raw_success = sum(1 for row in resumed_rows if row.get("success"))
    honest_success = sum(1 for row in resumed_rows if row.get("honest_success"))
    total_attempts = sum(int(row.get("attempts", 0) or 0) for row in resumed_rows)
    goal_start_failures = len(resumed_goal_start_rows)
    goal_start_repairs = sum(1 for row in resumed_rows if row.get("repair_success"))
    trigger_rows_total = len(resumed_trigger_rows)
    trigger_positive = sum(
        int(row.get("trigger_label") or row.get("can_apply") or 0)
        for row in resumed_trigger_rows
    )
    started_count = sum(1 for row in resumed_rows if row.get("started"))
    skipped_start_count = processed - started_count
    hard_residual_count = sum(
        1 for row in resumed_rows if row.get("follow_on_stage") == "hard_proof_solver"
    )
    bridge_invoked = sum(1 for row in resumed_rows if row.get("bridge_invoked"))
    bridge_progressed = sum(1 for row in resumed_rows if row.get("bridge_progressed"))
    bridge_closed = sum(1 for row in resumed_rows if row.get("bridge_closed"))
    summary_rows: list[dict[str, Any]] = list(resumed_rows)

    config_summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "budget": cfg.budget,
        "device": args.device,
        "lean_backend": lean_cfg.backend,
        "search_mode": cfg.search_mode,
        "temporal_mode": cfg.temporal_mode,
        "strategy_memory_path": cfg.strategy_memory_path,
        "cosine_rw_enabled": bool(args.cosine_rw),
        "cosine_rw_seq_enabled": bool(args.cosine_rw_seq),
        "cosine_rw_beam": cfg.cosine_rw_beam,
        "interleaved_bootstrap_enabled": cfg.interleaved_bootstrap_enabled,
        "family_classifier_path": cfg.family_classifier_path,
        "family_classifier_torch_path": cfg.family_classifier_torch_path,
        "cosine_apply_enabled": cfg.cosine_apply_enabled,
        "cosine_apply_gated": cfg.cosine_apply_gated,
        "exec_apply_selector_path": cfg.exec_apply_selector_path,
        "probe_selector_path": str(Path(args.selector).resolve()) if args.selector else "",
        "probe_selector_loaded": bool(selector is not None),
        "apply_trigger_path": cfg.apply_trigger_path,
        "apply_trigger_threshold": cfg.apply_trigger_threshold,
        "dr_ducky_enabled": cfg.dr_ducky_enabled,
        "dr_ducky_max_programs": cfg.dr_ducky_max_programs,
        "dr_ducky_max_rounds": cfg.dr_ducky_max_rounds,
        "dr_ducky_goal_limit": cfg.dr_ducky_goal_limit,
        "collect_trace": True,
        "mathlib_corpus": str(Path(args.mathlib_corpus).resolve()),
        "resume": bool(args.resume),
    }
    artifact_summary = {
        "sample_manifest_jsonl": str(sample_manifest_path),
        "details_jsonl": str(details_path),
        "goal_start_failures_jsonl": str(goal_start_path),
        "trigger_states_jsonl": str(trigger_path),
        "hardtail_bridge_rows_jsonl": str(bridge_path),
        "controller_decisions_jsonl": str(controller_path),
        "rarified_gap_packets_jsonl": str(rarified_path),
        "sample_summary_json": str(sample_summary_path),
    }

    _write_live_summary(
        summary_path,
        run_id="EXP-SOM-016-final-random-2000",
        status="resuming" if args.resume and processed else "starting",
        sample_metadata=sample_metadata,
        processed=processed,
        summary_rows=summary_rows,
        total_attempts=total_attempts,
        started_at=started_at,
        config_summary=config_summary,
        artifact_summary=artifact_summary,
    )
    _write_json(report_path, json.loads(summary_path.read_text()))
    _write_monitor(
        monitor_path,
        total=len(theorems),
        processed=processed,
        raw_success=raw_success,
        honest_success=honest_success,
        started=started_count,
        skipped_start=skipped_start_count,
        goal_start_failures=goal_start_failures,
        goal_start_repairs=goal_start_repairs,
        hard_residuals=hard_residual_count,
        trigger_rows=trigger_rows_total,
        total_attempts=total_attempts,
        started_at=started_at,
        current_theorem="(startup-resume)" if args.resume and processed else "(startup)",
        sample_manifest_path=sample_manifest_path,
        details_path=details_path,
        goal_start_path=goal_start_path,
        trigger_path=trigger_path,
        log_path=log_path,
        summary_path=summary_path,
        report_path=report_path,
        bridge_invoked=bridge_invoked,
        bridge_progressed=bridge_progressed,
        bridge_closed=bridge_closed,
        bridge_rows_path=bridge_path,
        controller_path=controller_path,
        rarified_path=rarified_path,
    )

    name_to_id = _build_theorem_id_map(conn) if cfg.accessible_premises else {}
    id_to_name = {}
    if selector is not None:
        id_to_name = {eid: name for eid, name in conn.execute("SELECT id, name FROM entities")}

    if lean._backend == "pantograph":
        lean._ensure_server()

    details_mode = "a" if args.resume else "w"
    start_mode = "a" if args.resume else "w"
    trigger_mode = "a" if args.resume else "w"
    bridge_mode = "a" if args.resume else "w"
    with (
        details_path.open(details_mode) as details_handle,
        goal_start_path.open(start_mode) as start_handle,
        trigger_path.open(trigger_mode) as trigger_handle,
        bridge_path.open(bridge_mode) as bridge_handle,
        controller_path.open(bridge_mode) as controller_handle,
        rarified_path.open(bridge_mode) as rarified_handle,
    ):
        for idx, thm in enumerate(theorems):
            theorem_id = str(thm.get("theorem_id", "") or "")
            if theorem_id in processed_theorem_ids:
                continue
            logger.info("[%d/%d] %s", idx + 1, len(theorems), theorem_id)
            _write_monitor(
                monitor_path,
                total=len(theorems),
                processed=processed,
                raw_success=raw_success,
                honest_success=honest_success,
                started=started_count,
                skipped_start=skipped_start_count,
                goal_start_failures=goal_start_failures,
                goal_start_repairs=goal_start_repairs,
                hard_residuals=hard_residual_count,
                trigger_rows=trigger_rows_total,
                total_attempts=total_attempts,
                started_at=started_at,
                current_theorem=theorem_id,
                sample_manifest_path=sample_manifest_path,
                details_path=details_path,
                goal_start_path=goal_start_path,
                trigger_path=trigger_path,
                log_path=log_path,
                summary_path=summary_path,
                report_path=report_path,
                bridge_invoked=bridge_invoked,
                bridge_progressed=bridge_progressed,
                bridge_closed=bridge_closed,
                bridge_rows_path=bridge_path,
                controller_path=controller_path,
                rarified_path=rarified_path,
            )

            accessible_id = name_to_id.get(theorem_id) if name_to_id else None
            initial_goal = _resolve_initial_goal(thm, lean)
            start_row: dict[str, Any] | None = None
            repaired_goal: str | None = None
            repair_result: Any | None = None

            if initial_goal is None:
                goal_start_failures += 1
                try:
                    repair_result = lean.goal_via_file_context(
                        theorem_id,
                        str(thm.get("file_path", "") or ""),
                        [],
                        "",
                        project_root=args.lean_project,
                    )
                    if repair_result.success:
                        repaired_goal = repair_result.goal_state
                except Exception:
                    repair_result = None
                    repaired_goal = None

                start_row = _build_goal_start_failure_row(
                    thm,
                    lean,
                    repaired_goal=repaired_goal,
                    repair_result=repair_result,
                )
                start_handle.write(json.dumps(start_row) + "\n")
                start_handle.flush()

                if repaired_goal is None:
                    theorem_row = _skipped_row(thm, start_row)
                    theorem_row["difficulty_bucket"] = _difficulty_bucket(theorem_row)
                    details_handle.write(json.dumps(theorem_row) + "\n")
                    details_handle.flush()
                    skipped_start_count += 1
                    processed += 1
                    summary_rows.append(theorem_row)
                    _write_live_summary(
                        summary_path,
                        run_id="EXP-SOM-016-final-random-2000",
                        status="running",
                        sample_metadata=sample_metadata,
                        processed=processed,
                        summary_rows=summary_rows,
                        total_attempts=total_attempts,
                        started_at=started_at,
                        config_summary=config_summary,
                        artifact_summary=artifact_summary,
                    )
                    _write_json(report_path, json.loads(summary_path.read_text()))
                    continue

                goal_start_repairs += 1
                initial_goal = repaired_goal

            t0 = time.perf_counter()
            try:
                def _timeout_handler(signum: int, frame: object) -> None:  # noqa: ARG001
                    raise TimeoutError("theorem search exceeded per-theorem timeout")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.per_theorem_timeout)
                try:
                    # Disable Dr. Ducky inside first-order search — the bridge
                    # runs Ducky separately on the residual.  Keeping it in the
                    # search loop lets expensive Ducky tactics (aesop, apply?)
                    # trigger heartbeat timeouts that kill theorems the base
                    # search would otherwise prove.
                    saved_ducky = cfg.dr_ducky_enabled
                    cfg.dr_ducky_enabled = False
                    try:
                        result = search(
                            theorem_id=theorem_id,
                            initial_goal=initial_goal,
                            pipeline=pipeline,
                            conn=conn,
                            lean=lean,
                            config=cfg,
                            accessible_theorem_id=accessible_id,
                            sentence_encoder=sentence_encoder,
                        )
                    finally:
                        cfg.dr_ducky_enabled = saved_ducky
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except (Exception, TimeoutError) as exc:
                logger.warning("Search error on %s: %s", theorem_id, exc)
                result = SearchResult(
                    theorem_id=theorem_id,
                    success=False,
                    tactics_used=[],
                    attempts=0,
                    goals_closed=0,
                    goals_remaining=1,
                    close_provenance=[],
                )
            elapsed = time.perf_counter() - t0

            theorem_row = _theorem_row(
                thm=thm,
                result=result,
                elapsed_s=elapsed,
                initial_goal=str(initial_goal),
                start_row=start_row,
            )
            theorem_row["difficulty_bucket"] = _difficulty_bucket(theorem_row)
            theorem_row = _with_first_order_snapshot(theorem_row)
            if _bridge_eligible(theorem_row):
                bridge_t0 = time.perf_counter()
                try:
                    def _bridge_timeout_handler(signum: int, frame: object) -> None:  # noqa: ARG001
                        raise TimeoutError("hardtail bridge exceeded per-theorem timeout")

                    old_handler = signal.signal(signal.SIGALRM, _bridge_timeout_handler)
                    signal.alarm(args.per_theorem_timeout)
                    try:
                        bridge_result = run_hardtail_bridge_on_row(
                            theorem_row,
                            packet=None,
                            controller_runtime=controller_runtime,
                            pipeline=pipeline,
                            search_config=cfg,
                            conn=conn,
                            lean=lean,
                            theorem_id_map=name_to_id if name_to_id else None,
                            sentence_encoder=sentence_encoder,
                            disabled_ducky_tactics=set(_BRIDGE_DISABLED_DUCKY_TACTICS),
                        )
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                except (Exception, TimeoutError) as exc:
                    logger.warning("Hardtail bridge error on %s: %s", theorem_id, exc)
                    bridge_result = None
                bridge_elapsed = time.perf_counter() - bridge_t0
                if bridge_result is not None:
                    bridge_payload = bridge_result.to_dict()
                    theorem_row = _apply_bridge_result(
                        theorem_row,
                        bridge_payload,
                        bridge_time_s=bridge_elapsed,
                    )
                    theorem_row["difficulty_bucket"] = _difficulty_bucket(theorem_row)
                    bridge_handle.write(json.dumps(bridge_payload) + "\n")
                    bridge_handle.flush()
                    controller_handle.write(
                        json.dumps(
                            {
                                "theorem_id": theorem_id,
                                "controller_decision": bridge_payload.get("controller_decision") or {},
                            }
                        )
                        + "\n"
                    )
                    controller_handle.flush()
                    if bridge_payload.get("rarified_gap_packet"):
                        rarified_handle.write(json.dumps(bridge_payload["rarified_gap_packet"]) + "\n")
                        rarified_handle.flush()
                    bridge_invoked += 1
                    bridge_progressed += int(bool(theorem_row.get("bridge_progressed")))
                    bridge_closed += int(bool(theorem_row.get("bridge_closed")))
            details_handle.write(json.dumps(theorem_row) + "\n")
            if (idx + 1) % max(args.flush_every, 1) == 0:
                details_handle.flush()
            else:
                details_handle.flush()

            processed += 1
            processed_theorem_ids.add(theorem_id)
            total_attempts += int(theorem_row.get("attempts", 0) or 0)
            if theorem_row.get("success"):
                raw_success += 1
            if theorem_row.get("honest_success"):
                honest_success += 1
            if theorem_row.get("started"):
                started_count += 1
            else:
                skipped_start_count += 1
            if theorem_row.get("follow_on_stage") == "hard_proof_solver":
                hard_residual_count += 1
            summary_rows.append(theorem_row)

            if selector is not None and selector_enc is not None and sentence_encoder is not None:
                premise_names: list[str] = []
                if accessible_id is not None:
                    premise_ids = get_accessible_premises(conn, accessible_id)
                    premise_names = [id_to_name[pid] for pid in premise_ids if pid in id_to_name]
                trigger_rows = extract_trigger_states(
                    theorem_id=theorem_id,
                    step_trace=list(getattr(result, "step_trace", [])),
                    theorem_success=result.success,
                    accessible_premises=premise_names,
                    selector=selector,
                    selector_enc=selector_enc,
                    sentence_encoder=sentence_encoder,
                    lean=lean if args.probe_lean else None,
                    probe_lean=args.probe_lean,
                    probe_k=args.probe_k,
                )
                for row in trigger_rows:
                    row["template_id"] = thm.get("template_id", "")
                    row["difficulty_band"] = thm.get("difficulty_band", "")
                    row["proof_steps"] = thm.get("proof_steps", 0)
                    row["split"] = thm.get("split", "")
                    row["difficulty_bucket"] = theorem_row["difficulty_bucket"]
                    trigger_handle.write(json.dumps(row) + "\n")
                    trigger_rows_total += 1
                    trigger_positive += int(row.get("trigger_label") or row.get("can_apply") or 0)
                if trigger_rows:
                    trigger_handle.flush()

            _write_live_summary(
                summary_path,
                run_id="EXP-SOM-016-final-random-2000",
                status="running",
                sample_metadata=sample_metadata,
                processed=processed,
                summary_rows=summary_rows,
                total_attempts=total_attempts,
                started_at=started_at,
                config_summary=config_summary,
                artifact_summary=artifact_summary,
            )
            _write_json(report_path, json.loads(summary_path.read_text()))

    report = json.loads(summary_path.read_text())
    report["status"] = "materializing" if not args.no_materialize else "complete"
    report["collection"] = {
        "goal_start_failures": goal_start_failures,
        "goal_start_repairs": goal_start_repairs,
        "trigger_rows": trigger_rows_total,
        "trigger_positive": trigger_positive,
    }

    if not args.no_materialize:
        bundle_dir = output_dir / "bundle"
        bundle_summary = build_hard_collection_bundle(
            inputs=[details_path],
            output_dir=bundle_dir,
            min_trace_length=args.min_trace_length,
            min_strategy_support=args.min_strategy_support,
        )
        bundle_rows = load_resolution_rows(bundle_dir / "collection_all.jsonl")
        hard_resolution_summary = materialize_hard_resolution_layer(
            rows=bundle_rows,
            output_dir=bundle_dir / "hard_resolution_layer",
            conn_or_db=conn,
            candidate_limit=args.hard_resolution_candidate_limit,
            exemplar_limit=args.hard_resolution_exemplar_limit,
        )
        second_order_summary = build_second_order_packet_freeze(
            run_dir=output_dir,
            output_dir=bundle_dir / "second_order_som",
        )
        second_order_feature_summary = build_second_order_feature_dataset(
            packets_path=bundle_dir / "second_order_som" / "second_order_packets.jsonl",
            output_dir=bundle_dir / "second_order_som" / "features",
        )
        report["artifacts"].update(
            {
                "bundle_dir": str(bundle_dir),
                "hard_resolution_layer_dir": str(bundle_dir / "hard_resolution_layer"),
                "second_order_som_dir": str(bundle_dir / "second_order_som"),
            }
        )
        report["collection"].update(
            {
                "bundle_summary": bundle_summary,
                "hard_resolution_summary": hard_resolution_summary,
                "second_order_packet_summary": second_order_summary,
                "second_order_feature_summary": second_order_feature_summary,
            }
        )

    report["status"] = "complete"
    _write_json(summary_path, report)
    _write_json(report_path, report)
    _write_monitor(
        monitor_path,
        total=len(theorems),
        processed=processed,
        raw_success=raw_success,
        honest_success=honest_success,
        started=started_count,
        skipped_start=skipped_start_count,
        goal_start_failures=goal_start_failures,
        goal_start_repairs=goal_start_repairs,
        hard_residuals=hard_residual_count,
        trigger_rows=trigger_rows_total,
        total_attempts=total_attempts,
        started_at=started_at,
        current_theorem="(complete)",
        sample_manifest_path=sample_manifest_path,
        details_path=details_path,
        goal_start_path=goal_start_path,
        trigger_path=trigger_path,
        log_path=log_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    lean.close()
    conn.close()

    print("=" * 72)
    print("EXP-SOM-016 Final Random Mathlib Benchmark")
    print("=" * 72)
    print(
        f"  honest_success={report['benchmark']['honest_success']}/"
        f"{report['benchmark']['total_theorems']}"
    )
    print(
        f"  started={report['benchmark']['started_theorems']} "
        f"skip={report['benchmark']['skipped_start']}"
    )
    print(
        f"  hard_residuals={report['benchmark']['residual_structure']['by_follow_on_stage'].get('hard_proof_solver', 0)}"
    )
    print(f"  details={details_path}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
