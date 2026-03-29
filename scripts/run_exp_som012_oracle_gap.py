"""EXP-SOM-012 Stage 2: oracle-gap audit on hard residuals.

The oracle-gap audit asks a narrower question than the full hard-proof solver:

- Can the reduced goal be started at all? (`startability`)
- Do we already know the residual family well enough to route it? (`routing`)
- If we had perfect access to the current symbolic packet, can one of the top
  candidate priors make progress? (`premise_or_unfold_choice`)
- Can a small family-specific closer bank finish the reduced goal without
  theorem replay? (`final_closer_choice`)

This script uses only symbolic packets plus Lean verification. It does not
train or invoke a new learned hard-proof controller.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark import _build_search_components
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text
from src.hard_resolution_layer import load_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _file_path_for_theorem(row: dict[str, Any], lean: Any) -> str:
    file_path = str(row.get("lean_path", "") or row.get("file_path", "") or "")
    if file_path:
        return file_path
    if getattr(lean, "_backend", "") != "pantograph" or getattr(lean, "_server", None) is None:
        return ""
    theorem_id = str(row.get("theorem_id", "") or "")
    if not theorem_id:
        return ""
    try:
        info = lean._server.env_inspect(theorem_id)
        module = info.get("module", "")
        if module:
            return module.replace(".", "/") + ".lean"
    except Exception:
        pass
    return ""


def _reduced_goal_candidates(row: dict[str, Any]) -> list[tuple[str, str]]:
    candidates = [
        ("last_goal", str(row.get("last_goal", "") or "").strip()),
        ("goal_state", str(row.get("goal_state", "") or "").strip()),
        ("theorem_statement", str(row.get("theorem_statement", "") or "").strip()),
        ("initial_goal", str(row.get("initial_goal", "") or "").strip()),
    ]
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate[1] and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _start_residual_goal(row: dict[str, Any], lean: Any) -> tuple[str | None, str, str]:
    theorem_id = str(row.get("theorem_id", "") or "")
    file_path = _file_path_for_theorem(row, lean)
    failures: list[str] = []
    for kind, text in _reduced_goal_candidates(row):
        if getattr(lean, "_backend", "") != "pantograph":
            return text, kind, ""
        try:
            return lean.goal_start(text, theorem_name=theorem_id, file_path=file_path), kind, ""
        except Exception as exc:
            failures.append(f"{kind}:{type(exc).__name__}")
    return None, "", ",".join(failures[:4])


def _domain_hints(theorem_id: str, goal_text: str) -> set[str]:
    source = " ".join([sanitize_goal_text(theorem_id or ""), sanitize_goal_text(goal_text or "")])
    hints: set[str] = set()
    checks = {
        "category_theory": ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"],
        "algebraic_geometry": ["AlgebraicGeometry", "Scheme", "LocallyRingedSpace", "PrimeSpectrum", "HomogeneousLocalization"],
        "abstract_algebra": ["IsIntegral", "traceMatrix", "Matrix.det", "discr", "FormallyUnramified", "HasRingHomProperty"],
        "geometric_analysis": ["Besicovitch", "dist", "Metric", "δ", "norm", "ball"],
        "witness_goal": ["sSup", "sInf", "iSup", "iInf", "range", "image", "∃", "Exists"],
        "membership_wall": [".carrier", "Submodule", "Ideal", "PrimeSpectrum", "HomogeneousLocalization"],
    }
    for label, markers in checks.items():
        if label == "membership_wall":
            if "∈" in source and any(marker in source for marker in markers):
                hints.add(label)
            continue
        if any(marker in source for marker in markers):
            hints.add(label)
    return hints


def closer_candidates(
    last_goal_bucket: str,
    reasoning_gap_family: str = "",
    goal_text: str = "",
    theorem_id: str = "",
) -> list[str]:
    shared = ["solve_by_elim", "aesop", "simp", "simpa", "assumption", "apply?"]
    bucket_specific = {
        "equality": ["rfl", "norm_num", "ring", "ext"],
        "inequality": ["linarith", "omega", "nlinarith", "norm_num", "gcongr", "positivity"],
        "false": ["contradiction", "tauto", "omega", "linarith"],
        "forall": ["intro", "solve_by_elim", "aesop", "simpa"],
        "exists": ["solve_by_elim", "aesop", "constructor"],
        "iff": ["constructor", "all_goals solve_by_elim", "aesop", "tauto"],
        "membership": ["simp", "simpa", "solve_by_elim", "aesop"],
        "subset": ["intro", "solve_by_elim", "simp", "aesop"],
        "atomic_prop": ["trivial", "solve_by_elim", "aesop", "decide"],
        "negation": ["intro", "contradiction", "solve_by_elim", "aesop"],
    }
    family_specific = {
        "local_ineq_close": ["gcongr", "positivity", "linarith", "nlinarith", "omega"],
        "witness_construction_close": ["use ?_", "refine ⟨?_, ?_⟩", "constructor", "aesop"],
        "exists_close": ["use 0", "refine ⟨0, ?_⟩", "constructor", "aesop"],
        "forward_context_close": ["apply", "exact", "apply?", "solve_by_elim", "simpa", "aesop"],
        "small_multigoal_side_conditions": ["assumption", "simpa", "infer_instance", "aesop"],
        "membership_close": ["simpa", "exact?", "apply?", "solve_by_elim", "aesop"],
        "local_eq_close": ["exact?", "apply?", "solve_by_elim", "simpa", "ext", "congr"],
    }
    out = _unique(family_specific.get(reasoning_gap_family, []) + bucket_specific.get(last_goal_bucket, []) + shared)
    if {"category_theory", "algebraic_geometry", "abstract_algebra", "membership_wall"} & _domain_hints(theorem_id, goal_text):
        banned = {"norm_num", "ring", "omega", "linarith", "nlinarith", "positivity"}
        out = [tactic for tactic in out if tactic not in banned]
    return out


def premise_tactic_candidates(lemma: str, reasoning_gap_family: str) -> list[str]:
    lemma = (lemma or "").strip()
    if not lemma:
        return []
    generic = [
        f"rw [{lemma}]",
        f"rw [← {lemma}]",
        f"simp [{lemma}]",
        f"simpa using {lemma}",
        f"exact {lemma}",
        f"apply {lemma}",
    ]
    family_specific: dict[str, list[str]] = {
        "local_eq_close": [f"exact {lemma}", f"apply {lemma}", f"simpa using {lemma}", f"rw [{lemma}]", f"rw [← {lemma}]"],
        "local_ineq_close": [f"rw [{lemma}]", f"simpa using {lemma}", f"apply {lemma}"],
        "membership_close": [f"exact {lemma}", f"apply {lemma}", f"simpa using {lemma}", f"simp [{lemma}]"],
        "subset_close": [f"simpa using {lemma}", f"apply {lemma}"],
        "small_multigoal_planner": [f"apply {lemma}", f"refine {lemma}", f"simp [{lemma}]"],
        "small_multigoal_side_conditions": [f"simpa using {lemma}", f"exact {lemma}", f"apply {lemma}"],
        "forward_context_close": [f"apply {lemma}", f"exact {lemma}", f"simpa using {lemma}"],
        "forall_close": [f"apply {lemma}", f"simpa using {lemma}"],
        "exists_close": [f"exact {lemma}", f"apply {lemma}", "use 0", "refine ⟨0, ?_⟩"],
        "witness_construction_close": [f"apply {lemma}", f"refine {lemma}", f"exact {lemma}"],
    }
    return _unique(family_specific.get(reasoning_gap_family, []) + generic)


def _tactic_progress(result: Any, original_goal: str) -> tuple[bool, bool]:
    if not getattr(result, "success", False):
        return False, False
    new_goals = list(getattr(result, "new_goals", []) or [])
    if not new_goals:
        return True, True
    if len(new_goals) == 1 and str(new_goals[0]) == original_goal:
        return False, False
    return True, False


def _probe_tactics(lean: Any, goal: str, tactics: list[str]) -> dict[str, Any]:
    for tactic in tactics:
        try:
            result = lean.try_tactic(goal, tactic)
        except Exception:
            continue
        progress, closed = _tactic_progress(result, goal)
        if progress:
            return {
                "progress": True,
                "closed": closed,
                "tactic": tactic,
                "new_goals": list(getattr(result, "new_goals", []) or []),
            }
    return {"progress": False, "closed": False, "tactic": "", "new_goals": []}


def _try_close_goal_bundle(
    lean: Any,
    goals: list[str],
    bucket: str,
    reasoning_gap_family: str = "",
    theorem_id: str = "",
) -> tuple[bool, list[str]]:
    chosen: list[str] = []
    if not goals or len(goals) > 3:
        return False, chosen
    for goal in goals:
        closers = closer_candidates(
            classify_goal_bucket(goal) or bucket,
            reasoning_gap_family=reasoning_gap_family,
            goal_text=goal,
            theorem_id=theorem_id,
        )
        probe = _probe_tactics(lean, goal, closers)
        if not probe["closed"]:
            return False, chosen
        chosen.append(str(probe["tactic"]))
    return True, chosen


def audit_oracle_row(row: dict[str, Any], packet: dict[str, Any] | None, lean: Any) -> dict[str, Any]:
    theorem_id = str(row.get("theorem_id", "") or "")
    started_goal, goal_kind, start_failure = _start_residual_goal(row, lean)
    last_goal_bucket = str(row.get("last_goal_bucket", "") or "other")
    family = str(row.get("reasoning_gap_family", "") or "")
    out = {
        "theorem_id": theorem_id,
        "hard_track": str(row.get("hard_track", "")),
        "residual_bucket": str(row.get("residual_bucket", "")),
        "reasoning_gap_family": family,
        "last_goal_bucket": last_goal_bucket,
        "startable": started_goal is not None,
        "start_goal_kind": goal_kind,
        "start_failure": start_failure,
        "routing_label_available": bool(family and family != "none"),
        "resolution_packet_available": bool(packet),
        "premise_progress": False,
        "premise_close": False,
        "premise_tactic": "",
        "premise_lemma": "",
        "closer_close": False,
        "closer_tactic": "",
        "combined_close": False,
        "combined_sequence": [],
    }
    if started_goal is None:
        return out

    closer_probe = _probe_tactics(
        lean,
        started_goal,
        closer_candidates(
            last_goal_bucket,
            reasoning_gap_family=family,
            goal_text=started_goal,
            theorem_id=theorem_id,
        ),
    )
    out["closer_close"] = bool(closer_probe["closed"])
    out["closer_tactic"] = str(closer_probe["tactic"])

    candidate_priors = list((packet or {}).get("candidate_priors", []) or [])
    for prior in candidate_priors[:5]:
        lemma = str(prior.get("lemma", "") or "").strip()
        if not lemma or lemma == theorem_id:
            continue
        tactics = premise_tactic_candidates(lemma, family)
        probe = _probe_tactics(lean, started_goal, tactics)
        if not probe["progress"]:
            continue
        out["premise_progress"] = True
        out["premise_tactic"] = str(probe["tactic"])
        out["premise_lemma"] = lemma
        if probe["closed"]:
            out["premise_close"] = True
            out["combined_close"] = True
            out["combined_sequence"] = [str(probe["tactic"])]
            break
        new_goals = [str(goal) for goal in probe["new_goals"] if str(goal).strip()]
        bundle_closed, chosen = _try_close_goal_bundle(
            lean,
            new_goals,
            last_goal_bucket,
            reasoning_gap_family=family,
            theorem_id=theorem_id,
        )
        if bundle_closed:
            out["combined_close"] = True
            out["combined_sequence"] = [str(probe["tactic"])] + chosen
            break

    return out


def summarize_oracle_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _count(key: str) -> int:
        return sum(1 for row in rows if row.get(key))

    by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_track: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        family = str(row.get("reasoning_gap_family", "") or "")
        track = str(row.get("hard_track", "") or "")
        for bucket in (family, track):
            target = by_family if bucket == family else by_track
            if bucket:
                target[bucket]["count"] += 1
                for key in (
                    "startable",
                    "resolution_packet_available",
                    "premise_progress",
                    "premise_close",
                    "closer_close",
                    "combined_close",
                ):
                    target[bucket][key] += int(bool(row.get(key)))

    return {
        "total_theorems": len(rows),
        "startability_oracle": _count("startable"),
        "routing_label_available": _count("routing_label_available"),
        "resolution_packets_available": _count("resolution_packet_available"),
        "premise_progress": _count("premise_progress"),
        "premise_close": _count("premise_close"),
        "final_closer_close": _count("closer_close"),
        "combined_close": _count("combined_close"),
        "by_reasoning_gap_family": {k: dict(v) for k, v in by_family.items()},
        "by_hard_track": {k: dict(v) for k, v in by_track.items()},
    }


class _TimeoutError(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _TimeoutError("oracle theorem timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="models/NAV-004_step5000.pt")
    parser.add_argument("--inputs", required=True, help="Hard residual JSONL input")
    parser.add_argument(
        "--resolution-packets",
        default="",
        help="Optional symbolic hard-resolution packet JSONL",
    )
    parser.add_argument("--output-dir", default="runs/exp_som012_oracle_gap")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="pantograph")
    parser.add_argument("--lean-project", default="data/lean_project")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--per-theorem-timeout", type=int, default=90)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    config.setdefault("lean", {})["project_root"] = args.lean_project
    config["lean"]["backend"] = args.backend
    config["lean"]["imports"] = ["Mathlib"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.inputs))
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    packets_by_theorem: dict[str, dict[str, Any]] = {}
    if args.resolution_packets:
        for packet in load_jsonl(Path(args.resolution_packets)):
            theorem_id = str(packet.get("theorem_id", "") or "")
            if theorem_id:
                packets_by_theorem[theorem_id] = packet

    _pipeline, _cfg, lean, _lean_cfg, conn = _build_search_components(
        config,
        Path(args.checkpoint),
        args.device,
    )
    del conn  # lean goal/tactic probes only
    if lean._backend == "pantograph":
        lean._ensure_server()

    started_at = time.time()
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        theorem_id = str(row.get("theorem_id", "") or "")
        logger.info("[%d/%d] oracle-gap %s", idx + 1, len(rows), theorem_id)
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(args.per_theorem_timeout)
            try:
                out = audit_oracle_row(row, packets_by_theorem.get(theorem_id), lean)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (Exception, _TimeoutError) as exc:
            logger.warning("Oracle-gap error on %s: %s", theorem_id, exc)
            out = {
                "theorem_id": theorem_id,
                "hard_track": str(row.get("hard_track", "")),
                "residual_bucket": str(row.get("residual_bucket", "")),
                "reasoning_gap_family": str(row.get("reasoning_gap_family", "")),
                "last_goal_bucket": str(row.get("last_goal_bucket", "")),
                "startable": False,
                "start_goal_kind": "",
                "start_failure": type(exc).__name__,
                "routing_label_available": bool(row.get("reasoning_gap_family")),
                "resolution_packet_available": theorem_id in packets_by_theorem,
                "premise_progress": False,
                "premise_close": False,
                "premise_tactic": "",
                "premise_lemma": "",
                "closer_close": False,
                "closer_tactic": "",
                "combined_close": False,
                "combined_sequence": [],
            }
        out_rows.append(out)

    summary = summarize_oracle_rows(out_rows)
    summary.update(
        {
            "experiment": "EXP-SOM-012-oracle-gap",
            "inputs": args.inputs,
            "resolution_packets": args.resolution_packets,
            "elapsed_s": round(time.time() - started_at, 2),
        }
    )
    (output_dir / "oracle_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in out_rows))
    (output_dir / "oracle_gap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
