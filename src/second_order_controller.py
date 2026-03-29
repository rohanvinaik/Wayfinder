from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _sorted_counter_keys(counter: dict[str, Any] | None) -> list[str]:
    if not counter:
        return []
    ranked = sorted(
        ((str(name or ""), int(count or 0)) for name, count in counter.items() if str(name or "").strip()),
        key=lambda item: (-item[1], item[0]),
    )
    return [name for name, _count in ranked]


@dataclass
class SecondOrderDecision:
    theorem_id: str
    packet_kind: str
    residual_bucket: str
    goal_bucket: str
    controller_mode: str = "deterministic_packet_policy_v1"
    invoke_first_ducky: bool = True
    first_pass_backends: list[str] = field(default_factory=list)
    first_pass_engines: list[str] = field(default_factory=list)
    search_budget: int = 120
    search_max_progress_steps: int = 8
    cosine_rw_beam: int = 5
    cosine_apply_beam: int = 5
    enable_family_router: bool = True
    enable_cosine_rw_seq: bool = True
    enable_cosine_simp: bool = True
    enable_interleaved_bootstrap: bool = True
    enable_cosine_apply: bool = True
    gated_cosine_apply: bool = True
    second_pass_backends: list[str] = field(default_factory=list)
    second_pass_engines: list[str] = field(default_factory=list)
    second_pass_goal_limit: int = 3
    rarified_target: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_GOAL_BUCKET_BACKENDS: dict[str, list[str]] = {
    "equality": ["egglog_eqsat", "rosette_proof_dsl", "lean_arith"],
    "inequality": ["lean_arith", "egglog_eqsat", "rosette_proof_dsl"],
    "membership": ["kodkod_relational", "rosette_proof_dsl"],
    "subset": ["kodkod_relational", "rosette_proof_dsl"],
    "exists": ["rosette_proof_dsl", "kodkod_relational"],
    "forall": ["rosette_proof_dsl", "egglog_eqsat"],
    "iff": ["rosette_proof_dsl", "egglog_eqsat"],
    "atomic_prop": ["rosette_proof_dsl"],
}

_GOAL_BUCKET_ENGINES: dict[str, list[str]] = {
    "equality": ["EqSatEngine", "ContextTransportEngine", "ArithEngine"],
    "inequality": ["ArithEngine", "ContextTransportEngine", "EqSatEngine"],
    "membership": ["FiniteFilterEngine", "ContextTransportEngine"],
    "subset": ["FiniteFilterEngine", "ContextTransportEngine"],
    "exists": ["WitnessEngine", "ContextTransportEngine", "FiniteFilterEngine"],
    "forall": ["ContextTransportEngine", "EqSatEngine"],
    "iff": ["ContextTransportEngine", "EqSatEngine"],
    "atomic_prop": ["ContextTransportEngine"],
}

_RESIDUAL_BUDGETS: dict[str, tuple[int, int]] = {
    "single_goal_near_miss": (90, 6),
    "single_goal_stall": (120, 8),
    "multi_goal_small_progress": (150, 10),
    "multi_goal_large_progress": (180, 12),
    "skipped_start": (0, 0),
}


def load_second_order_packet_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = str(row.get("theorem_id", "") or "")
            if theorem_id:
                out[theorem_id] = row
    return out


def derive_second_order_decision(
    row: dict[str, Any],
    packet: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> SecondOrderDecision:
    decision = _derive_packet_policy_decision(row, packet)
    if runtime is None:
        return decision
    return _apply_learned_runtime(decision, row=row, packet=packet, runtime=runtime)


def _derive_packet_policy_decision(
    row: dict[str, Any],
    packet: dict[str, Any] | None = None,
) -> SecondOrderDecision:
    theorem_id = str(row.get("theorem_id", "") or "")
    packet = packet or {}
    labels = packet.get("second_order_labels", {}) or {}
    ducky_surface = packet.get("ducky_outcome_surface", {}) or {}
    hard_surface = packet.get("hard_som_surface", {}) or {}
    residual_bucket = str(
        row.get("residual_bucket")
        or packet.get("residual_bucket")
        or ducky_surface.get("residual_bucket")
        or ""
    )
    goal_text = str(row.get("last_goal") or row.get("goal_state") or "")
    goal_bucket = str(
        row.get("last_goal_bucket")
        or packet.get("goal_bucket")
        or ducky_surface.get("goal_bucket")
        or classify_goal_bucket(goal_text)
    )
    search_budget, max_progress = _RESIDUAL_BUDGETS.get(residual_bucket, (120, 8))

    first_pass_backends = _unique(
        list(labels.get("backend_budget_targets") or [])
        + _sorted_counter_keys(ducky_surface.get("backend_family_counts"))
        + _GOAL_BUCKET_BACKENDS.get(goal_bucket, [])
    )
    first_pass_engines = _unique(
        list(labels.get("engine_family_budget_targets") or [])
        + _sorted_counter_keys(ducky_surface.get("engine_counts"))
        + _GOAL_BUCKET_ENGINES.get(goal_bucket, [])
    )

    resolution_family = str(packet.get("resolution_family", "") or hard_surface.get("resolution_family", "") or "")
    specialist_targets = list((hard_surface.get("proof_plan_geometry") or {}).get("specialist_targets") or [])
    lane_hints = list((hard_surface.get("proof_plan_geometry") or {}).get("lane_suppression_hints") or [])
    recursive_signal = "recursive" in resolution_family or "root" in sanitize_goal_text(goal_text)
    structural_signal = goal_bucket in {"membership", "subset", "exists"} or "structural_property" in " ".join(
        str(item) for item in specialist_targets
    )

    second_pass_backends = list(first_pass_backends)
    second_pass_engines = list(first_pass_engines)
    if recursive_signal:
        second_pass_backends = _unique(["symbolic_rewrite_vm", "rosette_proof_dsl"] + second_pass_backends)
        second_pass_engines = _unique(["RecursiveInvariantEngine", "ContextTransportEngine"] + second_pass_engines)
    elif structural_signal:
        second_pass_backends = _unique(["kodkod_relational", "rosette_proof_dsl"] + second_pass_backends)
        second_pass_engines = _unique(["FiniteFilterEngine", "WitnessEngine", "ContextTransportEngine"] + second_pass_engines)
    else:
        second_pass_backends = _unique(_GOAL_BUCKET_BACKENDS.get(goal_bucket, []) + second_pass_backends)
        second_pass_engines = _unique(_GOAL_BUCKET_ENGINES.get(goal_bucket, []) + second_pass_engines)

    return SecondOrderDecision(
        theorem_id=theorem_id,
        packet_kind=str(packet.get("packet_kind", "hard_residual") or "hard_residual"),
        residual_bucket=residual_bucket,
        goal_bucket=goal_bucket,
        invoke_first_ducky=bool(labels.get("invoke_ducky", True)),
        first_pass_backends=first_pass_backends or ["rosette_proof_dsl", "egglog_eqsat", "lean_arith"],
        first_pass_engines=first_pass_engines or ["ContextTransportEngine", "EqSatEngine", "ArithEngine"],
        search_budget=search_budget,
        search_max_progress_steps=max_progress,
        cosine_rw_beam=5 if goal_bucket in {"equality", "inequality", "iff"} else 3,
        cosine_apply_beam=6 if goal_bucket in {"forall", "atomic_prop", "iff"} else 5,
        enable_family_router=True,
        enable_cosine_rw_seq=True,
        enable_cosine_simp=goal_bucket in {"equality", "inequality", "atomic_prop", "iff"},
        enable_interleaved_bootstrap=True,
        enable_cosine_apply=goal_bucket not in {"membership", "subset"},
        gated_cosine_apply=True,
        second_pass_backends=second_pass_backends,
        second_pass_engines=second_pass_engines,
        second_pass_goal_limit=3 if residual_bucket.startswith("multi_goal") else 1,
        rarified_target=_rarified_target(goal_bucket, resolution_family, recursive_signal, structural_signal),
        provenance={
            "resolution_family": resolution_family,
            "specialist_targets": specialist_targets,
            "lane_suppression_hints": lane_hints,
            "projector_rejection_seen": bool(labels.get("projector_rejection_seen")),
        },
    )


def refine_second_pass_decision(
    decision: SecondOrderDecision,
    final_goals: list[str],
    *,
    search_trace: list[dict[str, Any]] | None = None,
    row: dict[str, Any] | None = None,
    packet: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> SecondOrderDecision:
    if not final_goals:
        return decision
    goal_bucket_counts: dict[str, int] = {}
    for goal in final_goals:
        bucket = classify_goal_bucket(goal)
        goal_bucket_counts[bucket] = goal_bucket_counts.get(bucket, 0) + 1
    dominant_bucket = sorted(goal_bucket_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    trace = search_trace or []
    close_lanes = [str(step.get("lane", "") or "") for step in trace if str(step.get("lane", "") or "")]
    refine_note = {
        "dominant_post_search_bucket": dominant_bucket,
        "post_search_goal_count": len(final_goals),
        "close_lanes_seen": _unique(close_lanes),
    }
    second_backends = _unique(_GOAL_BUCKET_BACKENDS.get(dominant_bucket, []) + list(decision.second_pass_backends))
    second_engines = _unique(_GOAL_BUCKET_ENGINES.get(dominant_bucket, []) + list(decision.second_pass_engines))
    target = _rarified_target(
        dominant_bucket,
        str(decision.provenance.get("resolution_family", "") or ""),
        "symbolic_rewrite_vm" in second_backends,
        dominant_bucket in {"membership", "subset", "exists"},
    )
    updated = SecondOrderDecision(**decision.to_dict())
    updated.second_pass_backends = second_backends
    updated.second_pass_engines = second_engines
    updated.second_pass_goal_limit = min(max(len(final_goals), 1), 3)
    updated.rarified_target = target
    updated.provenance = dict(decision.provenance)
    updated.provenance["post_search_refinement"] = refine_note
    if runtime is not None:
        synth_packet = _synthesized_runtime_packet(
            row=row or {},
            packet=packet or {},
            goal_bucket=dominant_bucket,
            final_goals=final_goals,
        )
        updated = _apply_learned_runtime(updated, row=row or {}, packet=synth_packet, runtime=runtime)
        updated.provenance = dict(updated.provenance)
        updated.provenance["post_search_refinement"] = refine_note
    return updated


def _top_or_threshold(scores: dict[str, float], *, threshold: float = 0.4, fallback_k: int = 2) -> list[str]:
    ranked = sorted(
        ((str(name or ""), float(score or 0.0)) for name, score in scores.items() if str(name or "").strip()),
        key=lambda item: (-item[1], item[0]),
    )
    chosen = [name for name, score in ranked if score >= threshold]
    if chosen:
        return chosen
    return [name for name, _score in ranked[:fallback_k]]


def _synthesized_runtime_packet(
    *,
    row: dict[str, Any],
    packet: dict[str, Any],
    goal_bucket: str,
    final_goals: list[str],
) -> dict[str, Any]:
    synth = dict(packet)
    synth.setdefault("packet_kind", packet.get("packet_kind", "hard_residual") or "hard_residual")
    synth["theorem_id"] = str(row.get("theorem_id", "") or synth.get("theorem_id", "") or "")
    synth["residual_bucket"] = str(row.get("residual_bucket", "") or synth.get("residual_bucket", "") or "")
    synth["goal_bucket"] = goal_bucket
    hard_surface = dict(synth.get("hard_som_surface", {}) or {})
    residual_geo = dict(hard_surface.get("residual_skeleton_geometry", {}) or {})
    residual_geo["goal_shape_features"] = {
        **dict(residual_geo.get("goal_shape_features", {}) or {}),
        "goal_count": len(final_goals),
        "char_len": max((len(goal) for goal in final_goals), default=0),
    }
    hard_surface["residual_skeleton_geometry"] = residual_geo
    synth["hard_som_surface"] = hard_surface
    return synth


def _apply_learned_runtime(
    decision: SecondOrderDecision,
    *,
    row: dict[str, Any],
    packet: dict[str, Any] | None,
    runtime: Any,
) -> SecondOrderDecision:
    packet_surface = packet or _synthesized_runtime_packet(
        row=row,
        packet={},
        goal_bucket=str(row.get("last_goal_bucket", "") or decision.goal_bucket),
        final_goals=[str(row.get("last_goal", "") or row.get("goal_state", "") or "")],
    )
    prediction = runtime.predict_packet(packet_surface)
    learned_backends = _top_or_threshold(prediction.get("backend_probs", {}), threshold=0.38, fallback_k=2)
    learned_engines = _top_or_threshold(prediction.get("engine_probs", {}), threshold=0.38, fallback_k=2)
    invoke_prob = float(prediction.get("invoke_prob", 0.0) or 0.0)
    progress_prob = float(prediction.get("progress_prob", 0.0) or 0.0)
    projector_prob = float(prediction.get("projector_rejection_prob", 0.0) or 0.0)

    updated = SecondOrderDecision(**decision.to_dict())
    updated.controller_mode = "learned_second_order_som_v1"
    updated.invoke_first_ducky = bool(updated.invoke_first_ducky or invoke_prob >= 0.35)
    updated.first_pass_backends = _unique(learned_backends + list(updated.first_pass_backends))
    updated.first_pass_engines = _unique(learned_engines + list(updated.first_pass_engines))
    updated.second_pass_backends = _unique(learned_backends + list(updated.second_pass_backends))
    updated.second_pass_engines = _unique(learned_engines + list(updated.second_pass_engines))
    if progress_prob >= 0.65:
        updated.search_budget += 30
        updated.search_max_progress_steps += 2
    elif progress_prob <= 0.25:
        updated.search_budget = max(updated.search_budget - 20, 60)
    if projector_prob >= 0.5:
        updated.enable_cosine_simp = True
    updated.provenance = dict(updated.provenance)
    updated.provenance["learned_runtime"] = {
        "invoke_prob": round(invoke_prob, 4),
        "progress_prob": round(progress_prob, 4),
        "projector_rejection_prob": round(projector_prob, 4),
        "packet_kind_probs": prediction.get("packet_kind_probs", {}),
        "engine_probs": prediction.get("engine_probs", {}),
        "backend_probs": prediction.get("backend_probs", {}),
    }
    return updated


def _rarified_target(
    goal_bucket: str,
    resolution_family: str,
    recursive_signal: bool,
    structural_signal: bool,
) -> str:
    if recursive_signal:
        return "recursive_invariant_microtheory"
    if structural_signal:
        return "relational_structural_closer"
    if goal_bucket in {"equality", "inequality"}:
        return "algebraic_transport_closer"
    if goal_bucket in {"forall", "iff", "atomic_prop"}:
        return "context_transport_closer"
    if "compiler" in resolution_family:
        return "compiler_context_repair"
    return "global_replanner_tail"
