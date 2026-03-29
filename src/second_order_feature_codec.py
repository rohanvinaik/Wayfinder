from __future__ import annotations

from typing import Any


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def collect_packet_tokens(packet: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    packet_kind = str(packet.get("packet_kind", "") or "")
    residual_bucket = str(packet.get("residual_bucket", "") or "")
    goal_bucket = str(packet.get("goal_bucket", "") or "")
    resolution_family = str(packet.get("resolution_family", "") or "")
    difficulty_band = str(packet.get("difficulty_band", "") or "")
    for prefix, value in (
        ("packet_kind", packet_kind),
        ("residual_bucket", residual_bucket),
        ("goal_bucket", goal_bucket),
        ("resolution_family", resolution_family),
        ("difficulty_band", difficulty_band),
    ):
        if value:
            tokens.append(f"{prefix}:{value}")

    hard_surface = safe_dict(packet.get("hard_som_surface"))
    residual_geo = safe_dict(hard_surface.get("residual_skeleton_geometry"))
    plan_geo = safe_dict(hard_surface.get("proof_plan_geometry"))
    prior_geo = safe_dict(hard_surface.get("prior_graph_geometry"))
    theorem_surface = safe_dict(prior_geo.get("theorem_surface"))
    ducky_surface = safe_dict(packet.get("ducky_outcome_surface"))

    for key in residual_geo.get("domain_hints", []) or []:
        if key:
            tokens.append(f"domain:{key}")
    for key in residual_geo.get("representation_pressures", []) or []:
        if key:
            tokens.append(f"pressure:{key}")
    for key in residual_geo.get("top_symbols", []) or []:
        if key:
            tokens.append(f"symbol:{key}")
    for key in plan_geo.get("candidate_methods", []) or []:
        if key:
            tokens.append(f"method:{key}")
    for key in plan_geo.get("specialist_targets", []) or []:
        if key:
            tokens.append(f"specialist:{key}")
    for key in plan_geo.get("lane_suppression_hints", []) or []:
        if key:
            tokens.append(f"suppress:{key}")
    for key in plan_geo.get("lane_history", []) or []:
        if key:
            tokens.append(f"lane:{key}")
    for key in theorem_surface.get("anchor_labels", []) or []:
        if key:
            tokens.append(f"anchor:{key}")
    for key in ducky_surface.get("engine_counts", {}) or {}:
        if key:
            tokens.append(f"engine_seen:{key}")
    for key in ducky_surface.get("backend_family_counts", {}) or {}:
        if key:
            tokens.append(f"backend_seen:{key}")
    return tokens


def numeric_packet_features(packet: dict[str, Any]) -> dict[str, float]:
    hard_surface = safe_dict(packet.get("hard_som_surface"))
    residual_geo = safe_dict(hard_surface.get("residual_skeleton_geometry"))
    plan_geo = safe_dict(hard_surface.get("proof_plan_geometry"))
    search_geo = safe_dict(plan_geo.get("search_control_geometry"))
    theorem_surface = safe_dict(safe_dict(hard_surface.get("prior_graph_geometry")).get("theorem_surface"))
    ducky_surface = safe_dict(packet.get("ducky_outcome_surface"))
    labels = safe_dict(packet.get("second_order_labels"))

    goal_shape = safe_dict(residual_geo.get("goal_shape_features"))
    theorem_shape = safe_dict(residual_geo.get("theorem_shape_features"))

    return {
        "goal_char_len": float(goal_shape.get("char_len", 0) or 0),
        "goal_token_len": float(goal_shape.get("token_len", 0) or 0),
        "goal_binder_count": float(goal_shape.get("binder_count", 0) or 0),
        "goal_forall_count": float(goal_shape.get("forall_count", 0) or 0),
        "goal_exists_count": float(goal_shape.get("exists_count", 0) or 0),
        "goal_eq_count": float(goal_shape.get("eq_count", 0) or 0),
        "goal_iff_count": float(goal_shape.get("iff_count", 0) or 0),
        "goal_membership_count": float(goal_shape.get("membership_count", 0) or 0),
        "goal_subset_count": float(goal_shape.get("subset_count", 0) or 0),
        "theorem_char_len": float(theorem_shape.get("char_len", 0) or 0),
        "theorem_token_len": float(theorem_shape.get("token_len", 0) or 0),
        "theorem_binder_count": float(theorem_shape.get("binder_count", 0) or 0),
        "symbol_count": float(residual_geo.get("symbol_count", 0) or 0),
        "domain_hint_count": float(len(residual_geo.get("domain_hints", []) or [])),
        "representation_pressure_count": float(len(residual_geo.get("representation_pressures", []) or [])),
        "candidate_method_count": float(len(plan_geo.get("candidate_methods", []) or [])),
        "lane_history_count": float(plan_geo.get("lane_count", 0) or 0),
        "bridge_pressure": float(plan_geo.get("bridge_pressure", 0) or 0),
        "representation_change_pressure": float(plan_geo.get("representation_change_pressure", 0) or 0),
        "step_count": float(search_geo.get("step_count", 0) or 0),
        "no_progress_steps": float(search_geo.get("no_progress_steps", 0) or 0),
        "no_progress_ratio": float(search_geo.get("no_progress_ratio", 0.0) or 0.0),
        "blank_lane_streak": float(search_geo.get("max_blank_lane_streak", 0) or 0),
        "identical_no_progress_streak": float(search_geo.get("max_identical_no_progress_streak", 0) or 0),
        "forward_rw_count": float(search_geo.get("forward_rw_count", 0) or 0),
        "backward_rw_count": float(search_geo.get("backward_rw_count", 0) or 0),
        "simp_count": float(search_geo.get("simp_count", 0) or 0),
        "bidirectional_rw_cycle": float(search_geo.get("bidirectional_rw_cycle", 0) or 0),
        "candidate_prior_count": float(safe_dict(hard_surface.get("prior_graph_geometry")).get("candidate_count", 0) or 0),
        "same_namespace_candidates": float(safe_dict(hard_surface.get("prior_graph_geometry")).get("same_namespace_candidates", 0) or 0),
        "accessible_premise_count": float(theorem_surface.get("accessible_premise_count", 0) or 0),
        "ducky_observed": float(bool(ducky_surface.get("observed"))),
        "ducky_started_count": float(ducky_surface.get("started_count", 0) or 0),
        "ducky_theorem_faithful_count": float(ducky_surface.get("theorem_faithful_count", 0) or 0),
        "ducky_progressed_count": float(ducky_surface.get("progressed_count", 0) or 0),
        "ducky_closed_count": float(ducky_surface.get("closed_count", 0) or 0),
        "ducky_compile_proxy_count": float(ducky_surface.get("compile_proxy_count", 0) or 0),
        "ducky_certificate_generation_count": float(ducky_surface.get("certificate_generation_count", 0) or 0),
        "ducky_projector_event_count": float(ducky_surface.get("projector_event_count", 0) or 0),
        "label_invoke_ducky": float(bool(labels.get("invoke_ducky"))),
        "label_projector_rejection_seen": float(bool(labels.get("projector_rejection_seen"))),
    }


def encode_packet_features(packet: dict[str, Any], metadata: dict[str, Any]) -> list[float]:
    token_vocab = list(metadata.get("token_vocab", []) or [])
    numeric_feature_names = list(metadata.get("numeric_feature_names", []) or [])
    token_index = {token: idx for idx, token in enumerate(token_vocab)}
    vector = [0.0] * (len(token_vocab) + len(numeric_feature_names))
    for token in collect_packet_tokens(packet):
        idx = token_index.get(token)
        if idx is not None:
            vector[idx] += 1.0
    numerics = numeric_packet_features(packet)
    for offset, name in enumerate(numeric_feature_names):
        vector[len(token_vocab) + offset] = float(numerics.get(name, 0.0))
    return vector
