"""Build trainable second-order SoM feature datasets from frozen packet surfaces.

The second-order packet freeze gives the controller a symbolic packet manifold.
This script converts those packets into dense numeric arrays plus supervised
targets so the second-order controller can be trained without re-reading raw
JSON at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stable_split(packet: dict[str, Any]) -> str:
    split = str(packet.get("split", "") or "").strip()
    return split if split else "train"


def _reshard_split(theorem_id: str, train_ratio: float = 0.8) -> str:
    digest = hashlib.sha1(theorem_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "eval"


def _collect_tokens(packet: dict[str, Any]) -> list[str]:
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

    hard_surface = _safe_dict(packet.get("hard_som_surface"))
    residual_geo = _safe_dict(hard_surface.get("residual_skeleton_geometry"))
    plan_geo = _safe_dict(hard_surface.get("proof_plan_geometry"))
    prior_geo = _safe_dict(hard_surface.get("prior_graph_geometry"))
    theorem_surface = _safe_dict(prior_geo.get("theorem_surface"))
    ducky_surface = _safe_dict(packet.get("ducky_outcome_surface"))

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


def _numeric_features(packet: dict[str, Any]) -> dict[str, float]:
    hard_surface = _safe_dict(packet.get("hard_som_surface"))
    residual_geo = _safe_dict(hard_surface.get("residual_skeleton_geometry"))
    plan_geo = _safe_dict(hard_surface.get("proof_plan_geometry"))
    search_geo = _safe_dict(plan_geo.get("search_control_geometry"))
    theorem_surface = _safe_dict(_safe_dict(hard_surface.get("prior_graph_geometry")).get("theorem_surface"))
    ducky_surface = _safe_dict(packet.get("ducky_outcome_surface"))
    labels = _safe_dict(packet.get("second_order_labels"))

    goal_shape = _safe_dict(residual_geo.get("goal_shape_features"))
    theorem_shape = _safe_dict(residual_geo.get("theorem_shape_features"))

    features = {
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
        "candidate_prior_count": float(_safe_dict(hard_surface.get("prior_graph_geometry")).get("candidate_count", 0) or 0),
        "same_namespace_candidates": float(_safe_dict(hard_surface.get("prior_graph_geometry")).get("same_namespace_candidates", 0) or 0),
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
    return features


def _collect_vocab(rows: list[dict[str, Any]]) -> list[str]:
    vocab = Counter()
    for row in rows:
        vocab.update(_collect_tokens(row))
    return [token for token, _count in sorted(vocab.items())]


def _engine_vocab(rows: list[dict[str, Any]]) -> list[str]:
    vocab: set[str] = set()
    backends: set[str] = set()
    for row in rows:
        labels = _safe_dict(row.get("second_order_labels"))
        vocab.update(str(item) for item in labels.get("engine_family_budget_targets", []) or [] if item)
        backends.update(str(item) for item in labels.get("backend_budget_targets", []) or [] if item)
    return sorted(vocab), sorted(backends)


def _packet_matrix(
    rows: list[dict[str, Any]],
    token_vocab: list[str],
    numeric_feature_names: list[str],
    engine_vocab: list[str],
    backend_vocab: list[str],
) -> dict[str, np.ndarray]:
    token_index = {token: idx for idx, token in enumerate(token_vocab)}
    engine_index = {token: idx for idx, token in enumerate(engine_vocab)}
    backend_index = {token: idx for idx, token in enumerate(backend_vocab)}

    features = np.zeros((len(rows), len(token_vocab) + len(numeric_feature_names)), dtype=np.float32)
    invoke_ducky = np.zeros(len(rows), dtype=np.int64)
    observed_progress = np.zeros(len(rows), dtype=np.int64)
    observed_close = np.zeros(len(rows), dtype=np.int64)
    projector_rejection_seen = np.zeros(len(rows), dtype=np.int64)
    packet_kind = np.zeros(len(rows), dtype=np.int64)
    engine_targets = np.zeros((len(rows), len(engine_vocab)), dtype=np.int64)
    backend_targets = np.zeros((len(rows), len(backend_vocab)), dtype=np.int64)

    theorem_ids: list[str] = []
    packet_kind_vocab = ["compiler_specialist", "hard_residual"]
    packet_kind_index = {name: idx for idx, name in enumerate(packet_kind_vocab)}

    for row_idx, row in enumerate(rows):
        theorem_ids.append(str(row.get("theorem_id", "") or ""))
        for token in _collect_tokens(row):
            if token in token_index:
                features[row_idx, token_index[token]] += 1.0
        numerics = _numeric_features(row)
        for offset, name in enumerate(numeric_feature_names):
            features[row_idx, len(token_vocab) + offset] = float(numerics.get(name, 0.0))

        labels = _safe_dict(row.get("second_order_labels"))
        invoke_ducky[row_idx] = int(bool(labels.get("invoke_ducky")))
        observed_progress[row_idx] = int(bool(labels.get("observed_progress")))
        observed_close[row_idx] = int(bool(labels.get("observed_close")))
        projector_rejection_seen[row_idx] = int(bool(labels.get("projector_rejection_seen")))
        packet_kind[row_idx] = packet_kind_index.get(str(row.get("packet_kind", "") or ""), 0)
        for name in labels.get("engine_family_budget_targets", []) or []:
            idx = engine_index.get(str(name))
            if idx is not None:
                engine_targets[row_idx, idx] = 1
        for name in labels.get("backend_budget_targets", []) or []:
            idx = backend_index.get(str(name))
            if idx is not None:
                backend_targets[row_idx, idx] = 1

    return {
        "features": features,
        "invoke_ducky": invoke_ducky,
        "observed_progress": observed_progress,
        "observed_close": observed_close,
        "projector_rejection_seen": projector_rejection_seen,
        "packet_kind": packet_kind,
        "engine_targets": engine_targets,
        "backend_targets": backend_targets,
        "theorem_ids": np.array(theorem_ids, dtype=object),
    }


def build_second_order_feature_dataset(
    packets_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _load_jsonl(packets_path)
    token_vocab = _collect_vocab(rows)
    engine_vocab, backend_vocab = _engine_vocab(rows)
    numeric_feature_names = sorted({name for row in rows for name in _numeric_features(row).keys()})

    train_rows = [row for row in rows if _stable_split(row) == "train"]
    eval_rows = [row for row in rows if _stable_split(row) != "train"]
    split_strategy = "source_split"
    if not train_rows or not eval_rows:
        split_strategy = "deterministic_theorem_reshard"
        train_rows = []
        eval_rows = []
        for row in rows:
            theorem_id = str(row.get("theorem_id", "") or "")
            if _reshard_split(theorem_id) == "train":
                train_rows.append(row)
            else:
                eval_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = _packet_matrix(train_rows, token_vocab, numeric_feature_names, engine_vocab, backend_vocab)
    eval_data = _packet_matrix(eval_rows, token_vocab, numeric_feature_names, engine_vocab, backend_vocab)

    np.savez_compressed(output_dir / "train.npz", **train_data)
    np.savez_compressed(output_dir / "eval.npz", **eval_data)

    summary = {
        "packet_version": "second_order_feature_dataset_v1",
        "packets_path": str(packets_path),
        "total_packets": len(rows),
        "train_packets": len(train_rows),
        "eval_packets": len(eval_rows),
        "feature_dim": int(len(token_vocab) + len(numeric_feature_names)),
        "token_vocab_size": len(token_vocab),
        "numeric_feature_count": len(numeric_feature_names),
        "split_strategy": split_strategy,
        "engine_vocab": engine_vocab,
        "backend_vocab": backend_vocab,
        "train_invoke_ducky": int(train_data["invoke_ducky"].sum()) if len(train_rows) else 0,
        "eval_invoke_ducky": int(eval_data["invoke_ducky"].sum()) if len(eval_rows) else 0,
        "train_observed_progress": int(train_data["observed_progress"].sum()) if len(train_rows) else 0,
        "eval_observed_progress": int(eval_data["observed_progress"].sum()) if len(eval_rows) else 0,
        "train_observed_close": int(train_data["observed_close"].sum()) if len(train_rows) else 0,
        "eval_observed_close": int(eval_data["observed_close"].sum()) if len(eval_rows) else 0,
    }
    metadata = {
        "packet_version": "second_order_feature_dataset_v1",
        "token_vocab": token_vocab,
        "numeric_feature_names": numeric_feature_names,
        "engine_vocab": engine_vocab,
        "backend_vocab": backend_vocab,
        "packet_kind_vocab": ["compiler_specialist", "hard_residual"],
        "summary": summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets",
        default="runs/exp_som012_hard_eval_r2/bundle/second_order_som/second_order_packets.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/exp_som012_hard_eval_r2/bundle/second_order_som/features",
    )
    args = parser.parse_args()
    summary = build_second_order_feature_dataset(Path(args.packets), Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
