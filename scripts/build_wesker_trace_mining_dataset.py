"""Build synthetic second-order SoM training data from LeanDojo tactic traces.

v2: Three fixes over v1 —
  1. Recover ``cases``/``induction``/``use``/``norm_cast`` etc from the canonical
     "other" family bucket via ``tactic_base`` field
  2. Multi-engine activation vectors calibrated from real bridge co-occurrence
  3. Realistic progress rate (~80%) instead of always-True

See docs/Research_Paper/Wesker/WESKER_SYNTHETIC_DATA.md for design rationale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tactic-base → engine/backend mapping (primary + "other" family recovery)
# ---------------------------------------------------------------------------

# Maps either canonical ``family`` or ``tactic_base`` to primary engine.
# The canonical pipeline assigns many tactics to family="other";
# we recover them here via tactic_base.

TACTIC_ENGINE_MAP: dict[str, str] = {
    # --- Already in canonical families ---
    "rw": "EqSatEngine",
    "simp": "ArithEngine",
    "exact": "ContextTransportEngine",
    "apply": "ContextTransportEngine",
    "refine": "WitnessEngine",
    "ext": "ContextTransportEngine",
    "funext": "ContextTransportEngine",
    "congr": "ContextTransportEngine",
    "cases": "RecursiveInvariantEngine",
    "induction": "RecursiveInvariantEngine",
    "rcases": "RecursiveInvariantEngine",
    "obtain": "WitnessEngine",
    "use": "WitnessEngine",
    "norm_num": "ArithEngine",
    "ring": "ArithEngine",
    "field_simp": "ArithEngine",
    "omega": "ArithEngine",
    "linarith": "ArithEngine",
    "nlinarith": "ArithEngine",
    "norm_cast": "FiniteFilterEngine",
    "push_cast": "FiniteFilterEngine",
    "decide": "FiniteFilterEngine",
    "aesop": "ContextTransportEngine",
    "tauto": "ContextTransportEngine",
    # --- Recovered from family="other" via tactic_base ---
    "by_cases": "RecursiveInvariantEngine",
    "split_ifs": "RecursiveInvariantEngine",
    "convert": "FiniteFilterEngine",
    "subst": "RecursiveInvariantEngine",
    "contrapose!": "RecursiveInvariantEngine",
    "contrapose": "RecursiveInvariantEngine",
    "classical": "RecursiveInvariantEngine",
    "haveI": "WitnessEngine",
    "letI": "WitnessEngine",
    "calc": "ContextTransportEngine",
    "replace": "EqSatEngine",
    "erw": "EqSatEngine",
    "gcongr": "ContextTransportEngine",
    "positivity": "ArithEngine",
    "filter_upwards": "ContextTransportEngine",
    "ext1": "ContextTransportEngine",
    "exacts": "ContextTransportEngine",
    "fun_prop": "ContextTransportEngine",
    "lift": "ContextTransportEngine",
    "infer_instance": "ContextTransportEngine",
    "set": "WitnessEngine",
    "existsi": "WitnessEngine",
    "inhabit": "WitnessEngine",
    "Abel": "ArithEngine",
    "polyrith": "ArithEngine",
    "interval_cases": "RecursiveInvariantEngine",
    "fin_cases": "RecursiveInvariantEngine",
    "match_target": "RecursiveInvariantEngine",
    "rcases_many": "RecursiveInvariantEngine",
    "per_cases": "RecursiveInvariantEngine",
}

TACTIC_BACKEND_MAP: dict[str, str] = {
    "rw": "egglog_eqsat",
    "simp": "lean_arith",
    "exact": "rosette_proof_dsl",
    "apply": "rosette_proof_dsl",
    "refine": "rosette_proof_dsl",
    "ext": "rosette_proof_dsl",
    "funext": "rosette_proof_dsl",
    "congr": "rosette_proof_dsl",
    "cases": "symbolic_rewrite_vm",
    "induction": "symbolic_rewrite_vm",
    "rcases": "symbolic_rewrite_vm",
    "obtain": "rosette_proof_dsl",
    "use": "rosette_proof_dsl",
    "norm_num": "lean_arith",
    "ring": "lean_arith",
    "field_simp": "lean_arith",
    "omega": "lean_arith",
    "linarith": "lean_arith",
    "nlinarith": "lean_arith",
    "norm_cast": "kodkod_relational",
    "push_cast": "kodkod_relational",
    "decide": "kodkod_relational",
    "aesop": "rosette_proof_dsl",
    "tauto": "rosette_proof_dsl",
    "by_cases": "symbolic_rewrite_vm",
    "split_ifs": "symbolic_rewrite_vm",
    "convert": "kodkod_relational",
    "subst": "symbolic_rewrite_vm",
    "contrapose!": "symbolic_rewrite_vm",
    "contrapose": "symbolic_rewrite_vm",
    "classical": "symbolic_rewrite_vm",
    "haveI": "rosette_proof_dsl",
    "letI": "rosette_proof_dsl",
    "calc": "rosette_proof_dsl",
    "replace": "egglog_eqsat",
    "erw": "egglog_eqsat",
    "gcongr": "rosette_proof_dsl",
    "positivity": "lean_arith",
    "filter_upwards": "rosette_proof_dsl",
    "ext1": "rosette_proof_dsl",
    "exacts": "rosette_proof_dsl",
    "fun_prop": "rosette_proof_dsl",
    "lift": "rosette_proof_dsl",
    "infer_instance": "rosette_proof_dsl",
    "set": "rosette_proof_dsl",
    "existsi": "rosette_proof_dsl",
    "inhabit": "rosette_proof_dsl",
    "Abel": "lean_arith",
    "polyrith": "lean_arith",
    "interval_cases": "symbolic_rewrite_vm",
    "fin_cases": "symbolic_rewrite_vm",
    "match_target": "symbolic_rewrite_vm",
    "rcases_many": "symbolic_rewrite_vm",
    "per_cases": "symbolic_rewrite_vm",
}

ENGINE_NAMES = [
    "ArithEngine",
    "ContextTransportEngine",
    "EqSatEngine",
    "FiniteFilterEngine",
    "RecursiveInvariantEngine",
    "WitnessEngine",
]

BACKEND_NAMES = [
    "egglog_eqsat",
    "kodkod_relational",
    "lean_arith",
    "rosette_proof_dsl",
    "symbolic_rewrite_vm",
]

# Families we exclude — pure structural setup, no engine-selection signal
STRUCTURAL_TACTICS = {
    "intro", "intros", "constructor", "left", "right",
    "trivial", "rfl", "assumption", "contradiction",
    "exact_mod_cast", "revert", "clear", "rename",
    "dsimp", "change", "show", "suffices",
}


# ---------------------------------------------------------------------------
# Co-occurrence model calibrated from real bridge data
# ---------------------------------------------------------------------------

# P(engine_j fires | primary engine = engine_i)
# Derived from 468 real bridge rows.  ContextTransport fires 92% of the time
# regardless — it's nearly always-on.  RecursiveInvariant fires 65%.

# Row = primary engine.  Columns = co-fire probability for each engine.
# Order: Arith, Context, EqSat, Finite, Recursive, Witness
COFIRE_MATRIX = np.array([
    # primary = Arith
    [1.000, 0.939, 0.854, 0.241, 0.679, 0.208],
    # primary = ContextTransport
    [0.461, 1.000, 0.650, 0.331, 0.641, 0.227],
    # primary = EqSat
    [0.622, 0.966, 1.000, 0.254, 0.649, 0.196],
    # primary = FiniteFilter
    [0.329, 0.923, 0.477, 1.000, 0.658, 0.348],
    # primary = RecursiveInvariant
    [0.475, 0.914, 0.624, 0.337, 1.000, 0.274],
    # primary = Witness
    [0.411, 0.916, 0.533, 0.505, 0.776, 1.000],
], dtype=np.float64)

# Real per-engine marginal activation rates from r6 bridge (468 hard rows)
REAL_MARGINAL = np.array([0.453, 0.923, 0.622, 0.331, 0.647, 0.229])

# Hard-tail primary engine weights (from canonical tactic data on failed theorems).
# This is the *oracle* engine distribution — what engine the GT tactic maps to.
# Co-occurrence then expands this to the full activation vector Ducky tries.
HARD_TAIL_PRIMARY_WEIGHTS = np.array([
    0.2539,  # ArithEngine
    0.2765,  # ContextTransportEngine
    0.3073,  # EqSatEngine
    0.0243,  # FiniteFilterEngine
    0.0334,  # RecursiveInvariantEngine
    0.1046,  # WitnessEngine
])

# Real progress rate on hard residuals
REAL_PROGRESS_RATE = 0.795  # 372/468 from r6 bridge


def _sample_engine_vector(primary_engine: str, rng: random.Random) -> list[str]:
    """Sample a multi-engine activation vector from the co-occurrence model."""
    try:
        primary_idx = ENGINE_NAMES.index(primary_engine)
    except ValueError:
        return [primary_engine] if primary_engine else []

    probs = COFIRE_MATRIX[primary_idx]
    active = []
    for j, p in enumerate(probs):
        if rng.random() < p:
            active.append(ENGINE_NAMES[j])
    # Guarantee the primary always fires
    if primary_engine not in active:
        active.append(primary_engine)
    return sorted(active)


def _engines_to_backends(engines: list[str]) -> list[str]:
    """Map active engines to their corresponding backends."""
    engine_to_backend = {
        "ArithEngine": "lean_arith",
        "ContextTransportEngine": "rosette_proof_dsl",
        "EqSatEngine": "egglog_eqsat",
        "FiniteFilterEngine": "kodkod_relational",
        "RecursiveInvariantEngine": "symbolic_rewrite_vm",
        "WitnessEngine": "rosette_proof_dsl",
    }
    backends = set()
    for e in engines:
        b = engine_to_backend.get(e)
        if b:
            backends.add(b)
    return sorted(backends)


# ---------------------------------------------------------------------------
# Goal-bucket derivation
# ---------------------------------------------------------------------------

def _goal_bucket(goal_shape: dict[str, Any]) -> str:
    if goal_shape.get("has_equality"):
        return "equality"
    if goal_shape.get("has_exists"):
        return "exists"
    if goal_shape.get("has_forall"):
        return "forall"
    if goal_shape.get("has_iff"):
        return "iff"
    if goal_shape.get("has_implication"):
        return "implication"
    target_head = str(goal_shape.get("target_head", "") or "")
    if target_head == "False":
        return "false"
    if target_head in ("Membership", "Mem", "mem"):
        return "membership"
    return "other"


def _residual_bucket(goal_shape: dict[str, Any]) -> str:
    goal_count = int(goal_shape.get("goal_count", 1) or 1)
    if goal_count == 1:
        return "single_goal_near_miss"
    if goal_count <= 3:
        return "multi_goal_small_progress"
    return "multi_goal_large_progress"


def _stable_split(theorem_id: str, train_ratio: float = 0.8) -> str:
    digest = hashlib.sha256(theorem_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "eval"


# ---------------------------------------------------------------------------
# Tactic resolution: family + tactic_base → engine
# ---------------------------------------------------------------------------

def _resolve_tactic(row: dict[str, Any]) -> tuple[str, str]:
    """Resolve a canonical row to (engine, backend) using family then tactic_base."""
    family = str(row.get("family", "") or "")
    tactic_base = str(row.get("tactic_base", "") or "")

    # Try canonical family first
    if family not in ("other", "") and family not in STRUCTURAL_TACTICS:
        engine = TACTIC_ENGINE_MAP.get(family)
        backend = TACTIC_BACKEND_MAP.get(family)
        if engine and backend:
            return engine, backend

    # Fall back to tactic_base (recovers "other" bucket)
    if tactic_base and tactic_base not in STRUCTURAL_TACTICS:
        engine = TACTIC_ENGINE_MAP.get(tactic_base)
        backend = TACTIC_BACKEND_MAP.get(tactic_base)
        if engine and backend:
            return engine, backend

    return "", ""


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _is_high_quality(row: dict[str, Any], min_goal_len: int = 30) -> bool:
    """Filter for non-trivial steps that carry engine-selection signal."""
    engine, _ = _resolve_tactic(row)
    if not engine:
        return False
    goal = str(row.get("goal_state_before", "") or "")
    if len(goal) < min_goal_len:
        return False
    return True


# ---------------------------------------------------------------------------
# Packet construction
# ---------------------------------------------------------------------------

def _build_synthetic_packet(
    row: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    family = str(row.get("family", "") or "")
    tactic_base = str(row.get("tactic_base", "") or "")
    theorem_id = str(row.get("theorem_full_name", "") or "")
    goal_shape = row.get("goal_shape_ir") or {}
    trigger = row.get("trigger_profile_ir") or {}

    primary_engine, _ = _resolve_tactic(row)
    gb = _goal_bucket(goal_shape)
    rb = _residual_bucket(goal_shape)

    # Multi-engine activation from co-occurrence model
    active_engines = _sample_engine_vector(primary_engine, rng)
    active_backends = _engines_to_backends(active_engines)

    # Realistic progress rate
    progressed = rng.random() < REAL_PROGRESS_RATE

    # Build hard_som_surface
    goal_text = str(row.get("goal_state_before", "") or "")
    local_names = goal_shape.get("local_names") or []
    resolved_family = tactic_base if family == "other" else family

    file_path = str(row.get("file_path", "") or "")
    domain_hints = []
    if file_path:
        parts = file_path.replace(".lean", "").split("/")
        for p in parts[:3]:
            if p and p != "Mathlib":
                domain_hints.append(p)

    trigger_features = trigger.get("features") or []
    specialist_targets = []
    for feat in trigger_features:
        kind = str(feat.get("kind", "") or "")
        value = str(feat.get("value", "") or "")
        if kind == "target_shape" and value:
            specialist_targets.append(f"{value}_specialist")

    hard_som_surface = {
        "theorem_id": theorem_id,
        "residual_bucket": rb,
        "goal_bucket": gb,
        "residual_skeleton_geometry": {
            "goal_shape_features": {
                "char_len": len(goal_text),
                "token_len": len(goal_text.split()),
                "binder_count": goal_text.count("\u2200") + goal_text.count("\u2203") + goal_text.count("fun"),
                "forall_count": int(goal_shape.get("has_forall", False)),
                "exists_count": int(goal_shape.get("has_exists", False)),
                "eq_count": int(goal_shape.get("has_equality", False)),
                "iff_count": int(goal_shape.get("has_iff", False)),
                "membership_count": 0,
                "subset_count": 0,
            },
            "theorem_shape_features": {
                "char_len": 0,
                "token_len": 0,
                "binder_count": 0,
            },
            "domain_hints": domain_hints,
            "representation_pressures": [],
            "top_symbols": local_names[:5],
            "symbol_count": len(local_names),
        },
        "proof_plan_geometry": {
            "candidate_methods": [resolved_family],
            "specialist_targets": specialist_targets,
            "lane_suppression_hints": [],
            "lane_history": [],
            "lane_count": int(row.get("step_index", 0) or 0),
            "bridge_pressure": 0.0,
            "representation_change_pressure": 0.0,
            "search_control_geometry": {
                "step_count": int(row.get("step_index", 0) or 0),
                "no_progress_steps": 0,
                "no_progress_ratio": 0.0,
                "max_blank_lane_streak": 0,
                "max_identical_no_progress_streak": 0,
                "forward_rw_count": 1 if resolved_family in ("rw", "replace", "erw") else 0,
                "backward_rw_count": 0,
                "simp_count": 1 if resolved_family == "simp" else 0,
                "bidirectional_rw_cycle": 0,
            },
        },
        "prior_graph_geometry": {
            "candidate_count": 0,
            "same_namespace_candidates": 0,
            "theorem_surface": {
                "accessible_premise_count": 0,
                "anchor_labels": domain_hints[:3],
            },
        },
    }

    engine_counts = {e: 1 for e in active_engines}
    backend_counts = {b: 1 for b in active_backends}

    ducky_outcome_surface = {
        "observed": True,
        "observation_count": 1,
        "started_count": 1,
        "theorem_faithful_count": 1,
        "progressed_count": 1 if progressed else 0,
        "closed_count": 0,
        "compile_proxy_count": 1,
        "certificate_generation_count": 1,
        "projector_event_count": 1,
        "observed_progress": progressed,
        "observed_close": False,
        "observed_start": True,
        "theorem_faithful_observed": True,
        "best_outcome": "progressed" if progressed else "started",
        "engine_counts": engine_counts,
        "backend_family_counts": backend_counts,
        "certificate_shape_counts": {},
        "projector_status_counts": {"projected": 1},
    }

    return {
        "packet_version": "second_order_controller_surface_v1",
        "packet_kind": "hard_residual",
        "packet_source": "wesker_trace_mining_v2",
        "theorem_id": theorem_id,
        "split": _stable_split(theorem_id),
        "difficulty_band": "",
        "residual_bucket": rb,
        "goal_bucket": gb,
        "resolution_family": resolved_family,
        "hard_som_surface": hard_som_surface,
        "ducky_outcome_surface": ducky_outcome_surface,
        "second_order_labels": {
            "invoke_ducky": True,
            "observed_progress": progressed,
            "observed_close": False,
            "engine_family_budget_targets": active_engines,
            "backend_budget_targets": active_backends,
            "projector_rejection_seen": False,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_hard_theorem_set(details_path: Path) -> set[str]:
    """Load theorem IDs that the first-order stack failed to prove."""
    hard = set()
    if not details_path.exists():
        return hard
    with details_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bucket = str(row.get("residual_bucket", "") or "")
            if bucket != "proved":
                tid = str(row.get("theorem_full_name", "") or "")
                if tid:
                    hard.add(tid)
    return hard


def _compute_enrichment_weights(
    canonical_path: Path,
    hard_theorems: set[str],
) -> dict[str, float]:
    """Compute hard-tail enrichment ratio per resolved tactic.

    Tactics that appear disproportionately in hard theorems get weight > 1.
    This biases the synthetic data toward the hard-tail distribution even
    when using the full canonical corpus.
    """
    hard_counts: Counter[str] = Counter()
    easy_counts: Counter[str] = Counter()

    with canonical_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = str(row.get("theorem_full_name", "") or "")
            family = str(row.get("family", "") or "")
            base = str(row.get("tactic_base", "") or "")
            resolved = base if family == "other" else family

            if tid in hard_theorems:
                hard_counts[resolved] += 1
            else:
                easy_counts[resolved] += 1

    total_hard = sum(hard_counts.values()) or 1
    total_easy = sum(easy_counts.values()) or 1

    weights: dict[str, float] = {}
    for tactic in set(hard_counts.keys()) | set(easy_counts.keys()):
        hard_rate = hard_counts[tactic] / total_hard
        easy_rate = easy_counts[tactic] / total_easy
        ratio = hard_rate / max(easy_rate, 1e-6)
        # Clamp enrichment to [0.3, 5.0] to avoid extreme over/under-sampling
        weights[tactic] = max(0.3, min(5.0, ratio))
    return weights


def build_trace_mining_dataset(
    canonical_path: Path,
    real_packets_path: Path | None,
    output_dir: Path,
    min_goal_len: int = 30,
    max_synthetic: int = 0,
    seed: int = 42,
    hard_details_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Load hard-tail theorem set for enrichment weighting
    hard_theorems: set[str] = set()
    enrichment_weights: dict[str, float] = {}
    if hard_details_path and hard_details_path.exists():
        hard_theorems = _load_hard_theorem_set(hard_details_path)
        enrichment_weights = _compute_enrichment_weights(canonical_path, hard_theorems)

    # Load and filter canonical steps
    synthetic_packets: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    filtered_count = 0
    enrichment_skipped = 0

    with canonical_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not _is_high_quality(row, min_goal_len=min_goal_len):
                filtered_count += 1
                continue

            # Apply enrichment weighting: skip easy-enriched steps probabilistically
            if enrichment_weights:
                family = str(row.get("family", "") or "")
                base = str(row.get("tactic_base", "") or "")
                resolved = base if family == "other" else family
                weight = enrichment_weights.get(resolved, 1.0)
                # For weight < 1 (easy-enriched), include with probability = weight
                # For weight >= 1 (hard-enriched), always include
                if weight < 1.0 and rng.random() > weight:
                    enrichment_skipped += 1
                    continue

            packet = _build_synthetic_packet(row, rng)
            family = row.get("family", "")
            tactic_base = row.get("tactic_base", "")
            resolved = tactic_base if family == "other" else family
            family_counts[resolved] += 1
            synthetic_packets.append(packet)

    # Optional cap with stratified sampling
    if max_synthetic > 0 and len(synthetic_packets) > max_synthetic:
        per_family_cap = max_synthetic // max(len(family_counts), 1)
        capped: list[dict[str, Any]] = []
        family_seen: Counter[str] = Counter()
        for packet in synthetic_packets:
            fam = packet.get("resolution_family", "")
            if family_seen[fam] < per_family_cap:
                capped.append(packet)
                family_seen[fam] += 1
        remaining = max_synthetic - len(capped)
        if remaining > 0:
            for packet in synthetic_packets:
                if packet not in capped:
                    capped.append(packet)
                    remaining -= 1
                    if remaining <= 0:
                        break
        synthetic_packets = capped

    # Load real packets if provided
    real_packets: list[dict[str, Any]] = []
    if real_packets_path and real_packets_path.exists():
        with real_packets_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    real_packets.append(json.loads(line))

    combined = real_packets + synthetic_packets

    # Write outputs
    synthetic_path = output_dir / "trace_mining_packets.jsonl"
    combined_path = output_dir / "combined_packets.jsonl"

    with synthetic_path.open("w") as handle:
        for packet in synthetic_packets:
            handle.write(json.dumps(packet) + "\n")

    with combined_path.open("w") as handle:
        for packet in combined:
            handle.write(json.dumps(packet) + "\n")

    # Stats
    train_real = sum(1 for p in real_packets if p.get("split") == "train")
    eval_real = sum(1 for p in real_packets if p.get("split") != "train")
    train_syn = sum(1 for p in synthetic_packets if p.get("split") == "train")
    eval_syn = sum(1 for p in synthetic_packets if p.get("split") != "train")

    engine_dist = Counter(
        t for p in synthetic_packets
        for t in (p.get("second_order_labels") or {}).get("engine_family_budget_targets", [])
    )
    backend_dist = Counter(
        t for p in synthetic_packets
        for t in (p.get("second_order_labels") or {}).get("backend_budget_targets", [])
    )

    # Verify multi-engine stats
    engines_per_packet = [
        len((p.get("second_order_labels") or {}).get("engine_family_budget_targets", []))
        for p in synthetic_packets
    ]

    summary = {
        "canonical_path": str(canonical_path),
        "real_packets_path": str(real_packets_path) if real_packets_path else None,
        "total_canonical_rows": filtered_count + len(synthetic_packets),
        "filtered_out": filtered_count,
        "synthetic_packets": len(synthetic_packets),
        "real_packets": len(real_packets),
        "combined_packets": len(combined),
        "train_real": train_real,
        "eval_real": eval_real,
        "train_synthetic": train_syn,
        "eval_synthetic": eval_syn,
        "train_combined": train_real + train_syn,
        "eval_combined": eval_real + eval_syn,
        "family_counts": dict(family_counts.most_common()),
        "engine_distribution": dict(engine_dist.most_common()),
        "backend_distribution": dict(backend_dist.most_common()),
        "avg_engines_per_packet": float(np.mean(engines_per_packet)) if engines_per_packet else 0.0,
        "hard_theorems_loaded": len(hard_theorems),
        "enrichment_skipped": enrichment_skipped,
        "seed": seed,
        "output_synthetic": str(synthetic_path),
        "output_combined": str(combined_path),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        default="data/canonical/canonical_residual_train.jsonl",
        help="Path to canonical residual training data",
    )
    parser.add_argument(
        "--real-packets",
        default="",
        help="Path to real second-order packets to merge with synthetic",
    )
    parser.add_argument(
        "--output-dir",
        default="data/wesker",
        help="Output directory for synthetic packets",
    )
    parser.add_argument(
        "--min-goal-len",
        type=int,
        default=30,
        help="Minimum goal text length to include",
    )
    parser.add_argument(
        "--max-synthetic",
        type=int,
        default=0,
        help="Cap on synthetic packets (0 = no cap)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for co-occurrence sampling",
    )
    parser.add_argument(
        "--hard-details",
        default="",
        help="Path to r6 details.jsonl for hard-tail enrichment weighting",
    )
    args = parser.parse_args()

    real_path = Path(args.real_packets) if args.real_packets else None
    hard_path = Path(args.hard_details) if args.hard_details else None
    summary = build_trace_mining_dataset(
        canonical_path=Path(args.canonical),
        real_packets_path=real_path,
        output_dir=Path(args.output_dir),
        min_goal_len=args.min_goal_len,
        max_synthetic=args.max_synthetic,
        seed=args.seed,
        hard_details_path=hard_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
