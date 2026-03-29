from __future__ import annotations

import json
from pathlib import Path

from src.second_order_controller import (
    derive_second_order_decision,
    load_second_order_packet_index,
    refine_second_pass_decision,
)


def test_load_second_order_packet_index(tmp_path: Path) -> None:
    path = tmp_path / "packets.jsonl"
    rows = [
        {"theorem_id": "A.x", "packet_kind": "hard_residual"},
        {"theorem_id": "B.y", "packet_kind": "compiler_specialist"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index = load_second_order_packet_index(path)
    assert sorted(index) == ["A.x", "B.y"]
    assert index["A.x"]["packet_kind"] == "hard_residual"


def test_derive_second_order_decision_uses_packet_surface() -> None:
    row = {
        "theorem_id": "T.eq",
        "residual_bucket": "single_goal_near_miss",
        "last_goal_bucket": "equality",
        "last_goal": "x = y",
    }
    packet = {
        "packet_kind": "hard_residual",
        "residual_bucket": "single_goal_near_miss",
        "goal_bucket": "equality",
        "resolution_family": "local_eq_close",
        "second_order_labels": {
            "invoke_ducky": True,
            "engine_family_budget_targets": ["EqSatEngine", "ContextTransportEngine"],
            "backend_budget_targets": ["egglog_eqsat", "rosette_proof_dsl"],
            "projector_rejection_seen": False,
        },
        "hard_som_surface": {
            "proof_plan_geometry": {
                "specialist_targets": ["human_calculator"],
                "lane_suppression_hints": ["avoid_bidirectional_rw_cycles"],
            }
        },
        "ducky_outcome_surface": {
            "backend_family_counts": {"egglog_eqsat": 3, "rosette_proof_dsl": 1},
            "engine_counts": {"EqSatEngine": 3, "ContextTransportEngine": 1},
        },
    }
    decision = derive_second_order_decision(row, packet)
    assert decision.invoke_first_ducky is True
    assert decision.search_budget == 90
    assert decision.enable_cosine_apply is True
    assert "egglog_eqsat" in decision.first_pass_backends
    assert "EqSatEngine" in decision.first_pass_engines
    assert decision.rarified_target == "algebraic_transport_closer"


def test_refine_second_pass_decision_promotes_relational_backends() -> None:
    row = {
        "theorem_id": "T.mem",
        "residual_bucket": "multi_goal_small_progress",
        "last_goal_bucket": "atomic_prop",
        "last_goal": "⊢ P",
    }
    decision = derive_second_order_decision(row, None)
    refined = refine_second_pass_decision(
        decision,
        ["x : α\n⊢ x ∈ s", "⊢ ∃ y, y ∈ s"],
        search_trace=[{"lane": "cosine_apply"}, {"lane": "interleaved_bootstrap"}],
    )
    assert refined.second_pass_goal_limit == 2
    assert "kodkod_relational" in refined.second_pass_backends
    assert refined.rarified_target == "relational_structural_closer"


def test_derive_second_order_decision_can_use_learned_runtime() -> None:
    class _FakeRuntime:
        def predict_packet(self, _packet):
            return {
                "invoke_prob": 0.9,
                "progress_prob": 0.8,
                "projector_rejection_prob": 0.1,
                "packet_kind_probs": {"hard_residual": 0.9, "compiler_specialist": 0.1},
                "engine_probs": {"FiniteFilterEngine": 0.8, "WitnessEngine": 0.7},
                "backend_probs": {"kodkod_relational": 0.9, "rosette_proof_dsl": 0.6},
            }

    row = {
        "theorem_id": "T.exists",
        "residual_bucket": "single_goal_stall",
        "last_goal_bucket": "exists",
        "last_goal": "⊢ ∃ y, y ∈ s",
    }
    decision = derive_second_order_decision(row, None, runtime=_FakeRuntime())
    assert decision.controller_mode == "learned_second_order_som_v1"
    assert decision.invoke_first_ducky is True
    assert "kodkod_relational" in decision.first_pass_backends
    assert "FiniteFilterEngine" in decision.first_pass_engines
