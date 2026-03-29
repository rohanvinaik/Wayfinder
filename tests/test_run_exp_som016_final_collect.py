from scripts.run_exp_som016_final_collect import (
    _apply_bridge_result,
    _bridge_eligible,
    _with_first_order_snapshot,
)


def test_bridge_eligible_only_for_started_unsolved_hard_rows() -> None:
    row = {
        "started": True,
        "success": False,
        "residual_bucket": "single_goal_near_miss",
    }
    assert _bridge_eligible(row) is True
    assert _bridge_eligible({**row, "success": True}) is False
    assert _bridge_eligible({**row, "started": False}) is False
    assert _bridge_eligible({**row, "residual_bucket": "proved"}) is False


def test_apply_bridge_result_preserves_first_order_and_updates_pipeline_state() -> None:
    first_order_row = _with_first_order_snapshot(
        {
            "theorem_id": "Foo.bar",
            "success": False,
            "honest_success": False,
            "success_category": "failed",
            "close_lane": "failed",
            "final_closer": "",
            "lane_sequence": "automation",
            "close_provenance": ["automation"],
            "attempts": 12,
            "time_s": 3.5,
            "goals_closed": 1,
            "goals_remaining": 1,
            "remaining_goals_snapshot": ["x = 0"],
            "last_goal": "x = 0",
            "last_goal_bucket": "equality",
            "residual_bucket": "single_goal_near_miss",
            "follow_on_stage": "hard_proof_solver",
            "reasoning_gap_family": "hard_proof_solver",
            "difficulty_bucket": "hard_local",
        }
    )
    bridge = {
        "started": True,
        "theorem_faithful": True,
        "progressed": True,
        "closed": True,
        "closed_by": "first_order_search",
        "final_goals": [],
        "initial_goal_count": 1,
        "controller_decision": {"controller_mode": "deterministic_packet_policy_v1"},
        "stage_trace": [{"stage": "first_order_search", "closed": True}],
        "ducky_pass_1": {"programs_considered": 4},
        "first_order_search": {
            "attempts": 6,
            "tactics_used": ["intros", "apply?"],
            "close_provenance": ["apply?"],
            "success": True,
        },
        "ducky_pass_2": None,
        "symbolic_close_pass_2": None,
        "rarified_gap_packet": None,
    }
    out = _apply_bridge_result(first_order_row, bridge, bridge_time_s=1.25)
    assert out["first_order_success"] is False
    assert out["success"] is True
    assert out["honest_success"] is True
    assert out["success_category"] == "bridge_success"
    assert out["close_lane"] == "hardtail_bridge:first_order_search"
    assert out["final_closer"] == "apply?"
    assert out["attempts"] == 22
    assert out["time_s"] == 4.75
    assert out["goals_remaining"] == 0
    assert out["bridge_invoked"] is True
    assert out["bridge_closed"] is True
