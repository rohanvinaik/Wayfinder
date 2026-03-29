from __future__ import annotations

import sqlite3

from src.dr_ducky_executor import DuckyExecutionResult, DuckyReplayState
from src.hardtail_bridge import run_hardtail_bridge_on_row
from src.proof_search import Pipeline, SearchConfig, SearchResult


class _DummyLean:
    pass


class _DummyPipeline:
    encoder = object()
    analyzer = object()
    bridge = object()
    navigator = object()


def _ducky_result(
    *,
    theorem_id: str,
    progressed: bool,
    closed: bool,
    goals_after: list[str],
    final_goal: str,
) -> DuckyExecutionResult:
    return DuckyExecutionResult(
        theorem_id=theorem_id,
        started=True,
        theorem_faithful=True,
        start_goal_kind="direct_goal",
        file_path="",
        replay_tier="live",
        replay_failure_category="",
        replay_failing_prefix_idx=-1,
        residual_bucket="single_goal_near_miss",
        goal_bucket="equality",
        specialist_targets=["human_calculator"],
        bank_priors=["eq_sat"],
        programs_considered=3,
        closed=closed,
        progressed=progressed,
        winning_program={"program_id": "p0"} if progressed else None,
        final_goal=final_goal,
        final_goal_bucket="equality",
        goals_after=goals_after,
        ledger_snapshot={},
        engine_outcomes=[],
        projector_outcomes=[],
        tried_programs=[],
    )


def test_run_hardtail_bridge_on_row_emits_rarified_gap(monkeypatch) -> None:
    row = {
        "theorem_id": "T.bridge",
        "residual_bucket": "single_goal_near_miss",
        "last_goal_bucket": "equality",
        "last_goal": "x = y",
        "goal_state": "x = y",
    }

    def fake_replay(_row, _lean):
        return DuckyReplayState(
            theorem_id="T.bridge",
            file_path="",
            goal_state="x = y",
            goal_kind="theorem_faithful",
            theorem_faithful=True,
            tier_used="B",
            replay_success=True,
        )

    call_counter = {"count": 0}

    def fake_ducky_on_goal(goal, **kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return _ducky_result(
                theorem_id="T.bridge",
                progressed=True,
                closed=False,
                goals_after=["h : x = y\n⊢ y = x"],
                final_goal="h : x = y\n⊢ y = x",
            )
        return _ducky_result(
            theorem_id="T.bridge",
            progressed=True,
            closed=False,
            goals_after=["h : x = y\n⊢ x = x"],
            final_goal="h : x = y\n⊢ x = x",
        )

    def fake_search_fn(**kwargs):
        return SearchResult(
            success=False,
            theorem_id="T.bridge",
            attempts=7,
            goals_closed=0,
            goals_remaining=1,
            progress_steps=1,
            close_provenance=["cosine_apply"],
            step_trace=[{"lane": "cosine_apply", "progress": True}],
            final_goals=["h : x = y\n⊢ x = x"],
        )

    monkeypatch.setattr("src.hardtail_bridge.replay_residual_state", fake_replay)
    monkeypatch.setattr("src.hardtail_bridge.run_ducky_on_goal", fake_ducky_on_goal)

    result = run_hardtail_bridge_on_row(
        row,
        packet=None,
        pipeline=Pipeline(
            encoder=_DummyPipeline.encoder,
            analyzer=_DummyPipeline.analyzer,
            bridge=_DummyPipeline.bridge,
            navigator=_DummyPipeline.navigator,
        ),
        search_config=SearchConfig(),
        conn=sqlite3.connect(":memory:"),
        lean=_DummyLean(),
        theorem_id_map={},
        sentence_encoder=None,
        search_fn=fake_search_fn,
    )

    assert result.started is True
    assert result.progressed is True
    assert result.closed is False
    assert result.first_order_search is not None
    assert result.ducky_pass_2 is not None
    assert result.symbolic_close_pass_2 is not None
    assert result.rarified_gap_packet is not None
    assert result.rarified_gap_packet["final_goal_count"] == 1
    assert result.stage_trace[-1]["stage"] == "post_ducky_symbolic_2"


def test_run_hardtail_bridge_on_row_clears_goals_on_ducky_close(monkeypatch) -> None:
    row = {
        "theorem_id": "T.ducky_close",
        "residual_bucket": "single_goal_stall",
        "last_goal_bucket": "forall",
        "last_goal": "⊢ True",
        "goal_state": "⊢ True",
    }

    def fake_replay(_row, _lean):
        return DuckyReplayState(
            theorem_id="T.ducky_close",
            file_path="",
            goal_state="⊢ True",
            goal_kind="theorem_faithful",
            theorem_faithful=True,
            tier_used="B",
            replay_success=True,
        )

    def fake_ducky_on_goal(_goal, **_kwargs):
        return _ducky_result(
            theorem_id="T.ducky_close",
            progressed=True,
            closed=True,
            goals_after=[],
            final_goal="",
        )

    monkeypatch.setattr("src.hardtail_bridge.replay_residual_state", fake_replay)
    monkeypatch.setattr("src.hardtail_bridge.run_ducky_on_goal", fake_ducky_on_goal)

    result = run_hardtail_bridge_on_row(
        row,
        packet=None,
        pipeline=Pipeline(
            encoder=_DummyPipeline.encoder,
            analyzer=_DummyPipeline.analyzer,
            bridge=_DummyPipeline.bridge,
            navigator=_DummyPipeline.navigator,
        ),
        search_config=SearchConfig(),
        conn=sqlite3.connect(":memory:"),
        lean=_DummyLean(),
        theorem_id_map={},
        sentence_encoder=None,
        search_fn=lambda **_kwargs: SearchResult(success=False, theorem_id="T.ducky_close"),
    )

    assert result.closed is True
    assert result.closed_by == "dr_ducky_pass_1"
    assert result.final_goal_count == 0
    assert result.final_goals == []
    ducky_stage = next(s for s in result.stage_trace if s.get("stage") == "dr_ducky_pass_1")
    assert ducky_stage["goals_after"] == []
