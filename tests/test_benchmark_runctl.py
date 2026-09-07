"""Intent tests for src/benchmark_runctl.py — the per-theorem watchdog and the resume plan that
`scripts/run_benchmark.py` (and through it the MCP `prove_corpus`) rely on."""

from __future__ import annotations

import signal
import time

import pytest

from src.benchmark_runctl import SearchTimeout, split_resume, theorem_watchdog


def test_watchdog_raises_when_the_budget_is_exceeded_and_restores_the_handler():
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(SearchTimeout):
        with theorem_watchdog(1):
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) is before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_watchdog_is_silent_when_the_work_finishes_in_time():
    with theorem_watchdog(5):
        time.sleep(0.05)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_watchdog_disabled_for_a_non_positive_budget():
    with theorem_watchdog(0):
        time.sleep(0.05)


def test_split_resume_keeps_recorded_rows_once_and_pends_the_rest_in_corpus_order():
    theorems = [{"theorem_id": t} for t in ("a", "b", "c", "d")]
    prior = [
        {"theorem_id": "b", "success": True},
        {"theorem_id": "zzz_not_in_corpus", "success": True},
        {"theorem_id": "b", "success": False},  # duplicate from an interrupted append
        {"theorem_id": "d", "success": False},
    ]
    pending, kept = split_resume(theorems, prior)
    assert [t["theorem_id"] for t in pending] == ["a", "c"]
    assert kept == [{"theorem_id": "b", "success": True}, {"theorem_id": "d", "success": False}]


def test_split_resume_with_nothing_prior_pends_everything():
    theorems = [{"theorem_id": "a"}, {"theorem_id": "b"}]
    assert split_resume(theorems, []) == (theorems, [])
    assert split_resume([], [{"theorem_id": "a"}]) == ([], [])
