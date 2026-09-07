"""Run control for the benchmark runner: the per-theorem watchdog and the resume plan.

Stdlib only, so the pure decision (`split_resume`) is testable and pinnable without loading the
search stack. `scripts/run_benchmark.py` consumes both; the MCP server's `prove_corpus` relies on
them (`--per-theorem-timeout`, `--resume`).
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Generator


class SearchTimeout(Exception):
    """The per-theorem watchdog fired: the search (or a hung kernel) exceeded its budget."""


@contextlib.contextmanager
def theorem_watchdog(seconds: int) -> Generator[None, None, None]:
    """SIGALRM watchdog around one theorem's search (the pattern of the exp scripts). A
    non-positive budget disables it. Raises `SearchTimeout` in the main thread when it fires; the
    previous handler and timer are always restored."""
    if seconds <= 0:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: object) -> None:
        raise SearchTimeout(f"theorem search exceeded the {seconds}s watchdog")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def split_resume(theorems: list[dict], prior: list[dict]) -> tuple[list[dict], list[dict]]:
    """The resume plan (pure). Prior rows whose `theorem_id` is among `theorems` are kept as
    results, in recorded order and once each (a duplicate row from an interrupted append is
    dropped); the theorems with no prior row are pending, in corpus order. Prior rows for
    theorems outside this corpus are not kept."""
    wanted = {t["theorem_id"] for t in theorems}
    kept: list[dict] = []
    seen: set[str] = set()
    for row in prior:
        theorem_id = str(row.get("theorem_id") or "")
        if theorem_id in wanted and theorem_id not in seen:
            kept.append(row)
            seen.add(theorem_id)
    pending = [t for t in theorems if t["theorem_id"] not in seen]
    return pending, kept
