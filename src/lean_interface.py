"""
Lean kernel interface — Pantograph-based tactic verification.

Provides try_tactic() and try_hammer() for sending tactics to the Lean 4
kernel and receiving success/failure with new goal states.

Phase 1: stub backend for development/testing.
Phase 2+: Pantograph LSP integration via user's existing Lean tooling.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.nav_contracts import TacticResult


@dataclass
class LeanConfig:
    """Configuration for Lean kernel connection."""

    backend: str = "stub"  # "stub", "pantograph", "lsp"
    timeout: int = 30
    hammer_timeout: int = 60
    project_root: str = ""


class LeanKernel:
    """Interface to the Lean 4 kernel for tactic verification.

    Args:
        config: Connection and timeout configuration.
    """

    def __init__(self, config: LeanConfig | None = None) -> None:
        self.config = config or LeanConfig()
        self._backend = self.config.backend

    def try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Send a tactic to the Lean kernel and get the result.

        Args:
            goal_state: Current goal state text.
            tactic: Tactic string to apply.

        Returns:
            TacticResult with success/failure, new goals, and error info.
        """
        if self._backend == "stub":
            return self._stub_try_tactic(goal_state, tactic)
        if self._backend == "pantograph":
            return self._pantograph_try_tactic(goal_state, tactic)
        msg = f"Unknown backend: {self._backend}"
        raise ValueError(msg)

    def try_hammer(
        self, goal_state: str, premises: list[str], timeout: int | None = None
    ) -> TacticResult:
        """Delegate to LeanHammer/Aesop with premise suggestions.

        Args:
            goal_state: Current goal state text.
            premises: Suggested premise names for the hammer.
            timeout: Override default hammer timeout.
        """
        t = timeout or self.config.hammer_timeout
        if self._backend == "stub":
            return self._stub_try_hammer(goal_state, premises, t)
        if self._backend == "pantograph":
            return self._pantograph_try_hammer(goal_state, premises, t)
        msg = f"Unknown backend: {self._backend}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Stub backend (Phase 1)
    # ------------------------------------------------------------------

    def _stub_try_tactic(self, _goal_state: str, tactic: str) -> TacticResult:
        """Stub: always fails. For development/testing without Lean."""
        return TacticResult(
            success=False,
            tactic=tactic,
            premises=[],
            new_goals=[],
            error_message="stub backend: no Lean kernel connected",
        )

    def _stub_try_hammer(
        self, _goal_state: str, premises: list[str], _timeout: int
    ) -> TacticResult:
        """Stub: always fails."""
        return TacticResult(
            success=False,
            tactic=f"aesop (premises: {len(premises)})",
            premises=premises,
            new_goals=[],
            error_message="stub backend: no Lean kernel connected",
        )

    # ------------------------------------------------------------------
    # Pantograph backend (Phase 2+)
    # ------------------------------------------------------------------

    def _pantograph_try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Pantograph: send tactic via Pantograph protocol."""
        raise NotImplementedError(
            "Pantograph backend is not yet implemented (Phase 2+). "
            "Use backend='stub' for offline testing or backend='axle' for Lane B verification. "
            "See docs/WAYFINDER_PLAN.md §3.1 for Pantograph integration plan."
        )

    def _pantograph_try_hammer(
        self, goal_state: str, premises: list[str], timeout: int
    ) -> TacticResult:
        """Pantograph: delegate to hammer via Pantograph."""
        raise NotImplementedError(
            "Pantograph backend is not yet implemented (Phase 2+). "
            "Use backend='stub' for offline testing or backend='axle' for Lane B verification. "
            "See docs/WAYFINDER_PLAN.md §3.1 for Pantograph integration plan."
        )
