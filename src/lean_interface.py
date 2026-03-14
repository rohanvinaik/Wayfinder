"""
Lean kernel interface — Pantograph-based tactic verification.

Provides try_tactic() and try_hammer() for sending tactics to the Lean 4
kernel and receiving success/failure with new goal states.

Backends:
    stub        — always fails, for development without Lean
    replay      — matches against registered ground-truth tactics
    pantograph  — real verification via PyPantograph (Lane A)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.nav_contracts import TacticResult

logger = logging.getLogger(__name__)


@dataclass
class LeanConfig:
    """Configuration for Lean kernel connection."""

    backend: str = "stub"  # "stub", "replay", "pantograph"
    timeout: int = 30
    hammer_timeout: int = 60
    project_root: str = ""
    imports: list[str] = field(default_factory=lambda: ["Init"])


class LeanKernel:
    """Interface to the Lean 4 kernel for tactic verification.

    Args:
        config: Connection and timeout configuration.
    """

    def __init__(self, config: LeanConfig | None = None) -> None:
        self.config = config or LeanConfig()
        self._backend = self.config.backend
        # Replay backend: stores ground-truth tactics per goal for offline eval
        self._replay_table: dict[str, list[str]] = {}
        # Pantograph backend: lazy-initialized server + goal state tracking
        self._server: Any | None = None
        self._goal_states: dict[str, Any] = {}

    def close(self) -> None:
        """Shut down the Pantograph server if running."""
        if self._server is not None:
            try:
                del self._server
            except Exception:
                pass
            self._server = None
            self._goal_states.clear()

    def register_ground_truth(self, goal_state: str, tactics: list[str]) -> None:
        """Register ground-truth tactics for replay backend.

        Args:
            goal_state: The goal state text.
            tactics: List of tactic strings that close this goal.
        """
        self._replay_table[goal_state] = tactics

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
        if self._backend == "replay":
            return self._replay_try_tactic(goal_state, tactic)
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
        if self._backend == "replay":
            return self._replay_try_hammer(goal_state, premises, t)
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
    # Replay backend (offline evaluation)
    # ------------------------------------------------------------------

    def _replay_try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Replay: succeed if tactic matches any registered ground-truth tactic.

        Matching is by base tactic name (first word), allowing the navigator
        to propose the right tactic even with different premise arguments.
        """
        gt_tactics = self._replay_table.get(goal_state, [])
        tactic_base = tactic.split()[0] if tactic.strip() else ""

        for gt in gt_tactics:
            gt_base = gt.split()[0] if gt.strip() else ""
            if tactic_base and gt_base and tactic_base == gt_base:
                return TacticResult(
                    success=True,
                    tactic=tactic,
                    premises=[],
                    new_goals=[],
                )

        return TacticResult(
            success=False,
            tactic=tactic,
            premises=[],
            new_goals=[],
            error_message=f"replay: tactic '{tactic_base}' not in ground truth",
        )

    def _replay_try_hammer(
        self, _goal_state: str, premises: list[str], _timeout: int
    ) -> TacticResult:
        """Replay: hammer always fails (only exact tactic matching supported)."""
        return TacticResult(
            success=False,
            tactic=f"aesop (premises: {len(premises)})",
            premises=premises,
            new_goals=[],
            error_message="replay backend: hammer not supported in replay mode",
        )

    # ------------------------------------------------------------------
    # Pantograph backend (Lane A — real Lean verification)
    # ------------------------------------------------------------------

    def _ensure_server(self) -> Any:
        """Lazy-initialize the Pantograph server."""
        if self._server is not None:
            return self._server

        try:
            from pantograph.server import Server  # type: ignore[import-untyped]
        except ImportError as e:
            msg = (
                "PyPantograph is not installed. Install with: "
                "pip install git+https://github.com/stanford-centaur/PyPantograph"
            )
            raise ImportError(msg) from e

        project = self.config.project_root or None
        self._server = Server(
            imports=self.config.imports,
            project_path=project,
            timeout=self.config.timeout,
        )
        logger.info(
            "Pantograph server started (project=%s, imports=%s)",
            project,
            self.config.imports,
        )
        return self._server

    def _get_or_create_goal(self, goal_state: str) -> Any:
        """Get a cached GoalState or create one from a type expression."""
        if goal_state in self._goal_states:
            return self._goal_states[goal_state]

        server = self._ensure_server()
        state = server.goal_start(goal_state)
        self._goal_states[goal_state] = state
        return state

    def goal_start(self, theorem_type: str) -> str:
        """Initialize a goal from a theorem type expression.

        Returns the goal state string for the initial goal.
        Use this at the start of a proof search session.
        """
        server = self._ensure_server()
        state = server.goal_start(theorem_type)
        goal_str = str(state.goals[0].target) if state.goals else theorem_type
        self._goal_states[goal_str] = state
        return goal_str

    def _pantograph_try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Apply a tactic via Pantograph and return the result."""
        from pantograph.server import TacticFailure  # type: ignore[import-untyped]

        try:
            state = self._get_or_create_goal(goal_state)
            server = self._ensure_server()
            new_state = server.goal_tactic(state, tactic=tactic)
        except TacticFailure as e:
            return TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message=str(e),
            )
        except Exception as e:
            return TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message=f"pantograph error: {e}",
            )

        new_goals = [str(g.target) for g in new_state.goals]

        # Cache each new goal state for subsequent tactic applications
        for goal in new_state.goals:
            goal_str = str(goal.target)
            if goal_str not in self._goal_states:
                self._goal_states[goal_str] = new_state

        return TacticResult(
            success=new_state.is_solved or len(new_goals) == 0,
            tactic=tactic,
            premises=[],
            new_goals=new_goals,
        )

    def _pantograph_try_hammer(
        self, goal_state: str, premises: list[str], timeout: int
    ) -> TacticResult:
        """Try hammer tactics (aesop, omega, simp, decide) with premises."""
        hammer_tactics = _build_hammer_tactics(premises)
        for hammer_tactic in hammer_tactics:
            result = self._pantograph_try_tactic(goal_state, hammer_tactic)
            if result.success:
                return TacticResult(
                    success=True,
                    tactic=hammer_tactic,
                    premises=premises,
                    new_goals=result.new_goals,
                )
        return TacticResult(
            success=False,
            tactic=f"hammer ({len(hammer_tactics)} tactics tried)",
            premises=premises,
            new_goals=[],
            error_message="all hammer tactics failed",
        )

    def gc(self) -> None:
        """Garbage-collect unused goal states in the Pantograph server."""
        if self._server is not None:
            try:
                self._server.gc()
            except Exception:
                pass
        self._goal_states.clear()


def _build_hammer_tactics(premises: list[str]) -> list[str]:
    """Build a list of hammer tactic strings to try."""
    tactics = []

    # aesop with premises as simp lemmas
    if premises:
        premise_list = ", ".join(premises[:16])
        tactics.append(f"aesop (add safe [{premise_list}])")

    tactics.append("aesop")
    tactics.append("omega")
    tactics.append("decide")

    if premises:
        premise_list = ", ".join(premises[:16])
        tactics.append(f"simp [{premise_list}]")
    tactics.append("simp")

    return tactics
