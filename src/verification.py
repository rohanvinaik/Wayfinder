"""
Lean proof verification interface.

Provides the deterministic verification layer for Balanced Sashimi.
In the proof domain, this replaces ShortcutForge's lint/parse/validate/compile
pipeline with Lean's kernel-level proof checking.

Phase 1: Stub interface (returns structural checks only).
Phase 2: Connect to Lean LSP via MathLinter integration.
Phase 3: Connect to Pantograph for goal-state-level tactic verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts import VerificationResult


@dataclass
class VerificationConfig:
    """Configuration for the verification backend."""

    backend: str = "stub"  # "stub", "lsp", "pantograph"
    timeout_s: float = 30.0
    lean_project_path: str | None = None


class ProofVerifier:
    """Verifies generated proofs against the Lean kernel.

    Supports multiple backends:
        - "stub": Structural checks only (no Lean server needed).
        - "lsp": Full verification via Lean Language Server.
        - "pantograph": Goal-state-level tactic verification.
    """

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()

    def verify(self, theorem_statement: str, proof_text: str) -> VerificationResult:
        """Verify a proof against the Lean kernel.

        Args:
            theorem_statement: The theorem to prove (Lean syntax).
            proof_text: The tactic proof body.

        Returns:
            VerificationResult with verified=True/False and diagnostics.
        """
        if self.config.backend == "stub":
            return self._verify_stub(theorem_statement, proof_text)
        if self.config.backend == "lsp":
            return self._verify_lsp(theorem_statement, proof_text)
        if self.config.backend == "pantograph":
            return self._verify_pantograph(theorem_statement, proof_text)
        raise ValueError(f"Unknown backend: {self.config.backend}")

    def verify_tactic(self, goal_state: str, tactic: str) -> VerificationResult:
        """Verify a single tactic application against a goal state.

        Only available with the "pantograph" backend.
        """
        if self.config.backend == "pantograph":
            return self._verify_tactic_pantograph(goal_state, tactic)
        return VerificationResult(
            verified=False,
            goal_state=goal_state,
            tactic_trace=[tactic],
            error_message="Tactic verification requires pantograph backend",
        )

    # -- Stub backend (Phase 1) --

    def _verify_stub(self, theorem_statement: str, proof_text: str) -> VerificationResult:
        """Structural checks only — no actual Lean verification."""
        tactics = [line.strip() for line in proof_text.strip().splitlines() if line.strip()]

        if not tactics:
            return VerificationResult(
                verified=False,
                goal_state=theorem_statement,
                tactic_trace=[],
                error_message="Empty proof",
            )

        if any(t == "sorry" for t in tactics):
            return VerificationResult(
                verified=False,
                goal_state=theorem_statement,
                tactic_trace=tactics,
                error_message="Proof contains sorry",
            )

        # Stub: assume structurally plausible proofs pass
        return VerificationResult(
            verified=True,
            goal_state=theorem_statement,
            tactic_trace=tactics,
            steps_used=len(tactics),
        )

    # -- LSP backend (Phase 2) --

    def _verify_lsp(self, theorem_statement: str, proof_text: str) -> VerificationResult:
        """Verify via Lean Language Server Protocol."""
        # TODO: Connect to MathLinter's LSP session
        raise NotImplementedError("LSP verification not yet implemented")

    # -- Pantograph backend (Phase 3) --

    def _verify_pantograph(self, theorem_statement: str, proof_text: str) -> VerificationResult:
        """Verify via Pantograph goal-state RPC."""
        # TODO: Connect to PyPantograph
        raise NotImplementedError("Pantograph verification not yet implemented")

    def _verify_tactic_pantograph(self, goal_state: str, tactic: str) -> VerificationResult:
        """Verify a single tactic via Pantograph."""
        raise NotImplementedError("Pantograph tactic verification not yet implemented")


def check_proof_structural(proof_text: str) -> dict[str, Any]:
    """Quick structural analysis of a proof (no Lean server needed).

    Returns dict with: has_sorry, tactic_count, uses_automation, etc.
    """
    tactics = [line.strip() for line in proof_text.strip().splitlines() if line.strip()]
    automation_tactics = {"simp", "omega", "linarith", "norm_num", "decide", "aesop", "tauto"}

    return {
        "tactic_count": len(tactics),
        "has_sorry": any(t == "sorry" for t in tactics),
        "uses_automation": any(
            t.split()[0] in automation_tactics for t in tactics if t
        ),
        "tactics": tactics,
    }
