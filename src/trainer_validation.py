"""Proof validation helpers for the training pipeline.

Replaces DSL validation with Lean proof structural checking.
Full Lean kernel verification requires the LSP/Pantograph backends (Phase 2+).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from src.lowering import lower_proof_to_lean, roundtrip_validate
from src.verification import ProofVerifier, VerificationConfig, check_proof_structural

_ValidationResult = namedtuple(
    "_ValidationResult",
    "lowered structurally_valid no_sorry verified tactic_count",
)


def validate_single_proof(example: Any, verifier: ProofVerifier | None = None) -> _ValidationResult:
    """Lower a ProofExample to Lean and run structural + optional kernel checks."""
    ok, err = roundtrip_validate(example)
    if not ok:
        return _ValidationResult(
            lowered=False,
            structurally_valid=False,
            no_sorry=False,
            verified=False,
            tactic_count=0,
        )

    proof_text = lower_proof_to_lean(example)
    structural = check_proof_structural(proof_text)

    verified = False
    if verifier is not None:
        try:
            result = verifier.verify(example.theorem_statement, proof_text)
            verified = result.verified
        except NotImplementedError:
            pass

    return _ValidationResult(
        lowered=True,
        structurally_valid=not structural["has_sorry"],
        no_sorry=not structural["has_sorry"],
        verified=verified,
        tactic_count=structural["tactic_count"],
    )


def get_default_verifier() -> ProofVerifier:
    """Return a stub verifier (no Lean server needed)."""
    return ProofVerifier(VerificationConfig(backend="stub"))
