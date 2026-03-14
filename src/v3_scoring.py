"""
v3 OTP-grounded scoring — bank-IDF weighting and constraint report composition.

Applies Minority Channel Advantage (MCA) theory: banks that rarely fire
non-zero carry more information per activation. Composes bank alignment,
critic distance, censor score, and anchor matching into a ConstraintReport.

No new neural modules — pure arithmetic on existing model outputs.

See PLAN §7.1 and DESIGN §12.3 for specification.
"""

from __future__ import annotations

import math
import sqlite3

from src.nav_contracts import BANK_NAMES, NavOutput
from src.v3_contracts import ActionCandidate, ConstraintReport


def compute_bank_idf(
    conn: sqlite3.Connection,
) -> dict[str, float]:
    """Compute bank-IDF weights from entity position frequencies.

    For each bank, IDF = log(total / count_nonzero). Banks that rarely
    fire non-zero get high IDF (MCA: rare signal = valuable signal).

    Args:
        conn: SQLite connection to proof_network.db.

    Returns:
        Mapping from bank name to IDF weight.
    """
    total = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM entity_positions").fetchone()[0]
    if total == 0:
        return {b: 1.0 for b in BANK_NAMES}

    idf: dict[str, float] = {}
    for bank in BANK_NAMES:
        row = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM entity_positions WHERE bank = ? AND sign != 0",
            (bank,),
        ).fetchone()
        nonzero = row[0] if row else 0
        # Smoothed IDF: avoid division by zero, dampen extreme values
        idf[bank] = math.log((total + 1) / (nonzero + 1)) + 1.0

    return idf


def apply_bank_idf(
    nav_output: NavOutput,
    bank_idf: dict[str, float],
) -> dict[str, float]:
    """Weight per-bank confidences by IDF.

    Returns adjusted bank scores: confidence * idf for each bank.
    Banks with zero direction are left at 0 (Informational Zero —
    transparent banks don't contribute to scoring).

    Args:
        nav_output: Navigator output with directions and confidences.
        bank_idf: Bank name to IDF weight mapping.

    Returns:
        Mapping from bank name to IDF-weighted score.
    """
    scores: dict[str, float] = {}
    for bank in BANK_NAMES:
        direction = nav_output.directions.get(bank, 0)
        if direction == 0:
            # Informational Zero: orthogonal, contributes nothing
            scores[bank] = 0.0
        else:
            confidence = nav_output.direction_confidences.get(bank, 0.0)
            idf = bank_idf.get(bank, 1.0)
            scores[bank] = confidence * idf
    return scores


def compute_otp_dimensionality(directions: dict[str, int]) -> int:
    """Count active (non-zero) banks — the OTP subspace dimensionality.

    Args:
        directions: Bank name to {-1, 0, +1} mapping.

    Returns:
        Number of non-zero banks (1-6).
    """
    return sum(1 for v in directions.values() if v != 0)


def build_constraint_report(
    bank_scores: dict[str, float],
    critic_distance: float,
    censor_score: float,
    anchor_alignment: float,
    weights: dict[str, float] | None = None,
) -> ConstraintReport:
    """Compose individual constraint channels into a ConstraintReport.

    The total_score is a weighted sum of the four channels.
    Higher total_score = better candidate (bank alignment and anchor
    matching are positive signals; critic distance and censor score
    are costs to minimize).

    Args:
        bank_scores: Per-bank IDF-weighted scores.
        critic_distance: Estimated remaining proof steps (lower = better).
        censor_score: P(failure) from censor (lower = better).
        anchor_alignment: IDF-weighted Jaccard with target anchors.
        weights: Override weights for {bank, critic, censor, anchor}.

    Returns:
        ConstraintReport with computed total_score.
    """
    w = weights or {"bank": 1.0, "critic": 0.5, "censor": 2.0, "anchor": 0.3}

    # Bank term: mean of IDF-weighted bank scores (higher = better alignment)
    active_scores = [s for s in bank_scores.values() if s > 0]
    bank_term = sum(active_scores) / max(len(active_scores), 1)

    # Total: maximize bank alignment and anchor matching,
    # minimize critic distance and censor violation
    total = (
        w.get("bank", 1.0) * bank_term
        + w.get("anchor", 0.3) * anchor_alignment
        - w.get("critic", 0.5) * critic_distance
        - w.get("censor", 2.0) * censor_score
    )

    return ConstraintReport(
        bank_scores=dict(bank_scores),
        critic_distance=critic_distance,
        censor_score=censor_score,
        anchor_alignment=anchor_alignment,
        total_score=total,
    )


def nav_output_to_candidates(
    nav_output: NavOutput,
    tactic_names: list[str],
    premise_names: list[str],
    bank_idf: dict[str, float],
) -> list[ActionCandidate]:
    """Convert navigator output + resolved entities to ActionCandidates.

    Args:
        nav_output: Navigator output with directions and confidences.
        tactic_names: Resolved tactic entity names.
        premise_names: Resolved premise entity names.
        bank_idf: Bank IDF weights for scoring.

    Returns:
        List of ActionCandidates with navigational_scores set.
    """
    scores = apply_bank_idf(nav_output, bank_idf)
    candidates = []
    for tactic in tactic_names:
        candidates.append(
            ActionCandidate(
                tactic=tactic,
                premises=list(premise_names),
                provenance="navigate",
                navigational_scores=dict(scores),
            )
        )
    return candidates
