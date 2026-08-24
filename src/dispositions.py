"""The disposition calculus — Detective's verdict discipline on search rows.

Wave 5 of THE_REFOUNDING (§7): every theorem row resolves to exactly one
named disposition, and the calculus CONSUMES the row's computed fields
(`honest_success`, `self_application_detected`, `residual_bucket`,
`censor_drop_count` — all produced upstream by `augment_result_entry` and
the search loop), never re-derives a narrower proxy. Two different "no"s
never collapse into one falsy value, and no disposition is ever promoted:

  closed_value        kernel-verified, honest — the value kill.
  closed_suspect      self-application closure — reported, never banked
                      (the run kill of theorem search).
  progressed          started and moved (goals closed) without closing.
  blocked_unproven    started, attempted, no close — NEVER promoted to
                      "unprovable"; resolved only by a richer read, a new
                      premise, or the residual hand-off.
  abstained_certified attempts were withheld under warrant (censor fences)
                      rather than spent and lost.
  skipped_start       the goal never built — the compiler-specialist track,
                      not a search verdict.
  infra               crash/timeout — the separate ledger, never a model
                      failure.

`next_action` is the consumed-signal controller: one imperative per
disposition, total, order-free (each row has exactly one disposition, so
there is nothing to outrank). The defect class this exists to prevent is
EXP-049's: a signal measured and never consumed.
"""

from __future__ import annotations

CLOSED_VALUE = "closed_value"
CLOSED_SUSPECT = "closed_suspect"
PROGRESSED = "progressed"
BLOCKED_UNPROVEN = "blocked_unproven"
ABSTAINED_CERTIFIED = "abstained_certified"
SKIPPED_START = "skipped_start"
INFRA = "infra"

DISPOSITIONS = (
    CLOSED_VALUE,
    CLOSED_SUSPECT,
    PROGRESSED,
    BLOCKED_UNPROVEN,
    ABSTAINED_CERTIFIED,
    SKIPPED_START,
    INFRA,
)

# One imperative per disposition — never a menu, never a score.
NEXT_ACTION = {
    CLOSED_VALUE: "settled",
    CLOSED_SUSPECT: "exclude_from_supervision",  # telemetry, not a solve; never train on it
    PROGRESSED: "continue_or_hand_off",  # budget remains -> continue; else render the cheat sheet
    BLOCKED_UNPROVEN: "hand_off_residual",  # the cheat sheet is the artifact; Aristotle/second-order consume it
    ABSTAINED_CERTIFIED: "audit_fences",  # withheld under warrant; the abstention ledger is the record
    SKIPPED_START: "fix_statement",  # the declaration or harness, never the prover
    INFRA: "repair_measurement",  # restart/rerun; not a verdict about the theorem
}


def goal_disposition(
    success: bool,
    honest_success: bool,
    started: bool,
    success_category: str,
    goals_closed: int,
    attempts: int,
    censor_drop_count: int,
) -> str:
    """Resolve one row's disposition from its computed fields.

    Precedence is the calculus, stated in order: an infrastructure failure
    outranks everything (a crash is not a verdict); an unbuilt goal is the
    compiler track; closures split honest/suspect on the already-computed
    `honest_success`; a started row that spent no attempts while the censor
    gate withheld families abstained under warrant; movement without
    closure is progress; the remainder is blocked — and UNPROVEN.
    """
    if success_category == "crashed":
        return INFRA
    if not started:
        return SKIPPED_START
    if success:
        return CLOSED_VALUE if honest_success else CLOSED_SUSPECT
    if attempts == 0 and censor_drop_count > 0:
        return ABSTAINED_CERTIFIED
    if goals_closed > 0:
        return PROGRESSED
    return BLOCKED_UNPROVEN


def next_action(disposition: str) -> str:
    """The ONE next action for a disposition. Unknown codes fail loudly —
    an unrecognized disposition is a vocabulary drift, never a default."""
    return NEXT_ACTION[disposition]


def disposition_of_row(row: dict) -> str:
    """The row adapter: consume `augment_result_entry`'s computed fields."""
    return goal_disposition(
        success=bool(row.get("success")),
        honest_success=bool(row.get("honest_success")),
        started=bool(row.get("started", True)),
        success_category=str(row.get("success_category", "")),
        goals_closed=int(row.get("goals_closed", 0) or 0),
        attempts=int(row.get("attempts", 0) or 0),
        censor_drop_count=int(row.get("censor_drop_count", 0) or 0),
    )
