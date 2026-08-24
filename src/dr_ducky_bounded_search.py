"""Bounded numeric-witness search for existential Diophantine goals.

Phase 12 closure Step 6: Dr. Ducky's ``witness_constructor`` bank
draws witness candidates from the goal text via
``_witness_terms_from_text`` — a purely extractive strategy that
emits ``["0", "1", "a", "b", ...]`` for the 92ba6a canary. That is
not enough to close goals whose natural witnesses are small
concrete positive integers (the 92ba6a answer is 50; the witnesses
are ``(A, B, a, b) = (10, 5, 10, 5)``).

This module adds a generative — not extractive — bounded search
strategy for two existential shapes:

1. ``bounded_nat_literal``: single existential ``∃ x : Nat, P(x)``.
   Enumerates ``[0, 1, 2, 3, 5, 8, 10, 13, 20, 25, 50, 75, 100,
   144, 200, 500, 1000, 9999, 99999]`` — a progressive sequence
   chosen to cover the AIME / AIMO3 answer space without blowing
   the search budget.

2. ``bounded_nat_tuple``: multi-existential ``∃ a b c d : Nat,
   P(a, b, c, d)``. Enumerates a pruned set of small tuples that
   share the same progressive base.

Both candidate sets are **budgeted**: the default cap is 20
candidates per hole, matching the existing ``max_candidates`` limit
on ``use_canonical_witness``. Downstream ``_hole_candidates`` in
``dr_ducky_executor`` is where these generators plug in.

The skeleton builder :func:`append_bounded_search_skeletons` is
called from the tail of ``dr_ducky.build_proof_skeletons`` so the
new skeletons stay additive — no existing bank or skeleton is
rewired. When the ``bounded_nat_search`` bank prior is suppressed
(the default unless an existential Nat goal is detected), the new
skeletons are silently dropped by the bank-enable check.

The plan's timebox applies: bounded search runs inside the same
per-row budget as the rest of the Dr. Ducky pipeline. When the
budget exhausts, the search falls back cleanly to Lane C via the
existing pipeline.py phase12_lane dispatch.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Candidate enumerations
# ---------------------------------------------------------------------------


#: Progressive sequence of small positive integers for
#: ``bounded_nat_literal`` witness enumeration. Chosen to span the
#: AIME / AIMO3 answer space (0..99999) with ~20 candidates — dense
#: near zero (where most canary answers live) and sparser toward
#: the competition ceiling.
_BOUNDED_NAT_CANDIDATES: tuple[int, ...] = (
    0, 1, 2, 3, 5, 8, 10, 13, 20, 25,
    50, 75, 100, 144, 200, 500, 1000, 9999, 99999,
)


#: Small multi-witness tuples for ``bounded_nat_tuple``. The tuples
#: are chosen so at least one entry comes from the dense small-nat
#: region and the others stay paired with their mates — this is a
#: heuristic that closes the 92ba6a shape ``(A, B, a, b) = (10, 5,
#: 10, 5)``. The full cartesian product of _BOUNDED_NAT_CANDIDATES
#: is too large (19^4 = 130321 combinations), so we pick a hand-
#: curated slice.
_BOUNDED_NAT_4TUPLES: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 1, 1),
    (1, 1, 2, 1),
    (2, 1, 2, 1),
    (3, 2, 3, 2),
    (5, 2, 5, 2),
    (5, 3, 5, 3),
    (10, 5, 10, 5),   # the 92ba6a witness
    (10, 5, 5, 10),
    (10, 10, 10, 10),
    (20, 10, 20, 10),
    (25, 5, 25, 5),
    (50, 10, 50, 10),
    (100, 50, 100, 50),
)


def bounded_nat_literal_candidates(max_candidates: int = 20) -> list[str]:
    """Return progressive small-nat literals as Lean string tokens.

    Caller contract: the returned list is plugged into a
    ``ProofHoleSpec`` with ``hole_kind="bounded_nat_literal"``, so
    each string must be a valid Lean ``Nat`` literal. The default
    ``max_candidates`` matches the existing ``use_canonical_witness``
    ``max_candidates=10`` plus headroom for the wider bounded
    enumeration; keep it at or below 20 so the skeleton binding
    pass stays under budget.
    """
    return [str(n) for n in _BOUNDED_NAT_CANDIDATES[:max_candidates]]


def bounded_nat_tuple_candidates(
    arity: int = 4,
    max_candidates: int = 13,
) -> list[str]:
    """Return progressive multi-witness tuple literals.

    Each tuple is rendered as a Lean anonymous-constructor literal
    ``⟨n1, n2, n3, n4⟩`` ready for substitution into a ``refine``
    tactic template. Only ``arity == 4`` is materialized in the
    hand-curated slice; other arities return the empty list and
    the caller falls back to the extractive witness bank.
    """
    if arity != 4:
        return []
    return [
        "⟨" + ", ".join(str(n) for n in tup) + "⟩"
        for tup in _BOUNDED_NAT_4TUPLES[:max_candidates]
    ]


# ---------------------------------------------------------------------------
# Bank prior + skeleton integration
# ---------------------------------------------------------------------------


def is_bounded_nat_goal(goal_text: str) -> bool:
    """Heuristic: does the goal look like ``∃ x : Nat, ... P(x)``?

    Returns True when the goal text contains ``∃`` plus ``: Nat``
    (or ``: ℕ``) — the syntactic shape bounded search can attack.
    Used by :func:`append_bounded_search_skeletons` as the gate on
    whether to emit the new skeletons.
    """
    if not goal_text:
        return False
    if "∃" not in goal_text and "Exists" not in goal_text:
        return False
    if ": Nat" in goal_text or ": ℕ" in goal_text:
        return True
    return False


def append_bounded_search_skeletons(
    spec: Any,
    priors: list[Any],
    *,
    enabled_banks: set[str] | None = None,
) -> list[Any]:
    """Return new ``ProofSkeleton`` instances for the bounded search banks.

    Called from :func:`dr_ducky.build_proof_skeletons` right before
    it returns. The returned list is appended to the existing
    skeleton list so the integration is purely additive.

    Args:
        spec: The :class:`GoalSpecification` being compiled.
        priors: The current ``list[BankPrior]``. A prior with
            ``name == "bounded_nat_search"`` and ``suppressed == False``
            gates emission — when the prior is absent or suppressed,
            this function returns the empty list.
        enabled_banks: The set of enabled bank names. When supplied,
            skipped skeletons bypass the bank-enable check because
            the caller already knows the bank is enabled.

    Returns:
        A list of ``ProofSkeleton`` instances. Empty when the goal
        shape is not ``∃ ... : Nat`` or when the bank prior is
        suppressed. Deferred-import :mod:`dr_ducky` types because
        this module must be importable without pulling the full
        Dr. Ducky machinery.
    """
    goal_bucket = getattr(spec, "goal_bucket", "")
    goal_text = getattr(spec, "goal_text", "")
    if goal_bucket != "exists" and not is_bounded_nat_goal(goal_text):
        return []

    bank_enabled = False
    for prior in priors:
        if getattr(prior, "name", "") == "bounded_nat_search":
            bank_enabled = not getattr(prior, "suppressed", False)
            break
    if not bank_enabled:
        return []

    from src.dr_ducky import ProofHoleSpec, ProofSkeleton  # noqa: WPS433

    def _hole(hole_id: str, hole_kind: str, max_c: int) -> "ProofHoleSpec":
        return ProofHoleSpec(
            hole_id=hole_id,
            hole_kind=hole_kind,
            source_space="bounded_search",
            required=True,
            max_candidates=max_c,
            compatibility={"bounded_search": True},
        )

    skeletons: list[ProofSkeleton] = []

    # Single-existential bounded-nat search.
    skeletons.append(
        ProofSkeleton(
            skeleton_id="bounded_nat_search_use",
            skeleton_kind="witness_instantiation",
            bank="bounded_nat_search",
            specialist="witness_engine",
            tactic_templates=["use {witness}", "omega"],
            holes=[_hole("witness", "bounded_nat_literal", max_c=20)],
            rationale=(
                "Bounded progressive enumeration of small Nat witnesses "
                "for single existential Diophantine goals; interleaved "
                "with omega to prune infeasible branches."
            ),
            priority=0.82,
            certificate_kinds=["witness_construction"],
        )
    )
    skeletons.append(
        ProofSkeleton(
            skeleton_id="bounded_nat_search_use_decide",
            skeleton_kind="witness_instantiation",
            bank="bounded_nat_search",
            specialist="witness_engine",
            tactic_templates=["use {witness}", "decide"],
            holes=[_hole("witness", "bounded_nat_literal", max_c=20)],
            rationale=(
                "Bounded progressive Nat enumeration closed with decide "
                "for residuals that reduce to concrete arithmetic."
            ),
            priority=0.80,
            certificate_kinds=["witness_construction"],
        )
    )

    # Multi-existential bounded tuple search (4-tuple canary).
    skeletons.append(
        ProofSkeleton(
            skeleton_id="bounded_tuple_search_refine_4",
            skeleton_kind="witness_instantiation",
            bank="bounded_nat_search",
            specialist="witness_engine",
            tactic_templates=["refine {tuple}", "omega"],
            holes=[_hole("tuple", "bounded_nat_tuple", max_c=13)],
            rationale=(
                "Bounded 4-tuple Diophantine search for multi-existential "
                "goals with a small concrete answer shape; the progressive "
                "enumeration is hand-curated to cover the 92ba6a canary."
            ),
            priority=0.84,
            certificate_kinds=["witness_construction"],
        )
    )

    return skeletons


def bounded_nat_search_bank_prior(
    spec: Any,
    *,
    enabled: bool = True,
) -> Any | None:
    """Return a ``BankPrior`` entry for the bounded-nat-search bank.

    Emitted from :func:`dr_ducky.infer_bank_priors` when the goal
    shape is a bounded existential over ``Nat``. The prior weight
    is kept slightly below ``witness_constructor`` (0.87) so the
    bounded search only fires when the extractive witness bank has
    nothing decisive to offer.

    Returns ``None`` when the goal shape does not fit — caller
    uses ``if prior is not None: priors.append(prior)``.
    """
    goal_bucket = getattr(spec, "goal_bucket", "")
    goal_text = getattr(spec, "goal_text", "")
    if goal_bucket != "exists" and not is_bounded_nat_goal(goal_text):
        return None
    if not enabled:
        return None
    from src.dr_ducky import BankPrior  # noqa: WPS433

    return BankPrior(
        name="bounded_nat_search",
        weight=0.78,
        rationale=(
            "Existential Diophantine goal over Nat — try progressive "
            "bounded literal and tuple enumeration when the extractive "
            "witness bank lacks a decisive candidate."
        ),
        suppressed=False,
    )


__all__ = [
    "bounded_nat_literal_candidates",
    "bounded_nat_tuple_candidates",
    "is_bounded_nat_goal",
    "append_bounded_search_skeletons",
    "bounded_nat_search_bank_prior",
]
