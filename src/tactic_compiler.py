"""Tactic compiler — the pure, learning-agnostic lowering stage of the pipeline.

This module is the single home of the family→tactic compilation grammar.
It is a total function of its inputs: give it a tactic family and a ranked
premise list — from ANY source: the cosine lane, the symbolic IDF lane, an
oracle, a hand-written list — and it returns the ordered Lean tactic
candidates the pipeline should attempt. No model, no I/O, no state.

This is the v1 of the compiler: the v0 template sketch (removed 2026-08)
predated the benchmark learning; these templates ARE that learning, and
every one of them is deterministic:

    rw    → per-premise direction inference (rw_scoper.infer_direction:
            backward-first for _symm/_comm-style names, forward-first when
            the premise head is visible in the goal target)
    exact → direct term application only (no wildcard escalation — the
            `exact p _` pattern is the documented 0%-success premise_direct
            failure mode; see the Dr. Ducky paper)
    apply → the escalation ladder: exact → refine ?_ ?_ → refine ?_ ?_ ?_
            → apply, per premise (from the EXP-047 "Function expected"
            failure analysis)
    simp  → bare simp first, then simp [top1] only (top-5 hinted simp
            underperforms top-1 in benchmarks — more hints hurt acceptance)

v2 adds the composite forms the traced corpus actually uses (the
EXP-RF-005b residual autopsy: 33/48 short-proof failures had their true
premises at IDF rank 1–4 while the v1 grammar could not utter the true
FORM — `rw [A, B]` chains, `simp [A, B]` sets, and the `simpa` family,
which was absent entirely, 7 true-tactic hits). The v1 single-hint simp
lesson was cosine-era evidence: noisy hints hurt; rank-correct symbolic
hints ARE the true form. Each v2 form is its own family so the censor
store can fence it on its own observed spine, never on the singles':

    rw_chain → ordered two-lemma rewrite chains over the top ranks
    simp_set → two- and three-hint simp over the top ranks
    simpa    → bare simpa, then simpa using/[] per premise

The statistical layer above this module chooses WHICH premises and WHICH
family to feed it (cosine ranking, ExecSelector, family classifier). The
grammar itself never depends on that layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rw_scoper import infer_direction


@dataclass
class TacticCandidate:
    """A compiled tactic candidate ready for Lean verification."""

    text: str
    family: str
    premises_used: list[str] = field(default_factory=list)
    confidence: float = 0.0


def compile_rw(
    premises: list[str],
    goal_target: str = "",
    max_premises: int = 4,
) -> list[TacticCandidate]:
    """Direction-aware rewrite candidates, per premise in rank order.

    For each premise, both orientations are emitted; the more likely one
    (per symbolic redex matching against the goal target) comes first.
    "ambiguous" is treated as forward-first, matching the live lane.
    """
    candidates: list[TacticCandidate] = []
    for p in premises[:max_premises]:
        forward = f"rw [{p}]"
        backward = f"rw [← {p}]"
        ordered = (
            [backward, forward]
            if infer_direction(p, goal_target) == "backward"
            else [forward, backward]
        )
        for text in ordered:
            candidates.append(TacticCandidate(text=text, family="rw", premises_used=[p]))
    return candidates


def compile_exact(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """Direct term application — one candidate per premise, nothing else."""
    return [
        TacticCandidate(text=f"exact {p}", family="exact", premises_used=[p])
        for p in premises[:max_premises]
    ]


def compile_apply(premises: list[str], max_premises: int = 4) -> list[TacticCandidate]:
    """The apply escalation ladder, per premise in rank order.

    exact closes completely on full unification; refine lets the
    elaborator fill implicits without "Function expected" failures;
    bare apply is the fallback that may leave metavariable holes.
    """
    candidates: list[TacticCandidate] = []
    for p in premises[:max_premises]:
        for text in (
            f"exact {p}",
            f"refine {p} ?_ ?_",
            f"refine {p} ?_ ?_ ?_",
            f"apply {p}",
        ):
            candidates.append(TacticCandidate(text=text, family="apply", premises_used=[p]))
    return candidates


def compile_simp(premises: list[str]) -> list[TacticCandidate]:
    """Bare simp first, then a single top-1 hint. Never multi-hint."""
    candidates = [TacticCandidate(text="simp", family="simp")]
    if premises:
        candidates.append(
            TacticCandidate(
                text=f"simp [{premises[0]}]",
                family="simp",
                premises_used=[premises[0]],
            )
        )
    return candidates


def compile_rw_chain(
    premises: list[str],
    goal_target: str = "",
    max_premises: int = 3,
) -> list[TacticCandidate]:
    """Ordered two-lemma rewrite chains — the dominant true form v1 missed.

    `rw [A, B]` requires A to fire first, so order matters and both orders
    are emitted, rank-first. The chain is atomic where sequential single
    rewrites wander: the intermediate state never re-enters lane scheduling.
    """
    candidates: list[TacticCandidate] = []
    top = premises[:max_premises]
    for i, a in enumerate(top):
        for j, b in enumerate(top):
            if i == j:
                continue
            candidates.append(
                TacticCandidate(
                    text=f"rw [{a}, {b}]", family="rw_chain", premises_used=[a, b]
                )
            )
    # Rank-order triples (EXP-RF-007 promotion: rw#3/#4+ recur in the
    # residual proofs and the corpus mode is 3+; v2 stopped at pairs).
    # Bounded: the rank-ordered triple plus its two adjacent swaps.
    if len(top) >= 3:
        a, b, c = top[0], top[1], top[2]
        for trip in ((a, b, c), (b, a, c), (a, c, b)):
            candidates.append(
                TacticCandidate(
                    text=f"rw [{trip[0]}, {trip[1]}, {trip[2]}]",
                    family="rw_chain",
                    premises_used=list(trip),
                )
            )
    return candidates


def compile_rw_at(
    premises: list[str],
    hypotheses: list[str],
    max_premises: int = 3,
    max_hyps: int = 3,
) -> list[TacticCandidate]:
    """Hypothesis-side rewriting — the axis the grammar never had.

    Promotion warrant (EXP-RF-007 proof corpus + §10.7): `rw [...] at h`
    recurs in 7/20 residual proofs while the v1/v2 grammar only ever
    rewrote the GOAL; corpus rw head share 18.4%. Forward orientation
    per (premise, hypothesis) pair, rank-first; `rwa` variant closes
    outright when the rewritten hypothesis IS the goal.
    """
    candidates: list[TacticCandidate] = []
    for p in premises[:max_premises]:
        for h in hypotheses[:max_hyps]:
            candidates.append(
                TacticCandidate(text=f"rw [{p}] at {h}", family="rw_at", premises_used=[p])
            )
            candidates.append(
                TacticCandidate(text=f"rwa [{p}] at {h}", family="rw_at", premises_used=[p])
            )
    return candidates


def compile_simp_set(premises: list[str]) -> list[TacticCandidate]:
    """Two- and three-hint simp over the top ranks.

    Supersedes the v1 single-hint ceiling for the symbolic lane: the
    cosine-era finding ("more hints hurt") measured noisy hints; the
    autopsy's true proofs use exactly these named-lemma sets.
    """
    candidates: list[TacticCandidate] = []
    if len(premises) >= 2:
        candidates.append(
            TacticCandidate(
                text=f"simp [{premises[0]}, {premises[1]}]",
                family="simp_set",
                premises_used=list(premises[:2]),
            )
        )
    if len(premises) >= 3:
        candidates.append(
            TacticCandidate(
                text=f"simp [{premises[0]}, {premises[1]}, {premises[2]}]",
                family="simp_set",
                premises_used=list(premises[:3]),
            )
        )
    return candidates


def compile_simpa(premises: list[str], max_premises: int = 3) -> list[TacticCandidate]:
    """The simpa family — absent from v1 entirely (7 true-tactic hits).

    Bare simpa first (closes `simp`-normal goals against a hypothesis),
    then per premise: `simpa using p` (transport p through simp into the
    goal) before `simpa [p]` (p as a simp hint).
    """
    candidates = [TacticCandidate(text="simpa", family="simpa")]
    for p in premises[:max_premises]:
        candidates.append(
            TacticCandidate(text=f"simpa using {p}", family="simpa", premises_used=[p])
        )
        candidates.append(
            TacticCandidate(text=f"simpa [{p}]", family="simpa", premises_used=[p])
        )
    return candidates


def compile_tactic(
    family: str,
    premises: list[str],
    goal_target: str = "",
    confidence: float = 0.0,
    max_premises: int = 4,
    hypotheses: list[str] | None = None,
) -> list[TacticCandidate]:
    """Compile ordered tactic candidates for one family.

    Unknown families return [] — the caller falls through to other lanes.
    `hypotheses` feeds the hypothesis-side families (rw_at); goal-side
    families ignore it.
    """
    if family == "rw":
        candidates = compile_rw(premises, goal_target, max_premises=max_premises)
    elif family == "exact":
        candidates = compile_exact(premises, max_premises=max_premises)
    elif family == "apply":
        candidates = compile_apply(premises, max_premises=max_premises)
    elif family == "simp":
        candidates = compile_simp(premises)
    elif family == "rw_chain":
        candidates = compile_rw_chain(premises, goal_target)
    elif family == "rw_at":
        candidates = compile_rw_at(premises, hypotheses or [])
    elif family == "simp_set":
        candidates = compile_simp_set(premises)
    elif family == "simpa":
        candidates = compile_simpa(premises)
    else:
        return []
    for c in candidates:
        c.confidence = confidence
    return candidates


def compile_from_predictions(
    family_ranking: list[tuple[str, float]],
    premises: list[str],
    goal_target: str = "",
    max_families: int = 3,
    max_candidates_per_family: int = 8,
) -> list[TacticCandidate]:
    """Compile candidates from a ranked family list — source-agnostic.

    The ranking may come from the learned family classifier, from the
    deterministic goal-bucket mapping, or from a uniform prior; the
    compilation is identical in every case.
    """
    all_candidates: list[TacticCandidate] = []
    for family, conf in family_ranking[:max_families]:
        candidates = compile_tactic(family, premises, goal_target, confidence=conf)
        all_candidates.extend(candidates[:max_candidates_per_family])
    return all_candidates
