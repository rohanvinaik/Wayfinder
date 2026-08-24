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


def compile_tactic(
    family: str,
    premises: list[str],
    goal_target: str = "",
    confidence: float = 0.0,
    max_premises: int = 4,
) -> list[TacticCandidate]:
    """Compile ordered tactic candidates for one family.

    Unknown families return [] — the caller falls through to other lanes.
    """
    if family == "rw":
        candidates = compile_rw(premises, goal_target, max_premises=max_premises)
    elif family == "exact":
        candidates = compile_exact(premises, max_premises=max_premises)
    elif family == "apply":
        candidates = compile_apply(premises, max_premises=max_premises)
    elif family == "simp":
        candidates = compile_simp(premises)
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
