"""The censor store — admissible negative constraints over goal signatures.

Wave 2 of THE_REFOUNDING (§3): the negative channel as governed fences, the
port of Detective's censor/κ/promotion machinery (itself measured first in
Regenesis) onto the proof-search domain. A censor asserts, with a recorded
spine of observed near-misses: *no goal matching this signature slice is
closed by this tactic family* — and is consulted BEFORE any Lean call (the
Monty Hall gate, Wesker Layer-2 style: eliminating a fenced family loses
nothing, by the censor's own warrant).

Governance, transported intact (NEG_SPEC Def. 9.3 / Prop. 14.7):

- **spine-sourced** — a candidate exists only as an aggregation of observed
  ATTEMPTED-AND-FAILED events from gate-off trace runs, with ZERO observed
  successes of the family on the slice, and carries its provenance. Mining
  from gate-off runs is what makes the guard structural: a censor can never
  be confirmed by its own effect (`self_confirming_cannot_certify`).
- **retained plurality** — admission never leaves a signature slice with no
  unfenced family (project invariant #5; the well-definedness condition).
- **the gate laws** (Peitho `resolve.gate`, THE_REFOUNDING §4.1b law 5):
  no signature → no fence (elimination requires positive fence evidence,
  never absence of evidence); the gate only ever REMOVES families, each
  removal carrying its censor's warrant.

κ is Detective's marginal coverage over failure events: a censor's worth is
the observed near-miss mass it fences beyond the adopted set; ranking is
greedy max-κ with the κ→0 knee as the stopping rule. v1 coverage is static
set algebra — the machine-checked submodular regime; the bridge/re-flow
dynamics (NEG_SPEC Thm 14.4) enter only when the corpus loop mutates the
graph, and the supermodular-degree measurement is EXP-RF-004's job, not an
assumption here.

The predicate binds on the TARGET-SURFACE dimension slice only
(`CENSOR_SLICE`): trace goals are target-surface text, so a censor mined
there must consult exactly those dimensions at runtime — binding it to the
full signature would silently never match (hypothesis-side reads differ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.structural_reads import GoalReads

# The dimensions a censor predicate may consult: computable from the
# target-surface text both at mining time and at runtime.
CENSOR_SLICE = ("structure", "domain", "automation", "size", "casts", "numerals")

# Lane → tactic family: the one vocabulary joining trace lanes, symbolic
# lanes, and censor predicates. Fences are keyed on the FAMILY.
FAMILY_OF_LANE = {
    "cosine_rw": "rw",
    "cosine_rw_seq": "rw",
    "symbolic_rw": "rw",
    "cosine_exact": "exact",
    "symbolic_exact": "exact",
    "cosine_simp": "simp",
    "symbolic_simp": "simp",
    "cosine_apply": "apply",
    "symbolic_apply": "apply",
    "structural_core": "structural",
    "solver_bootstrap": "structural",
    "interleaved_bootstrap": "structural",
    "automation": "automation",
    "learned": "learned",
    "dr_ducky": "ducky",
}

# Admission verdicts — named codes, never booleans (two different "no"s must
# not collapse into one falsy value).
ADMISSIBLE = "admissible"
INSUFFICIENT_SPINE = "insufficient_spine"
SUCCESS_OBSERVED = "success_observed"
PLURALITY_VIOLATED = "plurality_violated"


def slice_signature(reads: GoalReads) -> tuple[int, ...]:
    """The censor-visible signature: the CENSOR_SLICE signs, slice-ordered."""
    by_bank = {r.bank: r.sign for r in reads.reads}
    return tuple(by_bank[d] for d in CENSOR_SLICE)


def censor_key(goal_bucket: str, slice_sig: tuple[int, ...]) -> str:
    """The stable string key of a censor predicate (bucket + slice signs)."""
    signs = ",".join(str(s) for s in slice_sig)
    return f"{goal_bucket}|{signs}"


@dataclass(frozen=True)
class Censor:
    """One admissible fence, with its spine and its κ at adoption time."""

    key: str  # censor_key(bucket, slice_sig)
    family: str  # the fenced tactic family
    attempts: int  # observed attempted-and-failed events on the slice
    theorems: int  # distinct theorems contributing the spine
    successes: int  # observed successes of the family on the slice (0 by admission)
    kappa: int  # marginal failure-event coverage at adoption
    source: str  # the gate-off run(s) the spine was mined from
    note: str = ""


@dataclass
class CensorStore:
    """The adopted fences, keyed for the gate. A plain container — every
    decision about membership was made by the admission pipeline."""

    censors: list[Censor] = field(default_factory=list)

    def fences_by_key(self) -> dict[str, set[str]]:
        by_key: dict[str, set[str]] = {}
        for c in self.censors:
            by_key.setdefault(c.key, set()).add(c.family)
        return by_key


def spine_verdict(
    attempts: int,
    theorems: int,
    successes: int,
    min_attempts: int = 25,
    min_theorems: int = 8,
) -> str:
    """Clause (i): is the spine sufficient, and unconfirmed by any success?

    Returns a named code: ADMISSIBLE when the observed near-miss mass meets
    the floors and the family NEVER succeeded on the slice in the gate-off
    corpus; SUCCESS_OBSERVED when even one success exists (the fence would
    be false, not merely weak); INSUFFICIENT_SPINE otherwise.
    """
    if successes > 0:
        return SUCCESS_OBSERVED
    if attempts >= min_attempts and theorems >= min_theorems:
        return ADMISSIBLE
    return INSUFFICIENT_SPINE


def plurality_verdict(families_attempted: int, families_fenced_after: int) -> str:
    """Clause (ii): does the slice retain at least one unfenced family?

    `families_attempted` counts families ever observed attempting the slice;
    admission of this censor would leave `families_attempted −
    families_fenced_after` open. Zero open is the degenerate over-fenced
    regime — the quantity the theory calls undefined, not merely unsafe.
    """
    if families_fenced_after >= families_attempted:
        return PLURALITY_VIOLATED
    return ADMISSIBLE


def fenced_families(
    goal_bucket: str,
    slice_sig: tuple[int, ...] | None,
    fences_by_key: dict[str, set[str]],
) -> set[str]:
    """The Monty Hall gate: which families are fenced for this goal.

    Gate laws: a goal with no computable signature is fenced by NOTHING
    (conservative keep — elimination requires positive evidence); otherwise
    exactly the admitted fences for its key. The gate never adds, scores,
    or reorders — it only removes with a warrant.
    """
    if slice_sig is None:
        return set()
    return set(fences_by_key.get(censor_key(goal_bucket, slice_sig), ()))


def load_fences(path: str) -> dict[str, set[str]]:
    """Load an adopted censor store into the gate's key → fenced-families map.

    Reads only the `adopted` list — everything else in the artifact is
    provenance. A missing/empty path yields no fences (the gate-off default).
    """
    import json
    from pathlib import Path

    if not path or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text())
    fences: dict[str, set[str]] = {}
    for row in payload.get("adopted", []):
        fences.setdefault(row["key"], set()).add(row["family"])
    return fences


def marginal_kappa(candidate_events: frozenset, covered: set) -> int:
    """κ(c | S): the failure events this censor fences beyond the adopted set
    (Detective `kappa.marginal_coverage`, static-set regime)."""
    return len(candidate_events - covered)


def rank_by_kappa(
    candidates: dict[tuple[str, str], frozenset],
    min_kappa: int = 10,
) -> list[tuple[tuple[str, str], int]]:
    """Greedy max-κ adoption order with the κ-knee stopping rule.

    `candidates` maps (key, family) → its failure-event set. Returns the
    adoption sequence with each censor's marginal κ at adoption, stopping
    when the best remaining marginal κ falls below `min_kappa` (the knee —
    a principled halting condition, not a cap). Deterministic: ties break
    on the (key, family) pair.
    """
    remaining = dict(candidates)
    covered: set = set()
    adopted: list[tuple[tuple[str, str], int]] = []
    while remaining:
        best_pair, best_kappa = None, -1
        for pair in sorted(remaining):
            kappa = marginal_kappa(remaining[pair], covered)
            if kappa > best_kappa:
                best_pair, best_kappa = pair, kappa
        if best_pair is None or best_kappa < min_kappa:
            break
        covered |= remaining.pop(best_pair)
        adopted.append((best_pair, best_kappa))
    return adopted
