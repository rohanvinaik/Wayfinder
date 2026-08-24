"""Structural reads — the six banks as computed positions (THE_REFOUNDING §4).

The state estimator of the re-founded system: each bank reads a goal state
deterministically against a mined or documented norm, emitting a signed-ternary
position whose signature is simultaneously the coordinate, the σ_search test,
and the censor predicate vocabulary (one vocabulary, three consumers).

The algebra is Peitho's, transported precisely (THE_REFOUNDING §4.1b):

- a position is a FIVE-tuple — bank, ternary sign, magnitude (depth), the
  zero it was read against, and the evidence path — never a bare sign;
- the mined zero is median-of-active (non-participants cannot drag the norm);
- ``0`` is a first-class verdict ("this bank has no opinion here"), including
  the honest abstention of proof-side banks on an open goal;
- banks never fuse or average — the verdict layer is set-valued interference,
  where one confident opposition vetoes a pile of weak supports;
- ``discriminates`` is the executable form of the discrimination guarantee:
  two goal populations with different closing behavior sharing one signature
  is a structural bug fixed by a NEW orthogonal read, never a tuned threshold.

Goal-side rules are promoted from the extraction labeler
(scripts/extract_proof_network.py) and hard_data_tags — the ground-truth
generator becoming the runtime. Where the extraction rule consumed the PROOF
(tactic names: DEPTH, and the tactic halves of STRUCTURE/CONTEXT/
DECOMPOSITION), the open-goal read either substitutes the goal-side rule the
README's bank table documents, or abstains with the reason in evidence.

The estimator is pinned harder than any controller above it (Peitho DESIGN
§10): every pure decision here is Detective-converged and intent-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.tactic_maps import DOMAIN_PATTERNS  # shared vocabulary; slated for src/ in a later wave
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text

BANKS = ("structure", "domain", "depth", "automation", "context", "decomposition")

# Interference verdicts (Peitho otp, named codes — never booleans).
CONSTRUCTIVE = "constructive"
DESTRUCTIVE = "destructive"
AMBIGUOUS = "ambiguous"
SILENT = "silent"

# Goal-side marker vocabularies. Canonical home going forward; dr_ducky's
# private copies migrate here in a later wave (THE_REFOUNDING §8).
ABSTRACT_MARKERS = (
    "CategoryTheory",
    "Functor",
    "NatTrans",
    "Adjunction",
    "Limits",
    "AlgebraicGeometry",
    "essImage",
    "IsIso",
    "FormallyUnramified",
    "HasRingHomProperty",
    "Ordinal",
    "Cardinal",
)
DECIDABLE_MARKERS = ("Decidable", "decide", "Bool", "Nat.gcd", "%", "∣")
ARITH_MARKERS = (" + ", " - ", " * ", " / ", "^", "≤", "≥", " < ", " > ")
CONCRETE_TYPE_MARKERS = (": ℕ", ": ℤ", ": ℚ", ": Nat", ": Int", ": Fin", ": Bool")


@dataclass(frozen=True)
class BankRead:
    """One bank's position: sign for the signature, magnitude and norm for the
    ledger row, evidence for the audit trail. The Peitho DimensionPosition."""

    bank: str
    sign: int  # -1 | 0 | +1
    depth: float  # |deviation| or rule magnitude; 0.0 on abstention
    zero_state: float | None  # the mined/documented zero read against; None for rule reads
    evidence: str  # which rule or deviation produced the sign — the justification


@dataclass(frozen=True)
class ContextNorms:
    """The CONTEXT bank's mined zero and tolerance band.

    Defaults encode the documented mode (README bank table: "moderate (3-5
    hyps)" is the zero; "no hypotheses" and "rich (10+)" are the poles).
    Re-mine from a live corpus with :func:`mine_context_zero`.
    """

    zero: float = 4.0
    tol: float = 3.0


DEFAULT_NORMS = ContextNorms()


def ternary(deviation: float, tol: float) -> int:
    """Signed-ternary read of a deviation against a tolerance band (Peitho otp)."""
    if deviation > tol:
        return 1
    if deviation < -tol:
        return -1
    return 0


def mine_context_zero(hyp_counts: list[int]) -> float:
    """Median-of-active mined zero for hypothesis counts.

    Goals with zero hypotheses are the −1 pole by RULE (see read_context), so
    they are non-participants of the "context richness" property and are
    excluded here — the Peitho mined-zero discipline (a non-seller cannot drag
    the norm). Returns the DEFAULT zero when nothing is active.
    """
    active = sorted(c for c in hyp_counts if c > 0)
    if not active:
        return DEFAULT_NORMS.zero
    mid = len(active) // 2
    if len(active) % 2:
        return float(active[mid])
    return (active[mid - 1] + active[mid]) / 2.0


def count_hypotheses(goal_text: str) -> int:
    """Count hypothesis lines in a goal state (promoted from the extraction labeler)."""
    if not goal_text:
        return 0
    count = 0
    for line in goal_text.strip().split("\n"):
        if "⊢" in line or "|-" in line:
            break
        if ":" in line and line.strip():
            count += 1
    return count


def goal_target(goal_text: str) -> str:
    """The ⊢-side of a goal state; the whole text when no turnstile is present."""
    for marker in ("⊢", "|-"):
        if marker in goal_text:
            return goal_text.split(marker, 1)[1].strip()
    return goal_text.strip()


def read_structure(target: str, goal_bucket: str) -> BankRead:
    """STRUCTURE: −1 decidable/arithmetic · 0 equality/rewrite · +1 quantified/dependent."""
    binders = target.count("∀") + target.count("∃") + target.count("Π")
    dependent = "Type" in target or "Sort" in target
    if binders > 0 or dependent:
        why = f"{binders} binder(s)" + (" + dependent-type surface" if dependent else "")
        return BankRead("structure", 1, float(binders + (1 if dependent else 0)), None, why)
    decidable = any(m in target for m in DECIDABLE_MARKERS)
    arith = any(m in target for m in ARITH_MARKERS)
    if decidable or (arith and any(m in target for m in CONCRETE_TYPE_MARKERS)):
        why = "decidable surface" if decidable else "concrete arithmetic surface"
        return BankRead("structure", -1, 1.0, None, why)
    if goal_bucket in ("equality", "iff"):
        return BankRead("structure", 0, 0.0, None, f"{goal_bucket} goal — the rewrite mode")
    return BankRead("structure", 0, 0.0, None, "no structural pole marker fired")


def read_domain(goal_text: str) -> BankRead:
    """DOMAIN: vote of namespace-pattern matches in the goal (−1 concrete … +1 abstract).

    The extraction rule classified one theorem namespace; a goal carries many
    tokens, so the read is a VOTE over every matched pattern, tie → 0. The
    evidence lists the matches, so a surprising sign is auditable to its voters.
    """
    votes: list[tuple[str, int]] = []
    seen: set[str] = set()
    for token in goal_text.replace("(", " ").replace(")", " ").split():
        head = token.split(".", 1)[0].strip("↑,;:{}[]⟨⟩")
        if not head or head in seen:
            continue
        seen.add(head)
        for prefix, sign, _anchors in DOMAIN_PATTERNS:
            if head == prefix:
                votes.append((head, sign))
                break
    if not votes:
        return BankRead("domain", 0, 0.0, None, "no domain pattern matched")
    total = sum(sign for _name, sign in votes)
    sign = 1 if total > 0 else -1 if total < 0 else 0
    named = ", ".join(f"{n}({s:+d})" for n, s in votes[:6])
    return BankRead("domain", sign, float(abs(total)), None, f"vote of {len(votes)}: {named}")


def read_depth() -> BankRead:
    """DEPTH is a proof-side bank (proof length): an open goal has no proof yet.

    The honest read is abstention — the zero that means "no opinion", not
    "shallow". Any depth signal for an open goal must come from a NEW
    goal-side dimension with its own mined zero (the discrimination
    guarantee's fix-shape), never from repurposing this bank.
    """
    return BankRead("depth", 0, 0.0, None, "proof-length bank — no opinion for an open goal")


def read_automation(target: str) -> BankRead:
    """AUTOMATION: −1 hammerable surface · +1 abstract-manual surface · else 0."""
    abstract = [m for m in ABSTRACT_MARKERS if m in target]
    if abstract:
        return BankRead("automation", 1, float(len(abstract)), None, f"abstract markers: {', '.join(abstract[:4])}")
    decidable = any(m in target for m in DECIDABLE_MARKERS)
    concrete_arith = any(m in target for m in ARITH_MARKERS) and any(
        m in target for m in CONCRETE_TYPE_MARKERS
    )
    if decidable or concrete_arith:
        why = "decidable surface" if decidable else "concrete linear-arithmetic surface"
        return BankRead("automation", -1, 1.0, None, f"{why} — hammer-shaped")
    return BankRead("automation", 0, 0.0, None, "no automation pole marker fired")


def read_context(hyp_count: int, norms: ContextNorms = DEFAULT_NORMS) -> BankRead:
    """CONTEXT: a deviation read of hypothesis count against the mined zero.

    The zero-hypothesis goal is the −1 pole by rule (it is a different
    situation, not a small deviation); active goals read as signed deviation
    from the mined median-of-active within the tolerance band.
    """
    if hyp_count == 0:
        return BankRead("context", -1, float(norms.zero), norms.zero, "no hypotheses — the bare-goal pole")
    dev = hyp_count - norms.zero
    return BankRead(
        "context",
        ternary(dev, norms.tol),
        abs(dev),
        norms.zero,
        f"{hyp_count} hypotheses vs mined zero {norms.zero:g} (tol {norms.tol:g})",
    )


def read_decomposition(target: str, goal_bucket: str) -> BankRead:
    """DECOMPOSITION: +1 splittable surface · −1 atomic · else 0."""
    conj = target.count(" ∧ ")
    if conj or goal_bucket == "iff":
        why = f"{conj} conjunction(s)" if conj else "iff — splits into directions"
        return BankRead("decomposition", 1, float(conj or 1), None, why)
    if goal_bucket == "atomic_prop":
        return BankRead("decomposition", -1, 1.0, None, "atomic proposition — nothing to split")
    return BankRead("decomposition", 0, 0.0, None, "single-goal surface")


@dataclass(frozen=True)
class GoalReads:
    """The goal's ledger row, Wave-1 slice: raw facts + the six positions."""

    goal_bucket: str
    hyp_count: int
    reads: tuple[BankRead, ...]

    def signature(self) -> tuple[int, ...]:
        """The ternary signature, bank-ordered — the coordinate AND the justification key."""
        by_bank = {r.bank: r.sign for r in self.reads}
        return tuple(by_bank[b] for b in BANKS)

    def directions(self) -> dict[str, int]:
        """Bank → sign, the StructuredQuery vocabulary."""
        return {r.bank: r.sign for r in self.reads}

    def evidence(self) -> dict[str, str]:
        return {r.bank: r.evidence for r in self.reads}


def read_goal(goal_text: str, norms: ContextNorms = DEFAULT_NORMS) -> GoalReads:
    """Read all six banks for one goal state. Pure; total; every sign warranted."""
    clean = sanitize_goal_text(goal_text or "")
    bucket = classify_goal_bucket(clean)
    target = goal_target(clean)
    hyps = count_hypotheses(clean)
    reads = (
        read_structure(target, bucket),
        read_domain(clean),
        read_depth(),
        read_automation(target),
        read_context(hyps, norms),
        read_decomposition(target, bucket),
    )
    return GoalReads(goal_bucket=bucket, hyp_count=hyps, reads=reads)


def hamming(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> int:
    """Positions at which two signatures differ (Peitho position algebra)."""
    return sum(1 for a, b in zip(sig_a, sig_b) if a != b)


def discriminates(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> bool:
    """The discrimination guarantee, executable: do the signatures differ at all?

    False for two goals that close differently is a structural bug — the fix
    is a new orthogonal read off its own mined zero, never a tuned threshold.
    """
    return hamming(sig_a, sig_b) > 0


def interference(supports: int, opposes: int, support_min: int = 2, oppose_min: int = 1) -> str:
    """Set-valued verdict from ternary bank votes (Peitho otp; never an average).

    CONSTRUCTIVE — enough agreement, no opposition floor reached: emit.
    DESTRUCTIVE — opposition floor reached: censor-grade elimination.
    AMBIGUOUS — both fire, or a sub-floor mix: the escalate/disambiguate signal.
    SILENT — all orthogonal: no structural opinion; fall through to ranking.
    One confident opposition vetoes a pile of weak supports by construction.
    """
    constructive = supports >= support_min
    destructive = opposes >= oppose_min
    if constructive and destructive:
        return AMBIGUOUS
    if destructive:
        return DESTRUCTIVE
    if constructive:
        return CONSTRUCTIVE
    if supports > 0 and opposes > 0:
        return AMBIGUOUS
    return SILENT
