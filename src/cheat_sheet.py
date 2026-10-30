"""The cheat sheet — the residual as a transferable artifact (THE_REFOUNDING §6).

A failed or partial search stops discarding its computed understanding at
the hand-off: this renderer projects the goal-ledger row (the augmented
details row every search loop already emits) into a deterministic English
walk — the AIMO3 discipline, transported:

- **Byte-deterministic.** Same row, same bytes. Iteration orders come from
  the row's own list orderings; the only aggregations are sorted.
- **No hedging vocabulary.** No confidence, difficulty, or likelihood
  words; the sheet states what was computed, eliminated, and left open.
  `test_cheat_sheet.py::test_no_hedging_vocabulary` is the regression fence.
- **Typed input only.** The renderer consumes the row's computed fields and
  the structural-read vocabulary — no NL parsing, no model, no heuristics.
- **Graceful closed-enum fallback.** An unknown lane or family renders as
  its raw string; missing sections are omitted, never guessed.

Consumers: the Aristotle lane (`send_to_aristotle` folds the sheet into the
submission prompt — the measured Gotcha-3 fix), the second-order packet
writers, and the human audit trail.
"""

from __future__ import annotations

from collections import Counter

from src.censor_store import CENSOR_SLICE
from src.dispositions import disposition_of_row, next_action
from src.structural_reads import read_goal

_MAX_RESIDUAL_GOALS = 3
_MAX_TACTICS = 12

_DISPOSITION_SENTENCE = {
    "closed_value": "closed and kernel-verified (honest).",
    "closed_suspect": "closed by self-application — telemetry, not a solve.",
    "progressed": "moved without closing; the residual below is smaller than the start.",
    "blocked_unproven": "attempted without closing — UNPROVEN, not unprovable.",
    "abstained_certified": "attempts withheld under censor warrant.",
    "skipped_start": "the goal never built — a statement/harness item, not a search verdict.",
    "infra": "the measurement failed (crash/timeout) — not a verdict about the theorem.",
}


def residual_goals(row: dict) -> list[str]:
    """The open goals at the end of the trace (the row's own record)."""
    trace = row.get("step_trace") or []
    if trace:
        last = trace[-1]
        goals = last.get("open_goals_after")
        if goals:
            return list(goals)
    return list(row.get("final_goals") or [])


def _signature_walk(goal_text: str) -> list[str]:
    """One residual goal's structural read, each sign with its evidence.

    The read sees the whole goal (hypotheses included, when the text has them);
    the line shows its target.
    """
    reads = read_goal(goal_text)
    target = goal_text.rsplit("⊢", 1)[-1] if "⊢" in goal_text else goal_text
    lines = [f"goal ({reads.goal_bucket}): {' '.join(target.split())[:160]}"]
    for read in reads.reads:
        if read.sign == 0 and read.bank not in CENSOR_SLICE:
            continue  # non-slice zeros carry no hand-off content
        sign = {1: "+1", -1: "-1", 0: " 0"}[read.sign]
        lines.append(f"  {read.bank:14s} {sign}  {read.evidence}")
    return lines


def _lane_history(row: dict) -> list[str]:
    """Per-lane progressed/attempted counts from the trace — sorted, total."""
    progressed: Counter = Counter()
    attempted: Counter = Counter()
    for step in row.get("step_trace") or []:
        if step.get("progress") and step.get("lane"):
            progressed[step["lane"]] += 1
        for lane in step.get("lanes_tried") or []:
            attempted[lane] += 1
    lanes = sorted(set(progressed) | set(attempted))
    return [
        f"  {lane:22s} progressed {progressed.get(lane, 0)} / attempted {attempted.get(lane, 0)}"
        for lane in lanes
    ]


def _negative_channel(row: dict) -> list[str]:
    """The abstention ledger, grouped: which families were fenced, how often."""
    drops = row.get("censor_drops") or []
    by_family: Counter = Counter(d.get("family", "?") for d in drops)
    return [
        f"  {family}: fenced {count} time(s) by observed-failure censors"
        for family, count in sorted(by_family.items())
    ]


_HANDOFF_LINE = (
    "hand-off: prove the statement completely; reuse the verified progress "
    "where it applies; do not add axioms or `sorry`."
)


def render_cheat_sheet(row: dict, manifest_row: dict | None = None, handoff_line: str | None = None) -> str:
    """The full sheet: verdict, progress, reads walk, channels, hand-off.

    `handoff_line` replaces the closing instruction — the residual hand-off
    passes its own (fill the marked holes, keep every line), so the sheet never
    contradicts the skeleton it accompanies.
    """
    theorem_id = str(row.get("theorem_id", ""))
    disposition = disposition_of_row(row)
    sections: list[str] = [
        f"WAYFINDER CHEAT SHEET — {theorem_id}",
        f"verdict: {disposition} — {_DISPOSITION_SENTENCE.get(disposition, disposition)}",
        f"next action: {next_action(disposition)}",
    ]

    if manifest_row and manifest_row.get("full_declaration"):
        sections.append(f"statement:\n  {manifest_row['full_declaration'].strip()[:400]}")

    tactics = row.get("tactics_used") or []
    if tactics:
        listed = "\n".join(f"  {t}" for t in tactics[:_MAX_TACTICS])
        extra = len(tactics) - _MAX_TACTICS
        if extra > 0:
            listed += f"\n  … and {extra} more"
        sections.append(f"verified progress so far ({len(tactics)} step(s)):\n{listed}")

    goals = residual_goals(row)
    if goals:
        walk: list[str] = [f"residual: {len(goals)} open goal(s)"]
        for goal in goals[:_MAX_RESIDUAL_GOALS]:
            walk.extend(_signature_walk(goal))
        if len(goals) > _MAX_RESIDUAL_GOALS:
            walk.append(f"  … and {len(goals) - _MAX_RESIDUAL_GOALS} more")
        sections.append("\n".join(walk))

    lanes = _lane_history(row)
    if lanes:
        sections.append("lane history:\n" + "\n".join(lanes))

    fences = _negative_channel(row)
    if fences:
        sections.append("negative channel (do not retry these families on these shapes):\n" + "\n".join(fences))

    sections.append(handoff_line or _HANDOFF_LINE)
    return "\n\n".join(sections) + "\n"
