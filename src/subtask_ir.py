"""Derived move metadata for controller-facing proof planning.

This layer sits above ActionIR. It does not change tactic lowering or
executor semantics. Instead, it extracts a compact, auditable description of:

  1. Goal shape          -> what kind of local state are we in?
  2. Trigger profile     -> why is this family/operator admissible here?
  3. Subtask             -> what local transformation is this step attempting?

The design is intentionally lossy and controller-oriented. It is meant to
support planning, analysis, dataset construction, and future move-typing work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.tactic_canonicalizer import canonicalize
from src.tactic_ir import ActionIR, Direction, TermKind


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_LOCAL_LINE_RE = re.compile(r"^\s*([^\n:⊢]+?)\s*:\s*.+$")


@dataclass
class GoalShapeIR:
    """Lossy structural summary of the current goal state."""

    goal_count: int
    target: str
    target_head: str
    local_names: list[str] = field(default_factory=list)
    has_forall: bool = False
    has_implication: bool = False
    has_exists: bool = False
    has_equality: bool = False
    has_iff: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_count": self.goal_count,
            "target": self.target,
            "target_head": self.target_head,
            "local_names": self.local_names,
            "has_forall": self.has_forall,
            "has_implication": self.has_implication,
            "has_exists": self.has_exists,
            "has_equality": self.has_equality,
            "has_iff": self.has_iff,
        }


@dataclass
class TriggerFeatureIR:
    """One controller-facing reason a local move is admissible."""

    kind: str
    value: str
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "source": self.source}


@dataclass
class TriggerProfileIR:
    """Compact profile of move triggers for a step."""

    family: str
    primary_premise: str = ""
    features: list[TriggerFeatureIR] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "primary_premise": self.primary_premise,
            "features": [f.to_dict() for f in self.features],
        }


@dataclass
class SubtaskIR:
    """Lossy local objective derived from a proof step."""

    family: str
    kind: str
    summary: str
    primary_premise: str = ""
    local_inputs: list[str] = field(default_factory=list)
    expected_effect: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "kind": self.kind,
            "summary": self.summary,
            "primary_premise": self.primary_premise,
            "local_inputs": self.local_inputs,
            "expected_effect": self.expected_effect,
        }


def derive_goal_shape(goal_state: str) -> GoalShapeIR:
    """Build a coarse structural view of a Lean goal pretty-print."""
    lines = [line.rstrip() for line in (goal_state or "").splitlines()]
    goal_count = max(1, sum("⊢" in line for line in lines)) if goal_state else 0

    target = ""
    locals_block: list[str] = []
    for line in lines:
        if "⊢" in line:
            target = line.split("⊢", 1)[1].strip()
            break
        if line.strip():
            locals_block.append(line)

    local_names: list[str] = []
    for line in locals_block:
        match = _LOCAL_LINE_RE.match(line)
        if not match:
            continue
        prefix = match.group(1).strip()
        for token in prefix.split():
            if token and token not in {"instance", "let"} and token not in local_names:
                local_names.append(token)

    target_head = _target_head_symbol(target)
    return GoalShapeIR(
        goal_count=goal_count,
        target=target,
        target_head=target_head,
        local_names=local_names,
        has_forall="∀" in target,
        has_implication="→" in target,
        has_exists="∃" in target,
        has_equality=_has_equality(target),
        has_iff="↔" in target,
    )


def derive_move_metadata(
    *,
    goal_state: str,
    tactic_text: str,
    family: str,
    canonical_action_ir: str = "",
    annotated_premise: str = "",
    step_index: int = 0,
    prefix_tactics: list[str] | None = None,
) -> tuple[GoalShapeIR, TriggerProfileIR, SubtaskIR]:
    """Derive goal-shape, trigger-profile, and subtask metadata for one step."""
    prefix_tactics = prefix_tactics or []
    goal_shape = derive_goal_shape(goal_state)
    action = _parse_action(canonical_action_ir, tactic_text, family)
    primary_premise = _primary_premise(action, annotated_premise)
    trigger = derive_trigger_profile(
        goal_shape=goal_shape,
        family=family,
        action=action,
        primary_premise=primary_premise,
        step_index=step_index,
        prefix_len=len(prefix_tactics),
    )
    subtask = derive_subtask_ir(
        goal_shape=goal_shape,
        family=family,
        action=action,
        primary_premise=primary_premise,
        step_index=step_index,
    )
    return goal_shape, trigger, subtask


def derive_trigger_profile(
    *,
    goal_shape: GoalShapeIR,
    family: str,
    action: ActionIR | None,
    primary_premise: str,
    step_index: int,
    prefix_len: int,
) -> TriggerProfileIR:
    """Derive controller-facing trigger features from goal shape + action."""
    features: list[TriggerFeatureIR] = []

    if goal_shape.target_head:
        features.append(TriggerFeatureIR("target_head", goal_shape.target_head, "goal"))
    if goal_shape.has_equality:
        features.append(TriggerFeatureIR("target_shape", "equality", "goal"))
    if goal_shape.has_iff:
        features.append(TriggerFeatureIR("target_shape", "iff", "goal"))
    if goal_shape.has_implication:
        features.append(TriggerFeatureIR("binder_surface", "implication", "goal"))
    if goal_shape.has_forall:
        features.append(TriggerFeatureIR("binder_surface", "forall", "goal"))
    if goal_shape.local_names:
        features.append(TriggerFeatureIR("local_context", str(len(goal_shape.local_names)), "goal"))
    if prefix_len > 0 or step_index > 0:
        features.append(TriggerFeatureIR("sequential_context", "true", "proof_trace"))

    if action is None:
        return TriggerProfileIR(family=family, primary_premise=primary_premise, features=features)

    local_names = action.local_var_names()
    if local_names:
        features.append(
            TriggerFeatureIR("local_inputs", ",".join(local_names), "action_ir")
        )

    if family == "rw":
        rewrite_count = len(action.rewrites)
        features.append(TriggerFeatureIR("rewrite_count", str(rewrite_count), "action_ir"))
        direction = _rw_direction_label(action)
        features.append(TriggerFeatureIR("direction_prior", direction, "action_ir"))
        features.append(
            TriggerFeatureIR(
                "argument_mode",
                "applied" if _rw_has_applied_args(action) else "bare",
                "action_ir",
            )
        )
        if rewrite_count > 1:
            features.append(TriggerFeatureIR("composition", "rewrite_chain", "action_ir"))
    elif family in ("apply", "exact", "refine"):
        features.append(TriggerFeatureIR("term_shape", _term_shape(action), "action_ir"))
    elif family in ("simp", "simpa"):
        lemma_count = len(action.simp_lemmas)
        features.append(TriggerFeatureIR("simp_lemma_count", str(lemma_count), "action_ir"))
        if action.using_term is not None:
            features.append(TriggerFeatureIR("using_clause", "true", "action_ir"))

    return TriggerProfileIR(family=family, primary_premise=primary_premise, features=features)


def derive_subtask_ir(
    *,
    goal_shape: GoalShapeIR,
    family: str,
    action: ActionIR | None,
    primary_premise: str,
    step_index: int,
) -> SubtaskIR:
    """Derive a lossy local objective from the family and parsed action."""
    local_inputs = action.local_var_names() if action is not None else []

    if family == "rw":
        rewrite_count = len(action.rewrites) if action is not None else 0
        if rewrite_count > 1:
            kind = "rewrite_chain"
            summary = "execute an ordered local rewrite chain"
            effect = "advance the goal by repeated local normalization"
        elif action is not None and _rw_has_applied_args(action):
            kind = "specialize_rewrite"
            summary = "specialize a rewrite rule with local terms before rewriting"
            effect = "rewrite after instantiating the lemma with local context"
        elif action is not None and action.rewrites and action.rewrites[0].direction == Direction.BACKWARD:
            kind = "normalize_target_backward"
            summary = "rewrite the goal using a backward local equivalence"
            effect = "expose a more library-matchable target form"
        else:
            kind = "normalize_target_forward"
            summary = "rewrite the goal using a forward local equivalence"
            effect = "simplify or normalize the current target/hypotheses"
        if step_index > 0 and kind != "rewrite_chain":
            summary = "continue a local rewrite program from the current intermediate state"
        return SubtaskIR(
            family=family,
            kind=kind,
            summary=summary,
            primary_premise=primary_premise,
            local_inputs=local_inputs,
            expected_effect=effect,
        )

    if family == "apply":
        return SubtaskIR(
            family=family,
            kind="reduce_goal_by_lemma",
            summary="reduce the target to the obligations induced by a library lemma",
            primary_premise=primary_premise,
            local_inputs=local_inputs,
            expected_effect="replace the current target with lemma side goals",
        )

    if family == "exact":
        return SubtaskIR(
            family=family,
            kind="close_with_term",
            summary="close the current goal with an already-available term or theorem",
            primary_premise=primary_premise,
            local_inputs=local_inputs,
            expected_effect="discharge the current goal directly",
        )

    if family == "refine":
        return SubtaskIR(
            family=family,
            kind="construct_goal_skeleton",
            summary="refine the target with a partial proof term and leave residual holes",
            primary_premise=primary_premise,
            local_inputs=local_inputs,
            expected_effect="shape the proof state into smaller residual obligations",
        )

    if family in ("simp", "simpa"):
        return SubtaskIR(
            family=family,
            kind="simplify_with_context",
            summary="simplify the goal using a curated local lemma/context set",
            primary_premise=primary_premise,
            local_inputs=local_inputs,
            expected_effect="normalize the target and/or hypotheses with simp rules",
        )

    return SubtaskIR(
        family=family,
        kind="unmodeled",
        summary="family not yet mapped into motivated move types",
        primary_premise=primary_premise,
        local_inputs=local_inputs,
        expected_effect="unknown",
    )


def _parse_action(canonical_action_ir: str, tactic_text: str, family: str) -> ActionIR | None:
    text = canonical_action_ir.strip() or tactic_text.strip()
    if not text or family == "other":
        return None
    return canonicalize(text, family)


def _primary_premise(action: ActionIR | None, annotated_premise: str) -> str:
    if action is not None:
        name = action.primary_premise_name()
        if name:
            return name
    return annotated_premise.strip()


def _target_head_symbol(target: str) -> str:
    if not target:
        return ""
    text = target
    if text.startswith("∀") or text.startswith("∃"):
        return "binder"
    if "↔" in text:
        return "iff"
    if _has_equality(text):
        return "eq"
    for token in _IDENT_RE.findall(text):
        if token not in {"forall", "exists"}:
            return token
    stripped = text.strip()
    if not stripped:
        return ""
    first = stripped.split()[0]
    return first[:32]


def _has_equality(target: str) -> bool:
    if not target:
        return False
    return "=" in target and "==" not in target


def _rw_direction_label(action: ActionIR) -> str:
    dirs = {atom.direction for atom in action.rewrites}
    if dirs == {Direction.FORWARD}:
        return "forward"
    if dirs == {Direction.BACKWARD}:
        return "backward"
    return "mixed"


def _rw_has_applied_args(action: ActionIR) -> bool:
    for rewrite in action.rewrites:
        if rewrite.expr.kind not in {TermKind.CONST, TermKind.VAR}:
            return True
    return False


def _term_shape(action: ActionIR) -> str:
    if action.term is None:
        return "empty"
    return action.term.kind.value
