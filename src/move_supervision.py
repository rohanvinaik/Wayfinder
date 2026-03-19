"""Auxiliary supervision utilities for controller-facing move metadata.

Builds compact vocabularies from:
- `NavigationalExample.metadata` for step-level controller supervision
- `template_move_profile` summaries for theorem-level template supervision

and converts them into masked training targets for auxiliary supervision heads.

Design note:
- `goal_target_head` and `trigger_signature` are descriptive local-state features.
- `subtask_kind` is a planning/controller label.

The navigator should generally consume only the descriptive heads. Higher-level
slots such as template recognition or planning may additionally consume
`subtask_kind`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

IGNORE_INDEX = -100
ALL_MOVE_HEADS = ("subtask_kind", "goal_target_head", "trigger_signature")


def _top_vocab(counter: Counter[str], min_support: int, max_items: int | None = None) -> list[str]:
    """Return frequent items from a counter, ordered by support."""
    kept = [value for value, count in counter.most_common() if value and count >= min_support]
    if max_items is not None:
        kept = kept[:max_items]
    return kept


def _keep_goal_head(value: str) -> bool:
    """Filter obviously non-semantic head symbols from auxiliary supervision."""
    if not value:
        return False
    if value in {"eq", "iff", "binder", "False", "True"}:
        return True
    if value.isdigit():
        return False
    if len(value) == 1 and value.isalpha():
        return False
    return True


def _normalize_enabled_heads(enabled_heads: tuple[str, ...] | list[str] | None) -> set[str]:
    """Normalize a requested head set to the known head names."""
    if enabled_heads is None:
        return set(ALL_MOVE_HEADS)
    return {head for head in enabled_heads if head in ALL_MOVE_HEADS}


def _template_profile(row: Any) -> dict[str, Any]:
    """Return a normalized template move profile from a JSON row."""
    if isinstance(row, dict):
        return row.get("template_move_profile", {}) or {}
    return {}


def _profile_top_value(profile: dict[str, Any], key: str) -> str:
    """Return the first `value` field from an aggregated profile list."""
    items = profile.get(key, []) or []
    for item in items:
        value = str(item.get("value", "")).strip()
        if value:
            return value
    return ""


def _profile_trigger_values(profile: dict[str, Any]) -> list[str]:
    """Return trigger-signature values from a template move profile."""
    values: list[str] = []
    for item in profile.get("top_trigger_signatures", []) or []:
        value = str(item.get("value", "")).strip()
        if value:
            values.append(value)
    return values


@dataclass
class MoveSupervisionSpec:
    """Compact vocabularies for auxiliary move supervision."""

    subtask_vocab: list[str] = field(default_factory=list)
    goal_head_vocab: list[str] = field(default_factory=list)
    trigger_signature_vocab: list[str] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(self.subtask_vocab or self.goal_head_vocab or self.trigger_signature_vocab)

    def head_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        if self.subtask_vocab:
            sizes["subtask_kind"] = len(self.subtask_vocab)
        if self.goal_head_vocab:
            sizes["goal_target_head"] = len(self.goal_head_vocab)
        if self.trigger_signature_vocab:
            sizes["trigger_signature"] = len(self.trigger_signature_vocab)
        return sizes

    def target_types(self) -> dict[str, str]:
        kinds: dict[str, str] = {}
        if self.subtask_vocab:
            kinds["subtask_kind"] = "multiclass"
        if self.goal_head_vocab:
            kinds["goal_target_head"] = "multiclass"
        if self.trigger_signature_vocab:
            kinds["trigger_signature"] = "multilabel"
        return kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_vocab": self.subtask_vocab,
            "goal_head_vocab": self.goal_head_vocab,
            "trigger_signature_vocab": self.trigger_signature_vocab,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MoveSupervisionSpec:
        return cls(
            subtask_vocab=list(d.get("subtask_vocab", [])),
            goal_head_vocab=list(d.get("goal_head_vocab", [])),
            trigger_signature_vocab=list(d.get("trigger_signature_vocab", [])),
        )


def build_move_supervision_spec(
    examples: list,
    *,
    enabled_heads: tuple[str, ...] | list[str] | None = None,
    subtask_min_support: int = 25,
    goal_head_min_support: int = 100,
    max_goal_heads: int = 64,
    trigger_min_support: int = 250,
    max_trigger_signatures: int = 128,
) -> MoveSupervisionSpec:
    """Build auxiliary supervision vocabularies from nav examples."""
    heads = _normalize_enabled_heads(enabled_heads)
    subtask_counts: Counter[str] = Counter()
    goal_head_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()

    for ex in examples:
        metadata = getattr(ex, "metadata", {}) or {}
        subtask_kind = metadata.get("subtask_kind", "")
        if "subtask_kind" in heads and subtask_kind:
            subtask_counts[subtask_kind] += 1

        goal_head = metadata.get("goal_target_head", "")
        if "goal_target_head" in heads and _keep_goal_head(goal_head):
            goal_head_counts[goal_head] += 1

        if "trigger_signature" in heads:
            for signature in metadata.get("trigger_signature", []):
                if signature:
                    trigger_counts[signature] += 1

    return MoveSupervisionSpec(
        subtask_vocab=_top_vocab(subtask_counts, subtask_min_support)
        if "subtask_kind" in heads
        else [],
        goal_head_vocab=_top_vocab(goal_head_counts, goal_head_min_support, max_goal_heads)
        if "goal_target_head" in heads
        else [],
        trigger_signature_vocab=_top_vocab(
            trigger_counts, trigger_min_support, max_trigger_signatures
        )
        if "trigger_signature" in heads
        else [],
    )


def build_move_targets(
    examples: list,
    spec: MoveSupervisionSpec | None,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """Build masked auxiliary targets from a batch of nav examples."""
    if spec is None or not spec.has_any():
        return {}, {}, {}

    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    target_types = spec.target_types()

    if spec.subtask_vocab:
        lookup = {value: i for i, value in enumerate(spec.subtask_vocab)}
        rows = []
        mask_rows = []
        for ex in examples:
            metadata = getattr(ex, "metadata", {}) or {}
            subtask_kind = metadata.get("subtask_kind", "")
            if subtask_kind in lookup:
                rows.append(lookup[subtask_kind])
                mask_rows.append(True)
            else:
                rows.append(IGNORE_INDEX)
                mask_rows.append(False)
        targets["subtask_kind"] = torch.tensor(rows, dtype=torch.long, device=device)
        masks["subtask_kind"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    if spec.goal_head_vocab:
        lookup = {value: i for i, value in enumerate(spec.goal_head_vocab)}
        rows = []
        mask_rows = []
        for ex in examples:
            metadata = getattr(ex, "metadata", {}) or {}
            goal_head = metadata.get("goal_target_head", "")
            if goal_head in lookup:
                rows.append(lookup[goal_head])
                mask_rows.append(True)
            else:
                rows.append(IGNORE_INDEX)
                mask_rows.append(False)
        targets["goal_target_head"] = torch.tensor(rows, dtype=torch.long, device=device)
        masks["goal_target_head"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    if spec.trigger_signature_vocab:
        lookup = {value: i for i, value in enumerate(spec.trigger_signature_vocab)}
        matrix = torch.zeros(
            (len(examples), len(spec.trigger_signature_vocab)),
            dtype=torch.float32,
            device=device,
        )
        mask_rows = []
        for row_idx, ex in enumerate(examples):
            metadata = getattr(ex, "metadata", {}) or {}
            signatures = metadata.get("trigger_signature", [])
            has_metadata = bool(metadata)
            for signature in signatures:
                col_idx = lookup.get(signature)
                if col_idx is not None:
                    matrix[row_idx, col_idx] = 1.0
            mask_rows.append(has_metadata)
        targets["trigger_signature"] = matrix
        masks["trigger_signature"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    return targets, masks, target_types


def compute_move_metrics(
    move_logits: dict[str, torch.Tensor],
    move_targets: dict[str, torch.Tensor],
    move_masks: dict[str, torch.Tensor],
    move_target_types: dict[str, str],
) -> dict[str, float]:
    """Compute lightweight batch metrics for auxiliary heads."""
    metrics: dict[str, float] = {}
    for name, logits in move_logits.items():
        if name not in move_targets or name not in move_masks:
            continue
        mask = move_masks[name]
        if mask.numel() == 0 or not bool(mask.any().item()):
            continue
        if move_target_types.get(name) == "multiclass":
            preds = logits[mask].argmax(dim=-1)
            targets = move_targets[name][mask]
            metrics[f"{name}_acc"] = float((preds == targets).float().mean().item())
        elif move_target_types.get(name) == "multilabel":
            probs = torch.sigmoid(logits[mask])
            preds = probs > 0.5
            targets = move_targets[name][mask] > 0.5
            metrics[f"{name}_micro_acc"] = float((preds == targets).float().mean().item())
    return metrics


def build_template_move_supervision_spec(
    rows: list[dict[str, Any]],
    *,
    enabled_heads: tuple[str, ...] | list[str] | None = None,
    subtask_min_support: int = 25,
    goal_head_min_support: int = 100,
    max_goal_heads: int = 64,
    trigger_min_support: int = 250,
    max_trigger_signatures: int = 128,
) -> MoveSupervisionSpec:
    """Build auxiliary supervision vocabularies from theorem-level move profiles."""
    heads = _normalize_enabled_heads(enabled_heads)
    subtask_counts: Counter[str] = Counter()
    goal_head_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()

    for row in rows:
        profile = _template_profile(row)
        if not profile:
            continue

        subtask_kind = str(profile.get("dominant_subtask_kind", "")).strip()
        if "subtask_kind" in heads and subtask_kind:
            subtask_counts[subtask_kind] += 1

        goal_head = _profile_top_value(profile, "top_target_heads")
        if "goal_target_head" in heads and _keep_goal_head(goal_head):
            goal_head_counts[goal_head] += 1

        if "trigger_signature" in heads:
            for signature in _profile_trigger_values(profile):
                trigger_counts[signature] += 1

    return MoveSupervisionSpec(
        subtask_vocab=_top_vocab(subtask_counts, subtask_min_support)
        if "subtask_kind" in heads
        else [],
        goal_head_vocab=_top_vocab(goal_head_counts, goal_head_min_support, max_goal_heads)
        if "goal_target_head" in heads
        else [],
        trigger_signature_vocab=_top_vocab(
            trigger_counts, trigger_min_support, max_trigger_signatures
        )
        if "trigger_signature" in heads
        else [],
    )


def build_template_move_targets(
    rows: list[dict[str, Any]],
    spec: MoveSupervisionSpec | None,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """Build masked auxiliary targets from theorem-level move profiles."""
    if spec is None or not spec.has_any():
        return {}, {}, {}

    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    target_types = spec.target_types()

    if spec.subtask_vocab:
        lookup = {value: i for i, value in enumerate(spec.subtask_vocab)}
        rows_out = []
        mask_rows = []
        for row in rows:
            profile = _template_profile(row)
            subtask_kind = str(profile.get("dominant_subtask_kind", "")).strip()
            if subtask_kind in lookup:
                rows_out.append(lookup[subtask_kind])
                mask_rows.append(True)
            else:
                rows_out.append(IGNORE_INDEX)
                mask_rows.append(False)
        targets["subtask_kind"] = torch.tensor(rows_out, dtype=torch.long, device=device)
        masks["subtask_kind"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    if spec.goal_head_vocab:
        lookup = {value: i for i, value in enumerate(spec.goal_head_vocab)}
        rows_out = []
        mask_rows = []
        for row in rows:
            profile = _template_profile(row)
            goal_head = _profile_top_value(profile, "top_target_heads")
            if goal_head in lookup:
                rows_out.append(lookup[goal_head])
                mask_rows.append(True)
            else:
                rows_out.append(IGNORE_INDEX)
                mask_rows.append(False)
        targets["goal_target_head"] = torch.tensor(rows_out, dtype=torch.long, device=device)
        masks["goal_target_head"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    if spec.trigger_signature_vocab:
        lookup = {value: i for i, value in enumerate(spec.trigger_signature_vocab)}
        matrix = torch.zeros(
            (len(rows), len(spec.trigger_signature_vocab)),
            dtype=torch.float32,
            device=device,
        )
        mask_rows = []
        for row_idx, row in enumerate(rows):
            profile = _template_profile(row)
            signatures = _profile_trigger_values(profile)
            has_profile = bool(profile)
            for signature in signatures:
                col_idx = lookup.get(signature)
                if col_idx is not None:
                    matrix[row_idx, col_idx] = 1.0
            mask_rows.append(has_profile)
        targets["trigger_signature"] = matrix
        masks["trigger_signature"] = torch.tensor(mask_rows, dtype=torch.bool, device=device)

    return targets, masks, target_types
