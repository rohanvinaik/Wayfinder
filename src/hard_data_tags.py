from __future__ import annotations

import re
from typing import Any

_METAVAR_RE = re.compile(r"\?[A-Za-z][A-Za-z0-9_.]*")
_REPLAY_PREFIX_RE = re.compile(r"_wayfinder_replay[_A-Za-z0-9]*\.")
_REPLAY_DECL_RE = re.compile(r"_wayfinder_decl_\d+\.")
_META_WRAPPER_RE = re.compile(r"^(?:autoParam|optParam)\b")


def _balanced_prefix(text: str) -> tuple[str | None, str]:
    if not text.startswith("("):
        return None, text
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:idx], text[idx + 1 :]
    return None, text


def sanitize_goal_text(text: str) -> str:
    out = text or ""
    while True:
        cleaned = _REPLAY_PREFIX_RE.sub("", out)
        cleaned = _REPLAY_DECL_RE.sub("", cleaned)
        stripped = cleaned.strip()
        if _META_WRAPPER_RE.match(stripped):
            wrapper_rest = stripped.split(None, 1)[1] if " " in stripped else ""
            inner, _tail = _balanced_prefix(wrapper_rest.lstrip())
            if inner is not None:
                cleaned = inner.strip()
        if cleaned == out:
            return cleaned
        out = cleaned


def canonicalize_theorem_id(theorem_id: str) -> str:
    """Strip known replay noise and collapse duplicated namespace prefixes.

    Some benchmark rows carry malformed theorem identifiers such as
    `Foo.Bar.Foo.Bar.baz`. That is not a mathematical failure mode; it is an
    extraction / symbol-resolution bug. We canonicalize the obvious adjacent
    duplicate-prefix case so future runs can still resolve the symbol.
    """
    out = (theorem_id or "").strip()
    if not out:
        return ""
    out = _REPLAY_PREFIX_RE.sub("", out)
    out = _REPLAY_DECL_RE.sub("", out)
    parts = [part for part in out.split(".") if part]
    changed = True
    while changed and len(parts) >= 2:
        changed = False
        max_width = len(parts) // 2
        for width in range(max_width, 0, -1):
            collapsed = False
            for idx in range(len(parts) - (2 * width) + 1):
                if parts[idx : idx + width] == parts[idx + width : idx + (2 * width)]:
                    parts = parts[: idx + width] + parts[idx + (2 * width) :]
                    changed = True
                    collapsed = True
                    break
            if collapsed:
                break
    return ".".join(parts)


def _goal_target_text(goal: str) -> str:
    text = sanitize_goal_text((goal or "").strip())
    if "⊢" in text:
        return sanitize_goal_text(text.split("⊢", 1)[1].strip())
    return text


def _has_existential_substructure(text: str) -> bool:
    return "∃" in text or "Exists" in text


def _looks_like_witness_construction_goal(text: str) -> bool:
    target = _goal_target_text(text)
    if not target:
        return False
    if target.startswith("∃") or target.startswith("Exists"):
        return True
    if not _has_existential_substructure(target):
        return False
    witness_markers = [
        "sSup {",
        "sInf {",
        "iSup",
        "iInf",
        "range",
        "image",
        "Finset.image",
        "Set.range",
        "card =",
        "{",
    ]
    return any(marker in target for marker in witness_markers)


def _looks_like_side_condition_bundle(goals: list[str] | None) -> bool:
    items = [(_goal_target_text(goal)).strip() for goal in (goals or []) if str(goal).strip()]
    if len(items) < 2:
        return False
    normalized = [item.replace("\n", " ").strip() for item in items]
    repeated = len(set(normalized)) < len(normalized)
    simple_buckets = {"atomic_prop", "equality", "inequality", "membership", "subset", "forall", "iff", "false"}
    buckets = [classify_goal_bucket(item) for item in items]
    avg_token_len = sum(len(item.split()) for item in items) / max(len(items), 1)
    return repeated or (
        avg_token_len <= 12
        and sum(bucket in simple_buckets for bucket in buckets) >= max(len(buckets) - 1, 1)
    )


def _has_metavariable_text(text: str) -> bool:
    return bool(_METAVAR_RE.search(text or ""))


def _looks_like_bare_type_goal(text: str) -> bool:
    target = _goal_target_text(text)
    if not target:
        return False
    if _has_metavariable_text(target):
        return False
    compact = " ".join(target.replace("\n", " ").split())
    if compact in {"P", "Type", "Prop"}:
        return True
    tokens = compact.split()
    if len(tokens) <= 3 and all(token in {"→", "Type", "Prop", "ℝ", "ℕ", "ℤ"} or token[:1].isupper() for token in tokens):
        return True
    return False


def _looks_like_forward_context_goal(text: str) -> bool:
    target = _goal_target_text(text)
    if not target:
        return False
    markers = [
        "Function.Injective",
        "Function.Surjective",
        "Function.Bijective",
        "Ideal.Quotient",
        "IsScalarTower",
        "FormallyUnramified",
        ".comp",
        "Injective",
        "Surjective",
    ]
    if not any(marker in target for marker in markers):
        return False
    return any(
        marker in target
        for marker in ["Function.Injective", "Function.Surjective", "Function.Bijective", "Ideal.Quotient", ".comp"]
    )


def _looks_like_structural_property_goal(text: str) -> bool:
    target = _goal_target_text(text)
    if not target:
        return False
    markers = [
        "IsOpenMap",
        "HasRingHomProperty",
        "FormallyUnramified",
        "CategoryTheory.IsIso",
        "L.essImage",
        "IsIso",
        "OpenEmbedding",
        "AffineSpace",
    ]
    return any(marker in target for marker in markers)


def _looks_like_membership_wall(text: str) -> bool:
    target = _goal_target_text(text)
    if "∈" not in target:
        return False
    markers = [
        ".carrier",
        "carrier ",
        "carrier)",
        "Submodule",
        "Ideal",
        "Subring",
        "PrimeSpectrum",
        "HomogeneousLocalization",
    ]
    return any(marker in target for marker in markers)


def _looks_like_canonical_exists_witness_goal(text: str) -> bool:
    target = _goal_target_text(text)
    if not target.startswith("∃") and not target.startswith("Exists"):
        return False
    witness_markers = [
        "Injective",
        "Splitting",
        "IsZero",
        "0",
        "zero",
        "CategoryTheory",
        "Limits",
        "Nonempty",
    ]
    return any(marker in target for marker in witness_markers)


def trace_pathology_tags(
    step_trace: list[dict[str, Any]] | None,
    *,
    remaining_goals: list[str] | None = None,
) -> list[str]:
    trace = [entry for entry in (step_trace or []) if isinstance(entry, dict)]
    tags: list[str] = []
    if remaining_goals and any(_has_metavariable_text(str(goal)) for goal in remaining_goals):
        tags.append("metavariable_corruption")
    if remaining_goals and any(_looks_like_bare_type_goal(str(goal)) for goal in remaining_goals):
        tags.append("bare_type_side_goal")

    seen_signatures: list[str] = []
    idle_goal = ""
    idle_streak = 0
    max_idle_streak = 0
    for entry in trace:
        tactic = str(entry.get("tactic", "") or "")
        lane = str(entry.get("lane", "") or "")
        progress = bool(entry.get("progress"))
        goal_before = str(entry.get("goal_before", "") or "")
        goals_after = [str(goal) for goal in entry.get("open_goals_after", []) if str(goal).strip()]
        signature = " || ".join(sorted(" ".join(_goal_target_text(goal).split()) for goal in goals_after))
        goal_before_sig = " ".join(_goal_target_text(goal_before).split())
        if progress and signature and signature in seen_signatures[-3:]:
            tags.append("state_loop")
            if tactic.startswith("simp") or tactic.startswith("rw "):
                tags.append("definition_tug_of_war")
        if goals_after and any(_has_metavariable_text(goal) for goal in goals_after):
            tags.append("metavariable_corruption")
            if tactic.startswith("rw [←") or tactic.startswith("rw[←"):
                tags.append("backward_rewrite_metavariable")
        if goals_after and any(_looks_like_bare_type_goal(goal) for goal in goals_after):
            tags.append("bare_type_side_goal")
        if len(goals_after) >= 5:
            tags.append("goal_explosion")
        normalized = [" ".join(_goal_target_text(goal).split()) for goal in goals_after]
        if normalized and len(set(normalized)) < len(normalized):
            tags.append("duplicate_goal_progress")
            if tactic.startswith("simp") or tactic == "norm_num":
                tags.append("duplicate_goal_pseudo_progress")
        if (not progress) and (not lane.strip()) and goal_before_sig:
            if goal_before_sig == idle_goal:
                idle_streak += 1
            else:
                idle_goal = goal_before_sig
                idle_streak = 1
            max_idle_streak = max(max_idle_streak, idle_streak)
        else:
            idle_goal = ""
            idle_streak = 0
        if progress and signature:
            seen_signatures.append(signature)
    if max_idle_streak >= 4:
        tags.append("no_progress_plateau")
        tags.append("blank_lane_plateau")
    return list(dict.fromkeys(tags))


def classify_goal_bucket(goal: str) -> str:
    text = _goal_target_text(goal)
    if not text:
        return "empty"
    if text == "False":
        return "false"
    if text.startswith("∀"):
        return "forall"
    if text.startswith("∃") or text.startswith("Exists"):
        return "exists"
    if "↔" in text:
        return "iff"
    if " ≤ " in text or " ≥ " in text or " < " in text or " > " in text:
        return "inequality"
    if "⊆" in text:
        return "subset"
    if "∈" in text:
        return "membership"
    if "=" in text:
        return "equality"
    if text.startswith("¬") or " ¬" in text:
        return "negation"
    if "\n" in text:
        return "multiline"
    if len(text.split()) <= 3:
        return "atomic_prop"
    return "other"


def goal_bucket_tags(goal: str) -> list[str]:
    text = _goal_target_text(goal)
    tags: list[str] = []
    bucket = classify_goal_bucket(text)
    if bucket != "empty":
        tags.append(f"goal_bucket:{bucket}")
    if "\n" in text:
        tags.append("multiline_goal")
    if text == "False":
        tags.append("contradiction_target")
    if "⋯" in text or "..." in text:
        tags.append("has_ellipsis")
    if len(text) >= 200:
        tags.append("long_goal")
    if _has_existential_substructure(text) and bucket != "exists":
        tags.append("contains_existential_substructure")
    if _looks_like_witness_construction_goal(text):
        tags.append("witness_construction_pressure")
    if any(marker in text for marker in ["sSup", "sInf", "iSup", "iInf"]):
        tags.append("order_extremum_target")
    if any(marker in text for marker in ["range", "image", "Finset.image", "Set.range"]):
        tags.append("candidate_witness_source")
    if any(marker in text for marker in ["‖", "dist", "Metric", "δ", "abs"]):
        tags.append("metric_or_norm_goal")
    if any(marker in text for marker in ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"]):
        tags.append("category_theory_goal")
    if _looks_like_structural_property_goal(text):
        tags.append("structural_property_goal")
    if _looks_like_membership_wall(text):
        tags.append("opaque_membership_wall")
    if _looks_like_canonical_exists_witness_goal(text):
        tags.append("canonical_exists_witness")
    if any(marker in text for marker in ["IsIntegral", "traceMatrix", "Matrix.det", "discr"]):
        tags.append("local_hypothesis_bridge_goal")
    if _has_metavariable_text(text):
        tags.append("metavariable_goal")
    if _looks_like_bare_type_goal(text):
        tags.append("bare_type_side_goal")
    return tags


def attempt_band(attempts: int) -> str:
    if attempts >= 120:
        return "ge_120"
    if attempts >= 80:
        return "80_119"
    if attempts >= 40:
        return "40_79"
    if attempts > 0:
        return "1_39"
    return "0"


def classify_start_failure_family(
    *,
    failure_category: str,
    goal_text: str,
    module: str,
    theorem_line: int,
    context_features: dict[str, Any] | None = None,
    context_unsupported_kinds: list[str] | None = None,
) -> str:
    raw_text = goal_text or ""
    text = sanitize_goal_text(raw_text)
    context_features = context_features or {}
    context_unsupported_kinds = context_unsupported_kinds or []
    if "_wayfinder_replay" in raw_text or "_wayfinder_decl_" in raw_text:
        return "replay_namespace_leakage"
    if "open_scoped" in context_unsupported_kinds and failure_category in {"typeclass_missing", "goal_creation_fail"}:
        return "scoped_context_missing"
    if not module or theorem_line <= 0:
        return "metadata_missing"
    if failure_category == "universe_compilation_fail":
        return "universe_binder_pressure"
    if "⋯" in text or "..." in text:
        return "pretty_print_roundtrip"
    if int(context_features.get("variable", 0) or 0) >= 10 or int(context_features.get("open", 0) or 0) > 0:
        return "context_burden"
    if failure_category == "goal_creation_fail":
        return "goal_creation_generic"
    return "other_start_failure"


def start_failure_tags(
    *,
    failure_category: str,
    goal_text: str,
    module: str,
    theorem_line: int,
    context_features: dict[str, Any] | None = None,
    context_unsupported_kinds: list[str] | None = None,
) -> list[str]:
    raw_text = goal_text or ""
    text = sanitize_goal_text(raw_text)
    context_features = context_features or {}
    context_unsupported_kinds = context_unsupported_kinds or []
    tags: list[str] = [f"failure_category:{failure_category or 'unknown'}"]

    family = classify_start_failure_family(
        failure_category=failure_category,
        goal_text=text,
        module=module,
        theorem_line=theorem_line,
        context_features=context_features,
        context_unsupported_kinds=context_unsupported_kinds,
    )
    tags.append(f"start_failure_family:{family}")
    if not module or theorem_line <= 0:
        tags.append("module_metadata_missing")
    if "_wayfinder_replay" in raw_text or "_wayfinder_decl_" in raw_text:
        tags.append("replay_namespace_leakage")
    if "⋯" in text or "..." in text:
        tags.append("unsafe_pretty_print")
    if "Type" in text or failure_category == "universe_compilation_fail":
        tags.append("universe_pressure")
    if text.count("→") + text.count("->") >= 6:
        tags.append("arrow_heavy")
    if len(text) >= 200:
        tags.append("long_goal_pp")
    if int(context_features.get("variable", 0) or 0) >= 10:
        tags.append("variable_heavy_context")
    if int(context_features.get("open", 0) or 0) > 0:
        tags.append("open_directives_present")
    for kind in context_unsupported_kinds:
        tags.append(f"context_unsupported:{kind}")
    return tags


def classify_reasoning_gap_family(
    *,
    success: bool,
    started: bool,
    residual_bucket: str,
    last_goal_bucket: str,
    goal_text: str = "",
    remaining_goals: list[str] | None = None,
    pathology_tags: list[str] | None = None,
) -> str:
    target = _goal_target_text(goal_text)
    pathology_tags = pathology_tags or []
    if success:
        return "none"
    if not started:
        return "compiler_specialist"
    if any(
        tag in pathology_tags
        for tag in {
            "metavariable_corruption",
            "backward_rewrite_metavariable",
            "goal_explosion",
            "state_loop",
            "definition_tug_of_war",
            "no_progress_plateau",
            "blank_lane_plateau",
        }
    ):
        return "theorem_replanner"
    if residual_bucket == "single_goal_near_miss":
        if _looks_like_witness_construction_goal(target):
            return "witness_construction_close"
        if _looks_like_forward_context_goal(target):
            return "forward_context_close"
        mapping = {
            "false": "contradiction_close",
            "equality": "local_eq_close",
            "inequality": "local_ineq_close",
            "atomic_prop": "atomic_prop_close",
            "iff": "iff_close",
            "exists": "exists_close",
            "forall": "forall_close",
            "membership": "membership_close",
            "subset": "subset_close",
        }
        return mapping.get(last_goal_bucket, "local_goal_close")
    if residual_bucket == "single_goal_stall":
        return "single_goal_stall"
    if residual_bucket in {"multi_goal_small_progress", "multi_goal_small_stall"}:
        if _looks_like_side_condition_bundle(remaining_goals):
            return "small_multigoal_side_conditions"
        return "small_multigoal_planner"
    return "theorem_replanner"
