from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.benchmark_residuals import detect_self_application
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text
from src.nav_contracts import StructuredQuery
from src.proof_network import navigate

_SYMBOL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']*(?:\.[A-Za-z][A-Za-z0-9_']*)*")
_STOP_TOKENS = {
    "Type",
    "Sort",
    "Prop",
    "fun",
    "let",
    "by",
    "have",
    "show",
    "match",
    "if",
    "then",
    "else",
    "forall",
    "exists",
    "True",
    "False",
    "Nat",
    "Int",
    "Rat",
    "Real",
}

_FEATURE_LIBRARY: dict[str, list[str]] = {
    "local_eq_close": [
        "rewrite_chain",
        "extensionality",
        "normalization",
        "transport",
        "symmetric_unfolding",
        "hypothesis_injection",
        "close_before_unpack",
    ],
    "local_goal_close": [
        "local_lemma_selection",
        "targeted_rewrite",
        "terminal_closer",
    ],
    "forward_context_close": [
        "forward_context_chase",
        "local_hypothesis_unfold",
        "diagram_transport",
        "inheritance_reasoning",
    ],
    "contradiction_close": [
        "contradiction_search",
        "negation_push",
        "domain_discharge",
    ],
    "forall_close": [
        "binder_introduction",
        "pointwise_reduction",
        "extensionality",
    ],
    "exists_close": [
        "witness_construction",
        "constructor_apply",
        "residual_refine",
        "canonical_witness_selection",
        "existential_packaging",
    ],
    "membership_close": [
        "membership_rewrite",
        "subset_transport",
        "set_extensionality",
        "carrier_structure_exposure",
        "closure_lemma_retrieval",
        "smul_mul_operator_preservation",
    ],
    "subset_close": [
        "subset_intro",
        "membership_rewrite",
        "pointwise_transport",
    ],
    "local_ineq_close": [
        "inequality_solver_chain",
        "triangle_bound_search",
        "positivity_reasoning",
        "arith_normalization",
        "monotonicity_chain",
        "order_transport",
    ],
    "witness_construction_close": [
        "witness_construction",
        "range_image_realization",
        "set_builder_membership",
        "supremum_lower_bound",
    ],
    "iff_close": [
        "iff_intro",
        "bidirectional_rewrite",
        "logical_transport",
    ],
    "atomic_prop_close": [
        "local_fact_selection",
        "domain_lemma_pick",
        "terminal_exact",
    ],
    "small_multigoal_planner": [
        "subgoal_ordering",
        "shared_hypothesis_reuse",
        "bridge_lemma_synthesis",
        "case_coordination",
    ],
    "small_multigoal_side_conditions": [
        "subgoal_isolation",
        "side_condition_sweep",
        "micro_budget_partition",
        "cheap_goal_sweeps",
    ],
    "theorem_replanner": [
        "theorem_replan",
        "branch_pruning",
        "loop_escape",
        "metavariable_repair",
        "macro_strategy_shift",
        "intermediate_claim_synthesis",
    ],
}

_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "local_eq_close": ["eq", "congr", "ext", "cast", "simp", "rw", "det", "discr", "trace", "integral"],
    "local_goal_close": ["exact", "apply", "simp", "intro"],
    "forward_context_close": ["injective", "surjective", "comp", "tower", "quotient"],
    "contradiction_close": ["false", "not", "ne", "contra", "disjoint", "empty"],
    "forall_close": ["forall", "all", "ext", "funext", "intro"],
    "exists_close": ["exists", "choose", "existsi", "construct", "zero", "injective", "splitting"],
    "membership_close": ["mem", "subset", "union", "inter", "image", "preimage", "carrier", "submodule", "ideal"],
    "subset_close": ["subset", "mem", "inter", "union"],
    "local_ineq_close": ["lt", "le", "gt", "ge", "mono", "bound"],
    "witness_construction_close": ["exists", "choose", "range", "image", "sSup", "witness"],
    "iff_close": ["iff", "not", "imp", "mp", "mpr"],
    "atomic_prop_close": ["is", "has", "nonempty", "trivial"],
    "small_multigoal_planner": ["cases", "induction", "sum", "prod", "map", "comp"],
    "small_multigoal_side_conditions": ["antisymm", "infer", "instance", "iff", "isIso", "essImage"],
    "theorem_replanner": ["induction", "cases", "rec", "exists", "forall"],
}


def goal_shape_features(goal_text: str) -> dict[str, int]:
    text = sanitize_goal_text(goal_text or "")
    return {
        "char_len": len(text),
        "token_len": len(text.split()),
        "binder_count": text.count("∀") + text.count("∃") + text.count("fun "),
        "forall_count": text.count("∀"),
        "exists_count": text.count("∃"),
        "arrow_count": text.count("→") + text.count("->"),
        "iff_count": text.count("↔"),
        "eq_count": text.count("="),
        "neq_count": text.count("≠"),
        "and_count": text.count("∧"),
        "or_count": text.count("∨"),
        "not_count": text.count("¬"),
        "typeclass_count": text.count("[") + text.count("]"),
        "type_count": text.count("Type") + text.count("Sort"),
        "membership_count": text.count("∈"),
        "subset_count": text.count("⊆"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def extract_goal_symbols(text: str, limit: int = 16) -> list[str]:
    counts: Counter[str] = Counter()
    for token in _SYMBOL_RE.findall(text or ""):
        if token in _STOP_TOKENS:
            continue
        if len(token) <= 1:
            continue
        counts[token] += 1
        if "." in token:
            for piece in token.split("."):
                if len(piece) > 1 and piece not in _STOP_TOKENS:
                    counts[piece] += 1
    ranked = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
    return ranked[:limit]


def namespace_chain(theorem_id: str) -> list[str]:
    pieces = [piece for piece in theorem_id.split(".") if piece]
    out: list[str] = []
    for idx in range(1, len(pieces)):
        out.append(".".join(pieces[:idx]))
    return out


def closing_features(reasoning_gap_family: str, goal_text: str) -> list[str]:
    features = list(_FEATURE_LIBRARY.get(reasoning_gap_family, ["hard_residual_search"]))
    goal = sanitize_goal_text(goal_text or "")
    if "⋯" in goal or "..." in goal:
        features.append("pretty_print_safe_roundtrip")
    if "MeasureTheory" in goal or "volume" in goal or "lintegral" in goal:
        features.append("measure_rewrite")
    if "Set." in goal or " ∈ " in goal or "⊆" in goal:
        features.append("set_reasoning")
    if "toIcoMod" in goal or "AddCircle" in goal:
        features.append("normal_form_transport")
    if any(marker in goal for marker in ["sSup", "sInf", "iSup", "iInf"]) and "∃" in goal:
        features.append("witness_construction")
    if any(marker in goal for marker in ["‖", "dist", "Metric", "δ"]) and any(op in goal for op in ["≤", "≥", "<", ">"]):
        features.append("triangle_bound_search")
    if any(marker in goal for marker in ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"]):
        features.extend(["definition_unfolding", "domain_aware_lane_suppression"])
    if any(
        marker in goal
        for marker in [
            "IsIso",
            "IsOpenMap",
            "IsClosedMap",
            "Injective",
            "Surjective",
            "Bijective",
            "IsLocally",
            "Formally",
            "essImage",
            "quasiIso",
            "IsIdempotentComplete",
        ]
    ):
        features.extend(["structural_property_exact", "abstraction_preservation"])
    if any(marker in goal for marker in ["CategoryTheory.CategoryStruct.comp", "≫", "whisker", "associator", "leftUnitor"]):
        features.extend(["categorical_comp_normalization", "associativity_transport"])
    if any(marker in goal for marker in ["=ᶠ[", "Filter.atTop", "Tendsto", "IsEquivalent", "isLittleO", "isBigO"]):
        features.extend(["eventual_filter_normalization", "eventual_witness_construction"])
    if goal.startswith("∃") and any(marker in goal for marker in ["Injective", "Splitting", "IsZero", "CategoryTheory", "Limits"]):
        features.extend(["canonical_witness_selection", "zero_object_instantiation", "existential_packaging"])
    return list(dict.fromkeys(features))


def tactic_prefixes(tactics: list[Any]) -> list[str]:
    out: list[str] = []
    for tactic in tactics:
        if isinstance(tactic, str) and tactic.strip():
            out.append(tactic.strip().split()[0])
    return list(dict.fromkeys(out))


def build_dependency_profile(remaining_goals: list[str]) -> dict[str, Any]:
    goals = [goal for goal in remaining_goals if isinstance(goal, str) and goal.strip()]
    symbol_sets = [set(extract_goal_symbols(goal, limit=10)) for goal in goals]
    symbol_counts: Counter[str] = Counter()
    for symbols in symbol_sets:
        symbol_counts.update(symbols)
    shared_symbols = [symbol for symbol, count in symbol_counts.items() if count >= 2]
    normalized_goals = [" ".join(goal.split()) for goal in goals]
    repeated_goal_count = len(normalized_goals) - len(set(normalized_goals))
    avg_goal_token_len = round(
        sum(len(goal.split()) for goal in goals) / max(len(goals), 1),
        2,
    )
    simple_goal_buckets = {"atomic_prop", "equality", "inequality", "membership", "subset", "forall", "iff", "false"}
    goal_buckets = [classify_goal_bucket(goal) for goal in goals]
    side_condition_profile = (
        "repeated_small_goals"
        if goals and (
            repeated_goal_count > 0
            or (
                avg_goal_token_len <= 12
                and sum(bucket in simple_goal_buckets for bucket in goal_buckets) >= max(len(goal_buckets) - 1, 1)
            )
        )
        else "general_multigoal"
    )
    return {
        "remaining_goal_count": len(goals),
        "goal_buckets": goal_buckets,
        "goal_lengths": [len(goal.split()) for goal in goals[:8]],
        "goal_symbol_sets": [sorted(symbols)[:10] for symbols in symbol_sets[:8]],
        "shared_symbols": shared_symbols[:12],
        "shared_symbol_count": len(shared_symbols),
        "repeated_goal_count": repeated_goal_count,
        "avg_goal_token_len": avg_goal_token_len,
        "side_condition_profile": side_condition_profile,
        "coordination_class": (
            "independent"
            if len(goals) <= 1 or len(shared_symbols) == 0
            else "lightly_coupled"
            if len(shared_symbols) <= 3
            else "tightly_coupled"
        ),
        "dependency_density": round(
            len(shared_symbols) / max(sum(len(symbols) for symbols in symbol_sets), 1),
            4,
        ),
    }


def _domain_hints(theorem_id: str, text: str) -> list[str]:
    hints: list[str] = []
    source = " ".join([sanitize_goal_text(theorem_id), sanitize_goal_text(text)])
    checks = [
        ("measure", ["MeasureTheory", "lintegral", "volume", "ae", "AEMeasurable"]),
        ("set", ["Set.", "∈", "⊆", "image", "preimage", "union", "inter"]),
        ("algebraic_geometry", ["AlgebraicGeometry", "Scheme", "LocallyRingedSpace"]),
        (
            "abstract_algebra",
            [
                "IsIntegral",
                "traceMatrix",
                "Matrix.det",
                "discr",
                "Submodule",
                "Ideal",
                "PrimeSpectrum",
                "HomogeneousLocalization",
                "FormallyUnramified",
                "HasRingHomProperty",
            ],
        ),
        ("topology", ["Topological", "Continuous", "closure", "Open", "Closed"]),
        ("category_theory", ["CategoryTheory", "Functor", "Adjunction", "NatTrans", "IsIso", "essImage"]),
        ("eventual_filter", ["=ᶠ[", "Filter.atTop", "Tendsto", "IsEquivalent", "isLittleO", "isBigO"]),
        ("cardinal", ["Cardinal", "#", "countable", "mk_", "aleph"]),
        ("geometric_analysis", ["Besicovitch", "dist", "Metric", "δ", "norm", "ball"]),
        ("arithmetic_function", ["ArithmeticFunction", "moebius", "vonMangoldt", "divisors"]),
        ("group_morphism", ["MonoidHom", "RingHom", "LinearMap", "AddMonoid", "comp"]),
        ("order", ["≤", "≥", "<", ">", "Sup", "Inf", "Order"]),
        ("circle_transport", ["AddCircle", "toIcoMod"]),
    ]
    for label, markers in checks:
        if any(marker in source for marker in markers):
            hints.append(label)
    return hints


def infer_representation_pressures(
    goal_text: str,
    *,
    theorem_id: str = "",
    last_goal_bucket: str,
    remaining_goal_count: int = 1,
    remaining_goals: list[str] | None = None,
    search_pathology_tags: list[str] | None = None,
) -> list[str]:
    text = sanitize_goal_text(goal_text or "")
    state_text = sanitize_goal_text("\n".join([text] + [str(goal) for goal in (remaining_goals or [])]))
    pressures: list[str] = []
    domain_hints = _domain_hints(theorem_id, text)
    search_pathology_tags = search_pathology_tags or []
    if last_goal_bucket == "equality":
        pressures.append("canonical_normalization")
    if last_goal_bucket == "inequality":
        pressures.append("inequality_bounding")
    if last_goal_bucket in {"membership", "subset"} or "Set." in text:
        pressures.append("pointwise_set_reduction")
    if last_goal_bucket == "membership" and any(
        marker in text
        for marker in ["carrier", "Submodule", "Ideal", "Subring", "PrimeSpectrum", "HomogeneousLocalization"]
    ):
        pressures.append("opaque_membership_unfolding")
    if last_goal_bucket == "forall" or text.count("∀") > 0:
        pressures.append("binder_introduction")
    if last_goal_bucket == "exists" or text.count("∃") > 0:
        pressures.append("witness_exposure")
    if (last_goal_bucket == "exists" or text.startswith("∃")) and any(
        marker in text for marker in ["Injective", "Splitting", "IsZero", "zero", "CategoryTheory", "Limits"]
    ):
        pressures.append("canonical_object_witness")
    if any(marker in text for marker in ["sSup", "sInf", "iSup", "iInf"]) and "∃" in text:
        pressures.append("witness_construction")
    if last_goal_bucket == "false" or "¬" in text:
        pressures.append("contradiction_reduction")
    if any(marker in text for marker in ["↑", "↥", "cast", "Eq.mp", "Eq.mpr", "Subtype", "toIcoMod"]):
        pressures.append("transport_alignment")
    if any(marker in text for marker in ["MeasureTheory", "volume", "lintegral", "ae"]):
        pressures.append("measure_normalization")
    if last_goal_bucket == "inequality" and any(marker in text for marker in ["‖", "dist", "Metric", "abs", "δ"]):
        pressures.append("triangle_bound_search")
    if any(marker in text for marker in ["fun ", "⇑", "Function.", "LinearMap", "MonoidHom", ".comp"]):
        pressures.append("extensionality_reduction")
    if any(marker in text for marker in ["Function.Injective", "Function.Surjective", "Function.Bijective", "Ideal.Quotient", "IsScalarTower", ".comp"]):
        pressures.append("forward_context_chase")
    if any(marker in state_text for marker in ["IsOpenMap", "HasRingHomProperty", "FormallyUnramified", "IsIso", "essImage", "Matrix.det", "traceMatrix", "discr"]):
        pressures.append("structural_theorem_retrieval")
    if any(
        marker in text
        for marker in [
            "IsIso",
            "IsOpenMap",
            "IsClosedMap",
            "Injective",
            "Surjective",
            "Bijective",
            "IsLocally",
            "Formally",
            "essImage",
            "quasiIso",
            "IsIdempotentComplete",
        ]
    ):
        pressures.append("structural_property_closure")
    if any(marker in text for marker in ["IsIntegral", "traceMatrix", "Matrix.det", "discr"]):
        pressures.append("hypothesis_injection")
    if last_goal_bucket == "equality" and any(
        marker in text for marker in ["Matrix.det", "discr", "traceMatrix", "IsOpenMap", "HasRingHomProperty", "FormallyUnramified"]
    ):
        pressures.append("symmetric_unfolding")
    if "category_theory" in domain_hints and any(
        marker in state_text for marker in ["CategoryTheory.CategoryStruct.comp", "≫", "whisker", "associator", "leftUnitor"]
    ):
        pressures.append("categorical_composition_normalization")
    if any(marker in state_text for marker in ["=ᶠ[", "Filter.atTop", "Tendsto", "IsEquivalent", "isLittleO", "isBigO"]):
        pressures.append("eventual_filter_reasoning")
    if remaining_goal_count > 1:
        pressures.append("subgoal_coordination")
    if remaining_goal_count > 1 and build_dependency_profile(list(remaining_goals or [])).get("side_condition_profile") == "repeated_small_goals":
        pressures.append("side_condition_sweep")
    if "⋯" in text or "..." in text:
        pressures.append("pretty_print_reconstruction")
    if "category_theory" in domain_hints:
        pressures.extend(["definition_unfolding", "domain_solver_mismatch_risk"])
        if last_goal_bucket == "exists":
            pressures.append("zero_object_instantiation")
    if "algebraic_geometry" in domain_hints or "abstract_algebra" in domain_hints:
        pressures.extend(["close_before_unpack", "domain_solver_mismatch_risk"])
    if "cardinal" in domain_hints:
        pressures.append("antisymmetry_decomposition")
    if "geometric_analysis" in domain_hints and last_goal_bucket == "inequality":
        pressures.append("metric_bound_transport")
    if "metavariable_corruption" in search_pathology_tags:
        pressures.append("metavariable_repair")
    if "state_loop" in search_pathology_tags or "definition_tug_of_war" in search_pathology_tags:
        pressures.append("loop_escape")
    if "goal_explosion" in search_pathology_tags:
        pressures.append("branch_pruning")
    if any(tag in search_pathology_tags for tag in ["no_progress_plateau", "blank_lane_plateau"]):
        pressures.extend(["plateau_escape", "negative_memory_avoidance"])
    return list(dict.fromkeys(pressures))


def tactic_family_profile(tactics: list[str]) -> dict[str, int]:
    profile = Counter()
    for tactic in tactics:
        if tactic in {"rw", "simp", "simpa", "nth_rewrite"} or tactic.startswith("norm"):
            profile["rewrite"] += 1
        elif tactic in {"aesop", "grind", "omega", "linarith", "decide", "tauto"}:
            profile["automation"] += 1
        elif tactic in {"exact", "refine", "apply", "constructor", "convert"}:
            profile["closer"] += 1
        elif tactic in {"intro", "rintro", "ext", "funext"}:
            profile["binder"] += 1
        elif tactic in {"cases", "induction", "rcases"}:
            profile["decompose"] += 1
        else:
            profile["other"] += 1
    return dict(profile)


def build_residual_skeleton_geometry(row: dict[str, Any], remaining_goals: list[str]) -> dict[str, Any]:
    theorem_id = sanitize_goal_text(str(row.get("theorem_id", "")))
    goal_text = sanitize_goal_text(str(row.get("last_goal") or row.get("initial_goal") or ""))
    theorem_text = "\n".join(
        [
            sanitize_goal_text(str(row.get("theorem_statement", ""))),
            sanitize_goal_text(str(row.get("initial_goal", ""))),
            goal_text,
        ]
    )
    last_goal_bucket = str(row.get("last_goal_bucket") or classify_goal_bucket(goal_text))
    shape = goal_shape_features(goal_text)
    theorem_shape = goal_shape_features(theorem_text)
    symbol_list = extract_goal_symbols(theorem_text, limit=20)
    search_pathology_tags = list(row.get("search_pathology_tags", []) or [])
    return {
        "goal_bucket": last_goal_bucket,
        "goal_shape_features": shape,
        "theorem_shape_features": theorem_shape,
        "top_symbols": symbol_list,
        "symbol_count": len(symbol_list),
        "domain_hints": _domain_hints(theorem_id, theorem_text),
        "representation_pressures": infer_representation_pressures(
            goal_text,
            theorem_id=theorem_id,
            last_goal_bucket=last_goal_bucket,
            remaining_goal_count=len(remaining_goals),
            remaining_goals=remaining_goals,
            search_pathology_tags=search_pathology_tags,
        ),
        "search_pathology_tags": search_pathology_tags,
        "markers": {
            "has_ellipsis": int("⋯" in theorem_text or "..." in theorem_text),
            "has_cast_like": int(any(marker in theorem_text for marker in ["↑", "↥", "cast", "Subtype"])),
            "has_function_application": int(any(marker in theorem_text for marker in ["⇑", "fun ", "Function.", ".comp"])),
            "has_measure_terms": int(any(marker in theorem_text for marker in ["MeasureTheory", "volume", "lintegral", "ae"])),
            "has_set_terms": int(any(marker in theorem_text for marker in ["Set.", "∈", "⊆", "image", "preimage"])),
            "has_namespace_chain": int("." in theorem_id),
        },
    }


def build_search_control_geometry(row: dict[str, Any]) -> dict[str, Any]:
    trace = [entry for entry in (row.get("step_trace", []) or []) if isinstance(entry, dict)]
    blank_lane_streak = 0
    max_blank_lane_streak = 0
    identical_no_progress_streak = 0
    max_identical_no_progress_streak = 0
    trailing_blank_lane_streak = 0
    idle_goal = ""
    forward_rw_count = 0
    backward_rw_count = 0
    simp_count = 0
    no_progress_steps = 0
    trailing_goal = ""

    for entry in trace:
        tactic = str(entry.get("tactic", "") or "")
        lane = str(entry.get("lane", "") or "")
        progress = bool(entry.get("progress"))
        goal_before = " ".join(
            sanitize_goal_text(str(entry.get("goal_before", "") or "")).replace("\n", " ").split()
        )
        if tactic.startswith("rw [←"):
            backward_rw_count += 1
        elif tactic.startswith("rw ["):
            forward_rw_count += 1
        if tactic.startswith("simp"):
            simp_count += 1
        if not progress:
            no_progress_steps += 1
        if (not progress) and (not lane.strip()) and goal_before:
            if goal_before == idle_goal:
                blank_lane_streak += 1
                identical_no_progress_streak += 1
            else:
                idle_goal = goal_before
                blank_lane_streak = 1
                identical_no_progress_streak = 1
            trailing_blank_lane_streak = blank_lane_streak
            trailing_goal = goal_before
        else:
            idle_goal = ""
            blank_lane_streak = 0
            identical_no_progress_streak = 0
            trailing_blank_lane_streak = 0
            trailing_goal = ""
        max_blank_lane_streak = max(max_blank_lane_streak, blank_lane_streak)
        max_identical_no_progress_streak = max(max_identical_no_progress_streak, identical_no_progress_streak)

    return {
        "step_count": len(trace),
        "no_progress_steps": no_progress_steps,
        "no_progress_ratio": round(no_progress_steps / max(len(trace), 1), 4),
        "max_blank_lane_streak": max_blank_lane_streak,
        "max_identical_no_progress_streak": max_identical_no_progress_streak,
        "trailing_blank_lane_streak": trailing_blank_lane_streak,
        "plateau_detected": int(max_blank_lane_streak >= 4),
        "plateau_goal_signature": trailing_goal,
        "forward_rw_count": forward_rw_count,
        "backward_rw_count": backward_rw_count,
        "simp_count": simp_count,
        "bidirectional_rw_cycle": int(forward_rw_count > 0 and backward_rw_count > 0),
    }


def _infer_startability_actions(row: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    start_family = str(row.get("start_failure_family", "") or "")
    context_features = row.get("context_features", {})
    if not isinstance(context_features, dict):
        context_features = {}
    unsupported = {
        str(kind).strip()
        for kind in (row.get("context_unsupported_kinds", []) or [])
        if str(kind).strip()
    }

    if (
        str(row.get("module", "")).strip()
        or str(row.get("lean_path", "")).strip()
        or str(row.get("file_path", "")).strip()
        or int(row.get("theorem_line", 0) or 0) > 0
    ):
        actions.append("theorem_site_lookup")
    if start_family in {"metadata_missing", "replay_namespace_leakage"}:
        actions.extend(["symbol_name_canonicalization", "prefer_file_context_replay"])
    if int(context_features.get("variable", 0) or 0) > 0 or "variable" in unsupported:
        actions.append("replay_variable_block")
    if int(context_features.get("open", 0) or 0) > 0 or "open" in unsupported:
        actions.append("replay_open_namespaces")
    if int(context_features.get("open_scoped", 0) or 0) > 0 or "open_scoped" in unsupported:
        actions.extend(["replay_open_scopes", "prefer_file_context_replay"])
    if int(context_features.get("include", 0) or 0) > 0 or "include" in unsupported or "omit" in unsupported:
        actions.append("replay_include_omit")
    if (
        int(context_features.get("local_notation", 0) or 0) > 0
        or "local_notation" in unsupported
        or "notation" in unsupported
    ):
        actions.append("replay_local_notation")
    if int(context_features.get("local_attribute", 0) or 0) > 0 or "local_attribute" in unsupported:
        actions.append("replay_local_attributes")
    if start_family == "scoped_context_missing":
        actions.append("prefer_file_context_replay")
    return list(dict.fromkeys(actions))


def build_proof_plan_geometry(
    row: dict[str, Any],
    skeleton: dict[str, Any],
    decomposition_profile: dict[str, Any],
    closing_feature_list: list[str],
    search_control_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    search_control_geometry = search_control_geometry or {}
    tactics = tactic_prefixes(row.get("tactics_used", []))
    lane_sequence = [
        lane.strip()
        for lane in str(row.get("lane_sequence", "")).split("→")
        if lane.strip()
    ]
    reasoning_gap_family = str(row.get("reasoning_gap_family", ""))
    if reasoning_gap_family in {
        "local_eq_close",
        "local_goal_close",
        "forward_context_close",
        "contradiction_close",
        "forall_close",
        "exists_close",
        "membership_close",
        "local_ineq_close",
        "witness_construction_close",
        "atomic_prop_close",
    }:
        plan_level = "local_close"
    elif reasoning_gap_family in {"small_multigoal_planner", "small_multigoal_side_conditions"}:
        plan_level = "multigoal_coordination"
    else:
        plan_level = "replan"
    candidate_methods = list(
        dict.fromkeys(closing_feature_list + skeleton.get("representation_pressures", []))
    )
    if decomposition_profile.get("side_condition_profile") == "repeated_small_goals":
        candidate_methods.extend(["subgoal_isolation", "side_condition_sweep", "micro_budget_partition"])
    if "category_theory" in skeleton.get("domain_hints", []):
        candidate_methods.append("domain_aware_lane_suppression")
    if reasoning_gap_family == "witness_construction_close":
        candidate_methods.extend(["context_witness_mining", "range_image_realization"])
    if reasoning_gap_family == "exists_close":
        candidate_methods.extend(["canonical_witness_guess", "existential_packaging", "context_witness_mining"])
        if "category_theory" in skeleton.get("domain_hints", []):
            candidate_methods.extend(["zero_object_instantiation", "zero_morphism_packaging"])
    if reasoning_gap_family == "local_ineq_close":
        candidate_methods.extend(["inequality_specialist", "bound_transport"])
    if reasoning_gap_family == "local_eq_close":
        candidate_methods.extend(["symmetric_unfolding", "hypothesis_injection"])
    if reasoning_gap_family == "forward_context_close":
        candidate_methods.extend(["forward_context_chase", "hypothesis_unfold", "local_diagram_chase"])
    if reasoning_gap_family == "membership_close":
        candidate_methods.extend(["carrier_structure_exposure", "closure_lemma_retrieval", "subobject_unfolding"])
    if "structural_theorem_retrieval" in skeleton.get("representation_pressures", []):
        candidate_methods.extend(["close_before_unpack", "structural_exact_first"])
    if "metavariable_corruption" in skeleton.get("search_pathology_tags", []):
        candidate_methods.extend(["metavariable_repair", "branch_reset"])
    if "state_loop" in skeleton.get("search_pathology_tags", []):
        candidate_methods.extend(["loop_escape", "fold_unfold_suppression"])
    if "structural_property_closure" in skeleton.get("representation_pressures", []):
        candidate_methods.extend(["structural_property_exact", "close_before_unpack"])
    if "categorical_composition_normalization" in skeleton.get("representation_pressures", []):
        candidate_methods.extend(["categorical_comp_normalization", "associativity_transport"])
    if "eventual_filter_reasoning" in skeleton.get("representation_pressures", []):
        candidate_methods.extend(["eventual_filter_normalization", "eventual_witness_construction"])
    if bool(search_control_geometry.get("plateau_detected")):
        candidate_methods.extend(["negative_kline_retrieval", "plateau_bailout", "branch_blacklist_update"])
    if bool(search_control_geometry.get("bidirectional_rw_cycle")):
        candidate_methods.extend(["loop_escape", "branch_blacklist_update"])
    specialist_targets: list[str] = []
    lane_suppression_hints: list[str] = []
    if reasoning_gap_family == "local_ineq_close":
        specialist_targets.append("inequality_specialist")
    if reasoning_gap_family == "witness_construction_close":
        specialist_targets.append("witness_instantiation_specialist")
    if reasoning_gap_family == "exists_close":
        specialist_targets.append("witness_instantiation_specialist")
    if reasoning_gap_family == "small_multigoal_side_conditions":
        specialist_targets.append("side_condition_sweeper")
    if reasoning_gap_family == "small_multigoal_planner":
        specialist_targets.append("multigoal_planner")
    if reasoning_gap_family == "forward_context_close":
        specialist_targets.append("forward_reasoner")
    if reasoning_gap_family == "membership_close":
        specialist_targets.append("membership_specialist")
    if reasoning_gap_family == "theorem_replanner":
        specialist_targets.append("theorem_replanner")
    if "structural_property_closure" in skeleton.get("representation_pressures", []):
        specialist_targets.append("structural_property_specialist")
    if "categorical_composition_normalization" in skeleton.get("representation_pressures", []):
        specialist_targets.append("composition_normalizer")
    if "eventual_filter_reasoning" in skeleton.get("representation_pressures", []):
        specialist_targets.append("eventual_filter_specialist")
    if bool(search_control_geometry.get("plateau_detected")):
        specialist_targets.append("plateau_escape_replanner")
    if any(hint in skeleton.get("domain_hints", []) for hint in {"category_theory", "algebraic_geometry", "abstract_algebra"}):
        lane_suppression_hints.append("suppress_numeric_solver_lane")
    if "structural_theorem_retrieval" in skeleton.get("representation_pressures", []):
        lane_suppression_hints.append("prefer_structural_exact_before_unfold")
    if "backward_rewrite_metavariable" in skeleton.get("search_pathology_tags", []):
        lane_suppression_hints.append("suppress_backward_rewrite_expansion")
    if any(
        tag in skeleton.get("search_pathology_tags", [])
        for tag in {"definition_tug_of_war", "state_loop"}
    ):
        lane_suppression_hints.append("suppress_fold_unfold_ping_pong")
    if "cardinal" in skeleton.get("domain_hints", []):
        lane_suppression_hints.append("favor_side_condition_sweeps")
    if reasoning_gap_family == "membership_close":
        lane_suppression_hints.append("preserve_smul_membership_operators")
    if reasoning_gap_family == "exists_close" and "canonical_object_witness" in skeleton.get("representation_pressures", []):
        lane_suppression_hints.append("favor_canonical_witnesses_before_search")
    if bool(search_control_geometry.get("plateau_detected")):
        lane_suppression_hints.append("bail_out_identical_blank_lane_plateaus")
    if bool(search_control_geometry.get("bidirectional_rw_cycle")):
        lane_suppression_hints.append("avoid_bidirectional_rw_cycles")
    return {
        "plan_level": plan_level,
        "resolution_family": reasoning_gap_family,
        "candidate_methods": list(dict.fromkeys(candidate_methods)),
        "lane_history": lane_sequence,
        "lane_count": len(lane_sequence),
        "tactic_prefixes": tactics,
        "tactic_family_profile": tactic_family_profile(tactics),
        "attempt_band": str(row.get("attempt_band", "")),
        "search_control_geometry": search_control_geometry,
        "specialist_targets": specialist_targets,
        "lane_suppression_hints": lane_suppression_hints,
        "bridge_pressure": int(
            "bridge_lemma_synthesis" in closing_feature_list
            or decomposition_profile.get("shared_symbol_count", 0) > 0
        ),
        "representation_change_pressure": int(
            len(skeleton.get("representation_pressures", [])) > 0
        ),
    }


class _ProofGraphIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._entity_id_cache: dict[str, int | None] = {}
        self._entity_meta_cache: dict[int, dict[str, Any]] = {}
        self._anchor_map: dict[str, list[int]] | None = None
        self._entity_anchor_labels: dict[int, list[str]] = {}
        self._entity_anchor_ids: dict[int, list[int]] = {}
        self._position_cache: dict[int, dict[str, tuple[int, int]]] = {}
        self._relation_cache: dict[tuple[int, int], list[str]] = {}
        self._accessible_count_cache: dict[int, int] = {}

    def entity_id(self, name: str) -> int | None:
        if name not in self._entity_id_cache:
            row = self.conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
            self._entity_id_cache[name] = int(row[0]) if row else None
        return self._entity_id_cache[name]

    def entity_meta(self, entity_id: int) -> dict[str, Any]:
        if entity_id not in self._entity_meta_cache:
            row = self.conn.execute(
                "SELECT name, namespace, entity_type FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row:
                self._entity_meta_cache[entity_id] = {
                    "name": str(row[0]),
                    "namespace": str(row[1]),
                    "entity_type": str(row[2]),
                }
            else:
                self._entity_meta_cache[entity_id] = {
                    "name": "",
                    "namespace": "",
                    "entity_type": "",
                }
        return self._entity_meta_cache[entity_id]

    def anchor_ids_from_tokens(self, tokens: list[str]) -> list[int]:
        if self._anchor_map is None:
            self._anchor_map = defaultdict(list)
            for anchor_id, label in self.conn.execute("SELECT id, label FROM anchors").fetchall():
                self._anchor_map[_normalize_token(str(label))].append(int(anchor_id))
        ids: list[int] = []
        for token in tokens:
            ids.extend(self._anchor_map.get(_normalize_token(token), []))
        return list(dict.fromkeys(ids))

    def entity_anchor_labels(self, entity_id: int) -> list[str]:
        if entity_id not in self._entity_anchor_labels:
            rows = self.conn.execute(
                """
                SELECT a.label
                FROM entity_anchors ea
                JOIN anchors a ON a.id = ea.anchor_id
                WHERE ea.entity_id = ?
                """,
                (entity_id,),
            ).fetchall()
            self._entity_anchor_labels[entity_id] = [str(row[0]) for row in rows]
        return self._entity_anchor_labels[entity_id]

    def entity_anchor_ids(self, entity_id: int) -> list[int]:
        if entity_id not in self._entity_anchor_ids:
            rows = self.conn.execute(
                "SELECT anchor_id FROM entity_anchors WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()
            self._entity_anchor_ids[entity_id] = [int(row[0]) for row in rows]
        return self._entity_anchor_ids[entity_id]

    def positions(self, entity_id: int) -> dict[str, tuple[int, int]]:
        if entity_id not in self._position_cache:
            rows = self.conn.execute(
                "SELECT bank, sign, depth FROM entity_positions WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()
            self._position_cache[entity_id] = {
                str(bank): (int(sign), int(depth))
                for bank, sign, depth in rows
                if int(sign) != 0
            }
        return self._position_cache[entity_id]

    def relations(self, source_id: int, target_id: int) -> list[str]:
        key = (source_id, target_id)
        if key not in self._relation_cache:
            rows = self.conn.execute(
                """
                SELECT relation
                FROM entity_links
                WHERE (source_id = ? AND target_id = ?)
                   OR (source_id = ? AND target_id = ?)
                """,
                (source_id, target_id, target_id, source_id),
            ).fetchall()
            self._relation_cache[key] = [str(row[0]) for row in rows]
        return self._relation_cache[key]

    def accessible_premise_count(self, theorem_entity_id: int) -> int:
        if theorem_entity_id not in self._accessible_count_cache:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM accessible_premises WHERE theorem_id = ?",
                (theorem_entity_id,),
            ).fetchone()
            self._accessible_count_cache[theorem_entity_id] = int(row[0]) if row else 0
        return self._accessible_count_cache[theorem_entity_id]

    def theorem_surface(self, theorem_id: str) -> dict[str, Any]:
        theorem_entity_id = self.entity_id(theorem_id)
        if theorem_entity_id is None:
            return {
                "entity_id": None,
                "bank_signature": [],
                "anchor_labels": [],
                "accessible_premise_count": 0,
            }
        positions = self.positions(theorem_entity_id)
        return {
            "entity_id": theorem_entity_id,
            "bank_signature": [
                f"{bank}:{sign}/{depth}"
                for bank, (sign, depth) in sorted(positions.items())
            ],
            "anchor_labels": self.entity_anchor_labels(theorem_entity_id)[:16],
            "accessible_premise_count": self.accessible_premise_count(theorem_entity_id),
        }

    def build_query(self, theorem_id: str, residual_tokens: list[str]) -> tuple[StructuredQuery | None, int | None]:
        theorem_entity_id = self.entity_id(theorem_id)
        if theorem_entity_id is None:
            return None, None

        positions = self.positions(theorem_entity_id)
        if not positions:
            return None, theorem_entity_id

        bank_directions = {bank: sign for bank, (sign, _depth) in positions.items()}
        bank_confidences = {
            bank: min(depth / 3.0, 1.0) if depth > 0 else 0.35
            for bank, (_sign, depth) in positions.items()
        }
        prefer_anchors = list(dict.fromkeys(self.entity_anchor_ids(theorem_entity_id) + self.anchor_ids_from_tokens(residual_tokens)))
        prefer_weights = [1.0] * len(prefer_anchors)
        return (
            StructuredQuery(
                bank_directions=bank_directions,
                bank_confidences=bank_confidences,
                prefer_anchors=prefer_anchors,
                prefer_weights=prefer_weights,
                seed_entity_ids=[theorem_entity_id],
                accessible_theorem_id=theorem_entity_id,
            ),
            theorem_entity_id,
        )

    def discover_candidate_priors(
        self,
        theorem_id: str,
        residual_tokens: list[str],
        reasoning_gap_family: str,
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        query, theorem_entity_id = self.build_query(theorem_id, residual_tokens)
        if query is None or theorem_entity_id is None:
            return []

        theorem_prefixes = namespace_chain(theorem_id)
        theorem_anchor_labels = set(self.entity_anchor_labels(theorem_entity_id))
        family_keywords = _FAMILY_KEYWORDS.get(reasoning_gap_family, [])

        candidates = navigate(
            self.conn,
            query,
            limit=max(top_k * 3, top_k),
            entity_type="lemma",
        )
        enriched: list[dict[str, Any]] = []
        token_norms = {_normalize_token(token) for token in residual_tokens}

        for candidate in candidates:
            if candidate.entity_id == theorem_entity_id:
                continue
            meta = self.entity_meta(candidate.entity_id)
            candidate_name = meta["name"]
            candidate_namespace = meta["namespace"]
            candidate_norm = _normalize_token(candidate_name)
            shared_anchors = [
                label
                for label in self.entity_anchor_labels(candidate.entity_id)
                if label in theorem_anchor_labels
            ]
            symbol_overlap = [
                token
                for token in residual_tokens
                if _normalize_token(token) in candidate_norm
                or _normalize_token(token) in _normalize_token(candidate_namespace)
            ]
            relations = self.relations(theorem_entity_id, candidate.entity_id)
            namespace_match = any(
                candidate_namespace.startswith(prefix) for prefix in theorem_prefixes if prefix
            )
            keyword_matches = [
                keyword
                for keyword in family_keywords
                if keyword in candidate_name.lower() or keyword in candidate_namespace.lower()
            ]
            composite = (
                float(candidate.final_score)
                + 0.12 * len(shared_anchors)
                + 0.10 * len(symbol_overlap)
                + 0.08 * len(relations)
                + 0.06 * len(keyword_matches)
                + (0.05 if namespace_match else 0.0)
            )
            if not shared_anchors and not symbol_overlap and not relations and not namespace_match:
                if token_norms and not any(token in candidate_norm for token in token_norms):
                    continue
            enriched.append(
                {
                    "lemma": candidate_name,
                    "entity_id": int(candidate.entity_id),
                    "namespace": candidate_namespace,
                    "entity_type": meta["entity_type"],
                    "composite_score": round(composite, 4),
                    "navigate_score": round(float(candidate.final_score), 4),
                    "bank_score": round(float(candidate.bank_score), 4),
                    "anchor_score": round(float(candidate.anchor_score), 4),
                    "seed_score": round(float(candidate.seed_score), 4),
                    "shared_anchor_labels": shared_anchors[:8],
                    "symbol_overlap": symbol_overlap[:8],
                    "relation_support": relations[:6],
                    "namespace_match": namespace_match,
                    "family_keyword_matches": keyword_matches,
                }
            )

        enriched.sort(key=lambda row: (-float(row["composite_score"]), row["lemma"]))
        return enriched[:top_k]


def _row_namespace(row: dict[str, Any]) -> str:
    value = str(row.get("namespace_prefix", "")).strip()
    if value:
        return value
    theorem_id = str(row.get("theorem_id", ""))
    if "." in theorem_id:
        return theorem_id.split(".", 1)[0]
    return ""


def _shared_symbols(a: list[str], b: list[str]) -> list[str]:
    b_norm = {_normalize_token(token) for token in b}
    return [token for token in a if _normalize_token(token) in b_norm]


def discover_kline_exemplars(
    solved_rows: list[dict[str, Any]],
    target: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    target_symbols = extract_goal_symbols(
        "\n".join(
            [
                str(target.get("last_goal", "")),
                str(target.get("theorem_statement", "")),
                str(target.get("initial_goal", "")),
            ]
        )
    )
    target_tactics = tactic_prefixes(target.get("tactics_used", []))
    target_template = str(target.get("template_id", ""))
    target_namespace = _row_namespace(target)
    target_band = str(target.get("difficulty_band", ""))

    scored: list[dict[str, Any]] = []
    for row in solved_rows:
        if str(row.get("theorem_id", "")) == str(target.get("theorem_id", "")):
            continue
        score = 0.0
        reasons: list[str] = []
        row_namespace = _row_namespace(row)
        if target_namespace and row_namespace == target_namespace:
            score += 3.0
            reasons.append("same_namespace")
        if target_template and str(row.get("template_id", "")) == target_template:
            score += 2.0
            reasons.append("same_template")
        if target_band and str(row.get("difficulty_band", "")) == target_band:
            score += 0.75
            reasons.append("same_difficulty_band")
        row_tactics = tactic_prefixes(row.get("tactics_used", []))
        shared_tactics = [t for t in target_tactics if t in row_tactics]
        if shared_tactics:
            score += 0.5 * len(shared_tactics)
            reasons.append("shared_tactics")
        row_symbols = extract_goal_symbols(
            "\n".join(
                [
                    str(row.get("theorem_statement", "")),
                    str(row.get("initial_goal", "")),
                ]
            )
        )
        shared_symbols = _shared_symbols(target_symbols, row_symbols)
        if shared_symbols:
            score += min(2.0, 0.3 * len(shared_symbols))
            reasons.append("shared_symbols")
        if score <= 0:
            continue
        scored.append(
            {
                "theorem_id": str(row.get("theorem_id", "")),
                "namespace_prefix": row_namespace,
                "template_id": str(row.get("template_id", "")),
                "close_lane": str(row.get("close_lane", "")),
                "lane_sequence": str(row.get("lane_sequence", "")),
                "tactic_prefixes": row_tactics,
                "shared_symbols": shared_symbols[:8],
                "shared_tactics": shared_tactics,
                "score": round(score, 3),
                "score_reasons": reasons,
            }
        )
    scored.sort(key=lambda row: (-float(row["score"]), row["theorem_id"]))
    return scored[:top_k]


def build_prior_graph_geometry(
    theorem_id: str,
    candidate_priors: list[dict[str, Any]],
    graph_index: _ProofGraphIndex,
) -> dict[str, Any]:
    theorem_surface = graph_index.theorem_surface(theorem_id)
    relation_counts = Counter()
    namespace_counts = Counter()
    keyword_counts = Counter()
    anchor_overlap_max = 0
    for prior in candidate_priors:
        namespace = str(prior.get("namespace", ""))
        if namespace:
            namespace_counts[namespace.split(".", 1)[0]] += 1
        relation_counts.update(prior.get("relation_support", []))
        keyword_counts.update(prior.get("family_keyword_matches", []))
        anchor_overlap_max = max(anchor_overlap_max, len(prior.get("shared_anchor_labels", [])))
    return {
        "theorem_surface": theorem_surface,
        "candidate_count": len(candidate_priors),
        "same_namespace_candidates": sum(1 for prior in candidate_priors if prior.get("namespace_match")),
        "relation_support_counts": dict(relation_counts.most_common(8)),
        "top_candidate_roots": dict(namespace_counts.most_common(8)),
        "family_keyword_counts": dict(keyword_counts.most_common(8)),
        "max_anchor_overlap": anchor_overlap_max,
        "top_candidates": [
            {
                "lemma": prior.get("lemma", ""),
                "namespace": prior.get("namespace", ""),
                "score": prior.get("composite_score", 0.0),
            }
            for prior in candidate_priors[:5]
        ],
    }


def build_kline_geometry(exemplars: list[dict[str, Any]]) -> dict[str, Any]:
    lane_counts = Counter(str(exemplar.get("close_lane", "")) for exemplar in exemplars)
    shared_symbols = Counter(
        symbol
        for exemplar in exemplars
        for symbol in exemplar.get("shared_symbols", [])
    )
    return {
        "exemplar_count": len(exemplars),
        "same_namespace_hits": sum(
            1 for exemplar in exemplars if "same_namespace" in exemplar.get("score_reasons", [])
        ),
        "same_template_hits": sum(
            1 for exemplar in exemplars if "same_template" in exemplar.get("score_reasons", [])
        ),
        "close_lane_counts": dict(lane_counts.most_common(6)),
        "shared_symbol_hints": [symbol for symbol, _count in shared_symbols.most_common(10)],
        "top_exemplar_signatures": [
            {
                "namespace_prefix": str(exemplar.get("namespace_prefix", "")),
                "template_id": str(exemplar.get("template_id", "")),
                "close_lane": str(exemplar.get("close_lane", "")),
            }
            for exemplar in exemplars[:5]
        ],
    }


def discover_negative_kline_exemplars(
    failed_rows: list[dict[str, Any]],
    target: dict[str, Any],
    graph_index: _ProofGraphIndex,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    target_symbols = extract_goal_symbols(
        "\n".join(
            [
                str(target.get("last_goal", "")),
                str(target.get("theorem_statement", "")),
                str(target.get("initial_goal", "")),
            ]
        )
    )
    target_tactics = tactic_prefixes(target.get("tactics_used", []))
    target_template = str(target.get("template_id", ""))
    target_namespace = _row_namespace(target)
    target_family = str(target.get("reasoning_gap_family", ""))
    target_bucket = str(target.get("last_goal_bucket", ""))
    target_pathologies = set(target.get("search_pathology_tags", []) or [])
    target_surface = graph_index.theorem_surface(str(target.get("theorem_id", "")))
    target_bank_signature = set(target_surface.get("bank_signature", []))

    scored: list[dict[str, Any]] = []
    for row in failed_rows:
        if str(row.get("theorem_id", "")) == str(target.get("theorem_id", "")):
            continue
        if not bool(row.get("started")) or bool(row.get("success")):
            continue

        score = 0.0
        reasons: list[str] = []
        row_namespace = _row_namespace(row)
        row_family = str(row.get("reasoning_gap_family", ""))
        row_bucket = str(row.get("last_goal_bucket", ""))
        row_tactics = tactic_prefixes(row.get("tactics_used", []))
        row_pathologies = set(row.get("search_pathology_tags", []) or [])
        row_symbols = extract_goal_symbols(
            "\n".join(
                [
                    str(row.get("last_goal", "")),
                    str(row.get("theorem_statement", "")),
                    str(row.get("initial_goal", "")),
                ]
            )
        )
        shared_symbols = _shared_symbols(target_symbols, row_symbols)
        shared_tactics = [t for t in target_tactics if t in row_tactics]
        shared_pathologies = sorted(target_pathologies & row_pathologies)
        row_surface = graph_index.theorem_surface(str(row.get("theorem_id", "")))
        shared_bank_signature = [
            sig for sig in row_surface.get("bank_signature", []) if sig in target_bank_signature
        ]

        if row_family and row_family == target_family:
            score += 2.5
            reasons.append("same_family")
        if row_bucket and row_bucket == target_bucket:
            score += 1.25
            reasons.append("same_goal_bucket")
        if target_namespace and row_namespace == target_namespace:
            score += 2.0
            reasons.append("same_namespace")
        if target_template and str(row.get("template_id", "")) == target_template:
            score += 1.5
            reasons.append("same_template")
        if shared_symbols:
            score += min(2.0, 0.3 * len(shared_symbols))
            reasons.append("shared_symbols")
        if shared_tactics:
            score += min(1.5, 0.4 * len(shared_tactics))
            reasons.append("shared_tactics")
        if shared_pathologies:
            score += min(2.0, 0.6 * len(shared_pathologies))
            reasons.append("shared_pathologies")
        if shared_bank_signature:
            score += min(1.0, 0.25 * len(shared_bank_signature))
            reasons.append("shared_bank_signature")
        if score <= 0:
            continue

        scored.append(
            {
                "theorem_id": str(row.get("theorem_id", "")),
                "namespace_prefix": row_namespace,
                "template_id": str(row.get("template_id", "")),
                "reasoning_gap_family": row_family,
                "last_goal_bucket": row_bucket,
                "attempt_band": str(row.get("attempt_band", "")),
                "attempts": int(row.get("attempts", 0) or 0),
                "tactic_prefixes": row_tactics,
                "search_pathology_tags": sorted(row_pathologies),
                "shared_symbols": shared_symbols[:8],
                "shared_tactics": shared_tactics,
                "shared_pathologies": shared_pathologies,
                "shared_bank_signature": shared_bank_signature[:8],
                "score": round(score, 3),
                "score_reasons": reasons,
            }
        )
    scored.sort(key=lambda row: (-float(row["score"]), row["theorem_id"]))
    return scored[:top_k]


def build_negative_kline_geometry(exemplars: list[dict[str, Any]]) -> dict[str, Any]:
    pathology_counts = Counter(
        tag
        for exemplar in exemplars
        for tag in exemplar.get("search_pathology_tags", [])
    )
    tactic_counts = Counter(
        tactic
        for exemplar in exemplars
        for tactic in exemplar.get("tactic_prefixes", [])
    )
    return {
        "exemplar_count": len(exemplars),
        "same_namespace_hits": sum(
            1 for exemplar in exemplars if "same_namespace" in exemplar.get("score_reasons", [])
        ),
        "same_family_hits": sum(
            1 for exemplar in exemplars if "same_family" in exemplar.get("score_reasons", [])
        ),
        "plateau_hits": sum(
            1
            for exemplar in exemplars
            if any(
                tag in exemplar.get("search_pathology_tags", [])
                for tag in ["no_progress_plateau", "blank_lane_plateau"]
            )
        ),
        "top_pathologies": dict(pathology_counts.most_common(8)),
        "avoid_tactic_prefixes": [tactic for tactic, _count in tactic_counts.most_common(8)],
        "top_negative_signatures": [
            {
                "namespace_prefix": str(exemplar.get("namespace_prefix", "")),
                "resolution_family": str(exemplar.get("reasoning_gap_family", "")),
                "goal_bucket": str(exemplar.get("last_goal_bucket", "")),
            }
            for exemplar in exemplars[:5]
        ],
    }


def build_startability_surface(row: dict[str, Any]) -> dict[str, Any]:
    theorem_text = "\n".join(
        [
            str(row.get("theorem_statement", "")),
            str(row.get("initial_goal", "")),
        ]
    )
    return {
        "theorem_id": str(row.get("theorem_id", "")),
        "module": str(row.get("module", "")),
        "lean_path": str(row.get("lean_path", row.get("file_path", ""))),
        "theorem_line": int(row.get("theorem_line", 0) or 0),
        "start_failure_family": str(row.get("start_failure_family", "")),
        "start_failure_tags": list(row.get("start_failure_tags", [])),
        "goal_start_status": str(row.get("goal_start_status", "")),
        "failure_category": str(row.get("failure_category", "")),
        "context_features": row.get("context_features", {}),
        "context_unsupported_kinds": row.get("context_unsupported_kinds", []),
        "reconstruction_actions": _infer_startability_actions(row),
        "theorem_shape_features": goal_shape_features(theorem_text),
        "top_symbols": extract_goal_symbols(theorem_text, limit=20),
        "domain_hints": _domain_hints(str(row.get("theorem_id", "")), theorem_text),
    }


def build_hard_som_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_version": "hard_som_surface_v4",
        "theorem_id": str(packet.get("theorem_id", "")),
        "split": str(packet.get("split", "")),
        "difficulty_band": str(packet.get("difficulty_band", "")),
        "resolution_family": str(packet.get("resolution_family", "")),
        "goal_bucket": str(packet.get("last_goal_bucket", "")),
        "residual_bucket": str(packet.get("residual_bucket", "")),
        "residual_skeleton_geometry": packet.get("residual_skeleton_geometry", {}),
        "proof_plan_geometry": packet.get("proof_plan_geometry", {}),
        "prior_graph_geometry": packet.get("prior_graph_geometry", {}),
        "dependency_geometry": packet.get("decomposition_profile", {}),
        "kline_geometry": packet.get("kline_geometry", {}),
        "negative_kline_geometry": packet.get("negative_kline_geometry", {}),
        "search_control_geometry": packet.get("search_control_geometry", {}),
        "candidate_priors": packet.get("candidate_priors", [])[:8],
        "kline_exemplars": [
            {
                "namespace_prefix": exemplar.get("namespace_prefix", ""),
                "template_id": exemplar.get("template_id", ""),
                "close_lane": exemplar.get("close_lane", ""),
                "lane_sequence": exemplar.get("lane_sequence", ""),
                "tactic_prefixes": exemplar.get("tactic_prefixes", []),
                "shared_symbols": exemplar.get("shared_symbols", []),
                "shared_tactics": exemplar.get("shared_tactics", []),
                "score": exemplar.get("score", 0.0),
                "score_reasons": exemplar.get("score_reasons", []),
            }
            for exemplar in packet.get("kline_exemplars", [])[:4]
        ],
        "negative_kline_exemplars": [
            {
                "theorem_id": exemplar.get("theorem_id", ""),
                "namespace_prefix": exemplar.get("namespace_prefix", ""),
                "template_id": exemplar.get("template_id", ""),
                "reasoning_gap_family": exemplar.get("reasoning_gap_family", ""),
                "last_goal_bucket": exemplar.get("last_goal_bucket", ""),
                "tactic_prefixes": exemplar.get("tactic_prefixes", []),
                "search_pathology_tags": exemplar.get("search_pathology_tags", []),
                "shared_pathologies": exemplar.get("shared_pathologies", []),
                "score": exemplar.get("score", 0.0),
                "score_reasons": exemplar.get("score_reasons", []),
            }
            for exemplar in packet.get("negative_kline_exemplars", [])[:4]
        ],
        "symbolic_targets": {
            "closing_features": packet.get("closing_features", []),
            "candidate_methods": packet.get("proof_plan_geometry", {}).get("candidate_methods", []),
            "specialist_targets": packet.get("proof_plan_geometry", {}).get("specialist_targets", []),
            "lane_suppression_hints": packet.get("proof_plan_geometry", {}).get("lane_suppression_hints", []),
            "search_pathology_tags": packet.get("residual_skeleton_geometry", {}).get("search_pathology_tags", []),
            "avoid_tactic_prefixes": packet.get("negative_kline_geometry", {}).get("avoid_tactic_prefixes", []),
            "prior_lemma_names": [
                prior.get("lemma", "") for prior in packet.get("candidate_priors", [])[:5]
            ],
        },
        "dr_ducky_surface": packet.get("dr_ducky_surface", {}),
    }


def build_resolution_packet(
    row: dict[str, Any],
    solved_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    graph_index: _ProofGraphIndex,
    candidate_limit: int = 12,
    exemplar_limit: int = 5,
) -> dict[str, Any]:
    from src.dr_ducky import build_goal_capsule

    remaining_goals = row.get("remaining_goals_snapshot", [])
    if not isinstance(remaining_goals, list):
        remaining_goals = []
    theorem_id = str(row.get("theorem_id", ""))
    goal_text = str(row.get("last_goal") or "")
    reasoning_gap_family = str(row.get("reasoning_gap_family", ""))
    goal_symbols = extract_goal_symbols(
        "\n".join(
            [
                goal_text,
                str(row.get("theorem_statement", "")),
                str(row.get("initial_goal", "")),
                theorem_id,
            ]
        )
    )
    decomposition_profile = build_dependency_profile(remaining_goals)
    skeleton = build_residual_skeleton_geometry(row, remaining_goals)
    search_control_geometry = build_search_control_geometry(row)
    candidate_prior_rows = graph_index.discover_candidate_priors(
        theorem_id=theorem_id,
        residual_tokens=goal_symbols,
        reasoning_gap_family=reasoning_gap_family,
        top_k=candidate_limit,
    )
    exemplar_rows = discover_kline_exemplars(
        solved_rows,
        row,
        top_k=exemplar_limit,
    )
    negative_exemplar_rows = discover_negative_kline_exemplars(
        failed_rows,
        row,
        graph_index,
        top_k=exemplar_limit,
    )
    closing_feature_list = closing_features(reasoning_gap_family, goal_text)
    proof_plan = build_proof_plan_geometry(
        row,
        skeleton=skeleton,
        decomposition_profile=decomposition_profile,
        closing_feature_list=closing_feature_list,
        search_control_geometry=search_control_geometry,
    )
    packet = dict(row)
    packet["resolution_layer"] = "symbolic_hard_resolution_v2"
    packet["resolution_family"] = reasoning_gap_family or "hard_residual_search"
    packet["closing_features"] = closing_feature_list
    packet["goal_symbols"] = goal_symbols
    packet["namespace_chain"] = namespace_chain(theorem_id)
    packet["tactic_prefixes"] = tactic_prefixes(row.get("tactics_used", []))
    packet["decomposition_profile"] = decomposition_profile
    packet["residual_skeleton_geometry"] = skeleton
    packet["search_control_geometry"] = search_control_geometry
    packet["candidate_priors"] = candidate_prior_rows
    packet["prior_graph_geometry"] = build_prior_graph_geometry(
        theorem_id=theorem_id,
        candidate_priors=candidate_prior_rows,
        graph_index=graph_index,
    )
    packet["kline_exemplars"] = exemplar_rows
    packet["kline_geometry"] = build_kline_geometry(exemplar_rows)
    packet["negative_kline_exemplars"] = negative_exemplar_rows
    packet["negative_kline_geometry"] = build_negative_kline_geometry(negative_exemplar_rows)
    packet["proof_plan_geometry"] = proof_plan
    packet["symbolic_packet"] = {
        "discovery_sources": {
            "graph_seed": theorem_id,
            "goal_symbol_count": len(goal_symbols),
            "family_keywords": _FAMILY_KEYWORDS.get(reasoning_gap_family, []),
        },
        "goal_bucket": str(row.get("last_goal_bucket", "")),
        "resolution_family": packet["resolution_family"],
        "closing_features": packet["closing_features"],
        "residual_skeleton_geometry": packet["residual_skeleton_geometry"],
        "proof_plan_geometry": packet["proof_plan_geometry"],
        "specialist_targets": packet["proof_plan_geometry"].get("specialist_targets", []),
        "lane_suppression_hints": packet["proof_plan_geometry"].get("lane_suppression_hints", []),
        "prior_graph_geometry": packet["prior_graph_geometry"],
        "decomposition_profile": packet["decomposition_profile"],
        "kline_geometry": packet["kline_geometry"],
        "search_control_geometry": packet["search_control_geometry"],
        "negative_kline_geometry": packet["negative_kline_geometry"],
    }
    ducky_capsule = build_goal_capsule(row)
    packet["dr_ducky_surface"] = {
        "ducky_specialist_target": list(ducky_capsule.specialist_targets),
        "engine_family": list(ducky_capsule.allowed_engines),
        "backend_preferences": list(ducky_capsule.backend_preferences),
        "projector_markers": list(ducky_capsule.specification.projector_markers),
        "projector_policy": dict(ducky_capsule.projector_policy),
        "certificate_shape": sorted(
            {
                kind
                for skeleton in ducky_capsule.proof_skeletons
                for kind in (skeleton.certificate_kinds or [])
            }
        ),
        "proof_dsl_program_count": len(ducky_capsule.proof_dsl_programs),
        "solver_constraint_profile": {
            "count": len(ducky_capsule.solver_constraints),
            "kinds": sorted({constraint.constraint_kind for constraint in ducky_capsule.solver_constraints}),
        },
        "eqsat_plan": ducky_capsule.eqsat_plan.to_dict() if ducky_capsule.eqsat_plan is not None else None,
        "relational_surface": {
            "spec_count": len(ducky_capsule.relational_search_specs),
            "relation_symbols": sorted(
                {
                    symbol
                    for spec in ducky_capsule.relational_search_specs
                    for symbol in spec.relation_symbols
                }
            ),
        },
        "negative_geometry": {
            "suppression_hints": list(ducky_capsule.suppression_hints),
            "pathology_tags": list(ducky_capsule.specification.pathology_tags),
            "plateau_detected": ducky_capsule.specification.search_control.get("plateau_detected", 0),
        },
        "ledger_seed_summary": {
            "fact_count": len(ducky_capsule.ledger_seed.facts),
            "candidate_witness_count": len(ducky_capsule.ledger_seed.candidate_witnesses),
            "candidate_rewrite_count": len(ducky_capsule.ledger_seed.candidate_rewrites),
        },
    }
    return packet


def materialize_hard_resolution_layer(
    rows: list[dict[str, Any]],
    output_dir: Path,
    conn_or_db: sqlite3.Connection | str | Path,
    candidate_limit: int = 12,
    exemplar_limit: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    close_conn = False
    if isinstance(conn_or_db, sqlite3.Connection):
        conn = conn_or_db
    else:
        conn = sqlite3.connect(str(conn_or_db))
        close_conn = True

    try:
        graph_index = _ProofGraphIndex(conn)
        hard_rows = [
            row
            for row in rows
            if str(row.get("follow_on_stage", "")) == "hard_proof_solver"
        ]
        compiler_rows = [
            row
            for row in rows
            if str(row.get("follow_on_stage", "")) == "compiler_specialist"
        ]
        solved_rows = []
        failed_started_rows = []
        for row in rows:
            if not bool(row.get("success")):
                if bool(row.get("started")):
                    failed_started_rows.append(row)
                continue
            honest = row.get("honest_success")
            if honest is None:
                honest = not detect_self_application(row)
            if honest:
                solved_rows.append(row)
        packets = [
            build_resolution_packet(
                row=row,
                solved_rows=solved_rows,
                failed_rows=failed_started_rows,
                graph_index=graph_index,
                candidate_limit=candidate_limit,
                exemplar_limit=exemplar_limit,
            )
            for row in hard_rows
        ]

        _write_jsonl(output_dir / "resolution_packets.jsonl", packets)
        hard_som_packets = [build_hard_som_packet(packet) for packet in packets]
        _write_jsonl(output_dir / "hard_som_packets.jsonl", hard_som_packets)
        compiler_packets = []
        for row in compiler_rows:
            compiler_packets.append(
                {
                    "packet_version": "compiler_specialist_surface_v2",
                    "theorem_id": str(row.get("theorem_id", "")),
                    "split": str(row.get("split", "")),
                    "startability_surface": build_startability_surface(row),
                }
            )
        _write_jsonl(output_dir / "compiler_specialist_packets.jsonl", compiler_packets)

        family_dir = output_dir / "by_resolution_family"
        family_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for packet in packets:
            grouped[str(packet.get("resolution_family", ""))].append(packet)
        for family, entries in grouped.items():
            if family:
                _write_jsonl(family_dir / f"{family}.jsonl", entries)

        feature_counts = Counter()
        goal_symbol_counts = Counter()
        prior_counts = Counter()
        exemplar_counts = Counter()
        representation_pressures = Counter()
        plan_methods = Counter()
        domain_hints = Counter()
        search_pathologies = Counter()
        negative_pathologies = Counter()
        negative_avoid_tactics = Counter()
        compiler_actions = Counter()
        for packet in packets:
            feature_counts.update(packet.get("closing_features", []))
            goal_symbol_counts.update(packet.get("goal_symbols", [])[:8])
            prior_counts.update(prior["lemma"] for prior in packet.get("candidate_priors", [])[:3])
            exemplar_counts.update(
                exemplar["theorem_id"] for exemplar in packet.get("kline_exemplars", [])[:3]
            )
            representation_pressures.update(
                packet.get("residual_skeleton_geometry", {}).get("representation_pressures", [])
            )
            plan_methods.update(
                packet.get("proof_plan_geometry", {}).get("candidate_methods", [])
            )
            domain_hints.update(packet.get("residual_skeleton_geometry", {}).get("domain_hints", []))
            search_pathologies.update(
                packet.get("residual_skeleton_geometry", {}).get("search_pathology_tags", [])
            )
            negative_pathologies.update(
                packet.get("negative_kline_geometry", {}).get("top_pathologies", {})
            )
            negative_avoid_tactics.update(
                packet.get("negative_kline_geometry", {}).get("avoid_tactic_prefixes", [])
            )
        for packet in compiler_packets:
            compiler_actions.update(
                packet.get("startability_surface", {}).get("reconstruction_actions", [])
            )

        feature_inventory = {
            "by_resolution_family": {
                family: {
                    "count": len(entries),
                    "top_features": Counter(
                        feature
                        for entry in entries
                        for feature in entry.get("closing_features", [])
                    ).most_common(12),
                }
                for family, entries in grouped.items()
            },
            "top_goal_symbols": goal_symbol_counts.most_common(40),
        }
        candidate_inventory = {
            "top_candidate_priors": prior_counts.most_common(40),
        }
        exemplar_inventory = {
            "top_kline_exemplars": exemplar_counts.most_common(40),
        }
        surface_inventory = {
            "top_representation_pressures": representation_pressures.most_common(30),
            "top_plan_methods": plan_methods.most_common(30),
            "top_domain_hints": domain_hints.most_common(20),
            "top_search_pathologies": search_pathologies.most_common(20),
            "top_negative_pathologies": negative_pathologies.most_common(20),
            "top_negative_avoid_tactics": negative_avoid_tactics.most_common(20),
            "top_compiler_reconstruction_actions": compiler_actions.most_common(20),
        }

        (output_dir / "closing_feature_inventory.json").write_text(
            json.dumps(feature_inventory, indent=2)
        )
        (output_dir / "candidate_prior_inventory.json").write_text(
            json.dumps(candidate_inventory, indent=2)
        )
        (output_dir / "kline_exemplar_inventory.json").write_text(
            json.dumps(exemplar_inventory, indent=2)
        )
        (output_dir / "surface_inventory.json").write_text(
            json.dumps(surface_inventory, indent=2)
        )

        summary = {
            "total_rows": len(rows),
            "total_hard_packets": len(packets),
            "packets_with_candidate_priors": sum(
                1 for packet in packets if packet.get("candidate_priors")
            ),
            "packets_with_kline_exemplars": sum(
                1 for packet in packets if packet.get("kline_exemplars")
            ),
            "compiler_specialist_packets": len(compiler_packets),
            "by_resolution_family": dict(
                Counter(str(packet.get("resolution_family", "")) for packet in packets).most_common()
            ),
            "by_last_goal_bucket": dict(
                Counter(str(packet.get("last_goal_bucket", "")) for packet in packets).most_common()
            ),
            "top_closing_features": feature_counts.most_common(20),
            "top_representation_pressures": representation_pressures.most_common(20),
            "top_plan_methods": plan_methods.most_common(20),
            "top_domain_hints": domain_hints.most_common(20),
            "top_search_pathologies": search_pathologies.most_common(20),
            "top_negative_pathologies": negative_pathologies.most_common(20),
            "top_negative_avoid_tactics": negative_avoid_tactics.most_common(20),
            "top_compiler_reconstruction_actions": compiler_actions.most_common(20),
            "top_candidate_priors": prior_counts.most_common(20),
            "top_kline_exemplars": exemplar_counts.most_common(20),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if close_conn:
            conn.close()
