"""
Lean kernel interface — Pantograph-based tactic verification.

Provides try_tactic() and try_hammer() for sending tactics to the Lean 4
kernel and receiving success/failure with new goal states.

Backends:
    stub        — always fails, for development without Lean
    replay      — matches against registered ground-truth tactics
    pantograph  — real verification via PyPantograph (Lane A)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from src.lean_context_ir import extract_context_ir
from src.nav_contracts import LeanFeedback, TacticResult

logger = logging.getLogger(__name__)

_DECL_MODIFIERS = r"(?:(?:noncomputable|private|protected)\s+)*"


# ---------------------------------------------------------------------------
# Lean feedback classification
# ---------------------------------------------------------------------------


def _serialise_messages(messages: object) -> list[dict]:
    """Convert a list of Pantograph Message objects to plain dicts.

    Preserves pos/pos_end as structured dicts (line/column) rather than
    stringifying them, so repair logic can locate the error site.
    """
    if not isinstance(messages, list):
        return []
    result = []
    for m in messages:
        try:
            pos = getattr(m, "pos", None)
            pos_end = getattr(m, "pos_end", None)

            def _pos_dict(p: object) -> dict | None:
                if p is None:
                    return None
                if isinstance(p, dict):
                    return p
                return {
                    "line": getattr(p, "line", None),
                    "column": getattr(p, "column", None),
                }

            result.append(
                {
                    "severity": str(getattr(m, "severity", "error")),
                    "kind": getattr(m, "kind", None),
                    "data": str(getattr(m, "data", "")),
                    "pos": _pos_dict(pos),
                    "pos_end": _pos_dict(pos_end),
                }
            )
        except Exception:
            result.append({"data": str(m)})
    return result


def _classify_tactic_failure(exc: Exception) -> LeanFeedback:
    """
    Classify a Pantograph TacticFailure into a structured LeanFeedback object.

    TacticFailure is raised in three shapes:
      TacticFailure(result_dict)          — parseError
      TacticFailure([Message, ...])       — tactic elaboration error
      TacticFailure("hasSorry/Unsafe", messages)  — sorry/unsafe guard
    """
    args = exc.args

    # Shape 1: parseError — first arg is a dict with "parseError" key
    if args and isinstance(args[0], dict) and "parseError" in args[0]:
        return LeanFeedback(
            stage="tactic_parse",
            category="parse_error",
            messages=[{"data": str(args[0]["parseError"])}],
            raw_error=str(exc),
        )

    # Shape 2: hasSorry / hasUnsafe — first arg is a string sentinel
    if args and isinstance(args[0], str):
        sentinel = args[0].lower()
        if "sorry" in sentinel:
            cat = "generated_sorry"
        elif "unsafe" in sentinel:
            cat = "generated_unsafe"
        else:
            cat = "other"
        msgs = _serialise_messages(args[1] if len(args) > 1 else [])
        return LeanFeedback(
            stage="tactic_exec",
            category=cat,
            messages=msgs,
            raw_error=str(exc),
        )

    # Shape 3: list of Message objects
    messages_raw = args[0] if args and isinstance(args[0], list) else []
    msgs = _serialise_messages(messages_raw)

    # Classify by message content
    combined = " ".join(m.get("data", "") for m in msgs).lower()
    if "unknown identifier" in combined or "unknown constant" in combined:
        cat = "unknown_identifier"
    elif (
        "type mismatch" in combined
        or "application type mismatch" in combined
        or "could not unify" in combined
        or "failed to unify" in combined
    ):
        cat = "unification_mismatch"
    elif "failed to synthesize" in combined:
        # Distinguish typeclass failure from a mismatch that also mentions synthesis
        if "type mismatch" in combined or "application type mismatch" in combined:
            cat = "unification_mismatch"
        else:
            cat = "typeclass_missing"
    elif "function expected" in combined:
        cat = "unification_mismatch"
    else:
        cat = "other"

    return LeanFeedback(
        stage="tactic_exec",
        category=cat,
        messages=msgs,
        raw_error=str(exc),
    )


def _classify_compiler_feedback(
    exc: Exception,
    stage: str,
    fallback_category: str = "other",
) -> LeanFeedback:
    """Classify a generic Pantograph/compiler exception into LeanFeedback."""
    feedback = _classify_tactic_failure(exc)
    category = feedback.category
    raw_error = feedback.raw_error or str(exc)
    combined = " ".join(m.get("data", "") for m in feedback.messages).lower()
    raw_lower = raw_error.lower()

    if category == "other":
        if "cannot start goal" in raw_lower:
            category = "goal_creation_fail"
        elif "unknown identifier" in raw_lower or "unknown constant" in raw_lower:
            category = "unknown_identifier"
        elif "type mismatch" in raw_lower or "application type mismatch" in raw_lower:
            category = "unification_mismatch"
        elif "failed to synthesize" in raw_lower:
            category = "typeclass_missing"
        elif fallback_category != "other":
            category = fallback_category
    elif category == "typeclass_missing" and (
        "type mismatch" in combined or "application type mismatch" in combined
    ):
        category = "unification_mismatch"

    return replace(
        feedback,
        stage=stage,
        category=category,
        raw_error=raw_error,
    )


@dataclass
class LeanConfig:
    """Configuration for Lean kernel connection."""

    backend: str = "stub"  # "stub", "replay", "pantograph"
    timeout: int = 120  # seconds; Mathlib server startup takes ~30-40s
    hammer_timeout: int = 60
    project_root: str = ""
    imports: list[str] = field(default_factory=lambda: ["Init"])


@dataclass
class ReplayResult:
    """Result of creating a goal state via file-context replay."""

    success: bool
    goal_state: str
    goal_state_obj: Any | None
    tier_used: str  # "A" | "B" | "C"
    failure_category: str  # "" | "goal_creation_fail" | "prefix_replay_fail" | "goal_match_fail"
    failing_prefix_idx: int  # -1 if no failure
    crash_retries: int
    env_key: str
    goal_id: int = 0  # index into GoalState.goals for Site addressing
    feedback: LeanFeedback | None = None


class ServerCrashError(Exception):
    """Raised when the Pantograph server crashes (broken pipe, etc).

    Callers should catch this to distinguish infrastructure failures
    from semantic tactic failures.
    """


def qualify_tactic(
    tactic: str,
    accessible_names: set[str],
    suffix_index: dict[str, list[str]],
) -> str:
    """Qualify short identifiers in a tactic using accessible premises.

    For each token in the tactic that looks like an unqualified identifier
    (no dots, not a Lean keyword), check if it matches a suffix of any
    accessible premise name. If there's a unique match within the accessible
    set, replace it with the fully qualified name.

    This is conservative: ambiguous matches (multiple accessible premises
    with the same suffix) are left unqualified to avoid wrong substitutions.
    Local hypothesis names (short lowercase single-letter) are never touched.
    """

    _LEAN_KEYWORDS = frozenset(
        {
            "by",
            "at",
            "with",
            "using",
            "fun",
            "let",
            "have",
            "show",
            "do",
            "if",
            "then",
            "else",
            "match",
            "in",
            "where",
            "return",
            "true",
            "false",
            "intro",
            "intros",
            "apply",
            "apply?",
            "exact",
            "exact?",
            "rw",
            "simp",
            "constructor",
            "cases",
            "induction",
            "rfl",
            "trivial",
            "omega",
            "decide",
            "aesop",
            "solve_by_elim",
            "ring",
            "linarith",
            "norm_num",
            "ext",
            "congr",
            "assumption",
            "contradiction",
            "exfalso",
            "push_neg",
            "dsimp",
            "change",
            "refine",
            "use",
            "obtain",
            "rcases",
            "specialize",
            "revert",
            "clear",
            "rename_i",
            "calc",
            "conv",
            "rwa",
            "simpa",
            "simp_all",
            "tauto",
            "Abel",
            "field_simp",
            "norm_cast",
            "push_cast",
            "positivity",
            "gcongr",
            "rel",
            "nontriviality",
            "continuity",
            "measurability",
            "polyrith",
            "filter_upwards",
            "all_goals",
            "any_goals",
            "focus",
            "sorry",
            "only",
            "scoped",
        }
    )

    def _qualify_token(token: str) -> str:
        # Skip keywords, numbers, operators, short locals
        if token in _LEAN_KEYWORDS:
            return token
        if not token or not token[0].isalpha() and token[0] != "_":
            return token
        if "." in token:
            return token  # already qualified
        # Don't touch very short lowercase names (likely local hyps)
        if len(token) <= 2 and token[0].islower():
            return token

        candidates = suffix_index.get(token, [])
        if not candidates:
            return token

        # Filter to accessible premises
        accessible_matches = [c for c in candidates if c in accessible_names]
        if len(accessible_matches) == 1:
            return accessible_matches[0]

        # Ambiguous or no accessible match — leave unqualified
        return token

    # Tokenize: split on whitespace and common tactic punctuation,
    # preserving structure. We only qualify top-level identifiers,
    # not those inside brackets/parens (which are often expressions).
    # Simple approach: split by whitespace, qualify each token.
    tokens = tactic.split()
    qualified = []
    for tok in tokens:
        # Handle [name] and [← name] patterns
        if tok.startswith("[") or tok.startswith("←"):
            qualified.append(tok)
        elif tok.endswith("]") or tok.endswith(","):
            # Strip trailing punctuation, qualify, re-attach
            suffix = ""
            core = tok
            while core and core[-1] in "],":
                suffix = core[-1] + suffix
                core = core[:-1]
            qualified.append(_qualify_token(core) + suffix)
        else:
            qualified.append(_qualify_token(tok))

    return " ".join(qualified)


def build_suffix_index(entity_names: list[str]) -> dict[str, list[str]]:
    """Build a suffix → full_name index for name qualification."""
    idx: dict[str, list[str]] = {}
    for name in entity_names:
        parts = name.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            idx.setdefault(suffix, []).append(name)
    return idx


def build_local_alias_map(
    actual_goal_str: str,
    expected_goal_str: str,
) -> dict[str, str]:
    """Build a mapping from expected local names to actual Tier B names.

    Lean's load_sorry generates inaccessible names (e.g., 𝕜✝, E✝, inst✝)
    for binder variables, while LeanDojo traces use the source proof names
    (𝕜, E, inst). This function matches them by:
    1. Stripping ✝ suffixes and numeric daggers (✝¹, ✝²)
    2. Matching by stem
    3. Falling back to positional order for ambiguous stems

    Returns: {expected_name: actual_name} for names that differ.
    """

    def _parse_locals(goal_str: str) -> list[tuple[str, str]]:
        """Extract (name, type_prefix) pairs from a goal state string."""
        locals_list = []
        for line in goal_str.split("\n"):
            line = line.strip()
            if not line or line.startswith("⊢") or line.startswith("case "):
                continue
            if ":" in line:
                name_part = line.split(":")[0].strip()
                type_part = line.split(":", 1)[1].strip()[:40]
                # Handle "inst✝¹ : Foo" — name is the last word before ':'
                name = name_part.split()[-1] if name_part else ""
                if name:
                    locals_list.append((name, type_part))
        return locals_list

    def _strip_dagger(name: str) -> str:
        """Strip ✝ and numeric suffixes: 𝕜✝ → 𝕜, inst✝¹ → inst"""
        # Remove trailing ✝ and any digits/superscripts after it
        idx = name.find("✝")
        if idx >= 0:
            return name[:idx]
        return name

    actual_locals = _parse_locals(actual_goal_str)
    expected_locals = _parse_locals(expected_goal_str)

    alias_map: dict[str, str] = {}

    # First pass: match by stem (strip ✝)
    actual_by_stem: dict[str, list[str]] = {}
    for aname, _ in actual_locals:
        stem = _strip_dagger(aname)
        actual_by_stem.setdefault(stem, []).append(aname)

    for ename, _ in expected_locals:
        if ename in [a for a, _ in actual_locals]:
            continue  # name already matches exactly
        stem = _strip_dagger(ename)
        candidates = actual_by_stem.get(stem, [])
        if len(candidates) == 1 and candidates[0] != ename:
            alias_map[ename] = candidates[0]
        elif len(candidates) > 1:
            # Multiple candidates — try positional match
            pass  # leave unresolved for now

    return alias_map


def rewrite_tactic_locals(
    tactic: str,
    alias_map: dict[str, str],
) -> str:
    """Rewrite local hypothesis names in a tactic using an alias map.

    Replaces expected names (from LeanDojo traces) with actual names
    (from the Tier B goal state). Only replaces whole-word matches.
    """
    if not alias_map:
        return tactic
    import re as _re

    result = tactic
    # Sort by length descending to replace longer names first
    for expected, actual in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        # Whole-word replacement (with word boundary)
        result = _re.sub(r"\b" + _re.escape(expected) + r"\b", actual, result)
    return result


def _strip_daggers(s: str) -> str:
    """Strip ✝ and trailing superscript digits for alpha comparison."""
    return re.sub(r"✝[\d\u00b9\u00b2\u00b3\u2070-\u2079]*", "", s)


def _extract_local_names(goal_str: str) -> list[str]:
    """Extract local variable names from a goal state string."""
    names = []
    for line in goal_str.split("\n"):
        line = line.strip()
        if not line or line.startswith("⊢") or line.startswith("case "):
            continue
        if ":" in line:
            name = line.split(":")[0].strip().split()[-1]
            if name:
                names.append(name)
    return names


def _is_instance_like_name(name: str) -> bool:
    return name.startswith("inst") or "✝" in name


def _sanitize_goal_binder_name(name: str) -> str:
    cleaned = re.sub(r"✝[\d\u00b9\u00b2\u00b3\u2070-\u2079]*", "", name)
    cleaned = cleaned.strip()
    if not cleaned:
        return "_wf"
    return cleaned


def theorem_type_from_goal_pp(goal_str: str) -> str:
    """Synthesize a theorem type from a pretty-printed Lean goal state."""
    if not goal_str or "⊢" not in goal_str:
        return ""

    binders: list[str] = []
    target = ""
    used_names: set[str] = set()

    def _fresh_name(base: str) -> str:
        candidate = base
        idx = 1
        while candidate in used_names:
            idx += 1
            candidate = f"{base}_{idx}"
        used_names.add(candidate)
        return candidate

    for raw_line in goal_str.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("case "):
            continue
        if "⊢" in line:
            target = line.split("⊢", 1)[1].strip()
            continue
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        names = [part for part in lhs.strip().split() if part]
        binder_type = rhs.strip()
        if not names or not binder_type:
            continue
        if len(names) == 1 and _is_instance_like_name(names[0]):
            binders.append(f"[{_fresh_name('_inst')} : {binder_type}]")
        else:
            safe_names = [_fresh_name(_sanitize_goal_binder_name(name)) for name in names]
            binders.append(f"({' '.join(safe_names)} : {binder_type})")

    if not target:
        return ""
    if not binders:
        return target
    return f"∀ {' '.join(binders)}, {target}"


def _candidate_open_namespaces(theorem_name: str = "", file_path: str = "") -> list[str]:
    """Build conservative namespace-open candidates for theorem-type fallback."""
    candidates: list[str] = []

    if theorem_name:
        parts = [part for part in theorem_name.split(".") if part]
        if len(parts) >= 2:
            top = parts[0]
            if top and top not in candidates:
                candidates.append(top)

    if file_path:
        normalized = file_path.replace("\\", "/")
        if "/Mathlib/" in normalized:
            normalized = normalized.split("/Mathlib/", 1)[1]
        elif normalized.startswith("Mathlib/"):
            normalized = normalized[len("Mathlib/") :]
        if normalized.endswith(".lean"):
            normalized = normalized[:-5]
        parts = [part for part in normalized.split("/") if part]
        if parts:
            top = parts[0]
            if top and top not in candidates:
                candidates.append(top)

    return candidates


def _stem(name: str) -> str:
    """Strip ✝ suffix: 𝕜✝ → 𝕜, inst✝¹ → inst"""
    idx = name.find("✝")
    return name[:idx] if idx >= 0 else name


def _intro_missing_locals(server: Any, state: Any, expected_str: str) -> Any:
    """Introduce binders to align current state with expected locals.

    Introduces one binder at a time using the expected local name.
    Skips instance names and ✝-variants already present.
    """
    if not state.goals:
        return state

    expected_names = _extract_local_names(expected_str)
    current_names = {v.name for v in state.goals[0].variables if v.name}
    current_stems = {_stem(n) for n in current_names}

    for name in expected_names:
        if name in current_names:
            continue
        if _stem(name) in current_stems:
            continue
        if name.startswith("inst") or "✝" in name:
            continue
        try:
            new_state = server.goal_tactic(state, tactic=f"intro {name}")
            if not new_state.goals:
                break
            state = new_state
            current_names.add(name)
            current_stems.add(_stem(name))
        except Exception:
            break

    return state


def _normalize_namespaces(s: str) -> str:
    """Strip common namespace prefixes for fuzzy goal matching.

    Handles cases where trace uses short names but Pantograph emits
    fully-qualified: Set.Iio vs Iio, Polynomial.derivative vs derivative,
    Module.finrank vs finrank, Filter.Tendsto vs Tendsto, etc.
    """
    # Remove namespace prefixes from qualified identifiers.
    # Pattern: at word boundary, remove Foo.Bar. prefix before an identifier.
    # Conservative: only strip known-safe Lean/Mathlib prefixes.
    result = re.sub(
        r"\b(?:Set|Filter|Polynomial|Module|LinearMap|Submodule|"
        r"SimpleGraph|MeasureTheory|CategoryTheory|"
        r"Function|Finset|Multiset|List|Order|Nat|Int|"
        r"Real|Complex|ENNReal|NNReal|Metric|"
        r"WittVector|AlgebraicGeometry)\.",
        "",
        s,
    )
    return result


def _match_goal(goals: list, expected_str: str) -> tuple[int, str]:
    """Match a goal from a list against an expected goal state string.

    Returns (matched_index, goal_target_str). Returns (-1, "") on no match.
    Tries full match, target-only, then alpha-equivalent (strip ✝).
    """
    if not expected_str or not goals:
        return (-1, "")

    # Tier 1: full goal state match
    for i, goal in enumerate(goals):
        if str(goal) == expected_str:
            return (i, str(goal.target))

    # Tier 2: target-only exact match
    expected_target = ""
    for eline in expected_str.split("\n"):
        if "⊢" in eline:
            expected_target = eline.split("⊢", 1)[1].strip()
            break

    if expected_target:
        for i, goal in enumerate(goals):
            if str(goal.target) == expected_target:
                return (i, str(goal.target))

    # Tier 3: alpha-equivalent (strip ✝)
    norm_expected = _strip_daggers(expected_str)
    for i, goal in enumerate(goals):
        if _strip_daggers(str(goal)) == norm_expected:
            return (i, str(goal.target))

    # Tier 3b: alpha-equivalent target-only
    if expected_target:
        norm_target = _strip_daggers(expected_target)
        for i, goal in enumerate(goals):
            if _strip_daggers(str(goal.target)) == norm_target:
                return (i, str(goal.target))

    # Tier 4: namespace-fuzzy target match.
    # Strip common namespace prefixes that differ between trace and
    # Pantograph output (e.g., Iio vs Set.Iio, derivative vs
    # Polynomial.derivative, finrank vs Module.finrank).
    if expected_target:
        norm_exp = _normalize_namespaces(expected_target)
        for i, goal in enumerate(goals):
            norm_act = _normalize_namespaces(_strip_daggers(str(goal.target)))
            if norm_exp == norm_act:
                return (i, str(goal.target))

    return (-1, "")


def resolve_lean_path(file_path: str, project_root: str) -> str:
    """Resolve LeanDojo file_path to actual .lean file under project."""
    return os.path.join(project_root, ".lake/packages/mathlib", file_path)


def extract_file_header(lean_path: str, theorem_line: int) -> str:
    """Extract open/variable/namespace/section declarations before the theorem.

    DEPRECATED: use extract_active_context() instead for correct scope
    reconstruction. This function is kept for backward compatibility
    with tests but should not be used for Tier B wrappers.
    """
    with open(lean_path) as f:
        lines = f.readlines()
    header_lines = []
    for line in lines[: theorem_line - 1]:
        stripped = line.strip()
        _HDR_KWS = ("open", "variable", "namespace", "section", "end ")
        if any(stripped.startswith(kw) for kw in _HDR_KWS):
            header_lines.append(line.rstrip())
    return "\n".join(header_lines)


@dataclass
class ActiveContext:
    """Reconstructed active context at a source location."""

    prefix_lines: list[str]  # emit before the theorem wrapper
    suffix_lines: list[str]  # emit after (closing ends, reversed)
    inline_lines: list[str] = field(default_factory=list)  # inline ... in before decl


def extract_active_context(lean_path: str, theorem_line: int) -> ActiveContext:
    """Reconstruct the active Lean scope context at a source line.

    This is a thin compatibility wrapper over `src.lean_context_ir`.
    """
    ir = extract_context_ir(lean_path, theorem_line)
    return ActiveContext(
        prefix_lines=ir.prefix_lines,
        suffix_lines=ir.suffix_lines,
        inline_lines=ir.inline_lines,
    )


def _extract_decl_head(thm_decl: str) -> str:
    """Extract everything up to and including ':= by' from a theorem declaration.

    Returns the declaration head suitable for appending tactic lines + sorry.
    """
    match = re.search(r":=\s*by\b", thm_decl)
    if match:
        return thm_decl[: match.end()]
    match = re.search(r"\bby\b", thm_decl)
    if match:
        return thm_decl[: match.end()]
    # Fallback: strip trailing sorry/proof and append := by
    head = thm_decl.rstrip().rstrip("sorry").rstrip()
    if not head.endswith("by"):
        head += " := by"
    return head


def _resolve_source_file_path(
    *,
    project_root: str,
    module: str = "",
    file_path_hint: str = "",
    theorem_full_name: str = "",
) -> str:
    """Resolve a Lean source path from metadata, hints, and local source search."""
    candidates: list[str] = []
    if module:
        module_path = module.replace(".", "/") + ".lean"
        candidates.extend(
            [
                os.path.join(project_root, ".lake", "packages", "mathlib", module_path),
                os.path.join(project_root, module_path),
            ]
        )
    if file_path_hint:
        hint = file_path_hint
        if os.path.isabs(hint):
            candidates.append(hint)
        else:
            candidates.extend(
                [
                    os.path.join(project_root, hint),
                    os.path.join(project_root, ".lake", "packages", "mathlib", hint),
                ]
            )
    short_name = theorem_full_name.split(".")[-1] if theorem_full_name else ""

    def _contains_decl(path: str) -> bool:
        if not theorem_full_name:
            return True
        try:
            with open(path) as handle:
                lines = handle.readlines()
        except OSError:
            return False
        start_line, _ = _find_decl_bounds_in_source(
            lines,
            short_name,
            theorem_full_name=theorem_full_name,
        )
        return start_line > 0

    seen: set[str] = set()
    first_existing = ""
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            if not first_existing:
                first_existing = normalized
            if _contains_decl(normalized):
                return normalized

    search_roots: list[str] = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        candidate_dir = os.path.dirname(normalized)
        if candidate_dir and os.path.isdir(candidate_dir):
            search_roots.append(candidate_dir)
        stem_dir = os.path.splitext(normalized)[0]
        if stem_dir and os.path.isdir(stem_dir):
            search_roots.append(stem_dir)

    if theorem_full_name:
        global_roots = [
            os.path.join(project_root, ".lake", "packages", "mathlib", "Mathlib"),
            os.path.join(project_root, "Mathlib"),
        ]
        for root in global_roots:
            if os.path.isdir(root):
                search_roots.append(root)

    seen_dirs: set[str] = set()
    for root in search_roots:
        normalized_root = os.path.normpath(root)
        if normalized_root in seen_dirs:
            continue
        seen_dirs.add(normalized_root)
        for dirpath, _dirnames, filenames in os.walk(normalized_root):
            for filename in filenames:
                if not filename.endswith(".lean"):
                    continue
                candidate = os.path.join(dirpath, filename)
                if _contains_decl(candidate):
                    return candidate

    return first_existing


def _find_decl_bounds_in_source(
    all_lines: list[str],
    short_name: str,
    theorem_full_name: str = "",
) -> tuple[int, int]:
    """Best-effort source fallback when env_inspect cannot resolve a theorem.

    Returns 1-based (start_line, end_line). `(0, 0)` means not found.
    """
    if not all_lines or not short_name:
        return (0, 0)
    candidate_names: list[str] = []
    if theorem_full_name:
        parts = [part for part in theorem_full_name.split(".") if part]
        for i in range(len(parts)):
            candidate = ".".join(parts[i:])
            if candidate and candidate not in candidate_names:
                candidate_names.append(candidate)
    if short_name and short_name not in candidate_names:
        candidate_names.append(short_name)
    candidate_priority = {candidate: len(candidate) for candidate in candidate_names}
    normalized_priority = {
        re.sub(r"[^A-Za-z0-9]", "", candidate).lower(): len(candidate)
        for candidate in candidate_names
    }
    decl_re = re.compile(
        rf"^\s*(?:@\[[^\]]+\]\s*)*{_DECL_MODIFIERS}"
        r"(?:theorem|lemma|def|instance|abbrev|alias)\s+([^\s\(:\[]+)"
    )
    top_re = re.compile(
        rf"^\s*(?:@\[[^\]]+\]\s*)*{_DECL_MODIFIERS}"
        r"(?:theorem|lemma|def|instance|class|structure|inductive|abbrev|example|alias)\b"
    )
    start_idx = -1
    best_priority = -1
    for idx, line in enumerate(all_lines):
        match = decl_re.match(line)
        if not match:
            continue
        decl_name = match.group(1)
        normalized_decl_name = re.sub(r"[^A-Za-z0-9]", "", decl_name).lower()
        priority = candidate_priority.get(decl_name)
        if priority is None:
            priority = normalized_priority.get(normalized_decl_name, -1)
        if priority <= 0:
            continue
        if "." in decl_name and priority >= best_priority:
            start_idx = idx
            best_priority = priority
            break
        if priority > best_priority:
            start_idx = idx
            best_priority = priority
    if start_idx < 0:
        return (0, 0)
    start_indent = len(all_lines[start_idx]) - len(all_lines[start_idx].lstrip())
    end_idx = len(all_lines)
    for idx in range(start_idx + 1, len(all_lines)):
        line = all_lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= start_indent and (
            top_re.match(line) or stripped == "end" or stripped.startswith("end ")
        ):
            end_idx = idx
            break
    return (start_idx + 1, end_idx)


def _resolve_alias_target_in_source(
    all_lines: list[str],
    short_name: str,
    theorem_full_name: str = "",
) -> str:
    """Resolve a theorem alias to its target declaration from source text."""
    if not all_lines or not short_name:
        return ""
    candidate_names: list[str] = []
    if theorem_full_name:
        parts = [part for part in theorem_full_name.split(".") if part]
        for i in range(len(parts)):
            candidate = ".".join(parts[i:])
            if candidate and candidate not in candidate_names:
                candidate_names.append(candidate)
    if short_name and short_name not in candidate_names:
        candidate_names.append(short_name)
    normalized_candidates = {
        re.sub(r"[^A-Za-z0-9]", "", candidate).lower() for candidate in candidate_names
    }
    alias_re = re.compile(r"\balias\s+([^\s:=]+)\s*:=\s*([^\s]+)")
    for idx, line in enumerate(all_lines):
        if "alias" not in line:
            continue
        block_parts = [line.strip()]
        j = idx + 1
        while j < len(all_lines) and ":=" in " ".join(block_parts) and block_parts[-1].endswith(":="):
            nxt = all_lines[j].strip()
            if not nxt:
                break
            if re.match(
                r"^(?:theorem|lemma|def|instance|class|structure|inductive|abbrev|alias|example|section|namespace|end\b)",
                nxt,
            ):
                break
            block_parts.append(nxt)
            j += 1
        block = " ".join(block_parts)
        match = alias_re.search(block)
        if not match:
            continue
        alias_name = match.group(1)
        normalized_alias = re.sub(r"[^A-Za-z0-9]", "", alias_name).lower()
        if alias_name not in candidate_names and normalized_alias not in normalized_candidates:
            continue
        target = match.group(2).rstrip(",")
        if theorem_full_name and "." not in target and "." in theorem_full_name:
            prefix = theorem_full_name.rsplit(".", 1)[0]
            return f"{prefix}.{target}"
        return target
    return ""


class LeanKernel:
    """Interface to the Lean 4 kernel for tactic verification.

    Args:
        config: Connection and timeout configuration.
    """

    def __init__(self, config: LeanConfig | None = None) -> None:
        self.config = config or LeanConfig()
        self._backend = self.config.backend
        # Replay backend: stores ground-truth tactics per goal for offline eval
        self._replay_table: dict[str, list[str]] = {}
        # Pantograph backend: lazy-initialized server + goal state tracking
        self._server: Any | None = None
        self._goal_states: dict[tuple[str, str], Any] = {}  # (env_key, goal_str)
        # Tactic result cache: (env_key, goal_state, tactic) → TacticResult
        # Pure function — same env + goal + tactic = same result.
        # Eliminates redundant Lean IPC calls across search iterations.
        # key: (env_key, goal_str, tactic, goal_id)
        self._tactic_cache: dict[tuple[str, str, str, int], TacticResult] = {}
        self._current_env_key: str = ""  # default empty for backward compat
        self._server_contaminated: bool = False  # set after load_sorry-backed goal creation
        self._load_sorry_reset_count: int = 0
        self._last_goal_feedback: LeanFeedback | None = None
        # Prefix tactic qualification: resolve short names to qualified forms
        self._suffix_index: dict[str, list[str]] = {}
        self._accessible_names: set[str] = set()

    def close(self) -> None:
        """Shut down the Pantograph server if running."""
        if self._server is not None:
            try:
                self._server._close()
            except Exception:
                pass
            self._server = None
            self._goal_states.clear()
            self._tactic_cache.clear()

    def _prepare_for_new_example(self) -> None:
        """Reset per-example state after load_sorry without restarting every time.

        `frontend.distil` goal states are GC-able. Reuse the hot server and
        clear cached states first; fall back to periodic restarts as a safety
        valve against long-lived memory growth.
        """
        if not self._server_contaminated:
            return
        self.gc()
        self._server_contaminated = False
        self._load_sorry_reset_count += 1
        restart_every = 64
        if self._load_sorry_reset_count % restart_every == 0:
            logger.info(
                "Periodic Pantograph restart after %d load_sorry resets",
                self._load_sorry_reset_count,
            )
            self._restart_server()

    def register_ground_truth(self, goal_state: str, tactics: list[str]) -> None:
        """Register ground-truth tactics for replay backend.

        Args:
            goal_state: The goal state text.
            tactics: List of tactic strings that close this goal.
        """
        self._replay_table[goal_state] = tactics

    def try_tactic(
        self,
        goal_state: str,
        tactic: str,
        goal_id: int = 0,
    ) -> TacticResult:
        """Send a tactic to the Lean kernel and get the result.

        Args:
            goal_state: Current goal state text.
            tactic: Tactic string to apply.
            goal_id: Index into GoalState.goals for Site addressing.
                     Default 0 targets the first (or only) goal.

        Returns:
            TacticResult with success/failure, new goals, and error info.
        """
        if self._backend == "stub":
            return self._stub_try_tactic(goal_state, tactic)
        if self._backend == "replay":
            return self._replay_try_tactic(goal_state, tactic)
        if self._backend == "pantograph":
            return self._pantograph_try_tactic(goal_state, tactic, goal_id)
        msg = f"Unknown backend: {self._backend}"
        raise ValueError(msg)

    def try_hammer(
        self, goal_state: str, premises: list[str], timeout: int | None = None
    ) -> TacticResult:
        """Delegate to LeanHammer/Aesop with premise suggestions.

        Args:
            goal_state: Current goal state text.
            premises: Suggested premise names for the hammer.
            timeout: Override default hammer timeout.
        """
        t = timeout or self.config.hammer_timeout
        if self._backend == "stub":
            return self._stub_try_hammer(goal_state, premises, t)
        if self._backend == "replay":
            return self._replay_try_hammer(goal_state, premises, t)
        if self._backend == "pantograph":
            return self._pantograph_try_hammer(goal_state, premises, t)
        msg = f"Unknown backend: {self._backend}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Stub backend (Phase 1)
    # ------------------------------------------------------------------

    def _stub_try_tactic(self, _goal_state: str, tactic: str) -> TacticResult:
        """Stub: always fails. For development/testing without Lean."""
        return TacticResult(
            success=False,
            tactic=tactic,
            premises=[],
            new_goals=[],
            error_message="stub backend: no Lean kernel connected",
        )

    def _stub_try_hammer(
        self, _goal_state: str, premises: list[str], _timeout: int
    ) -> TacticResult:
        """Stub: always fails."""
        return TacticResult(
            success=False,
            tactic=f"aesop (premises: {len(premises)})",
            premises=premises,
            new_goals=[],
            error_message="stub backend: no Lean kernel connected",
        )

    # ------------------------------------------------------------------
    # Replay backend (offline evaluation)
    # ------------------------------------------------------------------

    def _replay_try_tactic(self, goal_state: str, tactic: str) -> TacticResult:
        """Replay: succeed if tactic matches any registered ground-truth tactic.

        Matching is by base tactic name (first word), allowing the navigator
        to propose the right tactic even with different premise arguments.
        """
        gt_tactics = self._replay_table.get(goal_state, [])
        tactic_base = tactic.split()[0] if tactic.strip() else ""

        for gt in gt_tactics:
            gt_base = gt.split()[0] if gt.strip() else ""
            if tactic_base and gt_base and tactic_base == gt_base:
                return TacticResult(
                    success=True,
                    tactic=tactic,
                    premises=[],
                    new_goals=[],
                )

        return TacticResult(
            success=False,
            tactic=tactic,
            premises=[],
            new_goals=[],
            error_message=f"replay: tactic '{tactic_base}' not in ground truth",
        )

    def _replay_try_hammer(
        self, _goal_state: str, premises: list[str], _timeout: int
    ) -> TacticResult:
        """Replay: hammer always fails (only exact tactic matching supported)."""
        return TacticResult(
            success=False,
            tactic=f"aesop (premises: {len(premises)})",
            premises=premises,
            new_goals=[],
            error_message="replay backend: hammer not supported in replay mode",
        )

    # ------------------------------------------------------------------
    # Pantograph backend (Lane A — real Lean verification)
    # ------------------------------------------------------------------

    def _ensure_server(self) -> Any:
        """Lazy-initialize the Pantograph server."""
        if self._server is not None:
            return self._server

        try:
            from pantograph.server import Server  # type: ignore[import-untyped]
        except ImportError as e:
            msg = (
                "PyPantograph is not installed. Install with: "
                "pip install git+https://github.com/stanford-centaur/PyPantograph"
            )
            raise ImportError(msg) from e

        project = self.config.project_root or None
        self._server = Server(
            imports=self.config.imports,
            project_path=project,
            timeout=self.config.timeout,
        )
        logger.info(
            "Pantograph server started (project=%s, imports=%s)",
            project,
            self.config.imports,
        )
        return self._server

    def _get_or_create_goal(self, goal_state: str) -> Any:
        """Get a cached GoalState or create one from a type expression.

        Falls back to load_sorry for universe-polymorphic types that
        goal_start cannot parse (e.g., ``Type u_1``).

        Raises ServerCrashError on unrecoverable server crashes so callers
        can distinguish infra failures from tactic failures.
        """
        cache_key = (self._current_env_key, goal_state)
        if cache_key in self._goal_states:
            return self._goal_states[cache_key]

        server = self._ensure_server()
        try:
            state = self._quiet_goal_start(server, goal_state)
        except Exception as e:
            if self._is_crash_error(e):
                self._restart_server()
                raise ServerCrashError(str(e)) from e
            # Fallback: wrap in a sorry'd theorem declaration.
            # This handles universe-polymorphic types that goal_start rejects.
            state = self._goal_via_sorry(server, goal_state)

        self._goal_states[cache_key] = state
        return state

    @staticmethod
    def _extract_universe_vars(text: str) -> list[str]:
        """Extract universe variable names from a type expression."""
        import re

        # Match u_1, u_2, ..., u, v, w (common single-letter universe names)
        uvars: set[str] = set()
        # Numbered: u_1, u_2, ...
        for m in re.finditer(r"\bu_(\d+)\b", text):
            uvars.add(f"u_{m.group(1)}")
        # Single-letter: Type u, Sort v (but not common identifiers)
        for m in re.finditer(r"(?:Type|Sort)\s+([a-z])\b", text):
            uvars.add(m.group(1))
        # max u v patterns
        for m in re.finditer(r"\bmax\s+([a-z](?:_\d+)?)\s+([a-z](?:_\d+)?)", text):
            uvars.add(m.group(1))
            uvars.add(m.group(2))
        return sorted(uvars)

    @staticmethod
    def _build_universe_prelude(uvars: list[str]) -> str:
        """Build a `universe u_1 u_2 ...` declaration."""
        if not uvars:
            return ""
        return f"universe {' '.join(uvars)}\n"

    def _goal_via_sorry(
        self,
        server: Any,
        goal_type: str,
        theorem_name: str = "",
        file_path: str = "",
    ) -> Any:
        """Create a GoalState via load_sorry — deterministic compiler cascade.

        Cascade (ordered by faithfulness):
        B2: theorem-type shell + explicit universe prelude (preserves polymorphism)
        B3: theorem-type shell + universe prelude + namespace opens
        C:  universe-erased fallback (least faithful, salvage only)
        """
        import re

        name = f"_wayfinder_goal_{len(self._goal_states)}"

        # Extract universe variables from the original type
        uvars = self._extract_universe_vars(goal_type)
        universe_prelude = self._build_universe_prelude(uvars)

        # Build conservative namespace open variants. Aggressive namespace
        # guesses from theorem ids can poison elaboration on stale metadata.
        open_prefixes: list[str] = [""]
        for namespace in _candidate_open_namespaces(theorem_name=theorem_name, file_path=file_path):
            prefix = f"open {namespace} in\n"
            if prefix not in open_prefixes:
                open_prefixes.insert(0, prefix)

        # --- Tier B2: explicit universe prelude + original type ---
        if uvars:
            for open_prefix in open_prefixes:
                src = (
                    f"section _wayfinder_goalstart\n"
                    f"{universe_prelude}"
                    f"{open_prefix}"
                    f"theorem {name} : {goal_type} := by\n  sorry\n"
                    f"end _wayfinder_goalstart"
                )
                try:
                    targets = server.load_sorry(src)
                    self._server_contaminated = True
                    if targets and targets[0].goal_state.goals:
                        return targets[0].goal_state
                except Exception as e:
                    self._server_contaminated = True
                    self._last_goal_feedback = _classify_compiler_feedback(
                        e,
                        stage="goal_creation",
                        fallback_category="unbound_universe",
                    )
                    if self._is_crash_error(e):
                        self._restart_server()
                        server = self._ensure_server()
                    continue

        # --- Tier B3: universe prelude + namespace opens (without section) ---
        for open_prefix in open_prefixes:
            src = (
                f"{universe_prelude}"
                f"{open_prefix}"
                f"theorem {name} : {goal_type} := by sorry"
            )
            try:
                targets = server.load_sorry(src)
                self._server_contaminated = True
                if targets and targets[0].goal_state.goals:
                    return targets[0].goal_state
            except Exception as e:
                self._server_contaminated = True
                self._last_goal_feedback = _classify_compiler_feedback(
                    e,
                    stage="goal_creation",
                    fallback_category="goal_creation_fail",
                )
                if self._is_crash_error(e):
                    self._restart_server()
                    server = self._ensure_server()
                continue

        # --- Tier C: universe-erased fallback (salvage only) ---
        cleaned = re.sub(r"\bu_\d+\b", "_", goal_type)
        cleaned = re.sub(r"(?<=Type )\bu\b", "_", cleaned)
        cleaned = re.sub(r"(?<=Sort )\bu\b", "_", cleaned)

        for open_prefix in open_prefixes:
            src = f"{open_prefix}theorem {name} : {cleaned} := by sorry"
            try:
                targets = server.load_sorry(src)
                self._server_contaminated = True
                if targets and targets[0].goal_state.goals:
                    return targets[0].goal_state
            except Exception as e:
                self._server_contaminated = True
                self._last_goal_feedback = _classify_compiler_feedback(
                    e,
                    stage="goal_creation",
                    fallback_category="universe_compilation_fail",
                )
                if self._is_crash_error(e):
                    self._restart_server()
                    server = self._ensure_server()
                continue

        logger.debug("load_sorry cascade failed for %s", name)
        return self._quiet_goal_start(server, goal_type)

    def goal_start(self, theorem_type: str, theorem_name: str = "", file_path: str = "") -> str:
        """Initialize a goal from a theorem type expression.

        Returns the goal state string for the initial goal.
        Falls back to load_sorry with namespace-open for universe-polymorphic types.
        """
        server = self._ensure_server()
        try:
            state = self._quiet_goal_start(server, theorem_type)
        except Exception as e:
            self._last_goal_feedback = _classify_compiler_feedback(
                e,
                stage="goal_creation",
                fallback_category="goal_creation_fail",
            )
            state = self._goal_via_sorry(server, theorem_type, theorem_name, file_path)

        goal_str = str(state.goals[0].target) if state.goals else theorem_type
        self._goal_states[(self._current_env_key, goal_str)] = state
        return goal_str

    def goal_start_from_pp(
        self,
        goal_state_pp: str,
        theorem_name: str = "",
        file_path: str = "",
    ) -> str:
        """Initialize a goal from a pretty-printed local-context goal state."""
        theorem_type = theorem_type_from_goal_pp(goal_state_pp)
        if not theorem_type:
            raise ValueError("could not synthesize theorem type from pretty goal state")
        return self.goal_start(theorem_type, theorem_name=theorem_name, file_path=file_path)

    @staticmethod
    def _quiet_goal_start(server: Any, expr: str) -> Any:
        with contextlib.redirect_stdout(io.StringIO()):
            return server.goal_start(expr)

    def _pantograph_try_tactic(
        self,
        goal_state: str,
        tactic: str,
        goal_id: int = 0,
    ) -> TacticResult:
        """Apply a tactic via Pantograph and return the result.

        Args:
            goal_state: Goal target text (used for caching and state lookup).
            tactic: Tactic string to apply.
            goal_id: Index into GoalState.goals for Site addressing.
                     Needed for multi-goal states after branching replay.

        Results are cached by (env_key, goal_state, tactic, goal_id).
        goal_id is part of the key because the same tactic on the same
        target text can produce different results when applied to
        different subgoals in a multi-goal state.
        """
        cache_key = (self._current_env_key, goal_state, tactic, goal_id)
        cached = self._tactic_cache.get(cache_key)
        if cached is not None:
            return cached

        from pantograph.expr import Site  # type: ignore[import-untyped]
        from pantograph.server import TacticFailure  # type: ignore[import-untyped]

        try:
            state = self._get_or_create_goal(goal_state)
            server = self._ensure_server()
            if goal_id > 0:
                site = Site(goal_id=goal_id)
                new_state = server.goal_tactic(state, tactic=tactic, site=site)
            else:
                new_state = server.goal_tactic(state, tactic=tactic)
        except TacticFailure as e:
            fail_result = TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message=str(e),
                feedback=_classify_tactic_failure(e),
            )
            self._tactic_cache[cache_key] = fail_result
            return fail_result
        except Exception as e:
            if self._is_crash_error(e):
                self._restart_server()
                raise ServerCrashError(str(e)) from e
            fail_result = TacticResult(
                success=False,
                tactic=tactic,
                premises=[],
                new_goals=[],
                error_message=f"pantograph error: {e}",
                feedback=LeanFeedback(
                    stage="tactic_exec",
                    category="other",
                    messages=[],
                    raw_error=f"pantograph error: {e}",
                ),
            )
            self._tactic_cache[cache_key] = fail_result
            return fail_result

        new_goals = [str(g.target) for g in new_state.goals]

        # Cache each new goal state for subsequent tactic applications
        for goal in new_state.goals:
            goal_str = str(goal.target)
            gs_key = (self._current_env_key, goal_str)
            if gs_key not in self._goal_states:
                self._goal_states[gs_key] = new_state

        # Success = Pantograph accepted the tactic (no TacticFailure raised).
        # is_solved / no goals → closed all; new_goals non-empty → structural
        # progress (intro, cases, etc.). Feedback encodes accepted_with_goals
        # vs none so downstream logic need not re-derive from new_goals.
        goals_closed = new_state.is_solved or not new_state.goals
        ok_result = TacticResult(
            success=goals_closed or new_goals != [goal_state],
            tactic=tactic,
            premises=[],
            new_goals=new_goals,
            feedback=LeanFeedback(
                stage="tactic_exec",
                category="none" if goals_closed else "accepted_with_goals",
                messages=[],
                raw_error="",
            ),
        )
        self._tactic_cache[cache_key] = ok_result
        return ok_result

    def _pantograph_try_hammer(
        self, goal_state: str, premises: list[str], timeout: int
    ) -> TacticResult:
        """Try hammer tactics (aesop, omega, simp, decide) with premises."""
        hammer_tactics = _build_hammer_tactics(premises)
        for hammer_tactic in hammer_tactics:
            result = self._pantograph_try_tactic(goal_state, hammer_tactic)
            if result.success:
                return TacticResult(
                    success=True,
                    tactic=hammer_tactic,
                    premises=premises,
                    new_goals=result.new_goals,
                    feedback=result.feedback,
                )
        return TacticResult(
            success=False,
            tactic=f"hammer ({len(hammer_tactics)} tactics tried)",
            premises=premises,
            new_goals=[],
            error_message="all hammer tactics failed",
        )

    def _is_crash_error(self, e: Exception) -> bool:
        """Check if an exception indicates a Pantograph server crash or corruption.

        Heartbeat timeouts at ``whnf`` leave the server in a corrupted state
        where subsequent ``load_sorry`` calls fail silently.  Treating these
        as crash errors forces an immediate restart rather than waiting for
        the periodic restart interval.

        KeyError from PyPantograph (e.g. ``'targets'``, ``'stateId'``) means
        the server returned a malformed response — the process is alive but
        state is corrupt.  Restart immediately rather than failing every
        subsequent theorem until the periodic restart interval.
        """
        if isinstance(e, (BrokenPipeError, ConnectionResetError, ProcessLookupError,
                          KeyError)):
            return True
        msg = str(e).lower()
        _CRASH_STRINGS = (
            "pipe closed", "broken pipe", "connection reset", "connection lost",
            "deterministic) timeout", "maximum number of heartbeats",
        )
        return any(s in msg for s in _CRASH_STRINGS)

    def _restart_server(self) -> None:
        """Restart the Pantograph server after a crash or contamination."""
        logger.warning("Restarting Pantograph server")
        if self._server is not None:
            try:
                self._server._close()  # terminates the subprocess
            except Exception:
                pass
            self._server = None
        self._goal_states.clear()
        self._tactic_cache.clear()
        self._server_contaminated = False
        self._ensure_server()

    @staticmethod
    def _extract_support_slice(
        all_lines: list[str],
        theorem_line: int,
        theorem_end_line: int,
        theorem_short_name: str,
    ) -> list[str]:
        """Extract supporting declarations needed for theorem elaboration.

        Scans upward from the theorem within the enclosing section/namespace,
        collecting: instance declarations, local instances, defs/abbrevs,
        scoped notation, local attributes, and include/omit directives.

        Returns renamed declarations (prefixed with _wf_support_) to avoid
        collision with already-imported Mathlib names.
        """
        import re

        pre_lines = all_lines[: theorem_line - 1]
        support: list[str] = []

        # Patterns to capture
        _SUPPORT_RE = re.compile(
            r"^\s*(?:"
            + _DECL_MODIFIERS + r"(?:def|abbrev|instance)\b"
            r"|"
            + _DECL_MODIFIERS + r"(?:theorem|lemma)\b"
            r"|local\s+instance\b"
            r"|local\s+macro(?::[A-Za-z_][A-Za-z0-9_]*)?\b"
            r"|local\s+notation\b"
            r"|scoped\s+(?:notation|prefix|infix|postfix|macro)\b"
            r"|attribute\s+\[.*?(?:local|instance).*?\]"
            r"|local\s+attribute\b"
            r"|include\s+"
            r"|omit\s+"
            r"|(?:letI|haveI)\b"
            r")"
        )

        # Find enclosing section/namespace boundary
        section_start = 0
        for i in range(len(pre_lines) - 1, -1, -1):
            stripped = pre_lines[i].strip()
            if stripped.startswith("section ") or stripped.startswith("namespace "):
                section_start = i
                break

        # Collect support declarations from section_start to theorem_line
        i = section_start
        while i < len(pre_lines):
            line = pre_lines[i]
            if _SUPPORT_RE.match(line):
                # Collect the full declaration (may span multiple lines)
                decl_lines = [line]
                # Declarations end at the next same-or-lesser-indent top-level
                # keyword. We must NOT cut on blank lines inside `where` blocks
                # or `⟨...⟩` expressions.
                indent = len(line) - len(line.lstrip())
                j = i + 1
                in_block = "where" in line or line.rstrip().endswith("⟨") or ":=" in line
                while j < len(pre_lines):
                    next_line = pre_lines[j]
                    next_stripped = next_line.strip()
                    next_indent = len(next_line) - len(next_line.lstrip())

                    # Blank line ends only if we're not in a block
                    if not next_stripped:
                        if not in_block:
                            break
                        # In a block: blank line is OK, but two consecutive blanks end it
                        if j + 1 < len(pre_lines) and not pre_lines[j + 1].strip():
                            break
                        decl_lines.append(next_line)
                        j += 1
                        continue

                    # Continuation lines inside the block
                    if next_indent > indent:
                        decl_lines.append(next_line)
                        in_block = in_block or "where" in next_line
                        j += 1
                        continue

                    # Same indent: new declaration or keyword → stop
                    if _SUPPORT_RE.match(next_line) or re.match(
                        r"\s*(?:theorem|lemma|section|namespace|end |set_option\b|open\b|alias\b|/--|--)",
                        next_line,
                    ):
                        break
                    if re.match(
                        r"\s*(?:attribute\b|local\s+attribute\b|scoped\b|notation\b|local\s+macro\b)",
                        next_line,
                    ):
                        break
                    # Same indent continuation (e.g., `| ...` patterns)
                    decl_lines.append(next_line)
                    j += 1

                decl_text = "".join(decl_lines)

                # Do NOT rename support declarations — instances need their
                # original names for typeclass resolution. Collisions with
                # imported Mathlib are handled by Lean's shadowing rules.
                support.append(decl_text.rstrip())
                i = j
            else:
                i += 1

        return support

    @staticmethod
    def _decl_name_from_text(text: str) -> str:
        match = re.search(
            rf"^\s*(?:@\[[^\]]+\]\s*)*{_DECL_MODIFIERS}"
            r"(?:theorem|lemma|def|instance|abbrev|alias)\s+([^\s\(:\[]+)",
            text,
            flags=re.MULTILINE,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _rewrite_identifier(text: str, old: str, new: str) -> str:
        ident_chars = r"[A-Za-z0-9_']"
        return re.sub(
            rf"(?<!{ident_chars}){re.escape(old)}(?!{ident_chars})",
            new,
            text,
        )

    @classmethod
    def _rename_support_declarations(
        cls,
        support_lines: list[str],
        theorem_decl: str,
    ) -> tuple[list[str], str]:
        """Rename local support declarations to avoid collision with imports."""
        mapping: dict[str, str] = {}
        for idx, support_decl in enumerate(support_lines):
            decl_name = cls._decl_name_from_text(support_decl)
            if not decl_name:
                continue
            sanitized = re.sub(r"[^A-Za-z0-9_']+", "_", decl_name).strip("_") or "decl"
            mapping[decl_name] = f"_wf_support_{idx}_{sanitized}"

        if not mapping:
            return support_lines, theorem_decl

        ordered_mapping = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)

        def _apply(text: str) -> str:
            rewritten = text
            for old, new in ordered_mapping:
                rewritten = cls._rewrite_identifier(rewritten, old, new)
            return rewritten

        return [_apply(support_decl) for support_decl in support_lines], _apply(theorem_decl)

    @staticmethod
    def _wrap_replay_namespace(source: str, seed: str) -> str:
        """Wrap replay source in a fresh namespace after header commands.

        This avoids collisions when replaying local support declarations that
        share names with already-imported Mathlib declarations.
        """
        lines = source.splitlines()
        if not lines:
            return source

        header: list[str] = []
        body_start = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("--")
                or stripped.startswith("/-")
                or stripped == "-/"
                or stripped.startswith("module")
                or stripped.startswith("import ")
            ):
                header.append(line)
                body_start = idx + 1
                continue
            body_start = idx
            break
        body = lines[body_start:]
        if not body:
            return source

        ns = f"_wayfinder_replay_{seed}"
        wrapped = header + [f"namespace {ns}"] + body + [f"end {ns}"]
        result = "\n".join(wrapped)
        if source.endswith("\n"):
            result += "\n"
        return result

    def _tier_b_file_wrapper(
        self,
        theorem_full_name: str,
        file_path: str,
        module_hint: str,
        thm_type: str | None,
        project_root: str,
        alias_depth: int = 0,
    ) -> Any | None:
        """Tier B: declaration-faithful start state via source + load_sorry.

        Uses env_inspect to get the declaring module and source coordinates,
        reads the original theorem declaration from the .lean source file,
        renames it to avoid collision, replaces the proof body with `by sorry`,
        wraps it in the active scope context, and loads via load_sorry.

        This produces the same post-`by` tactic state that LeanDojo traces
        start from — with binders introduced and local names preserved.
        """
        info: dict[str, Any] | None = None
        try:
            server = self._ensure_server()
            inspected = server.env_inspect(name=theorem_full_name)
            if isinstance(inspected, dict):
                info = inspected
        except Exception as e:
            if self._is_crash_error(e):
                self._restart_server()
            self._last_goal_feedback = _classify_compiler_feedback(
                e,
                stage="goal_creation",
                fallback_category="goal_creation_fail",
            )

        module = info.get("module", "") if isinstance(info, dict) else ""
        if not module and module_hint:
            module = module_hint
        src_start = info.get("sourceStart") if isinstance(info, dict) else None
        src_end = info.get("sourceEnd") if isinstance(info, dict) else None
        lean_path = _resolve_source_file_path(
            project_root=project_root,
            module=module,
            file_path_hint=file_path,
            theorem_full_name=theorem_full_name,
        )
        if not lean_path:
            return None

        start_line = src_start.get("line", 0) if isinstance(src_start, dict) else 0
        end_line = src_end.get("line", 0) if isinstance(src_end, dict) else 0

        try:
            with open(lean_path) as f:
                all_lines = f.readlines()
        except Exception:
            return None

        short_name = theorem_full_name.split(".")[-1]
        alias_target = _resolve_alias_target_in_source(
            all_lines,
            short_name,
            theorem_full_name=theorem_full_name,
        )
        if alias_target and alias_target != theorem_full_name and alias_depth < 4:
            logger.debug("Tier B alias redirect %s -> %s", theorem_full_name, alias_target)
            return self._tier_b_file_wrapper(
                alias_target,
                file_path,
                module_hint,
                thm_type,
                project_root,
                alias_depth=alias_depth + 1,
            )

        if start_line <= 0 or end_line <= 0:
            start_line, end_line = _find_decl_bounds_in_source(
                all_lines,
                short_name,
                theorem_full_name=theorem_full_name,
            )
        if start_line <= 0 or end_line <= 0 or end_line < start_line:
            return None

        # Extract the original declaration text
        decl_text = "".join(all_lines[start_line - 1 : end_line])
        if not decl_text.strip():
            return None

        # Rename declaration to avoid "already declared"
        fresh_name = f"_wayfinder_decl_{len(self._goal_states)}"
        renamed = decl_text
        decl_name_match = re.search(
            rf"^(\s*(?:@\[[^\]]+\]\s*)*{_DECL_MODIFIERS}"
            r"(?:theorem|lemma|def|instance)\s+)([^\s\(:\[]+)",
            renamed,
            flags=re.MULTILINE,
        )
        if decl_name_match:
            renamed = (
                renamed[: decl_name_match.start(2)]
                + fresh_name
                + renamed[decl_name_match.end(2) :]
            )

        # Replace proof body with `by sorry`
        # Find `:= by` or `:= by\n` and truncate
        by_match = re.search(r":=\s*by\b", renamed)
        if by_match:
            decl_head = renamed[: by_match.end()]
        else:
            # Non-tactic proof or where clause — try `:=` then append `by`
            eq_match = re.search(r":=", renamed)
            if eq_match:
                decl_head = renamed[: eq_match.end()] + " by"
            else:
                # No `:=` found — append
                decl_head = renamed.rstrip() + " := by"
        sorry_decl = decl_head + "\n  sorry"

        # Strip docstrings/attributes from the declaration head
        # (they appear before the keyword and can cause parse issues)
        sorry_decl = re.sub(r"/--.*?-/\s*", "", sorry_decl, flags=re.DOTALL)
        sorry_decl = re.sub(r"@\[.*?\]\s*", "", sorry_decl)

        # Reconstruct active scope context at declaration site
        ctx = extract_active_context(lean_path, start_line)

        # Build wrapper: prefix + inline modifiers + sorry'd decl + suffix.
        # Avoid a synthetic replay namespace here: it shadows real Mathlib names
        # like `WeierstrassCurve.Jacobian` and `QuadraticMap.polarBilin`, which
        # turns theorem-start into a self-inflicted elaboration failure.
        parts = list(ctx.prefix_lines) + list(ctx.inline_lines) + [sorry_decl] + list(
            ctx.suffix_lines
        )
        wrapper = "\n".join(parts)

        try:
            server = self._ensure_server()
            targets = server.load_sorry(wrapper)
            self._server_contaminated = True
            if targets and targets[0].goal_state.goals:
                return targets[0].goal_state
        except Exception as e:
            self._server_contaminated = True
            self._last_goal_feedback = _classify_compiler_feedback(
                e,
                stage="goal_creation",
                fallback_category="goal_creation_fail",
            )
            if self._is_crash_error(e):
                self._restart_server()
                self._server_contaminated = False
            logger.debug("Tier B decl wrapper failed for %s: %s", theorem_full_name, e)

        # --- Tier B1.5: support-slice replay ---
        # Instead of replaying the whole file, extract only the supporting
        # declarations (instances, defs, abbrevs, scoped notation) from the
        # same section/namespace that the theorem needs for elaboration.
        try:
            server = self._ensure_server()
            support_lines = self._extract_support_slice(
                all_lines, start_line, end_line, short_name
            )
            if support_lines:
                renamed_support_lines, renamed_sorry_decl = self._rename_support_declarations(
                    support_lines,
                    sorry_decl,
                )
                ctx = extract_active_context(lean_path, start_line)
                parts = (
                    list(ctx.prefix_lines)
                    + renamed_support_lines
                    + list(ctx.inline_lines)
                    + [renamed_sorry_decl]
                    + list(ctx.suffix_lines)
                )
                direct_wrapper = "\n".join(parts)
                try:
                    targets = server.load_sorry(direct_wrapper)
                    self._server_contaminated = True
                    if targets and targets[0].goal_state.goals:
                        logger.debug(
                            "Tier B1.5 direct support-slice succeeded for %s",
                            theorem_full_name,
                        )
                        return targets[0].goal_state
                except Exception as direct_err:
                    self._server_contaminated = True
                    if self._is_crash_error(direct_err):
                        self._restart_server()
                        self._server_contaminated = False
                        raise
                    logger.debug(
                        "Tier B1.5 direct support-slice failed for %s: %s",
                        theorem_full_name,
                        direct_err,
                    )

                wrapper = self._wrap_replay_namespace(direct_wrapper, fresh_name)
                targets = server.load_sorry(wrapper)
                self._server_contaminated = True
                if targets and targets[0].goal_state.goals:
                    logger.debug(
                        "Tier B1.5 wrapped support-slice succeeded for %s",
                        theorem_full_name,
                    )
                    return targets[0].goal_state
        except Exception as e_b15:
            self._server_contaminated = True
            if self._is_crash_error(e_b15):
                self._restart_server()
                self._server_contaminated = False
            logger.debug(
                "Tier B1.5 support-slice failed for %s: %s", theorem_full_name, e_b15
            )

        # --- Tier B4: bounded source replay ---
        # Replay the actual source from file start through the theorem,
        # renaming the theorem and replacing its proof with sorry.
        # This is the most faithful local fallback.
        try:
            server = self._ensure_server()
            # Find the source slice: everything from line 1 to end of theorem
            source_slice = "".join(all_lines[:end_line])

            # Rename the theorem to avoid "already declared"
            # Replace the LAST occurrence of the theorem keyword + short_name
            # (the declaration itself, not references in earlier code)
            rename_pattern = re.compile(
                rf"(theorem|lemma|def)\s+{re.escape(short_name)}\b",
            )
            matches = list(rename_pattern.finditer(source_slice))
            if matches:
                last_match = matches[-1]
                source_slice = (
                    source_slice[:last_match.start()]
                    + source_slice[last_match.start():last_match.end()].replace(
                        short_name, fresh_name
                    )
                    + source_slice[last_match.end():]
                )

            # Replace proof body with sorry
            by_match = re.search(r":=\s*by\b", source_slice[start_line - 1:])
            if by_match:
                # Count characters, not line offset
                char_offset = sum(len(line_text) for line_text in all_lines[:start_line - 1])
                by_match_full = re.search(r":=\s*by\b", source_slice[char_offset:])
                if by_match_full:
                    cut_char = char_offset + by_match_full.end()
                    source_slice = source_slice[:cut_char] + "\n  sorry"

            # Strip docstrings/attributes from the theorem declaration only
            # (keep file-level ones intact)
            source_slice = self._wrap_replay_namespace(source_slice, fresh_name)
            targets = server.load_sorry(source_slice)
            self._server_contaminated = True
            if targets and targets[0].goal_state.goals:
                logger.debug("Tier B4 source replay succeeded for %s", theorem_full_name)
                return targets[0].goal_state
        except Exception as e_b4:
            self._server_contaminated = True
            if self._is_crash_error(e_b4):
                self._restart_server()
                self._server_contaminated = False
            logger.debug("Tier B4 source replay failed for %s: %s", theorem_full_name, e_b4)

        return None

    def goal_via_file_context(
        self,
        theorem_full_name: str,
        file_path: str,
        prefix_tactics: list[str],
        expected_goal: str = "",
        project_root: str = "",
        module_hint: str = "",
        fallback_goal_pp: str = "",
        accessible_names: set[str] | None = None,
        prefix_goal_states: list[str] | None = None,
    ) -> ReplayResult:
        """Create a goal state using tiered strategy.

        Cascade order: B → A → C.
        - Tier B (primary): declaration-faithful load_sorry from source.
          Produces the post-`by` tactic state matching LeanDojo traces.
        - Tier A (fallback): env_inspect → goal_start. Fast but produces
          the raw theorem type, not the post-`by` state.
        - Tier C (step>0): sequential goal_tactic replay of prefix_tactics
          on the B or A base state.

        Args:
            theorem_full_name: Fully qualified theorem name.
            file_path: LeanDojo-style relative path (used as hint only;
                       actual path resolved via env_inspect module).
            prefix_tactics: Tactics to replay before the target step.
            expected_goal: Expected goal state text for matching.
            project_root: Root of Lean project (for file resolution).
        """
        crash_retries = 0
        prefix_hash = hashlib.md5("|".join(prefix_tactics).encode()).hexdigest()[:8]
        env_key = f"{file_path}:{theorem_full_name}:{prefix_hash}"
        self._current_env_key = env_key
        self._last_goal_feedback = None

        # Previous examples may have created load_sorry goal states. Those do
        # not require a full process restart; GC the old states and clear local
        # caches so the server can stay hot across examples.
        self._prepare_for_new_example()

        # ----------------------------------------------------------
        # Tier B (primary): declaration-faithful start state.
        # Uses env_inspect module + sourceStart/End to find the real
        # source declaration, renames it, replaces proof with sorry,
        # wraps in active scope context, loads via load_sorry.
        # ----------------------------------------------------------
        base_state = None
        tier_used = ""
        last_attempted_tier = "B"

        state = self._tier_b_file_wrapper(
            theorem_full_name,
            file_path,
            module_hint,
            None,
            project_root or self.config.project_root,
        )
        if state is not None:
            base_state = state
            tier_used = "B"

        # ----------------------------------------------------------
        # Tier A (fallback): env_inspect → goal_start.
        # Only tried when Tier B fails. Produces the raw theorem type,
        # not the post-`by` state — less faithful but broader coverage.
        # ----------------------------------------------------------
        if base_state is None:
            last_attempted_tier = "A"
            try:
                server = self._ensure_server()
                info = server.env_inspect(name=theorem_full_name)
                thm_type_obj = info.get("type") if isinstance(info, dict) else None
                thm_type = (
                    thm_type_obj.get("pp")
                    if isinstance(thm_type_obj, dict)
                    else str(thm_type_obj)
                    if thm_type_obj
                    else None
                )
                if thm_type:
                    goal_str = self.goal_start(thm_type, theorem_full_name, file_path)
                    base_state = self._goal_states.get((self._current_env_key, goal_str))
                    if base_state is not None:
                        tier_used = "A"
            except Exception as e:
                if self._is_crash_error(e):
                    crash_retries += 1
                    self._restart_server()
                logger.debug("Tier A failed for %s: %s", theorem_full_name, e)

        # ----------------------------------------------------------
        # Tier A2 (metadata fallback): synthesize theorem type from a
        # pretty-printed goal state when declaration metadata is stale.
        # ----------------------------------------------------------
        if base_state is None and fallback_goal_pp and "⊢" in fallback_goal_pp:
            last_attempted_tier = "A2"
            try:
                goal_str = self.goal_start_from_pp(
                    fallback_goal_pp,
                    theorem_name=theorem_full_name,
                    file_path=file_path,
                )
                base_state = self._goal_states.get((self._current_env_key, goal_str))
                if base_state is not None:
                    tier_used = "A2"
            except Exception as e:
                if self._is_crash_error(e):
                    crash_retries += 1
                    self._restart_server()
                logger.debug("Tier A2 failed for %s: %s", theorem_full_name, e)

        # Step-0: return base state directly.
        if not prefix_tactics:
            if base_state is not None:
                goal_str = str(base_state.goals[0].target)
                cache_key = (env_key, goal_str)
                self._goal_states[cache_key] = base_state
                return ReplayResult(
                    success=True,
                    goal_state=goal_str,
                    goal_state_obj=base_state,
                    tier_used=tier_used,
                    failure_category="",
                    failing_prefix_idx=-1,
                    crash_retries=crash_retries,
                    env_key=env_key,
                    feedback=LeanFeedback(
                        stage="goal_creation",
                        category="none",
                        messages=[],
                        raw_error="",
                    ),
                )
            return ReplayResult(
                success=False,
                goal_state="",
                goal_state_obj=None,
                tier_used=tier_used or last_attempted_tier,
                failure_category="goal_creation_fail",
                failing_prefix_idx=-1,
                crash_retries=crash_retries,
                env_key=env_key,
                feedback=self._last_goal_feedback
                or LeanFeedback(
                    stage="goal_creation",
                    category="goal_creation_fail",
                    messages=[],
                    raw_error="unable to create initial goal state",
                ),
            )

        # ----------------------------------------------------------
        # Tier C: Sequential goal_tactic replay (step>0, high-fidelity)
        # Per PLAN_2.md §2 Tier C: obtain theorem-level state via
        # Tier A or B, then apply each prefix tactic sequentially
        # via goal_tactic to build the real intermediate state.
        # ----------------------------------------------------------
        if base_state is None:
            return ReplayResult(
                success=False,
                goal_state="",
                goal_state_obj=None,
                tier_used="C",
                failure_category="goal_creation_fail",
                failing_prefix_idx=-1,
                crash_retries=crash_retries,
                env_key=env_key,
                feedback=self._last_goal_feedback
                or LeanFeedback(
                    stage="goal_creation",
                    category="goal_creation_fail",
                    messages=[],
                    raw_error="unable to create base state for prefix replay",
                ),
            )

        current_state = base_state
        server = self._ensure_server()
        acc = accessible_names or self._accessible_names

        from pantograph.server import TacticFailure as _TF  # type: ignore[import-untyped]

        # State-guided replay: process each prefix tactic with per-step
        # alignment against the expected intermediate goal states.
        pgs = prefix_goal_states or []

        for i, tac in enumerate(prefix_tactics):
            if not current_state.goals:
                return ReplayResult(
                    success=False,
                    goal_state="",
                    goal_state_obj=None,
                    tier_used="C",
                    failure_category="prefix_replay_fail",
                    failing_prefix_idx=i,
                    crash_retries=crash_retries,
                    env_key=env_key,
                    feedback=LeanFeedback(
                        stage="tactic_exec",
                        category="prefix_replay_fail",
                        messages=[],
                        raw_error="prefix replay exhausted goals",
                    ),
                )

            # Per-step expected state (the state BEFORE this tactic)
            step_expected = pgs[i] if i < len(pgs) else ""

            # Structural prelude: intro missing binders for this step
            if step_expected and current_state.goals:
                current_state = _intro_missing_locals(server, current_state, step_expected)

            # Build alias map for this step's local context
            actual_str = str(current_state.goals[0]) if current_state.goals else ""
            alias_map = build_local_alias_map(actual_str, step_expected)

            # Qualify global names + apply alias map to this tactic
            prepared_tac = tac
            if self._suffix_index and acc:
                prepared_tac = qualify_tactic(prepared_tac, acc, self._suffix_index)
            if alias_map:
                prepared_tac = rewrite_tactic_locals(prepared_tac, alias_map)

            # Apply the tactic
            try:
                current_state = server.goal_tactic(current_state, tactic=prepared_tac)
            except _TF as e:
                logger.debug(
                    "Tier C prefix[%d] TacticFailure for %s: %s",
                    i,
                    theorem_full_name,
                    e,
                )
                return ReplayResult(
                    success=False,
                    goal_state="",
                    goal_state_obj=None,
                    tier_used="C",
                    failure_category="prefix_replay_fail",
                    failing_prefix_idx=i,
                    crash_retries=crash_retries,
                    env_key=env_key,
                    feedback=replace(
                        _classify_tactic_failure(e),
                        category="prefix_replay_fail",
                    ),
                )
            except Exception as e:
                if self._is_crash_error(e):
                    crash_retries += 1
                    self._restart_server()
                return ReplayResult(
                    success=False,
                    goal_state="",
                    goal_state_obj=None,
                    tier_used="C",
                    failure_category="prefix_replay_fail",
                    failing_prefix_idx=i,
                    crash_retries=crash_retries,
                    env_key=env_key,
                    feedback=_classify_compiler_feedback(
                        e,
                        stage="tactic_exec",
                        fallback_category="prefix_replay_fail",
                    ),
                )

            # Post-step: select the correct subgoal for the next step.
            # The next expected state tells us which subgoal to target.
            if current_state.goals and len(current_state.goals) > 1:
                next_expected = ""
                if i + 1 < len(pgs):
                    next_expected = pgs[i + 1]
                elif i + 1 == len(prefix_tactics):
                    next_expected = expected_goal

                if next_expected:
                    _match_goal(current_state.goals, next_expected)
                    # Post-step subgoal selection noted but not acted on
                    # yet — Pantograph doesn't support goal reordering.

        # Final state: match against expected_goal
        if not current_state.goals:
            return ReplayResult(
                success=False,
                goal_state="",
                goal_state_obj=None,
                tier_used="C",
                failure_category="prefix_replay_fail",
                failing_prefix_idx=len(prefix_tactics) - 1,
                crash_retries=crash_retries,
                env_key=env_key,
                feedback=LeanFeedback(
                    stage="tactic_exec",
                    category="prefix_replay_fail",
                    messages=[],
                    raw_error="prefix replay solved all goals before expected target",
                ),
            )

        matched_idx, matched_goal_str = _match_goal(current_state.goals, expected_goal)

        if matched_idx < 0 and expected_goal:
            return ReplayResult(
                success=False,
                goal_state="",
                goal_state_obj=None,
                tier_used="C",
                failure_category="goal_match_fail",
                failing_prefix_idx=-1,
                crash_retries=crash_retries,
                env_key=env_key,
                feedback=LeanFeedback(
                    stage="tactic_exec",
                    category="goal_match_fail",
                    messages=[],
                    raw_error="no current subgoal matched the expected goal",
                ),
            )
        if matched_idx < 0:
            matched_idx = 0
            matched_goal_str = str(current_state.goals[0].target)

        cache_key = (env_key, matched_goal_str)
        self._goal_states[cache_key] = current_state

        return ReplayResult(
            success=True,
            goal_state=matched_goal_str,
            goal_state_obj=current_state,
            tier_used="C",
            failure_category="",
            failing_prefix_idx=-1,
            crash_retries=crash_retries,
            env_key=env_key,
            goal_id=matched_idx,
            feedback=LeanFeedback.success(),
        )

    def gc(self) -> None:
        """Garbage-collect unused goal states in the Pantograph server."""
        if self._server is not None:
            try:
                self._server.gc()
            except Exception:
                pass
        self._goal_states.clear()
        self._tactic_cache.clear()


def _build_hammer_tactics(premises: list[str]) -> list[str]:
    """Build a list of hammer tactic strings to try."""
    tactics = []

    # aesop with premises as simp lemmas
    if premises:
        premise_list = ", ".join(premises[:16])
        tactics.append(f"aesop (add safe [{premise_list}])")

    tactics.append("solve_by_elim")
    tactics.append("aesop")
    tactics.append("omega")
    tactics.append("decide")

    if premises:
        premise_list = ", ".join(premises[:16])
        tactics.append(f"simp [{premise_list}]")
    tactics.append("simp")
    tactics.append("apply?")

    return tactics
