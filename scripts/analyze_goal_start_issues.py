"""Mine goal-start issue signatures from structured trigger datasets.

This script inspects `goal_start_failure` rows and assembles theorem-level
issue lists using:

- recorded failure / repair feedback
- source-context extraction (`ContextIR`)
- cheap bounded source scans near the theorem site

It is intended to answer: "what cheap structural replay gaps remain?" without
rerunning Lean.

Examples:
    python -m scripts.analyze_goal_start_issues \
        --data data/apply_trigger_train_full.jsonl

    python -m scripts.analyze_goal_start_issues \
        --data data/apply_trigger_train_full.jsonl \
        --theorem-list runs/exp053_goal_repair/cosine_rw_only.jsonl \
        --only-unrepaired \
        --output-json runs/goal_start_issue_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.lean_context_ir import extract_context_ir

_RE_UNIVERSE = re.compile(r"\bu_\d+\b")
_RE_INSTANCE_BINDER = re.compile(r"\[[^\]]+\]")
_RE_VARIABLE = re.compile(r"^(?:variable|variables)\b")
_RE_LOCAL_INSTANCE = re.compile(r"^local\s+instance\b")
_RE_INSTANCE = re.compile(r"^instance\b")
_RE_ATTRIBUTE_LOCAL = re.compile(r"^(?:attribute\s+\[[^\]]*local[^\]]*\]|local attribute\b)")
_RE_ATTRIBUTE_INSTANCE = re.compile(
    r"^(?:attribute\s+\[[^\]]*instance[^\]]*\]|local attribute\s+\[[^\]]*instance[^\]]*\])"
)
_RE_SCOPED_NOTATION = re.compile(r"^(?:scoped notation\b|scoped\[)")
_RE_LOCAL_NOTATION = re.compile(r"^local notation\b")
_RE_OPEN_SCOPED = re.compile(r"^open scoped\b")
_RE_INCLUDE = re.compile(r"^include\b")
_RE_OMIT = re.compile(r"^omit\b")
_RE_LETI = re.compile(r"^(?:letI|haveI)\b")


@dataclass
class SourceScan:
    instance_lines: int = 0
    local_instance_lines: int = 0
    attribute_local_lines: int = 0
    attribute_instance_lines: int = 0
    scoped_notation_lines: int = 0
    local_notation_lines: int = 0
    open_scoped_lines: int = 0
    include_lines: int = 0
    omit_lines: int = 0
    leti_lines: int = 0
    multiline_variable_blocks: int = 0
    multiline_local_attribute_blocks: int = 0
    multiline_local_notation_blocks: int = 0
    window_start_line: int = 0
    window_end_line: int = 0


@dataclass
class IssueRecord:
    theorem_id: str
    namespace_prefix: str
    lean_path: str
    theorem_line: int
    status: str
    failure_categories: list[str]
    feedback_categories: list[str]
    repair_feedback_categories: list[str]
    theorem_type_excerpt: str
    issue_tags: list[str]
    context_unsupported: list[dict[str, str]]
    source_scan: dict[str, int]
    context_features: dict[str, int]
    evidence: dict[str, Any]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("row_type") == "goal_start_failure":
                rows.append(row)
    return rows


def _load_theorem_filter(path: Path) -> set[str]:
    theorem_ids: set[str] = set()
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                row = json.loads(raw)
                theorem_id = row.get("theorem_id") or row.get("theorem_full_name")
                if theorem_id:
                    theorem_ids.add(str(theorem_id))
            else:
                theorem_ids.add(raw)
    return theorem_ids


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        theorem_id = row.get("theorem_id", "")
        if theorem_id:
            grouped[theorem_id].append(row)
    return grouped


def _row_score(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(row.get("lean_path"))),
        int(bool(row.get("theorem_type"))),
        int(row.get("theorem_line", 0) > 0),
    )


def _pick_representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=_row_score)


def _source_scan(lean_path: str, theorem_line: int, window: int) -> SourceScan:
    scan = SourceScan()
    path = Path(lean_path)
    if theorem_line <= 0 or not lean_path or not path.exists():
        return scan

    try:
        lines = path.read_text().splitlines()
    except OSError:
        return scan

    start = max(1, theorem_line - window)
    end = max(0, theorem_line - 1)
    scan.window_start_line = start
    scan.window_end_line = end
    for idx in range(start, end + 1):
        stripped = lines[idx - 1].strip()
        if not stripped:
            continue
        next_is_indented = idx < len(lines) and (
            lines[idx].startswith("  ") or lines[idx].startswith("\t")
        )
        if _RE_LOCAL_INSTANCE.match(stripped):
            scan.local_instance_lines += 1
        elif _RE_INSTANCE.match(stripped):
            scan.instance_lines += 1
        if _RE_ATTRIBUTE_LOCAL.match(stripped):
            scan.attribute_local_lines += 1
            if next_is_indented:
                scan.multiline_local_attribute_blocks += 1
        if _RE_ATTRIBUTE_INSTANCE.match(stripped):
            scan.attribute_instance_lines += 1
        if _RE_SCOPED_NOTATION.match(stripped):
            scan.scoped_notation_lines += 1
        if _RE_LOCAL_NOTATION.match(stripped):
            scan.local_notation_lines += 1
            if next_is_indented:
                scan.multiline_local_notation_blocks += 1
        if _RE_OPEN_SCOPED.match(stripped):
            scan.open_scoped_lines += 1
        if _RE_INCLUDE.match(stripped):
            scan.include_lines += 1
        if _RE_OMIT.match(stripped):
            scan.omit_lines += 1
        if _RE_LETI.match(stripped):
            scan.leti_lines += 1
        if _RE_VARIABLE.match(stripped) and next_is_indented:
            scan.multiline_variable_blocks += 1
    return scan


def _compact_unsupported(lean_path: str, theorem_line: int) -> list[dict[str, str]]:
    path = Path(lean_path)
    if theorem_line <= 0 or not lean_path or not path.exists():
        return []
    try:
        ir = extract_context_ir(path, theorem_line)
    except OSError:
        return []
    return [
        {"kind": directive.kind, "reason": directive.reason, "text": directive.text}
        for directive in ir.unsupported
    ]


def _issue_tags(
    row: dict[str, Any],
    scan: SourceScan,
    unsupported: list[dict[str, str]],
) -> list[str]:
    tags: set[str] = set()

    failure_category = row.get("failure_category", "")
    feedback_category = (row.get("feedback") or {}).get("category", "")
    repair_feedback_category = (row.get("repair_feedback") or {}).get("category", "")
    theorem_type = row.get("theorem_type", "")

    if not row.get("lean_path") or int(row.get("theorem_line", 0)) <= 0:
        tags.add("missing_source_metadata")
    if failure_category == "goal_creation_fail" and not feedback_category:
        tags.add("opaque_goal_creation_failure")
    if "typeclass_missing" in {failure_category, feedback_category, repair_feedback_category}:
        tags.add("typeclass_elaboration_failure")
    if "unification_mismatch" in {failure_category, feedback_category, repair_feedback_category}:
        tags.add("unification_mismatch_goal_start")
    if "unknown_identifier" in {failure_category, feedback_category, repair_feedback_category}:
        tags.add("unknown_identifier_goal_start")
    if _RE_UNIVERSE.search(theorem_type):
        tags.add("universe_polymorphic_statement")
    if len(_RE_INSTANCE_BINDER.findall(theorem_type)) >= 4:
        tags.add("heavy_typeclass_statement")

    unsupported_pairs = {(u["kind"], u["reason"]) for u in unsupported}
    if ("variable", "probable_multiline") in unsupported_pairs:
        tags.add("multiline_variable_context")
    if ("local_attribute", "probable_multiline") in unsupported_pairs:
        tags.add("multiline_local_attribute_context")
    if ("local_notation", "probable_multiline") in unsupported_pairs or (
        "notation",
        "probable_multiline",
    ) in unsupported_pairs:
        tags.add("multiline_notation_context")
    if ("open", "inline_next_decl_only") in unsupported_pairs:
        tags.add("inline_open_context")
    if ("open_scoped", "inline_next_decl_only") in unsupported_pairs:
        tags.add("inline_open_scoped_context")

    if scan.multiline_variable_blocks > 0:
        tags.add("multiline_variable_context")
    if scan.multiline_local_attribute_blocks > 0:
        tags.add("multiline_local_attribute_context")
    if scan.multiline_local_notation_blocks > 0:
        tags.add("multiline_notation_context")
    if scan.scoped_notation_lines > 0:
        tags.add("scoped_notation_context")
    if scan.local_notation_lines > 0:
        tags.add("local_notation_context")
    if scan.open_scoped_lines > 0:
        tags.add("open_scoped_context")
    if scan.include_lines > 0 or scan.omit_lines > 0:
        tags.add("include_omit_context")
    if (
        scan.instance_lines > 0
        or scan.local_instance_lines > 0
        or scan.attribute_instance_lines > 0
        or scan.leti_lines > 0
    ):
        tags.add("same_file_instance_support")

    if row.get("repair_attempted") and not row.get("repair_success"):
        tags.add("repair_exhausted")
    if row.get("repair_success"):
        tags.add("repair_succeeded")

    return sorted(tags)


def _build_record(rows: list[dict[str, Any]], window: int) -> IssueRecord:
    row = _pick_representative(rows)
    lean_path = str(row.get("lean_path", ""))
    theorem_line = int(row.get("theorem_line", 0))
    scan = _source_scan(lean_path, theorem_line, window)
    unsupported = _compact_unsupported(lean_path, theorem_line)
    issue_tags = _issue_tags(row, scan, unsupported)

    failure_categories = sorted(
        {r.get("failure_category", "") for r in rows if r.get("failure_category")}
    )
    feedback_categories = sorted(
        {
            (r.get("feedback") or {}).get("category", "")
            for r in rows
            if (r.get("feedback") or {}).get("category", "")
        }
    )
    repair_feedback_categories = sorted(
        {
            (r.get("repair_feedback") or {}).get("category", "")
            for r in rows
            if (r.get("repair_feedback") or {}).get("category", "")
        }
    )
    unrepaired = any(r.get("repair_attempted") and not r.get("repair_success") for r in rows)
    repaired = any(r.get("repair_success") for r in rows)
    status = "unrepaired" if unrepaired else "repaired" if repaired else "failed"
    messages = [
        (r.get("feedback") or {}).get("messages") or []
        for r in rows
    ]
    repair_messages = [
        (r.get("repair_feedback") or {}).get("messages") or []
        for r in rows
    ]
    first_message = ""
    first_repair_message = ""
    for group in messages:
        if group:
            first_message = group[0].get("data", "")
            break
    for group in repair_messages:
        if group:
            first_repair_message = group[0].get("data", "")
            break

    return IssueRecord(
        theorem_id=str(row.get("theorem_id", "")),
        namespace_prefix=str(row.get("namespace_prefix", "")),
        lean_path=str(row.get("lean_path", "")),
        theorem_line=int(row.get("theorem_line", 0)),
        status=status,
        failure_categories=failure_categories,
        feedback_categories=feedback_categories,
        repair_feedback_categories=repair_feedback_categories,
        theorem_type_excerpt=str(row.get("theorem_type", ""))[:240],
        issue_tags=issue_tags,
        context_unsupported=unsupported,
        source_scan=asdict(scan),
        context_features={k: int(v) for k, v in (row.get("context_features") or {}).items()},
        evidence={
            "message_excerpt": first_message[:240],
            "repair_message_excerpt": first_repair_message[:240],
            "row_count": len(rows),
            "source_window": [scan.window_start_line, scan.window_end_line],
        },
    )


def _summarize(records: list[IssueRecord]) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[str]] = defaultdict(list)
    namespace_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for record in records:
        status_counts[record.status] += 1
        namespace_counts[record.namespace_prefix] += 1
        for tag in record.issue_tags:
            issue_counts[tag] += 1
            if len(issue_examples[tag]) < 12:
                issue_examples[tag].append(record.theorem_id)

    return {
        "issue_counts": dict(issue_counts.most_common()),
        "issue_examples": issue_examples,
        "namespace_counts": dict(namespace_counts.most_common(25)),
        "status_counts": dict(status_counts.most_common()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_summary(records: list[IssueRecord], summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("Goal-Start Issue Report")
    print("=" * 72)
    print(f"  Theorems: {len(records)}")
    print(f"  Status:   {summary['status_counts']}")
    print("\nTop issue tags:")
    for tag, count in list(summary["issue_counts"].items())[:20]:
        examples = ", ".join(summary["issue_examples"][tag][:4])
        print(f"  {tag:34s} {count:5d}  {examples}")
    print("\nTop namespaces:")
    for namespace, count in list(summary["namespace_counts"].items())[:15]:
        print(f"  {namespace:34s} {count:5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL dataset with goal_start_failure rows")
    parser.add_argument(
        "--theorem-list",
        help=(
            "Optional theorem-id filter. Accepts JSONL with theorem_id/"
            "theorem_full_name or plain text."
        ),
    )
    parser.add_argument(
        "--only-unrepaired",
        action="store_true",
        help="Keep only theorems whose recorded repairs did not succeed.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Source scan lookback window in lines (default: 120)",
    )
    parser.add_argument("--output-json", help="Optional JSON report path")
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    if args.theorem_list:
        theorem_filter = _load_theorem_filter(Path(args.theorem_list))
        rows = [row for row in rows if row.get("theorem_id") in theorem_filter]

    grouped = _group_rows(rows)
    records = [_build_record(theorem_rows, args.window) for theorem_rows in grouped.values()]

    if args.only_unrepaired:
        records = [record for record in records if record.status == "unrepaired"]

    records.sort(key=lambda record: (record.status, -len(record.issue_tags), record.theorem_id))
    summary = _summarize(records)
    _print_summary(records, summary)

    if args.output_json:
        payload = {
            "input": args.data,
            "theorem_filter": args.theorem_list or "",
            "only_unrepaired": bool(args.only_unrepaired),
            "window": args.window,
            "summary": summary,
            "records": [asdict(record) for record in records],
        }
        _write_json(Path(args.output_json), payload)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
