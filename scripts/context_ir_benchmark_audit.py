"""Audit ContextIR feature needs on a benchmark dataset.

The goal is to measure how much of the benchmark sits inside the current
source-context compiler's supported subset and which unsupported constructs are
most common at theorem declaration sites.

This script is intentionally static: it scans local source files and does not
start Pantograph. It is cheap enough to run before every replay-context change.

Usage:
    python -m scripts.context_ir_benchmark_audit --dataset data/canonical/rw0_eval.jsonl
    python -m scripts.context_ir_benchmark_audit --dataset data/canonical/canonical_eval_replayable.jsonl --limit 200 --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.lean_context_ir import extract_context_ir, find_decl_line


def _resolve_lean_path(project_root: Path, file_path: str) -> Path:
    return project_root / ".lake" / "packages" / "mathlib" / file_path


def run(dataset: Path, project_root: Path, limit: int | None = None) -> dict:
    examples = []
    with dataset.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            examples.append(json.loads(line))

    feature_examples: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    unsupported_kinds: Counter[str] = Counter()
    unsupported_texts: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    processed = 0
    for ex in examples:
        file_path = ex.get("file_path", "")
        full_name = ex.get("theorem_full_name", "")
        if not file_path or not full_name:
            failures["missing_file_or_name"] += 1
            continue

        lean_path = _resolve_lean_path(project_root, file_path)
        if not lean_path.exists():
            failures["file_missing"] += 1
            continue

        short_name = full_name.split(".")[-1]
        theorem_line = find_decl_line(lean_path, short_name)
        if theorem_line < 0:
            failures["decl_not_found"] += 1
            continue

        ir = extract_context_ir(lean_path, theorem_line)
        processed += 1

        seen = set()
        for directive in ir.active_directives:
            feature_counts[directive.kind] += 1
            seen.add(directive.kind)
        for kind in seen:
            feature_examples[kind] += 1

        for directive in ir.unsupported:
            unsupported_kinds[directive.kind] += 1
            unsupported_texts[directive.text] += 1

    report = {
        "dataset": str(dataset),
        "project_root": str(project_root),
        "requested_examples": len(examples),
        "processed_examples": processed,
        "failures": dict(failures.most_common()),
        "feature_counts": dict(feature_counts.most_common()),
        "examples_with_feature": dict(feature_examples.most_common()),
        "unsupported_kinds": dict(unsupported_kinds.most_common()),
        "top_unsupported_texts": [
            {"text": text, "count": count}
            for text, count in unsupported_texts.most_common(15)
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ContextIR needs on a benchmark set")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/canonical/rw0_eval.jsonl"),
        help="JSONL benchmark dataset with theorem_full_name and file_path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("data/lean_project"),
        help="Lean project root",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args.dataset, args.project_root, args.limit)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("ContextIR Benchmark Audit")
    print(f"Dataset: {report['dataset']}")
    print(f"Processed: {report['processed_examples']}/{report['requested_examples']}")
    if report["failures"]:
        print("\nFailures:")
        for kind, count in report["failures"].items():
            print(f"  {kind}: {count}")
    print("\nFeature counts:")
    for kind, count in report["feature_counts"].items():
        print(f"  {kind}: {count}")
    print("\nExamples with feature:")
    for kind, count in report["examples_with_feature"].items():
        print(f"  {kind}: {count}")
    print("\nUnsupported kinds:")
    for kind, count in report["unsupported_kinds"].items():
        print(f"  {kind}: {count}")
    print("\nTop unsupported texts:")
    for item in report["top_unsupported_texts"]:
        print(f"  {item['count']:>4}  {item['text']}")


if __name__ == "__main__":
    main()
