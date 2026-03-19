"""Static census of Lean source-context features in local Mathlib.

This is a cheap whole-corpus validator for the ContextIR compiler plan. It does
not start Pantograph or Lean; it scans source files and counts the context
constructs that theorem-faithful wrappers need to model.

Usage:
    python -m scripts.context_ir_census
    python -m scripts.context_ir_census --mathlib-root data/lean_project/.lake/packages/mathlib/Mathlib --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.lean_context_ir import classify_context_directive


def run(mathlib_root: Path, limit_files: int | None = None) -> dict:
    feature_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    files = sorted(mathlib_root.rglob("*.lean"))
    if limit_files is not None:
        files = files[:limit_files]

    for path in files:
        seen_in_file: set[str] = set()
        with path.open() as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped or stripped.startswith("--") or stripped.startswith("/-"):
                    continue
                kind, inline_only = classify_context_directive(stripped)
                if kind is not None:
                    feature_counts[kind] += 1
                    seen_in_file.add(kind)
                    if inline_only:
                        feature_counts["inline_only"] += 1
                elif stripped.startswith("namespace "):
                    feature_counts["namespace"] += 1
                    seen_in_file.add("namespace")
                elif stripped == "section" or stripped.startswith("section "):
                    feature_counts["section"] += 1
                    seen_in_file.add("section")
                elif stripped.startswith("noncomputable section"):
                    feature_counts["noncomputable_section"] += 1
                    seen_in_file.add("noncomputable_section")
                elif stripped == "end" or stripped.startswith("end "):
                    feature_counts["end"] += 1

        for kind in seen_in_file:
            file_counts[kind] += 1

    return {
        "mathlib_root": str(mathlib_root),
        "files_scanned": len(files),
        "feature_counts": dict(feature_counts.most_common()),
        "files_with_feature": dict(file_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Census Lean source-context features")
    parser.add_argument(
        "--mathlib-root",
        type=Path,
        default=Path("data/lean_project/.lake/packages/mathlib/Mathlib"),
        help="Root of local Mathlib checkout",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args.mathlib_root, args.limit_files)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("ContextIR Mathlib Census")
    print(f"Files scanned: {report['files_scanned']}")
    print(f"Root: {report['mathlib_root']}")
    print("\nFeature counts:")
    for kind, count in report["feature_counts"].items():
        print(f"  {kind}: {count}")
    print("\nFiles with feature:")
    for kind, count in report["files_with_feature"].items():
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
