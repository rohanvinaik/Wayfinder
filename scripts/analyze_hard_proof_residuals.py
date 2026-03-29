"""Analyze hard-proof residual worklists and summarize common structure.

This script is the data-analysis entry point for the staged hard-proof program.
It can analyze one or more hard-bucket JSONL files (for example the
`hard_proof_all.jsonl` outputs from multiple EXP-SOM-011 conditions), optionally
joining theorem-level narratives from `template_narrative_train_som.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _load_narratives(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        theorem_id = row.get("theorem_id")
        if theorem_id:
            out[str(theorem_id)] = row
    return out


def _namespace_prefix(theorem_id: str) -> str:
    if "." in theorem_id:
        return theorem_id.split(".", 1)[0]
    return "(root)"


def _primary_goal_geometry(goal: str) -> str:
    if not goal:
        return "unknown"
    text = goal.strip()
    if text.startswith("∀"):
        return "forall"
    if text.startswith("∃"):
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
    return "other"


def _goal_tags(goal: str) -> list[str]:
    if not goal:
        return ["unknown"]
    tags: list[str] = []
    checks = [
        ("forall", "∀"),
        ("exists", "∃"),
        ("iff", "↔"),
        ("equality", "="),
        ("inequality", " ≤ "),
        ("inequality", " ≥ "),
        ("inequality", " < "),
        ("inequality", " > "),
        ("membership", "∈"),
        ("subset", "⊆"),
        ("negation", "¬"),
    ]
    for tag, marker in checks:
        if marker in goal and tag not in tags:
            tags.append(tag)
    return tags or ["other"]


def _tactic_prefixes(tactics: list[Any]) -> list[str]:
    prefixes: list[str] = []
    for tactic in tactics:
        if isinstance(tactic, str) and tactic.strip():
            prefixes.append(tactic.strip().split()[0])
    return prefixes


def analyze_hard_proof_residuals(
    inputs: list[Path],
    narrative_path: Path | None = None,
    dedup_by_theorem: bool = True,
) -> dict[str, Any]:
    narratives = _load_narratives(narrative_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in inputs:
        for row in _load_jsonl(path):
            theorem_id = str(row.get("theorem_id", ""))
            if dedup_by_theorem and theorem_id:
                if theorem_id in seen:
                    continue
                seen.add(theorem_id)
            rows.append(row)

    by_namespace = Counter()
    by_geometry = Counter()
    by_tag = Counter()
    by_residual_bucket = Counter()
    by_hard_track = Counter()
    by_template = Counter()
    by_lane_sequence = Counter()
    by_tactic_prefix = Counter()
    by_pathology = Counter()
    examples_by_geometry: dict[str, list[str]] = defaultdict(list)

    matched_narratives = 0
    proof_steps: list[int] = []
    unique_premises: list[int] = []
    last_goal_available = 0

    for row in rows:
        theorem_id = str(row.get("theorem_id", ""))
        namespace = _namespace_prefix(theorem_id)
        goal = str(row.get("last_goal") or row.get("goal_state") or "")
        geometry = _primary_goal_geometry(goal)
        narrative = narratives.get(theorem_id)

        by_namespace[namespace] += 1
        by_geometry[geometry] += 1
        by_residual_bucket[str(row.get("residual_bucket", ""))] += 1
        by_hard_track[str(row.get("hard_track", ""))] += 1
        if row.get("lane_sequence"):
            by_lane_sequence[str(row.get("lane_sequence", ""))] += 1
        for tag in _goal_tags(goal):
            by_tag[tag] += 1
        for tactic_prefix in _tactic_prefixes(row.get("tactics_used", [])):
            by_tactic_prefix[tactic_prefix] += 1
        for tag in row.get("search_pathology_tags", []):
            by_pathology[str(tag)] += 1
        if theorem_id and len(examples_by_geometry[geometry]) < 10:
            examples_by_geometry[geometry].append(theorem_id)
        if row.get("last_goal_available"):
            last_goal_available += 1

        if narrative:
            matched_narratives += 1
            template_id = str(narrative.get("template_id", ""))
            if template_id:
                by_template[template_id] += 1
            history = narrative.get("proof_history_summary", {})
            proof_steps.append(int(history.get("total_steps", 0) or 0))
            unique_premises.append(int(history.get("unique_premises", 0) or 0))

    def _avg(values: list[int]) -> float:
        return round(sum(values) / max(len(values), 1), 2)

    return {
        "total_theorems": len(rows),
        "dedup_by_theorem": dedup_by_theorem,
        "last_goal_available": last_goal_available,
        "last_goal_coverage": round(last_goal_available / max(len(rows), 1), 4),
        "matched_narratives": matched_narratives,
        "narrative_coverage": round(matched_narratives / max(len(rows), 1), 4),
        "mean_proof_steps": _avg(proof_steps),
        "mean_unique_premises": _avg(unique_premises),
        "by_namespace_prefix": dict(by_namespace.most_common(25)),
        "by_primary_goal_geometry": dict(by_geometry.most_common()),
        "by_goal_tag": dict(by_tag.most_common()),
        "by_residual_bucket": dict(by_residual_bucket.most_common()),
        "by_hard_track": dict(by_hard_track.most_common()),
        "by_template_id": dict(by_template.most_common(20)),
        "by_lane_sequence": dict(by_lane_sequence.most_common(20)),
        "by_tactic_prefix": dict(by_tactic_prefix.most_common(20)),
        "by_search_pathology": dict(by_pathology.most_common()),
        "example_theorems_by_geometry": dict(examples_by_geometry),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Hard-bucket JSONL input(s)")
    parser.add_argument(
        "--narratives",
        default="data/template_narrative_train_som.jsonl",
        help="Optional theorem-level narrative dataset",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    parser.add_argument(
        "--no-dedup-by-theorem",
        action="store_true",
        help="Keep duplicate theorem_ids across multiple condition inputs",
    )
    args = parser.parse_args()

    summary = analyze_hard_proof_residuals(
        inputs=[Path(path) for path in args.inputs],
        narrative_path=Path(args.narratives) if args.narratives else None,
        dedup_by_theorem=not args.no_dedup_by_theorem,
    )
    print(json.dumps(summary, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
