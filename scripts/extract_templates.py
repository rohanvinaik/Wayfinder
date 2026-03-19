"""Extract proof strategy templates from the navigational training corpus.

Reads nav_training.jsonl, classifies each proof's tactic sequence into
a template, computes template statistics, and outputs:
  - data/template_taxonomy.json: template definitions + counts + move profiles
  - data/nav_train_templates.jsonl: augmented training data with template_id + theorem move profile

Usage:
    python -m scripts.extract_templates --data data/nav_training.jsonl --output-dir data/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.nav_contracts import NavigationalExample
from src.story_templates import (
    TEMPLATE_NAMES,
    TEMPLATE_TAXONOMY,
    classify_tactic_sequence,
    compute_bank_signature,
    get_template_index,
)


def _top_counts(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    """Serialize the top entries of a counter."""
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _theorem_group_key(ex: NavigationalExample) -> str:
    """Return the stable theorem grouping key for theorem-level aggregation."""
    return ex.theorem_key or ex.theorem_id


def _summarize_step_metadata(thm_examples: list[NavigationalExample]) -> dict[str, Any]:
    """Aggregate controller-facing move metadata across one theorem."""
    family_counts: Counter[str] = Counter()
    subtask_counts: Counter[str] = Counter()
    target_head_counts: Counter[str] = Counter()
    trigger_kind_counts: Counter[str] = Counter()
    trigger_signature_counts: Counter[str] = Counter()
    steps_with_metadata = 0

    for ex in thm_examples:
        metadata = ex.metadata or {}
        if not metadata:
            continue
        steps_with_metadata += 1

        local_family = metadata.get("local_family", "")
        if local_family:
            family_counts[local_family] += 1

        subtask_kind = metadata.get("subtask_kind", "")
        if subtask_kind:
            subtask_counts[subtask_kind] += 1

        target_head = metadata.get("goal_target_head", "")
        if target_head:
            target_head_counts[target_head] += 1

        for signature in metadata.get("trigger_signature", []):
            if not signature:
                continue
            trigger_signature_counts[signature] += 1
            trigger_kind_counts[signature.split("=", 1)[0]] += 1

    return {
        "steps_with_metadata": steps_with_metadata,
        "dominant_local_family": family_counts.most_common(1)[0][0] if family_counts else "",
        "dominant_subtask_kind": subtask_counts.most_common(1)[0][0] if subtask_counts else "",
        "top_local_families": _top_counts(family_counts),
        "top_subtasks": _top_counts(subtask_counts),
        "top_target_heads": _top_counts(target_head_counts),
        "top_trigger_kinds": _top_counts(trigger_kind_counts),
        "top_trigger_signatures": _top_counts(trigger_signature_counts),
    }


def _accumulate_template_move_stats(
    template_stats: dict[str, Counter[str] | int],
    thm_examples: list[NavigationalExample],
) -> None:
    """Accumulate raw step metadata into template-level counters."""
    for ex in thm_examples:
        metadata = ex.metadata or {}
        if not metadata:
            continue
        template_stats["steps_with_metadata"] += 1

        local_family = metadata.get("local_family", "")
        if local_family:
            family_counts = template_stats["families"]
            assert isinstance(family_counts, Counter)
            family_counts[local_family] += 1

        subtask_kind = metadata.get("subtask_kind", "")
        if subtask_kind:
            subtask_counts = template_stats["subtasks"]
            assert isinstance(subtask_counts, Counter)
            subtask_counts[subtask_kind] += 1

        target_head = metadata.get("goal_target_head", "")
        if target_head:
            target_head_counts = template_stats["target_heads"]
            assert isinstance(target_head_counts, Counter)
            target_head_counts[target_head] += 1

        for signature in metadata.get("trigger_signature", []):
            if not signature:
                continue
            trigger_signature_counts = template_stats["trigger_signatures"]
            trigger_kind_counts = template_stats["trigger_kinds"]
            assert isinstance(trigger_signature_counts, Counter)
            assert isinstance(trigger_kind_counts, Counter)
            trigger_signature_counts[signature] += 1
            trigger_kind_counts[signature.split("=", 1)[0]] += 1


def _classify_theorems(
    theorems: dict[str, list[NavigationalExample]],
) -> tuple[
    dict[str, str],
    Counter[str],
    dict[str, list[dict[str, int]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Counter[str] | int]],
]:
    """Classify each theorem's tactic sequence into a template."""
    theorem_templates: dict[str, str] = {}
    template_counts: Counter[str] = Counter()
    template_bank_sigs: dict[str, list[dict[str, int]]] = {name: [] for name in TEMPLATE_NAMES}
    theorem_move_profiles: dict[str, dict[str, Any]] = {}
    template_move_stats: dict[str, dict[str, Counter[str] | int]] = {
        name: {
            "steps_with_metadata": 0,
            "families": Counter(),
            "subtasks": Counter(),
            "target_heads": Counter(),
            "trigger_kinds": Counter(),
            "trigger_signatures": Counter(),
        }
        for name in TEMPLATE_NAMES
    }

    for theorem_id, thm_examples in theorems.items():
        thm_examples.sort(key=lambda e: e.step_index)
        tactic_seq = [ex.ground_truth_tactic for ex in thm_examples if ex.ground_truth_tactic]

        template_name = classify_tactic_sequence(tactic_seq) if tactic_seq else "DECIDE"
        theorem_templates[theorem_id] = template_name
        template_counts[template_name] += 1

        dir_seq = [ex.nav_directions for ex in thm_examples]
        if dir_seq:
            sig = compute_bank_signature(dir_seq)
            template_bank_sigs[template_name].append(sig)

        move_profile = _summarize_step_metadata(thm_examples)
        theorem_move_profiles[theorem_id] = move_profile

        template_stats = template_move_stats[template_name]
        _accumulate_template_move_stats(template_stats, thm_examples)

    return (
        theorem_templates,
        template_counts,
        template_bank_sigs,
        theorem_move_profiles,
        template_move_stats,
    )


def _build_taxonomy(
    template_counts: Counter[str],
    total_theorems: int,
    template_move_stats: dict[str, dict[str, Counter[str] | int]],
) -> dict[str, dict]:
    """Build taxonomy output dict."""
    taxonomy = {}
    for name in TEMPLATE_NAMES:
        count = template_counts.get(name, 0)
        info = TEMPLATE_TAXONOMY[name]
        move_stats = template_move_stats.get(name, {})
        family_counts = move_stats.get("families", Counter())
        subtask_counts = move_stats.get("subtasks", Counter())
        target_head_counts = move_stats.get("target_heads", Counter())
        trigger_kind_counts = move_stats.get("trigger_kinds", Counter())
        trigger_signature_counts = move_stats.get("trigger_signatures", Counter())
        taxonomy[name] = {
            "index": get_template_index(name),
            "pattern": info.pattern,
            "bank_signature": info.bank_signature,
            "tactic_patterns": info.tactic_patterns,
            "is_simple": info.is_simple,
            "count": count,
            "coverage_pct": round(100 * count / total_theorems, 1) if total_theorems > 0 else 0,
            "move_profile": {
                "steps_with_metadata": int(move_stats.get("steps_with_metadata", 0)),
                "top_local_families": _top_counts(family_counts)
                if isinstance(family_counts, Counter)
                else [],
                "top_subtasks": _top_counts(subtask_counts)
                if isinstance(subtask_counts, Counter)
                else [],
                "top_target_heads": _top_counts(target_head_counts)
                if isinstance(target_head_counts, Counter)
                else [],
                "top_trigger_kinds": _top_counts(trigger_kind_counts)
                if isinstance(trigger_kind_counts, Counter)
                else [],
                "top_trigger_signatures": _top_counts(trigger_signature_counts)
                if isinstance(trigger_signature_counts, Counter)
                else [],
            },
        }
    return taxonomy


def extract_templates(data_path: Path, output_dir: Path) -> dict:
    """Extract template labels from training data.

    Groups training examples by theorem_key, classifies each theorem's
    tactic sequence, and produces template taxonomy + augmented data.
    """
    print(f"Loading training data from {data_path}...")
    examples: list[NavigationalExample] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(NavigationalExample.from_dict(json.loads(line)))

    print(f"  Loaded {len(examples)} examples")

    # Group examples by theorem_key (fallback theorem_id for legacy rows)
    theorems: dict[str, list[NavigationalExample]] = {}
    for ex in examples:
        theorems.setdefault(_theorem_group_key(ex), []).append(ex)
    print(f"  {len(theorems)} unique theorems")

    # Classify
    theorem_templates, template_counts, _, theorem_move_profiles, template_move_stats = (
        _classify_theorems(theorems)
    )
    total_theorems = len(theorems)

    # Coverage report
    covered = sum(1 for c in template_counts.values() if c > 0)
    print(f"\nTemplate coverage: {covered}/{len(TEMPLATE_NAMES)} templates used")
    for name in TEMPLATE_NAMES:
        count = template_counts.get(name, 0)
        pct = 100 * count / total_theorems if total_theorems > 0 else 0
        print(f"  {name:25s}: {count:6d} ({pct:5.1f}%)")

    # Write taxonomy
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = _build_taxonomy(template_counts, total_theorems, template_move_stats)
    taxonomy_path = output_dir / "template_taxonomy.json"
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"\nTaxonomy written to {taxonomy_path}")

    # Write augmented training data with template_id
    augmented_path = output_dir / "nav_train_templates.jsonl"
    augmented_count = 0
    with open(augmented_path, "w") as f:
        for ex in examples:
            theorem_key = _theorem_group_key(ex)
            template_name = theorem_templates.get(theorem_key, "DECIDE")
            d = ex.to_dict()
            d["template_id"] = get_template_index(template_name)
            d["template_name"] = template_name
            d["template_move_profile"] = theorem_move_profiles.get(theorem_key, {})
            f.write(json.dumps(d) + "\n")
            augmented_count += 1
    print(f"Augmented data written to {augmented_path} ({augmented_count} examples)")

    return {
        "status": "complete",
        "theorems": total_theorems,
        "examples": len(examples),
        "taxonomy_path": str(taxonomy_path),
        "augmented_path": str(augmented_path),
        "template_counts": dict(template_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract proof strategy templates")
    parser.add_argument("--data", type=Path, default=Path("data/nav_training.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/"))
    args = parser.parse_args()

    result = extract_templates(args.data, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
