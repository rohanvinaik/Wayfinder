"""Extract proof strategy templates from the navigational training corpus.

Reads nav_train.jsonl, classifies each proof's tactic sequence into
a template, computes template statistics, and outputs:
  - data/template_taxonomy.json: template definitions + counts
  - data/nav_train_templates.jsonl: augmented training data with template_id

Usage:
    python -m scripts.extract_templates --data data/nav_train.jsonl --output-dir data/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.nav_contracts import NavigationalExample
from src.story_templates import (
    TEMPLATE_NAMES,
    TEMPLATE_TAXONOMY,
    classify_tactic_sequence,
    compute_bank_signature,
    get_template_index,
)


def _classify_theorems(
    theorems: dict[str, list[NavigationalExample]],
) -> tuple[dict[str, str], Counter[str], dict[str, list[dict[str, int]]]]:
    """Classify each theorem's tactic sequence into a template."""
    theorem_templates: dict[str, str] = {}
    template_counts: Counter[str] = Counter()
    template_bank_sigs: dict[str, list[dict[str, int]]] = {name: [] for name in TEMPLATE_NAMES}

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

    return theorem_templates, template_counts, template_bank_sigs


def _build_taxonomy(template_counts: Counter[str], total_theorems: int) -> dict[str, dict]:
    """Build taxonomy output dict."""
    taxonomy = {}
    for name in TEMPLATE_NAMES:
        count = template_counts.get(name, 0)
        info = TEMPLATE_TAXONOMY[name]
        taxonomy[name] = {
            "index": get_template_index(name),
            "pattern": info.pattern,
            "bank_signature": info.bank_signature,
            "tactic_patterns": info.tactic_patterns,
            "is_simple": info.is_simple,
            "count": count,
            "coverage_pct": round(100 * count / total_theorems, 1) if total_theorems > 0 else 0,
        }
    return taxonomy


def extract_templates(data_path: Path, output_dir: Path) -> dict:
    """Extract template labels from training data.

    Groups training examples by theorem_id, classifies each theorem's
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

    # Group examples by theorem_id
    theorems: dict[str, list[NavigationalExample]] = {}
    for ex in examples:
        theorems.setdefault(ex.theorem_id, []).append(ex)
    print(f"  {len(theorems)} unique theorems")

    # Classify
    theorem_templates, template_counts, _ = _classify_theorems(theorems)
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
    taxonomy = _build_taxonomy(template_counts, total_theorems)
    taxonomy_path = output_dir / "template_taxonomy.json"
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"\nTaxonomy written to {taxonomy_path}")

    # Write augmented training data with template_id
    augmented_path = output_dir / "nav_train_templates.jsonl"
    augmented_count = 0
    with open(augmented_path, "w") as f:
        for ex in examples:
            template_name = theorem_templates.get(ex.theorem_id, "DECIDE")
            d = ex.to_dict()
            d["template_id"] = get_template_index(template_name)
            d["template_name"] = template_name
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
    parser.add_argument("--data", type=Path, default=Path("data/nav_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/"))
    args = parser.parse_args()

    result = extract_templates(args.data, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
