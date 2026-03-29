"""Build a deterministic hard-theorem train/eval split from theorem narratives.

The goal is to create a harder training and benchmark corpus for the staged
hard-proof program using theorem-level complexity signals already present in the
proof corpus.

Hardness currently follows a simple, auditable rule:
- primary score: proof step count
- secondary score: unique premises
- banding by quantiles over theorem narratives
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hard_data_tags import canonicalize_theorem_id


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(max(int(q * (len(ordered) - 1)), 0), len(ordered) - 1)
    return float(ordered[idx])


def _hash_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _module_from_file_path(file_path: str) -> str:
    text = (file_path or "").strip()
    if not text:
        return ""
    if text.endswith(".lean"):
        text = text[:-5]
    return text.replace("/", ".")


def _load_metadata_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    index: dict[str, dict[str, str]] = {}
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            theorem_id = canonicalize_theorem_id(str(row.get("theorem_id", "")))
            if not theorem_id:
                continue
            file_path = str(row.get("file_path", "") or "")
            index[theorem_id] = {
                "file_path": file_path,
                "module": _module_from_file_path(file_path),
            }
    return index


def build_hard_theorem_split(
    input_path: Path,
    output_dir: Path,
    hard_quantile: float = 0.5,
    eval_fraction: float = 0.2,
    min_steps: int = 0,
    metadata_index_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(input_path)
    metadata_index = _load_metadata_index(metadata_index_path)
    scored: list[dict[str, Any]] = []
    metadata_hits = 0

    step_values: list[float] = []
    for row in rows:
        history = row.get("proof_history_summary", {})
        proof_steps = int(history.get("total_steps", 0) or 0)
        unique_premises = int(history.get("unique_premises", 0) or 0)
        complexity_score = math.exp(min(proof_steps, 12))
        theorem_id = canonicalize_theorem_id(str(row.get("theorem_id", "")))
        metadata = metadata_index.get(theorem_id, {})
        if metadata:
            metadata_hits += 1
        scored_row = {
            "theorem_id": theorem_id,
            "namespace_prefix": row.get("namespace_prefix", ""),
            "theorem_statement": row.get("theorem_statement", ""),
            "template_id": row.get("template_id", ""),
            "proof_steps": proof_steps,
            "unique_premises": unique_premises,
            "complexity_score_exp_steps": complexity_score,
            "file_path": row.get("file_path", "") or metadata.get("file_path", ""),
            "module": row.get("module", "") or metadata.get("module", ""),
        }
        scored.append(scored_row)
        step_values.append(float(proof_steps))

    median_steps = _quantile(step_values, 0.5)
    q75_steps = _quantile(step_values, 0.75)
    q90_steps = _quantile(step_values, 0.9)
    hard_threshold = max(_quantile(step_values, hard_quantile), float(min_steps))

    for row in scored:
        proof_steps = float(row["proof_steps"])
        if proof_steps >= q90_steps:
            difficulty_band = "expert"
        elif proof_steps >= q75_steps:
            difficulty_band = "hard"
        elif proof_steps >= median_steps:
            difficulty_band = "medium"
        else:
            difficulty_band = "easy"
        row["difficulty_band"] = difficulty_band
        row["hard_half"] = proof_steps >= hard_threshold
        row["split"] = "eval" if _hash_fraction(str(row["theorem_id"])) < eval_fraction else "train"

    hard_rows = [row for row in scored if row["hard_half"]]
    hard_train = [row for row in hard_rows if row["split"] == "train"]
    hard_eval = [row for row in hard_rows if row["split"] == "eval"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [
        ("hard_theorem_scores.jsonl", scored),
        ("hard_theorems_all.jsonl", hard_rows),
        ("hard_theorems_train.jsonl", hard_train),
        ("hard_theorems_eval.jsonl", hard_eval),
    ]:
        with (output_dir / name).open("w") as handle:
            for row in subset:
                handle.write(json.dumps(row) + "\n")

    summary = {
        "input_path": str(input_path),
        "total_theorems": len(scored),
        "hard_quantile": hard_quantile,
        "min_steps": min_steps,
        "median_steps": median_steps,
        "q75_steps": q75_steps,
        "q90_steps": q90_steps,
        "hard_threshold_steps": hard_threshold,
        "hard_total": len(hard_rows),
        "hard_train": len(hard_train),
        "hard_eval": len(hard_eval),
        "eval_fraction": eval_fraction,
        "metadata_index_path": str(metadata_index_path) if metadata_index_path is not None else "",
        "metadata_hits": metadata_hits,
        "metadata_hit_rate": round(metadata_hits / max(len(scored), 1), 4),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/template_narrative_train_som.jsonl",
        help="Theorem-level narrative dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="data/hard_split_som",
        help="Directory to write difficulty annotations and hard splits",
    )
    parser.add_argument(
        "--hard-quantile",
        type=float,
        default=0.5,
        help="Quantile threshold for the hard set (default: top half)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of hard theorems assigned to eval",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Optional proof-step floor for the hard set",
    )
    parser.add_argument(
        "--metadata-index",
        default="data/proof_network_entities_v3.jsonl",
        help="Optional theorem metadata index used to attach file_path/module",
    )
    args = parser.parse_args()

    summary = build_hard_theorem_split(
        Path(args.input),
        Path(args.output_dir),
        hard_quantile=args.hard_quantile,
        eval_fraction=args.eval_fraction,
        min_steps=args.min_steps,
        metadata_index_path=Path(args.metadata_index) if args.metadata_index else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
