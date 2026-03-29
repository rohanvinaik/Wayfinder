"""Build a minimal theorem-stratified SoM refresh corpus after compiler repairs.

This script reuses the broad original multi-step SoM corpus, but concentrates
additional mass on theorem families that were previously underrepresented due
to compiler/startability failures and on the currently observed hard residual
surface from the formal benchmark.

Outputs:
  - <output>/train.jsonl
  - <output>/eval.jsonl
  - <output>/train_features.npz
  - <output>/eval_features.npz
  - <output>/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.train_som_multistep import SPECIALIST_TO_IDX, _shape_to_vec, _step_context


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _stable_fraction(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _top_namespace(theorem_id: str) -> str:
    if not theorem_id:
        return "unknown"
    return theorem_id.split(".", 1)[0]


def _select_diverse_steps(
    theorem_rows: list[dict[str, Any]],
    *,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        theorem_rows,
        key=lambda row: (
            int(row.get("step_index", 0) or 0),
            str(row.get("specialist", "") or ""),
            str(row.get("tactic", "") or ""),
        ),
    )
    if len(rows) <= max_steps:
        return rows

    chosen: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()

    def add_row(row: dict[str, Any]) -> None:
        key = (int(row.get("step_index", 0) or 0), str(row.get("specialist", "") or ""))
        if key in seen_keys:
            return
        seen_keys.add(key)
        chosen.append(row)

    add_row(rows[0])
    add_row(rows[-1])

    first_by_specialist: dict[str, dict[str, Any]] = {}
    for row in rows:
        specialist = str(row.get("specialist", "") or "structural")
        first_by_specialist.setdefault(specialist, row)
    for specialist in ["rewrite", "structural", "solver", "apply", "closer"]:
        row = first_by_specialist.get(specialist)
        if row is not None and len(chosen) < max_steps:
            add_row(row)

    quantiles = [0.25, 0.5, 0.75]
    for q in quantiles:
        if len(chosen) >= max_steps:
            break
        idx = min(len(rows) - 1, max(0, int(round((len(rows) - 1) * q))))
        add_row(rows[idx])

    if len(chosen) < max_steps:
        step_values = [int(r.get("step_index", 0) or 0) for r in rows]
        ranked = sorted(
            rows,
            key=lambda row: (
                -abs((int(row.get("step_index", 0) or 0)) - (sum(step_values) / max(len(step_values), 1))),
                str(row.get("specialist", "") or ""),
            ),
        )
        for row in ranked:
            if len(chosen) >= max_steps:
                break
            add_row(row)

    return sorted(chosen[:max_steps], key=lambda row: int(row.get("step_index", 0) or 0))


def _collect_repair_theorem_ids(
    *,
    failure_paths: list[Path],
    benchmark_details: Path | None,
) -> tuple[set[str], set[str]]:
    repair_ids: set[str] = set()
    hard_ids: set[str] = set()

    for path in failure_paths:
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            theorem_id = str(row.get("theorem_full_name") or row.get("theorem_id") or "")
            if theorem_id:
                repair_ids.add(theorem_id)

    if benchmark_details is not None and benchmark_details.exists():
        for row in _load_jsonl(benchmark_details):
            theorem_id = str(row.get("theorem_full_name") or row.get("theorem_id") or "")
            if not theorem_id:
                continue
            difficulty = str(row.get("difficulty_bucket") or "")
            follow_on = str(row.get("follow_on_stage") or "")
            gap = str(row.get("reasoning_gap_family") or "")
            success = bool(row.get("honest_success") or row.get("success"))
            if (
                difficulty == "compiler_or_startability"
                or follow_on == "compiler_specialist"
                or gap == "compiler_specialist"
            ):
                repair_ids.add(theorem_id)
            if not success and difficulty not in {"resolved", "compiler_or_startability"}:
                hard_ids.add(theorem_id)

    return repair_ids, hard_ids


def _load_multistep_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_jsonl(path))
    return rows


def _group_by_theorem(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        theorem_id = str(row.get("theorem_full_name") or "")
        if theorem_id:
            grouped[theorem_id].append(row)
    return grouped


def _sample_background_theorems(
    *,
    theorem_ids: list[str],
    namespace_of: dict[str, str],
    focus_namespace_counts: Counter[str],
    desired_count: int,
    rng: random.Random,
) -> list[str]:
    if desired_count <= 0 or not theorem_ids:
        return []

    by_namespace: dict[str, list[str]] = defaultdict(list)
    for theorem_id in theorem_ids:
        by_namespace[namespace_of.get(theorem_id, "unknown")].append(theorem_id)

    selected: list[str] = []
    focus_total = sum(focus_namespace_counts.values())
    if focus_total > 0:
        for namespace, count in focus_namespace_counts.items():
            pool = by_namespace.get(namespace, [])
            if not pool:
                continue
            quota = max(1, round(desired_count * (count / focus_total)))
            take = min(len(pool), quota)
            selected.extend(rng.sample(pool, take))

    if len(selected) < desired_count:
        remaining = [tid for tid in theorem_ids if tid not in set(selected)]
        if remaining:
            take = min(len(remaining), desired_count - len(selected))
            selected.extend(rng.sample(remaining, take))

    rng.shuffle(selected)
    return selected[:desired_count]


def _target_text(row: dict[str, Any]) -> str:
    target = str(row.get("goal_target") or "")
    if target:
        return target[:300]
    goal_state = str(row.get("goal_state_before") or "")
    if "⊢" in goal_state:
        return goal_state.split("⊢")[-1].strip()[:300]
    return goal_state[:300]


def _build_feature_arrays(
    rows: list[dict[str, Any]],
    *,
    embed_cache: dict[str, np.ndarray],
    encoder: SentenceTransformer | None,
) -> dict[str, np.ndarray]:
    n = len(rows)
    goal_emb = np.zeros((n, 384), dtype=np.float32)
    goal_shape = np.zeros((n, 12), dtype=np.float32)
    step_context = np.zeros((n, 4), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int32)

    missing_targets: list[str] = []
    missing_indices: list[int] = []
    for i, row in enumerate(rows):
        target = _target_text(row)
        emb = embed_cache.get(target)
        if emb is None:
            missing_targets.append(target)
            missing_indices.append(i)
        else:
            goal_emb[i] = emb
        goal_shape[i] = _shape_to_vec(row.get("goal_shape", {}))
        step_context[i] = _step_context(row)
        labels[i] = SPECIALIST_TO_IDX.get(str(row.get("specialist") or "structural"), 1)

    if missing_targets:
        if encoder is None:
            raise RuntimeError(f"Missing {len(missing_targets)} goal embeddings and no encoder available")
        fresh = encoder.encode(missing_targets, normalize_embeddings=True)
        for idx, emb, target in zip(missing_indices, fresh, missing_targets, strict=False):
            goal_emb[idx] = emb
            embed_cache[target] = np.asarray(emb, dtype=np.float32)

    return {
        "goal_emb": goal_emb,
        "goal_shape": goal_shape,
        "step_context": step_context,
        "labels": labels,
    }


def build_delta_corpus(
    *,
    train_sources: list[Path],
    eval_sources: list[Path],
    failure_paths: list[Path],
    benchmark_details: Path | None,
    output_dir: Path,
    seed: int,
    max_steps_per_theorem: int,
    repair_dup: int,
    hard_dup: int,
    matched_background_theorems: int,
    random_background_theorems: int,
    eval_fraction: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    repair_ids, hard_ids = _collect_repair_theorem_ids(
        failure_paths=failure_paths,
        benchmark_details=benchmark_details,
    )

    all_rows = _load_multistep_rows(train_sources + eval_sources)
    theorem_rows = _group_by_theorem(all_rows)
    available_ids = set(theorem_rows)

    repair_ids &= available_ids
    hard_ids &= available_ids
    focus_ids = repair_ids | hard_ids

    namespace_of = {tid: _top_namespace(tid) for tid in available_ids}
    focus_namespace_counts = Counter(namespace_of[tid] for tid in focus_ids)

    background_pool = [tid for tid in available_ids if tid not in focus_ids]
    matched_bg = _sample_background_theorems(
        theorem_ids=background_pool,
        namespace_of=namespace_of,
        focus_namespace_counts=focus_namespace_counts,
        desired_count=matched_background_theorems,
        rng=rng,
    )
    matched_bg_set = set(matched_bg)
    random_pool = [tid for tid in background_pool if tid not in matched_bg_set]
    random_bg = rng.sample(
        random_pool,
        min(len(random_pool), random_background_theorems),
    ) if random_background_theorems > 0 else []
    background_ids = matched_bg + random_bg

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for theorem_id in sorted(focus_ids | set(background_ids)):
        split = "eval" if _stable_fraction(theorem_id) < eval_fraction else "train"
        rows = _select_diverse_steps(theorem_rows[theorem_id], max_steps=max_steps_per_theorem)
        multiplier = 1
        if theorem_id in repair_ids:
            multiplier = max(multiplier, repair_dup)
        elif theorem_id in hard_ids:
            multiplier = max(multiplier, hard_dup)

        target = eval_rows if split == "eval" else train_rows
        for _ in range(multiplier):
            for row in rows:
                out = dict(row)
                out["delta_bucket"] = (
                    "repair" if theorem_id in repair_ids
                    else "hard_current" if theorem_id in hard_ids
                    else "background"
                )
                out["delta_namespace"] = namespace_of.get(theorem_id, "unknown")
                out["delta_split"] = split
                target.append(out)

    embed_cache: dict[str, np.ndarray] = {}
    emb_npz = np.load("/Users/rohanvinaik/Projects/Wayfinder/data/som_goal_embeddings.npz", allow_pickle=True)
    targets = emb_npz["targets"]
    embeddings = emb_npz["embeddings"]
    for target, emb in zip(targets, embeddings, strict=False):
        embed_cache[str(target)] = np.asarray(emb, dtype=np.float32)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_features = _build_feature_arrays(train_rows, embed_cache=embed_cache, encoder=encoder)
    eval_features = _build_feature_arrays(eval_rows, embed_cache=embed_cache, encoder=encoder)

    train_jsonl = output_dir / "train.jsonl"
    eval_jsonl = output_dir / "eval.jsonl"
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(eval_jsonl, eval_rows)
    np.savez_compressed(output_dir / "train_features.npz", **train_features)
    np.savez_compressed(output_dir / "eval_features.npz", **eval_features)

    summary = {
        "seed": seed,
        "repair_theorems_available": len(repair_ids),
        "hard_theorems_available": len(hard_ids),
        "focus_theorems_total": len(focus_ids),
        "matched_background_theorems": len(matched_bg),
        "random_background_theorems": len(random_bg),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_theorems": len({row["theorem_full_name"] for row in train_rows}),
        "eval_theorems": len({row["theorem_full_name"] for row in eval_rows}),
        "train_buckets": dict(Counter(str(row.get("delta_bucket") or "") for row in train_rows)),
        "eval_buckets": dict(Counter(str(row.get("delta_bucket") or "") for row in eval_rows)),
        "train_specialists": dict(Counter(str(row.get("specialist") or "") for row in train_rows)),
        "eval_specialists": dict(Counter(str(row.get("specialist") or "") for row in eval_rows)),
        "focus_namespace_counts": dict(focus_namespace_counts),
        "top_train_namespaces": Counter(str(row.get("delta_namespace") or "") for row in train_rows).most_common(20),
        "top_eval_namespaces": Counter(str(row.get("delta_namespace") or "") for row in eval_rows).most_common(20),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", action="append", default=[
        "data/som_multistep_train.jsonl",
    ])
    parser.add_argument("--eval-source", action="append", default=[
        "data/som_multistep_eval.jsonl",
    ])
    parser.add_argument("--failure-path", action="append", default=[
        "runs/exp_som012_hard_eval_r2/goal_start_failures.jsonl",
        "runs/exp_som016_final_random2000_r1/goal_start_failures.jsonl",
        "runs/exp_som016_final_random2000_r2_stale_trace_semantics/goal_start_failures.jsonl",
    ])
    parser.add_argument("--benchmark-details", default="runs/exp_som016_final_random2000_r5_inline_bridge/details.jsonl")
    parser.add_argument("--output-dir", default="data/som_repair_delta_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-theorem", type=int, default=6)
    parser.add_argument("--repair-dup", type=int, default=3)
    parser.add_argument("--hard-dup", type=int, default=2)
    parser.add_argument("--matched-background-theorems", type=int, default=1800)
    parser.add_argument("--random-background-theorems", type=int, default=900)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()

    summary = build_delta_corpus(
        train_sources=[Path(p) for p in args.train_source],
        eval_sources=[Path(p) for p in args.eval_source],
        failure_paths=[Path(p) for p in args.failure_path],
        benchmark_details=Path(args.benchmark_details) if args.benchmark_details else None,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        max_steps_per_theorem=args.max_steps_per_theorem,
        repair_dup=args.repair_dup,
        hard_dup=args.hard_dup,
        matched_background_theorems=args.matched_background_theorems,
        random_background_theorems=args.random_background_theorems,
        eval_fraction=args.eval_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
