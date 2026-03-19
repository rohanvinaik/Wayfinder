#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_THEOREM_RUNS = [
    "rw37_results.json",
    "rw38_results.json",
    "rw39_results_v3.json",
    "rw41_results_v2.json",
    "rw42_results.json",
]

DEFAULT_COMPONENT_RUNS = [
    "simp0_bare_results.jsonl",
    "simp0_hints_results.jsonl",
    "apply0_results.jsonl",
    "refine0_results.jsonl",
]

SUCCESS_SUFFIX = "_success"
ATTEMPTS_SUFFIX = "_attempts"
TIME_SUFFIX = "_time_s"
LANE_SUFFIX = "_close_lane"


def _mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _condition_prefixes(row: dict[str, Any]) -> list[str]:
    return sorted(key[: -len(SUCCESS_SUFFIX)] for key in row if key.endswith(SUCCESS_SUFFIX))


def _activity_fields(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for key, value in row.items():
        if not key.startswith(prefix + "_"):
            continue
        suffix = key[len(prefix) + 1 :]
        if suffix in {"success", "attempts", "time_s", "close_lane"}:
            continue
        fields[suffix] = value
    return fields


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _condition_touched(prefix: str, row: dict[str, Any]) -> bool:
    for value in _activity_fields(prefix, row).values():
        if _is_nonempty(value):
            return True
    return False


def _sum_numeric_activity(prefix: str, row: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, value in _activity_fields(prefix, row).items():
        if isinstance(value, (int, float)):
            totals[key] = float(value)
    return totals


def _theorem_key(row: dict[str, Any]) -> str:
    source = row.get("source", "?")
    theorem = row.get("theorem_id", "?")
    return f"{source}::{theorem}"


def _classify_condition(
    baseline_successes: set[str],
    condition_successes: set[str],
    touched: set[str],
    total_calls: float,
    total_theorems: int,
) -> str:
    marginal_wins = len(condition_successes - baseline_successes)
    regressions = len(baseline_successes - condition_successes)
    calls_per_theorem = total_calls / total_theorems if total_theorems else 0.0
    if regressions > 0:
        return "harmful"
    if marginal_wins > 0:
        return "productive"
    if touched:
        return "supportive" if calls_per_theorem <= 4.0 else "dormant"
    return "inactive"


def analyze_theorem_runs(run_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lane_summary: dict[str, Any] = {"experiments": []}
    theorem_overlap: dict[str, Any] = {}
    transition_graph: dict[str, Any] = {"edges": []}

    edge_weights: Counter[tuple[str, str, str]] = Counter()

    for path in run_paths:
        data = _load_json(path)
        rows = data.get("results", [])
        if not rows:
            continue
        prefixes = _condition_prefixes(rows[0])
        total_theorems = len(rows)
        baseline_prefix = "baseline" if "baseline" in prefixes else prefixes[0]
        baseline_successes = {
            _theorem_key(row) for row in rows if bool(row.get(f"{baseline_prefix}{SUCCESS_SUFFIX}", False))
        }

        experiment_summary: dict[str, Any] = {
            "experiment": data.get("experiment", path.stem),
            "file": str(path),
            "n_theorems": total_theorems,
            "conditions": [],
        }

        for row in rows:
            theorem_overlap[_theorem_key(row)] = theorem_overlap.get(
                _theorem_key(row),
                {
                    "source": row.get("source"),
                    "theorem_id": row.get("theorem_id"),
                    "conditions": {},
                },
            )

        for prefix in prefixes:
            success_key = f"{prefix}{SUCCESS_SUFFIX}"
            attempts_key = f"{prefix}{ATTEMPTS_SUFFIX}"
            time_key = f"{prefix}{TIME_SUFFIX}"
            close_lane_key = f"{prefix}{LANE_SUFFIX}"

            successes = {_theorem_key(row) for row in rows if bool(row.get(success_key, False))}
            touched = {_theorem_key(row) for row in rows if _condition_touched(prefix, row)}
            attempts = [float(row.get(attempts_key, 0) or 0) for row in rows]
            times = [float(row.get(time_key, 0) or 0) for row in rows]
            close_lanes = Counter(
                row.get(close_lane_key, "failed") or "failed"
                for row in rows
                if bool(row.get(success_key, False))
            )

            numeric_activity = Counter()
            for row in rows:
                for key, value in _sum_numeric_activity(prefix, row).items():
                    numeric_activity[key] += value

            condition_summary = {
                "name": prefix,
                "proved": len(successes),
                "rate": round(len(successes) / total_theorems, 4),
                "marginal_wins_vs_baseline": len(successes - baseline_successes),
                "regressions_vs_baseline": len(baseline_successes - successes),
                "touched_theorems": len(touched),
                "mean_attempts": _mean(attempts),
                "mean_time_s": _mean(times),
                "close_lanes": dict(close_lanes),
                "activity_totals": dict(numeric_activity),
            }
            condition_summary["role"] = _classify_condition(
                baseline_successes,
                successes,
                touched,
                sum(attempts),
                total_theorems,
            )
            experiment_summary["conditions"].append(condition_summary)

            for row in rows:
                theorem_overlap[_theorem_key(row)]["conditions"][prefix] = {
                    "success": bool(row.get(success_key, False)),
                    "close_lane": row.get(close_lane_key),
                    "attempts": row.get(attempts_key, 0),
                    "time_s": row.get(time_key, 0),
                    "touched": _condition_touched(prefix, row),
                    "activity": _activity_fields(prefix, row),
                }

                if bool(row.get(success_key, False)):
                    final_lane = row.get(close_lane_key, "failed") or "failed"
                    if _condition_touched(prefix, row):
                        edge_weights[(prefix, final_lane, "success")] += 1
                elif _condition_touched(prefix, row):
                    edge_weights[(prefix, "failed", "touched_failed")] += 1

        lane_summary["experiments"].append(experiment_summary)

    for (src, dst, kind), weight in sorted(edge_weights.items()):
        transition_graph["edges"].append(
            {"from": src, "to": dst, "kind": kind, "weight": weight}
        )

    return lane_summary, theorem_overlap, transition_graph


def _component_condition_fields(row: dict[str, Any]) -> list[str]:
    names = []
    for key in row:
        if key.endswith("_accepted"):
            names.append(key[: -len("_accepted")])
    return sorted(set(names))


def analyze_component_runs(run_paths: list[Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {"benchmarks": []}
    for path in run_paths:
        rows = _load_jsonl(path)
        if not rows:
            continue
        sample = rows[0]
        conditions = _component_condition_fields(sample)
        subsets = sorted({row.get("subset", "all") for row in rows})
        benchmark = {
            "file": str(path),
            "rows": len(rows),
            "conditions": [],
            "subsets": [],
        }

        for subset in subsets:
            subset_rows = [row for row in rows if row.get("subset", "all") == subset]
            started = [row for row in subset_rows if row.get("goal_started")]
            started_n = len(started)
            subset_summary = {
                "subset": subset,
                "rows": len(subset_rows),
                "started": started_n,
                "start_rate": round(started_n / len(subset_rows), 4) if subset_rows else 0.0,
            }
            if any("gold_in_scope" in row for row in subset_rows):
                gold_vals = [bool(row.get("gold_in_scope")) for row in started if "gold_in_scope" in row]
                subset_summary["gold_in_scope_rate"] = (
                    round(sum(gold_vals) / len(gold_vals), 4) if gold_vals else None
                )
            if any("gold_hints_total" in row for row in subset_rows):
                hints_total = [int(row.get("gold_hints_total", 0) or 0) for row in started]
                hints_scope = [int(row.get("gold_hints_in_scope", 0) or 0) for row in started]
                subset_summary["gold_hint_recall_mean"] = (
                    round(sum(hints_scope) / sum(hints_total), 4) if sum(hints_total) else 0.0
                )
            benchmark["subsets"].append(subset_summary)

        for condition in conditions:
            accepted_key = f"{condition}_accepted"
            closed_key = f"{condition}_closed"
            accepted_rows = [row for row in rows if row.get("goal_started")]
            accepted = sum(bool(row.get(accepted_key)) for row in accepted_rows)
            closed = (
                sum(bool(row.get(closed_key)) for row in accepted_rows)
                if any(closed_key in row for row in rows)
                else None
            )
            summary_row = {
                "condition": condition,
                "accepted_started": accepted,
                "accepted_rate": round(accepted / len(accepted_rows), 4) if accepted_rows else 0.0,
            }
            if closed is not None:
                summary_row["closed_started"] = closed
                summary_row["closed_rate"] = round(closed / len(accepted_rows), 4) if accepted_rows else 0.0
            benchmark["conditions"].append(summary_row)

        summary["benchmarks"].append(benchmark)

    return summary


def build_findings(
    lane_summary: dict[str, Any],
    component_summary: dict[str, Any],
) -> str:
    productive = []
    supportive = []
    dormant = []
    for experiment in lane_summary["experiments"]:
        for condition in experiment["conditions"]:
            entry = (
                experiment["experiment"],
                condition["name"],
                condition["marginal_wins_vs_baseline"],
                condition["touched_theorems"],
                condition["mean_attempts"],
            )
            if condition["role"] == "productive":
                productive.append(entry)
            elif condition["role"] == "supportive":
                supportive.append(entry)
            elif condition["role"] == "dormant":
                dormant.append(entry)

    lines = [
        "# Meta Findings",
        "",
        "## Theorem-Search Lane Taxonomy",
    ]
    if productive:
        lines.append("- Productive lanes:")
        for experiment, name, wins, touched, attempts in productive:
            lines.append(
                f"  - `{experiment}` / `{name}`: +{wins} marginal theorem wins, {touched} touched theorems, {attempts:.2f} mean attempts."
            )
    if supportive:
        lines.append("- Supportive lanes:")
        for experiment, name, wins, touched, attempts in supportive:
            lines.append(
                f"  - `{experiment}` / `{name}`: {touched} touched theorems, +{wins} wins, {attempts:.2f} mean attempts."
            )
    if dormant:
        lines.append("- Dormant lanes:")
        for experiment, name, wins, touched, attempts in dormant:
            lines.append(
                f"  - `{experiment}` / `{name}`: {touched} touched theorems, +{wins} wins, {attempts:.2f} mean attempts."
            )

    lines.extend(
        [
            "",
            "## Component Benchmarks",
            "- Rewrite is the only currently productive theorem-winning family in search runs.",
            "- Interleaved bootstrap behaves as a systems baseline: active and cheaper, but not a new theorem-winning family.",
            "- `simp` is supportive/helper: real subgoal progress, no theorem-level lift yet.",
            "- `apply` is component-real but theorem-dormant on the current 50-theorem slice.",
            "- `refine` reads as a reranking/planning problem rather than a deployable cosine executor lane.",
            "",
            "## Theory-Level Direction",
            "- Promote a lane taxonomy: finisher (`rw`), scaffolder (`interleaved_bootstrap`), transformer/helper (`simp`), dormant specialist (`apply`).",
            "- Shift from globally enabled lanes toward residual-conditioned orchestration.",
            "- Treat `refine` as a typed reranking / skeleton-selection target instead of another always-on executor.",
            "",
            "## Benchmark Snapshots",
        ]
    )
    for benchmark in component_summary["benchmarks"]:
        best = max(benchmark["conditions"], key=lambda row: row.get("accepted_rate", 0.0))
        lines.append(
            f"- `{Path(benchmark['file']).name}`: best accepted-started condition is `{best['condition']}` at {best['accepted_rate']:.1%}."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run meta-analysis for local family experiments.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-prefix", type=Path, default=Path("runs/meta"))
    parser.add_argument("--theorem-runs", nargs="*", default=DEFAULT_THEOREM_RUNS)
    parser.add_argument("--component-runs", nargs="*", default=DEFAULT_COMPONENT_RUNS)
    args = parser.parse_args()

    theorem_paths = [args.runs_dir / name for name in args.theorem_runs if (args.runs_dir / name).exists()]
    component_paths = [args.runs_dir / name for name in args.component_runs if (args.runs_dir / name).exists()]

    lane_summary, theorem_overlap, transition_graph = analyze_theorem_runs(theorem_paths)
    component_summary = analyze_component_runs(component_paths)
    findings = build_findings(lane_summary, component_summary)

    outputs = {
        args.output_prefix.with_name(args.output_prefix.name + "_lane_summary.json"): lane_summary,
        args.output_prefix.with_name(args.output_prefix.name + "_component_summary.json"): component_summary,
        args.output_prefix.with_name(args.output_prefix.name + "_theorem_overlap.json"): theorem_overlap,
        args.output_prefix.with_name(args.output_prefix.name + "_transition_graph.json"): transition_graph,
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    findings_path = args.output_prefix.with_name(args.output_prefix.name + "_findings.md")
    findings_path.write_text(findings)

    print("Meta analysis complete")
    print(f"  theorem runs:   {len(theorem_paths)}")
    print(f"  component runs: {len(component_paths)}")
    for path in list(outputs) + [findings_path]:
        print(f"  wrote: {path}")


if __name__ == "__main__":
    main()
