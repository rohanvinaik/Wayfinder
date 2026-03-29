from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.benchmark_residuals import augment_result_entry
from src.dr_ducky import GoalCapsule, build_goal_capsule

LOCAL_BUCKETS = {
    "single_goal_near_miss",
    "single_goal_stall",
    "multi_goal_small_progress",
    "multi_goal_large_progress",
}

TARGETED_THEOREMS = [
    "Batteries.UnionFind.rootD_parent",
    "ModularGroup.tendsto_normSq_coprime_pair",
    "ModularGroup.smul_eq_lcRow0_add",
    "ModularGroup.eq_zero_of_mem_fdo_of_T_zpow_mem_fdo",
]

FAMILY_EXPECTATIONS: dict[str, dict[str, set[str]]] = {
    "local_eq_close": {
        "targets": {"human_calculator", "symbolic_sandbox"},
        "banks": {"eq_sat", "transport_normalizer", "hypothesis_injection", "extensionality_bridge"},
        "prescriptions": {"saturate_equality", "normalize_coercions", "inject_hypotheses", "reduce_pointwise"},
    },
    "local_ineq_close": {
        "targets": {"human_calculator", "symbolic_sandbox"},
        "banks": {"arith_nf", "solver_dispatch"},
        "prescriptions": {"normalize_arithmetic", "enter_symbolic_sandbox"},
    },
    "membership_close": {
        "targets": {"membership_surface_engine"},
        "banks": {"membership_exposure", "set_pointwise"},
        "prescriptions": {"expose_membership", "pointwise_set_reduction"},
    },
    "witness_construction_close": {
        "targets": {"witness_engine"},
        "banks": {"witness_constructor", "canonical_witness"},
        "prescriptions": {"construct_witness", "canonical_witness"},
    },
    "forward_context_close": {
        "targets": {"context_transport"},
        "banks": {"context_forward", "diagram_transport"},
        "prescriptions": {"forward_local_context"},
    },
    "forall_close": {
        "targets": {"binder_drilldown"},
        "banks": {"binder_instantiation", "extensionality_bridge"},
        "prescriptions": {"binder_drilldown", "reduce_pointwise"},
    },
    "iff_close": {
        "targets": {"logic_splitter"},
        "banks": {"iff_splitter"},
        "prescriptions": {"split_iff"},
    },
    "subset_close": {
        "targets": {"membership_surface_engine"},
        "banks": {"set_pointwise", "membership_exposure"},
        "prescriptions": {"pointwise_set_reduction", "expose_membership"},
    },
    "atomic_prop_close": {
        "targets": {"atomic_fact_engine", "structural_closer"},
        "banks": {"structural_close", "local_fact_selector"},
        "prescriptions": {"close_before_unpack", "select_local_fact"},
    },
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _has_enabled_bank(capsule: GoalCapsule, names: set[str]) -> bool:
    return any(prior.name in names and not prior.suppressed for prior in capsule.bank_priors)


def _has_prescription(capsule: GoalCapsule, names: set[str]) -> bool:
    return any(prescription.prescription_kind in names for prescription in capsule.prescriptions)


def _has_target(capsule: GoalCapsule, names: set[str]) -> bool:
    return any(target in names for target in capsule.specialist_targets)


def _case_payload(capsule: GoalCapsule) -> dict[str, Any]:
    return {
        "theorem_id": capsule.specification.theorem_id,
        "residual_bucket": capsule.specification.residual_bucket,
        "reasoning_gap_family": capsule.specification.reasoning_gap_family,
        "goal_bucket": capsule.specification.goal_bucket,
        "specialist_targets": capsule.specialist_targets,
        "suppression_hints": capsule.suppression_hints,
        "top_banks": [prior.to_dict() for prior in capsule.bank_priors[:6]],
        "prescriptions": [prescription.to_dict() for prescription in capsule.prescriptions[:6]],
    }


def _family_alignment(capsules: list[GoalCapsule]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, expected in FAMILY_EXPECTATIONS.items():
        subset = [capsule for capsule in capsules if capsule.specification.reasoning_gap_family == family]
        if not subset:
            out[family] = {"count": 0, "aligned": 0, "alignment_rate": 0.0}
            continue
        aligned = [
            capsule
            for capsule in subset
            if _has_target(capsule, expected["targets"])
            and (_has_enabled_bank(capsule, expected["banks"]) or _has_prescription(capsule, expected["prescriptions"]))
        ]
        out[family] = {
            "count": len(subset),
            "aligned": len(aligned),
            "alignment_rate": _pct(len(aligned), len(subset)),
            "examples": [_case_payload(capsule) for capsule in aligned[:2]],
        }
    return out


def _recursive_metrics(capsules: list[GoalCapsule]) -> dict[str, Any]:
    subset = [capsule for capsule in capsules if capsule.specification.signals.get("recursive_loop_risk")]
    success = [
        capsule
        for capsule in subset
        if "recursive_circuit_breaker" in capsule.specialist_targets
        and "suppress_repeat_rw" in capsule.suppression_hints
        and _has_prescription(capsule, {"recursive_loop_circuit_break", "bounded_unfold"})
        and not any(prior.name == "eq_sat" and not prior.suppressed for prior in capsule.bank_priors)
    ]
    return {
        "count": len(subset),
        "routed": len(success),
        "routing_rate": _pct(len(success), len(subset)),
        "examples": [_case_payload(capsule) for capsule in success[:3]],
    }


def _symbolic_sandbox_metrics(capsules: list[GoalCapsule]) -> dict[str, Any]:
    subset = [capsule for capsule in capsules if capsule.specification.signals.get("symbolic_sandbox_candidate")]
    success = [
        capsule
        for capsule in subset
        if "symbolic_sandbox" in capsule.specialist_targets
        and _has_prescription(capsule, {"enter_symbolic_sandbox"})
        and _has_enabled_bank(capsule, {"eq_sat", "transport_normalizer", "arith_nf", "solver_dispatch"})
    ]
    return {
        "count": len(subset),
        "routed": len(success),
        "routing_rate": _pct(len(success), len(subset)),
        "examples": [_case_payload(capsule) for capsule in success[:3]],
    }


def _domain_suppression_metrics(capsules: list[GoalCapsule]) -> dict[str, Any]:
    subset = [
        capsule
        for capsule in capsules
        if (
            "domain_solver_mismatch_risk" in capsule.specification.representation_pressures
            or (
                capsule.specification.signals.get("has_category_surface")
                and not capsule.specification.signals.get("has_numeric_surface")
            )
        )
    ]
    success = [capsule for capsule in subset if "suppress_numeric_solvers" in capsule.suppression_hints]
    return {
        "count": len(subset),
        "suppressed": len(success),
        "suppression_rate": _pct(len(success), len(subset)),
        "examples": [_case_payload(capsule) for capsule in success[:3]],
    }


def _targeted_cases(capsules: list[GoalCapsule]) -> dict[str, Any]:
    by_theorem = {capsule.specification.theorem_id: capsule for capsule in capsules}
    out: dict[str, Any] = {}
    for theorem_id in TARGETED_THEOREMS:
        capsule = by_theorem.get(theorem_id)
        if capsule is not None:
            out[theorem_id] = _case_payload(capsule)
    return out


def build_validation_summary(run_dir: Path) -> dict[str, Any]:
    details = [augment_result_entry(row) for row in _load_jsonl(run_dir / "details.jsonl")]
    local_rows = [row for row in details if str(row.get("residual_bucket", "")) in LOCAL_BUCKETS]
    capsules = [build_goal_capsule(row) for row in local_rows]
    bundle_dir = run_dir / "bundle/dr_ducky"
    bundle_summary = _load_json(bundle_dir / "summary.json")
    closure_validation = (
        _load_json(bundle_dir / "executor_validation_local20_vnext_closure_report.json")
        or _load_json(bundle_dir / "dr_ducky_closure_report.json")
    )
    executor_summary = (
        _load_json(bundle_dir / "executor_validation_local20_vnext_summary.json")
        or _load_json(bundle_dir / "executor_validation.json")
    )
    return {
        "run_dir": str(run_dir),
        "input_rows": len(details),
        "validated_rows": len(local_rows),
        "bundle_summary": bundle_summary,
        "executor_summary": executor_summary,
        "closure_validation": closure_validation,
        "recursive_circuit_breaker": _recursive_metrics(capsules),
        "symbolic_sandbox": _symbolic_sandbox_metrics(capsules),
        "domain_suppression": _domain_suppression_metrics(capsules),
        "family_alignment": _family_alignment(capsules),
        "targeted_cases": _targeted_cases(capsules),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    recursive = summary["recursive_circuit_breaker"]
    symbolic = summary["symbolic_sandbox"]
    domain = summary["domain_suppression"]
    bundle_summary = summary.get("bundle_summary") or {}
    executor_summary = summary.get("executor_summary") or {}
    closure_validation = summary.get("closure_validation") or {}
    lines = [
        "# Dr_Ducky Validation Report",
        "",
        "## Scope",
        "",
        "This report now separates the six validation layers required by the canonical Dr. Ducky architecture:",
        "",
        "1. routing validation",
        "2. theorem-faithful replay validation",
        "3. certificate generation",
        "4. projector compilation",
        "5. honest progress lift",
        "6. honest closure lift",
        "",
        "Run dir:",
        f"- `{summary['run_dir']}`",
        "",
        "## 1. Routing Validation",
        "",
        "Population:",
        f"- input rows: `{summary['input_rows']}`",
        f"- validated local rows: `{summary['validated_rows']}`",
        (f"- local capsules materialized: `{bundle_summary.get('total_capsules', summary['validated_rows'])}`" if bundle_summary else ""),
        "",
        "Key routing checks:",
        f"- recursive circuit-breaker routing: `{recursive['routed']}/{recursive['count']}` (`{recursive['routing_rate']}%`)",
        f"- symbolic sandbox routing: `{symbolic['routed']}/{symbolic['count']}` (`{symbolic['routing_rate']}%`)",
        f"- domain numeric-solver suppression: `{domain['suppressed']}/{domain['count']}` (`{domain['suppression_rate']}%`)",
        "",
        "Family alignment:",
    ]
    for family, result in summary["family_alignment"].items():
        lines.append(
            f"- `{family}`: `{result['aligned']}/{result['count']}` aligned (`{result['alignment_rate']}%`)"
        )
    if executor_summary:
        lines.extend(
            [
                "",
                "## 2. Theorem-Faithful Replay Validation",
                "",
                f"- replay sample rows: `{executor_summary.get('input_rows', executor_summary.get('total_rows', 0))}`",
                f"- theorem-faithful starts: `{executor_summary.get('theorem_faithful_starts', 0)}/{executor_summary.get('started', 0)}`",
            ]
        )
        by_replay_tier = executor_summary.get("by_replay_tier") or {}
        if by_replay_tier:
            lines.append("- replay tier mix:")
            for tier, count in by_replay_tier.items():
                lines.append(f"  - `{tier}`: `{count}`")
    if closure_validation:
        lines.extend(
            [
                "",
                "## 3. Certificate Generation",
                "",
                f"- generated certificates: `{closure_validation.get('certificate_generation_count', 0)}`",
                "",
                "## 4. Projector Compilation",
                "",
                f"- projector successes: `{closure_validation.get('projector_success_count', 0)}`",
                f"- projector rejections: `{closure_validation.get('projector_rejection_count', 0)}`",
                "",
                "## 5. Honest Progress Lift",
                "",
                f"- honest progress: `{closure_validation.get('honest_progress_count', 0)}`",
                f"- Lean compile proxy count: `{closure_validation.get('lean_compile_proxy_count', 0)}`",
                "",
                "## 6. Honest Closure Lift",
                "",
                f"- honest closures: `{closure_validation.get('honest_closure_count', 0)}`",
            ]
        )
    lines.extend(["", "## Targeted Cases", ""])
    for theorem_id, payload in summary["targeted_cases"].items():
        lines.append(f"### `{theorem_id}`")
        lines.append("")
        lines.append(f"- Residual bucket: `{payload['residual_bucket']}`")
        lines.append(f"- Gap family: `{payload['reasoning_gap_family']}`")
        lines.append(f"- Goal bucket: `{payload['goal_bucket']}`")
        lines.append(f"- Specialist targets: `{', '.join(payload['specialist_targets'])}`")
        lines.append(f"- Suppression hints: `{', '.join(payload['suppression_hints'])}`")
        lines.append(f"- Top prescriptions: `{', '.join(item['prescription_kind'] for item in payload['prescriptions'])}`")
        lines.append("")
    cleaned: list[str] = []
    for line in lines:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    summary = build_validation_summary(Path(args.run_dir))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2))
    markdown = render_markdown(summary)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
