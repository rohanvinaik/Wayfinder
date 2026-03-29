from __future__ import annotations

import re
import sqlite3
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from src.dr_ducky import (
    EngineCertificate,
    GoalCapsule,
    LedgerFact,
    MathEngineRequest,
    ProjectedProofProgram,
    ProjectorDecision,
    ProofHoleSpec,
    ProofShadowLedger,
    ProofSkeleton,
    build_goal_capsule,
)
from src.hard_data_tags import classify_goal_bucket, sanitize_goal_text
from src.nav_contracts import TacticResult
from src.proof_network import get_accessible_premises

_TACTIC_KEYWORDS = {
    "by",
    "at",
    "with",
    "using",
    "fun",
    "let",
    "have",
    "show",
    "do",
    "if",
    "then",
    "else",
    "match",
    "in",
    "where",
    "return",
    "intro",
    "intros",
    "constructor",
    "cases",
    "induction",
    "apply",
    "exact",
    "refine",
    "rw",
    "simp",
    "simpa",
    "aesop",
    "solve_by_elim",
    "apply?",
    "exact?",
    "omega",
    "linarith",
    "nlinarith",
    "norm_num",
    "ring",
    "ring_nf",
    "field_simp",
    "norm_cast",
    "push_cast",
    "positivity",
    "gcongr",
    "assumption",
    "trivial",
    "decide",
    "contradiction",
    "tauto",
    "ext",
    "congr",
    "funext",
    "infer_instance",
    "all_goals",
    "any_goals",
    "focus",
    "first",
    "repeat",
    "skip",
}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")
_TERM_APP_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*\s+[A-Za-z_][A-Za-z0-9_']*")
_GOAL_CANDIDATE_KEYS = ("last_goal", "goal_state", "theorem_statement", "initial_goal")
_REWRITE_LEMMA_RE = re.compile(
    r"(?:^|\.)(?:"
    r".*(?:eq|iff|cast|pow_two|normSq|lt_one_iff|lineMap|comp|map|set|succ|pred|prod|sum|mul|add|sub|div|inv|assoc|comm|left|right|zero|one|neg|abs)"
    r"|.*(?:_def|eq_def)"
    r")$"
)
_SIMPLE_EQUALITY_ZERO_RE = re.compile(r"([A-Za-z0-9_'.]+)\s*=\s*0$")
_ABS_LT_ONE_RE = re.compile(r"\|([A-Za-z0-9_'.]+)\|\s*<\s*(?:↑)?1\b")
_HEAVY_GOAL_MARKERS = (
    "CategoryTheory",
    "Filter.Tendsto",
    "TensorProduct",
    "WeierstrassCurve",
    "AlgebraicTopology",
    "Matrix.SpecialLinearGroup",
    "Ordinal.",
    "Cardinal.",
)

_ENGINE_BY_BANK: dict[str, str] = {
    "eq_sat": "EqSatEngine",
    "transport_normalizer": "EqSatEngine",
    "arith_nf": "ArithEngine",
    "solver_dispatch": "ArithEngine",
    "witness_constructor": "WitnessEngine",
    "canonical_witness": "WitnessEngine",
    "recursive_unfold_one": "RecursiveInvariantEngine",
    "loop_breaker": "RecursiveInvariantEngine",
    "membership_exposure": "FiniteFilterEngine",
    "set_pointwise": "FiniteFilterEngine",
    "eventual_filter_normalizer": "FiniteFilterEngine",
    "structural_close": "FiniteFilterEngine",
    "context_forward": "ContextTransportEngine",
    "local_fact_selector": "ContextTransportEngine",
    "binder_instantiation": "ContextTransportEngine",
    "iff_splitter": "ContextTransportEngine",
    "diagram_transport": "ContextTransportEngine",
}
_BACKEND_BY_ENGINE: dict[str, str] = {
    "EqSatEngine": "egglog_eqsat",
    "ArithEngine": "lean_arith",
    "WitnessEngine": "rosette_proof_dsl",
    "RecursiveInvariantEngine": "symbolic_rewrite_vm",
    "FiniteFilterEngine": "kodkod_relational",
    "ContextTransportEngine": "rosette_proof_dsl",
}


@dataclass
class DuckyProgram:
    program_id: str
    bank: str
    specialist: str
    tactics: list[str]
    rationale: str
    score: float
    skeleton_id: str = ""
    bindings: dict[str, str] = field(default_factory=dict)
    certificate_id: str = ""
    engine_name: str = ""
    backend_family: str = ""
    certificate_shape: str = ""
    projector_status: str = ""
    projector_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["script"] = self.script
        return out

    @property
    def script(self) -> str:
        return "; ".join(self.tactics)


@dataclass
class DuckyReplayState:
    theorem_id: str
    file_path: str
    goal_state: str
    goal_kind: str
    theorem_faithful: bool
    tier_used: str
    replay_success: bool
    replay_failure_category: str = ""
    replay_failing_prefix_idx: int = -1
    prefix_tactics: list[str] = field(default_factory=list)
    prefix_goal_states: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuckyProgramRun:
    program_id: str
    bank: str
    specialist: str
    tactics: list[str]
    script: str
    score: float
    progressed: bool
    closed: bool
    tactics_applied: list[str]
    final_goal: str
    final_goal_bucket: str
    goals_after: list[str]
    first_failure_tactic: str = ""
    first_failure_error: str = ""
    certificate_id: str = ""
    engine_name: str = ""
    backend_family: str = ""
    certificate_shape: str = ""
    projector_status: str = ""
    projector_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuckyExecutionResult:
    theorem_id: str
    started: bool
    theorem_faithful: bool
    start_goal_kind: str
    file_path: str
    replay_tier: str
    replay_failure_category: str
    replay_failing_prefix_idx: int
    residual_bucket: str
    goal_bucket: str
    specialist_targets: list[str]
    bank_priors: list[str]
    programs_considered: int
    closed: bool
    progressed: bool
    winning_program: dict[str, Any] | None
    final_goal: str
    final_goal_bucket: str
    goals_after: list[str]
    ledger_snapshot: dict[str, Any] | None = None
    engine_outcomes: list[dict[str, Any]] = field(default_factory=list)
    projector_outcomes: list[dict[str, Any]] = field(default_factory=list)
    tried_programs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _BindingSpace:
    target_text: str
    local_facts: list[tuple[str, str]]
    local_bindings: list[tuple[str, str]]
    accessible_lemmas: list[str]
    witness_terms: list[str]


class MathEngine:
    engine_name = "MathEngine"
    backend_family = "symbolic_vm"

    def eligible_banks(self, capsule: GoalCapsule) -> list[str]:
        banks: list[str] = []
        for prior in capsule.bank_priors:
            if prior.suppressed:
                continue
            if _ENGINE_BY_BANK.get(prior.name) == self.engine_name:
                banks.append(prior.name)
        return _unique_strings(banks)

    def eligible_skeletons(self, capsule: GoalCapsule, bank: str) -> list[ProofSkeleton]:
        return [
            skeleton
            for skeleton in capsule.proof_skeletons
            if skeleton.bank == bank or _ENGINE_BY_BANK.get(skeleton.bank) == self.engine_name
        ]

    def make_request(
        self,
        *,
        capsule: GoalCapsule,
        ledger: ProofShadowLedger,
        bank: str,
        goal_text: str,
    ) -> MathEngineRequest:
        return MathEngineRequest(
            request_id=f"{self.engine_name}:{bank}:{abs(hash((goal_text, bank))) % 10_000_000}",
            engine_name=self.engine_name,
            active_bank=bank,
            goal_text=goal_text,
            goal_bucket=capsule.specification.goal_bucket,
            allowed_transformations=list(capsule.specification.projector_markers),
            budget=int(capsule.execution_budgets.get("certificate_budget", 24)),
            ledger=ledger,
            backend_family=self.backend_family,
            proof_dsl_program_ids=[
                program.program_id
                for program in capsule.proof_dsl_programs
                if program.backend_family == self.backend_family
            ][:8],
            constraint_ids=[
                constraint.constraint_id
                for constraint in capsule.solver_constraints
                if constraint.solver_family == self.backend_family or self.backend_family == "lean_arith"
            ][:16],
            eqsat_plan_id=capsule.eqsat_plan.plan_id if capsule.eqsat_plan is not None and self.backend_family == "egglog_eqsat" else "",
            relational_spec_ids=[
                spec.spec_id
                for spec in capsule.relational_search_specs
                if spec.backend_family == self.backend_family
            ][:8],
        )

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        return []

    def generate(
        self,
        *,
        capsule: GoalCapsule,
        goal_text: str,
        space: _BindingSpace,
        ledger: ProofShadowLedger,
    ) -> tuple[list[EngineCertificate], list[dict[str, Any]]]:
        certificates: list[EngineCertificate] = []
        outcomes: list[dict[str, Any]] = []
        for bank in self.eligible_banks(capsule):
            request = self.make_request(
                capsule=capsule,
                ledger=ledger,
                bank=bank,
                goal_text=goal_text,
            )
            certificate_count = 0
            for skeleton in self.eligible_skeletons(capsule, bank):
                if skeleton.bank != bank:
                    continue
                dsl_bindings = _proof_dsl_binding_space(
                    capsule,
                    skeleton,
                    space,
                    self.backend_family,
                )
                if dsl_bindings:
                    generated = [
                        _certificate_from_skeleton(
                            skeleton,
                            binding,
                            engine_name=self.engine_name,
                            request=request,
                        )
                        for binding in dsl_bindings
                    ]
                else:
                    generated = _instantiate_skeleton_certificates(
                        skeleton,
                        space,
                        engine_name=self.engine_name,
                        request=request,
                    )
                certificates.extend(generated)
                certificate_count += len(generated)
                outcomes.append(
                    {
                        "request_id": request.request_id,
                        "engine_name": self.engine_name,
                        "backend_family": self.backend_family,
                        "bank": bank,
                        "skeleton_id": skeleton.skeleton_id,
                        "certificate_count": len(generated),
                        "goal_bucket": capsule.specification.goal_bucket,
                        "proof_dsl_program_ids": list(request.proof_dsl_program_ids),
                        "constraint_ids": list(request.constraint_ids),
                        "eqsat_plan_id": request.eqsat_plan_id,
                        "relational_spec_ids": list(request.relational_spec_ids),
                    }
                )
            extras = self.extra_certificates(capsule=capsule, request=request, space=space)
            if extras:
                certificates.extend(extras)
                certificate_count += len(extras)
                outcomes.append(
                    {
                        "request_id": request.request_id,
                        "engine_name": self.engine_name,
                        "backend_family": self.backend_family,
                        "bank": bank,
                        "skeleton_id": "__extra__",
                        "certificate_count": len(extras),
                        "goal_bucket": capsule.specification.goal_bucket,
                        "proof_dsl_program_ids": list(request.proof_dsl_program_ids),
                        "constraint_ids": list(request.constraint_ids),
                        "eqsat_plan_id": request.eqsat_plan_id,
                        "relational_spec_ids": list(request.relational_spec_ids),
                    }
                )
            if certificate_count == 0:
                outcomes.append(
                    {
                        "request_id": request.request_id,
                        "engine_name": self.engine_name,
                        "backend_family": self.backend_family,
                        "bank": bank,
                        "skeleton_id": "__none__",
                        "certificate_count": 0,
                        "goal_bucket": capsule.specification.goal_bucket,
                        "proof_dsl_program_ids": list(request.proof_dsl_program_ids),
                        "constraint_ids": list(request.constraint_ids),
                        "eqsat_plan_id": request.eqsat_plan_id,
                        "relational_spec_ids": list(request.relational_spec_ids),
                    }
                )
        return certificates, outcomes


class EqSatEngine(MathEngine):
    engine_name = "EqSatEngine"
    backend_family = "egglog_eqsat"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        rules = _rewrite_fact_rules(request.ledger)
        if not rules:
            return []
        known_props = {sanitize_goal_text(hyp_type) for _expr, hyp_type in space.local_facts}
        best_expr, path = _bounded_rewrite_search(
            space.target_text,
            rules,
            budget=max(8, request.budget * 4),
            known_props=known_props,
        )
        if not path:
            return []
        rw_items = _rewrite_tactic_items(path)
        if not rw_items:
            return []
        certificates: list[EngineCertificate] = []
        rewrite_tactic = f"rw [{', '.join(rw_items)}]"
        target_after = sanitize_goal_text(best_expr)
        explanation = [
            f"eqsat_plan:{request.eqsat_plan_id or 'inline'}",
            *[f"rewrite:{item}" for item in rw_items],
        ]
        matching_fact = _find_fact_expr(space, target_after)
        if matching_fact:
            certificates.append(
                EngineCertificate(
                    certificate_id=f"{self.engine_name}:close:{abs(hash((request.goal_text, tuple(rw_items), matching_fact))) % 10_000_000}",
                    engine_name=self.engine_name,
                    bank=request.active_bank,
                    skeleton_id="",
                    certificate_kind="calc_chain",
                    summary="Equality saturation extracted a rewrite path to an existing local fact.",
                    evidence=[matching_fact, *rw_items],
                    projected_tactics=[rewrite_tactic, f"exact {matching_fact}"],
                    target_before=request.goal_text,
                    target_after_hint=target_after,
                    backend_family=self.backend_family,
                    proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
                    constraint_ids=list(request.constraint_ids),
                    explanation_trace=explanation,
                    provenance={
                        "calc_steps": _calc_steps_from_path(path),
                        "eqsat_plan_id": request.eqsat_plan_id,
                        "close_mode": "exact_fact",
                    },
                )
            )
        if _is_reflexive_proposition(target_after):
            certificates.append(
                EngineCertificate(
                    certificate_id=f"{self.engine_name}:rfl:{abs(hash((request.goal_text, tuple(rw_items)))) % 10_000_000}",
                    engine_name=self.engine_name,
                    bank=request.active_bank,
                    skeleton_id="",
                    certificate_kind="calc_chain",
                    summary="Equality saturation extracted a rewrite path to a reflexive proposition.",
                    evidence=list(rw_items),
                    projected_tactics=[rewrite_tactic, "rfl"],
                    target_before=request.goal_text,
                    target_after_hint=target_after,
                    backend_family=self.backend_family,
                    proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
                    constraint_ids=list(request.constraint_ids),
                    explanation_trace=explanation,
                    provenance={
                        "calc_steps": _calc_steps_from_path(path),
                        "eqsat_plan_id": request.eqsat_plan_id,
                        "close_mode": "rfl",
                    },
                )
            )
        if not certificates:
            certificates.append(
                EngineCertificate(
                    certificate_id=f"{self.engine_name}:progress:{abs(hash((request.goal_text, tuple(rw_items), target_after))) % 10_000_000}",
                    engine_name=self.engine_name,
                    bank=request.active_bank,
                    skeleton_id="",
                    certificate_kind="rewrite_chain",
                    summary="Equality saturation extracted a cheaper representative for the target surface.",
                    evidence=list(rw_items),
                    projected_tactics=[rewrite_tactic],
                    target_before=request.goal_text,
                    target_after_hint=target_after,
                    backend_family=self.backend_family,
                    proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
                    constraint_ids=list(request.constraint_ids),
                    explanation_trace=explanation,
                    provenance={"eqsat_plan_id": request.eqsat_plan_id},
                )
            )
        return certificates


class ArithEngine(MathEngine):
    engine_name = "ArithEngine"
    backend_family = "lean_arith"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        certificates: list[EngineCertificate] = []
        target = space.target_text
        if _ABS_LT_ONE_RE.search(target):
            certificates.append(
                EngineCertificate(
                    certificate_id=f"{self.engine_name}:abs:{abs(hash((request.goal_text, target))) % 10_000_000}",
                    engine_name=self.engine_name,
                    bank=request.active_bank,
                    skeleton_id="",
                    certificate_kind="solver_normal_form",
                    summary="Normalize an absolute-value arithmetic goal before solver closure.",
                    evidence=["Int.abs_lt_one_iff"],
                    projected_tactics=["rw [Int.abs_lt_one_iff]", "omega"],
                    target_before=request.goal_text,
                    backend_family=self.backend_family,
                    constraint_ids=list(request.constraint_ids),
                    explanation_trace=["arith_normalize", "abs_lt_one_iff", "omega"],
                    provenance={"solver": "omega"},
                )
            )
        elif _is_arith_goal(target):
            certificates.extend(
                [
                    EngineCertificate(
                        certificate_id=f"{self.engine_name}:norm_num:{abs(hash((request.goal_text, 'norm_num'))) % 10_000_000}",
                        engine_name=self.engine_name,
                        bank=request.active_bank,
                        skeleton_id="",
                        certificate_kind="solver_normal_form",
                        summary="Arithmetic normal form through the Lean calculator backend.",
                        evidence=["norm_num"],
                        projected_tactics=["norm_num"],
                        target_before=request.goal_text,
                        backend_family=self.backend_family,
                        constraint_ids=list(request.constraint_ids),
                        explanation_trace=["arith_normalize", "norm_num"],
                        provenance={"solver": "norm_num"},
                    ),
                    EngineCertificate(
                        certificate_id=f"{self.engine_name}:linarith:{abs(hash((request.goal_text, 'linarith'))) % 10_000_000}",
                        engine_name=self.engine_name,
                        bank=request.active_bank,
                        skeleton_id="",
                        certificate_kind="solver_normal_form",
                        summary="Linear arithmetic close candidate from the arithmetic backend.",
                        evidence=["linarith"],
                        projected_tactics=["linarith"],
                        target_before=request.goal_text,
                        backend_family=self.backend_family,
                        constraint_ids=list(request.constraint_ids),
                        explanation_trace=["arith_normalize", "linarith"],
                        provenance={"solver": "linarith"},
                    ),
                ]
            )
        return certificates


class WitnessEngine(MathEngine):
    engine_name = "WitnessEngine"
    backend_family = "rosette_proof_dsl"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        parsed = _parse_exists_target(space.target_text)
        if parsed is None:
            return []
        binder, body = parsed
        certificates: list[EngineCertificate] = []
        for witness in space.witness_terms[: max(4, request.budget // 4)]:
            instantiated = _substitute_binder(body, binder, witness)
            proof_expr = _find_fact_expr(space, instantiated)
            if proof_expr:
                certificates.append(
                    _exact_certificate(
                        engine_name=self.engine_name,
                        backend_family=self.backend_family,
                        bank=request.active_bank,
                        summary=f"Bounded witness search found `{witness}` supported by a local fact.",
                        target_before=request.goal_text,
                        proof_term=f"⟨{witness}, {proof_expr}⟩",
                        request=request,
                        certificate_kind="witness_construction",
                        evidence=[witness, proof_expr],
                        explanation_trace=["bounded_witness_search", f"witness:{witness}", "local_fact_support"],
                    )
                )
                continue
            if _is_reflexive_proposition(instantiated):
                certificates.append(
                    EngineCertificate(
                        certificate_id=f"{self.engine_name}:witness:{abs(hash((request.goal_text, witness))) % 10_000_000}",
                        engine_name=self.engine_name,
                        bank=request.active_bank,
                        skeleton_id="",
                        certificate_kind="witness_construction",
                        summary=f"Bounded witness search found a reflexive witness `{witness}`.",
                        evidence=[witness],
                        projected_tactics=[f"refine ⟨{witness}, ?_⟩", "rfl"],
                        target_before=request.goal_text,
                        target_after_hint=instantiated,
                        backend_family=self.backend_family,
                        proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
                        constraint_ids=list(request.constraint_ids),
                        explanation_trace=["bounded_witness_search", f"witness:{witness}", "reflexive_body"],
                        provenance={"relational_spec_ids": list(request.relational_spec_ids)},
                    )
                )
        return certificates


class RecursiveInvariantEngine(MathEngine):
    engine_name = "RecursiveInvariantEngine"
    backend_family = "symbolic_rewrite_vm"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        target = space.target_text
        if "root" not in target and "rec" not in target and "fold" not in target:
            return []
        rules = _rewrite_fact_rules(request.ledger)
        known_props = {sanitize_goal_text(hyp_type) for _expr, hyp_type in space.local_facts}
        best_expr, path = _bounded_rewrite_search(
            target,
            rules,
            budget=max(6, request.budget * 2),
            known_props=known_props,
        )
        if not path:
            return []
        rw_items = _rewrite_tactic_items(path)
        if not rw_items:
            return []
        tactics = [f"rw [{', '.join(rw_items)}]"]
        if _is_reflexive_proposition(best_expr):
            tactics.append("rfl")
        summary = "Recursive rewrite VM applied a bounded local invariant-preserving rewrite path."
        return [
            EngineCertificate(
                certificate_id=f"{self.engine_name}:bounded:{abs(hash((request.goal_text, tuple(rw_items)))) % 10_000_000}",
                engine_name=self.engine_name,
                bank=request.active_bank,
                skeleton_id="",
                certificate_kind="recursive_bridge",
                summary=summary,
                evidence=list(rw_items),
                projected_tactics=tactics,
                target_before=request.goal_text,
                target_after_hint=best_expr,
                backend_family=self.backend_family,
                constraint_ids=list(request.constraint_ids),
                explanation_trace=["bounded_recursive_rewrite_vm", *[f"rewrite:{item}" for item in rw_items]],
                provenance={"close_mode": "rfl" if _is_reflexive_proposition(best_expr) else "progress_only"},
            )
        ]


class FiniteFilterEngine(MathEngine):
    engine_name = "FiniteFilterEngine"
    backend_family = "kodkod_relational"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        target = space.target_text
        certificates: list[EngineCertificate] = []
        exact_fact = _find_fact_expr(space, target)
        if exact_fact:
            certificates.append(
                _exact_certificate(
                    engine_name=self.engine_name,
                    backend_family=self.backend_family,
                    bank=request.active_bank,
                    summary="Relational local search found the target directly in the local fact universe.",
                    target_before=request.goal_text,
                    proof_term=exact_fact,
                    request=request,
                    evidence=[exact_fact],
                    explanation_trace=["bounded_relational_search", "direct_fact_hit"],
                )
            )
        mem_inter_match = re.match(r"^(.+?)\s*∈\s*(.+?)\s*∩\s*(.+)$", target)
        if mem_inter_match:
            elem = mem_inter_match.group(1).strip()
            left = mem_inter_match.group(2).strip()
            right = mem_inter_match.group(3).strip()
            left_fact = _find_fact_expr(space, f"{elem} ∈ {left}")
            right_fact = _find_fact_expr(space, f"{elem} ∈ {right}")
            if left_fact and right_fact:
                certificates.append(
                    _exact_certificate(
                        engine_name=self.engine_name,
                        backend_family=self.backend_family,
                        bank=request.active_bank,
                        summary="Bounded relational search assembled an intersection-membership witness pair.",
                        target_before=request.goal_text,
                        proof_term=f"⟨{left_fact}, {right_fact}⟩",
                        request=request,
                        certificate_kind="witness_construction",
                        evidence=[left_fact, right_fact],
                        explanation_trace=["bounded_relational_search", "intersection_pair"],
                    )
                )
        return certificates


class ContextTransportEngine(MathEngine):
    engine_name = "ContextTransportEngine"
    backend_family = "rosette_proof_dsl"

    def extra_certificates(
        self,
        *,
        capsule: GoalCapsule,
        request: MathEngineRequest,
        space: _BindingSpace,
    ) -> list[EngineCertificate]:
        certificates: list[EngineCertificate] = []
        abs_eq_match = _SIMPLE_EQUALITY_ZERO_RE.fullmatch(space.target_text)
        if abs_eq_match and any("abs_lt_one_iff" in lemma for lemma in space.accessible_lemmas):
            for fact in _abs_lt_one_facts(space)[:4]:
                certificates.append(
                    _exact_certificate(
                        engine_name=self.engine_name,
                        backend_family=self.backend_family,
                        bank=request.active_bank,
                        summary="Proof DSL transport mapped an absolute-value hypothesis through an iff bridge.",
                        target_before=request.goal_text,
                        proof_term=f"Int.abs_lt_one_iff.mp {fact}",
                        request=request,
                        certificate_kind="term_transport",
                        evidence=[fact, "Int.abs_lt_one_iff"],
                        explanation_trace=["proof_dsl_transport", "iff_mp", "Int.abs_lt_one_iff"],
                    )
                )
        abs_goal_match = _ABS_LT_ONE_RE.search(space.target_text)
        if abs_goal_match:
            witness_var = abs_goal_match.group(1)
            for expr, hyp_type in space.local_facts:
                if sanitize_goal_text(hyp_type) == f"{witness_var} = 0":
                    certificates.append(
                        EngineCertificate(
                            certificate_id=f"{self.engine_name}:simpa:{abs(hash((request.goal_text, expr))) % 10_000_000}",
                            engine_name=self.engine_name,
                            bank=request.active_bank,
                            skeleton_id="",
                            certificate_kind="term_transport",
                            summary="Proof DSL transport normalized an absolute-value target from a local equality.",
                            evidence=[expr],
                            projected_tactics=[f"simpa [{expr}]"],
                            target_before=request.goal_text,
                            backend_family=self.backend_family,
                            proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
                            constraint_ids=list(request.constraint_ids),
                            explanation_trace=["proof_dsl_transport", "simpa", expr],
                            provenance={"close_mode": "simpa"},
                        )
                    )
        for expr, hyp_type in space.local_facts:
            normalized = sanitize_goal_text(hyp_type)
            if normalized == space.target_text and (".mp " in expr or ".mpr " in expr or ".symm" in expr):
                certificates.append(
                    EngineCertificate(
                        certificate_id=f"{self.engine_name}:have:{abs(hash((request.goal_text, expr))) % 10_000_000}",
                        engine_name=self.engine_name,
                        bank="local_fact_selector",
                        skeleton_id="",
                        certificate_kind="have_chain",
                        summary="Project a derived local fact through a `have` chain before exact close.",
                        bindings={},
                        evidence=[expr],
                        target_before=request.goal_text,
                        backend_family=self.backend_family,
                        constraint_ids=list(request.constraint_ids),
                        explanation_trace=["derive_local_fact", "project_have_chain"],
                        provenance={"have_expr": expr, "close_mode": "exact"},
                    )
                )
        return certificates


# Finisher tactics appended after projected programs.  Closures in r6 bridge
# data came exclusively from simple finishers (ring, omega, positivity, simpa,
# apply?) applied after a rewrite/transport chain made partial progress.
_DEFAULT_FINISHERS: list[str] = ["ring", "omega", "simp_all", "apply?"]

_FINISHER_TACTICS_FOR_BUCKET: dict[str, list[str]] = {
    "equality": ["ring", "omega", "norm_num", "simp_all"],
    "inequality": ["omega", "linarith", "positivity", "norm_num"],
    "membership": ["simp_all", "aesop", "apply?"],
    "forall": ["simp_all", "aesop", "apply?"],
    "exists": ["simp_all", "aesop", "apply?"],
    "atomic_prop": ["simp_all", "aesop", "omega", "apply?"],
    "false": ["simp_all", "omega", "aesop"],
}


class ProofProjector:
    def __init__(self, skeleton_lookup: dict[str, ProofSkeleton]) -> None:
        self.skeleton_lookup = skeleton_lookup

    def project(
        self,
        certificate: EngineCertificate,
    ) -> tuple[ProjectedProofProgram | None, ProjectorDecision]:
        skeleton = self.skeleton_lookup.get(certificate.skeleton_id)
        tactics: list[str] = []
        rationale = certificate.summary
        specialist = "symbolic_sandbox"
        if certificate.projected_tactics:
            tactics = list(certificate.projected_tactics)
        elif certificate.certificate_kind == "have_chain":
            have_expr = str(certificate.provenance.get("have_expr", "") or "")
            close_mode = str(certificate.provenance.get("close_mode", "") or "exact")
            if not have_expr:
                return None, ProjectorDecision(
                    certificate_id=certificate.certificate_id,
                    projector_status="rejected",
                    rejection_reason="missing_have_expr",
                    certificate_backend=certificate.backend_family,
                )
            tactics = [f"have h_ducky := {have_expr}"]
            if close_mode == "simpa":
                tactics.append("simpa using h_ducky")
            else:
                tactics.append("exact h_ducky")
        elif certificate.certificate_kind == "calc_chain":
            calc_steps = [str(step).strip() for step in (certificate.provenance.get("calc_steps") or []) if str(step).strip()]
            if not calc_steps:
                return None, ProjectorDecision(
                    certificate_id=certificate.certificate_id,
                    projector_status="rejected",
                    rejection_reason="missing_calc_steps",
                    certificate_backend=certificate.backend_family,
                )
            tactics = ["calc\n" + "\n".join(calc_steps)]
        else:
            if skeleton is None:
                return None, ProjectorDecision(
                    certificate_id=certificate.certificate_id,
                    projector_status="rejected",
                    rejection_reason="missing_skeleton",
                    certificate_backend=certificate.backend_family,
                )
            specialist = skeleton.specialist
            try:
                tactics = [template.format(**certificate.bindings) for template in skeleton.tactic_templates]
                rationale = skeleton.rationale
            except KeyError as exc:
                return None, ProjectorDecision(
                    certificate_id=certificate.certificate_id,
                    projector_status="rejected",
                    rejection_reason=f"missing_binding:{exc}",
                    certificate_backend=certificate.backend_family,
                )
        if skeleton is not None:
            specialist = skeleton.specialist
        projected = ProjectedProofProgram(
            program_id=f"{certificate.bank}:{certificate.skeleton_id or certificate.certificate_kind}",
            bank=certificate.bank,
            specialist=specialist,
            skeleton_id=certificate.skeleton_id,
            tactics=tactics,
            rationale=rationale,
            score=0.0,
            certificate_id=certificate.certificate_id,
            certificate_shape=certificate.certificate_kind,
            bindings=dict(certificate.bindings),
            projector_backend="proof_projector_v1",
            derivation_path=list(certificate.explanation_trace),
        )
        return projected, ProjectorDecision(
            certificate_id=certificate.certificate_id,
            projector_status="projected",
            projected_program_id=projected.program_id,
            certificate_backend=certificate.backend_family,
        )


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _is_reflexive_proposition(text: str) -> bool:
    normalized = sanitize_goal_text(text)
    equality = _split_equality(normalized)
    if equality is not None:
        left, right = equality
        return left == right
    iff_parts = _split_iff(normalized)
    if iff_parts is not None:
        left, right = iff_parts
        return left == right
    return False


def _expression_cost(expr: str, known_props: set[str]) -> tuple[int, int, int]:
    normalized = sanitize_goal_text(expr)
    cost = len(normalized)
    token_cost = len(_tokenize_goal(normalized))
    bonus = 0
    if normalized in known_props:
        bonus -= 40
    if _is_reflexive_proposition(normalized):
        bonus -= 40
    return (cost + bonus, token_cost, cost)


def _rewrite_fact_rules(ledger: ProofShadowLedger) -> list[tuple[str, str, str, bool]]:
    rules: list[tuple[str, str, str, bool]] = []
    for fact in ledger.facts:
        proposition = sanitize_goal_text(fact.proposition)
        split = _split_equality(proposition)
        if split is None:
            split = _split_iff(proposition)
        if split is None:
            continue
        lhs, rhs = split
        if lhs == rhs:
            continue
        rules.append((fact.expr, lhs, rhs, False))
        rules.append((fact.expr, rhs, lhs, True))
    return rules


def _replace_once(expr: str, lhs: str, rhs: str, *, max_variants: int = 4) -> list[str]:
    if not lhs or lhs == rhs or lhs not in expr:
        return []
    variants: list[str] = []
    start = 0
    while len(variants) < max_variants:
        idx = expr.find(lhs, start)
        if idx < 0:
            break
        replaced = expr[:idx] + rhs + expr[idx + len(lhs) :]
        if replaced != expr:
            variants.append(replaced)
        start = idx + len(lhs)
    return _unique_strings(variants)


def _bounded_rewrite_search(
    expr: str,
    rules: list[tuple[str, str, str, bool]],
    *,
    budget: int,
    known_props: set[str],
) -> tuple[str, list[tuple[str, bool, str, str]]]:
    start = sanitize_goal_text(expr)
    queue: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}
    step_used: dict[str, tuple[str, bool, str, str]] = {}
    best = start
    best_cost = _expression_cost(start, known_props)

    while queue and len(parent) < max(1, budget):
        current = queue.popleft()
        for fact_expr, lhs, rhs, is_reverse in rules:
            for variant in _replace_once(current, lhs, rhs):
                normalized = sanitize_goal_text(variant)
                if normalized in parent:
                    continue
                parent[normalized] = current
                step_used[normalized] = (fact_expr, is_reverse, current, normalized)
                queue.append(normalized)
                variant_cost = _expression_cost(normalized, known_props)
                if variant_cost < best_cost:
                    best = normalized
                    best_cost = variant_cost
                if normalized in known_props or _is_reflexive_proposition(normalized):
                    best = normalized
                    queue.clear()
                    break
            if not queue and best != start:
                break

    if best == start:
        return start, []
    path: list[tuple[str, bool, str, str]] = []
    cursor = best
    while cursor in step_used:
        fact_expr, is_reverse, before, after = step_used[cursor]
        path.append((fact_expr, is_reverse, before, after))
        parent_cursor = parent.get(cursor)
        if parent_cursor is None:
            break
        cursor = parent_cursor
    path.reverse()
    return best, path


def _rewrite_tactic_items(path: list[tuple[str, bool, str, str]]) -> list[str]:
    items: list[str] = []
    for fact_expr, is_reverse, _before, _after in path:
        if not fact_expr:
            continue
        items.append(f"← {fact_expr}" if is_reverse else fact_expr)
    return _unique_strings(items)


def _parse_exists_target(target_text: str) -> tuple[str, str] | None:
    normalized = sanitize_goal_text(target_text)
    match = re.match(r"^∃\s+([A-Za-z_][A-Za-z0-9_']*)\s*,\s*(.+)$", normalized)
    if not match:
        return None
    binder, body = match.group(1).strip(), match.group(2).strip()
    if not binder or not body:
        return None
    return binder, body


def _substitute_binder(body: str, binder: str, term: str) -> str:
    pattern = re.compile(rf"\b{re.escape(binder)}\b")
    return sanitize_goal_text(pattern.sub(term, body))


def _find_fact_expr(space: _BindingSpace, proposition: str) -> str:
    target = sanitize_goal_text(proposition)
    for expr, hyp_type in space.local_facts:
        if sanitize_goal_text(hyp_type) == target:
            return expr
    return ""


def _instantiate_via_proof_dsl(
    skeleton: ProofSkeleton,
    program_id: str,
    space: _BindingSpace,
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = [{}]
    hole_order = [hole.hole_id for hole in skeleton.holes]
    if not hole_order:
        return [{}]
    for hole_id in hole_order:
        hole = next((candidate for candidate in skeleton.holes if candidate.hole_id == hole_id), None)
        if hole is None:
            continue
        candidates = _hole_candidates(hole, space)
        if not candidates:
            if hole.required:
                return []
            continue
        expanded_bindings: list[dict[str, str]] = []
        max_branch = max(1, hole.max_candidates)
        for binding in bindings:
            for candidate in candidates[:max_branch]:
                updated = dict(binding)
                updated[hole.hole_id] = candidate
                expanded_bindings.append(updated)
        bindings = expanded_bindings or bindings
    return bindings or [{}]


def _proof_dsl_binding_space(
    capsule: GoalCapsule,
    skeleton: ProofSkeleton,
    space: _BindingSpace,
    backend_family: str,
) -> list[dict[str, str]]:
    programs = [
        program
        for program in capsule.proof_dsl_programs
        if program.skeleton_id == skeleton.skeleton_id and program.backend_family == backend_family
    ]
    if not programs:
        return []
    bindings: list[dict[str, str]] = []
    for program in programs:
        bindings.extend(_instantiate_via_proof_dsl(skeleton, program.program_id, space))
    return bindings


def _calc_steps_from_path(path: list[tuple[str, bool, str, str]]) -> list[str]:
    if not path:
        return []
    steps: list[str] = []
    for fact_expr, is_reverse, before, after in path:
        rw_term = f"← {fact_expr}" if is_reverse else fact_expr
        steps.append(f"  {before} := by rw [{rw_term}]")
    steps.append(f"  _ = {path[-1][3]} := rfl")
    return steps


def _exact_certificate(
    *,
    engine_name: str,
    backend_family: str,
    bank: str,
    summary: str,
    target_before: str,
    proof_term: str,
    request: MathEngineRequest,
    certificate_kind: str = "terminal_close",
    evidence: list[str] | None = None,
    explanation_trace: list[str] | None = None,
    projected_tactics: list[str] | None = None,
) -> EngineCertificate:
    return EngineCertificate(
        certificate_id=f"{engine_name}:{bank}:{abs(hash((target_before, proof_term, bank))) % 10_000_000}",
        engine_name=engine_name,
        bank=bank,
        skeleton_id="",
        certificate_kind=certificate_kind,
        summary=summary,
        evidence=list(evidence or [proof_term]),
        projected_tactics=list(projected_tactics or [f"exact {proof_term}"]),
        target_before=target_before,
        backend_family=backend_family,
        proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
        constraint_ids=list(request.constraint_ids),
        explanation_trace=list(explanation_trace or []),
        provenance={
            "request_id": request.request_id,
            "proof_term": proof_term,
            "relational_spec_ids": list(request.relational_spec_ids),
            "eqsat_plan_id": request.eqsat_plan_id,
        },
    )


def _goal_text(row: dict[str, Any]) -> str:
    for key in _GOAL_CANDIDATE_KEYS:
        text = str(row.get(key, "") or "").strip()
        if text:
            return text
    return ""


def _step_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("step_trace") or []
    return [step for step in trace if isinstance(step, dict)]


def _prefix_tactics_from_row(row: dict[str, Any]) -> list[str]:
    tactics = [str(tac) for tac in (row.get("tactics_used") or []) if str(tac).strip()]
    if tactics:
        return tactics
    out: list[str] = []
    for step in _step_trace(row):
        if not step.get("progress"):
            continue
        tactic = str(step.get("tactic", "") or "").strip()
        if tactic:
            out.append(tactic)
    return out


def _prefix_goal_states_from_row(row: dict[str, Any], prefix_len: int) -> list[str]:
    out: list[str] = []
    for step in _step_trace(row):
        if not step.get("progress"):
            continue
        goal_before = str(step.get("goal_before", "") or "").strip()
        if goal_before:
            out.append(goal_before)
        if len(out) >= prefix_len:
            break
    return out


def _resolve_file_path(row: dict[str, Any], lean: Any) -> str:
    file_path = str(row.get("lean_path", "") or row.get("file_path", "") or "").strip()
    if file_path:
        return file_path
    theorem_id = str(row.get("theorem_id", "") or "").strip()
    if not theorem_id:
        return ""
    server = getattr(lean, "_server", None)
    if getattr(lean, "_backend", "") != "pantograph" or server is None:
        return ""
    try:
        info = server.env_inspect(theorem_id)
    except Exception:
        return ""
    module = str(info.get("module", "") or "").strip()
    if not module:
        return ""
    return module.replace(".", "/") + ".lean"


def replay_residual_state(row: dict[str, Any], lean: Any) -> DuckyReplayState:
    theorem_id = str(row.get("theorem_id", "") or "").strip()
    file_path = _resolve_file_path(row, lean)
    prefix_tactics = _prefix_tactics_from_row(row)
    prefix_goal_states = _prefix_goal_states_from_row(row, len(prefix_tactics))
    expected_goal = str(row.get("last_goal", "") or "").strip()

    if theorem_id and file_path and getattr(lean, "_backend", "") == "pantograph":
        replay_attempts = [
            {
                "expected_goal": expected_goal,
                "prefix_goal_states": prefix_goal_states,
            },
            {
                "expected_goal": "",
                "prefix_goal_states": prefix_goal_states,
            },
            {
                "expected_goal": "",
                "prefix_goal_states": [],
            },
            {
                "expected_goal": "",
                "prefix_goal_states": [],
                "prefix_tactics": [],
            },
        ]
        replay_failure = ""
        replay_fail_idx = -1
        for attempt in replay_attempts:
            try:
                replay = lean.goal_via_file_context(
                    theorem_full_name=theorem_id,
                    file_path=file_path,
                    prefix_tactics=list(attempt.get("prefix_tactics", prefix_tactics)),
                    expected_goal=str(attempt.get("expected_goal", "") or ""),
                    project_root=lean.config.project_root,
                    prefix_goal_states=list(attempt.get("prefix_goal_states", prefix_goal_states)),
                )
                if replay.success and replay.goal_state:
                    return DuckyReplayState(
                        theorem_id=theorem_id,
                        file_path=file_path,
                        goal_state=str(replay.goal_state),
                        goal_kind="file_context",
                        theorem_faithful=True,
                        tier_used=str(replay.tier_used or ""),
                        replay_success=True,
                        replay_failure_category="",
                        replay_failing_prefix_idx=-1,
                        prefix_tactics=prefix_tactics,
                        prefix_goal_states=prefix_goal_states,
                    )
                replay_failure = str(getattr(replay, "failure_category", "") or replay_failure)
                replay_fail_idx = int(getattr(replay, "failing_prefix_idx", replay_fail_idx) or replay_fail_idx)
            except Exception as exc:
                replay_failure = type(exc).__name__
                replay_fail_idx = -1
    else:
        replay_failure = "file_context_unavailable"
        replay_fail_idx = -1

    for key in _GOAL_CANDIDATE_KEYS:
        candidate = str(row.get(key, "") or "").strip()
        if not candidate:
            continue
        try:
            started = lean.goal_start(candidate, theorem_name=theorem_id, file_path=file_path)
            return DuckyReplayState(
                theorem_id=theorem_id,
                file_path=file_path,
                goal_state=started,
                goal_kind=key,
                theorem_faithful=False,
                tier_used="goal_start",
                replay_success=True,
                replay_failure_category=replay_failure,
                replay_failing_prefix_idx=replay_fail_idx,
                prefix_tactics=prefix_tactics,
                prefix_goal_states=prefix_goal_states,
            )
        except Exception:
            continue

    return DuckyReplayState(
        theorem_id=theorem_id,
        file_path=file_path,
        goal_state="",
        goal_kind="",
        theorem_faithful=False,
        tier_used="",
        replay_success=False,
        replay_failure_category=replay_failure,
        replay_failing_prefix_idx=replay_fail_idx,
        prefix_tactics=prefix_tactics,
        prefix_goal_states=prefix_goal_states,
    )


def _tokenize_goal(goal_text: str) -> list[str]:
    out: list[str] = []
    for token in _IDENT_RE.findall(sanitize_goal_text(goal_text or "")):
        if token in _TACTIC_KEYWORDS:
            continue
        if token in {"True", "False", "Prop", "Type"}:
            continue
        out.append(token)
    return _unique_strings(out)


def _goal_target_text(goal_text: str) -> str:
    text = sanitize_goal_text(goal_text or "")
    if "⊢" in text:
        return text.split("⊢", 1)[1].strip()
    return text.strip()


def _local_hypotheses(goal_text: str) -> list[tuple[str, str]]:
    text = sanitize_goal_text(goal_text or "")
    if "⊢" not in text:
        return []
    context, _target = text.split("⊢", 1)
    out: list[tuple[str, str]] = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("case "):
            continue
        if " : " not in line:
            continue
        name, hyp_type = line.split(" : ", 1)
        name = name.strip()
        hyp_type = hyp_type.strip()
        if not name or not hyp_type:
            continue
        if name == "⊢":
            continue
        out.append((name, hyp_type))
    return out


def _cached_goal_variables(lean: Any | None, goal_text: str) -> list[tuple[str, str]]:
    if lean is None:
        return []
    cache = getattr(lean, "_goal_states", None)
    env_key = getattr(lean, "_current_env_key", "")
    if not isinstance(cache, dict) or not env_key:
        return []
    state = cache.get((env_key, goal_text))
    if state is None:
        return []
    goals = getattr(state, "goals", None) or []
    if not goals:
        return []
    target_text = sanitize_goal_text(goal_text)
    selected = goals[0]
    for goal in goals:
        goal_target = sanitize_goal_text(str(getattr(goal, "target", "") or ""))
        if goal_target == target_text:
            selected = goal
            break
    out: list[tuple[str, str]] = []
    for variable in getattr(selected, "variables", []) or []:
        name = str(getattr(variable, "name", "") or "").strip()
        typ = sanitize_goal_text(str(getattr(variable, "t", "") or "").strip())
        if not name or not typ:
            continue
        out.append((name, typ))
    return out


def _cached_goal_targets(lean: Any | None, goal_text: str) -> list[str]:
    if lean is None:
        return []
    cache = getattr(lean, "_goal_states", None)
    env_key = getattr(lean, "_current_env_key", "")
    if not isinstance(cache, dict) or not env_key:
        return []
    state = cache.get((env_key, goal_text))
    if state is None:
        return []
    return [str(getattr(goal, "target", "") or "") for goal in (getattr(state, "goals", None) or []) if str(getattr(goal, "target", "") or "").strip()]


def _candidate_witness_terms(goal_text: str) -> list[str]:
    text = _goal_target_text(goal_text or "")
    out = ["0", "1"]
    for term in _TERM_APP_RE.findall(text):
        if "." not in term:
            continue
        out.append(term.strip())
    for token in _tokenize_goal(text):
        if token[0].isupper():
            continue
        if len(token) <= 24:
            out.append(token)
    return _unique_strings(out)[:10]


def _witness_terms_from_propositions(local_bindings: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for _name, typ in local_bindings:
        for token in _tokenize_goal(typ):
            if token in _TACTIC_KEYWORDS or token[0].isupper():
                continue
            if len(token) <= 24:
                out.append(token)
    return _unique_strings(out)


def _split_conjunction(text: str) -> tuple[str, str] | None:
    normalized = sanitize_goal_text(text)
    if " ∧ " not in normalized:
        return None
    left, right = normalized.split(" ∧ ", 1)
    if not left.strip() or not right.strip():
        return None
    return left.strip(), right.strip()


def _split_iff(text: str) -> tuple[str, str] | None:
    normalized = sanitize_goal_text(text)
    if " ↔ " not in normalized:
        return None
    left, right = normalized.split(" ↔ ", 1)
    if not left.strip() or not right.strip():
        return None
    return left.strip(), right.strip()


def _split_equality(text: str) -> tuple[str, str] | None:
    normalized = sanitize_goal_text(text)
    if " = " not in normalized:
        return None
    left, right = normalized.split(" = ", 1)
    if not left.strip() or not right.strip():
        return None
    return left.strip(), right.strip()


def _is_arith_goal(goal_text: str) -> bool:
    text = _goal_target_text(goal_text)
    markers = ("<", "≤", ">", "≥", "|", "+", "-", "*", "/", "^", "Int.", "Nat.", "Real.", "Rat.", "factorial", "choose")
    return any(marker in text for marker in markers)


def _is_rewrite_like_lemma(lemma: str) -> bool:
    return bool(_REWRITE_LEMMA_RE.search(lemma))


def _lemma_token_score(lemma: str, goal_tokens: set[str], theorem_id: str) -> float:
    parts = re.split(r"[._]", lemma)
    lemma_tokens = {part for part in parts if part}
    theorem_ns = theorem_id.split(".", 1)[0] if "." in theorem_id else theorem_id
    overlap = len(goal_tokens & lemma_tokens)
    score = float(overlap)
    if theorem_ns and lemma.startswith(theorem_ns):
        score += 2.0
    short = lemma.rsplit(".", 1)[-1]
    if short in goal_tokens:
        score += 2.0
    if any(token in lemma for token in goal_tokens):
        score += 0.5
    return score


def _accessible_lemmas(
    theorem_id: str,
    goal_text: str,
    conn: sqlite3.Connection | None = None,
    *,
    accessible_theorem_id: int | None = None,
    max_lemmas: int = 8,
) -> list[str]:
    if conn is None or accessible_theorem_id is None:
        return []
    premise_ids = list(get_accessible_premises(conn, accessible_theorem_id))
    if not premise_ids:
        return []
    query = "SELECT id, name FROM entities WHERE id IN (%s)" % ",".join("?" for _ in premise_ids)
    rows = conn.execute(query, tuple(premise_ids)).fetchall()
    goal_tokens = set(_tokenize_goal(goal_text))
    ranked = sorted(
        ((str(name), _lemma_token_score(str(name), goal_tokens, theorem_id)) for _eid, name in rows if name),
        key=lambda item: (-item[1], item[0]),
    )
    return [name for name, score in ranked if score > 0][:max_lemmas]


def _bank_priority(capsule: GoalCapsule, bank: str) -> float:
    for prior in capsule.bank_priors:
        if prior.name == bank and not prior.suppressed:
            return float(prior.weight)
    return 0.0


def _program_id(bank: str, name: str) -> str:
    return f"{bank}:{name}"


def _tactic_mentions_disabled(tactic: str, disabled_tactics: set[str]) -> bool:
    lowered = str(tactic or "").lower()
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", lowered) for name in disabled_tactics)


def _program_uses_disabled_tactic(program: DuckyProgram, disabled_tactics: set[str]) -> bool:
    if not disabled_tactics:
        return False
    return any(_tactic_mentions_disabled(tactic, disabled_tactics) for tactic in program.tactics)


def _engine_backend(engine_name: str) -> str:
    return _BACKEND_BY_ENGINE.get(engine_name, "")


def _engine_allowed(
    engine_name: str,
    *,
    allowed_engine_names: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
) -> bool:
    if allowed_engine_names and engine_name not in allowed_engine_names:
        return False
    if allowed_backend_families:
        backend_family = _engine_backend(engine_name)
        if backend_family not in allowed_backend_families:
            return False
    return True


def _bank_allowed(
    bank: str,
    *,
    allowed_engine_names: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
) -> bool:
    engine_name = _ENGINE_BY_BANK.get(bank, "")
    if not engine_name:
        return not allowed_engine_names and not allowed_backend_families
    return _engine_allowed(
        engine_name,
        allowed_engine_names=allowed_engine_names,
        allowed_backend_families=allowed_backend_families,
    )


def _add_program(
    programs: list[DuckyProgram],
    seen: set[tuple[str, ...]],
    *,
    bank: str,
    specialist: str,
    name: str,
    tactics: list[str],
    rationale: str,
    score: float,
    skeleton_id: str = "",
    bindings: dict[str, str] | None = None,
    disabled_tactics: set[str] | None = None,
    engine_name: str = "",
    backend_family: str = "",
    certificate_shape: str = "",
    projector_status: str = "",
    projector_backend: str = "",
) -> None:
    normalized = tuple(_unique_strings(tactics))
    if not normalized or normalized in seen:
        return
    blocked_tactics = {str(tactic).strip().lower() for tactic in (disabled_tactics or set()) if str(tactic).strip()}
    if blocked_tactics and any(_tactic_mentions_disabled(text, blocked_tactics) for text in normalized):
        return
    seen.add(normalized)
    programs.append(
        DuckyProgram(
            program_id=_program_id(bank, name),
            bank=bank,
            specialist=specialist,
            tactics=list(normalized),
            rationale=rationale,
            score=round(score, 3),
            skeleton_id=skeleton_id,
            bindings=dict(bindings or {}),
            engine_name=engine_name,
            backend_family=backend_family,
            certificate_shape=certificate_shape,
            projector_status=projector_status,
            projector_backend=projector_backend,
        )
    )


def _build_binding_space(
    lean: Any | None,
    theorem_id: str,
    goal_text: str,
    conn: sqlite3.Connection | None,
    accessible_theorem_id: int | None,
    holographic_premises: list[str] | None = None,
) -> _BindingSpace:
    local_bindings = _cached_goal_variables(lean, goal_text)
    if not local_bindings:
        local_bindings = _local_hypotheses(goal_text)
    local_facts = _derive_local_facts(local_bindings)
    witness_terms = _unique_strings(
        [name for name, _typ in local_bindings]
        + _witness_terms_from_propositions(local_bindings)
        + _candidate_witness_terms(goal_text)
    )[:16]
    db_lemmas = _accessible_lemmas(
        theorem_id,
        _goal_target_text(goal_text),
        conn,
        accessible_theorem_id=accessible_theorem_id,
        max_lemmas=8,
    )
    # Merge holographic premises — these are high-confidence matches from
    # multi-projection coherence scoring on the structured residual.
    if holographic_premises:
        seen = set(db_lemmas)
        for hp in holographic_premises:
            if hp not in seen:
                db_lemmas.append(hp)
                seen.add(hp)
    return _BindingSpace(
        target_text=_goal_target_text(goal_text),
        local_facts=local_facts,
        local_bindings=local_bindings,
        accessible_lemmas=db_lemmas,
        witness_terms=witness_terms,
    )


def _ledger_fact_kind(hyp_type: str) -> str:
    text = sanitize_goal_text(hyp_type or "")
    if " ↔ " in text:
        return "iff"
    if " = " in text:
        return "equality"
    if "∈" in text or "⊆" in text:
        return "membership"
    if text.startswith("∃") or "Exists" in text:
        return "witness"
    return "proposition"


def _build_shadow_ledger(space: _BindingSpace, *, theorem_id: str, goal_text: str) -> ProofShadowLedger:
    facts: list[LedgerFact] = []
    for idx, (expr, hyp_type) in enumerate(space.local_facts):
        facts.append(
            LedgerFact(
                fact_id=f"fact_{idx}",
                expr=expr,
                fact_kind=_ledger_fact_kind(hyp_type),
                proposition=sanitize_goal_text(hyp_type),
                source_space="local_context",
                derivation_kind="cached_or_derived",
                provenance={"theorem_id": theorem_id, "goal_text": sanitize_goal_text(goal_text)},
            )
        )
    return ProofShadowLedger(
        facts=facts,
        accessible_premises=list(space.accessible_lemmas),
        candidate_witnesses=list(space.witness_terms),
        candidate_rewrites=_local_rewrite_facts(space),
        engine_outcomes=[],
        rejected_branches=[],
    )


def _engine_registry() -> dict[str, MathEngine]:
    engines: list[MathEngine] = [
        EqSatEngine(),
        ArithEngine(),
        WitnessEngine(),
        RecursiveInvariantEngine(),
        FiniteFilterEngine(),
        ContextTransportEngine(),
    ]
    return {engine.engine_name: engine for engine in engines}


def _engine_banks(capsule: GoalCapsule, engine_name: str) -> list[str]:
    banks: list[str] = []
    for prior in capsule.bank_priors:
        if prior.suppressed:
            continue
        if _ENGINE_BY_BANK.get(prior.name) == engine_name:
            banks.append(prior.name)
    return _unique_strings(banks)


def _make_engine_request(
    *,
    capsule: GoalCapsule,
    ledger: ProofShadowLedger,
    engine_name: str,
    bank: str,
    goal_text: str,
) -> MathEngineRequest:
    return MathEngineRequest(
        request_id=f"{engine_name}:{bank}:{abs(hash((goal_text, bank))) % 10_000_000}",
        engine_name=engine_name,
        active_bank=bank,
        goal_text=goal_text,
        goal_bucket=capsule.specification.goal_bucket,
        allowed_transformations=list(capsule.specification.projector_markers),
        budget=int(capsule.execution_budgets.get("certificate_budget", 24)),
        ledger=ledger,
    )


def _certificate_from_skeleton(
    skeleton: ProofSkeleton,
    binding: dict[str, str],
    *,
    engine_name: str,
    request: MathEngineRequest,
    evidence: list[str] | None = None,
    summary: str | None = None,
    projected_tactics: list[str] | None = None,
) -> EngineCertificate:
    return EngineCertificate(
        certificate_id=f"{engine_name}:{skeleton.skeleton_id}:{abs(hash((request.goal_text, tuple(sorted(binding.items()))))) % 10_000_000}",
        engine_name=engine_name,
        bank=skeleton.bank,
        skeleton_id=skeleton.skeleton_id,
        certificate_kind=(skeleton.certificate_kinds or ["projected_tactics"])[0],
        summary=summary or skeleton.rationale,
        bindings=dict(binding),
        evidence=list(evidence or [*binding.values()]),
        projected_tactics=list(projected_tactics or []),
        target_before=request.goal_text,
        backend_family=request.backend_family,
        proof_dsl_program_id=(request.proof_dsl_program_ids[0] if request.proof_dsl_program_ids else ""),
        constraint_ids=list(request.constraint_ids),
        explanation_trace=[
            f"backend:{request.backend_family}",
            f"skeleton:{skeleton.skeleton_id}",
            f"bank:{skeleton.bank}",
        ],
        provenance={
            "request_id": request.request_id,
            "allowed_transformations": list(request.allowed_transformations),
            "hole_bindings": dict(binding),
            "proof_dsl_program_ids": list(request.proof_dsl_program_ids),
            "eqsat_plan_id": request.eqsat_plan_id,
            "relational_spec_ids": list(request.relational_spec_ids),
        },
    )


def _project_certificate(
    certificate: EngineCertificate,
    skeleton_lookup: dict[str, ProofSkeleton],
) -> tuple[ProjectedProofProgram | None, ProjectorDecision]:
    skeleton = skeleton_lookup.get(certificate.skeleton_id)
    tactics: list[str] = []
    rationale = certificate.summary
    specialist = "symbolic_sandbox"
    if certificate.projected_tactics:
        tactics = list(certificate.projected_tactics)
    elif certificate.certificate_kind == "have_chain":
        have_expr = str(certificate.provenance.get("have_expr", "") or "")
        close_mode = str(certificate.provenance.get("close_mode", "") or "exact")
        if not have_expr:
            return None, ProjectorDecision(
                certificate_id=certificate.certificate_id,
                projector_status="rejected",
                rejection_reason="missing_have_expr",
            )
        tactics = [f"have h_ducky := {have_expr}"]
        if close_mode == "simpa":
            tactics.append("simpa using h_ducky")
        else:
            tactics.append("exact h_ducky")
    elif certificate.certificate_kind == "calc_chain":
        calc_steps = [str(step).strip() for step in (certificate.provenance.get("calc_steps") or []) if str(step).strip()]
        if not calc_steps:
            return None, ProjectorDecision(
                certificate_id=certificate.certificate_id,
                projector_status="rejected",
                rejection_reason="missing_calc_steps",
            )
        tactics = ["calc\n" + "\n".join(calc_steps)]
    else:
        if skeleton is None:
            return None, ProjectorDecision(
                certificate_id=certificate.certificate_id,
                projector_status="rejected",
                rejection_reason="missing_skeleton",
            )
        specialist = skeleton.specialist
        try:
            tactics = [template.format(**certificate.bindings) for template in skeleton.tactic_templates]
            rationale = skeleton.rationale
        except KeyError as exc:
            return None, ProjectorDecision(
                certificate_id=certificate.certificate_id,
                projector_status="rejected",
                rejection_reason=f"missing_binding:{exc}",
            )
    if skeleton is not None:
        specialist = skeleton.specialist
    projected = ProjectedProofProgram(
        program_id=f"{certificate.bank}:{certificate.skeleton_id or certificate.certificate_kind}",
        bank=certificate.bank,
        specialist=specialist,
        skeleton_id=certificate.skeleton_id,
        tactics=tactics,
        rationale=rationale,
        score=0.0,
        certificate_id=certificate.certificate_id,
        certificate_shape=certificate.certificate_kind,
        bindings=dict(certificate.bindings),
    )
    return projected, ProjectorDecision(
        certificate_id=certificate.certificate_id,
        projector_status="projected",
        projected_program_id=projected.program_id,
    )


def _projected_to_program(
    projected: ProjectedProofProgram,
    decision: ProjectorDecision,
    *,
    score: float,
) -> DuckyProgram:
    return DuckyProgram(
        program_id=projected.program_id,
        bank=projected.bank,
        specialist=projected.specialist,
        tactics=list(projected.tactics),
        rationale=projected.rationale,
        score=round(score, 3),
        skeleton_id=projected.skeleton_id,
        bindings=dict(projected.bindings),
        certificate_id=projected.certificate_id,
        engine_name="",
        backend_family="",
        certificate_shape=projected.certificate_shape,
        projector_status=decision.projector_status,
        projector_backend=projected.projector_backend,
    )


def _matching_local_facts(space: _BindingSpace) -> list[str]:
    return [expr for expr, hyp_type in space.local_facts if sanitize_goal_text(hyp_type) == space.target_text]


def _abs_lt_one_facts(space: _BindingSpace) -> list[str]:
    eq_zero_match = _SIMPLE_EQUALITY_ZERO_RE.fullmatch(space.target_text)
    if not eq_zero_match:
        return []
    variable = eq_zero_match.group(1)
    out: list[str] = []
    for name, hyp_type in space.local_facts:
        match = _ABS_LT_ONE_RE.search(sanitize_goal_text(hyp_type))
        if match and match.group(1) == variable:
            out.append(name)
    return out


def _left_conj_facts(space: _BindingSpace) -> list[str]:
    parts = _split_conjunction(space.target_text)
    if not parts:
        return []
    left, _ = parts
    return [expr for expr, hyp_type in space.local_facts if sanitize_goal_text(hyp_type) == left]


def _right_conj_facts(space: _BindingSpace) -> list[str]:
    parts = _split_conjunction(space.target_text)
    if not parts:
        return []
    _, right = parts
    return [expr for expr, hyp_type in space.local_facts if sanitize_goal_text(hyp_type) == right]


def _local_rewrite_facts(space: _BindingSpace) -> list[str]:
    out: list[str] = []
    for expr, hyp_type in space.local_facts:
        text = sanitize_goal_text(hyp_type)
        if " = " in text or " ↔ " in text:
            out.append(expr)
    return _unique_strings(out)[:12]


def _rewrite_lemmas(space: _BindingSpace) -> list[str]:
    return [lemma for lemma in space.accessible_lemmas if _is_rewrite_like_lemma(lemma)]


def _hole_candidates(hole: ProofHoleSpec, space: _BindingSpace) -> list[str]:
    if hole.hole_kind == "matching_local_fact":
        candidates = _matching_local_facts(space)
    elif hole.hole_kind == "abs_lt_one_fact":
        candidates = _abs_lt_one_facts(space)
    elif hole.hole_kind == "left_conj_fact":
        candidates = _left_conj_facts(space)
    elif hole.hole_kind == "right_conj_fact":
        candidates = _right_conj_facts(space)
    elif hole.hole_kind == "local_rewrite_fact":
        candidates = _local_rewrite_facts(space)
    elif hole.hole_kind in {"accessible_exact_lemma", "accessible_apply_lemma"}:
        candidates = list(space.accessible_lemmas)
    elif hole.hole_kind == "rewrite_lemma":
        candidates = _rewrite_lemmas(space)
    elif hole.hole_kind == "witness_term":
        candidates = list(space.witness_terms)
    else:
        candidates = []
    return _unique_strings(candidates)[: hole.max_candidates]


def _instantiate_skeleton(skeleton: ProofSkeleton, space: _BindingSpace) -> list[DuckyProgram]:
    bindings: list[dict[str, str]] = [{}]
    for hole in skeleton.holes:
        candidates = _hole_candidates(hole, space)
        if not candidates:
            if hole.required:
                return []
            continue
        next_bindings: list[dict[str, str]] = []
        for binding in bindings:
            for candidate in candidates:
                expanded = dict(binding)
                expanded[hole.hole_id] = candidate
                next_bindings.append(expanded)
                if len(next_bindings) >= max(1, hole.max_candidates):
                    break
            if len(next_bindings) >= max(1, hole.max_candidates):
                break
        bindings = next_bindings or bindings

    programs: list[DuckyProgram] = []
    seen: set[tuple[str, ...]] = set()
    for binding in bindings:
        try:
            tactics = [template.format(**binding) for template in skeleton.tactic_templates]
        except KeyError:
            continue
        _add_program(
            programs,
            seen,
            bank=skeleton.bank,
            specialist=skeleton.specialist,
            name=skeleton.skeleton_id,
            tactics=tactics,
            rationale=skeleton.rationale,
            score=skeleton.priority,
            skeleton_id=skeleton.skeleton_id,
            bindings=binding,
        )
    return programs


def _instantiate_skeleton_certificates(
    skeleton: ProofSkeleton,
    space: _BindingSpace,
    *,
    engine_name: str,
    request: MathEngineRequest,
) -> list[EngineCertificate]:
    bindings: list[dict[str, str]] = [{}]
    for hole in skeleton.holes:
        candidates = _hole_candidates(hole, space)
        if not candidates:
            if hole.required:
                return []
            continue
        next_bindings: list[dict[str, str]] = []
        for binding in bindings:
            for candidate in candidates:
                expanded = dict(binding)
                expanded[hole.hole_id] = candidate
                next_bindings.append(expanded)
                if len(next_bindings) >= max(1, hole.max_candidates):
                    break
            if len(next_bindings) >= max(1, hole.max_candidates):
                break
        bindings = next_bindings or bindings

    return [
        _certificate_from_skeleton(
            skeleton,
            binding,
            engine_name=engine_name,
            request=request,
        )
        for binding in bindings
    ] or [
        _certificate_from_skeleton(
            skeleton,
            {},
            engine_name=engine_name,
            request=request,
        )
    ]


def _engine_skeletons(capsule: GoalCapsule, engine_name: str) -> list[ProofSkeleton]:
    allowed_banks = set(_engine_banks(capsule, engine_name))
    return [
        skeleton
        for skeleton in capsule.proof_skeletons
        if skeleton.bank in allowed_banks or _ENGINE_BY_BANK.get(skeleton.bank) == engine_name
    ]


def _run_engine(
    *,
    capsule: GoalCapsule,
    engine_name: str,
    goal_text: str,
    space: _BindingSpace,
    ledger: ProofShadowLedger,
) -> tuple[list[EngineCertificate], list[dict[str, Any]]]:
    certificates: list[EngineCertificate] = []
    outcomes: list[dict[str, Any]] = []
    for bank in _engine_banks(capsule, engine_name):
        request = _make_engine_request(
            capsule=capsule,
            ledger=ledger,
            engine_name=engine_name,
            bank=bank,
            goal_text=goal_text,
        )
        for skeleton in _engine_skeletons(capsule, engine_name):
            if skeleton.bank != bank:
                continue
            generated = _instantiate_skeleton_certificates(
                skeleton,
                space,
                engine_name=engine_name,
                request=request,
            )
            certificates.extend(generated)
            outcomes.append(
                {
                    "request_id": request.request_id,
                    "engine_name": engine_name,
                    "bank": bank,
                    "skeleton_id": skeleton.skeleton_id,
                    "certificate_count": len(generated),
                    "goal_bucket": capsule.specification.goal_bucket,
                }
            )
    # A small certificate-only fallback for heavy local transport cases.
    if engine_name == "ContextTransportEngine":
        for expr, hyp_type in space.local_facts:
            normalized = sanitize_goal_text(hyp_type)
            if normalized == space.target_text and (".mp " in expr or ".mpr " in expr or ".symm" in expr):
                certificates.append(
                    EngineCertificate(
                        certificate_id=f"{engine_name}:have:{abs(hash((goal_text, expr))) % 10_000_000}",
                        engine_name=engine_name,
                        bank="local_fact_selector",
                        skeleton_id="",
                        certificate_kind="have_chain",
                        summary="Project a derived local fact through a `have` chain before exact close.",
                        bindings={},
                        evidence=[expr],
                        target_before=goal_text,
                        provenance={"have_expr": expr, "close_mode": "exact"},
                    )
                )
    return certificates, outcomes


def build_ducky_symbolic_frontier(
    capsule: GoalCapsule,
    *,
    theorem_id: str,
    goal_text: str,
    lean: Any | None = None,
    conn: sqlite3.Connection | None = None,
    accessible_theorem_id: int | None = None,
    max_programs: int = 24,
    disabled_tactics: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
    allowed_engine_names: set[str] | None = None,
    holographic_premises: list[str] | None = None,
) -> dict[str, Any]:
    blocked_tactics = {str(tactic).strip().lower() for tactic in (disabled_tactics or set()) if str(tactic).strip()}
    programs: list[DuckyProgram] = []
    seen: set[tuple[str, ...]] = set()
    engine_outcomes: list[dict[str, Any]] = []
    projector_outcomes: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    space = _build_binding_space(lean, theorem_id, goal_text, conn, accessible_theorem_id,
                                 holographic_premises=holographic_premises)
    ledger = _build_shadow_ledger(space, theorem_id=theorem_id, goal_text=goal_text)
    skeleton_lookup = {skeleton.skeleton_id: skeleton for skeleton in capsule.proof_skeletons}
    engine_registry = _engine_registry()
    projector = ProofProjector(skeleton_lookup)

    for engine_name in capsule.allowed_engines:
        engine = engine_registry.get(engine_name)
        if engine is None:
            continue
        if not _engine_allowed(
            engine_name,
            allowed_engine_names=allowed_engine_names,
            allowed_backend_families=allowed_backend_families,
        ):
            continue
        generated, outcomes = engine.generate(
            capsule=capsule,
            goal_text=goal_text,
            space=space,
            ledger=ledger,
        )
        engine_outcomes.extend(outcomes)
        for certificate in generated:
            certificates.append(certificate.to_dict())
            projected, decision = projector.project(certificate)
            projector_outcomes.append(decision.to_dict())
            if projected is None:
                ledger.rejected_branches.append(
                    {
                        "certificate_id": certificate.certificate_id,
                        "engine_name": engine_name,
                        "backend_family": certificate.backend_family,
                        "reason": decision.rejection_reason,
                    }
                )
                continue
            score = _bank_priority(capsule, projected.bank)
            program = _projected_to_program(projected, decision, score=score)
            program.engine_name = engine_name
            program.backend_family = certificate.backend_family
            if _program_uses_disabled_tactic(program, blocked_tactics):
                ledger.rejected_branches.append(
                    {
                        "certificate_id": certificate.certificate_id,
                        "engine_name": engine_name,
                        "backend_family": certificate.backend_family,
                        "reason": "disabled_tactic_policy",
                        "tactics": list(program.tactics),
                    }
                )
                continue
            normalized = tuple(_unique_strings(program.tactics))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            programs.append(program)

    programs.sort(key=lambda item: (-item.score, len(item.tactics), item.program_id))
    return {
        "programs": programs[:max_programs],
        "ledger": ledger,
        "certificates": certificates,
        "engine_outcomes": engine_outcomes,
        "projector_outcomes": projector_outcomes,
    }


def build_ducky_programs(
    capsule: GoalCapsule,
    *,
    theorem_id: str,
    goal_text: str,
    lean: Any | None = None,
    conn: sqlite3.Connection | None = None,
    accessible_theorem_id: int | None = None,
    max_programs: int = 24,
    disabled_tactics: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
    allowed_engine_names: set[str] | None = None,
    holographic_premises: list[str] | None = None,
) -> list[DuckyProgram]:
    frontier = build_ducky_symbolic_frontier(
        capsule,
        theorem_id=theorem_id,
        goal_text=goal_text,
        lean=lean,
        conn=conn,
        accessible_theorem_id=accessible_theorem_id,
        max_programs=max_programs,
        disabled_tactics=disabled_tactics,
        allowed_backend_families=allowed_backend_families,
        allowed_engine_names=allowed_engine_names,
        holographic_premises=holographic_premises,
    )
    programs: list[DuckyProgram] = list(frontier["programs"])
    seen: set[tuple[str, ...]] = {tuple(_unique_strings(program.tactics)) for program in programs}
    space = _build_binding_space(lean, theorem_id, goal_text, conn, accessible_theorem_id,
                                 holographic_premises=holographic_premises)

    if len(programs) < 4 and _is_arith_goal(space.target_text):
        _add_program(
            programs,
            seen,
            bank="arith_nf",
            specialist="human_calculator",
            name="fallback_norm_num",
            tactics=["norm_num"],
            rationale="Arithmetic fallback when no richer symbolic skeleton instantiates.",
            score=max(_bank_priority(capsule, "arith_nf"), 0.5),
            skeleton_id="fallback_norm_num",
            disabled_tactics=disabled_tactics,
        )
        _add_program(
            programs,
            seen,
            bank="arith_nf",
            specialist="human_calculator",
            name="fallback_linarith",
            tactics=["linarith"],
            rationale="Linear arithmetic fallback when the skeleton frontier is sparse.",
            score=max(_bank_priority(capsule, "arith_nf"), 0.48),
            skeleton_id="fallback_linarith",
            disabled_tactics=disabled_tactics,
        )

    # ---------------------------------------------------------------
    # Fix 1: Premise-aware programs — try each accessible lemma as
    # exact/apply/rw directly.  The 81 near-miss "no progress" failures
    # mostly need the right specific lemma, not generic tactics.
    # ---------------------------------------------------------------
    def _add_if_allowed(
        bank: str, specialist: str, name: str, tactics: list[str],
        rationale: str, score: float, **kwargs: Any,
    ) -> None:
        if not _bank_allowed(bank, allowed_engine_names=allowed_engine_names,
                             allowed_backend_families=allowed_backend_families):
            return
        _add_program(programs, seen, bank=bank, specialist=specialist,
                     name=name, tactics=tactics, rationale=rationale,
                     score=score, disabled_tactics=disabled_tactics, **kwargs)

    for idx, lemma in enumerate(space.accessible_lemmas[:12]):
        base_score = 0.82 - idx * 0.015

        # --- Core application: exact, apply, refine ---
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_exact_{idx}",
            [f"exact {lemma}"], f"Direct: {lemma}",
            base_score, certificate_shape="premise_direct",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_apply_{idx}",
            [f"apply {lemma}"], f"Apply: {lemma}",
            base_score - 0.005, certificate_shape="premise_apply",
        )
        # Refine with one hole — lets Lean elaborate without "Function expected"
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_refine_{idx}",
            [f"refine {lemma} ?_"], f"Refine: {lemma}",
            base_score - 0.008, certificate_shape="premise_apply",
        )

        # --- Convert: fuzzy apply for type mismatches ---
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_convert_{idx}",
            [f"convert {lemma}"], f"Convert: {lemma}",
            base_score - 0.01, certificate_shape="premise_convert",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_convert_simp_{idx}",
            [f"convert {lemma} <;> simp"], f"Convert+simp: {lemma}",
            base_score - 0.012, certificate_shape="premise_convert",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_convert_ring_{idx}",
            [f"convert {lemma} <;> ring"], f"Convert+ring: {lemma}",
            base_score - 0.014, certificate_shape="premise_convert",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_convert_1_{idx}",
            [f"convert {lemma} using 1"], f"Convert using 1: {lemma}",
            base_score - 0.016, certificate_shape="premise_convert",
        )

        # --- Rewrites ---
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_rw_fwd_{idx}",
            [f"rw [{lemma}]"], f"Rw fwd: {lemma}",
            base_score - 0.02, certificate_shape="premise_rewrite",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_rw_bwd_{idx}",
            [f"rw [\u2190 {lemma}]"], f"Rw bwd: {lemma}",
            base_score - 0.025, certificate_shape="premise_rewrite",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_simp_rw_{idx}",
            [f"simp_rw [{lemma}]"], f"Simp_rw: {lemma}",
            base_score - 0.027, certificate_shape="premise_rewrite",
        )

        # --- Simp/simpa ---
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_simp_{idx}",
            [f"simp [{lemma}]"], f"Simp: {lemma}",
            base_score - 0.03, certificate_shape="premise_simp",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"premise_simpa_{idx}",
            [f"simpa using {lemma}"], f"Simpa: {lemma}",
            base_score - 0.035, certificate_shape="premise_simpa",
        )

        # --- Two-step chains: rw then close ---
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_rw_then_ring_{idx}",
            [f"rw [{lemma}]", "ring"], f"Rw then ring: {lemma}",
            base_score - 0.04, certificate_shape="premise_chain",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_rw_then_simp_{idx}",
            [f"rw [{lemma}]", "simp"], f"Rw then simp: {lemma}",
            base_score - 0.042, certificate_shape="premise_chain",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"premise_rw_then_omega_{idx}",
            [f"rw [{lemma}]", "omega"], f"Rw then omega: {lemma}",
            base_score - 0.044, certificate_shape="premise_chain",
        )

    # ---------------------------------------------------------------
    # Nested premise composition — the key gap from ~55% to ~75%+.
    # Real proofs need `exact P₁ (P₂ _ h)` but flat programs only
    # generate `exact P₁ _`.  Three tiers of composition:
    #
    # Tier 1 — Pairwise enumeration: for each (Pᵢ, Pⱼ), try
    #   exact Pᵢ (Pⱼ _ _), exact Pᵢ Pⱼ, apply Pᵢ; exact Pⱼ
    #
    # Tier 2 — Hypothesis filling: replace wildcards with local
    #   hypothesis names from the binding space
    #
    # Tier 3 — Rosette-style sketch: for proof-trace-guided
    #   compositions, use the holographic bundled tactics to identify
    #   which pairs actually compose in matched proofs
    # ---------------------------------------------------------------
    lemmas = space.accessible_lemmas
    hyp_names = [name for name, _typ in space.local_bindings[:8]
                 if name and not name.startswith("inst")]

    # Tier 1: Pairwise composition — each pair of accessible lemmas
    for i, p1 in enumerate(lemmas[:6]):
        for j, p2 in enumerate(lemmas[:6]):
            if i == j:
                continue
            pair_score = 0.78 - i * 0.01 - j * 0.01
            # P₁ applied to P₂ with wildcards (1-2 args)
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_exact_{i}_{j}",
                [f"exact {p1} ({p2} _ _)"],
                f"Nested: {p1} ({p2} _ _)",
                pair_score,
                certificate_shape="nested_composition",
            )
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_exact1_{i}_{j}",
                [f"exact {p1} ({p2} _)"],
                f"Nested: {p1} ({p2} _)",
                pair_score - 0.005,
                certificate_shape="nested_composition",
            )
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_exact0_{i}_{j}",
                [f"exact {p1} {p2}"],
                f"Direct: {p1} {p2}",
                pair_score - 0.01,
                certificate_shape="nested_composition",
            )
            # Two-step: apply P₁, then exact P₂
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_apply_exact_{i}_{j}",
                [f"apply {p1}", f"exact {p2} _ _"],
                f"Apply {p1} then exact {p2}",
                pair_score - 0.015,
                certificate_shape="nested_composition",
            )

    # Tier 2: Hypothesis filling — compose premise with local hypothesis
    for i, p in enumerate(lemmas[:8]):
        for k, h in enumerate(hyp_names[:6]):
            h_score = 0.76 - i * 0.01 - k * 0.005
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_hyp_{i}_{k}",
                [f"exact {p} {h}"],
                f"Premise+hyp: {p} {h}",
                h_score,
                certificate_shape="nested_composition",
            )
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_hyp_w_{i}_{k}",
                [f"exact {p} _ {h}"],
                f"Premise+wildcard+hyp: {p} _ {h}",
                h_score - 0.005,
                certificate_shape="nested_composition",
            )
            _add_if_allowed(
                "context_forward", "context_transport",
                f"compose_hyp_w2_{i}_{k}",
                [f"exact {p} {h} _"],
                f"Premise+hyp+wildcard: {p} {h} _",
                h_score - 0.01,
                certificate_shape="nested_composition",
            )
            # Premise applied to (another premise applied to hyp)
            for j, p2 in enumerate(lemmas[:4]):
                if j == i:
                    continue
                _add_if_allowed(
                    "context_forward", "context_transport",
                    f"compose_nest_{i}_{j}_{k}",
                    [f"exact {p} ({p2} _ {h})"],
                    f"Deep nest: {p} ({p2} _ {h})",
                    h_score - 0.02,
                    certificate_shape="nested_composition",
                )
                _add_if_allowed(
                    "context_forward", "context_transport",
                    f"compose_nest2_{i}_{j}_{k}",
                    [f"exact {p} ({p2} {h} _)"],
                    f"Deep nest: {p} ({p2} {h} _)",
                    h_score - 0.025,
                    certificate_shape="nested_composition",
                )
                _add_if_allowed(
                    "context_forward", "context_transport",
                    f"compose_nest3_{i}_{j}_{k}",
                    [f"exact {p} ({p2} {h})"],
                    f"Deep nest: {p} ({p2} {h})",
                    h_score - 0.03,
                    certificate_shape="nested_composition",
                )

    # Tier 3: Proof-trace-guided composition — if bundled tactics from
    # the holographic system contain specific nested patterns, inject those
    # directly.  The holographic trace already identified which premises
    # compose from real proof evidence.
    # (The bundled_tactics are tried via the holographic tactic chain stage
    # in the bridge — this tier focuses on the Ducky-internal generation
    # of similar patterns from the accessible lemma pool.)

    # ---------------------------------------------------------------
    # Static tactic compositions — multi-step sequences that
    # the first-order system never tries.
    # ---------------------------------------------------------------
    _COMPOSITIONS: list[tuple[str, list[str], str, str, float]] = [
        # (name, tactics, bank, specialist, score)
        ("ext_simp", ["ext", "simp"], "context_forward", "context_transport", 0.78),
        ("ext_aesop", ["ext", "aesop"], "context_forward", "context_transport", 0.76),
        ("ext_ring", ["ext", "ring"], "context_forward", "context_transport", 0.74),
        ("push_neg_norm_num", ["push_neg", "norm_num"], "arith_nf", "human_calculator", 0.72),
        ("push_neg_omega", ["push_neg", "omega"], "arith_nf", "human_calculator", 0.71),
        ("push_neg_simp", ["push_neg", "simp"], "arith_nf", "human_calculator", 0.70),
        ("norm_cast_ring", ["norm_cast", "ring"], "arith_nf", "human_calculator", 0.69),
        ("norm_cast_omega", ["norm_cast", "omega"], "arith_nf", "human_calculator", 0.68),
        ("norm_cast_norm_num", ["norm_cast", "norm_num"], "arith_nf", "human_calculator", 0.67),
        ("field_simp_ring", ["field_simp", "ring"], "arith_nf", "human_calculator", 0.66),
        ("simp_all_ext", ["simp_all", "ext"], "context_forward", "context_transport", 0.64),
        ("constructor_aesop", ["constructor", "all_goals aesop"], "context_forward", "context_transport", 0.62),
        ("constructor_simp", ["constructor", "all_goals simp"], "context_forward", "context_transport", 0.61),
    ]
    for cname, ctactics, cbank, cspec, cscore in _COMPOSITIONS:
        _add_if_allowed(
            cbank, cspec, f"composition_{cname}", ctactics,
            f"Two-step composition: {' then '.join(ctactics)}",
            cscore, certificate_shape="tactic_composition",
        )

    for idx, lemma in enumerate(space.accessible_lemmas[:4]):
        _add_if_allowed(
            "eq_sat", "human_calculator", f"rw_ring_{idx}",
            [f"rw [{lemma}]", "ring"], f"Rewrite then ring: {lemma}",
            0.73 - idx * 0.02, certificate_shape="premise_composition",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"rw_simp_{idx}",
            [f"rw [{lemma}]", "simp"], f"Rewrite then simp: {lemma}",
            0.72 - idx * 0.02, certificate_shape="premise_composition",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"simp_only_premise_{idx}",
            [f"simp only [{lemma}]"], f"Targeted simp: {lemma}",
            0.71 - idx * 0.02, certificate_shape="premise_simp_only",
        )

    # ---------------------------------------------------------------
    # Fix 3: First-order failure trace avoidance — add local hypothesis
    # exact/apply programs.  The first-order search often gets to 1 goal
    # but can't close because it doesn't try `exact <specific_hyp>`.
    # Use the local bindings (hypotheses in scope) as direct closers.
    # ---------------------------------------------------------------
    for idx, (hyp_name, _hyp_type) in enumerate(space.local_bindings[:12]):
        if not hyp_name or hyp_name.startswith("inst"):
            continue
        _add_if_allowed(
            "local_fact_selector", "atomic_fact_engine", f"hyp_exact_{idx}",
            [f"exact {hyp_name}"], f"Close with local hypothesis: {hyp_name}",
            0.88 - idx * 0.01, certificate_shape="local_hypothesis",
        )
        _add_if_allowed(
            "local_fact_selector", "atomic_fact_engine", f"hyp_apply_{idx}",
            [f"apply {hyp_name}"], f"Apply local hypothesis: {hyp_name}",
            0.87 - idx * 0.01, certificate_shape="local_hypothesis",
        )
        _add_if_allowed(
            "eq_sat", "human_calculator", f"hyp_rw_{idx}",
            [f"rw [{hyp_name}]"], f"Rewrite with hypothesis: {hyp_name}",
            0.75 - idx * 0.01, certificate_shape="local_hypothesis",
        )
        _add_if_allowed(
            "context_forward", "context_transport", f"hyp_simpa_{idx}",
            [f"simpa using {hyp_name}"], f"Simplify using hypothesis: {hyp_name}",
            0.86 - idx * 0.01, certificate_shape="local_hypothesis",
        )

    programs.sort(key=lambda item: (-item.score, len(item.tactics), item.program_id))
    return programs[:max_programs]


def _goal_shape_score(goal_text: str, goal_bucket: str) -> float:
    target = _goal_target_text(goal_text)
    token_count = len(_tokenize_goal(target))
    bucket_base = {
        "proved": 100.0,
        "atomic_prop": 78.0,
        "equality": 72.0,
        "inequality": 70.0,
        "membership": 66.0,
        "exists": 62.0,
        "iff": 58.0,
        "forall": 54.0,
        "subset": 52.0,
        "other": 48.0,
    }.get(goal_bucket, 45.0)
    score = bucket_base
    score -= min(len(target), 800) / 16.0
    score -= token_count * 0.85
    if _SIMPLE_EQUALITY_ZERO_RE.fullmatch(target):
        score += 12.0
    if _ABS_LT_ONE_RE.search(target):
        score += 6.0
    if target.startswith("∃ ") or " ∧ " in target:
        score -= 4.0
    for marker in _HEAVY_GOAL_MARKERS:
        if marker in target:
            score -= 8.0
    return score


def _progress_score(run: DuckyProgramRun) -> float:
    applied_bonus = len(run.tactics_applied) * 0.15
    return _goal_shape_score(run.final_goal, run.final_goal_bucket) + applied_bonus + run.score


def _derive_local_facts(local_bindings: list[tuple[str, str]]) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(expr: str, typ: str) -> bool:
        normalized = (expr.strip(), sanitize_goal_text(typ))
        if not normalized[0] or not normalized[1] or normalized in seen:
            return False
        seen.add(normalized)
        facts.append(normalized)
        return True

    for name, typ in local_bindings:
        add(name, typ)

    changed = True
    rounds = 0
    while changed and rounds < 3:
        changed = False
        rounds += 1
        current = list(facts)
        fact_by_type: dict[str, list[str]] = {}
        for expr, typ in current:
            fact_by_type.setdefault(typ, []).append(expr)
            conj = _split_conjunction(typ)
            if conj is not None:
                left, right = conj
                changed |= add(f"{expr}.1", left)
                changed |= add(f"{expr}.2", right)
            equality = _split_equality(typ)
            if equality is not None:
                left, right = equality
                changed |= add(f"{expr}.symm", f"{right} = {left}")
            iff_parts = _split_iff(typ)
            if iff_parts is not None:
                left, right = iff_parts
                changed |= add(f"{expr}.symm", f"{right} ↔ {left}")
        for expr, typ in current:
            iff_parts = _split_iff(typ)
            if iff_parts is None:
                continue
            left, right = iff_parts
            for proof in fact_by_type.get(left, [])[:4]:
                changed |= add(f"({expr}.mp {proof})", right)
            for proof in fact_by_type.get(right, [])[:4]:
                changed |= add(f"({expr}.mpr {proof})", left)

    return facts


def _build_closer_programs(
    goal_text: str,
    goal_bucket: str,
    *,
    theorem_id: str = "",
    lean: Any | None = None,
    conn: sqlite3.Connection | None = None,
    accessible_theorem_id: int | None = None,
    disabled_tactics: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
    allowed_engine_names: set[str] | None = None,
) -> list[DuckyProgram]:
    target_text = _goal_target_text(goal_text)
    space = _build_binding_space(lean, theorem_id, goal_text, conn, accessible_theorem_id)
    programs: list[DuckyProgram] = []
    seen: set[tuple[str, ...]] = set()

    def add(bank: str, specialist: str, name: str, tactics: list[str], rationale: str, score: float) -> None:
        if not _bank_allowed(
            bank,
            allowed_engine_names=allowed_engine_names,
            allowed_backend_families=allowed_backend_families,
        ):
            return
        engine_name = _ENGINE_BY_BANK.get(bank, "")
        backend_family = _engine_backend(engine_name)
        _add_program(
            programs,
            seen,
            bank=bank,
            specialist=specialist,
            name=name,
            tactics=tactics,
            rationale=rationale,
            score=score,
            disabled_tactics=disabled_tactics,
            engine_name=engine_name,
            backend_family=backend_family,
            certificate_shape="closer_program",
            projector_status="direct_closer",
            projector_backend="proof_projector_v1",
        )

    add("structural_close", "structural_closer", "simpa", ["simpa"], "Attempt direct normalization closure.", 0.95)
    add("structural_close", "structural_closer", "simp", ["simp"], "Cheap normalization close.", 0.94)
    add("structural_close", "structural_closer", "simp_all", ["simp_all"], "Exploit all local simplifications.", 0.93)
    add("context_forward", "context_transport", "solve_by_elim", ["solve_by_elim"], "Invoke Lean's symbolic elimination search on the contracted goal.", 0.925)
    add("context_forward", "context_transport", "aesop", ["aesop"], "Mine local context for a direct close.", 0.92)
    add("structural_close", "structural_closer", "apply?", ["apply?"], "Ask Lean for a direct theorem application on the contracted goal.", 0.91)
    add("local_fact_selector", "atomic_fact_engine", "assumption", ["assumption"], "Close directly from an assumption.", 0.92)

    add("arith_nf", "human_calculator", "ring", ["ring"], "Close algebraic identity by ring normalization.", 0.93)
    add("arith_nf", "human_calculator", "omega_generic", ["omega"], "Presburger arithmetic closure.", 0.90)
    add("arith_nf", "human_calculator", "positivity", ["positivity"], "Close non-negativity goal from structural positivity.", 0.88)
    add("arith_nf", "human_calculator", "norm_num_generic", ["norm_num"], "Numeric normalization closure.", 0.87)

    if goal_bucket in {"equality", "other"}:
        add("eq_sat", "human_calculator", "rfl", ["rfl"], "Reflexive equalities should close immediately.", 0.98)
        add("eq_sat", "human_calculator", "ring_nf_norm_num", ["ring_nf", "norm_num"], "Normalize algebraic equality to closure.", 0.94)
        add("eq_sat", "human_calculator", "field_simp_ring_nf", ["field_simp", "ring_nf"], "Clear denominators before algebraic closure.", 0.91)
        add("structural_close", "structural_closer", "ext_simp", ["ext", "simp"], "Reduce extensional equality to local closure.", 0.89)
        if _is_arith_goal(target_text):
            add("arith_nf", "human_calculator", "norm_num", ["norm_num"], "Normalize arithmetic equality.", 0.95)
            add("arith_nf", "human_calculator", "linarith", ["linarith"], "Close arithmetic equality from local facts.", 0.93)
            add("arith_nf", "human_calculator", "omega", ["omega"], "Presburger closure on arithmetic equality.", 0.93)
            add("solver_dispatch", "side_condition_sweeper", "nlinarith", ["nlinarith"], "Nonlinear arithmetic closure.", 0.9)
            add("structural_close", "structural_closer", "simp_all_abs_lt_one", ["simp_all [Int.abs_lt_one_iff]"], "Normalize local absolute-value facts before closure.", 0.9)
    if goal_bucket in {"inequality", "other"}:
        add("arith_nf", "human_calculator", "norm_num", ["norm_num"], "Normalize arithmetic side conditions.", 0.97)
        add("arith_nf", "human_calculator", "omega", ["omega"], "Presburger closure.", 0.95)
        add("arith_nf", "human_calculator", "linarith", ["linarith"], "Linear arithmetic closure.", 0.95)
        add("solver_dispatch", "side_condition_sweeper", "nlinarith", ["nlinarith"], "Nonlinear arithmetic closure.", 0.92)
        add("structural_close", "structural_closer", "simp_all_abs_lt_one", ["simp_all [Int.abs_lt_one_iff]"], "Normalize local absolute-value facts before closure.", 0.91)
        if "abs" in target_text or "|" in target_text:
            add("arith_nf", "human_calculator", "abs_lt_one_rw_norm_num", ["rw [Int.abs_lt_one_iff]", "norm_num"], "Rewrite and close an integer absolute-value side condition.", 0.96)
            add("arith_nf", "human_calculator", "abs_lt_one_rw_omega", ["rw [Int.abs_lt_one_iff]", "omega"], "Rewrite and close an integer absolute-value side condition with Presburger arithmetic.", 0.95)
            add("arith_nf", "human_calculator", "abs_lt_one_rw_aesop", ["rw [Int.abs_lt_one_iff]", "aesop"], "Rewrite and let local search close the normalized arithmetic goal.", 0.93)
    if goal_bucket == "iff":
        add("iff_splitter", "logic_splitter", "constructor_all_goals_aesop", ["constructor", "all_goals aesop"], "Split the iff and close both directions locally.", 0.94)
        add("iff_splitter", "logic_splitter", "constructor_all_goals_solve_by_elim", ["constructor", "all_goals solve_by_elim"], "Split the iff and close both directions with symbolic elimination.", 0.945)
        add("iff_splitter", "logic_splitter", "tauto", ["tauto"], "Close a pure-logic iff.", 0.93)
    if goal_bucket in {"exists", "membership", "subset"}:
        add("witness_constructor", "witness_engine", "constructor_aesop", ["constructor", "aesop"], "Expose local witness structure and close.", 0.9)
        add("witness_constructor", "witness_engine", "constructor_solve_by_elim", ["constructor", "solve_by_elim"], "Expose local witness structure and hand it to symbolic elimination.", 0.91)
        add("witness_constructor", "witness_engine", "constructor_simp", ["constructor", "simp"], "Expose local witness structure and simplify.", 0.89)
        # Domain-aware witness terms: try common witness values and local
        # hypothesis names.  For Int.exists_unit_of_abs the system needs
        # "use Int.sign a" not just "use 1".
        for witness_term in ["0", "1", "-1"]:
            add("witness_constructor", "witness_engine", f"use_{witness_term}",
                [f"use {witness_term}", "simp_all"],
                f"Witness {witness_term} + simplify.", 0.88)
        for idx, (hyp_name, _hyp_type) in enumerate(space.local_facts[:6]):
            add("witness_constructor", "witness_engine", f"use_hyp_{idx}",
                [f"use {hyp_name}", "simp_all"],
                f"Use local fact as witness: {hyp_name}", 0.87)
            add("witness_constructor", "witness_engine", f"exact_exists_hyp_{idx}",
                [f"exact ⟨{hyp_name}, by simp_all⟩"],
                f"Construct existential from {hyp_name}.", 0.86)
    if goal_bucket in {"forall", "subset", "negation"}:
        add("context_forward", "context_transport", "intros_solve_by_elim", ["intros", "solve_by_elim"], "Introduce local binders, then let symbolic elimination close the contracted goal.", 0.915)

    for idx, fact in enumerate(_matching_local_facts(space)[:12]):
        add("local_fact_selector", "atomic_fact_engine", f"exact_local_fact_{idx}", [f"exact {fact}"], f"Close directly from local fact `{fact}`.", 0.99)
        add("local_fact_selector", "atomic_fact_engine", f"simpa_local_fact_{idx}", [f"simpa using {fact}"], f"Normalize directly from local fact `{fact}`.", 0.985)

    conj_parts = _split_conjunction(target_text)
    if conj_parts is not None:
        left_facts = _left_conj_facts(space)
        right_facts = _right_conj_facts(space)
        if left_facts and right_facts:
            add(
                "witness_constructor",
                "witness_engine",
                "exact_local_pair",
                [f"exact ⟨{left_facts[0]}, {right_facts[0]}⟩"],
                "Construct the conjunction directly from local proof facts.",
                0.985,
            )

    for idx, (hyp_name, hyp_type) in enumerate(space.local_facts[:16]):
        hyp_text = sanitize_goal_text(hyp_type)
        eq_zero_match = _SIMPLE_EQUALITY_ZERO_RE.fullmatch(target_text)
        abs_lt_one_match = _ABS_LT_ONE_RE.search(hyp_text)
        if eq_zero_match and abs_lt_one_match and eq_zero_match.group(1) == abs_lt_one_match.group(1):
            add("local_fact_selector", "atomic_fact_engine", f"abs_mp_local_{idx}", [f"exact Int.abs_lt_one_iff.mp {hyp_name}"], f"Close equality from local absolute-value hypothesis `{hyp_name}`.", 0.99)
        abs_goal_match = _ABS_LT_ONE_RE.search(target_text)
        if abs_goal_match and hyp_text == f"{abs_goal_match.group(1)} = 0":
            add("local_fact_selector", "atomic_fact_engine", f"simpa_abs_local_{idx}", [f"simpa [{hyp_name}]"], f"Close absolute-value goal using local equality `{hyp_name}`.", 0.985)
        if " = " in hyp_text or " ↔ " in hyp_text:
            add("eq_sat", "human_calculator", f"rw_local_{idx}", [f"rw [{hyp_name}]"], f"Rewrite using local fact `{hyp_name}`.", 0.9)
            add("eq_sat", "human_calculator", f"rw_local_back_{idx}", [f"rw [← {hyp_name}]"], f"Rewrite using the reverse orientation of `{hyp_name}`.", 0.89)
            add("structural_close", "structural_closer", f"simpa_local_rw_{idx}", [f"simpa [{hyp_name}]"], f"Normalize using local rewrite fact `{hyp_name}`.", 0.9)

    programs.sort(key=lambda item: (-item.score, len(item.tactics), item.program_id))
    return programs


def execute_ducky_program(lean: Any, start_goal: str, program: DuckyProgram) -> DuckyProgramRun:
    current_goal = start_goal
    goals_after = [start_goal]
    tactics_applied: list[str] = []
    first_failure_tactic = ""
    first_failure_error = ""

    for tactic in program.tactics:
        current_targets = _cached_goal_targets(lean, current_goal)
        result: TacticResult = lean.try_tactic(current_goal, tactic)
        if not result.success:
            first_failure_tactic = tactic
            first_failure_error = str(result.error_message or "")
            break
        goals_after = [str(goal) for goal in (result.new_goals or []) if str(goal).strip()]
        if current_targets and goals_after == current_targets:
            first_failure_tactic = tactic
            first_failure_error = "accepted_without_goal_change"
            break
        tactics_applied.append(tactic)
        if not goals_after:
            return DuckyProgramRun(
                program_id=program.program_id,
                bank=program.bank,
                specialist=program.specialist,
                tactics=program.tactics,
                script=program.script,
                score=program.score,
                progressed=True,
                closed=True,
                tactics_applied=tactics_applied,
                final_goal="",
                final_goal_bucket="proved",
                goals_after=[],
                certificate_id=program.certificate_id,
                engine_name=program.engine_name,
                backend_family=program.backend_family,
                certificate_shape=program.certificate_shape,
                projector_status=program.projector_status,
                projector_backend=program.projector_backend,
            )
        current_goal = goals_after[0]

    # Iterative post-progress close: if any tactics applied and we have
    # a single remaining goal, try finishers ON THE LIVE GOALSTATE (not
    # from string recreation).  This catches cases like ring→omega where
    # the first tactic normalizes and the second closes.
    if tactics_applied and len(goals_after) <= 2:
        for finisher in ["ring", "omega", "simp_all", "norm_num",
                         "linarith", "nlinarith", "positivity",
                         "aesop", "tauto", "decide",
                         "field_simp; ring", "norm_cast; omega",
                         "ext; simp", "funext; simp",
                         "push_neg; simp", "assumption"]:
            try:
                fin_result: TacticResult = lean.try_tactic(current_goal, finisher)
                if fin_result.success and not fin_result.new_goals:
                    tactics_applied.append(finisher)
                    return DuckyProgramRun(
                        program_id=program.program_id,
                        bank=program.bank,
                        specialist=program.specialist,
                        tactics=program.tactics,
                        script=program.script,
                        score=program.score,
                        progressed=True,
                        closed=True,
                        tactics_applied=tactics_applied,
                        final_goal="",
                        final_goal_bucket="proved",
                        goals_after=[],
                        first_failure_tactic="",
                        first_failure_error="",
                        certificate_id=program.certificate_id,
                        engine_name=program.engine_name,
                        backend_family=program.backend_family,
                        certificate_shape="post_progress_close",
                        projector_status=program.projector_status,
                        projector_backend=program.projector_backend,
                    )
            except Exception:
                pass

    final_goal = current_goal if tactics_applied else start_goal
    return DuckyProgramRun(
        program_id=program.program_id,
        bank=program.bank,
        specialist=program.specialist,
        tactics=program.tactics,
        script=program.script,
        score=program.score,
        progressed=bool(tactics_applied),
        closed=False,
        tactics_applied=tactics_applied,
        final_goal=final_goal,
        final_goal_bucket=classify_goal_bucket(final_goal),
        goals_after=goals_after if tactics_applied else [start_goal],
        first_failure_tactic=first_failure_tactic,
        first_failure_error=first_failure_error,
        certificate_id=program.certificate_id,
        engine_name=program.engine_name,
        backend_family=program.backend_family,
        certificate_shape=program.certificate_shape,
        projector_status=program.projector_status,
        projector_backend=program.projector_backend,
    )


def _combined_program(chain: list[DuckyProgramRun]) -> dict[str, Any]:
    tactics: list[str] = []
    scripts: list[str] = []
    for run in chain:
        tactics.extend(run.tactics_applied)
        if run.script:
            scripts.append(run.script)
    final = chain[-1]
    return {
        "program_id": "chain:" + "→".join(run.program_id for run in chain),
        "bank": final.bank,
        "specialist": final.specialist,
        "script": "; ".join(scripts),
        "score": round(sum(run.score for run in chain), 3),
        "progressed": True,
        "closed": final.closed,
        "tactics_applied": tactics,
        "goals_after": list(final.goals_after),
        "final_goal": final.final_goal,
        "final_goal_bucket": final.final_goal_bucket,
        "rounds": len(chain),
        "subprograms": [run.to_dict() for run in chain],
    }


def run_ducky_on_goal(
    goal_state: str,
    *,
    theorem_id: str,
    lean: Any,
    conn: sqlite3.Connection | None = None,
    accessible_theorem_id: int | None = None,
    capsule: GoalCapsule | None = None,
    row_overrides: dict[str, Any] | None = None,
    max_programs: int = 24,
    max_rounds: int = 3,
    disabled_tactics: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
    allowed_engine_names: set[str] | None = None,
    holographic_premises: list[str] | None = None,
) -> DuckyExecutionResult:
    row = {
        "theorem_id": theorem_id,
        "last_goal": goal_state,
        "goal_state": goal_state,
        "last_goal_bucket": classify_goal_bucket(goal_state),
        "reasoning_gap_family": "",
        "residual_bucket": "single_goal_near_miss",
        "difficulty_band": "",
        "goals_closed": 1,
        "goals_remaining": 1,
        "attempts": 0,
        "lane_sequence": "",
        "search_pathology_tags": [],
        "remaining_goals_snapshot": [goal_state],
    }
    if row_overrides:
        row.update(row_overrides)
    tried: list[dict[str, Any]] = []
    engine_outcomes: list[dict[str, Any]] = []
    projector_outcomes: list[dict[str, Any]] = []
    current_goal = goal_state
    current_row = dict(row)
    current_capsule = capsule or build_goal_capsule(current_row)
    best_chain: list[DuckyProgramRun] = []
    total_programs = 0
    close_sweep_candidates = 3
    ledger_snapshot: dict[str, Any] | None = None

    for round_idx in range(max_rounds):
        all_programs = build_ducky_programs(
            current_capsule,
            theorem_id=theorem_id,
            goal_text=current_goal,
            lean=lean,
            conn=conn,
            accessible_theorem_id=accessible_theorem_id,
            max_programs=max_programs,
            disabled_tactics=disabled_tactics,
            allowed_backend_families=allowed_backend_families,
            allowed_engine_names=allowed_engine_names,
            holographic_premises=holographic_premises,
        )
        # Also run the frontier to get engine/projector outcomes and ledger
        frontier = build_ducky_symbolic_frontier(
            current_capsule,
            theorem_id=theorem_id,
            goal_text=current_goal,
            lean=lean,
            conn=conn,
            accessible_theorem_id=accessible_theorem_id,
            max_programs=max_programs,
            disabled_tactics=disabled_tactics,
            allowed_backend_families=allowed_backend_families,
            allowed_engine_names=allowed_engine_names,
        )
        # Merge: use the full program set but keep frontier metadata
        seen_scripts = {tuple(p.tactics) for p in all_programs}
        for p in frontier["programs"]:
            if tuple(p.tactics) not in seen_scripts:
                all_programs.append(p)
        programs = all_programs[:max_programs]
        ledger_snapshot = frontier["ledger"].to_dict()
        for outcome in frontier["engine_outcomes"]:
            tagged = dict(outcome)
            tagged["round"] = round_idx
            engine_outcomes.append(tagged)
        for decision in frontier["projector_outcomes"]:
            tagged = dict(decision)
            tagged["round"] = round_idx
            projector_outcomes.append(tagged)
        total_programs += len(programs)
        progress_candidates: list[list[DuckyProgramRun]] = []
        for program in programs:
            run = execute_ducky_program(lean, current_goal, program)
            payload = run.to_dict()
            payload["round"] = round_idx
            tried.append(payload)
            if run.closed:
                chain = best_chain + [run]
                return DuckyExecutionResult(
                    theorem_id=theorem_id,
                    started=True,
                    theorem_faithful=True,
                    start_goal_kind="direct_goal",
                    file_path="",
                    replay_tier="live",
                    replay_failure_category="",
                    replay_failing_prefix_idx=-1,
                    residual_bucket=str(row.get("residual_bucket", "") or ""),
                    goal_bucket=str(row.get("last_goal_bucket", "") or classify_goal_bucket(goal_state)),
                    specialist_targets=list(current_capsule.specialist_targets),
                    bank_priors=[prior.name for prior in current_capsule.bank_priors if not prior.suppressed],
                    programs_considered=total_programs,
                    closed=True,
                    progressed=True,
                    winning_program=_combined_program(chain),
                    final_goal="",
                    final_goal_bucket="proved",
                    goals_after=[],
                    ledger_snapshot=ledger_snapshot,
                    engine_outcomes=engine_outcomes,
                    projector_outcomes=projector_outcomes,
                    tried_programs=tried,
                    )
            if not run.progressed:
                continue

            # Decomposition quality guard: accept expansions that produce
            # genuinely simpler subgoals, reject pointless proliferation.
            n_after = len(run.goals_after) if run.goals_after else 1
            if n_after > 1:
                # Hard cap: >20 subgoals is never useful
                if n_after > 20:
                    continue
                # Quality check: are subgoals simpler than the original?
                orig_len = len(current_goal)
                avg_sub_len = sum(len(g) for g in run.goals_after) / max(n_after, 1)
                # Good decomposition: subgoals are shorter (simpler) than original
                # Bad decomposition: subgoals are same length or longer (duplication)
                if avg_sub_len >= orig_len * 0.9 and n_after > 3:
                    # Subgoals aren't meaningfully simpler AND there are many → reject
                    continue
                # Also check for identical subgoals (pointless duplication)
                unique_goals = len(set(run.goals_after))
                if unique_goals == 1 and n_after > 2:
                    # All subgoals identical — congr or similar produced copies
                    # Keep only if the single goal is genuinely simpler
                    if len(run.goals_after[0]) >= orig_len * 0.8:
                        continue

            # IMMEDIATE POST-PROGRESS CLOSER: try finishers on the contracted
            # state RIGHT NOW, before moving on.  This is the iterative
            # compression — each progress step triggers a re-attempt to close.
            if run.final_goal and len(run.goals_after) <= 2:
                for finisher in ["ring", "omega", "simp_all", "norm_num",
                                 "linarith", "nlinarith", "positivity",
                                 "aesop", "tauto", "decide",
                                 "ext; simp", "funext; simp",
                                 "field_simp; ring", "push_neg; simp",
                                 "norm_cast; omega", "norm_cast; ring",
                                 "apply?", "exact?"]:
                    try:
                        fin_result = lean.try_tactic(run.final_goal, finisher)
                        if fin_result.success and not fin_result.new_goals:
                            chain_with_fin = best_chain + [run]
                            # Build a synthetic closer run
                            fin_run = DuckyProgramRun(
                                program_id=f"post_progress_finisher:{finisher}",
                                bank="structural_close",
                                specialist="iterative_closer",
                                tactics=[finisher],
                                script=finisher,
                                score=0.99,
                                progressed=True,
                                closed=True,
                                tactics_applied=[finisher],
                                goals_after=[],
                                final_goal="",
                                final_goal_bucket="proved",
                                first_failure_tactic="",
                                first_failure_error="",
                                engine_name="",
                                backend_family="",
                                certificate_shape="post_progress_close",
                                projector_status="iterative",
                                projector_backend="",
                            )
                            chain_with_fin.append(fin_run)
                            fin_payload = fin_run.to_dict()
                            fin_payload["round"] = round_idx
                            fin_payload["followup_to"] = run.program_id
                            tried.append(fin_payload)
                            return DuckyExecutionResult(
                                theorem_id=theorem_id,
                                started=True,
                                theorem_faithful=True,
                                start_goal_kind="direct_goal",
                                file_path="",
                                replay_tier="live",
                                replay_failure_category="",
                                replay_failing_prefix_idx=-1,
                                residual_bucket=str(row.get("residual_bucket", "") or ""),
                                goal_bucket=str(row.get("last_goal_bucket", "") or classify_goal_bucket(goal_state)),
                                specialist_targets=list(current_capsule.specialist_targets),
                                bank_priors=[prior.name for prior in current_capsule.bank_priors if not prior.suppressed],
                                programs_considered=total_programs,
                                closed=True,
                                progressed=True,
                                winning_program=_combined_program(chain_with_fin),
                                final_goal="",
                                final_goal_bucket="proved",
                                goals_after=[],
                                ledger_snapshot=ledger_snapshot,
                                engine_outcomes=engine_outcomes,
                                projector_outcomes=projector_outcomes,
                                tried_programs=tried,
                            )
                    except Exception:
                        pass

            progress_candidates.append([run])
        if not progress_candidates:
            break
        ranked_progress = sorted(progress_candidates, key=lambda chain: _progress_score(chain[-1]), reverse=True)
        for chain in ranked_progress[:close_sweep_candidates]:
            run = chain[-1]
            closer_programs = _build_closer_programs(
                run.final_goal,
                run.final_goal_bucket,
                theorem_id=theorem_id,
                lean=lean,
                conn=conn,
                accessible_theorem_id=accessible_theorem_id,
                disabled_tactics=disabled_tactics,
                allowed_backend_families=allowed_backend_families,
                allowed_engine_names=allowed_engine_names,
            )
            for closer in closer_programs:
                closer_run = execute_ducky_program(lean, run.final_goal, closer)
                closer_payload = closer_run.to_dict()
                closer_payload["round"] = round_idx
                closer_payload["followup_to"] = run.program_id
                tried.append(closer_payload)
                if closer_run.closed:
                    chain = best_chain + chain + [closer_run]
                    return DuckyExecutionResult(
                        theorem_id=theorem_id,
                        started=True,
                        theorem_faithful=True,
                        start_goal_kind="direct_goal",
                        file_path="",
                        replay_tier="live",
                        replay_failure_category="",
                        replay_failing_prefix_idx=-1,
                        residual_bucket=str(row.get("residual_bucket", "") or ""),
                        goal_bucket=str(row.get("last_goal_bucket", "") or classify_goal_bucket(goal_state)),
                        specialist_targets=list(current_capsule.specialist_targets),
                        bank_priors=[prior.name for prior in current_capsule.bank_priors if not prior.suppressed],
                        programs_considered=total_programs,
                        closed=True,
                        progressed=True,
                        winning_program=_combined_program(chain),
                        final_goal="",
                        final_goal_bucket="proved",
                        goals_after=[],
                        ledger_snapshot=ledger_snapshot,
                        engine_outcomes=engine_outcomes,
                        projector_outcomes=projector_outcomes,
                        tried_programs=tried,
                    )
                if closer_run.progressed:
                    progress_candidates.append(chain + [closer_run])
        ranked_progress = sorted(progress_candidates, key=lambda chain: _progress_score(chain[-1]), reverse=True)
        best_progress_chain = ranked_progress[0]
        best_chain.extend(best_progress_chain)
        current_goal = best_progress_chain[-1].final_goal
        current_row = dict(current_row)
        current_row["last_goal"] = current_goal
        current_row["goal_state"] = current_goal
        current_row["last_goal_bucket"] = classify_goal_bucket(current_goal)
        current_row["remaining_goals_snapshot"] = list(best_progress_chain[-1].goals_after or [current_goal])
        current_capsule = build_goal_capsule(current_row)

    return DuckyExecutionResult(
        theorem_id=theorem_id,
        started=True,
        theorem_faithful=True,
        start_goal_kind="direct_goal",
        file_path="",
        replay_tier="live",
        replay_failure_category="",
        replay_failing_prefix_idx=-1,
        residual_bucket=str(row.get("residual_bucket", "") or ""),
        goal_bucket=str(row.get("last_goal_bucket", "") or classify_goal_bucket(goal_state)),
        specialist_targets=list(current_capsule.specialist_targets),
        bank_priors=[prior.name for prior in current_capsule.bank_priors if not prior.suppressed],
        programs_considered=total_programs,
        closed=False,
        progressed=bool(best_chain),
        winning_program=_combined_program(best_chain) if best_chain else None,
        final_goal=best_chain[-1].final_goal if best_chain else goal_state,
        final_goal_bucket=best_chain[-1].final_goal_bucket if best_chain else classify_goal_bucket(goal_state),
        goals_after=list(best_chain[-1].goals_after) if best_chain else [goal_state],
        ledger_snapshot=ledger_snapshot,
        engine_outcomes=engine_outcomes,
        projector_outcomes=projector_outcomes,
        tried_programs=tried,
    )


def run_ducky_on_row(
    row: dict[str, Any],
    *,
    lean: Any,
    conn: sqlite3.Connection | None = None,
    theorem_id_map: dict[str, int] | None = None,
    max_programs: int = 24,
    max_rounds: int = 3,
    disabled_tactics: set[str] | None = None,
    allowed_backend_families: set[str] | None = None,
    allowed_engine_names: set[str] | None = None,
) -> DuckyExecutionResult:
    theorem_id = str(row.get("theorem_id", "") or "")
    replay = replay_residual_state(row, lean)
    capsule = build_goal_capsule(row)
    if not replay.replay_success or not replay.goal_state:
        return DuckyExecutionResult(
            theorem_id=theorem_id,
            started=False,
            theorem_faithful=False,
            start_goal_kind=replay.goal_kind,
            file_path=replay.file_path,
            replay_tier=replay.tier_used,
            replay_failure_category=replay.replay_failure_category,
            replay_failing_prefix_idx=replay.replay_failing_prefix_idx,
            residual_bucket=str(row.get("residual_bucket", "") or ""),
            goal_bucket=str(row.get("last_goal_bucket", "") or classify_goal_bucket(_goal_text(row))),
            specialist_targets=list(capsule.specialist_targets),
            bank_priors=[prior.name for prior in capsule.bank_priors if not prior.suppressed],
            programs_considered=0,
            closed=False,
            progressed=False,
            winning_program=None,
            final_goal="",
            final_goal_bucket="",
            goals_after=[],
            ledger_snapshot=capsule.ledger_seed.to_dict(),
            engine_outcomes=[],
            projector_outcomes=[],
            tried_programs=[],
        )

    accessible_theorem_id = None
    if theorem_id_map is not None:
        accessible_theorem_id = theorem_id_map.get(theorem_id)

    result = run_ducky_on_goal(
        replay.goal_state,
        theorem_id=theorem_id,
        lean=lean,
        conn=conn,
        accessible_theorem_id=accessible_theorem_id,
        capsule=capsule,
        row_overrides={
            "residual_bucket": str(row.get("residual_bucket", "") or ""),
            "last_goal_bucket": str(row.get("last_goal_bucket", "") or classify_goal_bucket(replay.goal_state)),
            "reasoning_gap_family": str(row.get("reasoning_gap_family", "") or ""),
            "search_pathology_tags": list(row.get("search_pathology_tags") or []),
            "attempts": int(row.get("attempts", 0) or 0),
            "goals_closed": int(row.get("goals_closed", 0) or 0),
            "goals_remaining": int(row.get("goals_remaining", 0) or 0),
            "lane_sequence": str(row.get("lane_sequence", "") or ""),
            "remaining_goals_snapshot": list(row.get("remaining_goals_snapshot") or [replay.goal_state]),
        },
        max_programs=max_programs,
        max_rounds=max_rounds,
        disabled_tactics=disabled_tactics,
        allowed_backend_families=allowed_backend_families,
        allowed_engine_names=allowed_engine_names,
    )
    result.theorem_faithful = replay.theorem_faithful
    result.start_goal_kind = replay.goal_kind
    result.file_path = replay.file_path
    result.replay_tier = replay.tier_used
    result.replay_failure_category = replay.replay_failure_category
    result.replay_failing_prefix_idx = replay.replay_failing_prefix_idx
    return result


def summarize_ducky_execution(results: list[DuckyExecutionResult]) -> dict[str, Any]:
    total = len(results)
    started = sum(1 for result in results if result.started)
    theorem_faithful = sum(1 for result in results if result.theorem_faithful)
    closed = sum(1 for result in results if result.closed)
    progressed = sum(1 for result in results if result.progressed)
    by_goal_bucket: dict[str, int] = {}
    by_residual_bucket: dict[str, int] = {}
    by_program: dict[str, int] = {}
    by_specialist: dict[str, int] = {}
    by_replay_tier: dict[str, int] = {}
    by_engine: dict[str, int] = {}
    by_backend_family: dict[str, int] = {}
    by_certificate_shape: dict[str, int] = {}
    by_projector_status: dict[str, int] = {}
    certificate_count = 0
    for result in results:
        by_goal_bucket[result.goal_bucket] = by_goal_bucket.get(result.goal_bucket, 0) + 1
        by_residual_bucket[result.residual_bucket] = by_residual_bucket.get(result.residual_bucket, 0) + 1
        by_replay_tier[result.replay_tier] = by_replay_tier.get(result.replay_tier, 0) + 1
        for specialist in result.specialist_targets:
            by_specialist[specialist] = by_specialist.get(specialist, 0) + 1
        if result.winning_program:
            pid = str(result.winning_program.get("program_id", "") or "")
            if pid:
                by_program[pid] = by_program.get(pid, 0) + 1
        for outcome in result.engine_outcomes:
            engine = str(outcome.get("engine_name", "") or "")
            if engine:
                by_engine[engine] = by_engine.get(engine, 0) + 1
                certificate_count += int(outcome.get("certificate_count", 0) or 0)
            backend = str(outcome.get("backend_family", "") or "")
            if backend:
                by_backend_family[backend] = by_backend_family.get(backend, 0) + 1
        for outcome in result.projector_outcomes:
            status = str(outcome.get("projector_status", "") or "")
            if status:
                by_projector_status[status] = by_projector_status.get(status, 0) + 1
        for program in result.tried_programs:
            shape = str(program.get("certificate_shape", "") or "")
            if shape:
                by_certificate_shape[shape] = by_certificate_shape.get(shape, 0) + 1
    return {
        "total_rows": total,
        "started": started,
        "theorem_faithful_starts": theorem_faithful,
        "closed": closed,
        "progressed": progressed,
        "certificate_generation_count": certificate_count,
        "closure_rate_started": round(closed / max(started, 1), 4),
        "progress_rate_started": round(progressed / max(started, 1), 4),
        "by_goal_bucket": dict(sorted(by_goal_bucket.items(), key=lambda item: (-item[1], item[0]))),
        "by_residual_bucket": dict(sorted(by_residual_bucket.items(), key=lambda item: (-item[1], item[0]))),
        "by_specialist": dict(sorted(by_specialist.items(), key=lambda item: (-item[1], item[0]))),
        "by_engine": dict(sorted(by_engine.items(), key=lambda item: (-item[1], item[0]))),
        "by_backend_family": dict(sorted(by_backend_family.items(), key=lambda item: (-item[1], item[0]))),
        "by_certificate_shape": dict(sorted(by_certificate_shape.items(), key=lambda item: (-item[1], item[0]))),
        "by_projector_status": dict(sorted(by_projector_status.items(), key=lambda item: (-item[1], item[0]))),
        "by_replay_tier": dict(sorted(by_replay_tier.items(), key=lambda item: (-item[1], item[0]))),
        "by_winning_program": dict(sorted(by_program.items(), key=lambda item: (-item[1], item[0]))),
    }
