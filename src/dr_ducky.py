from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.hard_data_tags import (
    classify_goal_bucket,
    classify_reasoning_gap_family,
    goal_bucket_tags,
    sanitize_goal_text,
    trace_pathology_tags,
)
from src.hard_resolution_layer import (
    build_dependency_profile,
    build_residual_skeleton_geometry,
    build_search_control_geometry,
)

_COERCION_MARKERS = ("↑", "Nat.cast", "Int.cast", "Rat.cast", "OfNat.ofNat", "Subtype")
_NUMERIC_MARKERS = (" + ", " - ", " * ", " / ", "^", "≤", "≥", "<", ">", "dist", "‖", "abs")
_STRUCTURAL_MARKERS = (
    "IsIso",
    "Injective",
    "Surjective",
    "Bijective",
    "FormallyUnramified",
    "HasRingHomProperty",
    "IsOpenMap",
    "OpenEmbedding",
    "essImage",
)
_CATEGORY_MARKERS = ("CategoryTheory", "Functor", "NatTrans", "Adjunction", "Limits")
_MEMBERSHIP_MARKERS = (".carrier", "Submodule", "Ideal", "Subring", "PrimeSpectrum")
_WITNESS_MARKERS = ("∃", "Exists", "Nonempty", "zero", "image", "range", "sSup", "sInf", "iSup", "iInf")
_RECURSIVE_MARKERS = ("root", "fold", "rec", "iterate", "birthday", "succ", "pred")
_SYMBOLIC_SANDBOX_MARKERS = ("Complex.normSq", "sq", "pow", "Matrix.det", "traceMatrix", "discr")
_WITNESS_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")

_BANK_ENGINE_MAP: dict[str, str] = {
    "eq_sat": "EqSatEngine",
    "transport_normalizer": "EqSatEngine",
    "arith_nf": "ArithEngine",
    "solver_dispatch": "ArithEngine",
    "witness_constructor": "WitnessEngine",
    "canonical_witness": "WitnessEngine",
    "recursive_unfold_one": "RecursiveInvariantEngine",
    "loop_breaker": "RecursiveInvariantEngine",
    "eventual_filter_normalizer": "FiniteFilterEngine",
    "membership_exposure": "FiniteFilterEngine",
    "set_pointwise": "FiniteFilterEngine",
    "structural_close": "FiniteFilterEngine",
    "context_forward": "ContextTransportEngine",
    "local_fact_selector": "ContextTransportEngine",
    "binder_instantiation": "ContextTransportEngine",
    "iff_splitter": "ContextTransportEngine",
    "diagram_transport": "ContextTransportEngine",
}

_DEFAULT_PROJECTOR_POLICY = {
    "require_projected_program": True,
    "reject_no_goal_change": True,
    "prefer_have_chain": True,
    "prefer_calc_chain": True,
}

_DEFAULT_EXECUTION_BUDGETS = {
    "certificate_budget": 24,
    "projector_budget": 24,
    "execution_rounds": 3,
}

_ENGINE_BACKEND_FAMILY: dict[str, str] = {
    "EqSatEngine": "egglog_eqsat",
    "ArithEngine": "lean_arith",
    "WitnessEngine": "rosette_proof_dsl",
    "RecursiveInvariantEngine": "symbolic_rewrite_vm",
    "FiniteFilterEngine": "kodkod_relational",
    "ContextTransportEngine": "rosette_proof_dsl",
}


def _namespace_prefix(theorem_id: str) -> str:
    theorem = (theorem_id or "").strip()
    if "." in theorem:
        return theorem.split(".", 1)[0]
    return "(root)"


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _local_hypotheses_from_text(goal_text: str) -> list[tuple[str, str]]:
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
        name, typ = line.split(" : ", 1)
        if name.strip() and typ.strip():
            out.append((name.strip(), typ.strip()))
    return out


def _ledger_fact_kind(typ: str) -> str:
    text = sanitize_goal_text(typ)
    if " ↔ " in text:
        return "iff"
    if " = " in text:
        return "equality"
    if "∈" in text or "⊆" in text:
        return "membership"
    if text.startswith("∃") or "Exists" in text:
        return "witness"
    return "proposition"


def _witness_terms_from_text(text: str) -> list[str]:
    out = ["0", "1"]
    for token in _WITNESS_TOKEN_RE.findall(sanitize_goal_text(text or "")):
        if token[0].isupper():
            continue
        if len(token) <= 24:
            out.append(token)
    return _dedupe(out)[:16]


def infer_projector_markers(
    goal_bucket: str,
    *,
    signals: dict[str, Any],
    representation_pressures: list[str],
) -> list[str]:
    markers: list[str] = []
    if goal_bucket in {"equality", "inequality"} or signals.get("has_symbolic_sandbox_surface"):
        markers.append("rewrite_chain")
    if signals.get("has_witness_pressure") or goal_bucket == "exists":
        markers.append("witness_construction")
    if signals.get("recursive_loop_risk"):
        markers.append("recursive_bridge")
    if signals.get("structural_close_candidate") or "structural_property_closure" in representation_pressures:
        markers.append("structural_close")
    if goal_bucket in {"atomic_prop", "forall", "iff"} or "forward_context_chase" in representation_pressures:
        markers.append("have_chain")
    if goal_bucket == "equality" and signals.get("has_symbolic_sandbox_surface"):
        markers.append("calc_chain")
    return _dedupe(markers)


def infer_engine_eligibility(
    goal_bucket: str,
    *,
    signals: dict[str, Any],
    representation_pressures: list[str],
) -> list[str]:
    engines: list[str] = []
    if goal_bucket in {"equality", "other"} or signals.get("has_symbolic_sandbox_surface") or signals.get("coercion_count", 0) > 0:
        engines.append("EqSatEngine")
    if goal_bucket == "inequality" or signals.get("has_numeric_surface"):
        engines.append("ArithEngine")
    if goal_bucket == "exists" or signals.get("has_witness_pressure") or "canonical_object_witness" in representation_pressures:
        engines.append("WitnessEngine")
    if signals.get("has_recursive_pressure") or signals.get("recursive_loop_risk"):
        engines.append("RecursiveInvariantEngine")
    if goal_bucket in {"membership", "subset"} or "eventual_filter_reasoning" in representation_pressures or signals.get("structural_close_candidate"):
        engines.append("FiniteFilterEngine")
    if goal_bucket in {"atomic_prop", "forall", "iff", "membership", "equality", "inequality"} or "forward_context_chase" in representation_pressures:
        engines.append("ContextTransportEngine")
    return _dedupe(engines)


def build_ledger_seed(row: dict[str, Any], spec: GoalSpecification | None = None) -> ProofShadowLedger:
    goal_text = str(row.get("last_goal") or row.get("goal_state") or "")
    if spec is None:
        spec = build_goal_specification(row)
    local_hyps = _local_hypotheses_from_text(goal_text)
    facts = [
        LedgerFact(
            fact_id=f"seed_fact_{idx}",
            expr=name,
            fact_kind=_ledger_fact_kind(typ),
            proposition=sanitize_goal_text(typ),
            source_space="goal_text",
            derivation_kind="surface_extract",
            provenance={"theorem_id": spec.theorem_id},
        )
        for idx, (name, typ) in enumerate(local_hyps)
    ]
    candidate_rewrites = [
        fact.expr
        for fact in facts
        if fact.fact_kind in {"equality", "iff"}
    ][:12]
    return ProofShadowLedger(
        facts=facts,
        accessible_premises=[],
        candidate_witnesses=_witness_terms_from_text(spec.goal_text),
        candidate_rewrites=candidate_rewrites,
        engine_outcomes=[],
        rejected_branches=[],
    )


def allowed_engines_from_priors(priors: list[BankPrior]) -> list[str]:
    engines: list[str] = []
    for prior in priors:
        if prior.suppressed:
            continue
        engine = _BANK_ENGINE_MAP.get(prior.name)
        if engine:
            engines.append(engine)
    return _dedupe(engines)


def backend_preferences_for_spec(
    spec: GoalSpecification,
    allowed_engines: list[str],
) -> list[str]:
    backends: list[str] = []
    if "EqSatEngine" in allowed_engines:
        backends.append("egglog_eqsat")
    if any(engine in allowed_engines for engine in ("WitnessEngine", "ContextTransportEngine")):
        backends.append("rosette_proof_dsl")
    if "FiniteFilterEngine" in allowed_engines:
        backends.append("kodkod_relational")
    if "RecursiveInvariantEngine" in allowed_engines:
        backends.append("symbolic_rewrite_vm")
    if "ArithEngine" in allowed_engines:
        backends.append("lean_arith")
    if spec.goal_bucket in {"membership", "exists", "subset"}:
        backends.append("kodkod_relational")
    return _dedupe(backends)


def build_solver_constraints(
    spec: GoalSpecification,
    skeletons: list[ProofSkeleton],
    ledger_seed: ProofShadowLedger,
) -> list[ProofConstraint]:
    constraints: list[ProofConstraint] = []
    for fact in ledger_seed.facts[:16]:
        constraints.append(
            ProofConstraint(
                constraint_id=f"fact:{fact.fact_id}",
                constraint_kind=fact.fact_kind,
                expr=fact.proposition,
                source_space=fact.source_space,
                solver_family="rosette_proof_dsl",
                provenance={"expr": fact.expr, "derivation_kind": fact.derivation_kind},
            )
        )
    for skeleton in skeletons[:20]:
        for hole in skeleton.holes:
            constraints.append(
                ProofConstraint(
                    constraint_id=f"hole:{skeleton.skeleton_id}:{hole.hole_id}",
                    constraint_kind=hole.hole_kind,
                    expr=hole.hole_id,
                    source_space=hole.source_space,
                    solver_family="rosette_proof_dsl",
                    required=hole.required,
                    provenance={
                        "skeleton_id": skeleton.skeleton_id,
                        "bank": skeleton.bank,
                        "compatibility": dict(hole.compatibility),
                    },
                )
            )
    return constraints


def build_eqsat_plan(
    spec: GoalSpecification,
    allowed_engines: list[str],
    execution_budgets: dict[str, int],
) -> EqSatPlan | None:
    if "EqSatEngine" not in allowed_engines:
        return None
    rewrite_theories = ["core_eq"]
    if spec.goal_bucket == "equality":
        rewrite_theories.append("algebraic_eq")
    if spec.signals.get("coercion_count", 0) > 0:
        rewrite_theories.append("coercion_normalization")
    if "Complex.normSq" in spec.goal_text or "pow" in spec.goal_text or "^ 2" in spec.goal_text:
        rewrite_theories.append("ring_norm")
    return EqSatPlan(
        plan_id=f"eqsat:{spec.theorem_id or 'anon'}",
        backend_family="egglog_eqsat",
        rewrite_theories=_dedupe(rewrite_theories),
        extraction_cost_model="ast_size_then_symbol_weight",
        explanation_mode="proof_path",
        node_budget=int(execution_budgets.get("certificate_budget", 24)) * 8,
    )


def build_relational_search_specs(
    spec: GoalSpecification,
    ledger_seed: ProofShadowLedger,
) -> list[RelationalSearchSpec]:
    relation_symbols: list[str] = []
    text = spec.goal_text
    if "∈" in text:
        relation_symbols.append("membership")
    if "⊆" in text:
        relation_symbols.append("subset")
    if "Filter" in text or "=ᶠ" in text or "Tendsto" in text:
        relation_symbols.append("filter")
    if spec.goal_bucket == "exists":
        relation_symbols.append("witness")
    if not relation_symbols:
        return []
    universe_atoms = _dedupe(
        [fact.expr for fact in ledger_seed.facts[:12]] + ledger_seed.candidate_witnesses[:8]
    )[:16]
    return [
        RelationalSearchSpec(
            spec_id=f"rel:{spec.theorem_id or 'anon'}:{idx}",
            backend_family="kodkod_relational",
            relation_symbols=[symbol],
            universe_atoms=universe_atoms,
            bound_strategy="goal_surface_plus_local_facts",
            max_tuples=max(8, len(universe_atoms)),
            witness_roles=["candidate_witness"] if symbol == "witness" else [],
        )
        for idx, symbol in enumerate(_dedupe(relation_symbols))
    ]


def build_proof_dsl_programs(
    skeletons: list[ProofSkeleton],
    constraints: list[ProofConstraint],
) -> list[ProofDSLProgram]:
    programs: list[ProofDSLProgram] = []
    constraint_ids = [constraint.constraint_id for constraint in constraints]
    for skeleton in skeletons[:24]:
        backend_family = _ENGINE_BACKEND_FAMILY.get(_BANK_ENGINE_MAP.get(skeleton.bank, ""), "rosette_proof_dsl")
        steps = [
            ProofDSLStep(
                step_id=f"{skeleton.skeleton_id}:collect",
                op="collect_candidates",
                args={"hole_ids": [hole.hole_id for hole in skeleton.holes]},
                produces=["candidate_bindings"],
                backend_hint=backend_family,
            ),
            ProofDSLStep(
                step_id=f"{skeleton.skeleton_id}:project",
                op="project_certificate",
                args={"certificate_kinds": list(skeleton.certificate_kinds or ["projected_tactics"])},
                consumes=["candidate_bindings"],
                produces=["lean_program"],
                backend_hint="proof_projector_v1",
            ),
        ]
        extraction_policy = {
            "prefer_exact": skeleton.skeleton_kind in {"local_fact_transport", "premise_bridge"},
            "prefer_calc": "calc_chain" in skeleton.certificate_kinds,
        }
        programs.append(
            ProofDSLProgram(
                program_id=f"dsl:{skeleton.skeleton_id}",
                skeleton_id=skeleton.skeleton_id,
                backend_family=backend_family,
                steps=steps,
                constraint_ids=constraint_ids[: min(len(constraint_ids), 12)],
                extraction_policy=extraction_policy,
            )
        )
    return programs


def projector_policy_for_spec(spec: GoalSpecification) -> dict[str, Any]:
    policy = dict(_DEFAULT_PROJECTOR_POLICY)
    policy["prefer_calc_chain"] = bool(spec.goal_bucket == "equality" and spec.signals.get("has_symbolic_sandbox_surface"))
    policy["prefer_have_chain"] = bool(spec.goal_bucket in {"atomic_prop", "forall", "iff"} or "forward_context_chase" in spec.representation_pressures)
    policy["prefer_structural_close"] = bool(spec.signals.get("structural_close_candidate"))
    return policy


def execution_budgets_for_spec(spec: GoalSpecification) -> dict[str, int]:
    budgets = dict(_DEFAULT_EXECUTION_BUDGETS)
    if spec.signals.get("recursive_loop_risk"):
        budgets["execution_rounds"] = 4
    if spec.goal_bucket in {"exists", "membership"}:
        budgets["certificate_budget"] = 32
    return budgets


@dataclass
class BankPrior:
    name: str
    weight: float
    rationale: str
    suppressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoalPrescription:
    prescription_kind: str
    description: str
    rationale: str
    bank: str
    priority_band: str
    estimated_gain: float
    action_hint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LedgerFact:
    fact_id: str
    expr: str
    fact_kind: str
    proposition: str
    source_space: str
    derivation_kind: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofShadowLedger:
    facts: list[LedgerFact]
    accessible_premises: list[str]
    candidate_witnesses: list[str]
    candidate_rewrites: list[str]
    engine_outcomes: list[dict[str, Any]] = field(default_factory=list)
    rejected_branches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [fact.to_dict() for fact in self.facts],
            "accessible_premises": list(self.accessible_premises),
            "candidate_witnesses": list(self.candidate_witnesses),
            "candidate_rewrites": list(self.candidate_rewrites),
            "engine_outcomes": list(self.engine_outcomes),
            "rejected_branches": list(self.rejected_branches),
        }


@dataclass
class ProofConstraint:
    constraint_id: str
    constraint_kind: str
    expr: str
    source_space: str
    solver_family: str
    required: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofDSLStep:
    step_id: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    backend_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofDSLProgram:
    program_id: str
    skeleton_id: str
    backend_family: str
    steps: list[ProofDSLStep]
    constraint_ids: list[str] = field(default_factory=list)
    extraction_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "skeleton_id": self.skeleton_id,
            "backend_family": self.backend_family,
            "steps": [step.to_dict() for step in self.steps],
            "constraint_ids": list(self.constraint_ids),
            "extraction_policy": dict(self.extraction_policy),
        }


@dataclass
class EqSatPlan:
    plan_id: str
    backend_family: str
    rewrite_theories: list[str]
    extraction_cost_model: str
    explanation_mode: str
    node_budget: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationalSearchSpec:
    spec_id: str
    backend_family: str
    relation_symbols: list[str]
    universe_atoms: list[str]
    bound_strategy: str
    max_tuples: int
    witness_roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MathEngineRequest:
    request_id: str
    engine_name: str
    active_bank: str
    goal_text: str
    goal_bucket: str
    allowed_transformations: list[str]
    budget: int
    ledger: ProofShadowLedger
    backend_family: str = ""
    proof_dsl_program_ids: list[str] = field(default_factory=list)
    constraint_ids: list[str] = field(default_factory=list)
    eqsat_plan_id: str = ""
    relational_spec_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "engine_name": self.engine_name,
            "active_bank": self.active_bank,
            "goal_text": self.goal_text,
            "goal_bucket": self.goal_bucket,
            "allowed_transformations": list(self.allowed_transformations),
            "budget": self.budget,
            "ledger": self.ledger.to_dict(),
            "backend_family": self.backend_family,
            "proof_dsl_program_ids": list(self.proof_dsl_program_ids),
            "constraint_ids": list(self.constraint_ids),
            "eqsat_plan_id": self.eqsat_plan_id,
            "relational_spec_ids": list(self.relational_spec_ids),
        }


@dataclass
class EngineCertificate:
    certificate_id: str
    engine_name: str
    bank: str
    skeleton_id: str
    certificate_kind: str
    summary: str
    bindings: dict[str, str] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    projected_tactics: list[str] = field(default_factory=list)
    target_before: str = ""
    target_after_hint: str = ""
    backend_family: str = ""
    proof_dsl_program_id: str = ""
    constraint_ids: list[str] = field(default_factory=list)
    explanation_trace: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectedProofProgram:
    program_id: str
    bank: str
    specialist: str
    skeleton_id: str
    tactics: list[str]
    rationale: str
    score: float
    certificate_id: str
    certificate_shape: str
    bindings: dict[str, str] = field(default_factory=dict)
    projector_backend: str = "proof_projector_v1"
    derivation_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectorDecision:
    certificate_id: str
    projector_status: str
    projected_program_id: str = ""
    compiled: bool = False
    progressed: bool = False
    closed: bool = False
    rejection_reason: str = ""
    certificate_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofHoleSpec:
    hole_id: str
    hole_kind: str
    source_space: str
    required: bool = True
    max_candidates: int = 8
    compatibility: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofSkeleton:
    skeleton_id: str
    skeleton_kind: str
    bank: str
    specialist: str
    tactic_templates: list[str]
    holes: list[ProofHoleSpec]
    rationale: str
    priority: float
    certificate_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton_id": self.skeleton_id,
            "skeleton_kind": self.skeleton_kind,
            "bank": self.bank,
            "specialist": self.specialist,
            "tactic_templates": list(self.tactic_templates),
            "holes": [hole.to_dict() for hole in self.holes],
            "rationale": self.rationale,
            "priority": round(self.priority, 3),
            "certificate_kinds": list(self.certificate_kinds),
        }


@dataclass
class GoalSpecification:
    theorem_id: str
    namespace_prefix: str
    goal_text: str
    goal_bucket: str
    goal_tags: list[str]
    reasoning_gap_family: str
    residual_bucket: str
    difficulty_band: str
    goals_closed: int
    goals_remaining: int
    attempts: int
    lane_history: list[str]
    pathology_tags: list[str]
    signals: dict[str, Any]
    domain_hints: list[str]
    representation_pressures: list[str]
    dependency_profile: dict[str, Any]
    search_control: dict[str, Any]
    residual_geometry: dict[str, Any]
    projector_markers: list[str]
    engine_eligibility: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoalCapsule:
    specification: GoalSpecification
    specialist_targets: list[str]
    suppression_hints: list[str]
    bank_priors: list[BankPrior]
    prescriptions: list[GoalPrescription]
    proof_skeletons: list[ProofSkeleton]
    ledger_seed: ProofShadowLedger
    allowed_engines: list[str]
    projector_policy: dict[str, Any]
    execution_budgets: dict[str, int]
    priority_score: float = 0.0
    proof_dsl_programs: list[ProofDSLProgram] = field(default_factory=list)
    solver_constraints: list[ProofConstraint] = field(default_factory=list)
    eqsat_plan: EqSatPlan | None = None
    relational_search_specs: list[RelationalSearchSpec] = field(default_factory=list)
    backend_preferences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification.to_dict(),
            "specialist_targets": self.specialist_targets,
            "suppression_hints": self.suppression_hints,
            "bank_priors": [prior.to_dict() for prior in self.bank_priors],
            "prescriptions": [prescription.to_dict() for prescription in self.prescriptions],
            "proof_skeletons": [skeleton.to_dict() for skeleton in self.proof_skeletons],
            "ledger_seed": self.ledger_seed.to_dict(),
            "allowed_engines": list(self.allowed_engines),
            "projector_policy": dict(self.projector_policy),
            "execution_budgets": dict(self.execution_budgets),
            "proof_dsl_programs": [program.to_dict() for program in self.proof_dsl_programs],
            "solver_constraints": [constraint.to_dict() for constraint in self.solver_constraints],
            "eqsat_plan": self.eqsat_plan.to_dict() if self.eqsat_plan is not None else None,
            "relational_search_specs": [spec.to_dict() for spec in self.relational_search_specs],
            "backend_preferences": list(self.backend_preferences),
            "priority_score": round(self.priority_score, 3),
        }


def _signal_map(
    row: dict[str, Any],
    goal_text: str,
    *,
    goal_bucket: str,
    domain_hints: list[str],
    representation_pressures: list[str],
    dependency_profile: dict[str, Any],
    search_control: dict[str, Any],
) -> dict[str, Any]:
    theorem_id = str(row.get("theorem_id", "") or "")
    text = sanitize_goal_text(goal_text or "")
    namespace = _namespace_prefix(theorem_id)
    coercions = _count_markers(text, _COERCION_MARKERS)
    has_numeric = _has_any(text, _NUMERIC_MARKERS)
    has_structural = _has_any(text, _STRUCTURAL_MARKERS)
    has_category = namespace == "CategoryTheory" or _has_any(text, _CATEGORY_MARKERS)
    has_membership_wall = "∈" in text and _has_any(text, _MEMBERSHIP_MARKERS)
    has_witness_pressure = _has_any(text, _WITNESS_MARKERS)
    has_recursive_pressure = _has_any(text, _RECURSIVE_MARKERS)
    has_symbolic_sandbox_surface = has_numeric or coercions > 0 or _has_any(text, _SYMBOLIC_SANDBOX_MARKERS)
    lane_history = [lane for lane in str(row.get("lane_sequence", "") or "").split("→") if lane]
    recursive_loop_risk = bool(
        has_recursive_pressure
        and (
            "goal_explosion" in representation_pressures
            or "branch_pruning" in representation_pressures
            or search_control.get("plateau_detected")
            or any(
                tag in (row.get("search_pathology_tags") or [])
                for tag in ("duplicate_goal_progress", "duplicate_goal_pseudo_progress")
            )
        )
    )
    structural_close_candidate = bool(
        has_structural
        or has_category
        or "structural_property_closure" in representation_pressures
        or "close_before_unpack" in representation_pressures
        or "structural_theorem_retrieval" in representation_pressures
    )
    symbolic_sandbox_candidate = bool(
        has_symbolic_sandbox_surface
        and goal_bucket in {"equality", "inequality", "forall", "other", "membership"}
    )
    return {
        "text_length": len(text),
        "token_length": len(text.split()),
        "binder_count": text.count("∀") + text.count("∃"),
        "coercion_count": coercions,
        "has_numeric_surface": has_numeric,
        "has_structural_surface": has_structural,
        "has_category_surface": has_category,
        "has_membership_wall": has_membership_wall,
        "has_witness_pressure": has_witness_pressure,
        "has_recursive_pressure": has_recursive_pressure,
        "has_symbolic_sandbox_surface": has_symbolic_sandbox_surface,
        "symbolic_sandbox_candidate": symbolic_sandbox_candidate,
        "recursive_loop_risk": recursive_loop_risk,
        "structural_close_candidate": structural_close_candidate,
        "side_condition_cluster": dependency_profile.get("side_condition_profile") == "repeated_small_goals",
        "lane_count": len(lane_history),
        "namespace_prefix": namespace,
        "domain_hints": list(domain_hints),
    }


def build_goal_specification(row: dict[str, Any]) -> GoalSpecification:
    goal_text = str(row.get("last_goal") or row.get("goal_state") or "")
    goal_bucket = str(row.get("last_goal_bucket") or classify_goal_bucket(goal_text))
    pathology_tags = list(row.get("search_pathology_tags") or [])
    if not pathology_tags:
        pathology_tags = trace_pathology_tags(
            row.get("step_trace"),
            remaining_goals=row.get("remaining_goals_snapshot"),
        )
    reasoning_gap_family = str(row.get("reasoning_gap_family") or "")
    if not reasoning_gap_family:
        reasoning_gap_family = classify_reasoning_gap_family(
            success=bool(row.get("success")),
            started=bool(row.get("started", True)),
            residual_bucket=str(row.get("residual_bucket", "") or ""),
            last_goal_bucket=goal_bucket,
            goal_text=goal_text,
            remaining_goals=row.get("remaining_goals_snapshot"),
            pathology_tags=pathology_tags,
        )
    remaining_goals = [
        str(goal)
        for goal in (row.get("remaining_goals_snapshot") or [])
        if isinstance(goal, str) and str(goal).strip()
    ]
    if not remaining_goals and goal_text:
        remaining_goals = [goal_text]
    dependency_profile = build_dependency_profile(remaining_goals)
    skeleton_geometry = build_residual_skeleton_geometry(row, remaining_goals)
    search_control = build_search_control_geometry(row)
    signals = _signal_map(
        row,
        goal_text,
        goal_bucket=goal_bucket,
        domain_hints=list(skeleton_geometry.get("domain_hints", []) or []),
        representation_pressures=list(skeleton_geometry.get("representation_pressures", []) or []),
        dependency_profile=dependency_profile,
        search_control=search_control,
    )
    residual_geometry = {
        "domain_hints": list(skeleton_geometry.get("domain_hints", []) or []),
        "representation_pressures": list(skeleton_geometry.get("representation_pressures", []) or []),
        "dependency_profile": dependency_profile,
        "search_control_geometry": search_control,
        "pathology_tags": pathology_tags,
    }
    projector_markers = infer_projector_markers(
        goal_bucket,
        signals=signals,
        representation_pressures=list(skeleton_geometry.get("representation_pressures", []) or []),
    )
    engine_eligibility = infer_engine_eligibility(
        goal_bucket,
        signals=signals,
        representation_pressures=list(skeleton_geometry.get("representation_pressures", []) or []),
    )
    return GoalSpecification(
        theorem_id=str(row.get("theorem_id", "") or ""),
        namespace_prefix=signals["namespace_prefix"],
        goal_text=sanitize_goal_text(goal_text),
        goal_bucket=goal_bucket,
        goal_tags=goal_bucket_tags(goal_text),
        reasoning_gap_family=reasoning_gap_family,
        residual_bucket=str(row.get("residual_bucket", "") or ""),
        difficulty_band=str(row.get("difficulty_band", "") or ""),
        goals_closed=int(row.get("goals_closed", 0) or 0),
        goals_remaining=int(row.get("goals_remaining", 0) or 0),
        attempts=int(row.get("attempts", 0) or 0),
        lane_history=[lane for lane in str(row.get("lane_sequence", "") or "").split("→") if lane],
        pathology_tags=pathology_tags,
        signals=signals,
        domain_hints=list(skeleton_geometry.get("domain_hints", []) or []),
        representation_pressures=list(skeleton_geometry.get("representation_pressures", []) or []),
        dependency_profile=dependency_profile,
        search_control=search_control,
        residual_geometry=residual_geometry,
        projector_markers=projector_markers,
        engine_eligibility=engine_eligibility,
    )


def infer_specialist_targets(spec: GoalSpecification) -> list[str]:
    pressures = set(spec.representation_pressures)
    targets: list[str] = []
    if spec.signals["recursive_loop_risk"] or {"loop_escape", "branch_pruning", "plateau_escape"} & pressures:
        targets.append("recursive_circuit_breaker")
    if spec.signals["symbolic_sandbox_candidate"]:
        targets.append("symbolic_sandbox")
    if spec.goal_bucket in {"equality", "inequality"} or spec.reasoning_gap_family in {"local_eq_close", "local_ineq_close"}:
        targets.append("human_calculator")
    if spec.goal_bucket == "membership" or {"opaque_membership_unfolding", "pointwise_set_reduction"} & pressures:
        targets.append("membership_surface_engine")
    if spec.goal_bucket == "exists" or {"witness_exposure", "witness_construction", "canonical_object_witness"} & pressures:
        targets.append("witness_engine")
    if spec.reasoning_gap_family == "forward_context_close" or "forward_context_chase" in pressures:
        targets.append("context_transport")
    if spec.signals["structural_close_candidate"]:
        targets.append("structural_closer")
    if spec.goal_bucket == "forall" or "binder_introduction" in pressures:
        targets.append("binder_drilldown")
    if spec.goal_bucket == "iff":
        targets.append("logic_splitter")
    if spec.goal_bucket == "atomic_prop":
        targets.append("atomic_fact_engine")
    if spec.signals["side_condition_cluster"]:
        targets.append("side_condition_sweeper")
    if "metavariable_repair" in pressures:
        targets.append("metavariable_quarantine")
    if "eventual_filter_reasoning" in pressures:
        targets.append("filter_reasoner")
    return _dedupe(targets)


def infer_suppression_hints(spec: GoalSpecification) -> list[str]:
    pressures = set(spec.representation_pressures)
    hints: list[str] = []
    if spec.signals["recursive_loop_risk"]:
        hints.extend(
            [
                "suppress_repeat_rw",
                "suppress_repeat_norm_num",
                "suppress_full_recursive_unfold",
            ]
        )
    if spec.search_control.get("plateau_detected"):
        hints.append("suppress_blank_lane_retry")
    if "domain_solver_mismatch_risk" in pressures or (
        spec.signals["has_category_surface"] and not spec.signals["has_numeric_surface"]
    ):
        hints.append("suppress_numeric_solvers")
    if "close_before_unpack" in pressures or spec.signals["structural_close_candidate"]:
        hints.append("suppress_definition_unfold")
    if "metavariable_repair" in pressures or "metavariable_corruption" in spec.pathology_tags:
        hints.extend(["suppress_backward_rw", "suppress_bare_apply"])
    if spec.signals["side_condition_cluster"]:
        hints.append("suppress_global_replanner")
    if spec.signals["symbolic_sandbox_candidate"]:
        hints.append("prefer_symbolic_sandbox")
    return _dedupe(hints)


def infer_bank_priors(
    spec: GoalSpecification,
    specialist_targets: list[str] | None = None,
    suppression_hints: list[str] | None = None,
) -> list[BankPrior]:
    specialist_targets = specialist_targets or infer_specialist_targets(spec)
    suppression_hints = suppression_hints or infer_suppression_hints(spec)
    pressures = set(spec.representation_pressures)
    priors: dict[str, BankPrior] = {}

    def add(name: str, weight: float, rationale: str, *, suppressed: bool = False) -> None:
        existing = priors.get(name)
        candidate = BankPrior(name=name, weight=round(weight, 3), rationale=rationale, suppressed=suppressed)
        if existing is None:
            priors[name] = candidate
            return
        if candidate.suppressed and not existing.suppressed:
            priors[name] = candidate
            return
        if candidate.suppressed == existing.suppressed and candidate.weight > existing.weight:
            priors[name] = candidate

    def suppress(name: str, rationale: str, *, weight: float = 0.95) -> None:
        add(name, weight, rationale, suppressed=True)

    if spec.goal_bucket == "equality" or "human_calculator" in specialist_targets:
        weight = 0.92 + (0.03 if spec.signals["has_symbolic_sandbox_surface"] else 0.0)
        add("eq_sat", weight, "Equality residuals benefit from deterministic rewrite saturation in a sandbox.")
    if spec.signals["coercion_count"] > 0 or "transport_alignment" in pressures:
        add("transport_normalizer", 0.9, "Visible coercions and transports should be aligned before final closure.")
    if spec.goal_bucket == "inequality" or spec.signals["has_numeric_surface"]:
        add("arith_nf", 0.84, "Numeric/order surface suggests algebraic or linear normalization.")
        add("solver_dispatch", 0.74, "A lightweight arithmetic delegate may cheaply discharge the goal.")
    if spec.goal_bucket in {"membership", "subset"} or spec.signals["has_membership_wall"]:
        add("membership_exposure", 0.88, "Membership residuals often require exposing carrier or mem lemmas.")
        add("set_pointwise", 0.76, "Set-theoretic goals benefit from pointwise reduction after exposure.")
    if spec.goal_bucket == "exists" or spec.signals["has_witness_pressure"]:
        add("witness_constructor", 0.87, "Existential structure suggests canonical witness search.")
    if "canonical_object_witness" in pressures or spec.reasoning_gap_family == "witness_construction_close":
        add("canonical_witness", 0.83, "Domain structure suggests a small canonical witness family.")
    if spec.goal_bucket == "forall" or "binder_introduction" in pressures:
        add("binder_instantiation", 0.82, "Universal residuals often reduce under a controlled intro/instantiation pass.")
    if spec.goal_bucket == "iff":
        add("iff_splitter", 0.82, "Bidirectional logical residuals should be split and solved directionally.")
    if spec.goal_bucket == "atomic_prop":
        add("local_fact_selector", 0.78, "Atomic goals are often closed by local fact or instance selection.")
    elif spec.goal_bucket in {"equality", "inequality"}:
        add("local_fact_selector", 0.73, "Near-miss equalities and inequalities often close from one local bridge fact.")
    if spec.reasoning_gap_family == "forward_context_close" or "forward_context_chase" in pressures:
        add("context_forward", 0.9, "Reasoning gap indicates the missing move is a context bridge.")
        add("diagram_transport", 0.82, "Local transport lemmas should be mined before broad rewriting.")
        add("local_fact_selector", 0.81, "Context-bridge residues often collapse to a direct local fact after transport.")
    elif spec.goal_bucket in {"membership", "equality", "inequality"}:
        add("context_forward", 0.68, "Local hypotheses often bridge near-miss local residuals.")
    if spec.signals["structural_close_candidate"]:
        add("structural_close", 0.86, "Abstract structural goals should try close-before-unpack.")
    if "categorical_composition_normalization" in pressures:
        add("categorical_comp_normalizer", 0.84, "Category-theoretic composition chains need associativity transport, not raw rewriting.")
    if "hypothesis_injection" in pressures:
        add("hypothesis_injection", 0.82, "Injective/surjective or determinant-style goals need local hypothesis transport.")
    if "extensionality_reduction" in pressures or (spec.goal_bucket in {"equality", "forall"} and "⇑" in spec.goal_text):
        add("extensionality_bridge", 0.79, "Functional residuals should reduce pointwise before broad search.")
    if "eventual_filter_reasoning" in pressures:
        add("eventual_filter_normalizer", 0.8, "Eventual/filter obligations should be normalized explicitly.")
    if spec.signals["has_recursive_pressure"]:
        weight = 0.9 if spec.signals["recursive_loop_risk"] else 0.68
        add("recursive_unfold_one", weight, "Recursive markers justify bounded one-layer unfolding with re-evaluation.")
    if spec.search_control.get("plateau_detected") or "blank_lane_plateau" in spec.pathology_tags:
        weight = 0.91 if spec.signals["recursive_loop_risk"] else 0.78
        add("loop_breaker", weight, "Blank-lane or plateau traces justify forcing an explicit specialist move next.")
    if {"goal_explosion", "duplicate_goal_pseudo_progress"} & set(spec.pathology_tags):
        add("branch_pruning", 0.85, "Duplicate-goal pseudo-progress should trigger pruning instead of more search.")
    if "metavariable_repair" in pressures:
        add("metavariable_quarantine", 0.9, "Metavariable-corrupted branches should be quarantined before local repair.")
    if spec.signals["side_condition_cluster"]:
        add("side_condition_sweep", 0.84, "Repeated small side-goals should be swept together, not replanned globally.")

    if "suppress_numeric_solvers" in suppression_hints:
        suppress("solver_dispatch", "Abstract structural domains without numeric content should demote numeric solvers.")
        if not spec.signals["has_numeric_surface"]:
            suppress("arith_nf", "There is no real arithmetic payload to normalize in this domain.")
    if spec.signals["recursive_loop_risk"] and spec.goal_bucket == "equality":
        suppress("eq_sat", "Recursive loop risk means equality saturation over recursive symbols may repeat the same abyss.")
    if "suppress_definition_unfold" in suppression_hints:
        suppress("definition_unfold", "Structural goals should close before full definitional unpacking.")
    if "suppress_backward_rw" in suppression_hints:
        suppress("backward_rw", "Backward rewrites are unsafe under metavariable or loop pressure.")

    return sorted(priors.values(), key=lambda prior: (prior.suppressed, -prior.weight, prior.name))


def build_goal_prescriptions(
    spec: GoalSpecification,
    priors: list[BankPrior],
    specialist_targets: list[str] | None = None,
    suppression_hints: list[str] | None = None,
) -> list[GoalPrescription]:
    specialist_targets = specialist_targets or infer_specialist_targets(spec)
    suppression_hints = suppression_hints or infer_suppression_hints(spec)
    prescriptions: list[GoalPrescription] = []
    enabled = [prior for prior in priors if not prior.suppressed]
    enabled_names = {prior.name for prior in enabled}

    def add(
        kind: str,
        bank: str,
        description: str,
        rationale: str,
        gain: float,
        priority_band: str = "P1",
        **action_hint: Any,
    ) -> None:
        if bank not in enabled_names and bank not in {"definition_unfold", "backward_rw"}:
            return
        prescriptions.append(
            GoalPrescription(
                prescription_kind=kind,
                description=description,
                rationale=rationale,
                bank=bank,
                priority_band=priority_band,
                estimated_gain=round(gain, 3),
                action_hint=action_hint,
            )
        )

    top = enabled[0].name if enabled else "structural_close"

    if "recursive_circuit_breaker" in specialist_targets:
        add(
            "recursive_loop_circuit_break",
            "loop_breaker",
            "Stop generic lane replay and force a specialist local move next.",
            "Recursive plateaus are controller failures, not invitations to keep rewriting.",
            0.94,
            op="prune",
            target="goal",
        )
        add(
            "bounded_unfold",
            "recursive_unfold_one",
            "Permit at most one recursive unfolding layer before re-evaluating the capsule.",
            "Near misses should not drift back into unbounded recursive expansion.",
            0.9,
            op="unfold_once",
            target="goal",
        )
    if "symbolic_sandbox" in specialist_targets:
        bank = "transport_normalizer" if "transport_normalizer" in enabled_names else "eq_sat"
        add(
            "enter_symbolic_sandbox",
            bank,
            "Extract the algebraic surface into a deterministic symbolic sandbox before Lean replay.",
            "Human-style calculator work should happen off the free-form tactic path.",
            0.92,
            op="extract",
            target="goal",
        )
    if spec.signals["coercion_count"] > 0 or "transport_normalizer" in enabled_names:
        add(
            "normalize_coercions",
            "transport_normalizer",
            "Normalize coercions and transports before any extraction or exact-close attempt.",
            "Transport-heavy goals often become trivially closeable after canonicalization.",
            0.9,
            op="transport",
            target="goal",
        )
    if "eq_sat" in enabled_names:
        add(
            "saturate_equality",
            "eq_sat",
            "Run a bounded equality-saturation pass and extract the cheapest representative.",
            "Equality near-misses often fail because the runtime commits to rewrites too early.",
            0.9,
            op="rewrite",
            target="goal",
        )
    if "arith_nf" in enabled_names:
        add(
            "normalize_arithmetic",
            "arith_nf",
            "Normalize the arithmetic surface before any final close attempt.",
            "Inequalities and symbolic side conditions often reduce to a solver-ready normal form.",
            0.83,
            op="solver",
            target="goal",
        )
    if "membership_exposure" in enabled_names:
        add(
            "expose_membership",
            "membership_exposure",
            "Expose carrier and `mem_*` lemmas before generic rewriting.",
            "Opaque membership walls usually need representation exposure, not more search.",
            0.88,
            op="forward",
            target="hypothesis",
        )
    if "set_pointwise" in enabled_names:
        add(
            "pointwise_set_reduction",
            "set_pointwise",
            "Reduce the set goal to pointwise membership obligations.",
            "Subset and membership goals become manageable after pointwise reduction.",
            0.78,
            op="rewrite",
            target="goal",
        )
    if "witness_constructor" in enabled_names:
        add(
            "construct_witness",
            "witness_constructor",
            "Enumerate canonical witness candidates from local context and constructors.",
            "Near-miss existential goals are often blocked by witness naming, not theorem discovery.",
            0.87,
            op="construct_witness",
            target="goal",
        )
    if "canonical_witness" in enabled_names:
        add(
            "canonical_witness",
            "canonical_witness",
            "Try the smallest domain-canonical witness family before broad synthesis.",
            "Many existential leftovers have a small, structured witness basis.",
            0.82,
            priority_band="P2",
            op="construct_witness",
            target="witness",
        )
    if "binder_instantiation" in enabled_names:
        add(
            "binder_drilldown",
            "binder_instantiation",
            "Introduce or instantiate binders until the local computational core is visible.",
            "Many stalled `∀` goals hide a solvable symbolic nucleus underneath binders.",
            0.81,
            op="instantiate",
            target="binder",
        )
    if "iff_splitter" in enabled_names:
        add(
            "split_iff",
            "iff_splitter",
            "Split the bidirectional logical goal and solve the directions independently.",
            "Iff leftovers are usually two smaller local goals, not one coupled theorem problem.",
            0.81,
            op="split_iff",
            target="goal",
        )
    if "local_fact_selector" in enabled_names:
        add(
            "select_local_fact",
            "local_fact_selector",
            "Search the local context and instances for a direct closing fact.",
            "Atomic goals often close with one correct local fact rather than more planning.",
            0.78,
            op="retrieve",
            target="hypothesis",
        )
    if "context_forward" in enabled_names:
        add(
            "forward_local_context",
            "context_forward",
            "Mine injective/surjective/transport lemmas from local hypotheses before rewriting.",
            "The residual family suggests the missing move is a context bridge, not normalization.",
            0.84,
            op="forward",
            target="hypothesis",
        )
    if "structural_close" in enabled_names:
        add(
            "close_before_unpack",
            "structural_close",
            "Try high-level structural closure before unfolding definitions.",
            "Abstract structural properties tend to degrade under early definitional expansion.",
            0.85,
            op="close",
            target="goal",
        )
    if "categorical_comp_normalizer" in enabled_names:
        add(
            "normalize_categorical_comp",
            "categorical_comp_normalizer",
            "Normalize composition associativity and whiskering structure before lower-level rewriting.",
            "Category-theoretic equalities fail when associative transport is left implicit.",
            0.83,
            op="rewrite",
            target="goal",
        )
    if "hypothesis_injection" in enabled_names:
        add(
            "inject_hypotheses",
            "hypothesis_injection",
            "Move determinant, discriminant, or injectivity hypotheses into the target shape explicitly.",
            "Some equalities are blocked by local transport, not by a missing theorem plan.",
            0.8,
            op="forward",
            target="hypothesis",
        )
    if "extensionality_bridge" in enabled_names:
        add(
            "reduce_pointwise",
            "extensionality_bridge",
            "Reduce functional or map-level equalities to pointwise obligations.",
            "Function-shaped equalities should become local calculator goals first.",
            0.79,
            op="intro",
            target="binder",
        )
    if "eventual_filter_normalizer" in enabled_names:
        add(
            "normalize_eventual_filter",
            "eventual_filter_normalizer",
            "Push eventual/filter structure into a stable local normal form.",
            "Filter goals are locally rigid once their eventual structure is exposed.",
            0.79,
            op="normalize",
            target="goal",
        )
    if "side_condition_sweep" in enabled_names:
        add(
            "sweep_side_conditions",
            "side_condition_sweep",
            "Partition and sweep repeated small side-goals under a shared micro-budget.",
            "Cheap repeated goals should be solved together rather than escalated to replanning.",
            0.82,
            op="sweep",
            target="goal",
        )
    if "metavariable_quarantine" in enabled_names:
        add(
            "quarantine_metavariables",
            "metavariable_quarantine",
            "Quarantine the contaminated branch and reconstruct a clean local state before continuing.",
            "Metavariable corruption is a branch hygiene problem, not a mathematical one.",
            0.86,
            op="quarantine",
            target="goal",
        )

    if not prescriptions:
        add(
            "generic_local_close",
            top,
            "Apply the highest-priority local repair bank for this residual.",
            "The capsule still exposes deterministic structure even when no specialized rule fired.",
            0.6,
            op="close",
            target="goal",
        )

    prescriptions.sort(key=lambda item: (item.priority_band, -item.estimated_gain, item.prescription_kind))
    return prescriptions


def _prior_weight(priors: list[BankPrior], bank: str, default: float = 0.45) -> float:
    for prior in priors:
        if prior.name == bank and not prior.suppressed:
            return float(prior.weight)
    return default


def _make_hole(
    hole_id: str,
    hole_kind: str,
    source_space: str,
    *,
    required: bool = True,
    max_candidates: int = 8,
    **compatibility: Any,
) -> ProofHoleSpec:
    return ProofHoleSpec(
        hole_id=hole_id,
        hole_kind=hole_kind,
        source_space=source_space,
        required=required,
        max_candidates=max_candidates,
        compatibility=dict(compatibility),
    )


def build_proof_skeletons(
    spec: GoalSpecification,
    priors: list[BankPrior],
    specialist_targets: list[str] | None = None,
) -> list[ProofSkeleton]:
    specialist_targets = specialist_targets or infer_specialist_targets(spec)
    enabled_banks = {prior.name for prior in priors if not prior.suppressed}
    goal_text = spec.goal_text
    skeletons: list[ProofSkeleton] = []

    def add(
        skeleton_id: str,
        skeleton_kind: str,
        bank: str,
        specialist: str,
        tactic_templates: list[str],
        rationale: str,
        *,
        holes: list[ProofHoleSpec] | None = None,
        extra: float = 0.0,
        certificate_kinds: list[str] | None = None,
    ) -> None:
        if bank not in enabled_banks and bank not in {"structural_close", "context_forward", "local_fact_selector"}:
            return
        inferred_certificate_kinds = certificate_kinds or {
            "fixed_close": ["terminal_close"],
            "fixed_transform": ["rewrite_chain"],
            "fixed_normalize": ["normal_form"],
            "fixed_solver": ["solver_normal_form"],
            "premise_bridge": ["premise_bridge"],
            "rewrite_bridge": ["rewrite_chain"],
            "local_fact_transport": ["term_transport"],
            "binder_exposure": ["binder_transport"],
            "representation_exposure": ["structural_close"],
            "witness_frame": ["witness_frame"],
            "witness_instantiation": ["witness_construction"],
            "logical_split": ["have_chain"],
            "loop_circuit_break": ["recursive_bridge"],
            "recursive_bridge": ["recursive_bridge"],
            "local_fact_pair": ["witness_construction"],
        }.get(skeleton_kind, ["projected_tactics"])
        skeletons.append(
            ProofSkeleton(
                skeleton_id=skeleton_id,
                skeleton_kind=skeleton_kind,
                bank=bank,
                specialist=specialist,
                tactic_templates=list(tactic_templates),
                holes=list(holes or []),
                rationale=rationale,
                priority=round(_prior_weight(priors, bank) + extra, 3),
                certificate_kinds=list(inferred_certificate_kinds),
            )
        )

    # Always-on local calculator skeletons.
    add("simp_close", "fixed_close", "structural_close", "structural_closer", ["simp"], "Cheap local simplification.")
    add("simpa_close", "fixed_close", "structural_close", "structural_closer", ["simpa"], "Cheap normalization close.")
    add("simp_all_close", "fixed_close", "structural_close", "structural_closer", ["simp_all"], "Exploit all visible local simplifications.")
    add("aesop_close", "fixed_close", "context_forward", "context_transport", ["aesop"], "Mine local context before broader search.")

    if spec.goal_bucket in {"equality", "other"}:
        add("rfl_close", "fixed_close", "eq_sat", "human_calculator", ["rfl"], "Reflexive equalities should close immediately.", extra=0.07)
        add("congr_push", "fixed_transform", "eq_sat", "human_calculator", ["congr"], "Push congruence locally before theorem search.", extra=0.04)
        add("ring_nf_norm", "fixed_normalize", "eq_sat", "human_calculator", ["ring_nf"], "Normalize algebraic equalities to canonical form.", extra=0.08)
        add("ring_nf_norm_num", "fixed_normalize", "eq_sat", "human_calculator", ["ring_nf", "norm_num"], "Algebraic normalization followed by arithmetic cleanup.", extra=0.09)
        add("ring_close", "fixed_normalize", "eq_sat", "human_calculator", ["ring"], "Ring equalities often close directly.", extra=0.05)
        add("norm_cast_align", "fixed_normalize", "transport_normalizer", "symbolic_sandbox", ["norm_cast"], "Align casts before algebraic closure.", extra=0.06)
        add("push_cast_align", "fixed_normalize", "transport_normalizer", "symbolic_sandbox", ["push_cast"], "Push casts to a canonical surface.", extra=0.06)
        add("ext_reduce", "fixed_transform", "structural_close", "structural_closer", ["ext"], "Reduce extensional equality to pointwise obligations.", extra=0.03)
        add("ext_simp_reduce", "fixed_transform", "structural_close", "structural_closer", ["ext", "simp"], "Use extensionality then simplify.", extra=0.04)
        add(
            "exact_local_fact",
            "local_fact_transport",
            "local_fact_selector",
            "atomic_fact_engine",
            ["exact {fact}"],
            "Close directly from a local fact that already matches the target.",
            holes=[_make_hole("fact", "matching_local_fact", "local_context", match_target=True, max_candidates=12)],
            extra=0.12,
        )
        add(
            "simpa_local_fact",
            "local_fact_transport",
            "local_fact_selector",
            "atomic_fact_engine",
            ["simpa using {fact}"],
            "Normalize directly against a matching local fact.",
            holes=[_make_hole("fact", "matching_local_fact", "local_context", match_target=True, max_candidates=12)],
            extra=0.115,
        )
        add(
            "rw_local_fact",
            "rewrite_bridge",
            "eq_sat",
            "symbolic_sandbox",
            ["rw [{fact}]"],
            "Rewrite using a local equality or iff fact from the current proof state.",
            holes=[_make_hole("fact", "local_rewrite_fact", "local_context", max_candidates=10)],
            extra=0.075,
        )
        add(
            "rw_local_fact_back",
            "rewrite_bridge",
            "eq_sat",
            "symbolic_sandbox",
            ["rw [← {fact}]"],
            "Rewrite using the reverse orientation of a local equality fact.",
            holes=[_make_hole("fact", "local_rewrite_fact", "local_context", max_candidates=10)],
            extra=0.07,
        )
        add(
            "simpa_local_rewrite",
            "rewrite_bridge",
            "structural_close",
            "structural_closer",
            ["simpa [{fact}]"],
            "Normalize the target via a local equality/iff fact before closing.",
            holes=[_make_hole("fact", "local_rewrite_fact", "local_context", max_candidates=10)],
            extra=0.072,
        )
    if "pow" in goal_text or "^ 2" in goal_text:
        add("pow_two_rewrite", "rewrite_bridge", "eq_sat", "symbolic_sandbox", ["rw [pow_two]"], "Expose square structure explicitly.", extra=0.06)
    if "Complex.normSq" in goal_text:
        add("normsq_rewrite", "rewrite_bridge", "eq_sat", "symbolic_sandbox", ["rw [Complex.normSq]"], "Expose `Complex.normSq` before symbolic cleanup.", extra=0.08)
        add("normsq_rewrite_ring", "rewrite_bridge", "eq_sat", "symbolic_sandbox", ["rw [Complex.normSq]", "ring_nf"], "Expose norm square then normalize algebraically.", extra=0.09)

    if spec.goal_bucket in {"inequality", "other"}:
        add("norm_num_close", "fixed_solver", "arith_nf", "human_calculator", ["norm_num"], "Cheap arithmetic normalization.", extra=0.1)
        add("linarith_close", "fixed_solver", "arith_nf", "human_calculator", ["linarith"], "Linear arithmetic closure.", extra=0.07)
        add("omega_close", "fixed_solver", "arith_nf", "human_calculator", ["omega"], "Presburger closure for arithmetic side conditions.", extra=0.07)
        add("nlinarith_close", "fixed_solver", "solver_dispatch", "side_condition_sweeper", ["nlinarith"], "Nonlinear arithmetic closure.", extra=0.05)
        add("positivity_close", "fixed_solver", "solver_dispatch", "side_condition_sweeper", ["positivity"], "Positivity reasoning.", extra=0.04)
        add("gcongr_close", "fixed_solver", "solver_dispatch", "side_condition_sweeper", ["gcongr"], "Monotonicity or congruence on inequalities.", extra=0.04)
        add("norm_cast_linarith", "fixed_solver", "transport_normalizer", "symbolic_sandbox", ["norm_cast", "linarith"], "Normalize casts then solve linearly.", extra=0.09)
        add("push_cast_linarith", "fixed_solver", "transport_normalizer", "symbolic_sandbox", ["push_cast", "linarith"], "Push casts before linear closure.", extra=0.09)
        add("norm_num_linarith", "fixed_solver", "arith_nf", "human_calculator", ["norm_num", "linarith"], "Normalize local numerics before linear closure.", extra=0.09)
        add("norm_num_omega", "fixed_solver", "arith_nf", "human_calculator", ["norm_num", "omega"], "Normalize local numerics before Presburger closure.", extra=0.09)
    if spec.goal_bucket == "equality" or "abs" in goal_text or "|" in goal_text:
        add(
            "abs_lt_one_mp",
            "local_fact_transport",
            "local_fact_selector",
            "atomic_fact_engine",
            ["exact Int.abs_lt_one_iff.mp {fact}"],
            "Use a local absolute-value bound as a semantic bridge to the target equality.",
            holes=[_make_hole("fact", "abs_lt_one_fact", "local_context", max_candidates=8)],
            extra=0.12,
        )
    if "abs" in goal_text or "|" in goal_text:
        add("abs_lt_one_rewrite", "rewrite_bridge", "arith_nf", "human_calculator", ["rw [Int.abs_lt_one_iff]"], "Translate integer absolute-value bounds to equalities.", extra=0.1)
        add("abs_lt_one_rewrite_back", "rewrite_bridge", "arith_nf", "human_calculator", ["rw [← Int.abs_lt_one_iff]"], "Try the reverse absolute-value rewrite when already normalized.", extra=0.08)
        add("abs_lt_one_norm_num", "rewrite_bridge", "arith_nf", "human_calculator", ["rw [Int.abs_lt_one_iff]", "norm_num"], "Translate an absolute-value side condition then simplify.", extra=0.11)
        add("abs_lt_one_omega", "rewrite_bridge", "arith_nf", "human_calculator", ["rw [Int.abs_lt_one_iff]", "omega"], "Translate an absolute-value side condition then solve with Presburger arithmetic.", extra=0.11)
        add("simp_all_abs_lt_one", "fixed_close", "structural_close", "structural_closer", ["simp_all [Int.abs_lt_one_iff]"], "Normalize local absolute-value facts through the full context.", extra=0.08)

    if spec.goal_bucket == "forall":
        add("intro_binder", "binder_exposure", "binder_instantiation", "binder_drilldown", ["intro"], "Expose the symbolic core under one binder.", extra=0.1)
        add("intros_binders", "binder_exposure", "binder_instantiation", "binder_drilldown", ["intros"], "Expose the symbolic core under multiple binders.", extra=0.1)
        add("intros_aesop", "binder_exposure", "binder_instantiation", "binder_drilldown", ["intros", "aesop"], "Introduce locals then mine the local context.", extra=0.12)
        add("intros_simp", "binder_exposure", "binder_instantiation", "binder_drilldown", ["intros", "simp"], "Introduce locals and simplify.", extra=0.11)
        add("intros_simpa", "binder_exposure", "binder_instantiation", "binder_drilldown", ["intros", "simpa"], "Introduce locals and normalize to a closure shape.", extra=0.11)
        if spec.signals["has_symbolic_sandbox_surface"]:
            add("intros_norm_cast", "binder_exposure", "transport_normalizer", "symbolic_sandbox", ["intros", "norm_cast"], "Expose binders then normalize casts on the visible symbolic surface.", extra=0.1)
            add("intros_push_cast", "binder_exposure", "transport_normalizer", "symbolic_sandbox", ["intros", "push_cast"], "Expose binders then push casts into a stable local form.", extra=0.095)
            add("intros_ring_nf", "binder_exposure", "eq_sat", "symbolic_sandbox", ["intros", "ring_nf"], "Expose binders before algebraic normalization.", extra=0.09)
        if "Complex.normSq" in goal_text:
            add("intros_normsq", "binder_exposure", "eq_sat", "symbolic_sandbox", ["intros", "rw [Complex.normSq]"], "Expose binders before symbolic norm-square reduction.", extra=0.1)

    if spec.goal_bucket == "iff":
        add("constructor_split", "logical_split", "iff_splitter", "logic_splitter", ["constructor"], "Split the directions explicitly.", extra=0.11)
        add("constructor_all_goals_aesop", "logical_split", "iff_splitter", "logic_splitter", ["constructor", "all_goals aesop"], "Split and discharge each direction locally.", extra=0.13)
        add("constructor_all_goals_simp", "logical_split", "iff_splitter", "logic_splitter", ["constructor", "all_goals simp"], "Split and simplify both directions.", extra=0.12)
        add("tauto_close", "fixed_close", "iff_splitter", "logic_splitter", ["tauto"], "Pure-logic iff closure.", extra=0.08)

    if spec.goal_bucket in {"membership", "subset"}:
        add("membership_simp", "representation_exposure", "membership_exposure", "membership_surface_engine", ["simp"], "Expose carrier and membership representation.", extra=0.09)
        add("membership_simpa", "representation_exposure", "membership_exposure", "membership_surface_engine", ["simpa"], "Canonical membership normalization.", extra=0.09)
        add("membership_aesop", "representation_exposure", "membership_exposure", "membership_surface_engine", ["aesop"], "Use local membership facts.", extra=0.08)
        add("intro_simp_membership", "representation_exposure", "set_pointwise", "membership_surface_engine", ["intro", "simp"], "Reduce set membership under an introduced witness.", extra=0.08)

    if spec.goal_bucket == "exists":
        add("exists_constructor", "witness_frame", "witness_constructor", "witness_engine", ["constructor"], "Introduce explicit witness structure.", extra=0.08)
        add("exists_aesop", "witness_frame", "witness_constructor", "witness_engine", ["aesop"], "Let local witness facts close the existential tail.", extra=0.08)
        add(
            "use_canonical_witness",
            "witness_instantiation",
            "witness_constructor",
            "witness_engine",
            ["use {witness}"],
            "Try a canonical witness programmatically.",
            holes=[_make_hole("witness", "witness_term", "goal_surface", max_candidates=10)],
            extra=0.1,
        )
        add(
            "refine_canonical_witness",
            "witness_instantiation",
            "canonical_witness",
            "witness_engine",
            ["refine ⟨{witness}, ?_⟩"],
            "Try a canonical witness with one residual obligation.",
            holes=[_make_hole("witness", "witness_term", "goal_surface", max_candidates=10)],
            extra=0.08,
        )

    if spec.goal_bucket == "other" and " ∧ " in goal_text:
        add(
            "exact_local_pair",
            "local_fact_pair",
            "witness_constructor",
            "witness_engine",
            ["exact ⟨{left}, {right}⟩"],
            "Construct a conjunction directly from locally available proof facts.",
            holes=[
                _make_hole("left", "left_conj_fact", "local_context", max_candidates=8),
                _make_hole("right", "right_conj_fact", "local_context", max_candidates=8),
            ],
            extra=0.11,
        )

    if spec.goal_bucket == "atomic_prop":
        add("assumption_close", "fixed_close", "local_fact_selector", "atomic_fact_engine", ["assumption"], "Close from a local assumption.", extra=0.12)
        add("trivial_close", "fixed_close", "local_fact_selector", "atomic_fact_engine", ["trivial"], "Trivial proposition closure.", extra=0.08)
        add("decide_close", "fixed_close", "local_fact_selector", "atomic_fact_engine", ["decide"], "Decision-procedure close.", extra=0.07)
        add("infer_instance_close", "fixed_close", "local_fact_selector", "atomic_fact_engine", ["infer_instance"], "Close via typeclass search.", extra=0.07)

    # Premise- and context-driven skeletons.
    add(
        "exact_accessible_lemma",
        "premise_bridge",
        "context_forward",
        "context_transport",
        ["exact {lemma}"],
        "Use an accessible lemma as a direct closing bridge.",
        holes=[_make_hole("lemma", "accessible_exact_lemma", "accessible_premises", max_candidates=8)],
        extra=0.08,
    )
    add(
        "apply_accessible_lemma",
        "premise_bridge",
        "context_forward",
        "context_transport",
        ["apply {lemma}"],
        "Forward the local proof through an accessible lemma.",
        holes=[_make_hole("lemma", "accessible_apply_lemma", "accessible_premises", max_candidates=8)],
        extra=0.07,
    )
    add(
        "simpa_accessible_lemma",
        "premise_bridge",
        "context_forward",
        "context_transport",
        ["simpa using {lemma}"],
        "Normalize directly against an accessible lemma.",
        holes=[_make_hole("lemma", "accessible_exact_lemma", "accessible_premises", max_candidates=8)],
        extra=0.075,
    )
    add(
        "rw_accessible_fwd",
        "rewrite_bridge",
        "eq_sat",
        "symbolic_sandbox",
        ["rw [{lemma}]"],
        "Rewrite via an accessible lemma selected from the local theorem frontier.",
        holes=[_make_hole("lemma", "rewrite_lemma", "accessible_premises", max_candidates=8)],
        extra=0.055,
    )
    add(
        "rw_accessible_back",
        "rewrite_bridge",
        "eq_sat",
        "symbolic_sandbox",
        ["rw [← {lemma}]"],
        "Try the reverse orientation of an accessible rewrite lemma.",
        holes=[_make_hole("lemma", "rewrite_lemma", "accessible_premises", max_candidates=8)],
        extra=0.05,
    )

    if "recursive_circuit_breaker" in specialist_targets:
        add("bounded_aesop", "loop_circuit_break", "loop_breaker", "recursive_circuit_breaker", ["aesop"], "Break controller-level loops with a bounded local structural close.", extra=0.06)
        if "rootD" in goal_text or spec.theorem_id.startswith("Batteries.UnionFind"):
            add("unionfind_rootd_rw_back", "recursive_bridge", "recursive_unfold_one", "recursive_circuit_breaker", ["rw [← Batteries.UnionFind.rootD]"], "Fold the recursive surface back into the domain wrapper before continuing.", extra=0.11)
            add("unionfind_rootd_rw", "recursive_bridge", "recursive_unfold_one", "recursive_circuit_breaker", ["rw [Batteries.UnionFind.rootD]"], "Permit one domain-specific recursive exposure.", extra=0.08)
            add("unionfind_rootd_rw_simpa", "recursive_bridge", "recursive_unfold_one", "recursive_circuit_breaker", ["rw [Batteries.UnionFind.rootD]", "simpa"], "Expose one recursive layer then attempt immediate closure.", extra=0.1)
            add("unionfind_root_rw", "recursive_bridge", "recursive_unfold_one", "recursive_circuit_breaker", ["rw [Batteries.UnionFind.root]"], "Expose one structural root step without entering blind recursive expansion.", extra=0.09)

    skeletons.sort(key=lambda item: (-item.priority, len(item.tactic_templates), item.skeleton_id))
    return skeletons


def capsule_priority_score(
    spec: GoalSpecification,
    priors: list[BankPrior],
    specialist_targets: list[str] | None = None,
    suppression_hints: list[str] | None = None,
) -> float:
    specialist_targets = specialist_targets or infer_specialist_targets(spec)
    suppression_hints = suppression_hints or infer_suppression_hints(spec)
    top_weight = max((prior.weight for prior in priors if not prior.suppressed), default=0.0)
    score = 0.0
    score += spec.goals_closed * 3.0
    score -= max(spec.goals_remaining - 1, 0) * 2.5
    score += 8.0 if spec.residual_bucket == "single_goal_near_miss" else 0.0
    score += 4.0 if spec.goal_bucket in {"equality", "inequality", "membership", "exists"} else 0.0
    score += 4.5 if "human_calculator" in specialist_targets else 0.0
    score += 5.0 if "symbolic_sandbox" in specialist_targets else 0.0
    score += 6.0 if "recursive_circuit_breaker" in specialist_targets else 0.0
    score += top_weight * 10.0
    score += min(len(specialist_targets), 4) * 1.5
    score += min(len(suppression_hints), 4) * 0.75
    score -= min(spec.attempts, 256) / 32.0
    return round(score, 3)


def build_goal_capsule(row: dict[str, Any]) -> GoalCapsule:
    spec = build_goal_specification(row)
    specialist_targets = infer_specialist_targets(spec)
    suppression_hints = infer_suppression_hints(spec)
    priors = infer_bank_priors(spec, specialist_targets, suppression_hints)
    prescriptions = build_goal_prescriptions(spec, priors, specialist_targets, suppression_hints)
    proof_skeletons = build_proof_skeletons(spec, priors, specialist_targets)
    ledger_seed = build_ledger_seed(row, spec)
    allowed_engines = _dedupe(spec.engine_eligibility + allowed_engines_from_priors(priors))
    projector_policy = projector_policy_for_spec(spec)
    execution_budgets = execution_budgets_for_spec(spec)
    solver_constraints = build_solver_constraints(spec, proof_skeletons, ledger_seed)
    eqsat_plan = build_eqsat_plan(spec, allowed_engines, execution_budgets)
    relational_search_specs = build_relational_search_specs(spec, ledger_seed)
    backend_preferences = backend_preferences_for_spec(spec, allowed_engines)
    proof_dsl_programs = build_proof_dsl_programs(proof_skeletons, solver_constraints)
    score = capsule_priority_score(spec, priors, specialist_targets, suppression_hints)
    return GoalCapsule(
        specification=spec,
        specialist_targets=specialist_targets,
        suppression_hints=suppression_hints,
        bank_priors=priors,
        prescriptions=prescriptions,
        proof_skeletons=proof_skeletons,
        ledger_seed=ledger_seed,
        allowed_engines=allowed_engines,
        projector_policy=projector_policy,
        execution_budgets=execution_budgets,
        proof_dsl_programs=proof_dsl_programs,
        solver_constraints=solver_constraints,
        eqsat_plan=eqsat_plan,
        relational_search_specs=relational_search_specs,
        backend_preferences=backend_preferences,
        priority_score=score,
    )


def summarize_capsules(capsules: list[GoalCapsule]) -> dict[str, Any]:
    by_goal_bucket: dict[str, int] = {}
    by_bank: dict[str, int] = {}
    by_prescription: dict[str, int] = {}
    by_skeleton_kind: dict[str, int] = {}
    by_target: dict[str, int] = {}
    by_suppression: dict[str, int] = {}
    by_engine: dict[str, int] = {}
    by_projector_marker: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    by_constraint_kind: dict[str, int] = {}
    for capsule in capsules:
        bucket = capsule.specification.goal_bucket
        by_goal_bucket[bucket] = by_goal_bucket.get(bucket, 0) + 1
        for target in capsule.specialist_targets:
            by_target[target] = by_target.get(target, 0) + 1
        for hint in capsule.suppression_hints:
            by_suppression[hint] = by_suppression.get(hint, 0) + 1
        for prior in capsule.bank_priors:
            if prior.suppressed:
                continue
            by_bank[prior.name] = by_bank.get(prior.name, 0) + 1
        for engine in capsule.allowed_engines:
            by_engine[engine] = by_engine.get(engine, 0) + 1
        for backend in capsule.backend_preferences:
            by_backend[backend] = by_backend.get(backend, 0) + 1
        for marker in capsule.specification.projector_markers:
            by_projector_marker[marker] = by_projector_marker.get(marker, 0) + 1
        for constraint in capsule.solver_constraints:
            by_constraint_kind[constraint.constraint_kind] = by_constraint_kind.get(constraint.constraint_kind, 0) + 1
        for prescription in capsule.prescriptions:
            by_prescription[prescription.prescription_kind] = by_prescription.get(prescription.prescription_kind, 0) + 1
        for skeleton in capsule.proof_skeletons:
            by_skeleton_kind[skeleton.skeleton_kind] = by_skeleton_kind.get(skeleton.skeleton_kind, 0) + 1
    return {
        "total_capsules": len(capsules),
        "by_goal_bucket": dict(sorted(by_goal_bucket.items(), key=lambda item: (-item[1], item[0]))),
        "by_specialist_target": dict(sorted(by_target.items(), key=lambda item: (-item[1], item[0]))),
        "by_suppression_hint": dict(sorted(by_suppression.items(), key=lambda item: (-item[1], item[0]))),
        "by_enabled_bank": dict(sorted(by_bank.items(), key=lambda item: (-item[1], item[0]))),
        "by_allowed_engine": dict(sorted(by_engine.items(), key=lambda item: (-item[1], item[0]))),
        "by_backend_preference": dict(sorted(by_backend.items(), key=lambda item: (-item[1], item[0]))),
        "by_projector_marker": dict(sorted(by_projector_marker.items(), key=lambda item: (-item[1], item[0]))),
        "by_constraint_kind": dict(sorted(by_constraint_kind.items(), key=lambda item: (-item[1], item[0]))),
        "by_prescription_kind": dict(sorted(by_prescription.items(), key=lambda item: (-item[1], item[0]))),
        "by_skeleton_kind": dict(sorted(by_skeleton_kind.items(), key=lambda item: (-item[1], item[0]))),
        "top_capsules": [capsule.to_dict() for capsule in sorted(capsules, key=lambda item: (-item.priority_score, item.specification.theorem_id))[:10]],
    }
