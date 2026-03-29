"""EXP-049: Theorem-search integration with ExecSelector v1.

Tests whether the selector-gated apply lane improves theorem-level proof success
over the bootstrap-only baseline.

Conditions (50-theorem Mathlib set, same as EXP-3.2):
  baseline      — interleaved_bootstrap only (no cosine_apply)
  cosine_apply  — bootstrap + cosine_apply_gated (raw cosine, beam=5)
  selector_apply — bootstrap + cosine_apply_gated + ExecSelector v1 reranking (beam=1)

Primary metric: theorems proved (raw_success)
Secondary: apply-lane contributions (goals closed via cosine_apply / exec_selector_apply)

Claim being tested:
  selector improves valid apply-step selection (established: EXP-048)
  → does that translate to more theorem-level proofs?

Usage:
    python -m scripts.run_exp049_selector_search \\
        --config configs/wayfinder.yaml \\
        --checkpoint models/nav004/best.pt \\
        --selector models/apply_exec_selector_v1.pt \\
        --output runs/exp049_results/ \\
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scripts.run_benchmark import (  # type: ignore[attr-defined]
    _resolve_initial_goal,
    load_benchmark_theorems,
)
from src.lean_interface import LeanConfig, LeanKernel
from src.nav_model_factory import load_navigational_checkpoint
from src.proof_search import Pipeline, SearchConfig, search

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------------
# Build pipeline — uses load_navigational_checkpoint (same as run_benchmark)
# ---------------------------------------------------------------------------


def _build_pipeline(config: dict, checkpoint_path: str, device: str) -> Pipeline:
    _, modules = load_navigational_checkpoint(Path(checkpoint_path), config, device)
    return Pipeline(
        encoder=modules["encoder"],  # type: ignore[arg-type]
        analyzer=modules["analyzer"],  # type: ignore[arg-type]
        bridge=modules["bridge"],  # type: ignore[arg-type]
        navigator=modules["navigator"],  # type: ignore[arg-type]
    )


def _build_lean(config: dict) -> tuple[LeanKernel, LeanConfig]:
    lean_section = config.get("lean", {})
    search_cfg = config.get("search", {})
    lean_cfg = LeanConfig(
        backend=lean_section.get("backend", "stub"),
        hammer_timeout=search_cfg.get("hammer_timeout", 60),
        project_root=lean_section.get("project_root", ""),
        imports=lean_section.get("imports", ["Init"]),
    )
    return LeanKernel(lean_cfg), lean_cfg


# load_benchmark_theorems / _resolve_initial_goal imported from scripts.run_benchmark


# ---------------------------------------------------------------------------
# Per-condition search
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    name: str
    proved: int = 0
    total: int = 0
    started: int = 0
    skipped_start: int = 0
    apply_contributions: int = 0  # goals closed via apply lane
    total_attempts: int = 0
    elapsed_s: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)
    # Apply trigger/lane aggregate metrics
    trigger_fires: int = 0
    trigger_rejects: int = 0
    apply_attempts: int = 0
    apply_accepts: int = 0
    apply_goal_closes: int = 0
    theorems_with_apply_progress: int = 0

    @property
    def success_rate(self) -> float:
        return self.proved / max(self.total, 1)

    @property
    def started_success_rate(self) -> float:
        return self.proved / max(self.started, 1)


def run_condition(
    condition_name: str,
    theorems: list[dict[str, Any]],
    pipeline: Pipeline,
    lean: LeanKernel,
    conn: sqlite3.Connection,
    base_cfg: SearchConfig,
    override: dict[str, Any],
    name_to_id: dict[str, int],
    sentence_encoder: Any | None = None,
    exec_apply_selector: Any | None = None,
    exec_apply_encoder: Any | None = None,
    output_dir: Path | None = None,
) -> ConditionResult:
    # search_mode must be explicit per condition so that the learned lane and
    # cosine_rw lane (which fires whenever sentence_encoder is not None) are
    # not accidentally inherited by the baseline.
    cfg = SearchConfig(
        budget=base_cfg.budget,
        hammer_delegation=base_cfg.hammer_delegation,
        accessible_premises=base_cfg.accessible_premises,
        max_candidates_per_step=base_cfg.max_candidates_per_step,
        device=base_cfg.device,
        search_mode=override.get("search_mode", base_cfg.search_mode),
        temporal_mode=override.get("temporal_mode", base_cfg.temporal_mode),
        strategy_memory_path=override.get("strategy_memory_path", ""),
        cosine_rw_beam=base_cfg.cosine_rw_beam,
        interleaved_bootstrap_enabled=base_cfg.interleaved_bootstrap_enabled,
        cosine_apply_enabled=override.get("cosine_apply_enabled", False),
        cosine_apply_gated=override.get("cosine_apply_gated", False),
        cosine_apply_beam=override.get("cosine_apply_beam", 5),
        exec_apply_selector_path=override.get("exec_apply_selector_path", ""),
        exec_apply_selector_pool=override.get("exec_apply_selector_pool", 20),
        apply_trigger_path=override.get("apply_trigger_path", ""),
        apply_trigger_threshold=override.get("apply_trigger_threshold", 0.47),
        collect_trace=True,
    )
    # Suppress sentence encoder for conditions that must not activate cosine_rw.
    # cosine_rw fires automatically whenever env.sentence_encoder is not None,
    # so conditions without an explicit encoder override must receive None.
    if not override.get("use_sentence_encoder", False):
        sentence_encoder = None

    # Pre-initialize Pantograph server before the loop (same as run_benchmark)
    if lean._backend == "pantograph":
        lean._ensure_server()

    result = ConditionResult(name=condition_name, total=len(theorems))
    t0 = time.time()

    for i, thm in enumerate(theorems):
        theorem_id = thm.get("theorem_id", thm.get("full_name", ""))
        logger.info("[%s] [%d/%d] %s", condition_name, i + 1, len(theorems), theorem_id)

        accessible_id = name_to_id.get(theorem_id)

        initial_goal = _resolve_initial_goal(thm, lean)
        if initial_goal is None:
            logger.warning("Could not create goal for %s — skipping", theorem_id)
            result.skipped_start += 1
            result.results.append(
                {
                    "theorem_id": theorem_id,
                    "success": False,
                    "started": False,
                    "attempts": 0,
                    "tactics_used": [],
                    "close_provenance": [],
                }
            )
            continue

        result.started += 1

        try:
            import signal

            def _timeout_handler(signum: int, frame: object) -> None:  # noqa: ARG001
                raise TimeoutError("theorem search exceeded per-theorem timeout")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(300)  # 5 minute per-theorem timeout
            try:
                sr = search(
                    theorem_id=theorem_id,
                    initial_goal=initial_goal,
                    pipeline=pipeline,
                    conn=conn,
                    lean=lean,
                    config=cfg,
                    accessible_theorem_id=accessible_id,
                    sentence_encoder=sentence_encoder,
                    exec_apply_selector=exec_apply_selector,
                    exec_apply_encoder=exec_apply_encoder,
                )
                proved = sr.success
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (Exception, TimeoutError) as exc:
            logger.warning("Error on %s: %s", theorem_id, exc)
            proved = False
            sr = None  # type: ignore[assignment]

        if proved:
            result.proved += 1
        if sr is not None:
            result.total_attempts += getattr(sr, "attempts", 0)
            apply_contribs = sum(
                1
                for lane in getattr(sr, "close_provenance", [])
                if lane in ("cosine_apply", "exec_selector_apply")
            )
            result.apply_contributions += apply_contribs
            result.trigger_fires += getattr(sr, "trigger_fire_count", 0)
            result.trigger_rejects += getattr(sr, "trigger_reject_count", 0)
            result.apply_attempts += getattr(sr, "apply_attempt_count", 0)
            result.apply_accepts += getattr(sr, "apply_accept_count", 0)
            result.apply_goal_closes += getattr(sr, "apply_goal_close_count", 0)
            if getattr(sr, "apply_accept_count", 0) > 0:
                result.theorems_with_apply_progress += 1

        result.results.append(
            {
                "theorem_id": theorem_id,
                "success": proved,
                "started": True,
                "attempts": getattr(sr, "attempts", 0) if sr else 0,
                "tactics_used": getattr(sr, "tactics_used", []) if sr else [],
                "close_provenance": list(getattr(sr, "close_provenance", [])) if sr else [],
                "step_trace": list(getattr(sr, "step_trace", [])) if sr else [],
                "trigger_fire_count": getattr(sr, "trigger_fire_count", 0) if sr else 0,
                "trigger_reject_count": getattr(sr, "trigger_reject_count", 0) if sr else 0,
                "apply_attempt_count": getattr(sr, "apply_attempt_count", 0) if sr else 0,
                "apply_accept_count": getattr(sr, "apply_accept_count", 0) if sr else 0,
                "apply_goal_close_count": getattr(sr, "apply_goal_close_count", 0) if sr else 0,
            }
        )

        # Periodic flush every 50 theorems
        if (i + 1) % 50 == 0 and output_dir is not None:
            partial_path = output_dir / f"{condition_name}_partial.jsonl"
            with open(partial_path, "w") as pf:
                for r in result.results:
                    pf.write(json.dumps(r) + "\n")
            logger.info(
                "[%s] %d/%d flush: %d proved / %d started (%d skipped)",
                condition_name, i + 1, len(theorems),
                result.proved, result.started, result.skipped_start,
            )

    result.elapsed_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(conditions: list[ConditionResult]) -> None:
    print()
    print("=" * 78)
    print("Theorem-search integration")
    print("=" * 78)
    print(f"  {'Condition':<25} {'Proved':<8} {'Started':<10} {'Skip':<6} "
          f"{'Rate/total':<12} {'Rate/start':<12} {'Time'}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 8}")
    baseline_proved = conditions[0].proved if conditions else 0
    for c in conditions:
        delta = f" (Δ={c.proved - baseline_proved:+d})" if c.name != conditions[0].name else ""
        print(
            f"  {c.name:<25} {c.proved}/{c.total:<6} "
            f"{c.started}/{c.total:<8} {c.skipped_start:<6} "
            f"{100 * c.success_rate:.1f}%{'':<8} "
            f"{100 * c.started_success_rate:.1f}%{'':<8} "
            f"{c.elapsed_s:.0f}s{delta}"
        )

    # Apply lane detail (only for conditions with trigger activity)
    has_trigger = any(c.trigger_fires > 0 or c.trigger_rejects > 0 for c in conditions)
    if has_trigger:
        print()
        print("  Apply lane detail:")
        print(
            f"  {'Condition':<25} {'Trig fire':<11} {'Trig rej':<11} "
            f"{'Attempts':<11} {'Accepts':<11} {'GoalClose':<11} {'Thm w/prog'}"
        )
        print(f"  {'-' * 25} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11}")
        for c in conditions:
            if c.trigger_fires > 0 or c.apply_attempts > 0:
                print(
                    f"  {c.name:<25} {c.trigger_fires:<11} {c.trigger_rejects:<11} "
                    f"{c.apply_attempts:<11} {c.apply_accepts:<11} "
                    f"{c.apply_goal_closes:<11} {c.theorems_with_apply_progress}"
                )

    # Materially changed theorems (proved in one condition but not the other)
    if len(conditions) >= 2:
        base_proved = set(
            r["theorem_id"] for r in conditions[0].results if r["success"]
        )
        for c in conditions[1:]:
            c_proved = set(r["theorem_id"] for r in c.results if r["success"])
            gained = c_proved - base_proved
            lost = base_proved - c_proved
            if gained or lost:
                print(f"\n  {c.name} vs {conditions[0].name}:")
                if gained:
                    print(f"    Gained: {gained}")
                if lost:
                    print(f"    Lost:   {lost}")
                else:
                    print("    Lost:   (none)")

    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--selector", default="models/apply_exec_selector_v1.pt")
    parser.add_argument("--trigger", default="models/apply_trigger_v2.pt")
    parser.add_argument("--output", default="runs/exp049_results/")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["cosine_rw_only", "cosine_rw_trigger_apply"],
        help="Which conditions to run",
    )
    parser.add_argument(
        "--theorems", default="", help="Path to benchmark JSONL (overrides config evaluation paths)"
    )
    parser.add_argument(
        "--lean-project",
        default="",
        help="Path to Lean project root (overrides config lean.project_root)",
    )
    parser.add_argument(
        "--backend",
        default="",
        help="Lean backend: stub or pantograph (overrides config lean.backend)",
    )
    parser.add_argument(
        "--lean-imports",
        nargs="+",
        default=[],
        help="Lean imports list (overrides config lean.imports), e.g. Mathlib",
    )
    parser.add_argument(
        "--db", default="", help="Path to proof network DB (overrides config data.proof_network_db)"
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.theorems:
        config.setdefault("evaluation", {})["benchmark_theorems"] = args.theorems
        config["evaluation"].pop("mathlib_test_split", None)
    if args.lean_project:
        config.setdefault("lean", {})["project_root"] = args.lean_project
    if args.backend:
        config.setdefault("lean", {})["backend"] = args.backend
    if args.lean_imports:
        config.setdefault("lean", {})["imports"] = args.lean_imports
    if args.db:
        config.setdefault("data", {})["proof_network_db"] = args.db

    pipeline = _build_pipeline(config, args.checkpoint, args.device)
    lean, _ = _build_lean(config)
    conn = sqlite3.connect(config["data"]["proof_network_db"])

    name_to_id: dict[str, int] = {
        n: eid for eid, n in conn.execute("SELECT id, name FROM entities")
    }

    theorems = load_benchmark_theorems(config, args.limit or None)
    logger.info("Loaded %d theorems", len(theorems))

    search_cfg = config.get("search", {})
    base_cfg = SearchConfig(
        budget=args.budget,
        hammer_delegation=search_cfg.get("hammer_delegation", True),
        accessible_premises=True,
        max_candidates_per_step=search_cfg.get("max_candidates_per_step", 8),
        device=args.device,
        search_mode=search_cfg.get("search_mode", "full"),
        temporal_mode="off",
        cosine_rw_beam=search_cfg.get("cosine_rw_beam", 5),
        interleaved_bootstrap_enabled=True,
    )

    # Condition definitions.
    #
    # search_mode is explicit per condition so lane policy is unambiguous:
    #   baseline       — interleaved_bootstrap only; no cosine_rw, no learned
    #   cosine_apply   — same baseline + cosine_apply_gated (cosine_rw also
    #                    active because use_sentence_encoder=True, but that is
    #                    the intended baseline for the apply lane comparison)
    #   selector_apply — same as cosine_apply with ExecSelector v1 reranking
    #
    # use_sentence_encoder controls whether the shared encoder is forwarded to
    # search(); when False the encoder is suppressed so cosine_rw cannot fire.
    _condition_overrides: dict[str, dict[str, Any]] = {
        "baseline": {
            "search_mode": "no_learned",
            "use_sentence_encoder": False,
        },
        "cosine_apply": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "cosine_apply_enabled": True,
            "cosine_apply_gated": True,
            "cosine_apply_beam": 5,
        },
        "selector_apply": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "cosine_apply_enabled": True,
            "cosine_apply_gated": True,
            "cosine_apply_beam": 1,
            "exec_apply_selector_path": args.selector,
            "exec_apply_selector_pool": 20,
        },
        "trigger_apply": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "cosine_apply_enabled": True,
            "cosine_apply_gated": True,
            "cosine_apply_beam": 1,
            "exec_apply_selector_path": args.selector,
            "exec_apply_selector_pool": 20,
            "apply_trigger_path": args.trigger,
            "apply_trigger_threshold": 0.47,
        },
        # Clean A/B: same stack, only difference is trigger+apply
        "cosine_rw_only": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "cosine_apply_enabled": False,
        },
        "cosine_rw_trigger_apply": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "cosine_apply_enabled": True,
            "cosine_apply_gated": True,
            "cosine_apply_beam": 3,
            "exec_apply_selector_path": args.selector,
            "exec_apply_selector_pool": 20,
            "apply_trigger_path": args.trigger,
            "apply_trigger_threshold": 0.65,
        },
        # EXP-SOM-002: Arbiter conditions
        "arbiter_full": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "temporal_mode": "arbiter_full",
            "strategy_memory_path": "data/strategy_memory_som.json",
        },
        "arbiter_goal_only": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "temporal_mode": "arbiter_goal_only",
            "strategy_memory_path": "data/strategy_memory_som.json",
        },
        "arbiter_lane_only": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "temporal_mode": "arbiter_lane_only",
            "strategy_memory_path": "data/strategy_memory_som.json",
        },
        "arbiter_no_memory": {
            "search_mode": "no_learned",
            "use_sentence_encoder": True,
            "temporal_mode": "arbiter_full",
            "strategy_memory_path": "data/strategy_memory_empty.json",
        },
    }

    # Load sentence encoder once (needed for cosine_apply lane)
    sentence_encoder: Any | None = None
    needs_encoder = any(
        _condition_overrides.get(c, {}).get("cosine_apply_enabled", False) for c in args.conditions
    )
    if needs_encoder:
        try:
            from sentence_transformers import SentenceTransformer

            sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence encoder for cosine_apply lane")
        except Exception as e:
            logger.warning("Could not load sentence encoder: %s", e)

    # Pre-load ExecSelector once — avoids per-theorem model load cost in selector_apply.
    exec_apply_selector: Any | None = None
    exec_apply_encoder: Any | None = None
    needs_selector = any(
        _condition_overrides.get(c, {}).get("exec_apply_selector_path", "") for c in args.conditions
    )
    if needs_selector and Path(args.selector).exists():
        try:
            import torch
            import torch.nn as nn
            from sentence_transformers import SentenceTransformer as _ST

            class _ExecSelector(nn.Module):
                def __init__(self, in_dim: int, hidden: int) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden, hidden // 2),
                        nn.ReLU(),
                        nn.Linear(hidden // 2, 1),
                    )

                def score(self, x: "torch.Tensor") -> "torch.Tensor":
                    return torch.sigmoid(self.net(x).squeeze(-1))

            ckpt = torch.load(args.selector, map_location="cpu", weights_only=False)
            _emb_dim = ckpt.get("emb_dim", 384)
            _compat_dim = ckpt.get("compat_dim", 0)
            _in_dim = _emb_dim * 2 + 2 + _compat_dim
            sel = _ExecSelector(_in_dim, ckpt.get("hidden", 256))
            sel.load_state_dict(ckpt["model_state_dict"])
            sel.eval()
            exec_apply_selector = sel
            exec_apply_encoder = _ST(ckpt.get("encoder", "all-MiniLM-L6-v2"))
            logger.info("Loaded ExecSelector: %s", args.selector)
        except Exception as e:
            logger.warning("Could not load ExecSelector: %s", e)

    all_results: list[ConditionResult] = []
    for cond_name in args.conditions:
        override = _condition_overrides.get(cond_name, {})
        logger.info("--- Running condition: %s ---", cond_name)
        # Pass pre-loaded selector only to conditions that request it
        cond_selector = exec_apply_selector if override.get("exec_apply_selector_path") else None
        cond_sel_enc = exec_apply_encoder if override.get("exec_apply_selector_path") else None
        cr = run_condition(
            cond_name,
            theorems,
            pipeline,
            lean,
            conn,
            base_cfg,
            override,
            name_to_id,
            sentence_encoder=sentence_encoder,
            exec_apply_selector=cond_selector,
            exec_apply_encoder=cond_sel_enc,
            output_dir=Path(args.output),
        )
        all_results.append(cr)

        # Save per-condition JSONL
        out_path = Path(args.output) / f"{cond_name}.jsonl"
        with open(out_path, "w") as f:
            for r in cr.results:
                f.write(json.dumps(r) + "\n")
        logger.info(
            "%s: %d/%d proved (%.1f%%)",
            cond_name,
            cr.proved,
            cr.total,
            100 * cr.success_rate,
        )

    print_report(all_results)

    # Save summary JSON
    summary = {
        c.name: {
            "proved": c.proved,
            "total": c.total,
            "started": c.started,
            "skipped_start": c.skipped_start,
            "success_rate": round(c.success_rate, 4),
            "started_success_rate": round(c.started_success_rate, 4),
            "elapsed_s": round(c.elapsed_s, 1),
            "trigger_fires": c.trigger_fires,
            "trigger_rejects": c.trigger_rejects,
            "apply_attempts": c.apply_attempts,
            "apply_accepts": c.apply_accepts,
            "apply_goal_closes": c.apply_goal_closes,
            "theorems_with_apply_progress": c.theorems_with_apply_progress,
        }
        for c in all_results
    }
    with open(Path(args.output) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s/summary.json", args.output)


if __name__ == "__main__":
    main()
