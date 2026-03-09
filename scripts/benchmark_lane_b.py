"""Lane B (Axle) post-processing for benchmark results.

Extracted from run_benchmark.py to keep that file under the 400-line limit.
Handles Axle-based repair of failed theorems with partial progress.
"""

from __future__ import annotations

import asyncio


def _build_sorry_proof(entry: dict) -> str:
    """Build a minimal Lean proof attempt with sorries for open goals.

    This constructs a proof stub from the benchmark entry's tactics.
    The actual Lean file reconstruction depends on the theorem format.
    """
    theorem_id = entry.get("theorem_id", "unknown")
    tactics = entry.get("tactics_used", [])
    tactic_block = "\n  ".join(tactics) if tactics else "sorry"
    return f"import Mathlib\n\ntheorem {theorem_id} := by\n  {tactic_block}\n  sorry\n"


def run_lane_b(report: dict, axle_cfg: dict) -> dict:
    """Run Lane B (Axle) post-processing on benchmark results."""
    try:
        from src.proof_auditor import AuditorConfig, ProofAuditor, SuccessCategory
    except ImportError:
        print("  Warning: axiom-axle not installed, skipping Lane B")
        return report

    auditor_config = AuditorConfig(
        environment=axle_cfg.get("environment", "lean-4.28.0"),
        timeout_seconds=axle_cfg.get("timeout_seconds", 120),
        ignore_imports=axle_cfg.get("ignore_imports", False),
        cache_enabled=axle_cfg.get("cache_enabled", True),
        max_concurrency=axle_cfg.get("max_concurrency", 10),
        repair_terminal_tactics=axle_cfg.get(
            "repair_terminal_tactics", ["grind", "aesop", "simp", "omega", "decide"]
        ),
    )

    async def _lane_b() -> None:
        async with ProofAuditor(auditor_config) as auditor:
            failed = [r for r in report["details"] if not r["success"]]
            print(f"\n--- Lane B: Axle post-processing on {len(failed)} failed theorems ---")

            axle_repair_count = 0
            for entry in failed:
                if entry["goals_closed"] == 0:
                    continue

                result = await auditor.repair(
                    content=_build_sorry_proof(entry),
                )
                if result.success:
                    entry["success"] = True
                    entry["success_category"] = SuccessCategory.AXLE_REPAIR_ONLY.value
                    axle_repair_count += 1

            report["benchmark"]["axle_repair_only"] = axle_repair_count
            total_success = report["benchmark"]["raw_success"] + axle_repair_count
            report["benchmark"]["failed"] = report["benchmark"]["total_theorems"] - total_success

            cache = auditor.cache_stats()
            report["axle"] = {
                "repair_attempted": len([r for r in failed if r["goals_closed"] > 0]),
                "repair_succeeded": axle_repair_count,
                "cache_entries": cache["entries"],
            }
            print(f"  Axle repair: {axle_repair_count} additional theorems proved")

    asyncio.run(_lane_b())
    return report
