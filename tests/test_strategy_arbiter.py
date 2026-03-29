from __future__ import annotations

import unittest

from src.strategy_arbiter import StrategyArbiter
from src.temporal_controller import TemporalState


class TestStrategyArbiter(unittest.TestCase):
    def test_default_lane_order_contains_modern_full_stack(self) -> None:
        arbiter = StrategyArbiter(mode="full")
        state = TemporalState(
            theorem_id="Nat.add_assoc",
            open_goals=["⊢ a + b + c = a + (b + c)"],
        )
        decision = arbiter.decide(state)
        self.assertIn("cosine_apply", decision.lane_order)
        self.assertIn("learned", decision.lane_order)
        self.assertIn("dr_ducky", decision.lane_order)

    def test_category_theory_prefers_close_before_unpack(self) -> None:
        arbiter = StrategyArbiter(mode="full")
        state = TemporalState(
            theorem_id="CategoryTheory.Adjunction.isIso_counit_app_iff_mem_essImage",
            open_goals=["CategoryTheory.IsIso (h.counit.app X) ↔ L.essImage X"],
        )
        decision = arbiter.decide(state)
        self.assertEqual(decision.lane_order[0], "interleaved_bootstrap")
        self.assertLess(decision.lane_order.index("cosine_exact"), decision.lane_order.index("cosine_rw"))
        self.assertEqual(decision.lane_order[-1], "automation")

    def test_witness_goal_promotes_structural_lane(self) -> None:
        arbiter = StrategyArbiter(mode="full")
        state = TemporalState(
            theorem_id="Besicovitch.le_multiplicity_of_δ_of_fin",
            open_goals=["n ≤ sSup {N | ∃ s, s.card = N ∧ True}"],
        )
        decision = arbiter.decide(state)
        self.assertEqual(decision.lane_order[0], "interleaved_bootstrap")

    def test_membership_wall_prefers_exact_before_rewrite(self) -> None:
        arbiter = StrategyArbiter(mode="full")
        state = TemporalState(
            theorem_id="AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier.smul_mem",
            open_goals=["c • x ∈ AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f_deg q"],
        )
        decision = arbiter.decide(state)
        self.assertEqual(decision.lane_order[0], "cosine_exact")


if __name__ == "__main__":
    unittest.main()
