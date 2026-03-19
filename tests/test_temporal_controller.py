"""Tests for temporal controller — phase tracking, escalation, goal ordering."""

import unittest

from src.temporal_controller import (
    OrchestrationDecision,
    TemporalController,
    TemporalState,
)


class TestTemporalState(unittest.TestCase):
    def test_defaults(self):
        state = TemporalState(theorem_id="thm1")
        self.assertEqual(state.phase, "structural_setup")
        self.assertEqual(state.escalation_level, 0)
        self.assertEqual(state.budget_remaining, 600)
        self.assertEqual(state.open_goals, [])

    def test_goal_tracking(self):
        state = TemporalState(theorem_id="thm1", open_goals=["g0", "g1"])
        self.assertEqual(len(state.open_goals), 2)


class TestPhaseDetection(unittest.TestCase):
    def setUp(self):
        self.ctrl = TemporalController()

    def test_fresh_goal_is_structural_setup(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.phase, "structural_setup")

    def test_after_structural_failure_moves_to_local(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 4
        state.goal_lane_failures["g0"] = {"structural_core"}
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.phase, "local_close")

    def test_after_local_failure_moves_to_automation(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 7
        state.goal_lane_failures["g0"] = {"structural_core", "learned"}
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.phase, "automation_close")

    def test_many_failures_trigger_replan(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 12
        state.goal_lane_failures["g0"] = {"structural_core", "learned", "automation"}
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.phase, "repair_or_replan")


class TestGoalSelection(unittest.TestCase):
    def setUp(self):
        self.ctrl = TemporalController()

    def test_prefers_fresh_goals(self):
        state = TemporalState(theorem_id="t", open_goals=["g0", "g1", "g2"])
        state.goal_attempt_counts["g0"] = 5
        state.goal_attempt_counts["g1"] = 0
        state.goal_attempt_counts["g2"] = 3
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.next_goal_id, "g1")

    def test_single_goal_selected(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.next_goal_id, "g0")


class TestEscalation(unittest.TestCase):
    def setUp(self):
        self.ctrl = TemporalController(escalation_thresholds=(3, 6, 10))

    def test_level_0_at_start(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.escalation_level, 0)

    def test_level_1_after_threshold(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 4
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.escalation_level, 1)

    def test_level_3_at_max(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 15
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.escalation_level, 3)


class TestLaneOrder(unittest.TestCase):
    def setUp(self):
        self.ctrl = TemporalController()

    def test_structural_phase_structural_first(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.lane_order[0], "structural_core")

    def test_local_phase_learned_first(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_attempt_counts["g0"] = 4
        state.goal_lane_failures["g0"] = {"structural_core"}
        decision = self.ctrl.decide(state)
        self.assertEqual(decision.lane_order[0], "learned")


class TestReplan(unittest.TestCase):
    def setUp(self):
        self.ctrl = TemporalController()

    def test_no_replan_early(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = self.ctrl.decide(state)
        self.assertFalse(decision.replan)

    def test_replan_when_all_lanes_exhausted(self):
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        state.goal_lane_failures["g0"] = {"structural_core", "learned", "automation"}
        decision = self.ctrl.decide(state)
        self.assertTrue(decision.replan)


class TestUpdate(unittest.TestCase):
    def test_success_updates_state(self):
        ctrl = TemporalController()
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        ctrl.update(state, "g0", "structural_core", "intros", "intros", success=True)
        self.assertEqual(state.successful_tactics, ["intros"])
        self.assertEqual(state.prior_lanes, ["structural_core"])
        self.assertEqual(state.budget_remaining, 599)

    def test_failure_updates_state(self):
        ctrl = TemporalController()
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        ctrl.update(state, "g0", "learned", "exact", "exact foo", success=False)
        self.assertEqual(state.failed_tactics, ["exact foo"])
        self.assertIn("learned", state.goal_lane_failures["g0"])


class TestDecisionContract(unittest.TestCase):
    def test_decision_has_all_fields(self):
        ctrl = TemporalController()
        state = TemporalState(theorem_id="t", open_goals=["g0"])
        decision = ctrl.decide(state)
        self.assertIsInstance(decision, OrchestrationDecision)
        self.assertIsInstance(decision.next_goal_id, str)
        self.assertIn(decision.phase, [
            "structural_setup", "local_close", "automation_close", "repair_or_replan",
        ])
        self.assertIsInstance(decision.lane_order, list)
        self.assertIsInstance(decision.family_prior, list)
        self.assertIsInstance(decision.escalation_level, int)
        self.assertIsInstance(decision.budget_slice, int)
        self.assertIsInstance(decision.replan, bool)


if __name__ == "__main__":
    unittest.main()
