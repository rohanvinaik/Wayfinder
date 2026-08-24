"""Intent tests for the disposition calculus (Wave 5).

Pins the precedence order (infra > skipped > closed > abstained >
progressed > blocked), the honest/suspect split consuming the computed
`honest_success`, the never-promote rule, and Detective's two suggested
boundary witnesses (censor_drop_count == 1; goals_closed == 1)."""

import unittest

from src.dispositions import (
    ABSTAINED_CERTIFIED,
    BLOCKED_UNPROVEN,
    CLOSED_SUSPECT,
    CLOSED_VALUE,
    DISPOSITIONS,
    INFRA,
    NEXT_ACTION,
    PROGRESSED,
    SKIPPED_START,
    disposition_of_row,
    goal_disposition,
    next_action,
)


class PrecedenceTest(unittest.TestCase):
    def test_crash_outranks_everything_even_success_fields(self):
        self.assertEqual(goal_disposition(True, True, True, "crashed", 1, 5, 0), INFRA)

    def test_unbuilt_goal_is_the_compiler_track(self):
        self.assertEqual(goal_disposition(False, False, False, "failed", 0, 0, 0), SKIPPED_START)

    def test_closures_split_on_computed_honesty(self):
        self.assertEqual(goal_disposition(True, True, True, "raw_success", 1, 5, 0), CLOSED_VALUE)
        self.assertEqual(
            goal_disposition(True, False, True, "self_application", 1, 5, 0), CLOSED_SUSPECT
        )


class BoundaryWitnessTest(unittest.TestCase):
    """Detective's suggested boundary pins, hand-applied."""

    def test_single_censor_drop_with_zero_attempts_is_certified_abstention(self):
        self.assertEqual(
            goal_disposition(False, False, True, "failed", 0, 0, 1), ABSTAINED_CERTIFIED
        )

    def test_one_closed_goal_is_progress_not_blocked(self):
        self.assertEqual(goal_disposition(False, False, True, "failed", 1, 9, 0), PROGRESSED)

    def test_zero_attempts_without_drops_is_blocked_not_abstained(self):
        # Abstention requires a WARRANT (a recorded fence); a started row
        # with nothing spent and nothing withheld is simply blocked.
        self.assertEqual(
            goal_disposition(False, False, True, "failed", 0, 0, 0), BLOCKED_UNPROVEN
        )

    def test_attempts_spent_with_drops_is_not_abstention(self):
        # Abstention requires the attempts to have been WITHHELD, not spent.
        self.assertEqual(
            goal_disposition(False, False, True, "failed", 0, 9, 3), BLOCKED_UNPROVEN
        )


class NeverPromoteTest(unittest.TestCase):
    def test_blocked_is_unproven_never_unprovable(self):
        self.assertEqual(goal_disposition(False, False, True, "failed", 0, 4, 0), BLOCKED_UNPROVEN)
        self.assertEqual(next_action(BLOCKED_UNPROVEN), "hand_off_residual")

    def test_every_disposition_has_exactly_one_action(self):
        self.assertEqual(set(NEXT_ACTION), set(DISPOSITIONS))
        for code in DISPOSITIONS:
            self.assertTrue(next_action(code))

    def test_unknown_disposition_fails_loudly(self):
        with self.assertRaises(KeyError):
            next_action("maybe_provable")


class RowAdapterTest(unittest.TestCase):
    def test_adapter_consumes_augmented_fields(self):
        row = {
            "success": False,
            "honest_success": False,
            "started": True,
            "success_category": "failed",
            "goals_closed": 0,
            "attempts": 12,
            "censor_drop_count": 4,
        }
        self.assertEqual(disposition_of_row(row), BLOCKED_UNPROVEN)

    def test_adapter_defaults_do_not_invent_success(self):
        self.assertEqual(disposition_of_row({}), BLOCKED_UNPROVEN)


if __name__ == "__main__":
    unittest.main()
