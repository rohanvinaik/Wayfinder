"""Intent tests for the cheat-sheet renderer (Wave 4).

Pins the AIMO3 discipline as transported: byte-determinism, the no-hedging
vocabulary fence, section presence/omission, and the one-vocabulary
consumption of dispositions and structural reads."""

import unittest

from src.cheat_sheet import render_cheat_sheet, residual_goals

_ROW = {
    "theorem_id": "Demo.thm",
    "success": False,
    "honest_success": False,
    "started": True,
    "success_category": "failed",
    "goals_closed": 1,
    "attempts": 14,
    "censor_drop_count": 2,
    "tactics_used": ["intro h", "norm_cast"],
    "step_trace": [
        {
            "progress": True,
            "lane": "structural_core",
            "lanes_tried": ["structural_core", "symbolic_rw"],
            "open_goals_after": ["x : ℕ\n⊢ x + 0 = x"],
        }
    ],
    "censor_drops": [{"family": "exact"}, {"family": "exact"}],
}

# The regression fence, verbatim from the AIMO3 renderer's design law.
_HEDGING_WORDS = ("confidence", "difficult", "easy", "likely", "probably", "tractab", "hopefully")


class DeterminismTest(unittest.TestCase):
    def test_same_row_same_bytes(self):
        self.assertEqual(render_cheat_sheet(_ROW), render_cheat_sheet(dict(_ROW)))


class NoHedgingTest(unittest.TestCase):
    def test_no_hedging_vocabulary(self):
        sheet = render_cheat_sheet(_ROW).lower()
        for word in _HEDGING_WORDS:
            self.assertNotIn(word, sheet)


class SectionsTest(unittest.TestCase):
    def test_verdict_consumes_the_disposition_calculus(self):
        sheet = render_cheat_sheet(_ROW)
        self.assertIn("verdict: progressed", sheet)
        self.assertIn("next action: continue_or_hand_off", sheet)

    def test_negative_channel_names_fenced_families(self):
        sheet = render_cheat_sheet(_ROW)
        self.assertIn("exact: fenced 2 time(s)", sheet)

    def test_residual_walk_carries_read_evidence(self):
        sheet = render_cheat_sheet(_ROW)
        self.assertIn("residual: 1 open goal(s)", sheet)
        # A nonzero read renders with its evidence; non-slice zeros are
        # filtered (they carry no hand-off content).
        self.assertIn("coupling", sheet)
        self.assertIn("vs mined zero", sheet)
        self.assertNotIn("depth", sheet.split("residual:")[1].split("lane history")[0])

    def test_statement_section_only_with_manifest_row(self):
        bare = render_cheat_sheet(_ROW)
        with_stmt = render_cheat_sheet(_ROW, {"full_declaration": "theorem Demo.thm : True"})
        self.assertNotIn("statement:", bare)
        self.assertIn("statement:", with_stmt)

    def test_empty_row_still_renders_verdict_and_handoff(self):
        sheet = render_cheat_sheet({"theorem_id": "E"})
        self.assertIn("WAYFINDER CHEAT SHEET — E", sheet)
        self.assertIn("hand-off:", sheet)


class ResidualGoalsTest(unittest.TestCase):
    def test_trace_tail_wins_over_final_goals(self):
        row = {
            "step_trace": [{"open_goals_after": ["from_trace"]}],
            "final_goals": ["from_field"],
        }
        self.assertEqual(residual_goals(row), ["from_trace"])

    def test_empty_trace_tail_falls_back_to_final_goals(self):
        row = {"step_trace": [{"open_goals_after": []}], "final_goals": ["f"]}
        self.assertEqual(residual_goals(row), ["f"])


if __name__ == "__main__":
    unittest.main()
