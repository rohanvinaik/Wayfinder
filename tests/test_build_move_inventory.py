from __future__ import annotations

import unittest

from scripts.build_move_inventory import build_inventory


class TestBuildMoveInventory(unittest.TestCase):
    def test_groups_by_family_and_subtask(self):
        rows = [
            {
                "family": "apply",
                "subtask_kind": "reduce_goal_by_lemma",
                "subtask_summary": "reduce",
                "expected_effect": "side goals",
                "goal_target_head": "StrictMono",
                "primary_premise": "foo",
                "tactic_text": "apply foo",
                "theorem_full_name": "A",
                "step_index": 0,
                "trigger_profile_ir": {
                    "features": [
                        {"kind": "term_shape", "value": "const"},
                        {"kind": "local_context", "value": "3"},
                    ]
                },
            },
            {
                "family": "apply",
                "subtask_kind": "reduce_goal_by_lemma",
                "subtask_summary": "reduce",
                "expected_effect": "side goals",
                "goal_target_head": "StrictMono",
                "primary_premise": "bar",
                "tactic_text": "apply bar",
                "theorem_full_name": "B",
                "step_index": 1,
                "trigger_profile_ir": {
                    "features": [
                        {"kind": "term_shape", "value": "const"},
                    ]
                },
            },
        ]
        inventory = build_inventory(rows, {"apply"}, min_support=2)
        self.assertIn("apply", inventory["families"])
        entries = inventory["families"]["apply"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["subtask_kind"], "reduce_goal_by_lemma")
        self.assertEqual(entry["support"], 2)
        self.assertEqual(entry["top_target_heads"][0][0], "StrictMono")


if __name__ == "__main__":
    unittest.main()
