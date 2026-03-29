from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_exp_som012_hard_collect import _skipped_row, load_hard_theorems


class TestRunExpSom012HardCollect(unittest.TestCase):
    def test_load_hard_theorems_uses_theorem_statement_and_metadata(self) -> None:
        rows = [
            {
                "theorem_id": "Demo.Demo.hard",
                "theorem_statement": "x : Nat\n⊢ x = x",
                "template_id": "REWRITE_CHAIN",
                "proof_steps": 7,
                "unique_premises": 3,
                "difficulty_band": "hard",
                "hard_half": True,
                "split": "train",
                "file_path": "Mathlib/Demo.lean",
                "module": "Mathlib.Demo",
            },
            {
                "theorem_id": "Demo.other",
                "goal_state": "⊢ True",
                "split": "eval",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hard.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            loaded = load_hard_theorems(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["goal_state"], "x : Nat\n⊢ x = x")
        self.assertEqual(loaded[0]["theorem_statement"], "x : Nat\n⊢ x = x")
        self.assertEqual(loaded[0]["theorem_id"], "Demo.hard")
        self.assertEqual(loaded[0]["raw_theorem_id"], "Demo.Demo.hard")
        self.assertEqual(loaded[0]["template_id"], "REWRITE_CHAIN")
        self.assertEqual(loaded[0]["proof_steps"], 7)
        self.assertTrue(loaded[0]["hard_half"])
        self.assertEqual(loaded[0]["file_path"], "Mathlib/Demo.lean")
        self.assertEqual(loaded[0]["module"], "Mathlib.Demo")
        self.assertEqual(loaded[1]["goal_state"], "⊢ True")
        self.assertEqual(loaded[1]["split"], "eval")

    def test_load_hard_theorems_shuffle_is_deterministic(self) -> None:
        rows = [
            {"theorem_id": f"Demo.t{i}", "theorem_statement": f"⊢ p{i}"} for i in range(6)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hard.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            shuffled_a = load_hard_theorems(path, shuffle=True, seed=7)
            shuffled_b = load_hard_theorems(path, shuffle=True, seed=7)
            shuffled_c = load_hard_theorems(path, shuffle=True, seed=9)

        self.assertEqual(
            [row["theorem_id"] for row in shuffled_a],
            [row["theorem_id"] for row in shuffled_b],
        )
        self.assertNotEqual(
            [row["theorem_id"] for row in shuffled_a],
            [row["theorem_id"] for row in shuffled_c],
        )

    def test_skipped_row_carries_direct_tags(self) -> None:
        row = _skipped_row(
            {
                "theorem_id": "Demo.skip",
                "source": "hard_theorems",
                "goal_state": "",
                "theorem_statement": "⊢ False",
                "template_id": "DECIDE",
            },
            {
                "goal_start_status": "failed",
                "failure_category": "goal_creation_fail",
                "start_failure_family": "metadata_missing",
                "start_failure_tags": ["module_metadata_missing"],
                "module": "",
                "lean_path": "",
                "theorem_line": 0,
                "context_features": {},
                "context_unsupported_kinds": [],
                "feedback": None,
                "repair_attempted": False,
                "repair_success": False,
                "repair_goal_state": "",
                "repair_tier_used": "",
                "repair_failure_category": "",
                "repair_feedback": None,
            },
        )
        self.assertEqual(row["last_goal_bucket"], "empty")
        self.assertEqual(row["reasoning_gap_family"], "compiler_specialist")


if __name__ == "__main__":
    unittest.main()
