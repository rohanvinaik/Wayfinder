from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_hard_theorem_split import build_hard_theorem_split


class TestBuildHardTheoremSplit(unittest.TestCase):
    def test_build_hard_theorem_split_outputs_train_eval(self) -> None:
        rows = [
            {
                "theorem_id": "Demo.Demo.easy",
                "namespace_prefix": "Demo",
                "theorem_statement": "⊢ True",
                "template_id": "DECIDE",
                "proof_history_summary": {"total_steps": 1, "unique_premises": 0},
            },
            {
                "theorem_id": "Demo.medium",
                "namespace_prefix": "Demo",
                "theorem_statement": "⊢ x = x",
                "template_id": "REWRITE_CHAIN",
                "proof_history_summary": {"total_steps": 5, "unique_premises": 2},
            },
            {
                "theorem_id": "Demo.hard",
                "namespace_prefix": "Demo",
                "theorem_statement": "⊢ P ↔ Q",
                "template_id": "DECOMPOSE_AND_CONQUER",
                "proof_history_summary": {"total_steps": 9, "unique_premises": 4},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "narratives.jsonl"
            metadata_path = tmp / "entities.jsonl"
            out_dir = tmp / "out"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [
                        {
                            "theorem_id": "Demo.easy",
                            "file_path": "Mathlib/Demo.lean",
                        },
                        {
                            "theorem_id": "Demo.medium",
                            "file_path": "Mathlib/Demo.lean",
                        },
                        {
                            "theorem_id": "Demo.hard",
                            "file_path": "Mathlib/Demo.lean",
                        },
                    ]
                )
            )

            summary = build_hard_theorem_split(
                input_path,
                out_dir,
                hard_quantile=0.5,
                eval_fraction=0.5,
                metadata_index_path=metadata_path,
            )

            self.assertEqual(summary["total_theorems"], 3)
            self.assertGreaterEqual(summary["hard_total"], 1)
            self.assertEqual(summary["metadata_hits"], 3)

            hard_all = [json.loads(line) for line in (out_dir / "hard_theorems_all.jsonl").open()]

        self.assertTrue(all(row["hard_half"] for row in hard_all))
        self.assertTrue(all(row["file_path"] == "Mathlib/Demo.lean" for row in hard_all))
        self.assertTrue(all(row["module"] == "Mathlib.Demo" for row in hard_all))


if __name__ == "__main__":
    unittest.main()
