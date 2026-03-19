"""Tests for template extraction with SubtaskIR move metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_templates import _summarize_step_metadata, extract_templates
from src.nav_contracts import NavigationalExample


class TestSummarizeStepMetadata(unittest.TestCase):
    def test_aggregates_move_profile(self):
        examples = [
            NavigationalExample(
                theorem_id="Foo.bar",
                goal_state="⊢ a = a",
                metadata={
                    "local_family": "rw",
                    "subtask_kind": "rewrite_chain",
                    "goal_target_head": "Eq",
                    "trigger_signature": ["rewrite_count=2", "direction_prior=forward"],
                },
            ),
            NavigationalExample(
                theorem_id="Foo.bar",
                goal_state="⊢ True",
                metadata={
                    "local_family": "rw",
                    "subtask_kind": "rewrite_chain",
                    "goal_target_head": "Eq",
                    "trigger_signature": ["rewrite_count=2"],
                },
            ),
        ]

        profile = _summarize_step_metadata(examples)
        self.assertEqual(profile["steps_with_metadata"], 2)
        self.assertEqual(profile["dominant_local_family"], "rw")
        self.assertEqual(profile["dominant_subtask_kind"], "rewrite_chain")
        self.assertEqual(profile["top_local_families"][0], {"value": "rw", "count": 2})
        self.assertEqual(profile["top_trigger_kinds"][0], {"value": "rewrite_count", "count": 2})


class TestExtractTemplates(unittest.TestCase):
    def test_writes_template_move_profiles(self):
        examples = [
            NavigationalExample(
                theorem_id="Foo.bar",
                goal_state="⊢ a = a",
                step_index=0,
                total_steps=2,
                nav_directions={"structure": 0, "domain": 0, "depth": 0, "automation": 0, "context": 0, "decomposition": 0},
                ground_truth_tactic="rw",
                metadata={
                    "local_family": "rw",
                    "subtask_kind": "rewrite_chain",
                    "goal_target_head": "Eq",
                    "trigger_signature": ["rewrite_count=2", "direction_prior=forward"],
                },
            ),
            NavigationalExample(
                theorem_id="Foo.bar",
                goal_state="⊢ True",
                step_index=1,
                total_steps=2,
                nav_directions={"structure": 0, "domain": 0, "depth": 0, "automation": 0, "context": 0, "decomposition": 0},
                ground_truth_tactic="rfl",
                metadata={
                    "local_family": "exact",
                    "subtask_kind": "close_with_term",
                    "goal_target_head": "Eq",
                    "trigger_signature": ["term_shape=const"],
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "nav.jsonl"
            output_dir = Path(tmpdir) / "out"
            with open(data_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")

            result = extract_templates(data_path, output_dir)
            self.assertEqual(result["theorems"], 1)

            taxonomy = json.loads((output_dir / "template_taxonomy.json").read_text())
            used_templates = [name for name, info in taxonomy.items() if info["count"] > 0]
            self.assertEqual(len(used_templates), 1)
            template_name = used_templates[0]
            move_profile = taxonomy[template_name]["move_profile"]
            self.assertEqual(move_profile["steps_with_metadata"], 2)
            self.assertEqual(move_profile["top_local_families"][0]["count"], 1)

            rows = [json.loads(line) for line in (output_dir / "nav_train_templates.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["template_name"], template_name)
            self.assertEqual(rows[0]["template_move_profile"]["steps_with_metadata"], 2)
            self.assertEqual(rows[0]["template_move_profile"]["dominant_subtask_kind"], "rewrite_chain")

    def test_groups_by_theorem_key_not_display_name(self):
        examples = [
            NavigationalExample(
                theorem_id="aux",
                theorem_key="Mathlib/A.lean::aux",
                goal_state="⊢ P",
                step_index=0,
                total_steps=1,
                nav_directions={"structure": 0, "domain": 0, "depth": 0, "automation": 0, "context": 0, "decomposition": 0},
                ground_truth_tactic="rw",
            ),
            NavigationalExample(
                theorem_id="aux",
                theorem_key="Mathlib/B.lean::aux",
                goal_state="⊢ Q",
                step_index=0,
                total_steps=1,
                nav_directions={"structure": 0, "domain": 0, "depth": 0, "automation": 0, "context": 0, "decomposition": 0},
                ground_truth_tactic="simp",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "nav.jsonl"
            output_dir = Path(tmpdir) / "out"
            with open(data_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")

            result = extract_templates(data_path, output_dir)
            self.assertEqual(result["theorems"], 2)


if __name__ == "__main__":
    unittest.main()
