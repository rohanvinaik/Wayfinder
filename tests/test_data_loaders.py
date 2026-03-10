"""Tests for data.py JSONL loader functions."""

import json
import tempfile
import unittest
from pathlib import Path

from src.contracts import OODPrompt
from src.data import load_nav_examples_jsonl, load_ood_prompts_jsonl
from src.nav_contracts import NavigationalExample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ood_prompt(
    prompt: str = "Prove add_comm",
    label: str = "in_domain",
    category: str = "lean_goal",
    source: str = "leandojo",
) -> OODPrompt:
    return OODPrompt(prompt=prompt, label=label, category=category, source=source)


def _make_nav_example(
    theorem_id: str = "thm.alpha",
    step_index: int = 0,
    total_steps: int = 3,
    remaining_steps: int = 3,
    solvable: bool = True,
) -> NavigationalExample:
    return NavigationalExample(
        goal_state="a + b = b + a",
        theorem_id=theorem_id,
        step_index=step_index,
        total_steps=total_steps,
        nav_directions={
            "structure": 1,
            "domain": 0,
            "depth": -1,
            "automation": 0,
            "context": 1,
            "decomposition": -1,
        },
        anchor_labels=["Nat.add_comm", "Nat.succ"],
        ground_truth_tactic="ring",
        ground_truth_premises=["Nat.add_comm"],
        remaining_steps=remaining_steps,
        solvable=solvable,
        proof_history=["intro a", "intro b"],
    )


def _write_jsonl(records: list[dict], *, blank_lines: list[int] | None = None) -> Path:
    """Write dicts as JSONL to a temp file, optionally inserting blank lines."""
    blank_set = set(blank_lines or [])
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for i, rec in enumerate(records):
        if i in blank_set:
            tmp.write("\n")
        tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Trailing blank after last record if requested
    if len(records) in blank_set:
        tmp.write("\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ===================================================================
# load_ood_prompts_jsonl
# ===================================================================


class TestLoadOODPromptsJsonl(unittest.TestCase):
    """Tests for load_ood_prompts_jsonl."""

    def test_two_valid_records(self):
        p1 = _make_ood_prompt("Prove add_comm", "in_domain", "lean_goal", "leandojo")
        p2 = _make_ood_prompt("Write a poem", "ood", "general_chat", "synthetic")
        path = _write_jsonl([p1.to_dict(), p2.to_dict()])
        prompts = load_ood_prompts_jsonl(path)

        self.assertEqual(len(prompts), 2)
        # First record
        self.assertEqual(prompts[0].prompt, "Prove add_comm")
        self.assertEqual(prompts[0].label, "in_domain")
        self.assertEqual(prompts[0].category, "lean_goal")
        self.assertEqual(prompts[0].source, "leandojo")
        # Second record
        self.assertEqual(prompts[1].prompt, "Write a poem")
        self.assertEqual(prompts[1].label, "ood")
        self.assertEqual(prompts[1].category, "general_chat")
        self.assertEqual(prompts[1].source, "synthetic")

    def test_empty_lines_skipped(self):
        p = _make_ood_prompt()
        # blank line before record 0, and trailing blank after record 0
        path = _write_jsonl([p.to_dict()], blank_lines=[0, 1])
        prompts = load_ood_prompts_jsonl(path)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].prompt, p.prompt)

    def test_malformed_json_raises_valueerror(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.write("this is not json\n")
        tmp.close()
        with self.assertRaises(ValueError) as ctx:
            load_ood_prompts_jsonl(Path(tmp.name))
        self.assertIn("line 1", str(ctx.exception))

    def test_missing_required_key_raises_valueerror(self):
        # OODPrompt.from_dict requires "prompt" and "label"
        incomplete = {"prompt": "hello"}  # missing "label"
        path = _write_jsonl([incomplete])
        with self.assertRaises(ValueError) as ctx:
            load_ood_prompts_jsonl(path)
        self.assertIn("line 1", str(ctx.exception))

    def test_empty_file_returns_empty_list(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        self.assertEqual(load_ood_prompts_jsonl(Path(tmp.name)), [])

    def test_return_type_is_ood_prompt(self):
        p = _make_ood_prompt()
        path = _write_jsonl([p.to_dict()])
        prompts = load_ood_prompts_jsonl(path)
        self.assertIsInstance(prompts[0], OODPrompt)
        # Verify the loaded prompt preserves the original field values
        self.assertEqual(prompts[0].prompt, "Prove add_comm")
        self.assertEqual(prompts[0].label, "in_domain")
        self.assertEqual(prompts[0].category, "lean_goal")
        self.assertEqual(prompts[0].source, "leandojo")


# ===================================================================
# load_nav_examples_jsonl
# ===================================================================


class TestLoadNavExamplesJsonl(unittest.TestCase):
    """Tests for load_nav_examples_jsonl."""

    def test_two_valid_records(self):
        e1 = _make_nav_example(theorem_id="thm.alpha", step_index=0, total_steps=3)
        e2 = _make_nav_example(theorem_id="thm.beta", step_index=1, total_steps=5)
        path = _write_jsonl([e1.to_dict(), e2.to_dict()])
        examples = load_nav_examples_jsonl(path)

        self.assertEqual(len(examples), 2)
        # First record field checks
        self.assertEqual(examples[0].theorem_id, "thm.alpha")
        self.assertEqual(examples[0].step_index, 0)
        self.assertEqual(examples[0].total_steps, 3)
        self.assertEqual(examples[0].nav_directions["structure"], 1)
        self.assertEqual(examples[0].nav_directions["depth"], -1)
        self.assertEqual(examples[0].ground_truth_tactic, "ring")
        self.assertEqual(examples[0].ground_truth_premises, ["Nat.add_comm"])
        self.assertEqual(examples[0].anchor_labels, ["Nat.add_comm", "Nat.succ"])
        self.assertEqual(examples[0].solvable, True)
        self.assertEqual(examples[0].proof_history, ["intro a", "intro b"])
        # Second record
        self.assertEqual(examples[1].theorem_id, "thm.beta")
        self.assertEqual(examples[1].step_index, 1)
        self.assertEqual(examples[1].total_steps, 5)

    def test_empty_lines_skipped(self):
        e = _make_nav_example()
        path = _write_jsonl([e.to_dict()], blank_lines=[0, 1])
        examples = load_nav_examples_jsonl(path)
        self.assertEqual(len(examples), 1)

    def test_malformed_json_raises_valueerror(self):
        e = _make_nav_example()
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.write(json.dumps(e.to_dict()) + "\n")
        tmp.write("NOT JSON\n")
        tmp.close()
        with self.assertRaises(ValueError) as ctx:
            load_nav_examples_jsonl(Path(tmp.name))
        self.assertIn("line 2", str(ctx.exception))

    def test_missing_required_key_raises_valueerror(self):
        incomplete = {"goal_state": "x = x"}  # missing theorem_id, nav_directions
        path = _write_jsonl([incomplete])
        with self.assertRaises(ValueError) as ctx:
            load_nav_examples_jsonl(path)
        self.assertIn("line 1", str(ctx.exception))

    def test_nav_directions_exact_values(self):
        dirs = {
            "structure": -1,
            "domain": 1,
            "depth": 0,
            "automation": 1,
            "context": -1,
            "decomposition": 0,
        }
        e = NavigationalExample(
            goal_state="g",
            theorem_id="t",
            step_index=2,
            total_steps=4,
            nav_directions=dirs,
            anchor_labels=[],
            ground_truth_tactic="simp",
            ground_truth_premises=[],
            remaining_steps=2,
        )
        path = _write_jsonl([e.to_dict()])
        loaded = load_nav_examples_jsonl(path)[0]
        self.assertEqual(loaded.nav_directions, dirs)
        self.assertEqual(loaded.remaining_steps, 2)
        self.assertEqual(loaded.ground_truth_tactic, "simp")

    def test_empty_file_returns_empty_list(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        self.assertEqual(load_nav_examples_jsonl(Path(tmp.name)), [])

    def test_return_type_is_navigational_example(self):
        e = _make_nav_example()
        path = _write_jsonl([e.to_dict()])
        examples = load_nav_examples_jsonl(path)
        self.assertIsInstance(examples[0], NavigationalExample)
        # Verify the loaded example preserves original field values
        self.assertEqual(examples[0].theorem_id, "thm.alpha")
        self.assertEqual(examples[0].goal_state, "a + b = b + a")
        self.assertEqual(examples[0].step_index, 0)
        self.assertEqual(examples[0].total_steps, 3)
        self.assertEqual(examples[0].ground_truth_tactic, "ring")


if __name__ == "__main__":
    unittest.main()
