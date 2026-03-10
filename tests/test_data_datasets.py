"""Tests for data.py dataset classes: OODPromptDataset, NegativeBankDataset, NavigationalDataset."""

import json
import tempfile
import unittest
from pathlib import Path

from src.contracts import NegativeBankEntry, OODPrompt, ProofExample
from src.data import NavigationalDataset, NegativeBankDataset, OODPromptDataset
from src.nav_contracts import NavigationalExample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proof_example(theorem_id: str = "thm.alpha") -> ProofExample:
    return ProofExample(
        theorem_id=theorem_id,
        goal_state="a + b = b + a",
        theorem_statement="theorem add_comm (a b : Nat) : a + b = b + a",
        proof_text="by ring",
        tier1_tokens=["ring"],
        tier2_blocks=[],
        tier3_slots=[],
        metadata={"source": "test"},
    )


def _make_ood_prompt(
    prompt: str = "Prove add_comm",
    label: str = "in_domain",
    category: str = "lean_goal",
    source: str = "leandojo",
) -> OODPrompt:
    return OODPrompt(prompt=prompt, label=label, category=category, source=source)


def _make_negative_bank_entry(
    theorem_id: str = "thm.alpha",
    goal_state: str = "a + b = b + a",
    source: str = "failed_proof",
    error_tags: list[str] | None = None,
    with_negative: bool = True,
) -> NegativeBankEntry:
    positive = _make_proof_example(theorem_id)
    negative = _make_proof_example(theorem_id + ".neg") if with_negative else None
    return NegativeBankEntry(
        goal_state=goal_state,
        theorem_id=theorem_id,
        positive=positive,
        negative=negative,
        error_tags=error_tags or ["wrong_tactic"],
        source=source,
    )


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
    if len(records) in blank_set:
        tmp.write("\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ===================================================================
# OODPromptDataset
# ===================================================================


class TestOODPromptDataset(unittest.TestCase):
    """Tests for OODPromptDataset."""

    def _make_dataset(self, n: int = 3) -> tuple[OODPromptDataset, list[OODPrompt]]:
        prompts = [
            _make_ood_prompt(
                f"prompt_{i}", "in_domain" if i % 2 == 0 else "ood", f"cat_{i}", f"src_{i}"
            )
            for i in range(n)
        ]
        path = _write_jsonl([p.to_dict() for p in prompts])
        return OODPromptDataset(path), prompts

    def test_len(self):
        ds, prompts = self._make_dataset(3)
        self.assertEqual(len(ds), 3)

    def test_getitem_returns_ood_prompt(self):
        ds, prompts = self._make_dataset(2)
        item = ds[0]
        self.assertIsInstance(item, OODPrompt)
        self.assertEqual(item.prompt, "prompt_0")

    def test_getitem_correct_fields(self):
        ds, prompts = self._make_dataset(2)
        self.assertEqual(ds[0].prompt, "prompt_0")
        self.assertEqual(ds[0].label, "in_domain")
        self.assertEqual(ds[0].category, "cat_0")
        self.assertEqual(ds[0].source, "src_0")
        self.assertEqual(ds[1].prompt, "prompt_1")
        self.assertEqual(ds[1].label, "ood")
        self.assertEqual(ds[1].category, "cat_1")
        self.assertEqual(ds[1].source, "src_1")

    def test_path_attribute(self):
        ds, _ = self._make_dataset(1)
        self.assertIsInstance(ds.path, Path)
        self.assertTrue(ds.path.exists())
        # Path should end with .jsonl suffix
        self.assertEqual(ds.path.suffix, ".jsonl")
        # Dataset loaded 1 record, confirming path was used correctly
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].prompt, "prompt_0")

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        ds = OODPromptDataset(Path(tmp.name))
        self.assertEqual(len(ds), 0)


# ===================================================================
# NegativeBankDataset
# ===================================================================


class TestNegativeBankDataset(unittest.TestCase):
    """Tests for NegativeBankDataset."""

    def _make_dataset(
        self, n: int = 2, with_negative: bool = True
    ) -> tuple[NegativeBankDataset, list[NegativeBankEntry]]:
        entries = [
            _make_negative_bank_entry(
                theorem_id=f"thm.{i}",
                goal_state=f"goal_{i}",
                source="failed_proof" if i % 2 == 0 else "synthetic_mutation",
                error_tags=[f"tag_{i}"],
                with_negative=with_negative,
            )
            for i in range(n)
        ]
        path = _write_jsonl([e.to_dict() for e in entries])
        return NegativeBankDataset(path), entries

    def test_len(self):
        ds, _ = self._make_dataset(3)
        self.assertEqual(len(ds), 3)

    def test_getitem_returns_negative_bank_entry(self):
        ds, _ = self._make_dataset(1)
        self.assertIsInstance(ds[0], NegativeBankEntry)
        self.assertEqual(ds[0].theorem_id, "thm.0")

    def test_getitem_correct_fields(self):
        ds, entries = self._make_dataset(2)
        self.assertEqual(ds[0].theorem_id, "thm.0")
        self.assertEqual(ds[0].goal_state, "goal_0")
        self.assertEqual(ds[0].source, "failed_proof")
        self.assertEqual(ds[0].error_tags, ["tag_0"])
        self.assertEqual(ds[1].theorem_id, "thm.1")
        self.assertEqual(ds[1].source, "synthetic_mutation")

    def test_positive_is_proof_example(self):
        ds, _ = self._make_dataset(1)
        self.assertIsInstance(ds[0].positive, ProofExample)
        self.assertEqual(ds[0].positive.theorem_id, "thm.0")

    def test_negative_is_proof_example_when_present(self):
        ds, _ = self._make_dataset(1, with_negative=True)
        self.assertIsInstance(ds[0].negative, ProofExample)
        self.assertEqual(ds[0].negative.theorem_id, "thm.0.neg")

    def test_negative_is_none_when_absent(self):
        ds, _ = self._make_dataset(1, with_negative=False)
        self.assertIsNone(ds[0].negative)
        self.assertIsNotNone(ds[0].positive)
        self.assertEqual(ds[0].positive.theorem_id, "thm.0")

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        ds = NegativeBankDataset(Path(tmp.name))
        self.assertEqual(len(ds), 0)

    def test_path_attribute(self):
        ds, _ = self._make_dataset(1)
        self.assertIsInstance(ds.path, Path)
        self.assertTrue(ds.path.exists())
        # Path should end with .jsonl suffix
        self.assertEqual(ds.path.suffix, ".jsonl")
        # Dataset loaded 1 record, confirming path was used correctly
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].theorem_id, "thm.0")


# ===================================================================
# NavigationalDataset
# ===================================================================


class TestNavigationalDataset(unittest.TestCase):
    """Tests for NavigationalDataset."""

    def _write_nav_file(self) -> Path:
        """Create a file with examples of varying total_steps: 1, 2, 3, 5."""
        examples = [
            _make_nav_example(theorem_id="short1", total_steps=1, remaining_steps=1),
            _make_nav_example(theorem_id="short2", total_steps=2, remaining_steps=2),
            _make_nav_example(theorem_id="mid", total_steps=3, remaining_steps=3),
            _make_nav_example(theorem_id="long", total_steps=5, remaining_steps=5),
        ]
        return _write_jsonl([e.to_dict() for e in examples])

    def test_no_max_steps_loads_all(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path)
        self.assertEqual(len(ds), 4)

    def test_max_steps_filters(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path, max_steps=2)
        self.assertEqual(len(ds), 2)
        # Only short1 (1 step) and short2 (2 steps)
        ids = [ds[i].theorem_id for i in range(len(ds))]
        self.assertEqual(ids, ["short1", "short2"])

    def test_max_steps_boundary_inclusive(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path, max_steps=3)
        self.assertEqual(len(ds), 3)
        ids = [ds[i].theorem_id for i in range(len(ds))]
        self.assertEqual(ids, ["short1", "short2", "mid"])

    def test_max_steps_zero_excludes_all(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path, max_steps=0)
        self.assertEqual(len(ds), 0)

    def test_max_steps_very_large_includes_all(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path, max_steps=1000)
        self.assertEqual(len(ds), 4)

    def test_getitem_returns_navigational_example(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path)
        self.assertIsInstance(ds[0], NavigationalExample)
        self.assertEqual(ds[0].theorem_id, "short1")

    def test_getitem_correct_fields(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path)
        item = ds[0]
        self.assertEqual(item.theorem_id, "short1")
        self.assertEqual(item.total_steps, 1)
        self.assertEqual(item.nav_directions["structure"], 1)
        self.assertEqual(item.nav_directions["depth"], -1)
        self.assertEqual(item.ground_truth_tactic, "ring")
        self.assertEqual(item.anchor_labels, ["Nat.add_comm", "Nat.succ"])
        self.assertEqual(item.solvable, True)

    def test_path_attribute(self):
        path = self._write_nav_file()
        ds = NavigationalDataset(path)
        self.assertEqual(ds.path, path)

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.close()
        ds = NavigationalDataset(Path(tmp.name))
        self.assertEqual(len(ds), 0)


if __name__ == "__main__":
    unittest.main()
