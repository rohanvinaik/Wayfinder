"""Tests for data loading -- JSONL roundtrip."""

import tempfile
import unittest
from pathlib import Path

from src.contracts import ProofExample
from src.data import ProofDataset, load_proofs_jsonl, save_proofs_jsonl


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_proofs.jsonl"


class TestDataIO(unittest.TestCase):
    def test_load_fixture(self):
        examples = load_proofs_jsonl(FIXTURE_PATH)
        self.assertGreater(len(examples), 0)
        self.assertIsInstance(examples[0], ProofExample)
        self.assertTrue(examples[0].theorem_id)

    def test_save_load_roundtrip(self):
        examples = load_proofs_jsonl(FIXTURE_PATH)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
        count = save_proofs_jsonl(examples, path)
        self.assertEqual(count, len(examples))

        reloaded = load_proofs_jsonl(path)
        self.assertEqual(len(reloaded), len(examples))
        self.assertEqual(reloaded[0].theorem_id, examples[0].theorem_id)

    def test_dataset_class(self):
        ds = ProofDataset(FIXTURE_PATH)
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        self.assertIsInstance(ex, ProofExample)


if __name__ == "__main__":
    unittest.main()
