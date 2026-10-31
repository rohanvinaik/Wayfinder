"""Tests for censor provenance and staleness checking (S0.6)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.censor_store import (
    load_censor_store,
    staleness,
)


class StalenessVerdictsTest(unittest.TestCase):
    def test_unknown_when_commit_missing_or_none(self):
        self.assertEqual(staleness(None, []), "unknown")
        self.assertEqual(staleness("", []), "unknown")
        self.assertEqual(staleness(None, ["src/foo.py"]), "unknown")

    def test_unknown_when_changed_paths_none(self):
        self.assertEqual(staleness("8fad0f4", None), "unknown")

    def test_fresh_when_no_changed_paths(self):
        self.assertEqual(staleness("8fad0f4", []), "fresh")
        self.assertEqual(staleness("8fad0f4", set()), "fresh")
        self.assertEqual(staleness("8fad0f4", ()), "fresh")

    def test_stale_when_changed_paths_present(self):
        self.assertEqual(staleness("8fad0f4", ["src/proof_search.py"]), "stale")
        self.assertEqual(staleness("8fad0f4", {"src/censor_store.py"}), "stale")


class CensorLoaderPolicyTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.store_path = Path(self.td.name) / "store.json"
        self.store_content = {
            "version": 1,
            "mined_at_commit": "8fad0f4",
            "adopted": [
                {"key": "equality|0,0,0,-1,0,0", "family": "exact"}
            ]
        }
        self.store_path.write_text(json.dumps(self.store_content))

    def tearDown(self):
        self.td.cleanup()

    @mock.patch("src.censor_store.check_censor_store_staleness")
    def test_refuse_yields_no_fences_on_stale(self, mock_check):
        mock_check.return_value = "stale"
        fences, verdict = load_censor_store(str(self.store_path), stale_policy="refuse")
        self.assertEqual(verdict, "stale")
        self.assertEqual(fences, {})

    @mock.patch("src.censor_store.check_censor_store_staleness")
    def test_warn_yields_fences_on_stale(self, mock_check):
        mock_check.return_value = "stale"
        fences, verdict = load_censor_store(str(self.store_path), stale_policy="warn")
        self.assertEqual(verdict, "stale")
        self.assertEqual(fences, {"equality|0,0,0,-1,0,0": {"exact"}})

    @mock.patch("src.censor_store.check_censor_store_staleness")
    def test_fresh_yields_fences(self, mock_check):
        mock_check.return_value = "fresh"
        fences, verdict = load_censor_store(str(self.store_path), stale_policy="refuse")
        self.assertEqual(verdict, "fresh")
        self.assertEqual(fences, {"equality|0,0,0,-1,0,0": {"exact"}})


class MinerWriterTest(unittest.TestCase):
    def test_miner_writes_mined_at_commit_field(self):
        from scripts.mine_censors import current_commit
        commit = current_commit()
        self.assertIsInstance(commit, str)
        # Should be a hex sha or non-empty string when in a git repo
        if commit:
            self.assertEqual(len(commit), 40)


if __name__ == "__main__":
    unittest.main()
