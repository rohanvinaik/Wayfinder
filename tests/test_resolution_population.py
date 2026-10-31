"""Tests for resolution population freezing and P2 fixture integrity (§S0.7)."""

import hashlib
import json
import unittest
from pathlib import Path

from scripts.build_resolution_population import build_summary, collect_population


class ResolutionPopulationTest(unittest.TestCase):
    def test_population_reproducibility(self):
        records = collect_population()
        self.assertGreater(len(records), 0)
        # Check sorted by (theorem_id, id)
        keys = [(r["theorem_id"], r["id"]) for r in records]
        self.assertEqual(keys, sorted(keys))

        summary = build_summary(records)
        self.assertEqual(summary["total_rows"], len(records))

        # Check serialization sha matches
        serialized = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
        sha = hashlib.sha256(serialized).hexdigest()
        self.assertEqual(
            sha,
            "af6af99d7f43bb1ffb084fe77ef9f3cabd9db14779c5ce1ddab7b825cc25819f",
        )

    def test_p2_fixtures_exist(self):
        fixtures_dir = Path("tests/fixtures/resolution")
        expected = [
            "f1a_specimen_subset",
            "f1b_specimen_main",
            "f2a_herpderp_subset",
            "f2b_herpderp_main",
            "f3_misleading",
            "f4_renamed_candidate",
            "f5_dependent",
            "f6_two_location_simp",
            "f7_impossible_transport",
            "f8_continuation",
            "f9_opaque_definition",
            "f10_rejected_constant",
        ]
        for name in expected:
            d = fixtures_dir / name
            self.assertTrue(d.is_dir(), f"Missing fixture directory {name}")
            self.assertTrue((d / "file.lean").exists(), f"Missing file.lean in {name}")
            self.assertTrue((d / "step.txt").exists(), f"Missing step.txt in {name}")
            self.assertTrue((d / "expect.json").exists(), f"Missing expect.json in {name}")

    def test_p3_core_manifest(self):
        p3 = Path("data/eval/p3_core.jsonl")
        self.assertTrue(p3.exists())
        lines = [json.loads(l) for l in p3.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 6)


if __name__ == "__main__":
    unittest.main()
