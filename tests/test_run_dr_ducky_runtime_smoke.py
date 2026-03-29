from __future__ import annotations

import unittest

from scripts.run_dr_ducky_runtime_smoke import run_smoke


class TestRunDrDuckyRuntimeSmoke(unittest.TestCase):
    def test_run_smoke_reports_backend_closure(self) -> None:
        summary = run_smoke()
        self.assertEqual(summary["total_cases"], 5)
        self.assertGreaterEqual(summary["closed_cases"], 4)
        self.assertEqual(summary["progressed_cases"], 5)
        case_map = {case["case_id"]: case for case in summary["cases"]}
        self.assertTrue(case_map["eqsat_close"]["closed"])
        self.assertTrue(case_map["witness_close"]["closed"])
        self.assertTrue(case_map["relational_close"]["closed"])


if __name__ == "__main__":
    unittest.main()
