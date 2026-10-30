"""The capacity run's harness: a remote-only stream never claims a theorem no server can try."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

WF = Path(__file__).resolve().parents[1]


def _harness():
    spec = importlib.util.spec_from_file_location("decomposition_test_harness", WF / "scripts" / "decomposition_test.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        sys.argv = argv
    return mod


class CircuitBreakerTest(unittest.TestCase):
    def test_a_stream_whose_servers_do_not_answer_stops_instead_of_claiming(self):
        try:
            dt = _harness()
        except ModuleNotFoundError as e:  # the harness runs on the MCP host's python (it imports `mcp`)
            self.skipTest(f"the harness needs {e.name}")
        with mock.patch.object(dt.lp, "pool_config", return_value={"k": [{"name": "gone", "port": 1}]}):
            self.assertFalse(dt._pool_alive("k"))
            self.assertFalse(dt._await_servers("k", patience_s=0))

    def test_an_arm_with_no_answer_is_infrastructure(self):
        self.assertIn("unserved = not samples and not fills", (WF / "scripts" / "decomposition_test.py").read_text())


if __name__ == "__main__":
    unittest.main()
