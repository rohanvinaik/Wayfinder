from __future__ import annotations

import contextlib
import io
import unittest

from src.lean_interface import LeanKernel


class _VerboseServer:
    def goal_start(self, expr: str) -> str:
        print(f"Cannot start goal: {expr}")
        return f"goal:{expr}"


class TestLeanInterfaceQuietGoalStart(unittest.TestCase):
    def test_quiet_goal_start_suppresses_stdout_noise(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = LeanKernel._quiet_goal_start(_VerboseServer(), "X = Y")
        self.assertEqual(result, "goal:X = Y")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
