"""`local_prover_result` reads a run as running while its worker is alive — and a worker the MCP
process started, once it has exited, as finished: an unreaped child answers `kill(pid, 0)` like a
live process, and a finished run read as running forever."""

from __future__ import annotations

import subprocess
import sys
import time
import unittest

import wayfinder_mcp as m


class PidAliveTest(unittest.TestCase):
    def test_a_running_child_is_alive(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            self.assertTrue(m._pid_alive(p.pid))
        finally:
            p.kill()
            p.wait()

    def test_an_exited_child_is_finished_not_a_zombie_read_as_alive(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        time.sleep(1.0)  # it has exited; nothing has reaped it
        self.assertFalse(m._pid_alive(p.pid))

    def test_no_pid_is_not_alive(self):
        self.assertFalse(m._pid_alive(None))
        self.assertFalse(m._pid_alive("x"))


if __name__ == "__main__":
    unittest.main()
