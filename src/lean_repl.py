"""A persistent Lean REPL for the repair loop's checks — Mathlib imported once, not per compile.

Every `lake env lean` compile re-imports Mathlib (6–20 s); the loop checks a candidate, its
skeleton, the closer probe and the lift probes, so most of its time went to imports. Here each
check runs as one REPL command against the environment saved right after the file's imports
(never a chain of environments: a name declared by one check would clash with the next).

A Lean process kept alive drifts (Pantograph "starts to go a little weird and crashes silently"), so
each REPL is distrusted: recycled after `recycle_after` commands (it keeps every environment it
makes), a timeout on every command (the process is killed and restarted), and a canary — a known-
true and a known-false theorem — on every start and every `canary_every` checks. Any error is the
caller's cue to fall back to a fresh compile; final acceptance never runs here (the gate compiles
from scratch).

Returns what `proof_gate.run_lean_json` returns — {"returncode", "messages", "timed_out",
"tail"} — with message positions in the whole text's lines.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path

_IMPORT = re.compile(r"^\s*(import\s+\S.*)?$")  # an import line, or a blank one among them
_CANARY_TRUE = "example : (2 : ℕ) + 2 = 4 := by norm_num"
_CANARY_FALSE = "example : (2 : ℕ) + 2 = 5 := by norm_num"


class ReplError(RuntimeError):
    """The REPL could not answer (dead, timed out, failed its canary): compile from scratch instead."""


def _split(text: str) -> tuple[str, str, int]:
    """(the import header, the rest, the rest's first line's offset): imports must be the first command."""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and _IMPORT.match(lines[k]):
        k += 1
    header = "\n".join(ln for ln in lines[:k] if ln.strip())
    return header, "\n".join(lines[k:]), k


class LeanRepl:
    # the REPL keeps every environment a command makes (there is no command to free one): it grows
    # ~0.04 GB a command, so it is recycled often — 30 commands, then a 7 s re-import
    def __init__(self, project: Path, repl_bin: Path, lake_bin: str = "lake", recycle_after: int = 30,
                 canary_every: int = 15):  # fmt: skip
        self.project, self.repl_bin, self.lake_bin = Path(project), Path(repl_bin), lake_bin
        self.recycle_after, self.canary_every = recycle_after, canary_every
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.envs: dict[str, int] = {}
        self.count = 0
        self._env: dict | None = None

    # ── the process ──

    def _environment(self) -> dict:
        """The checking project's Lean environment (`lake env` once), so the REPL runs directly —
        no `lake` between us and it, nothing left behind when it is killed."""
        if self._env is None:
            out = subprocess.run([self.lake_bin, "env"], cwd=str(self.project), capture_output=True, text=True,
                                 timeout=120, check=True).stdout  # fmt: skip
            env = dict(os.environ)
            for line in out.splitlines():
                key, sep, value = line.partition("=")
                if sep and key.isidentifier():
                    env[key] = value
            self._env = env
        return self._env

    def _start(self) -> None:
        self.stop()
        self.proc = subprocess.Popen([str(self.repl_bin)], cwd=str(self.project), env=self._environment(),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)  # fmt: skip
        self.lines = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()
        self.envs, self.count = {}, 0
        self._canary()

    @staticmethod
    def _read(proc: subprocess.Popen, out: queue.Queue) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            out.put(line)
        out.put(None)  # the process is gone

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def _ask(self, cmd: dict, timeout: float) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise ReplError("the REPL is not running")
        self.proc.stdin.write(json.dumps(cmd) + "\n\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]
        got: list[str] = []
        while True:
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty:
                self.stop()
                raise ReplError(f"no answer in {timeout:.0f} s") from None
            if line is None:
                self.stop()
                raise ReplError("the REPL exited")
            if not line.strip() and got:
                break
            if line.strip():
                got.append(line)
        try:
            return json.loads("".join(got))
        except ValueError as e:
            self.stop()
            raise ReplError(f"unreadable answer: {''.join(got)[:200]!r}") from e

    def _env_for(self, header: str, timeout: float = 300) -> int:
        if header not in self.envs:
            r = self._ask({"cmd": header or "import Mathlib"}, timeout)
            if any(m.get("severity") == "error" for m in r.get("messages") or []) or "env" not in r:
                raise ReplError(f"the imports do not load: {json.dumps(r)[:200]}")
            self.envs[header] = int(r["env"])
        return self.envs[header]

    def _canary(self) -> None:
        env = self._env_for("import Mathlib")
        good = self._ask({"cmd": _CANARY_TRUE, "env": env}, 60)
        bad = self._ask({"cmd": _CANARY_FALSE, "env": env}, 60)
        if any(m.get("severity") == "error" for m in good.get("messages") or []) or not any(
            m.get("severity") == "error" for m in bad.get("messages") or []
        ):
            self.stop()
            raise ReplError("the REPL failed its canary")

    # ── a check ──

    def check(self, text: str, timeout: float = 300) -> dict:
        if self.proc is None or self.proc.poll() is not None or self.count >= self.recycle_after:
            self._start()
        elif self.count and self.count % self.canary_every == 0:
            self._canary()
        header, body, offset = _split(text)
        r = self._ask({"cmd": body, "env": self._env_for(header)}, timeout)
        self.count += 1
        messages = []
        for m in r.get("messages") or []:
            m = dict(m)
            if m.get("severity") == "info":  # `lean --json` says "information"; readers match on it
                m["severity"] = "information"
            if "maximum number of heartbeats" in str(m.get("data", "")):
                # where a heartbeat limit lands differs from a fresh compile's (elaboration near the limit
                # is not reproduced exactly here): such a check is settled by a fresh compile
                raise ReplError("a heartbeat timeout: confirm with a fresh compile")
            for key in ("pos", "endPos"):
                if isinstance(m.get(key), dict) and "line" in m[key]:
                    m[key] = {**m[key], "line": int(m[key]["line"]) + offset}
            if m.get("severity") == "warning" and "declaration uses" in str(m.get("data", "")):
                m.setdefault("kind", "hasSorry")
            messages.append(m)
        return {"returncode": 0, "messages": messages, "timed_out": False, "tail": ""}


class ReplPool:
    """One REPL per concurrent caller (the verification threads and the parallel skeleton
    repairs check side by side; a REPL answers one command at a time)."""

    def __init__(self, size: int, **kw):
        self.free: queue.Queue = queue.Queue()
        self.all = [LeanRepl(**kw) for _ in range(size)]
        for r in self.all:
            self.free.put(r)

    def check(self, text: str, timeout: float = 300) -> dict:
        repl = self.free.get()
        try:
            return repl.check(text, timeout)
        finally:
            self.free.put(repl)

    def stop(self) -> None:
        for r in self.all:
            r.stop()
