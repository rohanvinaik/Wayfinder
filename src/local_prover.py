"""The local prover lane — open-weights Lean provers on this machine, given only Wayfinder's residual.

`send_to_local_prover` (wayfinder_mcp.py) writes a job — the hand-off file, its open holes, the
statement's elaborated type — and starts this worker detached:

    python -m src.local_prover <run_dir>/job.json

The worker lifts each hole to a standalone lemma (`extract_goal *`, checked to re-elaborate),
reads how hard it looks from Wayfinder's own structural reads, and routes it: a light hole to
Goedel-Prover-V2-8B first and on to the 32B if the 8B leaves it, a heavy hole straight to the
32B. Each model is served by its own llama-server while it has holes — sized to the memory free
when it starts, stopped when it is done, and stopped early if the machine comes under memory
pressure, so other work keeps its RAM. Prompts follow Goedel-Prover-V2's own pipeline: its
"Complete the following Lean 4 code" request, then self-correction rounds that show the model
Lean's errors in the format it was trained on.

A candidate counts only once checked: the deny-list first (nothing that runs code, no `sorry`),
then the lifted lemma compiled with the model's proof, then the fill in place in the hand-off
file (elaborated statement and axioms checked). The worker only proposes: verified fills go to
`fills.json`, and `local_prover_result` applies them through the same path as `fill_hole`. Every
sample is logged with what happened to it, so a truncated, cancelled or yielded sample is never
read as the model getting the proof wrong.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import src.hole_fill as hf
import src.lean_repl as lr
import src.proof_gate as gate
import src.residual_handoff as rh
from src.structural_reads import goal_target, read_goal

GiB = 1024**3


# ── the models ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str  # GGUF file name under the job's gguf_dir
    quant: str
    layers: int
    kv_heads: int
    head_dim: int
    ctx: int = 40960  # the trained context (max_position_embeddings): 32K of proof plus the prompt

    def kv_bytes_per_token(self) -> int:
        """K and V for every layer in a q8_0 cache (32 values in 34 bytes)."""
        return self.layers * self.kv_heads * self.head_dim * 2 * 34 // 32


# Goedel-Prover-V2 (Qwen3 architecture). The 32B at Q4_K_M: decoding reads every weight per
# token, so on this machine the bandwidth, not the arithmetic, sets the speed, and Q8 would be
# about 1.75x slower for little accuracy.
MODELS = {
    "goedel-v2-8b": ModelSpec("goedel-v2-8b", "Goedel-Prover-V2-8B.Q8_0.gguf", "Q8_0", 36, 8, 128),
    "goedel-v2-32b": ModelSpec("goedel-v2-32b", "Goedel-Prover-V2-32B.Q4_K_M.gguf", "Q4_K_M", 64, 8, 128),
}
LIGHT = "goedel-v2-8b"
HEAVY = "goedel-v2-32b"
# Lean's budget for a candidate: Goedel-Prover-V2's own files set 400000; `0` (unlimited) would
# let one runaway candidate hold the checker forever.
HEARTBEATS = 400000
SKELETON_HOLES = 4  # a rejected proof with more failing steps than this is revised, not repaired
DRAFT_HOLES = 8  # a draft's own `sorry`s count too: Goedel's sketches leave one on every `have`
SEARCH_BUDGET = 200  # tactic attempts for Wayfinder's search on each lifted step
SEARCH_GOALS = Path(__file__).resolve().parents[1] / "scripts" / "search_goals.py"
REPL_BIN = Path(__file__).resolve().parents[1] / "data" / "repl_v428" / ".lake" / "build" / "bin" / "repl"
REPL_SIZE = 3  # REPLs per worker: the verification threads and the parallel skeleton repairs check side by side
# Errors this Mathlib raises where the model's 2024 Mathlib passed silently: a tactic run after its
# goal was already closed, or one that changed nothing. Deleting exactly that line cannot change what
# is proved — the tactic acted on nothing — and the result is compiled again. Only when every error
# is of these kinds, only inside the model's proof, at most REPAIR_ROUNDS times.
_REPAIRABLE = ("No goals to be solved", "made no progress")
REPAIR_ROUNDS = 3


def repairable_lines(messages: list[dict], first: int, last: int) -> set[int] | None:
    """The 1-based lines to delete, if every error is a no-goals / no-progress error on a line in
    [first, last]; else None."""
    lines: set[int] = set()
    for m in messages:
        if m.get("severity") != "error":
            continue
        line = int((m.get("pos") or {}).get("line", -1))
        if not any(k in str(m.get("data", "")) for k in _REPAIRABLE) or not first <= line <= last:
            return None
        lines.add(line)
    return lines or None


# ── routing: how hard does the hole look? ─────────────────────────────────────

HEAVY_AT = 3  # difficulty votes at which a hole goes straight to the 32B


def assess(goal_text: str, defs: Iterable[str], whole_statement: bool) -> dict:
    """How hard a hole looks, read from its goal, as votes with their reasons.

    The votes come from Wayfinder's structural reads (src/structural_reads.py) plus what the
    reads do not see: an existential target, the file's own definitions, a statement Wayfinder
    made no progress on. `tier` is "heavy" at HEAVY_AT votes or more.
    """
    reads = {r.bank: r for r in read_goal(goal_text).reads}
    target = goal_target(goal_text)
    votes: list[tuple[int, str]] = []

    def vote(fired: bool, n: int, why: str) -> None:
        if fired:
            votes.append((n, why))

    vote(reads["structure"].sign > 0, 1, f"quantified or dependent target ({reads['structure'].evidence})")
    vote(reads["automation"].sign > 0, 1, f"abstract surface ({reads['automation'].evidence})")
    vote(reads["automation"].sign < 0, -1, "hammer-shaped surface")
    vote(reads["domain"].sign > 0, 1, f"abstract domain ({reads['domain'].evidence})")
    vote(reads["size"].sign > 0, 1, f"large target ({reads['size'].evidence})")
    vote(reads["size"].sign < 0, -1, f"compact target ({reads['size'].evidence})")
    vote(reads["coupling"].sign < 0, 1, "the target shares no names with its hypotheses: library lemmas carry it")
    vote("∃" in target, 1, "existential target: a witness is needed")
    used = [d for d in defs if re.search(r"(?<![\w.'])" + re.escape(d) + r"(?![\w'])", goal_text)]
    vote(bool(used), 1, f"needs the file's own definitions ({', '.join(used[:3])})")
    vote(whole_statement, 1, "Wayfinder found no foothold: the hole is the whole statement")
    score = sum(n for n, _ in votes)
    return {"tier": "heavy" if score >= HEAVY_AT else "light", "score": score, "votes": [w for _, w in votes]}


def model_plan(assessment: dict, forced: str | None = None) -> list[str]:
    """The models a hole goes to, in order: a light hole escalates to the 32B if the 8B leaves it."""
    if forced:
        return [forced]
    return [HEAVY] if assessment["tier"] == "heavy" else [LIGHT, HEAVY]


# ── memory ────────────────────────────────────────────────────────────────────


def available_bytes() -> int:
    """Memory free for a new process: free, inactive, speculative and purgeable pages."""
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().available)
    except ImportError:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))  # type: ignore[union-attr]
        kinds = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        return page * sum(int(m.group(1)) for k in kinds if (m := re.search(rf"{k}:\s+(\d+)", out)))


def orphaned_servers(ps_lines: str | None = None) -> list[str]:
    """Pids of Goedel llama-servers whose worker is gone (reparented to launchd). A server whose
    worker is alive is never one: another stream may be sampling on it. The squishos router on
    port 8080 serves no Goedel GGUF, so it never matches."""
    if ps_lines is None:
        ps_lines = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in ps_lines.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "1" or "models/gguf/Goedel" not in parts[2]:
            continue
        if parts[2].split(maxsplit=1)[0].rsplit("/", 1)[-1] == "llama-server":
            found.append(parts[0])
    return found


def pressure_level() -> int:
    """The kernel's memory-pressure level: 1 normal, 2 warn, 4 critical (0 when unknown)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip() or 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def plan_slots(
    spec: ModelSpec, weights: int, available: int, reserve: int, max_slots: int, ctx: int | None = None
) -> int:
    """How many `ctx`-token sequences (default: the model's full context) fit beside the weights,
    keeping `reserve` bytes free."""
    overhead = int(1.5 * GiB)  # compute buffers and the server itself
    per_slot = (ctx or spec.ctx) * spec.kv_bytes_per_token()
    return max(0, min(max_slots, (available - reserve - weights - overhead) // per_slot))


# ── llama-server ──────────────────────────────────────────────────────────────


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def read_events(lines: Iterable[bytes], should_stop: Callable[[], bool]) -> dict:
    """Collect a streamed `/completion`: the text, and the final event's counts and timings.

    `cancelled` is set when `should_stop` fired before the server finished.
    """
    parts: list[str] = []
    final: dict = {}
    for raw in lines:
        if should_stop():
            return {"content": "".join(parts), "cancelled": True}
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: ") :])
        parts.append(str(event.get("content", "")))
        if event.get("stop"):
            final = event
            break
    else:
        # no final event: the server stopped mid-answer or the connection dropped, so what came
        # through is a fragment, not an answer to score
        raise ConnectionError("the stream ended before the server's final event")
    timings = final.get("timings") or {}
    return {
        "content": "".join(parts),
        "cancelled": False,
        "stop_type": final.get("stop_type"),
        "tokens_predicted": final.get("tokens_predicted"),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "truncated": final.get("truncated"),
        "tok_per_s": timings.get("predicted_per_second"),
        "gen_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"),
    }


class LlamaServer:
    """One llama-server for one model, on a free localhost port, for as long as its holes last."""

    def __init__(
        self, binary: str, model_path: Path, spec: ModelSpec, slots: int, log_path: Path, ctx: int | None = None
    ):
        self.binary, self.model_path, self.spec, self.slots, self.log_path = binary, model_path, spec, slots, log_path
        self.ctx = ctx or spec.ctx  # tokens per slot: the prompt and the answer
        self.name, self.quant = "local", spec.quant
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.stopping = False

    def command(self) -> list[str]:
        return [
            self.binary, "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.ctx * self.slots), "-np", str(self.slots), "-fa", "on", "-ctk", "q8_0",
            "-ctv", "q8_0", "-ngl", "999", "--no-context-shift",
        ]  # fmt: skip

    def start(self, timeout: float = 900) -> None:
        self.port = free_port()
        log = open(self.log_path, "ab")  # noqa: SIM115 — held for the server's lifetime
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if self._request("GET", "/health")[0] == 200:
                    return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError(f"llama-server did not become ready in {timeout:.0f}s; see {self.log_path}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        self.stopping = True
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, json.dumps(body) if body is not None else None,
                         {"Content-Type": "application/json"})  # fmt: skip
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, (json.loads(data) if data else {})
        finally:
            conn.close()

    def count_tokens(self, text: str) -> int:
        status, body = self._request("POST", "/tokenize", {"content": text}, timeout=120)
        if status != 200:
            raise RuntimeError(f"/tokenize answered {status}")
        return len(body.get("tokens") or [])

    def complete(self, prompt: str, n_predict: int, seed: int, should_stop: Callable[[], bool]) -> dict:
        """One sample, streamed so it can be cancelled; Goedel-Prover-V2's sampling (T=1, top-p 0.95)."""
        body = {
            "prompt": prompt, "n_predict": n_predict, "seed": seed, "stream": True, "cache_prompt": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0,
        }  # fmt: skip
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            conn.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"/completion answered {resp.status}: {resp.read()[:300]!r}")
            return read_events(resp, should_stop)
        finally:
            conn.close()  # closing mid-stream is how a sample is cancelled


class RemoteServer(LlamaServer):
    """A llama-server another machine runs for the same model, reached through a localhost tunnel
    (`ssh -L port:127.0.0.1:port node`): never started or stopped here, dropped when it fails."""

    def __init__(self, spec: ModelSpec, name: str, port: int, slots: int, ctx: int, quant: str):
        super().__init__("", Path(), spec, slots, Path(), ctx=ctx)
        self.name, self.quant, self.port = name, quant, port

    def start(self, timeout: float = 0) -> None:
        if self._request("GET", "/health", timeout=10)[0] != 200:
            raise RuntimeError(f"{self.name}: /health did not answer 200")

    def alive(self) -> bool:
        return True

    def stop(self) -> None:
        self.stopping = True


def pool_config() -> dict:
    """Machine-local remote servers per model: $WAYFINDER_LLM_POOL, else ~/.config/wayfinder/llm_pool.json.

    {"goedel-v2-8b": [{"name": "node", "port": 18081, "slots": 2, "ctx": 16384, "quant": "Q4_K_M"}]}
    """
    path = Path(os.environ.get("WAYFINDER_LLM_POOL") or Path.home() / ".config" / "wayfinder" / "llm_pool.json")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class Pool:
    """The servers one model's attempts go to: the local server first, then the remotes. Each slot
    serves one attempt at a time; a remote that fails is dropped and its attempt tried elsewhere."""

    def __init__(self, servers: list[LlamaServer]):
        self.servers = list(servers)
        self.local = self.servers[0]
        self.free = {id(s): s.slots for s in self.servers}
        self.dead: set[int] = set()
        self.cond = threading.Condition()

    @property
    def slots(self) -> int:
        return sum(s.slots for s in self.servers if id(s) not in self.dead)

    def acquire(self, exclude: set[int] | frozenset[int] = frozenset()) -> LlamaServer | None:
        with self.cond:
            while True:
                live = [s for s in self.servers if id(s) not in self.dead and id(s) not in exclude]
                if not live:
                    return None
                for srv in live:
                    if self.free[id(srv)] > 0:
                        self.free[id(srv)] -= 1
                        return srv
                self.cond.wait(timeout=5)

    def release(self, srv: LlamaServer) -> None:
        with self.cond:
            self.free[id(srv)] += 1
            self.cond.notify()

    def drop(self, srv: LlamaServer) -> None:
        with self.cond:
            self.dead.add(id(srv))
            self.cond.notify_all()


# ── prompts: Goedel-Prover-V2's own (Goedel-LM/Goedel-Prover-V2, src/utils.py) ─────

ROUND0 = (
    "Complete the following Lean 4 code:\n\n```lean4\n{statement}```\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
    "outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, "
    "and proof structures that will guide the construction of the final formal proof."
)
CORRECTION = (
    "The proof (Round {previous}) is not correct. Following is the compilation error message, where we use "
    "<error></error> to signal the position of the error.\n\n{errors}\n\nBefore producing the Lean 4 code to "
    "formally prove the given theorem, provide a detailed analysis of the error message."
)


def chatml(messages: list[dict]) -> str:
    """The text the models' chat template renders (Qwen3: no system turn, thinking on).

    An earlier assistant turn keeps only what follows its `</think>`, as the template drops a
    past turn's reasoning; the prompt ends by opening the assistant's turn.
    """
    out = []
    for m in messages:
        content = str(m["content"])
        if m["role"] == "assistant" and "</think>" in content:
            content = content.split("</think>")[-1].lstrip("\n")
        out.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
    return "".join(out) + "<|im_start|>assistant\n"


# ── a hole, ready for a model ─────────────────────────────────────────────────


@dataclass
class Hole:
    index: int
    goal: str  # the goal Lean printed at the hole
    statement: str = ""  # what the model proves: the theorem itself, or the lifted hole
    name: str = ""  # the name `statement` declares
    helper_name: str = ""  # the name its proof takes when it goes in as a helper lemma
    binders: list[tuple[str, str]] = field(default_factory=list)
    rename: str | None = None  # `rename_i …` restoring the lifted names in place ("" = none needed)
    assessment: dict = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    solved_by: str = ""
    note: str = ""
    outcomes: Counter = field(default_factory=Counter)
    retyped: bool = False  # lifted from the typed print, the plain one not fitting the hole
    tried: str = ""  # steps Wayfinder tried at this goal and withdrew (they closed nothing under them)


class Worker:
    def __init__(self, job_path: Path):
        self.job = json.loads(job_path.read_text())
        self.dir = job_path.parent
        (self.dir / "outputs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_reason = ""
        self.stop_kind = ""  # "requested" | "memory" | "server"
        self.counter = Counter()
        self.holes: list[Hole] = []
        self.fills: list[dict] = []
        self.repaired: dict[int, int] = {}
        self.models: dict[str, dict] = {}
        self.status: dict = {
            "run_id": self.job["run_id"], "theorem_id": self.job["theorem_id"], "pid": os.getpid(),
            "state": "starting", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": [],
        }  # fmt: skip
        file, decl = self.job["file"], self.job["declaration"]
        self.prefix = file[: file.rindex(decl)]
        self.target = self.job["target"]
        self.allowed = set(gate.STANDARD_AXIOMS) | set(self.job.get("allowed_axioms") or [])
        self.known = hf.declared_names(self.prefix) | {self.target}

    # ── bookkeeping ──

    def _write(self, name: str, data) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=list))
        tmp.replace(self.dir / name)

    def publish(self, **changes) -> None:
        with self.lock:
            self.status.update(changes)
            self.status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.status["holes"] = {
                str(h.index): {"assessment": h.assessment, "plan": h.plan, "solved_by": h.solved_by, "note": h.note,
                               "outcomes": dict(h.outcomes), "statement": h.statement}  # fmt: skip
                for h in self.holes
            }
            self.status["models"] = self.models
            self.status["stop_reason"] = self.stop_reason
            self._write("status.json", self.status)

    def note(self, text: str) -> None:
        with self.lock:
            self.status["notes"].append(text)

    def compile(self, text: str, timeout: int = 300) -> dict:
        """One check: through the persistent REPL when the job allows it (Mathlib imported once),
        else — and on any doubt about the REPL — a fresh `lake env lean`. The run's final
        acceptance never comes through here."""
        pool = self._repl_pool()
        if pool is not None:
            try:
                return pool.check(text, timeout)
            except Exception as e:  # noqa: BLE001 — any doubt: a fresh compile
                with self.lock:
                    self.repl_fallbacks = getattr(self, "repl_fallbacks", 0) + 1
                    if self.repl_fallbacks <= 3:
                        self.status["notes"].append(f"REPL fell back to a fresh compile: {type(e).__name__}: {e}"[:200])
        return gate.run_lean_json(self.job["lake_bin"], self.job["verify_project"], text, timeout=timeout)

    def _repl_pool(self) -> lr.ReplPool | None:
        # a live kill switch: this file turns the REPL off for every worker that starts after it appears
        if not self.job.get("repl", False) or not REPL_BIN.exists() or (REPL_BIN.parents[3] / "DISABLED").exists():
            return None
        with self.lock:
            if getattr(self, "repl_pool", None) is None:
                self.repl_pool = lr.ReplPool(REPL_SIZE, project=Path(self.job["verify_project"]), repl_bin=REPL_BIN,
                                             lake_bin=str(self.job["lake_bin"]))  # fmt: skip
            return self.repl_pool

    # ── lifting ──

    def prepare(self) -> None:
        defs = self.job.get("defs") or []
        forced = None if self.job.get("model") in (None, "", "auto") else self.job["model"]
        whole = self.job["status"] == "bare"
        for h in self.job["holes"]:
            hole = Hole(index=int(h["index"]), goal=str(h.get("lean_goal") or h.get("search_goal") or ""))
            hole.assessment = assess(hole.goal, defs, whole)
            hole.plan = model_plan(hole.assessment, forced)
            self.holes.append(hole)
        if whole:
            # The whole statement: the model proves the theorem itself, under its own binders.
            statement = self.job["statement"]
            for hole in self.holes:
                hole.statement, hole.name = statement, self.target
                hole.helper_name = hf.lifted_name(self.target, hole.index)
                hole.binders, hole.rename = hf.binders(statement), ""
            return
        self.publish(state="lifting")
        for k, steps in hf.tried_at(self.job["declaration"]).items():
            for hole in self.holes:
                if hole.index == k:
                    hole.tried = steps
        probe = self.prefix + hf.lifting_probe(self.job["declaration"], self.target)
        variants = hf.lifted_variants(self.compile(probe)["messages"], self.target)
        printed = {(k, i): st for k, pair in variants.items() for i, st in enumerate(pair) if st}
        block, ranges = hf.validation_block(printed)
        first = self.prefix.count("\n") + 1
        broken = hf.failing_ranges(self.compile(self.prefix + block)["messages"], ranges, first) if printed else {}
        for hole in self.holes:
            # the plain print if it elaborates back, else the one with types written out
            usable = [(i, st) for (k, i), st in sorted(printed.items()) if k == hole.index and (k, i) not in broken]
            if not any(k == hole.index for k, _ in printed):
                hole.note, hole.plan = "extract_goal printed no statement for this hole", []
            elif not usable:
                why = broken.get((hole.index, 1)) or broken.get((hole.index, 0))
                hole.note, hole.plan = f"the lifted statement does not re-elaborate: {why}", []
            else:
                i, statement = usable[0]
                if i == 0 and len(usable) > 1 and not self.fits(hole, statement):
                    # the plain print elaborates, but not as the goal it came from (a numeral or
                    # binder re-read at another type: ℕ where the goal has ℤ), so an answer to it
                    # cannot close the hole; the typed print says what the goal says
                    self.note(f"hole {hole.index}: the plain print does not fit its hole; lifted with types")
                    i, statement = usable[1]
                    hole.retyped = True
                hole.statement = statement if i == 0 else hf.untyped_name(statement, self.target, hole.index)
                hole.name = hole.helper_name = hf.lifted_name(self.target, hole.index)
                hole.binders = hf.binders(statement)
                hole.rename = hf.rename_line(hf.context_names(hole.goal), hole.binders)

    def fits(self, hole: Hole, statement: str) -> bool:
        """Whether a lifted statement, taken on `sorry`, closes its hole in the hand-off — by the same
        helper forms an answer is applied with, so a print that means something else fails here
        rather than after the model has proved it."""
        name = hf.lifted_name(self.target, hole.index)
        lifted = hf.binders(statement)
        rename = hf.rename_line(hf.context_names(hole.goal), lifted)
        if rename is None:
            fill = f"apply {name} <;> assumption"
        else:
            fill = "\n".join(([rename] if rename else []) + [hf.helper_call(name, lifted)])
        new_file, _ = hf.apply_fill(self.job["file"], self.job["declaration"], hole.index, fill,
                                    f"{statement} := by\n  sorry")  # fmt: skip
        run = self.compile(new_file)
        return not run["timed_out"] and not gate.errors_in(run["messages"])

    def formal(self, hole: Hole) -> str:
        # a residual's goal is a step of a theorem the model never sees otherwise: the theorem goes with
        # it as context; what Wayfinder tried and withdrew there goes too, as information, not as a step
        context = ""
        declaration = str((self.job.get("row") or {}).get("full_declaration", ""))
        if self.job.get("status") == "residual" and ":=" in declaration:
            try:
                header = rh.split_declaration(declaration)[0]
            except (ValueError, IndexError):
                header = ""
            header = re.sub(r"\s*:=\s*by\s*$", "", header).strip()
            if header and "-/" not in header:
                context = f"/- This goal is a step of the theorem below; Wayfinder proved the rest:\n{header} -/\n"
        note = (f"/- Wayfinder tried these steps on this goal and withdrew them, as nothing under them "
                f"closed: {hole.tried} -/\n") if hole.tried else ""  # fmt: skip
        return f"{self.prefix}{context}{note}{hole.statement} := by sorry"

    # ── checking one answer ──

    def verify(self, hole: Hole, raw: str, stop_type: str | None) -> dict:
        code = hf.extract_code(raw)
        if code is None:
            # cut off mid-proof: its unfinished block is still the model's plan, a draft to take forward
            cut = hf.cut_code(raw) if stop_type == "limit" else None
            return {"outcome": "truncated" if stop_type == "limit" else "no_code", **({"cut": cut} if cut else {})}
        bad = hf.code_findings(code)
        if bad:
            return {"outcome": "denied", "reasons": bad}
        code, rewrites = hf.modernize(code)
        res = self._check(hole, code)
        if rewrites:
            res["rewrites"] = rewrites
        return res

    def _check(self, hole: Hole, code: str) -> dict:
        cand = hf.candidate(code, hole.name, self.known)
        if cand is None:
            return {"outcome": "no_theorem"}
        helpers = [cand.scoped(h) for h in cand.helpers]
        bad = hf.fill_findings(cand.proof) + hf.helper_findings("\n\n".join(helpers))
        if bad:  # one whose only fault is `sorry` is a draft of the plan: the repair loop can take it forward
            return {"outcome": "denied", "reasons": bad, "draft": all("`sorry`" in r for r in bad)}
        head = f"set_option maxHeartbeats {HEARTBEATS} in\n" + cand.scoped(f"{hole.statement} :=")
        lemma = head + cand.proof
        above = self.prefix + "\n\n".join(helpers + [""])  # everything before the lemma
        standalone = above + lemma + "\n" + gate.probe_suffix(hole.name)
        run = self.compile(standalone)
        if run["timed_out"]:
            return {"outcome": "compile_timeout"}
        msgs = run["messages"]
        errors = gate.errors_in(msgs)
        # the proof's own lines: after the line its `:=` ends, through the lemma's last line
        first, last = (above + head).count("\n") + 2, (above + lemma).count("\n") + 1
        removed = 0
        for _ in range(REPAIR_ROUNDS):
            lines = repairable_lines(msgs, first, last) if errors else None
            if not lines:
                break
            kept = [ln for i, ln in enumerate(standalone.split("\n"), 1) if i not in lines]
            standalone, last, removed = "\n".join(kept), last - len(lines), removed + len(lines)
            run = self.compile(standalone)
            if run["timed_out"]:
                return {"outcome": "compile_timeout"}
            msgs = run["messages"]
            errors = gate.errors_in(msgs)
        if removed and not errors:
            lemma = "\n".join(standalone.split("\n")[above.count("\n") : last])
            cand.proof = lemma[len(head) :]
            self.repaired[hole.index] = removed
        axioms = gate.axioms_from_messages(msgs, hole.name)
        if errors or gate.uses_sorry(msgs) or axioms is None or not set(axioms) <= self.allowed:
            why = errors[:3] or [f"axioms: {axioms}" if axioms is not None else "no `#print axioms` report"]
            return {"outcome": "wrong_proof", "errors": why, "n_errors": len(errors),
                    "report": hf.error_report(standalone, msgs) or "\n".join(why)}  # fmt: skip
        res = self.in_place(hole, cand, helpers, lemma)
        if removed and not errors:
            res["repaired"] = f"removed {removed} tactic line(s) that ran with no goals or made no progress"
        return res

    # ── a rejected proof's skeleton: Goedel-Prover-V2 §5, with Wayfinder in the loop ──

    def skeleton_repair(self, hole: Hole, code: str, draft: bool = False) -> dict:
        """The rejected proof's plan kept — its `have` claims, case splits, witnesses — each step
        Lean rejected made a hole, and Wayfinder's closers on the holes, with `plausible` beside
        them (a refuted hole is a false claim of the model's). When every hole closes, the
        assembled proof goes through the one acceptance path, `_check`. Returns the outcome
        ("verified" | "open" | "no_skeleton") with what happened at each hole."""
        code, _ = hf.modernize(code)
        cand = hf.candidate(code, hole.name, self.known)
        if cand is None:
            return {"outcome": "no_skeleton", "why": "no theorem in the answer"}
        helpers = [cand.scoped(h) for h in cand.helpers]
        head = f"set_option maxHeartbeats {HEARTBEATS} in\n" + cand.scoped(f"{hole.statement} :=")
        above = self.prefix + "\n\n".join(helpers + [""])
        start = above.count("\n") + 1  # the file line where the lemma begins
        text, sk, marked = head + cand.proof, None, 0
        cap = DRAFT_HOLES if draft else SKELETON_HOLES
        steps: list[str] = []  # the model's failed steps, for seeding the closers
        if draft:  # the draft's own `sorry`s are holes from the start
            got = hf.mark_sorries(text)
            if got is None:
                return {"outcome": "no_skeleton", "why": "a sorry the skeleton cannot place"}
            text, marked = got
        for _ in range(3):  # the rejected proof, then up to two rounds on its skeleton
            run = self.compile(above + text + "\n")
            if run["timed_out"]:
                return {"outcome": "no_skeleton", "why": "compile timed out"}
            errors = [(int((m.get("pos") or {}).get("line", 0)) - start, int((m.get("pos") or {}).get("column", 0)),
                       str(m.get("data", ""))) for m in run["messages"] if m.get("severity") == "error"]  # fmt: skip
            if not errors:
                break
            sk = hf.skeleton(text, errors, cap)
            if sk is None:
                return {"outcome": "no_skeleton", "why": "nothing to keep", "errors": [e[2][:120] for e in errors[:3]]}
            steps += sk.replaced
            text, n = hf.renumber_holes(sk.text)
            if n > cap:
                return {"outcome": "no_skeleton", "why": f"{n} failing steps"}
        else:
            return {"outcome": "no_skeleton", "why": "the skeleton does not elaborate"}
        if sk is None and marked:  # a clean draft: Lean's only complaint is its own `sorry`s
            text, n = hf.renumber_holes(text)
            sk = hf.Skeleton(text=text, holes=n, replaced=["sorry"] * n)
        if sk is None:
            return {"outcome": "no_skeleton", "why": "Lean reports no errors"}
        text, n = hf.renumber_holes(text)
        decl = text.split("\n", 1)[1]  # triage_block puts its own heartbeat line on each copy
        # Wayfinder's battery, with the model's own failed steps fed in before the heavy closers
        base = rh.closers(rh.defs_mentioned(hole.goal, self.job.get("defs") or []), "")
        heavy = [c for c in base if c in rh._HEAVY_CLOSERS]
        seeds = list(dict.fromkeys(c for st in steps for c in hf.seeded_closers(st)))[:6]
        battery = [[c for c in base if c not in heavy] + seeds + heavy] * n
        # heartbeats per copy above the model's own 400K, so its seeded step gets more than it had, but
        # bounded: the probe runs two copies a hole, and the model waits on it
        probe = rh.triage_block(decl, hole.name, battery, heartbeats=600000)
        run = self.compile(above + probe.text, timeout=900)
        out = {"outcome": "open", "holes": n, "skeleton": decl, "replaced": sk.replaced,
               "claims": hf.claims_at_holes(decl)}  # fmt: skip
        if run["timed_out"]:
            return {**out, "why": "the closers timed out"}
        fates = rh.triage_outcomes(run["messages"], probe, start)
        closed = [f["closed_by"] for f in fates]
        out.update(closed=[c for c in closed if c], fills=closed, text=text, opens=cand.opens, helpers=cand.helpers,
                   refuted={k + 1: f["refuted"] for k, f in enumerate(fates) if f["refuted"]})  # fmt: skip
        out["partial"] = hf.fill_holes(decl, closed, left="still open: the step that was here failed")
        if all(closed):
            out.update(self.assemble(hole, out, closed))
            return out
        still = [k for k in range(1, n + 1) if not closed[k - 1] and k not in out["refuted"]]
        if still and self.job.get("search", True):
            out["lifts"] = self.lift_steps(hole, decl, closed, still, above)
        return out

    def assemble(self, hole: Hole, res: dict, fills: list) -> dict:
        """The rejected proof's skeleton with every hole filled, through the one acceptance path."""
        proof = hf.fill_holes(res["text"], fills).split(f"{hole.statement} :=", 1)[1]
        answer = (f"open {' '.join(res['opens'])}\n\n" if res["opens"] else "") + "\n\n".join(
            res["helpers"] + [f"{hole.statement} :={proof}"]
        )
        check = self._check(hole, answer)
        return {"outcome": "verified" if check["outcome"] == "verified" else "open", "check": check}

    def lift_steps(self, hole: Hole, decl: str, closed: list, still: list[int], above: str) -> dict[int, str]:
        """The skeleton's steps still open, each as a standalone lemma (`extract_goal` there, the
        plain print if it elaborates back, else the typed one), for Wayfinder's search."""
        partial, m = hf.renumber_holes(hf.fill_holes(decl, closed))  # the open holes, renumbered in order
        run = self.compile(above + hf.lifting_probe(partial, hole.name))
        variants = hf.lifted_variants(run["messages"], hole.name)
        printed = {(j, i): st for j, pair in variants.items() for i, st in enumerate(pair) if st}
        if not printed or m != len(still):
            return {}
        block, ranges = hf.validation_block(printed)
        broken = hf.failing_ranges(self.compile(above + block)["messages"], ranges, above.count("\n") + 1)
        out: dict[int, str] = {}
        for j, k in enumerate(still, 1):
            usable = [(i, st) for (jj, i), st in sorted(printed.items()) if jj == j and (jj, i) not in broken]
            if usable:
                i, st = usable[0]
                out[k] = st if i == 0 else hf.untyped_name(st, hole.name, j)
        return out

    def search_steps(self, hole: Hole, lifts: dict[str, str]) -> dict[str, str]:
        """Wayfinder's deterministic search (Pantograph-checked) on lifted steps, one kernel session
        for the batch: name -> the tactic block that proves it."""
        if not lifts:
            return {}
        row = self.job.get("row") or {}
        rows = [{"theorem_id": name, "full_declaration": f"{st} := by\n  sorry", "goal_state": hf.closed_type(st) or "",
                 "file_path": row.get("file_path", ""), "imports": row.get("imports") or [],
                 "opens": row.get("opens") or []} for name, st in lifts.items()]  # fmt: skip
        req, res = self.dir / f"search-{hole.index}.json", self.dir / f"search-{hole.index}-out.json"
        timeout = min(900, 90 + 90 * len(rows))
        req.write_text(json.dumps({"rows": rows, "budget": SEARCH_BUDGET, "timeout": timeout, "out": str(res)}))
        try:
            subprocess.run([sys.executable, str(SEARCH_GOALS), str(req)], cwd=str(SEARCH_GOALS.parents[1]),
                           capture_output=True, text=True, timeout=timeout + 60, check=False)  # fmt: skip
            found = json.loads(res.read_text())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
        out = {}
        for name, r in found.items():
            if r.get("proved") and r.get("declaration") and ":= by" in r["declaration"]:
                out[name] = r["declaration"].split(":= by", 1)[1].strip("\n")
        return out

    def repair_round(self, hole: Hole, spec: ModelSpec, failures: list[dict], rnd: int
                     ) -> tuple[bool, list[tuple[int, list[dict], str | None]]]:  # fmt: skip
        """After each round (unless the job sets `repair` false): the best rejected proofs' skeletons, their
        holes to Wayfinder's closers. A proof completed this way is accepted like any answer. A
        skeleton with steps still open goes back to the model to complete — its plan, Wayfinder's
        closes in place, `sorry` where steps failed (Goedel's own "complete the code" form); a
        false claim found in one goes into that proof's revision prompt instead. Returns (solved,
        the completion attempts for the next round)."""
        seen: set[str] = set()
        completions: list[tuple[int, list[dict], str | None]] = []
        proofs = sorted([f for f in failures if not f.get("draft")], key=lambda f: f["n_errors"])
        drafts = sorted([f for f in failures if f.get("draft")], key=lambda f: f["n_errors"])  # fewest sorries
        picked = proofs[: int(self.job.get("fix_attempts") or 4)] + drafts[:2]
        # the skeletons are compiled side by side: the model waits on them
        with ThreadPoolExecutor(max_workers=max(1, len(picked))) as ex:
            repaired = list(ex.map(lambda f: self.skeleton_repair(hole, hf.extract_code(f["raw"]) or "",
                                                                  draft=bool(f.get("draft"))), picked))  # fmt: skip
        # the steps neither the closers nor a refutation settled: Wayfinder's search, one batch for all
        lifts = {f"{hole.name}_wf_s{i}_{k}": re.sub(r"^(theorem|lemma)\s+\S+", f"theorem {hole.name}_wf_s{i}_{k}", st,
                                                    count=1)  # fmt: skip
                 for i, res in enumerate(repaired) for k, st in (res.get("lifts") or {}).items()}
        searched = self.search_steps(hole, lifts) if lifts and not self.stop.is_set() else {}
        for i, res in enumerate(repaired):
            found = {k: searched[f"{hole.name}_wf_s{i}_{k}"] for k in (res.get("lifts") or {})
                     if f"{hole.name}_wf_s{i}_{k}" in searched}  # fmt: skip
            if found:
                fills = [found.get(k + 1) or c for k, c in enumerate(res["fills"])]
                res["searched"] = len(found)
                res["partial"] = hf.fill_holes(res["skeleton"], fills, left="still open: the step that was here failed")
                if all(fills):
                    res.update(self.assemble(hole, res, fills))
        for f, res in zip(picked, repaired):
            if self.stop.is_set():
                break
            if res.get("skeleton") in seen:
                continue
            seen.add(res.get("skeleton") or "")
            sid = f"h{hole.index}-repair-{f['id']}"
            gen = {"server": "wayfinder", "quant": "-", "model": "wayfinder closers on a skeleton of " + f["id"]}
            if res["outcome"] == "verified":
                check = res["check"]
                self.log_sample(hole, spec, sid, rnd, f["id"], gen, check)
                hole.solved_by = sid
                source = (f"{f['id']}'s plan, its {res['holes']} failing step(s) closed by Wayfinder "
                          f"({len(res['closed'])} by its closers, {res.get('searched', 0)} by its search), "
                          f"run {self.job['run_id']}")  # fmt: skip
                with self.lock:
                    self.fills.append({"hole": hole.index, "model": "wayfinder", "quant": "-", "sample": sid,
                                       "round": rnd, "form": check["form"], "fill": check["fill"],
                                       "helpers": check["helpers"], "source": source})  # fmt: skip
                    self._write("fills.json", self.fills)
                return True, []
            self.log_sample(hole, spec, sid, rnd, f["id"], gen, {"outcome": f"repair_{res['outcome']}",
                                                                  "reasons": [res.get("why") or ""]})  # fmt: skip
            if res.get("partial") and not res.get("refuted"):
                ask = [{"role": "user", "content": ROUND0.format(statement=self.prefix + res["partial"])}]
                completions += [(rnd + 1, ask, sid)] * int(self.job.get("fix_samples") or 2)
            false = [f"`{res['claims'].get(k, f'step {k}')}` is false: {' '.join(str(c).split())[:300]}"
                     for k, c in (res.get("refuted") or {}).items()]  # fmt: skip
            if false:  # a refuted step is a false claim of the model's: say so in its revision
                note = "Checking this proof's steps one by one found these claims false:"
                f["report"] += "\n\n" + note + "\n" + "\n".join(false)
        return False, completions

    def in_place(self, hole: Hole, cand: hf.Candidate, helpers: list[str], lemma: str) -> dict:
        """The proof that passed standalone, put into the hand-off: inline behind `rename_i`, else
        as a helper lemma the hole applies. The first form that keeps the file sound wins."""
        as_helper = lemma if hole.helper_name == hole.name else _renamed(lemma, hole.name, hole.helper_name)
        forms: list[tuple[str, str, list[str]]] = []
        if hole.rename is not None:
            rename = [hole.rename] if hole.rename else []
            forms.append(("inline", "\n".join(rename + cand.tactic_lines()), helpers))
            forms.append(("helper", "\n".join(rename + [hf.helper_call(hole.helper_name, hole.binders)]),
                          helpers + [as_helper]))  # fmt: skip
        else:
            forms.append(("helper_apply", f"apply {hole.helper_name} <;> assumption", helpers + [as_helper]))
        left = len(self.job["holes"]) - 1
        reasons: list[str] = []
        for form, fill, extra in forms:
            new_file, _ = hf.apply_fill(self.job["file"], self.job["declaration"], hole.index, fill, "\n\n".join(extra))
            run = self.compile(new_file + gate.probe_suffix(self.target))
            if run["timed_out"]:
                reasons.append(f"{form}: compile timed out")
                continue
            why = hf.partial_findings(run["messages"], self.target, self.job.get("pp_type"),
                                      tuple(self.job.get("allowed_axioms") or ()), left)  # fmt: skip
            if not why:
                return {"outcome": "verified", "form": form, "fill": fill, "helpers": "\n\n".join(extra)}
            reasons.append(f"{form}: {why[0]}")
        return {"outcome": "in_place_failed", "reasons": reasons}

    # ── sampling ──

    def seed(self, sample_id: str) -> int:
        return int(hashlib.sha256(f"{self.job['run_id']}:{sample_id}".encode()).hexdigest()[:8], 16)

    def generate(self, pool: Pool, messages: list[dict], sample_id: str, cancel: threading.Event) -> dict:
        def halted() -> bool:
            return cancel.is_set() or self.stop.is_set()

        if halted():
            return {"outcome": self.halt_outcome()}
        prompt = chatml(messages)
        tried: set[int] = set()
        while True:
            srv = pool.acquire(exclude=tried)
            if srv is None:
                return {"outcome": "server_error", "error": "no server left in the pool"}
            try:
                n_prompt = pool.local.count_tokens(prompt)  # one tokenizer for every quant of the model
                budget = min(int(self.job["max_new_tokens"]), srv.ctx - n_prompt - 64)
                if budget < 1024:
                    return {"outcome": "context_full", "prompt_tokens": n_prompt}
                start = time.monotonic()
                out = srv.complete(prompt, budget, self.seed(sample_id), halted)
                break
            except (OSError, http.client.HTTPException, RuntimeError, ValueError) as e:
                if srv is pool.local:
                    return {"outcome": "server_error", "error": f"{type(e).__name__}: {e}"[:300]}
                pool.drop(srv)  # a remote that fails is out for the rest of this run; the attempt goes elsewhere
                tried.add(id(srv))
                self.note(f"remote {srv.name} dropped: {type(e).__name__}: {e}"[:200])
            finally:
                pool.release(srv)
        out.update(prompt_tokens=n_prompt, budget=budget, wall_s=round(time.monotonic() - start, 1),
                   server=srv.name, quant=srv.quant)  # fmt: skip
        (self.dir / "outputs" / f"{sample_id}.txt").write_text(out["content"])
        if out.pop("cancelled"):
            out["outcome"] = self.halt_outcome()
        else:
            out["outcome"] = "generated"
        return out

    def halt_outcome(self) -> str:
        """Why a sample stopped short: the run was stopped (asked, or for memory), or its hole was solved."""
        if self.stop.is_set():
            return "memory_yield" if self.stop_kind == "memory" else "stopped"
        return "cancelled"

    def log_sample(
        self, hole: Hole, spec: ModelSpec, sid: str, rnd: int, parent: str | None, gen: dict, res: dict
    ) -> None:
        outcome = res.get("outcome") or gen.get("outcome")
        rec = {"id": sid, "hole": hole.index, "model": spec.name, "quant": gen.get("quant") or spec.quant,
               "server": gen.get("server"), "round": rnd, "parent": parent,
               "outcome": outcome, "prompt_tokens": gen.get("prompt_tokens"), "budget": gen.get("budget"),
               "tokens": gen.get("tokens_predicted"),
               "stop_type": gen.get("stop_type"), "wall_s": gen.get("wall_s"), "tok_per_s": gen.get("tok_per_s"),
               "form": res.get("form"), "rewrites": res.get("rewrites"), "repaired": res.get("repaired"),
               "errors": res.get("errors") or res.get("reasons") or gen.get("error")}  # fmt: skip
        with self.lock:
            hole.outcomes[outcome] += 1
            with open(self.dir / "samples.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            m = self.models.setdefault(spec.name, {"samples": 0, "tokens": 0, "gen_seconds": 0.0})
            if gen.get("tokens_predicted"):
                m["samples"] += 1
                m["tokens"] += int(gen["tokens_predicted"])
                m["gen_seconds"] = round(m["gen_seconds"] + float(gen.get("gen_ms") or 0) / 1000, 1)
                m["tok_per_s"] = round(m["tokens"] / max(m["gen_seconds"], 1e-9), 2)
                blank = {"samples": 0, "tokens": 0, "gen_seconds": 0.0}
                per = m.setdefault("by_server", {}).setdefault(gen.get("server") or "local", blank)
                per["samples"] += 1
                per["tokens"] += int(gen["tokens_predicted"])
                per["gen_seconds"] = round(per["gen_seconds"] + float(gen.get("gen_ms") or 0) / 1000, 1)
        self.publish()

    def run_round(self, hole: Hole, pool: Pool, attempts: list[tuple[int, list[dict], str | None]],
                  failures: list[dict], seeds: list[tuple[str, list[dict], dict]] = ()) -> bool:  # fmt: skip
        """Sample every attempt (as many at once as the pool has slots), check each answer as it
        lands, and stop the rest the moment one is verified."""
        cancel = threading.Event()
        spec = pool.local.spec
        gen_pool = ThreadPoolExecutor(max_workers=max(1, pool.slots))
        ver_pool = ThreadPoolExecutor(max_workers=2)
        jobs: dict = {}
        for rnd, messages, parent in attempts:
            self.counter[(hole.index, spec.name)] += 1
            n = self.counter[(hole.index, spec.name)]
            sid = f"h{hole.index}-{spec.name.removeprefix('goedel-v2-')}-r{rnd}-s{n}"
            fut = gen_pool.submit(self.generate, pool, messages, sid, cancel)
            jobs[fut] = ("gen", sid, rnd, messages, parent, None)
        for seed_id, messages, gen in seeds:
            sid = f"h{hole.index}-seed-{seed_id}"
            (self.dir / "outputs" / f"{sid}.txt").write_text(gen["content"])  # a later tier may revise it in turn
            jobs[gen_pool.submit(dict, gen)] = ("gen", sid, 0, messages, None, None)
        pending = set(jobs)
        solved = False
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if self.stop.is_set():
                cancel.set()
            for fut in done:
                kind, sid, rnd, messages, parent, gen = jobs.pop(fut)
                if kind == "gen":
                    gen = fut.result()
                    if gen["outcome"] != "generated" or solved:
                        # an answer that lands after the hole is solved is not checked
                        superseded = solved and gen["outcome"] == "generated"
                        res = {"outcome": "superseded"} if superseded else {}
                        self.log_sample(hole, spec, sid, rnd, parent, gen, res)
                        continue
                    ver = ver_pool.submit(self.verify, hole, gen["content"], gen.get("stop_type"))
                    jobs[ver] = ("ver", sid, rnd, messages, parent, gen)
                    pending.add(ver)
                    continue
                res = fut.result()
                self.log_sample(hole, spec, sid, rnd, parent, gen, res)
                if res["outcome"] == "verified" and not solved:
                    solved = True
                    cancel.set()
                    hole.solved_by = sid
                    quant, where = gen.get("quant") or spec.quant, gen.get("server") or "local"
                    host = "local llama.cpp" if where == "local" else f"llama.cpp on {where}"
                    author = gen.get("model") or spec.name  # a revised-from answer keeps its own model
                    source = f"{author} ({quant}, {host}), sample {sid}, run {self.job['run_id']}"
                    with self.lock:
                        self.fills.append({"hole": hole.index, "model": author, "quant": quant, "sample": sid,
                                           "round": rnd, "form": res["form"], "fill": res["fill"],
                                           "helpers": res["helpers"], "source": source})  # fmt: skip
                        self._write("fills.json", self.fills)
                elif res.get("draft") or res.get("cut"):  # an unfinished plan: repaired, never revised
                    raw = gen["content"] if res.get("draft") else f"```lean4\n{res['cut']}\n```"
                    failures.append({"id": sid, "messages": messages, "raw": raw, "report": "",
                                     "n_errors": raw.count("sorry") if res.get("draft") else 50,
                                     "draft": "sorry" if res.get("draft") else "cut"})  # fmt: skip
                elif res["outcome"] == "wrong_proof":
                    failures.append({"id": sid, "messages": messages, "raw": gen["content"], "report": res["report"],
                                     "n_errors": res.get("n_errors", 0)})  # fmt: skip
        gen_pool.shutdown(wait=True)
        ver_pool.shutdown(wait=True)
        return solved

    def solve(self, hole: Hole, pool: Pool) -> bool:
        """Round 0 (`samples` attempts), then self-correction rounds on the best failures."""
        self.publish(state="sampling", hole=hole.index, model=pool.local.spec.name)
        base = [{"role": "user", "content": ROUND0.format(statement=self.formal(hole))}]
        attempts: list[tuple[int, list[dict], str | None]] = [(0, base, None)] * int(self.job["samples"])
        seeds = self.seeded(hole, base)
        failures: list[dict] = []
        completions: list[tuple[int, list[dict], str | None]] = []
        for rnd in range(int(self.job["correction_rounds"]) + 1):
            if rnd > 0:
                proofs = [f for f in failures if not f.get("draft")]  # a draft has no Lean report to revise from
                fixable = sorted(proofs, key=lambda f: f["n_errors"])[: int(self.job["fix_attempts"])]
                attempts = [
                    (rnd, _corrected(f, rnd), f["id"]) for f in fixable for _ in range(int(self.job["fix_samples"]))
                ] + completions
                seeds, completions = [], []
            if not attempts and not seeds:
                return False
            failures = []
            if self.run_round(hole, pool, attempts, failures, seeds):
                return True
            if self.job.get("repair", True) and failures:  # on unless a protocol turns it off
                try:
                    solved, completions = self.repair_round(hole, pool.local.spec, failures, rnd)
                except Exception as e:  # noqa: BLE001 — the repair loop never costs the theorem its rounds
                    self.note(f"repair round failed: {type(e).__name__}: {e}"[:300])
                    solved, completions = False, []
                if solved:
                    return True
            if self.stop.is_set():
                return False
        if completions and not self.stop.is_set():  # the last round's skeletons still get completed
            return self.run_round(hole, pool, completions, [], ())
        return False

    def seeded(self, hole: Hole, base: list[dict]) -> list[tuple[str, list[dict], dict]]:
        """The answers an earlier tier gave this hole and Lean rejected, to revise rather than redo
        (Goedel's self-correction starts from a failed proof and its errors). Only answers to the
        same statement: none when the earlier tier lifted the hole differently."""
        seed = self.job.get("seed") or {}
        if not seed:
            return []
        arm = Path(seed["arm"])
        try:
            prior = json.loads((arm / "status.json").read_text())
        except (OSError, ValueError):
            return []
        was = (prior.get("holes") or {}).get(str(hole.index)) or {}
        if "statement" in was:
            same = was["statement"] == hole.statement
        else:  # an arm from before statements were recorded: the lift changed only if one fell back to types
            note = f"hole {hole.index}: the plain print does not fit"
            same = hole.retyped == any(n.startswith(note) for n in prior.get("notes") or [])
        if not same:
            self.note(f"hole {hole.index}: the earlier tier answered another statement; no answers to revise")
            return []
        out = []
        for s in seed.get("samples") or []:
            f = arm / "outputs" / f"{s['id']}.txt"
            if str(s["id"]).startswith(f"h{hole.index}-") and f.exists():
                gen = {"outcome": "generated", "content": f.read_text(), "stop_type": s.get("stop_type") or "eos",
                       "quant": s.get("quant"), "server": s.get("server") or "local",
                       "model": s.get("model")}  # fmt: skip
                out.append((str(s["id"]), base, gen))
        return out

    # ── serving a model ──

    def watch(self, srv: LlamaServer) -> None:
        """Yield the machine: stop when memory pressure holds at warn or worse for 30 s."""
        strikes = 0
        while not srv.stopping and not self.stop.is_set():
            if not srv.alive():
                self.stop_reason, self.stop_kind = "llama-server exited unexpectedly", "server"
                self.stop.set()
                return
            level = pressure_level()
            strikes = strikes + 1 if level >= 2 else 0
            if strikes >= 3:
                self.stop_reason = f"memory pressure (level {level}) held for 30 s: stopped to give the RAM back"
                self.stop_kind = "memory"
                self.stop.set()
                return
            time.sleep(10)

    def remotes(self, model: str, spec: ModelSpec, ctx: int) -> list[LlamaServer]:
        """The pool file's servers for this model that answer now."""
        out: list[LlamaServer] = []
        # a job may name its own pool-file key, so a slow machine can serve a stream of its own
        for r in pool_config().get(self.job.get("pool_key") or model) or []:
            remote = RemoteServer(spec, str(r["name"]), int(r["port"]), int(r.get("slots", 1)),
                                  int(r.get("ctx", ctx)), str(r.get("quant", "?")))  # fmt: skip
            try:
                remote.start()
                out.append(remote)
            except (OSError, http.client.HTTPException, RuntimeError, ValueError) as e:
                self.note(f"remote {remote.name} not answering, left out: {type(e).__name__}: {e}"[:200])
        return out

    def serve_remote_only(self, model: str, holes: list[Hole]) -> None:
        """No local server: the pool is the other machines' servers (the first counts tokens)."""
        spec = MODELS[model]
        servers = self.remotes(model, spec, int(self.job.get("ctx") or spec.ctx))
        if not servers:
            for h in holes:
                h.note = f"{model}: remote-only run, and no remote server answered"
            return
        pool = Pool(servers)
        listed = [{"name": s.name, "slots": s.slots, "quant": s.quant, "ctx": s.ctx} for s in servers]
        self.models.setdefault(model, {"samples": 0, "tokens": 0, "gen_seconds": 0.0}).update(
            remote_only=True, servers=listed
        )
        self.publish(state="sampling", model=model)
        for h in holes:
            if self.stop.is_set():
                break
            h_start = time.monotonic()
            if self.solve(h, pool):
                h.note = f"verified after {time.monotonic() - h_start:.0f}s on {model} (remote)"

    def serve(self, model: str, holes: list[Hole]) -> None:
        if self.job.get("remote_only"):
            return self.serve_remote_only(model, holes)
        spec = MODELS[model]
        path = Path(self.job["gguf_dir"]) / spec.file
        if not path.exists():
            for h in holes:
                h.note = f"{model}: model file not downloaded ({path})"
            return
        avail = available_bytes()
        ctx = int(self.job.get("ctx") or spec.ctx)
        slots = plan_slots(spec, path.stat().st_size, avail, int(float(self.job["reserve_gb"]) * GiB),
                           int(self.job["max_slots"]), ctx=ctx)  # fmt: skip
        need = path.stat().st_size + ctx * spec.kv_bytes_per_token() + int(1.5 * GiB)
        if slots < 1:
            for h in holes:
                h.note = (f"{model}: not enough free memory (needs {need / GiB:.1f} GB plus a "
                          f"{self.job['reserve_gb']} GB reserve; {avail / GiB:.1f} GB free)")  # fmt: skip
            return
        srv = LlamaServer(self.job["llama_server"], path, spec, slots, self.dir / f"server-{model}.log", ctx=ctx)
        self.models.setdefault(model, {"samples": 0, "tokens": 0, "gen_seconds": 0.0}).update(
            slots=slots, ctx_per_slot=ctx, free_gb_at_start=round(avail / GiB, 1), quant=spec.quant
        )
        self.publish(state="loading", model=model)
        start = time.monotonic()
        srv.start()
        self.models[model]["load_seconds"] = round(time.monotonic() - start, 1)
        watcher = threading.Thread(target=self.watch, args=(srv,), daemon=True)
        watcher.start()
        servers: list[LlamaServer] = [srv, *self.remotes(model, spec, ctx)]
        pool = Pool(servers)
        self.models[model]["servers"] = [{"name": s.name, "slots": s.slots, "quant": s.quant, "ctx": s.ctx}
                                         for s in servers]  # fmt: skip
        try:
            for h in holes:
                if self.stop.is_set():
                    break
                h_start = time.monotonic()
                if self.solve(h, pool):
                    h.note = f"verified after {time.monotonic() - h_start:.0f}s on {model}"
        finally:
            srv.stop()

    def run(self) -> None:
        state = "failed"
        try:
            self.prepare()
            self.publish(state="routed")
            for model in (LIGHT, HEAVY):
                todo = [h for h in self.holes if h.plan and h.plan[0] == model and not h.solved_by and h.statement]
                if not todo or self.stop.is_set():
                    continue
                self.serve(model, todo)
                for h in todo:
                    if not h.solved_by and h.plan and h.plan[0] == model:
                        h.plan = h.plan[1:]  # on to the next model, if the plan has one
            ended = {"requested": "stopped", "memory": "yielded", "server": "failed"}
            state = ended.get(self.stop_kind, "stopped") if self.stop.is_set() else "done"
        except Exception:  # noqa: BLE001 — the run's record must say how it ended
            self.note(traceback.format_exc()[-2000:])
        finally:
            if getattr(self, "repl_pool", None) is not None:
                self.repl_pool.stop()
            self.publish(state=state, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                         verified=[f["hole"] for f in self.fills])  # fmt: skip


def _corrected(failure: dict, rnd: int) -> list[dict]:
    """The conversation for correction round `rnd`: the failed answer, then Lean's errors."""
    feedback = CORRECTION.format(previous=rnd - 1, errors=failure["report"])
    return failure["messages"] + [
        {"role": "assistant", "content": failure["raw"]},
        {"role": "user", "content": feedback},
    ]


def _renamed(declaration: str, name: str, new_name: str) -> str:
    head = re.compile(r"\b(theorem|lemma)\s+" + re.escape(name) + r"(?=[\s:({\[⦃.])")
    return head.sub(r"\1 " + new_name, declaration, count=1)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m src.local_prover <run_dir>/job.json", file=sys.stderr)
        return 2
    worker = Worker(Path(argv[0]))

    def stop(signum: int, _frame) -> None:
        # a stopped run still shuts its llama-server down and records how it ended
        worker.stop_reason, worker.stop_kind = f"stopped by request (signal {signum})", "requested"
        worker.stop.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
