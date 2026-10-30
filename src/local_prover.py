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
import src.proof_gate as gate
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
                               "outcomes": dict(h.outcomes)}  # fmt: skip
                for h in self.holes
            }
            self.status["models"] = self.models
            self.status["stop_reason"] = self.stop_reason
            self._write("status.json", self.status)

    def note(self, text: str) -> None:
        with self.lock:
            self.status["notes"].append(text)

    def compile(self, text: str, timeout: int = 300) -> dict:
        return gate.run_lean_json(self.job["lake_bin"], self.job["verify_project"], text, timeout=timeout)

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
        probe = self.prefix + hf.lifting_probe(self.job["declaration"], self.target)
        lifted = hf.lifted_from_messages(self.compile(probe)["messages"], self.target)
        block, ranges = hf.validation_block(lifted)
        first = self.prefix.count("\n") + 1
        broken = hf.failing_ranges(self.compile(self.prefix + block)["messages"], ranges, first) if lifted else {}
        for hole in self.holes:
            statement = lifted.get(hole.index)
            if statement is None:
                hole.note, hole.plan = "extract_goal printed no statement for this hole", []
            elif hole.index in broken:
                hole.note, hole.plan = f"the lifted statement does not re-elaborate: {broken[hole.index]}", []
            else:
                hole.statement = statement
                hole.name = hole.helper_name = hf.lifted_name(self.target, hole.index)
                hole.binders = hf.binders(statement)
                hole.rename = hf.rename_line(hf.context_names(hole.goal), hole.binders)

    def formal(self, hole: Hole) -> str:
        return f"{self.prefix}{hole.statement} := by sorry"

    # ── checking one answer ──

    def verify(self, hole: Hole, raw: str, stop_type: str | None) -> dict:
        code = hf.extract_code(raw)
        if code is None:
            return {"outcome": "truncated" if stop_type == "limit" else "no_code"}
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
        if bad:
            return {"outcome": "denied", "reasons": bad}
        lemma = f"set_option maxHeartbeats {HEARTBEATS} in\n" + cand.scoped(f"{hole.statement} :={cand.proof}")
        standalone = self.prefix + "\n\n".join(helpers + [lemma]) + "\n" + gate.probe_suffix(hole.name)
        run = self.compile(standalone)
        if run["timed_out"]:
            return {"outcome": "compile_timeout"}
        msgs = run["messages"]
        errors = gate.errors_in(msgs)
        axioms = gate.axioms_from_messages(msgs, hole.name)
        if errors or gate.uses_sorry(msgs) or axioms is None or not set(axioms) <= self.allowed:
            why = errors[:3] or [f"axioms: {axioms}" if axioms is not None else "no `#print axioms` report"]
            return {"outcome": "wrong_proof", "errors": why, "n_errors": len(errors),
                    "report": hf.error_report(standalone, msgs) or "\n".join(why)}  # fmt: skip
        return self.in_place(hole, cand, helpers, lemma)

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

    def generate(self, srv: LlamaServer, messages: list[dict], sample_id: str, cancel: threading.Event) -> dict:
        def halted() -> bool:
            return cancel.is_set() or self.stop.is_set()

        if halted():
            return {"outcome": self.halt_outcome()}
        prompt = chatml(messages)
        try:
            n_prompt = srv.count_tokens(prompt)
            budget = min(int(self.job["max_new_tokens"]), srv.ctx - n_prompt - 64)
            if budget < 1024:
                return {"outcome": "context_full", "prompt_tokens": n_prompt}
            start = time.monotonic()
            out = srv.complete(prompt, budget, self.seed(sample_id), halted)
        except (OSError, http.client.HTTPException, RuntimeError, ValueError) as e:
            return {"outcome": "server_error", "error": f"{type(e).__name__}: {e}"[:300]}
        out.update(prompt_tokens=n_prompt, budget=budget, wall_s=round(time.monotonic() - start, 1))
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
        rec = {"id": sid, "hole": hole.index, "model": spec.name, "quant": spec.quant, "round": rnd, "parent": parent,
               "outcome": outcome, "prompt_tokens": gen.get("prompt_tokens"), "budget": gen.get("budget"),
               "tokens": gen.get("tokens_predicted"),
               "stop_type": gen.get("stop_type"), "wall_s": gen.get("wall_s"), "tok_per_s": gen.get("tok_per_s"),
               "form": res.get("form"), "rewrites": res.get("rewrites"),
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
        self.publish()

    def run_round(self, hole: Hole, srv: LlamaServer, attempts: list[tuple[int, list[dict], str | None]],
                  failures: list[dict]) -> bool:  # fmt: skip
        """Sample every attempt (as many at once as the server has slots), check each answer as it
        lands, and stop the rest the moment one is verified."""
        cancel = threading.Event()
        spec = srv.spec
        gen_pool = ThreadPoolExecutor(max_workers=srv.slots)
        ver_pool = ThreadPoolExecutor(max_workers=2)
        jobs: dict = {}
        for rnd, messages, parent in attempts:
            self.counter[(hole.index, spec.name)] += 1
            n = self.counter[(hole.index, spec.name)]
            sid = f"h{hole.index}-{spec.name.removeprefix('goedel-v2-')}-r{rnd}-s{n}"
            jobs[gen_pool.submit(self.generate, srv, messages, sid, cancel)] = ("gen", sid, rnd, messages, parent, None)
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
                    source = f"{spec.name} ({spec.quant}, local llama.cpp), sample {sid}, run {self.job['run_id']}"
                    with self.lock:
                        self.fills.append({"hole": hole.index, "model": spec.name, "quant": spec.quant, "sample": sid,
                                           "round": rnd, "form": res["form"], "fill": res["fill"],
                                           "helpers": res["helpers"], "source": source})  # fmt: skip
                        self._write("fills.json", self.fills)
                elif res["outcome"] == "wrong_proof":
                    failures.append({"id": sid, "messages": messages, "raw": gen["content"], "report": res["report"],
                                     "n_errors": res.get("n_errors", 0)})  # fmt: skip
        gen_pool.shutdown(wait=True)
        ver_pool.shutdown(wait=True)
        return solved

    def solve(self, hole: Hole, srv: LlamaServer) -> bool:
        """Round 0 (`samples` attempts), then self-correction rounds on the best failures."""
        self.publish(state="sampling", hole=hole.index, model=srv.spec.name)
        base = [{"role": "user", "content": ROUND0.format(statement=self.formal(hole))}]
        attempts: list[tuple[int, list[dict], str | None]] = [(0, base, None)] * int(self.job["samples"])
        failures: list[dict] = []
        for rnd in range(int(self.job["correction_rounds"]) + 1):
            if rnd > 0:
                fixable = sorted(failures, key=lambda f: f["n_errors"])[: int(self.job["fix_attempts"])]
                attempts = [
                    (rnd, _corrected(f, rnd), f["id"]) for f in fixable for _ in range(int(self.job["fix_samples"]))
                ]
            if not attempts:
                return False
            failures = []
            if self.run_round(hole, srv, attempts, failures):
                return True
            if self.stop.is_set():
                return False
        return False

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

    def serve(self, model: str, holes: list[Hole]) -> None:
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
        try:
            for h in holes:
                if self.stop.is_set():
                    break
                h_start = time.monotonic()
                if self.solve(h, srv):
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
