"""The hand-off: Wayfinder's residual to Goedel, Goedel's answers to Lean, what Lean accepts back to
Wayfinder (docs/LOCAL_HANDOFF_PIPELINE.md §4).

1. Goedel-Prover-V2 sees the residual — the theorem, Wayfinder's validated partial proof, Lean's goal
   under each hole — in its own "complete the following Lean 4 code" form; a few samples see the
   bare statement, a path that does not start from Wayfinder's prefix.
2. A complete answer goes through the acceptance path. Every other answer is taken apart by Lean
   (`Worker.skeleton_repair`): the steps it accepts stay, each failing step becomes a hole, a `have`
   whose proof fails keeps its claim with a hole under it; Wayfinder's closers and `plausible` run on
   each hole, and a refuted claim drops its skeleton.
3. Wayfinder searches each remaining hole where it sits (scripts/search_holes.py); what it builds is
   spliced in and the merged proof recompiled.
4. The best partial proof, if it holds validated progress, is the next round's residual. Goedel is
   asked a fixed small number of times and never revises its own answers.

The job is the theorem's whole-statement job (its acceptance path, its gate) plus the residual:
`residual_file` and `residual_declaration`, as `prove` left them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import src.hole_fill as hf
import src.hole_search as hs
import src.proof_gate as gate

SEARCH_HOLES = Path(__file__).resolve().parents[1] / "scripts" / "search_holes.py"
ROUNDS = ((6, 2), (4, 0))  # per round: samples on the residual, samples on the bare statement
BUDGET_S = 900  # the whole hand-off, model loading included; `budget_s` in the job overrides it
HOLE_BUDGET = 200  # Wayfinder's tactic attempts per hole
KEEP = 2  # skeletons searched per round (with the residual itself kept as the fallback)


@dataclass
class Partial:
    """A partial proof of the theorem: its declaration (holes marked, Lean's goal under each), what
    goes above it in the file, and where it came from."""

    declaration: str
    above: str
    source: str

    @property
    def holes(self) -> int:
        return hs.open_holes(self.declaration)

    @property
    def file(self) -> str:
        return self.above + self.declaration + "\n"


class Handoff:
    def __init__(self, worker) -> None:
        self.w = worker
        self.hole = worker.holes[0]  # the theorem itself: its statement, its name
        self.started = time.monotonic()
        self.budget = float(worker.job.get("budget_s") or BUDGET_S)
        self.counter = 0
        decl = str(worker.job["residual_declaration"])
        rfile = str(worker.job["residual_file"])
        self.residual = Partial(decl, rfile[: rfile.rindex(decl)], "wayfinder")
        self.best = self.residual

    def left(self) -> float:
        return self.budget - (time.monotonic() - self.started)

    # ── the rounds ──

    def run(self, pool) -> bool:
        from src.local_prover import ROUND0  # the worker's module: one prompt template, one transport

        for rnd, (on_residual, on_bare) in enumerate(ROUNDS):
            if self.w.stop.is_set() or self.left() <= 0:
                break
            start_of_round = self.best
            residual_ask = [{"role": "user", "content": ROUND0.format(statement=self.best.file)}]
            bare_ask = [{"role": "user", "content": ROUND0.format(statement=self.w.formal(self.hole) + "\n")}]
            asks = [(rnd, residual_ask, "residual")] * on_residual + [(rnd, bare_ask, "bare")] * on_bare
            self.w.publish(state=f"round {rnd}: sampling", best_holes=self.best.holes)
            closed, answers = self.sample(pool, asks)
            if closed:
                return True
            self.w.publish(state=f"round {rnd}: taking answers apart")
            closed, parts = self.harvest(answers)
            if closed:
                return True
            parts.sort(key=lambda p: p.holes)
            for part in [*parts[:KEEP], *([self.best] if rnd == 0 else [])]:
                if self.w.stop.is_set() or self.left() <= 0:
                    break
                self.w.publish(state=f"round {rnd}: Wayfinder on {part.holes} hole(s) of {part.source}")
                merged = self.search_in_place(part)
                if merged.holes == 0 and self.accept(merged, rnd):
                    return True
                if merged.holes < self.best.holes:
                    self.best = merged
            if self.best is start_of_round:  # no validated progress this round: Goedel is not asked again
                break
            self.best = self.annotate(self.best)
            self.write_best()
        self.write_best()
        return False

    # ── Goedel's answers ──

    def sample(self, pool, asks: list[tuple[int, list[dict], str]]) -> tuple[bool, list[tuple[str, dict, dict]]]:
        """Every ask sampled (as many at once as the pool has slots), each answer checked as it lands;
        a verified one stops the rest. Returns (closed, the other answers with their checks)."""
        cancel = threading.Event()
        spec = pool.local.spec
        answers: list[tuple[str, dict, dict]] = []
        closed = False
        with ThreadPoolExecutor(max_workers=max(1, pool.slots)) as gen_pool, ThreadPoolExecutor(max_workers=2) as ver:
            jobs: dict = {}
            for rnd, messages, kind in asks:
                self.counter += 1
                sid = f"t-{kind}-r{rnd}-s{self.counter}"
                jobs[gen_pool.submit(self.w.generate, pool, messages, sid, cancel)] = ("gen", sid, rnd, None)
            pending = set(jobs)
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                if self.w.stop.is_set() or self.left() <= 0:
                    cancel.set()
                for fut in done:
                    kind, sid, rnd, gen = jobs.pop(fut)
                    if kind == "gen":
                        gen = fut.result()
                        if gen["outcome"] != "generated" or closed:
                            self.w.log_sample(self.hole, spec, sid, rnd, None, gen,
                                              {"outcome": "superseded"} if closed else {})  # fmt: skip
                            continue
                        f = ver.submit(self.w.verify, self.hole, gen["content"], gen.get("stop_type"))
                        jobs[f] = ("ver", sid, rnd, gen)
                        pending.add(f)
                        continue
                    res = fut.result()
                    self.w.log_sample(self.hole, spec, sid, rnd, None, gen, res)
                    if res["outcome"] == "verified" and not closed:
                        closed = True
                        cancel.set()
                        self.record(sid, rnd, res, {**gen, "model": gen.get("model") or spec.name},
                                    f"{gen.get('model') or spec.name} answer {sid}")  # fmt: skip
                    else:
                        answers.append((sid, gen, res))
        return closed, answers

    def harvest(self, answers: list[tuple[str, dict, dict]]) -> tuple[bool, list[Partial]]:
        """Each rejected answer or unfinished draft taken apart by Lean, its holes to Wayfinder's
        closers and `plausible`; a skeleton with a refuted claim is dropped. Returns (closed — the
        closers finished one —, the partial proofs)."""
        picked = [(sid, gen, res) for sid, gen, res in answers
                  if res.get("outcome") == "wrong_proof" or res.get("draft") or res.get("cut")]  # fmt: skip
        out: list[Partial] = []
        seen: set[str] = set()

        def one(item):
            sid, gen, res = item
            code = res.get("cut") or hf.extract_code(gen["content"]) or ""
            return sid, self.w.skeleton_repair(self.hole, code, draft=bool(res.get("draft") or res.get("cut")))

        with ThreadPoolExecutor(max_workers=max(1, min(len(picked), 4))) as ex:
            repaired = list(ex.map(one, picked))
        for sid, res in repaired:
            if res.get("outcome") == "verified":  # every hole closed by the closers, accepted as any answer
                how = f"{sid}'s plan, its holes closed by Wayfinder"
                self.record(f"{sid}-closers", 0, res["check"], {"model": "wayfinder"}, how)
                return True, []
            if res.get("outcome") != "open" or res.get("refuted") or res.get("skeleton") in seen:
                continue
            seen.add(res["skeleton"])
            decl = hf.renumber_holes(hf.fill_holes(res["skeleton"], res["fills"]))[0]
            opens = f"open {' '.join(res['opens'])}\n\n" if res.get("opens") else ""
            above = self.w.prefix + opens + "\n\n".join([*(res.get("helpers") or []), ""])
            out.append(Partial(decl, above, sid))
        return False, out

    # ── Wayfinder again ──

    def search_in_place(self, part: Partial) -> Partial:
        """Wayfinder's search on each hole of `part` where it sits; every block that keeps the file
        sound is spliced in."""
        holes = sorted(hf.hole_sites(part.declaration))
        if not holes:
            return part
        req = self.w.dir / f"holes-{part.source}.json"
        out = self.w.dir / f"holes-{part.source}-out.json"
        timeout = int(max(60, min(self.left(), 120 + 90 * len(holes))))
        req.write_text(json.dumps({"file": part.file, "declaration": part.declaration, "holes": holes,
                                   "row": self.w.job["row"], "budget": HOLE_BUDGET, "timeout": timeout,
                                   "out": str(out)}))  # fmt: skip
        try:
            subprocess.run([sys.executable, str(SEARCH_HOLES), str(req)], cwd=str(SEARCH_HOLES.parents[1]),
                           capture_output=True, text=True, timeout=timeout + 60, check=False)  # fmt: skip
            found = json.loads(out.read_text())
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            self.w.note(f"hole search on {part.source} failed: {type(e).__name__}: {e}"[:300])
            return part
        blocks = {k: found.get(str(k), {}).get("block") for k in holes}
        merged = self.splice(part, blocks)
        if self.sound(merged):
            return merged
        kept = {}  # one block breaks the file: keep each that is sound on its own
        for k, block in blocks.items():
            if block and self.sound(self.splice(part, {k: block})):
                kept[k] = block
        return self.splice(part, kept)

    @staticmethod
    def splice(part: Partial, blocks: dict[int, str | None]) -> Partial:
        fills = [blocks.get(k) for k in sorted(hf.hole_sites(part.declaration))]
        decl = hf.renumber_holes(hf.fill_holes(part.declaration, fills))[0]
        return Partial(decl, part.above, part.source + "+wayfinder")

    def sound(self, part: Partial) -> bool:
        """Whether the file elaborates with no error but its holes' `sorry`s."""
        run = self.w.compile(part.file, timeout=600)
        return not run["timed_out"] and not gate.errors_in(run["messages"])

    def annotate(self, part: Partial) -> Partial:
        """Lean's goal under each hole, as the next round's prompt shows it."""
        probe, at = hs.probe_declaration(part.declaration)
        run = self.w.compile(part.above + probe + "\n", timeout=600)
        if run["timed_out"]:
            return part
        first = part.above.count("\n")  # the 0-based line of the declaration in what was compiled
        goals: dict[int, str] = {}
        for m in run["messages"]:
            line = int((m.get("pos") or {}).get("line", 0)) - 1 - first
            if m.get("severity") == "info" or m.get("severity") == "information":
                if line in at and at[line] not in goals:
                    goals[at[line]] = str(m.get("data", ""))
        return Partial(hs.with_goals(part.declaration, goals), part.above, part.source)

    # ── outcomes ──

    def accept(self, part: Partial, rnd: int) -> bool:
        """A partial proof with no hole left, through the one acceptance path."""
        # what the answer adds above the theorem (its opens, its helpers); the residual's file shares
        # the whole-statement file's context (both are the row's file, the declaration replaced)
        extra = part.above[len(self.w.prefix) :] if part.above.startswith(self.w.prefix) else ""
        answer = extra + part.declaration
        res = self.w._check(self.hole, answer)
        if res.get("outcome") != "verified":
            self.w.note(f"{part.source}: every hole closed, but the proof was not accepted: {res.get('outcome')}")
            return False
        how = f"{part.source}: Goedel's plan, Wayfinder's search on its holes"
        self.record(part.source, rnd, res, {"model": "wayfinder"}, how)
        return True

    def record(self, sid: str, rnd: int, res: dict, gen: dict, how: str) -> None:
        with self.w.lock:
            self.w.fills.append({"hole": self.hole.index, "model": gen.get("model") or "?",
                                 "quant": gen.get("quant") or "-", "sample": sid, "round": rnd,
                                 "form": res["form"], "fill": res["fill"],
                                 "helpers": res["helpers"], "source": f"{how}, run {self.w.job['run_id']}",
                                 "whole": True})  # fmt: skip
            self.w._write("fills.json", self.w.fills)
        self.hole.solved_by = sid

    def write_best(self) -> None:
        self.w._write("residual.json", {"declaration": self.best.declaration, "file": self.best.file,
                                        "holes": self.best.holes, "source": self.best.source,
                                        "improved": self.best is not self.residual})  # fmt: skip
