"""The hand-off loop's control: Goedel asked a fixed number of times on Wayfinder's residual and never
on its own answers, only what Lean accepts carried forward, a round with no validated progress the
last, closure only through the acceptance path. Goedel's answers and Wayfinder's in-place results
are scripted here; the pieces they stand for have their own tests."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src import handoff_loop as hl

RESIDUAL = """theorem t (a : ℕ) : a + 0 = a ∧ 0 ≤ a := by
  constructor
  -- WAYFINDER RESIDUAL 1: close this goal
  · sorry
  -- WAYFINDER RESIDUAL 2: close this goal
  · sorry"""
ONE_HOLE = """theorem t (a : ℕ) : a + 0 = a ∧ 0 ≤ a := by
  constructor
  · simp
  -- WAYFINDER RESIDUAL 1: close this goal
  · sorry"""
CLOSED = """theorem t (a : ℕ) : a + 0 = a ∧ 0 ≤ a := by
  constructor
  · simp
  · omega"""
PREFIX = "import Mathlib\n\n"


class _Hole:
    index, name, statement, solved_by = 1, "t", "theorem t (a : ℕ) : a + 0 = a ∧ 0 ≤ a", ""


class _Worker:
    def __init__(self, checks: dict[str, str] | None = None):
        self.dir = Path(tempfile.mkdtemp())
        self.job = {"residual_declaration": RESIDUAL, "residual_file": PREFIX + RESIDUAL + "\n", "row": {},
                    "run_id": "r", "budget_s": 60}  # fmt: skip
        self.holes, self.prefix, self.fills = [_Hole()], PREFIX, []
        self.stop, self.lock = threading.Event(), threading.Lock()
        self.checks = checks or {}
        self.skeletons: dict[str, dict] = {}

    def formal(self, hole):
        return f"{self.prefix}{hole.statement} := by sorry"

    def publish(self, **_kw):
        pass

    def note(self, _text):
        pass

    def _write(self, name, data):
        (self.dir / name).write_text(json.dumps(data))

    def skeleton_repair(self, hole, code, draft=False):
        return self.skeletons[code]

    def _check(self, hole, answer):
        return {"outcome": self.checks.get(answer.strip(), "wrong_proof"), "form": "inline", "fill": "x", "helpers": ""}


class _Loop(hl.Handoff):
    """The loop with Goedel's answers and Wayfinder's in-place results scripted."""

    def __init__(self, worker, answers: list[list[tuple[str, dict, dict]]], searched: dict[str, str]):
        super().__init__(worker)
        self.answers, self.searched, self.prompts = list(answers), searched, []

    def sample(self, pool, asks):
        self.prompts.append([messages[0]["content"] for _, messages, _ in asks])
        batch = self.answers.pop(0) if self.answers else []
        for sid, gen, res in batch:
            if res.get("outcome") == "verified":
                self.record(sid, 0, res, gen, "scripted")
                return True, []
        return False, batch

    def search_in_place(self, part):
        return hl.Partial(self.searched.get(part.declaration, part.declaration), part.above, part.source + "+wf")

    def annotate(self, part):
        return part


def _answer(code: str) -> dict:
    return {"content": f"```lean4\n{code}\n```"}


class HandoffLoopTest(unittest.TestCase):
    def test_a_verified_answer_closes_the_run(self):
        w = _Worker()
        loop = _Loop(w, [[("s1", {"model": "g"}, {"outcome": "verified", "form": "inline", "fill": "f", "helpers": ""})]], {})
        self.assertTrue(loop.run(pool=None))
        self.assertEqual([f["whole"] for f in w.fills], [True])
        self.assertEqual(len(loop.prompts), 1)

    def test_a_round_without_validated_progress_is_the_last(self):
        w = _Worker()
        loop = _Loop(w, [[("s1", _answer("junk"), {"outcome": "no_code"})], []], {})
        self.assertFalse(loop.run(pool=None))
        self.assertEqual(len(loop.prompts), 1)  # Goedel is not asked again
        self.assertFalse(json.loads((w.dir / "residual.json").read_text())["improved"])

    def test_progress_goes_back_as_the_residual_never_as_goedels_answer(self):
        w = _Worker()
        w.skeletons["proof attempt"] = {"outcome": "open", "skeleton": RESIDUAL, "fills": ["simp", None],
                                        "opens": [], "helpers": []}  # fmt: skip
        loop = _Loop(w, [[("s1", _answer("proof attempt"), {"outcome": "wrong_proof"})], []], {})
        self.assertFalse(loop.run(pool=None))
        self.assertEqual(len(loop.prompts), 2)
        self.assertEqual([len(p) for p in loop.prompts], [8, 4])  # six + two, then four
        second = loop.prompts[1][0]
        self.assertIn("· simp", second)  # the closers' progress, in the residual Goedel sees
        self.assertNotIn("proof attempt", second)  # not its own earlier answer
        self.assertEqual(loop.best.holes, 1)

    def test_a_refuted_claim_drops_its_skeleton(self):
        w = _Worker()
        w.skeletons["false claim"] = {"outcome": "open", "skeleton": RESIDUAL, "fills": ["simp", None],
                                      "opens": [], "helpers": [], "refuted": {2: "a counterexample"}}  # fmt: skip
        loop = _Loop(w, [[("s1", _answer("false claim"), {"outcome": "wrong_proof"})]], {})
        self.assertFalse(loop.run(pool=None))
        self.assertEqual(loop.best.declaration, RESIDUAL)

    def test_no_hole_left_goes_through_the_acceptance_path(self):
        w = _Worker(checks={CLOSED: "verified"})
        loop = _Loop(w, [[("s1", _answer("junk"), {"outcome": "no_code"})]], {RESIDUAL: CLOSED})
        self.assertTrue(loop.run(pool=None))  # the residual itself, its holes closed by Wayfinder in place
        self.assertEqual(len(w.fills), 1)


if __name__ == "__main__":
    unittest.main()
