"""The local prover lane's pure parts: the prompt bytes against the models' own chat template,
the routing votes, the memory plan, and reading a streamed completion."""

from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.local_prover as lp

TEMPLATE = Path(__file__).parent / "fixtures" / "goedel_v2_chat_template.jinja"  # Goedel-LM/Goedel-Prover-V2


def _rendered(messages: list[dict]) -> str:
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(TEMPLATE.read_text()).render(messages=messages, add_generation_prompt=True)


@unittest.skipUnless(importlib.util.find_spec("jinja2"), "jinja2 not installed")
class PromptBytesTest(unittest.TestCase):
    """What is sent is exactly what the tokenizer's chat template would render."""

    def test_the_first_round_matches_the_template(self):
        msgs = [{"role": "user", "content": lp.ROUND0.format(statement="theorem t : True := by sorry")}]
        self.assertEqual(lp.chatml(msgs), _rendered(msgs))

    def test_a_correction_round_drops_the_past_turns_reasoning_as_the_template_does(self):
        answer = "<think>\nlong reasoning\n</think>\n\nPlan.\n```lean4\ntheorem t : True := by simp\n```"
        msgs = [
            {"role": "user", "content": lp.ROUND0.format(statement="theorem t : True := by sorry")},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": lp.CORRECTION.format(previous=0, errors="\nError 1: …\n")},
        ]
        self.assertEqual(lp.chatml(msgs), _rendered(msgs))
        self.assertNotIn("long reasoning", lp.chatml(msgs))


class PromptTextTest(unittest.TestCase):
    def test_the_requests_are_goedel_prover_v2s_own(self):
        first = lp.ROUND0.format(statement="S := by sorry")
        self.assertTrue(first.startswith("Complete the following Lean 4 code:\n\n```lean4\nS := by sorry```\n\n"))
        self.assertIn("provide a detailed proof plan", first)
        fix = lp.CORRECTION.format(previous=0, errors="E")
        self.assertTrue(fix.startswith("The proof (Round 0) is not correct."))
        self.assertTrue(fix.endswith("provide a detailed analysis of the error message."))

    def test_a_correction_carries_the_failed_answer_then_the_errors(self):
        failure = {"messages": [{"role": "user", "content": "q"}], "raw": "a", "report": "E"}
        conv = lp._corrected(failure, 2)
        self.assertEqual([m["role"] for m in conv], ["user", "assistant", "user"])
        self.assertIn("Round 1", conv[-1]["content"])


KAPPA_GOAL = (
    "α : Type u_1\ninst✝ : DecidableEq α\nn : ℕ\nlocalSuite : Fin n → Finset α\nκLocal : Fin n → ℕ\n"
    "hBound : ∀ (i : Fin n), #(localSuite i) ≤ κLocal i\n⊢ #(univ.biUnion localSuite) ≤ univ.sum κLocal"
)


class RoutingTest(unittest.TestCase):
    def test_a_compact_library_goal_is_light_and_escalates(self):
        a = lp.assess(KAPPA_GOAL, [], whole_statement=True)
        self.assertEqual(a["tier"], "light")
        self.assertEqual(lp.model_plan(a), [lp.LIGHT, lp.HEAVY])

    def test_an_existential_over_the_files_own_definitions_is_heavy(self):
        goal = "f : ℕ → ℕ\nh : ∀ n, obstructionClass f n = 0\n⊢ ∃ m, ∀ k ≥ m, obstructionClass f k ≤ f k * f (k + 1)"
        a = lp.assess(goal, ["obstructionClass"], whole_statement=True)
        self.assertEqual(a["tier"], "heavy", a)
        self.assertEqual(lp.model_plan(a), [lp.HEAVY])
        self.assertTrue(any("existential" in v for v in a["votes"]))

    def test_a_forced_model_is_the_whole_plan(self):
        self.assertEqual(lp.model_plan({"tier": "light"}, forced=lp.HEAVY), [lp.HEAVY])


class MemoryPlanTest(unittest.TestCase):
    def test_the_kv_cache_is_sized_from_the_architecture(self):
        self.assertEqual(lp.MODELS[lp.HEAVY].kv_bytes_per_token(), 64 * 8 * 128 * 2 * 34 // 32)

    def test_slots_fit_beside_the_weights_and_the_reserve(self):
        spec = lp.MODELS[lp.HEAVY]
        per_slot = spec.ctx * spec.kv_bytes_per_token()
        weights, reserve = 20 * lp.GiB, 6 * lp.GiB
        room = int(1.5 * lp.GiB) + weights + reserve
        self.assertEqual(lp.plan_slots(spec, weights, room + per_slot, reserve, 4), 1)
        self.assertEqual(lp.plan_slots(spec, weights, room + 3 * per_slot + 1, reserve, 4), 3)
        self.assertEqual(lp.plan_slots(spec, weights, room + 9 * per_slot, reserve, 4), 4)
        self.assertEqual(lp.plan_slots(spec, weights, room, reserve, 4), 0)

    def test_a_shorter_context_fits_more_slots(self):
        spec = lp.MODELS[lp.LIGHT]
        weights, reserve = 9 * lp.GiB, 6 * lp.GiB
        available = 29 * lp.GiB  # free memory measured on this machine before a run
        self.assertEqual(lp.plan_slots(spec, weights, available, reserve, 8), 4)  # full 40,960-token slots
        self.assertEqual(lp.plan_slots(spec, weights, available, reserve, 8, ctx=16384), 8)


def _event(**kw) -> bytes:
    return f"data: {json.dumps(kw)}\n".encode()


class StreamTest(unittest.TestCase):
    def test_the_text_and_the_final_counts_are_collected(self):
        lines = [_event(content="Plan"), b"\n", _event(content=" done"),
                 _event(content="", stop=True, stop_type="eos", tokens_predicted=2,
                        timings={"predicted_per_second": 12.5, "predicted_ms": 160})]  # fmt: skip
        out = lp.read_events(lines, lambda: False)
        self.assertEqual(out["content"], "Plan done")
        self.assertEqual((out["stop_type"], out["tokens_predicted"], out["tok_per_s"]), ("eos", 2, 12.5))
        self.assertFalse(out["cancelled"])

    def test_a_stop_request_cancels_mid_stream(self):
        calls = iter([False, True])
        out = lp.read_events([_event(content="a"), _event(content="b")], lambda: next(calls))
        self.assertTrue(out["cancelled"])
        self.assertEqual(out["content"], "a")


def _watcher() -> lp.Worker:
    w = lp.Worker.__new__(lp.Worker)  # no job: the watchdog only needs its stop state
    w.stop, w.stop_reason, w.stop_kind = threading.Event(), "", ""
    return w


class YieldTest(unittest.TestCase):
    """The worker gives the machine its RAM back rather than pushing it into pressure."""

    def test_pressure_held_for_thirty_seconds_stops_the_run(self):
        w, srv = _watcher(), SimpleNamespace(stopping=False, alive=lambda: True)
        with mock.patch.object(lp, "pressure_level", return_value=2), mock.patch.object(lp.time, "sleep"):
            w.watch(srv)
        self.assertTrue(w.stop.is_set())
        self.assertEqual(w.stop_kind, "memory")

    def test_a_brief_spike_does_not(self):
        w, srv = _watcher(), SimpleNamespace(stopping=False, alive=lambda: True)
        levels = iter([2, 2, 1, 2, 2, 1, 1])
        polls = iter(range(7))

        def sleep(_s):  # the server's own stop ends the watch after seven polls
            if next(polls) == 6:
                srv.stopping = True

        with mock.patch.object(lp, "pressure_level", side_effect=lambda: next(levels)), \
                mock.patch.object(lp.time, "sleep", side_effect=sleep):  # fmt: skip
            w.watch(srv)
        self.assertFalse(w.stop.is_set())

    def test_a_dead_server_ends_the_run(self):
        w = _watcher()
        w.watch(SimpleNamespace(stopping=False, alive=lambda: False))
        self.assertEqual(w.stop_kind, "server")


def _err(line: int, data: str) -> dict:
    return {"severity": "error", "pos": {"line": line, "column": 4}, "data": data}


class RepairTest(unittest.TestCase):
    """Only errors that say a tactic acted on nothing are repaired, and only inside the proof."""

    def test_no_goals_and_no_progress_lines_are_deleted(self):
        msgs = [_err(12, "No goals to be solved"), _err(14, "ring_nf made no progress at h₁"),
                {"severity": "warning", "pos": {"line": 3}, "data": "unused variable"}]  # fmt: skip
        self.assertEqual(lp.repairable_lines(msgs, 10, 20), {12, 14})

    def test_a_real_error_anywhere_means_no_repair(self):
        self.assertIsNone(lp.repairable_lines([_err(12, "No goals to be solved"), _err(13, "unsolved goals")], 10, 20))
        self.assertIsNone(lp.repairable_lines([_err(5, "No goals to be solved")], 10, 20))  # outside the proof
        self.assertIsNone(lp.repairable_lines([], 10, 20))


class PoolTest(unittest.TestCase):
    """Attempts take the local server's slots first, then the remotes'; a dropped remote is skipped."""

    def servers(self):
        local = SimpleNamespace(name="local", slots=2)
        remote = SimpleNamespace(name="air", slots=1)
        return local, remote, lp.Pool([local, remote])

    def test_local_first_then_remote_then_wait_for_a_release(self):
        local, remote, pool = self.servers()
        self.assertEqual(pool.slots, 3)
        self.assertIs(pool.acquire(), local)
        self.assertIs(pool.acquire(), local)
        self.assertIs(pool.acquire(), remote)
        pool.release(local)
        self.assertIs(pool.acquire(), local)

    def test_a_dropped_remote_is_left_out(self):
        local, remote, pool = self.servers()
        pool.drop(remote)
        self.assertEqual(pool.slots, 2)
        self.assertIsNone(pool.acquire(exclude={id(local)}))

    def test_the_pool_file_is_optional(self):
        with mock.patch.dict("os.environ", {"WAYFINDER_LLM_POOL": "/nonexistent/pool.json"}):
            self.assertEqual(lp.pool_config(), {})


class RenameTest(unittest.TestCase):
    def test_the_whole_statements_helper_takes_its_own_name(self):
        lemma = "set_option maxHeartbeats 400000 in\ntheorem kappa {n : ℕ} : n = n := by\n  rfl"
        self.assertIn("theorem kappa_wf_hole_1 {n", lp._renamed(lemma, "kappa", "kappa_wf_hole_1"))


if __name__ == "__main__":
    unittest.main()
