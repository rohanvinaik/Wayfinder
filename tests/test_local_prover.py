"""The local prover lane's pure parts: the prompt bytes against the models' own chat template,
the routing votes, the memory plan, and reading a streamed completion."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

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


class RenameTest(unittest.TestCase):
    def test_the_whole_statements_helper_takes_its_own_name(self):
        lemma = "set_option maxHeartbeats 400000 in\ntheorem kappa {n : ℕ} : n = n := by\n  rfl"
        self.assertIn("theorem kappa_wf_hole_1 {n", lp._renamed(lemma, "kappa", "kappa_wf_hole_1"))


if __name__ == "__main__":
    unittest.main()
