"""Characterization pin for _try_cosine_family's template generation.

Captures the EXACT tactic strings the live learned lane submits to Lean
per family, so the compilation grammar can be extracted into the pure
tactic_compiler without behavioral drift. These sequences encode
benchmark-validated policy:

- rw: per-premise direction inference (backward-first for _symm-style names)
- exact: direct term application only
- apply: the exact -> refine ?_ ?_ -> refine ?_ ?_ ?_ -> apply escalation ladder
- simp: bare simp first, then simp [top1] only (multi-hint hurts acceptance)

If one of these assertions breaks, either the refactor drifted (fix the
refactor) or the policy deliberately changed (update the pin AND the
thesis chapter's template table together).
"""

import unittest
from unittest import mock

try:  # proof_search pulls torch — skip in torch-less envs, never poison collection
    from src.proof_search import _SearchEnv, _SearchState, _try_cosine_family

    _HAVE_SEARCH = True
except ImportError:  # pragma: no cover
    _HAVE_SEARCH = False

if not _HAVE_SEARCH:
    raise unittest.SkipTest("src.proof_search requires torch")


class _FakeLean:
    """Records every tactic submitted; fails all of them by default."""

    def __init__(self, succeed_on=None):
        self.submitted = []
        self._succeed_on = succeed_on or set()

    def try_tactic(self, goal, tactic):
        self.submitted.append(tactic)
        success = tactic in self._succeed_on
        return mock.Mock(success=success, new_goals=[])


def _run_family(family, goal, premises, beam_width, succeed_on=None):
    lean = _FakeLean(succeed_on=succeed_on)
    env = _SearchEnv(conn=None, lean=lean, anchor_id_map={}, max_candidates=8)
    state = _SearchState(open_goals=[goal], theorem_id="TestThm.example")
    with mock.patch("src.proof_search._cosine_rank_premises", return_value=premises):
        closed = _try_cosine_family(
            family, goal, 0, state, env, accessible_theorem_id=None,
            beam_width=beam_width,
        )
    return lean.submitted, closed, state


class RwTemplateTest(unittest.TestCase):
    def test_backward_suffix_orders_backward_first(self):
        # add_comm ends in _comm -> infer_direction says backward
        submitted, _, _ = _run_family(
            "rw", "⊢ x + y = y + x", ["Nat.add_comm"], beam_width=1
        )
        self.assertEqual(submitted, ["rw [← Nat.add_comm]", "rw [Nat.add_comm]"])

    def test_goal_overlap_orders_forward_first(self):
        # succ_eq base tokens appear in the goal -> forward first
        submitted, _, _ = _run_family(
            "rw", "⊢ Nat.succ n = n + 1", ["Nat.succ_eq"], beam_width=1
        )
        self.assertEqual(submitted, ["rw [Nat.succ_eq]", "rw [← Nat.succ_eq]"])

    def test_beam_iterates_premises_in_rank_order(self):
        submitted, _, _ = _run_family(
            "rw", "⊢ x + y = y + x", ["Nat.add_comm", "Nat.mul_comm"], beam_width=2
        )
        self.assertEqual(
            submitted,
            [
                "rw [← Nat.add_comm]", "rw [Nat.add_comm]",
                "rw [← Nat.mul_comm]", "rw [Nat.mul_comm]",
            ],
        )

    def test_stops_at_first_success(self):
        submitted, closed, _ = _run_family(
            "rw", "⊢ x + y = y + x", ["Nat.add_comm", "Nat.mul_comm"],
            beam_width=2, succeed_on={"rw [Nat.add_comm]"},
        )
        self.assertTrue(closed)
        self.assertEqual(submitted, ["rw [← Nat.add_comm]", "rw [Nat.add_comm]"])


class ExactTemplateTest(unittest.TestCase):
    def test_direct_term_application_only(self):
        submitted, _, _ = _run_family(
            "exact", "⊢ P", ["lemma_a", "lemma_b"], beam_width=2
        )
        self.assertEqual(submitted, ["exact lemma_a", "exact lemma_b"])


class ApplyTemplateTest(unittest.TestCase):
    def test_escalation_ladder_per_premise(self):
        submitted, _, _ = _run_family(
            "apply", "⊢ P", ["lemma_a", "lemma_b"], beam_width=2
        )
        self.assertEqual(
            submitted,
            [
                "exact lemma_a",
                "refine lemma_a ?_ ?_",
                "refine lemma_a ?_ ?_ ?_",
                "apply lemma_a",
                "exact lemma_b",
                "refine lemma_b ?_ ?_",
                "refine lemma_b ?_ ?_ ?_",
                "apply lemma_b",
            ],
        )

    def test_apply_telemetry_records_every_attempt(self):
        _, _, state = _run_family("apply", "⊢ P", ["lemma_a"], beam_width=1)
        self.assertEqual(len(state._last_apply_candidates), 4)
        self.assertEqual(
            {rec["candidate"] for rec in state._last_apply_candidates}, {"lemma_a"}
        )


class SimpTemplateTest(unittest.TestCase):
    def test_bare_simp_first_then_single_hint(self):
        submitted, _, _ = _run_family(
            "simp", "⊢ P", ["lemma_a", "lemma_b"], beam_width=2
        )
        # Benchmark-validated policy: bare simp, then simp [top1] ONLY.
        self.assertEqual(submitted, ["simp", "simp [lemma_a]"])


class SelfApplicationFilterTest(unittest.TestCase):
    def test_theorem_id_filtered_from_ranking(self):
        lean = _FakeLean()
        env = _SearchEnv(conn=None, lean=lean, anchor_id_map={}, max_candidates=8)
        env.allow_self_application = False
        state = _SearchState(open_goals=["⊢ P"], theorem_id="TestThm.example")
        with mock.patch(
            "src.proof_search._cosine_rank_premises",
            return_value=["TestThm.example", "lemma_a"],
        ):
            _try_cosine_family("exact", "⊢ P", 0, state, env, None, beam_width=2)
        self.assertEqual(lean.submitted, ["exact lemma_a"])


if __name__ == "__main__":
    unittest.main()
