"""Intent tests for the censor store and the Monty Hall gate (Wave 2).

Pins the governance, not just the arithmetic: the zero-success guard, the
plurality floors, the gate laws (no signature → no fence; removal-only;
gate-time plurality backstop), the κ-knee ranking determinism, and the
end-to-end deterministic search step with fences live and the abstention
ledger populated.
"""

import unittest
from unittest import mock

from src.censor_store import (
    ADMISSIBLE,
    INSUFFICIENT_SPINE,
    PLURALITY_VIOLATED,
    SUCCESS_OBSERVED,
    censor_key,
    fenced_families,
    load_fences,
    marginal_kappa,
    plurality_verdict,
    rank_by_kappa,
    slice_signature,
    spine_verdict,
)

try:  # proof_search pulls torch, absent in Detective's tool env — skip, don't poison collection
    from src.proof_search import SearchConfig, _censored_lane_order, _SearchEnv, _SearchState

    _HAVE_SEARCH = True
except ImportError:  # pragma: no cover - exercised only in torch-less environments
    _HAVE_SEARCH = False
from src.structural_reads import read_goal


class SpineVerdictTest(unittest.TestCase):
    def test_one_success_kills_the_candidate(self):
        # The fence would be FALSE, not weak — a different "no" than a thin spine.
        self.assertEqual(
            spine_verdict(500, 50, successes=1, min_attempts=25, min_theorems=8),
            SUCCESS_OBSERVED,
        )

    def test_floors_are_inclusive(self):
        self.assertEqual(
            spine_verdict(25, 8, 0, min_attempts=25, min_theorems=8), ADMISSIBLE
        )
        self.assertEqual(
            spine_verdict(24, 8, 0, min_attempts=25, min_theorems=8), INSUFFICIENT_SPINE
        )
        self.assertEqual(
            spine_verdict(25, 7, 0, min_attempts=25, min_theorems=8), INSUFFICIENT_SPINE
        )


class PluralityTest(unittest.TestCase):
    def test_last_open_family_is_protected(self):
        self.assertEqual(plurality_verdict(3, 3), PLURALITY_VIOLATED)
        self.assertEqual(plurality_verdict(3, 2), ADMISSIBLE)


class GateLawsTest(unittest.TestCase):
    FENCES = {censor_key("equality", (0, 0, 0, -1, 0, 0)): {"exact", "learned"}}

    def test_no_signature_fences_nothing(self):
        self.assertEqual(fenced_families("equality", None, self.FENCES), set())

    def test_unknown_key_fences_nothing(self):
        self.assertEqual(fenced_families("exists", (1, 0, 0, 0, 0, 0), self.FENCES), set())

    def test_matching_key_returns_exact_fence_set(self):
        self.assertEqual(
            fenced_families("equality", (0, 0, 0, -1, 0, 0), self.FENCES),
            {"exact", "learned"},
        )


class KappaRankingTest(unittest.TestCase):
    def test_greedy_marginal_with_knee_stop(self):
        candidates = {
            ("k1", "exact"): frozenset({1, 2, 3, 4}),
            ("k2", "learned"): frozenset({3, 4, 5}),
            ("k3", "apply"): frozenset({5}),
        }
        adopted = rank_by_kappa(candidates, min_kappa=2)
        # k1 first (κ=4); k2 next with marginal κ=1 < 2 → knee stops adoption.
        self.assertEqual(adopted, [(("k1", "exact"), 4)])
        self.assertEqual(marginal_kappa(frozenset({3, 4, 5}), {1, 2, 3, 4}), 1)

    def test_deterministic_tie_break(self):
        candidates = {
            ("b", "rw"): frozenset({1, 2}),
            ("a", "rw"): frozenset({3, 4}),
        }
        adopted = rank_by_kappa(candidates, min_kappa=1)
        self.assertEqual([pair for pair, _k in adopted], [("a", "rw"), ("b", "rw")])


@unittest.skipUnless(_HAVE_SEARCH, "src.proof_search requires torch")
class RuntimeGateTest(unittest.TestCase):
    GOAL = "x : ℕ\nh : x + 2 = 5\n⊢ x = 3"

    def _fences_for(self, goal, families):
        reads = read_goal(goal)
        return {censor_key(reads.goal_bucket, slice_signature(reads)): set(families)}

    def _env(self, fences):
        env = _SearchEnv(conn=None, lean=mock.Mock(), anchor_id_map={}, max_candidates=8)
        env.censor_fences = fences
        return env

    def test_gate_off_by_default(self):
        env = self._env(None)
        state = _SearchState(open_goals=[self.GOAL])
        order = ["structural_core", "symbolic_exact", "symbolic_rw"]
        self.assertEqual(_censored_lane_order(order, self.GOAL, state, env), order)
        self.assertEqual(state._censor_drops, [])

    def test_fenced_family_is_removed_with_ledger_warrant(self):
        env = self._env(self._fences_for(self.GOAL, {"exact"}))
        state = _SearchState(open_goals=[self.GOAL])
        order = ["structural_core", "symbolic_exact", "symbolic_rw"]
        self.assertEqual(
            _censored_lane_order(order, self.GOAL, state, env),
            ["structural_core", "symbolic_rw"],
        )
        self.assertEqual(len(state._censor_drops), 1)
        drop = state._censor_drops[0]
        self.assertEqual((drop["lane"], drop["family"]), ("symbolic_exact", "exact"))

    def test_gate_time_plurality_backstop(self):
        # A fence set that would empty the order is not applied at all.
        env = self._env(self._fences_for(self.GOAL, {"exact", "rw"}))
        state = _SearchState(open_goals=[self.GOAL])
        order = ["symbolic_exact", "symbolic_rw"]
        self.assertEqual(_censored_lane_order(order, self.GOAL, state, env), order)
        self.assertEqual(state._censor_drops, [])

    def test_unfenced_signature_untouched(self):
        other = "⊢ ∀ x : Nat, x + 0 = x"
        env = self._env(self._fences_for(self.GOAL, {"exact"}))
        state = _SearchState(open_goals=[other])
        order = ["structural_core", "symbolic_exact"]
        self.assertEqual(_censored_lane_order(order, other, state, env), order)


class LoadFencesTest(unittest.TestCase):
    def test_missing_path_is_gate_off(self):
        self.assertEqual(load_fences(""), {})
        self.assertEqual(load_fences("/nonexistent/censors.json"), {})

    @unittest.skipUnless(_HAVE_SEARCH, "src.proof_search requires torch")
    def test_search_config_default_is_gate_off(self):
        self.assertEqual(SearchConfig().censor_store_path, "")


if __name__ == "__main__":
    unittest.main()
