"""Intent tests for the CAS solver tier in the Dr. Ducky executor.

Pins the integration contract: `cas_solve_goal` chains enter
`build_ducky_programs` as `solver_dispatch` programs with certificate
shape `cas_solver`, gated on arithmetic-shaped goals, and the tier is
advisory — a CAS failure degrades to the empty tier, never an error.
"""

import unittest
from unittest import mock

from src.dr_ducky import build_goal_capsule
from src.dr_ducky_executor import _cas_solver_programs, build_ducky_programs

_ARITH_ROW = {
    "theorem_id": "cas_smoke",
    "last_goal": "x : ℝ\nh1 : x + 2 = 5\n⊢ x = 3",
    "last_goal_bucket": "equality",
}


def _cas_programs(row, **kwargs):
    capsule = build_goal_capsule(row)
    programs = build_ducky_programs(
        capsule,
        theorem_id=row["theorem_id"],
        goal_text=row["last_goal"],
        max_programs=64,
        **kwargs,
    )
    return [p for p in programs if p.certificate_shape == "cas_solver"]


class CasSolverTierTest(unittest.TestCase):
    def test_arith_goal_gets_cas_programs(self):
        cas = _cas_programs(_ARITH_ROW)
        self.assertTrue(cas)
        for program in cas:
            self.assertEqual(program.bank, "solver_dispatch")
            self.assertEqual(program.specialist, "human_calculator")
            self.assertEqual(len(program.tactics), 1)

    def test_cas_scores_descend_below_local_fact_closers(self):
        # Since Wave C the capsule's bank weights scale scores across
        # families (program_score), so the pre-governance absolute ceiling
        # (0.85) no longer holds; the INTENT — a cached local fact beats a
        # speculative CAS chain on the same goal — is pinned relatively.
        capsule = build_goal_capsule(_ARITH_ROW)
        programs = build_ducky_programs(
            capsule,
            theorem_id=_ARITH_ROW["theorem_id"],
            goal_text=_ARITH_ROW["last_goal"],
            max_programs=64,
        )
        cas_scores = [p.score for p in programs if p.certificate_shape == "cas_solver"]
        hyp_scores = [
            p.score for p in programs if p.certificate_shape == "local_hypothesis"
            and p.tactics[0].startswith("exact ")
        ]
        self.assertEqual(cas_scores, sorted(cas_scores, reverse=True))
        self.assertTrue(hyp_scores, "the h1 hypothesis must yield exact closers")
        self.assertGreater(max(hyp_scores), max(cas_scores))

    def test_abstract_goal_stays_cas_free(self):
        row = {
            "theorem_id": "CategoryTheory.abstract_case",
            "last_goal": "⊢ CategoryTheory.Functor.map F f = g",
            "last_goal_bucket": "equality",
        }
        self.assertEqual(_cas_programs(row), [])

    def test_engine_filter_suppresses_cas_tier(self):
        # solver_dispatch maps to ArithEngine; excluding it must drop the tier.
        cas = _cas_programs(
            _ARITH_ROW, allowed_engine_names={"ContextTransportEngine"}
        )
        self.assertEqual(cas, [])

    def test_helper_caps_chain_count(self):
        chains = _cas_solver_programs(_ARITH_ROW["last_goal"], max_chains=3)
        self.assertLessEqual(len(chains), 3)

    def test_cas_failure_degrades_to_empty_tier(self):
        with mock.patch(
            "src.dr_ducky_executor.cas_solve_goal",
            side_effect=ValueError("signal only works in main thread"),
        ):
            self.assertEqual(_cas_solver_programs(_ARITH_ROW["last_goal"]), [])
            # And the full program build survives untouched.
            capsule = build_goal_capsule(_ARITH_ROW)
            programs = build_ducky_programs(
                capsule,
                theorem_id=_ARITH_ROW["theorem_id"],
                goal_text=_ARITH_ROW["last_goal"],
                max_programs=64,
            )
            self.assertTrue(programs)
            self.assertEqual(
                [p for p in programs if p.certificate_shape == "cas_solver"], []
            )


if __name__ == "__main__":
    unittest.main()
