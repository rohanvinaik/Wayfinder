from __future__ import annotations

import unittest

from src.proof_search import _suppress_numeric_solver_tactics


class TestProofSearchDomainFilters(unittest.TestCase):
    def test_category_theory_suppresses_numeric_solver_tactics(self) -> None:
        self.assertTrue(
            _suppress_numeric_solver_tactics(
                "CategoryTheory.Adjunction.isIso_counit_app_iff_mem_essImage",
                "CategoryTheory.IsIso (h.counit.app X) ↔ L.essImage X",
            )
        )

    def test_arithmetic_goal_keeps_numeric_solver_tactics(self) -> None:
        self.assertFalse(
            _suppress_numeric_solver_tactics(
                "ArithmeticFunction.sum_moebius_mul_log_eq",
                "∑ d ∈ n.divisors, ↑(μ d) * log d = -Λ n",
            )
        )

    def test_algebraic_geometry_suppresses_numeric_solver_tactics(self) -> None:
        self.assertTrue(
            _suppress_numeric_solver_tactics(
                "AlgebraicGeometry.AffineSpace.isOpenMap_over",
                "IsOpenMap (ConcreteCategory.hom (AffineSpace n S ↘ S).base)",
            )
        )

    def test_abstract_algebra_bridge_suppresses_numeric_solver_tactics(self) -> None:
        self.assertTrue(
            _suppress_numeric_solver_tactics(
                "Algebra.discr_eq_discr_of_toMatrix_coeff_isIntegral",
                "(Algebra.traceMatrix ℚ ⇑b).det = Algebra.discr ℚ ⇑b'",
            )
        )


if __name__ == "__main__":
    unittest.main()
