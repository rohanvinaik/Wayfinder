"""Intent pins for the Ducky-regime cell miner (THE_REFOUNDING §10.7).

The mining proposes; the read is authored by hand from the data. These
pins fix the proposal semantics: evidence-carrying cells, floors and
lift honored, degenerate inputs total.
"""

import unittest

from scripts.mine_ducky_regime import regime_cells

CELL_A = ("equality", (0, 0, 0, -1, 0, 0, 0, 0, 1, 0))
CELL_B = ("other", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))


class RegimeCellsTest(unittest.TestCase):
    def test_concentrated_cell_qualifies_with_evidence(self):
        cells = regime_cells(
            {CELL_A: 10}, {CELL_A: 20, CELL_B: 980}, 10, 1000, min_closes=3, min_ratio=3.0
        )
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["bucket"], "equality")
        self.assertEqual(cell["ducky_closes"], 10)
        self.assertEqual(cell["cell_close_rate"], 0.5)
        self.assertEqual(cell["lift"], 50.0)  # 0.5 vs base 0.01

    def test_floors_reject_thin_and_unlifted_cells(self):
        # Below min_closes: rejected regardless of purity.
        self.assertEqual(
            regime_cells({CELL_A: 2}, {CELL_A: 2}, 2, 100, min_closes=3), []
        )
        # At base rate (lift 1.0 < 3.0): rejected regardless of count.
        self.assertEqual(
            regime_cells({CELL_A: 50}, {CELL_A: 500}, 100, 1000, min_ratio=3.0), []
        )

    def test_degenerate_inputs_are_total(self):
        self.assertEqual(regime_cells({}, {}, 0, 0), [])
        self.assertEqual(regime_cells({CELL_A: 5}, {}, 5, 0), [])

    def test_sorted_by_closes_then_lift(self):
        cells = regime_cells(
            {CELL_A: 10, CELL_B: 4},
            {CELL_A: 20, CELL_B: 4, ("x", (1,)): 976},
            14,
            1000,
            min_closes=3,
            min_ratio=3.0,
        )
        self.assertEqual([c["ducky_closes"] for c in cells], [10, 4])


if __name__ == "__main__":
    unittest.main()
