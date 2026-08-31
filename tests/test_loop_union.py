"""Hand pins for the D3 counting law (scripts/loop_union.py).

Written from INTENT, not from behavior: each test states a clause of the
decided law — bank on the kernel's word, gate the evidence on run quality,
consume the disposition field, fail loudly on vocabulary drift.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.loop_union import bank_and_deltas, main  # noqa: E402


def _run(name, rows, defect_reason=None):
    return {"name": name, "defect_reason": defect_reason, "rows": rows}


class BankingLawTest(unittest.TestCase):
    def test_closed_value_banks_and_nothing_else_does(self):
        board = bank_and_deltas(
            [
                _run(
                    "r1",
                    [
                        ["A", "closed_value"],
                        ["B", "closed_suspect"],
                        ["C", "progressed"],
                        ["D", "blocked_unproven"],
                        ["E", "infra"],
                        ["F", "skipped_start"],
                        ["G", "abstained_certified"],
                    ],
                )
            ]
        )
        self.assertEqual(board["union"], ["A"])
        self.assertEqual(board["union_count"], 1)

    def test_union_is_distinct_across_runs_with_first_bank_provenance(self):
        board = bank_and_deltas(
            [
                _run("r1", [["A", "closed_value"], ["B", "closed_value"]]),
                _run("r2", [["B", "closed_value"], ["C", "closed_value"]]),
            ]
        )
        self.assertEqual(board["union"], ["A", "B", "C"])
        self.assertEqual(board["banked_by"], {"A": "r1", "B": "r1", "C": "r2"})

    def test_a_checked_proof_is_checked_forever(self):
        # A theorem banked by an earlier run stays banked even when a later
        # run over the same universe fails to close it.
        board = bank_and_deltas(
            [
                _run("r1", [["A", "closed_value"]]),
                _run("r2", [["A", "progressed"]]),
            ]
        )
        self.assertEqual(board["union"], ["A"])


class TwoLedgerRuleTest(unittest.TestCase):
    """The decision inside the decision: run quality gates EVIDENCE, not the BANK."""

    def test_defect_run_close_still_banks(self):
        board = bank_and_deltas(
            [
                _run("clean", [["A", "closed_value"]]),
                _run("broken", [["B", "closed_value"]], defect_reason="known defect"),
            ]
        )
        self.assertEqual(board["union"], ["A", "B"])
        self.assertEqual(board["banked_by"]["B"], "broken")

    def test_defect_run_is_excluded_from_flicker_evidence(self):
        board = bank_and_deltas(
            [
                _run("clean", [["A", "closed_value"], ["B", "closed_value"]]),
                # The defect run drops both banked theorems — a huge wobble
                # that must NOT enter the evidence band.
                _run(
                    "broken",
                    [["A", "progressed"], ["B", "progressed"]],
                    defect_reason="known defect",
                ),
            ]
        )
        broken = board["per_run"][1]
        self.assertEqual(broken["evidence"], "excluded")
        self.assertEqual(broken["flicker_drops"], ["A", "B"])  # still reported
        self.assertEqual(board["flicker_band"]["max_drops_in_one_run"], 0)  # never counted
        self.assertEqual(board["flicker_band"]["evidence_runs"], 0)  # clean run is bootstrap


class FlickerTest(unittest.TestCase):
    def test_drop_requires_membership_in_the_runs_own_universe(self):
        # A residual pass never saw the bulk theorems; not closing them is
        # not a flicker drop.
        board = bank_and_deltas(
            [
                _run("full", [["A", "closed_value"], ["B", "closed_value"]]),
                _run("residual", [["B", "progressed"], ["C", "closed_value"]]),
            ]
        )
        residual = board["per_run"][1]
        self.assertEqual(residual["flicker_drops"], ["B"])  # A is out of universe
        self.assertEqual(residual["newly_banked"], ["C"])

    def test_gains_and_drops_measured_against_the_standing_bank(self):
        board = bank_and_deltas(
            [
                _run("r1", [["A", "closed_value"]]),
                _run("r2", [["A", "progressed"], ["B", "closed_value"]]),
                _run("r3", [["A", "closed_value"], ["B", "progressed"]]),
            ]
        )
        r3 = board["per_run"][2]
        self.assertEqual(r3["newly_banked"], [])  # A was already banked by r1
        self.assertEqual(r3["flicker_drops"], ["B"])
        self.assertEqual(board["flicker_band"]["max_drops_in_one_run"], 1)
        self.assertEqual(board["flicker_band"]["max_gains_in_one_run"], 1)

    def test_flicker_is_only_defined_against_a_standing_bank(self):
        # The founding run and a new-lever run on a disjoint residual cannot
        # wobble: their closes are machinery signal, never band evidence.
        board = bank_and_deltas(
            [
                _run("founding", [["A", "closed_value"], ["B", "closed_value"]]),
                _run("new_lever_disjoint", [["C", "closed_value"], ["D", "progressed"]]),
                _run("re_measure", [["C", "progressed"], ["D", "closed_value"]]),
            ]
        )
        kinds = [r["evidence"] for r in board["per_run"]]
        self.assertEqual(kinds, ["no_standing_overlap", "no_standing_overlap", "counted"])
        self.assertEqual(board["flicker_band"]["evidence_runs"], 1)
        # The founding 2 and the disjoint 1 never enter the gains band.
        self.assertEqual(board["flicker_band"]["max_gains_in_one_run"], 1)
        self.assertEqual(board["flicker_band"]["max_drops_in_one_run"], 1)


class ConsumedSignalTest(unittest.TestCase):
    def test_unknown_disposition_fails_loudly(self):
        with self.assertRaises(ValueError):
            bank_and_deltas([_run("r1", [["A", "proved_kinda"]])])

    def test_report_without_disposition_field_is_refused(self):
        # End-to-end through main(): a pre-Wave-5 report (raw flags only,
        # no computed disposition) must be refused, never re-derived.
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "old_report.json"
            report.write_text(json.dumps({"details": [{"theorem_id": "A", "success": True, "honest_success": True}]}))
            with self.assertRaises(ValueError):
                main([str(report), "--out", str(Path(tmp) / "out.json")])


class EndToEndTest(unittest.TestCase):
    def test_main_writes_deterministic_scoreboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = Path(tmp) / "r1.json"
            r1.write_text(
                json.dumps(
                    {
                        "details": [
                            {"theorem_id": "A", "disposition": "closed_value"},
                            {"theorem_id": "B", "disposition": "progressed"},
                        ]
                    }
                )
            )
            r2 = Path(tmp) / "r2.json"
            r2.write_text(json.dumps({"details": [{"theorem_id": "B", "disposition": "closed_value"}]}))
            out = Path(tmp) / "union.json"
            self.assertEqual(main([str(r1), str(r2), "--out", str(out)]), 0)
            first = out.read_text()
            board = json.loads(first)
            self.assertEqual(board["union"], ["A", "B"])
            self.assertEqual(board["union_count"], 2)
            # Determinism: same inputs, same bytes.
            main([str(r1), str(r2), "--out", str(out)])
            self.assertEqual(out.read_text(), first)

    def test_defect_flag_reaches_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = Path(tmp) / "r1.json"
            r1.write_text(json.dumps({"details": [{"theorem_id": "A", "disposition": "closed_value"}]}))
            out = Path(tmp) / "union.json"
            main([str(r1), "--defect", str(r1), "--out", str(out)])
            board = json.loads(out.read_text())
            self.assertEqual(board["per_run"][0]["evidence"], "excluded")
            self.assertEqual(board["union"], ["A"])  # still banked


if __name__ == "__main__":
    unittest.main()
