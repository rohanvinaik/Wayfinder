"""loop_union — the loop's official scoreboard (D3: union accounting).

The counting law (CLOSING_DECISIONS D3, decided 2026-08-31):

  A checked proof is checked forever. The loop's official measure is the
  UNION of every kernel-verified honest close across all passes — a row
  banks iff its disposition is `closed_value` (the value kill). Single-run
  wobble (the measured ±3 flicker band) is ordering noise, reported beside
  the union so nobody mistakes wobble for signal.

  The two-ledger rule: run quality gates the EVIDENCE, never the BANK.
  A known-defect run's kernel-valid honest closes still enter the union
  (the kernel owns proof validity; the run's defect cannot un-check it),
  but the run is excluded from flicker-band and delta claims — as evidence
  about the machinery it is contaminated. This is Detective's
  value-kill ≠ run-kill discipline applied to theorem search.

The banking decision CONSUMES each row's computed `disposition` field
(Wave 5, `augment_result_entry`) — it never re-derives honesty from
raw success flags (§7.2, the consumed-signal law). A report whose rows
lack the field, or carry an unknown code, fails loudly: that is
vocabulary drift, never a default.

Usage:
    python scripts/loop_union.py                    # canonical manifest
    python scripts/loop_union.py A.json B.json      # explicit reports
    python scripts/loop_union.py --defect X.json    # mark X evidence-excluded
    python scripts/loop_union.py --out runs/loop_union.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dispositions import CLOSED_VALUE, DISPOSITIONS  # noqa: E402

# The canonical run manifest, in chronological order. `defect_reason` marks a
# run whose CODE carried a later-diagnosed defect: its closes still bank
# (two-ledger rule) but the run is excluded from flicker evidence.
DEFAULT_MANIFEST: tuple[tuple[str, str | None], ...] = (
    ("runs/loop_pass1_full2000.json", None),
    ("runs/loop_pass1b_untraced_identity.json", None),
    ("runs/loop_pass1c_traced.json", None),
    ("runs/loop_pass2_control.json", None),
    (
        "runs/loop_pass2_treated_v2store.json",
        "granularity transport defect (§10.8): lane fences passed into Ducky's internal generation",
    ),
    (
        "runs/loop_pass2_treated_v3store_pretransportfix.json",
        "granularity transport defect (§10.8): lane fences passed into Ducky's internal generation",
    ),
    ("runs/loop_pass2_treated.json", None),
    ("runs/loop_pass3_control.json", None),
    ("runs/loop_pass3b_plans.json", None),
    ("runs/loop_pass3c_diagnosis.json", None),
    (
        "runs/loop_pass3d_addressed.json",
        "falsified distinctive-only reserve (A-cycle finding 2): known-defect slate, -10 run",
    ),
)


def bank_and_deltas(runs: list[dict]) -> dict:
    """The pure counting decision over pre-extracted rows.

    Each run: {"name": str, "defect_reason": str | None,
               "rows": [[theorem_id, disposition], ...]}  (chronological order)

    Returns the scoreboard: the union bank (with first-banking provenance),
    per-run gains/flicker-drops against the standing bank, and the flicker
    band summarized over evidence runs only.
    """
    banked_by: dict[str, str] = {}
    per_run: list[dict] = []
    evidence_drop_counts: list[int] = []
    evidence_gain_counts: list[int] = []

    for run in runs:
        name = str(run["name"])
        defect_reason = run.get("defect_reason")
        closes: set[str] = set()
        universe: set[str] = set()
        for theorem_id, disposition in run["rows"]:
            if disposition not in DISPOSITIONS:
                raise ValueError(
                    f"{name}: unknown disposition {disposition!r} for {theorem_id!r} "
                    f"— vocabulary drift, refusing to guess"
                )
            universe.add(theorem_id)
            if disposition == CLOSED_VALUE:
                closes.add(theorem_id)

        standing = set(banked_by)
        gains = sorted(closes - standing)
        flicker_drops = sorted((standing & universe) - closes)
        for theorem_id in gains:
            banked_by[theorem_id] = name

        # Flicker is defined only against a standing bank: a run whose
        # universe shares nothing with what is already banked (the founding
        # run; a new-lever pass on a disjoint residual) cannot wobble — its
        # closes are machinery signal, not band evidence.
        if defect_reason:
            evidence = "excluded"
        elif not (standing & universe):
            evidence = "no_standing_overlap"
        else:
            evidence = "counted"
        if evidence == "counted":
            evidence_drop_counts.append(len(flicker_drops))
            evidence_gain_counts.append(len(gains))
        per_run.append(
            {
                "name": name,
                "universe_size": len(universe),
                "closed_value": len(closes),
                "newly_banked": gains,
                "flicker_drops": flicker_drops,
                "evidence": evidence,
                "defect_reason": defect_reason,
            }
        )

    return {
        "counting_law": (
            "union of closed_value dispositions across all runs; the kernel owns the bank, "
            "run quality gates only the evidence (value-kill != run-kill)"
        ),
        "union_count": len(banked_by),
        "union": sorted(banked_by),
        "banked_by": {k: banked_by[k] for k in sorted(banked_by)},
        "per_run": per_run,
        "flicker_band": {
            "evidence_runs": sum(1 for r in per_run if r["evidence"] == "counted"),
            "max_drops_in_one_run": max(evidence_drop_counts, default=0),
            "max_gains_in_one_run": max(evidence_gain_counts, default=0),
            "note": (
                "per-run wobble against the standing bank, evidence runs only; "
                "single-run deltas inside this band are ordering noise, not machinery signal"
            ),
        },
    }


def _extract_rows(report_path: Path) -> list[list[str]]:
    """I/O shell: pull (theorem_id, disposition) pairs from one report."""
    with report_path.open() as fh:
        report = json.load(fh)
    details = report.get("details")
    if not isinstance(details, list) or not details:
        raise ValueError(f"{report_path}: no `details` rows — not a loop pass report")
    rows: list[list[str]] = []
    for row in details:
        disposition = row.get("disposition")
        if not isinstance(disposition, str):
            raise ValueError(
                f"{report_path}: row {row.get('theorem_id')!r} carries no computed "
                f"`disposition` — pre-Wave-5 report; the union never re-derives from raw flags"
            )
        rows.append([str(row.get("theorem_id")), disposition])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="loop_union — the loop's official scoreboard (D3: union accounting)")
    parser.add_argument("reports", nargs="*", help="report JSONs (default: canonical manifest)")
    parser.add_argument(
        "--defect",
        action="append",
        default=[],
        metavar="PATH",
        help="mark this report evidence-excluded (closes still bank)",
    )
    parser.add_argument("--out", default="runs/loop_union.json")
    args = parser.parse_args(argv)

    if args.reports:
        manifest = [(p, "flagged via --defect" if p in args.defect else None) for p in args.reports]
    else:
        manifest = list(DEFAULT_MANIFEST)

    runs = []
    for path_str, defect_reason in manifest:
        path = (_REPO / path_str) if not Path(path_str).is_absolute() else Path(path_str)
        print(f"reading {path.name} ...", flush=True)
        runs.append(
            {
                "name": path.stem,
                "defect_reason": defect_reason,
                "rows": _extract_rows(path),
            }
        )

    scoreboard = bank_and_deltas(runs)
    scoreboard["generated_from"] = [p for p, _ in manifest]

    out_path = (_REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.write_text(json.dumps(scoreboard, indent=2) + "\n")

    print(f"\nUNION: {scoreboard['union_count']} distinct kernel-verified honest closes")
    for run in scoreboard["per_run"]:
        tag = "" if run["evidence"] == "counted" else f"  [{run['evidence']}]"
        print(
            f"  {run['name']}: closed_value={run['closed_value']}"
            f" gained={len(run['newly_banked'])} flicker_drops={len(run['flicker_drops'])}{tag}"
        )
    band = scoreboard["flicker_band"]
    print(
        f"flicker band (evidence runs): up to {band['max_drops_in_one_run']} drops / "
        f"{band['max_gains_in_one_run']} gains in a single run"
    )
    print(f"scoreboard -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
