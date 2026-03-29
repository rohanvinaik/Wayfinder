"""Materialize a symbolic hard-resolution layer from hard collection rows.

This stage sits between raw hard-proof collection and the learned hard SoM.
It builds structured prior packets for each hard residual:

- proof-graph candidate prior lemmas
- k-line-style solved-trace exemplars
- family-specific closing feature inventories
- specialist worklists grouped by residual family
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hard_resolution_layer import load_jsonl, materialize_hard_resolution_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Hard collection bundle dir containing collection_all.jsonl",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="Optional raw theorem-level JSONL inputs (used when --bundle-dir is empty)",
    )
    parser.add_argument(
        "--db",
        default="data/proof_network.db",
        help="Proof network DB used for prior-lemma discovery",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output dir (defaults to <bundle-dir>/hard_resolution_layer)",
    )
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--exemplar-limit", type=int, default=5)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else None
    if bundle_dir is not None:
        inputs = [bundle_dir / "collection_all.jsonl"]
        output_dir = Path(args.output_dir) if args.output_dir else bundle_dir / "hard_resolution_layer"
    else:
        inputs = [Path(path) for path in args.inputs]
        if not inputs:
            raise SystemExit("Provide --bundle-dir or at least one --inputs path")
        output_dir = Path(args.output_dir) if args.output_dir else Path("runs/hard_resolution_layer")

    rows = []
    for path in inputs:
        rows.extend(load_jsonl(path))

    summary = materialize_hard_resolution_layer(
        rows=rows,
        output_dir=output_dir,
        conn_or_db=Path(args.db),
        candidate_limit=args.candidate_limit,
        exemplar_limit=args.exemplar_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
