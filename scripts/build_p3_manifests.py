#!/usr/bin/env python3
"""Build P3 core evaluation manifest into data/eval/p3_core.jsonl (§5.2, §S0.7).

P3-core includes 6 check theorems:
- mathd_algebra_107 (minif2f_test.jsonl)
- amc12a_2021_p9 (minif2f_test.jsonl)
- aime_1983_p2 (minif2f_test.jsonl)
- sheaf_implies_local_objective (gap_T7_18_obstruction_intrinsic_manifest.jsonl)
- kappa_additive_bound (gap_T7_17_sheaf_gluing_manifest.jsonl)
- pigeonhole_cover (gap_T5_12_greedy_harmonic_manifest.jsonl)
"""

import json
from pathlib import Path

TARGETS = [
    ("data/minif2f_test.jsonl", "mathd_algebra_107"),
    ("data/minif2f_test.jsonl", "amc12a_2021_p9"),
    ("data/minif2f_test.jsonl", "aime_1983_p2"),
    ("data/gap_T7_18_obstruction_intrinsic_manifest.jsonl", "sheaf_implies_local_objective"),
    ("data/gap_T7_17_sheaf_gluing_manifest.jsonl", "kappa_additive_bound"),
    ("data/gap_T5_12_greedy_harmonic_manifest.jsonl", "pigeonhole_cover"),
]

OUT = Path("data/eval/p3_core.jsonl")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for filepath, tid in TARGETS:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Missing required manifest file: {filepath}")
        found = None
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("theorem_id", "")
            if row_id == tid or row_id == f"minif2f_test_{tid}":
                found = row
                break
        if found is None:
            raise ValueError(f"Theorem {tid} not found in {filepath}")
        rows.append(found)

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} P3-core theorems to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
