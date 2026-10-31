#!/usr/bin/env python3
"""Build frozen resolution population v1 and summary (§S0.7).

Walks runs/local_prover/ and runs/minif2f_v2/ for every samples.jsonl;
Filters to wrong_proof with non-empty errors classified as first_unknown, any_unknown, or type_mismatch_first;
Partitions by dev / held_out based on sha256(theorem_id);
Writes data/eval/resolution_population_v1.jsonl and data/eval/resolution_population_v1_summary.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOTS = [Path("runs/local_prover"), Path("runs/minif2f_v2")]
OUT_JSONL = Path("data/eval/resolution_population_v1.jsonl")
OUT_SUMMARY = Path("data/eval/resolution_population_v1_summary.json")


def is_type_mismatch_first(err: str) -> bool:
    return (
        err.startswith("Type mismatch")
        or err.startswith("type mismatch")
        or err.startswith("Application type mismatch")
        or "Function expected" in err
    )


def is_unknown(err: str) -> bool:
    return "Unknown identifier" in err or "Unknown constant" in err


def compute_split(theorem_id: str) -> str:
    h = hashlib.sha256(theorem_id.encode("utf-8")).hexdigest()
    return "dev" if int(h, 16) % 2 == 0 else "held_out"


def find_job_json(d: Path) -> Path | None:
    curr = d
    while curr != curr.parent:
        p = curr / "job.json"
        if p.exists():
            return p
        if curr.name in ("runs", ""):
            break
        curr = curr.parent
    return None


def collect_population() -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}

    for root_dir in ROOTS:
        if not root_dir.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            if "samples.jsonl" not in filenames:
                continue
            run_dir = Path(dirpath)
            sp = run_dir / "samples.jsonl"
            job_file = find_job_json(run_dir)
            if not job_file:
                continue
            try:
                job_data = json.loads(job_file.read_text())
                theorem_id = str(job_data["theorem_id"])
            except Exception:
                continue

            with open(sp, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    if row.get("outcome") != "wrong_proof":
                        continue
                    errors = row.get("errors") or []
                    if not errors:
                        continue

                    classes = []
                    if any(is_unknown(e) for e in errors):
                        classes.append("any_unknown")
                    if is_unknown(errors[0]):
                        classes.append("first_unknown")
                    if is_type_mismatch_first(errors[0]):
                        classes.append("type_mismatch_first")

                    if not classes:
                        continue

                    sid = str(row.get("id", ""))
                    if not sid:
                        continue

                    key = (theorem_id, sid)
                    if key in by_key:
                        continue

                    # Answer text check
                    ans_file = run_dir / "outputs" / f"{sid}.txt"
                    if not ans_file.exists() and job_file.parent != run_dir:
                        ans_file = job_file.parent / "outputs" / f"{sid}.txt"

                    answer_text = None
                    answer_missing = True
                    if ans_file.exists():
                        answer_text = ans_file.read_text()
                        answer_missing = False

                    record = {
                        "theorem_id": theorem_id,
                        "id": sid,
                        "split": compute_split(theorem_id),
                        "classes": sorted(classes),
                        "first_unknown": "first_unknown" in classes,
                        "any_unknown": "any_unknown" in classes,
                        "type_mismatch_first": "type_mismatch_first" in classes,
                        "errors": errors,
                        "run_dir": str(run_dir),
                        "sample": row,
                    }
                    if answer_missing:
                        record["answer_missing"] = True
                    else:
                        record["answer_text"] = answer_text

                    by_key[key] = record

    sorted_records = [by_key[k] for k in sorted(by_key.keys(), key=lambda x: (x[0], x[1]))]
    return sorted_records


def build_summary(records: list[dict]) -> dict:
    all_theorems = set()
    theorems_by_split = defaultdict(set)
    rows_by_split = Counter()

    class_names = ["first_unknown", "any_unknown", "type_mismatch_first"]
    by_class_split = {
        c: {
            "dev": {"rows": 0, "theorems": set()},
            "held_out": {"rows": 0, "theorems": set()},
            "theorem_counts": Counter(),
        }
        for c in class_names
    }

    for r in records:
        tid = r["theorem_id"]
        split = r["split"]
        all_theorems.add(tid)
        theorems_by_split[split].add(tid)
        rows_by_split[split] += 1

        for c in r["classes"]:
            by_class_split[c][split]["rows"] += 1
            by_class_split[c][split]["theorems"].add(tid)
            by_class_split[c]["theorem_counts"][tid] += 1

    summary = {
        "total_rows": len(records),
        "total_distinct_theorems": len(all_theorems),
        "by_split": {
            s: {
                "rows": rows_by_split[s],
                "distinct_theorems": len(theorems_by_split[s]),
            }
            for s in ("dev", "held_out")
        },
        "by_class_and_split": {},
    }

    for c in class_names:
        cdata = by_class_split[c]
        total_class_rows = cdata["dev"]["rows"] + cdata["held_out"]["rows"]
        total_class_theorems = len(cdata["dev"]["theorems"] | cdata["held_out"]["theorems"])
        largest_share = 0.0
        if total_class_rows > 0:
            max_count = cdata["theorem_counts"].most_common(1)[0][1]
            largest_share = round(max_count / total_class_rows, 4)

        summary["by_class_and_split"][c] = {
            "dev": {
                "rows": cdata["dev"]["rows"],
                "distinct_theorems": len(cdata["dev"]["theorems"]),
            },
            "held_out": {
                "rows": cdata["held_out"]["rows"],
                "distinct_theorems": len(cdata["held_out"]["theorems"]),
            },
            "total_rows": total_class_rows,
            "total_distinct_theorems": total_class_theorems,
            "largest_single_theorem_share": largest_share,
        }

    return summary


def main() -> int:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    records = collect_population()
    summary = build_summary(records)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    sha = hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()
    print(f"Wrote {len(records)} rows to {OUT_JSONL}")
    print(f"sha256({OUT_JSONL}): {sha}")
    print("Summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
