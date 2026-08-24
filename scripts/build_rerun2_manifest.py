"""Build rerun-2 manifest from rerun-1 failures.

Usage: python scripts/build_rerun2_manifest.py
"""
import json
from pathlib import Path

rerun1_details = Path("runs/minif2f_rerun/details.jsonl")
manifest = Path("data/minif2f_test.jsonl")
output = Path("data/minif2f_rerun2.jsonl")

# Get failures from rerun-1
rerun1_fails = set()
with open(rerun1_details) as f:
    for line in f:
        r = json.loads(line)
        if not r.get("success"):
            rerun1_fails.add(r.get("theorem_id", ""))

# Write manifest
count = 0
with open(manifest) as mf, open(output, "w") as out:
    for line in mf:
        thm = json.loads(line)
        if thm["theorem_id"] in rerun1_fails:
            out.write(line)
            count += 1

print(f"Wrote {count} theorems to {output}")
