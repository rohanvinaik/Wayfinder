"""EXP-055: Oracle closure audit on near-miss theorems.

For each near-miss theorem (1 remaining goal after apply progress),
test closure strategies on the final goal:
  1. exact? (oracle/teacher — expensive but finds the answer if it exists)
  2. exact <candidate> for each accessible premise
  3. apply <candidate> for each accessible premise
  4. refine <candidate> ?_ / refine <candidate> ?_ ?_ / refine <candidate> ?_ ?_ ?_
  5. simp / simp_all / omega / norm_num / ring / aesop

Records per-theorem: which strategy closes, what tactic, how long.

Usage:
    python -m scripts.run_near_miss_closure_audit \\
        --data data/near_miss_benchmark_29.jsonl \\
        --output runs/exp055_closure_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/near_miss_benchmark_29.jsonl")
    parser.add_argument("--output", default="runs/exp055_closure_audit.jsonl")
    parser.add_argument("--config", default="configs/wayfinder.yaml")
    parser.add_argument("--db", default="data/proof_network_v3.db")
    parser.add_argument("--lean-project", default="data/lean_project/")
    parser.add_argument("--exact-q-timeout", type=int, default=30)
    parser.add_argument("--max-premises", type=int, default=20)
    args = parser.parse_args()

    from src.lean_interface import LeanConfig, LeanKernel
    from src.proof_network import get_accessible_premises

    lean_cfg = LeanConfig(
        backend="pantograph",
        project_root=args.lean_project,
        imports=["Mathlib"],
    )
    lean = LeanKernel(lean_cfg)
    lean._ensure_server()

    conn = sqlite3.connect(args.db)
    id_to_name = {eid: name for eid, name in conn.execute("SELECT id, name FROM entities")}
    name_to_id = {name: eid for eid, name in id_to_name.items()}

    # Load near-miss data
    near_misses = []
    with open(args.data) as f:
        for line in f:
            near_misses.append(json.loads(line.strip()))

    logger.info("Loaded %d near-miss theorems", len(near_misses))

    # Automation closers (cheap, try first)
    _AUTO_CLOSERS = [
        "assumption", "rfl", "trivial", "exact?",
        "simp", "simp_all", "norm_num", "ring", "omega",
        "decide", "aesop", "tauto", "contradiction",
        "exact le_top", "exact bot_le", "exact le_refl _",
        "infer_instance",
    ]

    results = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as fout:
        for i, nm in enumerate(near_misses):
            tid = nm["theorem_id"]
            final_goal = nm["final_goal"]
            logger.info("[%d/%d] %s", i + 1, len(near_misses), tid)

            # First: need to reconstruct the goal state in Pantograph
            # Use goal_via_file_context to get the initial state, then replay
            # For now: try tactics directly on the final_goal text via goal_start
            goal_state = None
            try:
                goal_state = lean.goal_start(final_goal, theorem_name=tid)
            except Exception:
                pass

            if goal_state is None:
                logger.warning("  Cannot start final goal for %s", tid)
                results.append({
                    "theorem_id": tid,
                    "final_goal": final_goal[:100],
                    "goal_started": False,
                    "closed_by": None,
                    "closing_tactic": None,
                    "category": "goal_start_fail",
                })
                fout.write(json.dumps(results[-1]) + "\n")
                fout.flush()
                continue

            # Try automation closers
            closed_by = None
            closing_tactic = None
            category = "unclosed"

            for tac in _AUTO_CLOSERS:
                try:
                    t0 = time.time()
                    result = lean.try_tactic(goal_state, tac)
                    dt = time.time() - t0
                    if result.success and not result.new_goals:
                        closed_by = "auto"
                        closing_tactic = tac
                        category = f"auto_{tac.split()[0]}"
                        logger.info("  CLOSED by %s (%.1fs)", tac, dt)
                        break
                    elif result.success and result.new_goals:
                        # Opens subgoals — not a full close
                        pass
                except Exception:
                    pass

            # Try accessible premises with exact/apply/refine
            premises: list[str] = []
            if closed_by is None:
                accessible_id = name_to_id.get(tid)
                premises = []
                if accessible_id is not None:
                    premise_ids = get_accessible_premises(conn, accessible_id)
                    premises = [id_to_name[pid] for pid in premise_ids if pid in id_to_name][:args.max_premises]

                # Cosine rank premises
                if premises:
                    try:
                        from sentence_transformers import SentenceTransformer
                        enc = SentenceTransformer("all-MiniLM-L6-v2")
                        import numpy as np
                        goal_emb = enc.encode([goal_state], normalize_embeddings=True)
                        prem_embs = enc.encode(premises, normalize_embeddings=True)
                        scores = (goal_emb @ prem_embs.T).flatten()
                        ranked_idx = list(np.argsort(-scores))  # type: ignore[operator]
                        premises = [premises[j] for j in ranked_idx[:args.max_premises]]
                    except Exception:
                        pass

                for prem in premises:
                    for tac_template in [
                        f"exact {prem}",
                        f"apply {prem}",
                        f"refine {prem} ?_",
                        f"refine {prem} ?_ ?_",
                        f"refine {prem} ?_ ?_ ?_",
                    ]:
                        try:
                            result = lean.try_tactic(goal_state, tac_template)
                            if result.success and not result.new_goals:
                                closed_by = "premise"
                                closing_tactic = tac_template
                                category = f"premise_{tac_template.split()[0]}"
                                logger.info("  CLOSED by %s", tac_template)
                                break
                        except Exception:
                            pass
                    if closed_by:
                        break

            row = {
                "theorem_id": tid,
                "final_goal": final_goal[:200],
                "goal_started": True,
                "closed_by": closed_by,
                "closing_tactic": closing_tactic,
                "category": category,
                "premises_tried": len(premises) if closed_by is None and premises else 0,
            }
            results.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    closed = sum(1 for r in results if r["closed_by"])
    print("\n" + "=" * 60)
    print("EXP-055: Near-Miss Closure Audit")
    print("=" * 60)
    print(f"  Total: {len(results)}")
    print(f"  Goal started: {sum(1 for r in results if r['goal_started'])}")
    print(f"  Closed: {closed}")
    print(f"  Unclosed: {len(results) - closed}")
    print("\n  By category:")
    for cat, n in cats.most_common():
        print(f"    {cat:30s}: {n}")
    if closed:
        print("\n  Closing tactics:")
        for r in results:
            if r["closed_by"]:
                print(f"    {r['theorem_id'][:45]:45s} {r['closing_tactic']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
