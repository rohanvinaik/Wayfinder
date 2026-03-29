from __future__ import annotations

import json
from pathlib import Path

from scripts.run_exp_dd015_integrated_bridge import _select_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_select_rows_prefers_validated_progress_rows(tmp_path: Path) -> None:
    details = tmp_path / "details.jsonl"
    validated = tmp_path / "validated.jsonl"
    _write_jsonl(
        details,
        [
            {
                "theorem_id": "Batteries.UnionFind.rootD_parent",
                "residual_bucket": "multi_goal_large_progress",
                "goals_closed": 21,
                "goals_remaining": 20,
                "attempts": 541,
            },
            {
                "theorem_id": "Complex.eq_const_of_exists_le",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 31,
                "goals_remaining": 1,
                "attempts": 300,
            },
            {
                "theorem_id": "Affine.Simplex.Scalene.dist_ne",
                "residual_bucket": "single_goal_stall",
                "goals_closed": 0,
                "goals_remaining": 1,
                "attempts": 3,
            },
            {
                "theorem_id": "Equiv.Perm.disjoint_ofSubtype_of_memFixedPoints_self",
                "residual_bucket": "multi_goal_small_progress",
                "goals_closed": 10,
                "goals_remaining": 1,
                "attempts": 200,
            },
        ],
    )
    _write_jsonl(
        validated,
        [
            {
                "theorem_id": "Batteries.UnionFind.rootD_parent",
                "residual_bucket": "multi_goal_large_progress",
                "started": True,
                "progressed": True,
                "closed": False,
                "goals_after": ["g1", "g2", "g3", "g4"],
                "final_goal": "self.parent x = self.rootD x",
                "final_goal_bucket": "equality",
            },
            {
                "theorem_id": "Complex.eq_const_of_exists_le",
                "residual_bucket": "single_goal_near_miss",
                "started": True,
                "progressed": True,
                "closed": False,
                "goals_after": ["x = 0"],
                "final_goal": "x = 0",
                "final_goal_bucket": "equality",
            },
            {
                "theorem_id": "Affine.Simplex.Scalene.dist_ne",
                "residual_bucket": "single_goal_stall",
                "started": True,
                "progressed": True,
                "closed": False,
                "goals_after": ["False"],
                "final_goal": "False",
                "final_goal_bucket": "false",
            },
            {
                "theorem_id": "Equiv.Perm.disjoint_ofSubtype_of_memFixedPoints_self",
                "residual_bucket": "multi_goal_small_progress",
                "started": True,
                "progressed": False,
                "closed": False,
                "goals_after": ["False"],
                "final_goal": "False",
                "final_goal_bucket": "false",
            },
        ],
    )

    rows, source = _select_rows(
        rows_path=details,
        residual_buckets={
            "single_goal_near_miss",
            "single_goal_stall",
            "multi_goal_small_progress",
            "multi_goal_large_progress",
        },
        limit=3,
        selection_source="validated_progress",
        validated_seed_path=validated,
        allow_unvalidated_backfill=False,
    )

    assert source == "validated_progress"
    assert [row["theorem_id"] for row in rows] == [
        "Complex.eq_const_of_exists_le",
        "Affine.Simplex.Scalene.dist_ne",
        "Batteries.UnionFind.rootD_parent",
    ]


def test_select_rows_backfills_when_validated_rows_are_insufficient(tmp_path: Path) -> None:
    details = tmp_path / "details.jsonl"
    validated = tmp_path / "validated.jsonl"
    _write_jsonl(
        details,
        [
            {
                "theorem_id": "T_easy",
                "residual_bucket": "single_goal_near_miss",
                "goals_closed": 4,
                "goals_remaining": 1,
                "attempts": 10,
            },
            {
                "theorem_id": "T_backfill_1",
                "residual_bucket": "single_goal_stall",
                "goals_closed": 0,
                "goals_remaining": 1,
                "attempts": 3,
            },
            {
                "theorem_id": "T_backfill_2",
                "residual_bucket": "multi_goal_small_progress",
                "goals_closed": 7,
                "goals_remaining": 2,
                "attempts": 20,
            },
        ],
    )
    _write_jsonl(
        validated,
        [
            {
                "theorem_id": "T_easy",
                "residual_bucket": "single_goal_near_miss",
                "started": True,
                "progressed": True,
                "closed": False,
                "goals_after": ["x = 0"],
                "final_goal": "x = 0",
                "final_goal_bucket": "equality",
            }
        ],
    )

    rows, source = _select_rows(
        rows_path=details,
        residual_buckets={
            "single_goal_near_miss",
            "single_goal_stall",
            "multi_goal_small_progress",
            "multi_goal_large_progress",
        },
        limit=3,
        selection_source="validated_progress",
        validated_seed_path=validated,
        allow_unvalidated_backfill=True,
    )

    assert source == "validated_progress_backfill"
    assert [row["theorem_id"] for row in rows] == ["T_easy", "T_backfill_1", "T_backfill_2"]
