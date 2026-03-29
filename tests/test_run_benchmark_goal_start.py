from __future__ import annotations

from unittest.mock import MagicMock

from scripts.run_benchmark import _resolve_initial_goal


def test_resolve_initial_goal_prefers_file_context_before_goal_start() -> None:
    lean = MagicMock()
    lean._backend = "pantograph"
    lean._server = object()
    lean.config.project_root = "/tmp/project"
    lean.goal_via_file_context.return_value = MagicMock(
        success=True,
        goal_state="⊢ True",
    )

    thm = {
        "theorem_id": "Nat.add_assoc",
        "file_path": "Mathlib/Init/Algebra.lean",
        "module": "Mathlib.Init.Algebra",
        "goal_state": "a + b + c = a + (b + c)",
    }

    goal = _resolve_initial_goal(thm, lean)

    assert goal == "⊢ True"
    lean.goal_via_file_context.assert_called_once()
    assert lean.goal_via_file_context.call_args.kwargs["module_hint"] == "Mathlib.Init.Algebra"
    assert lean.goal_via_file_context.call_args.kwargs["fallback_goal_pp"] == "a + b + c = a + (b + c)"
    lean.goal_start.assert_not_called()
