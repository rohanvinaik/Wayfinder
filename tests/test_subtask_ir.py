from __future__ import annotations

import unittest

from src.subtask_ir import derive_goal_shape, derive_move_metadata
from src.tactic_canonicalizer import canonicalize


class TestGoalShapeIR(unittest.TestCase):
    def test_extracts_target_and_locals(self):
        goal = "\n".join(
            [
                "h : p = q",
                "x y : Nat",
                "⊢ q = p",
            ]
        )
        shape = derive_goal_shape(goal)
        self.assertEqual(shape.target, "q = p")
        self.assertEqual(shape.target_head, "eq")
        self.assertEqual(shape.local_names, ["h", "x", "y"])
        self.assertTrue(shape.has_equality)
        self.assertFalse(shape.has_iff)


class TestMoveMetadata(unittest.TestCase):
    def test_rw_forward_bare(self):
        goal_shape, trigger, subtask = derive_move_metadata(
            goal_state="h : a = b\n⊢ f a = f b",
            tactic_text="rw [h]",
            family="rw",
            canonical_action_ir="rw [h]",
            annotated_premise="h",
            step_index=0,
            prefix_tactics=[],
        )
        self.assertEqual(goal_shape.target_head, "eq")
        self.assertEqual(subtask.kind, "normalize_target_forward")
        kinds = {f.kind: f.value for f in trigger.features}
        self.assertEqual(kinds["rewrite_count"], "1")
        self.assertEqual(kinds["direction_prior"], "forward")
        self.assertEqual(kinds["argument_mode"], "bare")

    def test_rw_chain(self):
        _, trigger, subtask = derive_move_metadata(
            goal_state="⊢ a = c",
            tactic_text="rw [h1, h2]",
            family="rw",
            canonical_action_ir="rw [h1, h2]",
            annotated_premise="h1",
            step_index=0,
            prefix_tactics=[],
        )
        self.assertEqual(subtask.kind, "rewrite_chain")
        kinds = {f.kind for f in trigger.features}
        self.assertIn("composition", kinds)

    def test_rw_applied_args_maps_to_specialize(self):
        _, trigger, subtask = derive_move_metadata(
            goal_state="h : P x\n⊢ f x = g x",
            tactic_text="rw [foo h]",
            family="rw",
            canonical_action_ir="rw [foo h]",
            annotated_premise="foo",
            step_index=0,
            prefix_tactics=[],
        )
        self.assertEqual(subtask.kind, "specialize_rewrite")
        kinds = {f.kind: f.value for f in trigger.features}
        self.assertEqual(kinds["argument_mode"], "applied")

    def test_apply_maps_to_reduce_goal(self):
        _, trigger, subtask = derive_move_metadata(
            goal_state="⊢ P x",
            tactic_text="apply foo",
            family="apply",
            canonical_action_ir="apply foo",
            annotated_premise="foo",
            step_index=0,
            prefix_tactics=[],
        )
        self.assertEqual(subtask.kind, "reduce_goal_by_lemma")
        kinds = {f.kind for f in trigger.features}
        self.assertIn("term_shape", kinds)

    def test_rw_backward_is_backward_subtask(self):
        _, trigger, subtask = derive_move_metadata(
            goal_state="⊢ b = a",
            tactic_text="rw [← h]",
            family="rw",
            canonical_action_ir="rw [← h]",
            annotated_premise="h",
            step_index=0,
            prefix_tactics=[],
        )
        self.assertEqual(subtask.kind, "normalize_target_backward")
        kinds = {f.kind: f.value for f in trigger.features}
        self.assertEqual(kinds["direction_prior"], "backward")

    def test_action_ir_helpers_find_primary_premise(self):
        ir = canonicalize("apply foo.bar h", "apply")
        self.assertIsNotNone(ir)
        assert ir is not None
        self.assertEqual(ir.primary_premise_name(), "foo.bar")
        self.assertEqual(ir.local_var_names(), ["h"])


if __name__ == "__main__":
    unittest.main()
