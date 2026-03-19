"""Tests for auxiliary move supervision vocab/target builders."""

from __future__ import annotations

import unittest

import torch

from src.move_supervision import (
    MoveSupervisionSpec,
    build_move_supervision_spec,
    build_template_move_supervision_spec,
    build_template_move_targets,
    build_move_targets,
    compute_move_metrics,
)
from src.nav_contracts import NavigationalExample


def _example(**metadata) -> NavigationalExample:
    return NavigationalExample(
        goal_state="⊢ P",
        theorem_id="Foo.bar",
        nav_directions={"structure": 0},
        metadata=metadata,
    )


class TestMoveSupervisionSpec(unittest.TestCase):
    def test_build_spec_collects_vocab(self):
        examples = [
            _example(
                subtask_kind="rewrite_chain",
                goal_target_head="eq",
                trigger_signature=["target_head=eq", "rewrite_count=2"],
            ),
            _example(
                subtask_kind="rewrite_chain",
                goal_target_head="eq",
                trigger_signature=["target_head=eq"],
            ),
            _example(
                subtask_kind="close_with_term",
                goal_target_head="False",
                trigger_signature=["term_shape=const"],
            ),
        ]
        spec = build_move_supervision_spec(
            examples,
            subtask_min_support=1,
            goal_head_min_support=1,
            trigger_min_support=1,
            max_goal_heads=8,
            max_trigger_signatures=8,
        )
        self.assertIn("rewrite_chain", spec.subtask_vocab)
        self.assertIn("eq", spec.goal_head_vocab)
        self.assertIn("target_head=eq", spec.trigger_signature_vocab)

    def test_build_targets_masks_unknowns(self):
        spec = MoveSupervisionSpec(
            subtask_vocab=["rewrite_chain"],
            goal_head_vocab=["eq"],
            trigger_signature_vocab=["target_head=eq"],
        )
        examples = [
            _example(
                subtask_kind="rewrite_chain",
                goal_target_head="eq",
                trigger_signature=["target_head=eq"],
            ),
            _example(
                subtask_kind="other",
                goal_target_head="False",
                trigger_signature=["unknown=1"],
            ),
        ]
        targets, masks, target_types = build_move_targets(examples, spec, "cpu")
        self.assertEqual(target_types["subtask_kind"], "multiclass")
        self.assertTrue(bool(masks["subtask_kind"][0].item()))
        self.assertFalse(bool(masks["subtask_kind"][1].item()))
        self.assertEqual(int(targets["subtask_kind"][0].item()), 0)
        self.assertEqual(float(targets["trigger_signature"][0, 0].item()), 1.0)
        self.assertEqual(float(targets["trigger_signature"][1, 0].item()), 0.0)

    def test_build_spec_can_filter_to_descriptor_heads(self):
        examples = [
            _example(
                subtask_kind="rewrite_chain",
                goal_target_head="eq",
                trigger_signature=["target_head=eq", "rewrite_count=2"],
            ),
            _example(
                subtask_kind="close_with_term",
                goal_target_head="False",
                trigger_signature=["term_shape=const"],
            ),
        ]
        spec = build_move_supervision_spec(
            examples,
            enabled_heads=["goal_target_head", "trigger_signature"],
            subtask_min_support=1,
            goal_head_min_support=1,
            trigger_min_support=1,
            max_goal_heads=8,
            max_trigger_signatures=8,
        )
        self.assertEqual(spec.subtask_vocab, [])
        self.assertIn("eq", spec.goal_head_vocab)
        self.assertIn("target_head=eq", spec.trigger_signature_vocab)

    def test_compute_move_metrics(self):
        spec = MoveSupervisionSpec(
            subtask_vocab=["rewrite_chain"],
            goal_head_vocab=["eq"],
            trigger_signature_vocab=["target_head=eq"],
        )
        examples = [
            _example(
                subtask_kind="rewrite_chain",
                goal_target_head="eq",
                trigger_signature=["target_head=eq"],
            )
        ]
        targets, masks, target_types = build_move_targets(examples, spec, "cpu")
        logits = {
            "subtask_kind": torch.tensor([[3.0]]),
            "goal_target_head": torch.tensor([[2.0]]),
            "trigger_signature": torch.tensor([[5.0]]),
        }
        metrics = compute_move_metrics(logits, targets, masks, target_types)
        self.assertEqual(metrics["subtask_kind_acc"], 1.0)
        self.assertEqual(metrics["goal_target_head_acc"], 1.0)
        self.assertEqual(metrics["trigger_signature_micro_acc"], 1.0)

    def test_template_spec_and_targets(self):
        rows = [
            {
                "template_move_profile": {
                    "dominant_subtask_kind": "rewrite_chain",
                    "top_target_heads": [{"value": "eq", "count": 3}],
                    "top_trigger_signatures": [
                        {"value": "target_head=eq", "count": 3},
                        {"value": "sequential_context=true", "count": 2},
                    ],
                }
            },
            {
                "template_move_profile": {
                    "dominant_subtask_kind": "close_with_term",
                    "top_target_heads": [{"value": "False", "count": 2}],
                    "top_trigger_signatures": [{"value": "term_shape=const", "count": 2}],
                }
            },
        ]
        spec = build_template_move_supervision_spec(
            rows,
            subtask_min_support=1,
            goal_head_min_support=1,
            trigger_min_support=1,
            max_goal_heads=8,
            max_trigger_signatures=8,
        )
        self.assertIn("rewrite_chain", spec.subtask_vocab)
        self.assertIn("eq", spec.goal_head_vocab)
        self.assertIn("target_head=eq", spec.trigger_signature_vocab)

        targets, masks, target_types = build_template_move_targets(rows, spec, "cpu")
        self.assertEqual(target_types["subtask_kind"], "multiclass")
        self.assertTrue(bool(masks["subtask_kind"][0].item()))
        self.assertEqual(float(targets["trigger_signature"][0].sum().item()), 2.0)

    def test_template_spec_can_filter_out_subtask(self):
        rows = [
            {
                "template_move_profile": {
                    "dominant_subtask_kind": "rewrite_chain",
                    "top_target_heads": [{"value": "eq", "count": 3}],
                    "top_trigger_signatures": [{"value": "target_head=eq", "count": 3}],
                }
            }
        ]
        spec = build_template_move_supervision_spec(
            rows,
            enabled_heads=["goal_target_head", "trigger_signature"],
            subtask_min_support=1,
            goal_head_min_support=1,
            trigger_min_support=1,
        )
        self.assertEqual(spec.subtask_vocab, [])
        self.assertEqual(spec.goal_head_vocab, ["eq"])
        self.assertEqual(spec.trigger_signature_vocab, ["target_head=eq"])


if __name__ == "__main__":
    unittest.main()
