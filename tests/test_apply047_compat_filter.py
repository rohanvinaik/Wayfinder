"""Tests for EXP-APPLY-047 compatibility filtering helpers."""

from scripts.run_apply047_compat_filter import (
    _goal_shape_from_ir,
    is_compatible,
    prop_shape,
)


def test_prop_shape_detects_implication():
    assert prop_shape("P → Q") == "imp"


def test_goal_shape_ir_extracts_implication():
    assert _goal_shape_from_ir({"has_implication": True}) == "imp"


def test_implication_goal_accepts_implication_candidate():
    cand_type = "∀ {α : Type}, P α → Q α"
    assert is_compatible("", "imp", cand_type)


def test_implication_goal_rejects_pure_equality_candidate():
    cand_type = "∀ x : Nat, x = x"
    assert not is_compatible("", "imp", cand_type)
