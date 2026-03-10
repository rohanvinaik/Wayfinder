"""Tests for evaluate.py — pure helper functions (torch-free)."""

import unittest
from types import SimpleNamespace

from src.evaluate import _target_tier1_token


class TestTargetTier1Token(unittest.TestCase):
    def test_empty_tokens_returns_unk(self):
        ex = SimpleNamespace(tier1_tokens=[])
        self.assertEqual(_target_tier1_token(ex), "<UNK>")

    def test_missing_attribute_returns_unk(self):
        ex = SimpleNamespace()
        self.assertEqual(_target_tier1_token(ex), "<UNK>")

    def test_single_token_returns_it(self):
        ex = SimpleNamespace(tier1_tokens=["simp"])
        self.assertEqual(_target_tier1_token(ex), "simp")

    def test_two_tokens_returns_second(self):
        ex = SimpleNamespace(tier1_tokens=["apply", "exact"])
        self.assertEqual(_target_tier1_token(ex), "exact")

    def test_many_tokens_returns_second(self):
        ex = SimpleNamespace(tier1_tokens=["a", "b", "c", "d"])
        self.assertEqual(_target_tier1_token(ex), "b")


if __name__ == "__main__":
    unittest.main()
