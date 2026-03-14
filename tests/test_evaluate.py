"""Tests for src/evaluate.py — pure helper functions and NamedTuples (torch-free)."""

import unittest
from types import SimpleNamespace

from src.evaluate import _EvalPipeline, _infer_domain, _target_tier1_token, _VocabInfo

# ---------------------------------------------------------------------------
# _target_tier1_token
# ---------------------------------------------------------------------------


class TestTargetTier1Token(unittest.TestCase):
    """All edge cases for the BOS-skip token selector."""

    def test_no_attribute_returns_unk(self):
        """Missing tier1_tokens attribute -> '<UNK>'."""
        ex = SimpleNamespace()
        self.assertEqual(_target_tier1_token(ex), "<UNK>")

    def test_empty_list_returns_unk(self):
        """Empty tier1_tokens list -> '<UNK>'."""
        ex = SimpleNamespace(tier1_tokens=[])
        self.assertEqual(_target_tier1_token(ex), "<UNK>")

    def test_none_attribute_returns_unk(self):
        """tier1_tokens=None -> '<UNK>' (falsy)."""
        ex = SimpleNamespace(tier1_tokens=None)
        self.assertEqual(_target_tier1_token(ex), "<UNK>")

    def test_single_token_returns_that_token(self):
        """One token -> returns tokens[0] (no BOS to skip)."""
        ex = SimpleNamespace(tier1_tokens=["simp"])
        self.assertEqual(_target_tier1_token(ex), "simp")

    def test_two_tokens_returns_second(self):
        """Two tokens -> skip BOS at [0], return tokens[1]."""
        ex = SimpleNamespace(tier1_tokens=["BOS", "exact"])
        self.assertEqual(_target_tier1_token(ex), "exact")

    def test_many_tokens_returns_second(self):
        """Multiple tokens -> always returns tokens[1]."""
        ex = SimpleNamespace(tier1_tokens=["BOS", "apply", "intro", "simp", "EOS"])
        self.assertEqual(_target_tier1_token(ex), "apply")

    def test_three_tokens_returns_second(self):
        """Three tokens -> tokens[1]."""
        ex = SimpleNamespace(tier1_tokens=["a", "b", "c"])
        self.assertEqual(_target_tier1_token(ex), "b")

    def test_non_bos_two_tokens(self):
        """Even if first token is not 'BOS', still returns tokens[1]."""
        ex = SimpleNamespace(tier1_tokens=["intro", "exact"])
        self.assertEqual(_target_tier1_token(ex), "exact")


# ---------------------------------------------------------------------------
# _VocabInfo NamedTuple
# ---------------------------------------------------------------------------


class TestVocabInfo(unittest.TestCase):
    """Verify _VocabInfo construction and field access."""

    def test_construction_and_fields(self):
        tier1 = {"simp": 0, "apply": 1, "exact": 2}
        idx2tok = {0: "simp", 1: "apply", 2: "exact"}
        info = _VocabInfo(
            tier1_vocab=tier1,
            tier1_idx2token=idx2tok,
            tier2_vocab_size=128,
        )
        self.assertIs(info.tier1_vocab, tier1)
        self.assertIs(info.tier1_idx2token, idx2tok)
        self.assertEqual(info.tier2_vocab_size, 128)

    def test_is_namedtuple(self):
        info = _VocabInfo(
            tier1_vocab={},
            tier1_idx2token={},
            tier2_vocab_size=0,
        )
        self.assertIsInstance(info, tuple)
        self.assertEqual(len(info), 3)

    def test_field_names(self):
        self.assertEqual(
            _VocabInfo._fields,
            ("tier1_vocab", "tier1_idx2token", "tier2_vocab_size"),
        )

    def test_zero_tier2(self):
        info = _VocabInfo(tier1_vocab={"a": 0}, tier1_idx2token={0: "a"}, tier2_vocab_size=0)
        self.assertEqual(info.tier2_vocab_size, 0)


# ---------------------------------------------------------------------------
# _EvalPipeline NamedTuple
# ---------------------------------------------------------------------------


class TestEvalPipeline(unittest.TestCase):
    """Verify _EvalPipeline construction and field access."""

    def test_construction_and_fields(self):
        pipe = _EvalPipeline(
            encoder="enc",
            domain_gate="dg",
            goal_analyzer="ga",
            bridge="br",
            decoder="dec",
            checkpoint_meta={"step": 42},
        )
        self.assertEqual(pipe.encoder, "enc")
        self.assertEqual(pipe.domain_gate, "dg")
        self.assertEqual(pipe.goal_analyzer, "ga")
        self.assertEqual(pipe.bridge, "br")
        self.assertEqual(pipe.decoder, "dec")
        self.assertEqual(pipe.checkpoint_meta, {"step": 42})

    def test_field_names(self):
        self.assertEqual(
            _EvalPipeline._fields,
            ("encoder", "domain_gate", "goal_analyzer", "bridge", "decoder", "checkpoint_meta"),
        )

    def test_is_namedtuple(self):
        pipe = _EvalPipeline(
            encoder=None,
            domain_gate=None,
            goal_analyzer=None,
            bridge=None,
            decoder=None,
            checkpoint_meta={},
        )
        self.assertIsInstance(pipe, tuple)
        self.assertEqual(len(pipe), 6)


# ---------------------------------------------------------------------------
# _infer_domain (delegates to trainer_constants.infer_domain)
# ---------------------------------------------------------------------------


class TestInferDomain(unittest.TestCase):
    """Domain inference via keyword/tactic matching."""

    def test_algebra_keyword_in_goal(self):
        ex = SimpleNamespace(goal_state="G : Group A", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "algebra")

    def test_analysis_keyword_in_goal(self):
        ex = SimpleNamespace(goal_state="h : continuous f", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "analysis")

    def test_topology_keyword_in_goal(self):
        ex = SimpleNamespace(goal_state="h : IsCompact S", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "topology")

    def test_number_theory_keyword_in_statement(self):
        ex = SimpleNamespace(goal_state="", theorem_statement="theorem p_prime : prime p")
        self.assertEqual(_infer_domain(ex), "number_theory")

    def test_linear_algebra_keyword(self):
        ex = SimpleNamespace(goal_state="v : vector space", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "linear_algebra")

    def test_order_theory_keyword(self):
        ex = SimpleNamespace(goal_state="L : lattice A", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "order_theory")

    def test_set_theory_keyword(self):
        ex = SimpleNamespace(goal_state="h : A subset B", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "set_theory")

    def test_logic_keyword(self):
        ex = SimpleNamespace(goal_state="h : implies P Q", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "logic")

    def test_combinatorics_keyword(self):
        ex = SimpleNamespace(goal_state="h : permutation xs", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "combinatorics")

    def test_category_theory_keyword(self):
        ex = SimpleNamespace(goal_state="F : functor C D", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "category_theory")

    def test_tactic_hint_omega(self):
        """No keyword match, tactic 'omega' -> number_theory."""
        ex = SimpleNamespace(
            goal_state="n : Nat |- n = n",
            theorem_statement="",
            tier1_tokens=["omega"],
        )
        self.assertEqual(_infer_domain(ex), "number_theory")

    def test_tactic_hint_ring(self):
        """No keyword match, tactic 'ring' -> algebra."""
        ex = SimpleNamespace(
            goal_state="x : R |- x = x",
            theorem_statement="",
            tier1_tokens=["ring"],
        )
        self.assertEqual(_infer_domain(ex), "algebra")

    def test_tactic_hint_linarith(self):
        """No keyword match, tactic 'linarith' -> linear_algebra."""
        ex = SimpleNamespace(
            goal_state="h : a < b",
            theorem_statement="",
            tier1_tokens=["linarith"],
        )
        self.assertEqual(_infer_domain(ex), "linear_algebra")

    def test_tactic_hint_norm_num(self):
        """No keyword match, tactic 'norm_num' -> number_theory."""
        ex = SimpleNamespace(
            goal_state="|- (2 : Nat) + 3 = 5",
            theorem_statement="",
            tier1_tokens=["norm_num"],
        )
        self.assertEqual(_infer_domain(ex), "number_theory")

    def test_tactic_hint_field_simp(self):
        """No keyword match, tactic 'field_simp' -> algebra."""
        ex = SimpleNamespace(
            goal_state="x : F |- x / 1 = x",
            theorem_statement="",
            tier1_tokens=["field_simp"],
        )
        self.assertEqual(_infer_domain(ex), "algebra")

    def test_tactic_hint_positivity(self):
        """No keyword match, tactic 'positivity' -> analysis."""
        ex = SimpleNamespace(
            goal_state="|- 0 < 1",
            theorem_statement="",
            tier1_tokens=["positivity"],
        )
        self.assertEqual(_infer_domain(ex), "analysis")

    def test_fallback_general(self):
        """No keywords, no matching tactics -> 'general'."""
        ex = SimpleNamespace(
            goal_state="x : T |- x = x",
            theorem_statement="",
            tier1_tokens=["rfl"],
        )
        self.assertEqual(_infer_domain(ex), "general")

    def test_no_goal_no_statement_no_tactics(self):
        """Completely empty example -> 'general'."""
        ex = SimpleNamespace(goal_state="", theorem_statement="")
        self.assertEqual(_infer_domain(ex), "general")

    def test_keyword_takes_priority_over_tactic(self):
        """Keyword match wins even when tactic would suggest different domain."""
        ex = SimpleNamespace(
            goal_state="h : continuous f",
            theorem_statement="",
            tier1_tokens=["ring"],
        )
        # "continuous" matches analysis; tactic "ring" would give algebra
        self.assertEqual(_infer_domain(ex), "analysis")

    def test_missing_goal_attribute(self):
        """No goal_state attr at all -> still works (getattr default)."""
        ex = SimpleNamespace(theorem_statement="theorem p_prime : prime p")
        self.assertEqual(_infer_domain(ex), "number_theory")

    def test_none_goal_state(self):
        """goal_state=None -> handled by `or ''`."""
        ex = SimpleNamespace(goal_state=None, theorem_statement="")
        self.assertEqual(_infer_domain(ex), "general")


if __name__ == "__main__":
    unittest.main()
