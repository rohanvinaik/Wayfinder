"""Tests for extract_proof_network.py — pure extraction and position helpers."""

import unittest

from scripts.extract_proof_network import (
    _classify_domain,
    _collect_anchors,
    _compute_all_positions,
    _compute_automation_position,
    _compute_context_position,
    _compute_decomposition_position,
    _compute_depth_position,
    _compute_structure_position,
    _count_hypotheses,
    _extract_file_anchors,
    _extract_name_anchors,
    _extract_namespace_anchors,
    _extract_tactic_name,
    _extract_type_anchors,
    _extract_type_from_code,
    _extract_type_token_anchors,
    _infer_namespace,
    extract_entity,
    extract_premise_entity,
)


class TestExtractTacticName(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_extract_tactic_name("simp"), "simp")

    def test_with_args(self):
        self.assertEqual(_extract_tactic_name("apply Nat.add_comm"), "apply")

    def test_leading_paren(self):
        self.assertEqual(_extract_tactic_name("(intro h"), "intro")

    def test_empty(self):
        self.assertEqual(_extract_tactic_name(""), "")

    def test_whitespace_only(self):
        self.assertEqual(_extract_tactic_name("   "), "")

    def test_with_question_mark(self):
        self.assertEqual(_extract_tactic_name("exact?"), "exact?")

    def test_with_apostrophe(self):
        self.assertEqual(_extract_tactic_name("simp'"), "simp'")

    def test_leading_whitespace(self):
        self.assertEqual(_extract_tactic_name("  omega"), "omega")


class TestClassifyDomain(unittest.TestCase):
    def test_algebra(self):
        sign, anchors = _classify_domain("Mathlib.Algebra.Group")
        self.assertIsInstance(sign, int)
        self.assertIsInstance(anchors, tuple)
        self.assertTrue(len(anchors) > 0)

    def test_unknown_namespace(self):
        sign, anchors = _classify_domain("SomeRandom.Module")
        self.assertEqual(sign, 0)
        self.assertEqual(anchors, ("general",))

    def test_empty_namespace(self):
        sign, anchors = _classify_domain("")
        self.assertEqual(sign, 0)
        self.assertEqual(anchors, ("general",))


class TestComputeStructurePosition(unittest.TestCase):
    def test_simple_type_no_tactics(self):
        sign, depth = _compute_structure_position("Nat → Nat", [])
        self.assertEqual(sign, 0)  # no simplifiers or builders
        self.assertEqual(depth, 1)  # 1 arrow

    def test_complex_type(self):
        sign, depth = _compute_structure_position("∀ x y, x → y → x", [])
        self.assertEqual(depth, 3)  # 1 forall + 2 arrows = 3, capped at 3

    def test_simplifier_tactics(self):
        sign, depth = _compute_structure_position("Nat", ["simp", "norm_num"])
        self.assertEqual(sign, -1)

    def test_builder_tactics(self):
        sign, depth = _compute_structure_position("Nat", ["constructor", "intro"])
        self.assertEqual(sign, 1)

    def test_balanced(self):
        sign, _ = _compute_structure_position("Nat", ["simp", "constructor"])
        self.assertEqual(sign, 0)

    def test_depth_capped_at_3(self):
        # 10 arrows → still depth 3
        _, depth = _compute_structure_position("→" * 10, [])
        self.assertEqual(depth, 3)


class TestComputeDepthPosition(unittest.TestCase):
    def test_single_step(self):
        sign, depth = _compute_depth_position(1)
        self.assertEqual(sign, -1)
        self.assertEqual(depth, 2)

    def test_short_proof(self):
        sign, depth = _compute_depth_position(3)
        self.assertEqual(sign, -1)
        self.assertEqual(depth, 1)

    def test_medium_proof(self):
        sign, depth = _compute_depth_position(5)
        self.assertEqual(sign, 0)
        self.assertEqual(depth, 0)

    def test_long_proof(self):
        sign, depth = _compute_depth_position(10)
        self.assertEqual(sign, 1)
        self.assertEqual(depth, 1)

    def test_very_long_proof(self):
        sign, depth = _compute_depth_position(20)
        self.assertEqual(sign, 1)
        self.assertEqual(depth, 2)

    def test_zero_length(self):
        sign, depth = _compute_depth_position(0)
        self.assertEqual(sign, -1)
        self.assertEqual(depth, 2)

    def test_boundary_at_8(self):
        sign_8, depth_8 = _compute_depth_position(8)
        sign_9, depth_9 = _compute_depth_position(9)
        self.assertEqual(sign_8, 0)
        self.assertEqual(depth_8, 0)
        self.assertEqual(sign_9, 1)
        self.assertEqual(depth_9, 1)


class TestComputeAutomationPosition(unittest.TestCase):
    def test_all_auto(self):
        sign, _ = _compute_automation_position(["omega", "decide", "simp"])
        self.assertEqual(sign, -1)

    def test_all_manual(self):
        sign, _ = _compute_automation_position(["intro", "apply", "exact"])
        self.assertEqual(sign, 1)

    def test_empty(self):
        # auto=0, manual=0, total=max(0,1)=1, ratio=0/1=0.0 < 0.3 → sign=1
        sign, depth = _compute_automation_position([])
        self.assertEqual(sign, 1)
        self.assertEqual(depth, 1)  # min(max(0,1), 3) = 1

    def test_depth_capped_at_3(self):
        tactics = ["omega"] * 10
        _, depth = _compute_automation_position(tactics)
        self.assertEqual(depth, 3)


class TestComputeContextPosition(unittest.TestCase):
    def test_enrichers_dominate(self):
        sign, _ = _compute_context_position(2, ["have", "let"])
        self.assertEqual(sign, 1)

    def test_reducers_dominate(self):
        # "clear" is a CONTEXT_REDUCER, "revert" may not be — check actual sets
        from scripts.tactic_maps import CONTEXT_REDUCERS

        # Use known reducers
        reducers = list(CONTEXT_REDUCERS)[:2] if len(CONTEXT_REDUCERS) >= 2 else ["clear"]
        sign, _ = _compute_context_position(2, reducers)
        self.assertEqual(sign, -1)

    def test_balanced(self):
        sign, _ = _compute_context_position(2, [])
        self.assertEqual(sign, 0)

    def test_depth_is_hypothesis_count(self):
        _, depth = _compute_context_position(2, [])
        self.assertEqual(depth, 2)

    def test_depth_capped_at_3(self):
        _, depth = _compute_context_position(10, [])
        self.assertEqual(depth, 3)


class TestComputeDecompositionPosition(unittest.TestCase):
    def test_splitters_dominate(self):
        sign, _ = _compute_decomposition_position(["cases", "induction"])
        self.assertEqual(sign, 1)

    def test_closers_dominate(self):
        sign, _ = _compute_decomposition_position(["exact", "assumption"])
        self.assertEqual(sign, -1)

    def test_empty(self):
        sign, depth = _compute_decomposition_position([])
        self.assertEqual(sign, 0)
        self.assertEqual(depth, 0)


class TestCountHypotheses(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_count_hypotheses(""), 0)

    def test_no_hypotheses(self):
        self.assertEqual(_count_hypotheses("⊢ True"), 0)

    def test_one_hypothesis(self):
        self.assertEqual(_count_hypotheses("h : Nat\n⊢ True"), 1)

    def test_two_hypotheses(self):
        self.assertEqual(_count_hypotheses("h1 : Nat\nh2 : Bool\n⊢ True"), 2)

    def test_with_turnstile_arrow(self):
        self.assertEqual(_count_hypotheses("h : Nat\n|- True"), 1)


class TestInferNamespace(unittest.TestCase):
    def test_qualified(self):
        result = _infer_namespace("Mathlib.Algebra.Group.add_comm")
        self.assertEqual(result, "Mathlib.Algebra.Group")

    def test_no_namespace(self):
        self.assertEqual(_infer_namespace("add_comm"), "")

    def test_deep_nesting(self):
        self.assertEqual(
            _infer_namespace("Mathlib.Order.Basic.le_refl"),
            "Mathlib.Order.Basic",
        )


class TestExtractTypeAnchors(unittest.TestCase):
    def test_arrow_type(self):
        anchors = _extract_type_anchors("Nat → Nat")
        self.assertIn("implication", anchors)

    def test_universal(self):
        anchors = _extract_type_anchors("∀ x, x = x")
        self.assertIn("universal-quantifier", anchors)
        self.assertIn("equality", anchors)

    def test_nat_arithmetic(self):
        anchors = _extract_type_anchors("Nat.add_comm")
        self.assertIn("nat-arithmetic", anchors)

    def test_empty_type(self):
        anchors = _extract_type_anchors("")
        self.assertEqual(anchors, [])


class TestExtractNamespaceAnchors(unittest.TestCase):
    def test_multi_level(self):
        anchors = _extract_namespace_anchors("Mathlib.Algebra.Group.Basic")
        self.assertEqual(anchors[0], "ns:Mathlib")
        self.assertEqual(anchors[1], "ns:Mathlib.Algebra")
        self.assertEqual(anchors[2], "ns:Mathlib.Algebra.Group")
        self.assertEqual(anchors[3], "ns:Mathlib.Algebra.Group.Basic")
        self.assertEqual(len(anchors), 4)

    def test_empty(self):
        self.assertEqual(_extract_namespace_anchors(""), [])

    def test_single_level(self):
        anchors = _extract_namespace_anchors("Init")
        self.assertEqual(anchors, ["ns:Init"])

    def test_capped_at_4_levels(self):
        anchors = _extract_namespace_anchors("A.B.C.D.E.F")
        self.assertEqual(len(anchors), 4)


class TestExtractNameAnchors(unittest.TestCase):
    def test_underscore_split(self):
        anchors = _extract_name_anchors("Mathlib.Algebra.mul_comm")
        # "mul" and "comm" should appear
        self.assertIn("name:mul", anchors)
        self.assertIn("name:comm", anchors)

    def test_camel_case(self):
        anchors = _extract_name_anchors("addCommGroup")
        self.assertIn("name:add", anchors)
        self.assertIn("name:comm", anchors)

    def test_short_fragments_skipped(self):
        anchors = _extract_name_anchors("is_lt")
        # "lt" is only 2 chars → skipped
        self.assertNotIn("name:lt", anchors)

    def test_sorted_output(self):
        anchors = _extract_name_anchors("Mathlib.zeta_beta_alpha")
        self.assertEqual(anchors, sorted(anchors))


class TestExtractTypeTokenAnchors(unittest.TestCase):
    def test_type_tokens(self):
        anchors = _extract_type_token_anchors("Finset.sum (Group α) = Module β")
        self.assertIn("type:Finset", anchors)
        self.assertIn("type:Group", anchors)
        self.assertIn("type:Module", anchors)

    def test_short_tokens_skipped(self):
        anchors = _extract_type_token_anchors("Eq x y")
        self.assertNotIn("type:Eq", anchors)  # "Eq" is only 2 chars

    def test_empty(self):
        self.assertEqual(_extract_type_token_anchors(""), [])


class TestExtractFileAnchors(unittest.TestCase):
    def test_file_anchor(self):
        anchors = _extract_file_anchors("Mathlib/Algebra/Group/Basic.lean")
        self.assertIn("file:Mathlib/Algebra/Group/Basic", anchors)
        self.assertIn("dir:Mathlib/Algebra", anchors)

    def test_empty(self):
        self.assertEqual(_extract_file_anchors(""), [])


class TestComputeAllPositions(unittest.TestCase):
    def test_returns_six_banks(self):
        positions, _, _ = _compute_all_positions("Nat → Nat", ["simp"], "Mathlib.Algebra", [])
        expected_banks = {
            "structure",
            "domain",
            "depth",
            "automation",
            "context",
            "decomposition",
        }
        self.assertEqual(set(positions.keys()), expected_banks)

    def test_each_position_has_sign_and_depth(self):
        positions, _, _ = _compute_all_positions("Nat → Nat", ["simp"], "Mathlib.Algebra", [])
        for bank, pos in positions.items():
            self.assertIn("sign", pos, f"{bank} missing sign")
            self.assertIn("depth", pos, f"{bank} missing depth")
            self.assertIn(pos["sign"], (-1, 0, 1), f"{bank} sign out of range")
            self.assertGreaterEqual(pos["depth"], 0, f"{bank} negative depth")
            self.assertLessEqual(pos["depth"], 3, f"{bank} depth > 3")


class TestCollectAnchors(unittest.TestCase):
    def test_deduplication(self):
        anchors = _collect_anchors(
            ["algebra", "algebra"],
            "Nat → Nat",
            ["simp"],
            {"namespace": "Mathlib.Algebra", "theorem_id": "add_comm", "file_path": ""},
        )
        self.assertEqual(len(anchors), len(set(anchors)))

    def test_sorted_output(self):
        anchors = _collect_anchors(
            ["general"],
            "Nat",
            [],
            {"namespace": "", "theorem_id": "test", "file_path": ""},
        )
        self.assertEqual(anchors, sorted(anchors))


class TestExtractEntity(unittest.TestCase):
    def test_minimal_theorem(self):
        theorem = {
            "theorem_id": "test.add_zero",
            "theorem_statement": "∀ n : Nat, n + 0 = n",
            "tactics": ["omega"],
            "goal_states": ["n : Nat\n⊢ n + 0 = n"],
            "premises": ["Nat.add_zero"],
            "namespace": "test",
            "file_path": "Test/Basic.lean",
        }
        entity = extract_entity(theorem)
        self.assertEqual(entity["theorem_id"], "test.add_zero")
        self.assertEqual(entity["entity_type"], "lemma")
        self.assertEqual(entity["namespace"], "test")
        self.assertEqual(entity["premises"], ["Nat.add_zero"])
        self.assertEqual(entity["tactic_names"], ["omega"])
        self.assertEqual(entity["proof_length"], 1)
        self.assertIn("positions", entity)
        self.assertIn("anchors", entity)
        # positions has all 6 banks
        self.assertEqual(len(entity["positions"]), 6)

    def test_empty_theorem(self):
        entity = extract_entity({})
        self.assertEqual(entity["theorem_id"], "")
        self.assertEqual(entity["entity_type"], "lemma")
        self.assertEqual(entity["proof_length"], 0)


class TestExtractTypeFromCode(unittest.TestCase):
    """Tests for _extract_type_from_code (Stage 1 premise expansion)."""

    def test_theorem_declaration(self):
        code = "theorem Nat.add_comm (n m : Nat) : n + m = m + n := by omega"
        result = _extract_type_from_code(code)
        self.assertEqual(result, "n + m = m + n")

    def test_def_declaration(self):
        code = "def id {α : Sort u} (a : α) : α := a"
        result = _extract_type_from_code(code)
        self.assertEqual(result, "α")

    def test_no_colon_returns_empty(self):
        result = _extract_type_from_code("some text without colon")
        self.assertEqual(result, "")

    def test_where_clause_trimmed(self):
        code = "class Foo (α : Type) : Prop where\n  bar : α → α"
        result = _extract_type_from_code(code)
        self.assertEqual(result, "Prop")

    def test_empty_code(self):
        result = _extract_type_from_code("")
        self.assertEqual(result, "")

    def test_colon_inside_braces_skipped(self):
        code = "def f {α : Type} : α → α := id"
        result = _extract_type_from_code(code)
        # Should skip the colon inside {α : Type} and find the outer one
        self.assertEqual(result, "α → α")


class TestExtractPremiseEntity(unittest.TestCase):
    """Tests for extract_premise_entity (Stage 1 premise expansion)."""

    def test_basic_structure(self):
        entity = extract_premise_entity(
            full_name="Nat.add_comm",
            code="theorem Nat.add_comm (n m : Nat) : n + m = m + n := by omega",
            kind="commanddeclaration",
            file_path="Mathlib/Data/Nat/Basic.lean",
        )
        self.assertEqual(entity["theorem_id"], "Nat.add_comm")
        self.assertEqual(entity["entity_type"], "lemma")
        self.assertEqual(entity["provenance"], "premise_only")
        self.assertEqual(entity["declaration_kind"], "commanddeclaration")
        self.assertEqual(entity["tactic_names"], [])
        self.assertEqual(entity["premises"], [])
        self.assertEqual(entity["proof_length"], 0)

    def test_namespace_inferred(self):
        entity = extract_premise_entity(
            full_name="Mathlib.Order.Basic.le_refl",
            code="theorem le_refl : ∀ a, a ≤ a := fun a => le_refl a",
            kind="commanddeclaration",
            file_path="",
        )
        self.assertEqual(entity["namespace"], "Mathlib.Order.Basic")

    def test_positions_have_all_six_banks(self):
        entity = extract_premise_entity(
            full_name="test",
            code="def test : Nat := 0",
            kind="commanddeclaration",
            file_path="",
        )
        expected_banks = {"structure", "domain", "depth", "automation", "context", "decomposition"}
        self.assertEqual(set(entity["positions"].keys()), expected_banks)

    def test_tactic_banks_are_zero(self):
        """Banks requiring tactic info should be zero (Informational Zero)."""
        entity = extract_premise_entity(
            full_name="test",
            code="def test : Nat := 0",
            kind="commanddeclaration",
            file_path="",
        )
        for bank in ["depth", "automation", "context", "decomposition"]:
            self.assertEqual(entity["positions"][bank]["sign"], 0)
            self.assertEqual(entity["positions"][bank]["depth"], 0)

    def test_anchors_include_namespace(self):
        entity = extract_premise_entity(
            full_name="Mathlib.Topology.Basic.IsOpen",
            code="class IsOpen (s : Set α) : Prop",
            kind="commanddeclaration",
            file_path="Mathlib/Topology/Basic.lean",
        )
        ns_anchors = [a for a in entity["anchors"] if a.startswith("ns:")]
        self.assertTrue(len(ns_anchors) > 0)

    def test_anchors_include_name_fragments(self):
        entity = extract_premise_entity(
            full_name="AEMeasurable.comp_measurable",
            code="theorem AEMeasurable.comp_measurable : True := trivial",
            kind="commanddeclaration",
            file_path="",
        )
        name_anchors = [a for a in entity["anchors"] if a.startswith("name:")]
        self.assertTrue(len(name_anchors) > 0)
        self.assertIn("name:measurable", name_anchors)


if __name__ == "__main__":
    unittest.main()
