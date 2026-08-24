"""Intent tests for LeanExpr warranted emission (Wave 3).

Pins the emission contract: the value/expr coupling (lineage and source
built together, never drifting), the warrant header as evidence, the
environment as a first-class field with an explicit unpinned marker, the
sorry placeholder's visibility, and the projector-seam constructor's
direct-attribute discipline against the real ProjectedProofProgram.
"""

import tempfile
import unittest
from pathlib import Path

from src.dr_ducky import ProjectedProofProgram
from src.lean_expr import (
    EmissionWarrant,
    from_ducky_program,
    from_tactics,
    read_environment_pin,
    render_declaration,
    tactic_block,
    warrant_header,
)


class TacticBlockTest(unittest.TestCase):
    def test_lineage_renders_in_order_indented(self):
        self.assertEqual(tactic_block(["intro h", "simpa using h"]), "  intro h\n  simpa using h")

    def test_empty_lineage_is_a_visible_sorry(self):
        # Emission never invents content; the placeholder is explicit and the
        # sorryAx gate downstream rejects it.
        self.assertEqual(tactic_block([]), "  sorry")


class ValueExprCouplingTest(unittest.TestCase):
    def test_lineage_and_source_are_built_together(self):
        expr = from_tactics(["norm_cast", "exact h"], EmissionWarrant(theorem_id="T"))
        self.assertEqual(expr.value, ("norm_cast", "exact h"))
        self.assertEqual(expr.expr, tactic_block(expr.value))
        self.assertEqual(repr(expr), expr.expr)  # the source seam


class WarrantHeaderTest(unittest.TestCase):
    def test_header_carries_provenance_and_environment(self):
        warrant = EmissionWarrant(
            theorem_id="Nat.add_comm_demo",
            provenance=("structural_core", "symbolic_exact"),
            source_run="runs/demo.json",
        )
        header = warrant_header(warrant, "leanprover/lean4:v4.28.0 @ lean_project_v428")
        self.assertTrue(header.startswith("/- wayfinder warrant"))
        self.assertIn("structural_core -> symbolic_exact", header)
        self.assertIn("v4.28.0", header)
        self.assertIn("source_run: runs/demo.json", header)
        self.assertTrue(header.endswith("-/"))

    def test_empty_fields_are_omitted_not_blank(self):
        header = warrant_header(EmissionWarrant(theorem_id="T"), "")
        self.assertNotIn("engine:", header)
        self.assertNotIn("environment:", header)


class RenderDeclarationTest(unittest.TestCase):
    def _expr(self):
        return from_tactics(
            ["rfl"],
            EmissionWarrant(theorem_id="T", provenance=("symbolic_exact",)),
            imports=("import Mathlib",),
            opens=("open Nat",),
            environment="pin",
        )

    def test_section_order_is_warrant_imports_opens_declaration(self):
        text = render_declaration("theorem T : 1 = 1 := by", self._expr())
        idx = [
            text.index("/- wayfinder warrant"),
            text.index("import Mathlib"),
            text.index("open Nat"),
            text.index("theorem T : 1 = 1 := by\n  rfl"),
        ]
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(text.endswith("\n"))

    def test_warrant_can_be_omitted_for_probe_surfaces(self):
        text = render_declaration("theorem T : 1 = 1 := by", self._expr(), include_warrant=False)
        self.assertNotIn("warrant", text)
        self.assertIn("theorem T : 1 = 1 := by\n  rfl", text)


class DuckyProgramSeamTest(unittest.TestCase):
    def test_projector_fields_map_by_direct_attribute(self):
        program = ProjectedProofProgram(
            program_id="p1",
            bank="witness_constructor",
            specialist="witness_engine",
            skeleton_id="bounded_nat_search_use",
            tactics=["use 50", "omega"],
            rationale="bounded witness",
            score=0.84,
            certificate_id="c1",
            certificate_shape="witness_construction",
            bindings={},
            projector_backend="proof_projector_v1",
            derivation_path=[],
        )
        expr = from_ducky_program(program, theorem_id="T", environment="pin", source_run="r1")
        self.assertEqual(expr.value, ("use 50", "omega"))
        self.assertEqual(expr.warrant.bank, "witness_constructor")
        self.assertEqual(expr.warrant.certificate_id, "c1")
        self.assertEqual(expr.warrant.skeleton_id, "bounded_nat_search_use")
        self.assertEqual(expr.warrant.provenance, ("dr_ducky",))


class EnvironmentPinTest(unittest.TestCase):
    def test_real_project_pins_read(self):
        self.assertIn("v4.28.0", read_environment_pin("data/lean_project_v428"))
        self.assertIn("v4.27.0", read_environment_pin("data/lean_project"))

    def test_missing_toolchain_is_an_explicit_unpinned_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            pin = read_environment_pin(Path(tmp))
            self.assertTrue(pin.startswith("unpinned:"))


if __name__ == "__main__":
    unittest.main()
