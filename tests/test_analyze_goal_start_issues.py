from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analyze_goal_start_issues import _issue_tags, _source_scan


class TestAnalyzeGoalStartIssues(unittest.TestCase):
    def test_source_scan_detects_context_patterns(self) -> None:
        content = """\
section
variable
  {R : Type*}
open scoped BigOperators
scoped notation "Foo" => Nat
local notation "bar" => Nat.succ
  0
local instance : Inhabited Nat := ⟨0⟩
attribute [local instance] Nat.instInhabitedNat
include R
omit R
theorem demo : True := by
  trivial
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lean_path = Path(tmpdir) / "Demo.lean"
            lean_path.write_text(content)
            scan = _source_scan(str(lean_path), 12, 20)

        self.assertEqual(scan.multiline_variable_blocks, 1)
        self.assertEqual(scan.open_scoped_lines, 1)
        self.assertEqual(scan.scoped_notation_lines, 1)
        self.assertEqual(scan.local_notation_lines, 1)
        self.assertEqual(scan.local_instance_lines, 1)
        self.assertEqual(scan.attribute_instance_lines, 1)
        self.assertEqual(scan.include_lines, 1)
        self.assertEqual(scan.omit_lines, 1)

    def test_issue_tags_surface_likely_context_gaps(self) -> None:
        row = {
            "lean_path": "",
            "theorem_line": 0,
            "failure_category": "typeclass_missing",
            "feedback": {"category": "typeclass_missing"},
            "repair_feedback": {"category": ""},
            "theorem_type": (
                "∀ {R : Type u_1} [Semiring R] [Module R M] [SMul R M] "
                "[TopologicalSpace M], True"
            ),
            "repair_attempted": True,
            "repair_success": False,
        }
        unsupported = [
            {"kind": "variable", "reason": "probable_multiline", "text": "variable"},
            {
                "kind": "open_scoped",
                "reason": "inline_next_decl_only",
                "text": "open scoped Foo in",
            },
        ]
        scan = _source_scan("", 0, 20)
        scan.scoped_notation_lines = 1
        scan.multiline_variable_blocks = 1
        scan.instance_lines = 1
        tags = _issue_tags(row, scan, unsupported)

        self.assertIn("missing_source_metadata", tags)
        self.assertIn("typeclass_elaboration_failure", tags)
        self.assertIn("universe_polymorphic_statement", tags)
        self.assertIn("heavy_typeclass_statement", tags)
        self.assertIn("multiline_variable_context", tags)
        self.assertIn("inline_open_scoped_context", tags)
        self.assertIn("scoped_notation_context", tags)
        self.assertIn("same_file_instance_support", tags)
        self.assertIn("repair_exhausted", tags)


if __name__ == "__main__":
    unittest.main()
