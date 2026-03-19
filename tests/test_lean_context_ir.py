"""Tests for the Lean source-context parser."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.lean_context_ir import classify_context_directive, extract_context_ir, find_decl_line


class TestClassifyContextDirective(unittest.TestCase):
    def test_recognizes_open_scoped(self):
        kind, inline_only = classify_context_directive("open scoped MeasureTheory")
        self.assertEqual(kind, "open_scoped")
        self.assertFalse(inline_only)

    def test_marks_inline_only(self):
        kind, inline_only = classify_context_directive("variable (R : Type*) in")
        self.assertEqual(kind, "variable")
        self.assertTrue(inline_only)

    def test_ignores_non_directive(self):
        kind, inline_only = classify_context_directive("theorem foo : True := by")
        self.assertIsNone(kind)
        self.assertFalse(inline_only)


class TestExtractContextIR(unittest.TestCase):
    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False)
        tmp.write(textwrap.dedent(content))
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_extracts_scopes_and_directives(self):
        path = self._write(
            """
            namespace Foo
            open scoped Bar
            section Baz
            variable {α : Type*}
            local notation "ZZ" => Nat
            attribute [local simp] foo
            include α
            theorem demo : True := by
              trivial
            end Baz
            end Foo
            """
        )
        line = find_decl_line(path, "demo")
        ir = extract_context_ir(path, line)
        self.assertEqual(
            ir.prefix_lines,
            [
                "namespace Foo",
                "open scoped Bar",
                "section Baz",
                "variable {α : Type*}",
                'local notation "ZZ" => Nat',
                "attribute [local simp] foo",
                "include α",
            ],
        )
        self.assertEqual(ir.suffix_lines, ["end Baz", "end Foo"])
        counts = ir.feature_counts()
        self.assertEqual(counts["open_scoped"], 1)
        self.assertEqual(counts["local_notation"], 1)
        self.assertEqual(counts["local_attribute"], 1)
        self.assertEqual(counts["include"], 1)

    def test_records_inline_only_directive_as_unsupported(self):
        path = self._write(
            """
            section
            variable (R : Type*) in
            theorem demo : True := by
              trivial
            end
            """
        )
        line = find_decl_line(path, "demo")
        ir = extract_context_ir(path, line)
        self.assertEqual(ir.prefix_lines, ["section"])
        self.assertEqual(ir.suffix_lines, ["end"])
        self.assertEqual(len(ir.unsupported), 1)
        self.assertEqual(ir.unsupported[0].kind, "variable")
        self.assertEqual(ir.unsupported[0].reason, "inline_next_decl_only")

    def test_find_decl_line(self):
        path = self._write(
            """
            theorem first : True := by
              trivial

            lemma second : True := by
              trivial
            """
        )
        self.assertEqual(find_decl_line(path, "first"), 2)
        self.assertEqual(find_decl_line(path, "second"), 5)
        self.assertEqual(find_decl_line(path, "missing"), -1)


if __name__ == "__main__":
    unittest.main()
