"""The search kernel imports what the manifest rows declare, not a hard-coded `Mathlib`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wayfinder_mcp import _lean_imports  # noqa: E402


def test_rows_declare_their_own_modules_mathlib_first():
    rows = [
        {"imports": ["import Mathlib", "import Pabkit"]},
        {"imports": ["import Pabkit", "import Other"]},
    ]
    assert _lean_imports(rows) == ["Mathlib", "Pabkit", "Other"]


def test_mathlib_is_moved_first_even_when_declared_later():
    assert _lean_imports([{"imports": ["import Pabkit", "import Mathlib"]}]) == ["Mathlib", "Pabkit"]


def test_no_declared_imports_falls_back_to_mathlib():
    assert _lean_imports([{}]) == ["Mathlib"]
    assert _lean_imports([]) == ["Mathlib"]
    assert _lean_imports([{"imports": []}, {"imports": None}]) == ["Mathlib"]


def test_bare_module_names_and_blank_lines_are_accepted():
    assert _lean_imports([{"imports": ["Mathlib", " import Pabkit ", ""]}]) == ["Mathlib", "Pabkit"]
