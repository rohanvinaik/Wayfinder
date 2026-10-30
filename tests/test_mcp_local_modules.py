"""A calling repo's definitions module is inlined into the hosted-prover hand-off, not imported."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wayfinder_mcp as m  # noqa: E402


def test_split_imports_partitions_external_and_local_in_order():
    parts = m.split_imports(["import Mathlib", "import Pabkit", "import Other"], ["Pabkit", "Other"])
    assert parts == {"external": ["import Mathlib"], "local": ["Pabkit", "Other"]}


def test_split_imports_tolerates_bare_names_blanks_and_duplicates():
    parts = m.split_imports(["Mathlib", " import Pabkit ", "", None, "import Pabkit"], ["Pabkit"])
    assert parts == {"external": ["import Mathlib"], "local": ["Pabkit"]}


def test_split_imports_with_no_local_modules_keeps_everything_external():
    assert m.split_imports(["import Mathlib"], []) == {"external": ["import Mathlib"], "local": []}
    assert m.split_imports([], []) == {"external": [], "local": []}


def test_strip_import_lines_drops_only_import_lines():
    src = "/- header -/\nimport Mathlib\n\nopen Matrix\n\nnoncomputable def f (x : ℝ) : ℝ := x\n"
    assert m.strip_import_lines(src) == "/- header -/\n\nopen Matrix\n\nnoncomputable def f (x : ℝ) : ℝ := x"


def test_proof_lean_inlines_the_local_module_and_keeps_mathlib(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "Pabkit.lean").write_text(
        "import Mathlib\n\nopen Matrix\n\nnoncomputable def meanL (xs : List ℝ) : ℝ := 0\n"
    )
    (lib / "PabkitTheorems.lean").write_text(
        "import Mathlib\nimport Pabkit\n\ntheorem t (xs : List ℝ) : 0 ≤ meanL xs := by\n  sorry\n"
    )
    monkeypatch.setattr(m, "LEAN_PROJECT", tmp_path)
    row = {
        "theorem_id": "t",
        "imports": ["import Mathlib", "import Pabkit"],
        "opens": [],
        "file_path": "lib/PabkitTheorems.lean",
        "full_declaration": "theorem t (xs : List ℝ) : 0 ≤ meanL xs := by\n  sorry",
    }
    out = m._proof_lean(row)
    assert out.startswith("import Mathlib\n")
    assert "import Pabkit" not in out
    assert "noncomputable def meanL" in out
    assert out.index("open Matrix") < out.index("theorem t")
    assert out.rstrip().endswith("sorry")


def test_proof_lean_without_local_modules_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LEAN_PROJECT", tmp_path)
    row = {
        "theorem_id": "t",
        "imports": ["import Mathlib"],
        "opens": ["open Finset"],
        "file_path": "missing.lean",
        "full_declaration": "theorem t : True := by\n  sorry",
    }
    assert m._proof_lean(row) == "import Mathlib\nopen Finset\ntheorem t : True := by\n  sorry\n"


_PREAMBLE_SOURCE = """import Mathlib

open Finset

section Scratch
variable {β : Type*} [Inhabited β]
end Scratch

variable {α : Type*} [DecidableEq α]

def total (s : Finset ℕ) : ℕ := s.sum id

theorem target_thm (s : Finset α) : s ⊆ s := by
  sorry

variable {γ : Type*}

def after (n : ℕ) : ℕ := n + 1
"""


def test_the_preamble_carries_the_variables_in_scope_at_the_theorem(tmp_path, monkeypatch):
    (tmp_path / "src.lean").write_text(_PREAMBLE_SOURCE)
    monkeypatch.setattr(m, "LEAN_PROJECT", tmp_path)
    pre = m._def_preamble({"file_path": "src.lean", "theorem_line": 13})
    assert "variable {α : Type*} [DecidableEq α]" in pre
    assert pre.index("variable {α") < pre.index("def total")  # source order
    assert "Inhabited β" not in pre  # its section closed before the theorem
    assert "variable {γ" not in pre  # declared after the theorem
    assert "def after" in pre  # defs are carried wherever they sit, as before


def test_a_missing_source_gives_no_preamble(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LEAN_PROJECT", tmp_path)
    assert m._def_preamble({"file_path": "nope.lean"}) == ""

