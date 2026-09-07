"""A proofs/ home lists its flat modules as `roots`; Lake has no bare-`*` glob for that."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wayfinder_mcp as m  # noqa: E402


def test_proofs_roots_sorted_deduplicated_and_blank_free():
    assert m.proofs_roots(["b", "a", "a", " ", ""]) == 'roots = ["a", "b"]'
    assert m.proofs_roots([]) == "roots = []"


def test_save_solved_proof_regenerates_the_lakefile_from_the_directory(tmp_path):
    home = tmp_path / "repo" / "proofs"
    home.mkdir(parents=True)
    (home / "Pabkit.lean").write_text("import Mathlib\n")
    row = {"theorem_id": "meanL_nonneg", "proof_home": str(home)}
    saved = m._save_solved_proof(row, "import Mathlib\ntheorem meanL_nonneg : True := trivial\n")
    assert saved == home / "meanL_nonneg.lean"
    lakefile = (home / "lakefile.toml").read_text()
    assert 'roots = ["Pabkit", "meanL_nonneg"]' in lakefile
    assert 'defaultTargets = ["Proofs"]' in lakefile  # a bare `lake build` (what CI runs) builds the library
    assert 'globs = ["*"]' not in lakefile
    assert (home / "lean-toolchain").read_text().strip() == f"leanprover/lean4:{m.LEAN_VERSION}"
    # a second save picks up the new module too
    m._save_solved_proof(
        {"theorem_id": "stability_nonneg", "proof_home": str(home)}, "theorem stability_nonneg : True := trivial\n"
    )
    assert 'roots = ["Pabkit", "meanL_nonneg", "stability_nonneg"]' in (home / "lakefile.toml").read_text()
