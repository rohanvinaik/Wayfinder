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
    saved, note = m._save_solved_proof(row, "import Mathlib\ntheorem meanL_nonneg : True := trivial\n")
    assert saved == home / "meanL_nonneg.lean" and note == ""
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


_CURATED = """name = "Owner_proofs"
version = "0.1.0"

defaultTargets = ["Proofs"]

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "v4.28.0"

[[lean_lib]]
name = "Proofs"
globs = ["kept_thm"]
"""


def test_a_curated_lakefile_is_left_as_its_owner_wrote_it(tmp_path):
    home = tmp_path / "proofs"
    home.mkdir()
    (home / "kept_thm.lean").write_text("theorem kept_thm : True := trivial\n")
    (home / "draft.lean").write_text("theorem draft : False := sorry\n")  # kept out of the build by its owner
    (home / "lakefile.toml").write_text(_CURATED)
    saved, note = m._save_solved_proof(
        {"theorem_id": "new_thm", "proof_home": str(home)}, "theorem new_thm : True := trivial\n"
    )
    assert saved == home / "new_thm.lean" and saved.exists()
    assert (home / "lakefile.toml").read_text() == _CURATED
    assert "`lake build` does not build new_thm" in note
    # a module the owner's lakefile already lists: nothing to say
    (home / "kept_thm.lean").unlink()
    saved, note = m._save_solved_proof(
        {"theorem_id": "kept_thm", "proof_home": str(home)}, "theorem kept_thm : True := trivial\n"
    )
    assert saved == home / "kept_thm.lean" and note == ""
    assert (home / "lakefile.toml").read_text() == _CURATED


def test_a_different_proof_already_home_is_never_replaced(tmp_path, monkeypatch):
    home, aside = tmp_path / "proofs", tmp_path / "aside"
    monkeypatch.setattr(m, "PROOFS_ASIDE", aside)
    home.mkdir()
    owner = "theorem t : True := by\n  trivial\n"
    (home / "t.lean").write_text(owner)
    lakefile = m._lakefile(f"{tmp_path.name}_proofs", "Proofs", m.proofs_roots(["t"]))
    (home / "lakefile.toml").write_text(lakefile)
    saved, note = m._save_solved_proof({"theorem_id": "t", "proof_home": str(home)}, "theorem t : True := trivial\n")
    assert saved is None and "already holds a different proof; it was not replaced" in note
    assert (home / "t.lean").read_text() == owner and (home / "lakefile.toml").read_text() == lakefile
    [kept] = aside.glob("t-*.lean")
    assert kept.read_text() == "theorem t : True := trivial\n" and str(kept) in note
    # the same proof again is no replacement: it stands, nothing is set aside
    saved, note = m._save_solved_proof({"theorem_id": "t", "proof_home": str(home)}, owner)
    assert saved == home / "t.lean" and note == "" and len(list(aside.glob("t-*.lean"))) == 1
