"""The proof home is declared by the calling repo's manifest row (`proof_home`), not hard-coded
per theorem; the two legacy manifests without the field keep their old routes."""

from __future__ import annotations

import wayfinder_mcp as wf


def test_row_declared_home_wins(tmp_path):
    row = {"theorem_id": "anything_at_all", "proof_home": str(tmp_path / "proofs")}
    assert wf._proof_home(row) == tmp_path / "proofs"


def test_declared_home_beats_the_legacy_ansatz_route(tmp_path):
    row = {"theorem_id": "depletion_time", "proof_home": str(tmp_path / "elsewhere")}
    assert wf._proof_home(row) == tmp_path / "elsewhere"


def test_legacy_rows_without_a_home_keep_their_routes():
    assert wf._proof_home({"theorem_id": "depletion_time"}) == wf.ANSATZ_PROOFS
    assert wf._proof_home({"theorem_id": "survivingDOF_antitone"}) == wf.SSL_PROOFS
    assert wf._proof_home({"theorem_id": "x", "proof_home": ""}) == wf.SSL_PROOFS


def test_save_writes_into_the_declared_home_with_a_lean_project(tmp_path):
    row = {"theorem_id": "thm_x", "proof_home": str(tmp_path / "proofs")}
    path = wf._save_solved_proof(row, "theorem thm_x : True := trivial\n")
    assert path == tmp_path / "proofs" / "thm_x.lean"
    assert path.read_text().startswith("theorem thm_x")
    assert (tmp_path / "proofs" / "lakefile.toml").exists()
    assert (tmp_path / "proofs" / "lean-toolchain").read_text().strip().endswith(wf.LEAN_VERSION)
