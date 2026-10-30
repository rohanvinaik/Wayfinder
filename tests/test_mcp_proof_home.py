"""The proof home is declared by the calling repo's manifest row (`proof_home`), not hard-coded
per theorem; the two legacy manifests without the field keep their old routes."""

from __future__ import annotations

from pathlib import Path

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


def test_accept_home_never_falls_through_to_another_papers_folder(monkeypatch):
    monkeypatch.setattr(wf, "_legacy_ids", lambda: {"legacy_thm"})
    assert wf._accept_home({"theorem_id": "x", "proof_home": "/tmp/p"}) == Path("/tmp/p")
    assert wf._accept_home({"theorem_id": "new_thm"}) is None
    assert wf._accept_home({"theorem_id": "legacy_thm"}) == wf._proof_home({"theorem_id": "legacy_thm"})


_REFUSAL = (
    "No source files in a supported language were uploaded. Lean-only projects aren't supported "
    "right now as we focus on verified software and hardware."
)


def test_a_service_refusal_is_told_from_a_failed_attempt():
    assert wf._service_refusal(_REFUSAL)
    assert not wf._service_refusal("Proved rDiv_nonneg in Residual.lean, replacing the only sorry.")
    assert not wf._service_refusal(None)


def test_the_service_record_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "service_status.json"
    monkeypatch.setattr(wf, "ARISTOTLE_SERVICE", path)
    assert wf._load_service_status() == {}
    wf._save_service_status({"lean_only": "refused", "project_id": "p1"})
    assert wf._load_service_status()["lean_only"] == "refused"

