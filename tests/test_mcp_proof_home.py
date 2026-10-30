"""The proof home is declared by the calling repo's manifest row (`proof_home`), not hard-coded
per theorem; the two legacy manifests without the field keep their old routes."""

from __future__ import annotations

import json
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



# ── fill_hole: one acceptance path for every prover ──────────────────────────────

_DECL = """theorem t (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  -- WAYFINDER RESIDUAL 1: close this goal
  --   ⊢ p
  · sorry
  -- WAYFINDER RESIDUAL 2: close this goal
  --   ⊢ q
  · sorry
"""
_TYPE = "@t : ∀ (p q : Prop), p → q → p ∧ q"


def _fake_compile(text: str, timeout: int = 420) -> dict:
    """Lean, as far as the fill path reads it: errors where a fill says WRONG, `sorryAx` while a
    `sorry` is left, and the statement's type as recorded."""
    open_holes = "sorry" in text.split("#check")[0]
    msgs = [{"severity": "information", "data": _TYPE},
            {"severity": "information", "data": f"'t' depends on axioms: [propext{', sorryAx' if open_holes else ''}]"}]
    if open_holes:
        msgs.append({"severity": "warning", "data": "declaration uses `sorry`"})
    if "WRONG" in text:
        line = text.splitlines().index(next(ln for ln in text.splitlines() if "WRONG" in ln)) + 1
        msgs.append({"severity": "error", "pos": {"line": line, "column": 4}, "endPos": None, "data": "unknown tactic"})
    return {"returncode": 1 if "WRONG" in text else 0, "messages": msgs, "timed_out": False, "tail": ""}


def _fill_setup(tmp_path, monkeypatch) -> dict:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    monkeypatch.setattr(wf, "GOAL_LEDGER", ledger_dir)
    monkeypatch.setattr(wf, "_compile_v428", _fake_compile)
    row = {"theorem_id": "t", "proof_home": str(tmp_path / "proofs"),
           "full_declaration": "theorem t (p q : Prop) (hp : p) (hq : q) : p ∧ q := by\n  sorry"}
    handoff = {"status": "residual", "target": "t", "declaration": _DECL, "file": "import Mathlib\n\n" + _DECL,
               "pp_type": _TYPE, "statement_sha256": wf._statement_sha(row), "closures_kept": 1,
               "holes": [{"index": 1, "lean_goal": "⊢ p"}, {"index": 2, "lean_goal": "⊢ q"}]}
    (ledger_dir / "t.json").write_text(json.dumps({"detail": {}, "manifest_row": row, "handoff": handoff}))
    return row


def _ledger(tmp_path) -> dict:
    return json.loads((tmp_path / "ledger" / "t.json").read_text())["handoff"]


def test_a_fill_that_fails_changes_nothing(tmp_path, monkeypatch):
    row = _fill_setup(tmp_path, monkeypatch)
    before = _ledger(tmp_path)
    assert not wf._apply_fill("t", row, 3, "exact hp", "", "x")["accepted"]  # no such hole
    denied = wf._apply_fill("t", row, 1, "sorry", "", "x")
    assert not denied["accepted"] and denied["compiled"] is False
    wrong = wf._apply_fill("t", row, 1, "WRONG hp", "", "x")
    assert not wrong["accepted"] and "<error>" in wrong["lean_errors"]
    assert _ledger(tmp_path) == before


def test_fills_are_kept_one_hole_at_a_time_then_the_proof_is_saved(tmp_path, monkeypatch):
    row = _fill_setup(tmp_path, monkeypatch)
    first = wf._apply_fill("t", row, 1, "exact hp", "", "claude")
    assert first["accepted"] and first["holes_left"] == 1
    h = _ledger(tmp_path)
    assert h["status"] == "residual" and [x["index"] for x in h["holes"]] == [2]
    assert h["as_built"]["file"] == "import Mathlib\n\n" + _DECL  # what the final gate compares against
    assert h["fills"][0]["source"] == "claude"
    last = wf._apply_fill("t", row, 2, "exact hq", "", "goedel-v2-32b (Q4_K_M), sample h2-32b-r0-s1")
    assert last["accepted"] and last["closed"] and last["mode"] == "finished"
    saved = (tmp_path / "proofs" / "t.lean").read_text()
    assert "provenance: Wayfinder (1 verified closures, residual) + hole 1: claude; hole 2: goedel-v2-32b" in saved
    assert "sorry" not in saved.split("-/", 1)[1]
    assert _ledger(tmp_path)["status"] == "closed"


def test_a_local_run_s_fills_go_through_the_same_path(tmp_path, monkeypatch):
    row = _fill_setup(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    monkeypatch.setattr(wf, "LOCAL_RUNS", runs)
    as_built = wf._sha256(_ledger(tmp_path)["file"])
    for run_id, sha in (("stale", "0" * 64), ("fresh", as_built)):
        (runs / run_id).mkdir(parents=True)
        (runs / run_id / "job.json").write_text(json.dumps({"theorem_id": "t", "row": row, "as_built_sha256": sha}))
        (runs / run_id / "fills.json").write_text(json.dumps(
            [{"hole": 1, "fill": "exact hp", "helpers": "", "source": "goedel-v2-8b", "model": "goedel-v2-8b",
              "quant": "Q8_0", "sample": "h1-8b-r0-s1", "round": 0, "form": "inline"}]))
    stale = json.loads(wf.local_prover_result("stale"))
    assert not stale["applied"]["1"]["accepted"]
    fresh = json.loads(wf.local_prover_result("fresh"))
    assert fresh["applied"]["1"]["accepted"]
    again = json.loads(wf.local_prover_result("fresh"))  # applied once
    assert again["applied"] == fresh["applied"]
    assert [x["index"] for x in _ledger(tmp_path)["holes"]] == [2]


def test_a_statement_the_steps_made_harder_is_tried_whole(monkeypatch):
    """When the search's steps leave holes no closer takes, the closers get the statement itself."""
    row = {"theorem_id": "w", "file_path": "no/such/file.lean", "imports": ["import Mathlib"], "opens": [],
           "full_declaration": "theorem w (n : ℕ) : n + 0 = n := by\n  sorry"}
    calls = []

    def fake(text: str, timeout: int = 420) -> dict:
        calls.append(text)
        lines = text.splitlines()
        if "w_wf_close_1" in text:  # the triage probe: the second closer (`omega`) marks its success
            at = next(i for i, ln in enumerate(lines) if "w_wf_close_1" in ln) + 1
            msgs = [{"severity": "information", "pos": {"line": at}, "data": "wf_local:1"}]
        else:  # the one-step proof, with the gate's probe
            msgs = [{"severity": "information", "data": "w : ∀ (n : ℕ), n + 0 = n"},
                    {"severity": "information", "data": "'w' does not depend on any axioms"}]
        return {"returncode": 0, "timed_out": False, "tail": "", "messages": msgs}

    monkeypatch.setattr(wf, "_compile_v428", fake)
    header, _ = wf.rh.split_declaration(row["full_declaration"])
    out = wf._close_whole(row, header, "n : ℕ\n⊢ n + 0 = n")
    assert out is not None and out["prefix"] == ["omega"] and out["holes"] == [] and out["closures_kept"] == 0
    assert out["declaration"].rstrip().endswith("  omega") and len(calls) == 2
    assert out["pp_type"] == "w : ∀ (n : ℕ), n + 0 = n"
