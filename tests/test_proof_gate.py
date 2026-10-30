"""The acceptance gate: each check names the failure it rules out.

Pure-function pins (no Lean): the verdict logic is exercised on fabricated
`lean --json` runs, so every rejection path is covered without a kernel.
"""

from __future__ import annotations

import unittest

import src.proof_gate as g

SUBMITTED = """import Mathlib

def double (n : ℕ) : ℕ := 2 * n

theorem double_pos (n : ℕ) (h : 0 < n) : 0 < double n := by
  unfold double
  -- WAYFINDER RESIDUAL 1/1
  sorry
"""

FINISHED = """import Mathlib

def double (n : ℕ) : ℕ := 2 * n

lemma helper (n : ℕ) (h : 0 < n) : 0 < 2 * n := by omega

theorem double_pos (n : ℕ) (h : 0 < n) : 0 < double n := by
  unfold double
  exact helper n h
"""

TYPE = "@double_pos : forall (n : Nat), LT.lt Nat instLTNat 0 n -> LT.lt Nat instLTNat 0 (double n)"


def _run(
    *,
    axioms: str | None = "[propext]",
    errors: tuple[str, ...] = (),
    sorry: bool = False,
    rc: int | None = 0,
    check: str | None = TYPE,
    timed_out: bool = False,
) -> dict:
    msgs = [{"severity": "information", "data": check}] if check else []
    if axioms is not None:
        msgs.append({"severity": "information", "data": f"'double_pos' depends on axioms: {axioms}"})
    msgs += [{"severity": "error", "pos": {"line": 9, "column": 2}, "data": e} for e in errors]
    if sorry:
        msgs.append({"severity": "warning", "kind": "hasSorry", "data": "declaration uses `sorry`"})
    return {"returncode": rc, "messages": msgs, "timed_out": timed_out, "tail": ""}


class TextTest(unittest.TestCase):
    def test_strip_comments_keeps_strings_and_lines(self):
        src = 'def s := "a -- b" -- gone\n/- block /- nested -/ still -/ x'
        out = g.strip_comments(src)
        self.assertIn('"a -- b"', out)
        self.assertNotIn("gone", out)
        self.assertNotIn("nested", out)
        self.assertIn("x", out)
        self.assertEqual(out.count("\n"), 1)

    def test_chunks_and_names(self):
        chunks = g.top_level_chunks(FINISHED)
        names = [g.declaration_name(c) for c in chunks]
        self.assertEqual(names, ["", "double", "helper", "double_pos"])
        self.assertEqual(g.name_from_declaration("theorem aime_1983_p2 (x : ℝ) : x = x := by sorry"), "aime_1983_p2")

    def test_split_header_ignores_binder_defaults(self):
        head, body = g.split_header("theorem t (x : ℕ := 5) : x = x := by rfl")
        self.assertEqual(head.strip(), "theorem t (x : ℕ := 5) : x = x")
        self.assertEqual(body.strip(), "by rfl")


class StaticTest(unittest.TestCase):
    def test_clean_candidate(self):
        self.assertEqual(g.static_findings(FINISHED, SUBMITTED), [])

    def test_new_axiom_and_sorry_and_eval(self):
        bad = FINISHED + "\naxiom cheat : False\n#eval IO.println \"hi\"\ntheorem x : True := by admit\n"
        found = " | ".join(g.static_findings(bad, SUBMITTED))
        self.assertIn("new axiom: cheat", found)
        self.assertIn("code-executing command: #eval", found)
        self.assertIn("`admit` remains", found)

    def test_scoped_notation_is_a_syntax_extension(self):
        bad = FINISHED + "\nlocal notation \"⟪\" x \"⟫\" => x\n"
        self.assertIn("new syntax extension: notation", " ".join(g.static_findings(bad, SUBMITTED)))

    def test_kernel_bypass_is_by_token_not_substring(self):
        ok = FINISHED + "\ntheorem uses_name : True := by trivial -- unsafeCast mentioned only here\n"
        self.assertEqual(g.static_findings(ok, SUBMITTED), [])
        bad = FINISHED + "\nunsafe def boom : ℕ := 0\n"
        self.assertTrue(any("kernel bypass" in f for f in g.static_findings(bad, SUBMITTED)))

    def test_switching_the_kernel_check_off_is_a_bypass(self):
        bad = "set_option debug.skipKernelTC true\n" + FINISHED
        self.assertIn("kernel bypass (set_option debug.skipKernelTC)", g.static_findings(bad, SUBMITTED))
        scoped = FINISHED.replace(":= by", ":= by\n  set_option debug.skipKernelTC true in", 1)
        self.assertTrue(any("debug.skipKernelTC" in f for f in g.static_findings(scoped, SUBMITTED)))
        heartbeats = FINISHED.replace(":= by", ":= by\n  set_option maxHeartbeats 400000 in", 1)
        self.assertEqual(g.static_findings(heartbeats, SUBMITTED), [])


class FidelityTest(unittest.TestCase):
    def test_changed_definition_is_caught(self):
        changed = FINISHED.replace("2 * n", "n + 1", 1)
        self.assertTrue(g.preserved_findings(changed, SUBMITTED, "double_pos"))

    def test_changed_statement_is_caught(self):
        changed = FINISHED.replace("(h : 0 < n) : 0 < double n", "(h : 1 < n) : 0 < double n")
        self.assertEqual(g.header_findings(changed, SUBMITTED, "double_pos"), ["statement of `double_pos` changed"])

    def test_finish_vs_reprove(self):
        kept, missing, n = g.prefix_kept(FINISHED, SUBMITTED, "double_pos")
        self.assertEqual((kept, missing, n), (True, [], 1))
        reproved = FINISHED.replace("  unfold double\n", "  simp only [double]\n")
        kept, missing, _ = g.prefix_kept(reproved, SUBMITTED, "double_pos")
        self.assertFalse(kept)
        self.assertEqual(missing, ["unfold double"])

    def test_bare_sorry_has_no_prefix(self):
        bare = SUBMITTED.replace("  unfold double\n", "")
        self.assertEqual(g.prefix_kept(FINISHED, bare, "double_pos")[2], 0)


class VerdictTest(unittest.TestCase):
    def judge(self, returned=FINISHED, run=None, **kw):
        kw.setdefault("submitted_type", TYPE)
        return g.judge(returned=returned, submitted=SUBMITTED, target="double_pos", compile_run=run, **kw)

    def test_accepts_a_correct_finish(self):
        res = self.judge(run=_run())
        self.assertTrue(res.accepted, res.reasons)
        self.assertEqual(res.mode, "finished")
        self.assertEqual(res.axioms, ["propext"])

    def test_rejects_declared_axiom_dependency(self):
        res = self.judge(run=_run(axioms="[propext, Phase6.T5_12_greedy_approximation]"))
        self.assertFalse(res.accepted)
        self.assertIn("non-allowlisted axioms: Phase6.T5_12_greedy_approximation", res.reasons)

    def test_row_may_allow_an_axiom(self):
        res = self.judge(run=_run(axioms="[propext, rado_1962]"), allowed_axioms=("rado_1962",))
        self.assertTrue(res.accepted, res.reasons)

    def test_rejects_compile_errors_and_sorry(self):
        self.assertFalse(self.judge(run=_run(errors=("unknown identifier 'x'",), rc=1)).accepted)
        self.assertFalse(self.judge(run=_run(sorry=True)).accepted)

    def test_a_timeout_is_not_a_compile(self):
        res = self.judge(run=_run(check=None, axioms=None, rc=None, timed_out=True))
        self.assertFalse(res.checks["compiles"])
        self.assertIn("compile timed out", res.reasons)

    def test_statement_type_must_match_and_be_recorded(self):
        self.assertFalse(self.judge(run=_run(check=TYPE.replace("0 (double", "1 (double"))).accepted)
        res = self.judge(run=_run(), submitted_type=None)
        self.assertFalse(res.accepted)
        self.assertIn("no recorded statement type to compare against", res.reasons)

    def test_universe_polymorphic_type_is_read_back(self):
        msgs = [{"data": "@target_is_null.{u_1} : ∀ {Θ : Type u_1} (θ : Θ),\n  True"}]
        self.assertEqual(g.checked_type(msgs, "target_is_null"), "@target_is_null.{u_1} : ∀ {Θ : Type u_1} (θ : Θ), True")
        self.assertIsNone(g.checked_type(msgs, "target"))

    def test_static_failure_is_never_compiled(self):
        res = self.judge(returned=FINISHED + "\naxiom cheat : False\n", run=None)
        self.assertFalse(res.accepted)
        self.assertFalse(res.checks["compiles"])

    def test_strict_mode_rejects_a_reproof(self):
        reproved = FINISHED.replace("  unfold double\n", "  simp only [double]\n")
        self.assertTrue(self.judge(returned=reproved, run=_run()).accepted)
        res = self.judge(returned=reproved, run=_run(), require_finish=True)
        self.assertFalse(res.accepted)
        self.assertEqual(res.mode, "reproved")


if __name__ == "__main__":
    unittest.main()
