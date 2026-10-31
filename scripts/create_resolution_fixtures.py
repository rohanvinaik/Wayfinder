#!/usr/bin/env python3
"""Build P2 adversarial resolution fixtures F1a through F10 (§S0.7)."""

import json
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures/resolution")


def write_fixture(name: str, file_lean: str, step_txt: str, expect: dict):
    d = FIXTURES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "file.lean").write_text(file_lean.strip() + "\n")
    (d / "step.txt").write_text(step_txt.strip() + "\n")
    (d / "expect.json").write_text(json.dumps(expect, indent=2) + "\n")
    print(f"Wrote fixture {name}")


def main():
    SPECIMEN_PREFIX = """import Mathlib
open Finset

noncomputable def survivingDOF {D : ℕ} (lam : Fin D → ℝ) (t : ℝ) : ℕ :=
  (Finset.univ.filter (fun i => lam i < t)).card
"""

    F1A_LEAN = SPECIMEN_PREFIX + """
theorem survivingDOF_antitone {D : ℕ} (lam lam' : Fin D → ℝ) (t : ℝ)
    (h : ∀ i, lam i ≤ lam' i) :
    survivingDOF lam' t ≤ survivingDOF lam t := by
  have h_subset : (Finset.univ.filter (fun i => lam' i < t)) ⊆ (Finset.univ.filter (fun i => lam i < t)) := by
    sorry
  have h_main : survivingDOF lam' t ≤ survivingDOF lam t := by
    sorry
  exact h_main
"""

    F1A_STEP = """intro i hi
simp only [Finset.mem_filter, Finset.mem_univ, true_and_iff] at hi ⊢
-- We need to show that if `lam' i < t`, then `lam i < t`.
have h₁ : lam' i < t := by simpa using hi
have h₂ : lam i ≤ lam' i := h i
have h₃ : lam i < t := by linarith
exact h₃"""

    write_fixture(
        "f1a_specimen_subset",
        F1A_LEAN,
        F1A_STEP,
        {"outcome": "CLOSED", "fill_uses": ["true_and"]},
    )

    F1B_LEAN = SPECIMEN_PREFIX + """
theorem survivingDOF_antitone {D : ℕ} (lam lam' : Fin D → ℝ) (t : ℝ)
    (h : ∀ i, lam i ≤ lam' i) :
    survivingDOF lam' t ≤ survivingDOF lam t := by
  have h_subset : (Finset.univ.filter (fun i => lam' i < t)) ⊆ (Finset.univ.filter (fun i => lam i < t)) := by
    grind
  have h_main : survivingDOF lam' t ≤ survivingDOF lam t := by
    sorry
  exact h_main
"""

    F1B_STEP = """rw [survingDOF]
rw [survingDOF]
-- Use the fact that the cardinality of a subset is less than or equal to the cardinality of the original set.
exact Finset.card_le_of_subset h_subset"""

    write_fixture(
        "f1b_specimen_main",
        F1B_LEAN,
        F1B_STEP,
        {"outcome": "CLOSED", "fill_uses": ["survivingDOF", "Finset.card_le_card"]},
    )

    # F2a
    F2A_STEP = F1A_STEP.replace("true_and_iff", "blarg_17")
    write_fixture(
        "f2a_herpderp_subset",
        F1A_LEAN,
        F2A_STEP,
        {"outcome": "CLOSED", "fill_uses": ["true_and"]},
    )

    # F2b
    F2B_STEP = """rw [xq_zz_9]
rw [xq_zz_9]
-- Use the fact that the cardinality of a subset is less than or equal to the cardinality of the original set.
exact HerpDerp_theorem_47 h_subset"""
    write_fixture(
        "f2b_herpderp_main",
        F1B_LEAN,
        F2B_STEP,
        {"outcome": "CLOSED", "fill_uses": ["survivingDOF", "Finset.card_le_card"]},
    )

    # F3 misleading
    F3_STEP = """rw [survingDOF]
rw [survingDOF]
-- Use the fact that the cardinality of a subset is less than or equal to the cardinality of the original set.
exact Finset.card_lt_of_ssubset h_subset"""
    write_fixture(
        "f3_misleading",
        F1B_LEAN,
        F3_STEP,
        {"outcome": "CLOSED", "fill_uses": ["survivingDOF", "Finset.card_le_card"]},
    )

    # F4 renamed_candidate
    F4_LEAN = """import Mathlib

def wfP (n : ℕ) : Prop := n % 7 = 3

theorem HerpDerp_theorem_47 (n : ℕ) (h : n % 7 = 3) : wfP n := h

theorem wf_f4 (n : ℕ) (h : n % 7 = 3) : wfP n := by
  sorry
"""
    F4_STEP = "exact wfP_of_mod h"
    write_fixture(
        "f4_renamed_candidate",
        F4_LEAN,
        F4_STEP,
        {"candidate_first_page": "HerpDerp_theorem_47", "probe_outcome": "CLOSED"},
    )

    # F5 dependent
    F5_LEAN = """import Mathlib

theorem wf_f5 : ∃ n : ℕ, n = n := by
  sorry
"""
    F5_STEP = "exact missing_intro 3 rfl"
    write_fixture(
        "f5_dependent",
        F5_LEAN,
        F5_STEP,
        {
            "allowed_outcomes": ["CLOSED", "ROLE_UNDERCONSTRAINED", "BUDGET_EXHAUSTED"],
            "forbidden_outcomes": ["wrong_acceptance"],
        },
    )

    # F6 two_location_simp
    F6_LEAN = """import Mathlib

theorem wf_f6 (p q : Prop) (h : True ∧ p) (hq : p → q) : True ∧ q := by
  sorry
"""
    F6_STEP = """simp only [missing_rule] at h ⊢
exact hq h"""
    write_fixture(
        "f6_two_location_simp",
        F6_LEAN,
        F6_STEP,
        {"outcome": "CLOSED", "fill_uses": ["true_and"]},
    )

    # F7 impossible_transport
    F7_LEAN = """import Mathlib

theorem wf_f7 (a b c : ℕ) (h : a + b - c = 0) : a + b = c := by
  sorry
"""
    F7_STEP = "exact transport_lemma h"
    write_fixture(
        "f7_impossible_transport",
        F7_LEAN,
        F7_STEP,
        {"forbidden_outcomes": ["CLOSED"]},
    )

    # F8 continuation
    write_fixture(
        "f8_continuation",
        F1B_LEAN,
        F2B_STEP,
        {
            "outcome": "CLOSED",
            "resolution_page_size": 1,
            "resolution_max_pages": 64,
            "continuation": True,
        },
    )

    # F9 opaque_definition
    F9_STEP = "exact Finset.card_le_of_subset h_subset"
    write_fixture(
        "f9_opaque_definition",
        F1B_LEAN,
        F9_STEP,
        {"after_s6": "CLOSED"},
    )

    # F10 rejected_constant
    F10_STEP = """rw [survingDOF]
rw [survingDOF]
exact Finset.card_lt_card h_subset"""
    write_fixture(
        "f10_rejected_constant",
        F1B_LEAN,
        F10_STEP,
        {
            "outcome": "CLOSED",
            "fill_uses": ["Finset.card_le_card"],
            "excluded": ["Finset.card_lt_card"],
        },
    )


if __name__ == "__main__":
    main()
