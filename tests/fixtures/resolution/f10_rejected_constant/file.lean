import Mathlib
open Finset

noncomputable def survivingDOF {D : ℕ} (lam : Fin D → ℝ) (t : ℝ) : ℕ :=
  (Finset.univ.filter (fun i => lam i < t)).card

theorem survivingDOF_antitone {D : ℕ} (lam lam' : Fin D → ℝ) (t : ℝ)
    (h : ∀ i, lam i ≤ lam' i) :
    survivingDOF lam' t ≤ survivingDOF lam t := by
  have h_subset : (Finset.univ.filter (fun i => lam' i < t)) ⊆ (Finset.univ.filter (fun i => lam i < t)) := by
    grind
  have h_main : survivingDOF lam' t ≤ survivingDOF lam t := by
    sorry
  exact h_main
