/-
This file was edited by Aristotle.

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 37746312-ae50-429e-ae8d-09cde7ef0395

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>
-/

import Mathlib


/--
  Stratégie NX-33 : Preuve de structure pour Aristotle.
  Version : Pure Core 4.24.0
  Contrainte : SANS Nat.iterate.
--/

def collatz_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

theorem collatz_step_pair (n : ℕ) (h : n % 2 = 0) (hn : n > 0) :
  collatz_step n < n := by
  simp [collatz_step, h]
  apply Nat.div_lt_self hn
  exact Nat.le_refl 2

-- Structure pour test Aristotle
def NX33_Core_Structure := "Validated"