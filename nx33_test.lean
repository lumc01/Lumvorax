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
