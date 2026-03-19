# Rapport FSS Tc Binder — C36-P4

- **Tc nominal run** : 76.50 K
- **Tc estimé FSS** : N/A

- **Croisement Binder** : diff=inf
- **Statut** : NO_CROSSING

## Méthode

Cumulant de Binder g₄ calculé pour L = 8, 12, 16.
Tc FSS estimé par croisement g₄(T,L₁) = g₄(T,L₂) → interpolation.
Classe d'universalité supposée : 2D Ising-like (ν = 0.67, g* ≈ 0.611).

## Fichiers CSV

- `tests/fss_binder_cumulants.csv` — courbes g₄(T) pour L = 8, 12, 16
- `tests/fss_tc_estimate.json` — résultat FSS Tc

## Conclusion

FSS Binder STATUS=NO_CROSSING. Pas de croisement détecté.
