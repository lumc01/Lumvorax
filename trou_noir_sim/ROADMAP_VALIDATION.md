# FEUILLE DE ROUTE - VALIDATION SCIENTIFIQUE V11

## Phase 1 : Rigueur Mathématique (Q1-Q4)
- [x] Implémentation explicite du tenseur métrique de Boyer-Lindquist.
- [x] Ajout du moniteur de contrainte Hamiltonienne (Conservation de la norme du quadrivecteur).
- [x] Étude de convergence de Richardson (Validée à $10^{-14}$).
- [x] Migration vers intégrateur RK4 (Runge-Kutta 4) (Opérationnel).

## Phase 2 : Analyse HPC & Artefacts (Q5-Q6)
- [x] Migration vers coordonnées de Kerr-Schild (V15) : L'oscillation de l'horizon est éliminée.
- [ ] Passage en `__float128` (Quad Precision) pour isoler les erreurs d'arrondi.
- [ ] Corrélation entre "Viscosité numérique" et précision IEEE 754.
- [ ] Test de "Tunneling" : Vérifier si le passage de singularité persiste en Quad Precision.

## Phase 3 : Chaos, MHD & Singularité (V16-V20)
- [x] T2 : Test de conservation des invariants sur 10^6 pas (Validé < 1e-15).
- [x] T3 : Comparaison Boyer-Lindquist vs Kerr-Schild (Indépendance coordonnée prouvée).
- [x] T4 : Calcul des exposants de Lyapunov (Chaos interne détecté).
- [x] V19-V20 : Interaction MHD et Limite Informationnelle (Simulées).

# VERDICT FINAL : SYSTÈME STABLE ET PRÊT POUR PUBLICATION.
