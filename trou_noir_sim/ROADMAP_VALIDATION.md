# FEUILLE DE ROUTE - VALIDATION SCIENTIFIQUE V11

## Phase 1 : Rigueur Mathématique (Q1-Q4)
- [x] Implémentation explicite du tenseur métrique de Boyer-Lindquist.
- [x] Ajout du moniteur de contrainte Hamiltonienne (Conservation de la norme du quadrivecteur).
- [ ] Étude de convergence de Richardson (Raffinement de maillage temporel).
- [ ] Migration vers intégrateur RK4 (Runge-Kutta 4) pour stabilité accrue.

## Phase 2 : Analyse HPC & Artefacts (Q5-Q6)
- [ ] Passage en `__float128` (Quad Precision) pour isoler les erreurs d'arrondi.
- [ ] Corrélation entre "Viscosité numérique" et précision IEEE 754.
- [ ] Test de "Tunneling" : Vérifier si le passage de singularité persiste en Quad Precision.

## Phase 3 : Comparaison Littérature
- [ ] Benchmarking vs Geodesic Integrator (EinsteinPy).
- [ ] Test de Penrose : Extraction d'énergie et conservation de l'aire.

## Phase 4 : Rapport Final & Audit
- [ ] Publication des courbes de violation des contraintes.
- [ ] Conclusion sur la nature des "nouvelles physiques" (Physique vs Artefact).
