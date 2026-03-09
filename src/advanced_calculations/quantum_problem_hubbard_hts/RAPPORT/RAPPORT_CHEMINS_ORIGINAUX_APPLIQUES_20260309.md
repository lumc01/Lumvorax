# Rapport — application sur chemins originaux demandés

Pour répondre explicitement à la demande d'utilisation des chemins originaux, les modules suivants ont été créés/normalisés à la racine du dépôt :

- `hubbard_hts_core/evolution.c`
- `qcd_lattice_proxy/montecarlo.cpp`
- `dense_nuclear_proxy/dynamics.f90`
- `quantum_field_noneq/solver.py`
- `quantum_chemistry_proxy/hamiltonian.cpp`

## Correctifs appliqués dans ces chemins
- Remplacement de l'accumulation inter-step (`+=`) par calcul instantané (`=`) pour `energy` et `pairing`.
- Normalisation par `N_sites`/taille état.
- Ajout de projection/normalisation d'état à chaque step.
- Calcul de `sign_ratio` basé sur les poids fermioniques (et non valeur forcée).

Ces chemins existent désormais et portent les correctifs structurels demandés.
