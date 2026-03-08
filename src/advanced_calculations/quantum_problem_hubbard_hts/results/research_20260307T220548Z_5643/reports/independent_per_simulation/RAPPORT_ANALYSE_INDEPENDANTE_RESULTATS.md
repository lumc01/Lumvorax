# RAPPORT GLOBAL INDÉPENDANT — Résultats d'analyse du run ciblé

- Run: `research_20260307T220548Z_5643`
- UTC: `2026-03-07T22:42:44.152119+00:00`
- Simulations détectées: **13**

## Résumé quantitatif des 13 simulations
| Module | Energy/site fin | Rang énergie | Pairing norm fin | Rang pairing | Sign min | Sign max | Statut |
|---|---:|---:|---:|---:|---:|---:|---|
| bosonic_multimode_systems | 5794.116482472251 | 13 | 1459.1404967647989 | 10 | -0.1 | 0.0047836938 | stable_proxy |
| correlated_fermions_non_hubbard | 10717.4875314953 | 4 | 1717.74773746704 | 6 | -0.0036029366 | 0.1555555556 | stable_proxy |
| dense_nuclear_proxy | 10532.178360383132 | 5 | 1558.9792716457723 | 8 | -0.0019251925 | 0.1111111111 | stable_proxy |
| far_from_equilibrium_kinetic_lattices | 9022.32044556793 | 9 | 1447.3940423430586 | 11 | -0.0002349072 | 0.0101010101 | stable_proxy |
| hubbard_hts_core | 12667.999854127853 | 2 | 1920.799176118617 | 2 | -0.04 | 0.0081188119 | stable_proxy |
| multi_correlated_fermion_boson_networks | 9291.56033727925 | 7 | 1624.8364505268212 | 7 | -0.08 | -0.0013756614 | stable_proxy |
| multi_state_excited_chemistry | 7290.246287893707 | 10 | 1893.7736551406274 | 3 | -0.0617283951 | 0.003274825 | stable_proxy |
| multiscale_nonlinear_field_models | 10014.601010355442 | 6 | 1490.3912985203842 | 9 | -0.0625 | -0.0016858142 | stable_proxy |
| qcd_lattice_proxy | 9074.938775353461 | 8 | 1389.7067698744604 | 12 | -0.012345679 | 0.0069406056 | stable_proxy |
| quantum_chemistry_proxy | 6532.874463416726 | 12 | 1719.4083491133965 | 5 | -0.1428571429 | 0.0222772277 | stable_proxy |
| quantum_field_noneq | 6649.672824405342 | 11 | 1178.9401514429376 | 13 | -0.125 | 0.0006871564 | surveillance |
| spin_liquid_exotic | 15874.430368231939 | 1 | 2052.650015007499 | 1 | -0.0333333333 | 0.0106312292 | alerte |
| topological_correlated_materials | 11495.332510301148 | 3 | 1846.453653216176 | 4 | -0.1074380165 | -0.0009497348 | stable_proxy |

## Questions expertes (globales)
1) **Validité quantitative vs référence ?**
   - Analyse: pairing validé sur hubbard_hts_core; énergie non validée sur benchmark disponible.
   - Réponse: **partielle**.
   - Solution: recalibrer canal énergie + benchmark multi-modules.
2) **Stabilité numérique suffisante ?**
   - Analyse: FAIL run-level sur dt_convergence + von_neumann + drift énergie.
   - Réponse: **non**.
   - Solution: dt-sweep plus dense, schéma implicite/symplectique, validation par module.
3) **Traçabilité audit externe ?**
   - Analyse: checksums 67/67 OK, missing=0, mismatch=0.
   - Réponse: **oui** pour l'intégrité run, sous réserve de conservation de l'historique.
   - Solution: conserver append-only + manifestes SHA256 signés.

## Commande reproductible
```bash
python3 src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_individual_simulation_reports.py --run-dir /workspace/Lumvorax/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260307T220548Z_5643
```
