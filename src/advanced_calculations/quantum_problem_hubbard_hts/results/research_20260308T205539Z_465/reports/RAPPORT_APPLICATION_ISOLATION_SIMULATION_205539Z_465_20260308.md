# Rapport d’application immédiate — isolation stricte par simulation (run 205539Z_465)

- Horodatage UTC: 2026-03-08T23:11:46Z
- Objectif: vérifier et renforcer le principe “chaque simulation, ses tests, ses résultats, ses pourcentages”.

## 1) Vérification: est-ce déjà fait ?
### Introduction (thèse + contexte)
La séparation existe partiellement dans les artefacts, mais elle n’est pas homogène sur toute la suite.
### Développement (argumentation)
- De plus, les fichiers benchmark et stabilité sont module-spécifiques.
- Cependant, `new_tests_results.csv` reste global (sans colonne module), ce qui mélange l’interprétation.
- En outre, les pourcentages globaux doivent être distingués des pourcentages module-only.
### Conclusion (solution + clôture)
Donc, j’ai ajouté un audit d’isolation + un calcul de pourcentages strictement module-spécifiques.

## 2) Ce que j’ai ajouté (nouveau, non précisé explicitement)
- `integration_simulation_isolation_audit_205539.csv`: audit fichier par fichier de l’isolation.
- `integration_module_specific_percentages_205539.csv`: pourcentages calculés uniquement à partir des résultats propres à chaque module.
- `forensic_isolation_205539_summary.json`: résumé machine lisible de conformité/risques.
- Ce rapport explicatif d’application immédiate.

## 3) Résultats module par module (propres au module, pas globaux)
| module | progress% | remaining% | bench_pass%_module | num_pass%_module | own_validation%_module |
|---|---:|---:|---:|---:|---:|
| bosonic_multimode_systems | 57.35 | 42.65 | 0.00 | 0.00 | 0.00 |
| correlated_fermions_non_hubbard | 59.75 | 40.25 | 0.00 | 0.00 | 0.00 |
| dense_nuclear_proxy | 58.60 | 41.40 | 0.00 | 0.00 | 0.00 |
| far_from_equilibrium_kinetic_lattices | 56.47 | 43.53 | 0.00 | 0.00 | 0.00 |
| multi_correlated_fermion_boson_networks | 59.82 | 40.18 | 0.00 | 0.00 | 0.00 |
| multi_state_excited_chemistry | 69.44 | 30.56 | 0.00 | 0.00 | 0.00 |
| multiscale_nonlinear_field_models | 65.76 | 34.24 | 0.00 | 0.00 | 0.00 |
| qcd_lattice_proxy | 56.41 | 43.59 | 0.00 | 0.00 | 0.00 |
| quantum_chemistry_proxy | 69.91 | 30.09 | 0.00 | 0.00 | 0.00 |
| quantum_field_noneq | 63.43 | 36.57 | 0.00 | 0.00 | 0.00 |
| spin_liquid_exotic | 62.33 | 37.67 | 0.00 | 0.00 | 0.00 |
| topological_correlated_materials | 61.93 | 38.07 | 0.00 | 0.00 | 0.00 |
| hubbard_hts_core | 70.03 | 29.97 | 53.33 | 66.67 | 60.00 |

## 4) Réponse directe à ta contrainte “aucun test ne doit affecter les autres simulations”
- Déjà conforme pour: benchmark externes, stabilité numérique, progression module.
- Non totalement conforme pour: `new_tests_results.csv` (suite globale).
- Action appliquée: création d’un mécanisme de lecture/validation module-only indépendant des agrégats globaux.

## 5) Questions → Analyse → Réponse → Solution
### Q1. Les % sont-ils calculés par simulation ?
- Analyse: auparavant surtout global et mixte.
- Réponse: maintenant oui via `integration_module_specific_percentages_205539.csv`.
- Solution: utiliser ce fichier comme source principale pour validation par module.
### Q2. Les tests d’une simulation affectent-ils les autres ?
- Analyse: certains tests sont globaux.
- Réponse: partiellement encore, côté `new_tests_results.csv`.
- Solution: scinder en suites par module ou injecter la clé `module` dans tous les tests globaux.
### Q3. Que reste-t-il à faire ?
- Analyse: il manque une gate CI qui refuse un test sans module cible.
- Réponse: absence actuelle de cette gate.
- Solution: ajouter “module_scope_gate” bloquant.

## 6) Commandes reproductibles
```bash
git fetch origin --prune
python src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_cycle06_205539_isolation_enforcement_report.py
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T205539Z_465
sha256sum -c logs/forensic_isolation_205539_checksums.sha256
```
