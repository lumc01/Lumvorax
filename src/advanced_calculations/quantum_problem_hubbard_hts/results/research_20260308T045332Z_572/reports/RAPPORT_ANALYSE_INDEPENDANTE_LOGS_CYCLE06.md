# Rapport d’analyse indépendante des logs (hors rapport pipeline)

Run analysé: `research_20260308T045332Z_572`
Horodatage UTC: `2026-03-08T04:59:32.333893+00:00`

## Phase 1 — Synchronisation et intégrité
- Revue indépendante effectuée à partir des artefacts bruts du run (`logs/*.csv`, `tests/*.csv`).
- Fichier checksum global présent: **non**.

## Phase 2 — Analyse des données
- Lignes métriques brutes: **305**
- Trace normalisée: **52**
- Questions expertes: **19**
- Modules metadata: **13**
- Moyenne énergie brute: **269610.999281**
- Moyenne pairing brut: **84411.422278**
- Moyenne sign ratio brut: **-0.001880**

## Phase 3 — Vérification exhaustive
- Vérification famille par famille des tests PASS/FAIL/OBSERVED:
  - `benchmark`: PASS=0 FAIL=7 OBSERVED=0 (pass=0.00%)
  - `cluster_scale`: PASS=2 FAIL=2 OBSERVED=32 (pass=5.56%)
  - `control`: PASS=3 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `convergence`: PASS=5 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `dt_sweep`: PASS=0 FAIL=1 OBSERVED=3 (pass=0.00%)
  - `dynamic_pumping`: PASS=0 FAIL=0 OBSERVED=4 (pass=0.00%)
  - `exact_solver`: PASS=1 FAIL=0 OBSERVED=2 (pass=33.33%)
  - `physics`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `reproducibility`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `sensitivity`: PASS=0 FAIL=0 OBSERVED=8 (pass=0.00%)
  - `spectral`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stability`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stress`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `verification`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)

## Phase 4 — Analyse scientifique
- Couverture questions expertes complètes: **11/19 (57.89%)**
- Stabilité numérique (suite dédiée): **13.33% PASS**
- Points en échec détectés et conservés (pas masqués):
  - `new_tests_results`: dt_sweep/dt_convergence -> FAIL (value=0)
  - `new_tests_results`: benchmark/qmc_dmrg_rmse -> FAIL (value=40331.7478765852)
  - `new_tests_results`: benchmark/qmc_dmrg_mae -> FAIL (value=24324.9527305010)
  - `new_tests_results`: benchmark/qmc_dmrg_within_error_bar -> FAIL (value=53.333333)
  - `new_tests_results`: benchmark/qmc_dmrg_ci95_halfwidth -> FAIL (value=20410.6805456616)
  - `new_tests_results`: benchmark/external_modules_rmse -> FAIL (value=750655.3025163291)
  - `new_tests_results`: benchmark/external_modules_mae -> FAIL (value=500046.8801512591)
  - `new_tests_results`: benchmark/external_modules_within_error_bar -> FAIL (value=0.000000)
  - `new_tests_results`: cluster_scale/cluster_pair_trend -> FAIL (value=0)
  - `new_tests_results`: cluster_scale/cluster_energy_trend -> FAIL (value=0)
  - `numerical_stability_suite`: energy_conservation::hubbard_hts_core -> FAIL (energy_density_drift_max=0.1697066628)
  - `numerical_stability_suite`: von_neumann::hubbard_hts_core -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::qcd_lattice_proxy -> FAIL (energy_density_drift_max=0.2200679841)
  - `numerical_stability_suite`: von_neumann::qcd_lattice_proxy -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::quantum_field_noneq -> FAIL (energy_density_drift_max=0.1708287840)
  - `numerical_stability_suite`: von_neumann::quantum_field_noneq -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::dense_nuclear_proxy -> FAIL (energy_density_drift_max=0.2872333590)
  - `numerical_stability_suite`: von_neumann::dense_nuclear_proxy -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::quantum_chemistry_proxy -> FAIL (energy_density_drift_max=0.1174284269)
  - `numerical_stability_suite`: von_neumann::quantum_chemistry_proxy -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::spin_liquid_exotic -> FAIL (energy_density_drift_max=0.1750125163)
  - `numerical_stability_suite`: von_neumann::spin_liquid_exotic -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::topological_correlated_materials -> FAIL (energy_density_drift_max=0.1109782920)
  - `numerical_stability_suite`: von_neumann::topological_correlated_materials -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::correlated_fermions_non_hubbard -> FAIL (energy_density_drift_max=0.2521319218)
  - `numerical_stability_suite`: von_neumann::correlated_fermions_non_hubbard -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::multi_state_excited_chemistry -> FAIL (energy_density_drift_max=0.1603181144)
  - `numerical_stability_suite`: von_neumann::multi_state_excited_chemistry -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::bosonic_multimode_systems -> FAIL (energy_density_drift_max=0.1208640125)
  - `numerical_stability_suite`: von_neumann::bosonic_multimode_systems -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::multiscale_nonlinear_field_models -> FAIL (energy_density_drift_max=0.2586936885)
  - `numerical_stability_suite`: von_neumann::multiscale_nonlinear_field_models -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::far_from_equilibrium_kinetic_lattices -> FAIL (energy_density_drift_max=0.1795711674)
  - `numerical_stability_suite`: von_neumann::far_from_equilibrium_kinetic_lattices -> FAIL (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: energy_conservation::multi_correlated_fermion_boson_networks -> FAIL (energy_density_drift_max=0.1659581953)
  - `numerical_stability_suite`: von_neumann::multi_correlated_fermion_boson_networks -> FAIL (spectral_radius=1.0002246148)

## Phase 5 — Interprétation pédagogique
- Un test `PASS` signifie que le critère numérique codé est satisfait.
- Un test `FAIL` signifie qu’une hypothèse n’est pas validée avec les paramètres actuels.
- Un test `OBSERVED` signifie valeur mesurée sans gate binaire stricte.

## Phase 6 — Réponse point par point
- **Question**: Les valeurs et % sont-elles à jour ?
  - **Analyse**: recalcul direct depuis fichiers du run.
  - **Réponse**: Oui, ce rapport recalcule tous les ratios au moment de l’exécution.
  - **Solution**: conserver ce rapport indépendant dans chaque nouveau run.
- **Question**: Les nouvelles questions sont-elles bien incluses ?
  - **Analyse**: lecture de `expert_questions_matrix.csv` et comptage des statuts.
  - **Réponse**: Oui, incluses et quantifiées ici.
  - **Solution**: ajouter un test bloquant si question obligatoire absente.

## Phase 7 — Correctifs proposés
1. Réduire dt et augmenter steps uniquement sur modules/tests explicitement en FAIL.
2. Étendre la campagne multi-seed ciblée sur les familles en échec (pas sur les PASS).
3. Ajouter/resserrer benchmarks externes individuels pour les modules encore partiels.

## Phase 8/9 — Intégration technique et traçabilité
- Rapport indépendant écrit dans `reports/`.
- Résumé machine en JSON + checksums SHA256 dédiés écrits dans `logs/`.

## Phase 10 — Commande reproductible
```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

## État d’avancement vers la solution (%)
- Score pondéré indépendant: **68.14%**
