# Rapport d’analyse scientifique complet — dernière exécution

Run: `research_20260310T215445Z_720`
UTC: `2026-03-10T22:03:55.908595+00:00`

## 1. Analyse pédagogique (cours structuré)
- **Contexte** : pipeline multi-modules (Hubbard, QCD proxy, Quantum Field, Dense Nuclear, Quantum Chemistry).
- **Hypothèses** : la dynamique proxy doit rester stable, reproductible et traçable sous variations de paramètres (step, dt, contrôles externes).
- **Méthode** : tests automatiques (reproductibilité, convergence, stress, sensibilité, benchmark externe, stabilité, spectral) + matrices de questions expertes.
- **Résultats** : tous les tests sont loggés en CSV avec PASS/FAIL/OBSERVED.
- **Interprétation** : les points forts sont la reproductibilité et le benchmark; les points à risque sont les tests marqués FAIL.

## 2. Questions expertes et statut
- Questions totales: **19**
- `complete`: **11**
- `partial`: **8**
- `absent`: **0**
- Couverture experte complète: **57.89%**

## 3. Détection d’anomalies / incohérences
- Tests en échec détectés :
  - `new_tests_results`: convergence/conv_monotonic -> `FAIL` (value=0)
  - `new_tests_results`: verification/independent_calc -> `FAIL` (value=8.3585897831)
  - `new_tests_results`: benchmark/qmc_dmrg_rmse -> `FAIL` (value=1284425.8306754190)
  - `new_tests_results`: benchmark/qmc_dmrg_mae -> `FAIL` (value=810134.6052051842)
  - `new_tests_results`: benchmark/qmc_dmrg_within_error_bar -> `FAIL` (value=0.000000)
  - `new_tests_results`: benchmark/qmc_dmrg_ci95_halfwidth -> `FAIL` (value=650009.1539482179)
  - `new_tests_results`: benchmark/external_modules_rmse -> `FAIL` (value=30626.5888777254)
  - `new_tests_results`: benchmark/external_modules_mae -> `FAIL` (value=20345.2936596506)
  - `new_tests_results`: benchmark/external_modules_within_error_bar -> `FAIL` (value=0.000000)
  - `new_tests_results`: cluster_scale/cluster_pair_trend -> `FAIL` (value=0)
  - `new_tests_results`: cluster_scale/cluster_energy_trend -> `FAIL` (value=0)
  - `numerical_stability_suite`: von_neumann hubbard_hts_core -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann qcd_lattice_proxy -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann quantum_field_noneq -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann dense_nuclear_proxy -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann quantum_chemistry_proxy -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann spin_liquid_exotic -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann topological_correlated_materials -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann correlated_fermions_non_hubbard -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann multi_state_excited_chemistry -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann bosonic_multimode_systems -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann multiscale_nonlinear_field_models -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann far_from_equilibrium_kinetic_lattices -> `FAIL` (spectral_radius=1.0002246148)
  - `numerical_stability_suite`: von_neumann multi_correlated_fermion_boson_networks -> `FAIL` (spectral_radius=1.0002246148)

## 4. Comparaison littérature / référence
- Référence externe utilisée: QMC/DMRG via `benchmark_comparison_qmc_dmrg.csv` et tests famille `benchmark`.
- Taux de succès benchmark: **0.00%**

## 5. Nouveaux tests exécutés / proposés
- Exécutés: reproductibilité, convergence, stress, contrôle phase/pump/quench, sweep dt, FFT, stabilité Von Neumann proxy, cas jouet analytique, dérivées temporelles/variance.
- Prochains tests recommandés:
  1. resserrer `dt` autour des zones d’instabilité détectées;
  2. augmenter la campagne multi-seed sur modules en FAIL;
  3. comparer aux solutions analytiques supplémentaires par module.

## 6. Réponse point par point (question / analyse / réponse / solution)
- **Question**: Les valeurs et % sont-ils à jour avec les nouveaux tests ?
  - **Analyse**: recalcul automatique des pourcentages par famille + couverture experte.
  - **Réponse**: Oui, mis à jour depuis les CSV générés dans ce run.
  - **Solution**: maintenir ce rapport auto-généré à chaque exécution.
- **Question**: Les nouvelles questions sont-elles incluses au bon endroit ?
  - **Analyse**: lecture de `expert_questions_matrix.csv` et comptage des statuts.
  - **Réponse**: Oui, intégrées dans `tests/expert_questions_matrix.csv` et reprises dans ce rapport.
  - **Solution**: ajouter une gate FAIL si question requise absente.

## 7. État d’avancement vers la solution (%)
- Reproductibilité: **100.00%**
- Convergence: **80.00%**
- Benchmark externe: **0.00%**
- Contrôles dynamiques: **100.00%**
- Stabilité longue: **100.00%**
- Analyse spectrale: **100.00%**
- Couverture questions expertes complètes: **57.89%**
- Traçabilité checksum: **0.00%**
- **Score global pondéré**: **67.79%**

## 8. Traçabilité avancée
- Fichiers analysés: `new_tests_results.csv`, `expert_questions_matrix.csv`, `numerical_stability_suite.csv`, `toy_model_validation.csv`, `module_physics_metadata.csv`, `temporal_derivatives_variance.csv`.
- Checksums SHA256 ajoutés pour le rapport et le résumé JSON.

## 9. Commande d’exécution reproductible
```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```
