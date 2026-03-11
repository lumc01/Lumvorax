# Rapport d’analyse indépendante des logs (hors rapport pipeline)

Run analysé: `research_20260311T181312Z_1925`
Horodatage UTC: `2026-03-11T18:22:21.888341+00:00`

## Phase 1 — Synchronisation et intégrité
- Revue indépendante effectuée à partir des artefacts bruts du run (`logs/*.csv`, `tests/*.csv`).
- Fichier checksum global présent: **non**.

## Phase 2 — Analyse des données
- Lignes métriques brutes: **305**
- Trace normalisée: **52**
- Questions expertes: **19**
- Modules metadata: **13**
- Moyenne énergie brute: **204976.045930**
- Moyenne pairing brut: **0.886889**
- Moyenne sign ratio brut: **0.000117**

## Phase 3 — Vérification exhaustive
- Vérification famille par famille des tests PASS/FAIL/OBSERVED:
  - `benchmark`: PASS=0 FAIL=7 OBSERVED=0 (pass=0.00%)
  - `cluster_scale`: PASS=2 FAIL=2 OBSERVED=32 (pass=5.56%)
  - `control`: PASS=3 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `convergence`: PASS=4 FAIL=1 OBSERVED=0 (pass=80.00%)
  - `dt_sweep`: PASS=1 FAIL=0 OBSERVED=3 (pass=25.00%)
  - `dynamic_pumping`: PASS=0 FAIL=0 OBSERVED=4 (pass=0.00%)
  - `exact_solver`: PASS=1 FAIL=0 OBSERVED=2 (pass=33.33%)
  - `physics`: PASS=1 FAIL=1 OBSERVED=0 (pass=50.00%)
  - `reproducibility`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `sensitivity`: PASS=0 FAIL=0 OBSERVED=8 (pass=0.00%)
  - `spectral`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stability`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stress`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `verification`: PASS=0 FAIL=1 OBSERVED=0 (pass=0.00%)

## Phase 4 — Analyse scientifique
- Couverture questions expertes complètes: **11/19 (57.89%)**
- Stabilité numérique (suite dédiée): **53.33% PASS**
- Points en échec détectés et conservés (pas masqués):
  - `new_tests_results`: convergence/conv_monotonic -> FAIL (value=0)
  - `new_tests_results`: verification/independent_calc -> FAIL (value=76780167.6865008324)
  - `new_tests_results`: physics/energy_vs_U -> FAIL (value=0)
  - `new_tests_results`: benchmark/qmc_dmrg_rmse -> FAIL (value=827101.6758152760)
  - `new_tests_results`: benchmark/qmc_dmrg_mae -> FAIL (value=489514.2560637082)
  - `new_tests_results`: benchmark/qmc_dmrg_within_error_bar -> FAIL (value=0.000000)
  - `new_tests_results`: benchmark/qmc_dmrg_ci95_halfwidth -> FAIL (value=418571.1994309005)
  - `new_tests_results`: benchmark/external_modules_rmse -> FAIL (value=13971.9631590425)
  - `new_tests_results`: benchmark/external_modules_mae -> FAIL (value=7289.9767586145)
  - `new_tests_results`: benchmark/external_modules_within_error_bar -> FAIL (value=25.000000)
  - `new_tests_results`: cluster_scale/cluster_pair_trend -> FAIL (value=0)
  - `new_tests_results`: cluster_scale/cluster_energy_trend -> FAIL (value=0)
  - `numerical_stability_suite`: dt_sweep_extended::hubbard_hts_core -> FAIL (pairing_dt_2.00=0.2724181524)
  - `numerical_stability_suite`: energy_conservation::hubbard_hts_core -> FAIL (energy_density_drift_max=1.6811491139)
  - `numerical_stability_suite`: energy_conservation::qcd_lattice_proxy -> FAIL (energy_density_drift_max=0.0228974977)
  - `numerical_stability_suite`: energy_conservation::quantum_field_noneq -> FAIL (energy_density_drift_max=0.1808121645)
  - `numerical_stability_suite`: energy_conservation::dense_nuclear_proxy -> FAIL (energy_density_drift_max=0.0226575244)
  - `numerical_stability_suite`: energy_conservation::quantum_chemistry_proxy -> FAIL (energy_density_drift_max=0.2191537129)
  - `numerical_stability_suite`: energy_conservation::spin_liquid_exotic -> FAIL (energy_density_drift_max=0.0642715881)
  - `numerical_stability_suite`: energy_conservation::topological_correlated_materials -> FAIL (energy_density_drift_max=0.1071239402)
  - `numerical_stability_suite`: energy_conservation::correlated_fermions_non_hubbard -> FAIL (energy_density_drift_max=0.1306125356)
  - `numerical_stability_suite`: energy_conservation::multi_state_excited_chemistry -> FAIL (energy_density_drift_max=0.2242114965)
  - `numerical_stability_suite`: energy_conservation::bosonic_multimode_systems -> FAIL (energy_density_drift_max=0.0291621275)
  - `numerical_stability_suite`: energy_conservation::multiscale_nonlinear_field_models -> FAIL (energy_density_drift_max=0.1528298840)
  - `numerical_stability_suite`: energy_conservation::far_from_equilibrium_kinetic_lattices -> FAIL (energy_density_drift_max=0.0884909741)
  - `numerical_stability_suite`: energy_conservation::multi_correlated_fermion_boson_networks -> FAIL (energy_density_drift_max=0.0939023278)

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
- Score pondéré indépendant: **68.54%**
