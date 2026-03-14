# Rapport d’analyse indépendante des logs (hors rapport pipeline)

Run analysé: `research_20260314T163522Z_4296`
Horodatage UTC: `2026-03-14T16:35:48.431659+00:00`

## Phase 1 — Synchronisation et intégrité
- Revue indépendante effectuée à partir des artefacts bruts du run (`logs/*.csv`, `tests/*.csv`).
- Fichier checksum global présent: **non**.

## Phase 2 — Analyse des données
- Lignes métriques brutes: **305**
- Trace normalisée: **52**
- Questions expertes: **23**
- Modules metadata: **13**
- Moyenne énergie brute: **2.007009**
- Moyenne pairing brut: **0.664516**
- Moyenne sign ratio brut: **0.089337**

## Phase 3 — Vérification exhaustive
- Vérification famille par famille des tests PASS/FAIL/OBSERVED:
  - `benchmark`: PASS=5 FAIL=2 OBSERVED=0 (pass=71.43%)
  - `cluster_scale`: PASS=4 FAIL=0 OBSERVED=32 (pass=11.11%)
  - `control`: PASS=3 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `convergence`: PASS=5 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `dt_sweep`: PASS=1 FAIL=0 OBSERVED=3 (pass=25.00%)
  - `dynamic_pumping`: PASS=0 FAIL=0 OBSERVED=4 (pass=0.00%)
  - `exact_solver`: PASS=3 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `physics`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `reproducibility`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `sensitivity`: PASS=0 FAIL=0 OBSERVED=8 (pass=0.00%)
  - `spectral`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stability`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stress`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `verification`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)

## Phase 4 — Analyse scientifique
- Couverture questions expertes complètes: **16/23 (69.57%)**
- Stabilité numérique (suite dédiée): **56.67% PASS**
- Points en échec détectés et conservés (pas masqués):
  - `new_tests_results`: benchmark/external_modules_rmse -> FAIL (value=0.0853804832)
  - `new_tests_results`: benchmark/external_modules_mae -> FAIL (value=0.0748655687)
  - `numerical_stability_suite`: von_neumann::hubbard_hts_core -> FAIL (spectral_radius=1.0000279327)
  - `numerical_stability_suite`: von_neumann::qcd_lattice_fullscale -> FAIL (spectral_radius=1.0000252905)
  - `numerical_stability_suite`: von_neumann::quantum_field_noneq -> FAIL (spectral_radius=1.0000283924)
  - `numerical_stability_suite`: von_neumann::dense_nuclear_fullscale -> FAIL (spectral_radius=1.0000556598)
  - `numerical_stability_suite`: von_neumann::quantum_chemistry_fullscale -> FAIL (spectral_radius=1.0000394245)
  - `numerical_stability_suite`: von_neumann::spin_liquid_exotic -> FAIL (spectral_radius=1.0000514910)
  - `numerical_stability_suite`: von_neumann::topological_correlated_materials -> FAIL (spectral_radius=1.0000293288)
  - `numerical_stability_suite`: von_neumann::correlated_fermions_non_hubbard -> FAIL (spectral_radius=1.0000428436)
  - `numerical_stability_suite`: von_neumann::multi_state_excited_chemistry -> FAIL (spectral_radius=1.0000362141)
  - `numerical_stability_suite`: von_neumann::bosonic_multimode_systems -> FAIL (spectral_radius=1.0000043634)
  - `numerical_stability_suite`: von_neumann::multiscale_nonlinear_field_models -> FAIL (spectral_radius=1.0000620481)
  - `numerical_stability_suite`: von_neumann::far_from_equilibrium_kinetic_lattices -> FAIL (spectral_radius=1.0000269410)
  - `numerical_stability_suite`: von_neumann::multi_correlated_fermion_boson_networks -> FAIL (spectral_radius=1.0000239604)

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
- Score pondéré indépendant: **82.99%**
