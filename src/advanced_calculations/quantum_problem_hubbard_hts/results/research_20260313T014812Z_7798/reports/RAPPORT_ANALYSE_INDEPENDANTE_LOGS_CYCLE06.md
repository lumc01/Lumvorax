# Rapport d’analyse indépendante des logs (hors rapport pipeline)

Run analysé: `research_20260313T014812Z_7798`
Horodatage UTC: `2026-03-13T01:48:24.928529+00:00`

## Phase 1 — Synchronisation et intégrité
- Revue indépendante effectuée à partir des artefacts bruts du run (`logs/*.csv`, `tests/*.csv`).
- Fichier checksum global présent: **non**.

## Phase 2 — Analyse des données
- Lignes métriques brutes: **305**
- Trace normalisée: **52**
- Questions expertes: **19**
- Modules metadata: **13**
- Moyenne énergie brute: **2.007009**
- Moyenne pairing brut: **0.838499**
- Moyenne sign ratio brut: **-0.002308**

## Phase 3 — Vérification exhaustive
- Vérification famille par famille des tests PASS/FAIL/OBSERVED:
  - `benchmark`: PASS=0 FAIL=7 OBSERVED=0 (pass=0.00%)
  - `cluster_scale`: PASS=2 FAIL=2 OBSERVED=32 (pass=5.56%)
  - `control`: PASS=3 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `convergence`: PASS=5 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `dt_sweep`: PASS=1 FAIL=0 OBSERVED=3 (pass=25.00%)
  - `dynamic_pumping`: PASS=0 FAIL=0 OBSERVED=4 (pass=0.00%)
  - `exact_solver`: PASS=1 FAIL=0 OBSERVED=2 (pass=33.33%)
  - `physics`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `reproducibility`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `sensitivity`: PASS=0 FAIL=0 OBSERVED=8 (pass=0.00%)
  - `spectral`: PASS=1 FAIL=1 OBSERVED=0 (pass=50.00%)
  - `stability`: PASS=2 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `stress`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)
  - `verification`: PASS=1 FAIL=0 OBSERVED=0 (pass=100.00%)

## Phase 4 — Analyse scientifique
- Couverture questions expertes complètes: **13/19 (68.42%)**
- Stabilité numérique (suite dédiée): **100.00% PASS**
- Points en échec détectés et conservés (pas masqués):
  - `new_tests_results`: spectral/fft_dominant_amplitude -> FAIL (value=5.4941711929)
  - `new_tests_results`: benchmark/qmc_dmrg_rmse -> FAIL (value=1284423.9315556188)
  - `new_tests_results`: benchmark/qmc_dmrg_mae -> FAIL (value=810133.3234220213)
  - `new_tests_results`: benchmark/qmc_dmrg_within_error_bar -> FAIL (value=6.666667)
  - `new_tests_results`: benchmark/qmc_dmrg_ci95_halfwidth -> FAIL (value=650008.1928609951)
  - `new_tests_results`: benchmark/external_modules_rmse -> FAIL (value=30625.1038257904)
  - `new_tests_results`: benchmark/external_modules_mae -> FAIL (value=20344.1671958113)
  - `new_tests_results`: benchmark/external_modules_within_error_bar -> FAIL (value=0.000000)
  - `new_tests_results`: cluster_scale/cluster_pair_trend -> FAIL (value=0)
  - `new_tests_results`: cluster_scale/cluster_energy_trend -> FAIL (value=0)

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
- Score pondéré indépendant: **70.92%**
