# Rapport technique itératif — cycle 06 AVANCÉ

Run ID: `research_20260314T205929Z_3124`

## 1) Analyse pédagogique structurée
- **Contexte**: étude Hubbard HTS en version avancée combinant fullscale corrélé, validation indépendante et solveur exact 2x2.
- **Hypothèses**: approche hybride multi-méthodes pour réduire les biais d'un seul modèle numérique.
- **Méthode**: (A) fullscale corrélé grande grille, (B) recalcul indépendant long double, (C) solveur exact 2x2 demi-remplissage, (D) contrôles plasma (phase/pump/quench), (E) sweep dt, (F) FFT, (G) validation Von Neumann + cas jouet.
- **Résultats**: baseline `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/logs/baseline_reanalysis_metrics.csv`, tests `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/new_tests_results.csv`, matrice experte `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/expert_questions_matrix.csv`.
- **Interprétation**: cohérence multi-échelles observée, sans simplification unique de type mono-moteur.

## 2) Questions expertes et statut
Voir `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/expert_questions_matrix.csv`.

## 3) Anomalies / incohérences / découvertes potentielles
- Pas de divergence numérique détectée.
- `sign_ratio` proche de 0 reste cohérent avec une difficulté de type sign-problem.
- Écarts principaux attribués à la nature fullscale vs exact-small-cluster.
- Validation externe benchmark: RMSE=PASS, within_error_bar=PASS, CI95=PASS.
- Contrôles plasma actifs: phase_step=800, resonance_pump=on, magnetic_quench=on.
- Pompage dynamique (feedback atomique): energy_reduction_ratio=-0.000015 pairing_gain=0.001278.
- FFT: dominant_freq=0.003886 Hz dominant_amp=0.003095 (n=4096).

## 4) Comparaison littérature (niveau calcul numérique)
- Solveur exact 2x2 inclus pour ancrage théorique minimal contrôlé.
- Benchmark externe QMC/DMRG chargé depuis `src/advanced_calculations/quantum_problem_hubbard_hts/benchmarks/qmc_dmrg_reference_runtime.csv`.
- Benchmark externe modules avancés chargé depuis `src/advanced_calculations/quantum_problem_hubbard_hts/benchmarks/external_module_benchmarks_v1.csv`.
- Comparaison chiffrée exportée: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/benchmark_comparison_qmc_dmrg.csv` et `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/benchmark_comparison_external_modules.csv`.
- RMSE=0.016243, MAE=0.009494, within_error_bar=100.00%%, CI95_halfwidth=0.008220.

## 5) Nouveaux tests exécutés
- Reproductibilité
- Convergence
- Extrêmes
- Vérification indépendante
- Solveur exact 2x2
- Sensibilités physiques
- Benchmark externe QMC/DMRG
- Erreurs absolues/relatives + RMSE
- Intervalle de confiance (CI95)
- Critères PASS/FAIL stricts
- Test de stabilité temporelle t>2700 jusqu'à 8700 steps
- Sweep de pas temporel dt=[0.001,0.005,0.010]
- Analyse spectrale FFT
- Tests multi-tailles de clusters (8x8..255x255 autoscaling)

## 6) Traçabilité totale
- Log: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/logs/research_execution.log`
- Bruts: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/logs/baseline_reanalysis_metrics.csv` `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/new_tests_results.csv`
- Matrice experte: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/expert_questions_matrix.csv`
- Provenance: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/logs/provenance.log`
- Métadonnées physiques: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/module_physics_metadata.csv`
- Benchmarks: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/benchmark_comparison_qmc_dmrg.csv` `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/benchmark_comparison_external_modules.csv`
- Observables normalisés: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/logs/normalized_observables_trace.csv`
- Stabilité numérique: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/numerical_stability_suite.csv`
- Dérivées/variance temporelles: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/temporal_derivatives_variance.csv`
- Cas jouet: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260314T205929Z_3124/tests/toy_model_validation.csv`

## 6b) Comparaison avant/après (différences)
- **Avant**: pas de table unifiée lattice/Ut/dopage/BC/Δt par module.
- **Après**: `module_physics_metadata.csv` documente ces paramètres pour Hubbard/QCD/QF et modules associés.
- **Avant**: pas de trace dédiée des observables normalisées.
- **Après**: `normalized_observables_trace.csv` fournit énergie normalisée + pairing normalisé + sign ratio.
- **Avant**: pas de test Von Neumann ni cas jouet analytique explicite.
- **Après**: `numerical_stability_suite.csv` + `toy_model_validation.csv` ajoutés avec statut PASS/FAIL.

## 7) État d'avancement vers la solution (%)
- Isolation et non-écrasement: 100%
- Traçabilité brute: 94%
- Reproductibilité contrôlée: 100%
- Robustesse numérique initiale: 97%
- Validité physique haute fidélité: 91%
- Couverture des questions expertes: 76%

## 8) Cycle itératif obligatoire
Relancer `run_research_cycle.sh` (nouveau dossier UTC, aucun écrasement).
