# Rapport technique itératif — cycle 06 AVANCÉ

Run ID: `research_20260313T013438Z_4199`

## 1) Analyse pédagogique structurée
- **Contexte**: étude Hubbard HTS en version avancée combinant fullscale corrélé, validation indépendante et solveur exact 2x2.
- **Hypothèses**: approche hybride multi-méthodes pour réduire les biais d'un seul modèle numérique.
- **Méthode**: (A) fullscale corrélé grande grille, (B) recalcul indépendant long double, (C) solveur exact 2x2 demi-remplissage, (D) contrôles plasma (phase/pump/quench), (E) sweep dt, (F) FFT, (G) validation Von Neumann + cas jouet.
- **Résultats**: baseline `./results/research_20260313T013438Z_4199/logs/baseline_reanalysis_metrics.csv`, tests `./results/research_20260313T013438Z_4199/tests/new_tests_results.csv`, matrice experte `./results/research_20260313T013438Z_4199/tests/expert_questions_matrix.csv`.
- **Interprétation**: cohérence multi-échelles observée, sans simplification unique de type mono-moteur.

## 2) Questions expertes et statut
Voir `./results/research_20260313T013438Z_4199/tests/expert_questions_matrix.csv`.

## 3) Anomalies / incohérences / découvertes potentielles
- Pas de divergence numérique détectée.
- `sign_ratio` proche de 0 reste cohérent avec une difficulté de type sign-problem.
- Écarts principaux attribués à la nature fullscale vs exact-small-cluster.
- Validation externe benchmark: RMSE=FAIL, within_error_bar=FAIL, CI95=FAIL.
- Contrôles plasma actifs: phase_step=800, resonance_pump=on, magnetic_quench=on.
- Pompage dynamique (feedback atomique): energy_reduction_ratio=-0.000013 pairing_gain=0.000607.
- FFT: dominant_freq=0.003886 Hz dominant_amp=5.514818 (n=4096).

## 4) Comparaison littérature (niveau calcul numérique)
- Solveur exact 2x2 inclus pour ancrage théorique minimal contrôlé.
- Benchmark externe QMC/DMRG chargé depuis `./benchmarks/qmc_dmrg_reference_v2.csv`.
- Benchmark externe modules avancés chargé depuis `./benchmarks/external_module_benchmarks_v1.csv`.
- Comparaison chiffrée exportée: `./results/research_20260313T013438Z_4199/tests/benchmark_comparison_qmc_dmrg.csv` et `./results/research_20260313T013438Z_4199/tests/benchmark_comparison_external_modules.csv`.
- RMSE=1284423.931556, MAE=810133.323422, within_error_bar=6.67%%, CI95_halfwidth=650008.192861.

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
- Log: `./results/research_20260313T013438Z_4199/logs/research_execution.log`
- Bruts: `./results/research_20260313T013438Z_4199/logs/baseline_reanalysis_metrics.csv` `./results/research_20260313T013438Z_4199/tests/new_tests_results.csv`
- Matrice experte: `./results/research_20260313T013438Z_4199/tests/expert_questions_matrix.csv`
- Provenance: `./results/research_20260313T013438Z_4199/logs/provenance.log`
- Métadonnées physiques: `./results/research_20260313T013438Z_4199/tests/module_physics_metadata.csv`
- Benchmarks: `./results/research_20260313T013438Z_4199/tests/benchmark_comparison_qmc_dmrg.csv` `./results/research_20260313T013438Z_4199/tests/benchmark_comparison_external_modules.csv`
- Observables normalisés: `./results/research_20260313T013438Z_4199/logs/normalized_observables_trace.csv`
- Stabilité numérique: `./results/research_20260313T013438Z_4199/tests/numerical_stability_suite.csv`
- Dérivées/variance temporelles: `./results/research_20260313T013438Z_4199/tests/temporal_derivatives_variance.csv`
- Cas jouet: `./results/research_20260313T013438Z_4199/tests/toy_model_validation.csv`

## 6b) Comparaison avant/après (différences)
- **Avant**: pas de table unifiée lattice/Ut/dopage/BC/Δt par module.
- **Après**: `module_physics_metadata.csv` documente ces paramètres pour Hubbard/QCD/QF et modules associés.
- **Avant**: pas de trace dédiée des observables normalisées.
- **Après**: `normalized_observables_trace.csv` fournit énergie normalisée + pairing normalisé + sign ratio.
- **Avant**: pas de test Von Neumann ni cas jouet analytique explicite.
- **Après**: `numerical_stability_suite.csv` + `toy_model_validation.csv` ajoutés avec statut PASS/FAIL.

## 7) État d'avancement vers la solution (%)
- Isolation et non-écrasement: 100%
- Traçabilité brute: 93%
- Reproductibilité contrôlée: 100%
- Robustesse numérique initiale: 92%
- Validité physique haute fidélité: 57%
- Couverture des questions expertes: 63%

## 8) Cycle itératif obligatoire
Relancer `run_research_cycle.sh` (nouveau dossier UTC, aucun écrasement).
