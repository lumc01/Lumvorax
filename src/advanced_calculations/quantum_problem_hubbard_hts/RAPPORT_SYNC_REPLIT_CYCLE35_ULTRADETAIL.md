# RAPPORT_SYNC_REPLIT_CYCLE35_ULTRADETAIL

Run analysé: `research_20260314T065135Z_7551`

## Phase 1 — Synchronisation / intégrité
- total_runs_audited: 35
- runs_with_missing_files: 0

## Phase 2 — Résultats par problème (pourcentages exacts)
| Problème | Progression | Reste à valider |
|---|---:|---:|
| bosonic_multimode_systems | 75.00% | 25.00% |
| correlated_fermions_non_hubbard | 75.00% | 25.00% |
| dense_nuclear_fullscale | 75.00% | 25.00% |
| far_from_equilibrium_kinetic_lattices | 75.00% | 25.00% |
| hubbard_hts_core | 75.00% | 25.00% |
| multi_correlated_fermion_boson_networks | 75.00% | 25.00% |
| multi_state_excited_chemistry | 75.00% | 25.00% |
| multiscale_nonlinear_field_models | 75.00% | 25.00% |
| qcd_lattice_fullscale | 75.00% | 25.00% |
| quantum_chemistry_fullscale | 75.00% | 25.00% |
| quantum_field_noneq | 75.00% | 25.00% |
| spin_liquid_exotic | 75.00% | 25.00% |
| topological_correlated_materials | 75.00% | 25.00% |

## Phase 3 — Vérification exhaustive
- tests_critiques: PASS=11, OBSERVED=1, FAIL=0
- Détails T1..T12:
  - T1_finite_size_scaling_coverage: PASS (11 sizes: [56, 64, 72, 80, 81, 90, 96, 99, 100, 120, 121])
  - T2_parameter_sweep_u_over_t: PASS (12 values: [4.0625, 4.533333, 5.384615, 6.571429, 7.047619, 7.090909, 7.166667, 8.0, 8.666667, 11.666667, 12.857143, 13.75])
  - T3_temperature_sweep_coverage: PASS (9 values: [5.7, 40.0, 60.0, 80.0, 95.0, 110.0, 130.0, 155.0, 180.0])
  - T4_boundary_condition_traceability: PASS (['open', 'periodic'])
  - T5_qmc_dmrg_crosscheck: PASS (within_error_bar=15/15 (100.00%);trend_pairing=0.9927;trend_energy=1.0000)
  - T6_sign_problem_watchdog: OBSERVED (median(|sign_ratio|)=0.104167)
  - T7_energy_pairing_scaling: PASS (min_abs_pearson=0.616810)
  - T8_critical_minimum_window: PASS (hubbard_hts_core:ok; qcd_lattice_fullscale:ok; quantum_field_noneq:ok; dense_nuclear_fullscale:ok; quantum_chemistry_fullscale:ok; spin_liquid_exotic:ok; topological_correlated_materials:ok; correlated_fermions_non_hubbard:ok; multi_state_excited_chemistry:ok; bosonic_multimode_systems:ok; multiscale_nonlinear_field_models:ok; far_from_equilibrium_kinetic_lattices:ok; multi_correlated_fermion_boson_networks:ok)
  - T9_dt_sensitivity_index: PASS (max_dt_sensitivity_fullscale=0.001210)
  - T10_spatial_correlations_required: PASS (rows=65 from integration_spatial_correlations.csv)
  - T11_entropy_required: PASS (rows=13 from integration_entropy_observables.csv)
  - T12_alternative_solver_required: PASS (rows=16; global_status=NA; independent_status=NA)

## Phase 4 — Analyse scientifique
- Drift inter-run (dernier vs précédent):
  - energy: max_abs_diff=0.0, mean_abs_diff=0.0
  - pairing: max_abs_diff=0.0, mean_abs_diff=0.0
  - sign_ratio: max_abs_diff=0.0, mean_abs_diff=0.0
  - elapsed_ns: max_abs_diff=482135425.0, mean_abs_diff=204423275.13157895
  - cpu_percent: max_abs_diff=23.72, mean_abs_diff=23.64438596491228
  - mem_percent: max_abs_diff=0.10000000000000009, mean_abs_diff=0.03201754385964917

## Phase 5 — Métriques bas niveau (runtime/hardware fullscale)
| Problème | Qubits fullscale | Module % | CPU% | MEM% | calc/s | latence ns/step |
|---|---:|---:|---:|---:|---:|---:|
| bosonic_multimode_systems | 80 | 6.30 | 17.10 | 80.38 | 126174.25 | 756529.55 |
| correlated_fermions_non_hubbard | 90 | 7.75 | 17.10 | 80.38 | 112339.74 | 853067.08 |
| dense_nuclear_fullscale | 72 | 5.60 | 17.10 | 80.38 | 135306.00 | 703871.90 |
| far_from_equilibrium_kinetic_lattices | 99 | 8.56 | 17.10 | 80.38 | 101758.75 | 941770.00 |
| hubbard_hts_core | 100 | 10.39 | 17.10 | 80.38 | 98418.20 | 979783.89 |
| multi_correlated_fermion_boson_networks | 100 | 8.52 | 17.10 | 80.38 | 102192.48 | 937772.88 |
| multi_state_excited_chemistry | 81 | 6.71 | 17.10 | 80.38 | 124170.74 | 770327.83 |
| multiscale_nonlinear_field_models | 96 | 7.87 | 17.10 | 80.38 | 105840.37 | 903740.00 |
| qcd_lattice_fullscale | 81 | 6.48 | 17.10 | 80.38 | 122768.46 | 777516.82 |
| quantum_chemistry_fullscale | 56 | 4.64 | 17.10 | 80.38 | 171289.46 | 557270.41 |
| quantum_field_noneq | 64 | 5.12 | 17.10 | 80.38 | 147867.75 | 644076.19 |
| spin_liquid_exotic | 120 | 11.21 | 17.10 | 80.38 | 84475.44 | 1138246.15 |
| topological_correlated_materials | 121 | 10.84 | 17.10 | 80.38 | 83849.62 | 1144906.76 |

## Phase 6 — Réponse point par point (question/analyse/réponse/solution)
### Q1. Où en est chaque problème précisément ?
- Analyse: voir tableau progression détaillée ci-dessus.
- Réponse: progression entre 79.41% et 93.03% selon problème.
- Solution: pousser les items T7/T8 via campagnes ciblées (U/t, T, dt, fenêtre critique).
### Q2. Combien il reste pour valider à 100% ?
- Analyse: reste = 100 - solution_progress_percent par problème.
- Réponse: reste entre 6.97% et 20.59%.
- Solution: automatiser dt/2-dt-2dt + alignment minimum critique multi-problèmes.
### Q3. Quoi de `analysechatgpt2.md` est déjà intégré / non intégré ?
- Intégré:
  - Rebond/minimum critique (T8=PASS)
  - Hypothèse artefact numérique via fullscale dt (T9=PASS)
  - Corrélations spatiales/fullscale corrélations 2-points (T10=PASS)
  - Validation structure multi-échelle/scaling (T7=PASS)
  - Fullscale entropie pour compétition de phases/pseudogap (T11=PASS)
- Non intégré / partiellement intégré:
  - Flux RG inversé: nécessite campagne dédiée multi-U/t, multi-tail... + solveurs indépendants

## Phase 7 — Correctifs proposés (simultanés)
1. Ajouter gate stricte T7>0.98 par problème + fail explicite.
2. Ajouter campagne automatique centrée sur 600-800 pas avec résolution fine.
3. Ajout d’un garde checksum obligatoire dans tous les runs (aucun missing autorisé).
4. Ajouter sweep lattice 12x12,14x14,16x16 en standard.

## Phase 8 — Intégration technique
- Rapport ultra-détaillé cycle35 généré automatiquement.
- Aucun ancien rapport modifié.

## Phase 9 — Traçabilité
- readiness_global: 81.32%
- rollout_status: SHADOW_BLOCKED / rollback=ENABLED

## Phase 10 — Commandes exactes reproductibles
```bash
ROOT="src/advanced_calculations/quantum_problem_hubbard_hts"
RUN_DIR="$(ls -1 "$ROOT/results" | rg "^research_" | tail -n 1)"
RUN_PATH="$ROOT/results/$RUN_DIR"
python3 "$ROOT/tools/post_run_v4next_integration_status.py" "$RUN_PATH"
python3 "$ROOT/tools/v4next_rollout_controller.py" "$RUN_PATH" full
python3 "$ROOT/tools/post_run_v4next_rollout_progress.py" "$RUN_PATH"
python3 "$ROOT/tools/post_run_v4next_realtime_evolution.py" "$ROOT" "$RUN_PATH"
python3 "$ROOT/tools/post_run_low_level_telemetry.py" "$RUN_PATH"
python3 "$ROOT/tools/post_run_advanced_observables_pack.py" "$RUN_PATH"
python3 "$ROOT/tools/post_run_chatgpt_critical_tests.py" "$RUN_PATH"
python3 "$ROOT/tools/post_run_problem_solution_progress.py" "$RUN_PATH"
python3 "$ROOT/tools/post_run_authenticity_audit.py" "$ROOT" "$RUN_PATH"
python3 "$ROOT/tools/exhaustive_replit_audit.py" --results-dir "$ROOT/results" --out-csv "$ROOT/AUDIT_EXHAUSTIF_REPLIT_RUNS.csv" --out-drift-csv "$ROOT/AUDIT_EXHAUSTIF_REPLIT_DRIFT.csv"
python3 "$ROOT/tools/post_run_cycle35_exhaustive_report.py" "$ROOT" "$RUN_PATH"
```
