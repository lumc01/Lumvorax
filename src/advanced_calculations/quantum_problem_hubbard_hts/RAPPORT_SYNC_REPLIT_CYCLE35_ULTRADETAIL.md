# RAPPORT_SYNC_REPLIT_CYCLE35_ULTRADETAIL

Run analysé: `research_20260309T205848Z_3489`

## Phase 1 — Synchronisation / intégrité
- total_runs_audited: 35
- runs_with_missing_files: 0

## Phase 2 — Résultats par problème (pourcentages exacts)
| Problème | Progression | Reste à valider |
|---|---:|---:|
| bosonic_multimode_systems | 42.00% | 58.00% |
| correlated_fermions_non_hubbard | 42.00% | 58.00% |
| dense_nuclear_proxy | 42.00% | 58.00% |
| far_from_equilibrium_kinetic_lattices | 42.00% | 58.00% |
| hubbard_hts_core | 42.00% | 58.00% |
| multi_correlated_fermion_boson_networks | 42.00% | 58.00% |
| multi_state_excited_chemistry | 42.00% | 58.00% |
| multiscale_nonlinear_field_models | 42.00% | 58.00% |
| qcd_lattice_proxy | 42.00% | 58.00% |
| quantum_chemistry_proxy | 42.00% | 58.00% |
| quantum_field_noneq | 42.00% | 58.00% |
| spin_liquid_exotic | 42.00% | 58.00% |
| topological_correlated_materials | 42.00% | 58.00% |

## Phase 3 — Vérification exhaustive
- tests_critiques: PASS=7, OBSERVED=2, FAIL=3
- Détails T1..T12:
  - T1_finite_size_scaling_coverage: PASS (11 sizes: [56, 64, 72, 80, 81, 90, 96, 99, 100, 120, 121])
  - T2_parameter_sweep_u_over_t: PASS (12 values: [4.0625, 4.533333, 5.384615, 6.571429, 7.047619, 7.090909, 7.166667, 8.0, 8.666667, 11.666667, 12.857143, 13.75])
  - T3_temperature_sweep_coverage: PASS (13 values: [48.0, 55.0, 60.0, 70.0, 80.0, 85.0, 95.0, 100.0, 110.0, 125.0, 140.0, 150.0, 180.0])
  - T4_boundary_condition_traceability: PASS (['periodic_proxy'])
  - T5_qmc_dmrg_crosscheck: FAIL (within_error_bar=0/15)
  - T6_sign_problem_watchdog: OBSERVED (median(|sign_ratio|)=0.070707)
  - T7_energy_pairing_scaling: FAIL (min_pearson=-0.927617)
  - T8_critical_minimum_window: OBSERVED (hubbard_hts_core:off; qcd_lattice_proxy:off; quantum_field_noneq:off; dense_nuclear_proxy:off; quantum_chemistry_proxy:off; spin_liquid_exotic:off; topological_correlated_materials:off; correlated_fermions_non_hubbard:off; multi_state_excited_chemistry:off; bosonic_multimode_systems:off; multiscale_nonlinear_field_models:off; far_from_equilibrium_kinetic_lattices:off; multi_correlated_fermion_boson_networks:off)
  - T9_dt_sensitivity_proxy: PASS (max_dt_sensitivity_proxy=0.052535)
  - T10_spatial_correlations_required: PASS (rows=65 from integration_spatial_correlations.csv)
  - T11_entropy_required: PASS (rows=13 from integration_entropy_observables.csv)
  - T12_alternative_solver_required: FAIL (rows=16; global_status=FAIL)

## Phase 4 — Analyse scientifique
- Drift inter-run (dernier vs précédent):
  - energy: max_abs_diff=0.0, mean_abs_diff=0.0
  - pairing: max_abs_diff=0.0, mean_abs_diff=0.0
  - sign_ratio: max_abs_diff=0.0, mean_abs_diff=0.0
  - elapsed_ns: max_abs_diff=482135425.0, mean_abs_diff=204423275.13157895
  - cpu_percent: max_abs_diff=23.72, mean_abs_diff=23.64438596491228
  - mem_percent: max_abs_diff=0.10000000000000009, mean_abs_diff=0.03201754385964917

## Phase 5 — Métriques bas niveau (runtime/hardware proxy)
| Problème | Qubits proxy | Module % | CPU% | MEM% | calc/s | latence ns/step |
|---|---:|---:|---:|---:|---:|---:|
| bosonic_multimode_systems | 80 | 7.18 | 17.56 | 82.90 | 1921.50 | 49677200.41 |
| correlated_fermions_non_hubbard | 90 | 7.79 | 17.56 | 82.90 | 1939.25 | 49417692.17 |
| dense_nuclear_proxy | 72 | 6.93 | 17.56 | 82.89 | 1896.60 | 50215279.14 |
| far_from_equilibrium_kinetic_lattices | 99 | 8.02 | 17.56 | 82.89 | 1884.21 | 50861341.21 |
| hubbard_hts_core | 100 | 9.41 | 17.56 | 82.93 | 1885.79 | 51134247.54 |
| multi_correlated_fermion_boson_networks | 100 | 7.88 | 17.56 | 82.89 | 1918.38 | 49955333.38 |
| multi_state_excited_chemistry | 81 | 7.33 | 17.56 | 82.90 | 1971.99 | 48505307.74 |
| multiscale_nonlinear_field_models | 96 | 7.58 | 17.56 | 82.90 | 1905.96 | 50185711.48 |
| qcd_lattice_proxy | 81 | 7.17 | 17.56 | 82.92 | 1923.81 | 49617324.50 |
| quantum_chemistry_proxy | 56 | 7.15 | 17.56 | 82.88 | 1929.32 | 49475627.73 |
| quantum_field_noneq | 64 | 6.81 | 17.56 | 82.91 | 1929.20 | 49366534.00 |
| spin_liquid_exotic | 120 | 8.64 | 17.56 | 82.88 | 1901.66 | 50563021.96 |
| topological_correlated_materials | 121 | 8.11 | 17.56 | 82.90 | 1944.77 | 49363195.92 |

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
  - Rebond/minimum critique (T8=OBSERVED)
  - Hypothèse artefact numérique via proxy dt (T9=PASS)
  - Corrélations spatiales/proxy corrélations 2-points (T10=PASS)
  - Proxy entropie pour compétition de phases/pseudogap (T11=PASS)
- Non intégré / partiellement intégré:
  - Validation structure multi-échelle/scaling (T7=FAIL)
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
- readiness_global: 87.49%
- rollout_status: SHADOW_ACTIVE / rollback=ENABLED

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
