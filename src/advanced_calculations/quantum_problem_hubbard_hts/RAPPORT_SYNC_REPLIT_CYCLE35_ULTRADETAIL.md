# RAPPORT_SYNC_REPLIT_CYCLE35_ULTRADETAIL

Run analysé: `research_20260307T162452Z_12025`

## Phase 1 — Synchronisation / intégrité
- total_runs_audited: 35
- runs_with_missing_files: 0

## Phase 2 — Résultats par problème (pourcentages exacts)
| Problème | Progression | Reste à valider |
|---|---:|---:|
| dense_nuclear_proxy | 81.60% | 18.40% |
| hubbard_hts_core | 93.03% | 6.97% |
| qcd_lattice_proxy | 79.41% | 20.59% |
| quantum_chemistry_proxy | 92.91% | 7.09% |
| quantum_field_noneq | 86.43% | 13.57% |

## Phase 3 — Vérification exhaustive
- tests_critiques: PASS=10, OBSERVED=1, FAIL=1
- Détails T1..T12:
  - T1_finite_size_scaling_coverage: PASS (5 sizes: [56, 64, 72, 81, 100])
  - T2_parameter_sweep_u_over_t: PASS (5 values: [4.0625, 5.384615, 8.0, 12.857143, 13.75])
  - T3_temperature_sweep_coverage: PASS (5 values: [60.0, 80.0, 95.0, 140.0, 180.0])
  - T4_boundary_condition_traceability: PASS (['periodic_proxy'])
  - T5_qmc_dmrg_crosscheck: PASS (within_error_bar=15/15)
  - T6_sign_problem_watchdog: PASS (median(|sign_ratio|)=0.001567)
  - T7_energy_pairing_scaling: FAIL (min_pearson=0.796421)
  - T8_critical_minimum_window: OBSERVED (hubbard_hts_core:ok; qcd_lattice_proxy:off; quantum_field_noneq:ok; dense_nuclear_proxy:off; quantum_chemistry_proxy:ok)
  - T9_dt_sensitivity_proxy: PASS (max_dt_sensitivity_proxy=0.055052)
  - T10_spatial_correlations_required: PASS (rows=25 from integration_spatial_correlations.csv)
  - T11_entropy_required: PASS (rows=5 from integration_entropy_observables.csv)
  - T12_alternative_solver_required: PASS (rows=16; global_status=PASS)

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
| dense_nuclear_proxy | 72 | 18.34 | 2.30 | 3.53 | 2106.45 | 45212686.33 |
| hubbard_hts_core | 100 | 25.14 | 2.28 | 3.57 | 2073.86 | 46497151.39 |
| qcd_lattice_proxy | 81 | 19.18 | 2.29 | 3.56 | 2114.57 | 45141314.68 |
| quantum_chemistry_proxy | 56 | 19.07 | 2.30 | 3.51 | 2127.19 | 44873562.27 |
| quantum_field_noneq | 64 | 18.27 | 2.29 | 3.54 | 2114.21 | 45046694.48 |

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
- readiness_global: 93.71%
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
