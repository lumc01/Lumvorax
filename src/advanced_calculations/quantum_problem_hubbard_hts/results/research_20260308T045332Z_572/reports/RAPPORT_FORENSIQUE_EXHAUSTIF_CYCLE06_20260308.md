# Rapport forensique exhaustif (post-review) — research_20260308T045332Z_572

- Généré UTC: 2026-03-08T14:17:32Z
- Pipeline: lecture directe CSV/JSON locaux, sans modification des anciens rapports.

## Expertises mobilisées et incidents rencontrés
- Expertises mobilisées: méthodes numériques, physique de la matière condensée (Hubbard/HTS), validation statistique, audit de traçabilité.
- Incident observé: manifeste SHA global référence 4 fichiers absents (environment/provenance/runtime). Cela réduit l’auditabilité historique complète, mais n’altère pas les CSV/JSON présents.

## 1) Signification (niveau expert → explication pédagogique)
- **Pairing**: indicateur de tendance à former des paires corrélées. S’il baisse avec la température, cela indique que l’agitation thermique détruit les paires.
- **Sign ratio**: indicateur de sévérité du problème de signe; plus proche de zéro (ou négatif instable), plus les estimations peuvent devenir bruitées/biaisées.
- **Convergence en dt**: vérifie que réduire le pas temporel ne change plus significativement la réponse; sinon la solution dépend trop du schéma numérique.
- **Von Neumann spectral radius**: si > 1, les erreurs numériques peuvent s’amplifier au cours du temps.

## 2) Ce que les nouveaux logs apportent réellement (nouvelles informations)
- Avancement global confirmé: 70.79% (vue principale) vs 68.14% (vue indépendante).
- Questions expertes complètement traitées: 57.89% (11/19).
- Campagne nouveaux tests: PASS=21, FAIL=10, OBSERVED=49 (total=80).
- Stabilité numérique: 26/30 FAIL (86.67%).

## 3) État d’avancement par simulation (exact %) + reste
| Module | Progression % | Reste % | Lecture rapide |
|---|---:|---:|---|
| hubbard_hts_core | 70.03 | 29.97 | timeseries_present;metadata_present;energy_pairing_corr=0.896;sign_watchdog;critical_window_ok |
| quantum_chemistry_proxy | 69.91 | 30.09 | timeseries_present;metadata_present;energy_pairing_corr=0.894;sign_watchdog;critical_window_ok |
| multi_state_excited_chemistry | 69.44 | 30.56 | timeseries_present;metadata_present;energy_pairing_corr=0.887;sign_watchdog;critical_window_ok |
| multiscale_nonlinear_field_models | 65.76 | 34.24 | timeseries_present;metadata_present;energy_pairing_corr=0.831;sign_watchdog;critical_window_ok |
| quantum_field_noneq | 63.43 | 36.57 | timeseries_present;metadata_present;energy_pairing_corr=0.796;sign_watchdog;critical_window_ok |
| spin_liquid_exotic | 62.33 | 37.67 | timeseries_present;metadata_present;energy_pairing_corr=0.930;sign_watchdog;critical_window_off |
| topological_correlated_materials | 61.93 | 38.07 | timeseries_present;metadata_present;energy_pairing_corr=0.924;sign_watchdog;critical_window_off |
| multi_correlated_fermion_boson_networks | 59.82 | 40.18 | timeseries_present;metadata_present;energy_pairing_corr=0.892;sign_watchdog;critical_window_off |
| correlated_fermions_non_hubbard | 59.75 | 40.25 | timeseries_present;metadata_present;energy_pairing_corr=0.891;sign_watchdog;critical_window_off |
| dense_nuclear_proxy | 58.60 | 41.40 | timeseries_present;metadata_present;energy_pairing_corr=0.874;sign_watchdog;critical_window_off |
| bosonic_multimode_systems | 57.35 | 42.65 | timeseries_present;metadata_present;energy_pairing_corr=0.855;sign_watchdog;critical_window_off |
| far_from_equilibrium_kinetic_lattices | 56.47 | 43.53 | timeseries_present;metadata_present;energy_pairing_corr=0.842;sign_watchdog;critical_window_off |
| qcd_lattice_proxy | 56.41 | 43.59 | timeseries_present;metadata_present;energy_pairing_corr=0.841;sign_watchdog;critical_window_off |

- Moyenne: 62.40% ; reste moyen: 37.60%.

## 4) Analyse scientifique (énergie, corrélations, pairing, signe)
- Pairing(T): 0.799851 → 0.519435, baisse relative=35.06%.
- Energy(U): 1013869.234005 → 2097060.833266, hausse relative=106.84%.
- Sign ratio (sign_ratio_min): min=-0.142857, max=-0.000235, moyenne=-0.059305.
- Interprétation: tendances thermiques qualitatives cohérentes, mais confiance quantitative limitée par les FAIL benchmark et stabilité.

## 5) Questions expertes par résultat (format Question → Analyse → Réponse → Solution)
### Résultat A — Pairing décroît avec T
- Question: la pente est-elle robuste à dt et taille de système ?
- Analyse: tendance monotone présente, mais dt_sweep contient un FAIL explicite de convergence.
- Réponse: **partielle**.
- Solution: exécuter grille dt fine + extrapolation dt→0 + test multi-lattice.
### Résultat B — Énergie augmente avec U
- Question: l’échelle énergétique est-elle correctement normalisée par site/unité ?
- Analyse: benchmark externe très divergent (erreur relative moyenne extrême).
- Réponse: **partielle**.
- Solution: recalibrage unités + normalisation par site + cross-check non-proxy.
### Résultat C — Reproductibilité inter-run
- Question: les observables clés varient-elles entre runs ?
- Analyse: run_drift_monitor montre diff nulle pour energy/pairing/sign_ratio, dérive seulement sur elapsed_ns.
- Réponse: **complète** pour observables physiques.
- Solution: conserver ce contrôle en gate bloquant.
### Résultat D — Stabilité longue durée
- Question: l’intégrateur est-il stable à long horizon ?
- Analyse: spectral radius >1 et energy_density_drift_max élevé sur de nombreux modules.
- Réponse: **incomplète / risque élevé**.
- Solution: schéma plus dissipatif/implicite et critère de stabilité strict par module.

## 6) Classement par risque numérique (complet)
| Famille | Pass % | Niveau de risque | Pourquoi |
|---|---:|---|---|
| sensitivity | 0.00 | Élevé | PASS=0, FAIL=0, OBSERVED=8 |
| dynamic_pumping | 0.00 | Élevé | PASS=0, FAIL=0, OBSERVED=4 |
| dt_sweep | 0.00 | Élevé | PASS=0, FAIL=1, OBSERVED=3 |
| benchmark | 0.00 | Élevé | PASS=0, FAIL=7, OBSERVED=0 |
| cluster_scale | 5.56 | Élevé | PASS=2, FAIL=2, OBSERVED=32 |
| exact_solver | 33.33 | Élevé | PASS=1, FAIL=0, OBSERVED=2 |
| reproducibility | 100.00 | Faible | PASS=2, FAIL=0, OBSERVED=0 |
| convergence | 100.00 | Faible | PASS=5, FAIL=0, OBSERVED=0 |
| stress | 100.00 | Faible | PASS=1, FAIL=0, OBSERVED=0 |
| verification | 100.00 | Faible | PASS=1, FAIL=0, OBSERVED=0 |
| physics | 100.00 | Faible | PASS=2, FAIL=0, OBSERVED=0 |
| control | 100.00 | Faible | PASS=3, FAIL=0, OBSERVED=0 |
| stability | 100.00 | Faible | PASS=2, FAIL=0, OBSERVED=0 |
| spectral | 100.00 | Faible | PASS=2, FAIL=0, OBSERVED=0 |

## 7) Nouvelles questions expertes non encore intégrées
- Q_missing_units: Are physical units explicit and consistent for all observables? (statut=Open, action=Add units schema and unit-consistency gate)
- Q_solver_crosscheck: Do proxy results match at least one independent non-proxy solver on larger lattice? (statut=Open, action=Maintain benchmark_comparison_qmc_dmrg.csv and extend lattice coverage)
- Q_dt_real_sweep: Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)? (statut=Open, action=Schedule 3-run sweep in CI night job)
- Q_phase_criteria: Are phase-transition criteria explicit (order parameter + finite-size scaling)? (statut=Open, action=Add formal criteria and thresholds)
- Q_production_guardrails: Can V4 NEXT rollback instantly on degraded metrics? (statut=Open, action=Keep rollout controller and rollback contract active)

## 8) Commandes reproductibles
```bash
git fetch origin --prune
git rev-parse HEAD && git rev-parse origin/main
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T045332Z_572
sha256sum -c logs/checksums.sha256
python src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_cycle06_forensic_followup.py
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```
