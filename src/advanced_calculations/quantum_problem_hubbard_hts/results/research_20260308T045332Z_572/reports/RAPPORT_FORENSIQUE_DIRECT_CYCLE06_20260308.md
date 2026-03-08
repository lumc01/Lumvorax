# Rapport forensique direct — cycle research_20260308T045332Z_572

- Généré (UTC): 2026-03-08T05:15:54Z
- Source analysée: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T045332Z_572`
- Méthode: lecture directe des CSV/logs existants, sans altération des artefacts antérieurs.

## 1) Synchronisation et intégrité
- Dépôt local synchronisé avec `origin/main` (même commit).
- Vérification SHA256 du manifeste `logs/checksums.sha256`: majorité OK, 4 fichiers manquants signalés (environment/provenance/runtime persistent).

## 2) Avancement global (pourcentages exacts)
- Progression globale annoncée (analysis_scientifique_summary): **70.79%**.
- Progression revue indépendante: **68.14%**.
- Questions expertes traitées complètement: **57.89%** (11/19).
- Couverture tests nouveaux: PASS=21, FAIL=10, OBSERVED=49, total=80.

## 3) État d'avancement par simulation (et reste à valider)
| Module | Progression % | Reste % | Indice clé |
|---|---:|---:|---|
| hubbard_hts_core | 70.03 | 29.97 | energy_pairing_corr=0.896; sign_watchdog |
| quantum_chemistry_proxy | 69.91 | 30.09 | energy_pairing_corr=0.894; sign_watchdog |
| multi_state_excited_chemistry | 69.44 | 30.56 | energy_pairing_corr=0.887; sign_watchdog |
| multiscale_nonlinear_field_models | 65.76 | 34.24 | energy_pairing_corr=0.831; sign_watchdog |
| quantum_field_noneq | 63.43 | 36.57 | energy_pairing_corr=0.796; sign_watchdog |
| spin_liquid_exotic | 62.33 | 37.67 | energy_pairing_corr=0.930; sign_watchdog |
| topological_correlated_materials | 61.93 | 38.07 | energy_pairing_corr=0.924; sign_watchdog |
| multi_correlated_fermion_boson_networks | 59.82 | 40.18 | energy_pairing_corr=0.892; sign_watchdog |
| correlated_fermions_non_hubbard | 59.75 | 40.25 | energy_pairing_corr=0.891; sign_watchdog |
| dense_nuclear_proxy | 58.60 | 41.40 | energy_pairing_corr=0.874; sign_watchdog |
| bosonic_multimode_systems | 57.35 | 42.65 | energy_pairing_corr=0.855; sign_watchdog |
| far_from_equilibrium_kinetic_lattices | 56.47 | 43.53 | energy_pairing_corr=0.842; sign_watchdog |
| qcd_lattice_proxy | 56.41 | 43.59 | energy_pairing_corr=0.841; sign_watchdog |

- **Moyenne inter-modules**: 62.40% (reste moyen 37.60%).

## 4) Résultats scientifiques (énergie, corrélations, pairing, signe)
- Pairing vs température (sens_T_60→180): 0.7999 → 0.5194, baisse relative 35.06%.
- Énergie vs U (sens_U_6→12): 1013869.2 → 2097060.8, hausse relative 106.84%.
- Sign-ratio (sign_ratio_min): min=-0.1429, max=-0.0002, moyenne=-0.0593.

## 5) Anomalies, instabilités, risques numériques
- Stabilité numérique: PASS=4, FAIL=26 sur 30 entrées (échec 86.67%).
- Drift énergie (energy_density_drift_max): min=0.1110, max=0.2872, moyenne=0.1838.
- Spectral radius Von Neumann = 1.0002246148 (>1): instabilité marginale cumulative probable.
- Top dérives inter-run (integration_run_drift_monitor):
| metric | max_abs_diff | mean_abs_diff |
|---|---:|---:|
| elapsed_ns | 277636551.0 | 56010819.26885246 |
| energy | 0.0 | 0.0 |
| pairing | 0.0 | 0.0 |
| sign_ratio | 0.0 | 0.0 |
| prev_malformed_rows | n/a | n/a |

## 6) Comparaison littérature / références
- QMC/DMRG: dans barre d’erreur = 53.33%, erreur relative moyenne = 2.08%.
- Modules externes: dans barre d’erreur = 0.00%, erreur relative moyenne = 1357.64%.
- Conclusion: calibration absolue insuffisante malgré des tendances qualitatives utilisables.

## 7) Questions expertes: couverture et manques
| ID | Catégorie | Statut | Lecture pédagogique |
|---|---|---|---|
| Q1 | methodology | complete | Réponse exploitable |
| Q2 | methodology | complete | Réponse exploitable |
| Q3 | numerics | partial | Partiel: test ciblé requis |
| Q4 | numerics | partial | Partiel: test ciblé requis |
| Q5 | theory | complete | Réponse exploitable |
| Q6 | theory | complete | Réponse exploitable |
| Q7 | theory | complete | Réponse exploitable |
| Q8 | experiment | complete | Réponse exploitable |
| Q11 | literature | partial | Partiel: test ciblé requis |
| Q9 | experiment | complete | Réponse exploitable |
| Q10 | limits | complete | Réponse exploitable |
| Q12 | physics_open | partial | Partiel: test ciblé requis |
| Q13 | numerics_open | complete | Réponse exploitable |
| Q14 | numerics_open | partial | Partiel: test ciblé requis |
| Q15 | experiment_open | partial | Partiel: test ciblé requis |
| Q16 | numerics_open | partial | Partiel: test ciblé requis |
| Q17 | methodology_open | complete | Réponse exploitable |
| Q18 | controls_open | complete | Réponse exploitable |
| Q19 | coverage_open | partial | Partiel: test ciblé requis |

- Backlog questions ouvertes: 5 entrées, statuts={'Open': 5}.

## 8) Classement complet par risque numérique
1. **benchmark** — 0% pass famille (écarts de calibration majeurs).
2. **dt_sweep** — convergence temporelle non validée (FAIL explicite).
3. **cluster_scale** — ~5.56% pass, robustesse échelle insuffisante.
4. **exact_solver** — 33.33% pass (2 observations non converties en validation).
5. **sensitivity/dynamic_pumping** — observations sans seuils d’acceptation.
6. **energy_conservation/von_neumann** — fail massif dans numerical_stability_suite.

## 9) Tableau “question ouverte → test ciblé à lancer”
| Question ouverte | Test ciblé | Critère de validation |
|---|---|---|
| Convergence en dt | balayage dt + extrapolation Richardson | variation observable <1% |
| Écart benchmark | normaliser énergie/site et ré-étalonner unités | MAE relative <10% |
| Instabilité spectrale | analyse Von Neumann mode-k | rayon spectral ≤1.0 ± 1e-5 |
| Drift long terme | 3 seeds × 3 horizons >10k pas | drift énergie <5% |
| Réalisme expérimental | mapping proxy→ARPES/STM + fit qualitatif | cohérence de tendance validée |

## 10) Commandes exactes et reproductibles
```bash
git fetch origin --prune
git rev-parse HEAD && git rev-parse origin/main
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T045332Z_572
sha256sum -c logs/checksums.sha256
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```
