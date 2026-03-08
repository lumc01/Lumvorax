# Rapport manuel: système de temps, différenciation step vs temps physique, et impacts benchmark (Cycle06)

- Horodatage UTC: 2026-03-08T19:46:55Z
- Nature: analyse manuelle (non-résumé) du code/artefacts existants + plan d’intégration complémentaire.
- Contrainte respectée: aucun ancien rapport/log modifié; nouveaux fichiers sidecar uniquement.

## Expertises mobilisées et erreurs rencontrées en cours de route
- Expertises mobilisées: méthodes numériques, stabilité d’intégrateurs explicites, benchmarking physique (QMC/DMRG), ingénierie de traçabilité.
- Erreur rencontrée: la commande de cycle complet peut générer des artefacts hors périmètre très volumineux; l’analyse a donc été réalisée manuellement à partir des résultats existants pour conserver un patch propre.

## 1) Vérification: est-ce déjà appliqué ?
- Le run contient des tests `dt_sweep` avec `dt_0.001`, `dt_0.005`, `dt_0.010`; cependant, le critère `dt_convergence` est **FAIL**.
- Le schéma est `euler_explicit` sur tous les modules et dt metadata standard = 0.010000.
- Conséquence: la séparation conceptuelle `step` (index numérique) vs `temps physique` n’est pas encore verrouillée par un contrat de validation indépendant.

## 2) Conséquences du système de temps actuel
### Introduction (thèse + contexte)
Le pipeline mélange potentiellement la lecture opérationnelle des `steps` (progression numérique) et l’interprétation physique du temps continu.
### Développement (argumentation)
De plus, `dt_convergence` échoue, ce qui indique une dépendance de la solution au pas de temps. En outre, le drift énergie moyen est 0.1838.
Cependant, les dérives inter-run sur energy/pairing/sign_ratio sont nulles, alors que `elapsed_ns` dérive: cela isole un effet performance plutôt qu’un effet physique.
Néanmoins, le rayon spectral moyen (1.000224615) est > 1, donc une amplification d’erreur cumulative reste plausible.
### Conclusion (solution + clôture)
Donc, sans séparation explicite entre temps numérique et temps physique, les comparaisons benchmark énergie sont fragilisées; ainsi il faut imposer une couche de conversion et des tests d’invariance dédiés.

## 3) Différenciation demandée: step (temps numérique) vs temps physique (avant/après)
| État | Step numérique | Temps physique | Règle de couplage | Risque |
|---|---|---|---|---|
| AVANT (actuel) | index d’itération `step` | implicite via `dt` | couplage direct non audité | confusion sémantique + biais benchmark |
| APRÈS (à intégrer) | `step_index` strictement discret | `t_phys = step_index * dt_phys` | conversion dédiée, testée, versionnée | séparation complète et contrôle mutuel |

## 4) Analyse manuelle des résultats benchmark (par problème, structure complète)
- QMC/DMRG: 7/15 points hors barres.
- Externes: 16/16 points hors barres.

### Problème A — Benchmark énergie Hubbard (QMC/DMRG)
**Introduction (thèse + contexte)**: la tendance globale existe, mais plusieurs points énergie dépassent les barres d’erreur.
**Développement (argumentation)**: de plus, les erreurs absolues énergie dépassent les tolérances; cependant, les points pairing Hubbard restent majoritairement dans barres.
**Conclusion (solution + clôture)**: donc, recalibrer les unités énergie et normaliser par site avant re-validation QMC/DMRG.

### Problème B — Benchmarks modules externes
**Introduction (thèse + contexte)**: tous les points externes sont hors barres, ce qui invalide la calibration inter-modèles.
**Développement (argumentation)**: en outre, les erreurs relatives sont très élevées; néanmoins, des tendances qualitatives internes restent exploitables.
**Conclusion (solution + clôture)**: ainsi, implémenter un pont de mapping observable+unité+volume avant toute comparaison quantitative.

### Problème C — Interprétation des tests OBSERVED
**Introduction (thèse + contexte)**: OBSERVED signifie “mesuré” mais non “validé”.
**Développement (argumentation)**: de même, ces tests donnent de l’information (sensibilité, cluster, pumping), cependant sans seuil PASS/FAIL ils ne bloquent pas les régressions.
**Conclusion (solution + clôture)**: de cette manière, convertir OBSERVED critiques en gates avec critères explicites.

## 5) Questions d’experts + réponses révélées + inconnues restantes
- Q: le temps physique est-il indépendant du step numérique ? R: **partiel** (pas de test d’invariance explicite).
- Q: la stabilité dt est-elle démontrée ? R: **non** (`dt_convergence` FAIL).
- Q: le benchmark externe est-il quantitativement valide ? R: **non** (0% dans barres).
- Q: découverte physique nouvelle confirmée ? R: **non confirmé**; les anomalies sont plus compatibles avec artefacts numériques/calibration.

## 6) Nouveaux tests à exécuter immédiatement (liste exhaustive)
| ID | Question | Test | Critère |
|---|---|---|---|
| TS1 | step indépendant du temps physique ? | invariance à dt compensé (`N*dt` constant) | delta observables <1% |
| TS2 | convergence temporelle robuste ? | grille dt logarithmique + extrapolation dt→0 | pente stable, erreur <1% |
| TS3 | séparation mutuelle complète ? | double horloge (`step_index`, `t_phys`) + audit | 0 confusion de champs |
| TS4 | benchmark énergie recalibré ? | énergie/site + mapping unités | >=90% within_error_bar |
| TS5 | robustesse long terme ? | horizon >10k steps + spectral radius | drift<5%, rayon<=1 |

## 7) Commandes exactes et reproductibles
```bash
git fetch origin --prune
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T045332Z_572
sha256sum -c logs/checksums.sha256
python src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_cycle06_time_system_manual_analysis.py
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

## 8) Données chiffrées clés (avant/après conceptuel)
- overall_progress_pct: 70.789474
- dt_sweep_values: [('dt_0.001', 0.9039900751, 'OBSERVED'), ('dt_0.005', 0.7125458373, 'OBSERVED'), ('dt_0.010', 0.6837103128, 'OBSERVED')]
- energy_drift_mean: 0.183753
- spectral_radius_mean: 1.000224615
- benchmark_qmc_fail_pct: 46.67%
- benchmark_external_fail_pct: 100.00%
