# Addendum forensique pédagogique — Validation de la dernière correction (sans réexécution)

## 1) Méthode appliquée (ce qui a été fait exactement)
- Synchronisation avec le dépôt distant `https://github.com/lumc01/Lumvorax.git` via `git fetch origin --prune` (mise à jour locale des branches distantes).
- Lecture **directe** des CSV/logs du run demandé, sans relancer de simulation et sans modifier les anciens rapports Markdown.
- Extraction manuelle/programmée des métriques de validation (PASS/FAIL/OBSERVED), puis interprétation module par module.

## 2) Verdict scientifique global sur la correction récente
**Conclusion courte : non, la correction n’est pas validée à 100% scientifiquement à ce stade.**

### Introduction (thèse + contexte)
La correction récente améliore la traçabilité et la couverture structurelle (fichiers, métadonnées, exécution multi-modules), cependant la validation physique forte (accord solveurs indépendants, cohérence des lois d’échelle) reste insuffisante.

### Développement (argumentation)
- De plus, le tableau global de couverture donne **42/148 PASS = 28.3784%**.
- En outre, les tests critiques affichent 7 PASS, 2 OBSERVED, 3 FAIL (soit 58,33% PASS strict sur ce sous-ensemble).
- Cependant, le cross-check QMC/DMRG est à **0/15 = 0.00%** dans les barres d’erreur (échec critique).
- De même, les benchmarks externes sont à **0/16 = 0.00%** dans les barres d’erreur.
- Néanmoins, les gates d’intégrité/coverage sont PASS, ce qui signifie: pipeline structuré correct, mais contenu scientifique encore insuffisant.

### Conclusion (solution + clôture)
Donc, la correction est **partiellement acceptable** sur la forme (ingénierie des tests), mais **non acceptable à 100%** sur le fond scientifique tant que T5/T7/T12 restent en échec.

## 3) Pourcentage d’avancement et reste à faire (par simulation / problème)
Hypothèse de score fournie par le pipeline: `solution_progress_percent` par problème.

| Problème | Avancement (%) | Reste (%) | Corrélation énergie-pairing | Diagnostic pédagogique |
|---|---:|---:|---:|---|
| bosonic_multimode_systems | 42.00 | 58.00 | -0.859 | anti-corrélation forte (proxy non validé physiquement) |
| correlated_fermions_non_hubbard | 42.00 | 58.00 | -0.691 | anti-corrélation faible à modérée |
| dense_nuclear_proxy | 42.00 | 58.00 | -0.925 | anti-corrélation forte (proxy non validé physiquement) |
| far_from_equilibrium_kinetic_lattices | 42.00 | 58.00 | -0.838 | anti-corrélation modérée |
| hubbard_hts_core | 42.00 | 58.00 | -0.869 | anti-corrélation forte (proxy non validé physiquement) |
| multi_correlated_fermion_boson_networks | 42.00 | 58.00 | -0.729 | anti-corrélation modérée |
| multi_state_excited_chemistry | 42.00 | 58.00 | -0.580 | anti-corrélation faible à modérée |
| multiscale_nonlinear_field_models | 42.00 | 58.00 | -0.764 | anti-corrélation modérée |
| qcd_lattice_proxy | 42.00 | 58.00 | -0.928 | anti-corrélation forte (proxy non validé physiquement) |
| quantum_chemistry_proxy | 42.00 | 58.00 | -0.869 | anti-corrélation forte (proxy non validé physiquement) |
| quantum_field_noneq | 42.00 | 58.00 | -0.640 | anti-corrélation faible à modérée |
| spin_liquid_exotic | 42.00 | 58.00 | -0.895 | anti-corrélation forte (proxy non validé physiquement) |
| topological_correlated_materials | 42.00 | 58.00 | -0.668 | anti-corrélation faible à modérée |

Lecture simple: chaque problème est actuellement à **42,00%**, donc **58,00% restant** pour une validation complète.

## 4) Détails des tests critiques (nouveaux tests de la dernière correction)
| Test | Statut | Pourcentage/valeur | Interprétation pédagogique | Action de déblocage |
|---|---|---|---|---|
| T1_finite_size_scaling_coverage | PASS | 11 sizes: [56, 64, 72, 80, 81, 90, 96, 99, 100, 120, 121] | pré-requis structurel rempli. | Add larger lattices if FAIL (e.g. 12x12,16x16) |
| T2_parameter_sweep_u_over_t | PASS | 12 values: [4.0625, 4.533333, 5.384615, 6.571429, 7.047619, 7.090909, 7.166667, 8.0, 8.666667, 11.666667, 12.857143, 13.75] | pré-requis structurel rempli. | Extend sweep grid if FAIL |
| T3_temperature_sweep_coverage | PASS | 13 values: [48.0, 55.0, 60.0, 70.0, 80.0, 85.0, 95.0, 100.0, 110.0, 125.0, 140.0, 150.0, 180.0] | pré-requis structurel rempli. | Extend T grid if FAIL |
| T4_boundary_condition_traceability | PASS | ['periodic_proxy'] | pré-requis structurel rempli. | Populate boundary_conditions in metadata |
| T5_qmc_dmrg_crosscheck | FAIL | within_error_bar=0/15 | 0% d’accord aux barres d’erreur => modèle non calibré face à référence indépendante. | Refresh benchmark tables or fix model if FAIL |
| T6_sign_problem_watchdog | OBSERVED | median(|sign_ratio|)=0.070707 | signal surveillé, pas encore un succès strict. | Keep auditing if OBSERVED |
| T7_energy_pairing_scaling | FAIL | min_pearson=-0.927617 | corrélation min négative alors que seuil >0.98 demandé. | Investigate outlier problem if FAIL |
| T8_critical_minimum_window | OBSERVED | hubbard_hts_core:off; qcd_lattice_proxy:off; quantum_field_noneq:off; dense_nuclear_proxy:off; quantum_chemistry_proxy:off; spin_liquid_exotic:off; topological_correlated_materials:off; correlated_fermions_non_hubbard:off; multi_state_excited_chemistry:off; bosonic_multimode_systems:off; multiscale_nonlinear_field_models:off; far_from_equilibrium_kinetic_lattices:off; multi_correlated_fermion_boson_networks:off | signal surveillé, pas encore un succès strict. | Re-check time-step normalization if OBSERVED |
| T9_dt_sensitivity_proxy | PASS | max_dt_sensitivity_proxy=0.052535 | pré-requis structurel rempli. | Run explicit dt sweep if FAIL |
| T10_spatial_correlations_required | PASS | rows=65 from integration_spatial_correlations.csv | pré-requis structurel rempli. | Generate via post_run_advanced_observables_pack.py |
| T11_entropy_required | PASS | rows=13 from integration_entropy_observables.csv | pré-requis structurel rempli. | Generate via post_run_advanced_observables_pack.py |
| T12_alternative_solver_required | FAIL | rows=16; global_status=FAIL | campagne solveur alternatif insuffisante pour valider la robustesse. | Generate via post_run_advanced_observables_pack.py + benchmark ingestion |

## 5) Cours pédagogique problème par problème (ce qui bloque et comment débloquer)
### bosonic_multimode_systems
**Introduction (thèse + contexte)**
Ce module simule un système `bosonic_proxy` sur réseau `10x8`, avec `U/t=8.666667`, dopage `0.060000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.859` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### correlated_fermions_non_hubbard
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `10x9`, avec `U/t=7.166667`, dopage `0.180000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.691` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=2 (50.00%), OBSERVED=2 (50.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### dense_nuclear_proxy
**Introduction (thèse + contexte)**
Ce module simule un système `mixed_proxy` sur réseau `9x8`, avec `U/t=13.750000`, dopage `0.300000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.925` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=2 (50.00%), OBSERVED=2 (50.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### far_from_equilibrium_kinetic_lattices
**Introduction (thèse + contexte)**
Ce module simule un système `field_proxy` sur réseau `11x9`, avec `U/t=8.000000`, dopage `0.090000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.838` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### hubbard_hts_core
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `10x10`, avec `U/t=8.000000`, dopage `0.200000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.869` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### multi_correlated_fermion_boson_networks
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `10x10`, avec `U/t=7.047619`, dopage `0.140000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.729` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=0 (0.00%), OBSERVED=4 (100.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### multi_state_excited_chemistry
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `9x9`, avec `U/t=4.533333`, dopage `0.220000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.580` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### multiscale_nonlinear_field_models
**Introduction (thèse + contexte)**
Ce module simule un système `field_proxy` sur réseau `12x8`, avec `U/t=6.571429`, dopage `0.100000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.764` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### qcd_lattice_proxy
**Introduction (thèse + contexte)**
Ce module simule un système `gauge_field` sur réseau `9x9`, avec `U/t=12.857143`, dopage `0.100000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.928` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=0 (0.00%), OBSERVED=4 (100.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### quantum_chemistry_proxy
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `8x7`, avec `U/t=4.062500`, dopage `0.400000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.869` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### quantum_field_noneq
**Introduction (thèse + contexte)**
Ce module simule un système `field_proxy` sur réseau `8x8`, avec `U/t=5.384615`, dopage `0.050000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.640` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=0 (0.00%), OBSERVED=4 (100.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### spin_liquid_exotic
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `12x10`, avec `U/t=11.666667`, dopage `0.120000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.895` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=1 (25.00%), OBSERVED=3 (75.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

### topological_correlated_materials
**Introduction (thèse + contexte)**
Ce module simule un système `fermionic_proxy` sur réseau `11x11`, avec `U/t=7.090909`, dopage `0.150000`, schéma `euler_explicit` et `dt=0.010000`.

**Développement (argumentation)**
De plus, l’avancement déclaré est **42,00%** (reste **58,00%**).
Cependant, l’indice `energy_pairing_corr=-0.668` montre une anti-corrélation (au lieu d’une cohérence forte positive exigée par T7).
En outre, sur les tests forensiques spécifiques au module: PASS=2 (50.00%), OBSERVED=2 (50.00%), FAIL=0 (0.00%).
Néanmoins, les modules indépendants sont marqués PASS 4/4, ce qui garantit une exécution technique, mais pas la justesse physique finale.

**Conclusion (solution + clôture)**
Donc, pour débloquer ce problème: (1) recalibrer paramètres physiques contre référence externe, (2) refaire cross-check QMC/DMRG avec barres d’erreur réalistes, (3) valider fenêtre critique 600–800 pas et critères de transition de phase explicites.

## 6) Nouvelles choses non intégrées (gap analysis)
- Unités physiques explicites et contrôle de cohérence des unités: **non intégré**.
- Validation dt réelle (dt/2, dt, 2dt sur exécutions indépendantes): **non intégrée**.
- Critères formels de transition de phase (order parameter + finite-size scaling): **non intégrés**.
- Accord solveurs indépendants à grande taille de réseau: **non intégré** (0% within error bars).
- Garde-fous de production / rollback en dégradation: **ouvert** tant que métriques scientifiques restent faibles.

## 7) Traduction pédagogique des termes techniques
- **PASS**: test réussi sans ambiguïté.
- **OBSERVED**: mesure observée/surveillée, mais pas au niveau d’un critère strict de validation.
- **FAIL**: test en échec, blocage explicite.
- **Cross-check QMC/DMRG**: comparaison avec méthodes indépendantes reconnues pour vérifier que le simulateur ne « raconte pas une histoire fausse ».
- **Error bar (barre d’erreur)**: marge d’incertitude acceptable; si le modèle sort hors de cette marge, l’accord scientifique n’est pas validé.
- **Finite-size scaling**: vérifier qu’un résultat tient quand la taille du système change (important pour extrapoler vers des systèmes plus grands).

## 8) Questions d’experts, réponses révélées par les résultats, et anomalies
| Question expert | Réponse révélée par ce run | Statut |
|---|---|---|
| Les résultats restent-ils stables sur plusieurs tailles de réseau ? | Oui pour la couverture de tailles (11 tailles), mais pas suffisant seul pour valider la physique. | Partiel |
| L’accord avec solveurs indépendants est-il démontré ? | Non, 0/15 dans les barres d’erreur QMC/DMRG. | Bloquant |
| Les lois d’échelle énergie–pairing sont-elles confirmées ? | Non, corrélation min négative, seuil >0.98 non atteint. | Bloquant |
| Le régime sign problem est-il sous contrôle ? | Surveillé (OBSERVED), médiane 0.070707 au-dessus de la zone « très dure » <0.01. | À surveiller |
| Les observables avancées (corrélations, entropie) existent-elles ? | Oui, T10/T11 PASS. | Positif |

Anomalies notables: écarts relatifs élevés vs benchmarks, notamment sur l’énergie (rel_error ≈ 1.0 dans modules externes), ce qui suggère une possible incompatibilité d’échelle/unité ou de convention d’observable.

## 9) Comparaison à l’état de l’art/littérature (niveau prudent)
Sans importer de nouvelles sources externes dans ce run, la comparaison disponible est celle des benchmarks internes « référence vs modèle ».
- En pratique, une validation de littérature exige qu’une part significative des points tombe dans les barres d’erreur de références indépendantes.
- Ici, le taux observé est 0%, donc la revendication de conformité à la littérature ne peut pas être soutenue scientifiquement à ce stade.

## 10) Nouveaux tests ciblés à lancer ensuite (sans exécution ici)
| Question ouverte | Test ciblé recommandé | Critère de succès |
|---|---|---|
| Unités physiques cohérentes ? | `unit-consistency-gate` sur toutes colonnes d’observables | 100% unités présentes + conversion cohérente |
| Cross-check solveurs indépendants ? | Campagne QMC/DMRG sur réseaux plus grands + mapping strict des observables | >80% points dans error bars |
| Sensibilité temporelle réelle ? | Triplet d’exécutions `(dt/2, dt, 2dt)` avec seeds multiples | écart < seuil défini par observable |
| Transition de phase explicite ? | Test combiné order-parameter + finite-size scaling + intervalle de confiance | seuils publiés et franchis |
| Robustesse production ? | Test de rollback automatique sur dégradation métrique | rollback < latence cible et sans perte d’intégrité |

## 11) Algorithmes/méthodes identifiés et rôle exact
- **Intégration temporelle :** `euler_explicit` (mise à jour explicite pas à pas, simple mais potentiellement sensible au choix de `dt`).
- **Solveurs indépendants (validation croisée) :** QMC, DMRG, ARPES, STM (utilisés comme points de contrôle externes).
- **Métriques forensiques :** proxy de Lyapunov (instabilité dynamique), bootstrap CI95 (robustesse statistique), RMSE cross-validation (réalisme du modèle), pente finite-size (cohérence d’échelle).
- **Gates qualité :** intégrité CSV, couverture modules, présence glossaire/metadata, traçabilité claims/confidence.

## 12) Synthèse finale exécutive
Ainsi, la correction récente est **utile** (meilleure instrumentation), mais **insuffisante** pour une validation scientifique complète. La priorité absolue est de transformer les échecs T5/T7/T12 en PASS et d’augmenter `solution_progress_percent` au-delà de 80% pour chaque problème avant de conclure à une acceptation robuste.
