# Rapport forensique complet — validation/invalidation des nouveaux tests (run research_20260308T205539Z_465)

- Horodatage UTC: 2026-03-08T21:12:00Z
- Méthode: lecture directe des CSV/JSON locaux, extraction manuelle assistée (sans modification des anciens rapports).

## Phase 1 — Synchronisation & intégrité
- Progression globale (sci): 70.789474%.
- Progression indépendante: 68.142690%.
- Couverture nouveaux tests: PASS=21, FAIL=10, OBSERVED=49, TOTAL=80.

## Phase 2-3 — Analyse exhaustive A→Z module par module (%, reste)
| Module | Progression % | Reste % | Evidence |
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

- Moyenne inter-modules: 62.40% ; reste moyen 37.60%.

## Phase 4 — Analyse scientifique (énergie, corrélations, pairing, sign ratio)
- Drift énergie (energy_density_drift_max): min=0.110978, max=0.287233, moyenne=0.183753.
- Rayon spectral Von Neumann: min=1.000224615, max=1.000224615, moyenne=1.000224615 (>1 indique risque d’amplification).
- Benchmark QMC/DMRG invalidé partiellement: 7/15 hors barres (46.67%).
- Benchmark modules externes invalidé: 16/16 hors barres (100.00%).

## Phase 5 — Cours pédagogique des termes (niveau débutant)
- **problem**: Nom du module simulé. De plus, lecture: Chaque groupe de lignes correspond à un module. Cependant, risque: Comparer modules sans normalisation peut induire une fausse interprétation.
- **step**: Indice d'avancement numérique. De plus, lecture: +100 = incrément de simulation. Cependant, risque: step n'est pas forcément un temps physique réel.
- **energy**: Observable énergétique interne. De plus, lecture: Peut être négative/positive selon le modèle. Cependant, risque: Sans unités/normalisation, pas de conclusion physique forte.
- **pairing**: Observable de corrélation. De plus, lecture: Ici souvent cumulative. Cependant, risque: Ne pas interpréter comme ordre supraconducteur sans définition formelle.
- **sign_ratio**: Indicateur statistique de signe. De plus, lecture: Proche de 0 peut indiquer compensation des signes. Cependant, risque: Ne suffit pas seul pour conclure sur le sign-problem.
- **cpu_percent**: Charge CPU système. De plus, lecture: Valeur stable = performance stable. Cependant, risque: N'apporte pas de preuve de validité physique.
- **mem_percent**: Usage RAM système. De plus, lecture: Valeur stable = pas de fuite évidente. Cependant, risque: N'évalue pas la justesse scientifique.
- **elapsed_ns**: Temps écoulé en nanosecondes. De plus, lecture: Croissance ~linéaire = coût local stable. Cependant, risque: Ne prouve pas la complexité asymptotique globale.

## Phase 6 — Questions expertes, réponses, solutions
| question_id | question | statut | réponse de ce run | solution |
|---|---|---|---|---|
| Q1 | Le seed est-il contrôlé ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q2 | Deux solveurs indépendants concordent-ils ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q3 | Convergence multi-échelle testée ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q4 | Stabilité aux extrêmes validée ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q5 | Pairing décroît avec T ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q6 | Énergie croît avec U ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q7 | Solveur exact 2x2 exécuté ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q8 | Traçabilité run+UTC ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q11 | Benchmark externe QMC/DMRG (plus de points + clusters) validé ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q9 | Données brutes préservées ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q10 | Cycle itératif explicitement défini ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q12 | Mécanisme physique exact du plasma clarifié ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q13 | Stabilité pour t > 2700 validée ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q14 | Dépendance au pas temporel (dt) testée ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q15 | Comparaison aux expériences réelles (ARPES/STM) ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q16 | Analyse Von Neumann exécutée ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |
| Q17 | Paramètres physiques module-par-module explicités ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q18 | Pompage dynamique (feedback atomique) inclus et tracé ? | complete | Réponse complète dans les artefacts | Conserver gate |
| Q19 | Nouveaux modules avancés CPU/RAM intégrés et benchmarkés individuellement ? | partial | Réponse partielle: validation supplémentaire requise | Ajouter test ciblé + critère quantitatif |

## Phase 7 — Benchmark failures: explication individuelle (Introduction / Développement / Conclusion)
### Groupe QMC/DMRG
#### BF001 — hubbard_hts_core / energy (T=95.000000, U=4.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=652800.0000000000 et modèle=665599.8539853615; abs_error=12799.8539853615 > error_bar=12000.0000000000. Cependant, rel_error=1.96%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF002 — hubbard_hts_core / energy (T=95.000000, U=6.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=1013875.0000000000 et modèle=1056071.8083606362; abs_error=42196.8083606362 > error_bar=9000.0000000000. Cependant, rel_error=4.16%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF003 — hubbard_hts_core / energy (T=95.000000, U=8.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=1374940.0000000000 et modèle=1408808.6816832828; abs_error=33868.6816832828 > error_bar=9000.0000000000. Cependant, rel_error=2.46%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF004 — hubbard_hts_core / energy (T=95.000000, U=10.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=1736003.0000000000 et modèle=1678572.9204088624; abs_error=57430.0795911376 > error_bar=9000.0000000000. Cependant, rel_error=3.31%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF005 — hubbard_hts_core / energy (T=95.000000, U=12.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=2097068.0000000000 et modèle=1986945.4706086284; abs_error=110122.5293913716 > error_bar=9000.0000000000. Cependant, rel_error=5.25%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF006 — hubbard_hts_core / energy (T=95.000000, U=14.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=2458130.0000000000 et modèle=2513276.6561204088; abs_error=55146.6561204088 > error_bar=12000.0000000000. Cependant, rel_error=2.24%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF007 — hubbard_hts_core / energy (T=95.000000, U=16.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=2819200.0000000000 et modèle=2765890.3892134754; abs_error=53309.6107865246 > error_bar=14000.0000000000. Cependant, rel_error=1.89%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

### Groupe External modules
#### BF008 — spin_liquid_exotic / pairing (T=55.000000, U=10.500000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.6840000000 et modèle=0.8262403823; abs_error=0.1422403823 > error_bar=0.0900000000. Cependant, rel_error=20.80%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF009 — spin_liquid_exotic / energy (T=55.000000, U=10.500000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=56120.0000000000 et modèle=1789210.6792104105; abs_error=1733090.6792104105 > error_bar=9500.0000000000. Cependant, rel_error=3088.19%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF010 — topological_correlated_materials / pairing (T=70.000000, U=7.800000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.6360000000 et modèle=0.7736075411; abs_error=0.1376075411 > error_bar=0.0900000000. Cependant, rel_error=21.64%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF011 — topological_correlated_materials / energy (T=70.000000, U=7.800000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=56760.0000000000 et modèle=1320912.8372668442; abs_error=1264152.8372668442 > error_bar=9500.0000000000. Cependant, rel_error=2227.19%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF012 — correlated_fermions_non_hubbard / pairing (T=85.000000, U=8.600000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.5650000000 et modèle=0.7399247335; abs_error=0.1749247335 > error_bar=0.0900000000. Cependant, rel_error=30.96%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF013 — correlated_fermions_non_hubbard / energy (T=85.000000, U=8.600000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=42900.0000000000 et modèle=1074881.8518466770; abs_error=1031981.8518466770 > error_bar=9500.0000000000. Cependant, rel_error=2405.55%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF014 — multi_state_excited_chemistry / pairing (T=48.000000, U=6.800000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.7490000000 et modèle=0.8603385169; abs_error=0.1113385169 > error_bar=0.0900000000. Cependant, rel_error=14.86%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF015 — multi_state_excited_chemistry / energy (T=48.000000, U=6.800000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=36920.0000000000 et modèle=680620.3616473363; abs_error=643700.3616473363 > error_bar=9500.0000000000. Cependant, rel_error=1743.50%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF016 — bosonic_multimode_systems / pairing (T=110.000000, U=5.200000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.5210000000 et modèle=0.7166394136; abs_error=0.1956394136 > error_bar=0.1000000000. Cependant, rel_error=37.55%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF017 — bosonic_multimode_systems / energy (T=110.000000, U=5.200000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=11200.0000000000 et modèle=497023.1416568772; abs_error=485823.1416568772 > error_bar=9500.0000000000. Cependant, rel_error=4337.71%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF018 — multiscale_nonlinear_field_models / pairing (T=125.000000, U=9.200000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.4220000000 et modèle=0.6677829491; abs_error=0.2457829491 > error_bar=0.1000000000. Cependant, rel_error=58.24%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF019 — multiscale_nonlinear_field_models / energy (T=125.000000, U=9.200000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=30240.0000000000 et modèle=997210.7847945302; abs_error=966970.7847945302 > error_bar=9500.0000000000. Cependant, rel_error=3197.65%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF020 — far_from_equilibrium_kinetic_lattices / pairing (T=150.000000, U=8.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.2860000000 et modèle=0.5969764049; abs_error=0.3109764049 > error_bar=0.1100000000. Cependant, rel_error=108.73%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF021 — far_from_equilibrium_kinetic_lattices / energy (T=150.000000, U=8.000000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=35250.0000000000 et modèle=1026562.3412401110; abs_error=991312.3412401110 > error_bar=9500.0000000000. Cependant, rel_error=2812.23%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF022 — multi_correlated_fermion_boson_networks / pairing (T=100.000000, U=7.400000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=0.4870000000 et modèle=0.6962215230; abs_error=0.2092215230 > error_bar=0.1000000000. Cependant, rel_error=42.96%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

#### BF023 — multi_correlated_fermion_boson_networks / energy (T=100.000000, U=7.400000)
**Introduction (thèse + contexte)**: ce point de benchmark est hors tolérance.
**Développement (argumentation)**: de plus, référence=56130.0000000000 et modèle=939846.5570258966; abs_error=883716.5570258966 > error_bar=9500.0000000000. Cependant, rel_error=1574.41%, ce qui empêche la validation quantitative.
**Conclusion (solution + clôture)**: donc, recalibrer unités+normalisation, puis relancer benchmark ciblé avant interprétation physique finale.

## Phase 7bis — Pour chaque OBSERVED: points forts et critiques
| test_family | test_id | point fort | critique | action |
|---|---|---|---|---|
| exact_solver | hubbard_2x2_ground_u4 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| exact_solver | hubbard_2x2_ground_u8 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| sensitivity | sens_T_60 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_T_95 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_T_130 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_T_180 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_U_6 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_U_8 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_U_10 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| sensitivity | sens_U_12 | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| dynamic_pumping | feedback_loop_atomic | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| dynamic_pumping | feedback_loop_atomic | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| dynamic_pumping | feedback_loop_atomic | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| dynamic_pumping | feedback_loop_atomic | tendance utile mesurée | pas de seuil de succès explicite | formaliser critère PASS/FAIL |
| dt_sweep | dt_0.001 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| dt_sweep | dt_0.005 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| dt_sweep | dt_0.010 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_8x8 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_8x8 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_10x10 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_10x10 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_12x12 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_12x12 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_14x14 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_14x14 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_16x16 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_16x16 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_18x18 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_18x18 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_24x24 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_24x24 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_26x26 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_26x26 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_28x28 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_28x28 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_32x32 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_32x32 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_36x36 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_36x36 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_64x64 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_64x64 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_66x66 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_66x66 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_68x68 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_68x68 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_128x128 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_128x128 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_255x255 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |
| cluster_scale | cluster_255x255 | couverture exploratoire riche | validation incomplète/instable | convertir en gate bloquant |

## Phase 8-10 — Nouvelles choses non intégrées + tests à lancer
### Questions ouvertes non intégrées
| question_id | question | action recommandée |
|---|---|---|
| Q_missing_units | Are physical units explicit and consistent for all observables? | Add units schema and unit-consistency gate |
| Q_solver_crosscheck | Do proxy results match at least one independent non-proxy solver on larger lattice? | Maintain benchmark_comparison_qmc_dmrg.csv and extend lattice coverage |
| Q_dt_real_sweep | Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)? | Schedule 3-run sweep in CI night job |
| Q_phase_criteria | Are phase-transition criteria explicit (order parameter + finite-size scaling)? | Add formal criteria and thresholds |
| Q_production_guardrails | Can V4 NEXT rollback instantly on degraded metrics? | Keep rollout controller and rollback contract active |

### Algorithmes utilisés par simulation (source metadata)
| module | integration_scheme | dt | field_type | utilité |
|---|---|---:|---|---|
| hubbard_hts_core | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| qcd_lattice_proxy | euler_explicit | 0.010000 | gauge_field | intégration temporelle proxy des observables corrélées |
| quantum_field_noneq | euler_explicit | 0.010000 | field_proxy | intégration temporelle proxy des observables corrélées |
| dense_nuclear_proxy | euler_explicit | 0.010000 | mixed_proxy | intégration temporelle proxy des observables corrélées |
| quantum_chemistry_proxy | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| spin_liquid_exotic | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| topological_correlated_materials | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| correlated_fermions_non_hubbard | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| multi_state_excited_chemistry | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |
| bosonic_multimode_systems | euler_explicit | 0.010000 | bosonic_proxy | intégration temporelle proxy des observables corrélées |
| multiscale_nonlinear_field_models | euler_explicit | 0.010000 | field_proxy | intégration temporelle proxy des observables corrélées |
| far_from_equilibrium_kinetic_lattices | euler_explicit | 0.010000 | field_proxy | intégration temporelle proxy des observables corrélées |
| multi_correlated_fermion_boson_networks | euler_explicit | 0.010000 | fermionic_proxy | intégration temporelle proxy des observables corrélées |

### Commandes exactes reproductibles
```bash
git fetch origin --prune
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T205539Z_465
sha256sum -c logs/checksums.sha256
python src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_cycle06_205539_forensic_full_report.py
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```
