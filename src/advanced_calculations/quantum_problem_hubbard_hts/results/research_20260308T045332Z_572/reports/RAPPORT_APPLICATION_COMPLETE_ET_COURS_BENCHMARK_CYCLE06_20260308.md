# Rapport d’intégration complète A→Z + cours pédagogique benchmark (Cycle06)

- Horodatage UTC: 2026-03-08T19:34:51Z
- Portée: application immédiate des solutions de `RAPPORT_FORENSIQUE_EXHAUSTIF_CYCLE06_20260308.md` sous forme de plan exécutable, traçable et détaillé.
- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.

## 0) Actions d’intégration appliquées immédiatement
1. Synchronisation dépôt distant et collecte des branches distantes.
2. Vérification d’intégrité SHA du run Cycle06.
3. Extraction exhaustive des échecs benchmark (QMC/DMRG + modules externes).
4. Génération d’un plan d’action détaillé par blocage, avec test cible et critère d’acceptation.
5. Génération de ce cours pédagogique complet + fichiers de traçabilité JSON/SHA.

## 1) Cours débutant: explication claire des termes techniques
### Introduction (thèse + contexte)
Dans ce pipeline, chaque module simule un système corrélé. Les métriques servent à vérifier **deux choses**: la cohérence physique et la robustesse numérique.
### Développement (argumentation)
- **problem**: Nom du module simulé. De plus, lecture: Chaque groupe de lignes correspond à un module. Cependant, risque: Comparer modules sans normalisation peut induire une fausse interprétation.
- **step**: Indice d'avancement numérique. De plus, lecture: +100 = incrément de simulation. Cependant, risque: step n'est pas forcément un temps physique réel.
- **energy**: Observable énergétique interne. De plus, lecture: Peut être négative/positive selon le modèle. Cependant, risque: Sans unités/normalisation, pas de conclusion physique forte.
- **pairing**: Observable de corrélation. De plus, lecture: Ici souvent cumulative. Cependant, risque: Ne pas interpréter comme ordre supraconducteur sans définition formelle.
- **sign_ratio**: Indicateur statistique de signe. De plus, lecture: Proche de 0 peut indiquer compensation des signes. Cependant, risque: Ne suffit pas seul pour conclure sur le sign-problem.
- **cpu_percent**: Charge CPU système. De plus, lecture: Valeur stable = performance stable. Cependant, risque: N'apporte pas de preuve de validité physique.
- **mem_percent**: Usage RAM système. De plus, lecture: Valeur stable = pas de fuite évidente. Cependant, risque: N'évalue pas la justesse scientifique.
- **elapsed_ns**: Temps écoulé en nanosecondes. De plus, lecture: Croissance ~linéaire = coût local stable. Cependant, risque: Ne prouve pas la complexité asymptotique globale.
### Conclusion (solution + clôture)
Donc, une valeur isolée n’est jamais suffisante: il faut croiser observable, stabilité numérique et benchmark externe.

## 2) Algorithmes utilisés et rôle exact pour chaque simulation
| Module | Schéma numérique | Rôle scientifique | Risque principal |
|---|---|---|---|
| hubbard_hts_core | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=8.000000, doping=0.200000 | instabilité dt |
| qcd_lattice_proxy | euler_explicit (dt=0.010000) | gauge_field, U/t=12.857143, doping=0.100000 | instabilité dt |
| quantum_field_noneq | euler_explicit (dt=0.010000) | field_proxy, U/t=5.384615, doping=0.050000 | instabilité dt |
| dense_nuclear_proxy | euler_explicit (dt=0.010000) | mixed_proxy, U/t=13.750000, doping=0.300000 | instabilité dt |
| quantum_chemistry_proxy | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=4.062500, doping=0.400000 | instabilité dt |
| spin_liquid_exotic | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=11.666667, doping=0.120000 | instabilité dt |
| topological_correlated_materials | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=7.090909, doping=0.150000 | instabilité dt |
| correlated_fermions_non_hubbard | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=7.166667, doping=0.180000 | instabilité dt |
| multi_state_excited_chemistry | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=4.533333, doping=0.220000 | instabilité dt |
| bosonic_multimode_systems | euler_explicit (dt=0.010000) | bosonic_proxy, U/t=8.666667, doping=0.060000 | instabilité dt |
| multiscale_nonlinear_field_models | euler_explicit (dt=0.010000) | field_proxy, U/t=6.571429, doping=0.100000 | instabilité dt |
| far_from_equilibrium_kinetic_lattices | euler_explicit (dt=0.010000) | field_proxy, U/t=8.000000, doping=0.090000 | instabilité dt |
| multi_correlated_fermion_boson_networks | euler_explicit (dt=0.010000) | fermionic_proxy, U/t=7.047619, doping=0.140000 | instabilité dt |

## 3) Résultats benchmark qui ne passent pas — explication individuelle (unité par unité)
### 3A. Échecs benchmark QMC/DMRG
- Total points QMC/DMRG: 15 ; échecs individuels: 7.
#### QMC_FAIL_01 — hubbard_hts_core / energy à T=95.000000, U=4.000000
**Introduction (thèse + contexte)**
Le modèle produit `665599.8539853615` alors que la référence QMC/DMRG est `652800.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `12799.8539853615` pour une barre d’erreur `12000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `1.96%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_02 — hubbard_hts_core / energy à T=95.000000, U=6.000000
**Introduction (thèse + contexte)**
Le modèle produit `1056071.8083606362` alors que la référence QMC/DMRG est `1013875.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `42196.8083606362` pour une barre d’erreur `9000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `4.16%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_03 — hubbard_hts_core / energy à T=95.000000, U=8.000000
**Introduction (thèse + contexte)**
Le modèle produit `1408808.6816832828` alors que la référence QMC/DMRG est `1374940.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `33868.6816832828` pour une barre d’erreur `9000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `2.46%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_04 — hubbard_hts_core / energy à T=95.000000, U=10.000000
**Introduction (thèse + contexte)**
Le modèle produit `1678572.9204088624` alors que la référence QMC/DMRG est `1736003.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `57430.0795911376` pour une barre d’erreur `9000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `3.31%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_05 — hubbard_hts_core / energy à T=95.000000, U=12.000000
**Introduction (thèse + contexte)**
Le modèle produit `1986945.4706086284` alors que la référence QMC/DMRG est `2097068.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `110122.5293913716` pour une barre d’erreur `9000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `5.25%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_06 — hubbard_hts_core / energy à T=95.000000, U=14.000000
**Introduction (thèse + contexte)**
Le modèle produit `2513276.6561204088` alors que la référence QMC/DMRG est `2458130.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `55146.6561204088` pour une barre d’erreur `12000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `2.24%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

#### QMC_FAIL_07 — hubbard_hts_core / energy à T=95.000000, U=16.000000
**Introduction (thèse + contexte)**
Le modèle produit `2765890.3892134754` alors que la référence QMC/DMRG est `2819200.0000000000`.
**Développement (argumentation)**
De plus, l’erreur absolue vaut `53309.6107865246` pour une barre d’erreur `14000.0000000000`. Cependant, `within_error_bar=0`, donc ce point est hors tolérance.
Également, l’erreur relative est `1.89%`, ce qui indique un écart non négligeable pour une validation quantitative stricte.
**Conclusion (solution + clôture)**
Donc, il faut recalibrer les unités énergétiques et imposer une normalisation par site avant toute conclusion physique finale.

### 3B. Échecs benchmark modules externes
- Total points externes: 16 ; échecs individuels: 16.
#### EXT_FAIL_01 — spin_liquid_exotic / pairing à T=55.000000, U=10.500000
**Introduction (thèse + contexte)**
La sortie modèle `0.8262403823` diverge de la référence externe `0.6840000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.1422403823` dépasse la barre `0.0900000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `20.80%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_02 — spin_liquid_exotic / energy à T=55.000000, U=10.500000
**Introduction (thèse + contexte)**
La sortie modèle `1789210.6792104105` diverge de la référence externe `56120.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `1733090.6792104105` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `3088.19%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_03 — topological_correlated_materials / pairing à T=70.000000, U=7.800000
**Introduction (thèse + contexte)**
La sortie modèle `0.7736075411` diverge de la référence externe `0.6360000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.1376075411` dépasse la barre `0.0900000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `21.64%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_04 — topological_correlated_materials / energy à T=70.000000, U=7.800000
**Introduction (thèse + contexte)**
La sortie modèle `1320912.8372668442` diverge de la référence externe `56760.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `1264152.8372668442` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `2227.19%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_05 — correlated_fermions_non_hubbard / pairing à T=85.000000, U=8.600000
**Introduction (thèse + contexte)**
La sortie modèle `0.7399247335` diverge de la référence externe `0.5650000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.1749247335` dépasse la barre `0.0900000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `30.96%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_06 — correlated_fermions_non_hubbard / energy à T=85.000000, U=8.600000
**Introduction (thèse + contexte)**
La sortie modèle `1074881.8518466770` diverge de la référence externe `42900.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `1031981.8518466770` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `2405.55%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_07 — multi_state_excited_chemistry / pairing à T=48.000000, U=6.800000
**Introduction (thèse + contexte)**
La sortie modèle `0.8603385169` diverge de la référence externe `0.7490000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.1113385169` dépasse la barre `0.0900000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `14.86%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_08 — multi_state_excited_chemistry / energy à T=48.000000, U=6.800000
**Introduction (thèse + contexte)**
La sortie modèle `680620.3616473363` diverge de la référence externe `36920.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `643700.3616473363` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `1743.50%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_09 — bosonic_multimode_systems / pairing à T=110.000000, U=5.200000
**Introduction (thèse + contexte)**
La sortie modèle `0.7166394136` diverge de la référence externe `0.5210000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.1956394136` dépasse la barre `0.1000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `37.55%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_10 — bosonic_multimode_systems / energy à T=110.000000, U=5.200000
**Introduction (thèse + contexte)**
La sortie modèle `497023.1416568772` diverge de la référence externe `11200.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `485823.1416568772` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `4337.71%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_11 — multiscale_nonlinear_field_models / pairing à T=125.000000, U=9.200000
**Introduction (thèse + contexte)**
La sortie modèle `0.6677829491` diverge de la référence externe `0.4220000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.2457829491` dépasse la barre `0.1000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `58.24%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_12 — multiscale_nonlinear_field_models / energy à T=125.000000, U=9.200000
**Introduction (thèse + contexte)**
La sortie modèle `997210.7847945302` diverge de la référence externe `30240.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `966970.7847945302` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `3197.65%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_13 — far_from_equilibrium_kinetic_lattices / pairing à T=150.000000, U=8.000000
**Introduction (thèse + contexte)**
La sortie modèle `0.5969764049` diverge de la référence externe `0.2860000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.3109764049` dépasse la barre `0.1100000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `108.73%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_14 — far_from_equilibrium_kinetic_lattices / energy à T=150.000000, U=8.000000
**Introduction (thèse + contexte)**
La sortie modèle `1026562.3412401110` diverge de la référence externe `35250.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `991312.3412401110` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `2812.23%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_15 — multi_correlated_fermion_boson_networks / pairing à T=100.000000, U=7.400000
**Introduction (thèse + contexte)**
La sortie modèle `0.6962215230` diverge de la référence externe `0.4870000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `0.2092215230` dépasse la barre `0.1000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `42.96%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

#### EXT_FAIL_16 — multi_correlated_fermion_boson_networks / energy à T=100.000000, U=7.400000
**Introduction (thèse + contexte)**
La sortie modèle `939846.5570258966` diverge de la référence externe `56130.0000000000`.
**Développement (argumentation)**
En outre, l’erreur absolue `883716.5570258966` dépasse la barre `9500.0000000000`; ainsi `within_error_bar=0`.
De même, l’erreur relative atteint `1574.41%`, suggérant une incohérence d’échelle ou de mapping d’observable.
**Conclusion (solution + clôture)**
Ainsi, la solution prioritaire est un pont de calibration inter-modèles (mapping observable + unité + facteur de volume).

## 4) Pour chaque OBSERVED: points forts, critiques, et interprétation
- Nombre de tests OBSERVED: 49.
| test_family | test_id | point fort | critique | action recommandée |
|---|---|---|---|---|
| exact_solver | hubbard_2x2_ground_u4 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| exact_solver | hubbard_2x2_ground_u8 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| sensitivity | sens_T_60 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_T_95 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_T_130 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_T_180 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_U_6 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_U_8 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_U_10 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| sensitivity | sens_U_12 | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| dynamic_pumping | feedback_loop_atomic | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| dynamic_pumping | feedback_loop_atomic | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| dynamic_pumping | feedback_loop_atomic | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| dynamic_pumping | feedback_loop_atomic | mesure disponible et cohérente en tendance | pas de seuil PASS/FAIL formalisé | ajouter critère de validation numérique |
| dt_sweep | dt_0.001 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| dt_sweep | dt_0.005 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| dt_sweep | dt_0.010 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_8x8 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_8x8 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_10x10 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_10x10 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_12x12 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_12x12 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_14x14 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_14x14 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_16x16 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_16x16 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_18x18 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_18x18 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_24x24 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_24x24 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_26x26 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_26x26 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_28x28 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_28x28 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_32x32 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_32x32 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_36x36 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_36x36 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_64x64 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_64x64 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_66x66 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_66x66 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_68x68 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_68x68 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_128x128 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_128x128 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_255x255 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |
| cluster_scale | cluster_255x255 | couverture exploratoire utile | validation incomplète (ou fail associé) | convertir en test bloquant avec seuil |

## 5) Questions d’experts, réponses révélées par les résultats, et manques
- Questions complètes: 11; partielles: 8.
- Questions déjà répondues: reproductibilité inter-run, tendance pairing(T), tendance energy(U), présence de contrôles dynamiques.
- Questions encore ouvertes: cohérence d’unités, cross-check non-proxy large lattice, stabilité dt réelle, critères de transition de phase, rollback production.

## 6) Découvertes potentielles, anomalies et statut scientifique
- Tendance robuste observée: pairing décroît avec T (test physics PASS).
- Tendance robuste observée: energy croît avec U (test physics PASS).
- Anomalie majeure: benchmark externe 0% within_error_bar (écart structurel).
- Anomalie numérique: dt_convergence FAIL et stabilité Von Neumann >1 (dans les logs de stabilité).
- Interprétation prudente: ces anomalies sont plus compatibles avec un artefact numérique/calibration qu’avec une découverte physique confirmée.

## 7) Comparaison aux simulations de référence disponibles
- Références utilisées dans les artefacts: `qmc_dmrg_reference` et `external_modules`.
- Constat: la composante pairing du coeur Hubbard est globalement proche des références (dans barres), cependant les composantes énergie et modules externes sont hors tolérance.
- Donc, la cohérence qualitative est présente, néanmoins la validité quantitative globale n’est pas atteinte.

## 8) Nouveaux tests à exécuter (liste exhaustive et reproductible)
| ID | Objectif | Méthodologie | Variables | Critère de validation |
|---|---|---|---|---|
| T1 | Calibration unités énergie | normaliser énergie/site puis refit | energy, abs_error | >=90% points within_error_bar |
| T2 | Convergence temporelle stricte | dt/2, dt, 2dt sur 3 seeds | pairing, energy | variation <1% |
| T3 | Cross-check solveur indépendant | comparer proxy vs solveur non-proxy | pairing, energy | rel_error <10% |
| T4 | Stabilite longue durée | horizon >10k pas | drift énergie, spectral radius | drift <5% et rayon <=1 |
| T5 | Transition de phase | finite-size scaling + ordre paramètre | exponents, chi2 | RMSE fit <0.05 |

## 9) Commandes exactes
```bash
git fetch origin --prune
cd src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260308T045332Z_572
sha256sum -c logs/checksums.sha256
python src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_cycle06_benchmark_course_report.py
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```
