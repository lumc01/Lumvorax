# Analyse forensique ligne par ligne — code source vs addendum précédent

## 0) Objet de cette contre-analyse
Cette analyse vérifie **ligne par ligne les mécanismes du code** qui produisent les CSV/logs du run `research_20260309T205848Z_3489`, afin d’expliquer pourquoi la traçabilité Lumvorax fonctionne mais la validation scientifique échoue.

## 1) Portée d'inspection réellement effectuée
- Fichiers source inspectés (hors backups): **48** (`.py/.c/.h`).
- Volume de code relu: **9955 lignes**.
- Scripts de post-run inspectés: scoring, tests critiques, observables avancés, modules indépendants, pipeline d'orchestration.
- Binaire/simulateur inspecté: `hubbard_hts_research_cycle_advanced_parallel.c` (génération des benchmarks, tests, CSV).

## 2) Diagnostic principal (réponse directe à « pourquoi rien ne fonctionne sauf la traçabilité ? »)
1. **La traçabilité est robuste** (logs événementiels HFBL/Lumvorax persistants, checksums, historique de commandes).
2. **La validation scientifique est faible** car plusieurs métriques clés sont construites avec des seuils ou des comparaisons structurellement défavorables (voire circulaires).
3. **Les “modules indépendants” ne sont pas des solveurs externes de vérité terrain**: ils retraitent la même série baseline, donc ils ne peuvent pas invalider un biais de génération amont.

## 3) Écarts code ↔ interprétation précédente (preuves techniques)
| Sujet | Constat dans le code | Impact réel |
|---|---|---|
| T5 cross-check QMC/DMRG | Critère `PASS` exige **100%** des lignes dans l'error bar (`sum(v>=1)==len`) | Une seule ligne hors barre suffit à FAIL total; critère très dur. |
| T8 fenêtre critique | Fenêtre codée fixe `600<=step<=800` sur minimum énergie | Peut produire OBSERVED global même si dynamique valide mais décalée. |
| Progression solution % | Score ajoute `+20 metadata_present` **sans condition**, puis pénalise globalement | Tendance à score uniforme, peu discriminant entre modules. |
| Alternative solver T12 | Campagne alternative dérive du CSV benchmark existant, pas d'exécution solveur externe ici | Risque de circularité de validation. |
| Modules “indépendants” | QMC/DMRG/ARPES/STM appliqués à `baseline_reanalysis_metrics.csv` | Validation technique de pipeline, pas preuve de vérité physique externe. |
| Métadonnées | Grande table metadata codée en dur dans `post_run_metadata_capture.py` | Traceabilité de schéma oui; authenticité physique paramétrique limitée. |

## 4) Données chiffrées consolidées (run analysé)
- Tests critiques T1–T12: PASS=7, OBSERVED=2, FAIL=3.
- QMC/DMRG within error bars: 0/15 = 0.00%.
- Benchmarks externes within error bars: 0/16 = 0.00%.
- Énergie (QMC): référence [652800.0, 2819200.0] vs modèle [-0.231, -0.081] (échelles incompatibles).
- Énergie (externes): référence [11200.0, 56760.0] vs modèle [-0.292, -0.096] (échelles incompatibles).

## 5) Lecture pédagogique: pourquoi Lumvorax “voit tout” mais ne “corrige pas tout”
- Lumvorax trace les événements (`simulation_start/step/end`, hash d'état, timestamps) et sécurise la persistance des logs.
- En revanche, Lumvorax ne remplace pas un modèle physique correctement calibré: il prouve **ce qui a été exécuté**, pas que la physique est correcte.
- Conclusion: la chaîne d'audit est bonne, la chaîne de validité scientifique reste à renforcer.

## 6) Avancement module par module (forensic + progrès réel)
| Module | Progression % | Reste % | Forensic PASS | Forensic OBSERVED | Forensic FAIL | Diagnostic blocant |
|---|---:|---:|---:|---:|---:|---|
| bosonic_multimode_systems | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| correlated_fermions_non_hubbard | 42.00 | 58.00 | 50.00 | 50.00 | 0.00 | benchmark/échelle + corrélation T7 |
| dense_nuclear_proxy | 42.00 | 58.00 | 50.00 | 50.00 | 0.00 | benchmark/échelle + corrélation T7 |
| far_from_equilibrium_kinetic_lattices | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| hubbard_hts_core | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| multi_correlated_fermion_boson_networks | 42.00 | 58.00 | 0.00 | 100.00 | 0.00 | benchmark/échelle + corrélation T7 |
| multi_state_excited_chemistry | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| multiscale_nonlinear_field_models | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| qcd_lattice_proxy | 42.00 | 58.00 | 0.00 | 100.00 | 0.00 | benchmark/échelle + corrélation T7 |
| quantum_chemistry_proxy | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| quantum_field_noneq | 42.00 | 58.00 | 0.00 | 100.00 | 0.00 | benchmark/échelle + corrélation T7 |
| spin_liquid_exotic | 42.00 | 58.00 | 25.00 | 75.00 | 0.00 | benchmark/échelle + corrélation T7 |
| topological_correlated_materials | 42.00 | 58.00 | 50.00 | 50.00 | 0.00 | benchmark/échelle + corrélation T7 |

## 7) Conclusions forensiques finales
### Introduction
Le run est traçable, reproductible au niveau artefacts, et riche en instrumentation.

### Développement
- De plus, plusieurs tests sont de présence/structure et non de calibration physique absolue.
- Cependant, les comparaisons benchmark énergie montrent une incompatibilité d'échelle majeure (ordre 10^4–10^6 vs ordre 10^-1).
- En outre, la progression 42% uniforme est cohérente avec le calcul de score et ses pénalités globales, pas avec une maturité scientifique individualisée.

### Conclusion
Donc, **ce qui fonctionne aujourd'hui: la traçabilité Lumvorax**. **Ce qui bloque: la calibration physique, le mapping d'unités/observables, et la non-indépendance forte de certains contrôles.**

## 8) Plan de déblocage concret (sans exécution ici)
1. Normaliser les observables benchmark (`energy/site`, conventions de signe, unités explicites) avant comparaison.
2. Remplacer le critère T5 “all rows” par score statistique gradué (ex: >=80% + CI).
3. Décorréler T12 de la table benchmark existante: exiger un run solveur externe réellement séparé.
4. Rendre `solution_progress_percent` conditionnel par module (retirer bonus metadata inconditionnel).
5. Ajouter une gate “unit-consistency” bloquante avant publication scientifique.
