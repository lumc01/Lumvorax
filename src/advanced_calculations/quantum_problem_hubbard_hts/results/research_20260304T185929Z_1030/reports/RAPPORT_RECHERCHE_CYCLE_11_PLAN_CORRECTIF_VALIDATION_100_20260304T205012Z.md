# Rapport technique itératif — cycle 11 (plan correctif pour viser une validation robuste et prévenir les FAIL futurs)

**Nom exact du rapport :** `RAPPORT_RECHERCHE_CYCLE_11_PLAN_CORRECTIF_VALIDATION_100_20260304T205012Z.md`

## Réponse directe à ta question
Tu demandes: *quoi intégrer de nouveau pour corriger tous les FAIL (1.B,1.C,1.D,2,3,4) et anticiper les FAIL futurs ?*

Réponse d’expert, honnête: **on ne peut pas garantir 100% de vérité physique absolue**, mais on peut atteindre un **niveau de validation scientifique très élevé et falsifiable** avec des garde-fous stricts.

## 1) Ce qu’il faut intégrer immédiatement (P0)
1. **Schéma de métadonnées physiques obligatoire** dans chaque run: `U/t`, taille lattice, BC, pas de temps, Hamiltonien, seed, version algorithme.
2. **Writer transactionnel** (tmp->fsync->rename) + checksum/footer pour éliminer les lignes tronquées.
3. **Validation de complétude** (manifest des modules + steps attendus) avant génération du rapport.
4. **Observables normalisées**: `pairing_per_site`, susceptibilité, corrélateur connecté + unités explicites.
5. **Statistiques robustes**: CI95, bootstrap, autocorrélation, ESS pour éviter les conclusions fragiles.
6. **Gates automatiques**: pas de claim physique si prérequis non satisfaits.

## 2) Corrections précises par section demandée
### 1.B (pairing)
- **Fail actuel**: pairing brut cumulatif, non interprétable comme ordre supraconducteur.
- **Correction**: définir mathématiquement l’observable (ordre paramètre, corrélateur), normaliser par site/volume, publier formule et unités dans le log.
- **Critère PASS**: cohérence avec cas analytique jouet + stabilité de la valeur normalisée au scaling de taille.

### 1.C (sign_ratio)
- **Fail actuel**: sign_ratio observé sans cadre statistique suffisant.
- **Correction**: ajouter ESS, iat, bootstrap CI, courbe sign_vs_beta_L.
- **Critère PASS**: diagnostic statistique complet publiable; pas de conclusion sans CI.

### 1.D (temps/complexité)
- **Fail actuel**: inférence complexité basée sur un profil local.
- **Correction**: sweep de tailles (N) et fit de l’exposant de scaling.
- **Critère PASS**: exponent stable + R² élevé + reproductibilité multi-run.

### 2 (validé technique vs physique)
- **Fail actuel**: claims physiques sans tous les paramètres.
- **Correction**: gate “Physics Claim Ready” bloquant toute conclusion si métadonnées/benchmarks absents.
- **Critère PASS**: tous les champs physiques et validations croisées présents.

### 3 (diagnostic probable)
- **Fail actuel**: cause d’instabilité non isolée.
- **Correction**: protocole d’ablation factoriel (dt, intégrateur, forcing, normalisation).
- **Critère PASS**: facteurs dominants quantifiés (sensibilité/ANOVA).

### 4 (message clair engineering vs science)
- **Fail actuel**: confusion possible entre robustesse logicielle et validation physique.
- **Correction**: double badge dans chaque rapport: `Tech Gate` et `Physics Gate`.
- **Critère PASS**: séparation explicite et auditée des deux statuts.

## 3) Anticipation des FAIL futurs (prévention active)
- CI bloquante sur: intégrité CSV, complétude modules/steps, métadonnées obligatoires, stats minimales, checksums, provenance.
- Nightly stress tests (100 runs) pour détecter intermittences d’écriture.
- Alertes automatiques si dérive des métriques ou augmentation du taux d’échec.
- Régression tests sur observables normalisées et cas analytiques de référence.

## 4) Ce que je sais vs ce que je ne sais pas (transparence)
- **Je sais**: les FAIL actuels sont traitables par ingénierie de données + méthodologie statistique + gouvernance des claims.
- **Je ne sais pas encore**: si le modèle produira une preuve physique forte après corrections; cela dépendra des résultats des nouveaux tests et benchmarks.
- **Donc**: je propose un chemin de validation robuste, pas une promesse dogmatique de “100% vrai”.

## 5) Artefacts techniques créés pour exécution immédiate
- Plan d’intégration: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260304T185929Z_1030/tests/cycle11_integration_plan_20260304T205012Z.csv`
- Matrice fail->fix->prevention: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260304T185929Z_1030/tests/cycle11_fail_to_fix_matrix_20260304T205012Z.csv`
- Provenance + checksums ajoutés pour audit externe.
