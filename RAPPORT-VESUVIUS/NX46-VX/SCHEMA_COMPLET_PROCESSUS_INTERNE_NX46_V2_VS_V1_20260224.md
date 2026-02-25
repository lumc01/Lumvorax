# SCHÉMA COMPLET — Processus interne NX46-V2 vs ancienne version V1
Date: 2026-02-24

Objectif de ce document: expliquer **de manière hiérarchique** (processus, sous-processus, étape, phase, sous-phase, sous-étape, point, sous-point) le fonctionnement interne de la nouvelle version V2 comparée à l’ancienne V1.

---

## 0) Vue d’ensemble comparative (V1 vs V2)

### 0.1 Architecture générale commune
- Les deux notebooks gardent la même base unifiée de blocs techniques:
  - bloc source V61.5,
  - bloc source V144.2,
  - bloc source V7.7,
  - bloc source V7.6.

### 0.2 Différence stratégique
- V1: forte préparation “teachers” (registre + garde), mais sans manifeste dépendances complet ni protocole V2 de synchronisation.
- V2: reprend V1 **et** ajoute:
  - protocole de copie/rollback en tête,
  - versioning explicite `NOTEBOOK_VERSION = 'V2'`,
  - manifeste exact des dépendances,
  - plan d’apprentissage V2 en 4 phases,
  - synchronisation du versioning des logs forensic (`NX46-VX V2`, `log_version_tag`).

### 0.3 Résumé opérationnel
- V1 = “prépare et verrouille”.
- V2 = “prépare, verrouille, vérifie dépendances, formalise phases, synchronise logs/version”.

---

## 1) Processus P0 — Gouvernance version et rollback

### 1.1 Sous-processus P0.1 — Version notebook
#### Étape 1.1.1
- V1: en-tête notebook générique NX46 VX.
- V2: en-tête explicitement marqué `(V2)`.

#### Étape 1.1.2
- V2 introduit `NOTEBOOK_VERSION = 'V2'` pour tracer la version active côté runtime.

### 1.2 Sous-processus P0.2 — Politique de sauvegarde
#### Étape 1.2.1
- V2 contient un protocole “copies préalables” (backup immutable V1 + backup V2 pré-update).

#### Étape 1.2.2
- Effet métier: rétrogradation contrôlée possible sans écraser l’historique.

---

## 2) Processus P1 — Entrées enseignants (9 modèles concurrents)

### 2.1 Sous-processus P1.1 — Référentiel enseignants
#### Étape 2.1.1
- V1 et V2: `TEACHER_MODELS_REGISTRY` à 9 entrées.

#### Étape 2.1.2
- Chaque teacher contient:
  - `teacher_id`,
  - source, 
  - référence de poids (`weight_ref`).

### 2.2 Sous-processus P1.2 — Vérification stricte du minimum requis
#### Étape 2.2.1
- V1 et V2: `assert_9_teacher_models_ready(...)` bloque si `<9` résolus.

#### Étape 2.2.2
- Sous-point critique:
  - c’est un verrou de conformité,
  - pas encore la preuve d’un transfert effectif déjà exécuté.

### 2.3 Sous-processus P1.3 — Références concurrent exactes
#### Étape 2.3.1
- V1 et V2 maintiennent `COMPETITOR_MODELS_FOUND_EXACT` pour tracer les références concurrentes détectées.

---

## 3) Processus P2 — Dépendances et contrat d’environnement

### 3.1 Sous-processus P2.1 — Manifest exact des fichiers dépendants
#### Étape 3.1.1
- V2 ajoute `REQUIRED_DEPENDENCY_FILENAMES` (noms exacts) pour verrouiller l’environnement.

#### Étape 3.1.2
- V1 ne portait pas ce manifeste complet en dur.

### 3.2 Sous-processus P2.2 — Découverte et validation runtime
#### Étape 3.2.1
- V2 ajoute:
  - `discover_paths(...)`
  - `validate_dependency_manifest_exact_names(...)`

#### Étape 3.2.2
- Sous-point pratique:
  - scan multi-racines,
  - liste des dépendances manquantes,
  - visibilité immédiate avant exécution lourde.

### 3.3 Sous-processus P2.3 — Artefacts teacher TIFF/LUM
#### Étape 3.3.1
- V2 référence explicitement:
  - `competitor_teacher_1407735.tif`
  - `competitor_teacher_1407735.lum`

#### Étape 3.3.2
- Sous-point métier:
  - l’intention d’utiliser le teacher TIFF est formalisée dans le plan,
  - l’exécution réelle d’apprentissage reste dépendante du run effectif.

---

## 4) Processus P3 — Plan d’apprentissage (orchestration en phases)

### 4.1 Sous-processus P3.1 — Formalisation explicite V2
#### Étape 4.1.1
- V2 introduit `NX46_V2_TRAINING_PLAN` avec 4 phases.

#### Étape 4.1.2
- Structure:
  1. distillation enseignants,
  2. entraînement supervisé train,
  3. ultra-fine-tuning guidé test vs TIFF concurrent,
  4. inférence principale + soumission.

### 4.2 Sous-processus P3.2 — Différence avec V1
#### Étape 4.2.1
- V1: logique teachers présente, mais plan V2 moins explicité/structuré côté contrat global.

#### Étape 4.2.2
- V2: séquencement pédagogique explicite pour éviter confusion d’ordre.

---

## 5) Processus P4 — Noyau pipeline de traitement (communs V1/V2)

### 5.1 Sous-processus P4.1 — Ingestion et découverte données
#### Étape 5.1.1
- découverte input,
- inventaire train/test,
- validation de layout.

### 5.2 Sous-processus P4.2 — Prétraitement & features
#### Étape 5.2.1
- normalisations / extractions multi-features,
- gestion slices / batching.

### 5.3 Sous-processus P4.3 — Apprentissage et calibration
#### Étape 5.3.1
- entraînement supervisé,
- calibrage ratio/seuils,
- simulation F1 vs ratio.

### 5.4 Sous-processus P4.4 — Post-process et export
#### Étape 5.4.1
- fusion adaptative par slices,
- topologie/hysteresis 3D,
- écriture TIFF + ZIP submission.

### 5.5 Sous-processus P4.5 — Conclusion comparative
#### Étape 5.5.1
- Ces blocs techniques restent globalement communs.
- La différence V2 porte surtout sur la **gouvernance** (versioning, dépendances, plan explicite, logs synchronisés).

---

## 6) Processus P5 — Observabilité et forensic (changement majeur V2)

### 6.1 Sous-processus P5.1 — Version runtime dans les logs
#### Étape 6.1.1
- Ancien marquage legacy (V144.2/V132) retiré de la version active.

#### Étape 6.1.2
- V2: `self.version = 'NX46-VX V2'`.

### 6.2 Sous-processus P5.2 — Nommage cohérent des artefacts logs
#### Étape 6.2.1
- V2 ajoute `self.log_version_tag = 'nx46vx_v2'`.

#### Étape 6.2.2
- Les fichiers forensic/runtime sont générés via ce tag:
  - roadmap,
  - execution logs,
  - memory tracker,
  - execution metadata,
  - merkle jsonl,
  - forensic report.

### 6.3 Sous-processus P5.3 — Builder forensic synchronisé
#### Étape 6.3.1
- Passage à `_build_v2_forensic_report`.

#### Étape 6.3.2
- Flag export aligné:
  - `export_forensic_v2_report`,
  - variable `NX46VX_V2_EXPORT_FORENSIC_REPORT`,
  - fallback legacy maintenu pour compatibilité.

### 6.4 Sous-processus P5.4 — Valeur pratique
#### Étape 6.4.1
- Réduction du risque “code en V2 mais logs en ancienne version”.

#### Étape 6.4.2
- Meilleure traçabilité des audits et comparaisons inter-runs.

---

## 7) Processus P6 — Schéma décisionnel GO / NO-GO (lecture métier)

### 7.1 Sous-processus P6.1 — GO technique
#### Étape 7.1.1
- teachers déclarés,
- dépendances présentes,
- pipeline exécutable,
- logs versionnés cohérents.

### 7.2 Sous-processus P6.2 — GO performance
#### Étape 7.2.1
- vérifier proximité vs TIFF concurrent (IoU/Dice/XOR).

#### Étape 7.2.2
- si proximité insuffisante: pré-exécutions seulement, pas soumission officielle.

---

## 8) Schéma ultra-hiérarchique “point / sous-point” (compact)

1. Versioning global
   1.1 Version notebook
      1.1.1 V1 générique
      1.1.2 V2 explicitée
   1.2 Rollback
      1.2.1 backup immutable
      1.2.2 backup pré-update
2. Teachers
   2.1 Registry x9
      2.1.1 IDs + refs
      2.1.2 source tracking
   2.2 Guard strict
      2.2.1 assert `<9` => blocage
      2.2.2 conformité ≠ entraînement déjà fait
3. Dépendances
   3.1 Manifest exact
      3.1.1 noms figés
      3.1.2 scan multi-racines
   3.2 Teacher TIFF/LUM
      3.2.1 présence référencée
      3.2.2 usage réel conditionné par run
4. Phases apprentissage
   4.1 Distillation
   4.2 Train
   4.3 Ultra-finetune test guidé
   4.4 Inference finale
5. Forensic
   5.1 version runtime alignée
   5.2 tag logs unique
   5.3 artefacts homogènes
6. Décision
   6.1 GO technique
   6.2 GO score/proximité

---

## 9) Conclusion pédagogique
- **Ancienne version (V1):** excellente base de contrôle teachers, mais encore orientée préparation/verrou.
- **Nouvelle version (V2):** transforme cette base en architecture plus gouvernée et auditable, avec dépendances explicites, phases formalisées, et logs synchronisés sur la version active.
- Résultat: on comprend mieux “qui fait quoi, quand, et avec quelle trace”.
