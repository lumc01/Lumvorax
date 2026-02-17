# PLAN FEUILLE DE ROUTE V6 — RÉPONSES EXPERTES ET INTÉGRATION A→Z

## 0) Préambule de vérité

Les artefacts suivants demandés n'ont pas été trouvés localement dans ce dépôt:
- `RAPPORT-VESUVIUS/output_logs_vesuvius/v4-outlput-logs--nx46-vesuvius-core-kaggle-ready/PLAN_FEUILLE_DE_ROUTE_V4_REPONSES_EXPERTES.md`
- `RAPPORT-VESUVIUS/output_logs_vesuvius/v5-outlput-logs--nx46-vesuvius-core-kaggle-ready`

Ce plan V6 est donc construit à partir:
1. du code V5 actuel,
2. des logs/zip V4 disponibles,
3. du notebook de référence `nx46-vesuvius-challenge-surface-detection.ipynb`.

---

## 1) Ce qui a été réellement appliqué en V5

### 1.1 Corrigé et intégré
- Sortie masque TIFF en `uint8` **0/255** (et non 0/1).
- Publication multi-chemins du `submission.zip` (incluant `nx46_vesuvius/submission.zip`).
- Validation stricte des membres `.tif` attendus dans le zip.
- `finalize_forensics` forcé à 100% avant sauvegarde d'état.
- Télémétrie enrichie (`training_strategy`, alias de soumission, inventaire dataset).

### 1.2 Partiellement intégré
- Stratégie fallback sans labels (quantile probe): présente, mais pas encore calibrée par benchmark LB multi-runs.

### 1.3 Non intégré (à faire en V6)
- Lecture ligne-par-ligne A→Z de toute la racine (`src`, C/C++, Python, `.md`).
- Fusion technologique inter-projets (patterns mémoire, parsers, compression, sécurité, forensic cross-domain).
- Version strictement "sans dépendances ML externes" (décorrélée de numpy/tifffile/imagecodecs si exigé).
- Jeu de tests end-to-end local simulant plusieurs layouts Kaggle + validation pixel-range automatisée.

---

## 2) Exigence V6 cible (produit final)

Objectif: **NX46 V6 autonome**, robuste Kaggle, intégrant les technologies internes du dépôt, avec:
- pipeline 100% offline,
- sortie standardisée Kaggle,
- forensic complet,
- zéro régression des correctifs v3/v4/v5,
- et réduction maximale des dépendances non natives.

---

## 3) Plan V6 complet (phases + livrables)

## Phase A — Audit total racine (A→Z)
- Scanner tous les sous-projets racine et cataloguer:
  - patterns mémoire/allocation,
  - I/O TIFF/ZIP,
  - forensic/logging/merkle,
  - accélérations CPU,
  - utilitaires robustesse.
- Livrable: `RAPPORT-VESUVIUS/AUDIT_TOTAL_RACINE_V6.md`.

## Phase B — Matrice d'intégration technologique
- Construire une matrice: `source_module -> capacité -> candidat intégration V6 -> risque`.
- Prioriser les blocs qui améliorent robustesse sans dépendances ML externes.
- Livrable: `RAPPORT-VESUVIUS/MATRICE_INTEGRATION_TECHNOLOGIQUE_V6.md`.

## Phase C — Refonte noyau V6
- Créer `RAPPORT-VESUVIUS/src_vesuvius/nx46-vesuvius-core-kaggle-ready-v6.py`.
- Conserver compatibilité Kaggle (chemins + format 0/255 + zip strict).
- Injecter les briques internes retenues (allocation/mémoire/forensic).

## Phase D — Validation multi-niveaux
- Tests unitaires: format TIFF, zip members, chemins submit.
- Tests intégration: layouts dataset multiples.
- Tests forensic: cohérence état/logs/merkle/bit capture.
- Livrable: `RAPPORT-VESUVIUS/VALIDATION_V6_COMPLETE.md`.

## Phase E — Comparaison avant/après
- Comparer V4 vs V5 vs V6:
  - structure artefacts,
  - cohérence roadmap,
  - robustesse soumission,
  - stabilité exécution.
- Livrable: `RAPPORT-VESUVIUS/COMPARATIF_V4_V5_V6.md`.

---

## 4) Questions expertes à couvrir dans V6 (et preuves attendues)

1. **Le zip de soumission est-il toujours détectable par Kaggle quel que soit le runner?**
   - Preuve: existence contrôlée sur chemins alias + checksum.
2. **Les masques sont-ils réellement en 0/255 sur tous les cas?**
   - Preuve: histogrammes min/max/unique-values.
3. **La pipeline reste-t-elle conforme si aucun label train n'est présent?**
   - Preuve: fallback strategy + logs d'activation.
4. **A-t-on supprimé des fonctions héritées critiques?**
   - Preuve: matrice de traçabilité fonctionnelle v3/v4/v5/v6.
5. **Le système peut-il fonctionner sans stack ML externe avancée?**
   - Preuve: mode dégradé documenté + tests passants.

---

## 5) Statut d'avancement V6 (initial)

- Phase A: 0%
- Phase B: 0%
- Phase C: 0%
- Phase D: 0%
- Phase E: 0%

Ce document constitue le démarrage officiel V6.
