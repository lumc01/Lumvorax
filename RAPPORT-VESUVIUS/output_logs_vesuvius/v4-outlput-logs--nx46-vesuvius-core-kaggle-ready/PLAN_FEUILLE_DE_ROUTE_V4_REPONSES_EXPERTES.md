# PLAN FEUILLE DE ROUTE V4 — RÉPONSES EXPERTES (RECONSTRUCTION LOCALE)

> Statut: fichier reconstruit localement à partir de l'historique de session, en l'absence de synchronisation GitHub disponible dans cet environnement.

## 1) Réponses expertes consolidées (issues de l'historique)

- Le pipeline v3/v4 observé n'est pas un CNN end-to-end pur: c'est un pipeline adaptatif data-driven avec allocation dynamique et projection énergétique.
- La sortie de soumission correcte côté Kaggle est liée au format masque binaire `uint8` **0/255** et à un chemin d'output détectable.
- Le plan V4 a été étendu pour couvrir:
  - détection de matériaux (encre/fibre/fond/artefacts),
  - explicabilité du raisonnement,
  - preuve d'apprentissage réel,
  - trajectoire 100% native (sans dépendance à des modèles concurrents externes).

## 2) Architecture cible "100% native" (synthèse)

### Phase 6 — Tête matériaux
- Ajouter une tête multiclasse matériaux (encre / fibre papyrus / fond / artefacts).
- Produire des cartes de probabilité par classe et des métriques de séparation inter-classes.

### Phase 7 — Auto-apprentissage profond natif
- Pré-entraînement auto-supervisé volumique (MVR/CSC/FOP) from-scratch offline.
- Interdictions strictes:
  - aucun checkpoint externe,
  - aucune API distante,
  - aucun fallback opaque non traçable.

### Phase 8 — Validation comparative stricte
- Benchmark AGNN natif vs hybride interne vs baseline CNN.
- Critères:
  - score Kaggle,
  - stabilité run-to-run,
  - coût CPU/mémoire,
  - qualité forensic (preuves d'intégrité).

## 3) Preuves anti-smoke à exiger

- `selfsup_loss_curve.json`
- `material_head_metrics.csv`
- `native_training_manifest.json`
- courbes d'ablation (avec/sans auto-supervision)
- preuves de reproductibilité (seed + variance inter-runs)
- validation de calibration des seuils et robustesse sur layouts différents

## 4) Questions expertes (et réponses attendues)

1. Le système apprend-il réellement ?
   - Réponse attendue: oui, si perte auto-supervisée diminue + amélioration mesurée sur tête compétition.
2. Détecte-t-il autre chose que l'encre ?
   - Réponse attendue: oui, si la tête matériaux sépare encre/fibre/fond/artefacts avec métriques dédiées.
3. Est-il autonome ?
   - Réponse attendue: oui, si entraînement from-scratch offline sans dépendance modèle externe.
4. Pourquoi confiance dans la soumission ?
   - Réponse attendue: zip strict validé + masques 0/255 + traçabilité complète des logs.

## 5) Plan d'exécution sprint (S1 → S4)

- **S1**: backbone auto-supervisé natif + instrumentation forensic.
- **S2**: tête matériaux + métriques multi-classes.
- **S3**: tête compétition + calibration seuils + format soumission Kaggle strict.
- **S4**: benchmark final + dossier preuves + décision go/no-go.

## 6) Rappel critique

Ce document est une reconstruction fidèle des points mentionnés dans l'historique, et doit être remplacé par la version GitHub source dès qu'un accès réseau valide au remote est disponible.
