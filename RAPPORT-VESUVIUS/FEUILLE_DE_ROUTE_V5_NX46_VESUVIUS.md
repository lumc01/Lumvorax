# Feuille de route V5 — NX46 Vesuvius

## 1) Ce qui a été compris depuis l'historique V4

- La version v4 a livré des artefacts Kaggle valides (`submission.zip`, logs HFBL, validation des membres `.tif`) mais avec une incohérence d'état: `roadmap_percent.finalize_forensics` restait à `60.0` dans `state.json` malgré exécution complète.
- La chaîne de compatibilité attendue est: logs forensic + `nx-46-vesuvius-core.log` + `nx46-vesuvius-core-kaggle-ready.log` + `RkF4XakI.txt` + `UJxLRsEE.txt`.
- Le dataset détecté sur Kaggle est `competition_test_images` avec `train_images`, `train_labels`, `test_images`; le zip final attendu doit contenir exactement les noms `.tif` de `test_images`.
- L'utilisateur signale un risque de régression par suppression de code entre versions. La stratégie V5 retenue est **réintégration + durcissement**, pas simplification.

## 2) Diagnostic des problèmes actuels (v4)

1. **Incohérence roadmap**
   - `state.json` capture `finalize_forensics=60.0` au lieu de `100.0`.
2. **Coût de calcul évitable**
   - Le pipeline peut charger des stacks d'entraînement lourds avant de confirmer la présence de labels.
3. **Traçabilité stratégie d'entraînement**
   - Pas assez explicite dans l'état final sur mode d'entraînement utilisé (`supervised` vs fallback).
4. **Risque de runtime très long**
   - Absence de borne d'entraînement configurable pour contrôles rapides/itératifs.

## 3) Plan V5 (nouveau)

### Phase A — Stabilisation forensique (100%)
- Corriger l'ordre des updates roadmap pour figer `finalize_forensics=100.0` avant écriture `state.json`.
- Conserver tous les fichiers de compatibilité v4.

### Phase B — Réduction des régressions runtime (100%)
- Ajouter une pré-vérification de labels (`_quick_has_label`) pour éviter les chargements inutiles.
- Ajouter `NX46_MAX_TRAIN_ITEMS` (env var) pour limiter l'entraînement si nécessaire.

### Phase C — Transparence d'entraînement (100%)
- Enregistrer `training_strategy`, `train_items_with_labels`, `train_items_skipped_no_label`.
- Ajouter fallback quantile probe quand aucun label n'est disponible.

### Phase D — Validation stricte Kaggle (100%)
- Maintenir génération `submission.zip` + validation stricte des noms `.tif`.
- Publier le zip sur chemins alias (`submission.zip` et `nx46_vesuvius/submission.zip`) pour compatibilité notebook/Kaggle Submit Output.
- Forcer les pixels masque au format `uint8` en plage `0..255`.
- Maintenir logs forensic, Merkle, bit capture, inventaire dataset.

## 4) Ce qu'il reste à faire après V5

- Exécuter V5 sur Kaggle et comparer score public LB contre v4.
- Si score stable/hausse: promouvoir v5 comme base de production.
- Si score baisse: augmenter `NX46_MAX_TRAIN_ITEMS` et recalibrer `threshold_quantile`.

## 5) Livrable V5

- Script: `RAPPORT-VESUVIUS/src_vesuvius/nx46-vesuvius-core-kaggle-ready-v5.py`.
