# NX46 V7.5 — Campagne 10 runs (application immédiate)

## Objectif
Exécuter 10 variantes de calibrage seuil/poids 3D pour améliorer le rappel sans casser la conformité.

## Matrice utilisée
- Source: `RAPPORT-VESUVIUS/campaign_run_matrix_v61_3_v7_5.json`
- 10 runs (`v7.5-run-01` à `v7.5-run-10`) avec variation:
  - `NX46_THRESHOLD_QUANTILE`
  - `NX46_SCORE_BLEND_3D_WEIGHT`
  - `NX46_Z_SMOOTHING_RADIUS`

## Implémentation appliquée dans v7.5
- Ajout `threshold_quantile` configurable par environnement.
- Ajout `NX46_RUN_TAG` pour traçabilité.
- Ajout mode `NX46_DRY_RUN=1` pour campagne préflight locale 10/10.

## Résultat d’exécution ici
- 10/10 runs préflight local exécutés (retour 0).
- Limitation: scoring Kaggle non exécutable ici (absence de runtime Kaggle + chemins `/kaggle/input/*`).

---

## Mise à jour score Kaggle (retour utilisateur)
- Soumission réussie `nx46_vesuvius_core_kaggle_ready` (Version 14): **Public Score = 0.303**.
- Référence NX47 V61.2: **0.387**.

Écart observé: **-0.084** vs baseline 0.387.

---

## Analyse rapide de cause probable (avec artefacts offline)
- Le `results.zip` v7.5 est conforme (TIFF 3D `uint8` binaire `0/1`).
- Densité finale de soumission (`ink_ratio`) observée: **0.02341797** (≈2.34%), très faible.
- Cette sous-densité est cohérente avec un risque de **rappel insuffisant** (under-segmentation).

---

## Questions critiques à traiter au prochain run
1. Quel `threshold_quantile` permet de remonter la densité entre 0.06 et 0.12 sans exploser les faux positifs ?
2. Le `score_blend_3d_weight=0.78` est-il trop conservateur pour les tranches faibles ?
3. Quels z-slices concentrent la perte de rappel (analyse `ink_ratio_by_slice`) ?
4. Quelle combinaison seuil + smoothing garde le format conforme et remonte le score LB > 0.303 ?

---

## Feuille de route corrective (itération courte)

### Étape 1 — Diagnostic densité
- Exporter et versionner `ink_ratio_global` + `ink_ratio_by_slice` + percentiles de probas.
- Définir une plage cible de densité validée par rapport aux runs scorés.

### Étape 2 — Mini A/B/C calibration
- Run A: quantile plus permissif (+recall)
- Run B: blend 3D réduit
- Run C: combinaison A+B

### Étape 3 — Go/No-Go
- **GO** si score > 0.303 ET conformité zip/tiff inchangée.
- **NO-GO** sinon; conserver la meilleure config scorée.
