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
