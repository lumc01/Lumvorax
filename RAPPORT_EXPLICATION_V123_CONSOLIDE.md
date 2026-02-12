# RAPPORT D'EXPLICATION V123 CONSOLIDÉ

## 1) Lecture claire des derniers résultats V122 fournis

Les logs V122 montrent explicitement les métriques suivantes sur `1407735.tif` :

- neurones actifs: début `0`, milieu `6`, fin `6`
- dataset découvert: dossiers `.` `deprecated_train_images` `deprecated_train_labels` `test_images` `train_images` `train_labels`
- types de fichiers: `.tif` (1613), `.csv` (2)
- calculs par seconde: `6,441,120` ops/s fichier et `3,327,176` ops/s global
- pixels traités: `32,768,000`
- pixels ancrés détectés: `0`
- pixels papyrus sans ancre: `3,072`
- matériaux détectés: `1`
- tranches traitées: `320`
- golden nonce détectés: `11`
- patterns détectés: `1`
- découvertes inconnues: `0`
- anomalies détectées: `52`

Interprétation rapide:

- Le masque final est piloté surtout par la calibration de ratio (`0.03`) et non par l'hystérésis (`0.0` sur ce cas).
- Le neurone devient dense rapidement (6/6 poids non nuls) puis se stabilise.
- Le pipeline est fonctionnel mais mono-neurone, ratio quasi fixe, sans gouvernance évolutive multi-agents.

## 2) Ce que V123 consolide dans une seule version

La V123 garde la base V122 (forensic + Merkle + hardware + inventaire dataset) et ajoute:

1. **Méta-neurones compétitifs** (`meta_neurons`) avec sélection par objectif CE/sparsité/proxy-F1.
2. **Ratio adaptatif slice-map** via scan de candidats (`ratio_candidates`) et sélection automatique.
3. **Mémoire évolutive** (`NX47EvolutionMemory`) : historique F1 proxy + ratios.
4. **Learning rate adaptatif** selon stagnation historique.
5. **Auto-pruning intelligent** par quantile de faibles poids.
6. **Mutation contrôlée** en cas de stagnation sur fenêtre F1.
7. **Simulation 100 volumes** optionnelle (`V123_RUN_SIMULATION_100=1`) avec résumé F1.
8. **Compteurs globaux enrichis**: candidats méta-neurones, événements pruning/mutation, ratio moyen sélectionné.

## 3) Conseils de pilotage opérationnel

- Si recall trop bas: élargir `ratio_candidates` vers `0.10-0.16`.
- Si bruit/FP trop haut: augmenter `pruning_quantile` et réduire `mutation_noise`.
- Si système stable mais plafonne: activer simulation + augmenter `meta_neurons` à 4-5.
- Si dérive temporelle: remonter `f1_stagnation_window` pour lisser les décisions de mutation.

## 4) Cible V123 = système complet unique

Avec cette consolidation, une seule version répond à la fois à:

- audit dataset (dossiers + types)
- audit hardware/système réel
- métriques segmentation avancées (pixels/anchors/materials/patterns/golden/anomalies)
- performance compute (ops estimées + ops/s)
- dynamique neuronale (début/milieu/fin + méta-neurones + mutation/pruning)
- traçabilité forensic (jsonl Merkle + signatures nanoseconde)

