# RAPPORT EXPERT — NX46-VX V2 (analyse profonde, A→Z)

Date: 2026-02-25

## 1) Mise à jour dépôt distant (GitHub)

- Remote `origin` configuré vers `https://github.com/lumc01/Lumvorax.git`.
- `git fetch --prune` exécuté avec succès (branches distantes récupérées).

## 2) Artefacts V2 réellement présents

Le dossier demandé existe localement et contient un paquet complet de pré-exécution:

- `results.zip`
- notebook `nx46-vx-unified-kaggle-v2.ipynb`
- log brut `nx46-vx-unified-kaggle-v1.log`
- artefacts extraits: `nx46vx_v2_execution_metadata.json`, `nx46vx_v2_execution_logs.json`, `submission.zip`, `submission_masks/1407735.tif`, etc.

Constat: la pré-exécution V2 est techniquement complète côté génération d’artefacts (`READY: /kaggle/working/submission.zip`).

## 3) Est-ce que les « professeurs » ont enseigné le neurone ?

### Réponse courte
**Partiellement oui (au sens pipeline exécuté), mais pas au niveau d’un apprentissage validé par F1/IoU de validation.**

### Preuves internes
- `learning_percent_real = 100.0`
- `epochs_configured = 0`, `epochs_observed = 6`, `epochs_effective = 6`
- `files_supervised_mode = 1`
- mais `val_f1_mean_supervised = 0.0`, `val_iou_mean_supervised = 0.0`, `val_fbeta` de scan = 0.0

Interprétation:
- Le système **a bien déroulé** une procédure d’apprentissage (au moins en mode formel/forensic),
- mais les métriques de validation supervisée restent nulles dans ce run (probable limite de protocole/échantillonnage/objectif défini pour la pré-exécution).

## 4) Qu’a-t-on détecté exactement ?

`global_stats` V2 indique:

- `materials_detected = 933`
- `patterns_detected = 933`
- `golden_nonce_detected = 11`
- `anomalies_detected = 52`
- `pixels_anchor_detected = 0`
- `pixels_papyrus_without_anchor = 6144`
- `f1_ratio_curve_best_f1 = 0.686275490593166` au ratio `0.20591836734693877`

Important:
- La courbe F1/ratio interne (optimisation de ratio) est bonne **dans son espace interne**,
- mais cela ne garantit pas une hausse du score Kaggle externe.

## 5) Pourquoi `global_progress_percent = 33.333333333333336` semble bloqué ?

Ce n’est pas une preuve de freeze global; c’est surtout un **indicateur agrégé par macro-étapes** qui est réémis pendant des sous-étapes répétées.

Dans les logs, on observe des sauts typiques:
- `4.1666666667` (≈ 1/24)
- `16.6666666667` (≈ 4/24)
- `20.8333333333` (≈ 5/24)
- `33.3333333333` (≈ 8/24)

Diagnostic:
1. Le compteur global est **quantifié** (pondération fixe des macro-blocs).
2. Pendant `preflight_train/load_pair`, la boucle réécrit souvent la même fraction globale.
3. Le chiffre n’est donc pas “temps réel continu”, mais “jalon de bloc”.

## 6) Pourquoi peu de neurones « activés » dans la phase LLLLL ?

Tu vois deux échelles différentes:

1. **Niveau méta/supervisé compact** (dans metadata):
   - `active_neurons_start_total = 0`
   - `active_neurons_mid_total = 6`
   - `active_neurons_end_total = 6`
   - `meta_neuron_candidates = 45`

2. **Niveau moteur volumique** (forensic_ultra.log):
   - `SLAB_ALLOCATION` affiche des millions de neurones dynamiques (ex: 3,098,856)

Donc: ce n’est pas contradictoire. Le “6” concerne un sous-espace (neurones effectifs de la tête/optimisation), alors que le moteur volumique manipule un grand réservoir opérationnel.

## 7) Pourquoi `meta_neurons: 3` ?

`meta_neurons` est un **hyperparamètre de design** (petit nombre de contrôleurs méta) pour:
- limiter la variance,
- stabiliser les mises à jour,
- éviter un surcoût en pré-exécution.

Ce paramètre n’est pas censé varier “tout seul” dans un run si aucun mécanisme d’auto-expansion n’est activé.

## 8) Pourquoi certaines valeurs restent constantes ?

Tu as cité:
- `dust_min_size: 24`
- `mutation_noise: 0.015`
- `golden_nonce_topk: 11`
- `convergence_patience: 5`

C’est normal: ce sont des **hyperparamètres de configuration (BOOT)**, pas des métriques dynamiques. Ils restent fixes tant qu’aucune logique d’auto-tuning ne les modifie en runtime.

Sens fonctionnel:
- `dust_min_size=24`: retire petits composants parasites (<24 px).
- `mutation_noise=0.015`: amplitude de bruit d’exploration des poids.
- `golden_nonce_topk=11`: nombre de candidats top conservés.
- `convergence_patience=5`: nb d’itérations sans gain avant arrêt/convergence.

## 9) Pourquoi le score Kaggle reste ~0.303 malgré run « propre » ?

Faits disponibles:
- V1/V6: score **0.303** (succès)
- V2v4: **Submission Scoring Error** (non scoré)
- masque final identique entre versions locales (`sha256` préfixe `92590f075047...`), et ZIP de même structure (1 fichier `1407735.tif`).

Conclusion probable:
1. Plateau de performance (masque très similaire entre runs scorés).
2. L’erreur “Scoring Error” V2v4 est probablement un incident de soumission côté exécution spécifique (artefact/session Kaggle), plus qu’un changement de contenu masque.

## 10) Point critique: couverture d’entraînement

Dans `train_dataset_audit`:
- paires découvertes: `786`
- paires sélectionnées pour entraînement: `32`
- couverture: `4.0712468193%`

Donc même si `learning_percent_real=100%` (processus exécuté), la **couverture effective** reste faible sur ce run de préflight.

## 11) Décodage clair des champs BOOT (ceux que tu as fournis)

- `target_active_ratio=0.03`: cible de sparsité/activation globale (3%).
- `max_layers=320`: profondeur max de volume traité (slices z).
- `overlay_stride=8`: pas d’échantillonnage spatial pour overlays/features.
- `ratio_candidates=[...]`: grille testée pour ratio d’activation/post-seuil.
- `pruning_quantile=0.25`: coupe bas 25% (faible importance).
- `f1_stagnation_window=5`: fenêtre de stagnation F1.
- `supervised_epochs=0`: config explicite; le moteur a néanmoins observé des étapes effectives via logique interne.
- `auto_epoch_safety_cap=0`: pas de cap auto supplémentaire.
- `threshold_scan=[0.35..0.6]`: grille de seuils évalués.
- `use_unet_25d=true`, `unet_in_slices=7`, `unet_epochs=2`: chemin UNet 2.5D actif (léger).
- `strict_no_fallback=true`: interdit des fallback “silencieux”.
- `require_train_completion_100=true`: contrainte de complétion de process.
- `adapt_train_threshold_to_dataset_size=true`: adaptation seuil selon taille dataset.

## 12) Verdict expert (en profondeur)

1. **Pipeline V2 pré-exécution**: valide techniquement (artefacts + forensic + submission générée).
2. **Apprentissage utile vers score public**: pas encore démontré (plateau 0.303, métriques val supervisées nulles dans ce run).
3. **`global_progress_percent`**: comportement attendu d’un compteur par jalons, pas un vrai pourcentage continu de wall-time.
4. **Constantes BOOT**: normal qu’elles ne bougent pas; ce sont des knobs de config.
5. **Prochain levier majeur**: augmenter la couverture d’entraînement effective (32/786 → bien plus), sinon plateau probable.

## 13) Checklist d’experts à appliquer ensuite

- Vérifier la soumission V2v4 qui a échoué avec le même script d’export, en comparant session Kaggle, logs stderr Kaggle et checksum du zip soumis.
- Forcer un run avec couverture entraînement plus élevée (au moins >30% des paires), puis comparer densité masque + score.
- Séparer clairement: métriques internes (forensic) vs score compétition final (leaderboard).
- Ajouter un indicateur `global_progress_monotonic_stage` pour éviter la confusion du 33.333%. 
