# RAPPORT D’ANALYSE — V140 / V144 / V61.2 / V7.4 / Transfer Learning / Vision IR

Date: 2026-02-18

## Expertises mobilisées (notification explicite)
- **ML Engineering (segmentation 3D, calibration, pipeline Kaggle)**
- **Computer Vision scientifique (TIFF multipage, densité de masques, contraste structurel)**
- **MLOps/Kaggle Ops (format de soumission, robustesse offline, packaging reproductible)**
- **Forensic debugging (corrélation logs ↔ code ↔ artefacts)**
- **Stratégie R&D (transfer learning multi-modèles, plan d’expériences, risques de dérive)**

---

## 1) Résultats récupérés et constats factuels

### 1.1 V140
- `results.zip` contient bien un `submission.zip`.
- Le TIFF de soumission est **2D** `(320, 320)` en `uint8` avec valeurs `0/1`.
- Cela confirme la cause principale du blocage de score/soumission déjà observée historiquement: format 2D au lieu de volume multipage.

### 1.2 V144
- `results.zip` ne contient **pas** de `submission.zip`.
- Le log `nx47-vesu-kernel-new-v144.log` montre un échec précoce:
  - `RuntimeError: Offline dependency directory not found for imagecodecs`.
- Donc V144 échoue avant packaging, d’où absence de soumission.

### 1.3 V61.2
- `results.zip` contient `submission.zip`.
- TIFF de soumission en **3D** `(320, 320, 320)` `uint8` `0/1`.

### 1.4 V7.4 (dans logs V61.3)
- Le `results.zip` du dossier `v7.4-outlput-logs...` contient une soumission `submission_masks/1407735.tif`.
- TIFF en **3D** `(320, 320, 320)` `uint8` `0/1`.

### 1.5 Scores demandés (V61.2 et V7.4)
- **Problème rencontré (notification obligatoire):** aucun score Kaggle *post-submission* explicite (Public/Private LB) n’est présent dans les artefacts locaux inspectés.
- Conséquence: impossible de fournir des “nouveaux scores” certifiés sans exécution/consultation LB Kaggle.

---

## 2) Corrections implémentées dans le code

## 2.1 Correction V140 (soumission)
- Conversion explicite du masque 2D en volume 3D multipage avant écriture TIFF:
  - `mask2d -> uint8 0/255`
  - réplication sur l’axe Z (`vol.shape[0]`) pour sortir un TIFF `(Z,H,W)`.
- Compression ZIP passée en `ZIP_DEFLATED` (alignement pratique avec pipelines scorés).

## 2.2 Correction V144 (réintégration + robustesse)
- Réintégration complète du code source base V140 dans V144 (suppression de l’état cassé).
- Suppression implicite de l’injection invalide qui avait introduit des lignes non-Python.
- Durcissement du chargement offline des dépendances:
  - vérification de présence des packages avant installation,
  - exploration de plusieurs chemins de wheels Kaggle,
  - échec explicite conservé si vraiment introuvable.
- Assouplissement du blocage imagecodecs:
  - warning forensique + fallback (au lieu d’arrêt immédiat).
- Alignement de la sortie de soumission sur le format 3D multipage également dans V144.

---

## 3) Analyse de ton observation Transfer Learning (9 modèles Kaggle)

Réponse courte: **oui, c’est une piste sérieuse**, à condition de structurer l’apprentissage multi-source proprement.

### 3.1 Pourquoi ça peut améliorer
- Les 9 modèles peuvent apprendre des **priors complémentaires** (textures fines, bruit, relief local, patterns d’encre).
- Le fine-tuning final sur votre distribution peut augmenter la robustesse au domaine Vesuvius.

### 3.2 Risques techniques
- **Negative transfer** si certains modèles sont hors-domaine.
- **Divergence de normalisation** (préprocessing incompatible entre modèles).
- **Overfit compétition** si calibration mal verrouillée.

### 3.3 Recommandation d’implémentation future (V61.2 + V7.4)
1. Phase A: pré-entraînement “feature bank” multi-modèles (frozen backbone, têtes séparées).
2. Phase B: distillation vers un modèle cible unique (teacher ensemble -> student).
3. Phase C: fine-tuning progressif (unfreeze par blocs) + early stopping strict.
4. Phase D: calibration de seuil par volume (pas seuil global unique).
5. Phase E: ablations obligatoires (retirer 1 modèle à la fois pour mesurer contribution réelle).

---

## 4) Analyse “vision infrarouge type James Webb”

### 4.1 Position scientifique réaliste
- Le “James Webb” est une métaphore utile: il faut viser **plus de sensibilité spectrale et structurelle**, pas juste “plus de puissance”.
- Sur Vesuvius, l’équivalent pratique est:
  - fusion multi-échelles,
  - attention 3D anisotrope,
  - canaux dérivés “pseudo-spectraux” (gradients, Laplacien, réponses de fréquence locale),
  - supervision par cohérence volumique.

### 4.2 Ce qui est prometteur
- Encoders 3D hybrides CNN+Transformer légers.
- Heads de segmentation avec contrainte de continuité inter-slices.
- Post-traitement morphologique piloté par incertitude (au lieu d’un seuil fixe).

### 4.3 Ce qui est à éviter
- Ajouter des modules “atomiques/microscopiques” non reliés à un signal mesurable.
- Complexifier sans protocole d’ablation et validation stable.

---

## 5) Problèmes rencontrés pendant cette session (notification)
1. **Absence de scores Kaggle nouveaux V61.2/V7.4** dans les artefacts disponibles localement.
2. **V144 cassée à l’exécution** (dépendance offline `imagecodecs` introuvable dans le run fourni).
3. **Désalignement format soumission V140** (2D au lieu de 3D multipage).

---

## 6) Prochaines actions recommandées
1. Exécuter V140 corrigée puis V144 corrigée sur Kaggle (offline + soumission).
2. Capturer et archiver pour chaque run:
   - hash `submission.zip`,
   - forme TIFF `(Z,H,W)`, dtype et plage de valeurs,
   - score Public/Private LB.
3. Lancer un mini-campaign Transfer Learning (3 configurations) avec protocole d’ablation strict.

