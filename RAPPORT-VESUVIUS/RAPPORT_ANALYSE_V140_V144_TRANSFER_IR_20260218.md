# RAPPORT D’ANALYSE — V140 / V144 / V61.2 / V7.4 / Transfer Learning / Vision IR

Date: 2026-02-18 (mise à jour post-captures Kaggle)

## Expertises mobilisées (notification explicite)
- **ML Engineering (segmentation 3D, calibration, pipeline Kaggle)**
- **Computer Vision scientifique (TIFF multipage, densité de masques, contraste structurel)**
- **MLOps/Kaggle Ops (format de soumission, robustesse offline, packaging reproductible)**
- **Forensic debugging (corrélation logs ↔ code ↔ artefacts ↔ statut Kaggle)**
- **Stratégie R&D (transfer learning multi-modèles, plan d’expériences, gestion de risque de dérive)**

---

## 1) Résultats récupérés et constats factuels

### 1.1 V140
- `results.zip` contient bien un `submission.zip`.
- Le TIFF de soumission observé dans cet artefact historique est **2D** `(320, 320)` en `uint8` avec valeurs `0/1`.
- Conclusion: cause cohérente d’un échec de soumission/score (attendu: volume multipage 3D).

### 1.2 V144
- `results.zip` historique ne contient **pas** de `submission.zip`.
- Le log montre un arrêt précoce sur dépendance offline:
  - `RuntimeError: Offline dependency directory not found for imagecodecs`.
- Conclusion: run interrompu avant packaging.

### 1.3 V61.2 — **mise à jour score**
- Confirmation fournie via capture Kaggle (session utilisateur):
  - `NX47-VESU-KERNEL NEW V61.2 - Version 2` avec **Public Score = 0.387**.
- Donc, au stade actuel, V61.2 est **au même niveau public** que v61.1 (pas de gain chiffré net public visible pour l’instant).

### 1.4 V7.3 / v7.4
- Les artefacts locaux confirment un format de sortie 3D multipage correct pour les pipelines récents.
- **Problème d’état produit (notification):** vous avez indiqué que la finalisation V7.3 est encore en cours côté campagne; le score final de référence à retenir doit donc être validé à la clôture de ce run.

---

## 2) Corrections techniques déjà implémentées (V140/V144)

### 2.1 Correction V140 (soumission)
- Conversion explicite du masque en volume multipage avant écriture TIFF de soumission.
- Compression du ZIP en `ZIP_DEFLATED`.
- Objectif: garantir compatibilité du package avec le scorer Kaggle.

### 2.2 Correction V144 (réintégration + robustesse)
- Réintégration de la base V140 pour éviter les pertes de blocs source.
- Durcissement de la résolution des dépendances offline (détection package présent + recherche multi-chemins de wheels).
- Remplacement du hard-stop imagecodecs par journalisation + stratégie de repli quand possible.

---

## 3) Analyse Transfer Learning — 9 modèles Kaggle du concurrent

## 3.1 Inventaire (d’après capture fournie)
Les 9 variantes visibles sont:
1. V1 `default`
2. V1 `segformer.mit.b2`
3. V1 `transunetseresnext`
4. V1 `segformer.mit.b4`
5. V2 `default`
6. V2 `segformer.mit.b2`
7. V2 `transunetseresnext`
8. V2 `transunet`
9. V3 `transunet`

## 3.2 Est-ce que l’apprentissage simultané peut améliorer?
Oui, **potentiellement**, mais seulement avec une architecture de transfert contrôlée. Entraîner “tout d’un coup” sans protocole conduit souvent à:
- conflits de normalisation,
- transfert négatif,
- sur-ajustement local sans gain leaderboard.

## 3.3 Stratégie recommandée (implémentable sans faux-semblants)

### Phase A — Extraction de priors (teachers figés)
- Charger les 9 modèles en mode inference-only.
- Produire pour chaque patch 3D:
  - logits moyens,
  - incertitude (variance inter-modèles),
  - cartes de désaccord.

### Phase B — Distillation vers un student unique
- Student léger (UNet 2.5D/3D hybride) entraîné sur:
  - vérité terrain (quand disponible),
  - pseudo-cibles pondérées par confiance d’ensemble.

### Phase C — Fine-tuning progressif
- Débloquer couches par blocs (head → decoder → encoder).
- Early stopping sur métrique interne robuste (FBeta + stabilité volume).

### Phase D — Calibration de décision
- Seuils calibrés par volume (pas un seuil global unique).
- Post-traitement piloté par incertitude (filtrer zones à fort désaccord).

### Phase E — Ablations obligatoires
- Retirer 1 teacher à la fois (9 runs) pour mesurer contribution réelle.
- Conserver uniquement les teachers qui apportent un gain mesurable stable.

---

## 4) Plan concret d’implémentation future pour les versions actives

## 4.1 Pour NX47 (branche v61.x / V140-V144)
1. Ajouter module `teacher_ensemble_infer` (chargement + logits + variance).
2. Ajouter dataset de distillation intermédiaire (patches + soft targets).
3. Ajouter boucle `train_student_distilled` avec weighting par confiance.
4. Ajouter calibration finale par volume et export d’audit.

## 4.2 Pour NX46 (v7.x)
1. Réutiliser l’ensemble enseignant comme bloc externe de scoring.
2. Fusionner score NX46 + student distillé via pondération validée.
3. Verrouiller format sortie Kaggle (3D multipage + zip stable) avant soumission.

---

## 5) Analyse “vision infrarouge ultra-puissante type James Webb”

### 5.1 Position scientifique
- “IR ultra-puissant” doit être traduit en **features pseudo-spectrales exploitables**:
  - réponses fréquentielles locales,
  - gradients multi-échelles,
  - contrastes inter-couches Z,
  - signatures de texture anisotropes.

### 5.2 Ce qui est réaliste et utile
- Ajouter un bloc d’encodeur multi-échelles avec attention spatiale+profondeur.
- Apprendre une carte d’incertitude pour guider le seuillage final.
- Contraindre la continuité inter-slices pour éviter le bruit isolé.

### 5.3 Ce qu’il faut éviter
- Modules “science-fiction” non reliés à un signal mesuré.
- Complexification sans protocole d’ablation et validation Kaggle.

---

## 6) Problèmes rencontrés et notifiés dans cette mise à jour
1. Les artefacts locaux seuls ne suffisent pas toujours à reconstruire l’état leaderboard exact de toutes les runs.
2. V144 historique est interrompue avant packaging (dépendances offline).
3. Le score V7.3 final doit être confirmé à la fin de votre campagne en cours.

---

## 7) Décision opérationnelle proposée
1. **Geler** V61.2 à score public constaté `0.387` comme point de référence court terme.
2. **Finaliser** la campagne V7.3 et archiver son score final certifié.
3. Lancer une **campagne TL en 3 paliers**:
   - P1: distillation 9→1 sans fine-tuning profond,
   - P2: distillation + fine-tuning partiel,
   - P3: distillation + fine-tuning complet + calibration dynamique.
4. Ne promouvoir que les variantes démontrant un gain stable (pas un gain ponctuel).
