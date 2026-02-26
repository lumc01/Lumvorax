# Cours pédagogique A→Z — Meta-neurones (V5 / V6 / future V7)

## 1) Définition opérationnelle (dans CE projet)
Dans Lumvorax/NX46-VX, un **meta-neurone** n'est pas un neurone biologique réel ni un neurone deep-learning unique: c'est une **unité de décision de haut niveau** qui pilote la sélection de signaux, la calibration de seuil, la traçabilité (Merkle), et la robustesse Kaggle (format soumission + offline dependencies).

En pratique, la logique est visible dans:
- les configs V6 (`meta_neurons`, `threshold_scan`, `ratio_candidates`, etc.),
- la chaîne forensic (`merkle`, `bit trace`, événements),
- la phase de scoring/projection (2.5D/3D),
- la phase de packaging strict (`submission.zip`, TIFF LZW, binaire 0/1 ou 0/255).

## 2) Anatomie pédagogique (analogie biologique → architecture logicielle)

### 2.1 Schéma anatomique simplifié

```text
[Entrées sensorielles]
  TIFF 3D (Z,H,W) + labels
          |
          v
[Pré-traitement / normalisation]
          |
          v
[Meta-neurone #1: perception]
  - gradients, contrastes, proxies matière
          |
          v
[Meta-neurone #2: décision]
  - score projection
  - calibration seuil / ratios
          |
          v
[Meta-neurone #3: contrôle exécutif]
  - règles Kaggle
  - format zip/tiff
  - logs forensic (merkle + bit trace)
          |
          v
[Sorties motrices]
  submission.zip + état + métriques + manifestes
```

### 2.2 Correspondance "neuro" vs "code"
- **Dendrites (collecte)** ↔ lecture de volumes TIFF multi-pages et auto-discovery de layout dataset.
- **Soma (intégration)** ↔ fusion des signaux (gradients + contraste + projection 3D/2.5D).
- **Axone (décision)** ↔ seuillage final + masque binaire.
- **Synapses (mémoire/traçabilité)** ↔ Merkle chain + bit capture + state/metrics.
- **Homéostasie** ↔ fallback offline deps + validation stricte des sorties.

## 3) Où s'intègre le meta-neurone dans l'exécution (pipeline réel)

### Étape A — Boot environnement (offline)
- Installation depuis dataset de wheels (`nx47-dependencies`) avec fallback de chemins.
- Objectif: garantir exécution Kaggle sans internet runtime.

### Étape B — Découverte dataset
- Détection de plusieurs layouts (`test_images`, `train/test fragments`, legacy).
- Inventaire des assets + journal des tentatives.

### Étape C — Perception / scoring
- V3/V7 utilisent un noyau natif de projection d'énergie d'encre (gradients Z/Y/X).
- V7.7 ajoute score 3D natif + smoothing Z + blend pondéré (et 2.5D optionnel).

### Étape D — Calibration / apprentissage
- Seuil appris sur labels si disponibles.
- Fallback quantile si supervision incomplète.
- En V6, présence de stratégie mixte avec features multi-signaux + 2.5D Torch optionnel.

### Étape E — Décision binaire et export
- Conversion masque binaire (`0_1` ou `0_255`) + validation stricte du contenu zip.
- Vérification shape Z/H/W et domaine de valeurs.

### Étape F — Forensic / preuve
- Logs multi-fichiers, metrics CSV, state JSON, Merkle chain, manifestes de dépendances.

## 4) Comparaison détaillée V5 vs V6 vs V7 (future)

## 4.1 État des artefacts V5
- Le fichier source `v5.py` présent dans le repo est vide (0 ligne) : impossible d'auditer son code local exact.
- Le comportement historique "v5-compatible" est mentionné dans les objectifs V7.7 (continuité des sorties/forensics).

## 4.2 V6 (code notebook unifié)
Forces:
- stack hybride (NumPy + CuPy + Torch optionnel),
- mécanismes offline install,
- modèle de continuité NX + Merkle signatures,
- paramétrage riche (meta_neurons, threshold scan, contraintes de couverture train, etc.).

Fragilités observées:
- complexité opérationnelle élevée,
- dépendance à chemins Kaggle stricts,
- validations environnement parfois bloquantes (`dataset_root_required_but_not_found`).

## 4.3 Future V7 (orientation actuelle du repo: v7.7)
Forces:
- exécution plus robuste et déterministe côté Kaggle,
- scoring 3D natif explicitement contrôlé (blend + smoothing),
- validation de contenu soumission (forme + binaire) plus stricte,
- manifestes de dépendances et outputs forensic enrichis.

Compromis:
- il faut calibrer finement la densité de prédiction (risque de sous-segmentation),
- il faut conserver la traçabilité sans re-complexifier l'intégration.

## 5) Lecture des logs (ce qu'ils racontent vraiment)

### 5.1 Signal de stabilité
- Logs V7 montrent pipeline "forensic + validation zip" très structuré.
- Le mode strict empêche de pousser un zip invalide.

### 5.2 Signal de risque qualité
- Le rapport comparatif des scores montre un problème de densité prédite (under-segmentation) entre versions.
- Même avec un pipeline propre, une densité trop faible peut dégrader le score Kaggle.

### 5.3 Signal de risque intégration
- Un rapport strict V13 indique un échec de chargement `liblumvorax.so` (symbole manquant).
- Un rapport V6 binaire montre un échec de résolution du `dataset_root`.

=> Conclusion: les "meta-neurones" doivent optimiser **à la fois** précision de segmentation **et** résilience d'exécution.

## 6) Processus internet / offline (clarification)
Le design cible est **offline-first**:
1. Les wheels sont pré-packagés dans un dataset Kaggle.
2. Le notebook installe localement (sans index externe).
3. Le run lit les inputs Kaggle mountés.
4. Le run produit `submission.zip` + traces.

Donc le "processus internet" est surtout en **amont** (préparation du dataset de dépendances), pas pendant l'inférence.

## 7) Différences conceptuelles avec d'autres approches existantes

### 7.1 vs U-Net supervisé pur
- U-Net pur apprend end-to-end un mapping image→masque.
- Meta-neurone Lumvorax combine heuristiques physiques + calibration + contrôle forensic + conformité compétition.

### 7.2 vs pipeline heuristique pur sans logs
- Heuristique pure: rapide mais peu auditable.
- Lumvorax: ajoute chaîne de preuve (Merkle, metrics, state) et règles anti-régression de format.

### 7.3 vs stack C/C++ custom native
- Native custom peut être plus rapide, mais plus fragile en Kaggle.
- V7 orienté Python pur privilégie reproductibilité et validation continue.

## 8) Recommandation pédagogique finale
Pour comprendre "la totalité" du fonctionnement, il faut suivre ces 3 axes en parallèle:
1. **Axe calcul**: comment le score est construit (gradients, blending, seuils).
2. **Axe décision**: comment on passe du score au masque final (calibration densité / seuil).
3. **Axe gouvernance**: comment on garantit que le résultat est traçable, valide Kaggle, et reproductible.

C'est l'intersection de ces 3 axes qui constitue le "meta-neurone" opérationnel dans ce projet.
