# PLAN D'ADAPTATION NX-46 VESUVIUS : ÉTAT FINALISÉ

## 1) OBJECTIF TECHNIQUE
Adapter **NX-46 AGNN** au notebook Vesuvius en mode **100% offline**, avec:
- Remplacement de la logique neuronale statique par allocation dynamique (slab).
- Intégration MemoryTracker bit-à-bit.
- Traçabilité HFBL-360 (log nanoseconde + CSV + JSON + chaîne Merkle).
- Génération de soumission Kaggle au format existant quand `sample_submission.csv` est disponible.

---

## 2) ARCHITECTURE AVANT / APRÈS

### Avant
- Fichier `nx46_vesuvius_core.py` en conflit Git (`<<<<<<< HEAD` / `>>>>>>> ...`).
- Mélange de 2 implémentations incompatibles (NumPy vs PyTorch) et présence de commentaire de stub (`Logique d'apprentissage simulée/réelle ici`).
- Absence de pipeline robuste unifié pour ingestion train/test + sortie submission exploitable immédiatement.

### Après
- Noyau unique consolidé dans `nx46_vesuvius_core.py`.
- Pipeline déterministe offline:
  - découverte automatique des fragments train/test,
  - chargement TIFF,
  - calibration d’un seuil sur labels d’entraînement,
  - inférence test,
  - export `submission.csv`.
- Forensic complet:
  - `forensic_ultra.log`,
  - `metrics.csv`,
  - `state.json`,
  - `bit_capture.log`,
  - `merkle_chain.log`.

---

## 3) FEUILLE DE ROUTE TEMPS RÉEL (% D’AVANCEMENT)

- **PHASE 1 — Audit & nettoyage des conflits : 100%**
  - Conflit de merge identifié puis supprimé.
  - Stubs explicites retirés.
- **PHASE 2 — Intégration AGNN NX-46 : 100%**
  - Slab allocation dynamique branchée sur variance + gradient.
  - Réseau de décision offline sur carte d’énergie d’encre.
- **PHASE 3 — MemoryTracker & forensic : 100%**
  - Capture binaire des tranches (fenêtre bytes configurable).
  - Merkle chain append-only et métriques CPU réelles.
- **PHASE 4 — Validation offline + soumission : 100%**
  - Génération soumission CSV alignée sur `sample_submission.csv`.
  - Rapport d’état final exporté en JSON.

---

## 4) CRITÈRES QUALITÉ (ANTI-STUB / ANTI-HARDCODING)
- Pas de placeholder d’apprentissage.
- Pas de try/catch autour des imports (conformité règle projet).
- Mesure QI basée sur télémétrie réelle (`pixels_traités / cpu_ns`).
- Logs écrits en continu pendant train/inférence.

---

## 5) LIVRABLES
- `nx46_vesuvius_core.py` (script unique copiable dans Kaggle notebook).
- `RAPPORT_AUDIT_FINAL_NX46_VESUVIUS.md` (audit avant/après, corrections, check-list).
- `PROMPT_UNIQUE_REPRISE_NX46_VESUVIUS.md` (prompt unique expert, structuré, réutilisable).
