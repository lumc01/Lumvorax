# RAPPORT ANALYSE PROFONDE V5 — EXÉCUTION, LOGS, AVANT/APRÈS, PÉDAGOGIE

## 1) Disponibilité des artefacts demandés

### 1.1 Artefacts trouvés
- V4 présent et exploitable: logs + `results.zip`.
- Notebook de référence présent: `output_logs_vesuvius/nx46-vesuvius-challenge-surface-detection.ipynb`.
- Code V5 présent: `src_vesuvius/nx46-vesuvius-core-kaggle-ready-v5.py`.

### 1.2 Artefacts manquants pour l'analyse demandée
- Dossier demandé non trouvé localement:
  - `RAPPORT-VESUVIUS/output_logs_vesuvius/v5-outlput-logs--nx46-vesuvius-core-kaggle-ready`
- Fichier demandé non trouvé localement:
  - `.../PLAN_FEUILLE_DE_ROUTE_V4_REPONSES_EXPERTES.md`

Conséquence: l'analyse V5 runtime ci-dessous est **forensic par inférence** (code V5 + baseline V4), pas un audit de logs V5 réellement exécutés.

---

## 2) Baseline V4 observée (avant)

## 2.1 Ce que V4 faisait bien
- Génération `submission.zip`.
- Validation zip membre `.tif` attendue.
- Forensic riche (state, merkle, bit capture, metrics, inventaire dataset).

## 2.2 Faiblesses V4 objectivées
- `finalize_forensics` restait à `60.0` dans l'état final.
- `training_strategy` absent de l'état final.
- Chemin de soumission potentiellement ambigu selon runner Kaggle.
- Encodage masque pas explicitement forcé à 0/255 dans la version antérieure.

---

## 3) Correctifs V5 observables dans le code (après)

## 3.1 Soumission Kaggle durcie
- `submission.zip` produit puis publié sur chemins alias critiques.
- Évite l'erreur: `Could not find provided output file nx46_vesuvius/submission.zip`.

## 3.2 Format masque explicite
- Conversion en `uint8` binaire 0/255 avant écriture TIFF LZW.

## 3.3 Forensic/état améliorés
- journal `SUBMISSION_PATHS_PUBLISHED`.
- ajout `submission_zip_aliases` dans l'état final.
- `finalize_forensics` poussé à 100% avant `finalize()`.
- ajout `training_strategy` dans le résultat final.

## 3.4 Robustesse pipeline
- préfiltrage des items train sans label (`_quick_has_label`).
- fallback calibration quand aucun label train exploitable.

---

## 4) Comparaison pédagogique avant/après

## 4.1 Chemin de sortie
- Avant (V4): un chemin principal de zip.
- Après (V5): chemin principal + alias publiés pour compatibilité notebook/submit output.

## 4.2 Valeurs pixel masque
- Avant: ambigu selon implémentation (risque 0/1).
- Après: imposé à 0/255, compatible conventions d'évaluation TIFF binaire.

## 4.3 Roadmap finalisée
- Avant: `finalize_forensics=60.0` enregistré.
- Après: `finalize_forensics=100.0` avant écriture état.

## 4.4 Explicabilité entraînement
- Avant: pas de stratégie explicitée.
- Après: `training_strategy` et compte des items train/labels.

---

## 5) Réponses "questions expertes" (style feuille V4)

1. **Comment prouver que la sortie Kaggle est retrouvable ?**
   - Vérifier existence des alias publiés dans l'état + logs event `SUBMISSION_PATHS_PUBLISHED`.

2. **Comment prouver que les pixels sont 0/255 ?**
   - Lire un TIFF du zip, contrôler `np.unique(mask)` ∈ `{0,255}`.

3. **Comment prouver absence de régression de validation zip ?**
   - Contrôler `zip_members_validated == true`, `missing=[]`, `unexpected=[]`.

4. **Comment prouver résilience sans labels ?**
   - Vérifier `training_strategy=fallback_quantile_probe` quand labels absents.

5. **Comment prouver cohérence forensic finale ?**
   - Vérifier roadmap complète à 100% et présence du bundle logs complet.

---

## 6) Limitations et prochaines actions

- Sans dossier de résultats `v5-outlput-logs...`, impossible de produire un "avant/après runtime V5" chiffré réel.
- Action immédiate recommandée:
  1) ajouter/commiter les artefacts v5 demandés,
  2) relancer ce rapport pour version **preuves runtime complètes**.

---

## 7) Mini protocole de vérification V5 dès réception des logs

- Extraire `results.zip`.
- Vérifier `state.json`:
  - `submission_zip_aliases` présent,
  - `training_strategy` présent,
  - `roadmap_percent.finalize_forensics == 100.0`.
- Ouvrir `submission.zip`:
  - noms `.tif` strictement égaux aux fichiers test attendus.
- Lire un masque:
  - uniques = `{0,255}`.

