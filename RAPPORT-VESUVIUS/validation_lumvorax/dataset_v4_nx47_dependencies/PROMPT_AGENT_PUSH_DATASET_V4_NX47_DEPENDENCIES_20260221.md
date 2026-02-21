# PROMPT AGENT — Push Dataset V4 NX47 Dépendances (procédure unique, chemins exacts)

Tu dois **uniquement** pousser la V4 préparée et produire un rapport d’exécution détaillé.

## 0) Répertoire source unique (NE PAS CHANGER)
- Racine bundle V4: `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies`
- Dossier artefacts à pousser: `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle`
- TIFF concurrent source obligatoire: `RAPPORT-VESUVIUS/exemple-soumision-notebook-concurrent/results.zip`

## 1) Vérifications préalables obligatoires
1. `kaggle --version`
2. `test -f ~/.kaggle/kaggle.json`
3. `kaggle datasets files ndarray2000/nx47-dependencies --page-size 200`

Si une commande échoue => STOP immédiat + NO-GO.

## 2) Construire / vérifier le bundle V4 (si non déjà fait)
Depuis la racine du repo:

```bash
python3 RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/build_dataset_v4_bundle.py \
  --repo-root . \
  --output-dir RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle \
  --wheel-source /kaggle/input/datasets/ndarray2000/nx47-dependencies \
  --wheel-source RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies

python3 RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/verify_dataset_v4_bundle.py \
  --dir RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle
```

Critère GO local:
- `verify_dataset_v4_bundle.py` doit afficher `"status": "ok"`.

## 3) Contrôle assets obligatoires V4
Le bundle doit contenir **sans exception**:
- wheels V1+V2 requis,
- `liblumvorax_replit.so`,
- `competitor_teacher_1407735.tif` (extrait de `results.zip/submission.zip`),
- `competitor_teacher_1407735.lum` (conversion du TIFF),
- `dataset_v4_build_report.json`.

## 4) Push Kaggle dataset V4
- Dataset cible: `ndarray2000/nx47-dependencies`
- Dossier à pousser: `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle`

Commande:
```bash
kaggle datasets version \
  -p RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle \
  -m "V4 complete NX47 dependencies: wheels union + rebuilt liblumvorax_replit.so + competitor TIFF/LUM teacher assets + build manifest"
```

## 5) Post-check obligatoire
```bash
kaggle datasets files ndarray2000/nx47-dependencies --page-size 200
```

Confirmer présence:
- `liblumvorax_replit.so`
- `competitor_teacher_1407735.tif`
- `competitor_teacher_1407735.lum`
- wheels V1+V2 requis
- `dataset_v4_build_report.json`

## 6) Exécuter le test V3 complet sur Kaggle (copier-coller)
Script exact:
- `RAPPORT-VESUVIUS/validation_lumvorax/notebook_validation_lumvorax_dependances_360_kaggle_single_cell_V3_COMPLETE_20260221.py`

Attendus minimaux:
- `status: ok`
- `events_count > 0`
- `module_headers > 0` (si sources montées)
- `write_compression_used = LZW` en mode strict
- aucun fallback en mode strict

## 7) Rapport final obligatoire (sans omission)
Créer:
- `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/RAPPORT_AGENT_PUSH_DATASET_V4_EXECUTION.md`

Le rapport doit contenir:
1. Date/heure de chaque étape.
2. Commandes exactes exécutées.
3. Sorties console importantes (succès/erreurs).
4. Version dataset Kaggle publiée (ID/version).
5. Liste des fichiers détectés post-push.
6. Tous les problèmes rencontrés et comment ils ont été traités.
7. Verdict final GO/NO-GO.

## 8) Interdictions
- Ne pas changer le nom du dataset cible.
- Ne pas pousser un autre dossier que `.../dataset_v4_nx47_dependencies/bundle`.
- Ne pas masquer une erreur (si échec => le rapport doit le dire explicitement).
