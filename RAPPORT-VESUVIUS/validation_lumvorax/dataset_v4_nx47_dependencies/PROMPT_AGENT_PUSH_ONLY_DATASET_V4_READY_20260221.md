# PROMPT AGENT — PUSH ONLY dataset NX47 dépendances V4 (bundle déjà produit)

Tu ne dois **PAS** reconstruire. Le bundle V4 est déjà préparé localement.

## 1) Chemins exacts
- Bundle prêt: `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle`
- Vérif locale: `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/verify_dataset_v4_bundle.py`
- Dataset cible: `ndarray2000/nx47-dependencies`

## 2) Pré-check obligatoire
```bash
kaggle --version
test -f ~/.kaggle/kaggle.json
python3 RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/verify_dataset_v4_bundle.py \
  --dir RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle
```

## 3) Push (unique action)
```bash
kaggle datasets version \
  -p RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/bundle \
  -m "V4 ready bundle: wheels+so+competitor_tiff_lum+manifest"
```

## 4) Post-check
```bash
kaggle datasets files ndarray2000/nx47-dependencies --page-size 200
```

## 5) Rapport obligatoire
Créer:
- `RAPPORT-VESUVIUS/validation_lumvorax/dataset_v4_nx47_dependencies/RAPPORT_AGENT_PUSH_ONLY_V4_EXECUTION.md`

Le rapport doit contenir:
1. horodatage de chaque commande,
2. sortie de `verify_dataset_v4_bundle.py`,
3. résultat du push (version publiée),
4. listing post-push,
5. incidents/erreurs rencontrés,
6. verdict final GO/NO-GO.
