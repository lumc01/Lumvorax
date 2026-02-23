# Diagnostic total Lumvorax ↔ Kaggle (NX47 Dependencies V13)

## Contexte
Le rapport `lumvorax_360_validation_report_v6_binary.json` montre un statut `ok_with_warnings`, non conforme à l'objectif "zéro warning".

## Problèmes identifiés

1. **Chargement natif `.so` échoué**
   - Erreur: `undefined symbol: neural_config_create_default`
   - Cause: la librairie `liblumvorax.so` exportée dans le dataset dépend d'un symbole absent au runtime Kaggle.

2. **Roundtrip TIFF LZW non strictement garanti**
   - Erreur constatée: `"<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"`.
   - Cause probable: incohérence ABI/runtime entre environnements Python Kaggle et wheel `imagecodecs` du dataset.

3. **Validation trop permissive dans le validateur V6**
   - Le validateur pouvait produire `ok_with_warnings`.
   - Le chargement `.so` n'était pas bloquant (`enforce_so_load=false` par défaut).
   - Le roundtrip pouvait fallback vers `ADOBE_DEFLATE`/`NONE`, masquant les incompatibilités LZW.

4. **Alignement de version incomplet**
   - Métadonnées dataset/kernel encore marquées V7 alors que la cible opérationnelle est V13.

## Correctifs appliqués

1. **Mode strict V13 implémenté dans le validateur Kaggle**
   - Nouveau report name: `lumvorax_dependency_360_kaggle_single_cell_v13_strict`.
   - Fichier de sortie report migré vers `lumvorax_360_validation_report_v13_strict.json`.

2. **`so_load` rendu bloquant en mode strict**
   - Si le `.so` ne charge pas, la validation échoue immédiatement (`RuntimeError`).
   - `ENFORCE_SO_LOAD` activé par défaut si `STRICT_NO_FALLBACK=1`.

3. **Compression TIFF durcie (zéro fallback)**
   - Plan de compression limité à LZW uniquement.
   - Si backend LZW indisponible (`imagecodecs.lzw_encode` absent), échec immédiat.
   - Si `tifffile.imwrite(..., compression="LZW")` échoue, échec immédiat.

4. **Scanner de compatibilité wheel/runtime ajouté**
   - Vérification de compatibilité des tags wheel via `packaging.tags.sys_tags` + `parse_wheel_filename`.
   - Toute wheel incompatible avec le runtime Kaggle force l'état `dataset_artifacts.status = fail`.

5. **Versions Kaggle synchronisées en V13**
   - Kernel metadata: `LUMVORAX V13 Certification Test`.
   - Dataset metadata: `NX47 Dependencies V13`.

## Recommandations opérationnelles pour publier la vraie V13 dataset

1. **Rebuilder et republier `liblumvorax.so`** avec dépendances complètes, sans symbole non résolu (`neural_config_create_default`).
2. **Publier des wheels compatibles runtime Kaggle cible** (tags réellement matchés par `sys_tags()` du notebook final).
3. **N'inclure qu'une seule version de `tifffile`** pour éviter ambiguïté d'installation.
4. **Exécuter le validateur V13 strict dans Kaggle** et n'accepter que `status="ok"` avec `warnings=[]`.

## Critère d'acceptation final (zéro warning)

- `status == "ok"`
- `warnings_count == 0`
- `so_load_status == "ok"`
- `roundtrip_status == "ok"`
- `dataset_artifacts.wheel_compatibility.incompatible_wheels == []`
