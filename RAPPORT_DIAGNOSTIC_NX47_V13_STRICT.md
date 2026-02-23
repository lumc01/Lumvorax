# Diagnostic total Lumvorax ↔ Kaggle (NX47 Dependencies V13)

## Contexte
Le rapport historique `lumvorax_360_validation_report_v6_binary.json` montrait `ok_with_warnings`, ce qui viole la certification "zéro warning".

## Problèmes racines confirmés

1. **Échec de chargement natif `.so`**
   - Erreur observée: `undefined symbol: neural_config_create_default`.
   - Impact: la dépendance native n'est pas réellement utilisable même si le fichier est présent.

2. **LZW non garanti au runtime Kaggle**
   - Erreur observée: `"<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"`.
   - Cause typique: mismatch de wheel/tags Python runtime (ex: wheel non compatible runtime effectif).

3. **Validation précédente trop permissive**
   - Acceptait des warnings au lieu d'un échec bloquant.
   - Autres compressions (fallback) pouvaient masquer le vrai défaut LZW.

4. **Synchronisation versionnelle incomplète**
   - Le branding test/dataset était encore en V7.

## Correctifs appliqués (V13 strict réel)

1. **Validation par packages requis + compatibilité tags runtime**
   - Le validateur vérifie chaque package requis (`imagecodecs`, `numpy`, `tifffile`, etc.).
   - Chaque package doit avoir **au moins une wheel compatible** avec `packaging.tags.sys_tags()`.
   - Toute absence/incompatibilité requise déclenche `status=fail`.

2. **Installation offline sélectionnée sur wheel compatible**
   - La sélection n'est plus figée sur un seul nom de wheel.
   - Le validateur choisit automatiquement la meilleure wheel compatible trouvée dans le dataset.

3. **LZW durci sans fallback**
   - Vérifie explicitement `imagecodecs.lzw_encode` et `imagecodecs.lzw_decode`.
   - Si indisponible: échec immédiat.
   - `tifffile.imwrite(..., compression='LZW')` doit fonctionner réellement.

4. **`so_load` bloquant en strict**
   - `liblumvorax.so` doit se charger via `ctypes.CDLL`.
   - Sinon échec immédiat, aucun pass "avec warnings".

5. **Synchronisation V7 → V13 mise à jour**
   - Test kernel Kaggle renommé en `LUMVORAX V13 Certification Test`.
   - Dataset metadata renommée en `NX47 Dependencies V13`.

## Versions compatibles LZW / imagecodecs (règle fiable)

> Ce n'est pas un simple "nom de version" fixe: la compatibilité dépend du triplet **Python ABI + plateforme + manylinux tag**.

- `imagecodecs` doit être compatible avec les tags runtime Kaggle exposés par `sys_tags()`.
- Pour Python 3.12 Kaggle, préférer:
  - wheels `cp312-*` (ou `abi3` réellement supporté),
  - manylinux compatible avec l'image Kaggle utilisée.
- Même logique pour `numpy`, `pillow`, `scipy`, `scikit-image`.

## Critère d'acceptation final (zéro warning, zéro erreur)

- `status == "ok"`
- `warnings_count == 0`
- `so_load_status == "ok"`
- `roundtrip_status == "ok"`
- `missing_required_packages == []`
- `incompatible_required_packages == []`

## Plan de publication recommandé

1. Rebuild `liblumvorax.so` (sans symbole non résolu).
2. Publier dataset V13 avec wheels réellement compatibles tags Kaggle runtime.
3. Garder une seule version par package critique (éviter ambiguïtés).
4. Lancer le validateur V13 strict dans Kaggle et publier uniquement si verdict final est `ok` sans warning.
