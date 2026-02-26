# Audit complet V6 → V7 (Lumvorax)

## Portée et méthode
- Scan exhaustif des fichiers via index local (`docs/audit/file_inventory.tsv`).
- Génération d'un arbre de répertoires de substitution (`tree_like_L2.txt` et `tree_like_L4.txt`) car la commande `tree` n'est pas disponible dans l'environnement.
- Analyse ciblée du notebook de base V6: `RAPPORT-VESUVIUS/NX46-VX/result-nx46-vx-v6/nx46-vx-unified-kaggle-v6.ipynb`.

## Chiffres globaux (A→Z, sans exclusion)
- Nombre total de fichiers: **33242**.
- Nombre de fichiers Markdown (`.md`): **1159**.
- Top extensions: `.py`=11741, `.h`=10524, `.pyi`=2058, `<no_ext>`=2003, `.md`=1159, `.txt`=756, `.tif`=753, `.so`=622, `.c`=552, `.blob`=548, `.png`=250, `.json`=222, `.mat`=110, `.csv`=106, `.cpp`=92.

## Diagnostic par sous-projet (racine)
| Sous-projet | Nb fichiers | Signatures technologiques observées (top extensions) |
|---|---:|---|
| `.venv` | 24790 | .py:10729, .h:9809, .pyi:1260, <no_ext>:791, .so:622 |
| `.pythonlibs` | 1859 | .py:804, .pyi:798, <no_ext>:118, .c:59, .txt:25 |
| `RAPPORT-VESUVIUS` | 1595 | .tif:751, .c:177, .h:149, .md:140, .py:73 |
| `.ccls-cache` | 1096 | .blob:548, .h:462, .c:37, .tcc:20, .0@bit:1 |
| `test_persistence.db` | 1000 | <no_ext>:1000 |
| `attached_assets` | 629 | .txt:422, .png:175, .lean:8, .py:6, .ipynb:5 |
| `src` | 396 | .c:178, .h:80, .o:43, .lean:30, .cpp:27 |
| `RAPPORT_IAMO3` | 324 | .md:268, .txt:42, .json:8, .sh:3, .png:2 |
| `reports` | 265 | .md:256, .json:3, .csv:3, <no_ext>:1, .dat:1 |
| `logs_AIMO3` | 163 | .csc:60, .json:42, .csv:36, .md:13, .bin:6 |
| `trou_noir_sim` | 59 | .c:20, <no_ext>:12, .md:10, .h:4, .json:4 |
| `.local` | 32 | .bin:30, <no_ext>:1, .json:1 |
| `evidence` | 27 | .txt:25, .json:2 |
| `kaggle_outputs` | 26 | .json:11, .bin:8, .parquet:7 |
| `PREUVE_IAMO` | 22 | .txt:15, .md:7 |
| `logs` | 20 | .txt:15, .json:1, .parquet:1, <no_ext>:1, .c:1 |
| `RAPPORT` | 18 | .md:18 |
| `dataset` | 16 | .py:10, .csv:3, .zip:1, .pdf:1, .proto:1 |
| `DATASET` | 16 | .py:10, .csv:3, .pdf:1, .txt:1, .proto:1 |
| `v28_forensic_logs` | 16 | .json:8, .bin:8 |
| `DÉSACTIVÉ` | 15 | .c:7, .h:7, .md:1 |
| `nx47_dependencies_v3` | 13 | .whl:12, .json:1 |
| `lum-vorax-dependencies` | 11 | .whl:10, .json:1 |
| `tests` | 11 | .py:11 |
| `temp_v1_v2` | 10 | .whl:10 |
| `results` | 9 | .json:8, .csv:1 |
| `kaggle_kernels` | 8 | .json:4, .py:4 |
| `v44v1` | 8 | .md:2, .png:2, .sha512:2, .json:1, .csv:1 |
| `forensic_analysis_nx47_v2` | 7 | .md:7 |
| `iamo3_results` | 7 | .json:3, .bin:3, .parquet:1 |
| `nfl_results` | 7 | .json:3, .bin:3, .parquet:1 |
| `modules` | 6 | .py:5, .json:1 |
| `build_kaggle` | 5 | .json:3, .md:1, .py:1 |
| `final_v4_output` | 5 | .json:2, .bin:2, .parquet:1 |
| `iamo3_results_v2` | 5 | .json:2, .bin:2, .parquet:1 |

## État de la base V6 (détection de fonctionnalités perf)
- ❌ `torch.compile`
- ❌ `autocast`
- ❌ `GradScaler`
- ❌ `channels_last`
- ❌ `pin_memory`
- ❌ `prefetch_factor`
- ❌ `persistent_workers`
- ❌ `num_workers`
- ❌ `cudnn.benchmark`
- ❌ `torch.jit`
- ❌ `onnx`
- ❌ `tensorrt`
- ❌ `xformers`
- ❌ `flash_attn`
- ❌ `numba`
- ❌ `nvidia.dali`
- ❌ `dask`
- ❌ `polars`
- ❌ `zarr`
- ❌ `memmap`
- ❌ `parquet`
- ✅ `cupy`
- ✅ `cupyx`
- ✅ `ray`
- ✅ `tta`
- ✅ `ensemble`

## Gisements d'optimisation disponibles dans le repo mais non intégrés en V6
1. **Pipeline mémoire/disque haute performance côté C**: `mmap_persistence`, `zero_copy_allocator`, `slab_allocator`, `lz4_compressor`, `lockfree_queue`, `simd_batch_processor` présents sous `src/optimization/` mais absents du notebook V6.
2. **Optimisation PyTorch runtime**: absence de `torch.compile`, AMP (`autocast`/`GradScaler`), `channels_last`, et réglages DataLoader (`pin_memory`, `persistent_workers`, `prefetch_factor`).
3. **Format de données orienté throughput**: absence de `memmap`/`zarr`/`parquet` détectée dans V6 alors que le repo contient des assets volumineux (`.tif`, `.blob`) qui bénéficieraient de streaming et de cache.
4. **Compilation/export**: pas de `torch.jit`/ONNX/TensorRT détecté dans V6 (voie potentielle pour inférence Kaggle).
5. **Vectorisation CPU/GPU**: présence de composants SIMD C mais pas de pont Python explicite dans V6.

## Plan V7 recommandé (priorité exécution Kaggle)
### P0 (impact immédiat)
- Ajouter AMP + `torch.compile` (avec fallback) + `channels_last` sur les modèles CNN.
- Revoir DataLoader (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`) avec micro-benchmark intégré au notebook.
- Introduire cache local `numpy.memmap` pour patches/tiles TIF prétraités.

### P1 (impact fort, effort moyen)
- Brancher `src/optimization/lz4_compression` + `mmap_io` pour sérialiser et relire rapidement features intermédiaires.
- Intégrer un mode infer ONNX Runtime (CPU/GPU selon dispo) pour réduire latence prédictive.
- Utiliser pipeline asynchrone (préfetch CPU + transfert GPU non bloquant).

### P2 (R&D)
- Créer bindings Python (ctypes/cffi/pybind11) pour exploiter `simd_optimizer` et `simd_batch_processor`.
- Évaluer Dask/Ray seulement si charge multi-image massive (sinon overhead trop élevé en notebook Kaggle).

## Livrables générés
- `docs/audit/file_inventory.tsv` (inventaire exhaustif fichier par fichier).
- `docs/audit/md_inventory.txt` (liste exhaustive `.md`).
- `docs/audit/tree_like_L2.txt` et `docs/audit/tree_like_L4.txt` (arborescences).
- `docs/v7_audit_inventory.json` (statistiques structurées).
