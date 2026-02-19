# Validation système LUM/VORAX — exécution locale

- Timestamp: 2026-02-19 19:18:32
- Durée: 0.5966 s

## Résumé
- ✅ **source_indentation**: {"ok": true}
- ⏳ **native_sources**: {"c_candidates_checked": ["/kaggle/working/src/vorax/vorax_operations.c", "/kaggle/working/src/lum/lum_core.c", "/kaggle/working/src/logger/lum_logger.c", "src/vorax/vorax_operations.c", "src/lum/lum_core.c", "src/logger/lum_logger.c"], "c_sources_found": ["src/vorax/vorax_operations.c", "src/lum/lum_core.c", "src/logger/lum_logger.c"], "native_3d_c_sources_present": true}
- ⏳ **native_compile_attempt**: {"ok": false, "output": "", "reason_if_empty": "missing /kaggle/working sources or gcc failure"}
- ✅ **lum_roundtrip_unit**: {"ok": true, "shape": [4, 12, 10], "dtype": "float32", "payload_sha512_prefix": "b73a033f5092362549475f67"}
- ✅ **python_integration_smoke**: {"ok": true, "stats": {"files_processed": 1, "pixels_processed": 4608, "ink_detected": 552, "mean_density": 0.11979166666666667, "active_neurons_start_total": 4608, "active_neurons_mid_total": 1613, "active_neurons_end_total": 950, "mutation_events": 0, "pruning_events": 1, "files_autonomous_fallback": 0, "lum_bridge_enabled": false}, "artifacts": {"submission_zip": "/tmp/tmpvk_bos64/out/submission.zip", "submission_parquet": "/tmp/tmpvk_bos64/out/submission.parquet", "metadata": "/tmp/tmpvk_bos64/out/v134_execution_metadata.json", "roadmap": "/tmp/tmpvk_bos64/out/v134_roadmap_realtime.json"}, "artifacts_exist": {"submission_zip": true, "submission_parquet": true, "metadata": true, "roadmap": true}}

## Conclusion experte
- Le pipeline Python 3D + format `.lum` est validé localement.
- Le moteur C 3D natif n'est pas confirmé à 100% dans cet environnement tant que les sources `.c` et/ou `.so` ne sont pas disponibles et compilables.
- Les artefacts de preuve machine sont dans `validation_results.json`.
