# RAPPORT COMPARATIF RÉEL — V126 vs V130 vs V131
## Périmètre et preuves utilisées
- Kernels analysés localement : `nx47_vesu_kernel_v126.py`, `nx47_vesu_kernel_v130.py`, `nx47_vesu_kernel_v131.py`.
- Logs disponibles localement :
  - `nx47-vesu-kernel-new-v130.log` : présent (142 lignes).
  - `nx47-vesu-kernel-new-v131.log` : absent.

> Note vérité: dans ce workspace, le log V131 distant n'est pas présent. Le rapport V131 runtime ci-dessous ne peut pas être recalculé sans ce fichier.

## 1) Résultats réels extraits du log V130
| KPI | Valeur réelle |
|---|---:|
| `files_processed` | 1 |
| `slices_processed` | 320 |
| `pixels_processed` | 32768000 |
| `pixels_anchor_detected` | 0 |
| `pixels_papyrus_without_anchor` | 6144 |
| `materials_detected` | 933 |
| `patterns_detected` | 933 |
| `golden_nonce_detected` | 11 |
| `unknown_discoveries` | 0 |
| `anomalies_detected` | 52 |
| `calc_ops_estimated` | 27852800 |
| `meta_neuron_candidates` | 45 |
| `mutation_events` | 0 |
| `pruning_events` | 1 |
| `val_f1_mean_supervised` | 0.0 |
| `val_iou_mean_supervised` | 0.0 |
| `best_threshold_mean_supervised` | 0.35 |
| `FILE_DONE.elapsed_s` | 3.123 |
| `FILE_DONE.calc_per_sec` | 8920019.62472221 |
| `probability_audit.max` | 0.1458509862422943 |

## 2) État V131 runtime (log distant demandé)
- Le log `nx47-vesu-kernel-new-v131.log` **n'est pas présent localement** dans ce dépôt de travail.
- Conséquence: je ne peux pas confirmer par preuve locale les métriques runtime V131 (GLOBAL_STATS/FILE_DONE/EXEC_COMPLETE) tant que ce fichier n'est pas sync ici.

## 3) Avant / Après code (V126 -> V130 -> V131)
| Capacité | V126 | V130 | V131 |
|---|---:|---:|---:|
| `supervised_train` | ✅ | ✅ | ✅ |
| `threshold_scan` | ✅ | ✅ | ✅ |
| `golden_nonce_topk` | ✅ | ✅ | ✅ |
| `hysteresis_topology_3d` | ✅ | ✅ | ✅ |
| `train_unet_25d_supervised` | ❌ | ✅ | ✅ |
| `forensic_parser_v130` | ❌ | ❌ | ✅ |
| `forensic_report_v131` | ❌ | ❌ | ✅ |

### Lecture synthétique
- **V126**: pipeline supervisé + calibration + golden nonces + hysteresis déjà présents.
- **V130**: conserve V126 + ajoute branche 2.5D U-Net plus structurée.
- **V131**: conserve V130 et ajoute couche forensic dédiée (`_parse_v130_log_summary`, `_build_v131_forensic_report`) et métriques probabilistes globales.

## 4) Écarts critiques vis-à-vis des analyses précédentes
1. Les constats principaux remontés (anchor=0, val_f1=0, 1 fichier traité, anomalies=52, golden_nonce=11) sont **confirmés pour V130** via log réel.
2. Le fichier runtime V131 demandé n'étant pas localement disponible, la validation “résultats V131 réels” reste **à compléter dès synchronisation du log**.
3. Le code V131 contient désormais les hooks pour formaliser ce rapport automatiquement dès que le log source est présent.

## 5) Actions réalisées par moi dans ce cycle
- Vérification présence logs et kernels.
- Parsing forensic du log V130 local (142 lignes, événements JSON, GLOBAL_STATS).
- Revue comparative des trois kernels (V126/V130/V131) sur les briques clés.
- Rédaction de ce rapport “avant/après” fondé sur preuves locales.
