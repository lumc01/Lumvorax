# RAPPORT FINAL V2 — Copy NQubit NX sans écrasement

## Exécution V2 créée
- Dossier V2: `src/quantum/vkernel_nqubit_v2/`
- Dernier run: `20260302_183549`
- Commande Replit:
```bash
cd /workspace/Lumvorax && python3 src/quantum/vkernel_nqubit_v2/run_vkernel_nqubit_comparison_v2.py
```

## Réponse claire sur "combien de qubits avant"
- Avant (baseline origine C): `baseline_qubits_simulated_proxy = 3.0` (mesuré via lignes ajoutées dans `hardware_metrics.log`).
- Nouveau simulateur NX V2: `nqubits_simulated = 504000.0`.
- Ratio réel V2: `ratio_nqubit_vs_baseline_proxy = 168000.0`.

## Autres métriques générées
- `nqubits_per_sec = 10446787.33`
- `nqubit_avg_score = 0.9609706`
- `baseline_qubit_avg_score = 0.940176696`
- `nqubit_win_rate = 0.652777778`

## Rapports précédents retrouvés (avec dates de référence)
- src/quantum/ANALYSE_REMOTE_DEPOT_ET_LOGS_VKERNEL_NQUBIT_20260302.md — mtime_utc=2026-03-02T18:34:07Z
- src/quantum/PLAN_ANALYSE_INTEGRATION_V6_NQUBIT_SIMULATEUR.md — mtime_utc=2026-03-02T18:34:01Z
- src/quantum/RAPPORT_COMPARATIF_DETAILLE_QUbits_NQUBIT_20260302.md — mtime_utc=2026-03-02T18:34:07Z

## Anciennes exécutions `src/quantum/results_vkernel_compare`
- 20260302_172657: baseline=4.0 nqubits=504000.0 win_rate=0.652777778
- 20260302_180157: baseline=4.0 nqubits=504000.0 win_rate=0.652777778
- 20260302_180853: baseline=4.0 nqubits=504000.0 win_rate=0.652777778

## Différences technologiques claires
- Origine `v_kernel_quantum.c`: simulation courte, 3 métriques loguées.
- V2 NX copy: simulation multi-scénarios/multi-steps, score comparatif, débit `nqubits_per_sec`, log forensic ns.

## Statut A→Z (sans régression)
- Rien d'existant n'a été écrasé.
- Tout est produit dans un nouveau dossier V2 + runs horodatés.
- Les anciennes sorties restent intactes.
