# V2 NQubit NX (copy non destructive)

Cette V2 est créée **dans un nouveau dossier** pour ne rien casser:
- `src/quantum/vkernel_nqubit_v2/`

## Commande Replit (nouvelle version V2)
```bash
cd /workspace/Lumvorax && python3 src/quantum/vkernel_nqubit_v2/run_vkernel_nqubit_comparison_v2.py
```

## Garanties
- Aucun fichier historique n'est effacé.
- Les artefacts V2 sont écrits uniquement dans:
  - `src/quantum/vkernel_nqubit_v2/results/<timestamp>/`
- L'index V2 est:
  - `src/quantum/vkernel_nqubit_v2/LATEST_V2.json`

## Ce que V2 ajoute
- Mesure `nqubits_per_sec`.
- Détection automatique des lignes métriques ajoutées par la baseline dans `logs_AIMO3/v46/hardware_metrics.log`.
- Références automatiques vers les anciens rapports `src/quantum` avec date UTC (`mtime`).
- Références vers les runs précédents `results_vkernel_compare`.
