# Module isolé — Hubbard HTS (supraconductivité haute température)

Ce dossier est **indépendant** des modules existants et sert à traiter, en séquence, les problèmes suivants sans toucher aux autres versions:
1. Hubbard HTS (problème principal)
2. QCD sur réseau (proxy)
3. Dynamique hors équilibre (proxy)
4. Matière nucléaire dense (proxy)
5. Chimie quantique haute précision (proxy)

## Garanties d'isolation et de traçabilité
- Chaque exécution crée un dossier unique: `results/<timestamp_pid>/`
- Logs et rapports séparés:
  - `logs/execution.log` (lignes numérotées)
  - `logs/metrics.csv` (métriques pas-à-pas en ns)
  - `logs/hardware_snapshot.log` (infos système bas niveau `/proc`)
  - `reports/rapport_pedagogique.md`
- **Aucun écrasement**: ID de run unique par timestamp + PID.
- Avant chaque correction/exécution via script, une copie versionnée est créée dans `backups/version_YYYYMMDD_HHMMSS/`.

## Exécution
```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_hubbard_hts.sh 99 99
```

## Notes
- Implémentation volontairement en **C** uniquement.
- Le script vise une charge élevée CPU/RAM, mais la consommation réelle dépend des limites OS/conteneur.
