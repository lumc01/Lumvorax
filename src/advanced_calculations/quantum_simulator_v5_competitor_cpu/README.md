# Quantum Simulator V5 Competitor CPU Pack

Nouvelle copie isolée pour intégrer les 6 concurrents CPU immédiatement exploitables,
sans toucher aux versions V2/V3/V4 existantes.

## Concurrents intégrés (CPU)
1. Qiskit Aer
2. quimb (Tensor Network MPS)
3. Qulacs
4. MQT DDSIM
5. ProjectQ
6. QuTiP

## Exécution Replit exacte
```bash
bash /workspace/Lumvorax/src/advanced_calculations/quantum_simulator_v5_competitor_cpu/run_on_replit_v5_competitors.sh /workspace/Lumvorax 30 360 1400 36 0 0
```

Paramètres:
1. `ROOT`
2. `RUNS`
3. `SCENARIOS`
4. `STEPS`
5. `MAX_QUBITS_WIDTH`
6. `PLAN_ONLY` (`1` = clone + plan sans benchmark snippet)
7. `SKIP_INSTALL` (`1` = n'installe pas pip)

## Mode rapide de validation (sans install)
```bash
bash /workspace/Lumvorax/src/advanced_calculations/quantum_simulator_v5_competitor_cpu/run_on_replit_v5_competitors.sh /workspace/Lumvorax 1 20 40 36 1 1
```

## Artefacts générés
- `src/advanced_calculations/quantum_simulator_v5_competitor_cpu/results/<run_id>/competitor_cpu_results.csv`
- `src/advanced_calculations/quantum_simulator_v5_competitor_cpu/results/<run_id>/competitor_cpu_summary.json`
- `src/advanced_calculations/quantum_simulator_v5_competitor_cpu/results/<run_id>/competitor_cpu_summary.md`
