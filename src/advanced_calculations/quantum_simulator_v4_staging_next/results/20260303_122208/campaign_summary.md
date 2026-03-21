# Quantum Simulator V3 Staging Campaign Summary

- run_id: `20260303_122208`
- runs_per_mode: **30**
- scenarios: **360**
- steps: **1400**
- max_qubits_width: **36**

## A/B officiel vs staging
- officiel smoke rc: `0`
- staging fusion gate pass: `True`
- gate latency p95 max ns: `300000.0`
- gate latency p99 max ns: `900000.0`

## Modes
### hardware_preferred
- count: 30
- win_rate_mean: 0.651574
- win_rate_std: 0.021869
- win_rate_ci95: [0.643749, 0.659400]
- throughput_mean: 8589807.935
- latency_p95_mean_ns: 198881.7
- latency_p99_mean_ns: 224827.4
- max_qubits_width: 36.0

### deterministic_seeded
- count: 30
- win_rate_mean: 0.656296
- win_rate_std: 0.024023
- win_rate_ci95: [0.647700, 0.664893]
- throughput_mean: 8159860.643
- latency_p95_mean_ns: 205932.7
- latency_p99_mean_ns: 236095.7
- max_qubits_width: 36.0

### baseline_neutralized
- count: 30
- win_rate_mean: 0.774259
- win_rate_std: 0.018743
- win_rate_ci95: [0.767552, 0.780966]
- throughput_mean: 7924898.907
- latency_p95_mean_ns: 209656.3
- latency_p99_mean_ns: 230005.3
- max_qubits_width: 36.0

## Intégrité manifest/signature (style V6)
- manifest build rc: 0
- sign rc: 0
- verify rc: 0

## Fichiers produits
- campaign_runs.csv
- scenario_losses_frequency.csv
- failure_reasons_frequency.csv
- campaign_summary.json
- manifest_forensic_v3.json + .sig

- throughput_regression_flag: False

## Watchlist scénarios fragiles
- watchlist: 74, 117, 133