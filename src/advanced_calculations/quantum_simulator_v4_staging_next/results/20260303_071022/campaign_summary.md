# Quantum Simulator V3 Staging Campaign Summary

- run_id: `20260303_071022`
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
- win_rate_mean: 0.663241
- win_rate_std: 0.020509
- win_rate_ci95: [0.655902, 0.670580]
- throughput_mean: 8604443.795
- latency_p95_mean_ns: 198700.7
- latency_p99_mean_ns: 219552.3
- max_qubits_width: 36.0

### deterministic_seeded
- count: 30
- win_rate_mean: 0.656296
- win_rate_std: 0.024023
- win_rate_ci95: [0.647700, 0.664893]
- throughput_mean: 7951806.914
- latency_p95_mean_ns: 209930.3
- latency_p99_mean_ns: 238991.0
- max_qubits_width: 36.0

### baseline_neutralized
- count: 30
- win_rate_mean: 0.774259
- win_rate_std: 0.018743
- win_rate_ci95: [0.767552, 0.780966]
- throughput_mean: 7868658.992
- latency_p95_mean_ns: 214037.7
- latency_p99_mean_ns: 250209.6
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