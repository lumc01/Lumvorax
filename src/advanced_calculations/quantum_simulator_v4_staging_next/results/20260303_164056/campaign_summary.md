# Quantum Simulator V3 Staging Campaign Summary

- run_id: `20260303_164056`
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
- win_rate_mean: 0.652130
- win_rate_std: 0.029456
- win_rate_ci95: [0.641589, 0.662670]
- throughput_mean: 10123866.818
- latency_p95_mean_ns: 168672.4
- latency_p99_mean_ns: 187033.8
- max_qubits_width: 36.0

### deterministic_seeded
- count: 30
- win_rate_mean: 0.656296
- win_rate_std: 0.024023
- win_rate_ci95: [0.647700, 0.664893]
- throughput_mean: 9965170.890
- latency_p95_mean_ns: 173786.0
- latency_p99_mean_ns: 191627.4
- max_qubits_width: 36.0

### baseline_neutralized
- count: 30
- win_rate_mean: 0.774259
- win_rate_std: 0.018743
- win_rate_ci95: [0.767552, 0.780966]
- throughput_mean: 9940653.950
- latency_p95_mean_ns: 172757.0
- latency_p99_mean_ns: 194391.8
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