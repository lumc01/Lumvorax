# Quantum Simulator V3 Staging Campaign Summary

- run_id: `20260303_051554`
- runs_per_mode: **30**
- scenarios: **360**
- steps: **1400**

## A/B officiel vs staging
- officiel smoke rc: `0`
- staging fusion gate pass: `True`
- gate latency p95 max ns: `300000.0`
- gate latency p99 max ns: `900000.0`

## Modes
### hardware_preferred
- count: 30
- win_rate_mean: 0.655741
- win_rate_std: 0.029335
- win_rate_ci95: [0.645243, 0.666238]
- throughput_mean: 8587502.052
- latency_p95_mean_ns: 192804.7
- latency_p99_mean_ns: 214552.1

### deterministic_seeded
- count: 30
- win_rate_mean: 0.656296
- win_rate_std: 0.024023
- win_rate_ci95: [0.647700, 0.664893]
- throughput_mean: 8213471.310
- latency_p95_mean_ns: 199029.6
- latency_p99_mean_ns: 220376.1

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

## Watchlist scénarios fragiles
- watchlist: 74, 117, 133