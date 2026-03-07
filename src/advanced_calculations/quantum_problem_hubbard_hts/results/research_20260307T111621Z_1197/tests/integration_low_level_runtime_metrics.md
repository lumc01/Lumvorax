# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `5627473209`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `15.98`
- avg_mem_percent_global: `66.43`

## Architecture (mode FULL V4 NEXT)
```mermaid
flowchart LR
  A[run_research_cycle.sh] --> B[metadata_capture]
  B --> C[cycle_guard]
  C --> D[physics_readiness_pack]
  D --> E[v4next_integration_status]
  E --> F[v4next_rollout_controller full]
  F --> G[v4next_rollout_progress]
  G --> H[v4next_realtime_evolution]
  H --> I[chatgpt_critical_tests]
  I --> J[authenticity_audit]
  J --> K[checksums/audit]
```

## Module runtime share
```mermaid
pie title Module runtime share (%)
    "dense_nuclear_proxy (18.72%)" : 18.72
    "hubbard_hts_core (24.97%)" : 24.97
    "qcd_lattice_proxy (19.07%)" : 19.07
    "quantum_chemistry_proxy (19.11%)" : 19.11
    "quantum_field_noneq (18.13%)" : 18.13
```
