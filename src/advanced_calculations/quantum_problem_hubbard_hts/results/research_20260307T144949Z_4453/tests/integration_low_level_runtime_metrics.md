# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `8047491869`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `18.61`
- avg_mem_percent_global: `75.68`

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
    "dense_nuclear_proxy (18.14%)" : 18.14
    "hubbard_hts_core (25.33%)" : 25.33
    "qcd_lattice_proxy (19.20%)" : 19.20
    "quantum_chemistry_proxy (19.18%)" : 19.18
    "quantum_field_noneq (18.15%)" : 18.15
```
