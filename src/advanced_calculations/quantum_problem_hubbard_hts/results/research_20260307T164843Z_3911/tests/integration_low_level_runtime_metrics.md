# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `5569674829`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `15.78`
- avg_mem_percent_global: `59.67`

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
    "dense_nuclear_proxy (18.26%)" : 18.26
    "hubbard_hts_core (24.94%)" : 24.94
    "qcd_lattice_proxy (18.96%)" : 18.96
    "quantum_chemistry_proxy (18.69%)" : 18.69
    "quantum_field_noneq (19.15%)" : 19.15
```
