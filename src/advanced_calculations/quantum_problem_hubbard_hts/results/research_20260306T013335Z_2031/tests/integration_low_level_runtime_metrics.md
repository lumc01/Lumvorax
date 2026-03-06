# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `6552919389`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `19.23`
- avg_mem_percent_global: `63.79`

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
    "dense_nuclear_proxy (17.51%)" : 17.51
    "hubbard_hts_core (24.52%)" : 24.52
    "qcd_lattice_proxy (19.10%)" : 19.10
    "quantum_chemistry_proxy (18.62%)" : 18.62
    "quantum_field_noneq (20.26%)" : 20.26
```
