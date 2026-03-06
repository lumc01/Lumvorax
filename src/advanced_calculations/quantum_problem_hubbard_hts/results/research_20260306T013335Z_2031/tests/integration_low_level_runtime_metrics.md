# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `5047213732`
- total_qubits_simulated_proxy: `317`
- avg_cpu_percent_global: `15.38`
- avg_mem_percent_global: `51.05`

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
    "dense_nuclear_pr (0.00%)" : 0.00
    "dense_nuclear_proxy (17.07%)" : 17.07
    "hubbard_hts_core (31.83%)" : 31.83
    "qcd_lattice_proxy (24.80%)" : 24.80
    "quantum_field_noneq (26.30%)" : 26.30
```
