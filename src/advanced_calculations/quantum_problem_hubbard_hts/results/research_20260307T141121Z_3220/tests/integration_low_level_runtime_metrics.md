# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `6463547948`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `17.78`
- avg_mem_percent_global: `74.37`

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
    "dense_nuclear_proxy (18.29%)" : 18.29
    "hubbard_hts_core (26.34%)" : 26.34
    "qcd_lattice_proxy (19.99%)" : 19.99
    "quantum_chemistry_proxy (17.85%)" : 17.85
    "quantum_field_noneq (17.53%)" : 17.53
```
