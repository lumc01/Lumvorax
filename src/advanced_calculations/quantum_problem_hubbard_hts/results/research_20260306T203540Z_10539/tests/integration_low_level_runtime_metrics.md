# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `6380546631`
- total_qubits_simulated_proxy: `373`
- avg_cpu_percent_global: `15.04`
- avg_mem_percent_global: `72.72`

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
    "dense_nuclear_proxy (19.51%)" : 19.51
    "hubbard_hts_core (22.81%)" : 22.81
    "qcd_lattice_proxy (18.44%)" : 18.44
    "quantum_chemistry_proxy (22.06%)" : 22.06
    "quantum_field_noneq (17.18%)" : 17.18
```
