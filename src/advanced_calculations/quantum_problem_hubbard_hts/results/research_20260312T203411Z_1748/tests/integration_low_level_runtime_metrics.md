# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `250417209`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.49`
- avg_mem_percent_global: `61.10`

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
    "bosonic_multimode_systems (6.49%)" : 6.49
    "correlated_fermions_non_hubbard (7.72%)" : 7.72
    "dense_nuclear_fullscale (5.66%)" : 5.66
    "far_from_equilibrium_kinetic_lattices (8.39%)" : 8.39
    "hubbard_hts_core (10.25%)" : 10.25
    "multi_correlated_fermion_boson_networks (8.56%)" : 8.56
    "multi_state_excited_chemistry (6.77%)" : 6.77
    "multiscale_nonlinear_field_models (7.93%)" : 7.93
    "qcd_lattice_fullscale (6.52%)" : 6.52
    "quantum_chemistry_fullscale (4.72%)" : 4.72
    "quantum_field_noneq (4.98%)" : 4.98
    "spin_liquid_exotic (11.19%)" : 11.19
    "topological_correlated_materials (10.82%)" : 10.82
```
