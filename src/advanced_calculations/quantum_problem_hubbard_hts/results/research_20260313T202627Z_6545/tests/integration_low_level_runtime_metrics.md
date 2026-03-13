# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `265336377`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `13.62`
- avg_mem_percent_global: `49.20`

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
    "bosonic_multimode_systems (6.30%)" : 6.30
    "correlated_fermions_non_hubbard (7.82%)" : 7.82
    "dense_nuclear_fullscale (5.51%)" : 5.51
    "far_from_equilibrium_kinetic_lattices (8.63%)" : 8.63
    "hubbard_hts_core (10.21%)" : 10.21
    "multi_correlated_fermion_boson_networks (8.67%)" : 8.67
    "multi_state_excited_chemistry (6.76%)" : 6.76
    "multiscale_nonlinear_field_models (7.98%)" : 7.98
    "qcd_lattice_fullscale (6.42%)" : 6.42
    "quantum_chemistry_fullscale (4.68%)" : 4.68
    "quantum_field_noneq (5.05%)" : 5.05
    "spin_liquid_exotic (11.15%)" : 11.15
    "topological_correlated_materials (10.81%)" : 10.81
```
