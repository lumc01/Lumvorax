# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `25131995380`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.63`
- avg_mem_percent_global: `76.63`

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
    "bosonic_multimode_systems (7.27%)" : 7.27
    "correlated_fermions_non_hubbard (7.87%)" : 7.87
    "dense_nuclear_fullscale (6.82%)" : 6.82
    "far_from_equilibrium_kinetic_lattices (7.83%)" : 7.83
    "hubbard_hts_core (9.26%)" : 9.26
    "multi_correlated_fermion_boson_networks (7.83%)" : 7.83
    "multi_state_excited_chemistry (7.53%)" : 7.53
    "multiscale_nonlinear_field_models (7.50%)" : 7.50
    "qcd_lattice_fullscale (7.17%)" : 7.17
    "quantum_chemistry_fullscale (7.14%)" : 7.14
    "quantum_field_noneq (6.97%)" : 6.97
    "spin_liquid_exotic (8.57%)" : 8.57
    "topological_correlated_materials (8.26%)" : 8.26
```
